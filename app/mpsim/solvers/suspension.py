"""Particle dynamics of a dense suspension in simple shear (lubrication DEM).

Why this exists
---------------
The voxel flow solve of solvers/viscosity.py resolves the hydrodynamics of
one particle arrangement, but not the resin films between particles thinner
than a voxel, nor contacts, friction or the forces between particle
surfaces. Near the maximum packing those decide the viscosity, and they are
also what makes a fine filler stiffer than a coarse one at the same content:
creeping flow alone has no length scale, so 1 um and 10 um spheres give the
same relative viscosity in it. Here each particle moves, turns and collides
in a sheared periodic box, and the forces that carry a length of their own -
surface roughness, van der Waals attraction - make the result depend on the
particle size.

The model (Mari, Seto, Morris and Denn, J. Rheol. 58 (2014) 1693; Ness and
Sun, Phys. Rev. E 91 (2015) 012201)
-------------------------------------------------------------------------
* Zero Reynolds number: no inertia; every particle is force- and torque-free.
  Each step solves R . V = F for the particle velocities and spins relative
  to the imposed simple shear U = gd y e_x, Omega = -gd/2 e_z.
* R: the Stokes drag of each particle (6 pi mu a, 8 pi mu a^3) and, for pairs
  whose surfaces are closer than LUB_CUT times their mean radius, the
  leading lubrication terms (Jeffrey and Onishi 1984): the squeeze resistance
  6 pi mu R*^2 / h along the line of centres and the log(1/h) shear
  resistance on the relative velocity of the two surfaces. Both are shifted
  to zero at the cut-off. h is regularised by the roughness delta (a length):
  the resin film never closes below it, and at h < 0 the asperities touch.
  Each pair term is the gradient of a dissipation c_n v_n^2 + c_t |v_t|^2, so
  R stays symmetric positive definite.
* Contacts (h < 0): a stiff normal spring and a tangential spring capped by
  Coulomb friction mu_f (Cundall and Strack 1979), its history kept per
  contact. The springs are implicit (backward Euler): over a step their
  force grows by k dt times the relative surface velocity, which enters R
  beside the lubrication, so a stiff contact does not limit the step.
* Van der Waals attraction A R* / (6 h^2), A the Hamaker constant of the
  filler across the resin, cut at h_min (closest approach of the surfaces).
* Lees-Edwards boundaries: the periodic images above and below slide with
  the shear.
* Stress: sigma = (1/V) sum over pairs of d (x) F (d from i to j, F the pair
  force on i) for lubrication, contacts and van der Waals, plus the stresslet
  of each sphere in the shear (Einstein's 5/2 phi) and the resin.

Units: length = the largest radius a0, time = 1/gd, viscosity = mu (the resin
at the shear rate), so a force is in mu gd a0^2. Physical inputs are made
dimensionless in Suspension.

Particles of other shapes (cubes, platelets, rods, polyhedra ...) are rigid
clumps of overlapping spheres - the multi-sphere method of DEM - whose
members feel the same pair laws (RigidSuspension).

Computation (Taichi): the same kernels run on the CPU (all cores) or an NVIDIA
GPU (CUDA), in double precision. Each particle keeps the list of its
neighbours with a higher index (fixed slots), so every pair is handled by
one thread and the contact history needs no locks. The linear system is
solved matrix-free by preconditioned conjugate gradients.
"""
# no "from __future__ import annotations": Taichi reads the kernel argument
# annotations (ti.template(), ti.f64) as objects
import math
import time

import numpy as np

LUB_CUT = 0.25          # lubrication acts below this gap / mean radius of the pair
KN_FACTOR = 1000.0      # contact stiffness against the strongest attraction (see Suspension)
# scatter of the mean viscosity between runs from other random packings,
# relative: RUN_SCATTER / sqrt(spheres x averaged strain). Fitted to 50 vol%
# silica, 1 and 10 um: 500 spheres to a strain of 5 scattered by 3.6 % and
# 4.9 % between three packings each (the 8 blocks of one run said 1.4 and
# 3.4 %), 150 spheres to a strain of 2 by 17 % and 32 % between five
RUN_SCATTER = 2.5
# pairs are listed this far (in largest radii) beyond the range of their
# forces; the lists are rebuilt before any pair could have closed half of it.
# Absolute, so pairs of small spheres have the same margin as large ones
# (a skin of 0.05 of the pair's own radius let two spheres 0.15 of the
# largest close past it between rebuilds)
SKIN = 0.05
# arch: the running Taichi backend; prefer: what it was started for;
# cuda_failed: why a run on the GPU failed (the process then stays on the CPU)
_TI = {"arch": None, "prefer": None, "cuda_failed": None, "said": None, "logged": None}


def init_backend(prefer="auto", log=None):
    """Starts Taichi for the particle dynamics: CUDA first - an NVIDIA GPU
    when one answers - and the CPU otherwise ('cpu' asks for the CPU only).
    Kept for the process; a later request changes it only when it asks for
    the CPU on a GPU process, or for the GPU on a process that had been told
    to use the CPU. Returns the backend name ('cuda' or 'x64')."""
    import taichi as ti
    want_gpu = prefer != "cpu"
    cur = _TI["arch"]
    if cur is not None:
        switch = (not want_gpu and cur == "cuda") or (want_gpu and cur != "cuda" and _TI["prefer"] == "cpu"
                                                      and not _TI["cuda_failed"])
        if not switch:
            return cur
        ti.reset()
        _TI["arch"] = None
    order = ["cuda", "cpu"] if (want_gpu and not _TI["cuda_failed"]) else ["cpu"]
    last, why = None, None
    for name in order:
        try:
            ti.init(arch=getattr(ti, name), default_fp=ti.f64, default_ip=ti.i32, offline_cache=True,
                    log_level=ti.ERROR, random_seed=0)
            got = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
            if name != "cpu" and got in ("x64", "arm64"):
                ti.reset()                                     # no GPU: Taichi fell back silently
                why = "no NVIDIA GPU with CUDA found"
                continue
            _TI["arch"], _TI["prefer"] = got, prefer
            if got == "cuda":
                said = "particle dynamics on the GPU (CUDA)"
            else:
                reason = ("as set" if not want_gpu else
                          f"the GPU run failed: {_TI['cuda_failed']}" if _TI["cuda_failed"] else why or "")
                said = f"particle dynamics on the CPU{f' ({reason})' if reason else ''}"
            _TI["said"] = said
            # said once per change, not at every run (each run starts Taichi again)
            if log and said != _TI.get("logged"):
                log("    " + said)
                _TI["logged"] = said
            return got
        except Exception as e:                                 # noqa: BLE001
            last = e
            if name == "cuda":
                why = f"CUDA did not start: {str(e).splitlines()[0][:160] if str(e) else type(e).__name__}"
            try:
                ti.reset()
            except Exception:                                  # noqa: BLE001
                pass
    raise RuntimeError(f"Taichi could not start ({last})")


def cuda_device_count():
    """NVIDIA GPUs the CUDA driver reports, asked without starting Taichi
    (0 when there is no driver or no device)."""
    import ctypes
    import sys
    try:
        lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        if lib.cuInit(0) != 0:
            return 0
        n = ctypes.c_int(0)
        return int(n.value) if lib.cuDeviceGetCount(ctypes.byref(n)) == 0 else 0
    except Exception:                                          # noqa: BLE001
        return 0


def device_text(prefer="auto", log=None):
    """Where the particle dynamics runs, for the stage line: starts the
    backend if it is not running (and logs the choice)."""
    if init_backend(prefer, log) == "cuda":
        return "the NVIDIA GPU (CUDA)"
    if prefer == "cpu":
        return "the CPU (as set)"
    said = _TI.get("said") or ""
    return "the CPU (" + said.split("(", 1)[1] if "(" in said else "the CPU"


def release():
    """Frees every Taichi field (the backend starts again on the next use)."""
    import taichi as ti
    if _TI["arch"] is None:
        return
    try:
        ti.reset()
    except Exception:                                          # noqa: BLE001
        pass
    _TI["arch"] = None


def fall_back_to_cpu(reason, log=None):
    """A run on the GPU failed: the CPU from now on in this process."""
    import taichi as ti
    _TI["cuda_failed"] = str(reason).splitlines()[0][:200] if str(reason) else "error"
    try:
        ti.reset()
    except Exception:                                          # noqa: BLE001
        pass
    _TI["arch"] = None
    return init_backend("auto", log)


class NeighbourOverflow(RuntimeError):
    """A particle has more neighbours than its list holds."""


def _kernels(n, maxn, nc=0):
    """Taichi fields and kernels for n particles with at most maxn
    higher-index neighbours each; nc cells per box edge for the neighbour
    search (below 3 every pair is checked)."""
    import taichi as ti
    vec = ti.types.vector(3, ti.f64)
    F3 = lambda shape: ti.Vector.field(3, ti.f64, shape=shape)        # noqa: E731

    x, u, w = F3(n), F3(n), F3(n)
    rad = ti.field(ti.f64, shape=n)
    sqa = ti.field(ti.f64, shape=n)                  # sqrt of the dimensionless Hamaker constant
    fu, fw, bu, bw, ru, rw, pu, pw = (F3(n) for _ in range(8))
    Mu = ti.Matrix.field(3, 3, ti.f64, shape=n)      # inverse translational block of R
    Mw = ti.Matrix.field(3, 3, ti.f64, shape=n)      # inverse rotational block
    scal = ti.field(ti.f64, shape=8)
    nb = ti.field(ti.i32, shape=(n, maxn))
    nbc = ti.field(ti.i32, shape=n)
    nb2 = ti.field(ti.i32, shape=(n, maxn))
    nbc2 = ti.field(ti.i32, shape=n)
    over = ti.field(ti.i32, shape=())
    pd, pn, xi, xi2 = F3((n, maxn)), F3((n, maxn)), F3((n, maxn)), F3((n, maxn))
    ph = ti.field(ti.f64, shape=(n, maxn))
    pcn = ti.field(ti.f64, shape=(n, maxn))
    pct = ti.field(ti.f64, shape=(n, maxn))
    pkn = ti.field(ti.f64, shape=(n, maxn))          # k_n dt of a contact (implicit spring)
    pkt = ti.field(ti.f64, shape=(n, maxn))          # k_t dt
    pfc = F3((n, maxn))
    sig = ti.field(ti.f64, shape=8)
    par = ti.field(ti.f64, shape=16)
    # the cell grid of the neighbour search: particles sorted by cell
    # (counting sort), a cell at least as wide as the longest pair kept
    use_cells = nc >= 3
    ncell = nc * nc * nc if use_cells else 1
    ccount = ti.field(ti.i32, shape=ncell)
    cstart = ti.field(ti.i32, shape=ncell)
    cfill = ti.field(ti.i32, shape=ncell)
    corder = ti.field(ti.i32, shape=n)
    cellof = ti.field(ti.i32, shape=n)
    # par: 0 L, 1 Lees-Edwards offset, 2 roughness delta, 3 h_min, 4 (unused),
    #      5 k_n, 6 k_t, 7 mu_f, 8 pair range / mean radius, 9 skin, 10 van der Waals range,
    #      12 adhesion W / (mu gd a0)

    @ti.func
    def sep(i, j):
        L = par[0]
        d = x[j] - x[i]
        ky = ti.round(d[1] / L)
        d[1] -= ky * L
        d[0] -= ky * par[1]
        d[0] -= ti.round(d[0] / L) * L
        d[2] -= ti.round(d[2] / L) * L
        return d

    @ti.kernel
    def find_pairs():
        """New neighbour lists in nb2, carrying the tangential springs of
        contacts that persist into xi2."""
        over[None] = 0
        for i in range(n):
            c = 0
            for j in range(i + 1, n):
                d = sep(i, j)
                h = d.norm() - rad[i] - rad[j]
                if h < par[8] * 0.5 * (rad[i] + rad[j]) + par[9]:
                    if c < maxn:
                        nb2[i, c] = j
                        old = vec(0.0, 0.0, 0.0)
                        for m in range(nbc[i]):
                            if nb[i, m] == j:
                                old = xi[i, m]
                        xi2[i, c] = old
                        c += 1
                    else:
                        over[None] = 1
            nbc2[i] = c

    @ti.func
    def cell_index(v, cs):
        L = par[0]
        c = int(ti.floor((v - ti.floor(v / L) * L) / cs))
        return ti.min(ti.max(c, 0), nc - 1)

    @ti.kernel
    def bin_count():
        for c in range(ncell):
            ccount[c] = 0
            cfill[c] = 0
        for i in range(n):
            cs = par[0] / nc
            c = (cell_index(x[i][0], cs) * nc + cell_index(x[i][1], cs)) * nc + cell_index(x[i][2], cs)
            cellof[i] = c
            ti.atomic_add(ccount[c], 1)

    @ti.kernel
    def bin_scan():
        acc = 0
        ti.loop_config(serialize=True)
        for c in range(ncell):
            cstart[c] = acc
            acc += ccount[c]

    @ti.kernel
    def bin_fill():
        for i in range(n):
            c = cellof[i]
            k = ti.atomic_add(cfill[c], 1)
            corder[cstart[c] + k] = i

    @ti.kernel
    def find_pairs_cells():
        """find_pairs on the cell grid: the 27 cells around a particle; across
        the sheared faces (y) the cells are those under the image shifted by
        the Lees-Edwards offset. Lists in ascending index, as find_pairs."""
        over[None] = 0
        for i in range(n):
            cs = par[0] / nc
            p = x[i]
            cy0 = cell_index(p[1], cs)
            cz0 = cell_index(p[2], cs)
            c = 0
            for a in range(3):
                jy = cy0 + a - 1
                xs = p[0]
                if jy < 0:
                    jy += nc
                    xs += par[1]
                elif jy >= nc:
                    jy -= nc
                    xs -= par[1]
                cx0 = cell_index(xs, cs)
                for b in range(3):
                    jx = (cx0 + b - 1 + nc) % nc
                    for e in range(3):
                        jz = (cz0 + e - 1 + nc) % nc
                        cc = (jx * nc + jy) * nc + jz
                        for k in range(cstart[cc], cstart[cc] + ccount[cc]):
                            j = corder[k]
                            if j > i:
                                h = sep(i, j).norm() - rad[i] - rad[j]
                                if h < par[8] * 0.5 * (rad[i] + rad[j]) + par[9]:
                                    if c < maxn:
                                        nb2[i, c] = j
                                        c += 1
                                    else:
                                        over[None] = 1
            for q0 in range(1, c):                   # insertion sort: ascending j
                v = nb2[i, q0]
                q = q0
                while q > 0:
                    if nb2[i, q - 1] > v:
                        nb2[i, q] = nb2[i, q - 1]
                        q -= 1
                    else:
                        break
                nb2[i, q] = v
            for m2 in range(c):
                j = nb2[i, m2]
                old = vec(0.0, 0.0, 0.0)
                for m in range(nbc[i]):
                    if nb[i, m] == j:
                        old = xi[i, m]
                xi2[i, m2] = old
            nbc2[i] = c

    @ti.kernel
    def swap_pairs():
        for i in range(n):
            nbc[i] = nbc2[i]
            for m in range(nbc2[i]):
                nb[i, m] = nb2[i, m]
                xi[i, m] = xi2[i, m]

    @ti.kernel
    def geometry(dt: ti.f64):
        for i in range(n):
            for m in range(nbc[i]):
                j = nb[i, m]
                d = sep(i, j)
                r = d.norm()
                ai, aj = rad[i], rad[j]
                h = r - ai - aj
                rs = ai * aj / (ai + aj)
                hc = LUB_CUT * 0.5 * (ai + aj)
                he = ti.max(h, 0.0) + par[2]
                cn = 0.0
                ct = 0.0
                if he < hc:
                    cn = 6.0 * math.pi * rs * rs * (1.0 / he - 1.0 / hc)
                    ct = 6.0 * math.pi * (4.0 / 15.0) * ai * aj * (2.0 * ai * ai + ai * aj + 2.0 * aj * aj) \
                        / (ai + aj) ** 3 * ti.log(hc / he)
                pd[i, m] = d
                pn[i, m] = d / r
                ph[i, m] = h
                pcn[i, m] = cn
                pct[i, m] = ct
                kn_ = 0.0
                kt_ = 0.0
                if h < 0.0:
                    kn_ = par[5] * dt
                    if par[7] > 0.0:
                        kt_ = par[6] * dt
                pkn[i, m] = kn_
                pkt[i, m] = kt_
                if h >= 0.0:
                    xi[i, m] = vec(0.0, 0.0, 0.0)               # the contact opened: the spring is gone

    @ti.func
    def relv(i, m, j, Ui, Wi, Uj, Wj):
        """Normal and tangential relative velocity of the two surfaces."""
        nv = pn[i, m]
        vs = (Uj - Ui) - (rad[i] * Wi + rad[j] * Wj).cross(nv)
        vn = nv.dot(vs)
        return vn, vs - vn * nv

    @ti.func
    def lub(i, m, j, Ui, Wi, Uj, Wj):
        """Force on i (j gets minus it) linear in the motions: lubrication and
        the step's increment of the contact springs."""
        vn, vt = relv(i, m, j, Ui, Wi, Uj, Wj)
        return (pcn[i, m] + pkn[i, m]) * vn * pn[i, m] + (pct[i, m] + pkt[i, m]) * vt

    @ti.func
    def lub_only(i, m, j, Ui, Wi, Uj, Wj):
        vn, vt = relv(i, m, j, Ui, Wi, Uj, Wj)
        return pcn[i, m] * vn * pn[i, m] + pct[i, m] * vt

    @ti.func
    def spring_inc(i, m, j, Ui, Wi, Uj, Wj):
        vn, vt = relv(i, m, j, Ui, Wi, Uj, Wj)
        return pkn[i, m] * vn * pn[i, m] + pkt[i, m] * vt

    @ti.kernel
    def apply_R(su: ti.template(), sw: ti.template()):
        """fu, fw = R . (su, sw)."""
        for i in range(n):
            a = rad[i]
            fu[i] = 6.0 * math.pi * a * su[i]
            fw[i] = 8.0 * math.pi * a * a * a * sw[i]
        for i in range(n):
            for m in range(nbc[i]):
              if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                j = nb[i, m]
                f = lub(i, m, j, su[i], sw[i], su[j], sw[j])
                nv = pn[i, m]
                fu[i] -= f
                fu[j] += f
                fw[i] -= (rad[i] * nv).cross(f)
                fw[j] -= (rad[j] * nv).cross(f)

    @ti.kernel
    def precond():
        """The 3x3 diagonal blocks of R (drag plus the pair terms), inverted."""
        I3 = ti.Matrix.identity(ti.f64, 3)
        for i in range(n):
            a = rad[i]
            Mu[i] = 6.0 * math.pi * a * I3
            Mw[i] = 8.0 * math.pi * a * a * a * I3
        for i in range(n):
            for m in range(nbc[i]):
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    j = nb[i, m]
                    nv = pn[i, m]
                    nn = nv.outer_product(nv)
                    bt = (pcn[i, m] + pkn[i, m]) * nn + (pct[i, m] + pkt[i, m]) * (I3 - nn)
                    br = (pct[i, m] + pkt[i, m]) * (I3 - nn)
                    ti.atomic_add(Mu[i], bt)
                    ti.atomic_add(Mu[j], bt)
                    ti.atomic_add(Mw[i], rad[i] * rad[i] * br)
                    ti.atomic_add(Mw[j], rad[j] * rad[j] * br)
        for i in range(n):
            Mu[i] = Mu[i].inverse()
            Mw[i] = Mw[i].inverse()

    @ti.kernel
    def rhs_forces():
        """bu, bw: the forces at zero motion relative to the shear - the
        lubrication of the imposed relative motion, contacts with friction
        and van der Waals; pfc keeps the non-hydrodynamic pair force."""
        for i in range(n):
            bu[i] = vec(0.0, 0.0, 0.0)
            bw[i] = vec(0.0, 0.0, 0.0)
        W0 = vec(0.0, 0.0, -0.5)
        Z = vec(0.0, 0.0, 0.0)
        for i in range(n):
            for m in range(nbc[i]):
              if True:
                j = nb[i, m]
                nv = pn[i, m]
                d = pd[i, m]
                h = ph[i, m]
                f = Z
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    f = lub(i, m, j, Z, W0, vec(d[1], 0.0, 0.0), W0)
                fc = Z
                aij = sqa[i] * sqa[j]
                if aij > 0.0 and h < par[10]:
                    hv = ti.max(h, par[3])
                    fc += aij * rad[i] * rad[j] / (rad[i] + rad[j]) / (6.0 * hv * hv) * nv
                ft = Z
                if par[12] > 0.0 and h < par[3]:
                    # adhesion of touching surfaces (DMT pull-off 2 pi W R*), held
                    # down to the closest approach so the pair can snap together
                    fc += 2.0 * math.pi * par[12] * rad[i] * rad[j] / (rad[i] + rad[j]) * nv
                if h < 0.0:
                    fc += par[5] * h * nv                       # the normal spring pushes i away from j
                    if par[7] > 0.0:
                        s = xi[i, m] - xi[i, m].dot(nv) * nv
                        ft = par[6] * s
                        fmax = par[7] * par[5] * (-h)
                        fl = ft.norm()
                        if fl > fmax:
                            ft *= fmax / fl
                            s = ft / par[6]                      # sliding: the spring stays at the cap
                        xi[i, m] = s
                bu[i] += f + fc + ft
                bu[j] -= f + fc + ft
                bw[i] += (rad[i] * nv).cross(f + ft)
                bw[j] += (rad[j] * nv).cross(f + ft)
                pfc[i, m] = fc + ft

    @ti.kernel
    def cg_start():
        """r = b - R u (fu holds R u), p = z = M^-1 r; scal: 0 r.z, 1 b.b, 2 r.r."""
        scal[0] = 0.0
        scal[1] = 0.0
        scal[2] = 0.0
        for i in range(n):
            ru[i] = bu[i] - fu[i]
            rw[i] = bw[i] - fw[i]
            pu[i] = Mu[i] @ ru[i]
            pw[i] = Mw[i] @ rw[i]
            scal[0] += ru[i].dot(pu[i]) + rw[i].dot(pw[i])
            scal[1] += bu[i].dot(bu[i]) + bw[i].dot(bw[i])
            scal[2] += ru[i].dot(ru[i]) + rw[i].dot(rw[i])

    @ti.kernel
    def cg_a():
        """fu = R p (drag + pairs), then alpha = r.z / p.R p into scal[5]."""
        for i in range(n):
            a = rad[i]
            fu[i] = 6.0 * math.pi * a * pu[i]
            fw[i] = 8.0 * math.pi * a * a * a * pw[i]
        for i in range(n):
            for m in range(nbc[i]):
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    j = nb[i, m]
                    f = lub(i, m, j, pu[i], pw[i], pu[j], pw[j])
                    nv = pn[i, m]
                    fu[i] -= f
                    fu[j] += f
                    fw[i] -= (rad[i] * nv).cross(f)
                    fw[j] -= (rad[j] * nv).cross(f)
        scal[3] = 0.0
        for i in range(n):
            scal[3] += pu[i].dot(fu[i]) + pw[i].dot(fw[i])
        scal[5] = ti.select(scal[3] > 0.0, scal[0] / ti.max(scal[3], 1e-300), 0.0)

    @ti.kernel
    def cg_b():
        """u += alpha p, r -= alpha R p; new r.z gives beta; p = z + beta p."""
        al = scal[5]
        scal[2] = 0.0
        scal[6] = 0.0
        for i in range(n):
            u[i] += al * pu[i]
            w[i] += al * pw[i]
            ru[i] -= al * fu[i]
            rw[i] -= al * fw[i]
            scal[2] += ru[i].dot(ru[i]) + rw[i].dot(rw[i])
            scal[6] += ru[i].dot(Mu[i] @ ru[i]) + rw[i].dot(Mw[i] @ rw[i])
        beta = ti.select(scal[0] > 0.0, scal[6] / ti.max(scal[0], 1e-300), 0.0)
        scal[0] = scal[6]
        for i in range(n):
            pu[i] = Mu[i] @ ru[i] + beta * pu[i]
            pw[i] = Mw[i] @ rw[i] + beta * pw[i]

    @ti.kernel
    def set_offset(v: ti.f64):
        par[1] = v

    @ti.kernel
    def umax() -> ti.f64:
        mx = 0.0
        for i in range(n):
            ti.atomic_max(mx, u[i].norm())
        return mx

    @ti.kernel
    def move(dt: ti.f64):
        """Tangential springs grow with the relative surface velocity of the
        contact; positions move with the shear flow (Lees-Edwards wrap)."""
        L = par[0]
        W0 = vec(0.0, 0.0, -0.5)
        for i in range(n):
            for m in range(nbc[i]):
              if ph[i, m] < 0.0 and par[7] > 0.0:
                j = nb[i, m]
                nv = pn[i, m]
                d = pd[i, m]
                dU = (u[j] - u[i]) + vec(d[1], 0.0, 0.0)
                vs = dU - (rad[i] * (w[i] + W0) + rad[j] * (w[j] + W0)).cross(nv)
                xi[i, m] += (vs - vs.dot(nv) * nv) * dt
        for i in range(n):
            p = x[i]
            p[0] += (u[i][0] + (p[1] - 0.5 * L)) * dt
            p[1] += u[i][1] * dt
            p[2] += u[i][2] * dt
            if p[1] >= L:
                p[1] -= L
                p[0] -= par[1]
            elif p[1] < 0.0:
                p[1] += L
                p[0] += par[1]
            p[0] -= ti.floor(p[0] / L) * L
            p[2] -= ti.floor(p[2] / L) * L
            x[i] = p

    @ti.kernel
    def stress():
        """sig: 0 lubrication, 1 contact + friction + van der Waals (shear
        stress times V), 2 contacts, 3 largest overlap, 4 umax."""
        for s in range(8):
            sig[s] = 0.0
        W0 = vec(0.0, 0.0, -0.5)
        for i in range(n):
            for m in range(nbc[i]):
              if True:
                j = nb[i, m]
                d = pd[i, m]
                fl = vec(0.0, 0.0, 0.0)
                fk = vec(0.0, 0.0, 0.0)
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    Ui, Wi, Uj, Wj = u[i], w[i] + W0, u[j] + vec(d[1], 0.0, 0.0), w[j] + W0
                    fl = lub_only(i, m, j, Ui, Wi, Uj, Wj)
                    fk = spring_inc(i, m, j, Ui, Wi, Uj, Wj)
                fc = pfc[i, m] + fk
                sig[0] += 0.5 * (d[0] * fl[1] + d[1] * fl[0])
                sig[1] += 0.5 * (d[0] * fc[1] + d[1] * fc[0])
                if ph[i, m] < 0.0:
                    sig[2] += 1.0
                    ti.atomic_max(sig[3], -ph[i, m])
        for i in range(n):
            ti.atomic_max(sig[4], u[i].norm())

    @ti.kernel
    def push_apart(rate: ti.f64):
        """Packing: overlapping spheres are pushed apart (no flow)."""
        for i in range(n):
            for m in range(nbc[i]):
              if True:
                j = nb[i, m]
                d = sep(i, j)
                r = d.norm()
                h = r - rad[i] - rad[j]
                if h < 0.0:
                    mv = 0.5 * rate * h * d / r
                    ti.atomic_add(x[i], mv)
                    ti.atomic_add(x[j], -mv)
        L = par[0]
        for i in range(n):
            p = x[i]
            for c in ti.static(range(3)):
                p[c] -= ti.floor(p[c] / L) * L
            x[i] = p

    @ti.kernel
    def max_overlap() -> ti.f64:
        mo = 0.0
        for i in range(n):
            for m in range(nbc[i]):
              if True:
                j = nb[i, m]
                h = sep(i, j).norm() - rad[i] - rad[j]
                ti.atomic_max(mo, -h)
        return mo

    return dict(locals())


class Suspension:
    """A periodic box of spheres sheared at unit rate in the xy plane.

    radii: the sphere radii (any unit; scaled to the largest = 1).
    phi: volume fraction. delta: roughness, hmin: closest approach of the
    surfaces for van der Waals, hamaker: A / (mu gd a0^3) (one number, or one
    per particle - a pair takes sqrt(A_i A_j)), vdw_range: where the
    attraction is cut (all in units of the largest radius). mu_f: friction
    coefficient (0 frictionless). kn: contact stiffness (mu gd a0)."""

    def __init__(self, radii, phi, seed=1, backend="auto", log=None, delta=1e-3, hmin=None, hamaker=0.0,
                 vdw_range=0.1, mu_f=0.0, kn=1.0e4, kt=None, maxn=None, adhesion=0.0):
        self.log = log or (lambda *a: None)
        self.backend = init_backend(backend, self.log)
        r = np.asarray(radii, float)
        r = r / r.max()
        self.n = n = len(r)
        self.L = ((4.0 / 3.0) * math.pi * np.sum(r ** 3) / phi) ** (1.0 / 3.0)
        self.phi = phi
        self.r = r
        self.nbod = n                    # every sphere is a particle of its own
        self.intrinsic = 2.5             # Einstein
        self.rc_off = 0.0                # no member off its particle's centre
        # pairs are kept to a surface gap of `gap` mean radii of the pair plus SKIN
        gap = max(LUB_CUT, 2.0 * vdw_range)
        if maxn is None:
            # the largest sphere (radius 1) packed round with the smallest
            # (rmin): the centres within (1 + rmin)(1 + gap/2) + SKIN at most at
            # the densest packing, 0.74, with half again for the random start
            rmin = float(r.min())
            shell = (((1.0 + 0.5 * gap) * (1.0 + rmin) + SKIN) ** 3 - 1.0) / rmin ** 3
            maxn = int(min(4096, max(48, 1.5 * 0.74 * shell)))
        self.maxn = maxn
        # cells at least as wide as the longest pair kept, 2 + gap + SKIN (largest radius 1)
        self.nc = int(self.L // (2.0 + gap + SKIN))
        self.memory_bytes = n * maxn * 168 + n * 700 + 12 * self.nc ** 3
        self.K = K = _kernels(n, maxn, self.nc)
        K["rad"].from_numpy(r)
        rng = np.random.default_rng(seed)
        K["x"].from_numpy(rng.random((n, 3)) * self.L)
        hk = np.broadcast_to(np.asarray(hamaker, float), (n,))
        K["sqa"].from_numpy(np.sqrt(np.maximum(hk, 0.0)))
        self._params(delta, hmin, hk, adhesion, kn, kt, mu_f, vdw_range, gap)
        self.strain = 0.0

    def _params(self, delta, hmin, hk, adhesion, kn, kt, mu_f, vdw_range, gap):
        """The pair-law parameters (par) of the kernels."""
        self.par = np.zeros(16)
        self.par[0] = self.L
        self.par[2] = delta
        self.par[3] = hmin if hmin is not None else delta
        # the strongest attraction a contact has to hold: van der Waals at the
        # closest approach and adhesion, for the largest pair (R* <= 1/2)
        hm = hmin if hmin is not None else delta
        f_att = float(np.max(hk)) * 0.5 / (6.0 * hm * hm) + 2.0 * math.pi * float(adhesion) * 0.5
        # 1000x the strongest attraction: the overlap of a contact held by it and
        # pressed by the shear stays near 1 % of the largest radius (100x left
        # 6 % with AlN at 1 um); the implicit springs keep any stiffness stable
        kn = max(float(kn), KN_FACTOR * f_att)
        self.par[5] = kn
        self.par[6] = kt if kt is not None else 0.5 * kn
        # the springs are implicit in R (k dt joins the lubrication), so their
        # stiffness does not limit the step
        self.par[7] = mu_f
        self.par[10] = vdw_range
        self.par[12] = adhesion
        self.par[8] = gap
        self.par[9] = SKIN
        self.K["par"].from_numpy(self.par)

    def _push_par(self):
        self.K["par"].from_numpy(self.par)

    def positions(self):
        """Sphere centres (box units; the members of the rigid particles)."""
        return self.K["x"].to_numpy()

    def speeds(self):
        """Speed of every sphere relative to the shear flow."""
        return np.linalg.norm(self.K["u"].to_numpy(), axis=1)

    def _placed(self):
        """After a move: nothing to do for free spheres."""

    def neighbours(self):
        K = self.K
        if K["use_cells"]:
            K["bin_count"]()
            K["bin_scan"]()
            K["bin_fill"]()
            K["find_pairs_cells"]()
        else:
            K["find_pairs"]()
        if K["over"][None]:
            raise NeighbourOverflow(f"more than {self.maxn} neighbours for a particle")
        K["swap_pairs"]()

    def pack(self, tol=1e-4, max_iter=200000):
        """Random non-overlapping start: random centres, overlaps pushed apart
        until the largest is below tol (radii units). Returns the iterations."""
        K = self.K
        mo = 0.0
        for it in range(max_iter):
            if it % 20 == 0:
                self.neighbours()
                mo = K["max_overlap"]()
                if mo < tol:
                    return it
            K["push_apart"](1.0)
        raise RuntimeError(f"the packing did not relax (largest overlap {mo:.3g})")

    # tol and the step were checked on 200 spheres at phi 0.5 (strain 0-0.5):
    # tol 1e-5 gives eta as 1e-7 to 4 digits; a step of 2e-3 radii (against
    # 5e-4) changes it by 0.2 %, four times faster
    def solve(self, tol=1e-5, maxit=5000, check=8):
        """Preconditioned CG on R . V = F (motions relative to the shear),
        warm-started from the last motion. The residual is read back every
        `check` iterations (each read waits for the device)."""
        K = self.K
        K["rhs_forces"]()
        K["precond"]()
        K["apply_R"](K["u"], K["w"])
        K["cg_start"]()
        sc = K["scal"]
        bnorm = math.sqrt(sc[1]) or 1.0
        if math.sqrt(sc[2]) <= tol * bnorm:
            return 0
        it = 0
        while it < maxit:
            for _ in range(check):
                K["cg_a"]()
                K["cg_b"]()
            it += check
            if math.sqrt(sc[2]) <= tol * bnorm:
                break
        return it

    def run(self, strain, every=0.05, progress=None, should_stop=None, max_disp=2e-3, dt_max=5e-3,
            frames=0.0, frame_every=0.02, live=None):
        """Shears the box to the given total strain. Returns the time series
        of the relative viscosity and its parts; with frames > 0 also the
        positions and speeds (relative to the shear) over that last strain."""
        K = self.K
        V = self.L ** 3
        # rigid particles: a member up to rc_off from its centre moves with the
        # particle, not with the shear at its own place - a pair of members
        # closes at up to 2.6 + 3 rc_off per unit strain instead of 2.6, so the
        # strain step shrinks by as much
        reach = 2.6 + 3.0 * self.rc_off
        dt_max = dt_max * 2.6 / reach
        out = {"strain": [], "eta": [], "eta_lub": [], "eta_contact": [], "contacts_per_particle": [],
               "max_overlap": [], "cg_iterations": []}
        step = 0
        next_rec = self.strain
        t0 = time.time()
        umax = 1.0
        # the neighbour lists hold pairs up to 0.05 mean radii beyond the
        # lubrication range: rebuilt before any pair could have closed half
        # of that - by the shear (relative speed <= gap distance <= 2.6 radii)
        # or by the motion relative to it
        moved = 1e9
        dt_prev = 1e-4
        f_from = strain - frames if frames > 0 else math.inf
        next_frame = f_from
        out["frames"], out["frames_speed"] = [], []
        while self.strain < strain:
            if moved > 0.025:
                self.neighbours()
                moved = 0.0
            K["geometry"](dt_prev)
            its = self.solve()
            umax = max(K["umax"](), 1e-6)
            if self.strain >= next_frame:
                out["frames"].append(self.positions().astype(np.float32))
                out["frames_speed"].append(self.speeds().astype(np.float32))
                next_frame += frame_every
            if self.strain >= next_rec - 1e-12:
                K["stress"]()
                s = K["sig"].to_numpy()
                lub, con = s[0] / V, s[1] / V
                eta = 1.0 + self.intrinsic * self.phi + lub + con
                out["strain"].append(self.strain)
                out["eta"].append(eta)
                out["eta_lub"].append(lub)
                out["eta_contact"].append(con)
                out["contacts_per_particle"].append(2.0 * s[2] / self.nbod)
                out["max_overlap"].append(float(s[3]))
                out["cg_iterations"].append(its)
                next_rec += every
                # the contacts stiffen until no surface overlaps more than 2 % of
                # the largest radius: on AlN at 1 um, 6 % overlap read 96 where
                # 1 % read 129 and 0.08 % 128 (strain 0.1-0.5)
                if s[3] > 0.02:
                    self.par[5] *= 1.5
                    self.par[6] *= 1.5
                    self._push_par()
                if progress:
                    progress(self.strain, eta, its)
                if live is not None and live.due():
                    live(self, eta)
            # the step: no particle moves more than max_disp relative to the
            # shear (the affine part is exact)
            dt = min(dt_max, max_disp / umax)
            if dt > 1.5 * dt_prev:
                dt = 1.5 * dt_prev                           # the implicit springs assumed dt_prev
            dt_prev = dt
            # the offset of the step's end first: a particle crossing the
            # sheared faces lands where the image boxes are then (with the
            # offset before, it landed L dt off, up to 4 % of a radius)
            self.strain += dt
            self.par[1] = (self.strain * self.L) % self.L
            K["set_offset"](self.par[1])
            K["move"](dt)
            self._placed()
            moved += dt * (reach + 2.0 * umax)
            step += 1
            if should_stop is not None and step % 50 == 0 and should_stop():
                raise InterruptedError
        out["seconds"] = time.time() - t0
        out["steps"] = step
        out["frame_strain"] = frame_every
        return out


def _rigid_kernels(n, nbod, maxn, nc=0):
    """Taichi fields and kernels for rigid particles made of overlapping
    spheres (multi-sphere clumps): n member spheres in nbod bodies. The pair
    laws act between member spheres of different bodies, as between spheres
    in _kernels; the unknowns are the velocity and spin of every body, and a
    member moves with its body, U + Omega x r. A force on a member acts at its
    contact point c (from the body centre), so it turns the body by c x F.
    One-member bodies at their centre are the spheres of _kernels."""
    import taichi as ti
    vec = ti.types.vector(3, ti.f64)
    F3 = lambda shape: ti.Vector.field(3, ti.f64, shape=shape)        # noqa: E731
    M3 = lambda shape: ti.Matrix.field(3, 3, ti.f64, shape=shape)     # noqa: E731

    # members
    x, rr, q0 = F3(n), F3(n), F3(n)                  # position, offset from the body centre (world, body frame)
    rad = ti.field(ti.f64, shape=n)
    sqa = ti.field(ti.f64, shape=n)
    bod = ti.field(ti.i32, shape=n)
    msp = ti.field(ti.f64, shape=n)                  # speed relative to the shear (viewer)
    # bodies
    X = F3(nbod)
    Rot = M3(nbod)                                   # body frame to world: r = Rot q0
    u, w = F3(nbod), F3(nbod)
    fu, fw, bu, bw, ru, rw, pu, pw = (F3(nbod) for _ in range(8))
    Mu, Mw = M3(nbod), M3(nbod)
    dU = ti.field(ti.f64, shape=nbod)                # drag 6 pi a_eq, 8 pi a_eq^3
    dW = ti.field(ti.f64, shape=nbod)
    scal = ti.field(ti.f64, shape=8)
    nb = ti.field(ti.i32, shape=(n, maxn))
    nbc = ti.field(ti.i32, shape=n)
    nb2 = ti.field(ti.i32, shape=(n, maxn))
    nbc2 = ti.field(ti.i32, shape=n)
    over = ti.field(ti.i32, shape=())
    pd, pn, xi, xi2 = F3((n, maxn)), F3((n, maxn)), F3((n, maxn)), F3((n, maxn))
    ph = ti.field(ti.f64, shape=(n, maxn))
    pcn = ti.field(ti.f64, shape=(n, maxn))
    pct = ti.field(ti.f64, shape=(n, maxn))
    pkn = ti.field(ti.f64, shape=(n, maxn))
    pkt = ti.field(ti.f64, shape=(n, maxn))
    pfc = F3((n, maxn))
    sig = ti.field(ti.f64, shape=8)
    par = ti.field(ti.f64, shape=16)
    use_cells = nc >= 3
    ncell = nc * nc * nc if use_cells else 1
    ccount = ti.field(ti.i32, shape=ncell)
    cstart = ti.field(ti.i32, shape=ncell)
    cfill = ti.field(ti.i32, shape=ncell)
    corder = ti.field(ti.i32, shape=n)
    cellof = ti.field(ti.i32, shape=n)

    @ti.func
    def sep(i, j):
        L = par[0]
        d = x[j] - x[i]
        ky = ti.round(d[1] / L)
        d[1] -= ky * L
        d[0] -= ky * par[1]
        d[0] -= ti.round(d[0] / L) * L
        d[2] -= ti.round(d[2] / L) * L
        return d

    @ti.func
    def cross_mat(c):
        return ti.Matrix([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]], ti.f64)

    @ti.func
    def wrap(p):
        L = par[0]
        if p[1] >= L:
            p[1] -= L
            p[0] -= par[1]
        elif p[1] < 0.0:
            p[1] += L
            p[0] += par[1]
        p[0] -= ti.floor(p[0] / L) * L
        p[2] -= ti.floor(p[2] / L) * L
        return p

    @ti.kernel
    def members():
        """Member positions and offsets from the bodies."""
        for i in range(n):
            b = bod[i]
            r_ = Rot[b] @ q0[i]
            rr[i] = r_
            x[i] = wrap(X[b] + r_)

    @ti.kernel
    def find_pairs():
        over[None] = 0
        for i in range(n):
            c = 0
            for j in range(i + 1, n):
                if bod[j] != bod[i]:
                    d = sep(i, j)
                    h = d.norm() - rad[i] - rad[j]
                    if h < par[8] * 0.5 * (rad[i] + rad[j]):
                        if c < maxn:
                            nb2[i, c] = j
                            old = vec(0.0, 0.0, 0.0)
                            for m in range(nbc[i]):
                                if nb[i, m] == j:
                                    old = xi[i, m]
                            xi2[i, c] = old
                            c += 1
                        else:
                            over[None] = 1
            nbc2[i] = c

    @ti.func
    def cell_index(v, cs):
        L = par[0]
        c = int(ti.floor((v - ti.floor(v / L) * L) / cs))
        return ti.min(ti.max(c, 0), nc - 1)

    @ti.kernel
    def bin_count():
        for c in range(ncell):
            ccount[c] = 0
            cfill[c] = 0
        for i in range(n):
            cs = par[0] / nc
            c = (cell_index(x[i][0], cs) * nc + cell_index(x[i][1], cs)) * nc + cell_index(x[i][2], cs)
            cellof[i] = c
            ti.atomic_add(ccount[c], 1)

    @ti.kernel
    def bin_scan():
        acc = 0
        ti.loop_config(serialize=True)
        for c in range(ncell):
            cstart[c] = acc
            acc += ccount[c]

    @ti.kernel
    def bin_fill():
        for i in range(n):
            c = cellof[i]
            k = ti.atomic_add(cfill[c], 1)
            corder[cstart[c] + k] = i

    @ti.kernel
    def find_pairs_cells():
        over[None] = 0
        for i in range(n):
            cs = par[0] / nc
            p = x[i]
            cy0 = cell_index(p[1], cs)
            cz0 = cell_index(p[2], cs)
            c = 0
            for a in range(3):
                jy = cy0 + a - 1
                xs = p[0]
                if jy < 0:
                    jy += nc
                    xs += par[1]
                elif jy >= nc:
                    jy -= nc
                    xs -= par[1]
                cx0 = cell_index(xs, cs)
                for b in range(3):
                    jx = (cx0 + b - 1 + nc) % nc
                    for e in range(3):
                        jz = (cz0 + e - 1 + nc) % nc
                        cc = (jx * nc + jy) * nc + jz
                        for k in range(cstart[cc], cstart[cc] + ccount[cc]):
                            j = corder[k]
                            if j > i and bod[j] != bod[i]:
                                h = sep(i, j).norm() - rad[i] - rad[j]
                                if h < par[8] * 0.5 * (rad[i] + rad[j]):
                                    if c < maxn:
                                        nb2[i, c] = j
                                        c += 1
                                    else:
                                        over[None] = 1
            for q1 in range(1, c):
                v = nb2[i, q1]
                q = q1
                while q > 0:
                    if nb2[i, q - 1] > v:
                        nb2[i, q] = nb2[i, q - 1]
                        q -= 1
                    else:
                        break
                nb2[i, q] = v
            for m2 in range(c):
                j = nb2[i, m2]
                old = vec(0.0, 0.0, 0.0)
                for m in range(nbc[i]):
                    if nb[i, m] == j:
                        old = xi[i, m]
                xi2[i, m2] = old
            nbc2[i] = c

    @ti.kernel
    def swap_pairs():
        for i in range(n):
            nbc[i] = nbc2[i]
            for m in range(nbc2[i]):
                nb[i, m] = nb2[i, m]
                xi[i, m] = xi2[i, m]

    @ti.kernel
    def geometry(dt: ti.f64):
        for i in range(n):
            for m in range(nbc[i]):
                j = nb[i, m]
                d = sep(i, j)
                r = d.norm()
                ai, aj = rad[i], rad[j]
                h = r - ai - aj
                hc = LUB_CUT * 0.5 * (ai + aj)
                he = ti.max(h, 0.0) + par[2]
                rs = ai * aj / (ai + aj)
                cn = 0.0
                ct = 0.0
                if he < hc:
                    cn = 6.0 * math.pi * rs * rs * (1.0 / he - 1.0 / hc)
                    ct = 6.0 * math.pi * (4.0 / 15.0) * ai * aj * (2.0 * ai * ai + ai * aj + 2.0 * aj * aj) \
                        / (ai + aj) ** 3 * ti.log(hc / he)
                pd[i, m] = d
                pn[i, m] = d / r
                ph[i, m] = h
                pcn[i, m] = cn
                pct[i, m] = ct
                kn_ = 0.0
                kt_ = 0.0
                if h < 0.0:
                    kn_ = par[5] * dt
                    if par[7] > 0.0:
                        kt_ = par[6] * dt
                pkn[i, m] = kn_
                pkt[i, m] = kt_
                if h >= 0.0:
                    xi[i, m] = vec(0.0, 0.0, 0.0)

    @ti.func
    def relv(i, m, j, Ui, Wi, Uj, Wj):
        """Normal and tangential relative velocity of the two member surfaces
        (Ui, Uj: the member centres' velocities; Wi, Wj: the bodies' spins)."""
        nv = pn[i, m]
        vs = (Uj - Ui) - (rad[i] * Wi + rad[j] * Wj).cross(nv)
        vn = nv.dot(vs)
        return vn, vs - vn * nv

    @ti.func
    def lub(i, m, j, Ui, Wi, Uj, Wj):
        vn, vt = relv(i, m, j, Ui, Wi, Uj, Wj)
        return (pcn[i, m] + pkn[i, m]) * vn * pn[i, m] + (pct[i, m] + pkt[i, m]) * vt

    @ti.func
    def lub_only(i, m, j, Ui, Wi, Uj, Wj):
        vn, vt = relv(i, m, j, Ui, Wi, Uj, Wj)
        return pcn[i, m] * vn * pn[i, m] + pct[i, m] * vt

    @ti.func
    def spring_inc(i, m, j, Ui, Wi, Uj, Wj):
        vn, vt = relv(i, m, j, Ui, Wi, Uj, Wj)
        return pkn[i, m] * vn * pn[i, m] + pkt[i, m] * vt

    @ti.kernel
    def apply_R(su: ti.template(), sw: ti.template()):
        """fu, fw = R . (su, sw) (body motions relative to the shear); pu, pw
        hold a copy of them."""
        for b in range(nbod):
            pu[b] = su[b]
            pw[b] = sw[b]
            fu[b] = dU[b] * su[b]
            fw[b] = dW[b] * sw[b]
        for i in range(n):
            for m in range(nbc[i]):
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    j = nb[i, m]
                    bi, bj = bod[i], bod[j]
                    f = lub(i, m, j, pu[bi] + pw[bi].cross(rr[i]), pw[bi], pu[bj] + pw[bj].cross(rr[j]), pw[bj])
                    nv = pn[i, m]
                    fu[bi] -= f
                    fu[bj] += f
                    fw[bi] -= (rr[i] + rad[i] * nv).cross(f)
                    fw[bj] += (rr[j] - rad[j] * nv).cross(f)

    @ti.kernel
    def precond():
        """3x3 translational and rotational blocks of each body's R,
        inverted: a pair's block C at contact point c turns the body by
        -[c]x C [c]x."""
        I3 = ti.Matrix.identity(ti.f64, 3)
        for b in range(nbod):
            Mu[b] = dU[b] * I3
            Mw[b] = dW[b] * I3
        for i in range(n):
            for m in range(nbc[i]):
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    j = nb[i, m]
                    nv = pn[i, m]
                    nn = nv.outer_product(nv)
                    C = (pcn[i, m] + pkn[i, m]) * nn + (pct[i, m] + pkt[i, m]) * (I3 - nn)
                    Ci = cross_mat(rr[i] + rad[i] * nv)
                    Cj = cross_mat(rr[j] - rad[j] * nv)
                    ti.atomic_add(Mu[bod[i]], C)
                    ti.atomic_add(Mu[bod[j]], C)
                    ti.atomic_add(Mw[bod[i]], -(Ci @ C @ Ci))
                    ti.atomic_add(Mw[bod[j]], -(Cj @ C @ Cj))
        for b in range(nbod):
            Mu[b] = Mu[b].inverse()
            Mw[b] = Mw[b].inverse()

    @ti.kernel
    def rhs_forces():
        """bu, bw: forces and torques on the bodies at zero motion relative to
        the shear - each body carried with the flow at its centre and turned
        with its vorticity; pfc keeps the non-hydrodynamic pair force."""
        for b in range(nbod):
            bu[b] = vec(0.0, 0.0, 0.0)
            bw[b] = vec(0.0, 0.0, 0.0)
        W0 = vec(0.0, 0.0, -0.5)
        Z = vec(0.0, 0.0, 0.0)
        for i in range(n):
            for m in range(nbc[i]):
                j = nb[i, m]
                bi, bj = bod[i], bod[j]
                nv = pn[i, m]
                D = pd[i, m] - rr[j] + rr[i]                 # body centre to body centre
                h = ph[i, m]
                f = Z
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    f = lub(i, m, j, W0.cross(rr[i]), W0, vec(D[1], 0.0, 0.0) + W0.cross(rr[j]), W0)
                fc = Z
                aij = sqa[i] * sqa[j]
                if aij > 0.0 and h < par[10]:
                    hv = ti.max(h, par[3])
                    fc += aij * rad[i] * rad[j] / (rad[i] + rad[j]) / (6.0 * hv * hv) * nv
                ft = Z
                if par[12] > 0.0 and h < par[3]:
                    fc += 2.0 * math.pi * par[12] * rad[i] * rad[j] / (rad[i] + rad[j]) * nv
                if h < 0.0:
                    fc += par[5] * h * nv
                    if par[7] > 0.0:
                        s = xi[i, m] - xi[i, m].dot(nv) * nv
                        ft = par[6] * s
                        fmax = par[7] * par[5] * (-h)
                        fl = ft.norm()
                        if fl > fmax:
                            ft *= fmax / fl
                            s = ft / par[6]
                        xi[i, m] = s
                F = f + fc + ft
                bu[bi] += F
                bu[bj] -= F
                bw[bi] += (rr[i] + rad[i] * nv).cross(F)
                bw[bj] -= (rr[j] - rad[j] * nv).cross(F)
                pfc[i, m] = fc + ft

    @ti.kernel
    def cg_start():
        scal[0] = 0.0
        scal[1] = 0.0
        scal[2] = 0.0
        for b in range(nbod):
            ru[b] = bu[b] - fu[b]
            rw[b] = bw[b] - fw[b]
            pu[b] = Mu[b] @ ru[b]
            pw[b] = Mw[b] @ rw[b]
            scal[0] += ru[b].dot(pu[b]) + rw[b].dot(pw[b])
            scal[1] += bu[b].dot(bu[b]) + bw[b].dot(bw[b])
            scal[2] += ru[b].dot(ru[b]) + rw[b].dot(rw[b])

    @ti.kernel
    def cg_a():
        for b in range(nbod):
            fu[b] = dU[b] * pu[b]
            fw[b] = dW[b] * pw[b]
        for i in range(n):
            for m in range(nbc[i]):
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    j = nb[i, m]
                    bi, bj = bod[i], bod[j]
                    f = lub(i, m, j, pu[bi] + pw[bi].cross(rr[i]), pw[bi], pu[bj] + pw[bj].cross(rr[j]), pw[bj])
                    nv = pn[i, m]
                    fu[bi] -= f
                    fu[bj] += f
                    fw[bi] -= (rr[i] + rad[i] * nv).cross(f)
                    fw[bj] += (rr[j] - rad[j] * nv).cross(f)
        scal[3] = 0.0
        for b in range(nbod):
            scal[3] += pu[b].dot(fu[b]) + pw[b].dot(fw[b])
        scal[5] = ti.select(scal[3] > 0.0, scal[0] / ti.max(scal[3], 1e-300), 0.0)

    @ti.kernel
    def cg_b():
        al = scal[5]
        scal[2] = 0.0
        scal[6] = 0.0
        for b in range(nbod):
            u[b] += al * pu[b]
            w[b] += al * pw[b]
            ru[b] -= al * fu[b]
            rw[b] -= al * fw[b]
            scal[2] += ru[b].dot(ru[b]) + rw[b].dot(rw[b])
            scal[6] += ru[b].dot(Mu[b] @ ru[b]) + rw[b].dot(Mw[b] @ rw[b])
        beta = ti.select(scal[0] > 0.0, scal[6] / ti.max(scal[0], 1e-300), 0.0)
        scal[0] = scal[6]
        for b in range(nbod):
            pu[b] = Mu[b] @ ru[b] + beta * pu[b]
            pw[b] = Mw[b] @ rw[b] + beta * pw[b]

    @ti.kernel
    def set_offset(v: ti.f64):
        par[1] = v

    @ti.kernel
    def umax() -> ti.f64:
        """The fastest member relative to the shear (its body's motion)."""
        mx = 0.0
        for i in range(n):
            b = bod[i]
            s = (u[b] + w[b].cross(rr[i])).norm()
            msp[i] = s
            ti.atomic_max(mx, s)
        return mx

    @ti.kernel
    def move(dt: ti.f64):
        """Tangential springs grow with the relative surface velocity; the
        bodies move with the shear at their centre and their own motion, and
        turn with the vorticity and their own spin (exact rotation, kept
        orthonormal). members() then places the member spheres."""
        L = par[0]
        W0 = vec(0.0, 0.0, -0.5)
        for i in range(n):
            for m in range(nbc[i]):
                if ph[i, m] < 0.0 and par[7] > 0.0:
                    j = nb[i, m]
                    bi, bj = bod[i], bod[j]
                    nv = pn[i, m]
                    D = pd[i, m] - rr[j] + rr[i]
                    Oi = w[bi] + W0
                    Oj = w[bj] + W0
                    dV = (u[bj] + vec(D[1], 0.0, 0.0) + Oj.cross(rr[j])) - (u[bi] + Oi.cross(rr[i]))
                    vs = dV - (rad[i] * Oi + rad[j] * Oj).cross(nv)
                    xi[i, m] += (vs - vs.dot(nv) * nv) * dt
        for b in range(nbod):
            p = X[b]
            p[0] += (u[b][0] + (p[1] - 0.5 * L)) * dt
            p[1] += u[b][1] * dt
            p[2] += u[b][2] * dt
            X[b] = wrap(p)
            om = (w[b] + W0) * dt
            th = om.norm()
            if th > 1e-14:
                K = cross_mat(om / th)
                R = ti.Matrix.identity(ti.f64, 3) + ti.sin(th) * K + (1.0 - ti.cos(th)) * (K @ K)
                Q = R @ Rot[b]
                # Gram-Schmidt on the columns: the round-off of many small turns
                c0 = vec(Q[0, 0], Q[1, 0], Q[2, 0]).normalized()
                c1 = vec(Q[0, 1], Q[1, 1], Q[2, 1])
                c1 = (c1 - c1.dot(c0) * c0).normalized()
                c2 = c0.cross(c1)
                Rot[b] = ti.Matrix.cols([c0, c1, c2])

    @ti.kernel
    def stress():
        """sig: 0 lubrication, 1 contact + friction + van der Waals (shear
        stress times V, from the body-centre separations), 2 member contacts,
        3 largest overlap, 4 umax."""
        for s in range(8):
            sig[s] = 0.0
        W0 = vec(0.0, 0.0, -0.5)
        for i in range(n):
            for m in range(nbc[i]):
                j = nb[i, m]
                bi, bj = bod[i], bod[j]
                D = pd[i, m] - rr[j] + rr[i]
                fl = vec(0.0, 0.0, 0.0)
                fk = vec(0.0, 0.0, 0.0)
                if pcn[i, m] > 0.0 or pkn[i, m] > 0.0:
                    Oi = w[bi] + W0
                    Oj = w[bj] + W0
                    Ui = u[bi] + Oi.cross(rr[i])
                    Uj = u[bj] + vec(D[1], 0.0, 0.0) + Oj.cross(rr[j])
                    fl = lub_only(i, m, j, Ui, Oi, Uj, Oj)
                    fk = spring_inc(i, m, j, Ui, Oi, Uj, Oj)
                fc = pfc[i, m] + fk
                sig[0] += 0.5 * (D[0] * fl[1] + D[1] * fl[0])
                sig[1] += 0.5 * (D[0] * fc[1] + D[1] * fc[0])
                if ph[i, m] < 0.0:
                    sig[2] += 1.0
                    ti.atomic_max(sig[3], -ph[i, m])
        for b in range(nbod):
            ti.atomic_max(sig[4], u[b].norm())

    @ti.kernel
    def max_overlap() -> ti.f64:
        mo = 0.0
        for i in range(n):
            for m in range(nbc[i]):
                j = nb[i, m]
                h = sep(i, j).norm() - rad[i] - rad[j]
                ti.atomic_max(mo, -h)
        return mo

    @ti.kernel
    def dissipation() -> ti.f64:
        """Power dissipated by the lubrication and the drag at the current
        solution (for the self-test: it equals the shear stress times the
        rate when only lubrication acts)."""
        W0 = vec(0.0, 0.0, -0.5)
        e = 0.0
        for b in range(nbod):
            e += dU[b] * u[b].dot(u[b]) + dW[b] * w[b].dot(w[b])
        for i in range(n):
            for m in range(nbc[i]):
                if pcn[i, m] > 0.0:
                    j = nb[i, m]
                    bi, bj = bod[i], bod[j]
                    D = pd[i, m] - rr[j] + rr[i]
                    Oi = w[bi] + W0
                    Oj = w[bj] + W0
                    vn, vt = relv(i, m, j, u[bi] + Oi.cross(rr[i]), Oi,
                                  u[bj] + vec(D[1], 0.0, 0.0) + Oj.cross(rr[j]), Oj)
                    e += pcn[i, m] * vn * vn + pct[i, m] * vt.dot(vt)
        return e

    return dict(locals())


class RigidSuspension(Suspension):
    """Rigid particles of any shape in the sheared box of Suspension, each a
    clump of overlapping spheres - the multi-sphere method of DEM (Favier,
    Abbaspour-Fard, Kremmer and Raji, Eng. Comput. 16 (1999) 467;
    Kruggel-Emden, Rickelt, Wirtz and Scherer, Powder Technol. 188 (2008)
    153). Every pair law of Suspension (lubrication, contacts with friction,
    van der Waals, adhesion) acts between member spheres of different
    particles; the particles move and turn as rigid bodies: R . V = F is
    solved for every body's velocity and spin, a member moving with
    U + Omega x r. Each body has the Stokes drag of its volume-equivalent
    sphere, and its own stresslet in the shear enters the viscosity as
    [eta] phi with the [eta] of its shape.

    Lengths in units of the largest member radius. mq (n, 3), mr (n,): the
    members' offsets in their body's frame and their radii; mb (n,): the
    body of every member; X (nbod, 3), Rot (nbod, 3, 3): the packed bodies'
    centres and body-to-world rotations in a box of side L; a_eq (nbod,):
    volume-equivalent radii; hamaker: per member."""

    def __init__(self, mq, mr, mb, X, Rot, a_eq, L, phi, intrinsic=2.5, backend="auto", log=None, delta=1e-3,
                 hmin=None, hamaker=0.0, vdw_range=0.1, mu_f=0.0, kn=1.0e4, kt=None, maxn=None, adhesion=0.0):
        self.log = log or (lambda *a: None)
        self.backend = init_backend(backend, self.log)
        mq = np.asarray(mq, float)
        mr = np.asarray(mr, float)
        mb = np.asarray(mb, np.int32)
        X = np.mod(np.asarray(X, float), float(L))
        Rot = np.asarray(Rot, float)
        self.n = n = len(mr)
        self.nbod = nbod = len(X)
        self.L = float(L)
        self.phi = phi
        self.r = mr
        self.mb = mb
        self.intrinsic = float(intrinsic)
        self.rc_off = float(np.max(np.linalg.norm(mq, axis=1))) if n else 0.0
        gap = max(LUB_CUT, 2.0 * vdw_range)
        if maxn is None:
            pos = X[mb] + np.einsum("nij,nj->ni", Rot[mb], mq)
            maxn = int(min(4096, max(32, 1.5 * _neighbour_count(pos, mr, mb, self.L, gap) + 16)))
        self.maxn = maxn
        self.nc = int(self.L // (2.0 + gap + SKIN))
        self.memory_bytes = n * maxn * 168 + n * 120 + nbod * 700 + 12 * self.nc ** 3
        self.K = K = _rigid_kernels(n, nbod, maxn, self.nc)
        K["rad"].from_numpy(mr)
        K["q0"].from_numpy(mq)
        K["bod"].from_numpy(mb)
        K["X"].from_numpy(X)
        K["Rot"].from_numpy(Rot)
        a_eq = np.asarray(a_eq, float)
        K["dU"].from_numpy(6.0 * math.pi * a_eq)
        K["dW"].from_numpy(8.0 * math.pi * a_eq ** 3)
        hk = np.broadcast_to(np.asarray(hamaker, float), (n,))
        K["sqa"].from_numpy(np.sqrt(np.maximum(hk, 0.0)))
        self._params(delta, hmin, hk, adhesion, kn, kt, mu_f, vdw_range, gap)
        K["members"]()
        self.strain = 0.0

    def pack(self, tol=1e-4, max_iter=0):
        """The bodies come packed (pack_clumps): the neighbour lists only,
        and the largest overlap left, which the contacts then clear."""
        self.neighbours()
        mo = self.K["max_overlap"]()
        if mo > 0.02:
            self.log(f"    the packed particles overlap by up to {mo:.3f} member radii; the contacts push them apart")
        return 0

    def _placed(self):
        self.K["members"]()

    def speeds(self):
        return self.K["msp"].to_numpy()


def _neighbour_count(pos, r, body, L, gap):
    """The longest neighbour list - members of other bodies with a higher
    index within reach - of member spheres at pos in a periodic box of side
    L (the lists are sized from it before the kernels are built)."""
    import itertools
    from scipy.spatial import cKDTree
    pos = np.mod(np.asarray(pos, float), L)
    pos[pos >= L] = 0.0
    tree = cKDTree(pos, boxsize=L)
    reach = 2.0 * float(r.max()) * (1.0 + 0.5 * gap) + SKIN
    best = 0
    for a in range(0, len(r), 20000):
        idx = np.arange(a, min(len(r), a + 20000))
        lists = tree.query_ball_point(pos[idx], reach)
        lens = np.fromiter((len(v) for v in lists), np.int64, len(idx))
        if not lens.sum():
            continue
        j = np.fromiter(itertools.chain.from_iterable(lists), np.int64, int(lens.sum()))
        i = np.repeat(idx, lens)
        k = (j > i) & (body[j] != body[i])
        i, j = i[k], j[k]
        d = pos[j] - pos[i]
        d -= L * np.round(d / L)
        h = np.linalg.norm(d, axis=1) - r[i] - r[j]
        i = i[h < gap * 0.5 * (r[i] + r[j]) + SKIN]
        if len(i):
            best = max(best, int(np.bincount(i - a).max()))
    return best


def pack_clumps(bodies, phi, bound_m=0.0, seed=1, log=None, should_stop=None):
    """The rigid particles of `bodies` packed without overlap at phi, for
    RigidSuspension (lengths in units of the largest member radius a0, which
    is returned in metres). bodies: 'templates' [(centres, radii) in m at
    scale 1] and per body its template 'tpl', 'scale', rotation 'rot' (rows:
    the body axes in the world), 'rotmode' (-1 sphere, 0 free, 1 axis kept,
    2 axis in the xy plane), 'circ' circumradius at scale 1 (m), true 'vol'
    (m3) and 'area' (m2), 'phase'. A bound layer bound_m thickens every
    member, and the content by area x thickness. Packed by the structure
    generator's rigid-clump packing (generate.clump_pack) - or, when bodies
    carries 'X' (centres, m) and 'box_m', taken as they are: the particles of
    a generated RVE, already apart (phi is then their own content)."""
    from .. import generate as GEN
    b = float(bound_m)
    tpl = np.asarray(bodies["tpl"], np.int64)
    s = np.asarray(bodies["scale"], float)
    vol = np.asarray(bodies["vol"], float)
    area = np.asarray(bodies["area"], float)
    tq = [np.asarray(q, float).reshape(-1, 3) for q, _ in bodies["templates"]]
    tr = [np.asarray(r_, float).ravel() for _, r_ in bodies["templates"]]
    tmax = np.array([r_.max() for r_ in tr])
    a0 = float(np.max(tmax[tpl] * s)) + b
    given = bodies.get("X") is not None
    if given:
        phi = float(vol.sum()) / float(bodies["box_m"]) ** 3
    L = (float(vol.sum()) / float(phi)) ** (1.0 / 3.0) / a0
    phi_eff = float(phi) * float(np.sum(vol + area * b) / vol.sum())
    templates = [(q / a0, r_ / a0) for q, r_ in zip(tq, tr)]
    nbod = len(tpl)
    rng = np.random.default_rng(seed)
    C = rng.random((nbod, 3)) * L
    Rm = np.ascontiguousarray(np.asarray(bodies["rot"], float))
    ph = np.asarray(bodies["phase"], np.int64)
    t0 = time.time()
    if given:
        C, g = np.asarray(bodies["X"], float) / a0, 1.0
    else:
        C, Rm, g = GEN.clump_pack(float(L), C, Rm, s.copy(), tpl, templates, np.full(nbod, 2.0 * b / a0), ph,
                                  np.zeros(int(ph.max()) + 1, np.int64), np.asarray(bodies["rotmode"], np.int64),
                                  np.asarray(bodies["circ"], float) / a0, log=log, should_stop=should_stop,
                                  recover=True)
    if g < 0.995:
        raise RuntimeError(f"these particles do not pack without overlap at {100 * phi_eff:.1f} vol% (the packing "
                           f"stopped at {100 * phi_eff * g ** 3:.1f} vol%): the content is at or above their maximum "
                           f"packing")
    counts = np.array([len(tr[t]) for t in tpl])
    mb = np.repeat(np.arange(nbod), counts).astype(np.int32)
    mq = np.concatenate([templates[t][0] * sc for t, sc in zip(tpl, s)])
    mr = np.concatenate([templates[t][1] * sc for t, sc in zip(tpl, s)]) + b / a0
    a_eq = (3.0 * (vol + area * b) / (4.0 * math.pi)) ** (1.0 / 3.0) / a0
    if log:
        log(f"    {nbod} particles as {len(mr)} member spheres, "
            + ("as placed in the RVE" if given else f"packed in {time.time() - t0:.1f} s")
            + f" (box {L * a0 * 1e6:.4g} µm)")
    return {"mq": mq, "mr": mr, "mb": mb, "X": np.mod(C, L), "Rot": np.ascontiguousarray(np.transpose(Rm, (0, 2, 1))),
            "a_eq": a_eq, "L": L, "a0": a0, "phi_eff": phi_eff, "pack_scale": float(g)}


# --------------------------------------------------------------- physical units
H_PLANCK = 6.62607015e-34
KB = 1.380649e-23
NU_E = 3.0e15            # main electronic absorption frequency (UV), Israelachvili ch. 13
# visible refractive index n_D of the library materials (CRC Handbook,
# refractiveindex.info); None: conducting (metals, carbon) or unknown
N_OPT = {"epoxy": 1.57, "bt": 1.60, "polyimide": 1.70, "pbo": 1.68, "bcb": 1.56, "su8": 1.59, "silicone": 1.40,
         "ppe": 1.57, "lcp": 1.64, "ptfe": 1.35, "peek": 1.65, "parylene_c": 1.64,
         "sio2": 1.458, "quartz": 1.544, "sicoh": 1.40, "glass_bsg": 1.47, "eglass": 1.55, "al2o3": 1.76,
         "aln": 2.15, "hbn": 1.80, "si3n4": 2.02, "mgo": 1.736, "zno": 2.00, "beo": 1.72, "sic": 2.65,
         "diamond": 2.42, "tio2": 2.60, "batio3": 2.40, "nizn_ferrite": 2.30}
A_CONDUCTOR = 1.0e-19    # metal or carbon particles across a polymer (J), Israelachvili table 13.3 order


def hamaker(filler_id, resin_id, eps_f=None, eps_r=None, sigma_f=0.0, T=298.15):
    """Hamaker constant of two filler particles across the resin (J): the
    Lifshitz theory in the Tabor-Winterton form (Israelachvili, Intermolecular
    and Surface Forces, eq. 13.16) from the refractive indices and the static
    permittivities; conductors take A_CONDUCTOR. None when the optical data
    are missing."""
    if sigma_f and sigma_f > 1.0:
        return A_CONDUCTOR
    n1, n3 = N_OPT.get(filler_id), N_OPT.get(resin_id)
    if n1 is None or n3 is None:
        return None
    e1 = float(eps_f) if eps_f else n1 * n1
    e3 = float(eps_r) if eps_r else n3 * n3
    zero = 0.75 * KB * T * ((e1 - e3) / (e1 + e3)) ** 2
    disp = 3.0 * H_PLANCK * NU_E / (16.0 * math.sqrt(2.0)) * (n1 * n1 - n3 * n3) ** 2 / (n1 * n1 + n3 * n3) ** 1.5
    return zero + disp


class _LiveDem:
    """The particle dynamics' frames for the live view (mpsim/live.Slot):
    positions in um inside the box and speeds relative to the shear; above
    40 000 spheres a slab across z, chosen at the start."""

    def __init__(self, slot, a0):
        self.slot, self.a0, self.keep = slot, a0, None

    def due(self):
        return self.slot.due()

    def start(self, s):
        r = s.r * self.a0 * 1e6
        L = s.L * self.a0 * 1e6
        self.keep = None
        if s.n > 40000:
            z = np.mod(s.positions()[:, 2] * self.a0 * 1e6, L)
            t = L * 40000.0 / s.n
            self.keep = np.abs(z - 0.5 * L) < 0.5 * t
        info = getattr(self.slot, "static_info", None) or {}
        ph = info.get("phase")
        if ph is not None and getattr(s, "mb", None) is not None and len(ph) == s.nbod:
            ph = np.asarray(ph)[s.mb]                     # the phase of each member sphere
        ph = np.zeros(s.n, np.uint8) if ph is None or len(ph) != s.n else np.asarray(ph, np.uint8)
        keep = self.keep if self.keep is not None else slice(None)
        self.slot.particles_static(r[keep], ph[keep], L, info.get("names") or ["Filler"],
                                   info.get("colors") or ["#4c9be8"], shown_of=s.n)

    def __call__(self, s, eta):
        L = s.L * self.a0 * 1e6
        x = np.mod(s.positions() * self.a0 * 1e6, L)
        v = s.speeds()
        if self.keep is not None:
            x, v = x[self.keep], v[self.keep]
        self.slot.particles(x, v, s.strain, eta)


def shear_viscosity(radii_m, phi, mu_resin, gd, roughness_m=5e-9, hmin_m=1e-9, hamaker_J=0.0, mu_f=0.5,
                    strain=5.0, settle=1.0, seed=1, backend="auto", log=None, progress=None, should_stop=None,
                    bound_m=0.0, adhesion_J_m2=0.0, frames=None, live=None, bodies=None):
    """See _shear_viscosity. live: a mpsim.live.Slot for the live view (its
    static_info may give the phase names, colours and the phase of every
    sphere: {"names", "colors", "phase"}). On the GPU first; a run that fails there (the
    driver, the GPU's memory) is redone on the CPU, and the process stays
    on the CPU. out['gpu_fallback'] says why, when it happened."""
    kw = dict(roughness_m=roughness_m, hmin_m=hmin_m, hamaker_J=hamaker_J, mu_f=mu_f, strain=strain, settle=settle,
              seed=seed, log=log, progress=progress, should_stop=should_stop, bound_m=bound_m,
              adhesion_J_m2=adhesion_J_m2, frames=frames, live=live, bodies=bodies)
    try:
        out = _shear_viscosity(radii_m, phi, mu_resin, gd, backend=backend, **kw)
    except InterruptedError:
        raise
    except Exception as e:                                     # noqa: BLE001
        if _TI["arch"] != "cuda":
            raise
        if log:
            log(f"    the run on the GPU failed ({str(e).splitlines()[0][:200] if str(e) else type(e).__name__}); "
                f"it is redone on the CPU")
        fall_back_to_cpu(e, log)
        out = _shear_viscosity(radii_m, phi, mu_resin, gd, backend="auto", **kw)
    out["gpu_fallback"] = _TI["cuda_failed"]
    return out


def _shear_viscosity(radii_m, phi, mu_resin, gd, roughness_m=5e-9, hmin_m=1e-9, hamaker_J=0.0, mu_f=0.5,
                     strain=5.0, settle=1.0, seed=1, backend="auto", log=None, progress=None, should_stop=None,
                     bound_m=0.0, adhesion_J_m2=0.0, frames=None, live=None, bodies=None):
    """Relative viscosity of spheres of the given radii (m) at volume fraction
    phi in a resin of viscosity mu_resin (Pa s) sheared at gd (1/s), from the
    particle dynamics. The surface roughness and the closest approach are
    lengths, so their weight against the particle - and the van der Waals
    attraction against the viscous force, A / (mu gd a^3) - changes with the
    particle size. Averaged over the strain after `settle`.

    bound_m: a resin layer bound to the surface that moves with the particle
    (adsorbed resin, coupling agent): the radius for the flow and the
    contacts grows by it, and the volume fraction with it - by (1 + b/a)^3,
    which a fine filler feels most. adhesion_J_m2: work of adhesion W of two
    touching surfaces across the resin (hydrogen bonding of untreated silica
    and the like), a pull-off force 2 pi W R*; against the viscous force it
    scales as W / (mu gd a).

    bodies: particles that are not all spheres (see pack_clumps; radii_m are
    then their volume-equivalent radii, hamaker_J one per particle or one
    for all). They are rigid clumps of spheres (RigidSuspension), with the
    intrinsic viscosity bodies['intrinsic'] for their own stresslet."""
    r0 = np.asarray(radii_m, float)
    rigid = None
    if bodies is None:
        r = r0 + float(bound_m)
        phi = float(phi) * float(np.sum(r ** 3) / np.sum(r0 ** 3))
        a0 = float(r.max())
    else:
        rigid = pack_clumps(bodies, float(phi), bound_m=float(bound_m), seed=seed, log=log, should_stop=should_stop)
        phi, a0 = rigid["phi_eff"], rigid["a0"]
        r = rigid["mr"] * a0
    maxn = None
    lv = _LiveDem(live, a0) if live is not None else None
    while True:
        # the fields of an earlier run are freed first (Taichi keeps every
        # field until it is reset)
        release()
        hk = np.asarray(hamaker_J, float) / (mu_resin * gd * a0 ** 3)
        common = dict(backend=backend, log=log, delta=roughness_m / a0, hmin=max(hmin_m, 1e-12) / a0, mu_f=mu_f,
                      adhesion=float(adhesion_J_m2) / (mu_resin * gd * a0), maxn=maxn)
        if rigid is None:
            s = Suspension(r / a0, phi, seed=seed, hamaker=hk, **common)
        else:
            hk_m = np.broadcast_to(hk, (len(rigid["X"]),))[rigid["mb"]]
            s = RigidSuspension(rigid["mq"], rigid["mr"], rigid["mb"], rigid["X"], rigid["Rot"], rigid["a_eq"],
                                rigid["L"], phi, intrinsic=float(bodies.get("intrinsic", 2.5)), hamaker=hk_m, **common)
        if log and (len(r) >= 20000 or rigid is not None):
            log(f"    {len(r)} spheres{f' in {s.nbod} rigid particles' if rigid is not None else ''}, {s.maxn} "
                f"neighbour slots each, cell grid {s.nc}³: about {s.memory_bytes / 2 ** 30:.2f} GB on the "
                f"{'GPU' if s.backend == 'cuda' else 'CPU'}")
        try:
            s.pack()
            # frames of the last strain for the viewer (frames=0: none)
            if lv is not None:
                lv.start(s)
            out = s.run(strain, every=0.02, progress=progress, should_stop=should_stop,
                        frames=min(1.0, strain / 2) if frames is None else frames, live=lv)
            break
        except NeighbourOverflow:
            if s.maxn >= 4096:
                raise
            maxn = min(4096, 2 * s.maxn)
            if log:
                log(f"    a particle had more than {s.maxn} neighbours: restarted with {maxn}")
    st, e = np.asarray(out["strain"]), np.asarray(out["eta"])
    # a strain no longer than the settling one (1 is accepted): its second half
    if float(strain) <= settle:
        settle = 0.5 * float(strain)
    m = st >= settle
    blocks = [b.mean() for b in np.array_split(e[m], 8) if len(b)]
    # the standard error: contact networks build and break over a strain of
    # order one, so the blocks of one run scatter less than whole runs from
    # other packings do. The largest of the 8 blocks, batches of one strain
    # and the measured scatter between runs (RUN_SCATTER) is taken.
    eta_mean = float(e[m].mean())
    span = max(float(st[-1]) - settle, 0.1)
    se = float(np.std(blocks) / math.sqrt(len(blocks)))
    nb = int(span // 1.0)
    if nb >= 3:
        batches = [e[(st >= settle + k) & (st < settle + k + 1.0)].mean() for k in range(nb)]
        se = max(se, float(np.std(batches, ddof=1) / math.sqrt(nb)))
    se = max(se, RUN_SCATTER / math.sqrt(s.nbod * span) * eta_mean)
    if out["frames"]:
        out["frames_um"] = np.stack(out.pop("frames")) * (a0 * 1e6)
        out["frames_speed"] = np.stack(out["frames_speed"])
    else:
        out.pop("frames")
        out["frames_um"] = None
    # the spheres the viewer draws (the members of rigid particles) and the
    # particle each belongs to
    out["anim_radii_m"] = np.asarray(r, float)
    out["anim_body"] = np.asarray(s.mb if rigid is not None else np.arange(len(r)), np.int64)
    out.update(eta_mean=eta_mean, eta_se=se,
               n=int(s.nbod), members=int(len(r)), rigid=rigid is not None, intrinsic=float(s.intrinsic),
               pack_scale=(rigid or {}).get("pack_scale"), box_um=float(s.L * a0 * 1e6), backend=s.backend,
               delta_rel=float(roughness_m / a0), phi_effective=float(phi),
               adhesion_rel=float(adhesion_J_m2) / (mu_resin * gd * a0),
               hamaker_rel=float(np.max(np.asarray(hamaker_J, float)) / (mu_resin * gd * a0 ** 3)))
    return out
