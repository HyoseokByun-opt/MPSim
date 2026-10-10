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
# arch: the running Taichi backend; prefer: what it was started for;
# cuda_failed: why a run on the GPU failed (the process then stays on the CPU)
_TI = {"arch": None, "prefer": None, "cuda_failed": None}


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
            if log:
                if got == "cuda":
                    log("    particle dynamics on the GPU (CUDA)")
                else:
                    reason = ("as set" if not want_gpu else
                              f"the GPU run failed: {_TI['cuda_failed']}" if _TI["cuda_failed"] else why or "")
                    log(f"    particle dynamics on the CPU{f' ({reason})' if reason else ''}")
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
    #      5 k_n, 6 k_t, 7 mu_f, 8 neighbour gap / mean radius, 10 van der Waals range,
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
                                if h < par[8] * 0.5 * (rad[i] + rad[j]):
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
        # pairs are kept to a surface gap of `gap` mean radii of the pair
        gap = max(LUB_CUT, 2.0 * vdw_range) + 0.05
        if maxn is None:
            # the largest sphere (radius 1) packed round with the smallest
            # (rmin): the centres within (1 + rmin)(1 + gap/2) at most at the
            # densest packing, 0.74, with half again for the random start
            rmin = float(r.min())
            shell = (((1.0 + 0.5 * gap) * (1.0 + rmin)) ** 3 - 1.0) / rmin ** 3
            maxn = int(min(4096, max(48, 1.5 * 0.74 * shell)))
        self.maxn = maxn
        # cells at least as wide as the longest pair kept, 2 + gap (largest radius 1)
        self.nc = int(self.L // (2.0 + gap))
        self.memory_bytes = n * maxn * 168 + n * 700 + 12 * self.nc ** 3
        self.K = K = _kernels(n, maxn, self.nc)
        K["rad"].from_numpy(r)
        rng = np.random.default_rng(seed)
        K["x"].from_numpy(rng.random((n, 3)) * self.L)
        hk = np.broadcast_to(np.asarray(hamaker, float), (n,))
        K["sqa"].from_numpy(np.sqrt(np.maximum(hk, 0.0)))
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
        K["par"].from_numpy(self.par)
        self.strain = 0.0

    def _push_par(self):
        self.K["par"].from_numpy(self.par)

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
            frames=0.0, frame_every=0.02):
        """Shears the box to the given total strain. Returns the time series
        of the relative viscosity and its parts; with frames > 0 also the
        positions and speeds (relative to the shear) over that last strain."""
        K = self.K
        V = self.L ** 3
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
                out["frames"].append(K["x"].to_numpy().astype(np.float32))
                out["frames_speed"].append(np.linalg.norm(K["u"].to_numpy(), axis=1).astype(np.float32))
                next_frame += frame_every
            if self.strain >= next_rec - 1e-12:
                K["stress"]()
                s = K["sig"].to_numpy()
                lub, con = s[0] / V, s[1] / V
                eta = 1.0 + 2.5 * self.phi + lub + con
                out["strain"].append(self.strain)
                out["eta"].append(eta)
                out["eta_lub"].append(lub)
                out["eta_contact"].append(con)
                out["contacts_per_particle"].append(2.0 * s[2] / self.n)
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
            # the step: no particle moves more than max_disp relative to the
            # shear (the affine part is exact)
            dt = min(dt_max, max_disp / umax)
            if dt > 1.5 * dt_prev:
                dt = 1.5 * dt_prev                           # the implicit springs assumed dt_prev
            dt_prev = dt
            K["move"](dt)
            moved += dt * (2.6 + 2.0 * umax)
            self.strain += dt
            self.par[1] = (self.strain * self.L) % self.L
            K["set_offset"](self.par[1])
            step += 1
            if should_stop is not None and step % 50 == 0 and should_stop():
                raise InterruptedError
        out["seconds"] = time.time() - t0
        out["steps"] = step
        out["frame_strain"] = frame_every
        return out


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


def shear_viscosity(radii_m, phi, mu_resin, gd, roughness_m=5e-9, hmin_m=1e-9, hamaker_J=0.0, mu_f=0.5,
                    strain=5.0, settle=1.0, seed=1, backend="auto", log=None, progress=None, should_stop=None,
                    bound_m=0.0, adhesion_J_m2=0.0, frames=None):
    """See _shear_viscosity. On the GPU first; a run that fails there (the
    driver, the GPU's memory) is redone on the CPU, and the process stays
    on the CPU. out['gpu_fallback'] says why, when it happened."""
    kw = dict(roughness_m=roughness_m, hmin_m=hmin_m, hamaker_J=hamaker_J, mu_f=mu_f, strain=strain, settle=settle,
              seed=seed, log=log, progress=progress, should_stop=should_stop, bound_m=bound_m,
              adhesion_J_m2=adhesion_J_m2, frames=frames)
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
                     bound_m=0.0, adhesion_J_m2=0.0, frames=None):
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
    scales as W / (mu gd a)."""
    r0 = np.asarray(radii_m, float)
    r = r0 + float(bound_m)
    phi = float(phi) * float(np.sum(r ** 3) / np.sum(r0 ** 3))
    a0 = float(r.max())
    maxn = None
    while True:
        # the fields of an earlier run are freed first (Taichi keeps every
        # field until it is reset)
        release()
        s = Suspension(r / a0, phi, seed=seed, backend=backend, log=log, delta=roughness_m / a0,
                       hmin=max(hmin_m, 1e-12) / a0, hamaker=np.asarray(hamaker_J, float) / (mu_resin * gd * a0 ** 3),
                       mu_f=mu_f, adhesion=float(adhesion_J_m2) / (mu_resin * gd * a0), maxn=maxn)
        if log and len(r) >= 20000:
            log(f"    {len(r)} spheres, {s.maxn} neighbour slots each, cell grid {s.nc}³: about "
                f"{s.memory_bytes / 2 ** 30:.2f} GB on the {'GPU' if s.backend == 'cuda' else 'CPU'}")
        try:
            s.pack()
            # frames of the last strain for the viewer (frames=0: none)
            out = s.run(strain, every=0.02, progress=progress, should_stop=should_stop,
                        frames=min(1.0, strain / 2) if frames is None else frames)
            break
        except NeighbourOverflow:
            if s.maxn >= 4096:
                raise
            maxn = min(4096, 2 * s.maxn)
            if log:
                log(f"    a particle had more than {s.maxn} neighbours: restarted with {maxn}")
    st, e = np.asarray(out["strain"]), np.asarray(out["eta"])
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
    se = max(se, RUN_SCATTER / math.sqrt(len(r) * span) * eta_mean)
    if out["frames"]:
        out["frames_um"] = np.stack(out.pop("frames")) * (a0 * 1e6)
        out["frames_speed"] = np.stack(out["frames_speed"])
    else:
        out.pop("frames")
        out["frames_um"] = None
    out.update(eta_mean=eta_mean, eta_se=se,
               n=int(len(r)), box_um=float(s.L * a0 * 1e6), backend=s.backend,
               delta_rel=float(roughness_m / a0), phi_effective=float(phi),
               adhesion_rel=float(adhesion_J_m2) / (mu_resin * gd * a0),
               hamaker_rel=float(np.max(np.asarray(hamaker_J, float)) / (mu_resin * gd * a0 ** 3)))
    return out
