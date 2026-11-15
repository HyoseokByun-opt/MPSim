"""Compression of a dense suspension of spheres with lubrication films of their real width and frictional contacts
(5.10.0): the particle dynamics of solvers/suspension.py in a box that is shortened along z.

Why this exists
---------------
The compression of solvers/compress.py solves the resin's flow on the voxels: a film between two close spheres is
one voxel wide whatever its real width, overlaps after a step are pushed apart, and nothing acts where two surfaces
slide on each other. At loadings above about 40 vol% its stress therefore depends on the voxel, and it has no
friction at all. Here the same spheres are pressed as rigid bodies with the pair laws of the viscosity analysis:

* zero Reynolds number: every sphere is force- and torque-free; each step solves R . V = F for the velocities and
  spins relative to the imposed flow, a pure strain E = diag(e_x, e_y, -1) x rate (sides free: 1/2, 1/2; x free
  and y held: 1, 0) - the box keeps its volume, as the resin is incompressible and there are no pores;
* R: Stokes drag (6 pi mu a, 8 pi mu a^3) and, for pairs closer than LUB_CUT mean radii, the squeeze resistance
  6 pi mu R*^2 / h and the log(1/h) shear resistance (Jeffrey and Onishi 1984), shifted to zero at the cut-off; h
  is the real gap, regularised by the surface roughness;
* contacts (h < 0): a stiff normal spring and a tangential spring capped by Coulomb friction mu_f (Cundall and
  Strack 1979) with its history per contact, both implicit over the step;
* van der Waals attraction A R* / (6 h^2) and the adhesion of touching surfaces, as in the viscosity analysis;
* the box: periodic, its sides L_a(t) = L_a(0) exp(E_aa t); positions move affinely with it plus the motion solved;
* stress: sigma = (1/V) sum d (x) F over the pairs (lubrication, contacts, attraction) plus the stresslet of every
  sphere in the strain (2 mu E (1 + 5/2 phi)). The axial stress reported is sigma_zz - sigma_free, the stress along
  z with the free side at zero stress (the pressure is whatever keeps the volume).

The pair laws are those of suspension.py (Mari, Seto, Morris and Denn, J. Rheol. 58 (2014) 1693), written here for
NumPy/Numba on the CPU and a deforming box; suspension.py itself (simple shear, Lees-Edwards) is not touched.

Reductions: spheres only; one resin of one viscosity (at the equivalent shear rate), no pores, no cure; at least
one side free. Other cases stay with the flow solve of compress.py alone.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

LUB_CUT = 0.25          # lubrication acts below this gap / mean radius of the pair (suspension.py)
SKIN = 0.05
KN_FACTOR = 1000.0
LATERAL = {"xy": (0.5, 0.5), "x": (1.0, 0.0)}


@njit(cache=True)
def _sep(x, i, j, L, d):
    for a in range(3):
        v = x[j, a] - x[i, a]
        v -= L[a] * math.floor(v / L[a] + 0.5)
        d[a] = v


@njit(cache=True)
def _pairs(x, rad, L, gap, skin):
    """Every pair i < j whose surfaces are within gap mean radii plus the skin."""
    n = x.shape[0]
    d = np.empty(3)
    cap = max(64, 16 * n)
    pi = np.empty(cap, np.int64)
    pj = np.empty(cap, np.int64)
    m = 0
    for i in range(n):
        for j in range(i + 1, n):
            _sep(x, i, j, L, d)
            r = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            if r - rad[i] - rad[j] < gap * 0.5 * (rad[i] + rad[j]) + skin:
                if m == cap:
                    cap *= 2
                    p2 = np.empty(cap, np.int64)
                    p2[:m] = pi[:m]
                    pi = p2
                    p2 = np.empty(cap, np.int64)
                    p2[:m] = pj[:m]
                    pj = p2
                pi[m], pj[m] = i, j
                m += 1
    return pi[:m].copy(), pj[:m].copy()


@njit(cache=True)
def _geometry(x, rad, L, pi, pj, par, dt, pd, pn, ph, pcn, pct, pkn, pkt, xi):
    d = np.empty(3)
    for m in range(pi.shape[0]):
        i, j = pi[m], pj[m]
        _sep(x, i, j, L, d)
        r = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        ai, aj = rad[i], rad[j]
        h = r - ai - aj
        rs = ai * aj / (ai + aj)
        hc = LUB_CUT * 0.5 * (ai + aj)
        he = max(h, 0.0) + par[2]
        cn, ct = 0.0, 0.0
        if he < hc:
            cn = 6.0 * math.pi * rs * rs * (1.0 / he - 1.0 / hc)
            ct = (6.0 * math.pi * (4.0 / 15.0) * ai * aj * (2.0 * ai * ai + ai * aj + 2.0 * aj * aj) / (ai + aj) ** 3
                  * math.log(hc / he))
        for a in range(3):
            pd[m, a] = d[a]
            pn[m, a] = d[a] / r
        ph[m] = h
        pcn[m] = cn
        pct[m] = ct
        kn_, kt_ = 0.0, 0.0
        if h < 0.0:
            kn_ = par[5] * dt
            if par[7] > 0.0:
                kt_ = par[6] * dt
        pkn[m] = kn_
        pkt[m] = kt_
        if h >= 0.0:
            xi[m, 0], xi[m, 1], xi[m, 2] = 0.0, 0.0, 0.0


@njit(cache=True)
def _pair_force(m, ai, aj, pn, cn, ct, Ui, Wi, Uj, Wj, f):
    """f = cn v_n n + ct v_t: the force on i for the motions given (v: the relative velocity of the surfaces)."""
    nx, ny, nz = pn[m, 0], pn[m, 1], pn[m, 2]
    sx = ai * Wi[0] + aj * Wj[0]
    sy = ai * Wi[1] + aj * Wj[1]
    sz = ai * Wi[2] + aj * Wj[2]
    vx = (Uj[0] - Ui[0]) - (sy * nz - sz * ny)
    vy = (Uj[1] - Ui[1]) - (sz * nx - sx * nz)
    vz = (Uj[2] - Ui[2]) - (sx * ny - sy * nx)
    vn = nx * vx + ny * vy + nz * vz
    f[0] = cn * vn * nx + ct * (vx - vn * nx)
    f[1] = cn * vn * ny + ct * (vy - vn * ny)
    f[2] = cn * vn * nz + ct * (vz - vn * nz)


@njit(cache=True)
def _apply(rad, pi, pj, pn, pcn, pct, pkn, pkt, su, sw, fu, fw):
    """fu, fw = R . (su, sw)."""
    n = rad.shape[0]
    f = np.empty(3)
    for i in range(n):
        a = rad[i]
        for c in range(3):
            fu[i, c] = 6.0 * math.pi * a * su[i, c]
            fw[i, c] = 8.0 * math.pi * a * a * a * sw[i, c]
    for m in range(pi.shape[0]):
        if pcn[m] > 0.0 or pkn[m] > 0.0:
            i, j = pi[m], pj[m]
            _pair_force(m, rad[i], rad[j], pn, pcn[m] + pkn[m], pct[m] + pkt[m], su[i], sw[i], su[j], sw[j], f)
            nx, ny, nz = pn[m, 0], pn[m, 1], pn[m, 2]
            tx, ty, tz = ny * f[2] - nz * f[1], nz * f[0] - nx * f[2], nx * f[1] - ny * f[0]
            for c in range(3):
                fu[i, c] -= f[c]
                fu[j, c] += f[c]
            fw[i, 0] -= rad[i] * tx
            fw[i, 1] -= rad[i] * ty
            fw[i, 2] -= rad[i] * tz
            fw[j, 0] -= rad[j] * tx
            fw[j, 1] -= rad[j] * ty
            fw[j, 2] -= rad[j] * tz


@njit(cache=True)
def _precond(rad, pi, pj, pn, pcn, pct, pkn, pkt, Mu, Mw):
    n = rad.shape[0]
    for i in range(n):
        a = rad[i]
        for c in range(3):
            for e in range(3):
                Mu[i, c, e] = 0.0
                Mw[i, c, e] = 0.0
            Mu[i, c, c] = 6.0 * math.pi * a
            Mw[i, c, c] = 8.0 * math.pi * a * a * a
    for m in range(pi.shape[0]):
        if pcn[m] > 0.0 or pkn[m] > 0.0:
            i, j = pi[m], pj[m]
            cn, ct = pcn[m] + pkn[m], pct[m] + pkt[m]
            for c in range(3):
                for e in range(3):
                    nn = pn[m, c] * pn[m, e]
                    tt = (1.0 if c == e else 0.0) - nn
                    bt = cn * nn + ct * tt
                    br = ct * tt
                    Mu[i, c, e] += bt
                    Mu[j, c, e] += bt
                    Mw[i, c, e] += rad[i] * rad[i] * br
                    Mw[j, c, e] += rad[j] * rad[j] * br
    for i in range(n):
        Mu[i] = np.linalg.inv(Mu[i])
        Mw[i] = np.linalg.inv(Mw[i])


@njit(cache=True)
def _rhs(rad, sqa, pi, pj, pd, pn, ph, pcn, pct, pkn, pkt, xi, par, E, bu, bw, pfc):
    """The forces at zero motion relative to the strain: lubrication of the imposed relative motion, contacts with
    friction, van der Waals, adhesion; pfc keeps the non-hydrodynamic pair force."""
    n = rad.shape[0]
    for i in range(n):
        for c in range(3):
            bu[i, c] = 0.0
            bw[i, c] = 0.0
    Z = np.zeros(3)
    dU = np.empty(3)
    f = np.empty(3)
    fc = np.empty(3)
    ft = np.empty(3)
    for m in range(pi.shape[0]):
        i, j = pi[m], pj[m]
        nx, ny, nz = pn[m, 0], pn[m, 1], pn[m, 2]
        h = ph[m]
        f[0], f[1], f[2] = 0.0, 0.0, 0.0
        if pcn[m] > 0.0 or pkn[m] > 0.0:
            for c in range(3):
                dU[c] = E[c] * pd[m, c]
            _pair_force(m, rad[i], rad[j], pn, pcn[m] + pkn[m], pct[m] + pkt[m], Z, Z, dU, Z, f)
        fc[0], fc[1], fc[2] = 0.0, 0.0, 0.0
        ft[0], ft[1], ft[2] = 0.0, 0.0, 0.0
        rs = rad[i] * rad[j] / (rad[i] + rad[j])
        aij = sqa[i] * sqa[j]
        if aij > 0.0 and h < par[10]:
            hv = max(h, par[3])
            g = aij * rs / (6.0 * hv * hv)
            fc[0] += g * nx
            fc[1] += g * ny
            fc[2] += g * nz
        if par[12] > 0.0 and h < par[3]:
            g = 2.0 * math.pi * par[12] * rs
            fc[0] += g * nx
            fc[1] += g * ny
            fc[2] += g * nz
        if h < 0.0:
            g = par[5] * h
            fc[0] += g * nx
            fc[1] += g * ny
            fc[2] += g * nz
            if par[7] > 0.0:
                sn = xi[m, 0] * nx + xi[m, 1] * ny + xi[m, 2] * nz
                s0, s1, s2 = xi[m, 0] - sn * nx, xi[m, 1] - sn * ny, xi[m, 2] - sn * nz
                ft[0], ft[1], ft[2] = par[6] * s0, par[6] * s1, par[6] * s2
                fmax = par[7] * par[5] * (-h)
                fl = math.sqrt(ft[0] * ft[0] + ft[1] * ft[1] + ft[2] * ft[2])
                if fl > fmax:
                    sc = fmax / fl
                    ft[0] *= sc
                    ft[1] *= sc
                    ft[2] *= sc
                    s0, s1, s2 = ft[0] / par[6], ft[1] / par[6], ft[2] / par[6]
                xi[m, 0], xi[m, 1], xi[m, 2] = s0, s1, s2
        gx, gy, gz = f[0] + ft[0], f[1] + ft[1], f[2] + ft[2]
        for c in range(3):
            v = f[c] + fc[c] + ft[c]
            bu[i, c] += v
            bu[j, c] -= v
            pfc[m, c] = fc[c] + ft[c]
        tx, ty, tz = ny * gz - nz * gy, nz * gx - nx * gz, nx * gy - ny * gx
        bw[i, 0] += rad[i] * tx
        bw[i, 1] += rad[i] * ty
        bw[i, 2] += rad[i] * tz
        bw[j, 0] += rad[j] * tx
        bw[j, 1] += rad[j] * ty
        bw[j, 2] += rad[j] * tz


@njit(cache=True)
def _solve(rad, pi, pj, pn, pcn, pct, pkn, pkt, bu, bw, u, w, tol, maxit):
    """Preconditioned conjugate gradients on R . V = F, warm-started from u, w. Returns the iterations."""
    n = rad.shape[0]
    Mu = np.empty((n, 3, 3))
    Mw = np.empty((n, 3, 3))
    _precond(rad, pi, pj, pn, pcn, pct, pkn, pkt, Mu, Mw)
    fu = np.empty((n, 3))
    fw = np.empty((n, 3))
    _apply(rad, pi, pj, pn, pcn, pct, pkn, pkt, u, w, fu, fw)
    ru, rw = bu - fu, bw - fw
    bn = math.sqrt((bu * bu).sum() + (bw * bw).sum())
    if bn == 0.0:
        bn = 1.0
    zu = np.empty((n, 3))
    zw = np.empty((n, 3))
    for i in range(n):
        zu[i] = Mu[i] @ ru[i]
        zw[i] = Mw[i] @ rw[i]
    pu, pw = zu.copy(), zw.copy()
    rz = (ru * zu).sum() + (rw * zw).sum()
    it = 0
    while it < maxit and math.sqrt((ru * ru).sum() + (rw * rw).sum()) > tol * bn:
        _apply(rad, pi, pj, pn, pcn, pct, pkn, pkt, pu, pw, fu, fw)
        pq = (pu * fu).sum() + (pw * fw).sum()
        if pq <= 0.0:
            break
        al = rz / pq
        u += al * pu
        w += al * pw
        ru -= al * fu
        rw -= al * fw
        for i in range(n):
            zu[i] = Mu[i] @ ru[i]
            zw[i] = Mw[i] @ rw[i]
        rz2 = (ru * zu).sum() + (rw * zw).sum()
        be = rz2 / rz
        rz = rz2
        pu = zu + be * pu
        pw = zw + be * pw
        it += 1
    return it


@njit(cache=True)
def _stress(rad, pi, pj, pd, pn, ph, pcn, pct, pkn, pkt, pfc, u, w, E, out):
    """out: 0-2 sum d_a F_a of the lubrication, 3-5 of contacts + friction + attraction, 6 contacts, 7 largest
    overlap."""
    for s in range(8):
        out[s] = 0.0
    fl = np.empty(3)
    fk = np.empty(3)
    Uj = np.empty(3)
    for m in range(pi.shape[0]):
        i, j = pi[m], pj[m]
        fl[0], fl[1], fl[2] = 0.0, 0.0, 0.0
        fk[0], fk[1], fk[2] = 0.0, 0.0, 0.0
        if pcn[m] > 0.0 or pkn[m] > 0.0:
            for c in range(3):
                Uj[c] = u[j, c] + E[c] * pd[m, c]
            _pair_force(m, rad[i], rad[j], pn, pcn[m], pct[m], u[i], w[i], Uj, w[j], fl)
            _pair_force(m, rad[i], rad[j], pn, pkn[m], pkt[m], u[i], w[i], Uj, w[j], fk)
        for c in range(3):
            out[c] += pd[m, c] * fl[c]
            out[3 + c] += pd[m, c] * (pfc[m, c] + fk[c])
        if ph[m] < 0.0:
            out[6] += 1.0
            if -ph[m] > out[7]:
                out[7] = -ph[m]


@njit(cache=True)
def _move(x, L, rad, pi, pj, pd, pn, ph, xi, u, w, E, par, dt):
    """The tangential springs grow with the relative surface velocity; the positions move with the strain and the
    motion solved; the box with the strain."""
    if par[7] > 0.0:
        for m in range(pi.shape[0]):
            if ph[m] < 0.0:
                i, j = pi[m], pj[m]
                nx, ny, nz = pn[m, 0], pn[m, 1], pn[m, 2]
                sx = rad[i] * w[i, 0] + rad[j] * w[j, 0]
                sy = rad[i] * w[i, 1] + rad[j] * w[j, 1]
                sz = rad[i] * w[i, 2] + rad[j] * w[j, 2]
                vx = (u[j, 0] - u[i, 0]) + E[0] * pd[m, 0] - (sy * nz - sz * ny)
                vy = (u[j, 1] - u[i, 1]) + E[1] * pd[m, 1] - (sz * nx - sx * nz)
                vz = (u[j, 2] - u[i, 2]) + E[2] * pd[m, 2] - (sx * ny - sy * nx)
                vn = vx * nx + vy * ny + vz * nz
                xi[m, 0] += (vx - vn * nx) * dt
                xi[m, 1] += (vy - vn * ny) * dt
                xi[m, 2] += (vz - vn * nz) * dt
    for a in range(3):
        s = math.exp(E[a] * dt)
        L[a] *= s
        for i in range(x.shape[0]):
            v = x[i, a] * s + u[i, a] * dt
            x[i, a] = v - L[a] * math.floor(v / L[a])


def compress(centres_um, radii_um, box_um, mu_resin, rate, compression, lateral="xy", roughness_m=5e-9, hmin_m=1e-9,
             hamaker_J=0.0, mu_f=0.25, adhesion_J_m2=0.0, records=40, max_disp=2e-3, dt_max=5e-3, tol=1e-5,
             progress=None, should_stop=None, log=None):
    """Presses the spheres (centres, radii in µm; periodic box in µm) by `compression` (0-1) along z at the strain
    rate `rate` (1/s) in a resin of viscosity mu_resin (Pa s). lateral: "xy" both sides free, "x" x free and y held.
    Returns {"compression", "sigma_zz_Pa" (axial stress with the free side at zero stress, negative in compression),
    its parts "hydro_Pa" (resin and films) and "contact_Pa" (contacts, friction, attraction), "contacts_per_particle",
    "max_overlap" (largest radii), "eta_compression_Pa_s" (-stress / rate, mean over the second half), "centres_um",
    "box_um", "steps", "seconds", ...}."""
    if lateral not in LATERAL:
        raise ValueError("the contact dynamics need at least one free side")
    t0 = time.time()
    log = log or (lambda *a: None)
    r_um = np.asarray(radii_um, float)
    a0 = float(r_um.max())
    rad = np.ascontiguousarray(r_um / a0)
    n = len(rad)
    L = np.asarray(box_um, float) / a0
    L0 = L.copy()
    x = np.ascontiguousarray(np.mod(np.asarray(centres_um, float) / a0, L))
    V = float(np.prod(L))
    phi = float((4.0 / 3.0) * math.pi * np.sum(rad ** 3) / V)
    ex, ey = LATERAL[lateral]
    E = np.array([ex, ey, -1.0])
    a0_m = a0 * 1e-6
    unit = mu_resin * rate                                   # stress unit, Pa
    delta = roughness_m / a0_m
    hmin = max(hmin_m, 1e-12) / a0_m
    hk = np.broadcast_to(np.asarray(hamaker_J, float), (n,)) / (mu_resin * rate * a0_m ** 3)
    adh = float(adhesion_J_m2) / (mu_resin * rate * a0_m)
    vdw_range = 0.1
    gap = max(LUB_CUT, 2.0 * vdw_range)
    par = np.zeros(16)
    par[2], par[3] = delta, hmin
    f_att = float(np.max(hk)) * 0.5 / (6.0 * hmin * hmin) + 2.0 * math.pi * adh * 0.5
    par[5] = max(1.0e4, KN_FACTOR * f_att)
    par[6] = 0.5 * par[5]
    par[7], par[10], par[12] = float(mu_f), vdw_range, adh
    sqa = np.ascontiguousarray(np.sqrt(np.maximum(hk, 0.0)))
    u = np.zeros((n, 3))
    w = np.zeros((n, 3))
    bu = np.zeros((n, 3))
    bw = np.zeros((n, 3))
    t_end = -math.log(1.0 - float(compression))
    rec_every = t_end / max(int(records), 1)
    out = {k: [] for k in ("compression", "sigma_zz_Pa", "hydro_Pa", "contact_Pa", "contacts_per_particle", "max_overlap",
                           "cg_iterations")}
    pi = pj = None
    xi = np.zeros((0, 3))
    keys = np.zeros(0, np.int64)
    moved, dt_prev, t, step, next_rec = 1e9, 1e-4, 0.0, 0, 0.0
    sig = np.zeros(8)
    einstein = 2.0 * (1.0 + 2.5 * phi) * (E[2] - E[0])       # the spheres' stresslets and the resin, sigma_zz - sigma_xx
    while True:
        if should_stop is not None and should_stop():
            raise InterruptedError
        if moved > 0.025:
            pi, pj = _pairs(x, rad, L, gap, SKIN)
            k2 = pi * n + pj
            xi2 = np.zeros((len(pi), 3))
            if len(keys) and len(k2):
                pos = np.searchsorted(keys, k2)
                ok = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == k2)
                xi2[ok] = xi[pos[ok]]
            keys, xi = k2, xi2
            m = len(pi)
            pd, pn, pfc = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros((m, 3))
            ph, pcn, pct, pkn, pkt = (np.zeros(m) for _ in range(5))
            moved = 0.0
        _geometry(x, rad, L, pi, pj, par, dt_prev, pd, pn, ph, pcn, pct, pkn, pkt, xi)
        _rhs(rad, sqa, pi, pj, pd, pn, ph, pcn, pct, pkn, pkt, xi, par, E, bu, bw, pfc)
        its = _solve(rad, pi, pj, pn, pcn, pct, pkn, pkt, bu, bw, u, w, tol, 5000)
        umax = max(float(np.sqrt((u * u).sum(axis=1)).max()), 1e-6)
        if t >= next_rec - 1e-12 or t >= t_end:
            _stress(rad, pi, pj, pd, pn, ph, pcn, pct, pkn, pkt, pfc, u, w, E, sig)
            if lateral == "xy":
                lub = (sig[2] - 0.5 * (sig[0] + sig[1])) / V
                con = (sig[5] - 0.5 * (sig[3] + sig[4])) / V
            else:
                lub = (sig[2] - sig[0]) / V
                con = (sig[5] - sig[3]) / V
            out["compression"].append(1.0 - L[2] / L0[2])
            out["hydro_Pa"].append((einstein + lub) * unit)
            out["contact_Pa"].append(con * unit)
            out["sigma_zz_Pa"].append((einstein + lub + con) * unit)
            out["contacts_per_particle"].append(2.0 * sig[6] / n)
            out["max_overlap"].append(float(sig[7]))
            out["cg_iterations"].append(int(its))
            next_rec += rec_every
            if sig[7] > 0.02:                                  # the contacts stiffen until no overlap exceeds 2 %
                par[5] *= 1.5
                par[6] *= 1.5
            if progress:
                progress(out["compression"][-1], out["sigma_zz_Pa"][-1], its)
        if t >= t_end:
            break
        dt = min(dt_max, max_disp / umax)
        if dt > 1.5 * dt_prev:
            dt = 1.5 * dt_prev                                 # the implicit springs assumed dt_prev
        if t + dt > t_end:
            dt = max(t_end - t, 1e-12)
        dt_prev = dt
        _move(x, L, rad, pi, pj, pd, pn, ph, xi, u, w, E, par, dt)
        t += dt
        step += 1
        moved += dt * (umax + 2.6)
    s = np.asarray(out["sigma_zz_Pa"])
    half = s[len(s) // 2:]
    res = {k: [float(v) for v in vals] for k, vals in out.items()}
    res.update({"eta_compression_Pa_s": float(-half.mean() / rate), "eta_start_Pa_s": float(-s[0] / rate),
                "eta_end_Pa_s": float(-s[-1] / rate), "centres_um": x * a0, "box_um": (L * a0).tolist(),
                "steps": int(step), "seconds": time.time() - t0, "n": int(n), "phi": phi, "mu_f": float(mu_f),
                "roughness_nm": float(roughness_m * 1e9), "lateral": lateral, "rate_1_s": float(rate),
                "mu_resin_Pa_s": float(mu_resin), "kn": float(par[5]),
                "trouton": 3.0 if lateral == "xy" else 4.0})
    log(f"    contact dynamics: {n} spheres at {100 * phi:.1f} vol%, friction {mu_f:g}, roughness {roughness_m * 1e9:g} nm: "
        f"axial stress {-s[0]:.4g} -> {-s[-1]:.4g} Pa over {100 * compression:.0f} % ({step} steps, "
        f"{res['contacts_per_particle'][-1]:.2f} contacts per sphere at the end, {res['seconds']:.0f} s)")
    return res
