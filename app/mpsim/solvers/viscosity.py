"""Viscosity and flowability of a filled liquid (uncured underfill, mould
compound, adhesive or paste).

Effective viscosity on the RVE
------------------------------
Creeping (Stokes) flow of a Newtonian resin around rigid particles obeys the
equations of incompressible linear elasticity with velocity for displacement,
strain rate for strain and viscosity for the shear modulus. The RVE is
therefore solved by the FANS voxel finite elements (solvers/fans.py) with
B-bar elements (no volumetric locking at incompressibility), a uniform bulk
penalty for incompressibility, and particles as a fluid far more viscous than
the resin (RIGID_RATIO, rigid for the flow). A unit macroscopic shear rate in
each plane gives the shear viscosity of the suspension; the mean stress over
the resin viscosity is the relative viscosity mu_r. Pores (air bubbles) are a
fluid of vanishing viscosity, which lowers mu_r as it should.

The voxel solve resolves the hydrodynamics of the particle arrangement, not
the lubrication film between two particles closer than a voxel; near the
maximum packing fraction it therefore under-predicts (as every grid method
does), and the Krieger-Dougherty curve below carries the divergence.

Closed forms (references in the guide, chapter "Viscosity and flowability")
- Einstein (1906): mu_r = 1 + 2.5 phi; Batchelor (1977): + 6.2 phi^2
- Hashin-Shtrikman lower bound for rigid spheres in an incompressible matrix:
  mu_r >= 1 + 2.5 phi / (1 - phi)
- Krieger-Dougherty (1959): mu_r = (1 - phi/phi_m)^(-[eta] phi_m)
- intrinsic viscosity [eta]: 2.5 for spheres; prolate spheroids (Simha 1940,
  large aspect), oblate spheroids / platelets (Simha; thin discs 32 r / 15 pi);
  any other shape from a dilute RVE of the same particles
- Chateau, Ovarlez and Trung (J. Rheol. 52 (2008) 489): a suspension in a
  non-Newtonian resin, sigma_s(g) = (1 - phi) A sigma_m(A g) with
  A = sqrt(mu_r / (1 - phi)) the mean shear-rate amplification in the resin
- capillary underfill between parallel plates (Washburn 1921; Han and Wang,
  IEEE CPMT-B 20 (1997) 424): t = 3 mu L^2 / (h gamma cos theta)
- settling: Stokes velocity times the hindered-settling factor (1 - phi)^4.65
  (Richardson and Zaki 1954)

Maximum packing fraction phi_m
- spheres of any size distribution: Farr and Groot (J. Chem. Phys. 131 (2009)
  244104), the random close packing from a mapping onto rods on a line; it
  reproduced their 3D packings of bidisperse (ratios to 1:10), tridisperse
  and log-normal spheres within about 0.01, and the exact limit of infinite
  size ratio
- Desmond and Weeks (Phys. Rev. E 90 (2014) 022204): phi_RCP = 0.634 +
  0.0658 delta + 0.0857 S delta^2 from the polydispersity and skewness of the
  radii, fitted for delta <= 0.4 (shown as a check where it applies)
- other shapes: the jammed packing of the same particles (generate.
  jamming_fraction)
- friction: frictional particles jam in shear below the frictionless random
  close packing (0.585 for spheres against 0.64; Boyer, Guazzelli and
  Pouliquen, Phys. Rev. Lett. 107 (2011) 188301); FRICTION_FACTOR scales
  phi_m by that ratio when the option asks for it
"""
from __future__ import annotations

import math

import numpy as np

from . import fans as FA

RIGID_RATIO = 1.0e3        # particle / resin viscosity: rigid for the flow (checked in the self-test)
BUBBLE_RATIO = 1.0e-4      # pore / resin viscosity
PENALTY = 1.0e2            # bulk penalty / largest viscosity: incompressible to 1 %


def phase_viscosities(table, mu_resin=1.0):
    """Shear viscosity per label for the flow solve: the resin (matrix, and any
    liquid interphase) mu_resin, solid particles RIGID_RATIO x, pores BUBBLE_RATIO x."""
    mu = np.empty(len(table))
    for i, t in enumerate(table):
        if t["kind"] == "matrix":
            mu[i] = mu_resin
        elif float(t["props"].get("E", 1.0)) <= 0:
            mu[i] = BUBBLE_RATIO * mu_resin
        else:
            mu[i] = RIGID_RATIO * mu_resin
    return mu


def effective_viscosity(labels, mu, tol=1e-6, maxiter=20000, progress=None, should_stop=None, workers=-1,
                        keep_field=True, extension=True):
    """Relative viscosity of the suspension (mu given per label, the resin
    1). Returns shear viscosities in the yz, xz, xy planes, the extensional
    viscosity over 3 (= the shear viscosity of an isotropic fluid), their
    isotropic mean, the local shear-rate magnification field of the xy shear
    and the solver record."""
    mu = np.asarray(mu, float)
    lam = np.full_like(mu, PENALTY * float(mu.max()))
    sol = FA.FANS(labels, None, None, None, workers=workers, lam_mu=(lam, mu), bbar=True)
    out = {"shear": {}, "stats": {}}
    field = None
    for k, plane in ((3, "yz"), (4, "xz"), (5, "xy")):
        Ev = np.zeros(6)
        Ev[k] = 1.0                         # engineering shear rate 1
        cb = (lambda it, res, plane=plane: progress(plane, it, res)) if progress else None
        sbar, st, _ = sol.solve(Ev, 0.0, tol, maxiter, cb, should_stop,
                                keep_strain=keep_field and plane == "xy", keep_disp=keep_field and plane == "xy")
        out["shear"][plane] = float(sbar[k])
        out["stats"][plane] = st
        if keep_field and plane == "xy":
            e = sol.last_strain
            # local shear rate sqrt(2 D':D') over the applied one (D = strain
            # rate with tensor shear components)
            tr = (e[0] + e[1] + e[2]) / 3.0
            dd = ((e[0] - tr) ** 2 + (e[1] - tr) ** 2 + (e[2] - tr) ** 2
                  + 0.5 * (e[3] ** 2 + e[4] ** 2 + e[5] ** 2))
            field = np.sqrt(2.0 * dd).astype(np.float32)
            sol.last_strain = None
            del e, tr, dd
            out["velocity"] = shear_velocity(sol.last_disp)
            sol.last_disp = None
    if extension:
        Ev = np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0])       # uniaxial, volume-preserving
        cb = (lambda it, res: progress("extension", it, res)) if progress else None
        sbar, st, _ = sol.solve(Ev, 0.0, tol, maxiter, cb, should_stop)
        out["extension"] = float((sbar[0] - 0.5 * (sbar[1] + sbar[2])) / 3.0)
        out["stats"]["extension"] = st
    vals = list(out["shear"].values()) + ([out["extension"]] if extension else [])
    out["mu_r"] = float(np.mean(list(out["shear"].values())))
    out["mu_r_all"] = float(np.mean(vals))
    out["field"] = field
    return out


def shear_velocity(u):
    """Resin velocity of simple shear in the xy plane at unit shear rate, at
    the voxel centres, (nx, ny, nz, 3) in voxels per unit time.

    The solve applies pure straining (E_xy = E_yx = 1/2) and returns the
    periodic fluctuation u' at the voxel corners. A rigid rotation of the whole
    RVE (W_xy = -W_yx = 1/2) is a stress-free Stokes flow in which every
    torque-free particle turns with the fluid, so adding it gives simple shear
    without another solve: v = (y - y0, 0, 0) + u'."""
    if u is None:
        return None
    uc = np.zeros(u.shape[1:] + (3,), np.float32)
    for c in range(3):
        a = u[c]
        s = a + np.roll(a, -1, 0)
        s = s + np.roll(s, -1, 1)
        s = s + np.roll(s, -1, 2)
        uc[..., c] = 0.125 * s
    ny = u.shape[2]
    y = (np.arange(ny) + 0.5 - 0.5 * ny).astype(np.float32)
    uc[..., 0] += y[None, :, None]
    return uc


# ------------------------------------------------------------ closed forms
def einstein(phi):
    return 1.0 + 2.5 * phi


def batchelor(phi):
    return 1.0 + 2.5 * phi + 6.2 * phi * phi


def hs_lower(phi):
    """Hashin-Shtrikman lower bound, rigid spheres in an incompressible
    matrix (exact to first order: Einstein)."""
    return 1.0 + 2.5 * phi / max(1.0 - phi, 1e-12)


def krieger_dougherty(phi, phi_m, eta):
    phi = np.asarray(phi, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(phi < phi_m, (1.0 - phi / phi_m) ** (-eta * phi_m), np.inf)


def maron_pierce(phi, phi_m):
    phi = np.asarray(phi, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(phi < phi_m, (1.0 - phi / phi_m) ** -2.0, np.inf)


def kd_phi_m_from(mu_r, phi, eta):
    """phi_m for which Krieger-Dougherty passes through (phi, mu_r) with the
    given [eta] (bisection; None when no phi_m in (phi, 1) fits)."""
    if mu_r <= 1.0 or phi <= 0:
        return None
    f = lambda pm: (1.0 - phi / pm) ** (-eta * pm) - mu_r
    lo, hi = phi * (1 + 1e-9), 1.0
    if f(lo) < 0 or f(hi) > 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


FG_F = 0.7654              # Farr-Groot free-length parameter: equal spheres 0.6435
FRICTION_FACTOR = 0.585 / 0.64


def rcp_farr_groot(d, w=None, n_rods=20000, f=FG_F):
    """Random close packing fraction of spheres with diameters d and number
    weights w (Farr and Groot 2009). A line through the packing cuts each
    sphere of diameter D into a rod of length L < D with probability density
    2 L / D^2, hit in proportion to D^2: the rods are taken at equidistant
    quantiles of that length distribution and packed on a line by the
    greedy rule of the paper (longest first, each into the largest gap,
    a free length f L_min kept between two rods)."""
    import heapq
    d = np.asarray(d, float).ravel()
    w = np.ones_like(d) if w is None else np.asarray(w, float).ravel()
    keep = (d > 0) & (w > 0)
    d, w = d[keep], w[keep]
    o = np.argsort(d)
    d, w = d[o], w[o]
    a = w * d * d
    a /= a.sum()
    # tail of the rod-length distribution: T(L) = sum_{D > L} a (1 - L^2 / D^2)
    sa = np.concatenate([np.cumsum(a[::-1])[::-1], [0.0]])
    sb = np.concatenate([np.cumsum((a / (d * d))[::-1])[::-1], [0.0]])

    def tail(L):
        k = np.searchsorted(d, L, side="right")
        return sa[k] - L * L * sb[k]
    q = (n_rods - np.arange(1, n_rods + 1) + 0.5) / n_rods      # T(L_i), decreasing in i
    lo = np.zeros(n_rods)
    hi = np.full(n_rods, d[-1])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        big = tail(mid) > q
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    L = np.sort(0.5 * (lo + hi))[::-1]
    gaps = [-f * L[0]]
    for Lj in L[1:]:
        g = -heapq.heappop(gaps)
        heapq.heappush(gaps, -f * Lj)
        heapq.heappush(gaps, -max(g - (1.0 + f) * Lj, f * Lj))
    return float(L.sum() / (L.sum() - sum(gaps)))


def rcp_desmond_weeks(d, w=None):
    """(phi_RCP, delta, S) of Desmond and Weeks 2014 from the number-weighted
    polydispersity delta and skewness S of the radii; fitted for delta <= 0.4."""
    d = np.asarray(d, float).ravel()
    w = np.ones_like(d) if w is None else np.asarray(w, float).ravel()
    w = w / w.sum()
    m = float(np.dot(w, d))
    dr = d - m
    v = float(np.dot(w, dr * dr))
    delta = math.sqrt(max(v, 0.0)) / m
    if delta < 1e-6:
        # equal spheres: the spread is round-off, and so would be its skewness
        delta, S = 0.0, 0.0
    else:
        S = float(np.dot(w, dr ** 3)) / v ** 1.5
    return 0.634 + 0.0658 * delta + 0.0857 * S * delta * delta, delta, S


def intrinsic_viscosity(shape, aspect=1.0):
    """[eta] of a dilute suspension of randomly oriented rigid particles, where
    a closed form exists: 2.5 for spheres (Einstein), Simha's limits for
    prolate (rods, aspect = length / diameter >= 10) and oblate particles
    (platelets, diameter / thickness >= 10; thin discs 32 r / 15 pi). None
    otherwise: the analysis then computes [eta] on a dilute RVE of the same
    particles (intrinsic_from_rve)."""
    p = float(aspect or 1.0)
    if shape in ("sphere", "network") or abs(p - 1.0) < 1e-9 and shape in ("spheroid",):
        return 2.5
    if p >= 10.0:
        lp = math.log(2.0 * p)
        return p * p / (15.0 * (lp - 1.5)) + p * p / (5.0 * (lp - 0.5)) + 14.0 / 15.0
    if p <= 0.1:
        return 32.0 / (15.0 * math.pi * p)
    return None


def intrinsic_from_rve(mu_r_dilute, phi_dilute):
    """[eta] from a dilute RVE: mu_r = 1 + [eta] phi + O(phi^2)."""
    return (mu_r_dilute - 1.0) / max(phi_dilute, 1e-12)


# ------------------------------------------------------------- rheology
def matrix_flow(model, gd):
    """Viscosity of the resin [Pa s] at shear rate gd [1/s] for a model dict:
    {"type": "newtonian", "mu": } | {"type": "power", "K":, "n":}
    | {"type": "carreau", "mu0":, "mu_inf":, "lam":, "n":}
    | {"type": "cross", "mu0":, "mu_inf":, "lam":, "n":}."""
    gd = np.maximum(np.asarray(gd, float), 1e-12)
    t = (model or {}).get("type", "newtonian")
    if t == "power":
        return model["K"] * gd ** (model["n"] - 1.0)
    if t == "carreau":
        return model["mu_inf"] + (model["mu0"] - model["mu_inf"]) * (1 + (model["lam"] * gd) ** 2) ** ((model["n"] - 1) / 2)
    if t == "cross":
        return model["mu_inf"] + (model["mu0"] - model["mu_inf"]) / (1 + (model["lam"] * gd) ** (1 - model["n"]))
    return np.full_like(gd, float(model.get("mu", 1.0)))


def suspension_flow(model, mu_r, phi, gd, tau_y=0.0):
    """Viscosity of the suspension at shear rate gd: Chateau-Ovarlez-Trung,
    sigma_s(g) = (1 - phi) A sigma_m(A g), A = sqrt(mu_r / (1 - phi)); a resin
    yield stress tau_y becomes sqrt((1 - phi) mu_r) tau_y. mu_r may be one
    value or one per shear rate (from the particle dynamics at several rates)."""
    gd = np.maximum(np.asarray(gd, float), 1e-12)
    A = np.sqrt(np.maximum(np.asarray(mu_r, float), 1.0) / max(1.0 - phi, 1e-9))
    if A.ndim == 0:
        A = float(A)
    sig_m = matrix_flow(model, A * gd) * A * gd + (tau_y if tau_y else 0.0)
    sig_s = (1.0 - phi) * A * sig_m
    return sig_s / gd, A


def underfill_time(mu, length_m, gap_m, gamma_N_m, theta_deg):
    """Capillary filling time between parallel plates (Washburn)."""
    c = math.cos(math.radians(theta_deg))
    if c <= 0:
        return math.inf
    return 3.0 * mu * length_m ** 2 / (gap_m * gamma_N_m * c)


def settling(d_m, rho_p, rho_f, mu_f, phi, time_s, g=9.81):
    """Stokes settling of one particle size with hindered settling."""
    v0 = 2.0 * (rho_p - rho_f) * g * (0.5 * d_m) ** 2 / (9.0 * mu_f)
    v = v0 * max(1.0 - phi, 0.0) ** 4.65
    return {"v0_m_s": v0, "v_m_s": v, "distance_m": v * time_s}
