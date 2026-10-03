"""Closed-form references shown next to every numerical result.

Two kinds, kept apart on purpose:

* **rigorous bounds** (Wiener, Hashin-Shtrikman, Schapery) - a correct full-field
  answer can never fall outside them, so they are used as automatic checks;
* **mean-field models** (Maxwell-Garnett / Mori-Tanaka with shape and
  orientation, Bruggeman, Turner, Kerner) - approximations whose distance from
  the full-field answer is itself the information.

Conductivity-type functions work for k, sigma, eps_r and mu_r alike.
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as G


# =========================================================================
# conductivity-type
# =========================================================================
def wiener(phi, k):
    phi = np.asarray(phi, float)
    k = np.asarray(k, float)
    upper = float(np.sum(phi * k))
    with np.errstate(divide="ignore"):
        inv = np.sum(np.where(phi > 0, phi / np.maximum(k, 1e-300), 0.0))
    lower = float(1.0 / inv) if inv > 0 else 0.0
    return lower, upper


def hashin_shtrikman(phi, k):
    """n-phase HS bounds for a statistically isotropic mixture."""
    phi = np.asarray(phi, float)
    k = np.asarray(k, float)
    present = phi > 0
    kp = k[present]
    ph = phi[present]

    def A(k0):
        s = np.sum(ph / (kp + 2.0 * k0)) if k0 > 0 else np.sum(
            np.where(kp > 0, ph / np.maximum(kp, 1e-300), np.inf))
        return float(1.0 / s - 2.0 * k0) if np.isfinite(s) and s > 0 else 0.0
    return A(float(kp.min())), A(float(kp.max()))


def depolarization(aspect):
    """Depolarisation factor along the symmetry axis of a spheroid of aspect
    ratio c/a (needle -> 0, sphere 1/3, disc -> 1)."""
    p = float(aspect)
    if abs(p - 1.0) < 1e-6:
        return 1.0 / 3.0
    if p > 1.0:
        e = math.sqrt(1.0 - 1.0 / (p * p))
        return (1.0 - e * e) / (2.0 * e ** 3) * (math.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
    e = math.sqrt(1.0 / (p * p) - 1.0)
    return (1.0 + e * e) / e ** 3 * (e - math.atan(e))


def mori_tanaka(km, phases):
    """Maxwell-Garnett / Mori-Tanaka effective tensor diagonal.

    phases: list of dicts {phi, k, aspect, orientation, r_int(optional, in
    units of 1/k * m), a_eq_m(optional)}. The interfacial resistance enters by
    the Hasselman-Johnson equivalent inclusion k* = k/(1 + R k / a), exact for
    spheres and an approximation for other shapes.
    """
    phi_m = 1.0 - sum(p["phi"] for p in phases)
    out = []
    for comp in range(3):
        num = 0.0
        den = phi_m
        for p in phases:
            if p["phi"] <= 0:
                continue
            k = p["k"]
            if p.get("r_int") and p.get("a_eq_m"):
                k = k / (1.0 + p["r_int"] * k / p["a_eq_m"])
            Lp = depolarization(p.get("aspect", 1.0))
            Lt = 0.5 * (1.0 - Lp)
            Ap = km / (km + Lp * (k - km))
            At = km / (km + Lt * (k - km))
            m2 = G.orientation_moments(p.get("orientation", "iso"))[comp]
            A = At + (Ap - At) * m2
            num += p["phi"] * (k - km) * A
            den += p["phi"] * A
        out.append(km + num / den)
    return out


def bruggeman(phi, k):
    """Symmetric effective-medium root (spheres), n phases incl. the matrix."""
    phi = np.asarray(phi, float)
    k = np.maximum(np.asarray(k, float), 1e-300)

    def f(x):
        return float(np.sum(phi * (k - x) / (k + 2.0 * x)))
    lo, hi = float(k.min()), float(k.max())
    if hi / lo < 1.0 + 1e-12:
        return lo
    llo, lhi = math.log(lo), math.log(hi)
    for _ in range(200):
        mid = 0.5 * (llo + lhi)
        if f(math.exp(mid)) > 0:
            llo = mid
        else:
            lhi = mid
    return math.exp(0.5 * (llo + lhi))


# =========================================================================
# elastic / thermal expansion
# =========================================================================
def bulk_shear(E, nu):
    E = np.asarray(E, float)
    nu = np.asarray(nu, float)
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def hs_elastic(phi, K, Gm):
    """n-phase Hashin-Shtrikman (Walpole) bounds on K and G."""
    phi = np.asarray(phi, float)
    K = np.asarray(K, float)
    Gm = np.asarray(Gm, float)
    pr = phi > 0
    phi, K, Gm = phi[pr], K[pr], Gm[pr]

    def Kb(G0):
        s = np.sum(phi / (K + 4.0 / 3.0 * G0))
        return float(1.0 / s - 4.0 / 3.0 * G0) if s > 0 and np.isfinite(s) else 0.0

    def zeta(K0, G0):
        return G0 / 6.0 * (9.0 * K0 + 8.0 * G0) / (K0 + 2.0 * G0) if (K0 + 2 * G0) > 0 else 0.0

    def Gb(z):
        with np.errstate(divide="ignore"):
            s = np.sum(phi / (Gm + z))
        return float(1.0 / s - z) if s > 0 and np.isfinite(s) else 0.0
    return {"K": (Kb(Gm.min()), Kb(Gm.max())),
            "G": (Gb(zeta(K.min(), Gm.min())), Gb(zeta(K.max(), Gm.max())))}


def mori_tanaka_elastic_spheres(phi, K, Gm, alpha, matrix=0):
    """Mori-Tanaka K, G and a Levin-consistent CTE for spherical inclusions."""
    phi = np.asarray(phi, float)
    K = np.asarray(K, float)
    Gm = np.asarray(Gm, float)
    alpha = np.asarray(alpha, float)
    Km, Gmm = K[matrix], Gm[matrix]
    b = (Km + 4.0 / 3.0 * Gmm) / (K + 4.0 / 3.0 * Gmm)
    zeta = Gmm * (9.0 * Km + 8.0 * Gmm) / (6.0 * (Km + 2.0 * Gmm))
    c = (Gmm + zeta) / (Gm + zeta)
    K_mt = float(np.sum(phi * K * b) / np.sum(phi * b))
    G_mt = float(np.sum(phi * Gm * c) / np.sum(phi * c))
    den = np.sum(phi * K * b)
    a_mt = float(np.sum(phi * K * b * alpha) / den) if den > 0 else float(alpha[matrix])
    return K_mt, G_mt, a_mt


def cte_models(phi, E, nu, alpha, matrix=0):
    phi = np.asarray(phi, float)
    K, Gm = bulk_shear(E, nu)
    alpha = np.asarray(alpha, float)
    out = {"ROM": float(np.sum(phi * alpha))}
    sK = np.sum(phi * K)
    out["Turner"] = float(np.sum(phi * K * alpha) / sK) if sK > 0 else float("nan")
    try:
        K_mt, G_mt, a_mt = mori_tanaka_elastic_spheres(phi, K, Gm, alpha, matrix)
        out["Kerner / Mori-Tanaka (spheres)"] = a_mt
    except Exception:                                          # noqa: BLE001
        pass
    present = [i for i in range(len(phi)) if phi[i] > 0]
    if len(present) == 2 and all(K[i] > 0 for i in present):
        i1, i2 = present
        hs = hs_elastic(phi, K, Gm)
        vals = []
        for Kstar in hs["K"]:
            if Kstar <= 0:
                continue
            f = (1.0 / Kstar - 1.0 / K[i1]) / (1.0 / K[i2] - 1.0 / K[i1])
            vals.append(float(alpha[i1] + (alpha[i2] - alpha[i1]) * f))
        if len(vals) == 2:
            out["Schapery bounds"] = (min(vals), max(vals))
    return out
