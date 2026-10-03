"""Shielding effectiveness of a homogenised layer (exact, normal incidence).

The transfer-matrix code is the one validated in EMI/Claude_v1 (3D FDTD
agreed with it to 0.51 dB over 1.5 decades of conductivity). It takes the
effective sigma, eps_r and mu_r that the RVE solves produced, so the dB number
is traceable to the microstructure.

Quasi-static homogenisation assumes the microstructure is much smaller than
both the wavelength in the material and the skin depth inside the filler.
`validity` reports the frequencies at which those assumptions start to fail.
"""
from __future__ import annotations

import math

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * math.pi
C0 = 1.0 / math.sqrt(EPS0 * MU0)
Z0 = math.sqrt(MU0 / EPS0)


def _eps_c(f, eps_r, sigma):
    return EPS0 * eps_r - 1j * sigma / (2.0 * math.pi * f)


def _gamma(f, eps_r, mu_r, sigma):
    return 1j * 2.0 * math.pi * f * np.sqrt(MU0 * mu_r * _eps_c(f, eps_r, sigma))


def _eta(f, eps_r, mu_r, sigma):
    return np.sqrt(MU0 * mu_r / _eps_c(f, eps_r, sigma))


def slab_transmission(f, layers, z_left=Z0, z_right=Z0):
    """Cascade of lossy layers (thickness_m, eps_r, mu_r, sigma).

    Each layer matrix is factored as (e^{gd}/2) * Mtilde with u = e^{-2gd},
    and the prefactor is carried in the log domain, so a stack thousands of
    skin depths thick stays exact instead of overflowing.
    Returns (S11, log10|S21|).
    """
    f = np.atleast_1d(np.asarray(f, float))
    a = np.ones(f.shape, complex)
    b = np.zeros(f.shape, complex)
    c = np.zeros(f.shape, complex)
    d = np.ones(f.shape, complex)
    log_scale = np.zeros(f.shape, complex)
    for t, eps_r, mu_r, sigma in layers:
        g = _gamma(f, eps_r, mu_r, sigma)
        eta = _eta(f, eps_r, mu_r, sigma)
        gd = g * t
        u = np.exp(-2.0 * gd)
        la, lb, lc, ld = 1.0 + u, eta * (1.0 - u), (1.0 - u) / eta, 1.0 + u
        a, b, c, d = a * la + b * lc, a * lb + b * ld, c * la + d * lc, c * lb + d * ld
        log_scale = log_scale + gd - math.log(2.0)
    denom = a * z_right + b + c * z_left * z_right + d * z_left
    numer = a * z_right + b - c * z_left * z_right - d * z_left
    s11 = numer / denom
    log_s21 = np.log(2.0 * np.sqrt(z_left * z_right)) - log_scale - np.log(denom)
    return s11, np.real(log_s21) / math.log(10.0)


def schelkunoff_split(f, sigma, t, eps_r, mu_r):
    """Reflection / absorption / multiple-reflection terms that add up to the
    exact SE."""
    f = np.atleast_1d(np.asarray(f, float))
    eta = _eta(f, eps_r, mu_r, sigma)
    gd = _gamma(f, eps_r, mu_r, sigma) * t
    se_r = 20.0 * np.log10(np.abs((eta + Z0) ** 2 / (4.0 * eta * Z0)))
    se_a = (20.0 / math.log(10.0)) * np.real(gd)
    rho = (eta - Z0) / (eta + Z0)
    se_m = 20.0 * np.log10(np.abs(1.0 - rho ** 2 * np.exp(-2.0 * gd)))
    return se_r, se_a, se_m


def skin_depth(f, sigma, eps_r=1.0, mu_r=1.0):
    alpha = np.real(_gamma(np.atleast_1d(np.asarray(f, float)), eps_r, mu_r, sigma))
    return np.where(alpha > 0, 1.0 / np.maximum(alpha, 1e-300), np.inf)


def spectrum(sigma, eps_r, mu_r, thickness_m, f_min, f_max, n=160):
    f = np.logspace(math.log10(f_min), math.log10(f_max), n)
    s11, l21 = slab_transmission(f, [(thickness_m, eps_r, mu_r, sigma)])
    se = -20.0 * l21
    r, a, m = schelkunoff_split(f, sigma, thickness_m, eps_r, mu_r)
    R = np.abs(s11) ** 2
    T = 10.0 ** (2.0 * l21)
    return {
        "freq_hz": f.tolist(), "se_db": se.tolist(),
        "se_r_db": r.tolist(), "se_a_db": a.tolist(), "se_m_db": m.tolist(),
        "reflectance": R.tolist(), "transmittance": T.tolist(),
        "absorptance": np.clip(1.0 - R - T, 0.0, 1.0).tolist(),
        "skin_depth_m": skin_depth(f, sigma, eps_r, mu_r).tolist(),
    }


def validity(eps_r, mu_r, sigma_eff, rve_len_m, filler_dims):
    """Frequencies above which the quasi-static RVE answer should be doubted.

    * wavelength in the composite < 10 x RVE edge
    * skin depth in a conductive filler < its smallest dimension
      (filler_dims: list of (name, sigma, mu_r, min_dim_m))
    """
    n_eff = math.sqrt(max(eps_r * mu_r, 1.0))
    f_wave = C0 / (n_eff * 10.0 * rve_len_m)
    out = {"f_wavelength_hz": f_wave, "filler": []}
    for name, sig, mu, dmin in filler_dims:
        if sig <= 0 or dmin <= 0:
            continue
        # delta = 1/sqrt(pi f mu sigma) = dmin
        f_skin = 1.0 / (math.pi * MU0 * mu * sig * dmin * dmin)
        out["filler"].append({"name": name, "f_skin_hz": f_skin})
    return out
