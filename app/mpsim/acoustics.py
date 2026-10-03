"""Sound absorption of a rigid-frame porous layer (Johnson-Champoux-Allard).

The five JCA parameters come from the RVE solves rather than from fits:

    porosity φ                    open porosity of the voxel structure
    static flow resistivity σ     μ / K           (PuMA Stokes permeability)
    tortuosity α∞                 φ · F = τ       (PuMA continuum tortuosity;
                                  the electrical formation factor F of the
                                  pore space equals D0 / D_eff)
    viscous length Λ              2 ∫ |∇φ_e|² dV / ∫_S |∇φ_e|² dS
                                  (Johnson, Koplik & Dashen 1987), evaluated on
                                  the same potential field as the tortuosity
    thermal length Λ'             2 V_pore / S    (Champoux & Allard 1991)

The dynamic density and bulk modulus follow Allard & Atalla, "Propagation of
Sound in Porous Media", 2nd ed. (2009), eqs. 5.50-5.52, with an e^{jωt} time
convention. A hard-backed layer of thickness d has the surface impedance
Z_s = -j Z_c cot(k_c d) and the normal-incidence absorption α = 1 - |R|².

For straight cylindrical pores of radius r, Λ = Λ' = r and α∞ = 1 exactly;
the self-test checks the field-weighted Λ against that case.
"""
from __future__ import annotations

import math

import numpy as np


def _neighbour_gradient_sq(C, pore, axis, h, periodic):
    """Squared derivative of C along `axis`, using pore voxels only: central
    where both neighbours are pore, one-sided where one is, zero at a wall
    normal (the no-flux condition)."""
    Cp = np.roll(C, -1, axis)
    Cm = np.roll(C, 1, axis)
    pp = np.roll(pore, -1, axis)
    pm = np.roll(pore, 1, axis)
    if not periodic:
        last = [slice(None)] * 3
        last[axis] = -1
        first = [slice(None)] * 3
        first[axis] = 0
        pp[tuple(last)] = False
        pm[tuple(first)] = False
    g = np.where(pp & pm, (Cp - Cm) / (2.0 * h),
                 np.where(pp, (Cp - C) / h, np.where(pm, (C - Cm) / h, 0.0)))
    return g * g


def viscous_length(C, pore_mask, voxel_m, direction, area_ratio=1.0):
    """Λ from a potential field solved in the pore space with the macroscopic
    gradient along `direction` (no flux through the walls).

    area_ratio = true interface area / voxel-face interface area corrects the
    staircase surface (PuMA's marching-cubes area over the face count)."""
    C = np.asarray(C, float)
    pore = np.asarray(pore_mask, bool)
    d = "xyz".index(direction)
    h = float(voxel_m)
    g2 = np.zeros(C.shape)
    for ax in range(3):
        g2 += _neighbour_gradient_sq(C, pore, ax, h, periodic=(ax != d))
    vol = float(g2[pore].sum()) * h ** 3
    surf = 0.0
    for ax in range(3):
        for shift in (-1, 1):
            nb = np.roll(pore, shift, ax)
            wall = pore & ~nb
            if ax == d:
                edge = [slice(None)] * 3
                edge[ax] = -1 if shift == -1 else 0
                wall[tuple(edge)] = False
            surf += float(g2[wall].sum()) * h * h
    surf *= area_ratio
    return 2.0 * vol / surf if surf > 0 else float("nan")


def voxel_face_area(pore_mask, voxel_m):
    pore = np.asarray(pore_mask, bool)
    n = 0
    for ax in range(3):
        n += int(np.count_nonzero(pore != np.roll(pore, -1, axis=ax)))
    return n * voxel_m ** 2


def jca(freq, phi, sigma, alpha_inf, lam, lam_p, gas):
    """Dynamic density ρ_eff and bulk modulus K_eff (e^{jωt})."""
    w = 2.0 * math.pi * np.asarray(freq, float)
    rho0 = gas["density_kg_m3"]
    eta = gas["viscosity_Pa_s"]
    g = gas["gamma"]
    pr = gas["prandtl"]
    P0 = gas["pressure_Pa"]
    G = np.sqrt(1.0 + 1j * 4.0 * alpha_inf ** 2 * eta * rho0 * w / (sigma ** 2 * lam ** 2 * phi ** 2))
    rho_eff = alpha_inf * rho0 / phi * (1.0 + sigma * phi / (1j * w * rho0 * alpha_inf) * G)
    Gp = np.sqrt(1.0 + 1j * rho0 * w * pr * lam_p ** 2 / (16.0 * eta))
    K_eff = (g * P0 / phi) / (g - (g - 1.0) / (1.0 + 8.0 * eta / (1j * lam_p ** 2 * pr * w * rho0) * Gp))
    return rho_eff, K_eff


def absorption(freq, thickness_m, phi, sigma, alpha_inf, lam, lam_p, gas):
    """Normal-incidence absorption of a hard-backed layer."""
    f = np.asarray(freq, float)
    w = 2.0 * math.pi * f
    rho_eff, K_eff = jca(f, phi, sigma, alpha_inf, lam, lam_p, gas)
    Zc = np.sqrt(rho_eff * K_eff)
    kc = w * np.sqrt(rho_eff / K_eff)
    kc = np.where(np.imag(kc) > 0, -kc, kc)            # decaying wave for e^{j(ωt - kx)}
    Zc = np.where(np.real(Zc) < 0, -Zc, Zc)
    Zs = -1j * Zc / np.tan(kc * thickness_m)
    Z0 = gas["density_kg_m3"] * gas["sound_speed_m_s"]
    R = (Zs - Z0) / (Zs + Z0)
    alpha = 1.0 - np.abs(R) ** 2
    return {"freq_hz": f.tolist(), "alpha": np.clip(alpha, 0.0, 1.0).tolist(),
            "z_real": (np.real(Zs) / Z0).tolist(), "z_imag": (np.imag(Zs) / Z0).tolist()}


def nrc(freq, alpha):
    """Noise reduction coefficient: mean absorption at 250, 500, 1000 and
    2000 Hz, rounded to the nearest 0.05 (ASTM C423)."""
    f = np.asarray(freq, float)
    a = np.asarray(alpha, float)
    bands = [250.0, 500.0, 1000.0, 2000.0]
    if f[0] > bands[0] or f[-1] < bands[-1]:
        return None
    v = np.mean([np.interp(math.log10(b), np.log10(f), a) for b in bands])
    return round(v * 20.0) / 20.0
