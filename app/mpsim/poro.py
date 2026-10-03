"""Pore-space analyses - the counterpart of a porous-media analysis module.

All image operations come from established open-source packages:

* PoreSpy (Gostick et al., JOSS 2019)
    morphological drainage (mercury intrusion porosimetry and capillary
    pressure curves, Hilpert & Miller 2001), SNOW2 pore-network extraction
    (Gostick 2017)
* SciPy ndimage
    Euclidean distance transform and connected components

Physics on top of the image results is closed-form and cited where used:
Young-Laplace / Washburn capillary pressure, the kinetic-theory mean free
path, Knudsen diffusivity and the Bosanquet interpolation.

Conventions: the RVE is periodic for transport, but intrusion experiments
invade a sample from its faces, so the porosimetry here is non-periodic with
inlets on the faces - as in a physical measurement.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import ndimage

from .morphology import _ps, _quiet

KB = 1.380649e-23
R_GAS = 8.314462618


# =========================================================================
# porosity by type
# =========================================================================
def porosity_types(void_mask):
    """Total, open (part of a cluster that spans the periodic cell) and
    closed porosity."""
    from .solvers.conduction import percolating_mask, wrapping_axes
    void = np.asarray(void_mask, bool)
    total = float(void.mean())
    open_mask = percolating_mask(void)
    return {"total": total, "open": float(open_mask.mean()),
            "closed": float((void & ~open_mask).mean()),
            "connected_axes": dict(zip("xyz", wrapping_axes(void)))}, open_mask


def fraction_profiles(labels, n_labels, voxel_um):
    """Volume fraction of every label along x, y and z (slice averages)."""
    lab = np.asarray(labels)
    out = {"position_um": {}, "profiles": {}}
    for ax, name in enumerate("xyz"):
        n = lab.shape[ax]
        out["position_um"][name] = ((np.arange(n) + 0.5) * voxel_um).tolist()
        prof = []
        for li in range(n_labels):
            m = (lab == li)
            other = tuple(a for a in range(3) if a != ax)
            prof.append(m.mean(axis=other).tolist())
        out["profiles"][name] = prof
    return out


# =========================================================================
# capillary physics
# =========================================================================
def capillary_pressure(diameter_m, gamma, theta_deg):
    """Young-Laplace pressure of a spherical meniscus in a pore of diameter d:
    P = 4 γ |cos θ| / d (Washburn form)."""
    return 4.0 * gamma * abs(math.cos(math.radians(theta_deg))) / np.maximum(diameter_m, 1e-300)


def _inscribed_radius_vox(dt):
    """Inscribed radius from the EDT (centre-to-centre distance to the nearest
    solid voxel). The bias depends on how the pore is aligned with the grid:
    a 5-voxel slot gives dt = 3 (true 2.5), a 16-voxel channel centred between
    voxels gives max dt = 7.6 (true 8). Measured on both, the plain EDT is
    unbiased to within a tenth of a voxel on average, so it is used as is."""
    return np.maximum(dt, 0.5)


def max_through_sphere(void, dt, axis):
    """Radius [voxels] of the largest sphere whose centre can travel from the
    face at index 0 to the opposite face along `axis` (the bubble-point pore).
    Bisection over the distinct distance-transform values; each test is one
    connected-component labelling of {dt >= r}."""
    vals = np.unique(dt[void])
    vals = vals[vals > 0]
    if vals.size == 0:
        return None

    def through(r):
        centres = dt >= r
        lab, n = ndimage.label(centres)
        if n == 0:
            return False
        a = np.unique(np.take(lab, 0, axis=axis))
        b = np.unique(np.take(lab, lab.shape[axis] - 1, axis=axis))
        return np.intersect1d(a[a > 0], b[b > 0]).size > 0
    if not through(vals[0]):
        return None
    lo, hi = 0, vals.size - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if through(vals[mid]):
            lo = mid
        else:
            hi = mid - 1
    return float(vals[lo])


def porosimetry(void_mask, voxel_um, gamma=0.485, theta_deg=140.0, steps=40,
                directions="xyz", log=None, want_field=True):
    """Intrusion porosimetry of the pore space with PoreSpy's morphological
    drainage (sphere insertion from all six faces) and the through-pore
    (bubble-point) diameter per direction.

    Returns (result dict, diameter map in um or None). The diameter map holds,
    for every intruded voxel, the diameter of the largest sphere that reached
    it from outside - the pore-entry (throat) size that controls intrusion.
    """
    ps = _ps()
    t0 = time.time()
    void = np.asarray(void_mask, bool)
    h_m = voxel_um * 1e-6
    dt = ndimage.distance_transform_edt(void)
    r_vox = _inscribed_radius_vox(dt)
    pc = np.where(void, capillary_pressure(2.0 * r_vox * h_m, gamma, theta_deg), np.inf)
    fin = pc[void]
    res = {"fluid_gamma_N_m": gamma, "contact_angle_deg": theta_deg}
    if fin.size == 0:
        return res, None
    pmin, pmax = float(fin.min()), float(fin.max())
    pressures = np.geomspace(pmin * 0.999, pmax * 1.001, int(steps)).tolist()
    inlets = np.zeros_like(void)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = 0
        inlets[tuple(sl)] = True
        sl[ax] = -1
        inlets[tuple(sl)] = True
    inlets &= void
    with _quiet():
        dr = ps.simulations.drainage(im=void, pc=pc, inlets=inlets, steps=pressures)
    P = np.asarray(dr.pc, float)
    Snw = np.asarray(dr.snwp, float)
    ok = np.isfinite(P) & np.isfinite(Snw) & (P > 0)
    P, Snw = P[ok], Snw[ok]
    order = np.argsort(P)
    P, Snw = P[order], np.maximum.accumulate(Snw[order])
    d_um = 4.0 * gamma * abs(math.cos(math.radians(theta_deg))) / P * 1e6
    res["intrusion"] = {"pressure_Pa": P.tolist(), "saturation": Snw.tolist(), "diameter_um": d_um.tolist()}
    # pore-entry size distribution: volume intruded between successive pressures
    if P.size >= 2:
        dS = np.diff(np.concatenate([[0.0], Snw]))
        logd = np.log10(d_um)
        width = np.abs(np.gradient(logd)) if logd.size > 1 else np.ones_like(logd)
        res["entry_size_distribution"] = {
            "diameter_um": d_um.tolist(), "dS": dS.tolist(),
            "dS_dlogd": (dS / np.maximum(width, 1e-12)).tolist()}
        if Snw[-1] > 0.5:
            res["d50_um"] = float(10 ** np.interp(0.5, Snw, logd))
        if Snw[-1] > 0.1:
            res["d10_um"] = float(10 ** np.interp(0.1, Snw, logd))
        if Snw[-1] > 0.9:
            res["d90_um"] = float(10 ** np.interp(0.9, Snw, logd))
        res["threshold_pressure_Pa"] = float(P[np.argmax(Snw > 0.01)]) if (Snw > 0.01).any() else None
    res["max_intrusion"] = float(Snw[-1]) if Snw.size else 0.0
    # extrusion: the wetting phase re-enters from the faces as the pressure
    # falls (PoreSpy morphological imbibition); the gap to the intrusion
    # curve is the ink-bottle hysteresis
    try:
        with _quiet():
            im_ = ps.simulations.imbibition(im=void, pc=pc, inlets=inlets, steps=pressures)
        Pe = np.asarray(im_.pc, float)
        Se = np.asarray(im_.snwp, float)
        ok = np.isfinite(Pe) & np.isfinite(Se) & (Pe > 0)
        Pe, Se = Pe[ok], Se[ok]
        order = np.argsort(Pe)
        if Pe.size >= 2:
            res["extrusion"] = {"pressure_Pa": Pe[order].tolist(), "saturation": Se[order].tolist()}
    except Exception as e:                                          # noqa: BLE001
        res["extrusion_error"] = f"{type(e).__name__}: {e}"
    # through-pores
    bt = {}
    for d in directions:
        ax = "xyz".index(d)
        r = max_through_sphere(void, dt, ax)
        if r is None:
            bt[d] = None
            continue
        dia_m = 2.0 * float(_inscribed_radius_vox(np.array(r))) * h_m
        bt[d] = {"diameter_um": dia_m * 1e6,
                 "pressure_fluid_Pa": float(capillary_pressure(dia_m, gamma, theta_deg)),
                 "bubble_point_water_Pa": float(capillary_pressure(dia_m, 0.0728, 0.0)),
                 "bubble_point_ipa_Pa": float(capillary_pressure(dia_m, 0.0217, 0.0))}
    res["through_pore"] = bt
    field = None
    if want_field:
        imp = np.asarray(dr.im_pc, float)
        inv = void & np.isfinite(imp) & (imp > 0)
        field = np.zeros(void.shape, np.float32)
        field[inv] = (4.0 * gamma * abs(math.cos(math.radians(theta_deg))) / imp[inv] * 1e6).astype(np.float32)
    res["seconds"] = time.time() - t0
    if log:
        log(f"    porosimetry: {P.size} pressure steps, maximum intrusion {100*res['max_intrusion']:.1f} %, "
            f"{time.time()-t0:.1f} s")
    return res, field


# =========================================================================
# pore network
# =========================================================================
def pore_network(void_mask, voxel_um, log=None, max_view=20000):
    """SNOW2 pore network (PoreSpy): pores as watershed regions of the
    distance transform, throats as the shared faces between them."""
    ps = _ps()
    t0 = time.time()
    void = np.asarray(void_mask, bool)
    with _quiet():
        snow = ps.networks.snow2(void.astype(np.int32), voxel_size=voxel_um, boundary_width=0)
    net = snow.network
    coords = np.asarray(net["pore.coords"], float) + 0.5 * voxel_um
    conns = np.asarray(net["throat.conns"], np.int64).reshape(-1, 2)
    n_p, n_t = coords.shape[0], conns.shape[0]
    zc = np.bincount(conns.ravel(), minlength=n_p) if n_t else np.zeros(n_p, int)
    pd_ins = np.asarray(net.get("pore.inscribed_diameter", np.zeros(n_p)), float)
    pd_eq = np.asarray(net.get("pore.equivalent_diameter", np.zeros(n_p)), float)
    td_ins = np.asarray(net.get("throat.inscribed_diameter", np.zeros(n_t)), float)
    td_eq = np.asarray(net.get("throat.equivalent_diameter", np.zeros(n_t)), float)
    tl = np.asarray(net.get("throat.total_length", net.get("throat.direct_length", np.zeros(n_t))), float)

    def dist(v, bins=16, weights=None):
        v = np.asarray(v, float)
        ok = np.isfinite(v) & (v > 0)
        if not ok.any():
            return None
        lo, hi = float(v[ok].min()), float(v[ok].max())
        if hi <= lo:
            hi = lo * 1.01 + 1e-12
        hist, edges = np.histogram(v[ok], bins=bins, range=(lo, hi),
                                   weights=None if weights is None else np.asarray(weights, float)[ok])
        return {"center": (0.5 * (edges[1:] + edges[:-1])).tolist(),
                "fraction": (hist / max(hist.sum(), 1e-300)).tolist(),
                "mean": float(np.average(v[ok], weights=None if weights is None else np.asarray(weights, float)[ok])),
                "d50": float(np.median(v[ok]))}
    res = {
        "n_pores": int(n_p), "n_throats": int(n_t),
        "coordination_mean": float(zc.mean()) if n_p else 0.0,
        "coordination_max": int(zc.max()) if n_p else 0,
        "isolated_pores": int((zc == 0).sum()),
        "pore_density_per_mm3": float(n_p / (np.prod(void.shape) * (voxel_um * 1e-3) ** 3)),
        "pore_inscribed_diameter_um": dist(pd_ins),
        "pore_equivalent_diameter_um": dist(pd_eq),
        "throat_inscribed_diameter_um": dist(td_ins),
        "throat_equivalent_diameter_um": dist(td_eq),
        "throat_length_um": dist(tl),
        "coordination_distribution": {"center": list(range(int(zc.max()) + 1)) if n_p else [],
                                      "fraction": (np.bincount(zc) / max(n_p, 1)).tolist() if n_p else []},
        "seconds": time.time() - t0,
    }
    view = None
    if n_p:
        keep = np.arange(n_p)
        if n_p > max_view:
            keep = np.sort(np.random.default_rng(0).choice(n_p, max_view, replace=False))
        idx = -np.ones(n_p, np.int64)
        idx[keep] = np.arange(keep.size)
        tsel = (idx[conns[:, 0]] >= 0) & (idx[conns[:, 1]] >= 0) if n_t else np.zeros(0, bool)
        view = {"pores": {"xyz_um": np.round(coords[keep], 4).tolist(),
                          "diameter_um": np.round(pd_ins[keep], 5).tolist(),
                          "coordination": zc[keep].tolist()},
                "throats": {"conns": idx[conns[tsel]].tolist() if n_t else [],
                            "diameter_um": np.round(td_ins[tsel], 5).tolist() if n_t else []}}
    if log:
        log(f"    pore network: {n_p} pores, {n_t} throats, mean coordination {res['coordination_mean']:.2f} "
            f"({time.time()-t0:.1f} s)")
    return res, view


# =========================================================================
# gas kinetics
# =========================================================================
def mean_free_path(T, p, d_molecule_m):
    return KB * T / (math.sqrt(2.0) * math.pi * d_molecule_m ** 2 * p)


def knudsen(d_pore_m, gas, T):
    """Knudsen number, Knudsen diffusivity D_K = d/3 · sqrt(8RT/(πM)) and the
    Bosanquet pore diffusivity 1/D = 1/D_m + 1/D_K."""
    lam = mean_free_path(T, gas["pressure_Pa"], gas["molecule_diameter_nm"] * 1e-9)
    kn = lam / max(d_pore_m, 1e-300)
    v_mean = math.sqrt(8.0 * R_GAS * T / (math.pi * gas["molar_mass_g_mol"] * 1e-3))
    dk = d_pore_m / 3.0 * v_mean
    dm = gas["diffusivity_m2_s"]
    return {"mean_free_path_nm": lam * 1e9, "knudsen_number": kn, "d_pore_um": d_pore_m * 1e6,
            "D_knudsen_m2_s": dk, "D_molecular_m2_s": dm,
            "D_bosanquet_m2_s": 1.0 / (1.0 / dm + 1.0 / dk),
            "regime": "molecular" if kn < 0.01 else "transition" if kn < 10 else "Knudsen"}


def gas_conductivity_factor(kn, gamma=1.4, prandtl=0.71, accommodation=1.0):
    """Kaganer's rarefied-gas conduction factor k_g/k_g0 = 1/(1 + 2 β Kn),
    β = 2(2 - a)/a · γ/(γ + 1) / Pr (a: thermal accommodation coefficient)."""
    beta = 2.0 * (2.0 - accommodation) / accommodation * gamma / (gamma + 1.0) / prandtl
    return 1.0 / (1.0 + 2.0 * beta * kn), beta
