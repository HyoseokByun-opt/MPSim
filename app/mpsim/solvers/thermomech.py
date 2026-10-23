"""Thermo-mechanical coupling of a film: the temperature field of heat flowing
through it, the thermal stress that field leaves from the stress-free
temperature, and the warpage of the film.

The film (x and y periodic, z its real thickness) lies between a hot face
(z = 0, the die side) and a cooler one. Three steps:

1. Heat. The steady conduction through the film between the two face
   temperatures is solved on the voxels (solvers/conduction.solve_film_z,
   with the interface and contact resistances of the thermal analysis); it
   is linear, so the temperature is T_top + (T_bottom - T_top) u of the unit
   drop's field u.
2. Stress. Every voxel takes the thermal strain of its own temperature
   from the stress-free one (the cure temperature): the expansion
   coefficient integrated over the temperature range, alpha1 below the
   material's Tg and alpha2 above, and the stiffness at its temperature (E
   below Tg, the rubbery E above); a voxel that ends below Tg counts the
   part of its path above Tg at alpha2 E_r / E (the stress the rubbery
   resin built up there). The film is held in its plane and free
   through its thickness (void layers above and below in the periodic
   voxel finite elements, solvers/fans.py), which gives the stress of the
   held film; a second solve at a unit in-plane strain gives each voxel's
   stress per unit strain.
3. Warpage. Classical laminate theory on the slab averages of those two
   solutions: the film (and a substrate under it, a homogeneous layer such
   as the die) is released in its plane and in bending until the in-plane
   force and the moment vanish, which gives the in-plane strain and the
   curvature; the stress of every voxel is then that of the held film plus
   its stress per unit strain times the released strain (eps0 + kappa z).

In-plane isotropy is assumed for the release (x and y averaged). Elastic:
no viscoelastic relaxation and no cure shrinkage. The mechanics do not act
back on the heat (thermoelastic heating and the change of shape are
negligible at these scales).
"""
from __future__ import annotations

import time

import numpy as np

from . import conduction as CD
from . import fans as FA


def thermal_strain(T, T_ref, alpha1, alpha2=None, tg=None):
    """Isotropic thermal strain from T_ref to T (arrays; ppm/K inputs):
    alpha integrated over the range, alpha1 below Tg, alpha2 above."""
    T = np.asarray(T, float)
    if tg is None or alpha2 is None:
        return alpha1 * (T - T_ref) * 1e-6

    def F(t):
        return np.where(t <= tg, alpha1 * t, alpha1 * tg + alpha2 * (t - tg))
    return (F(T) - F(np.asarray(T_ref, float))) * 1e-6


def principal_max(s):
    """Largest principal stress of a Voigt field (6, ...)."""
    sh = s.shape[1:]
    m = np.empty(sh + (3, 3), np.float32)
    m[..., 0, 0], m[..., 1, 1], m[..., 2, 2] = s[0], s[1], s[2]
    m[..., 1, 2] = m[..., 2, 1] = s[3]
    m[..., 0, 2] = m[..., 2, 0] = s[4]
    m[..., 0, 1] = m[..., 1, 0] = s[5]
    return np.linalg.eigvalsh(m.reshape(-1, 3, 3))[:, 2].reshape(sh).astype(np.float32)


def von_mises(s):
    return np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
                   + 3.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)).astype(np.float32)


def laminate(Q, sT, h, z, substrate=None, support="free"):
    """In-plane strain eps0 and curvature kappa (1/m) of the released film.
    Q, sT: slab averages per film layer of the biaxial stiffness (Pa per unit
    strain) and of the held film's thermal stress (Pa); h: layer thickness,
    z: layer centres (m, from the film's bottom face). substrate: dict with
    Q (Pa), sT (Pa, its held thermal stress) and t (m), under z = 0."""
    if support == "held":
        return 0.0, 0.0
    A = float(np.sum(Q) * h)
    B = float(np.sum(Q * z) * h)
    D = float(np.sum(Q * z * z) * h)
    N = float(np.sum(sT) * h)
    M = float(np.sum(sT * z) * h)
    if substrate is not None:
        t, Qs, ss = substrate["t"], substrate["Q"], substrate["sT"]
        A += Qs * t
        B += Qs * (-0.5 * t * t)
        D += Qs * (t ** 3 / 3.0)
        N += ss * t
        M += ss * (-0.5 * t * t)
    eps0, kappa = np.linalg.solve(np.array([[A, B], [B, D]]), -np.array([N, M]))
    return float(eps0), float(kappa)


def run(labels, table, h_um, opts, k_vals, rpair=None, cmask=None, rcpair=None, log=None, progress=None,
        should_stop=None, tol=1e-6, keep_fields=True, compare_uniform=True, callback=None):
    """labels: film labels (nx, ny, nz); table: label table (props E [GPa],
    nu, alpha [ppm/K], optional tg [C], E_r [GPa], alpha2 [ppm/K]); opts:
    T_ref, T_bottom, T_top (C), support ("free", "substrate", "held"),
    substrate {E, nu, alpha, thickness_um, name}, span_mm. k_vals,
    rpair, cmask, rcpair: the thermal analysis's conduction inputs."""
    log = log or (lambda *a: None)
    progress = progress or (lambda *a: None)
    t0 = time.time()
    h = float(h_um) * 1e-6
    nx, ny, nz = labels.shape
    Tref, Tb, Tt = float(opts["T_ref"]), float(opts["T_bottom"]), float(opts["T_top"])
    # 1. heat: the unit drop's field, scaled to the face temperatures
    progress("heat", 0.0)
    r = CD.solve_film_z(labels, np.asarray(k_vals, float), rpair=rpair, cmask=cmask, rcpair=rcpair, tol=1e-8,
                        should_stop=should_stop,
                        callback=(lambda it, res: callback("Heat flow through the film", it, res, 1e-8)) if callback else None)
    heat = {"iterations": int(r["iterations"]), "residual": float(r["residual"]), "converged": bool(r["converged"])}
    u = r["u"]
    T = (Tt + (Tb - Tt) * u).astype(np.float32)
    q_through = float(r["k_eff"]) * (Tb - Tt) / (nz * h)            # W/m^2
    k_eff = float(r["k_eff"])
    del r, u
    out = _mechanics(labels, table, h, T, Tref, opts, log, progress, should_stop, tol, keep_fields, callback)
    out.update({"heat": heat, "k_eff": k_eff, "q_W_m2": q_through, "T_min": float(T.min()), "T_max": float(T.max()),
                "T_mean": float(T.mean())})
    if keep_fields:
        out["T"] = T
    # the same film at its mean temperature: how much the gradient adds
    if compare_uniform and abs(Tb - Tt) > 1e-9:
        progress("uniform", 0.9)
        u_out = _mechanics(labels, table, h, np.full(T.shape, out["T_mean"], np.float32), Tref, opts, log,
                           lambda *a: None, should_stop, tol, False,
                           (lambda nm, it, res, tg: callback(nm + " (uniform temperature)", it, res, tg)) if callback else None)
        out["uniform"] = {k: u_out[k] for k in ("kappa_1_m", "warpage_um", "eps0", "stress_stats", "above_tg")}
    out["seconds"] = time.time() - t0
    return out


def _mechanics(labels, table, h, T, Tref, opts, log, progress, should_stop, tol, keep_fields, callback=None):
    nx, ny, nz = labels.shape
    nlab = len(table)
    props = [t["props"] for t in table]
    E = np.array([float(p.get("E") or 0.0) for p in props])
    nu = np.array([float(p.get("nu") or 0.3) for p in props])
    a1 = np.array([float(p.get("alpha") or 0.0) for p in props])
    tg = [p.get("tg") if (p.get("tg") not in (None, "") and p.get("E_r") and p.get("alpha2")) else None for p in props]
    # glassy and rubbery variants of every label; a void layer above and below
    E_ext = list(E) + [float(props[i].get("E_r") or E[i]) for i in range(nlab)] + [0.0]
    nu_ext = list(nu) + [min(float(nu[i]), 0.49) for i in range(nlab)] + [0.2]
    lab = labels.astype(np.int32)
    above = np.zeros(labels.shape, bool)
    eig = np.zeros(labels.shape, np.float64)
    for i in range(nlab):
        m = labels == i
        if not m.any():
            continue
        if tg[i] is not None:
            hot = m & (T > float(tg[i]))
            above |= hot
            lab[hot] = nlab + i
            # a voxel that ends rubbery spent its whole path rubbery: alpha2
            # above Tg at the rubbery modulus. One that ends glassy was
            # rubbery for the part of its path above Tg, which adds the
            # stress E_r alpha2 per kelvin there: at its glassy modulus that
            # is an expansion coefficient of alpha2 E_r / E (incremental
            # thermo-elasticity, sigma = integral of E alpha dT)
            a2 = float(props[i]["alpha2"])
            cold = m & ~hot
            eig[hot] = thermal_strain(T[hot], Tref, a1[i], a2, float(tg[i]))
            eig[cold] = thermal_strain(T[cold], Tref, a1[i], a2 * float(props[i]["E_r"]) / max(E[i], 1e-30),
                                       float(tg[i]))
        else:
            eig[m] = thermal_strain(T[m], Tref, a1[i])
    void = 2 * nlab
    npad = 2
    labp = np.full((nx, ny, nz + 2 * npad), void, np.int32)
    labp[:, :, npad:npad + nz] = lab
    eigp = np.zeros(labp.shape)
    eigp[:, :, npad:npad + nz] = eig
    solid_E = np.array(E_ext)
    sol = FA.FANS(labp, solid_E, np.array(nu_ext), None, 1e-4)
    # a unit equi-biaxial in-plane strain (stiffness per layer) and the held
    # film's thermal stress
    progress("stiffness", 0.2)
    cb1 = (lambda it, res: callback("Film stiffness (unit in-plane strain)", it, res, tol)) if callback else None
    cb2 = (lambda it, res: callback("Thermal stress of the held film", it, res, tol)) if callback else None
    _, st1, sb = sol.solve(np.array([1.0, 1.0, 0, 0, 0, 0]), 0.0, tol, 20000, cb1, should_stop, keep_stress=True)
    progress("thermal stress", 0.5)
    _, st2, sT = sol.solve(np.zeros(6), 0.0, tol, 20000, cb2, should_stop, keep_stress=True, eig=eigp)
    del sol
    sb = sb[:, :, :, npad:npad + nz] * 1e9                     # Pa per unit strain
    sT = sT[:, :, :, npad:npad + nz] * 1e9                     # Pa
    Q = 0.5 * (sb[0] + sb[1]).mean(axis=(0, 1))
    sTz = 0.5 * (sT[0] + sT[1]).mean(axis=(0, 1))
    z = (np.arange(nz) + 0.5) * h
    sup = opts.get("support", "free")
    subs = None
    if sup == "substrate":
        s_ = opts["substrate"]
        Qs = float(s_["E"]) * 1e9 / (1.0 - float(s_["nu"]))
        th_s = thermal_strain(float(opts["T_bottom"]), Tref, float(s_["alpha"]))
        subs = {"t": float(s_["thickness_um"]) * 1e-6, "Q": Qs, "sT": -Qs * float(th_s)}
    eps0, kappa = laminate(Q, sTz, h, z, subs, sup)
    # every voxel: held stress + stress per unit strain x released strain
    zz = z[None, None, :]
    s = sT + sb * (eps0 + kappa * zz)[None]
    s *= 1e-6                                                   # MPa
    vm = von_mises(s)
    p1 = principal_max(s)
    span = float(opts.get("span_mm") or 10.0) * 1e-3
    warp = kappa * span * span / 8.0 * 1e6                      # µm, centre to edge
    stats = {}
    for i in np.unique(labels):
        m = labels == i
        stats[int(i)] = {"name": table[i]["name"], "vm_mean": float(vm[m].mean()), "vm_p999": float(np.percentile(vm[m], 99.9)),
                         "p1_max": float(np.percentile(p1[m], 99.9)), "p1_mean": float(p1[m].mean()),
                         "szz_p999": float(np.percentile(np.abs(s[2][m]), 99.9)),
                         "T_mean": float(T[m].mean()), "T_max": float(T[m].max()),
                         "above_tg": float(np.mean(above[m])) if tg[int(i)] is not None else None}
    s_sub = None
    if subs is not None:
        ts = subs["t"]
        th_s = -subs["sT"] / subs["Q"]
        s_sub = {"top_MPa": subs["Q"] * (eps0 - th_s) * 1e-6, "bottom_MPa": subs["Q"] * (eps0 - kappa * ts - th_s) * 1e-6}
    # the film's mean in-plane stress, slab by slab (MPa), after the release
    prof = (0.5 * (s[0] + s[1])).mean(axis=(0, 1))
    out = {"eps0": eps0, "kappa_1_m": kappa, "radius_m": (1.0 / kappa) if abs(kappa) > 1e-30 else None,
           "warpage_um": warp, "span_mm": span * 1e3, "stress_stats": stats, "substrate_stress": s_sub,
           "above_tg": float(np.mean(above[np.isin(labels, [i for i in range(nlab) if tg[i] is not None])]))
           if any(t is not None for t in tg) else None,
           "profile": {"z_um": (z * 1e6).tolist(), "Q_GPa": (Q * 1e-9).tolist(), "s_held_MPa": (sTz * 1e-6).tolist(),
                       "s_MPa": prof.tolist(), "T_C": T.mean(axis=(0, 1)).tolist()},
           "iterations": [st1["iterations"], st2["iterations"]],
           "converged": [bool(st1["converged"]), bool(st2["converged"])],
           "residual": [float(st1["residual"]), float(st2["residual"])]}
    if keep_fields:
        out["fields"] = {"vm": vm, "p1": p1, "szz": s[2].astype(np.float32), "sxx": s[0].astype(np.float32),
                         "above": above}
    log(f"    thermo-mechanical: κ {kappa:.4g} 1/m (warpage {warp:.4g} µm over {span * 1e3:g} mm), in-plane strain "
        f"{eps0:.4g}; FANS {st1['iterations']} + {st2['iterations']} iterations")
    return out
