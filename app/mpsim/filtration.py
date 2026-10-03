"""Filtration of an aerosol by the structure.

Particles are followed one by one through the solved Stokes velocity field, so
the efficiency comes out of the same geometry and the same flow solution as the
permeability, not out of a correlation:

    convection     the fluid velocity interpolated at the particle position
    inertia        a relaxation time tau = rho_p d_p^2 Cc / (18 mu); a heavy or
                   fast particle cannot follow a bend, which is impaction
    Brownian       a random step of sqrt(2 D dt) per axis with the
                   Stokes-Einstein diffusivity D = k_B T Cc / (3 pi mu d_p);
                   this is what captures the smallest particles
    interception   a particle is caught when its *surface* reaches the solid,
                   that is when its centre comes within d_p/2 of it, which is
                   what captures the largest particles

Because the first mechanism grows and the last falls with particle size, the
efficiency has a minimum: the most penetrating particle size (MPPS), the number
a filter is actually specified by.

Reported per size: single-pass efficiency, penetration, and the quality factor
-ln(P)/dp, which weighs the efficiency against the pressure drop it costs. The
pressure drop itself comes from the computed permeability (Darcy), so the
quality factor is consistent with the flow solve.

Not modelled: electrostatic capture, particle rebound, loading (the structure
does not change as particles deposit), and gravity, which is negligible for
the sub-micrometre sizes that decide a filter.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

K_B = 1.380649e-23


def cunningham(d_p_m, mean_free_path_m):
    """Slip correction: a particle smaller than the mean free path of the gas
    feels less drag than Stokes' law predicts."""
    kn = 2.0 * mean_free_path_m / np.maximum(d_p_m, 1e-12)
    return 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / np.maximum(kn, 1e-12)))


def _sample(field, pts):
    """Trilinear sample of a vector field at voxel coordinates (n, 3)."""
    c = pts.T
    return np.stack([ndimage.map_coordinates(field[..., k], c, order=1, mode="nearest")
                     for k in range(3)], axis=1)


def _seed(mask, flux, axis, n, rng):
    """Start positions on the inlet face, chosen in proportion to the flow that
    enters there - an aerosol arrives with the air, not uniformly over a face."""
    pm = np.take(mask, 0, axis=axis)
    pf = np.abs(np.take(flux, 0, axis=axis))
    cand = np.argwhere(pm & (pf > 0))
    if cand.shape[0] == 0:
        cand = np.argwhere(pm)
        if cand.shape[0] == 0:
            return None
        w = np.ones(cand.shape[0])
    else:
        w = np.array([pf[tuple(c)] for c in cand], float)
    w = w / w.sum()
    pick = rng.choice(cand.shape[0], size=n, replace=True, p=w)
    other = [k for k in range(3) if k != axis]
    pos = np.zeros((n, 3))
    pos[:, other[0]] = cand[pick, 0] + rng.random(n)
    pos[:, other[1]] = cand[pick, 1] + rng.random(n)
    pos[:, axis] = 0.5
    return pos


def track(u_field, mask, dist_vox, voxel_m, d_p_m, gas, U0, axis=0, n_particles=200,
          temperature_K=298.15, rho_p=1000.0, seed=0, max_steps=20000, keep_tracks=0):
    """Follow `n_particles` of one diameter and return the captured fraction."""
    mu = float(gas["viscosity_Pa_s"])
    lam = float(gas.get("mean_free_path_m") or 6.8e-8)
    cc = float(cunningham(d_p_m, lam))
    tau = rho_p * d_p_m ** 2 * cc / (18.0 * mu)
    D = K_B * temperature_K * cc / (3.0 * np.pi * mu * d_p_m)
    r_p_vox = 0.5 * d_p_m / voxel_m                      # interception radius

    shape = np.array(u_field.shape[:3])
    rng = np.random.default_rng(int(seed))
    pos = _seed(mask, u_field[..., axis], axis, n_particles, rng)
    if pos is None:
        return None
    vel = _sample(u_field, pos)

    umax = float(np.abs(u_field[mask]).max()) if mask.any() else 0.0
    if umax <= 0:
        return None
    # one step must not cross a voxel, must resolve the inertial relaxation and
    # must not let the Brownian step jump over the solid
    dt = 0.3 * voxel_m / umax
    if tau > 0:
        dt = min(dt, 0.3 * tau)
    if D > 0:
        dt = min(dt, 0.3 * (0.5 * voxel_m) ** 2 / D)
    hi = shape - 1e-6

    lateral = [k for k in range(3) if k != axis]
    alive = np.ones(len(pos), bool)
    captured = np.zeros(len(pos), bool)
    passed = np.zeros(len(pos), bool)
    returned = np.zeros(len(pos), bool)
    tracks = [[p.copy()] for p in pos[:keep_tracks]]
    caught_track = np.zeros(max(keep_tracks, 0), bool)
    for _ in range(int(max_steps)):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        uf = _sample(u_field, pos[idx])
        # exponential integrator for the drag: exact for a constant fluid
        # velocity over the step, and stable when tau is far below dt
        decay = np.exp(-dt / tau) if tau > 0 else 0.0
        v = uf + (vel[idx] - uf) * decay
        step = v * dt / voxel_m                           # metres -> voxels
        if D > 0:
            step += rng.normal(0.0, np.sqrt(2.0 * D * dt) / voxel_m, size=(idx.size, 3))
        nxt = pos[idx] + step
        vel[idx] = v
        # the RVE repeats sideways, so a particle leaving through a lateral face
        # re-enters on the opposite one instead of being counted as lost
        for k in lateral:
            nxt[:, k] = np.mod(nxt[:, k], shape[k])
        along = nxt[:, axis]
        # leaving downstream is penetration; diffusing back out of the inlet is
        # not - such a particle never entered the filter and must not be counted
        # as having passed through it
        out_far = along > hi[axis]
        out_back = along < 0.0
        gone = out_far | out_back
        inb = ~gone
        vox = np.clip(nxt, 0, hi).astype(int)
        # the distance transform gives the clearance to the solid, so one
        # lookup decides interception for any particle size
        clear = dist_vox[vox[:, 0], vox[:, 1], vox[:, 2]]
        hit = inb & (clear <= r_p_vox)
        pos[idx] = np.where(inb[:, None], nxt, pos[idx])
        for j, i in enumerate(idx):
            if i < keep_tracks and inb[j]:
                tracks[i].append(pos[i].copy())
                if hit[j]:
                    caught_track[i] = True
        captured[idx[hit]] = True
        passed[idx[out_far]] = True
        returned[idx[out_back]] = True
        alive[idx[hit | gone]] = False
    n = len(pos)
    resolved = int(captured.sum() + passed.sum())
    eff = float(captured.sum()) / resolved if resolved else None
    return {"diameter_m": float(d_p_m), "efficiency": eff,
            "penetration": (1.0 - eff) if eff is not None else None,
            "n_particles": int(n), "captured": int(captured.sum()), "passed": int(passed.sum()),
            "returned_to_inlet": int(returned.sum()), "still_inside": int(alive.sum()),
            "relaxation_time_s": float(tau), "diffusivity_m2_s": float(D), "cunningham": cc,
            "stokes_dt_s": float(dt),
            "tracks": [np.asarray(t) for t in tracks], "track_captured": caught_track.tolist()}


def analyse(u_field, mask, voxel_um, gas, options, K_m2=None, log=None, want_tracks=6):
    """Efficiency over a range of particle sizes, with the MPPS and quality factor."""
    t0 = time.time()
    voxel_m = float(voxel_um) * 1e-6
    axis = "xyz".index(options.get("direction", "x"))
    u = np.asarray(u_field, float)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {"error": "The pore space is empty"}, []

    # the solve uses a unit pressure gradient; scale it so the superficial
    # velocity through the sample is the face velocity the user specified
    U0 = float(options.get("face_velocity_m_s", 0.05))
    mean_u = float(u[..., axis].mean())                   # over the whole RVE = superficial
    if abs(mean_u) < 1e-300:
        return {"error": "The velocity field is zero"}, []
    u = u * (U0 / mean_u)

    dist_vox = ndimage.distance_transform_edt(mask).astype(np.float32)
    diameters = [float(d) for d in (options.get("diameters_um")
                 or [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])]
    n_particles = int(options.get("n_particles", 200))
    rho_p = float(options.get("particle_density_kg_m3", 1000.0))
    T = float(options.get("temperature_K", 298.15))
    thickness_m = mask.shape[axis] * voxel_m

    rows, paths = [], []
    for i, d_um in enumerate(sorted(diameters)):
        # a few trajectories are kept for every size: which size is the
        # interesting one only becomes clear once the efficiencies are known
        r = track(u, mask, dist_vox, voxel_m, d_um * 1e-6, gas, U0, axis=axis,
                  n_particles=n_particles, temperature_K=T, rho_p=rho_p, seed=1000 + i,
                  keep_tracks=want_tracks)
        if r is None:
            continue
        # Stokes number on the pore scale: the classic impaction parameter
        d_pore = 2.0 * float(np.percentile(dist_vox[mask], 50)) * voxel_m
        r["stokes_number"] = float(r["relaxation_time_s"] * U0 / d_pore) if d_pore > 0 else None
        r["peclet"] = float(U0 * d_pore / r["diffusivity_m2_s"]) if r["diffusivity_m2_s"] > 0 else None
        r["diameter_um"] = d_um
        for t, caught in zip(r.pop("tracks"), r.pop("track_captured")):
            if len(t) > 3:
                pts = (t + 0.5) * voxel_um
                keep = slice(None) if len(pts) <= 250 else slice(0, None, int(np.ceil(len(pts) / 250)))
                paths.append({"points_um": [[round(float(c), 4) for c in p] for p in pts[keep]],
                              "captured": bool(caught), "diameter_um": d_um})
        rows.append(r)
        if log:
            log(f"    filtration {d_um:g} µm: efficiency {100*r['efficiency']:.1f} % "
                f"(Stk {r['stokes_number']:.3g}, Pe {r['peclet']:.3g})")

    out = {"face_velocity_m_s": U0, "n_particles": n_particles, "direction": options.get("direction", "x"),
           "particle_density_kg_m3": rho_p, "thickness_um": thickness_m * 1e6,
           "by_size": rows, "seconds": time.time() - t0}
    if K_m2 and K_m2 > 0:
        dp = float(gas["viscosity_Pa_s"]) * U0 * thickness_m / K_m2
        out["pressure_drop_Pa"] = dp
        for r in rows:
            p = max(r["penetration"], 1e-9)
            r["quality_factor_1_Pa"] = float(-np.log(p) / dp) if dp > 0 else None
    eff = [(r["diameter_um"], r["efficiency"]) for r in rows if r["efficiency"] is not None]
    if eff:
        d_mpps, e_mpps = min(eff, key=lambda t: t[1])
        out["mpps_um"] = d_mpps
        out["mpps_efficiency"] = e_mpps
        out["worst_penetration"] = 1.0 - e_mpps
    return out, paths
