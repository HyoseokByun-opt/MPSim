"""Field lines of a vector field on the voxel grid.

A contour says how strong a field is somewhere; a field line says where the
transport actually goes - which channels carry the heat, the current or the
flow, and which parts of the structure are bypassed. The lines are integrated
here on the server so that one set of polylines feeds the interactive viewer,
the report figures and any later export; vtk.js has no stream tracer.

Integration is second order (midpoint rule) on the unit-direction field with
trilinear sampling of the components, which keeps a line inside a channel only
a few voxels wide. Lines start on the face where the field enters, and stop at
the far face, at a dead end (the local magnitude collapses), when they leave
the phase, or after a step budget.

Array convention as everywhere else: index [x, y, z], vectors as (nx, ny, nz, 3).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _sample(vec, pts):
    """Trilinear sample of the three components at voxel coordinates (n, 3)."""
    c = pts.T
    return np.stack([ndimage.map_coordinates(vec[..., k], c, order=1, mode="nearest")
                     for k in range(3)], axis=1)


def _unit(v, floor):
    n = np.sqrt((v ** 2).sum(1))
    out = np.zeros_like(v)
    ok = n > floor
    out[ok] = v[ok] / n[ok, None]
    return out, n


def trace(vec, mask, voxel_um, axis, n_lines=40, step=0.6, max_steps=None,
          max_points=220, seed=0, seed_quantile=0.5):
    """Field lines through the sample along `axis`.

    Returns (lines, summary). Each line is {"points_um": [[x, y, z], ...],
    "speed": [...]}; the summary reports how many lines reached the far face
    and the mean line length divided by the straight distance - a tortuosity
    measured on the field rather than on the geometry.
    """
    vec = np.asarray(vec, float)
    if vec.ndim != 4 or vec.shape[-1] != 3:
        raise ValueError("a vector field of shape (nx, ny, nz, 3) is required")
    shape = np.array(vec.shape[:3])
    mask = np.ones(tuple(shape), bool) if mask is None else np.asarray(mask, bool)
    if not mask.any():
        return [], {}
    mag = np.sqrt((vec ** 2).sum(-1))
    scale = float(np.percentile(mag[mask], 99))
    if not np.isfinite(scale) or scale <= 0:
        return [], {}
    floor = 1e-4 * scale
    # follow the field in the direction of net transport: a unit gradient may
    # drive the flux either way, and the line should still cross the sample.
    # Lines follow sign * vec, whose net component along `axis` is positive,
    # so they always enter at index 0 and leave at the far face. (Seeding at
    # the far face for a negative field, as before, sent every line out of the
    # box on its first step: the finite-volume flux, whose sign convention is
    # the opposite of PuMA's, drew no lines at all.)
    sign = 1.0 if float(vec[..., axis][mask].mean()) >= 0 else -1.0
    inlet = 0

    pm = np.take(mask, inlet, axis=axis)
    pg = np.take(mag, inlet, axis=axis)
    if not pm.any():
        return [], {}
    # seed where the field enters strongly; in a composite the matrix carries
    # almost nothing and lines seeded there would die at once
    thr = max(floor, float(np.quantile(pg[pm], seed_quantile)))
    cand = np.argwhere(pm & (pg > thr))
    if cand.shape[0] < max(4, n_lines // 4):
        cand = np.argwhere(pm & (pg > floor))
    if cand.shape[0] == 0:
        return [], {}
    rng = np.random.default_rng(seed)
    if cand.shape[0] > n_lines:
        cand = cand[rng.choice(cand.shape[0], n_lines, replace=False)]
    other = [k for k in range(3) if k != axis]
    pos = np.zeros((cand.shape[0], 3))
    pos[:, other[0]] = cand[:, 0] + 0.5
    pos[:, other[1]] = cand[:, 1] + 0.5
    pos[:, axis] = inlet + 0.5

    if max_steps is None:
        max_steps = int(6 * shape[axis] / step)
    hi = shape - 1e-6
    alive = np.ones(pos.shape[0], bool)
    tracks = [[p.copy()] for p in pos]
    speeds = [[float(s)] for s in np.sqrt((_sample(vec, pos) ** 2).sum(1))]
    for _ in range(int(max_steps)):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        u0, n0 = _unit(sign * _sample(vec, pos[idx]), floor)
        u1, n1 = _unit(sign * _sample(vec, pos[idx] + 0.5 * step * u0), floor)
        cand = pos[idx] + step * u1
        # A streamline of an incompressible flow cannot cross a pore wall, and
        # a flux line cannot leave its conducting phase: where the discrete
        # step would, the step is too long near the wall, so it is shortened
        # rather than the line being ended there. Leaving the box is different -
        # that is the line reaching a face, and it ends.
        good = np.zeros(idx.size, bool)
        for _halving in range(4):
            todo = ~good
            inside = np.all((cand >= 0) & (cand <= hi), axis=1)
            vox = np.clip(cand, 0, hi).astype(int)
            here = inside & mask[vox[:, 0], vox[:, 1], vox[:, 2]]
            good |= todo & here
            shrink = todo & ~here & inside
            if not shrink.any():
                break
            cand[shrink] = pos[idx][shrink] + 0.5 * (cand[shrink] - pos[idx][shrink])
        keep = (n0 > floor) & (n1 > floor) & good
        for j, i in enumerate(idx):
            if keep[j]:
                pos[i] = cand[j]
                tracks[i].append(pos[i].copy())
                speeds[i].append(float(n1[j]))
        alive[idx] = keep

    straight = float((shape[axis] - 1) * voxel_um)
    lines, lengths, through = [], [], []
    far = shape[axis] - 1.0
    for tr, sp in zip(tracks, speeds):
        if len(tr) < 3:
            continue
        pts = np.asarray(tr) * voxel_um
        seg = float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1)).sum())
        lengths.append(seg)
        # a line either crosses the sample or ends in a dead end; only the ones
        # that cross measure a transport path, so they are marked as such
        crossed = bool(abs(tr[-1][axis] - far) < 1.5)
        if crossed:
            through.append(seg)
        keep = slice(None) if len(pts) <= max_points else slice(0, None, int(np.ceil(len(pts) / max_points)))
        lines.append({"points_um": [[round(float(c), 4) for c in p] for p in pts[keep]],
                      "speed": [round(float(v), 6) for v in np.asarray(sp)[keep]],
                      "length_um": seg, "through": crossed})
    # only lines that cross measure a transport path; with none, a mean length
    # would read as a tortuosity below one, which is not a meaningful number
    summary = {"n_lines": len(lines), "n_through": len(through),
               "mean_length_um": float(np.mean(lengths)) if lengths else None,
               "tortuosity": float(np.mean(through) / straight) if through and straight else None,
               "straight_um": straight}
    return lines, summary
