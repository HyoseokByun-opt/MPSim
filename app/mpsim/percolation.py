"""Percolation paths and geodesic tortuosity of a phase.

For every requested direction the phase is separated into

    spanning     part of a cluster that touches the inlet and the outlet face,
                 i.e. a path through the sample exists
    dead end     connected to one face only
    isolated     touching neither face

and, inside the spanning cluster, the shortest path from face to face is
measured with a geodesic (Dijkstra) distance transform. The geometric
tortuosity is the mean shortest-path length of the outlet voxels divided by the
straight distance - the quantity a percolation-path analysis reports, and a
lower bound for the diffusive tortuosity factor of the same structure.

Connectivity follows the physics: a fluid cannot pass through a corner, so the
pore space uses face connectivity (6), whereas solid particles touching at an
edge or corner do conduct in a voxel finite-element model, so solid phases use
full connectivity (26).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

SOLID, ISOLATED, DEAD_END, SPANNING = 0.0, 1.0, 2.0, 3.0


def _structure(full):
    return np.ones((3, 3, 3), bool) if full else ndimage.generate_binary_structure(3, 1)


def classify(mask, axis, full_connectivity=False):
    """Label map of the phase for one direction (see module docstring)."""
    mask = np.asarray(mask, bool)
    out = np.zeros(mask.shape, np.float32)
    if not mask.any():
        return out, None
    lab, n = ndimage.label(mask, structure=_structure(full_connectivity))
    if n == 0:
        return out, None
    first = np.unique(np.take(lab, 0, axis=axis))
    last = np.unique(np.take(lab, lab.shape[axis] - 1, axis=axis))
    first = set(int(v) for v in first if v > 0)
    last = set(int(v) for v in last if v > 0)
    span = first & last
    touch = first | last
    kind = np.zeros(n + 1, np.float32)
    for i in range(1, n + 1):
        kind[i] = SPANNING if i in span else (DEAD_END if i in touch else ISOLATED)
    out = kind[lab]
    spanning = np.isin(lab, list(span)) if span else None
    return out, spanning


def geodesic(spanning, axis, voxel_um):
    """Shortest path length from the inlet face through the spanning cluster.

    Returns (distance map in um with NaN outside the cluster, tortuosity).
    """
    from skimage.graph import MCP_Geometric
    costs = np.where(spanning, 1.0, np.inf)
    sl = [slice(None)] * 3
    sl[axis] = 0
    inlet = np.zeros(spanning.shape, bool)
    inlet[tuple(sl)] = spanning[tuple(sl)]
    starts = np.argwhere(inlet)
    if starts.size == 0:
        return None, None
    mcp = MCP_Geometric(costs, fully_connected=True)
    cum, _ = mcp.find_costs([tuple(s) for s in starts])
    sl[axis] = spanning.shape[axis] - 1
    outlet = np.zeros(spanning.shape, bool)
    outlet[tuple(sl)] = spanning[tuple(sl)]
    ends = cum[outlet]
    ends = ends[np.isfinite(ends)]
    if ends.size == 0:
        return None, None
    straight = spanning.shape[axis] - 1
    tau = float(ends.mean() / straight)
    dist = np.where(np.isfinite(cum), cum * voxel_um, np.nan).astype(np.float32)
    return dist, tau


def _spanning_centres(dt, r, axis):
    """Connected set of sphere centres of radius r that reaches both faces."""
    centres = dt >= r
    if not centres.any():
        return None, 0
    lab, n = ndimage.label(centres, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return None, 0
    first = set(int(v) for v in np.unique(np.take(lab, 0, axis=axis)) if v > 0)
    last = set(int(v) for v in np.unique(np.take(lab, lab.shape[axis] - 1, axis=axis)) if v > 0)
    span = sorted(first & last)
    if not span:
        return None, 0
    return np.isin(lab, span), len(span)


def _shortest_path(region, axis, voxel_um):
    """Shortest path through `region` from the inlet face to the outlet face.

    Returns (polyline in um, length in um, tortuosity) using a geodesic
    (Dijkstra) traversal with 26 neighbours and Euclidean step costs.
    """
    from skimage.graph import MCP_Geometric
    sl = [slice(None)] * 3
    sl[axis] = 0
    inlet = np.zeros(region.shape, bool)
    inlet[tuple(sl)] = region[tuple(sl)]
    starts = np.argwhere(inlet)
    if starts.size == 0:
        return None, None, None
    mcp = MCP_Geometric(np.where(region, 1.0, np.inf), fully_connected=True)
    cum, _ = mcp.find_costs([tuple(s) for s in starts])
    sl[axis] = region.shape[axis] - 1
    outlet = np.zeros(region.shape, bool)
    outlet[tuple(sl)] = region[tuple(sl)]
    cand = np.argwhere(outlet & np.isfinite(cum))
    if cand.size == 0:
        return None, None, None
    end = cand[np.argmin([cum[tuple(c)] for c in cand])]
    path = np.asarray(mcp.traceback(tuple(end)), float)
    length = float(cum[tuple(end)]) * voxel_um
    straight = (region.shape[axis] - 1) * voxel_um
    poly = [[float((p[0] + 0.5) * voxel_um), float((p[1] + 0.5) * voxel_um), float((p[2] + 0.5) * voxel_um)]
            for p in path]
    return poly, length, (length / straight if straight else None)


def _max_radius(dt, axis):
    """Largest sphere radius [voxels] whose centre can still cross the sample.

    Bisection over the distinct distance-transform values; each test is one
    connected-component labelling of {dt >= r}.
    """
    vals = np.unique(dt[dt > 0])
    if vals.size == 0 or _spanning_centres(dt, float(vals[0]), axis)[0] is None:
        return None
    lo, hi = 0, vals.size - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _spanning_centres(dt, float(vals[mid]), axis)[0] is not None:
            lo = mid
        else:
            hi = mid - 1
    return float(vals[lo])


def _decimate(poly, max_points):
    if not poly or max_points <= 0 or len(poly) <= max_points:
        return poly
    step = int(np.ceil(len(poly) / max_points))
    out = poly[::step]
    if (len(poly) - 1) % step:
        out.append(poly[-1])
    return out


def sphere_paths(mask, axis, voxel_um, dt=None, levels=8, path_rows=3, max_points=300):
    """Which particle sizes pass through the sample, largest first.

    A rigid sphere of radius r travels wherever its centre stays at least r
    away from the solid, so the reachable centres are {dt >= r}; the sphere
    crosses the sample when that set has a component touching both faces. The
    first row is therefore the largest particle that passes at all, and every
    row carries the route its centre takes - the path itself, not only the
    statement that one exists.
    """
    mask = np.asarray(mask, bool)
    if dt is None:
        dt = ndimage.distance_transform_edt(mask)
    r_max = _max_radius(dt, axis)
    if r_max is None:
        return []
    # the exact value matters: rounding the top radius up by a fraction of a
    # voxel empties {dt >= r} and drops the largest sphere from the list
    radii = np.linspace(r_max, 0.5, max(2, levels)) if r_max > 0.5 else np.array([r_max])
    rows = []
    for i, r in enumerate(radii):
        region, n_paths = _spanning_centres(dt, float(r), axis)
        if region is None:
            continue
        poly, length, tau = _shortest_path(region, axis, voxel_um)
        rows.append({"diameter_um": 2.0 * float(r) * voxel_um, "radius_voxels": float(r),
                     "n_paths": int(n_paths), "path_length_um": length, "tortuosity": tau,
                     "centre_fraction": float(region.sum() / max(mask.sum(), 1)),
                     # only the widest routes are drawn; the rest would only
                     # make the result file larger
                     "polyline_um": _decimate(poly, max_points) if (poly and i < path_rows) else None})
    return rows


def analyse(mask, voxel_um, directions="xyz", full_connectivity=False, want_fields=True,
            want_paths=False, path_levels=8):
    """Percolation classes, geodesic tortuosity and passing particle sizes."""
    t0 = time.time()
    mask = np.asarray(mask, bool)
    res = {"volume_fraction": float(mask.mean()), "connectivity": 26 if full_connectivity else 6,
           "by_direction": {}}
    fields = {}
    dt = ndimage.distance_transform_edt(mask) if want_paths and mask.any() else None
    for d in directions:
        ax = "xyz".index(d)
        cls, spanning = classify(mask, ax, full_connectivity)
        entry = {"percolates": bool(spanning is not None and spanning.any())}
        if want_paths and dt is not None:
            try:
                entry["passable_spheres"] = sphere_paths(mask, ax, voxel_um, dt=dt, levels=path_levels)
            except Exception as e:                                   # noqa: BLE001
                entry["sphere_path_error"] = f"{type(e).__name__}: {e}"
        if spanning is not None and spanning.any():
            entry["spanning_fraction"] = float(spanning.sum() / max(mask.sum(), 1))
            try:
                dist, tau = geodesic(spanning, ax, voxel_um)
                entry["geodesic_tortuosity"] = tau
                if want_fields and dist is not None:
                    fields[f"geodesic_{d}"] = dist
            except Exception as e:                                   # noqa: BLE001
                entry["geodesic_error"] = f"{type(e).__name__}: {e}"
        entry["dead_end_fraction"] = float((cls == DEAD_END).sum() / max(mask.sum(), 1))
        entry["isolated_fraction"] = float((cls == ISOLATED).sum() / max(mask.sum(), 1))
        res["by_direction"][d] = entry
        if want_fields:
            fields[f"class_{d}"] = cls
    res["seconds"] = time.time() - t0
    return res, fields
