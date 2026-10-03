"""Grain and filler analysis - particle statistics of the generated structure.

Two sources, reported side by side:

* the **particle list** of the generator (exact geometry of every particle):
  count, number density, equivalent-diameter distribution, sphericity,
  aspect ratio, orientation tensor and Hermans order parameters, nearest-
  neighbour spacing and the Clark-Evans dispersion index;
* the **voxel image** (what the solvers see): connected clusters, periodic
  percolation, PuMA structure-tensor orientation of elongated phases and the
  PoreSpy local thickness of the matrix between particles (the matrix
  ligament or film thickness that governs conduction in highly filled
  composites).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import geometry as G
from . import morphology as MO
from . import shapes as SH


def _quantiles(v, w=None):
    v = np.asarray(v, float)
    if v.size == 0:
        return {}
    order = np.argsort(v)
    v = v[order]
    w = np.ones_like(v) if w is None else np.asarray(w, float)[order]
    c = np.cumsum(w) / w.sum()
    q = lambda p: float(np.interp(p, c, v))
    return {"d10": q(0.1), "d50": q(0.5), "d90": q(0.9), "mean": float(np.average(v, weights=w))}


def _aspect(code, a, b):
    if code == 0:
        return np.ones_like(a)
    if code == 1:
        return b / a
    if code == 2:
        return b / a
    return (b + a) / a


def analyse(spec, info, labels, table, h_um, want_structure_tensor=True, log=None):
    from scipy.spatial import cKDTree
    from .solvers.conduction import wrapping_axes
    t0 = time.time()
    parts = info["particles"]
    L = float(info["L_um"])
    out = {"phases": {}}
    for li, t in enumerate(table):
        if t["kind"] != "core":
            continue
        i = t["phase"]
        ph = spec["phases"][i]
        if ph["shape"] == G.NETWORK:
            continue
        sel = parts["phase"] == i
        n = int(sel.sum())
        entry = {"name": t["name"], "shape": ph["shape"], "n_particles": n}
        if n == 0:
            out["phases"][str(li)] = entry
            continue
        codes = parts["codes"][sel]
        Ps = parts["P"][sel]
        cen = np.mod(parts["centers"][sel], L)
        ax = parts["axes"][sel]
        code = int(codes[0])
        meta = info.get("poly_meta")
        V = np.array([SH.volume(int(c), P, meta) for c, P in zip(codes, Ps)])
        A = np.array([SH.surface(int(c), P, meta) for c, P in zip(codes, Ps)])
        a = np.array([SH.circumradius(int(c), P) for c, P in zip(codes, Ps)])
        eqd = (6.0 * V / math.pi) ** (1.0 / 3.0)
        psi = math.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0) / A
        ar = np.array([SH.aspect(int(c), P, meta) for c, P in zip(codes, Ps)])
        entry.update({
            "number_density_per_mm3": n / (L * 1e-3) ** 3,
            "eq_diameter_um": {"number": _quantiles(eqd), "volume": _quantiles(eqd, V)},
            "sphericity": float(np.mean(psi)),
            "aspect_ratio": float(np.mean(ar)),
            "specific_surface_true_1pm": float(A.sum() / (L ** 3) * 1e6),
        })
        hist, edges = np.histogram(eqd, bins=14)
        vhist, _ = np.histogram(eqd, bins=edges, weights=V)
        entry["size_histogram"] = {"center_um": (0.5 * (edges[1:] + edges[:-1])).tolist(),
                                   "number": (hist / hist.sum()).tolist(),
                                   "volume": (vhist / max(vhist.sum(), 1e-300)).tolist()}
        if code != 0:
            T = (ax.T @ ax) / n
            ev = np.sort(np.linalg.eigvalsh(T))[::-1]
            entry["orientation_tensor"] = T.tolist()
            entry["orientation_eigenvalues"] = ev.tolist()
            entry["hermans"] = {d: float(1.5 * T[k, k] - 0.5) for k, d in enumerate("xyz")}
        if n >= 2:
            tree = cKDTree(cen, boxsize=L * (1 + 1e-12))
            dist, idx = tree.query(cen, k=2)
            nn = dist[:, 1]
            rho = n / L ** 3
            entry["nearest_neighbour_um"] = {"mean": float(nn.mean()), "cv": float(nn.std() / nn.mean())}
            entry["clark_evans_index"] = float(nn.mean() / (0.55396 * rho ** (-1.0 / 3.0)))
            if code == 0:
                gap = nn - a - a[idx[:, 1]]
                entry["surface_gap_um"] = {"mean": float(gap.mean()), "min": float(gap.min()),
                                           "d10": float(np.percentile(gap, 10))}
        ids = [li] + [j for j, u in enumerate(table) if u["kind"] == "shell" and u["phase"] == i]
        mask = np.isin(labels, ids)
        cl = MO.clusters(mask)
        entry["voxel"] = {"volume_fraction": float(mask.mean()), "clusters": cl,
                          "particles_per_cluster": n / max(cl["n_clusters"], 1),
                          "percolates": dict(zip("xyz", wrapping_axes(mask)))}
        if want_structure_tensor and code != 0 and (np.mean(ar) >= 2.0):
            try:
                from .solvers import puma_backend as PB
                if PB.available():
                    Ast = PB.orientation_tensor(labels, ids, h_um * 1e-6)
                    if Ast is not None:
                        entry["orientation_tensor_image"] = Ast.tolist()
            except Exception as e:                                  # noqa: BLE001
                entry["orientation_tensor_image_error"] = str(e)
        del mask
        out["phases"][str(li)] = entry
    matrix = np.asarray(labels) == 0
    lt = None
    if matrix.any() and not matrix.all():
        lt = MO.local_thickness(matrix)
        sd = MO.size_distribution(lt, matrix, h_um, bins=14)
        out["matrix_ligament"] = sd
    out["seconds"] = time.time() - t0
    if log:
        log(f"    grain analysis: {sum(p.get('n_particles', 0) for p in out['phases'].values())} particles "
            f"({time.time()-t0:.1f} s)")
    return out, lt
