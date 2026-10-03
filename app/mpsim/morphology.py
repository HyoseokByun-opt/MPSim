"""Structure analysis - porosity, pore and particle sizes, specific surface.

Everything here is computed by established open-source packages:

* PoreSpy (PMEAL, Gostick et al., JOSS 2019)
    local thickness / pore- and particle-size distribution, chord lengths,
    two-point correlation
* NASA PuMA
    specific surface area (marching-cubes isosurface)
* SciPy ndimage
    connected components

PoreSpy filters are not periodic, while the RVE is. The local-thickness map is
therefore computed on a copy padded by wrap-around by the largest inscribed
radius and cropped back, so a pore cut by the box face is measured whole.
"""
from __future__ import annotations

import contextlib
import io
import math
import time
import warnings

import numpy as np
from scipy import ndimage


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _ps():
    with _quiet():
        import porespy
        try:
            porespy.settings.tqdm["disable"] = True
        except Exception:                                       # noqa: BLE001
            pass
    return porespy


def available():
    try:
        _ps()
        return True
    except Exception:                                           # noqa: BLE001
        return False


def version():
    try:
        return _ps().__version__
    except Exception:                                           # noqa: BLE001
        return None


def local_thickness(mask):
    """Periodic local thickness (diameter of the largest inscribed sphere
    covering each voxel), in voxels. Zero outside the mask."""
    ps = _ps()
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    dt = ndimage.distance_transform_edt(np.pad(mask, 2, mode="wrap"))[2:-2, 2:-2, 2:-2]
    pad = int(min(math.ceil(float(dt.max())) + 2, min(mask.shape) // 2))
    padded = np.pad(mask, pad, mode="wrap")
    with _quiet():
        lt = ps.filters.local_thickness(padded, method="dt", smooth=True)
    sl = tuple(slice(pad, pad + s) for s in mask.shape)
    # PoreSpy returns the inscribed radius measured between voxel centres,
    # which sits half a voxel inside the real surface. Measured on voxelised
    # spheres of d = 12 / 20 / 30 voxels: raw 2r = 10.4 / 18.5 / 28.6,
    # with the half-voxel offset 11.4 / 19.5 / 29.6. Report the corrected
    # diameter.
    lt = np.asarray(lt[sl], np.float32)
    return np.where(mask, 2.0 * (lt + 0.5), 0.0).astype(np.float32)


def size_distribution(lt_vox, mask, voxel_um, bins=12):
    """Volume-weighted distribution of local thickness (diameter), in um."""
    vals = np.asarray(lt_vox)[np.asarray(mask, bool)] * voxel_um
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + voxel_um
    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(vals, bins=edges)
    frac = hist / hist.sum()
    centers = 0.5 * (edges[1:] + edges[:-1])
    cdf = np.cumsum(frac)
    d50 = float(np.interp(0.5, cdf, centers))
    d10 = float(np.interp(0.1, cdf, centers))
    d90 = float(np.interp(0.9, cdf, centers))
    return {"diameter_um": centers.tolist(), "volume_fraction": frac.tolist(),
            "cdf": cdf.tolist(), "d10_um": d10, "d50_um": d50, "d90_um": d90,
            "mean_um": float(vals.mean())}


def chord_lengths(mask, voxel_um, bins=20):
    """Mean chord length along x, y and z (PoreSpy apply_chords)."""
    ps = _ps()
    out = {}
    for ax in range(3):
        with _quiet():
            ch = ps.filters.apply_chords(np.asarray(mask, bool), axis=ax, trim_edges=True, label=False)
            # chords are straight runs along `ax`: connect neighbours along that axis only
            st = np.zeros((3, 3, 3), bool)
            idx = [1, 1, 1]
            for k in (0, 1, 2):
                idx[ax] = k
                st[tuple(idx)] = True
            lab, n = ndimage.label(ch, structure=st)
        if n == 0:
            out["xyz"[ax]] = None
            continue
        lengths = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * voxel_um
        out["xyz"[ax]] = {"mean_um": float(np.mean(lengths)), "median_um": float(np.median(lengths)),
                          "n": int(n)}
    return out


def two_point(mask, voxel_um, bins=60):
    """Two-point correlation S2(r) and the distance where it decays to within
    1/e of phi^2 - a correlation length for the microstructure."""
    ps = _ps()
    with _quiet():
        tpc = ps.metrics.two_point_correlation(np.asarray(mask, bool), voxel_size=voxel_um, bins=bins)
    r = np.asarray(getattr(tpc, "distance", getattr(tpc, "bin_centers", [])), float)
    p = np.asarray(getattr(tpc, "probability_scaled", getattr(tpc, "probability", [])), float)
    if r.size == 0:
        return None
    phi = float(np.mean(mask))
    s0, sinf = phi, phi * phi
    corr_len = None
    if p.size and s0 > sinf:
        target = sinf + (s0 - sinf) / math.e
        below = np.nonzero(p <= target)[0]
        if below.size:
            corr_len = float(r[below[0]])
    return {"r_um": r.tolist(), "s2": p.tolist(), "correlation_length_um": corr_len}


def clusters(mask):
    """Connected components (6-connectivity) of a PERIODIC phase: a particle
    cut by a box face is one cluster, not two."""
    lab, n = ndimage.label(np.asarray(mask, bool))
    if n == 0:
        return {"n_clusters": 0, "largest_fraction": 0.0}
    parent = np.arange(n + 1)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    for ax in range(3):
        hi = np.take(lab, lab.shape[ax] - 1, axis=ax)
        lo = np.take(lab, 0, axis=ax)
        m = (hi > 0) & (lo > 0)
        if m.any():
            for p, q in np.unique(np.stack([hi[m], lo[m]], axis=1), axis=0):
                rp, rq = find(p), find(q)
                if rp != rq:
                    parent[rq] = rp
    roots = np.array([find(i) for i in range(n + 1)])
    sizes = np.bincount(roots[lab.ravel()], minlength=n + 1)
    sizes[0] = 0
    real = sizes[sizes > 0]
    return {"n_clusters": int(real.size), "largest_fraction": float(real.max() / real.sum())}


def analyse_phase(labels, phase_ids, voxel_um, want_psd=True, log=None):
    """Morphology of the union of `phase_ids` (e.g. all pores, or one filler)."""
    from .solvers.conduction import wrapping_axes
    t0 = time.time()
    mask = np.isin(labels, phase_ids)
    res = {"volume_fraction": float(mask.mean())}
    res["clusters"] = clusters(mask)
    res["percolates"] = dict(zip("xyz", wrapping_axes(mask)))
    lt = None
    if want_psd and mask.any():
        lt = local_thickness(mask)
        res["size_distribution"] = size_distribution(lt, mask, voxel_um)
    try:
        res["chords"] = chord_lengths(mask, voxel_um)
    except Exception as e:                                          # noqa: BLE001
        res["chords"] = {"error": str(e)}
    try:
        res["two_point"] = two_point(mask, voxel_um)
    except Exception as e:                                          # noqa: BLE001
        res["two_point"] = {"error": str(e)}
    res["seconds"] = time.time() - t0
    return res, lt
