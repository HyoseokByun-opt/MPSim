"""Particle shapes: parameters, analytic measures and random sampling.

Every particle is described by the same eight numbers - centre, unit symmetry
axis u, and two semi-dimensions (a, b) - so one rasteriser handles all shapes:

    shape            code   a                  b
    sphere            0     radius             (unused)
    spheroid          1     equatorial radius  polar semi-axis (along u)
    cylinder          2     radius             half length (along u)
    spherocylinder    3     radius             half length of the straight part

A spheroid with aspect = b/a > 1 is a needle, < 1 a platelet. A cylinder whose
length is below its diameter is a disc. So "rod", "fibre", "flake" and "disc"
are all covered without separate code paths.

All lengths are micrometres.
"""
from __future__ import annotations

import math

import numpy as np

SHAPE_CODES = {"sphere": 0, "spheroid": 1, "cylinder": 2, "spherocylinder": 3}
NETWORK = "network"
UNIT_TO_UM = {"nm": 1e-3, "um": 1.0, "mm": 1e3}


def to_um(value, unit):
    return float(value) * UNIT_TO_UM[unit]


def semi_axes(shape, size):
    """(a, b) in um for a size dict already converted to um."""
    d = float(size["d"])
    if d <= 0:
        raise ValueError("The size (diameter) must be greater than zero")
    a = 0.5 * d
    if shape == "sphere":
        return a, 0.0
    if shape == "spheroid":
        ar = float(size.get("aspect", 1.0))
        if ar <= 0:
            raise ValueError("The aspect ratio must be greater than zero")
        return a, a * ar
    if shape == "cylinder":
        length = float(size["length"])
        if length <= 0:
            raise ValueError("The length must be greater than zero")
        return a, 0.5 * length
    if shape == "spherocylinder":
        length = float(size["length"])
        if length < d:
            raise ValueError("The total length of a spherocylinder must not be smaller than its diameter")
        return a, 0.5 * (length - d)
    raise ValueError(f"Unknown shape: {shape}")


def volume(code, a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if code == 0:
        return 4.0 / 3.0 * math.pi * a ** 3
    if code == 1:
        return 4.0 / 3.0 * math.pi * a * a * b
    if code == 2:
        return math.pi * a * a * 2.0 * b
    return math.pi * a * a * 2.0 * b + 4.0 / 3.0 * math.pi * a ** 3


def surface(code, a, b):
    """Analytic surface area. Used to correct the voxel staircase when an
    interfacial resistance is applied (the voxelised surface is ~1.5x larger)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if code == 0:
        return 4.0 * math.pi * a * a
    if code == 1:
        out = np.empty(np.broadcast(a, b).shape)
        aa, bb = np.broadcast_arrays(a, b)
        for idx in np.ndindex(out.shape) if out.shape else [()]:
            ai, bi = float(aa[idx]), float(bb[idx])
            if abs(bi - ai) < 1e-9 * ai:
                s = 4.0 * math.pi * ai * ai
            elif bi > ai:                                   # prolate
                e = math.sqrt(1.0 - (ai / bi) ** 2)
                s = 2.0 * math.pi * ai * ai * (1.0 + bi / (ai * e) * math.asin(e))
            else:                                           # oblate
                e = math.sqrt(1.0 - (bi / ai) ** 2)
                s = 2.0 * math.pi * ai * ai * (1.0 + (1.0 - e * e) / e * math.atanh(e))
            out[idx] = s
        return out if out.shape else float(out)
    if code == 2:
        return 2.0 * math.pi * a * a + 2.0 * math.pi * a * 2.0 * b
    return 4.0 * math.pi * a * a + 2.0 * math.pi * a * 2.0 * b


def min_dim(code, a, b):
    """Smallest feature a voxel grid has to resolve."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if code == 0 or code == 3:
        return 2.0 * a
    return 2.0 * np.minimum(a, b)


def max_extent(code, a, b):
    """Largest end-to-end length of the particle."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if code == 0:
        return 2.0 * a
    if code == 1:
        return 2.0 * np.maximum(a, b)
    if code == 2:
        return np.sqrt((2.0 * a) ** 2 + (2.0 * b) ** 2)
    return 2.0 * b + 2.0 * a


def eq_diameter(code, a, b):
    return (6.0 * volume(code, a, b) / math.pi) ** (1.0 / 3.0)


def aspect_ratio(shape, size):
    """Length / diameter along the symmetry axis (1 for spheres); for the v4
    shapes, the longest over the shortest extent."""
    if shape not in SHAPE_CODES:
        from . import shapes as SH
        return SH.phase_stats(shape, size, None)["aspect"]
    if shape == "sphere":
        return 1.0
    if shape == "spheroid":
        return float(size.get("aspect", 1.0))
    return float(size["length"]) / float(size["d"])


# ---------------------------------------------------------------- sampling
def _lognormal_params(cv):
    """(mu, sigma) of ln(scale) for a log-normal diameter of mean 1 and
    coefficient of variation cv (number basis)."""
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    return -0.5 * sigma * sigma, sigma


# the smallest size factor a normal distribution draws: a broad one (CV
# above 0.5) would otherwise reach zero or negative diameters
NORMAL_FLOOR = 0.05


def _scale_of(dist, z):
    """Size factor(s) of the standard normal variable z: 1 + CV z for a
    normal distribution, exp(mu + sigma z) for a log-normal one."""
    cv = float(dist.get("cv", 0.0) or 0.0)
    if dist.get("type") == "normal":
        return 1.0 + cv * z
    mu, sigma = _lognormal_params(cv)
    return np.exp(mu + sigma * z)


def _z_of(dist, s):
    """The standard normal variable of the size factor s (inverse of _scale_of)."""
    cv = float(dist.get("cv", 0.0) or 0.0)
    if dist.get("type") == "normal":
        return (s - 1.0) / cv
    mu, sigma = _lognormal_params(cv)
    return (math.log(s) - mu) / sigma


def _z_window(dist):
    """The window of the standard normal variable that the size factors are
    drawn from: the bounds of the distribution (dist['lo'], dist['hi'], size
    factors), or +-2 standard deviations when none is given; a normal
    distribution never below NORMAL_FLOOR."""
    lo, hi = dist.get("lo"), dist.get("hi")
    if lo is None and hi is None:
        zl, zh = -2.0, 2.0
    else:
        zl = _z_of(dist, lo) if lo else -8.0
        zh = _z_of(dist, hi) if hi else 8.0
    if dist.get("type") == "normal":
        zl = max(zl, _z_of(dist, NORMAL_FLOOR))
    return max(zl, -8.0), min(zh, 8.0)


def sample_scales(n, dist, rng):
    """Multiplicative size factors of the nominal size.

    'lognormal': the coefficient of variation of the diameter (number basis)
    around a number mean of 1, cut at the bounds 'lo' and 'hi' (factors of the
    nominal size) or, without bounds, at +-2 standard deviations of the
    underlying normal, so that a rare tiny particle cannot silently demand a
    ten times finer grid. The mean and CV describe the distribution before the
    cut.
    'normal': the diameter itself normal, mean 1 and standard deviation CV,
    cut in the same way (and never below NORMAL_FLOOR).
    'table': a measured distribution in bins of the size factor ('edges'),
    with number probabilities 'p'; log-uniform inside a bin. Its mean is 1
    (spec.py scales the nominal size to the number mean).
    """
    kind = (dist or {}).get("type", "mono")
    if kind == "table":
        e = np.asarray(dist["edges"], float)
        pr = np.asarray(dist["p"], float)
        k = rng.choice(len(pr), size=n, p=pr / pr.sum())
        u = rng.random(n)
        return np.exp(np.log(e[k]) + u * (np.log(e[k + 1]) - np.log(e[k])))
    cv = float((dist or {}).get("cv", 0.0) or 0.0)
    if kind not in ("lognormal", "normal") or cv <= 0.0:
        return np.ones(n)
    zl, zh = _z_window(dist)
    z = rng.standard_normal(n)
    bad = (z < zl) | (z > zh)
    while bad.any():
        z[bad] = rng.standard_normal(int(bad.sum()))
        bad = (z < zl) | (z > zh)
    return _scale_of(dist, z)


def scale_bounds(dist):
    """(smallest, largest) size factor the distribution can draw."""
    kind = (dist or {}).get("type", "mono")
    if kind == "table":
        e = np.asarray(dist["edges"], float)
        pr = np.asarray(dist["p"], float)
        nz = np.flatnonzero(pr > 0)
        return float(e[nz[0]]), float(e[nz[-1] + 1])
    cv = float((dist or {}).get("cv", 0.0) or 0.0)
    if kind not in ("lognormal", "normal") or cv <= 0.0:
        return 1.0, 1.0
    zl, zh = _z_window(dist)
    return float(_scale_of(dist, zl)), float(_scale_of(dist, zh))


# ------------------------------------------------- size distributions (PSD)
def psd_bins(d, v, form="auto", basis="volume", d_min=None, d_max=None):
    """A measured particle-size distribution as number probabilities in bins.

    d: diameters (ascending); v: the measured values - per channel
    ('differential', any scale, e.g. % per channel) or passing below d
    ('cumulative', % or fraction); form 'auto' takes a non-decreasing column
    that ends near its maximum as cumulative. basis: 'volume' (laser
    diffraction) or 'number' (counting). Returns (edges, p_number, info): bin
    edges in the unit of d and number probabilities, with the bounds applied
    (bins cut at d_min, d_max)."""
    d = np.asarray(d, float)
    v = np.asarray(v, float)
    ok = np.isfinite(d) & np.isfinite(v) & (d > 0)
    d, v = d[ok], v[ok]
    order = np.argsort(d)
    d, v = d[order], v[order]
    if len(d) < 2:
        raise ValueError("a size distribution needs at least two diameters")
    if np.any(np.diff(d) <= 0):
        raise ValueError("the diameters of a size distribution must all differ")
    v = np.maximum(v, 0.0)
    if form == "auto":
        form = "cumulative" if (np.all(np.diff(v) >= -1e-9) and v[-1] > 0 and v[0] <= 0.5 * v[-1]) else "differential"
    if form == "cumulative":
        # passing below d: the bins lie between the listed diameters, the first
        # one below the smallest diameter by the same ratio as the next
        r0 = d[1] / d[0]
        edges = np.concatenate([[d[0] / r0], d])
        frac = np.diff(np.concatenate([[0.0], v]))
        frac = np.maximum(frac, 0.0)
    else:
        # per channel, the diameter in the middle of its channel (geometric)
        mid = np.sqrt(d[1:] * d[:-1])
        edges = np.concatenate([[d[0] ** 2 / mid[0]], mid, [d[-1] ** 2 / mid[-1]]])
        frac = v.copy()
    if frac.sum() <= 0:
        raise ValueError("the size distribution holds no particles")
    # bounds: cut the bins (a bin partly outside keeps its share inside)
    lo = float(d_min) if d_min else edges[0]
    hi = float(d_max) if d_max else edges[-1]
    if hi <= lo:
        raise ValueError("the upper bound of the size distribution must exceed the lower bound")
    a, b = np.maximum(edges[:-1], lo), np.minimum(edges[1:], hi)
    keep = b > a
    if not keep.any():
        raise ValueError("no part of the size distribution lies between its bounds")
    share = np.where(keep, np.log(np.maximum(b, 1e-300) / np.maximum(a, 1e-300)) / np.log(edges[1:] / edges[:-1]), 0.0)
    frac = frac * np.clip(share, 0.0, 1.0)
    a, b, frac = a[keep], b[keep], frac[keep]
    e2 = np.concatenate([a[:1], b])
    if np.any(np.abs(a[1:] - b[:-1]) > 1e-12 * b[:-1]):
        raise ValueError("the size distribution has a gap between its bins")
    mid = np.sqrt(e2[1:] * e2[:-1])
    # a volume distribution counts particles by their volume: number ~ volume / d^3
    pn = frac / mid ** 3 if basis == "volume" else frac
    pn = pn / pn.sum()
    return e2, pn, {"form": form, "basis": basis}


def table_moments(edges, pn):
    """Number mean diameter of binned number probabilities (log-uniform in a bin)."""
    a, b = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
    mean_bin = (b - a) / np.log(b / a)
    return float(np.dot(pn, mean_bin))


def dist_summary(d_nominal, dist, n=40000, seed=11):
    """What a size distribution draws, for the planner and the browser:
    number mean, standard deviation and median, the volume median, the
    extremes, and number and volume densities over log d for a chart."""
    s = np.sort(sample_scales(n, dist, np.random.default_rng(seed))) * float(d_nominal)
    w = s ** 3
    cw = np.cumsum(w) / w.sum()
    dv = lambda q: float(s[min(int(np.searchsorted(cw, q)), len(s) - 1)])  # noqa: E731
    lo, hi = float(s[0]), float(s[-1])
    if hi > lo * 1.0001:
        bins = np.geomspace(lo, hi, 41)
        hn, _ = np.histogram(s, bins=bins)
        hv, _ = np.histogram(s, bins=bins, weights=w)
        dlog = np.log10(bins[1:] / bins[:-1])
        curve = {"d": np.sqrt(bins[1:] * bins[:-1]).tolist(),
                 "number": (hn / hn.sum() / dlog).tolist(), "volume": (hv / hv.sum() / dlog).tolist()}
    else:
        curve = None
    return {"number_mean": float(s.mean()), "number_median": float(np.median(s)), "std": float(s.std()),
            "cv": float(s.std() / s.mean()), "d10v": dv(0.10), "d50v": dv(0.50), "d90v": dv(0.90),
            "d_min": lo, "d_max": hi, "curve": curve}


def sample_axes(n, mode, spread_deg, rng):
    """Unit symmetry axes.

    iso   uniform on the sphere
    x/y/z aligned with that axis, with a Gaussian tilt of `spread_deg`
    xy    axis lying in the xy plane (uniform azimuth), tilted out of plane
          by `spread_deg`
    """
    spread = math.radians(float(spread_deg or 0.0))
    if mode == "iso":
        v = rng.standard_normal((n, 3))
    elif mode in ("x", "y", "z"):
        t = rng.standard_normal((n, 2)) * spread
        v = np.column_stack([t[:, 0], t[:, 1], np.ones(n)])
        if mode == "x":
            v = v[:, [2, 0, 1]]
        elif mode == "y":
            v = v[:, [0, 2, 1]]
    elif mode == "xy":
        phi = rng.random(n) * 2.0 * math.pi
        tilt = rng.standard_normal(n) * spread
        v = np.column_stack([np.cos(phi) * np.cos(tilt),
                             np.sin(phi) * np.cos(tilt), np.sin(tilt)])
    else:
        raise ValueError(f"Unknown orientation mode: {mode}")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def orientation_moments(mode):
    """<(u.e_x)^2>, <(u.e_y)^2>, <(u.e_z)^2> for the analytical models
    (spread ignored)."""
    return {"iso": (1 / 3, 1 / 3, 1 / 3), "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0),
            "xy": (0.5, 0.5, 0.0)}[mode]


def phase_size_stats(shape, size_um, dist):
    """Deterministic size statistics used by the RVE planner.

    Quantiles are evaluated on the truncated distribution, so they are what
    the generator will actually draw.
    """
    if shape not in SHAPE_CODES:
        from . import shapes as SH
        return SH.phase_stats(shape, size_um, dist)
    code = SHAPE_CODES[shape]
    a, b = semi_axes(shape, size_um)
    lo, hi = scale_bounds(dist)
    rng = np.random.default_rng(20260915)
    s = sample_scales(20000, dist, rng)
    V = volume(code, a * s, b * s)
    md = min_dim(code, a * s, b * s)
    return {
        "code": code,
        "a": a, "b": b,
        "min_dim_p10": float(np.quantile(md, 0.10)),
        "min_dim_min": float(min_dim(code, a * lo, b * lo)),
        "eqd_max": float(eq_diameter(code, a * hi, b * hi)),
        "extent_max": float(max_extent(code, a * hi, b * hi)),
        "extent_median": float(max_extent(code, a, b)),
        "eqd_median": float(eq_diameter(code, a, b)),
        "mean_volume": float(V.mean()),
        "scale_min": lo, "scale_max": hi,
    }
