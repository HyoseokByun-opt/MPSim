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
def sample_scales(n, dist, rng):
    """Multiplicative size factors with mean 1.

    'lognormal' uses the coefficient of variation of the diameter and is
    truncated at +-2 standard deviations of the underlying normal, so a rare
    tiny particle cannot silently demand a ten times finer grid.
    """
    kind = (dist or {}).get("type", "mono")
    cv = float((dist or {}).get("cv", 0.0) or 0.0)
    if kind != "lognormal" or cv <= 0.0:
        return np.ones(n)
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma
    z = rng.standard_normal(n)
    bad = np.abs(z) > 2.0
    while bad.any():
        z[bad] = rng.standard_normal(int(bad.sum()))
        bad = np.abs(z) > 2.0
    return np.exp(mu + sigma * z)


def scale_bounds(dist):
    kind = (dist or {}).get("type", "mono")
    cv = float((dist or {}).get("cv", 0.0) or 0.0)
    if kind != "lognormal" or cv <= 0.0:
        return 1.0, 1.0
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma
    return math.exp(mu - 2.0 * sigma), math.exp(mu + 2.0 * sigma)


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
