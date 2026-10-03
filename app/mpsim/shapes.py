"""Particle shapes (v4): one rigid-body description for every shape.

A particle is a shape code, six shape parameters P, a rotation R and a centre.
R holds the particle's local axes as rows, so local = R @ (x - c) and a world
direction of local axis j is R[j]. Every shape is tested point-by-point in its
local frame, which lets one rasteriser and one packing engine handle them all:

    code  shape            P[0]        P[1]         P[2]        P[3]
    0     sphere           r
    1     spheroid         a (eq.)     b (along z)
    2     cylinder         r           half length
    3     spherocylinder   r           half straight length
    4     cuboid           half x      half y       half z               (cube: equal)
    5     superellipsoid   semi x      semi y       semi z      exponent n
    6     polyhedron       circumradius template id
    7     helix            wire r      coil radius  rise/rad    swept angle

The superellipsoid |x/a|^n + |y/b|^n + |z/c|^n <= 1 is the "meta-particle" of
DEM codes: n = 2 is an ellipsoid, larger n rounds towards a box with soft
edges. A polyhedron is the intersection of half-spaces n.x <= d (convex); its
planes live in a template table so particles of one phase can share a set of
irregular shapes. A helix is a wire swept along a coil around local z, with
rounded ends.

`inset` shrinks a shape by a normal distance (a shell's inner boundary) and a
negative inset grows it (a separation gap). All lengths are micrometres.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from . import geometry as G

CODES = {"sphere": 0, "spheroid": 1, "cylinder": 2, "spherocylinder": 3,
         "cube": 4, "cuboid": 4, "superellipsoid": 5, "polyhedron": 6, "helix": 7}
USER_SHAPES = ["sphere", "spheroid", "cylinder", "spherocylinder", "cube", "cuboid",
               "superellipsoid", "polyhedron", "helix"]
POLY_KINDS = ["tetrahedron", "octahedron", "dodecahedron", "icosahedron", "irregular"]
MAX_FACES = 64


# =========================================================================
# point tests (numba)
# =========================================================================
@njit(cache=True, fastmath=True)
def helix_dist2(x, y, z, r_c, c, th_max):
    """Squared distance from a point to the centreline (Rc cos t, Rc sin t, c t - H/2), t in [0, th_max]."""
    H = c * th_max
    zz = z + 0.5 * H
    phi = math.atan2(y, x)
    if c > 1e-12:
        k0 = int(math.floor((zz / c - phi) / (2.0 * math.pi) + 0.5))
    else:
        k0 = 0
    best = 1e300
    for dk in range(-1, 2):
        t = phi + 2.0 * math.pi * (k0 + dk)
        if t < 0.0:
            t = 0.0
        elif t > th_max:
            t = th_max
        for _it in range(6):
            ct = math.cos(t)
            st = math.sin(t)
            px = x - r_c * ct
            py = y - r_c * st
            pz = zz - c * t
            g1 = -(px * (-r_c * st) + py * (r_c * ct) + pz * c)
            g2 = (r_c * r_c + c * c) - (px * (-r_c * ct) + py * (-r_c * st))
            if g2 <= 1e-12:
                break
            t2 = t - g1 / g2
            if t2 < 0.0:
                t2 = 0.0
            elif t2 > th_max:
                t2 = th_max
            if abs(t2 - t) < 1e-10:
                t = t2
                break
            t = t2
        px = x - r_c * math.cos(t)
        py = y - r_c * math.sin(t)
        pz = zz - c * t
        d2 = px * px + py * py + pz * pz
        if d2 < best:
            best = d2
    return best


@njit(cache=True, fastmath=True)
def inside_local(code, x, y, z, P, planes, nfaces, inset):
    if code == 0:
        r = P[0] - inset
        return r > 0.0 and x * x + y * y + z * z <= r * r
    if code == 1:
        a = P[0] - inset
        b = P[1] - inset
        if a <= 0.0 or b <= 0.0:
            return False
        return (x * x + y * y) / (a * a) + z * z / (b * b) <= 1.0
    if code == 2:
        r = P[0] - inset
        hl = P[1] - inset
        if r <= 0.0 or hl <= 0.0 or z > hl or z < -hl:
            return False
        return x * x + y * y <= r * r
    if code == 3:
        r = P[0] - inset
        if r <= 0.0:
            return False
        zc = z
        if zc > P[1]:
            zc = P[1]
        elif zc < -P[1]:
            zc = -P[1]
        dz = z - zc
        return x * x + y * y + dz * dz <= r * r
    if code == 4:
        return abs(x) <= P[0] - inset and abs(y) <= P[1] - inset and abs(z) <= P[2] - inset
    if code == 5:
        a = P[0] - inset
        b = P[1] - inset
        c = P[2] - inset
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            return False
        n = P[3]
        return (abs(x) / a) ** n + (abs(y) / b) ** n + (abs(z) / c) ** n <= 1.0
    if code == 6:
        s = P[0]
        t = int(P[1])
        for f in range(nfaces[t]):
            if planes[t, f, 0] * x + planes[t, f, 1] * y + planes[t, f, 2] * z > s * planes[t, f, 3] - inset:
                return False
        return True
    if code == 7:
        r = P[0] - inset
        if r <= 0.0:
            return False
        rc = P[1]
        rho = math.sqrt(x * x + y * y)
        if (rho - rc) * (rho - rc) > r * r:
            return False
        H = P[2] * P[3]
        if z < -0.5 * H - r or z > 0.5 * H + r:
            return False
        return helix_dist2(x, y, z, rc, P[2], P[3]) <= r * r
    return False


@njit(cache=True, fastmath=True)
def local_half_extents(code, P, planes, nfaces):
    """Half extents of the particle's box in its own frame."""
    if code == 0:
        return P[0], P[0], P[0]
    if code == 1:
        return P[0], P[0], P[1]
    if code == 2:
        return P[0], P[0], P[1]
    if code == 3:
        return P[0], P[0], P[1] + P[0]
    if code == 4 or code == 5:
        return P[0], P[1], P[2]
    if code == 6:
        return P[0], P[0], P[0]
    if code == 7:
        e = P[1] + P[0]
        return e, e, 0.5 * P[2] * P[3] + P[0]
    return P[0], P[0], P[0]


# =========================================================================
# polyhedron templates
# =========================================================================
def _platonic_vertices(kind):
    p = (1 + 5 ** 0.5) / 2
    if kind == "tetrahedron":
        v = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif kind == "octahedron":
        v = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif kind == "icosahedron":
        v = [(0, s1, s2 * p) for s1 in (-1, 1) for s2 in (-1, 1)] + \
            [(s1, s2 * p, 0) for s1 in (-1, 1) for s2 in (-1, 1)] + \
            [(s2 * p, 0, s1) for s1 in (-1, 1) for s2 in (-1, 1)]
    elif kind == "dodecahedron":
        v = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)] + \
            [(0, s1 / p, s2 * p) for s1 in (-1, 1) for s2 in (-1, 1)] + \
            [(s1 / p, s2 * p, 0) for s1 in (-1, 1) for s2 in (-1, 1)] + \
            [(s2 * p, 0, s1 / p) for s1 in (-1, 1) for s2 in (-1, 1)]
    else:
        raise ValueError(kind)
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1).max()


def hull_planes(vertices):
    """Unique outward planes (n, d) of the convex hull, scaled to circumradius 1."""
    from scipy.spatial import ConvexHull
    v = np.asarray(vertices, float)
    v = v - v.mean(axis=0)
    v /= np.linalg.norm(v, axis=1).max()
    hull = ConvexHull(v)
    eq = hull.equations                        # n.x + e <= 0 inside, |n| = 1
    planes = []
    for n0, n1, n2, e in eq:
        cand = np.array([n0, n1, n2, -e])
        if not any(np.allclose(cand, p, atol=1e-7) for p in planes):
            planes.append(cand)
    return np.array(planes), float(hull.volume), float(hull.area)


def irregular_vertices(n_vertices, rng, jitter=0.35):
    """Random convex polyhedron: points on a sphere with radial jitter, like a
    crushed-grain template. More vertices -> rounder grain."""
    v = rng.standard_normal((max(4, int(n_vertices)), 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= (1.0 - jitter * rng.random((len(v), 1)))
    return v


class PolyTable:
    """Plane table shared by all polyhedral particles of one structure."""

    def __init__(self):
        self.planes = []
        self.meta = []

    def add(self, kind, n_vertices=12, rng=None, n_variants=1):
        ids = []
        for k in range(n_variants if kind == "irregular" else 1):
            v = (irregular_vertices(n_vertices, rng) if kind == "irregular" else _platonic_vertices(kind))
            pl, vol, area = hull_planes(v)
            if len(pl) > MAX_FACES:
                raise ValueError(f"a polyhedron with {len(pl)} faces exceeds the limit of {MAX_FACES}")
            self.planes.append(pl)
            self.meta.append({"kind": kind, "volume": vol, "area": area,
                              "inradius": float(pl[:, 3].min()), "faces": len(pl)})
            ids.append(len(self.planes) - 1)
        return ids

    def arrays(self):
        T = max(1, len(self.planes))
        arr = np.zeros((T, MAX_FACES, 4))
        nf = np.zeros(T, np.int64)
        for t, pl in enumerate(self.planes):
            arr[t, :len(pl)] = pl
            nf[t] = len(pl)
        return arr, nf


EMPTY_PLANES = (np.zeros((1, MAX_FACES, 4)), np.zeros(1, np.int64))


# =========================================================================
# shape parameters from a size spec
# =========================================================================
def params(shape, size, poly_ids=None):
    """(code, P) of the nominal particle from a size dict in um."""
    d = float(size["d"])
    P = np.zeros(6)
    if shape in ("sphere", "spheroid", "cylinder", "spherocylinder"):
        a, b = G.semi_axes(shape, size)
        P[0], P[1] = a, b
    elif shape == "cube":
        P[0] = P[1] = P[2] = 0.5 * d
    elif shape == "cuboid":
        P[0], P[1], P[2] = 0.5 * d, 0.5 * float(size["ly"]), 0.5 * float(size["lz"])
    elif shape == "superellipsoid":
        P[0], P[1], P[2] = 0.5 * d, 0.5 * float(size["ly"]), 0.5 * float(size["lz"])
        P[3] = float(size.get("n", 4.0))
    elif shape == "polyhedron":
        P[0] = 0.5 * d
        P[1] = float((poly_ids or [0])[0])
    elif shape == "helix":
        coil_d, pitch, turns = float(size["coil_d"]), float(size["pitch"]), float(size["turns"])
        P[0] = 0.5 * d
        P[1] = 0.5 * coil_d
        P[2] = pitch / (2.0 * math.pi)
        P[3] = 2.0 * math.pi * turns
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return CODES[shape], P


def validate(shape, size):
    """Raise ValueError on sizes that do not describe a solid."""
    d = float(size["d"])
    if d <= 0:
        raise ValueError("The size must be greater than zero")
    if shape in ("sphere", "spheroid", "cylinder", "spherocylinder"):
        G.semi_axes(shape, size)
    elif shape in ("cuboid", "superellipsoid"):
        if float(size.get("ly", 0)) <= 0 or float(size.get("lz", 0)) <= 0:
            raise ValueError("All three edge lengths must be greater than zero")
        if shape == "superellipsoid" and not (2.0 <= float(size.get("n", 4)) <= 40.0):
            raise ValueError("The blockiness exponent must lie between 2 (ellipsoid) and 40 (almost a box)")
    elif shape == "helix":
        coil_d, pitch, turns = float(size.get("coil_d", 0)), float(size.get("pitch", 0)), float(size.get("turns", 0))
        if coil_d <= 0 or turns <= 0 or pitch < 0:
            raise ValueError("A helix needs a coil diameter and a number of turns greater than zero")
        if coil_d < d:
            raise ValueError("The coil diameter must be at least the wire diameter")


# ---------------------------------------------------------------- measures
def _gamma(x):
    return math.gamma(x)


def volume(code, P, poly_meta=None):
    """Analytic volume of one particle (P already scaled)."""
    if code in (0, 1, 2, 3):
        return float(G.volume(code, P[0], P[1]))
    if code == 4:
        return 8.0 * P[0] * P[1] * P[2]
    if code == 5:
        n = P[3]
        return 8.0 * P[0] * P[1] * P[2] * _gamma(1 + 1 / n) ** 3 / _gamma(1 + 3 / n)
    if code == 6:
        return poly_meta[int(P[1])]["volume"] * P[0] ** 3
    if code == 7:
        length = P[3] * math.hypot(P[1], P[2])
        return math.pi * P[0] ** 2 * length + 4.0 / 3.0 * math.pi * P[0] ** 3
    raise ValueError(code)


def surface(code, P, poly_meta=None):
    if code in (0, 1, 2, 3):
        return float(G.surface(code, P[0], P[1]))
    if code == 4:
        return 8.0 * (P[0] * P[1] + P[1] * P[2] + P[0] * P[2])
    if code == 5:
        return _superellipsoid_area(P[0], P[1], P[2], P[3])
    if code == 6:
        return poly_meta[int(P[1])]["area"] * P[0] ** 2
    if code == 7:
        length = P[3] * math.hypot(P[1], P[2])
        return 2.0 * math.pi * P[0] * length + 4.0 * math.pi * P[0] ** 2
    raise ValueError(code)


def _superellipsoid_area(a, b, c, n, m=240):
    """Surface area by quadrature over the parametric form (Barr)."""
    e = 2.0 / n
    u = np.linspace(-math.pi / 2, math.pi / 2, m)
    v = np.linspace(-math.pi, math.pi, 2 * m)
    U, V = np.meshgrid(u, v, indexing="ij")
    sp = lambda w, p: np.sign(np.sin(w)) * np.abs(np.sin(w)) ** p
    cp = lambda w, p: np.sign(np.cos(w)) * np.abs(np.cos(w)) ** p
    X = a * cp(U, e) * cp(V, e)
    Y = b * cp(U, e) * sp(V, e)
    Z = c * sp(U, e)
    Xu, Xv = np.gradient(X, u, v)
    Yu, Yv = np.gradient(Y, u, v)
    Zu, Zv = np.gradient(Z, u, v)
    cr = np.sqrt((Yu * Zv - Zu * Yv) ** 2 + (Zu * Xv - Xu * Zv) ** 2 + (Xu * Yv - Yu * Xv) ** 2)
    return float(np.trapz(np.trapz(cr, v, axis=1), u))


def min_dim(code, P, poly_meta=None):
    if code == 0 or code == 3:
        return 2.0 * P[0]
    if code in (1, 2):
        return 2.0 * min(P[0], P[1])
    if code in (4, 5):
        return 2.0 * min(P[0], P[1], P[2])
    if code == 6:
        return 2.0 * P[0] * poly_meta[int(P[1])]["inradius"]
    if code == 7:
        return 2.0 * P[0]
    raise ValueError(code)


def circumradius(code, P):
    if code == 0:
        return P[0]
    if code == 1:
        return max(P[0], P[1])
    if code == 2:
        return math.hypot(P[0], P[1])
    if code == 3:
        return P[0] + P[1]
    if code == 4:
        return math.sqrt(P[0] ** 2 + P[1] ** 2 + P[2] ** 2)
    if code == 5:
        # the corner of the rounded box: along (1,1,1)/sqrt3 scaled by the axes
        n = P[3]
        t = 3.0 ** (-1.0 / n)
        return max(max(P[0], P[1], P[2]), math.sqrt((P[0] * t) ** 2 + (P[1] * t) ** 2 + (P[2] * t) ** 2))
    if code == 6:
        return P[0]
    if code == 7:
        return math.hypot(P[1] + P[0], 0.5 * P[2] * P[3] + P[0])
    raise ValueError(code)


def aspect(code, P, poly_meta=None):
    """Longest over shortest extent (1 for spheres)."""
    return max(1.0, 2.0 * circumradius(code, P) / max(min_dim(code, P, poly_meta), 1e-30))


# ---------------------------------------------------------------- rotations
def sample_rotations(n, mode, spread_deg, rng):
    """Rotation matrices (rows = local axes in the world). The local z axis
    follows the orientation mode; the spin about it is uniform, so 'iso' is a
    uniformly random rotation."""
    z = G.sample_axes(n, mode, spread_deg, rng)
    tmp = rng.standard_normal((n, 3))
    x = tmp - np.sum(tmp * z, axis=1, keepdims=True) * z
    bad = np.linalg.norm(x, axis=1) < 1e-8
    x[bad] = np.cross(z[bad], [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------- templates
def clump_template(code, P, planes, nfaces, max_spheres=40, coverage=0.975, res=44):
    """Spheres whose union approximates the shape (for packing only).

    The shape is sampled on a fine grid in its own frame; spheres are picked
    greedily at the deepest uncovered point with the distance to the surface
    as radius, until `coverage` of the volume is covered. Wire-like shapes
    (cylinder, spherocylinder, helix) are chains of spheres along the axis,
    which is exact for the spherocylinder and the helix.
    """
    from scipy import ndimage
    if code == 0:
        return np.zeros((1, 3)), np.array([P[0]])
    if code == 3:
        r, hl = P[0], P[1]
        m = max(2, int(math.ceil(2 * hl / (0.5 * r))) + 1)
        zs = np.linspace(-hl, hl, m)
        return np.column_stack([np.zeros(m), np.zeros(m), zs]), np.full(m, r)
    if code == 7:
        r, rc, c, th = P[0], P[1], P[2], P[3]
        length = th * math.hypot(rc, c)
        m = max(2, int(math.ceil(length / (0.5 * r))) + 1)
        t = np.linspace(0.0, th, m)
        xyz = np.column_stack([rc * np.cos(t), rc * np.sin(t), c * t - 0.5 * c * th])
        return xyz, np.full(m, r)
    hx, hy, hz = local_half_extents(code, P, planes, nfaces)
    ext = max(hx, hy, hz)
    h = 2.0 * ext / res
    ax = [np.arange(-e + 0.5 * h, e, h) for e in (hx, hy, hz)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    ins = _inside_grid(code, X.ravel(), Y.ravel(), Z.ravel(), P, planes, nfaces).reshape(X.shape)
    if not ins.any():
        return np.zeros((1, 3)), np.array([0.5 * min_dim(code, P, [{"inradius": 0.5}])])
    edt = ndimage.distance_transform_edt(np.pad(ins, 1)) [1:-1, 1:-1, 1:-1] * h
    pts = np.column_stack([X[ins], Y[ins], Z[ins]])
    depth = edt[ins]
    covered = np.zeros(len(pts), bool)
    cen, rad = [], []
    while len(cen) < max_spheres and covered.mean() < coverage:
        cand = np.where(~covered, depth, -1.0)
        k = int(np.argmax(cand))
        if cand[k] <= 0.5 * h:
            break
        cen.append(pts[k])
        rad.append(depth[k])
        covered |= np.sum((pts - pts[k]) ** 2, axis=1) <= depth[k] ** 2
    return np.array(cen), np.array(rad)


def _inside_grid(code, x, y, z, P, planes, nfaces):
    out = np.zeros(len(x), bool)
    _inside_many(code, x, y, z, P, planes, nfaces, out)
    return out


@njit(cache=True)
def _inside_many(code, x, y, z, P, planes, nfaces, out):
    for i in range(len(x)):
        out[i] = inside_local(code, x[i], y[i], z[i], P, planes, nfaces, 0.0)


# ---------------------------------------------------------------- planner stats
def phase_stats(shape, size_um, dist, poly_meta=None):
    """Size statistics of one phase for the RVE planner (all shapes)."""
    if poly_meta is None and shape == "polyhedron":
        tab = PolyTable()
        rng = np.random.default_rng(20260929)
        ids = tab.add(size_um.get("poly", "icosahedron"), n_vertices=int(size_um.get("n_vertices", 14)),
                      rng=rng, n_variants=min(4, int(size_um.get("variants", 8))))
        poly_meta = tab.meta
        code, P = params(shape, size_um, ids)
    else:
        code, P = params(shape, size_um, [0])
    lo, hi = G.scale_bounds(dist)
    rng = np.random.default_rng(20260915)
    s = G.sample_scales(4000, dist, rng)
    lin = {0: [0], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [0, 1, 2], 5: [0, 1, 2], 6: [0], 7: [0, 1, 2]}[code]

    def at(scale):
        Q = P.copy()
        Q[lin] *= scale
        return Q
    vol1 = volume(code, P, poly_meta)
    md1 = min_dim(code, P, poly_meta)
    ext1 = 2.0 * circumradius(code, P)
    eqd1 = (6.0 * vol1 / math.pi) ** (1.0 / 3.0)
    return {
        "code": code, "P": P,
        "min_dim_p10": float(md1 * np.quantile(s, 0.10)),
        "min_dim_min": float(md1 * lo),
        "eqd_max": float(eqd1 * hi), "extent_max": float(ext1 * hi),
        "extent_median": float(ext1), "eqd_median": float(eqd1),
        "mean_volume": float(vol1 * np.mean(s ** 3)),
        "scale_min": lo, "scale_max": hi,
        "aspect": float(aspect(code, P, poly_meta)),
    }
