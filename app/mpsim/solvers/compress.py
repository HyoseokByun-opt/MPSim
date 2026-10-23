"""Compression of a periodic RVE while its resin flows: three-dimensional
creeping flow on the voxel grid, step by step on the structure as it deforms.

The RVE is a point inside a compound pressed slowly (a moulded or laminated
part far from its edges): it is shortened along z at a constant strain rate,
and its sides either expand freely (no lateral stress) or are held. Each
step solves the periodic Stokes flow of the whole RVE on the voxel grid -
every resin with its own viscosity, the filler spheres as rigid bodies, the
pores as a compressible, nearly stress-free gas - with the voxel finite
elements of the elasticity solver in their viscous form (fans.py: velocity
in place of displacement, viscosity in place of the shear modulus, a bulk
penalty for the incompressible resin, B-bar elements against volumetric
locking; the same equations as the voxel viscosity of solvers/viscosity.py).
The flow then moves the structure by one step:

- each sphere by the mean velocity of its voxels (a rigid translation;
  spheres need no rotation), with overlaps from the step pushed apart,
- the resins and the pores with the flow, through the map from every voxel
  of the deformed RVE back to the voxel of the initial structure it came
  from (accumulated over the steps, so that a motion smaller than a voxel
  per step is not lost to rounding),
- the box by the macroscopic strain of the step.

Macroscopic conditions: the strain rate along z is given. A free side has
zero mean stress across it; a held side has no strain. With both sides held
the volume lost must come from the pores (an incompressible resin cannot be
squeezed into less volume). The macroscopic strain is combined from load
cases that keep the volume (planar x - y, axisymmetric about z) and, with
pores, one that changes it; without pores the mean pressure is the free
multiplier of incompressibility, fixed by the free sides.

Output per step: the macroscopic stress, the strain of every resin region,
the von Mises stress and the pressure in the voxels, the pore content, and
the faces and middle sections of the box for the viewer.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from . import fans as FA

# viscosity ratios: the spheres 1000 times the largest resin viscosity (rigid
# for the flow, as in solvers/viscosity.py); a pore resists closing with 1 % of
# the resin's stress. In a dense packing whose spheres touch, bridged by
# one-voxel resin films, the stress depends on the spheres' rigidity (300
# times the resin gave 28 % less at 40 vol%), so they are kept this stiff.
RIGID_RATIO = 1.0e3        # sphere / largest resin viscosity
BUBBLE_RATIO = 1.0e-2      # pore / smallest resin viscosity
PENALTY = 1.0e2            # bulk penalty / largest viscosity

LATERAL = {"xy": (True, True), "x": (True, False), "none": (False, False)}

# 26 directions around a sphere: the resin that surrounds it
_DIRS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
                 float)
_DIRS /= np.linalg.norm(_DIRS, axis=1)[:, None]


@njit(cache=True)
def _voxel_volumes(X, V, C, comp, hv, nx, ny, nz, T, P):
    """Marker volume per voxel: all (T) and of the compressible classes (P)."""
    for m in range(X.shape[0]):
        i = int(math.floor(X[m, 0] / hv[0])) % nx
        j = int(math.floor(X[m, 1] / hv[1])) % ny
        k = int(math.floor(X[m, 2] / hv[2])) % nz
        T[i, j, k] += V[m]
        if comp[C[m]]:
            P[i, j, k] += V[m]


@njit(parallel=True, cache=True)
def _advect(X, V, C, comp, T, P, x, div, s, hv, h, dt, Lnew):
    """Moves the markers X (M, 3) (µm) one step: the box stretch s and the
    velocity fluctuation x (3, nx, ny, nz; voxels per second at the voxel
    corners of a grid of voxel size hv), interpolated trilinearly. The
    incompressible materials keep their volume; the volume change of a voxel
    (the divergence of its finite element) goes to the compressible markers
    in it (the pores), which T and P (marker volume per voxel: all, pores)
    divide among them."""
    nx, ny, nz = x.shape[1], x.shape[2], x.shape[3]
    for m in prange(X.shape[0]):
        gx = X[m, 0] / hv[0]
        gy = X[m, 1] / hv[1]
        gz = X[m, 2] / hv[2]
        i0 = int(math.floor(gx))
        j0 = int(math.floor(gy))
        k0 = int(math.floor(gz))
        wx = gx - i0
        wy = gy - j0
        wz = gz - k0
        ia = i0 % nx
        ja = j0 % ny
        ka = k0 % nz
        ib = (ia + 1) % nx
        jb = (ja + 1) % ny
        kb = (ka + 1) % nz
        for c in range(3):
            u = ((1 - wx) * ((1 - wy) * ((1 - wz) * x[c, ia, ja, ka] + wz * x[c, ia, ja, kb])
                             + wy * ((1 - wz) * x[c, ia, jb, ka] + wz * x[c, ia, jb, kb]))
                 + wx * ((1 - wy) * ((1 - wz) * x[c, ib, ja, ka] + wz * x[c, ib, ja, kb])
                         + wy * ((1 - wz) * x[c, ib, jb, ka] + wz * x[c, ib, jb, kb])))
            v = X[m, c] * s[c] + u * h * dt
            X[m, c] = v - math.floor(v / Lnew[c]) * Lnew[c]
        if comp[C[m]] and P[ia, ja, ka] > 0:
            t = T[ia, ja, ka]
            f = (t * math.exp(div[ia, ja, ka] * dt) - (t - P[ia, ja, ka])) / P[ia, ja, ka]
            V[m] *= max(f, 0.0)


@njit(cache=True)
def _deposit(X, V, cls, nk, hv, shape, out):
    """Marker volumes per class and voxel of the current grid."""
    nx, ny, nz = shape[0], shape[1], shape[2]
    for m in range(X.shape[0]):
        i = int(math.floor(X[m, 0] / hv[0])) % nx
        j = int(math.floor(X[m, 1] / hv[1])) % ny
        k = int(math.floor(X[m, 2] / hv[2])) % nz
        out[cls[m], (i * ny + j) * nz + k] += V[m]


def assign_volumes(frac, target, max_iter=80):
    """Labels (class index per voxel) from the volume fraction of every class
    in every voxel, so that every class keeps its volume (target, in voxels):
    the largest fraction wins, shifted by a per-class bias adjusted until the
    totals match. A label by the voxel centre alone would keep a boundary
    that moves less than half a voxel per step where it was, and lose that
    layer's volume."""
    K, M = frac.shape
    best = np.argmax(frac, axis=0)
    if K == 1:
        return best
    mixed = np.flatnonzero(frac.max(axis=0) < 0.999)
    if not len(mixed):
        return best
    sub = frac[:, mixed]
    fixed = np.bincount(np.delete(best, mixed), minlength=K) if len(mixed) < M else np.zeros(K)
    tgt = target - fixed
    b = np.zeros(K)
    lab = best[mixed]
    step = 0.5
    for it in range(max_iter):
        lab = np.argmax(sub + b[:, None], axis=0)
        # only classes present in the voxel may win it
        cnt = np.bincount(lab, minlength=K)
        err = tgt - cnt
        if np.all(np.abs(err) <= 0.5):
            break
        b += step * err / max(len(mixed), 1) * 4.0
        b -= b.mean()
        step *= 0.97
    lab = np.where(sub[lab, np.arange(len(mixed))] > 0, lab, best[mixed])
    out = best.copy()
    out[mixed] = lab
    return out


def _node_to_centre(u):
    """Voxel-centre mean of a nodal field (3, nx, ny, nz), nodes at the lower
    corners."""
    out = np.empty_like(u)
    for c in range(3):
        s = u[c] + np.roll(u[c], -1, 0)
        s = s + np.roll(s, -1, 1)
        s = s + np.roll(s, -1, 2)
        out[c] = 0.125 * s
    return out


def push_apart(c, r, L, max_iter=200, tol=1e-4):
    """Moves overlapping spheres apart (periodic box L), each by half the
    overlap along the line of centres. Returns the largest overlap left
    (in units of the smaller radius)."""
    if len(c) < 2:
        return 0.0
    rmax = float(r.max())
    worst = 0.0
    for _ in range(max_iter):
        c %= L
        tree = cKDTree(c, boxsize=L)
        pairs = tree.query_pairs(2.0 * rmax, output_type="ndarray")
        if not len(pairs):
            return 0.0
        i, j = pairs[:, 0], pairs[:, 1]
        d = c[j] - c[i]
        d -= np.round(d / L) * L
        dist = np.linalg.norm(d, axis=1)
        ov = r[i] + r[j] - dist
        hit = ov > tol * np.minimum(r[i], r[j])
        if not hit.any():
            return 0.0
        i, j, d, dist, ov = i[hit], j[hit], d[hit], np.maximum(dist[hit], 1e-12), ov[hit]
        worst = float(np.max(ov / np.minimum(r[i], r[j])))
        n = d / dist[:, None]
        dx = 0.5 * (ov[:, None] + 1e-9) * n
        np.add.at(c, i, -dx)
        np.add.at(c, j, dx)
    return worst


R_GAS = 8.314462618


class Cure:
    """The cure of every resin while the RVE is compressed: Kamal-Sourour
    kinetics d(alpha)/dt = (k1 + k2 alpha^m)(1 - alpha)^n, k_i = A_i
    exp(-E_i / R T), and the Castro-Macosko viscosity
    eta = eta0(T) (alpha_g / (alpha_g - alpha))^(A + B alpha), which diverges at
    the gel point alpha_g. The temperature is uniform in the RVE (heat crosses
    a few tens of µm in milliseconds) and follows the mould through a lumped
    exchange: rho c dT/dt = sum(heat of each resin x its rate) - rho c (T -
    T_mould) / tau, tau = rho c L^2 / (pi^2 k) for a part of thickness L.

    kin: {label: {E1, A1, E2, A2 (J/mol, 1/s), m, n, ag, A, B, Ea (J/mol)}};
    eta0: {label: viscosity of the uncured resin at the mould temperature};
    heat: {label: heat released per unit conversion per volume of the RVE,
    J/m^3}; rhocp: J/m^3 K; tau: s (0 or None: isothermal)."""

    def __init__(self, kin, eta0, T_mold_C, heat, rhocp, tau, alpha0=0.0):
        self.labels = sorted(kin)
        self.kin = kin
        self.eta0 = eta0
        self.Tm = float(T_mold_C) + 273.15
        self.T = self.Tm
        self.t = 0.0
        self.a = np.full(len(self.labels), float(alpha0))
        self.heat = np.array([float(heat.get(li, 0.0)) for li in self.labels])
        self.rhocp = float(rhocp)
        self.tau = float(tau or 0.0)

    def _rates(self, a, T):
        r = np.empty(len(self.labels))
        for i, li in enumerate(self.labels):
            k = self.kin[li]
            ai = min(max(a[i], 0.0), 1.0)
            k1 = k["A1"] * math.exp(-k["E1"] / (R_GAS * T))
            k2 = k["A2"] * math.exp(-k["E2"] / (R_GAS * T))
            r[i] = (k1 + k2 * ai ** k["m"]) * (1.0 - ai) ** k["n"] if ai < 1.0 else 0.0
        return r

    def _deriv(self, a, T):
        r = self._rates(a, T)
        dT = float(np.dot(self.heat, r)) / self.rhocp - ((T - self.Tm) / self.tau if self.tau > 0 else 0.0)
        if self.tau <= 0:
            dT = 0.0
        return r, dT

    def advance(self, dt):
        """Fourth-order Runge-Kutta in sub-steps of at most 1 % conversion."""
        t_left = float(dt)
        while t_left > 0:
            r, _ = self._deriv(self.a, self.T)
            rmax = float(r.max()) if len(r) else 0.0
            h = t_left if rmax <= 0 else min(t_left, 0.01 / rmax)
            if self.tau > 0:
                h = min(h, 0.2 * self.tau)
            k1a, k1T = self._deriv(self.a, self.T)
            k2a, k2T = self._deriv(self.a + 0.5 * h * k1a, self.T + 0.5 * h * k1T)
            k3a, k3T = self._deriv(self.a + 0.5 * h * k2a, self.T + 0.5 * h * k2T)
            k4a, k4T = self._deriv(self.a + h * k3a, self.T + h * k3T)
            self.a = np.minimum(self.a + h / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a), 1.0)
            self.T += h / 6.0 * (k1T + 2 * k2T + 2 * k3T + k4T)
            t_left -= h
        self.t += dt

    def alpha(self, li):
        return float(self.a[self.labels.index(li)])

    def gelled(self, li):
        return self.alpha(li) >= 0.999 * self.kin[li]["ag"]

    def all_gelled(self):
        return all(self.gelled(li) for li in self.labels)

    def viscosity(self, li):
        """Castro-Macosko viscosity now (inf at and past the gel point)."""
        k = self.kin[li]
        a = self.alpha(li)
        if a >= 0.999 * k["ag"]:
            return math.inf
        f = math.exp(k.get("Ea", 0.0) / R_GAS * (1.0 / self.T - 1.0 / self.Tm))
        return self.eta0[li] * f * (k["ag"] / (k["ag"] - a)) ** (k["A"] + k["B"] * a)

    def dt_for(self, dalpha):
        """The time for the fastest resin to cure by dalpha, or to reach its
        gel point, whichever is first."""
        r = self._rates(self.a, self.T)
        best = math.inf
        for i, li in enumerate(self.labels):
            ag = self.kin[li]["ag"]
            if self.a[i] >= 0.999 * ag or r[i] <= 0:
                continue
            best = min(best, min(dalpha, 0.999 * ag - self.a[i] + 1e-6) / r[i])
        return best


class Compress:
    """base: table label of every voxel of the initial N x N x N structure
    without the spheres (the resin regions, the pores; under a sphere, the
    resin it sits in). spheres: centres (n, 3) and radii (n,) in µm, their
    table labels (n,) and a raster(shape, centres) -> (owner, labels) drawing
    them on a grid. mu: viscosity per table label for the resins (Pa s),
    None for others; pore: bool per table label. h: voxel (µm)."""

    def __init__(self, base, h, mu, pore, centres, radii, sphere_label, raster, lateral="xy", compression=0.3,
                 rate=1.0, steps=None, log=None, should_stop=None, progress=None, tol=1e-4, frames=True,
                 cure=None, stress_limit=None, t_max=None):
        base = np.ascontiguousarray(base, np.uint8)
        self.h = float(h)
        self.L0 = np.array(base.shape, float) * self.h
        self.L = self.L0.copy()
        self.A = np.ones(3)                      # accumulated stretch of the box
        self.shape = tuple(base.shape)
        # the materials as markers, 2 x 2 x 2 per voxel, each with its volume
        self.cls_lab = np.unique(base)
        lut = np.zeros(256, np.int64)
        lut[self.cls_lab] = np.arange(len(self.cls_lab))
        n = 2
        g = [(np.arange(nn * n) + 0.5) * (self.h / n) for nn in self.shape]
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        self.mX = np.ascontiguousarray(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        del X, Y, Z
        ii = (self.mX / self.h).astype(np.int64)
        self.mC = lut[base[ii[:, 0], ii[:, 1], ii[:, 2]]].astype(np.int64)
        self.mV = np.full(len(self.mX), (self.h / n) ** 3)
        del ii
        self.base_labels = np.unique(base)
        self.pore = np.asarray(pore, bool)
        resin = np.array([m is not None for m in mu])
        self.resin = resin
        mu_r = np.array([float(m) if m is not None else np.nan for m in mu])
        lo, hi = float(np.nanmin(mu_r)), float(np.nanmax(mu_r))
        self.mu = np.where(resin, mu_r, np.where(self.pore, BUBBLE_RATIO * lo, RIGID_RATIO * hi))
        self.lam = np.where(self.pore, BUBBLE_RATIO * lo, PENALTY * hi)
        self.mu0 = self.mu.copy()
        self.mu_cap = RIGID_RATIO * hi             # a gelled resin: as stiff as the spheres
        self.c = np.array(centres, float).reshape(-1, 3)
        self.r = np.asarray(radii, float)
        self.slab = np.asarray(sphere_label, np.int64)
        self.raster = raster
        self.free = LATERAL[lateral]
        self.lateral = lateral
        self.comp = float(compression)
        self.rate = float(rate)
        eps_tot = -math.log(1.0 - self.comp)
        self.n_steps = int(steps or max(4, min(40, math.ceil(eps_tot / 0.05))))
        self.dt = eps_tot / self.n_steps / self.rate
        self.log = log or (lambda *a: None)
        self.should_stop = should_stop
        self.progress = progress or (lambda *a: None)
        self.tol = tol
        self.keep_frames = frames
        self.has_pores = bool(self.pore[self.cls_lab].any())
        # the cure: viscosities that rise with conversion; the closing rate
        # set, or limited by an axial stress the press can apply
        self.cure = cure
        self.stress_limit = float(stress_limit) if stress_limit else None
        self.t_max = float(t_max) if t_max else None

    def cure_viscosities(self):
        """Every curing resin at its viscosity now, its bulk penalty with it.
        The rest follows the resins that still flow, so that the contrasts
        stay those of the start: the spheres - and a resin past its gel
        point - RIGID_RATIO times the stiffest of them, the pores
        BUBBLE_RATIO times the softest. (Held at the start's values, a resin
        near its gel point reached the spheres' viscosity and the RVE became
        one uniform material.)"""
        eta = {li: self.cure.viscosity(li) for li in self.cure.labels}
        flowing = [v for v in eta.values() if math.isfinite(v)]
        flowing += [float(self.mu0[li]) for li in range(len(self.mu))
                    if self.resin[li] and li not in eta and math.isfinite(self.mu0[li])]
        if not flowing:
            return                                     # every resin gelled: the run stops
        hi, lo = max(flowing), min(flowing)
        rigid = RIGID_RATIO * hi
        for li, v in eta.items():
            self.mu[li] = v if math.isfinite(v) else rigid
            self.lam[li] = PENALTY * self.mu[li]
        solid = ~self.resin & ~self.pore
        self.mu[solid] = rigid
        self.lam[solid] = PENALTY * hi
        self.mu[self.pore] = BUBBLE_RATIO * lo
        self.lam[self.pore] = BUBBLE_RATIO * lo
        self.mu_cap = rigid

    def _mean_thinning(self, lab, dt, n=16):
        """Mean over the next dt of eta(now) / eta(t) of the curing resins
        (their volume-weighted viscosity), from a copy of the cure."""
        import copy
        c = copy.deepcopy(self.cure)
        w = np.array([float(np.count_nonzero(lab == li)) for li in c.labels])
        w = w / max(w.sum(), 1.0)

        def eta_mix(cc):
            return float(sum(wi * min(cc.viscosity(li), self.mu_cap) for wi, li in zip(w, cc.labels)))
        e0 = eta_mix(c)
        vals = [1.0]
        for _ in range(n):
            c.advance(dt / n)
            vals.append(e0 / eta_mix(c))
        return float(np.trapz(vals, dx=1.0 / n))

    @staticmethod
    def _scale(flow, f):
        """A flow solved at a unit rate, at the rate f."""
        for k in ("e", "x", "eps", "div", "stress", "sigma"):
            if flow.get(k) is not None:
                flow[k] = flow[k] * (np.float32(f) if k == "stress" else f)
        flow["P"] = flow["P"] * f

    # ------------------------------------------------------------------
    def field(self):
        """Table labels of the current grid without the spheres: the marker
        volumes of every material in every voxel, the labels then given so
        that each material keeps its volume. A voxel no marker reached takes
        the label of the nearest voxel that has one."""
        from scipy import ndimage
        K = len(self.cls_lab)
        M = int(np.prod(self.shape))
        hv = self.L / np.array(self.shape)
        vol = np.zeros((K, M))
        _deposit(self.mX, self.mV, self.mC, K, hv, np.array(self.shape, np.int64), vol)
        tot = vol.sum(axis=0)
        empty = tot <= 0
        frac = vol / np.where(empty, 1.0, tot)[None, :]
        k = assign_volumes(frac, vol.sum(axis=1) / float(np.prod(hv)))
        if empty.any():
            e3 = empty.reshape(self.shape)
            idx = ndimage.distance_transform_edt(e3, return_distances=False, return_indices=True)
            k = k.reshape(self.shape)[tuple(idx)].ravel()
        return self.cls_lab[k].astype(np.uint8).reshape(self.shape)

    def grid_centres(self):
        """Sphere centres on the voxel grid (the box drawn at h per voxel)."""
        return self.c * (np.array(self.shape) * self.h / self.L)

    def labels(self):
        """(labels of the structure, the materials without the spheres, the
        sphere of every voxel)."""
        fld = self.field()
        owner, slab = self.raster(self.shape, self.grid_centres())
        lab = fld.copy()
        inside = owner > 0
        lab[inside] = slab[owner[inside]]
        return lab, fld, owner

    @staticmethod
    def flow_labels(lab, fld, owner):
        """The labels the flow is solved on: where two spheres touch, the
        voxel layer between them is resin, so that touching spheres do not
        move as one rigid body (the resin film between them is thinner than a
        voxel, and the flow needs one to let them slide)."""
        if not (owner > 0).any():
            return lab
        out = lab.copy()
        for ax in range(3):
            nb = np.roll(owner, -1, axis=ax)
            m = (owner > 0) & (nb > 0) & (owner != nb)
            out[m] = fld[m]
        return out

    def surroundings(self, fld):
        """The resin around each sphere: the most frequent resin label at 26
        points half a voxel outside it."""
        n = len(self.c)
        if not n:
            return np.zeros(0, np.int64)
        hv = self.L / np.array(self.shape)
        pts = (self.c[:, None, :] + (self.r[:, None, None] + 0.75 * self.h) * _DIRS[None, :, :]).reshape(-1, 3)
        idx = np.floor(pts / hv).astype(np.int64) % np.array(self.shape)
        labs = fld[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(n, -1).astype(np.int64)
        out = np.empty(n, np.int64)
        for i in range(n):
            v = labs[i][self.resin[labs[i]]]
            out[i] = np.bincount(v).argmax() if len(v) else -1
        return out

    # ------------------------------------------------------------------
    def solve(self, lab):
        """The flow of one state: macroscopic strain rates, stress, the
        velocity fluctuation at the voxel corners (voxels per second) and the
        stress fields."""
        sol = FA.FANS(lab, None, None, None, lam_mu=(self.lam, self.mu), bbar=True)
        fx, fy = self.free
        # load cases: planar x - y, axisymmetric about z (both volume-keeping)
        # and the volume change (only with pores)
        cases = {"axi": np.array([-0.5, -0.5, 1.0, 0, 0, 0])}
        if fx or fy:
            cases["pl"] = np.array([1.0, -1.0, 0.0, 0, 0, 0])
        vol = self.has_pores and bool((self.pore[lab]).any())
        if vol:
            cases["vol"] = np.array([1.0, 1.0, 1.0, 0, 0, 0])
        S, U, its = {}, {}, 0
        prev = getattr(self, "_prev", {})
        if prev and getattr(self, "_prev_shape", None) != self.shape:
            prev = {k: self._resample(v) for k, v in prev.items()}
        for k, E in cases.items():
            # the previous step's solution of the same load case is a close
            # first guess (the structure moved little)
            sbar, st, _ = sol.solve(E, 0.0, self.tol, 20000, None, self.should_stop, keep_disp=True, x0=prev.get(k))
            S[k], U[k] = sbar[:3], sol.last_disp.copy()
            its += st["iterations"]
        self._prev = {k: v for k, v in U.items()}
        self._prev_shape = self.shape
        ez = -self.rate
        # e = a pl + b axi + c vol;  e_z = b + c = ez
        # e_x = a - b/2 + c,  e_y = -a - b/2 + c
        a, b, c = 0.0, ez, 0.0
        P = 0.0                                  # the free pressure of an incompressible RVE

        def sig(a_, b_, c_):
            s = b_ * S["axi"]
            if "pl" in S:
                s = s + a_ * S["pl"]
            if "vol" in S:
                s = s + c_ * S["vol"]
            return s
        if not vol:
            if fx and fy:
                # sigma_xx = sigma_yy (both equal the pressure) -> a
                dpl = S["pl"][0] - S["pl"][1]
                a = -b * (S["axi"][0] - S["axi"][1]) / dpl if abs(dpl) > 0 else 0.0
                P = sig(a, b, 0.0)[0]
            elif fx:
                a = -b / 2.0                      # e_y = -a - b/2 = 0
                P = sig(a, b, 0.0)[0]
            else:
                raise ValueError("Both sides held and no pores: an incompressible resin cannot be compressed")
        else:
            # unknowns (a, c); b = ez - c
            def resid(a_, c_):
                s = sig(a_, ez - c_, c_)
                r = []
                if fx and fy:
                    r = [s[0], s[1]]
                elif fx:
                    r = [s[0], -a_ - (ez - c_) / 2.0 + c_]          # sigma_x = 0, e_y = 0
                else:
                    r = [a_ - (ez - c_) / 2.0 + c_, -a_ - (ez - c_) / 2.0 + c_]   # e_x = e_y = 0
                return np.array(r)
            r0 = resid(0.0, 0.0)
            J = np.column_stack([resid(1.0, 0.0) - r0, resid(0.0, 1.0) - r0])
            a, c = np.linalg.solve(J, -r0)
            b = ez - c
        e = np.array([a - b / 2.0 + c, -a - b / 2.0 + c, b + c])
        x = b * U["axi"]
        if "pl" in U:
            x += a * U["pl"]
        if "vol" in U:
            x += c * U["vol"]
        U.clear()
        eps = np.empty((6,) + self.shape)
        FA._strain(x, sol.lab, sol.Bbar, eps)
        eps[:3] += e[:, None, None, None]
        div = (eps[0] + eps[1] + eps[2]).astype(np.float64)
        stress = np.empty((6,) + self.shape, np.float32)
        for m in np.unique(sol.lab):
            mask = sol.lab == m
            stress[:, mask] = (sol.C[m] @ eps[:, mask]).astype(np.float32)
        stress[:3] -= np.float32(P)
        mean_s = stress.reshape(6, -1).mean(axis=1).astype(float)
        return {"e": e, "x": x, "eps": eps, "div": div, "stress": stress, "sigma": mean_s, "iterations": its,
                "P": P}

    def _resample(self, v):
        """A nodal field of the previous grid on the current one: the box was
        stretched onto itself, so node i of the new grid sits at i n_old /
        n_new of the old (periodic, trilinear)."""
        from scipy import ndimage
        old = v.shape[1:]
        coords = np.meshgrid(*[np.arange(n) * (o / n) for n, o in zip(self.shape, old)], indexing="ij")
        out = np.empty((3,) + self.shape)
        for c in range(3):
            out[c] = ndimage.map_coordinates(v[c], coords, order=1, mode="grid-wrap")
        return out

    @staticmethod
    def von_mises(s):
        return np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
                       + 3.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)).astype(np.float32)

    # ------------------------------------------------------------------
    def move(self, flow, owner):
        """Advances the structure by one step of the flow."""
        dt, h = self.dt, self.h
        x = flow["x"]
        e = flow["e"]
        hv = self.L / np.array(self.shape)
        # the spheres: the mean velocity of their voxels
        if len(self.c):
            vc = _node_to_centre(x)
            ids = owner.ravel()
            n = len(self.c)
            cnt = np.bincount(ids, minlength=n + 1)[1:n + 1]
            v = np.zeros((n, 3))
            for k in range(3):
                v[:, k] = np.bincount(ids, weights=vc[k].ravel(), minlength=n + 1)[1:n + 1] / np.maximum(cnt, 1)
            v *= h                                # µm/s
            v += e[None, :] * self.c
            self.c = self.c + v * dt
        s = np.exp(e * dt)
        shp_prev = self.shape
        self.L = self.L * s
        self.A = self.A * s
        self.shape = tuple(max(4, int(round(l / h))) for l in self.L)
        # the materials: the markers with the flow, their volumes with its
        # divergence
        comp = self.pore[self.cls_lab]
        T = np.zeros(shp_prev)
        Pv = np.zeros(shp_prev)
        _voxel_volumes(self.mX, self.mV, self.mC, comp, hv, shp_prev[0], shp_prev[1], shp_prev[2], T, Pv)
        _advect(self.mX, self.mV, self.mC, comp, T, Pv, x, flow["div"], s, hv, h, dt, self.L)
        # the resin is incompressible: the small volume change its bulk
        # penalty lets through belongs to the pores - all markers together
        # fill the box exactly
        pm = comp[self.mC]
        if pm.any():
            excess = float(self.mV.sum()) - float(np.prod(self.L))
            pv = float(self.mV[pm].sum())
            if pv > 0:
                self.mV[pm] *= max(pv - excess, 0.0) / pv
        if len(self.c):
            ov = push_apart(self.c, self.r, self.L)
            self.c %= self.L
            return ov
        return 0.0

    # ------------------------------------------------------------------
    def run(self):
        t0 = time.time()
        series = {"compression": [], "sigma_zz_Pa": [], "sigma_xx_Pa": [], "sigma_yy_Pa": [], "strain_x": [],
                  "strain_y": [], "strain_z": [], "pore_fraction": [], "iterations": []}
        labels_all = sorted(set(self.cls_lab.tolist()) | set(self.slab.tolist()))
        reg = {int(li): {"strain": np.zeros(6), "strain_all": np.zeros(6), "vm": [], "volume": [],
                         "volume_all": []} for li in labels_all if self.resin[li]}
        frames = []
        if self.cure is not None:
            for k in ("time_s", "T_C", "rate_1_s"):
                series[k] = []
            series["alpha"] = {int(li): [] for li in self.cure.labels}
            series["eta_Pa_s"] = {int(li): [] for li in self.cure.labels}
        eps_target = -math.log(1.0 - self.comp)
        eps_done = 0.0
        stop_why = None
        rate_set = self.rate
        lab, fld, owner = self.labels()
        around0 = self.surroundings(fld)
        vol0 = {li: float(np.count_nonzero(lab == li)) * self.h ** 3 for li in reg}
        pore0 = float(np.mean(self.pore[lab]))
        max_ov = 0.0
        flow = None
        step = 0
        while True:
            if self.should_stop is not None and self.should_stop():
                raise InterruptedError
            if self.cure is not None:
                # solve at a unit rate, then close at the rate the press can
                self.cure_viscosities()
                self.rate = 1.0
                flow = self.solve(self.flow_labels(lab, fld, owner))
                s_unit = abs(float(flow["sigma"][2]))
                rate = rate_set
                if self.stress_limit and s_unit > 0:
                    rate = min(rate_set, self.stress_limit / s_unit)
                self._scale(flow, rate)
                self.rate = rate
            else:
                flow = self.solve(self.flow_labels(lab, fld, owner))
            vm = self.von_mises(flow["stress"])
            # the region of every sphere: the resin around it; a region is
            # its resin with the spheres in it
            sreg = np.concatenate([[-1], self.surroundings(fld), [-1]])
            in_reg = np.where(owner > 0, sreg[owner], lab)
            comp = 1.0 - self.L[2] / self.L0[2]
            series["compression"].append(float(comp))
            series["sigma_zz_Pa"].append(float(flow["sigma"][2]))
            series["sigma_xx_Pa"].append(float(flow["sigma"][0]))
            series["sigma_yy_Pa"].append(float(flow["sigma"][1]))
            series["strain_x"].append(float(math.log(self.A[0])))
            series["strain_y"].append(float(math.log(self.A[1])))
            series["strain_z"].append(float(math.log(self.A[2])))
            series["pore_fraction"].append(float(np.mean(self.pore[lab])))
            series["iterations"].append(int(flow["iterations"]))
            if self.cure is not None:
                series["time_s"].append(float(self.cure.t))
                series["T_C"].append(float(self.cure.T - 273.15))
                series["rate_1_s"].append(float(self.rate))
                for li in self.cure.labels:
                    series["alpha"][int(li)].append(self.cure.alpha(li))
                    series["eta_Pa_s"][int(li)].append(float(min(self.cure.viscosity(li), 1e30)))
            hv3 = float(np.prod(self.L / np.array(self.shape)))
            for li, rr in reg.items():
                m = lab == li
                nv = int(m.sum())
                rr["volume"].append(nv * hv3)
                rr["volume_all"].append(int(np.count_nonzero(in_reg == li)) * hv3)
                rr["vm"].append(float(vm[m].mean()) if nv else 0.0)
            # the von Mises stress of the resins (a rigid sphere carries the
            # load it is given; a pore none)
            vm_r = np.where(self.resin[lab] & (owner == 0), vm, np.float32(np.nan)).astype(np.float32)
            if self.keep_frames:
                frames.append(self.frame(lab, vm_r))
            self.progress(step, self.n_steps, comp, flow["sigma"][2])
            cure_txt = ""
            if self.cure is not None:
                cure_txt = (f", t {self.cure.t:.4g} s, {self.cure.T - 273.15:.4g} °C, rate {self.rate:.3g} 1/s, cure "
                            + ", ".join(f"{self.cure.alpha(li):.3f}" for li in self.cure.labels))
            self.log(f"    step {step}{'' if self.cure is not None else '/' + str(self.n_steps)}: compression {100 * comp:.1f} %, box "
                     + " × ".join(f"{v:.3g}" for v in self.L) + f" µm, σzz {flow['sigma'][2]:.4g} Pa, "
                     f"lateral strain rates {flow['e'][0] / max(self.rate, 1e-300):+.3f} / {flow['e'][1] / max(self.rate, 1e-300):+.3f}, "
                     f"{flow['iterations']} iterations" + cure_txt)
            if self.cure is None:
                if step == self.n_steps:
                    break
            else:
                if eps_done >= eps_target * (1.0 - 1e-6):
                    stop_why = "compressed"
                elif self.cure.all_gelled():
                    stop_why = "gelled"
                elif self.t_max and self.cure.t >= self.t_max:
                    stop_why = "time"
                elif self.rate < 1e-6 * max(rate_set, 1e-300):
                    # the closing has stopped: at the gel point, or short of it
                    near = all(self.cure.alpha(li) >= 0.98 * self.cure.kin[li]["ag"] for li in self.cure.labels)
                    stop_why = "gelled" if near else "stalled"
                elif step >= 200:
                    stop_why = "steps"
                if stop_why:
                    break
                # the step: at most 5 % strain and 4 % conversion, never past
                # the gel point of a resin or the time limit
                dt = min(0.05, eps_target - eps_done) / self.rate
                dt = min(dt, self.cure.dt_for(0.04))
                if self.t_max:
                    dt = min(dt, self.t_max - self.cure.t)
                self.dt = max(dt, 1e-9)
            # the strain of each resin over the step: the mean strain rate of
            # its voxels times the step
            for li, rr in reg.items():
                m = lab == li
                if m.any():
                    rr["strain"] += flow["eps"][:, m].mean(axis=1) * self.dt
                m = in_reg == li
                if m.any():
                    rr["strain_all"] += flow["eps"][:, m].mean(axis=1) * self.dt
            if self.cure is not None and self.stress_limit and self.rate < rate_set * (1.0 - 1e-9):
                # held at the stress limit, the rate falls as the resins
                # thicken during the step: the mean of eta(start) / eta(t)
                # over the step (the curing resins' volume-weighted viscosity)
                f = self._mean_thinning(lab, self.dt)
                rate_new = min(rate_set, self.rate * f)
                self._scale(flow, rate_new / self.rate)
                self.rate = rate_new
            flow["eps"] = None
            max_ov = max(max_ov, self.move(flow, owner))
            if self.cure is not None:
                self.cure.advance(self.dt)
                eps_done += self.rate * self.dt
            step += 1
            lab, fld, owner = self.labels()
        around1 = self.surroundings(fld)
        out = {"series": series, "frames": frames, "labels": lab, "field": fld, "owner": owner,
               "centres": self.c.copy(), "centres_grid": self.grid_centres(), "L_um": self.L.tolist(),
               "shape": list(self.shape), "von_mises": vm_r,
               "von_mises_all": vm,
               "pressure": (-(flow["stress"][0] + flow["stress"][1] + flow["stress"][2]) / 3.0).astype(np.float32),
               "regions": {li: {"strain": rr["strain"].tolist(), "strain_all": rr["strain_all"].tolist(),
                                "vm_Pa": rr["vm"], "volume_um3": rr["volume"], "volume_all_um3": rr["volume_all"],
                                "volume0_um3": vol0[li]} for li, rr in reg.items()},
               "pore_fraction": [pore0, float(np.mean(self.pore[lab]))],
               "crossed": int(np.count_nonzero((around0 >= 0) & (around1 >= 0) & (around0 != around1))),
               "around0": around0.tolist(), "around1": around1.tolist(),
               "max_overlap": max_ov, "steps": step, "seconds": time.time() - t0, "stop": stop_why}
        return out

    def frame(self, lab, vm):
        """The three visible faces of the box (top z, front y = 0, side
        x = end) and the three middle sections, labels and von Mises."""
        nx, ny, nz = lab.shape
        f = {"shape": [nx, ny, nz], "L_um": self.L.tolist()}
        for key, sl in (("top", (slice(None), slice(None), nz - 1)), ("front", (slice(None), 0, slice(None))),
                        ("side", (nx - 1, slice(None), slice(None))),
                        ("mxz", (slice(None), ny // 2, slice(None))), ("mxy", (slice(None), slice(None), nz // 2)),
                        ("myz", (nx // 2, slice(None), slice(None)))):
            f[key] = np.ascontiguousarray(lab[sl]).astype(np.uint8)
            f[key + "_vm"] = np.ascontiguousarray(vm[sl]).astype(np.float32)
        return f
