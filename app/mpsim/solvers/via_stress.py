"""Thermal stress of a through via in a glass (or other) substrate (5.8.0).

A plated via - a wall thickness t at the via's ends and throw * t in its middle (a parabola between), around a
core that is the composite of the RVE (its stiffness and expansion from the cte analysis), copper (a fully plated
via), empty (a conformal via) or a material entered - in a disc of the substrate. A uniform temperature change from
the stress-free state (the cure of the fill) to the temperature of use; all interfaces bonded; linear elastic.

The finite element model (axisymmetric, small strain):
* half the via along its axis: z = 0 is the substrate's mid-plane (symmetry, u_z = 0), z = H the free top surface;
  r = 0 the axis (u_r = 0 for a filled core), the disc's rim r = R free;
* a structured mesh mapped onto the three regions (core | wall | substrate), graded towards the wall's surfaces
  and towards the top, where the bimaterial corners at the hole's edge concentrate the stress;
* 9-node Lagrange quadrilaterals (biquadratic), 3 x 3 Gauss points, the thermal strain alpha dT as an
  eigenstrain; a sparse direct solve (SuperLU).
Stresses are read at fixed points (the substrate at the interface at mid-height; the substrate's top surface at
distances from the hole's edge; along the interface) rather than at the corners themselves, where linear
elasticity is singular. The mid-height is checked against the generalized plane strain solution of concentric
cylinders (Lame), and the model against FEniCSx (DOLFINx) with P2 triangles (self-test).

An optional backend runs the same model in FEniCSx (DOLFINx) when an installation is found (fenicsx_tgv.py).

Units inside: um, MPa, K.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

GP = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
GW = np.array([5.0, 8.0, 5.0]) / 9.0


def wall_profile(t_end, throw, z, H):
    s = np.asarray(z, float) / H
    return t_end * (throw + (1.0 - throw) * s ** 2)


def _graded(length, h0, q, hmax):
    """Element edges from 0 to length: the first size h0 growing by q up to hmax, then uniform."""
    xs = [0.0]
    h = h0
    while xs[-1] + h < length:
        xs.append(xs[-1] + h)
        h = min(h * q, hmax)
    if length - xs[-1] < 0.3 * min(h, hmax) and len(xs) > 1:
        xs[-1] = length
    else:
        xs.append(length)
    return np.array(xs)


def _two_sided(n_half, q):
    """Parametric edges on [0, 1], fine at both ends (geometric with ratio q)."""
    h = q ** np.arange(n_half)
    e = np.concatenate([[0.0], np.cumsum(np.concatenate([h, h[::-1]]))])
    return e / e[-1]


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


class Model:
    """Mesh, solve, evaluate. mats: dict region -> (E MPa, nu, alpha 1/K); regions 'core' (or None), 'wall', 'sub'."""

    def __init__(self, a, H, t_end, throw, R, mats, core, h_min=0.15, q=1.2, h_max=25.0):
        self.a, self.H, self.t_end, self.throw, self.R = a, H, t_end, throw, R
        self.core = core
        self.mats = mats
        t_top = t_end
        # z (element corners), fine at the top
        zt = _graded(H, h_min, q, h_max)                        # distance below the top
        self.z_e = np.sort(H - zt)
        self.z_e[0] = 0.0
        # columns: parametric edges
        cols = []
        if core:
            b_top = a - t_top
            sc = _graded(b_top, h_min, q, max(2.0, b_top / 12.0))
            cols.append(("core", np.sort(1.0 - sc / b_top)))       # fine towards s = 1 (the wall)
        nh = max(3, int(math.ceil(math.log(max(t_top / (2 * h_min) * (q - 1) + 1, 1.0001)) / math.log(q))))
        nh = min(nh, 10)
        cols.append(("wall", _two_sided(nh, q)))
        sg = _graded(R - a, h_min, q, h_max)
        cols.append(("sub", sg / (R - a)))
        self.cols = cols
        self._mesh()

    # ---------------------------------------------------------------- mesh
    def _rmap(self, region, s, z):
        b = self.a - wall_profile(self.t_end, self.throw, z, self.H)
        if region == "core":
            return s * b
        if region == "wall":
            return b + s * (self.a - b)
        return self.a + s * (self.R - self.a)

    def _mesh(self):
        # node columns at element corners and mid-points
        nodes_s = []                                          # (region, s) per node column
        elem_region = []
        for k, (reg, se) in enumerate(self.cols):
            sm = np.empty(2 * len(se) - 1)
            sm[0::2] = se
            sm[1::2] = 0.5 * (se[:-1] + se[1:])
            start = 0 if k == 0 else 1                         # shared with the previous column's last node
            nodes_s += [(reg, s) for s in sm[start:]]
            elem_region += [reg] * (len(se) - 1)
        ze = self.z_e
        zn = np.empty(2 * len(ze) - 1)
        zn[0::2] = ze
        zn[1::2] = 0.5 * (ze[:-1] + ze[1:])
        self.zn = zn
        ni, nj = len(nodes_s), len(zn)
        R_ = np.empty((ni, nj))
        for i, (reg, s) in enumerate(nodes_s):
            R_[i] = self._rmap(reg, s, zn)
        self.ni, self.nj = ni, nj
        self.xr = R_.ravel()                                   # node index = i * nj + j
        self.xz = np.tile(zn, ni)
        ner, nez = (ni - 1) // 2, (nj - 1) // 2
        self.ner, self.nez = ner, nez
        ie, je = np.meshgrid(np.arange(ner), np.arange(nez), indexing="ij")
        ie, je = ie.ravel(), je.ravel()
        a_, b_ = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        a_, b_ = a_.ravel(), b_.ravel()                        # local node k = 3 a + b
        self.conn = ((2 * ie[:, None] + a_[None, :]) * nj + 2 * je[:, None] + b_[None, :])
        self.eregion = np.array(elem_region)[ie]
        self.ie, self.je = ie, je
        self.la, self.lb = a_, b_

    # ---------------------------------------------------------------- shape functions
    @staticmethod
    def _L(x):
        return np.stack([0.5 * x * (x - 1), 1 - x * x, 0.5 * x * (x + 1)], -1)

    @staticmethod
    def _dL(x):
        return np.stack([x - 0.5, -2 * x, x + 0.5], -1)

    def _shape(self, xi, eta):
        """N (..., 9), dN/dxi, dN/deta for local node k = 3 a + b (a along r, b along z)."""
        Lx, Ly, dLx, dLy = self._L(xi), self._L(eta), self._dL(xi), self._dL(eta)
        N = Lx[..., self.la] * Ly[..., self.lb]
        dxi = dLx[..., self.la] * Ly[..., self.lb]
        deta = Lx[..., self.la] * dLy[..., self.lb]
        return N, dxi, deta

    def _D(self):
        lam = np.empty(len(self.eregion))
        mu = np.empty_like(lam)
        al = np.empty_like(lam)
        for reg, (E, nu, alpha) in self.mats.items():
            m = self.eregion == reg
            lam[m], mu[m] = lame(E, nu)
            al[m] = alpha
        return lam, mu, al

    def _B(self, e, xi, eta):
        """B (n, 4, 18) and r at (xi, eta) of the elements e; rows err, ezz, ett, grz; dofs (ur, uz) per node."""
        N, dxi, deta = self._shape(np.asarray(xi, float), np.asarray(eta, float))
        X = np.stack([self.xr[self.conn[e]], self.xz[self.conn[e]]], -1)       # (n, 9, 2)
        if N.ndim == 1:
            N, dxi, deta = (np.broadcast_to(v, (len(e), 9)) for v in (N, dxi, deta))
        J = np.empty((len(e), 2, 2))
        J[:, 0, 0] = np.einsum("nk,nk->n", dxi, X[..., 0])
        J[:, 0, 1] = np.einsum("nk,nk->n", dxi, X[..., 1])
        J[:, 1, 0] = np.einsum("nk,nk->n", deta, X[..., 0])
        J[:, 1, 1] = np.einsum("nk,nk->n", deta, X[..., 1])
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        inv = np.stack([np.stack([J[:, 1, 1], -J[:, 0, 1]], -1), np.stack([-J[:, 1, 0], J[:, 0, 0]], -1)], -2) / det[:, None, None]
        dr = inv[:, 0, 0, None] * dxi + inv[:, 0, 1, None] * deta
        dz = inv[:, 1, 0, None] * dxi + inv[:, 1, 1, None] * deta
        r = np.einsum("nk,nk->n", N, X[..., 0])
        B = np.zeros((len(e), 4, 18))
        B[:, 0, 0::2] = dr
        B[:, 1, 1::2] = dz
        on_axis = r < 1e-9
        rr = np.where(on_axis, 1.0, r)
        B[:, 2, 0::2] = np.where(on_axis[:, None], dr, N / rr[:, None])
        B[:, 3, 0::2] = dz
        B[:, 3, 1::2] = dr
        return B, r, det

    # ---------------------------------------------------------------- solve
    def solve(self, dT):
        t0 = time.time()
        lam, mu, al = self._D()
        ne = len(self.eregion)
        D = np.zeros((ne, 4, 4))
        for i in range(3):
            for j in range(3):
                D[:, i, j] = lam
            D[:, i, i] = lam + 2 * mu
        D[:, 3, 3] = mu
        eth = np.zeros((ne, 4))
        eth[:, :3] = (al * dT)[:, None]
        Ke = np.zeros((ne, 18, 18))
        fe = np.zeros((ne, 18))
        e = np.arange(ne)
        for xi, wx in zip(GP, GW):
            for eta, wy in zip(GP, GW):
                B, r, det = self._B(e, xi, eta)
                w = wx * wy * r * det
                DB = np.einsum("nij,njk->nik", D, B)
                Ke += np.einsum("nji,njk->nik", B, DB) * w[:, None, None]
                fe += np.einsum("nji,nj->ni", B, np.einsum("nij,nj->ni", D, eth)) * w[:, None]
        dof = np.empty((ne, 18), int)
        dof[:, 0::2] = 2 * self.conn
        dof[:, 1::2] = 2 * self.conn + 1
        nd = 2 * self.ni * self.nj
        K = sparse.coo_matrix((Ke.ravel(), (np.repeat(dof, 18, 1).ravel(), np.tile(dof, (1, 18)).ravel())), shape=(nd, nd)).tocsr()
        f = np.zeros(nd)
        np.add.at(f, dof.ravel(), fe.ravel())
        fixed = list(2 * np.where(self.xz < 1e-12)[0] + 1)               # u_z = 0 on the mid-plane
        if self.core:
            fixed += list(2 * np.where(self.xr < 1e-12)[0])              # u_r = 0 on the axis
        fixed = np.unique(fixed)
        free = np.setdiff1d(np.arange(nd), fixed)
        u = np.zeros(nd)
        u[free] = spsolve(K[free][:, free].tocsc(), f[free])
        self.u, self.dT = u, dT
        self.lam, self.mu, self.al = lam, mu, al
        self.n_dofs, self.n_elem, self.time_s = int(len(free)), int(ne), time.time() - t0
        return self

    # ---------------------------------------------------------------- evaluation
    def _locate(self, r, z, prefer):
        """Element and local (xi, eta) of points (r, z), preferring the region `prefer` on its boundaries."""
        r, z = np.atleast_1d(np.asarray(r, float)), np.atleast_1d(np.asarray(z, float))
        je = np.clip(np.searchsorted(self.z_e, z, side="right") - 1, 0, self.nez - 1)
        z0, z1 = self.z_e[je], self.z_e[je + 1]
        eta = 2 * (z - z0) / (z1 - z0) - 1
        out_e, out_xi = np.empty(len(r), int), np.empty(len(r))
        cand = np.arange(self.ner)
        for p in range(len(r)):
            es = cand * self.nez + je[p]
            Ly = self._L(eta[p])
            # r of the three node lines of every element of this row at eta
            rl = np.einsum("b,nab->na", Ly, self.xr[self.conn[es]].reshape(-1, 3, 3))
            inside = (rl[:, 0] <= r[p] + 1e-9) & (rl[:, 2] >= r[p] - 1e-9)
            idx = np.where(inside)[0]
            if not len(idx):
                raise ValueError(f"point r = {r[p]:.4g} um, z = {z[p]:.4g} um lies outside the model")
            pick = idx[0]
            if prefer is not None:
                pr = [k for k in idx if self.eregion[es[k]] == prefer]
                if pr:
                    pick = pr[0]
            c = rl[pick]
            # r(xi) = c0 L0 + c1 L1 + c2 L2: Newton from the linear guess
            x = np.clip(2 * (r[p] - c[0]) / max(c[2] - c[0], 1e-30) - 1, -1, 1)
            for _ in range(30):
                Lx, dLx = self._L(x), self._dL(x)
                g = Lx @ c - r[p]
                dg = dLx @ c
                x = float(np.clip(x - g / dg, -1, 1))
                if abs(g) < 1e-12 * max(1.0, abs(r[p])):
                    break
            out_e[p], out_xi[p] = es[pick], x
        return out_e, out_xi, eta

    def stress(self, r, z, prefer=None):
        """Stress (n, 4): rr, zz, tt, rz at points."""
        e, xi, eta = self._locate(r, z, prefer)
        B, rr, _ = self._B(e, xi, eta)
        dof = np.empty((len(e), 18), int)
        dof[:, 0::2] = 2 * self.conn[e]
        dof[:, 1::2] = 2 * self.conn[e] + 1
        eps = np.einsum("nij,nj->ni", B, self.u[dof])
        lam, mu, al = self.lam[e], self.mu[e], self.al[e]
        em = eps.copy()
        em[:, :3] -= (al * self.dT)[:, None]
        tr = em[:, 0] + em[:, 1] + em[:, 2]
        s = np.empty_like(em)
        s[:, 0] = lam * tr + 2 * mu * em[:, 0]
        s[:, 1] = lam * tr + 2 * mu * em[:, 1]
        s[:, 2] = lam * tr + 2 * mu * em[:, 2]
        s[:, 3] = mu * em[:, 3]
        return s

    def displacement(self, r, z, prefer=None):
        e, xi, eta = self._locate(r, z, prefer)
        N, _, _ = self._shape(xi, eta)
        return np.stack([np.einsum("nk,nk->n", N, self.u[2 * self.conn[e]]), np.einsum("nk,nk->n", N, self.u[2 * self.conn[e] + 1])], -1)

    def field_nodes(self, rmax, zmin):
        """Element-local nodal stresses of the elements in a window (for maps): points (m, 2), p1 (m,), triangles."""
        keep = np.where((self.xr[self.conn[:, 0]] < rmax) & (self.xz[self.conn[:, 8]] > zmin))[0]
        xi = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], float)
        eta = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], float)
        pts, vals, tris, regs = [], [], [], []
        base = 0
        sub = [(0, 3, 4, 1), (1, 4, 5, 2), (3, 6, 7, 4), (4, 7, 8, 5)]
        for e in keep:
            ee = np.full(9, e)
            B, r, _ = self._B(ee, xi, eta)
            dof = np.empty((9, 18), int)
            dof[:, 0::2] = 2 * self.conn[ee]
            dof[:, 1::2] = 2 * self.conn[ee] + 1
            eps = np.einsum("nij,nj->ni", B, self.u[dof])
            em = eps.copy()
            em[:, :3] -= self.al[e] * self.dT
            tr = em[:, :3].sum(1)
            s = np.stack([self.lam[e] * tr + 2 * self.mu[e] * em[:, 0], self.lam[e] * tr + 2 * self.mu[e] * em[:, 1],
                          self.lam[e] * tr + 2 * self.mu[e] * em[:, 2], self.mu[e] * em[:, 3]], -1)
            pts.append(np.stack([self.xr[self.conn[e]], self.xz[self.conn[e]]], -1))
            vals.append(principal(s))
            regs.append(np.full(9, self.eregion[e]))
            for q0, q1, q2, q3 in sub:
                tris += [(base + q0, base + q1, base + q2), (base + q0, base + q2, base + q3)]
            base += 9
        return np.concatenate(pts), np.concatenate(vals), np.array(tris), np.concatenate(regs)


def principal(s):
    rr, zz, tt, rz = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    c, d = 0.5 * (rr + zz), np.sqrt((0.5 * (rr - zz)) ** 2 + rz ** 2)
    return np.max(np.c_[c + d, c - d, tt], axis=1)


def mises(s):
    rr, zz, tt, rz = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    return np.sqrt(0.5 * ((rr - zz) ** 2 + (zz - tt) ** 2 + (tt - rr) ** 2) + 3 * rz ** 2)


def lame_cylinders(layers, dT):
    """Concentric cylinders, generalized plane strain (uniform axial strain, zero axial force), free outside (and
    inside for a hollow innermost layer): layers [(r_in, r_out, E, nu, alpha)] -> stress(r) dict."""
    n = len(layers)
    lam = [lame(E, nu)[0] for _, _, E, nu, _ in layers]
    mu = [lame(E, nu)[1] for _, _, E, nu, _ in layers]
    beta = [(3 * lm + 2 * m) * al for lm, m, (_, _, _, _, al) in zip(lam, mu, layers)]
    N = 2 * n + 1
    M = np.zeros((N, N))
    rhs = np.zeros(N)
    row = 0

    def sr(i, r):
        c = np.zeros(N)
        c[2 * i], c[2 * i + 1], c[-1] = 2 * (lam[i] + mu[i]), -2 * mu[i] / r ** 2, lam[i]
        return c, beta[i] * dT
    if layers[0][0] == 0.0:
        M[row, 1] = 1.0
    else:
        M[row], rhs[row] = sr(0, layers[0][0])
    row += 1
    for i in range(n - 1):
        r = layers[i][1]
        M[row, 2 * i], M[row, 2 * i + 1], M[row, 2 * i + 2], M[row, 2 * i + 3] = r, 1 / r, -r, -1 / r
        row += 1
        c1, b1 = sr(i, r)
        c2, b2 = sr(i + 1, r)
        M[row], rhs[row] = c1 - c2, b1 - b2
        row += 1
    M[row], rhs[row] = sr(n - 1, layers[-1][1])
    row += 1
    for i, (ri, ro, *_x) in enumerate(layers):
        A = math.pi * (ro ** 2 - ri ** 2)
        M[row, 2 * i] += 2 * lam[i] * A
        M[row, -1] += (lam[i] + 2 * mu[i]) * A
        rhs[row] += beta[i] * dT * A
    x = np.linalg.solve(M, rhs)

    def st(r):
        r = np.asarray(r, float)
        out = {k: np.full(r.shape, np.nan) for k in ("rr", "tt", "zz")}
        for i, (ri, ro, *_x) in enumerate(layers):
            m = (r >= ri) & (r <= ro)
            A, Bc, e0 = x[2 * i], x[2 * i + 1], x[-1]
            out["rr"][m] = 2 * (lam[i] + mu[i]) * A - 2 * mu[i] * Bc / r[m] ** 2 + lam[i] * e0 - beta[i] * dT
            out["tt"][m] = 2 * (lam[i] + mu[i]) * A + 2 * mu[i] * Bc / r[m] ** 2 + lam[i] * e0 - beta[i] * dT
            out["zz"][m] = 2 * lam[i] * A + (lam[i] + 2 * mu[i]) * e0 - beta[i] * dT
        return out
    return st


TOP_D = [1.0, 2.0, 5.0, 10.0, 20.0]
DEPTHS = np.r_[np.geomspace(0.05, 30.0, 60), np.linspace(32.0, 880.0, 40)]


def solve_case(a, H, t_end, throw, R, core, mats, dT, h_min=0.15):
    """One via: core in ('fill', 'cu', 'none'); mats: wall, sub, fill as (E MPa, nu, alpha 1/K)."""
    m = {"wall": mats["wall"], "sub": mats["sub"]}
    if core == "fill":
        m["core"] = mats["fill"]
    elif core == "cu":
        m["core"] = mats["wall"]
    mod = Model(a, H, t_end, throw, R, m, core != "none", h_min=h_min).solve(dT)
    t_mid = throw * t_end
    out = {"t_um": t_end, "t_mid_um": t_mid, "core": core, "n_dofs": mod.n_dofs, "n_elem": mod.n_elem, "time_s": mod.time_s}
    s_mid = mod.stress([a], [0.0], prefer="sub")[0]
    out["mid"] = {"rr": float(s_mid[0]), "tt": float(s_mid[2]), "zz": float(s_mid[1])}
    st = mod.stress(a + np.array(TOP_D), np.full(len(TOP_D), H), prefer="sub")
    out["top_p1"] = principal(st).tolist()
    out["top_d_um"] = TOP_D
    sd = mod.stress(np.full(len(DEPTHS), a), H - DEPTHS, prefer="sub")
    out["depth_um"] = DEPTHS.tolist()
    out["depth_p1"] = principal(sd).tolist()
    out["depth_rz"] = sd[:, 3].tolist()
    sw = mod.stress([a - 0.5 * t_mid], [0.0], prefer="wall")
    out["wall_mises_mid"] = float(mises(sw)[0])
    rs = np.linspace(0.0 if core != "none" else a - t_end, a, 40)[1:-1]
    uz = mod.displacement(rs, np.full(len(rs), H), prefer="wall" if core == "none" else None)[:, 1]
    uz_far = mod.displacement([R], [H], prefer="sub")[0, 1]
    out["top_dz_nm"] = float((uz.mean() - uz_far) * 1000.0)
    out["glass_max"] = max(out["mid"]["rr"], out["top_p1"][2])
    # the generalized plane strain solution at mid-height
    lay = []
    if core != "none":
        lay.append((0.0, a - t_mid) + tuple(m["core"]))
        lay.append((a - t_mid, a) + tuple(m["wall"]))
    else:
        lay.append((a - t_mid, a) + tuple(m["wall"]))
    lay.append((a, R) + tuple(m["sub"]))
    ana = lame_cylinders(lay, dT)(np.array([a + 1e-9]))
    out["lame_mid_rr"] = float(ana["rr"][0])
    out["_model"] = mod
    return out


def analyse(so, a_um, length_um, t_end, throw, sweep, fill_props, wall_props, log=print, map_png=None):
    """so: the stress options; fill_props/wall_props (E GPa, nu, alpha ppm/K) or None for an empty core.
    Returns the result entry (JSON-able)."""
    t0 = time.time()
    dT = so["T_use"] - so["T_free"]
    mats = {"wall": (wall_props[0] * 1000.0, wall_props[1], wall_props[2] * 1e-6),
            "sub": (so["sub_E"] * 1000.0, so["sub_nu"], so["sub_alpha"] * 1e-6)}
    core = "none"
    if fill_props is not None:
        mats["fill"] = (fill_props[0] * 1000.0, fill_props[1], fill_props[2] * 1e-6)
        core = "fill"
    R = so["R_factor"] * a_um
    H = 0.5 * length_um
    h = {"standard": 0.15, "fine": 0.075}.get(so.get("mesh", "standard"), 0.15)

    def strip(c):
        c = dict(c)
        c.pop("_model", None)
        return c
    main = solve_case(a_um, H, t_end, throw, R, core, mats, dT, h)
    mod = main["_model"]
    out = {"dT": dT, "T_free": so["T_free"], "T_use": so["T_use"], "R_um": R, "core": core,
           "materials": {k: {"E_GPa": v[0] / 1000.0, "nu": v[1], "alpha_ppm": v[2] * 1e6} for k, v in mats.items()},
           "main": strip(main), "solver": "MPSim FEM (axisymmetric, 9-node quadrilaterals)"}
    out["lame_dev"] = abs(main["mid"]["rr"] / main["lame_mid_rr"] - 1.0) if main["lame_mid_rr"] else 0.0
    sweep = sorted(set(list(sweep or []) + [t_end]))
    comp = {}
    if so.get("compare", True):
        kinds = [("fill", core)] if core == "fill" and so.get("core") != "cu" else []
        kinds += [("none", "none"), ("cu", "cu")]
        for lab, kind in kinds:
            comp[lab] = []
            if lab == "cu":
                # a fully plated via is copper whatever the wall: one solve, shown at every wall thickness
                one = strip(main) if so.get("core") == "cu" else strip(solve_case(a_um, H, t_end, throw, R, "cu", mats, dT, h))
                comp[lab] = [dict(one, t_um=t, t_mid_um=throw * t) for t in sweep]
                continue
            for t in sweep:
                if lab == "fill" and t == t_end:
                    comp[lab].append(strip(main))
                    continue
                comp[lab].append(strip(solve_case(a_um, H, t, throw, R, kind, mats, dT, h)))
    out["compare"] = comp
    if map_png:
        try:
            _map(mod, a_um, H, main, map_png, so)
            out["map_png"] = map_png
        except Exception as e:  # noqa: BLE001
            log(f"   via stress map not drawn: {e}")
    out["time_s"] = time.time() - t0
    log(f"   via thermal stress (dT {dT:g} K): substrate at the interface mid-height {main['mid']['rr']:.1f} MPa, "
        f"top surface 5 um from the hole {main['top_p1'][2]:.1f} MPa; Lame check {100 * out['lame_dev']:.2f} %; "
        f"{main['n_dofs']} dofs, {out['time_s']:.1f} s")
    return out


def _map(mod, a, H, main, path, so):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    depth = 40.0
    pts, p1, tris, reg = mod.field_nodes(rmax=a + 70.0, zmin=H - depth - 10.0)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    tri = Triangulation(pts[:, 0], pts[:, 1] - H, tris)
    # the colours follow the substrate (what may crack); the copper above that range shows saturated
    sub_ = p1[reg == "sub"]
    lim = max(float(np.percentile(sub_, 99.7)) if len(sub_) else 50.0, 50.0)
    im = ax.tripcolor(tri, np.clip(p1, -0.5 * lim, lim), shading="gouraud", cmap="jet", vmin=-0.5 * lim, vmax=lim)
    zz = np.linspace(H - depth - 10.0, H, 60)
    ax.plot(a - wall_profile(main["t_um"], mod.throw, zz, H), zz - H, color="white", lw=0.8)
    ax.axvline(a, color="white", lw=0.8)
    ax.set_xlim(max(0.0, a - 40.0), a + 70)
    ax.set_ylim(-depth, 0)
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("depth below the top (µm)")
    ax.set_title(f"Largest principal stress near the via's top (ΔT {so['T_use'] - so['T_free']:g} K)", fontsize=9)
    ax.text(a + 2, -depth + 2, "substrate", color="white", fontsize=8)
    cb = fig.colorbar(im, ax=ax, extend="both")
    cb.set_label("MPa (colours set by the substrate)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
