"""Elasticity and thermal expansion on the body-fitted tetrahedral mesh (5.10.0).

The mesh of tetfem.build_mesh / tetmesh.build_mesh with linear (P1) displacement elements:

* the periodic cell problem: u = E.x + w, w periodic, for the six unit strains E (Voigt order xx yy zz yz xz xy,
  engineering shear) and for the thermal eigenstrain alpha dT of every material; C* from the energy,
  alpha* = -C*^-1 sigma_th;
* a film of matrix between two particles thinner than the mesh is an interface element, as in the conduction: a
  layer of matrix of the film's width held between two stiff faces, so its stiffness per area is
  (lambda + 2 mu) / width across it and mu / width along it (the thin-layer limit; the width is the film's,
  halved on each side of its middle vertex). Overlapping particles share their vertex (bonded);
* conjugate gradients preconditioned with smoothed-aggregation multigrid (PyAMG's hierarchy on the rigid-body
  modes, cycled on all cores by mgsolve.py), the seven loads in one block.

A mesh made without periodic wrap is held at its faces (u = E.x there, w = 0): the problem FEniCSx solves on the
same mesh in solvers/fenicsx_tet.py, to the solvers' tolerance.

Reductions: interfaces are bonded (no debonding, no sliding - a cured compound at small strain); linear
tetrahedra are stiffer than quadratic ones in bending and lock for a nearly incompressible material
(nu > 0.49: use the voxel solver with B-bar elements); the film has no thermal strain of its own.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit
from scipy import sparse

from . import tetfem as TF

NAMES = ["xx", "yy", "zz", "yz", "xz", "xy"]
_SH = np.array([[1, 2], [0, 2], [0, 1]], np.int64)


@njit(cache=True)
def _find(indptr, indices, r, c):
    lo, hi = indptr[r], indptr[r + 1] - 1
    while lo <= hi:
        m = (lo + hi) >> 1
        if indices[m] == c:
            return m
        if indices[m] < c:
            lo = m + 1
        else:
            hi = m - 1
    return -1


@njit(cache=True)
def _sig(k, lam, mu, s):
    """The stress of unit strain k in an isotropic element, as a 3 x 3 array."""
    for i in range(3):
        for j in range(3):
            s[i, j] = 0.0
    if k < 3:
        for i in range(3):
            s[i, i] = lam
        s[k, k] += 2.0 * mu
    else:
        i, j = _SH[k - 3, 0], _SH[k - 3, 1]
        s[i, j] = mu
        s[j, i] = mu


@njit(cache=True)
def _fill(tets, points, per, L, lam, mu, bt, mat, indptr, indices, data, F, c0, s0):
    """Stiffness blocks, the loads of the six strains and the thermal strain, and the volume sums."""
    e = np.empty((3, 3))
    g = np.empty((4, 3))
    s = np.empty((3, 3))
    vol_sum = 0.0
    for t in range(tets.shape[0]):
        vol = TF._grads(tets, t, points, per, L, e, g)
        if vol <= 0.0:
            continue
        vol_sum += vol
        la, m_, b_ = lam[mat[t]], mu[mat[t]], bt[mat[t]]
        for a in range(4):
            va = tets[t, a]
            for q in range(4):
                pos = _find(indptr, indices, va, tets[t, q])
                gg = g[a, 0] * g[q, 0] + g[a, 1] * g[q, 1] + g[a, 2] * g[q, 2]
                for i in range(3):
                    for j in range(3):
                        data[pos, i, j] += vol * (la * g[a, i] * g[q, j] + m_ * g[a, j] * g[q, i])
                    data[pos, i, i] += vol * m_ * gg
            for k in range(6):
                _sig(k, la, m_, s)
                for i in range(3):
                    F[va, i, k] -= vol * (s[i, 0] * g[a, 0] + s[i, 1] * g[a, 1] + s[i, 2] * g[a, 2])
            for i in range(3):
                F[va, i, 6] += vol * b_ * g[a, i]
        for k in range(3):
            s0[k] -= vol * b_
            for k2 in range(3):
                c0[k, k2] += vol * la
            c0[k, k] += vol * 2.0 * m_
            c0[k + 3, k + 3] += vol * m_
    return vol_sum


@njit(cache=True)
def _springs(sp, sw, sn, Mn, Mt, indptr, indices, data):
    for q in range(sp.shape[0]):
        a, b = sp[q, 0], sp[q, 1]
        nx, ny, nz = sn[q, 0], sn[q, 1], sn[q, 2]
        nr = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nr <= 0.0 or sw[q] <= 0.0:
            continue
        n = (nx / nr, ny / nr, nz / nr)
        paa, pbb = _find(indptr, indices, a, a), _find(indptr, indices, b, b)
        pab, pba = _find(indptr, indices, a, b), _find(indptr, indices, b, a)
        for i in range(3):
            for j in range(3):
                kij = sw[q] * ((Mn - Mt) * n[i] * n[j] + (Mt if i == j else 0.0))
                data[paa, i, j] += kij
                data[pbb, i, j] += kij
                data[pab, i, j] -= kij
                data[pba, i, j] -= kij


@njit(cache=True)
def _stress(tets, points, per, L, lam, mu, bt, mat, u, Ev, dT, out):
    """Stress per tetrahedron (xx yy zz yz xz xy) of E + grad_s(u) - alpha dT."""
    e = np.empty((3, 3))
    g = np.empty((4, 3))
    H = np.empty((3, 3))
    for t in range(tets.shape[0]):
        vol = TF._grads(tets, t, points, per, L, e, g)
        for i in range(6):
            out[t, i] = 0.0
        if vol <= 0.0:
            continue
        for i in range(3):
            for j in range(3):
                H[i, j] = 0.0
        for a in range(4):
            va = tets[t, a]
            for i in range(3):
                for j in range(3):
                    H[i, j] += u[3 * va + i] * g[a, j]
        exx, eyy, ezz = Ev[0] + H[0, 0], Ev[1] + H[1, 1], Ev[2] + H[2, 2]
        gyz, gxz, gxy = Ev[3] + H[1, 2] + H[2, 1], Ev[4] + H[0, 2] + H[2, 0], Ev[5] + H[0, 1] + H[1, 0]
        la, m_ = lam[mat[t]], mu[mat[t]]
        tr = la * (exx + eyy + ezz) - bt[mat[t]] * dT
        out[t, 0] = tr + 2.0 * m_ * exx
        out[t, 1] = tr + 2.0 * m_ * eyy
        out[t, 2] = tr + 2.0 * m_ * ezz
        out[t, 3] = m_ * gyz
        out[t, 4] = m_ * gxz
        out[t, 5] = m_ * gxy


@njit(cache=True)
def raster(tets, points, per, L, h, cells, val, out, hit):
    """val (per tetrahedron, k columns) at the voxel centres: every voxel takes the tetrahedron its centre lies in."""
    X = np.empty((4, 3))
    T = np.empty((3, 3))
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    idx = np.empty(3, np.int64)
    for t in range(tets.shape[0]):
        for a in range(3):
            X[0, a] = points[tets[t, 0], a]
        for m in range(1, 4):
            for a in range(3):
                d = points[tets[t, m], a] - X[0, a]
                if per[a]:
                    d -= L[a] * math.floor(d / L[a] + 0.5)
                X[m, a] = X[0, a] + d
        for a in range(3):
            mn = min(X[0, a], X[1, a], X[2, a], X[3, a])
            mx = max(X[0, a], X[1, a], X[2, a], X[3, a])
            lo[a] = int(math.ceil(mn / h - 0.5 - 1e-9))
            hi[a] = int(math.floor(mx / h - 0.5 + 1e-9))
            for m in range(3):
                T[a, m] = X[m + 1, a] - X[0, a]
        det = (T[0, 0] * (T[1, 1] * T[2, 2] - T[1, 2] * T[2, 1]) - T[0, 1] * (T[1, 0] * T[2, 2] - T[1, 2] * T[2, 0])
               + T[0, 2] * (T[1, 0] * T[2, 1] - T[1, 1] * T[2, 0]))
        if det == 0.0 or lo[0] > hi[0] or lo[1] > hi[1] or lo[2] > hi[2]:
            continue
        inv = 1.0 / det
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    rx, ry, rz = (i + 0.5) * h - X[0, 0], (j + 0.5) * h - X[0, 1], (k + 0.5) * h - X[0, 2]
                    b1 = (rx * (T[1, 1] * T[2, 2] - T[1, 2] * T[2, 1]) - T[0, 1] * (ry * T[2, 2] - T[1, 2] * rz)
                          + T[0, 2] * (ry * T[2, 1] - T[1, 1] * rz)) * inv
                    b2 = (T[0, 0] * (ry * T[2, 2] - T[1, 2] * rz) - rx * (T[1, 0] * T[2, 2] - T[1, 2] * T[2, 0])
                          + T[0, 2] * (T[1, 0] * rz - ry * T[2, 0])) * inv
                    b3 = (T[0, 0] * (T[1, 1] * rz - ry * T[2, 1]) - T[0, 1] * (T[1, 0] * rz - ry * T[2, 0])
                          + rx * (T[1, 0] * T[2, 1] - T[1, 1] * T[2, 0])) * inv
                    if b1 < -1e-9 or b2 < -1e-9 or b3 < -1e-9 or b1 + b2 + b3 > 1.0 + 1e-9:
                        continue
                    idx[0], idx[1], idx[2] = i, j, k
                    ok = True
                    for a in range(3):
                        if idx[a] < 0 or idx[a] >= cells[a]:
                            if per[a]:
                                idx[a] %= cells[a]
                            else:
                                ok = False
                    if not ok:
                        continue
                    c = (idx[0] * cells[1] + idx[1]) * cells[2] + idx[2]
                    hit[c] = 1
                    for q in range(val.shape[1]):
                        out[c, q] = val[t, q]


def _pattern(mesh):
    """The node graph (CSR, sorted): every pair of vertices that share a tetrahedron or a film spring."""
    tets = mesh["tets"]
    nv = len(mesh["points"])
    it = np.int32 if nv < 2 ** 31 else np.int64
    A = None
    for a in range(0, len(tets), TF.CHUNK):
        tt = tets[a:a + TF.CHUNK]
        rows = np.repeat(tt, 4, axis=1).ravel().astype(it)
        cols = np.tile(tt, (1, 4)).ravel().astype(it)
        Ac = sparse.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(nv, nv)).tocsr()
        Ac.data[:] = 1
        A = Ac if A is None else A + Ac
    sp = mesh.get("springs")
    if sp is not None and len(sp):
        a, b = sp[:, 0], sp[:, 1]
        A = A + sparse.coo_matrix((np.ones(4 * len(a), np.int8), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))),
                                  shape=(nv, nv)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


def lame(E, nu):
    E, nu = np.asarray(E, float), np.asarray(nu, float)
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def assemble(mesh, E, nu, alpha, void_ratio=1e-4):
    """{"K" (BSR, 3 x 3 blocks), "F" (loads: six strains and dT), "c0", "s0", "vol", moduli used}."""
    E = np.array(E, float)
    nu = np.array(nu, float)
    solid = E > 0
    void = np.nonzero(~solid)[0].tolist()
    if not solid.all():
        E[~solid] = void_ratio * E[solid].min()
        nu[~solid] = 0.0
    lam, mu = lame(E, nu)
    al = np.zeros(len(E)) if alpha is None else np.array(alpha, float)
    al[~solid] = 0.0
    bt = (3 * lam + 2 * mu) * al
    nv = len(mesh["points"])
    indptr, indices = _pattern(mesh)
    data = np.zeros((len(indices), 3, 3))
    F = np.zeros((nv, 3, 7))
    c0 = np.zeros((6, 6))
    s0 = np.zeros(6)
    vol = _fill(mesh["tets"], mesh["points"], mesh["per"], mesh["L"], lam, mu, bt, mesh["mat"].astype(np.int64), indptr,
                indices, data, F, c0, s0)
    sp = mesh.get("springs")
    if sp is not None and len(sp):
        sn = mesh.get("spring_n")
        if sn is None:
            raise ValueError("this mesh carries no film normals")
        _springs(sp, mesh["spring_w"], sn, float(lam[0] + 2 * mu[0]), float(mu[0]), indptr, indices, data)
    K = sparse.bsr_matrix((data, indices, indptr), shape=(3 * nv, 3 * nv), blocksize=(3, 3))
    return {"K": K, "F": F.reshape(3 * nv, 7), "c0": c0, "s0": s0, "vol": vol, "lam": lam, "mu": mu, "bt": bt,
            "E_used": E, "nu_used": nu, "void_phases": void}


def _rigid(points, L):
    nv = len(points)
    x = points - 0.5 * np.asarray(L)
    B = np.zeros((3 * nv, 6))
    for i in range(3):
        B[i::3, i] = 1.0
    B[1::3, 3], B[2::3, 3] = -x[:, 2], x[:, 1]
    B[0::3, 4], B[2::3, 4] = x[:, 2], -x[:, 0]
    B[0::3, 5], B[1::3, 5] = -x[:, 1], x[:, 0]
    return B


class Solver:
    """K with its multigrid, for a block of right-hand sides: PyAMG's hierarchy on the rigid-body modes, cycled on
    all cores (mgsolve.py)."""

    def __init__(self, K, points, L, free=None):
        import pyamg
        from . import mgsolve as MG
        self.MG = MG
        self.free = free
        Kb = K if free is None else K.tocsr()[free][:, free].tobsr(blocksize=(3, 3))
        B = _rigid(points, L)
        if free is not None:
            B = B[free]
        ml = pyamg.smoothed_aggregation_solver(Kb, B=B, max_coarse=600, keep=False, strength="symmetric",
                                               smooth=("energy", {"krylov": "cg", "maxiter": 2, "degree": 1}))
        self.complexity = float(ml.operator_complexity())
        self.M = MG.Multigrid(ml, degree=3)
        del ml, Kb
        self.how = "AMG, all cores"

    def solve(self, Bm, tol, maxiter, callback=None, should_stop=None):
        """Bm (n, k). Returns (X, per-column statistics)."""
        bb = Bm if self.free is None else Bm[self.free]
        X, its, res, ok, sec = self.MG.pcg(None, bb, self.M, tol=tol, maxiter=maxiter, callback=callback,
                                           should_stop=should_stop)
        if self.free is not None:
            U = np.zeros(Bm.shape)
            U[self.free] = X
            X = U
        st = [{"iterations": int(its[c]), "residual": float(res[c]), "converged": bool(ok[c]), "target": float(tol),
               "seconds": sec / max(Bm.shape[1], 1), "preconditioner": self.how} for c in range(Bm.shape[1])]
        return X, st


def homogenize(mesh, E, nu, alpha, tol=1e-6, maxiter=3000, void_ratio=1e-4, progress=None, should_stop=None,
               keep_thermal_stress=False, stress_bc="th", loads=None, log=None):
    """C* (6 x 6), alpha* (6) and, on request, the thermal stress for dT = 1 K on the voxels (stress_bc "th": the RVE
    held at its size; "free": free to expand by alpha* dT). The keys of fans.homogenize."""
    t0 = time.time()
    sysm = assemble(mesh, E, nu, alpha, void_ratio)
    K, F, vol = sysm["K"], sysm["F"], sysm["vol"]
    points, per, L = mesh["points"], mesh["per"], mesh["L"]
    nv = len(points)
    free = None
    if not per.all():
        if per.any():
            raise ValueError("elasticity on the tetrahedral mesh needs a mesh periodic in all directions, or in none")
        eps = 1e-9 * L
        inner = np.all((points > eps) & (points < L - eps), axis=1)
        free = np.flatnonzero(np.repeat(inner, 3))
    sol = Solver(K, points, L, free)
    del K
    sysm.pop("K")
    t_setup = time.time() - t0
    if log:
        log(f"    tetrahedral elasticity: {3 * nv:,} unknowns, multigrid set up in {t_setup:.1f} s "
            f"(operator complexity {sol.complexity:.2f}); the loads are solved together on all cores")
    loads = list(loads or NAMES + ["th"])
    cols = [k for k, nm in enumerate(NAMES) if nm in loads] + ([6] if "th" in loads else [])
    X, sts = sol.solve(np.ascontiguousarray(F[:, cols]), tol, maxiter, progress, should_stop)
    if free is None:
        X = X - np.tile(X.reshape(nv, 3, len(cols)).mean(axis=0), (nv, 1))
    C = np.zeros((6, 6))
    U = {}
    stats = {}
    for q, k in enumerate(cols):
        stats["th" if k == 6 else NAMES[k]] = sts[q]
        if k < 6:
            U[k] = X[:, q]
    for k in U:
        for i in range(6):
            C[i, k] = (sysm["c0"][i, k] - float(F[:, i] @ U[k])) / vol
    out = {"C": 0.5 * (C + C.T), "C_raw": C, "stats": stats, "E_used": sysm["E_used"].tolist(),
           "nu_used": sysm["nu_used"].tolist(), "void_phases": sysm["void_phases"], "method": "tet-fe",
           "n_dof": int(3 * nv), "n_tets": int(len(mesh["tets"])), "setup_seconds": t_setup,
           "preconditioner": sol.how}
    if "th" in loads:
        uth = X[:, len(cols) - 1]
        sth = np.array([(sysm["s0"][i] - float(F[:, i] @ uth)) / vol for i in range(6)])
        out["sigma_th"] = sth
        out["alpha"] = -np.linalg.solve(out["C"], sth)
        if keep_thermal_stress:
            Ev = np.zeros(6)
            u = uth
            if stress_bc == "free" and len(U) == 6:
                Ev = out["alpha"].copy()
                u = uth + sum(Ev[k] * U[k] for k in range(6))
            sig = np.empty((len(mesh["tets"]), 6))
            _stress(mesh["tets"], points, per, L, sysm["lam"], sysm["mu"], sysm["bt"], mesh["mat"].astype(np.int64),
                    np.ascontiguousarray(u), Ev, 1.0, sig)
            cells = mesh["cells"]
            grid = np.zeros((int(np.prod(cells)), 6))
            hit = np.zeros(int(np.prod(cells)), np.int8)
            raster(mesh["tets"], points, per, L, float(mesh["h"]), cells, sig, grid, hit)
            out["thermal_stress"] = np.moveaxis(grid.reshape(tuple(cells) + (6,)), -1, 0)
            out["stress_tets"] = sig
            out["u_thermal"] = np.ascontiguousarray(u)
    out["asymmetry"] = float(np.abs(C - C.T).max() / max(np.abs(C).max(), 1e-300))
    out["seconds"] = time.time() - t0
    return out
