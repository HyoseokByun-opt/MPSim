"""Conjugate gradients with an algebraic-multigrid V-cycle on all cores, for several right-hand sides at once (5.10.0).

PyAMG builds the smoothed-aggregation hierarchy (operators, prolongations); its own cycle and SciPy's products run
on one core. Here the cycle is run with kernels compiled for all cores (Numba), on a block of right-hand sides in
one pass over the matrices: Chebyshev smoothing on the diagonally scaled operator (three products, no ordering
dependence, so it is the same on any number of cores), a dense pseudo-inverse on the coarsest level, and
conjugate gradients with one step length per right-hand side.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange
from scipy import sparse


@njit(cache=True, parallel=True)
def _mm(indptr, indices, data, X, Y):
    """Y = A X for a block X (n, k)."""
    k = X.shape[1]
    for i in prange(indptr.shape[0] - 1):
        for c in range(k):
            Y[i, c] = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            a = data[p]
            j = indices[p]
            for c in range(k):
                Y[i, c] += a * X[j, c]


@njit(cache=True, parallel=True)
def _resid(indptr, indices, data, X, Bk, Rk):
    """R = B - A X."""
    k = X.shape[1]
    for i in prange(indptr.shape[0] - 1):
        for c in range(k):
            Rk[i, c] = Bk[i, c]
        for p in range(indptr[i], indptr[i + 1]):
            a = data[p]
            j = indices[p]
            for c in range(k):
                Rk[i, c] -= a * X[j, c]


@njit(cache=True, parallel=True)
def _cheb_step(X, D, Rk, dinv, a, b):
    """D = a D + b dinv R;  X += D."""
    k = X.shape[1]
    for i in prange(X.shape[0]):
        for c in range(k):
            D[i, c] = a * D[i, c] + b * dinv[i] * Rk[i, c]
            X[i, c] += D[i, c]


@njit(cache=True, parallel=True)
def _dots(A, Bm, out):
    k = A.shape[1]
    n = A.shape[0]
    nthr = 64
    acc = np.zeros((nthr, k))
    chunk = (n + nthr - 1) // nthr
    for t in prange(nthr):
        for i in range(t * chunk, min(n, (t + 1) * chunk)):
            for c in range(k):
                acc[t, c] += A[i, c] * Bm[i, c]
    for c in range(k):
        s = 0.0
        for t in range(nthr):
            s += acc[t, c]
        out[c] = s


@njit(cache=True, parallel=True)
def _axpy(Y, a, X):
    """Y += X * a (a per column)."""
    k = X.shape[1]
    for i in prange(X.shape[0]):
        for c in range(k):
            Y[i, c] += a[c] * X[i, c]


@njit(cache=True, parallel=True)
def _xpay(X, b, Y):
    """Y = X + Y * b (b per column)."""
    k = X.shape[1]
    for i in prange(X.shape[0]):
        for c in range(k):
            Y[i, c] = X[i, c] + b[c] * Y[i, c]


def _csr(A):
    A = sparse.csr_matrix(A)
    A.sort_indices()
    it = np.int32 if A.nnz < 2 ** 31 - 1 else np.int64             # the column indices: half the memory where they fit
    return A.indptr.astype(np.int64), A.indices.astype(it), np.ascontiguousarray(A.data, float)


class Multigrid:
    """A PyAMG hierarchy as a preconditioner for blocks of vectors. degree: Chebyshev steps before and after."""

    def __init__(self, ml, degree=3):
        from pyamg.util.linalg import approximate_spectral_radius
        self.degree = int(degree)
        self.lv = []
        for q, lev in enumerate(ml.levels):
            A = sparse.csr_matrix(lev.A)
            d = {"A": _csr(A), "n": A.shape[0]}
            if q < len(ml.levels) - 1:
                dg = A.diagonal().astype(float)
                dg[dg == 0] = 1.0
                d["dinv"] = 1.0 / dg
                Dm = sparse.diags(d["dinv"])
                d["rho"] = float(approximate_spectral_radius(Dm @ A, tol=1e-2, maxiter=12, restart=1))
                d["P"] = _csr(lev.P)
                d["R"] = _csr(lev.R)
            else:
                Ad = A.toarray()
                d["pinv"] = np.linalg.pinv(Ad, rcond=1e-10, hermitian=True)
            self.lv.append(d)
        self.work = {}

    def _w(self, q, k, name):
        key = (q, k, name)
        if key not in self.work:
            self.work[key] = np.zeros((self.lv[q]["n"], k))
        return self.work[key]

    def _smooth(self, q, X, Bk, zero):
        """Chebyshev on dinv A, eigenvalues in [rho / 30, 1.1 rho]."""
        lv = self.lv[q]
        k = X.shape[1]
        Rk, D = self._w(q, k, "r"), self._w(q, k, "d")
        hi, lo = 1.1 * lv["rho"], lv["rho"] / 30.0
        theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
        sigma = theta / delta
        rho_old = 1.0 / sigma
        if zero:
            Rk[:] = Bk
        else:
            _resid(*lv["A"], X, Bk, Rk)
        D[:] = 0.0
        _cheb_step(X, D, Rk, lv["dinv"], 0.0, 1.0 / theta)
        for _ in range(self.degree - 1):
            _resid(*lv["A"], X, Bk, Rk)
            rho_new = 1.0 / (2.0 * sigma - rho_old)
            _cheb_step(X, D, Rk, lv["dinv"], rho_new * rho_old, 2.0 * rho_new / delta)
            rho_old = rho_new

    def _cycle(self, q, X, Bk):
        lv = self.lv[q]
        if q == len(self.lv) - 1:
            X[:] = lv["pinv"] @ Bk
            return
        k = X.shape[1]
        X[:] = 0.0
        self._smooth(q, X, Bk, True)
        Rk = self._w(q, k, "r2")
        _resid(*lv["A"], X, Bk, Rk)
        Bc, Xc = self._w(q + 1, k, "b"), self._w(q + 1, k, "x")
        _mm(*lv["R"], Rk, Bc)
        self._cycle(q + 1, Xc, Bc)
        _mm(*lv["P"], Xc, Rk)
        X += Rk
        self._smooth(q, X, Bk, False)

    def apply(self, Rk, Z):
        self._cycle(0, Z, Rk)


def pcg(A, Bm, M, tol=1e-8, maxiter=3000, callback=None, should_stop=None, X0=None):
    """A X = B for a block B (n, k): one conjugate-gradient recurrence per column, the products shared.
    A None: the operator of the hierarchy's finest level (no second copy).
    Returns (X, iterations per column, residuals, converged)."""
    t0 = time.time()
    a = M.lv[0]["A"] if A is None else _csr(A)
    Bm = np.ascontiguousarray(Bm, float)
    n, k = Bm.shape
    X = np.zeros((n, k)) if X0 is None else np.ascontiguousarray(X0, float).copy()
    Rk = np.empty((n, k))
    _resid(*a, X, Bm, Rk)
    Z = np.zeros((n, k))
    Pk = np.zeros((n, k))
    Q = np.zeros((n, k))
    bn = np.empty(k)
    _dots(Bm, Bm, bn)
    bn = np.sqrt(bn)
    bn[bn == 0] = 1.0
    rz = np.empty(k)
    rz_new = np.empty(k)
    pq = np.empty(k)
    rr = np.empty(k)
    its = np.zeros(k, np.int64)
    _dots(Rk, Rk, rr)
    res = np.sqrt(rr) / bn
    active = res > tol
    it = 0
    if active.any():
        M.apply(Rk, Z)
        Pk[:] = Z
        _dots(Rk, Z, rz)
    while active.any() and it < maxiter:
        if should_stop is not None and should_stop():
            raise InterruptedError
        _mm(*a, Pk, Q)
        _dots(Pk, Q, pq)
        alpha = np.where(active & (pq > 0), rz / np.where(pq > 0, pq, 1.0), 0.0)
        _axpy(X, alpha, Pk)
        _axpy(Rk, -alpha, Q)
        it += 1
        its[active] = it
        _dots(Rk, Rk, rr)
        res = np.where(active, np.sqrt(rr) / bn, res)
        active = active & (res > tol)
        if callback is not None and it % 5 == 0:
            callback(it, float(res.max()))
        if not active.any():
            break
        M.apply(Rk, Z)
        _dots(Rk, Z, rz_new)
        beta = np.where(active & (rz != 0), rz_new / np.where(rz != 0, rz, 1.0), 0.0)
        rz[:] = rz_new
        _xpay(Z, beta, Pk)
    _resid(*a, X, Bm, Rk)
    _dots(Rk, Rk, rr)
    res = np.sqrt(rr) / bn
    return X, its, res, res <= tol * 1.5, time.time() - t0
