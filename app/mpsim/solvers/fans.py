"""Thermo-elastic homogenisation on voxel finite elements, solved by
FFT-preconditioned conjugate gradients (FANS).

The discretisation is PuMA's: one trilinear hexahedron per voxel, full 2x2x2
Gauss integration, periodic boundary, the macroscopic strain applied as a
mean and the periodic displacement fluctuation solved for. Only the solver
differs. PuMA solves K u = f with MINRES preconditioned by the diagonal of K,
and the number of iterations of such a solve grows with the grid edge: on a
500-cubed RVE it ran for more than 30 hours without reaching 1e-5.

Here the preconditioner is the exact inverse of the same finite-element
operator for a homogeneous reference material, applied in Fourier space
(the operator of a homogeneous material on a periodic grid is diagonal there,
one 3 x 3 block per frequency). The iteration count then depends on the
stiffness contrast of the phases, not on the grid (Leuschner and Fritzen,
"Fourier-Accelerated Nodal Solvers (FANS) for homogenization problems",
Comput. Mech. 62 (2018) 359-392). The product K u is matrix-free: each
voxel's 24 x 24 element matrix (one per material) applied to its 24
displacements; the threads take whole x-slices of voxels in two colour
classes (even and odd slices), so that no two add to the same node.

Voigt order xx, yy, zz, yz, xz, xy with engineering shear strain, as in
elastic.py and the rest of the program.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy import fft as sfft

from .elastic import lame

GAUSS = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))
# local node n = a + 2 b + 4 c sits at the voxel corner (a, b, c)
NODE_OFF = np.array([[n & 1, (n >> 1) & 1, (n >> 2) & 1] for n in range(8)], np.int64)


def iso_C(lam, mu):
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    for i in range(3):
        C[i, i] = lam + 2.0 * mu
    for i in range(3, 6):
        C[i, i] = mu
    return C


def b_matrix(x, y, z):
    """Strain-displacement matrix (6 x 24) of the unit voxel at (x, y, z)."""
    B = np.zeros((6, 24))
    for n in range(8):
        a, b, c = NODE_OFF[n]
        fx = x if a else 1.0 - x
        fy = y if b else 1.0 - y
        fz = z if c else 1.0 - z
        dx = (1.0 if a else -1.0) * fy * fz
        dy = fx * (1.0 if b else -1.0) * fz
        dz = fx * fy * (1.0 if c else -1.0)
        j = 3 * n
        B[0, j] = dx
        B[1, j + 1] = dy
        B[2, j + 2] = dz
        B[3, j + 1] = dz
        B[3, j + 2] = dy
        B[4, j] = dz
        B[4, j + 2] = dx
        B[5, j] = dy
        B[5, j + 1] = dx
    return B


def element(C, bbar=False):
    """Element stiffness (24 x 24, unit voxel, 2x2x2 Gauss) and the
    voxel-averaged strain-displacement matrix (6 x 24).

    bbar: the volumetric strain taken at the voxel centre at every Gauss
    point (Hughes' B-bar), which keeps the trilinear element free of volumetric
    locking when the material is (nearly) incompressible - the creeping flow
    of a filled resin, solved by the same equations with velocity in place of
    displacement and viscosity in place of the shear modulus."""
    K = np.zeros((24, 24))
    Bc = b_matrix(0.5, 0.5, 0.5)
    m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    vol_c = m @ Bc
    for gz in GAUSS:
        for gy in GAUSS:
            for gx in GAUSS:
                B = b_matrix(gx, gy, gz)
                if bbar:
                    B = B + np.outer(m, vol_c - m @ B) / 3.0
                K += 0.125 * B.T @ C @ B
    return K, Bc


def stencil(K):
    """The 27 3 x 3 blocks of a homogeneous material's global operator: the
    force at a node from the displacement of the node at offset d."""
    M = np.zeros((3, 3, 3, 3, 3))
    for n in range(8):
        for m in range(8):
            d = NODE_OFF[m] - NODE_OFF[n]
            M[d[0] + 1, d[1] + 1, d[2] + 1] += K[3 * n:3 * n + 3, 3 * m:3 * m + 3]
    return M


def _classes(n):
    """Element indices of one axis in colour classes whose elements share no
    node: even, odd, and for an odd count the last element on its own."""
    if n % 2 == 0:
        return [np.arange(0, n, 2), np.arange(1, n, 2)]
    return [np.arange(0, n - 1, 2), np.arange(1, n - 1, 2), np.array([n - 1])]


@njit(parallel=True, fastmath=True, cache=True)
def _apply_pass(u, out, lab, Kmat, ii, jj, kk):
    nx, ny, nz = lab.shape
    for t in prange(ii.size):
        i = ii[t]
        ue = np.empty(24)
        for j in jj:
            for k in kk:
                m = lab[i, j, k]
                for n in range(8):
                    a = i + (n & 1)
                    if a == nx:
                        a = 0
                    b = j + ((n >> 1) & 1)
                    if b == ny:
                        b = 0
                    c = k + ((n >> 2) & 1)
                    if c == nz:
                        c = 0
                    ue[3 * n] = u[0, a, b, c]
                    ue[3 * n + 1] = u[1, a, b, c]
                    ue[3 * n + 2] = u[2, a, b, c]
                for n in range(8):
                    a = i + (n & 1)
                    if a == nx:
                        a = 0
                    b = j + ((n >> 1) & 1)
                    if b == ny:
                        b = 0
                    c = k + ((n >> 2) & 1)
                    if c == nz:
                        c = 0
                    for d in range(3):
                        r = 3 * n + d
                        s = 0.0
                        for q in range(24):
                            s += Kmat[m, r, q] * ue[q]
                        out[d, a, b, c] += s


@njit(parallel=True, fastmath=True, cache=True)
def _force_pass(out, lab, fmat, ii, jj, kk):
    """Adds each voxel's 24 nodal forces fmat[material] to its nodes."""
    nx, ny, nz = lab.shape
    for t in prange(ii.size):
        i = ii[t]
        for j in jj:
            for k in kk:
                m = lab[i, j, k]
                for n in range(8):
                    a = i + (n & 1)
                    if a == nx:
                        a = 0
                    b = j + ((n >> 1) & 1)
                    if b == ny:
                        b = 0
                    c = k + ((n >> 2) & 1)
                    if c == nz:
                        c = 0
                    for d in range(3):
                        out[d, a, b, c] += fmat[m, 3 * n + d]


@njit(parallel=True, fastmath=True, cache=True)
def _strain(u, lab, Bbar, eps):
    """Voxel-averaged strain of the displacement fluctuation (6 per voxel)."""
    nx, ny, nz = lab.shape
    for i in prange(nx):
        ue = np.empty(24)
        for j in range(ny):
            for k in range(nz):
                for n in range(8):
                    a = i + (n & 1)
                    if a == nx:
                        a = 0
                    b = j + ((n >> 1) & 1)
                    if b == ny:
                        b = 0
                    c = k + ((n >> 2) & 1)
                    if c == nz:
                        c = 0
                    ue[3 * n] = u[0, a, b, c]
                    ue[3 * n + 1] = u[1, a, b, c]
                    ue[3 * n + 2] = u[2, a, b, c]
                for r in range(6):
                    s = 0.0
                    for q in range(24):
                        s += Bbar[r, q] * ue[q]
                    eps[r, i, j, k] = s


@njit(parallel=True, fastmath=True, cache=True)
def _precond(R0, R1, R2, M, e1, e2, e3):
    """In place: the residual's Fourier coefficients times the inverse of the
    reference operator's 3 x 3 symbol (zero for the mean)."""
    n1, n2, n3 = R0.shape
    for i in prange(n1):
        for j in range(n2):
            for k in range(n3):
                if i == 0 and j == 0 and k == 0:
                    R0[i, j, k] = 0.0
                    R1[i, j, k] = 0.0
                    R2[i, j, k] = 0.0
                    continue
                a00 = a01 = a02 = a10 = a11 = a12 = a20 = a21 = a22 = 0j
                for a in range(3):
                    for b in range(3):
                        ph_ab = e1[i, a] * e2[j, b]
                        for c in range(3):
                            ph = ph_ab * e3[k, c]
                            a00 += M[a, b, c, 0, 0] * ph
                            a01 += M[a, b, c, 0, 1] * ph
                            a02 += M[a, b, c, 0, 2] * ph
                            a10 += M[a, b, c, 1, 0] * ph
                            a11 += M[a, b, c, 1, 1] * ph
                            a12 += M[a, b, c, 1, 2] * ph
                            a20 += M[a, b, c, 2, 0] * ph
                            a21 += M[a, b, c, 2, 1] * ph
                            a22 += M[a, b, c, 2, 2] * ph
                # 3 x 3 inverse (Hermitian positive definite away from k = 0)
                c00 = a11 * a22 - a12 * a21
                c01 = a02 * a21 - a01 * a22
                c02 = a01 * a12 - a02 * a11
                c10 = a12 * a20 - a10 * a22
                c11 = a00 * a22 - a02 * a20
                c12 = a02 * a10 - a00 * a12
                c20 = a10 * a21 - a11 * a20
                c21 = a01 * a20 - a00 * a21
                c22 = a00 * a11 - a01 * a10
                det = a00 * c00 + a01 * c10 + a02 * c20
                if abs(det) < 1e-300:
                    R0[i, j, k] = 0.0
                    R1[i, j, k] = 0.0
                    R2[i, j, k] = 0.0
                    continue
                x0, x1, x2 = R0[i, j, k], R1[i, j, k], R2[i, j, k]
                R0[i, j, k] = (c00 * x0 + c01 * x1 + c02 * x2) / det
                R1[i, j, k] = (c10 * x0 + c11 * x1 + c12 * x2) / det
                R2[i, j, k] = (c20 * x0 + c21 * x1 + c22 * x2) / det


@njit(parallel=True, fastmath=True, cache=True)
def _dot(a, b):
    s = 0.0
    n = a.size
    fa = a.ravel()
    fb = b.ravel()
    for t in prange(n):
        s += fa[t] * fb[t]
    return s


@njit(parallel=True, fastmath=True, cache=True)
def _axpy(y, a, x):
    fy = y.ravel()
    fx = x.ravel()
    for t in prange(fy.size):
        fy[t] += a * fx[t]


@njit(parallel=True, fastmath=True, cache=True)
def _xpay(x, b, y):
    """y = x + b y"""
    fy = y.ravel()
    fx = x.ravel()
    for t in prange(fy.size):
        fy[t] = fx[t] + b * fy[t]


class FANS:
    """Periodic voxel-FE thermo-elasticity of a label map.

    E, nu, alpha: per label (alpha in 1/K). Phases with E <= 0 (pores) get
    void_ratio times the smallest positive modulus."""

    def __init__(self, labels, E, nu, alpha=None, void_ratio=1e-4, workers=-1, lam_mu=None, bbar=False):
        """lam_mu: (lambda, mu) per label in place of E and nu (the viscous
        problem: mu the shear viscosity, lambda a large bulk penalty); bbar:
        B-bar elements (see element)."""
        if lam_mu is not None:
            lam, mu = (np.asarray(a, float).copy() for a in lam_mu)
            E = mu * (3 * lam + 2 * mu) / (lam + mu)
            nu = lam / (2 * (lam + mu))
            self.void_phases = []
        else:
            E = np.asarray(E, float).copy()
            nu = np.asarray(nu, float).copy()
            solid = E > 0
            if not solid.any():
                raise ValueError("No phase has a non-zero stiffness")
            self.void_phases = np.nonzero(~solid)[0].tolist()
            E[~solid] = void_ratio * E[solid].min()
            nu[~solid] = 0.2
            lam, mu = lame(E, nu)
        self.E_used, self.nu_used = E, nu
        self.alpha = np.zeros(len(E)) if alpha is None else np.asarray(alpha, float)
        self.lab = np.ascontiguousarray(labels, dtype=np.int32)
        self.shape = self.lab.shape
        self.workers = workers
        self.bbar = bool(bbar)
        self.C = np.array([iso_C(l, m) for l, m in zip(lam, mu)])
        Ks, Bbar = [], None
        for Cm in self.C:
            K, Bbar = element(Cm, self.bbar)
            Ks.append(K)
        self.Kmat = np.ascontiguousarray(Ks)
        self.Bbar = Bbar
        # reference medium: geometric means of the shear and bulk moduli of
        # the phases present (the contrast to either end is then balanced)
        present = np.unique(self.lab)
        K_b = lam + 2.0 * mu / 3.0
        mu0 = float(np.exp(np.mean(np.log(mu[present]))))
        kb0 = float(np.exp(np.mean(np.log(K_b[present]))))
        K0, _ = element(iso_C(kb0 - 2.0 * mu0 / 3.0, mu0), self.bbar)
        self.M = np.ascontiguousarray(stencil(K0)).astype(np.float64)
        self.e = []
        for n in self.shape:
            th = 2.0 * np.pi * np.fft.fftfreq(n)
            self.e.append(np.ascontiguousarray(np.exp(1j * th[:, None] * np.array([-1.0, 0.0, 1.0])[None, :])))
        n3 = self.shape[2] // 2 + 1
        th3 = 2.0 * np.pi * np.arange(n3) / self.shape[2]
        self.e[2] = np.ascontiguousarray(np.exp(1j * th3[:, None] * np.array([-1.0, 0.0, 1.0])[None, :]))
        # threads own x-slices; slices of one class share no node
        jj, kk = np.arange(self.shape[1]), np.arange(self.shape[2])
        self.passes = [(a, jj, kk) for a in _classes(self.shape[0])]

    # ------------------------------------------------------------------
    def K(self, u, out=None):
        if out is None:
            out = np.zeros_like(u)
        else:
            out.fill(0.0)
        for ii, jj, kk in self.passes:
            _apply_pass(u, out, self.lab, self.Kmat, ii, jj, kk)
        return out

    def P(self, r):
        w = self.workers
        R = [sfft.rfftn(r[c], workers=w) for c in range(3)]
        _precond(R[0], R[1], R[2], self.M, self.e[0], self.e[1], self.e[2])
        z = np.empty_like(r)
        for c in range(3):
            z[c] = sfft.irfftn(R[c], s=self.shape, workers=w)
        return z

    def rhs(self, E_voigt, dT=0.0):
        """Nodal forces of a macroscopic strain E and a thermal eigenstrain
        alpha dT: f = - sum_e Bbar^T C_e (E - alpha_e dT [1 1 1 0 0 0])."""
        E_voigt = np.asarray(E_voigt, float)
        fmat = np.empty((len(self.C), 24))
        for m, Cm in enumerate(self.C):
            e = E_voigt.copy()
            e[:3] -= self.alpha[m] * dT
            fmat[m] = -(self.Bbar.T @ (Cm @ e))
        b = np.zeros((3,) + self.shape)
        for ii, jj, kk in self.passes:
            _force_pass(b, self.lab, fmat, ii, jj, kk)
        return b

    def solve(self, E_voigt, dT=0.0, tol=1e-5, maxiter=5000, callback=None, should_stop=None,
              keep_stress=False, keep_strain=False, keep_disp=False):
        """Mean stress (Voigt) for a macroscopic strain and a temperature
        change; the per-voxel stress (6, nx, ny, nz) when keep_stress, the
        per-voxel total strain in self.last_strain when keep_strain, and the
        nodal displacement fluctuation (3, nx, ny, nz; node i,j,k at the lower
        corner of voxel i,j,k, unit voxel) in self.last_disp when keep_disp."""
        t0 = time.time()
        b = self.rhs(E_voigt, dT)
        bn = math.sqrt(_dot(b, b))
        x = np.zeros_like(b)
        it, res = 0, 0.0
        if bn > 0:
            r = b
            z = self.P(r)
            p = z.copy()
            rz = _dot(r, z)
            q = np.empty_like(b)
            for it in range(1, maxiter + 1):
                self.K(p, q)
                pq = _dot(p, q)
                if pq <= 0:
                    break
                a = rz / pq
                _axpy(x, a, p)
                _axpy(r, -a, q)
                res = math.sqrt(_dot(r, r)) / bn
                if callback is not None and (it % 2 == 0 or res < tol):
                    callback(it, res)
                if res < tol:
                    break
                if should_stop is not None and it % 5 == 0 and should_stop():
                    raise InterruptedError
                z = self.P(r)
                rz_new = _dot(r, z)
                _xpay(z, rz_new / rz, p)
                rz = rz_new
            del q, p, z
        eps = np.empty((6,) + self.shape)
        _strain(x, self.lab, self.Bbar, eps)
        self.last_disp = x if keep_disp else None
        del x
        E_voigt = np.asarray(E_voigt, float)
        self.last_strain = (eps + E_voigt[:, None, None, None]) if keep_strain else None
        sbar = np.zeros(6)
        stress = np.empty((6,) + self.shape) if keep_stress else None
        for m in np.unique(self.lab):
            mask = self.lab == m
            e = eps[:, mask] + E_voigt[:, None]
            e[:3] -= self.alpha[m] * dT
            s = self.C[m] @ e
            sbar += s.sum(axis=1)
            if keep_stress:
                stress[:, mask] = s
        sbar /= self.lab.size
        info = {"iterations": it, "residual": float(res), "converged": bool(bn == 0 or res < tol),
                "seconds": time.time() - t0}
        return sbar, info, stress


def homogenize(labels, E, nu, alpha, tol=1e-5, maxiter=5000, void_ratio=1e-4, workers=-1,
               progress=None, should_stop=None, keep_thermal_stress=False, loads=None, stress_bc="th",
               C_known=None, bbar=False):
    """C* (6 x 6), alpha* (6) and, on request, the thermal stress field for
    dT = 1 K (stress_bc "th": the RVE held at its size, "free": free to
    expand by alpha* dT, one more solve). progress(load, it, res) is called
    on the way. loads: subset of ("xx","yy","zz","yz","xz","xy","th").
    C_known: the stiffness of the same RVE and moduli from an earlier solve,
    so an eigenstrain (another alpha) needs the "th" load only."""
    sol = FANS(labels, E, nu, alpha, void_ratio, workers, bbar=bbar)
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    loads = list(loads or names + ["th"])
    if C_known is not None:
        loads = [x for x in loads if x not in names]
    C = np.array(C_known, float) if C_known is not None else np.zeros((6, 6))
    stats = {}
    for k, nm in enumerate(names):
        if nm not in loads:
            continue
        Ev = np.zeros(6)
        Ev[k] = 1.0
        cb = (lambda it, res, nm=nm: progress(nm, it, res)) if progress else None
        sbar, st, _ = sol.solve(Ev, 0.0, tol, maxiter, cb, should_stop)
        C[:, k] = sbar
        stats[nm] = st
    out = {"C": 0.5 * (C + C.T), "C_raw": C, "stats": stats, "E_used": sol.E_used.tolist(),
           "nu_used": sol.nu_used.tolist(), "void_phases": sol.void_phases}
    if "th" in loads:
        cb = (lambda it, res: progress("th", it, res)) if progress else None
        keep = keep_thermal_stress and stress_bc != "free"
        sth, st, stress = sol.solve(np.zeros(6), 1.0, tol, maxiter, cb, should_stop, keep)
        stats["th"] = st
        out["sigma_th"] = sth
        out["alpha"] = -np.linalg.solve(out["C"], sth)
        if keep_thermal_stress and stress_bc == "free":
            # the RVE expands freely: macroscopic strain alpha* dT, zero mean stress
            cb = (lambda it, res: progress("free", it, res)) if progress else None
            _, st, stress = sol.solve(out["alpha"], 1.0, tol, maxiter, cb, should_stop, True)
            stats["free"] = st
        out["thermal_stress"] = stress
    out["asymmetry"] = float(np.abs(C - C.T).max() / max(np.abs(C).max(), 1e-300))
    return out
