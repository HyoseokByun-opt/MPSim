"""Frequency-resolved quasi-static homogenisation with complex coefficients.

At angular frequency w every region carries the complex admittivity

    kappa = sigma + j w eps0 eps_r (1 - j tan d)
          = (sigma + w eps0 eps_r tan d) + j w eps0 eps_r ,

and the RVE is solved for div(kappa grad phi) = 0 exactly as for a steady
conductivity - same cell-centred finite volume, same harmonic face values -
only in complex arithmetic. The effective kappa_eff(w) then holds both the
conduction and the displacement current of the composite at that frequency,
with their interaction: the interfacial (Maxwell-Wagner) polarisation of a
conducting filler below percolation, a conducting network above it, and
metal particles acting as the perfect conductors they are at radio
frequencies. Combining a DC conductivity solve with a separate dielectric
solve misses all three.

Valid while the RVE is much smaller than the wavelength in the material and
the skin depth inside the filler exceeds the particle size (eddy currents
inside a particle are not modelled); emi.validity reports those limits.

The system is complex symmetric (not Hermitian), so it is solved with the
conjugate orthogonal conjugate gradient method (COCG: CG with unconjugated
inner products), preconditioned by the exact inverse of the periodic
Laplacian scaled by a complex reference admittivity.

Boundary conditions (`bc`):
    "periodic"  all three directions periodic (a bulk RVE)
    "film"      x, y periodic, the z faces insulated (in-plane solve of a film)
    "plates_x"  fixed potentials on the x faces, the y and z faces insulated:
                the unit cell between the PEC (x) and PMC (y) walls of a
                full-wave simulation at normal incidence, E along x.
    "plates"    the same with the plates on the faces normal to `direction`
                (y for the magnetic field of that simulation).
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy import fft as sfft

EPS0 = 8.8541878128e-12


@njit(parallel=True, cache=True)
def _faces_c(labels, vals, axis, out):
    n0, n1, n2 = labels.shape
    for i0 in prange(n0):
        i = np.int64(i0)
        for j in range(n1):
            for k in range(n2):
                ni, nj, nk = i, j, k
                if axis == 0:
                    ni = i + 1 if i + 1 < n0 else 0
                elif axis == 1:
                    nj = j + 1 if j + 1 < n1 else 0
                else:
                    nk = k + 1 if k + 1 < n2 else 0
                a = labels[i, j, k]
                b = labels[ni, nj, nk]
                if a == b:
                    out[i, j, k] = vals[a]
                else:
                    out[i, j, k] = 1.0 / (0.5 / vals[a] + 0.5 / vals[b])


@njit(parallel=True, cache=True)
def _apply_c(u, kx, ky, kz, out):
    n0, n1, n2 = u.shape
    for i0 in prange(n0):
        i = np.int64(i0)
        ip = i + 1 if i + 1 < n0 else 0
        im = i - 1 if i > 0 else n0 - 1
        for j in range(n1):
            jp = j + 1 if j + 1 < n1 else 0
            jm = j - 1 if j > 0 else n1 - 1
            for k in range(n2):
                kp = k + 1 if k + 1 < n2 else 0
                km = k - 1 if k > 0 else n2 - 1
                c = u[i, j, k]
                out[i, j, k] = -(kx[i, j, k] * (u[ip, j, k] - c)
                                 - kx[im, j, k] * (c - u[im, j, k])
                                 + ky[i, j, k] * (u[i, jp, k] - c)
                                 - ky[i, jm, k] * (c - u[i, jm, k])
                                 + kz[i, j, k] * (u[i, j, kp] - c)
                                 - kz[i, j, km] * (c - u[i, j, km]))


class _Precond:
    """(k0 L)^-1 with L the 7-point Laplacian on the grid's own boundaries:
    periodic -> FFT, insulated -> cosine transform, fixed potential half a cell
    beyond the last cell -> sine transform, one choice per axis."""

    def __init__(self, shape, k0, kinds, workers):
        self.shape, self.k0, self.kinds, self.w = shape, complex(k0), kinds, workers
        lam = 0.0
        for ax, (n, kind) in enumerate(zip(shape, kinds)):
            if kind == "periodic":
                l = 4.0 * np.sin(np.pi * np.arange(n) / n) ** 2
            elif kind == "neumann":
                l = 4.0 * np.sin(np.pi * np.arange(n) / (2.0 * n)) ** 2
            else:
                l = 4.0 * np.sin(np.pi * (np.arange(n) + 1) / (2.0 * n)) ** 2
            sh = [1, 1, 1]
            sh[ax] = n
            lam = lam + l.reshape(sh)
        lam = np.broadcast_to(lam, shape).copy()
        self.singular = all(k != "dirichlet" for k in kinds)
        if self.singular:
            lam[0, 0, 0] = 1.0
        inv = 1.0 / lam
        if self.singular:
            inv[0, 0, 0] = 0.0
        self.inv = inv

    def _fwd(self, a):
        for ax, kind in enumerate(self.kinds):
            if kind == "periodic":
                a = sfft.fft(a, axis=ax, workers=self.w)
            elif kind == "neumann":
                a = sfft.dct(a, type=2, axis=ax, norm="ortho", workers=self.w)
            else:
                a = sfft.dst(a, type=2, axis=ax, norm="ortho", workers=self.w)
        return a

    def _inv(self, a):
        for ax, kind in reversed(list(enumerate(self.kinds))):
            if kind == "periodic":
                a = sfft.ifft(a, axis=ax, workers=self.w)
            elif kind == "neumann":
                a = sfft.idct(a, type=2, axis=ax, norm="ortho", workers=self.w)
            else:
                a = sfft.idst(a, type=2, axis=ax, norm="ortho", workers=self.w)
        return a

    def __call__(self, r):
        # the real transforms act on the real and imaginary parts alike
        return self._inv(self._fwd(r) * self.inv) / self.k0


def _threads():
    from .conduction import _threads as t
    return t()


def solve(labels, kappa, direction, bc="periodic", tol=1e-7, maxiter=20000, should_stop=None, workers=None):
    """Effective complex admittivity along `direction` for a unit macroscopic
    field. kappa: complex value per label. Returns {"kappa": complex,
    "column": complex[3], "iterations", "residual", "converged", "seconds"}.

    For bc="plates_x" the direction must be 0: the two x faces are held at
    potentials 1 and 0 (through the half cell next to them) and kappa_eff is
    the current through the plates per unit area and field."""
    if bc == "plates" and direction != 0:
        # plates on the faces normal to `direction`: the same problem with that
        # axis moved first
        r = solve(np.moveaxis(labels, direction, 0), kappa, 0, bc="plates_x", tol=tol, maxiter=maxiter,
                  should_stop=should_stop, workers=workers)
        col = np.zeros(3, np.complex128)
        col[direction] = r["kappa"]
        return dict(r, column=col)
    if bc == "plates":
        bc = "plates_x"
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    kap = np.asarray(kappa, dtype=np.complex128)
    if np.any(np.abs(kap) == 0):
        raise ValueError("an admittivity of zero")
    scale = complex(np.exp(np.mean(np.log(kap))))          # O(1) numbers
    v = kap / scale
    shape = labels.shape
    kf = [np.empty(shape, np.complex128) for _ in range(3)]
    for ax in range(3):
        _faces_c(labels, v, ax, kf[ax])
    if bc == "film":
        kinds = ["periodic", "periodic", "neumann"]
        kf[2][:, :, -1] = 0.0
    elif bc == "plates_x":
        if direction != 0:
            raise ValueError("plates_x solves along x")
        kinds = ["dirichlet", "neumann", "neumann"]
        kf[1][:, -1, :] = 0.0
        kf[2][:, :, -1] = 0.0
    else:
        kinds = ["periodic", "periodic", "periodic"]
    t0 = time.time()
    n0 = shape[0]
    if bc == "plates_x":
        # potential phi itself; the x faces tie to the plates through half a cell
        kf[0][-1, :, :] = 0.0                      # no periodic wrap in x
        gb = 2.0 * v[labels[0, :, :]]              # plate at x = 0, potential 1
        gt = 2.0 * v[labels[-1, :, :]]             # plate at x = L, potential 0
        b = np.zeros(shape, np.complex128)
        b[0, :, :] = gb * 1.0

        def A(x, out):
            _apply_c(x, kf[0], kf[1], kf[2], out)
            out[0, :, :] += gb * x[0, :, :]
            out[-1, :, :] += gt * x[-1, :, :]
    else:
        b = np.empty(shape, np.complex128)
        kd = kf[direction]
        b[...] = kd - np.roll(kd, 1, axis=direction)
        b -= b.mean()

        def A(x, out):
            _apply_c(x, kf[0], kf[1], kf[2], out)
            out -= out.mean()
    # reference admittivity: volume-weighted geometric mean of the regions
    frac = np.bincount(labels.ravel(), minlength=len(v)) / labels.size
    k0 = complex(np.exp(np.sum(frac * np.log(v))))
    M = _Precond(shape, k0, kinds, _threads() if workers is None else workers)

    u = np.zeros(shape, np.complex128)
    r = b.copy()
    z = M(r)
    if M.singular:
        z -= z.mean()
    p = z.copy()
    Ap = np.empty(shape, np.complex128)
    rho = complex(np.sum(r * z))
    bnorm = float(np.linalg.norm(b)) or 1.0
    it, res, converged = 0, 1.0, False
    for it in range(1, maxiter + 1):
        A(p, Ap)
        pAp = complex(np.sum(p * Ap))
        if pAp == 0:
            break
        alpha = rho / pAp
        u += alpha * p
        r -= alpha * Ap
        res = float(np.linalg.norm(r)) / bnorm
        if res < tol:
            converged = True
            break
        if should_stop is not None and it % 25 == 0 and should_stop():
            raise InterruptedError
        z = M(r)
        if M.singular:
            z -= z.mean()
        rho_new = complex(np.sum(r * z))
        p = z + (rho_new / rho) * p
        rho = rho_new
    if bc == "plates_x":
        # current through the x = 0 plate per cell, over the face, for a unit
        # potential drop across n0 cells: kappa_eff = I / (A * 1 / L)
        cur = np.sum(gb * (1.0 - u[0, :, :]))
        kx = complex(cur / (shape[1] * shape[2]) * n0)
        col = np.array([kx, 0.0, 0.0], np.complex128) * scale
    else:
        col = np.zeros(3, np.complex128)
        for ax in range(3):
            g = np.roll(u, -1, axis=ax) - u
            if ax == direction:
                g = g + 1.0
            col[ax] = np.mean(kf[ax] * g)
        col = col * scale
    return {"kappa": complex(col[direction]), "column": col, "iterations": it, "residual": res,
            "converged": converged, "seconds": time.time() - t0}


def admittivity(table, f_hz):
    """Complex admittivity per label at frequency f."""
    w = 2.0 * math.pi * f_hz
    out = []
    for t in table:
        p = t["props"]
        eps = float(p.get("eps_r", 1.0))
        out.append(float(p.get("sigma", 0.0)) + w * EPS0 * eps * float(p.get("tan_d", 0.0)) + 1j * w * EPS0 * eps)
    return np.array(out, np.complex128)


def cap_contrast(labels, kap, direction, cap=1e6, cache=None):
    """Limit |kappa| contrast to `cap` so the Krylov solve converges, on the
    side that keeps the answer exact in the limit (phases unchanged), as
    conduction.contrast_policy does for real values:

    * the best conductors do not percolate along `direction`: they are lowered
      to cap x the least admitting region - the perfect-conductor limit that
      the true answer has already reached (a metal filler at 1 MHz is 10^11
      times more admitting than the resin around it);
    * they percolate: the network carries the current and the rest is raised
      to max / cap; the displacement part of kappa_eff is then not resolved
      (it is below 1/cap of the conduction part), and the result says so.
    Returns (kappa, policy)."""
    kap = np.asarray(kap, np.complex128).copy()
    a = np.abs(kap)
    top, bot = float(a.max()), float(a.min())
    if top / bot <= cap:
        return kap, "none"
    cache = {} if cache is None else cache
    if "wraps" not in cache:
        from .conduction import wrapping_axes
        high = np.nonzero(a >= top / 1e3)[0]
        cache["wraps"] = wrapping_axes(np.isin(labels, high))
    if cache["wraps"][direction]:
        low = a < top / cap
        kap[low] = kap[low] / a[low] * (top / cap)
        return kap, "network"
    hi = a > bot * cap
    kap[hi] = kap[hi] / a[hi] * (bot * cap)
    return kap, "dispersed"


def spectrum(labels, table, freqs, directions=(0, 1), bc="periodic", tol=1e-7, should_stop=None, log=None,
             pre=None):
    """kappa_eff(f) averaged over the given directions, at every frequency.
    Returns (kappa array, list of per-frequency records). pre: {(frequency
    index, direction): result} solved beforehand in the job's pool."""
    out, recs = [], []
    cache = {}
    pre = pre or {}
    for i, f in enumerate(freqs):
        vals, pols, its = [], [], []
        for d in directions:
            kap, pol = cap_contrast(labels, admittivity(table, f), d, cache=cache)
            r = pre.get((i, d))
            if r is None:
                r = solve(labels, kap, d, bc=bc, tol=tol, should_stop=should_stop)
            vals.append(r["kappa"])
            pols.append(pol)
            its.append(r["iterations"])
            if log:
                log(f"    f = {f:.3g} Hz, {'xyz'[d]}: kappa = {r['kappa'].real:.4g} + j{r['kappa'].imag:.4g} S/m "
                    f"({r['iterations']} iterations{'' if r['converged'] else ', NOT converged'}"
                    f"{'' if pol == 'none' else ', contrast limited: ' + pol})")
        k = complex(np.mean(vals))
        out.append(k)
        recs.append({"f_hz": float(f), "kappa_re": k.real, "kappa_im": k.imag, "policy": pols,
                     "iterations": its})
    return np.array(out, np.complex128), recs
