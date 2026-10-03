"""Steady conduction homogenisation on a periodic voxel grid (built-in solver).

One operator serves four properties, because they are the same equation:

    thermal      div(k   grad T)   = 0
    electrical   div(sigma grad V) = 0
    dielectric   div(eps grad phi) = 0      (quasi-static)
    magnetic     div(mu  grad psi) = 0      (quasi-static)

Discretisation (from the TIM study, where it was verified against series and
parallel laminates): cell-centred finite volume, harmonic-mean face values -
exact for a sharp interface lying on a face - and the split T = G.x + u with
u periodic, so the RVE is periodic in all three directions. The system is
solved by conjugate gradients preconditioned with the exact inverse of the
constant-coefficient periodic Laplacian (FFT).

Additions over the TIM solver:
* numba stencil kernels (parallel) instead of np.roll temporaries;
* an interfacial resistance on faces between two phases,
      1/k_face = 1/(2k_a) + 1/(2k_b) + R/h ;
* the energy of the converged field split by phase, which gives the loss
  tangent of a composite dielectric (first order in tan delta) and a built-in
  Hill-Mandel check: the phase energies must add up to k_eff exactly;
* a periodic percolation test for the contrast policy;
* a film (v4): periodic in x and y, the real thickness in z. In-plane the top
  and bottom faces are insulated (no flux through them); through the
  thickness the two faces are held at fixed potentials, as between the plates
  of a measurement - `solve_film_z`.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange
from scipy import fft as sfft
from scipy import ndimage


# =========================================================================
# kernels
# =========================================================================
@njit(parallel=True, cache=True, fastmath=True)
def _apply(u, kx, ky, kz, out):
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


@njit(parallel=True, cache=True)
def _faces(labels, vals, rpair, cmask, rcpair, use_c, axis, out):
    """Face value between cell (i,j,k) and its +axis neighbour.

    Two kinds of interface resistance: rpair[a, b] on every face between
    regions a and b (a filler and the matrix), and rcpair[a, b] only on faces
    flagged in cmask - where two different particles touch, which may be two
    particles of the same region."""
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
                    res = 1.0 / vals[a]
                else:
                    res = 0.5 / vals[a] + 0.5 / vals[b] + rpair[a, b]
                if use_c and (cmask[i, j, k] >> axis) & 1:
                    res += rcpair[a, b]
                out[i, j, k] = 1.0 / res


@njit(parallel=True, cache=True)
def _rhs(kd, axis, out):
    n0, n1, n2 = kd.shape
    for i0 in prange(n0):
        i = np.int64(i0)
        for j in range(n1):
            for k in range(n2):
                if axis == 0:
                    prev = kd[i - 1 if i > 0 else n0 - 1, j, k]
                elif axis == 1:
                    prev = kd[i, j - 1 if j > 0 else n1 - 1, k]
                else:
                    prev = kd[i, j, k - 1 if k > 0 else n2 - 1]
                out[i, j, k] = kd[i, j, k] - prev


@njit(parallel=True, cache=True)
def _mean_flux(u, kf, axis, add):
    n0, n1, n2 = u.shape
    total = 0.0
    for i0 in prange(n0):
        i = np.int64(i0)
        ip = i + 1 if i + 1 < n0 else 0
        row = 0.0
        for j in range(n1):
            jp = j + 1 if j + 1 < n1 else 0
            for k in range(n2):
                kp = k + 1 if k + 1 < n2 else 0
                if axis == 0:
                    g = u[ip, j, k] - u[i, j, k]
                elif axis == 1:
                    g = u[i, jp, k] - u[i, j, k]
                else:
                    g = u[i, j, kp] - u[i, j, k]
                row += kf[i, j, k] * (g + add)
        total += row
    return total / (n0 * n1 * n2)


@njit(cache=True)
def _energy(u, kx, ky, kz, labels, vals, direction, nlab):
    """Energy of the converged field, split by phase and interface.

    On a face of resistance r = 1/(2k_a) + 1/(2k_b) + R the flux q dissipates
    q^2/(2k_a) in cell a, q^2/(2k_b) in cell b and q^2 R in the interface.
    Summed over all faces this is exactly sum q g = N k_eff (unit gradient).
    """
    n0, n1, n2 = u.shape
    E = np.zeros(nlab)
    e_int = 0.0
    for i in range(n0):
        ip = i + 1 if i + 1 < n0 else 0
        for j in range(n1):
            jp = j + 1 if j + 1 < n1 else 0
            for k in range(n2):
                kp = k + 1 if k + 1 < n2 else 0
                a = labels[i, j, k]
                c = u[i, j, k]
                for ax in range(3):
                    if ax == 0:
                        b = labels[ip, j, k]
                        g = u[ip, j, k] - c
                        kf = kx[i, j, k]
                    elif ax == 1:
                        b = labels[i, jp, k]
                        g = u[i, jp, k] - c
                        kf = ky[i, j, k]
                    else:
                        b = labels[i, j, kp]
                        g = u[i, j, kp] - c
                        kf = kz[i, j, k]
                    if kf == 0.0:
                        continue
                    if ax == direction:
                        g += 1.0
                    q = kf * g
                    q2 = q * q
                    E[a] += 0.5 * q2 / vals[a]
                    E[b] += 0.5 * q2 / vals[b]
                    # whatever resistance the face carries beyond its two
                    # half-cells is interface (filler-matrix or contact)
                    extra = 1.0 / kf - 0.5 / vals[a] - 0.5 / vals[b]
                    if extra > 0.0:
                        e_int += q2 * extra
    return E, e_int


# =========================================================================
# preconditioner and CG
# =========================================================================
class FFTPreconditioner:
    """Exact inverse of k0 * (-Laplacian_7pt) on a periodic grid.

    zbc selects the z boundary of a film: "periodic" (FFT), "neumann" -
    insulated faces, the cosine transform (DCT-II) - or "dirichlet" - faces
    at a fixed potential half a cell beyond the last cells, the sine transform
    (DST-II); x and y stay periodic (FFT)."""

    def __init__(self, shape, k0, workers=-1, zbc="periodic"):
        self.shape = shape
        self.workers = workers
        self.zbc = zbc
        n0, n1, n2 = shape
        l0 = 4.0 * np.sin(np.pi * np.arange(n0) / n0) ** 2
        if zbc == "periodic":
            l1 = 4.0 * np.sin(np.pi * np.arange(n1) / n1) ** 2
            l2 = 4.0 * np.sin(np.pi * np.arange(n2 // 2 + 1) / n2) ** 2
        else:
            l1 = 4.0 * np.sin(np.pi * np.arange(n1 // 2 + 1) / n1) ** 2
            m = np.arange(n2) + (1 if zbc == "dirichlet" else 0)
            l2 = 4.0 * np.sin(np.pi * m / (2.0 * n2)) ** 2
        lam = k0 * (l0[:, None, None] + l1[None, :, None] + l2[None, None, :])
        singular = zbc != "dirichlet"
        if singular:
            lam[0, 0, 0] = 1.0
        inv = 1.0 / lam
        if singular:
            inv[0, 0, 0] = 0.0
        self.inv = inv

    def __call__(self, r):
        w = self.workers
        if self.zbc == "periodic":
            R = sfft.rfftn(r, workers=w)
            R *= self.inv
            return sfft.irfftn(R, s=self.shape, workers=w)
        tr, itr = (sfft.dct, sfft.idct) if self.zbc == "neumann" else (sfft.dst, sfft.idst)
        R = tr(r, type=2, axis=2, norm="ortho", workers=w)
        R = sfft.rfftn(R, axes=(0, 1), workers=w)
        R *= np.transpose(self.inv, (0, 1, 2))
        R = sfft.irfftn(R, s=self.shape[:2], axes=(0, 1), workers=w)
        return itr(R, type=2, axis=2, norm="ortho", workers=w)


def _threads():
    """The thread count the job was given (the queue sets NUMBA_NUM_THREADS, a
    parallel-solve worker MPSIM_THREADS); -1 (all cores) would ignore the
    machine setting."""
    import os
    try:
        return max(1, int(os.environ.get("MPSIM_THREADS") or os.environ.get("NUMBA_NUM_THREADS")
                          or os.cpu_count() or 1))
    except ValueError:
        return -1


def solve_direction(labels, vals, direction, rpair=None, tol=1e-7, maxiter=20000,
                    callback=None, should_stop=None, workers=None, cmask=None, rcpair=None,
                    flux_h=None, film=False):
    """Periodic cell problem for a unit macroscopic gradient along `direction`.

    labels : (n0,n1,n2) uint8 phase map;  vals : value per label (> 0)
    rpair  : (nlab,nlab) interfacial resistance divided by the voxel size, in
             the same units as 1/vals (m K/W for thermal), or None.
    cmask  : uint8 bit mask of particle-particle contact faces (bit 0/1/2 =
             +x/+y/+z face), with rcpair the contact resistance / voxel size.
    flux_h : voxel size [m]; when given, the cell-centred flux vector for a
             1 K (1 V) drop across the RVE is returned as "q" (float32).
    film   : the grid is a film (periodic in x and y only). Through the
             thickness (direction 2) this calls solve_film_z; in-plane the
             faces between the top and bottom layers carry nothing, so the
             two film faces are insulated.
    Returns a dict with the effective column, the energy split, the field and
    solver statistics.
    """
    if film and direction == 2:
        return solve_film_z(labels, vals, rpair=rpair, tol=tol, maxiter=maxiter, callback=callback,
                            should_stop=should_stop, workers=workers, cmask=cmask, rcpair=rcpair,
                            flux_h=flux_h)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    vals = np.asarray(vals, dtype=np.float64)
    nlab = len(vals)
    if np.any(vals <= 0):
        raise ValueError("Property values must be positive (apply the contrast limit first)")
    # work in O(1) numbers; everything is linear, so rescale at the end
    scale = math.sqrt(float(vals.min()) * float(vals.max()))
    v = vals / scale
    rp = (np.zeros((nlab, nlab)) if rpair is None
          else np.asarray(rpair, dtype=np.float64) * scale)
    use_c = cmask is not None and rcpair is not None and float(np.max(rcpair)) > 0.0
    rc = (np.asarray(rcpair, dtype=np.float64) * scale) if use_c else np.zeros((nlab, nlab))
    cm = np.ascontiguousarray(cmask, dtype=np.uint8) if use_c else np.zeros((1, 1, 1), np.uint8)
    shape = labels.shape
    kf = [np.empty(shape) for _ in range(3)]
    for ax in range(3):
        _faces(labels, v, rp, cm, rc, use_c, ax, kf[ax])
    if film:
        kf[2][:, :, -1] = 0.0              # the insulated film faces
    b = np.empty(shape)
    _rhs(kf[direction], direction, b)
    b -= b.mean()

    fpos = [k[k > 0] for k in kf]
    fmin = min(float(k.min()) for k in fpos)
    fmax = max(float(k.max()) for k in fpos)
    k0 = math.sqrt(fmin * fmax)
    M = FFTPreconditioner(shape, k0, workers=_threads() if workers is None else workers,
                          zbc="neumann" if film else "periodic")

    u = np.zeros(shape)
    r = b.copy()
    z = M(r)
    z -= z.mean()
    p = z.copy()
    Ap = np.empty(shape)
    rz = float(np.vdot(r, z))
    bnorm = float(np.linalg.norm(b)) or 1.0
    t0 = time.time()
    it, res = 0, 0.0
    converged = bnorm == 1.0 and float(np.linalg.norm(b)) == 0.0
    history = []
    if not converged:
        for it in range(1, maxiter + 1):
            _apply(p, kf[0], kf[1], kf[2], Ap)
            Ap -= Ap.mean()
            pAp = float(np.vdot(p, Ap))
            if pAp <= 0:
                break
            alpha = rz / pAp
            u += alpha * p
            r -= alpha * Ap
            res = float(np.linalg.norm(r)) / bnorm
            if it % 5 == 0 or it == 1:
                history.append((it, res))
            if res < tol:
                converged = True
                break
            if callback is not None and it % 10 == 0:
                callback(it, res)
            if should_stop is not None and it % 25 == 0 and should_stop():
                raise InterruptedError
            z = M(r)
            z -= z.mean()
            rz_new = float(np.vdot(r, z))
            p = z + (rz_new / rz) * p
            rz = rz_new
    u -= u.mean()

    col = np.array([_mean_flux(u, kf[ax], ax, 1.0 if ax == direction else 0.0)
                    for ax in range(3)]) * scale
    E, e_int = _energy(u, kf[0], kf[1], kf[2], labels, v, direction, nlab)
    q = None
    if flux_h is not None:
        # face flux k_f (du + unit gradient) averaged onto the cells; the RVE
        # drop is n voxel-gradients, so one voxel carries 1/n of 1 K
        n = labels.shape[direction]
        q = np.empty(shape + (3,), np.float32)
        for ax in range(3):
            g = np.roll(u, -1, axis=ax) - u
            if ax == direction:
                g += 1.0
            qf = kf[ax] * g
            q[..., ax] = -0.5 * (qf + np.roll(qf, 1, axis=ax)) * (scale / (n * flux_h))
            del g, qf
    ncell = float(labels.size)
    E_total = (float(E.sum()) + e_int) / ncell            # = k_eff/scale
    energy_gap = abs(E_total - col[direction] / scale) / max(abs(col[direction] / scale), 1e-300)
    return {
        "column": col,
        "k_eff": float(col[direction]),
        "energy_frac": (E / max(E.sum() + e_int, 1e-300)).tolist(),
        "energy_frac_interface": float(e_int / max(E.sum() + e_int, 1e-300)),
        "hill_mandel_gap": float(energy_gap),
        "iterations": it, "residual": res, "converged": bool(converged),
        "seconds": time.time() - t0, "history": history,
        "u": u, "q": q,
    }


@njit(parallel=True, cache=True, fastmath=True)
def _apply_film_z(u, kx, ky, kz, gb, gt, out):
    """The operator with the z faces of the film cut and each face tied to its
    plate through the half cell next to it (conductance gb, gt)."""
    _apply(u, kx, ky, kz, out)
    n0, n1, n2 = u.shape
    for i0 in prange(n0):
        i = np.int64(i0)
        for j in range(n1):
            out[i, j, 0] += gb[i, j] * u[i, j, 0]
            out[i, j, n2 - 1] += gt[i, j] * u[i, j, n2 - 1]


def solve_film_z(labels, vals, rpair=None, tol=1e-7, maxiter=20000, callback=None, should_stop=None,
                 workers=None, cmask=None, rcpair=None, flux_h=None):
    """Through-thickness conduction of a film between two plates.

    x and y are periodic; the bottom face is held at 1, the top face at 0
    (a plate at each face, touching every cell of the layer next to it through
    that half cell). k_zz = (heat through the film per area) x thickness, for a
    unit drop across the film; the column also gives the x and y flux the
    drop drives. The potential itself is solved for (no periodic split), so
    "u" is the full field, 1 at the bottom face and 0 at the top.
    """
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    vals = np.asarray(vals, dtype=np.float64)
    nlab = len(vals)
    if np.any(vals <= 0):
        raise ValueError("Property values must be positive (apply the contrast limit first)")
    scale = math.sqrt(float(vals.min()) * float(vals.max()))
    v = vals / scale
    rp = (np.zeros((nlab, nlab)) if rpair is None
          else np.asarray(rpair, dtype=np.float64) * scale)
    use_c = cmask is not None and rcpair is not None and float(np.max(rcpair)) > 0.0
    rc = (np.asarray(rcpair, dtype=np.float64) * scale) if use_c else np.zeros((nlab, nlab))
    cm = np.ascontiguousarray(cmask, dtype=np.uint8) if use_c else np.zeros((1, 1, 1), np.uint8)
    shape = labels.shape
    n0, n1, n2 = shape
    kf = [np.empty(shape) for _ in range(3)]
    for ax in range(3):
        _faces(labels, v, rp, cm, rc, use_c, ax, kf[ax])
    kf[2][:, :, -1] = 0.0
    gb = 2.0 * v[labels[:, :, 0]]          # half a cell between the plate and the first layer
    gt = 2.0 * v[labels[:, :, -1]]
    b = np.zeros(shape)
    b[:, :, 0] = gb * 1.0                  # bottom plate at 1, top plate at 0

    fpos = [k[k > 0] for k in kf]
    k0 = math.sqrt(min(float(k.min()) for k in fpos) * max(float(k.max()) for k in fpos))
    M = FFTPreconditioner(shape, k0, workers=_threads() if workers is None else workers, zbc="dirichlet")
    u = M(b)                               # a good start: the constant-coefficient solution
    Ap = np.empty(shape)
    _apply_film_z(u, kf[0], kf[1], kf[2], gb, gt, Ap)
    r = b - Ap
    z = M(r)
    p = z.copy()
    rz = float(np.vdot(r, z))
    bnorm = float(np.linalg.norm(b)) or 1.0
    t0 = time.time()
    it, res = 0, float(np.linalg.norm(r)) / bnorm
    converged = res < tol
    history = []
    while not converged and it < maxiter:
        it += 1
        _apply_film_z(p, kf[0], kf[1], kf[2], gb, gt, Ap)
        pAp = float(np.vdot(p, Ap))
        if pAp <= 0:
            break
        alpha = rz / pAp
        u += alpha * p
        r -= alpha * Ap
        res = float(np.linalg.norm(r)) / bnorm
        if it % 5 == 0 or it == 1:
            history.append((it, res))
        if res < tol:
            converged = True
            break
        if callback is not None and it % 10 == 0:
            callback(it, res)
        if should_stop is not None and it % 25 == 0 and should_stop():
            raise InterruptedError
        z = M(r)
        rz_new = float(np.vdot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new

    # flux through each z plane (all equal at convergence); the bottom plate's is used
    q_bottom = float(np.sum(gb * (1.0 - u[:, :, 0]))) / (n0 * n1)
    q_top = float(np.sum(gt * u[:, :, -1])) / (n0 * n1)
    kzz = q_bottom * n2
    # x and y flux driven by the drop (the off-diagonal terms), per unit gradient
    qx = -float(np.mean(kf[0] * (np.roll(u, -1, axis=0) - u))) * n2
    qy = -float(np.mean(kf[1] * (np.roll(u, -1, axis=1) - u))) * n2
    col = np.array([qx, qy, kzz]) * scale
    # energy by phase: the dissipation q^2 / k of every face, split between its
    # two half cells, plus the half cells next to the plates
    E = np.zeros(nlab)
    e_int = 0.0
    for ax in range(3):
        g = np.roll(u, -1, axis=ax) - u
        q = kf[ax] * g
        a_lab = labels
        b_lab = np.roll(labels, -1, axis=ax)
        live = kf[ax] > 0
        q2 = (q * q)[live]
        la, lb = a_lab[live], b_lab[live]
        E += np.bincount(la, weights=0.5 * q2 / v[la], minlength=nlab)
        E += np.bincount(lb, weights=0.5 * q2 / v[lb], minlength=nlab)
        extra = (1.0 / kf[ax][live]) - 0.5 / v[la] - 0.5 / v[lb]
        e_int += float(np.sum(q2 * np.maximum(extra, 0.0)))
    qb = gb * (1.0 - u[:, :, 0])
    qt = gt * u[:, :, -1]
    E += np.bincount(labels[:, :, 0].ravel(), weights=(qb * qb / gb).ravel(), minlength=nlab)
    E += np.bincount(labels[:, :, -1].ravel(), weights=(qt * qt / gt).ravel(), minlength=nlab)
    # a unit drop over the film: total dissipation = k_zz / n2 per cell area
    E_total = (float(E.sum()) + e_int) / (n0 * n1)
    energy_gap = abs(E_total - q_bottom) / max(abs(q_bottom), 1e-300)
    q = None
    if flux_h is not None:
        # cell-centred flux for 1 K across the film, W/m^2 (V/m x S/m ...)
        q = np.empty(shape + (3,), np.float32)
        for ax in range(2):
            qf = -kf[ax] * (np.roll(u, -1, axis=ax) - u)
            q[..., ax] = 0.5 * (qf + np.roll(qf, 1, axis=ax)) * (scale / flux_h)
        fz = kf[2] * (u - np.roll(u, -1, axis=2))        # flux across each +z face
        below = np.concatenate([qb[:, :, None], fz[:, :, :-1]], axis=2)
        above = np.concatenate([fz[:, :, :-1], qt[:, :, None]], axis=2)
        q[..., 2] = 0.5 * (below + above) * (scale / flux_h)
    return {
        "column": col,
        "k_eff": float(col[2]),
        "energy_frac": (E / max(E.sum() + e_int, 1e-300)).tolist(),
        "energy_frac_interface": float(e_int / max(E.sum() + e_int, 1e-300)),
        "hill_mandel_gap": float(energy_gap),
        "plate_balance": abs(q_top - q_bottom) / max(abs(q_bottom), 1e-300),
        "iterations": it, "residual": res, "converged": bool(converged),
        "seconds": time.time() - t0, "history": history,
        "u": u, "q": q, "absolute_potential": True,
    }


# =========================================================================
# percolation (periodic)
# =========================================================================
def wrapping_axes(mask):
    """Does a 6-connected cluster of `mask` wrap around the periodic cell?

    Returns [wrap_x, wrap_y, wrap_z]. Labels are computed without periodicity
    and then glued across the three periodic faces with union-find that tracks
    each cluster's offset in cell periods; a cycle whose net offset is non-zero
    along an axis is an infinite (percolating) cluster along that axis.
    """
    mask = np.asarray(mask, bool)
    lab, n = ndimage.label(mask)
    wraps = [False, False, False]
    if n == 0:
        return wraps
    parent = np.arange(n + 1)
    offset = np.zeros((n + 1, 3), np.int64)

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root = x
        acc = np.zeros(3, np.int64)
        for node in reversed(path):
            acc = acc + offset[node]
            offset[node] = acc
            parent[node] = root
        return root

    for ax in range(3):
        hi = np.take(lab, lab.shape[ax] - 1, axis=ax)
        lo = np.take(lab, 0, axis=ax)
        m = (hi > 0) & (lo > 0)
        if not m.any():
            continue
        pairs = np.unique(np.stack([hi[m], lo[m]], axis=1), axis=0)
        e = np.zeros(3, np.int64)
        e[ax] = 1
        for p, q in pairs:
            rp = find(p)
            rq = find(q)
            op = offset[p] if p != rp else np.zeros(3, np.int64)
            oq = offset[q] if q != rq else np.zeros(3, np.int64)
            if rp == rq:
                d = op + e - oq
                for k in range(3):
                    if d[k] != 0:
                        wraps[k] = True
            else:
                parent[rq] = rp
                offset[rq] = op + e - oq
    return wraps


def contrast_policy(labels, vals, direction, cap, wraps_cache):
    """Limit the value contrast so CG converges, choosing the side that keeps
    the answer exact in the limit.

    If the high-value phases percolate along `direction`, the network carries
    the flux and the low values are raised to max/cap (their contribution is
    ~1/cap relative). Otherwise the low phase is continuous and the high values
    are lowered to min*cap - the perfectly-conducting-inclusion limit, which
    the true answer has already saturated to.
    """
    vals = np.asarray(vals, dtype=float).copy()
    vmax = float(vals.max())
    floor = vmax * 1e-30
    vals = np.maximum(vals, floor)
    vmin = float(vals.min())
    if vmax / vmin <= cap:
        return vals, "none"
    key = "wraps"
    if key not in wraps_cache:
        high = np.nonzero(vals >= vmax / 1e3)[0]
        wraps_cache[key] = wrapping_axes(np.isin(labels, high))
    if wraps_cache[key][direction]:
        return np.maximum(vals, vmax / cap), "network"
    return np.minimum(vals, vmin * cap), "dispersed"


def percolating_mask(mask):
    """Voxels of `mask` that belong to a cluster wrapping the periodic cell.

    Flow and diffusion can only use open porosity: a closed cavity (the core of
    a hollow particle, an isolated pore) carries no net transport but a Stokes
    solver still returns a recirculating field inside it. Same union-find with
    period offsets as `wrapping_axes`, but it remembers which clusters wrap.
    """
    mask = np.asarray(mask, bool)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(mask.shape, bool)
    parent = np.arange(n + 1)
    offset = np.zeros((n + 1, 3), np.int64)

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        root = x
        acc = np.zeros(3, np.int64)
        for node in reversed(path):
            acc = acc + offset[node]
            offset[node] = acc
            parent[node] = root
        return root

    wrapping = []
    for ax in range(3):
        hi = np.take(lab, lab.shape[ax] - 1, axis=ax)
        lo = np.take(lab, 0, axis=ax)
        m = (hi > 0) & (lo > 0)
        if not m.any():
            continue
        e = np.zeros(3, np.int64)
        e[ax] = 1
        for p, q in np.unique(np.stack([hi[m], lo[m]], axis=1), axis=0):
            rp, rq = find(p), find(q)
            op = offset[p] if p != rp else np.zeros(3, np.int64)
            oq = offset[q] if q != rq else np.zeros(3, np.int64)
            if rp == rq:
                if np.any(op + e - oq != 0):
                    wrapping.append(rp)
            else:
                parent[rq] = rp
                offset[rq] = op + e - oq
    if not wrapping:
        return np.zeros(mask.shape, bool)
    roots = np.array([find(i) for i in range(n + 1)])
    keep = np.zeros(n + 1, bool)
    keep[np.unique([find(r) for r in wrapping])] = True
    keep[0] = False
    return keep[roots][lab]
