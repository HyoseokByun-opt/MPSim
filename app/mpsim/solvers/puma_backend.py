"""NASA PuMA backend - the default solver for every transport and elastic property.

PuMA (Ferguson et al., SoftwareX 2021; conda-forge package "puma") is the
reference open-source code for voxel homogenisation. What is used from it:

* conductivity (FV)   compute_thermal_conductivity / compute_electrical_conductivity
                      - the same operator serves k, sigma, eps_r and mu_r
* elasticity (FE)     periodic Q1 element-by-element solver
                      (Lopes et al., Comput. Mater. Sci. 219, 2023)
* permeability (FE)   Stokes flow, periodic sides
* tortuosity (FV)     continuum diffusion in the pore phase
* surface area        marching-cubes isosurface

PuMA has no thermal-expansion load. `ThermoElasticFE` adds one *inside* PuMA's
own finite-element formulation rather than beside it:

    element load   b_e = -alpha_e (m_B[xx] + m_B[yy] + m_B[zz])
    element stress s_e = -m_B . x_e - alpha_e C_e : I

m_B = <C B> is PuMA's own element stress-strain matrix, and the element
matrices, periodic DOF map and Krylov solver are PuMA's. Two exact
rearrangements are also made, and the self-test compares the result with
unmodified PuMA: the load and stress loops are vectorised, and the matrix-free
product is done material by material, so it no longer builds a
(24, 24, n_elem) array on every iteration (1.2 GB at 64^3).
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import re
import time
import warnings

import numpy as np

_PUMA = None
_PUMA_ERR = None


def _load():
    global _PUMA, _PUMA_ERR
    if _PUMA is not None or _PUMA_ERR is not None:
        return _PUMA
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pumapy
        _PUMA = pumapy
    except Exception as e:                                    # noqa: BLE001
        _PUMA_ERR = e
    return _PUMA


def available():
    return _load() is not None


def version():
    if _load() is None:
        return None
    try:
        from importlib.metadata import version as v
        return v("puma")
    except Exception:                                         # noqa: BLE001
        return getattr(_PUMA, "__version__", "unknown")


def unavailable_reason():
    _load()
    return None if _PUMA_ERR is None else f"{type(_PUMA_ERR).__name__}: {_PUMA_ERR}"


def require():
    pm = _load()
    if pm is None:
        raise RuntimeError(
            "NASA PuMA could not be imported. The conda-forge package 'puma' is required "
            "(the PyPI package 'pumapy' is a different program).\nCause: " + str(_PUMA_ERR))
    return pm


def set_log_dir(path):
    """PuMA writes a log file per solve into settings['log_location'] (default
    'logs' in the working directory). Keep those files inside the run folder."""
    pm = _load()
    if pm is None:
        return
    try:
        os.makedirs(path, exist_ok=True)
        pm.settings["log_location"] = path
    except Exception:                                         # noqa: BLE001
        pass


# =========================================================================
# stdout -> progress
# =========================================================================
class _Tap(io.TextIOBase):
    """PuMA reports its Krylov iterations on stdout:
        Iteration: 12, driving modified residual = 5.59 --> target = 0.00032
    Turn those into callbacks (with a 0..1 fraction on a log scale between the
    first residual and the target) and keep the other lines as a log."""
    # CG/BiCGSTAB: "residual = r --> target = t"; MINRES: "residual (r1, r2) --> target = t"
    _it = re.compile(r"[Ii]teration\s*[:=]?\s*(\d+).*?[Rr]esidual\s*[=:(]?\s*([0-9.eE+\-]+)"
                     r"(?:.*?target\s*=\s*([0-9.eE+\-]+))?")
    _mem = re.compile(r"memory requirement for simulation:\s*([0-9.]+)\s*([KMG]B)")

    def __init__(self, progress=None, log=None, should_stop=None, tag=""):
        self.progress = progress
        self.log = log
        self.should_stop = should_stop
        self.tag = tag
        self.buf = ""
        self.last_emit = 0.0
        self.lines = []
        self.first = None
        self.last_res = None
        self.target = None
        self.iterations = 0
        self.memory_mb = None

    def writable(self):
        return True

    def write(self, s):
        self.buf += s
        while True:
            cuts = [i for i in (self.buf.find("\n"), self.buf.find("\r")) if i >= 0]
            if not cuts:
                break
            cut = min(cuts)
            line, self.buf = self.buf[:cut], self.buf[cut + 1:]
            self._line(line.strip())
        return len(s)

    def flush(self):
        pass

    def _line(self, line):
        if not line:
            return
        m = self._it.search(line)
        if m:
            it, res = int(m.group(1)), float(m.group(2))
            tgt = float(m.group(3)) if m.group(3) else None
            if self.first is None or it <= 1:
                self.first = res
            self.last_res, self.target, self.iterations = res, tgt, it
            if self.should_stop is not None and it % 5 == 0 and self.should_stop():
                raise InterruptedError
            now = time.time()
            if self.progress and now - self.last_emit > 0.4:
                frac = None
                if tgt and self.first and self.first > tgt and res > 0:
                    frac = max(0.0, min(1.0, math.log(self.first / res) / math.log(self.first / tgt)))
                self.progress(self.tag, it, res, tgt, frac)
                self.last_emit = now
            return
        mm = self._mem.search(line)
        if mm:
            mult = {"KB": 1e-3, "MB": 1.0, "GB": 1e3}[mm.group(2)]
            self.memory_mb = float(mm.group(1)) * mult
        self.lines.append(line)
        if self.log and len(line) < 160 and not line.startswith(("Time to", "Solving Ax")):
            self.log("    [PuMA] " + line)

    def converged(self):
        if self.target is None or self.last_res is None:
            return None
        return self.last_res <= self.target * (1 + 1e-9)


@contextlib.contextmanager
def _tapped(progress=None, log=None, should_stop=None, tag=""):
    tap = _Tap(progress, log, should_stop, tag)
    with contextlib.redirect_stdout(tap), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield tap


def _workspace(labels, voxel_m):
    pm = require()
    ws = pm.Workspace.from_array(np.ascontiguousarray(labels, dtype=np.uint16))
    ws.voxel_length = float(voxel_m)
    return ws


# =========================================================================
# conductivity-type properties
# =========================================================================
def conductivity(labels, values, direction, voxel_m, kind="thermal", tol=1e-6,
                 maxiter=50000, solver_type="cg", progress=None, log=None,
                 should_stop=None, keep_fields=True, method="fe"):
    """PuMA conduction for one macroscopic gradient direction.

    method="fe"  periodic Q1 finite elements (Lopes et al. 2023): fully
                 periodic, so a rigid translation of the RVE leaves k_eff
                 unchanged (measured: identical to 4 decimals for a sphere
                 array shifted by half a cell). The default.
    method="fv"  PuMA finite volume: periodic sides but fixed potential on the
                 two faces normal to `direction`. Fast, but those faces cut
                 particles and the same shifted array moves by +-5 %
                 (measured), so it is offered only as a quick option.

    Returns dict(column, k_eff, T, q, energy_frac, converged, ...).
    """
    pm = require()
    values = np.asarray(values, float)
    # PuMA's stopping test is not scale-free: with sigma ~ 1e-13 the first
    # residual is already below target and it stops after one iteration
    # (measured: 2.15e-13 instead of 1.78e-13). The problem is linear, so
    # solve with O(1) values and scale the answer back.
    pos = values[values > 0]
    scale = math.sqrt(float(pos.min()) * float(pos.max())) if pos.size else 1.0
    vn = values / scale
    ws = _workspace(labels, voxel_m)
    t0 = time.time()
    d = "xyz".index(direction)
    shape = labels.shape
    if method == "fe":
        cmap = pm.AnisotropicConductivityMap()
        for i in np.unique(labels):
            cmap.add_isotropic_material((int(i), int(i)), float(vn[i]))
        Cls = _fe_cond_class()
        with _tapped(progress, log, should_stop, tag=direction) as tap:
            solver = Cls(ws, cmap, direction, tol, maxiter, "minres", True, True)
            solver.error_check()
            solver.compute()
        col = np.asarray(solver.keff, float) * scale
        L_m = shape[d] * voxel_m
        q = solver.q * (scale / L_m)                   # W/m^2 for 1 K across the RVE
        n = shape[d]
        ramp = (np.arange(n) / n).reshape([n if a == d else 1 for a in range(3)])
        T = ramp - solver.T / n                         # macroscopic ramp + periodic fluctuation
        del solver
        lab_i, q_i = np.asarray(labels), q
    else:
        cmap = pm.IsotropicConductivityMap()
        for i, v in enumerate(vn):
            cmap.add_material((i, i), float(v))
        fn = (pm.compute_thermal_conductivity if kind == "thermal"
              else pm.compute_electrical_conductivity)
        with _tapped(progress, log, should_stop, tag=direction) as tap:
            keff, T, q = fn(ws, cmap, direction, side_bc="p", tolerance=tol,
                            maxiter=maxiter, solver_type=solver_type,
                            display_iter=True, method="fv")
        col = np.asarray(keff, float) * scale
        q = np.asarray(q) * scale
        T = np.asarray(T)
        # PuMA returns zero flux on the two fixed-potential slices by
        # construction (LowDk v5); keep them out of the energy split
        n = shape[d]
        sl = [slice(None)] * 3
        sl[d] = slice(1, n - 1)
        sl = tuple(sl)
        lab_i, q_i = np.asarray(labels)[sl], q[sl]
    q2 = (q_i ** 2).sum(axis=-1)
    E = np.zeros(len(values))
    for i in range(len(values)):
        m = lab_i == i
        if m.any() and values[i] > 0:
            E[i] = float(q2[m].sum()) / (2.0 * values[i])
    Et = E.sum()
    return {
        "column": col, "k_eff": float(col[d]), "method": method,
        "T": T if keep_fields else None, "q": q if keep_fields else None,
        "energy_frac": (E / Et if Et > 0 else E).tolist(),
        "iterations": tap.iterations, "residual": tap.last_res, "target": tap.target,
        "converged": tap.converged(), "memory_mb": tap.memory_mb,
        "seconds": time.time() - t0,
    }


# =========================================================================
# multi-core FE: index tables, product and MINRES
# =========================================================================
# A PuMA FE solve takes the same time on 1 or 18 threads (measured, guide 3.7):
# every part of it is single-threaded Python or NumPy. Profile of one 112^3
# conduction solve (85 MINRES iterations): PuMA's index-table loops 5.1 s,
# scipy's MINRES vector work 3.4 s, the matrix-free product 2.0 s (8 threads).
# With `_FE_PARALLEL` all three run on the cores and give PuMA's numbers:
#
# * index tables - PuMA builds the element-material map and the element DOF
#   table in Python loops over every voxel; the same index formulas are
#   evaluated as array expressions (identical arrays, checked by the
#   self-test);
# * product      - the same operator summed per DOF instead of scattered per
#   element: each DOF adds up the rows of the elements that contain it, so the
#   DOFs are independent and a numba prange spreads them over the cores;
# * MINRES       - scipy's MINRES (Paige-Saunders), the same recurrences and
#   stopping tests in the same order, with the vector updates of each
#   iteration fused into three parallel kernels. The iteration stays in
#   Python, so progress lines and stop requests work as before.
#
# Element matrices, right-hand side, periodic DOF map and Krylov method stay
# PuMA's; results agree with the single-threaded path to rounding (1e-14).
import sys
from numba import njit, prange

_FE_PARALLEL = True


def set_fe_parallel(on):
    """Multi-core FE path (True) or PuMA's single-threaded one (False)."""
    global _FE_PARALLEL
    _FE_PARALLEL = bool(on)


def _fe_tables(lx, ly, lz, per_node):
    """PuMA's periodic Q1 DOF table (fe_conductivity / fe_elasticity
    `initialize`) evaluated as array expressions."""
    n_el = lx * ly * lz
    n_els = lx * ly
    n_nodes = (lx + 1) * (ly + 1)
    n = np.arange(n_nodes * (lz + 1), dtype=np.int64)
    i = n % n_nodes
    dof_map = (i - i // (ly + 1) - ly * ((i % (ly + 1)) // ly)) % n_els + ((n // n_nodes) % lz) * n_els
    e = np.arange(n_el, dtype=np.int64)
    es = e % n_els
    N1 = 1 + es + es // ly + (e // n_els) * n_nodes
    N3 = N1 + ly
    N2 = N3 + 1
    N4 = N1 - 1
    nodes = (N1, N2, N3, N4, N1 + n_nodes, N2 + n_nodes, N3 + n_nodes, N4 + n_nodes)
    if per_node == 1:
        rows = [dof_map[Nk] for Nk in nodes]
    else:
        rows = []
        for Nk in nodes:
            d = dof_map[Nk] + 1                 # fe_elasticity numbers nodes from 1
            rows += [d * 3 - 3, d * 3 - 2, d * 3 - 1]
    return np.stack(rows).astype(np.uint32)


def _elem_mat_map(matrix):
    """elemMatMap[i + j*ly + k*lx*ly] = matrix[j, i, k] (PuMA's loop order)."""
    return np.ascontiguousarray(matrix.transpose(2, 0, 1)).reshape(-1).astype(int)


@njit(parallel=True, cache=True)
def _mr_scale(src, s, dst):
    for i in prange(src.size):
        dst[i] = s * src[i]


@njit(parallel=True, cache=True)
def _mr_k1(y, r1, v, c):
    acc = 0.0
    for i in prange(y.size):
        yi = y[i] - c * r1[i]
        y[i] = yi
        acc += v[i] * yi
    return acc


@njit(parallel=True, cache=True)
def _mr_k2(y, r2, a):
    acc = 0.0
    for i in prange(y.size):
        yi = y[i] - a * r2[i]
        y[i] = yi
        acc += yi * yi
    return acc


@njit(parallel=True, cache=True)
def _mr_k2p(y, r2, a, dinv, z):
    acc = 0.0
    for i in prange(y.size):
        yi = y[i] - a * r2[i]
        y[i] = yi
        zi = dinv[i] * yi
        z[i] = zi
        acc += yi * zi
    return acc


@njit(parallel=True, cache=True)
def _dof_diag(ptr, inc_e, inc_a, mat, K, out):
    for d in prange(out.size):
        s = 0.0
        for p in range(ptr[d], ptr[d + 1]):
            a = inc_a[p]
            s += K[mat[inc_e[p]], a, a]
        out[d] = s


@njit(parallel=True, cache=True)
def _mr_k3(v, w1, w2, wout, x, oldeps, delta, denom, phi):
    acc = 0.0
    for i in prange(v.size):
        wi = (v[i] - oldeps * w1[i] - delta * w2[i]) * denom
        wout[i] = wi
        xi = x[i] + phi * wi
        x[i] = xi
        acc += xi * xi
    return acc


_STOP = "relres"
_PRECOND = "jacobi"


def set_fe_precond(mode):
    """"jacobi" (default): MINRES preconditioned by the diagonal of PuMA's
    matrix; "none": plain MINRES as PuMA runs it. The equations and their
    solution are the same; only the number of iterations changes."""
    global _PRECOND
    _PRECOND = mode


def set_fe_stop(mode):
    """"relres": stop when ||b - Ax|| / ||b|| <= tol (default); "scipy": scipy's
    own test ||r|| / (||A|| ||x||) <= tol, which PuMA uses."""
    global _STOP
    _STOP = mode


def _minres_parallel(matvec_into, b, tol, maxiter, dinv=None, floor=0.0):
    """scipy.sparse.linalg.minres (1.11) with M = I, x0 = 0, shift = 0: the same
    recurrences; returns (x, info) as scipy does. Writes PuMA's
    MinResSolverDisplay line every iteration, the driving residual first.

    Stopping test. scipy stops on ||r|| / (||A|| ||x||) <= tol, and ||x||, the
    periodic potential over all nodes, grows with the grid: measured on one
    50 vol% structure at contrast 150, tol 1e-6 stopped after 96, 70 and 58
    iterations on 64^3, 128^3 and 192^3, and k came out +0.4 %, +7.5 % and
    +32 % above the converged value. The relative residual ||r|| / ||b||
    (phibar / beta1 in MINRES) does not depend on the grid size, so it is the
    test used unless set_fe_stop("scipy") asks for PuMA's.

    dinv: the inverse diagonal of A for a Jacobi preconditioner (scipy's
    preconditioned MINRES with M = diag(A)); the residual is then measured in
    the preconditioned norm.

    floor: an absolute residual below which the system counts as solved (a
    round-off multiple of the matrix scale). A homogeneous RVE has b = 0 up to
    round-off, and ||r|| / ||b|| cannot fall below round-off relative to it."""
    n = b.size
    eps = np.finfo(float).eps
    x = np.zeros(n)
    r1 = np.array(b, dtype=float, copy=True)
    zb = None
    if dinv is not None:
        zb = dinv * r1
        beta1 = float(np.dot(r1, zb))
    else:
        beta1 = float(np.dot(r1, r1))
    if beta1 <= 0.0:
        return x, 0
    beta1 = math.sqrt(beta1)
    if beta1 <= floor:
        return x, 0
    pool = [r1, np.empty(n), np.empty(n)]      # r1, r2 and the next product rotate through these
    v = np.empty(n)
    w, w1, w2 = np.zeros(n), np.zeros(n), np.zeros(n)
    r2 = r1
    istop = itn = 0
    oldb, beta, dbar, epsln, phibar = 0.0, beta1, 0.0, 0.0, beta1
    rhs1, rhs2, tnorm2, gmax, gmin = beta1, 0.0, 0.0, 0.0, np.finfo(float).max
    cs, sn = -1.0, 0.0
    while itn < maxiter:
        itn += 1
        _mr_scale(r2 if zb is None else zb, 1.0 / beta, v)
        y = next(a for a in pool if a is not r1 and a is not r2)
        matvec_into(v, y)
        alfa = _mr_k1(y, r1, v, beta / oldb if itn >= 2 else 0.0)
        if zb is None:
            bsq = _mr_k2(y, r2, alfa / beta)
        else:
            bsq = _mr_k2p(y, r2, alfa / beta, dinv, zb)
        r1, r2 = r2, y
        oldb = beta
        if bsq < 0:
            raise ValueError("non-symmetric matrix")
        beta = math.sqrt(bsq)
        tnorm2 += alfa ** 2 + oldb ** 2 + beta ** 2
        if itn == 1 and beta / beta1 <= 10 * eps:
            istop = -1
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        root = math.hypot(gbar, dbar)
        gamma = max(math.hypot(gbar, beta), eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        denom = 1.0 / gamma
        xx = _mr_k3(v, w2, w, w1, x, oldeps, delta, denom, phi)
        w1, w2, w = w2, w, w1
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z
        Anorm = math.sqrt(tnorm2)
        ynorm = math.sqrt(xx)
        epsx = Anorm * ynorm * eps
        rnorm = phibar
        test1 = math.inf if (ynorm == 0 or Anorm == 0) else rnorm / (Anorm * ynorm)
        test2 = math.inf if Anorm == 0 else root / Anorm
        Acond = gmax / gmin
        if _STOP == "relres":
            relres = rnorm / beta1
            if relres <= tol or rnorm <= floor:
                istop = 1
            elif itn >= maxiter:
                istop = 6
            sys.stdout.write(f"\rIteration: {itn}, driving either residual ({relres:0.3e}, {test2:0.3e}) "
                             f"--> target = {tol:0.3e}")
            if istop != 0:
                break
            continue
        if istop == 0:
            if 1 + test2 <= 1:
                istop = 2
            if 1 + test1 <= 1:
                istop = 1
            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1
        sys.stdout.write(f"\rIteration: {itn}, driving either residual ({test1:0.10f}, {test2:0.10f}) "
                         f"--> target = {tol:0.10f}")
        if istop != 0:
            break
    return x, (maxiter if istop == 6 else 0)


def _solve_parallel(self):
    """PropertySolver.solve for the multi-core path (MINRES, no preconditioner)."""
    print(f"Solving Ax=b using {self.solver_type} solver", end='')
    print()
    dinv = None
    floor = 0.0
    if hasattr(self.Amat, "diagonal"):
        dg = self.Amat.diagonal()
        pos = np.where(dg > 0, dg, 0.0)
        if _PRECOND == "jacobi":
            dinv = np.where(dg > 0, 1.0 / np.where(dg > 0, dg, 1.0), 1.0)
            floor = 1e-13 * math.sqrt(float(pos.sum()))          # ||diag||, M^-1 norm
        else:
            floor = 1e-13 * float(np.linalg.norm(pos))
        del dg, pos
    self.x, info = _minres_parallel(self.Amat.matvec_into, np.ascontiguousarray(self.bvec, dtype=float),
                                    self.tolerance, self.maxiter, dinv, floor)
    if info > 0:
        raise Exception("Convergence to tolerance not achieved.")
    if self.del_matrices:
        del self.Amat, self.bvec, self.initial_guess, self.M
    print(" ... Done")


def _fe_compute(self):
    """PuMA's ConductivityFE/ElasticityFE.compute, step for step, except that
    it calls self.solve(): PuMA's calls super().solve(), which would skip the
    subclass's solver."""
    from pumapy.utilities.timer import Timer
    from pumapy.utilities.generic_checks import estimate_max_memory
    t = Timer()
    estimate_max_memory("elasticity_fe", self.ws.matrix.shape, self.solver_type, self.need_to_orient,
                        mf=self.matrix_free)
    self.initialize()
    self.compute_rhs()
    self.assemble_Amatrix()
    print("Time to assemble matrices: ", t.elapsed())
    t.reset()
    self.solve()
    print("Time to solve: ", t.elapsed())
    self.compute_effective_coefficient()
    self.solve_time = t.elapsed()


def _use_parallel_solve(self):
    return (_FE_PARALLEL and self.solver_type == "minres" and self.M is None
            and self.initial_guess is None and hasattr(self.Amat, "matvec_into"))


@njit(parallel=True, cache=True)
def _dof_matvec(x, y, ptr, inc_e, inc_a, dofs, mat, K):
    nd = y.shape[0]
    nloc = dofs.shape[1]
    for d in prange(nd):
        s = 0.0
        for p in range(ptr[d], ptr[d + 1]):
            e = inc_e[p]
            a = inc_a[p]
            Km = K[mat[e], a]
            de = dofs[e]
            for b in range(nloc):
                s += Km[b] * x[de[b]]
        y[d] = s


@njit(cache=True)
def _incidence(dofs, ptr, inc_e, inc_a):
    """DOF -> (element, local index) lists in CSR form, elements ascending
    (the order of a stable argsort, built by counting in O(n))."""
    ne, nloc = dofs.shape
    nd = ptr.shape[0] - 1
    for e in range(ne):
        for a in range(nloc):
            ptr[dofs[e, a] + 1] += 1
    for d in range(nd):
        ptr[d + 1] += ptr[d]
    fill = ptr[:-1].copy()
    for e in range(ne):
        for a in range(nloc):
            d = dofs[e, a]
            q = fill[d]
            inc_e[q] = e
            inc_a[q] = a
            fill[d] = q + 1


def _parallel_operator(pElemDOFNum, elemMatMap, m_K, nd):
    """LinearOperator of PuMA's assembled-by-elements matrix, parallel over DOFs.
    pElemDOFNum (nloc, n_elem), elemMatMap (n_elem,), m_K (nloc, nloc, n_mat)."""
    from scipy.sparse.linalg import LinearOperator
    nloc, ne = pElemDOFNum.shape
    it = np.int32 if max(nd, ne) < 2 ** 31 - 1 else np.int64
    dofs = np.ascontiguousarray(pElemDOFNum.T, dtype=it)          # (n_elem, nloc)
    ptr = np.zeros(nd + 1, np.int64)
    inc_e = np.empty(dofs.size, it)
    inc_a = np.empty(dofs.size, np.int8)
    _incidence(dofs, ptr, inc_e, inc_a)
    K = np.ascontiguousarray(np.transpose(m_K, (2, 0, 1)), dtype=np.float64)   # (n_mat, nloc, nloc)
    mat = np.ascontiguousarray(elemMatMap, dtype=np.int32)

    def matvec(x):
        x = np.ascontiguousarray(np.ravel(x), dtype=np.float64)
        y = np.empty(nd)
        _dof_matvec(x, y, ptr, inc_e, inc_a, dofs, mat, K)
        return y

    def matvec_into(x, y):
        _dof_matvec(x, y, ptr, inc_e, inc_a, dofs, mat, K)

    def diagonal():
        out = np.empty(nd)
        _dof_diag(ptr, inc_e, inc_a, mat, K, out)
        return out
    op = LinearOperator(shape=(nd, nd), matvec=matvec, dtype=float)
    op.matvec_into = matvec_into
    op.diagonal = diagonal
    return op


_CCLS = None


def _fe_cond_class():
    """PuMA's periodic Q1 conduction (ConductivityFE) with the element loops
    vectorised and a material-by-material matrix-free product; PuMA's own
    builds an (8, 8, n_elem) array on every iteration. Same element matrices,
    same DOF map, same MINRES."""
    global _CCLS
    if _CCLS is not None:
        return _CCLS
    require()
    from pumapy.physics_models.finite_element.fe_conductivity import ConductivityFE
    from scipy.sparse.linalg import LinearOperator

    class ConductivityFEFast(ConductivityFE):
        def initialize(self):
            if self.need_to_orient or not _FE_PARALLEL:
                return super().initialize()
            print("Initializing indexing matrices ... ", flush=True, end='')
            self.nElems = self.len_x * self.len_y * self.len_z
            self.nElemS = self.len_x * self.len_y
            for i in range(self.cond_map.get_size()):
                low, high, _ = self.cond_map.get_material(i)
                self.ws[np.logical_and(self.ws.matrix >= low, self.ws.matrix <= high)] = low
            keys, props = self.mat_cond.keys(), self.mat_cond.values()
            del self.mat_cond
            self.mat_cond = dict()
            for counter, (key, prop) in enumerate(sorted(zip(keys, props))):
                self.mat_cond[counter] = prop
                self.ws.matrix[self.ws.matrix == key] = counter
            self.elemMatMap = _elem_mat_map(self.ws.matrix)
            self.create_element_matrices(onlyB=False)
            self.pElemDOFNum = _fe_tables(self.len_x, self.len_y, self.len_z, 1)
            print("Done")

        def compute(self):
            return _fe_compute(self)

        def solve(self):
            if _use_parallel_solve(self):
                return _solve_parallel(self)
            return super().solve()

        def compute_rhs(self):
            w = self.m_B[self.axis][:, self.elemMatMap]
            self.bvec = np.bincount(self.pElemDOFNum.ravel().astype(np.int64), weights=w.ravel(),
                                    minlength=self.nElems).astype(float)
            del self.m_B

        def assemble_Amatrix(self):
            if _FE_PARALLEL:
                self.Amat = _parallel_operator(self.pElemDOFNum, self.elemMatMap, self.m_K, self.nElems)
                return
            nmat = self.m_K.shape[2]
            groups = [np.nonzero(self.elemMatMap == m)[0] for m in range(nmat)]
            idx = [self.pElemDOFNum[:, g].astype(np.int64) for g in groups]
            flat = [ix.ravel() for ix in idx]
            nd = self.nElems
            mK = self.m_K

            def matvec(x):
                x = np.ravel(x)
                y = np.zeros(nd)
                for m in range(nmat):
                    if groups[m].size:
                        y += np.bincount(flat[m], weights=(mK[:, :, m] @ x[idx[m]]).ravel(), minlength=nd)
                return y
            self.Amat = LinearOperator(shape=(nd, nd), matvec=matvec, dtype=float)

        def compute_effective_coefficient(self):
            del self.m_K
            self.create_element_matrices(onlyB=True)
            lx, ly, lz = self.len_x, self.len_y, self.len_z
            self.T = self.x.reshape(lz, lx, ly).transpose(1, 2, 0).copy()
            t = np.zeros(8)
            t[{0: [1, 2, 5, 6], 1: [2, 3, 6, 7], 2: [4, 5, 6, 7]}[self.axis]] = 1.0
            xe = t[:, None] - self.x[self.pElemDOFNum.astype(np.int64)]
            qf = np.empty((self.nElems, 3))
            for c in range(3):
                qf[:, c] = np.einsum("ie,ie->e", self.m_B[c][:, self.elemMatMap], xe)
            del xe
            del self.m_B
            self.q = np.ascontiguousarray(qf.reshape(lz, lx, ly, 3).transpose(1, 2, 0, 3))
            self.keff = qf.mean(axis=0).tolist()

    _CCLS = ConductivityFEFast
    return _CCLS


# =========================================================================
# thermo-elasticity
# =========================================================================
_CLS = None


def _thermo_class():
    global _CLS
    if _CLS is not None:
        return _CLS
    require()
    from pumapy.physics_models.finite_element.fe_elasticity import ElasticityFE
    from scipy.sparse.linalg import LinearOperator

    class ThermoElasticFE(ElasticityFE):
        """PuMA periodic Q1 elasticity; direction 'th' is a unit thermal load."""

        def __init__(self, workspace, elast_map, direction, alpha_by_id, tolerance,
                     maxiter, solver_type, display_iter):
            self.thermal = direction == "th"
            super().__init__(workspace, elast_map, "x" if self.thermal else direction,
                             tolerance, maxiter, solver_type, display_iter, True)
            self.alpha_by_id = dict(alpha_by_id)

        def initialize(self):
            if self.need_to_orient or not _FE_PARALLEL:
                return super().initialize()
            print("Initializing indexing matrices ... ", flush=True, end='')
            self.nElems = self.len_x * self.len_y * self.len_z
            self.nDOFs = self.nElems * 3
            self.nElemS = self.len_x * self.len_y
            for i in range(self.elast_map.get_size()):
                low, high, _ = self.elast_map.get_material(i)
                self.ws[np.logical_and(self.ws.matrix >= low, self.ws.matrix <= high)] = low
            keys, props = self.mat_elast.keys(), self.mat_elast.values()
            del self.mat_elast
            self.mat_elast = dict()
            for counter, (key, prop) in enumerate(sorted(zip(keys, props))):
                self.mat_elast[counter] = prop
                self.ws.matrix[self.ws.matrix == key] = counter
            self.elemMatMap = _elem_mat_map(self.ws.matrix)
            self.create_element_matrices(onlyB=False)
            self.pElemDOFNum = _fe_tables(self.len_x, self.len_y, self.len_z, 3)
            print("Done")

        def compute(self):
            return _fe_compute(self)

        def solve(self):
            if _use_parallel_solve(self):
                return _solve_parallel(self)
            return super().solve()

        def _material_tables(self):
            # PuMA remaps material ids to 0..n-1 in sorted order of the ids
            ids = sorted(self.alpha_by_id)
            present = sorted(self.mat_elast.keys())
            self.alpha_mat = np.array([self.alpha_by_id[ids[m]] for m in present], float)
            self.C_mat = np.stack([self.create_C(self.mat_elast[m]) for m in present], axis=-1)

        def compute_rhs(self):
            self._material_tables()
            if self.thermal:
                vec = -(self.m_B[0] + self.m_B[1] + self.m_B[2]) * self.alpha_mat[None, :]
            else:
                vec = self.m_B[self.axis]
            w = vec[:, self.elemMatMap]
            self.bvec = np.bincount(self.pElemDOFNum.ravel().astype(np.int64),
                                    weights=w.ravel(), minlength=self.nDOFs).astype(float)
            del self.m_B

        def assemble_Amatrix(self):
            if _FE_PARALLEL:
                self.Amat = _parallel_operator(self.pElemDOFNum, self.elemMatMap, self.m_K, self.nDOFs)
                return
            nmat = self.m_K.shape[2]
            groups = [np.nonzero(self.elemMatMap == m)[0] for m in range(nmat)]
            idx = [self.pElemDOFNum[:, g].astype(np.int64) for g in groups]
            flat = [ix.ravel() for ix in idx]
            nd = self.nDOFs
            mK = self.m_K

            def matvec(x):
                x = np.ravel(x)
                y = np.zeros(nd)
                for m in range(nmat):
                    if groups[m].size == 0:
                        continue
                    ye = mK[:, :, m] @ x[idx[m]]
                    y += np.bincount(flat[m], weights=ye.ravel(), minlength=nd)
                return y
            self.Amat = LinearOperator(shape=(nd, nd), matvec=matvec, dtype=float)

        def compute_effective_coefficient(self):
            del self.m_K
            self.create_element_matrices(onlyB=True)
            t = np.zeros(24)
            if not self.thermal:
                pattern = {0: [3, 6, 15, 18], 1: [7, 10, 19, 22], 2: [14, 17, 20, 23],
                           3: [6, 9, 18, 21], 4: [12, 15, 18, 21], 5: [8, 11, 20, 23]}
                t[pattern[self.axis]] = 1.0
            xe = t[:, None] - self.x[self.pElemDOFNum.astype(np.int64)]
            sf = np.empty((self.nElems, 6))
            for c in range(6):
                sf[:, c] = np.einsum("ie,ie->e", self.m_B[c][:, self.elemMatMap], xe)
            del xe
            if self.thermal:
                cI = self.C_mat[:, 0:3, :].sum(axis=1)                 # (6, nmat)
                sf -= (cI * self.alpha_mat[None, :]).T[self.elemMatMap]
            lx, ly, lz = self.len_x, self.len_y, self.len_z
            field = sf.reshape(lz, lx, ly, 6).transpose(1, 2, 0, 3)
            self.s = np.ascontiguousarray(field[..., 0:3])
            self.t = np.ascontiguousarray(field[..., 3:6])
            self.u = None
            self.Ceff = sf.mean(axis=0).tolist()
            del self.m_B

    _CLS = ThermoElasticFE
    return _CLS


def _iso_material(emap, i, E, nu):
    """An isotropic phase with mu on the shear diagonal.

    PuMA's own add_isotropic_material puts 2 mu there (Mandel convention)
    while its strain-displacement matrix uses engineering shear strain, so
    every phase entered the solve with twice its shear modulus; v4 halved
    the shear output afterwards, which corrects a homogeneous RVE only. On a
    25 vol% alumina/epoxy RVE C11 came out 12 % and the bulk modulus 11 %
    too high, the CTE 8 % too low (v5 check against solvers/fans.py, which
    reproduces PuMA exactly when given the doubled shear modulus).
    The full matrix keeps the same elements with the right material."""
    lam = nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    d = lam + 2.0 * mu
    emap.add_material((i, i), d, lam, lam, 0.0, 0.0, 0.0, d, lam, 0.0, 0.0, 0.0, d, 0.0, 0.0, 0.0,
                      mu, 0.0, 0.0, mu, 0.0, mu)


def elasticity_map():
    pm = require()
    if hasattr(pm, "experimental") and hasattr(pm.experimental, "ElasticityMap"):
        return pm.experimental.ElasticityMap()
    from pumapy.physics_models.utils.property_maps import ElasticityMap
    return ElasticityMap()


def _elastic_inputs(E, nu, alpha, void_ratio):
    E = np.asarray(E, float).copy()
    nu = np.asarray(nu, float).copy()
    alpha = np.asarray(alpha, float)
    solid = E > 0
    void_phases = np.nonzero(~solid)[0].tolist()
    if void_phases:
        E[~solid] = void_ratio * E[solid].min()
        nu[~solid] = 0.2
    return E, nu, alpha, void_phases


def elastic_load(labels, E, nu, alpha, voxel_m, direction, void_ratio=1e-4, tol=1e-6,
                 maxiter=50000, solver_type="minres", progress=None, should_stop=None):
    """One load case of the thermo-elastic problem (a unit Mandel strain along
    `direction`, or the thermal load "th"). Returns (Ceff column in Mandel
    notation, normal stress field, Mandel shear stress field, stats) - the
    building block that thermo_elastic assembles and the parallel pool runs."""
    E, nu, alpha, _ = _elastic_inputs(E, nu, alpha, void_ratio)
    present = np.unique(labels)
    Cls = _thermo_class()
    ws = _workspace(labels, voxel_m)
    emap = elasticity_map()
    for i in present:
        _iso_material(emap, int(i), float(E[i]), float(nu[i]))
    solver = Cls(ws, emap, direction, {int(i): float(alpha[i]) for i in present},
                 tol, maxiter, solver_type, True)
    t0 = time.time()
    with _tapped(progress, None, should_stop, tag=direction) as tap:
        solver.error_check()
        solver.compute()
    st = {"load": direction, "seconds": time.time() - t0, "iterations": tap.iterations,
          "residual": tap.last_res, "target": tap.target, "converged": tap.converged()}
    return np.asarray(solver.Ceff, float), solver.s, solver.t, st


class _Done:
    """A load case solved elsewhere (the parallel pool)."""

    def __init__(self, Ceff, s, t):
        self.Ceff, self.s, self.t = Ceff, s, t


def thermo_elastic(labels, E, nu, alpha, voxel_m, void_ratio=1e-4, tol=1e-6,
                   maxiter=50000, solver_type="minres", progress=None, log=None,
                   should_stop=None, keep_field="th", pre_loads=None):
    """C* [units of E] and alpha* [units of alpha] of a periodic RVE.

    Voids (E = 0) get E = void_ratio * (smallest solid E) so the stiffness
    matrix stays non-singular; the ratio is returned with the result.
    `keep_field` selects which load's stress field is returned. "th" is the
    thermal load at zero macroscopic strain (the RVE held rigidly); "free" is
    the same temperature change with zero macroscopic *stress* - the composite
    expands by alpha* dT and only the mismatch between phases is left, which is
    what a filler feels inside a free film. It is built by superposition:
    sigma_free = sigma_th + sum_k (alpha*_k) sigma_k, so the six strain-load
    fields are kept (float32) until alpha* is known.
    """
    E = np.asarray(E, float).copy()
    nu = np.asarray(nu, float).copy()
    alpha = np.asarray(alpha, float)
    solid = E > 0
    void_phases = np.nonzero(~solid)[0].tolist()
    if void_phases:
        E[~solid] = void_ratio * E[solid].min()
        nu[~solid] = 0.2
    present = np.unique(labels)
    Cls = _thermo_class()
    names = ["x", "y", "z", "yz", "xz", "xy"]
    C = np.zeros((6, 6))
    stats = []
    fields = None
    pre_loads = pre_loads or {}

    def one(direction):
        if direction in pre_loads:
            Ceff, s_, t_, st_ = pre_loads[direction]
            return _Done(Ceff, s_, t_), st_
        ws = _workspace(labels, voxel_m)
        emap = elasticity_map()
        for i in present:
            _iso_material(emap, int(i), float(E[i]), float(nu[i]))
        solver = Cls(ws, emap, direction, {int(i): float(alpha[i]) for i in present},
                     tol, maxiter, solver_type, True)
        t0 = time.time()
        with _tapped(progress, None, should_stop, tag=direction) as tap:
            solver.error_check()
            solver.compute()
        return solver, {"load": direction, "seconds": time.time() - t0,
                        "iterations": tap.iterations, "residual": tap.last_res,
                        "target": tap.target, "converged": tap.converged()}

    # PuMA works in Mandel (Kelvin) notation: shear strains and shear stresses
    # both carry a factor sqrt(2), which keeps the matrix symmetric and puts 2G
    # on the shear diagonal (its isotropic material has 2*mu there). Everything
    # downstream - the reports, the engineering constants, the stress maps - is
    # Voigt with engineering shear strain and physical stress, so all of it is
    # converted here: C_v = D^-1 C D^-1, sigma = D^-1 sigma~, D = diag(1,1,1,r2,r2,r2).
    # Taken as it came, G was twice too large (a homogeneous E 3.2 GPa, nu 0.35
    # RVE returned 2.370 GPa against 1.185), the normal-shear couplings sqrt(2)
    # too large, and the shear-stress maps and von Mises likewise. E, the Poisson
    # ratios, the bulk modulus, the CTE and the normal stresses were unaffected.
    # with _iso_material the output is plain Voigt (engineering shear): the
    # Mandel conversion below is kept as an identity
    dm = np.ones(6)
    unit = []                   # strain-load stress fields, only for "free"
    for k, d in enumerate(names):
        s, st = one(d)
        C[:, k] = s.Ceff
        stats.append(st)
        if keep_field == d and s.s is not None:
            fields = (s.s, s.t * dm[3])
        if keep_field == "free":
            unit.append((np.asarray(s.s, np.float32), np.asarray(s.t * dm[3], np.float32)))
        del s
    s, st = one("th")
    sig_th = np.asarray(s.Ceff, float) * dm
    stats.append(st)
    if keep_field in ("th", "free") and s.s is not None:
        fields = (s.s, s.t * dm[3])
    del s
    C = C * np.outer(dm, dm)
    asym = float(np.abs(C - C.T).max() / max(np.abs(C).max(), 1e-300))
    Csym = 0.5 * (C + C.T)
    alpha_star = -np.linalg.solve(Csym, sig_th)
    if keep_field == "free" and fields is not None:
        # load k was a unit Mandel strain; the free expansion is the Voigt
        # (engineering) strain alpha*, i.e. Mandel amplitude dm_k * alpha*_k
        amp = dm * alpha_star
        sn, sh = np.array(fields[0], float), np.array(fields[1], float)
        for k, (un, us) in enumerate(unit):
            sn += amp[k] * un
            sh += amp[k] * us
        fields = (sn, sh)
        del unit
    # PuMA's frame is ours mirrored in y, with its shear rows in the order
    # xy, xz, yz: its "yz" entries are our xy ones and both change sign
    # (v5 check against solvers/fans.py: equal to 1e-9 after this mapping;
    # v4 reported G_xy as G_yz and the other way round).
    P = [0, 1, 2, 5, 4, 3]
    sgn = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    Csym = (sgn[:, None] * sgn[None, :]) * Csym[np.ix_(P, P)]
    alpha_star = sgn * alpha_star[P]
    sig_th = sgn * sig_th[P]
    if fields is not None:
        tp = np.asarray(fields[1])
        fields = (fields[0], np.stack([-tp[..., 2], tp[..., 1], -tp[..., 0]], axis=-1))
    return {"C": Csym, "alpha": alpha_star, "sigma_th": sig_th, "asymmetry": asym,
            "stats": stats, "fields": fields, "void_phases": void_phases,
            "E_used": E.tolist(), "nu_used": nu.tolist(), "void_ratio": void_ratio}


# =========================================================================
# porous transport and morphology
# =========================================================================
def permeability(labels, void_ids, voxel_m, direction, tol=1e-6, maxiter=50000,
                 progress=None, log=None, should_stop=None):
    """Darcy permeability [m^2] from PuMA's periodic Stokes FE solver."""
    pm = require()
    lab = np.asarray(labels)
    binary = np.where(np.isin(lab, void_ids), 0, 1).astype(np.uint16)
    ws = _workspace(binary, voxel_m)
    t0 = time.time()
    with _tapped(progress, log, should_stop, tag=direction) as tap:
        keff, (ux, uy, uz) = pm.compute_permeability(
            ws, (1, 1), direction=direction, tol=tol, maxiter=maxiter,
            solver_type="minres", display_iter=True, matrix_free=True,
            output_fields=True)
    return {"K": np.asarray(keff, float), "u": (ux, uy, uz), "seconds": time.time() - t0,
            "iterations": tap.iterations, "converged": tap.converged()}


def tortuosity(labels, void_ids, voxel_m, direction, tol=1e-6, maxiter=50000,
               progress=None, log=None, should_stop=None):
    """Tortuosity factor and normalised effective diffusivity of the pore
    space (PuMA continuum tortuosity, periodic sides)."""
    pm = require()
    lab = np.asarray(labels)
    binary = np.where(np.isin(lab, void_ids), 0, 1).astype(np.uint16)
    ws = _workspace(binary, voxel_m)
    t0 = time.time()
    with _tapped(progress, log, should_stop, tag=direction) as tap:
        eta, deff, poro, Cfield = pm.compute_continuum_tortuosity(
            ws, (0, 0), direction, side_bc="p", tolerance=tol, maxiter=maxiter,
            solver_type="cg", display_iter=True)
    d = "xyz".index(direction)
    return {"tortuosity": float(eta[d]), "d_eff": float(np.asarray(deff)[d]),
            "porosity": float(poro), "C": np.asarray(Cfield),
            "seconds": time.time() - t0, "converged": tap.converged()}


def surface_area(labels, phase_ids, voxel_m):
    """Interface area [m^2] and specific surface area [1/m] of a phase set."""
    pm = require()
    lab = np.asarray(labels)
    binary = np.where(np.isin(lab, phase_ids), 1, 0).astype(np.uint16)
    ws = _workspace(binary, voxel_m)
    with _tapped():
        area, specific = pm.compute_surface_area(ws, (1, 1))
    return float(area), float(specific)


def mean_intercept_length(labels, phase_ids, voxel_m):
    """PuMA mean intercept length [m] of a phase set along x, y and z: the
    mean chord of the phase between two interface crossings."""
    pm = require()
    lab = np.asarray(labels)
    binary = np.where(np.isin(lab, phase_ids), 0, 1).astype(np.uint16)
    ws = _workspace(binary, voxel_m)
    with _tapped():
        mil = pm.compute_mean_intercept_length(ws, (0, 0))
    return [float(v) for v in mil]


def radiation(labels, void_ids, voxel_m, sources=200, rays=500, should_stop=None):
    """Extinction coefficient beta [1/m] of the pore space along x, y and z by
    PuMA's ray casting (opaque solid, rays emitted from random pore sources,
    periodic domain)."""
    pm = require()
    lab = np.asarray(labels)
    binary = np.where(np.isin(lab, void_ids), 0, 1).astype(np.uint16)
    ws = _workspace(binary, voxel_m)
    t0 = time.time()
    with _tapped(None, None, should_stop):
        beta, beta_std, dist = pm.experimental.compute_radiation(ws, (0, 0), int(sources), int(rays))
    beta = np.atleast_1d(np.asarray(beta, float))
    std = np.atleast_1d(np.asarray(beta_std, float))
    out = {"beta_1pm": beta.tolist(), "beta_std_1pm": std.tolist(), "seconds": time.time() - t0,
           "sources": int(sources), "rays": int(rays)}
    try:
        # PuMA returns the ray lengths in voxels (its fit divides beta by
        # ws.voxel_length), so convert with the voxel size, not with 1e6
        d = np.asarray(dist, float).ravel() * float(voxel_m)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            hi = float(np.percentile(d, 99.5))
            hist, edges = np.histogram(d[d <= hi], bins=40)
            out["free_path"] = {"length_um": (0.5 * (edges[1:] + edges[:-1]) * 1e6).tolist(),
                                "probability": (hist / max(hist.sum(), 1)).tolist(),
                                "mean_um": float(d.mean() * 1e6)}
    except Exception:                                         # noqa: BLE001
        pass
    return out


def orientation_tensor(labels, phase_ids, voxel_m, sigma=0.7, rho=1.4):
    """Second-order orientation tensor <n n> of a phase from PuMA's structure
    tensor analysis of the voxel image (the local direction of least
    grey-value variation: the fibre axis for rods, an in-plane direction for
    platelets).

    The analysis runs on the Euclidean distance transform (edt=True): inside a
    binary phase the grey value is constant, and without it every vector
    comes back as (1, 0, 0) - measured on z- and x-aligned rods, which then
    give A_zz = 0.94 and A_xx = 0.94 respectively."""
    pm = require()
    lab = np.asarray(labels)
    binary = np.where(np.isin(lab, phase_ids), 1, 0).astype(np.uint16)
    ws = _workspace(binary, voxel_m)
    with _tapped():
        pm.compute_orientation_st(ws, (1, 1), sigma=sigma, rho=rho, edt=True)
    ori = np.asarray(ws.orientation, float)
    m = binary.astype(bool)
    v = ori[m]
    del ori
    nrm = np.linalg.norm(v, axis=1)
    v = v[nrm > 1e-12] / nrm[nrm > 1e-12, None]
    if v.shape[0] == 0:
        return None
    A = (v.T @ v) / v.shape[0]
    return A
