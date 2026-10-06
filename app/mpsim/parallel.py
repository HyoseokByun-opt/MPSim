"""Independent solves of one job, run side by side.

How the cores of a run are shared (measured on a 14-core PC, guide "Cores"):
a PuMA FE solve on the multi-core path scales to 4.3x on 14 threads (one
elastic load at 64^3, converged: 36.0 s on 1 thread, 13.7 s on 4, 8.3 s on
14), but per core one thread per solve does the most work (36 core-seconds
against 55 at 4 threads). A run therefore gives each independent problem a worker first -
a thermal + dielectric + CTE run has 3 + 3 + 7 of them (3 directions each, 7
elastic load cases) - and shares the cores left over among the workers as
threads. `plan_split` picks the split: for every worker count up to
min(cores, problems, what fits in memory) it schedules the problems longest
first on workers of cores // workers threads, using the measured speed-up of
one solve, and keeps the split that finishes first. Elastic load cases are
dispatched first so the longest solves do not finish last on one worker.

Since v5.1 every independent solve of a run goes into the same pool, not
only conduction and PuMA's elastic loads: the Stokes and diffusion solves
of each direction, the EMI admittivity at each frequency and direction, the
moisture diffusion of each direction, the ray casting, the structure
analyses of the first realisation (size distributions, percolation,
porosimetry, pore network, grain statistics) and the conduction solves of
the resolution check on the finer grid. A permeability
solve keeps one core busy for minutes (PuMA's matrix-free MINRES); three of
them and the diffusion solves now run at once on the cores the job was
given, instead of one after another (v5.0.1 solved 3 x 340 s in turn on an
80^3 porous RVE). Each kind has its own cost and its own gain from threads
(TASK_COST, TASK_PAR), and a solve starts only when its memory fits beside
the solves already running, so a heavy Stokes solve and many light FV
solves share the pool without overcommitting the memory.

The pool is started once per job. The structure is written once to a .npy
file that every worker memory-maps; each worker returns its small numbers and
writes its fields (temperature, flux, stress) to .npy files the main process
loads for the viewer. A stop request terminates the workers at once.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np


# --------------------------------------------------------------- worker side
_LIMITS = None


def _init(threads, log_dir):
    # NUMBA_NUM_THREADS is not changed here: a worker whose main module
    # imported numba before this initializer has already launched its pool,
    # and every later compile re-reads the variable and fails when it
    # differs. The pool keeps its size; set_num_threads limits how many of
    # its threads a parallel loop uses (per calling thread, and the tasks run
    # on this one).
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    os.environ["MPSIM_THREADS"] = str(threads)
    os.environ["MPSIM_WORKER_LOG_DIR"] = log_dir or ""
    import numba
    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
    # numpy (and with it MKL / OpenBLAS) was loaded before this initializer,
    # with the job's thread count in the environment: eight workers of one
    # thread each then multiplied vectors on eight threads apiece (an EMI batch
    # kept 10.5 cores busy on 8). threadpoolctl caps the pools already loaded.
    global _LIMITS
    try:
        from threadpoolctl import threadpool_limits
        _LIMITS = threadpool_limits(limits=max(1, int(threads)))
    except Exception:                                                  # noqa: BLE001
        _LIMITS = None
    import warnings
    warnings.filterwarnings("ignore")


class _Reporter:
    """A worker's iterations, sent to the main process at most twice a second
    so the convergence view can draw every solve running side by side.
    Accepts both callback forms: (it, res) and (tag, it, res, target, frac)."""

    def __init__(self, q, name, target):
        self.q, self.name, self.target, self.t = q, name, target, 0.0

    def __call__(self, *a):
        if self.q is None:
            return
        if len(a) == 2:
            it, res, tgt = a[0], a[1], self.target
        else:
            it, res = a[1], a[2]
            tgt = a[3] if len(a) > 3 and a[3] is not None else self.target
        now = time.time()
        if now - self.t < 0.5:
            return
        self.t = now
        try:
            self.q.put_nowait((self.name, int(it), float(res), float(tgt) if tgt is not None else None))
        except Exception:                                              # noqa: BLE001
            pass


def _save(tmp, tag, arr):
    if arr is None:
        return None
    p = os.path.join(tmp, f"{tag}.npy")
    np.save(p, np.asarray(arr))
    return p


def _worker_labels(t, PB):
    labels = np.ascontiguousarray(np.load(t["labels"], mmap_mode="r"))
    if os.environ.get("MPSIM_WORKER_LOG_DIR"):
        PB.set_log_dir(os.path.join(os.environ["MPSIM_WORKER_LOG_DIR"], f"w{os.getpid()}"))
    return labels


def stokes_task(t):
    """One permeability direction: PuMA's periodic Stokes solve in the open
    pores. The velocity components go back as files (the viewer, the
    hydraulic tortuosity and the filtration stage read them)."""
    from .solvers import puma_backend as PB
    labels = _worker_labels(t, PB)
    rep = _Reporter(t.get("q"), t.get("name") or f"Stokes flow {t['d']}", t["tol"])
    pr = PB.permeability(labels, t["void"], t["voxel_m"], t["d"], tol=t["tol"], progress=rep, should_stop=None)
    comps = [np.asarray(c, float) for c in pr["u"] if c is not None]
    if comps and comps[0].ndim == 4:
        comps = [comps[0][..., k] for k in range(comps[0].shape[-1])]
    tag = f"stokes_{t['d']}"
    return {"kind": "stokes", "d": t["d"], "K": np.asarray(pr["K"], float),
            "stats": {"seconds": pr["seconds"], "iterations": pr["iterations"], "converged": pr["converged"]},
            "u_paths": [_save(t["tmp"], f"{tag}_u{k}", c) for k, c in enumerate(comps)]}


def tort_task(t):
    """One diffusion direction: PuMA's continuum tortuosity in the open pores."""
    from .solvers import puma_backend as PB
    labels = _worker_labels(t, PB)
    rep = _Reporter(t.get("q"), t.get("name") or f"Diffusion {t['d']}", t["tol"])
    tr = PB.tortuosity(labels, t["void"], t["voxel_m"], t["d"], tol=t["tol"], progress=rep, should_stop=None)
    C = tr.pop("C", None)
    return {"kind": "tort", "d": t["d"], "r": tr,
            "C_path": _save(t["tmp"], f"tort_{t['d']}_C", C) if t["fields"] else None}


def emi_task(t):
    """The complex admittivity at one frequency along one direction."""
    from .solvers import complex_cond as CC
    labels = np.ascontiguousarray(np.load(t["labels"], mmap_mode="r"))
    r = CC.solve(labels, t["kap"], t["d"], bc=t["bc"], tol=t["tol"], should_stop=None)
    return {"kind": "emi", "i": t["i"], "d": t["d"],
            "r": {"kappa": complex(r["kappa"]), "iterations": r["iterations"], "converged": bool(r["converged"])}}


def radiation_task(t):
    """Ray casting of the pore space (one task: PuMA's casting runs on one core)."""
    from .solvers import puma_backend as PB
    labels = _worker_labels(t, PB)
    return {"kind": "radiation", "r": PB.radiation(labels, t["void"], t["voxel_m"], t["sources"], t["rays"], None)}


def struct_task(t):
    """One structure analysis of the first realisation: the size
    distributions of a phase, its percolation, the porosimetry, the pore
    network or the grain statistics. Its log lines come back with it."""
    from .solvers import puma_backend as PB
    labels = _worker_labels(t, PB)
    what, h, lines = t["what"], t["h"], []
    out = {"kind": "struct", "what": what, "gkey": t.get("gkey"), "lines": lines}

    def save(tag, a):
        return _save(t["tmp"], f"{what}_{t.get('gkey') or ''}_{tag}", a) if a is not None else None
    if what == "morph":
        from . import morphology as MO
        mres, lt = MO.analyse_phase(labels, t["ids"], h)
        try:
            area, sv = PB.surface_area(labels, t["ids"], h * 1e-6)
            mres["surface_area_m2"], mres["specific_surface_1pm"] = area, sv
            mres["mean_intercept_length_um"] = [v * 1e6 for v in PB.mean_intercept_length(labels, t["ids"], h * 1e-6)]
        except Exception as e:                                          # noqa: BLE001
            mres["surface_error"] = str(e)
        out["res"], out["lt_path"] = mres, save("lt", lt) if t["fields"] else None
    elif what == "perc":
        from . import percolation as PC
        pres, pfields = PC.analyse(np.isin(labels, t["ids"]), h, t["dirs"], full_connectivity=t["full"],
                                   want_fields=t["fields"], want_paths=not t["full"])
        out["res"] = pres
        out["field_paths"] = {k: save(k, v) for k, v in (pfields or {}).items()}
    elif what == "poros":
        from . import poro as PO
        pres, pfield = PO.porosimetry(np.isin(labels, t["void"]), h, t["gamma"], t["theta"], t["steps"], t["dirs"],
                                      log=lines.append)
        out["res"], out["field_path"] = pres, save("mip", pfield)
    elif what == "network":
        from . import poro as PO
        out["res"], out["view"] = PO.pore_network(np.isin(labels, t["void"]), h, log=lines.append)
    elif what == "grains":
        from . import grains as GR
        gres, glt = GR.analyse(t["spec"], t["ginfo"], labels, t["table"], h, log=lines.append)
        out["res"], out["lt_path"] = gres, save("glt", glt)
    return out


def cond_task(t):
    """One conduction-type solve (one property, one direction)."""
    from .solvers import puma_backend as PB
    from .solvers import conduction as CD
    labels = np.load(t["labels"], mmap_mode="r")
    labels = np.ascontiguousarray(labels)
    if os.environ.get("MPSIM_WORKER_LOG_DIR"):
        PB.set_log_dir(os.path.join(os.environ["MPSIM_WORKER_LOG_DIR"], f"w{os.getpid()}"))
    tag = f"{t['key']}_{t['d']}"
    out = {"key": t["key"], "d": t["d"]}
    t0 = time.time()
    rep = _Reporter(t.get("q"), t.get("name") or tag, t["tol"])
    primary, _, check = t["backend"].partition("+")
    if primary in ("fe", "puma_fv"):
        r = PB.conductivity(labels, t["vals"], t["d"], t["voxel_m"], kind=t["kind"], tol=t["tol"],
                            progress=rep, log=None, should_stop=None, keep_fields=t["fields"],
                            method="fe" if primary == "fe" else "fv")
        if check == "fv":
            try:
                xc = CD.solve_direction(labels, t["vals"], "xyz".index(t["d"]), tol=1e-8)
                r["crosscheck"], r["crosscheck_method"] = xc["k_eff"], "fv"
            except Exception as e:                                    # noqa: BLE001
                out["note"] = f"cross-check FV failed: {e}"
        out["T_path"] = _save(t["tmp"], tag + "_T", r.pop("T", None)) if t["fields"] else None
        out["q_path"] = _save(t["tmp"], tag + "_q", r.pop("q", None)) if t["fields"] else None
        r.pop("T", None)
        r.pop("q", None)
    else:
        cm = np.load(t["cmask"]) if t.get("cmask") else None
        r = CD.solve_direction(labels, t["vals"], "xyz".index(t["d"]), tol=t["tol"], rpair=t.get("rpair"),
                               cmask=cm, rcpair=t.get("rcpair"), callback=rep,
                               flux_h=t["voxel_m"] if t["fields"] else None, film=bool(t.get("film")))
        r["method"] = ("fv-film" if t.get("film") else
                       "fv-interface" if (t.get("rpair") is not None or t.get("rcpair") is not None) else "fv-periodic")
        if check == "fe":
            try:
                xc = PB.conductivity(labels, t["vals"], t["d"], t["voxel_m"], kind=t["kind"], tol=t["tol"],
                                     progress=None, log=None, should_stop=None, keep_fields=False, method="fe")
                r["crosscheck"], r["crosscheck_method"] = xc["k_eff"], "fe"
            except Exception as e:                                    # noqa: BLE001
                out["note"] = f"cross-check PuMA FE failed: {e}"
        out["u_path"] = _save(t["tmp"], tag + "_u", r.pop("u", None)) if t["fields"] else None
        out["q_path"] = _save(t["tmp"], tag + "_q", r.pop("q", None)) if t["fields"] else None
        r.pop("u", None)
        r.pop("q", None)
    r["worker_seconds"] = time.time() - t0
    out["r"] = r
    return out


def elastic_task(t):
    """One elastic load case (a unit strain or the thermal load)."""
    from .solvers import puma_backend as PB
    labels = np.ascontiguousarray(np.load(t["labels"], mmap_mode="r"))
    if os.environ.get("MPSIM_WORKER_LOG_DIR"):
        PB.set_log_dir(os.path.join(os.environ["MPSIM_WORKER_LOG_DIR"], f"w{os.getpid()}"))
    rep = _Reporter(t.get("q"), t.get("name") or f"elastic {t['load']}", t["tol"])
    Ceff, s, tt, st = PB.elastic_load(labels, t["E"], t["nu"], t["alpha"], t["voxel_m"], t["load"],
                                      tol=t["tol"], progress=rep)
    tag = f"el_{t['load']}"
    return {"load": t["load"], "Ceff": np.asarray(Ceff, float), "stats": st,
            "s_path": _save(t["tmp"], tag + "_s", np.asarray(s, np.float32)) if t["fields"] else None,
            "t_path": _save(t["tmp"], tag + "_t", np.asarray(tt, np.float32)) if t["fields"] else None}


# ----------------------------------------------------------------- main side
class SolvePool:
    """A process pool for the independent solves of one job."""

    def __init__(self, workers, threads_each, log_dir=None):
        self.workers = int(workers)
        self.threads_each = int(threads_each)
        self.tmp = tempfile.mkdtemp(prefix="mpsim_par_")
        ctx = mp.get_context("spawn")
        self.ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_init,
                                      initargs=(self.threads_each, log_dir))
        self._labels_path = None
        # the workers' iterations come back through a managed queue
        try:
            self.mgr = ctx.Manager()
            self.q = self.mgr.Queue()
        except Exception:                                              # noqa: BLE001
            self.mgr, self.q = None, None

    def share(self, labels, name="labels"):
        p = os.path.join(self.tmp, f"{name}.npy")
        np.save(p, np.ascontiguousarray(labels))
        return p

    def run(self, jobs, should_stop=None, on_done=None, on_progress=None, budget_gb=None):
        """jobs: list of (function, task dict), in the order they should start.
        Returns results in the same order. on_progress(name, it, res, target)
        for every report of a worker.

        A job starts when a worker is free and, with a memory budget, when its
        task's "mem_gb" fits beside the jobs already running (the first job
        always starts); the next job that fits is taken, so light solves fill
        the room a heavy one leaves.

        A job that raises, or that a broken pool (a worker killed, e.g. out of
        memory) cannot run, comes back as {"failed": message, "name": ...}:
        the caller leaves it to the stage that solves it in turn."""
        for _, t in jobs:
            t["q"] = self.q if on_progress is not None else None
        out = [None] * len(jobs)
        queue = list(range(len(jobs)))
        running = {}
        held = [0.0]

        def drain():
            while self.q is not None and on_progress is not None:
                try:
                    item = self.q.get_nowait()
                except Exception:                                      # noqa: BLE001
                    break
                on_progress(*item)

        finished = [0]

        def fail(i, e):
            out[i] = {"failed": f"{type(e).__name__}: {e}", "name": jobs[i][1].get("name")}

        def report(i):
            if isinstance(out[i], dict) and "_units" in jobs[i][1]:
                out[i]["_units"] = jobs[i][1]["_units"]
            finished[0] += 1
            if on_done is not None:
                on_done(out[i], finished[0], len(jobs))

        def start():
            for i in list(queue):
                if len(running) >= self.workers:
                    break
                m = float(jobs[i][1].get("mem_gb") or 0.0)
                if running and budget_gb is not None and held[0] + m > budget_gb:
                    continue
                fn, t = jobs[i]
                queue.remove(i)
                try:
                    running[self.ex.submit(fn, t)] = (i, m)
                except Exception as e:                                 # noqa: BLE001
                    # the pool is broken: nothing more starts here
                    fail(i, e)
                    report(i)
                    continue
                held[0] += m
        start()
        while running:
            done, _ = wait(set(running), timeout=0.5, return_when=FIRST_COMPLETED)
            drain()
            for f in done:
                i, m = running.pop(f)
                held[0] -= m
                try:
                    out[i] = f.result()
                except Exception as e:                                 # noqa: BLE001
                    fail(i, e)
                report(i)
            if should_stop is not None and should_stop():
                self.kill()
                raise InterruptedError
            start()
        return out

    def kill(self):
        for p in list(getattr(self.ex, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:                                          # noqa: BLE001
                pass
        self.ex.shutdown(wait=False, cancel_futures=True)

    def close(self):
        try:
            self.ex.shutdown(wait=True, cancel_futures=True)
        finally:
            if getattr(self, "mgr", None) is not None:
                try:
                    self.mgr.shutdown()
                except Exception:                                      # noqa: BLE001
                    pass
            shutil.rmtree(self.tmp, ignore_errors=True)


# Speed-up of one FE solve on t threads, S(t) = 1 / ((1 - p) + p / t) with
# p = 0.85 fitted to one elastic load at 64^3 on the 14-core PC (measured
# 2.6x on 4 threads, 3.6x on 8, 4.3x on 14; the fit gives 2.8, 3.9, 4.7).
PAR_FRACTION = 0.85
# Relative cost of one solve (one thread): an elastic load case carries three
# displacement DOFs per node and a 24 x 24 element matrix - measured ~5x a
# conduction FE solve per voxel; the periodic FV solver with its FFT
# preconditioner is ~20x faster than the conduction FE solve (112^3: 1.4 s
# against 30.5 s on one thread).
# Measured in v5.1 on 80-100^3 RVEs with 8 threads: a PuMA Stokes direction
# ~30x a conduction FE solve and on one core whatever the threads (matrix-free
# MINRES in NumPy); a PuMA diffusion direction ~1x on about two cores; one EMI
# admittivity (one frequency, one direction) ~0.4 on about three; PuMA's ray
# casting on one core.
# The structure analyses, from the same runs: the size distributions of a
# phase ~1x (on most cores), percolation ~1.2x and porosimetry ~1.4x (on two
# or three), the pore network ~1x, the grain statistics ~3x (on most cores).
TASK_COST = {"elastic": 5.0, "fe": 1.0, "fv": 0.05, "stokes": 30.0, "tort": 1.0, "emi": 0.4,
             "radiation": 0.5, "morph": 1.0, "perc": 1.2, "poros": 1.4, "network": 1.0, "grains": 3.0}
# Fraction of each kind that runs in parallel on a worker's threads.
TASK_PAR = {"elastic": PAR_FRACTION, "fe": PAR_FRACTION, "fv": 0.6, "stokes": 0.1, "tort": 0.5, "emi": 0.6,
            "radiation": 0.05, "morph": 0.8, "perc": 0.5, "poros": 0.6, "network": 0.6, "grains": 0.85}


def speedup(t, p=PAR_FRACTION):
    return 1.0 / ((1.0 - p) + p / max(1, int(t)))


def makespan(costs, workers, threads, pars=None):
    """Longest-processing-time-first schedule of `costs` on `workers` equal
    workers of `threads` threads each; returns the finishing time. pars: the
    parallel fraction of each task (PAR_FRACTION when not given)."""
    load = [0.0] * workers
    pars = pars or [PAR_FRACTION] * len(costs)
    for c, p in sorted(zip(costs, pars), reverse=True):
        i = load.index(min(load))
        load[i] += c / speedup(threads, p)
    return max(load)


def plan_split(cores, costs, max_workers, pars=None):
    """(workers, threads each) that finishes `costs` soonest on `cores` cores
    with at most `max_workers` workers (the memory cap). Every split from one
    worker with all the cores to one thread per worker is scored with the
    longest-first schedule; a tie goes to fewer workers (less memory).
    Examples (7 elastic loads + 6 FV conduction solves): 4 cores -> 4 x 1,
    16 cores -> 8 x 2, 64 cores -> 7 x 9; with room for only 3 workers,
    4 cores -> 2 x 2."""
    cores = max(1, int(cores))
    best = None
    for w in range(1, max(1, min(cores, len(costs), int(max_workers))) + 1):
        t = cores // w
        m = makespan(costs, w, t, pars)
        if best is None or m < best[0] * (1 - 1e-9):
            best = (m, w, t)
    return best[1], best[2]


def plan_workers(threads, n_tasks, mem_per_solve_gb, overhead_gb=0.45):
    """How many solves to run at once: no more than the job's cores, the
    number of independent problems, and what fits in the free memory now
    (each worker also carries ~0.45 GB of Python, NumPy and PuMA)."""
    w = max(1, min(int(threads), int(n_tasks)))
    try:
        import psutil
        free = psutil.virtual_memory().available / 2 ** 30
        per = max(mem_per_solve_gb, 0.05) + overhead_gb
        w = max(1, min(w, int((0.85 * free) // per)))
    except Exception:                                                  # noqa: BLE001
        pass
    return w
