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

    def run(self, jobs, should_stop=None, on_done=None, on_progress=None):
        """jobs: list of (function, task dict). Returns results in the same order.
        on_progress(name, it, res, target) for every report of a worker."""
        for _, t in jobs:
            t["q"] = self.q if on_progress is not None else None
        futs = {self.ex.submit(fn, t): i for i, (fn, t) in enumerate(jobs)}
        out = [None] * len(jobs)
        pending = set(futs)

        def drain():
            while self.q is not None and on_progress is not None:
                try:
                    item = self.q.get_nowait()
                except Exception:                                      # noqa: BLE001
                    break
                on_progress(*item)
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            drain()
            for f in done:
                out[futs[f]] = f.result()
                if on_done is not None:
                    on_done(out[futs[f]], len(jobs) - len(pending), len(jobs))
            if should_stop is not None and should_stop():
                self.kill()
                raise InterruptedError
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
TASK_COST = {"elastic": 5.0, "fe": 1.0, "fv": 0.05}


def speedup(t, p=PAR_FRACTION):
    return 1.0 / ((1.0 - p) + p / max(1, int(t)))


def makespan(costs, workers, threads):
    """Longest-processing-time-first schedule of `costs` on `workers` equal
    workers of `threads` threads each; returns the finishing time."""
    load = [0.0] * workers
    f = 1.0 / speedup(threads)
    for c in sorted(costs, reverse=True):
        i = load.index(min(load))
        load[i] += c * f
    return max(load)


def plan_split(cores, costs, max_workers):
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
        m = makespan(costs, w, t)
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
