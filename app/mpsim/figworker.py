"""Figures rendered beside the solves.

A run used to render its report figures only after the last analysis, and on
a 500-cubed RVE that took longer than some of the analyses. The figures of
each analysis are now queued as soon as it finishes and drawn by a separate
process while the next analysis solves; a listener thread reports each one
the moment it is written, so the browser shows it then. The end of the run
waits for the queue and renders only what is still missing, so the total
time is not longer than before. Since 5.5.1 two or three processes draw from
the one queue (as the free memory allows), so the figures still due when the
last analysis ends take a half or a third of the time.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time


def _loop(q_in, q_out, threads):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = str(threads)
    from . import visual as V
    while True:
        t = q_in.get()
        if t is None:
            break
        name, kind, view_dir, req, out = t
        t0 = time.time()
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            (V.slice_figure if kind == "slice" else V.render_scene)(view_dir, req, out)
            q_out.put((name, True, None, time.time() - t0))
        except Exception as e:                                       # noqa: BLE001
            q_out.put((name, False, f"{type(e).__name__}: {e}", time.time() - t0))


def _n_procs():
    """Figure processes: up to three, a core of the run each beyond the
    solves, and about 1.5 GB of free memory each (a large scene)."""
    try:
        cores = int(os.environ.get("NUMBA_NUM_THREADS") or 0) or (os.cpu_count() or 1)
    except ValueError:
        cores = os.cpu_count() or 1
    try:
        import psutil
        free = psutil.virtual_memory().available
    except Exception:                                                # noqa: BLE001
        free = 4e9
    return max(1, min(3, cores // 4, int(free // 1.5e9)))


class FigureWorker:
    """Background processes that draw the figures as they come (from one
    queue); on_done(name, ok, error, seconds, title, group) is called from a
    listener thread as each one finishes."""

    def __init__(self, on_done=None, threads=2, procs=None):
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out = ctx.Queue(), ctx.Queue()
        n = procs or _n_procs()
        self.procs = [ctx.Process(target=_loop, args=(self.q_in, self.q_out, threads), daemon=True)
                      for _ in range(n)]
        for p in self.procs:
            p.start()
        self.proc = self.procs[0]
        self.queued = {}            # name -> (title, group)
        self.done = {}              # name -> (ok, error, seconds)
        self.lock = threading.Lock()
        self.on_done = on_done
        self._stop = False
        self.listener = threading.Thread(target=self._listen, daemon=True)
        self.listener.start()

    def _listen(self):
        while not self._stop:
            try:
                name, ok, err, sec = self.q_out.get(timeout=0.5)
            except queue.Empty:
                if not self.alive():
                    break
                continue
            except (EOFError, OSError):
                break
            with self.lock:
                self.done[name] = (ok, err, sec)
                title, group = self.queued.get(name, ("", ""))
            if self.on_done is not None:
                try:
                    self.on_done(name, ok, err, sec, title, group)
                except Exception:                                    # noqa: BLE001
                    pass

    def submit(self, name, kind, view_dir, req, out, title="", group=""):
        with self.lock:
            if name in self.queued:
                return False
            self.queued[name] = (title, group)
        self.q_in.put((name, kind, view_dir, req, out))
        return True

    def alive(self):
        return any(p.is_alive() for p in self.procs)

    def pending(self):
        with self.lock:
            return [n for n in self.queued if n not in self.done]

    def finish(self, timeout=1800.0, should_stop=None):
        """Waits for every queued figure (or the timeout), then stops the
        process. Returns {name: (ok, error, seconds)}."""
        t0 = time.time()
        while self.pending() and time.time() - t0 < timeout and self.alive():
            if should_stop is not None and should_stop():
                break
            time.sleep(0.25)
        self.close()
        with self.lock:
            return dict(self.done)

    def close(self):
        self._stop = True
        try:
            for _ in self.procs:
                self.q_in.put(None)
            for p in self.procs:
                p.join(timeout=10)
        except Exception:                                            # noqa: BLE001
            pass
        for p in self.procs:
            if p.is_alive():
                p.terminate()
