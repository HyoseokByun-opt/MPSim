"""Publication renders on demand, outside the web process.

VTK off-screen rendering wants one OpenGL context on one thread. The web
server has many threads, so renders go to a single spawned process that owns
VTK and serves requests one at a time. If it dies it is restarted on the next
request.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import uuid


def _serve(req_q, res_q):
    import traceback
    from mpsim import visual as V
    while True:
        item = req_q.get()
        if item is None:
            return
        rid, kind, view_dir, req, out_png = item
        try:
            if kind == "slice":
                V.slice_figure(view_dir, req, out_png)
            else:
                V.render_scene(view_dir, req, out_png)
            res_q.put((rid, True, out_png))
        except Exception:                                      # noqa: BLE001
            res_q.put((rid, False, traceback.format_exc()))


class RenderService:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.req_q = None
        self.res_q = None
        self.pending = {}

    def _ensure(self):
        if self.proc is not None and self.proc.is_alive():
            return
        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue()
        self.res_q = ctx.Queue()
        self.proc = ctx.Process(target=_serve, args=(self.req_q, self.res_q), daemon=True)
        self.proc.start()

    def render(self, kind, view_dir, req, out_png, timeout=180):
        with self.lock:                     # one render at a time, in order
            self._ensure()
            rid = uuid.uuid4().hex
            self.req_q.put((rid, kind, view_dir, req, out_png))
            t0 = time.time()
            while time.time() - t0 < timeout:
                try:
                    got, ok, payload = self.res_q.get(timeout=1.0)
                except queue.Empty:
                    if not self.proc.is_alive():
                        self.proc = None
                        raise RuntimeError("The render process terminated")
                    continue
                if got == rid:
                    if not ok:
                        raise RuntimeError(payload)
                    return payload
            raise TimeoutError("The render timed out")
