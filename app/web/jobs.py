"""Simulation jobs for the web server.

Three facts of a shared analysis server shape it:

* **the browser may leave.** All run state lives here and on disk; a
  reconnecting browser replays the event history from a sequence number.
* **a bad run must not take the server with it.** Every job runs in its own
  spawned child process; if it dies the job is marked failed and the server
  keeps serving.
* **the machine has one set of cores.** Jobs queue and run one at a time by
  default.

Finished runs are re-registered from disk at start-up, so a server restart
does not lose anyone's results.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import traceback
import uuid

MAX_EVENTS = 6000


def _child(spec, out_dir, out_q, stop_ev, threads):
    """Runs one job. Everything it says goes down `out_q`."""
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    def send(kind, **payload):
        try:
            out_q.put((kind, payload))
        except Exception:                                     # noqa: BLE001
            pass

    try:
        if spec.get("_kind") == "calibration":
            from mpsim import calibrate
            summary = calibrate.run(spec, out_dir,
                                    log=lambda *a: send("line", text=" ".join(str(x) for x in a)),
                                    event=lambda kind, **p: send(kind, **p),
                                    should_stop=stop_ev.is_set, limits=spec.get("_limits"))
        else:
            from mpsim import pipeline
            summary = pipeline.run(spec, out_dir,
                                   log=lambda *a: send("line", text=" ".join(str(x) for x in a)),
                                   event=lambda kind, **p: send(kind, **p),
                                   should_stop=stop_ev.is_set)
        send("finished", summary=summary)
    except InterruptedError:
        send("line", text="!! The run was stopped by the user.")
        send("finished", summary={"aborted": True})
    except Exception as e:                                    # noqa: BLE001
        send("line", text="!! The run terminated with an error:")
        for ln in traceback.format_exc().splitlines():
            send("line", text="   " + ln)
        send("finished", summary={"error": str(e)})


class Job:
    def __init__(self, jid, owner, spec, out_dir, meta, queue_=None, folder=None, case=""):
        self.id = jid
        self.owner = owner
        self.spec = spec
        self.out_dir = out_dir
        self.meta = meta
        self.queue = queue_
        # Which work folder this run belongs to, as an absolute path, and which
        # case inside it. Runs from every folder opened this session stay in the
        # queue - switching folders must not lose a running job - so the list is
        # filtered by these instead.
        self.folder = folder
        self.case = case or ""
        self.state = "queued"          # queued running done failed stopped
        self.created = time.time()
        self.started = None
        self.ended = None
        self.seq = 0
        self.events = []
        self.lock = threading.Lock()
        self.proc = None
        self.stop_ev = None
        self.summary = None
        self.stage = ""
        self.stage_detail = ""
        self.progress = 0.0
        self.solver = None             # {name, it, res, target, frac}
        self.history = {}              # solve name -> [(t, res)] in the current stage
        self.figures = []              # figures drawn while the run goes on
        self.partial = {}              # property -> headline numbers as they land
        self.lines = []

    def add(self, kind, payload):
        with self.lock:
            self.seq += 1
            self.events.append((self.seq, kind, payload))
            if len(self.events) > MAX_EVENTS:
                del self.events[:len(self.events) - MAX_EVENTS]
            return self.seq

    def since(self, after):
        with self.lock:
            return [e for e in self.events if e[0] > after]

    def snapshot(self, with_lines=True):
        pos = self.queue.position(self) if (self.queue and self.state == "queued") else 0
        return {
            "id": self.id, "state": self.state, "meta": self.meta,
            "queue_position": pos, "stage": self.stage, "stage_detail": self.stage_detail,
            "progress": self.progress, "solver": self.solver,
            "history": {k: v[-400:] for k, v in self.history.items()}, "partial": self.partial,
            "figures": self.figures,
            "summary": self.summary,
            "elapsed_s": ((self.ended or time.time()) - (self.started or time.time())) if self.started else 0.0,
            "created": self.created, "seq": self.seq,
            "lines": self.lines[-400:] if with_lines else [],
        }


def total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 2 ** 30
    except Exception:                                          # noqa: BLE001
        return 0.0


class JobQueue:
    def __init__(self, concurrency=1, threads=None, mem_budget_gb=None):
        self.jobs = {}
        self.order = []
        self.lock = threading.Lock()
        self.concurrency = max(1, int(concurrency))
        cores = os.cpu_count() or 2
        # remembered so that changing the number of parallel jobs later can keep
        # dividing the cores by itself, unless the user has pinned a count
        self.auto_threads = not threads
        self.threads = threads or self.default_threads(cores, self.concurrency)
        # Parallel runs share one machine's memory as well as its cores. Counting
        # jobs alone let three runs of ~4 GB each start together on a 15.5 GB
        # PC; it paged to disk and a two-minute solve took fifty. A run carries
        # the planner's estimate of its peak memory, and the next one waits while
        # the estimates of those running would exceed this budget. 70 % of the
        # RAM leaves room for the operating system and the browser.
        self.ram_gb = total_ram_gb()
        self.mem_budget_gb = mem_budget_gb if mem_budget_gb else round(0.7 * self.ram_gb, 1)
        self.estimator = None          # set by the server: form -> GB
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------- public
    @staticmethod
    def _dirname(meta):
        """What to call a run's directory.

        The date is already the work folder's name and the study is already the
        case's, so the one thing left worth reading off a run directory is the
        analysis it holds - which is what anyone scanning the case in a file
        manager is looking for. A DOE case adds its point number, because
        sixty-four runs of the same analysis otherwise differ only by a suffix.
        """
        an = meta.get("analyses") or []
        if isinstance(an, dict):
            an = [k for k, v in an.items() if v]
        base = "+".join(str(a) for a in an) or "run"
        if meta.get("doe"):
            # A DOE case is named for the values it varies ("content 50 - d 5"),
            # which is the only thing that tells two of its runs apart.
            label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(meta.get("name") or "")).strip("-.")
            base = "doe-" + (label or base)
        base = re.sub(r"[^A-Za-z0-9_+.-]+", "-", base).strip("-+.")[:60]
        return base or "run"

    def submit(self, owner, spec, out_dir, meta, folder=None, case="", into=None, replaces=None):
        """Start a run. `out_dir` is the case directory it is written into;
        `folder` is the work folder that case belongs to.

        `into` continues an existing run directory instead of making one (its
        structure and results are reused; see pipeline.run), and `replaces` is
        the id of the finished run whose place in the list the new one takes."""
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        base = self._dirname(meta)
        out = None
        if into:
            out = into
            # named after everything it now holds ("thermal" -> "thermal+dielectric")
            if os.path.basename(into) != base:
                parent = os.path.dirname(into)
                for i in range(1, 1000):
                    cand = os.path.join(parent, base if i == 1 else f"{base}-{i:02d}")
                    if os.path.exists(cand):
                        continue
                    try:
                        os.rename(into, cand)
                        out = cand
                    except OSError:
                        pass
                    break
        # Created rather than merely checked for, so two submits landing at the
        # same instant cannot pick the same name and share a directory.
        for i in range(1, 1000 if out is None else 0):
            cand = os.path.join(out_dir, base if i == 1 else f"{base}-{i:02d}")
            try:
                os.makedirs(cand, exist_ok=False)
                out = cand
                break
            except FileExistsError:
                continue
        if out is None:
            out = os.path.join(out_dir, f"{base}-{jid}")
            os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "spec.json"), "w", encoding="utf-8") as fh:
            json.dump(spec, fh, ensure_ascii=False, indent=1)
        with open(os.path.join(out, "job.json"), "w", encoding="utf-8") as fh:
            json.dump({"id": jid, "owner": owner, "meta": meta, "case": case,
                       "dir": os.path.basename(out), "created": time.time()},
                      fh, ensure_ascii=False)
        job = Job(jid, owner, spec, out, meta, self,
                  folder=os.path.abspath(folder or os.path.dirname(out_dir)), case=case)
        with self.lock:
            if replaces in self.jobs:
                self.jobs.pop(replaces)
                if replaces in self.order:
                    self.order.remove(replaces)
            self.jobs[jid] = job
            self.order.append(jid)
        return job

    def _restore_one(self, d, dirname, folder, case):
        try:
            with open(os.path.join(d, "job.json"), encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            return 0
        # The directory is named after the analysis and so is not unique across
        # cases; the id inside the file is. Older runs were named by their id,
        # and for those the two are the same.
        jid = info.get("id") or dirname
        if jid in self.jobs:
            return 0
        spec = {}
        try:
            with open(os.path.join(d, "spec.json"), encoding="utf-8") as fh:
                spec = json.load(fh)
        except (OSError, ValueError):
            pass
        job = Job(jid, info.get("owner"), spec, d, info.get("meta") or {}, self,
                  folder=folder, case=info.get("case", case))
        job.created = info.get("created", os.path.getmtime(d))
        res = os.path.join(d, "result.json")
        if os.path.exists(res):
            try:
                with open(res, encoding="utf-8") as fh:
                    job.summary = json.load(fh)
                job.state = "done" if not job.summary.get("error") else "failed"
                job.progress = 1.0
            except (OSError, ValueError):
                job.state = "failed"
                job.summary = {"error": "result.json could not be read"}
        elif not os.path.exists(os.path.join(d, "run.log")):
            # Never started: it was still waiting when the server stopped. It
            # goes back in the queue at its original place, so a restart does
            # not turn a DOE's unstarted cases into failures.
            job.state = "queued"
            with self.lock:
                self.jobs[jid] = job
                self.order.append(jid)
            return 1
        else:
            job.state = "failed"
            job.summary = {"error": "The run was interrupted by a server restart"}
        job.started = job.ended = job.created
        logp = os.path.join(d, "run.log")
        if os.path.exists(logp):
            with open(logp, encoding="utf-8", errors="replace") as fh:
                job.lines = fh.read().splitlines()[-400:]
        with self.lock:
            self.jobs[jid] = job
            self.order.append(jid)
        return 1

    def restore(self, runs_dir):
        """Register finished runs found in a work folder.

        A work folder holds case folders and a case folder holds runs, so the
        scan goes two levels deep. A run directory sitting straight in the work
        folder is still picked up - that is the layout of anything made before
        cases existed, and quietly losing sight of it would be worse than an
        untidy tree. Such a run simply has no case name.
        """
        if not os.path.isdir(runs_dir):
            return 0
        folder = os.path.abspath(runs_dir)
        n = 0
        for name in sorted(os.listdir(runs_dir)):
            d = os.path.join(runs_dir, name)
            if not os.path.isdir(d):
                continue
            if os.path.exists(os.path.join(d, "job.json")):
                n += self._restore_one(d, name, folder, "")
                continue
            try:
                subs = sorted(os.listdir(d))
            except OSError:
                continue
            for sub in subs:
                dd = os.path.join(d, sub)
                if os.path.isdir(dd) and os.path.exists(os.path.join(dd, "job.json")):
                    n += self._restore_one(dd, sub, folder, name)
        return n

    def get(self, jid):
        return self.jobs.get(jid)

    def for_owner(self, owner, limit=40, folder=None):
        """Runs to show in the list: this owner's, in this work folder.

        `folder` is the point of the work-folder model - the list is the
        contents of one folder, not everything the server has ever run. A job
        still in flight in another folder keeps running and is still reachable
        by id; it is simply not listed here.
        """
        want = os.path.abspath(folder) if folder else None
        with self.lock:
            # runs without an owner (command line, copied in) are visible to all
            mine = [j for j in self.jobs.values()
                    if (owner is None or j.owner in (owner, None))
                    and (want is None or j.folder == want)]
        mine.sort(key=lambda j: j.created, reverse=True)
        return mine[:limit]

    def running(self):
        return [j for j in self.jobs.values() if j.state == "running"]

    def position(self, job):
        with self.lock:
            waiting = sorted((self.jobs[i] for i in self.order if self.jobs[i].state == "queued"),
                             key=lambda j: j.created)
        return waiting.index(job) + 1 if job in waiting else 0

    def stop(self, job):
        if job.state == "queued":
            job.state = "stopped"
            job.ended = time.time()
            job.summary = {"aborted": True}
            job.add("finished", {"summary": job.summary})
            return True
        if job.state == "running" and job.stop_ev is not None:
            job.stop_ev.set()
            job.add("line", {"text": "A stop has been requested; the run stops after the current iteration."})
            return True
        return False

    @staticmethod
    def default_threads(cores, concurrency):
        """A quarter of the machine per run by default, and never more than an
        equal share when several runs go at once. Within a run the cores go to
        its independent problems first (mpsim.parallel) and what is left over
        to the threads of each solve: measured on a 14-core PC (guide, "Cores"),
        a PuMA FE solve scales to about 4x on 14 threads, but one thread per
        solve is still the most work per core."""
        return max(1, min(cores // 4, cores // max(1, concurrency)))

    def configure(self, concurrency=None, threads=None):
        """Change how many jobs run at once and how many cores each one gets.

        Safe while work is in flight. `_loop` re-reads `concurrency` on every
        pass, so raising it starts the next queued job within a third of a
        second, and lowering it never kills a running job - it only stops new
        ones from starting until the count has fallen. `threads` is read when a
        job starts, so it applies to the next job rather than to a running one.
        """
        cores = os.cpu_count() or 2
        with self.lock:
            if concurrency is not None:
                self.concurrency = max(1, min(int(concurrency), 64))
            if threads is not None:
                t = int(threads)
                # 0 hands the choice back to the server: cores split evenly
                self.auto_threads = t <= 0
                self.threads = self.default_threads(cores, self.concurrency) if t <= 0 else max(1, min(t, 4 * cores))
            elif concurrency is not None and self.auto_threads:
                self.threads = self.default_threads(cores, self.concurrency)
        return self.settings()

    def settings(self):
        run = self.running()
        return {"concurrency": self.concurrency, "threads": self.threads,
                "auto_threads": bool(self.auto_threads), "cores": os.cpu_count() or 2,
                "running": len(run),
                "queued": sum(1 for j in self.jobs.values() if j.state == "queued"),
                "ram_gb": round(self.ram_gb, 1), "mem_budget_gb": self.mem_budget_gb,
                "mem_reserved_gb": round(sum(self._expected(j) for j in run), 2)}

    # --------------------------------------------------------------- pump
    def _mem(self, job):
        """A run's peak-memory estimate. A queued run without one - submitted
        before estimates were recorded, or restored from disk - is estimated the
        first time it is a candidate, with the server's planner."""
        meta = job.meta if isinstance(job.meta, dict) else {}
        m = meta.get("mem_gb")
        if m is None and getattr(self, "estimator", None):
            try:
                m = float(self.estimator(job.spec))
            except Exception:                                  # noqa: BLE001
                m = 0.0
            meta["mem_gb"] = m
        try:
            return float(m or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # Measured on this study's solves: a running job's resident memory was 1.31-1.62
    # times the planner's estimate, plus a fixed ~0.3 GB for the interpreter,
    # NumPy and PuMA. The queue plans with what a run will really take.
    MEM_SCALE, MEM_BASE = 1.4, 0.3

    def _expected(self, job):
        m = self._mem(job)
        return self.MEM_SCALE * m + self.MEM_BASE if m else 0.0

    def _fits(self, job):
        """Whether `job` can start next to the ones already running.

        First in, first out: the oldest waiting run is the only candidate, so a
        large run is never overtaken indefinitely by small ones behind it. A run
        whose own estimate exceeds the budget still starts once the machine is
        otherwise idle - it was warned about at planning time, and refusing it
        outright would leave it queued forever.

        Two conditions, because each misses something. The estimates of the
        runs in flight must fit the budget - a run that has just started has not
        allocated yet, so free memory alone would let a second one in behind it.
        And the memory actually free must cover the new run - the budget cannot
        see what the rest of the machine (browser, editor) is using.
        """
        mine = self._expected(job)        # estimated now even when it will run alone,
        running = self.running()          # so that it counts once it is running
        if not running or not self.mem_budget_gb:
            return True
        if sum(self._expected(j) for j in running) + mine > self.mem_budget_gb:
            return False
        try:
            import psutil
            if psutil.virtual_memory().available / 2 ** 30 < mine + 1.0:
                return False
        except Exception:                                      # noqa: BLE001
            pass
        return True

    def _loop(self):
        while True:
            try:
                if len(self.running()) < self.concurrency:
                    with self.lock:
                        waiting = [self.jobs[i] for i in self.order if self.jobs[i].state == "queued"]
                    if waiting:
                        nxt = min(waiting, key=lambda j: j.created)
                        if self._fits(nxt):
                            self._start(nxt)
            except Exception:                                  # noqa: BLE001
                traceback.print_exc()
            time.sleep(0.3)

    def _start(self, job):
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        stop_ev = ctx.Event()
        proc = ctx.Process(target=_child, args=(job.spec, job.out_dir, out_q, stop_ev, self.threads),
                           daemon=False)
        job.proc = proc
        job.stop_ev = stop_ev
        job.state = "running"
        job.started = time.time()
        job.add("state", {"state": "running"})
        proc.start()
        threading.Thread(target=self._reader, args=(job, out_q), daemon=True).start()

    def _reader(self, job, out_q):
        logf = open(os.path.join(job.out_dir, "run.log"), "a", encoding="utf-8")
        try:
            while True:
                try:
                    kind, payload = out_q.get(timeout=1.0)
                except queue.Empty:
                    if job.proc is not None and not job.proc.is_alive():
                        break
                    continue
                except Exception:                              # noqa: BLE001
                    break
                if kind == "line":
                    logf.write(payload.get("text", "") + "\n")
                    logf.flush()
                self._apply(job, kind, payload)
                if kind == "finished":
                    break
        finally:
            logf.close()
        job.proc.join(timeout=15)
        if job.state == "running":
            job.state = "failed"
            job.summary = {"error": "The solver process terminated abnormally (possibly insufficient memory)"}
            job.add("finished", {"summary": job.summary})
        job.ended = time.time()

    @staticmethod
    def _apply(job, kind, payload):
        if kind == "line":
            job.lines.append(payload.get("text", ""))
            if len(job.lines) > 3000:
                del job.lines[:1000]
        elif kind == "stage":
            job.stage = payload.get("name", "")
            job.stage_detail = payload.get("detail", "")
            job.progress = float(payload.get("progress") if payload.get("progress") is not None else job.progress)
            job.history = {}
            job.solver = None
        elif kind == "solver":
            job.solver = payload
            # one series per solve: the six strains of an elastic run, or the
            # problems a run solves side by side, each keep their own curve
            ser = job.history.setdefault(payload.get("name") or "Residual", [])
            ser.append((round(time.time() - (job.started or time.time()), 2), payload.get("res")))
            if len(ser) > 800:
                del ser[:len(ser) - 600]
            if len(job.history) > 40:
                job.history.pop(next(iter(job.history)))
            if payload.get("progress") is not None:
                job.progress = float(payload["progress"])
        elif kind == "progress":
            # progress within a stage (a calibration's solves done so far);
            # unlike a new stage it keeps the convergence history
            if payload.get("progress") is not None:
                job.progress = float(payload["progress"])
            if payload.get("detail") is not None:
                job.stage_detail = payload["detail"]
            return
        elif kind == "partial":
            job.partial[payload.get("key")] = payload.get("value")
        elif kind == "figure":
            job.figures.append(payload)
        elif kind == "finished":
            job.summary = payload.get("summary") or {}
            if job.summary.get("error"):
                job.state = "failed"
            elif job.summary.get("aborted"):
                job.state = "stopped"
            else:
                job.state = "done"
                job.progress = 1.0
        job.add(kind, payload)
