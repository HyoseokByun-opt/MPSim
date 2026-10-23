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
import shutil
import threading
import time
import traceback
import uuid

MAX_EVENTS = 6000
# written into a run's directory when it is stopped (see JobQueue.stop)
STOPPED_FILE = "stopped.json"
TMP_DIR = "_tmp"                 # a run's scratch files (see _child)


def kill_tree(pid):
    """End a process and every process it started (pool workers, the
    particle dynamics' process, their managers)."""
    try:
        import psutil
        try:
            top = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        procs = top.children(recursive=True) + [top]
        for p in procs:
            try:
                p.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(procs, timeout=5)
        return
    except ImportError:
        pass
    if os.name == "nt":
        import subprocess
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _child(spec, out_dir, out_q, stop_ev, threads, gate=None):
    """Runs one job. Everything it says goes down `out_q`.

    `gate` (grant event, granted cores): the run builds its structure on
    `threads` cores, then waits before its first solve until the queue has
    the memory for it, and solves on the cores the queue grants - more than it
    started with when the other runs leave them free. Numba's pool is
    therefore sized for every core and limited to `threads` until then."""
    import warnings
    warnings.filterwarnings("ignore")
    cores = os.cpu_count() or int(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(max(int(threads), cores) if gate else threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ["MPSIM_JOB_CORES"] = str(threads)
    # scratch files of the run (the solve pool's shared grids) in its own
    # directory: a stop ends the process at once and removes them with it
    os.environ["MPSIM_TMP"] = os.path.join(out_dir, TMP_DIR)

    now = {"n": int(threads)}

    def send(kind, **payload):
        # the queue gives a run more or fewer cores as the others start and
        # finish (JobQueue._rebalance), while it builds its structure and
        # while it solves: taken up here, between two reports, outside any
        # parallel loop; the solve pool reads MPSIM_JOB_CORES twice a second
        if gate is not None:
            try:
                v = int(gate[1].value or 0)
                if v and v != now["n"]:
                    import numba
                    numba.set_num_threads(max(1, min(v, numba.config.NUMBA_NUM_THREADS)))
                    os.environ["MPSIM_JOB_CORES"] = str(v)
                    now["n"] = v
            except Exception:                                 # noqa: BLE001
                pass
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
            if gate is not None:
                grant_ev, grant_n = gate
                try:
                    import numba
                    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
                except Exception:                             # noqa: BLE001
                    pass

                def wait_gate():
                    """Before the first solve: wait until the queue grants the
                    memory, then take the cores it grants."""
                    send("gate")
                    t0 = time.time()
                    while not grant_ev.wait(0.3):
                        if stop_ev.is_set():
                            raise InterruptedError
                    n = int(grant_n.value or threads)
                    os.environ["MPSIM_JOB_CORES"] = str(n)
                    try:
                        import numba
                        numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))
                    except Exception:                         # noqa: BLE001
                        pass
                    try:
                        from threadpoolctl import threadpool_limits
                        threadpool_limits(limits=n)
                    except Exception:                         # noqa: BLE001
                        pass
                    return {"waited_s": time.time() - t0, "cores": n}
                pipeline.set_gate(wait_gate)
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
    finally:
        shutil.rmtree(os.path.join(out_dir, TMP_DIR), ignore_errors=True)


# what a restored run keeps of its result.json in memory
SUMMARY_KEYS = ("name", "headline", "check_counts", "elapsed_s", "error", "preview", "aborted", "analyses", "backend")


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
        self.stopping = False          # a stop was asked for: the end is "stopped", not "failed"
        self.threads = None            # the cores it was started with
        self.summary = None
        self.stage = ""
        self.stage_detail = ""
        self.progress = 0.0
        self.solver = None             # {name, it, res, target, frac}
        self.history = {}              # solve name -> [(t, res)] in the current stage
        self.figures = []              # figures drawn while the run goes on
        self.partial = {}              # property -> headline numbers as they land
        self.lines = []
        # a gated run (meta "gated"): "structure" while it builds its
        # structure, "gate" while it waits for the memory of its solves,
        # "solve" once granted. A run that is not gated reserves its peak
        # memory from the start ("solve").
        self.phase = None
        self.gate_t = None              # when it asked for its solve memory
        self.grant = None               # (grant event, granted cores) of its process

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
        wait = self.queue.wait_reason(self) if (self.queue and self.state == "queued") else ""
        return {
            "id": self.id, "state": self.state, "meta": self.meta,
            "queue_position": pos, "wait": wait, "threads": self.threads,
            "stage": self.stage, "stage_detail": self.stage_detail,
            "progress": self.progress, "solver": self.solver,
            "history": {k: v[-400:] for k, v in self.history.items()}, "partial": self.partial,
            "figures": self.figures,
            "summary": self.summary,
            "elapsed_s": ((self.ended or time.time()) - (self.started or time.time())) if self.started else 0.0,
            "created": self.created, "seq": self.seq, "phase": self.phase,
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
        self.mem_budget_auto = not mem_budget_gb
        self.estimator = None          # set by the server: form -> GB
        # also ask the PC how much memory is free (off only in the self-test)
        self.check_free = True
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
            # (the label alone: the full name starts with the base case's name and
            # would be cut at 60 characters before the values)
            label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(meta.get("doe_label") or meta.get("name") or "")).strip("-.")
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
        if not os.path.exists(res) and os.path.exists(os.path.join(d, STOPPED_FILE)):
            job.state = "stopped"
            job.summary = {"aborted": True}
        elif os.path.exists(res):
            try:
                with open(res, encoding="utf-8") as fh:
                    full = json.load(fh)
                # only what the run list and the reuse need: a whole workspace
                # of results would otherwise sit in memory
                job.summary = {k: full.get(k) for k in SUMMARY_KEYS if k in full}
                del full
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

    def for_owner(self, owner, limit=40, folder=None, folders=None, everyone=False):
        """Runs to show in the list: this owner's (every run, `everyone`), in
        this work folder (`folder`) or in any of `folders`.

        A job still in flight in another folder keeps running and is still
        reachable by id; it is simply not listed here.
        """
        want = os.path.abspath(folder) if folder else None
        among = {os.path.normcase(os.path.abspath(f)) for f in folders} if folders else None
        with self.lock:
            # runs without an owner (command line, copied in) are visible to all
            mine = [j for j in self.jobs.values()
                    if (everyone or owner is None or j.owner in (owner, None))
                    and (want is None or j.folder == want)
                    and (among is None or os.path.normcase(j.folder or "") in among)]
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
        """Stop a run now.

        A queued run is taken out of the queue. A running one is told to stop
        (the stop event, which the solvers read between their iterations) and,
        if its process has not ended within STOP_GRACE seconds, the process is
        ended with everything it started - the solve pool's workers, the
        particle dynamics' process. Waiting for the next check let a stop wait
        for a whole packing stage or a structure surface; a stop is meant to
        stop."""
        with self.lock:
            if job.state == "queued":
                job.state = "stopped"
                job.ended = time.time()
                job.summary = {"aborted": True}
                job.add("finished", {"summary": job.summary})
                self._mark_stopped(job)
                return True
        if job.state == "running" and job.stop_ev is not None:
            if job.stopping:
                return True
            job.stopping = True
            job.stop_ev.set()
            msg = "!! Stopped by the user."
            job.lines.append(msg)
            job.add("line", {"text": msg})
            try:
                with open(os.path.join(job.out_dir, "run.log"), "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass
            self._mark_stopped(job)
            threading.Thread(target=self._end, args=(job,), daemon=True).start()
            return True
        return False

    def stop_many(self, jobs):
        """Stop several runs (a DOE study): the queued ones first, so that none
        of them starts in the place a stopped one leaves."""
        jobs = list(jobs)
        n = 0
        for j in sorted(jobs, key=lambda j: j.state != "queued"):
            n += bool(self.stop(j))
        return n

    STOP_GRACE = 1.0

    @staticmethod
    def _mark_stopped(job):
        # read back by restore(): a stopped run stays "stopped" after a restart
        try:
            with open(os.path.join(job.out_dir, STOPPED_FILE), "w", encoding="utf-8") as fh:
                json.dump({"stopped": time.time()}, fh)
        except OSError:
            pass

    def _end(self, job):
        """End a stopped run's process and its children, after STOP_GRACE."""
        p = job.proc
        if p is None:
            return
        p.join(timeout=self.STOP_GRACE)
        if p.is_alive():
            kill_tree(p.pid)
            p.join(timeout=5)
        shutil.rmtree(os.path.join(job.out_dir, TMP_DIR), ignore_errors=True)

    @staticmethod
    def default_threads(cores, concurrency):
        """All of the machine's cores, shared equally by the runs set to go at
        once (5.5; before, a quarter of the PC per run left three quarters idle
        beside a single run). Within a run the cores go to its independent
        problems and the threads of each solve (mpsim.parallel), and the cores
        of finished solves to those still running."""
        return max(1, cores // max(1, concurrency))

    def configure(self, concurrency=None, threads=None, mem_budget_gb=None):
        """Change how many jobs run at once and how many cores each one gets.

        Safe while work is in flight. `_loop` re-reads `concurrency` on every
        pass, so raising it starts the next queued job within a third of a
        second, and lowering it never kills a running job - it only stops new
        ones from starting until the count has fallen. `threads` is read when a
        job starts, so it applies to the next job rather than to a running one.
        `mem_budget_gb`: the memory the runs in flight may reserve together; 0
        switches the check off, a negative value returns to 70 % of the RAM.
        """
        cores = os.cpu_count() or 2
        with self.lock:
            if mem_budget_gb is not None:
                v = float(mem_budget_gb)
                self.mem_budget_auto = v < 0
                self.mem_budget_gb = round(0.7 * self.ram_gb, 1) if v < 0 else max(0.0, v)
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
                "mem_budget_auto": bool(getattr(self, "mem_budget_auto", True)),
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

    def _expected(self, job, phase=None):
        """The memory a run is expected to hold: its structure's while it
        builds it or waits at the gate, its peak once it solves."""
        phase = phase or (job.phase if job.state == "running" else None) or self._first_phase(job)
        if phase in ("structure", "gate"):
            st = (job.meta or {}).get("mem_struct_gb") if isinstance(job.meta, dict) else None
            if st is not None:
                return self.MEM_SCALE * float(st) + self.MEM_BASE
        m = self._mem(job)
        return self.MEM_SCALE * m + self.MEM_BASE if m else 0.0

    @staticmethod
    def _first_phase(job):
        meta = job.meta if isinstance(job.meta, dict) else {}
        return "structure" if (meta.get("gated") and meta.get("mem_struct_gb") is not None
                               and not job.spec.get("preview") and job.spec.get("_kind") != "calibration") else "solve"

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
        mine = self._expected(job, self._first_phase(job))   # a gated run: its structure only
        running = self.running()          # estimated now even when it will run alone, so that it counts
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

    def gate_reason(self, job):
        """Why a run with its structure built waits before solving."""
        if job.state != "running" or job.phase != "gate":
            return ""
        need = self._expected(job, "solve")
        held = sum(self._expected(j) for j in self.running() if j is not job)
        free = None
        try:
            import psutil
            free = psutil.virtual_memory().available / 2 ** 30
        except Exception:                                      # noqa: BLE001
            pass
        return (f"structure built · waiting for memory to solve: needs about {need:.1f} GB, the other runs hold "
                f"{held:.1f} of the {self.mem_budget_gb:.1f} GB budget"
                + (f", {free:.1f} GB free on the PC" if free is not None else ""))

    def wait_reason(self, job):
        """Why a queued run has not started yet, for the run list: the first
        in line waits for a free slot or for memory, the others behind it."""
        with self.lock:
            waiting = sorted((self.jobs[i] for i in self.order if self.jobs[i].state == "queued"),
                             key=lambda j: j.created)
        if job not in waiting:
            return ""
        head = waiting[0]
        run = self.running()
        if len(run) >= self.concurrency:
            why = f"all {self.concurrency} places in use"
        else:
            mine = self._expected(head, self._first_phase(head))
            held = sum(self._expected(j) for j in run)
            why = ""
            if run and self.mem_budget_gb and held + mine > self.mem_budget_gb:
                why = (f"waiting for memory: needs about {mine:.1f} GB, the runs in progress hold "
                       f"{held:.1f} of the {self.mem_budget_gb:.1f} GB budget")
            elif run and self.mem_budget_gb:
                try:
                    import psutil
                    free = psutil.virtual_memory().available / 2 ** 30
                    if free < mine + 1.0:
                        why = f"waiting for memory: needs about {mine:.1f} GB, {free:.1f} GB are free on the PC"
                except Exception:                              # noqa: BLE001
                    pass
            why = why or "starting"
        pos = waiting.index(job) + 1
        return why if pos == 1 else f"{pos - 1} ahead · {why}"

    def _loop(self):
        while True:
            try:
                if len(self.running()) < self.concurrency:
                    # picked and started under the lock, so that a stop of the
                    # queued runs (stop_many) cannot be overtaken by a start
                    with self.lock:
                        waiting = [self.jobs[i] for i in self.order if self.jobs[i].state == "queued"]
                        if waiting:
                            nxt = min(waiting, key=lambda j: j.created)
                            if self._fits(nxt):
                                self._start(nxt)
                with self.lock:
                    self._open_gates()
                    self._rebalance()
            except Exception:                                  # noqa: BLE001
                traceback.print_exc()
            time.sleep(0.3)

    def _open_gates(self):
        """Let runs that have built their structure start solving, in the
        order they asked, while their solve memory fits the budget. A run
        whose solve does not fit yet lets later, lighter ones through only
        when it would still fit beside all of them once the solves that were
        under way before it asked have finished - so a heavy solve is never
        overtaken for good, and light ones use the room meanwhile (a heavy
        solve on one run while light ones solve on others)."""
        at_gate = sorted((j for j in self.jobs.values() if j.state == "running" and j.phase == "gate"),
                         key=lambda j: j.gate_t or 0.0)
        if not at_gate:
            return
        head = at_gate[0]
        head_need = self._expected(head, "solve")
        for j in at_gate:
            need = self._expected(j, "solve")
            others = [x for x in self.running() if x is not j]
            held = sum(self._expected(x) for x in others)
            solving = [x for x in others if x.phase == "solve"]
            fits = not self.mem_budget_gb or held + need <= self.mem_budget_gb
            if not fits and j is head and not solving:
                # nobody solves: the first in line goes, as a run alone always
                # starts (the others' structures are all that is held)
                fits = True
            if fits and self.mem_budget_gb and solving and self.check_free:
                try:
                    import psutil
                    # the structure's share of this run is already allocated
                    if psutil.virtual_memory().available / 2 ** 30 < need - self._expected(j, "structure") + 1.0:
                        fits = False
                except Exception:                              # noqa: BLE001
                    pass
            if fits and j is not head and self.mem_budget_gb:
                # backfill: the head must still fit beside every run that has
                # passed the gate after it asked, and this one
                after = sum(self._expected(x, "solve") for x in solving if (x.gate_t or 0) > (head.gate_t or 0))
                if head_need + after + need > self.mem_budget_gb:
                    fits = False
            if fits:
                self._grant(j)

    def _rebalance(self):
        """The cores of the gated runs that build their structure or solve:
        what the other runs leave, in equal shares (a run waiting at the gate
        uses none). A run takes its new share between two reports - its own
        threads and its solve pool's - so the cores of a finished run go to
        the others, and a run that starts solving beside one already solving
        does not get one core while the first keeps all of them."""
        run = self.running()
        active = [j for j in run if j.grant is not None and j.phase in ("structure", "solve")]
        if not active:
            return
        cores = os.cpu_count() or 2
        fixed = sum(int(j.threads or 0) for j in run if j.grant is None)
        avail = max(len(active), cores - fixed)
        q, r = divmod(avail, len(active))
        for k, j in enumerate(sorted(active, key=lambda x: x.started or 0.0)):
            share = max(1, q + (1 if k < r else 0))
            if int(j.threads or 0) != share:
                j.threads = share
                try:
                    j.grant[1].value = share
                except Exception:                              # noqa: BLE001
                    pass

    def _grant(self, job):
        """Open the gate: the run solves from now on, on its equal share of
        the cores (_rebalance)."""
        job.phase = "solve"
        self._rebalance()
        if job.grant is not None:
            ev, val = job.grant
            if not val.value:
                val.value = int(job.threads or 1)
            ev.set()

    def _threads_for(self, job):
        """The cores a starting run gets. Automatic: the machine's share for
        `concurrency` runs at once - but when memory lets fewer than that run
        side by side (an Accurate grid of 470^3 reserves ~35 GB each), the
        cores those runs cannot use go to the ones that do run, instead of
        three quarters of the PC waiting idle beside a single run."""
        if not self.auto_threads:
            return self.threads
        cores = os.cpu_count() or 2
        side = self.side_by_side(job)
        # fewer runs than the set number at once - by memory, or because no
        # more are queued: those runs share the cores the others would have
        # had, as far as the runs already going leave them free (never below
        # the equal share)
        waiting = sum(1 for j in self.jobs.values() if j.state == "queued")
        going = [j for j in self.jobs.values() if j.state == "running" and j is not job]
        n = max(1, min(side, self.concurrency, len(going) + max(1, waiting)))
        free = cores - sum(int(j.threads or 0) for j in going)
        return max(self.threads if side >= self.concurrency else 1,
                   min(cores // n, max(free, 1)))

    def side_by_side(self, job):
        """How many runs the size of `job` the memory budget holds at once
        (at most `concurrency`)."""
        mine = self._expected(job)
        if not self.mem_budget_gb or mine <= 0:
            return self.concurrency
        return max(1, min(self.concurrency, int(self.mem_budget_gb // mine)))

    def _start(self, job):
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        stop_ev = ctx.Event()
        job.threads = self._threads_for(job)
        job.phase = self._first_phase(job)
        gate = None
        if job.phase == "structure":
            gate = (ctx.Event(), ctx.Value("i", 0))
            job.grant = gate
        proc = ctx.Process(target=_child, args=(job.spec, job.out_dir, out_q, stop_ev, job.threads, gate),
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
            if job.stopping:
                job.state = "stopped"
                job.summary = {"aborted": True}
            else:
                job.state = "failed"
                job.summary = {"error": "The solver process terminated abnormally (possibly insufficient memory)"}
            job.add("finished", {"summary": job.summary})
        job.ended = time.time()

    @staticmethod
    def _apply(job, kind, payload):
        if kind == "gate":
            # the structure is built: the run waits for the memory of its solves
            job.phase = "gate"
            job.gate_t = time.time()
            return
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
