"""The web front-end: the engine over HTTP.

It drives `mpsim/` exactly as the command line does - the engine has no idea
which front-end is asking - so a browser and a script cannot disagree about a
result.

Design notes:

* **Owners.** Every browser gets an id in a signed cookie; its run list shows
  its own jobs. A result link can still be opened by anyone who has it, which
  is how a result is shared with a colleague.
* **The browser is optional.** Run state lives in `jobs.py` and on disk.
* **No CDN.** vtk.js and all CSS/JS ship with the page; an analysis server
  often sits on an isolated network.
* **Access.** `--token` is a door lock for a lab network, not authentication.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import secrets
import shutil
import time
import zipfile
from importlib import metadata

from flask import (Flask, abort, jsonify, render_template, request, send_file,
                   send_from_directory, session, Response)
from flask.json.provider import DefaultJSONProvider

from mpsim import __version__
from mpsim import materials as M
from mpsim import rve as R
from mpsim import spec as S

from .doe import DOEStore
from .jobs import JobQueue
from .render_worker import RenderService

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PRESETS = os.path.join(ROOT, "mpsim", "data", "presets.json")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _clean(o.item())
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
    except Exception:                                          # noqa: BLE001
        pass
    return o


class SafeJSON(DefaultJSONProvider):
    """NaN is not JSON; a browser's JSON.parse rejects the whole response."""
    ensure_ascii = False
    sort_keys = False

    def dumps(self, obj, **kw):
        kw.setdefault("ensure_ascii", False)
        return json.dumps(_clean(obj), **kw)


def versions():
    """Package versions without importing the heavy packages into the web
    process. conda-forge's NASA PuMA is not registered under a pip name, so
    its version is read from pumapy/version.py."""
    import importlib.util
    import re
    out = {"mpsim": __version__}
    for dist in ("puma", "porespy", "pyvista", "vtk", "scikit-rf", "numba", "numpy", "scipy", "flask"):
        try:
            out[dist] = metadata.version(dist)
        except Exception:                                      # noqa: BLE001
            out[dist] = None
    if out["puma"] is None:
        try:
            import glob
            import sys
            hits = glob.glob(os.path.join(sys.prefix, "conda-meta", "puma-[0-9]*.json"))
            if hits:
                m = re.match(r"puma-([0-9][^-]*)-", os.path.basename(hits[0]))
                out["puma"] = m.group(1) if m else "installed"
            elif importlib.util.find_spec("pumapy") is not None:
                out["puma"] = "installed"
        except Exception:                                      # noqa: BLE001
            pass
    return out


# The workspace the program was last pointed at, kept beside the program so a
# restart (or 2_RUN_LOCAL.bat) opens it again, as a simulation project keeps
# its work directory.
SETTINGS = os.path.join(ROOT, "..", "settings.json")
DEFAULT_WORKSPACE = os.path.abspath(os.path.join(ROOT, "..", "workspace"))


def load_settings():
    try:
        with open(SETTINGS, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(d):
    try:
        with open(SETTINGS, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _persistent_secret(workspace):
    """Session secret kept in the workspace. A new random secret at every start
    would invalidate every browser's owner cookie, and users would lose their
    run lists whenever the server restarts."""
    os.makedirs(workspace, exist_ok=True)
    path = os.path.join(workspace, ".session_secret")
    try:
        with open(path, encoding="ascii") as fh:
            s = fh.read().strip()
        if len(s) >= 32:
            return s
    except OSError:
        pass
    s = secrets.token_hex(32)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(s)
    return s


def needed_analyses(spec):
    return S.required_analyses(spec["analyses"])


def create_app(workspace=None, token=None, concurrency=1, limits=None):
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"),
                template_folder=os.path.join(HERE, "templates"))
    app.json = SafeJSON(app)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    # an updated page template takes effect without restarting the server,
    # like the static scripts it references
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    remembered = load_settings().get("workspace")
    if not workspace and remembered and os.path.isdir(remembered):
        workspace = remembered
    app.config["WORKSPACE"] = os.path.abspath(workspace or DEFAULT_WORKSPACE)
    os.makedirs(app.config["WORKSPACE"], exist_ok=True)
    app.config["SECRET_KEY"] = os.environ.get("MPSIM_SECRET") or _persistent_secret(app.config["WORKSPACE"])
    app.config["TOKEN"] = token
    lim = dict(R.DEFAULT_LIMITS)
    lim.update(limits or {})
    app.config["LIMITS"] = lim
    app.queue = JobQueue(concurrency=concurrency)
    app.doe = DOEStore(app.config["WORKSPACE"])
    app.render = RenderService()
    app.versions = versions()

    # ------------------------------------------------------- work folders
    # A run belongs to a work folder chosen before it starts, instead of
    # landing in one shared pile. A folder is an ordinary directory, so it
    # sorts by name in the file manager - which is why a new one is named for
    # today by default - and a finished study can be zipped or moved whole.
    def wf_state():
        return os.path.join(app.config["WORKSPACE"], "workfolders.json")

    def wf_load():
        try:
            with open(wf_state(), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = {}
        return {"current": d.get("current") or "",
                "recent": [p for p in (d.get("recent") or []) if isinstance(p, str)]}

    def wf_save(state):
        try:
            with open(wf_state(), "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=1)
        except OSError:
            pass                                   # a read-only workspace still runs

    def wf_resolve(path):
        """A bare name becomes a directory in the workspace; an absolute path is
        taken as given, so a large study can live on another disk."""
        p = (path or "").strip().strip('"')
        if not p:
            raise ValueError("The work folder is missing")
        if not os.path.isabs(p):
            bad = set(p) & set('\\/:*?"<>|')
            if bad:
                raise ValueError("A folder name must not contain " + " ".join(sorted(bad)))
            p = os.path.join(app.config["WORKSPACE"], p)
        return os.path.abspath(p)

    def wf_count(p):
        """Runs in a work folder, counting both layouts: runs inside case
        folders, and any bare run left directly in the folder."""
        n = 0
        try:
            entries = os.listdir(p)
        except OSError:
            return 0
        for d in entries:
            full = os.path.join(p, d)
            if not os.path.isdir(full):
                continue
            if os.path.exists(os.path.join(full, "job.json")):
                n += 1
                continue
            try:
                n += sum(1 for s in os.listdir(full)
                         if os.path.exists(os.path.join(full, s, "job.json")))
            except OSError:
                pass
        return n

    def wf_use(path, create=True):
        p = wf_resolve(path)
        if not os.path.isdir(p):
            if not create:
                raise ValueError(f"There is no work folder at {p}")
            os.makedirs(p, exist_ok=True)
        app.config["RUNS_DIR"] = p
        # runs already in the folder join the list; ones from other folders stay
        # in the queue untouched, so switching never disturbs work in flight
        n = app.queue.restore(p)
        # Open the newest case in the folder, so reopening a folder lands where
        # the work was left rather than on an empty one. The directory is read
        # directly rather than through case_list(), which reports the currently
        # open case even when it belongs to the folder being left behind.
        app.config["CASE"] = ""
        found = []
        try:
            for name in os.listdir(p):
                d = os.path.join(p, name)
                if os.path.isdir(d) and not os.path.exists(os.path.join(d, "job.json")):
                    found.append((os.path.getmtime(d), name))
        except OSError:
            pass
        app.config["CASE"] = max(found)[1] if found else "case-01"
        os.makedirs(os.path.join(p, app.config["CASE"]), exist_ok=True)
        st = wf_load()
        st["current"] = p
        st["recent"] = [p] + [q for q in st["recent"]
                              if os.path.normcase(q) != os.path.normcase(p)]
        st["recent"] = st["recent"][:12]
        wf_save(st)
        return p, n

    # ------------------------------------------------------------- cases
    # Inside a work folder, each case is its own directory and every run of
    # that case accumulates under it. One folder is therefore a session's whole
    # body of work, already sorted: cases side by side, runs in time order
    # within each.
    def case_resolve(name):
        n = (name or "").strip().strip('"')
        if not n:
            raise ValueError("The case name is missing")
        bad = set(n) & set('\\/:*?"<>|')
        if bad:
            raise ValueError("A case name must not contain " + " ".join(sorted(bad)))
        if n in (".", ".."):
            raise ValueError("That is not a usable case name")
        return n[:80]

    def case_dir(name=None):
        c = case_resolve(name if name is not None else app.config.get("CASE") or "")
        return os.path.join(app.config["RUNS_DIR"], c)

    def case_list():
        root = app.config["RUNS_DIR"]
        out = []
        try:
            names = sorted(os.listdir(root))
        except OSError:
            names = []
        for n in names:
            d = os.path.join(root, n)
            if not os.path.isdir(d) or os.path.exists(os.path.join(d, "job.json")):
                continue                      # a bare run, not a case
            try:
                runs = sum(1 for s in os.listdir(d)
                           if os.path.exists(os.path.join(d, s, "job.json")))
            except OSError:
                runs = 0
            out.append({"name": n, "runs": runs, "current": n == app.config.get("CASE")})
        if app.config.get("CASE") and not any(c["name"] == app.config["CASE"] for c in out):
            out.insert(0, {"name": app.config["CASE"], "runs": 0, "current": True})
        return out

    def case_use(name, create=True):
        c = case_resolve(name)
        d = os.path.join(app.config["RUNS_DIR"], c)
        if not os.path.isdir(d):
            if not create:
                raise ValueError(f"There is no case called {c} in this work folder")
            os.makedirs(d, exist_ok=True)
        app.config["CASE"] = c
        return c

    def case_info():
        return {"current": app.config.get("CASE") or "", "cases": case_list(),
                "workfolder": app.config["RUNS_DIR"],
                "dir": os.path.join(app.config["RUNS_DIR"], app.config.get("CASE") or "")}

    def wf_info():
        cur = app.config["RUNS_DIR"]
        seen, rows = set(), []

        def add(p):
            key = os.path.normcase(os.path.abspath(p))
            if key in seen or not os.path.isdir(p):
                return
            seen.add(key)
            rows.append({"path": os.path.abspath(p), "name": os.path.basename(p.rstrip("\\/")) or p,
                         "runs": wf_count(p),
                         "current": key == os.path.normcase(cur)})
        add(cur)
        for p in wf_load()["recent"]:
            add(p)
        # anything else already sitting in the workspace, so a folder made by
        # hand in Explorer shows up without having to be typed in
        try:
            for d in sorted(os.listdir(app.config["WORKSPACE"])):
                full = os.path.join(app.config["WORKSPACE"], d)
                if os.path.isdir(full) and wf_count(full):
                    add(full)
        except OSError:
            pass
        rows.sort(key=lambda r: (not r["current"], r["name"].lower()))
        return {"current": cur, "name": os.path.basename(cur), "runs": wf_count(cur),
                "workspace": app.config["WORKSPACE"], "folders": rows}

    # ----------------------------------------------------------- workspace
    def ws_use(path, create=True):
        """Point the program at another workspace: the root every work folder
        given by name is made in, and where the DOE studies and the list of
        work folders are kept. Runs in flight finish where they started. The
        work folder opened is the one last used in that workspace (today's
        folder for a new one). Remembered across restarts."""
        p = (path or "").strip().strip('"')
        if not p:
            raise ValueError("The workspace path is missing")
        if not os.path.isabs(p):
            raise ValueError("Give the workspace as a full path, e.g. D:\\SimData\\MPSim")
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            if not create:
                raise ValueError(f"There is no folder at {p}")
            os.makedirs(p, exist_ok=True)
        probe = os.path.join(p, ".mpsim_write_test")
        try:
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
        except OSError as e:
            raise ValueError(f"The workspace {p} cannot be written to ({e})")
        app.config["WORKSPACE"] = p
        app.doe = DOEStore(p)
        st = wf_load()
        try:
            out = wf_use(st["current"] or time.strftime("%Y-%m-%d"))
        except (ValueError, OSError):
            out = wf_use(time.strftime("%Y-%m-%d"))
        s = load_settings()
        s["workspace"] = p
        s["recent_workspaces"] = ([p] + [q for q in s.get("recent_workspaces") or []
                                         if os.path.normcase(q) != os.path.normcase(p)])[:10]
        save_settings(s)
        return out

    def ws_info():
        s = load_settings()
        rec = [q for q in s.get("recent_workspaces") or [] if os.path.isdir(q)]
        cur = app.config["WORKSPACE"]
        if not any(os.path.normcase(q) == os.path.normcase(cur) for q in rec):
            rec.insert(0, cur)
        if not any(os.path.normcase(q) == os.path.normcase(DEFAULT_WORKSPACE) for q in rec):
            rec.append(DEFAULT_WORKSPACE)
        return {"workspace": cur, "default": DEFAULT_WORKSPACE, "recent": rec}

    _st = wf_load()
    try:
        runs_dir, restored = wf_use(_st["current"] or time.strftime("%Y-%m-%d"))
    except (ValueError, OSError):
        runs_dir, restored = wf_use(time.strftime("%Y-%m-%d"))
    print(f"  workspace:   {app.config['WORKSPACE']}")
    print(f"  work folder: {runs_dir} ({restored} runs restored)")

    # ------------------------------------------------------------ helpers
    def owner():
        sid = session.get("sid")
        if not sid:
            sid = secrets.token_hex(8)
            session["sid"] = sid
            session.permanent = True
        return sid

    def owner_or_none():
        """The established session, or None for a client that keeps no cookie.

        A browser stores the signed cookie and keeps one identity. A script, the
        command line or curl stores nothing, so `owner()` would mint a fresh
        throwaway id on every single request and stamp it on the run. Nothing can
        ever match that id again, and the run becomes invisible in the interface
        for good - which is what happened to a whole sweep of command-line runs.
        The queue already treats an ownerless run as visible to everyone, and
        that is what a run submitted outside a browser should be.
        """
        return session.get("sid")

    def job_or_404(jid):
        job = app.queue.get(jid)
        if job is None:
            abort(404, "Run not found")
        return job

    def safe_path(base, name):
        p = os.path.abspath(os.path.join(base, name))
        if not p.startswith(os.path.abspath(base) + os.sep):
            abort(400)
        return p

    # ------------------------------------------------------------- guard
    @app.before_request
    def _guard():
        tok = app.config["TOKEN"]
        if not tok or request.endpoint in ("static", "healthz"):
            return None
        given = request.args.get("token") or request.headers.get("X-Token")
        if given == tok:
            session["auth"] = tok
        if session.get("auth") != tok:
            return Response("<h3>An access token is required</h3><form><input name='token' placeholder='token'>"
                            "<button>Open</button></form>", status=401, mimetype="text/html")
        return None

    # -------------------------------------------------------------- pages
    def asset_version():
        """The program version and the newest change to a page file, so a
        browser never keeps an old script after an update of the same version."""
        sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        try:
            t = max(os.path.getmtime(os.path.join(sd, f)) for f in os.listdir(sd) if f.endswith((".js", ".css")))
        except (OSError, ValueError):
            t = 0
        return f"{__version__}.{int(t)}"

    @app.get("/")
    def index():
        owner()
        return render_template("index.html", version=asset_version())

    @app.get("/healthz")
    def healthz():
        return "ok"

    @app.get("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.get("/api/health")
    def api_health():
        return jsonify({"version": __version__, "versions": app.versions,
                        "cores": os.cpu_count(), "concurrency": app.queue.concurrency,
                        "limits": app.config["LIMITS"],
                        "queued": sum(1 for j in app.queue.jobs.values() if j.state == "queued"),
                        "running": len(app.queue.running())})

    @app.get("/api/materials")
    def api_materials():
        return jsonify(M.load_library())

    @app.get("/api/presets")
    def api_presets():
        with open(PRESETS, encoding="utf-8") as fh:
            return jsonify(json.load(fh))

    @app.get("/api/analyses")
    def api_analyses():
        return jsonify(S.ANALYSES)

    def run_tags(spec, plan=None):
        """What a run is, for the run list: structure only or which analyses,
        the phases in one line, and the grid."""
        def size(ph):
            d = (ph.get("size_um") or {}).get("d")
            if d is None:
                return ""
            return f" {d * 1000:.0f} nm" if d < 1 else f" {d:.3g} µm"
        phases = " + ".join(f"{ph.get('name', '')} {ph.get('shape', '')}{size(ph)} {100 * float(ph.get('vf', 0)):.3g} %"
                            for ph in spec.get("phases") or [])
        mat = (spec.get("matrix") or {}).get("name", "")
        tags = {"kind": "structure" if (spec.get("preview") or not spec.get("analyses")) else "analysis",
                "structure": (f"{phases} in {mat}" if phases else mat).strip()}
        if plan:
            n, nz = plan.get("N"), plan.get("Nz") if plan.get("film") else None
            if n:
                tags["grid"] = f"{n}³" if not nz else f"{n}×{n}×{nz}"
        return tags

    def est_mem_gb(form):
        """The planner's peak-memory estimate for a run, kept with the job so
        the queue can hold parallel runs inside the machine's RAM."""
        try:
            spec = S.normalize(copy.deepcopy(form))
            from mpsim.pipeline import _plan_phases
            plan = R.plan(_plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"]), needed_analyses(spec), app.config["LIMITS"])
            est_mem_gb.last_plan = plan
            return round(max((plan.get("memory_gb") or {"-": 0.0}).values()), 3)
        except Exception:                                   # noqa: BLE001
            est_mem_gb.last_plan = None
            return 0.0

    app.queue.estimator = est_mem_gb

    @app.post("/api/plan")
    def api_plan():
        form = copy.deepcopy(request.get_json(force=True, silent=True) or {})
        an = form.get("analyses") or {}
        if not any(an.values()):
            form["analyses"] = {"thermal": True}
        try:
            spec = S.normalize(form)
        except ValueError as e:
            return jsonify({"ok": False, "errors": str(e).split("\n")})
        from mpsim.pipeline import _plan_phases
        plan = R.plan(_plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"]), needed_analyses(spec), app.config["LIMITS"])
        return jsonify({"ok": True, "plan": plan, "composition": spec.get("composition"),
                        "labels": [{"name": t["name"], "kind": t["kind"]} for t in spec["labels"]],
                        "phases": [{"name": p["name"], "vf": p["vf"]} for p in spec["phases"]]})

    # --------------------------------------------------------------- jobs
    @app.get("/api/settings")
    def api_settings_get():
        return jsonify(app.queue.settings())

    @app.post("/api/settings")
    def api_settings_set():
        """How many analyses run at once, and how many cores each one uses.

        The right answer depends on the machine and on the work: many small
        cases finish sooner side by side, one large case wants every core. Only
        the person at the keyboard knows which it is, so it is set here rather
        than fixed at start-up.
        """
        b = request.get_json(force=True, silent=True) or {}
        return jsonify(app.queue.configure(concurrency=b.get("concurrency"),
                                           threads=b.get("threads")))

    def find_reuse(spec, prefer=None):
        """The finished run of this case whose structure a new submission can
        use, and whether the submission can simply continue it.

        A run keeps the structure it generated (pipeline.save_structure). The
        same geometry, grid, contact handling and seed make the same key; the
        run then continues when nothing else differs either (the same
        materials and options, more analyses) or when it holds no analysis yet
        (a "Generate structure" run). Otherwise a new run is made that reuses
        the structure. The run the user has open is preferred."""
        if not spec["options"].get("reuse_runs", True):
            return None, False
        from mpsim import pipeline as P
        try:
            need = S.required_analyses(set(spec["analyses"]))
            plan = R.plan(P._plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"]), need,
                          app.config["LIMITS"])
            plan["skin"] = P.film_skin(spec, plan) if plan.get("film") else None
            key = P.structure_key(P._generator_phases(spec), plan, spec["options"]["contacts"],
                                  spec["rve"]["seed"], P.wants_contact_faces(spec))
        except Exception:                                            # noqa: BLE001
            return None, False
        case = app.config.get("CASE") or ""
        cands = [j for j in app.queue.for_owner(owner_or_none(), limit=100000, folder=app.config["RUNS_DIR"])
                 if j.case == case and j.state == "done" and not (j.meta or {}).get("doe")
                 and os.path.exists(os.path.join(j.out_dir, "structure", key + ".npz"))]
        if not cands:
            return None, False
        cands.sort(key=lambda j: (j.id != prefer, -j.created))
        sig = P.run_signature(spec)
        for j in cands:
            summ = j.summary or {}
            if summ.get("preview") or not (j.meta or {}).get("analyses"):
                return j, True
            # the inputs as that run normalised them (what pipeline.run
            # compares against): after a version that adds options, a run no
            # longer matches and a new one is made, keeping the old results
            try:
                with open(os.path.join(j.out_dir, "spec.normalized.json"), encoding="utf-8") as fh:
                    old = json.load(fh)
            except (OSError, ValueError):
                continue
            if P.run_signature(old) == sig:
                return j, True
        return cands[0], False

    @app.post("/api/jobs")
    def api_submit():
        form = request.get_json(force=True, silent=True) or {}
        prefer = form.pop("_continue", None)
        try:
            spec = S.normalize(copy.deepcopy(form))
        except ValueError as e:
            return jsonify({"ok": False, "errors": str(e).split("\n")}), 400
        form["_limits"] = app.config["LIMITS"]
        meta = {"name": spec["name"], "analyses": spec["analyses"], "mem_gb": est_mem_gb(form)}
        meta.update(run_tags(spec, getattr(est_mem_gb, "last_plan", None)))
        src, append = find_reuse(spec, prefer)
        info = {}
        if src is not None and append:
            old = [a for a in ((src.meta or {}).get("analyses") or [])]
            meta["analyses"] = sorted(set(old) | set(spec["analyses"]))
            meta["kind"] = "analysis" if meta["analyses"] else "structure"
            form["_reuse"] = {"append": True}
            job = app.queue.submit(owner_or_none(), form, case_dir(), meta, folder=app.config["RUNS_DIR"],
                                   case=app.config.get("CASE") or "", into=src.out_dir, replaces=src.id)
            info = {"continued": src.id, "kept": old}
        else:
            if src is not None:
                form["_reuse"] = {"from": [src.out_dir]}
                info = {"structure_from": src.id}
            job = app.queue.submit(owner_or_none(), form, case_dir(), meta,
                                   folder=app.config["RUNS_DIR"], case=app.config.get("CASE") or "")
        return jsonify({"ok": True, "id": job.id, **info})

    @app.get("/api/workfolder")
    def api_wf_get():
        return jsonify(wf_info())

    @app.get("/api/workspace")
    def api_ws_get():
        return jsonify(ws_info())

    @app.post("/api/workspace")
    def api_ws_set():
        """Change the workspace (the root of all work folders); kept for the
        next start. Runs in flight are not touched."""
        b = request.get_json(force=True, silent=True) or {}
        try:
            ws_use(b.get("path"), create=bool(b.get("create", True)))
        except (ValueError, OSError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, **ws_info(), "workfolder": wf_info()})

    @app.post("/api/workfolder")
    def api_wf_set():
        """Open a work folder, making it the one new runs are written to.

        Runs already in flight are not touched - they finish into the folder
        they were started in. Only the list the browser shows changes.
        """
        b = request.get_json(force=True, silent=True) or {}
        try:
            wf_use(b.get("path"), create=bool(b.get("create", True)))
        except (ValueError, OSError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, **wf_info()})

    @app.get("/api/case")
    def api_case_get():
        return jsonify(case_info())

    @app.post("/api/case")
    def api_case_set():
        """Open, or create, a case inside the current work folder. New runs are
        written under it; runs already in flight are not moved."""
        b = request.get_json(force=True, silent=True) or {}
        try:
            case_use(b.get("name"), create=bool(b.get("create", True)))
        except (ValueError, OSError) as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, **case_info()})

    @app.get("/api/jobs")
    def api_list():
        me = owner()
        rows = []
        for j in app.queue.for_owner(me, folder=app.config["RUNS_DIR"]):
            if "kind" not in j.meta:
                # a run from before the tags: read them from its input once
                try:
                    with open(os.path.join(j.out_dir, "spec.normalized.json"), encoding="utf-8") as fh:
                        sp = json.load(fh)
                    pl = None
                    try:
                        with open(os.path.join(j.out_dir, "result.json"), encoding="utf-8") as fh:
                            pl = json.load(fh).get("plan")
                    except (OSError, ValueError):
                        pass
                    j.meta.update(run_tags(sp, pl))
                except (OSError, ValueError):
                    j.meta["kind"] = "analysis" if j.meta.get("analyses") else "structure"
            rows.append({"id": j.id, "state": j.state, "name": j.meta.get("name"),
                         "kind": j.meta.get("kind"), "structure": j.meta.get("structure"),
                         "grid": j.meta.get("grid"), "doe_label": j.meta.get("doe_label"),
                         # the case folder this run lives in, so the list can be
                         # grouped the way the directory tree is
                         "case": j.case,
                         # the study a DOE case belongs to, so the run list can
                         # fold a study into one entry instead of 64 loose rows
                         "doe": j.meta.get("doe"),
                         "analyses": j.meta.get("analyses"), "created": j.created,
                         "progress": j.progress, "stage": j.stage,
                         "elapsed_s": ((j.ended or time.time()) - j.started) if j.started else 0,
                         "headline": (j.summary or {}).get("headline") if j.state == "done" else None,
                         "queue_position": app.queue.position(j) if j.state == "queued" else 0})
        return jsonify(rows)

    @app.get("/api/jobs/<jid>")
    def api_job(jid):
        return jsonify(job_or_404(jid).snapshot())

    @app.get("/api/jobs/<jid>/events")
    def api_events(jid):
        job = job_or_404(jid)
        after = int(request.args.get("after", 0))
        ev = job.since(after)[-800:]
        return jsonify({"seq": job.seq, "state": job.state, "progress": job.progress,
                        "stage": job.stage, "stage_detail": job.stage_detail, "solver": job.solver,
                        "queue_position": app.queue.position(job) if job.state == "queued" else 0,
                        "elapsed_s": ((job.ended or time.time()) - job.started) if job.started else 0,
                        "events": [{"seq": s, "kind": k, "data": p} for s, k, p in ev
                                   if k in ("line", "stage", "partial", "finished", "state", "figure",
                                            "partial_result")],
                        "history": {k: v[-400:] for k, v in job.history.items()}})

    @app.post("/api/jobs/<jid>/stop")
    def api_stop(jid):
        return jsonify({"ok": app.queue.stop(job_or_404(jid))})

    @app.delete("/api/jobs/<jid>")
    def api_delete(jid):
        job = job_or_404(jid)
        # an ownerless run belongs to no one in particular - the command line,
        # or a workspace copied in - so anyone who can see it may remove it
        if job.owner is not None and job.owner != owner():
            abort(403, "Runs of other users cannot be deleted")
        if job.state in ("running", "queued"):
            abort(409, "A running or queued run must be stopped first")
        shutil.rmtree(job.out_dir, ignore_errors=True)
        with app.queue.lock:
            app.queue.jobs.pop(jid, None)
            if jid in app.queue.order:
                app.queue.order.remove(jid)
        return jsonify({"ok": True})

    @app.get("/api/jobs/<jid>/result")
    def api_result(jid):
        job = job_or_404(jid)
        p = os.path.join(job.out_dir, "result.json")
        if not os.path.exists(p) and request.args.get("partial"):
            # the analyses finished so far, while the run goes on
            p = os.path.join(job.out_dir, "result.partial.json")
        if not os.path.exists(p):
            abort(404, "No result is available yet")
        return send_file(p, mimetype="application/json", max_age=0)

    import threading
    _upgrade_lock, _upgraded = threading.Lock(), set()

    @app.get("/api/jobs/<jid>/view/<path:name>")
    def api_view(jid, name):
        job = job_or_404(jid)
        view = os.path.join(job.out_dir, "view")
        p = safe_path(view, name)
        if not os.path.exists(p):
            abort(404)
        mt = "application/json" if name.endswith(".json") else "application/octet-stream"
        if name == "meta.json" and job.state in ("done", "failed", "stopped") and jid not in _upgraded:
            # an earlier run painted the particles with the resin's
            # shear rate from inside the fillers (all zero): sampled again
            # once from the resin side, out of the files the run kept
            with _upgrade_lock:
                if jid not in _upgraded:
                    try:
                        from mpsim import visual as VIS
                        VIS.upgrade_matrix_fields(view)
                    except Exception as e:                              # noqa: BLE001
                        app.logger.warning("view of %s not upgraded: %s", jid, e)
                    _upgraded.add(jid)
        if name.endswith(".gz"):
            # a gzip file is sent as such and the browser inflates it: the
            # voxel volume crosses the network at a fraction of its size
            resp = send_file(p, mimetype="application/octet-stream", max_age=3600, conditional=False)
            resp.headers["Content-Encoding"] = "gzip"
            return resp
        # meta.json changes when a run is continued; the binaries carry its
        # build stamp in their URL instead
        return send_file(p, mimetype=mt, max_age=0 if name == "meta.json" else 3600)

    def _view_meta(job):
        with open(os.path.join(job.out_dir, "view", "meta.json"), encoding="utf-8") as fh:
            return json.load(fh)

    @app.get("/api/jobs/<jid>/fslice/<key>/<axis>/<int:k>")
    def api_field_slice(jid, key, axis, k):
        """One section of a field at the solver's resolution (float32, first
        in-plane axis fastest), from the 16-bit copy the run kept."""
        import numpy as np
        from mpsim import visual as VIS
        job = job_or_404(jid)
        if axis not in ("x", "y", "z"):
            abort(400)
        try:
            a = VIS.field_slice(os.path.join(job.out_dir, "view"), _view_meta(job), key, axis, k)
        except (OSError, ValueError):
            a = None
        if a is None:
            abort(404)
        return Response(np.ascontiguousarray(a.T).astype("<f4").tobytes(), mimetype="application/octet-stream",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/jobs/<jid>/vol8/<key>")
    def api_field_volume(jid, key):
        """A field on every voxel at 8 bits for the browser's voxel volume
        (made once from the 16-bit copy, then kept)."""
        from mpsim import visual as VIS
        job = job_or_404(jid)
        view = os.path.join(job.out_dir, "view")
        try:
            fn = VIS.field_volume8(view, _view_meta(job), key)
        except (OSError, ValueError):
            fn = None
        if not fn:
            abort(404)
        resp = send_file(os.path.join(view, fn), mimetype="application/octet-stream", max_age=3600, conditional=False)
        resp.headers["Content-Encoding"] = "gzip"
        return resp

    @app.get("/api/jobs/<jid>/lslice/<axis>/<int:k>")
    def api_label_slice(jid, axis, k):
        """One section of the structure at the solver's resolution (uint8,
        first in-plane axis fastest), read from structure_labels.tif - the
        browser volume is coarser than the solver's grid on a large RVE."""
        import numpy as np
        from mpsim import visual as VIS
        job = job_or_404(jid)
        if axis not in ("x", "y", "z"):
            abort(400)
        a = VIS.label_slice(os.path.join(job.out_dir, "view"), axis, k)
        if a is None:
            abort(404)
        return Response(np.ascontiguousarray(a.T).tobytes(), mimetype="application/octet-stream",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/jobs/<jid>/figures/<name>")
    def api_figure(jid, name):
        job = job_or_404(jid)
        p = safe_path(os.path.join(job.out_dir, "figures"), name)
        if not os.path.exists(p):
            abort(404)
        return send_file(p, mimetype="image/png", max_age=600)

    @app.post("/api/jobs/<jid>/render")
    def api_render(jid):
        job = job_or_404(jid)
        body = request.get_json(force=True, silent=True) or {}
        kind = body.get("kind", "scene")
        req = body.get("req") or {}
        view = os.path.join(job.out_dir, "view")
        if not os.path.exists(os.path.join(view, "meta.json")):
            abort(404, "No visualisation data is available")
        out = os.path.join(job.out_dir, "figures", f"export_{int(time.time()*1000)}.png")
        try:
            app.render.render("slice" if kind == "slice" else "scene", view, req, out)
        except Exception as e:                                  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)[-2000:]}), 500
        return send_file(out, mimetype="image/png", as_attachment=False, max_age=0)

    # ---------------------------------------------------------- calibration
    @app.post("/api/calibration/theory")
    def api_cal_theory():
        """Theory values for the editor (mismatch-model interfacial
        resistances) and, with measurements and unknowns, the effective-medium
        preview: what these measurements can determine, before any solve."""
        from mpsim import calibrate as CA
        b = request.get_json(force=True, silent=True) or {}
        out = {}
        try:
            out["theory"] = CA.theory(b.get("form") or b.get("base") or {})
        except Exception as e:                                       # noqa: BLE001
            out["theory_error"] = str(e)
        if b.get("measurements") and b.get("unknowns"):
            try:
                out["preview"] = CA.preview(b)
            except Exception as e:                                   # noqa: BLE001
                out["preview_error"] = str(e)
        return jsonify(out)

    @app.post("/api/calibration")
    def api_cal_submit():
        from mpsim import calibrate as CA
        job = request.get_json(force=True, silent=True) or {}
        job["_kind"] = "calibration"
        try:
            CA._validate(job)
            S.normalize(copy.deepcopy(CA.fill_props(job.get("base") or {})))
        except ValueError as e:
            return jsonify({"ok": False, "errors": str(e).split("\n")}), 400
        job["_limits"] = app.config["LIMITS"]
        # structures of earlier runs in this case are reused when they match
        job["_structure_dirs"] = [j.out_dir for j in app.queue.for_owner(owner_or_none(), limit=200,
                                                                          folder=app.config["RUNS_DIR"])
                                  if j.state == "done" and os.path.isdir(os.path.join(j.out_dir, "structure"))]
        names = ", ".join(u.get("label") or u.get("path") for u in job.get("unknowns") or [])
        meta = {"name": job.get("name") or f"Calibration of {names}", "analyses": ["calibration"],
                "kind": "calibration", "structure": f"{len(job['measurements'])} measurement(s) · {names}",
                "mem_gb": 0.5}
        jb = app.queue.submit(owner_or_none(), job, case_dir(), meta, folder=app.config["RUNS_DIR"],
                              case=app.config.get("CASE") or "")
        return jsonify({"ok": True, "id": jb.id})

    # ---------------------------------------------------------- DOE studies
    @app.post("/api/doe/parameters")
    def api_doe_params():
        from mpsim import doe as DOE
        form = request.get_json(force=True, silent=True) or {}
        return jsonify(DOE.parameter_catalogue(form))

    @app.post("/api/doe")
    def api_doe_create():
        payload = request.get_json(force=True, silent=True) or {}
        form = payload.get("form") or {}
        form["_limits"] = app.config["LIMITS"]
        payload["form"] = form
        try:
            study = app.doe.create(owner_or_none(), payload, app.queue, case_dir(), S.normalize,
                                   folder=app.config["RUNS_DIR"], case_name=app.config.get("CASE") or "",
                                   estimate=est_mem_gb)
        except ValueError as e:
            return jsonify({"ok": False, "errors": str(e).split("\n")}), 400
        return jsonify({"ok": True, "id": study["id"], "n_cases": len(study["cases"])})

    @app.get("/api/doe")
    def api_doe_list():
        me = owner()
        out = []
        for s in app.doe.all(me):
            rows, _ = app.doe.rows(s, app.queue)
            counts = {}
            for r in rows:
                counts[r["state"]] = counts.get(r["state"], 0) + 1
            out.append({"id": s["id"], "name": s["name"], "created": s["created"], "design": s["design"],
                        "n_cases": len(s["cases"]), "counts": counts})
        return jsonify(out)

    @app.get("/api/doe/<sid>")
    def api_doe_get(sid):
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        return jsonify(app.doe.summary(s, app.queue))

    @app.get("/api/doe/<sid>/csv")
    def api_doe_csv(sid):
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        data = app.doe.csv(s, app.queue).encode("utf-8-sig")
        name = (s["name"] or sid).replace("/", "_")[:60]
        return Response(data, mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})

    @app.post("/api/doe/<sid>/optimize")
    def api_doe_optimize(sid):
        """Key design factors, a surrogate of the study and its optimum."""
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        body = request.get_json(force=True, silent=True) or {}
        response = body.get("response")
        if not response:
            return jsonify({"ok": False, "error": "No response quantity was chosen"}), 400
        rows, metrics = app.doe.rows(s, app.queue)
        from mpsim import optimise as OPT
        try:
            out = OPT.analyse(s, rows, response, goal=body.get("goal") or "max",
                              target=body.get("target"), seed=int(body.get("seed") or 0),
                              options=body.get("options") or None)
        except Exception as e:                                  # noqa: BLE001
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
        out["ok"] = True
        out["metrics"] = metrics
        out["study"] = {"id": s["id"], "name": s["name"], "design": s["design"]}
        return jsonify(out)

    @app.get("/api/doe/<sid>/kernels")
    def api_doe_kernels(sid):
        """The surrogate models that can be chosen, for the tuning step."""
        from mpsim import optimise as OPT
        return jsonify({"kernels": [{"id": k, "label": v[0]} for k, v in OPT.KERNELS.items()]})

    @app.post("/api/doe/<sid>/pareto")
    def api_doe_pareto(sid):
        """Several objectives at once: the trade-off front (NSGA-II)."""
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        body = request.get_json(force=True, silent=True) or {}
        responses = body.get("responses") or []
        goals = body.get("goals") or []
        if len(responses) < 2:
            return jsonify({"ok": False, "error": "Two or more responses are needed for a trade-off"}), 400
        if len(goals) != len(responses):
            goals = (goals + ["max"] * len(responses))[:len(responses)]
        rows, metrics = app.doe.rows(s, app.queue)
        from mpsim import optimise as OPT
        try:
            out = OPT.pareto(s, rows, responses, goals, targets=body.get("targets") or None,
                             seed=int(body.get("seed") or 0), options=body.get("options") or None,
                             pop=int(body.get("pop") or 64),
                             generations=int(body.get("generations") or 60))
        except Exception as e:                                  # noqa: BLE001
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
        out["ok"] = not out.get("error")
        out["metrics"] = metrics
        out["study"] = {"id": s["id"], "name": s["name"], "design": s["design"]}
        return jsonify(out)

    @app.post("/api/doe/<sid>/point")
    def api_doe_point(sid):
        """Queue one run at a chosen point: the verification of an optimum.

        A predicted optimum is only a prediction; this runs the real solver
        there so the prediction can be compared with a measurement.
        """
        import copy as _copy
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        body = request.get_json(force=True, silent=True) or {}
        values = body.get("values") or {}
        if not values:
            return jsonify({"ok": False, "error": "No parameter values were given"}), 400
        from mpsim import doe as DOE
        base = _copy.deepcopy(s["base_form"])
        base["_limits"] = app.config["LIMITS"]
        case = {"values": {k: float(v) for k, v in values.items()}, "label": body.get("label") or "optimum"}
        form = DOE.apply_case(base, case)
        try:
            S.normalize(_copy.deepcopy(form))
        except ValueError as e:
            return jsonify({"ok": False, "errors": str(e).split("\n")}), 400
        meta = {"name": form["name"], "analyses": s.get("analyses") or [], "doe": sid,
                "mem_gb": est_mem_gb(form)}
        job = app.queue.submit(owner_or_none(), form, case_dir(), meta,
                               folder=app.config["RUNS_DIR"], case=app.config.get("CASE") or "")
        return jsonify({"ok": True, "id": job.id, "name": form["name"]})

    @app.post("/api/doe/<sid>/retry")
    def api_doe_retry(sid):
        """Queue again every case of a study whose run failed, was stopped or has
        gone missing.

        The design is kept; each such case is re-run with its own values and the
        study is pointed at the new run, so the table, the surrogate and the
        optimiser see it as if it had worked the first time. A retry lands next
        to the run it replaces, in that run's case folder, rather than in
        whatever case happens to be open.
        """
        import copy as _copy
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        from mpsim import doe as DOE
        done = []
        for case in s["cases"]:
            old = app.queue.get(case["job_id"])
            if old is not None and old.state not in ("failed", "stopped"):
                continue
            form = DOE.apply_case(_copy.deepcopy(s["base_form"]), {"values": case["values"], "label": case["label"]})
            form["_limits"] = app.config["LIMITS"]
            meta = {"name": form["name"], "analyses": s.get("analyses") or [], "doe": sid,
                    "mem_gb": est_mem_gb(form)}
            if old is not None:
                where, folder, cname = os.path.dirname(old.out_dir), old.folder, old.case
            else:
                where, folder, cname = case_dir(), app.config["RUNS_DIR"], app.config.get("CASE") or ""
            job = app.queue.submit(owner_or_none(), form, where, meta, folder=folder, case=cname)
            done.append({"label": case["label"], "old": case["job_id"], "new": job.id})
            case["job_id"] = job.id
        if done:
            app.doe.save(s)
        return jsonify({"ok": True, "requeued": len(done), "cases": done})

    @app.delete("/api/doe/<sid>")
    def api_doe_delete(sid):
        s = app.doe.get(sid)
        if s is None:
            abort(404, "DOE study not found")
        if s.get("owner") not in (owner(), None):
            abort(403, "DOE studies of other users cannot be deleted")
        if request.args.get("runs") == "1":
            for case in s["cases"]:
                job = app.queue.get(case["job_id"])
                if job is None:
                    continue
                if job.state in ("running", "queued"):
                    app.queue.stop(job)
                    continue
                shutil.rmtree(job.out_dir, ignore_errors=True)
                with app.queue.lock:
                    app.queue.jobs.pop(job.id, None)
                    if job.id in app.queue.order:
                        app.queue.order.remove(job.id)
        app.doe.delete(sid)
        return jsonify({"ok": True})

    @app.get("/api/jobs/<jid>/report")
    def api_report(jid):
        job = job_or_404(jid)
        p = os.path.join(job.out_dir, "report.html")
        if not os.path.exists(p):
            abort(404, "No report is available yet")
        return send_file(p, mimetype="text/html", max_age=0)

    @app.get("/api/jobs/<jid>/download/<what>")
    def api_download(jid, what):
        job = job_or_404(jid)
        d = job.out_dir
        names = {"layer_card": "layer_card.json", "result": "result.json", "structure": "structure.tif",
                 "structure_labels": "structure_labels.tif", "structure_legend": "structure_legend.csv",
                 "report": "report.html", "spec": "spec.json"}
        base = (job.meta.get("name") or job.id).replace("/", "_").replace("\\", "_")[:60]
        if what in names:
            p = os.path.join(d, names[what])
            if not os.path.exists(p):
                abort(404)
            return send_file(p, as_attachment=True, download_name=f"{base}_{names[what]}")
        if what == "structure_set":
            # the three files another program needs to rebuild the same model
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for n in ("structure.tif", "structure_labels.tif", "structure_legend.csv"):
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        zf.write(p, n)
            buf.seek(0)
            return send_file(buf, as_attachment=True, download_name=f"{base}_structure.zip",
                             mimetype="application/zip")
        if what == "bundle":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for n in ("result.json", "layer_card.json", "spec.json", "spec.normalized.json",
                          "structure.tif", "structure_labels.tif", "structure_legend.csv", "report.html", "run.log"):
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        zf.write(p, n)
                figs = os.path.join(d, "figures")
                if os.path.isdir(figs):
                    for n in os.listdir(figs):
                        zf.write(os.path.join(figs, n), f"figures/{n}")
            buf.seek(0)
            return send_file(buf, as_attachment=True, download_name=f"{base}_bundle.zip",
                             mimetype="application/zip")
        abort(404)

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    def _err(e):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": getattr(e, "description", str(e))}), e.code
        return e

    return app
