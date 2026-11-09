"""DOE studies: a set of runs created from one base form and a design.

A study is a small JSON file in `workspace/doe/`; the runs themselves are
ordinary jobs in the queue, so a study survives a server restart exactly as
single runs do and its cases can be opened, compared and deleted individually.
A copy of the file is kept beside the runs (doe_study_<id>.json in their case
folder), so that a folder opened in Runs - on another PC, in another
workspace - comes back as the study it was (5.6.7).
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import re
import time
import uuid

from mpsim import doe as DOE


# a size parameter of a phase (phases.<i>.size.<key>, or the standard deviation of d or of a dimension with a
# distribution of its own) and the dimension of the size report (grains.size_report) it is realised as; units of
# the form against micrometres
SIZE_PATH = re.compile(r"^phases\.(\d+)\.(?:size\.(d|length|aspect|aspect_y|ly|lz)|dist\.(std)"
                       r"|dists\.(length|aspect|aspect_y|ly|lz)\.std)$")
LEN_UNIT = {"um": 1.0, "µm": 1.0, "nm": 1e-3, "mm": 1e3}
MIRROR = "doe_study_{}.json"               # the copy of a study beside its runs
MIRROR_RE = re.compile(r"doe_study_(.+)\.json")


def realised(study, r):
    """What a finished case really holds, beside what was asked for: the
    filler content and, for every parameter (or linked entry) that sets a
    content or a share of the filler, the value of the generated structure.
    A dense packing can fall short of the target (BN stacks at half the
    filler: 51 of 55 vol%); a study is read against these, not the inputs.
    A parameter that sets a particle size gets the size realised: the number
    mean of that dimension over the particles placed (the standard deviation
    for one that sets the spread), against the same statistic of the
    particles the input draws (the distribution after its cut at ±2σ or the
    bounds), in the unit of the parameter.
    -> {key: (label, unit, value, the target the run aimed at)}"""
    comp = r.get("composition") or {}
    labs = comp.get("labels") or []
    sp = r.get("spec") or {}
    slabs = sp.get("labels") or []
    phases = sp.get("phases") or []
    if not labs or len(labs) != len(slabs) or not phases:
        return {}
    vf, wt = {}, {}
    for lab, sl in zip(labs, slabs):
        i = sl.get("phase")
        if i is None or sl.get("kind") not in ("core", "shell"):
            continue
        vf[i] = vf.get(i, 0.0) + float(lab.get("vf") or 0.0)
        wt[i] = wt.get(i, 0.0) + float(lab.get("wt") or 0.0)
    base = study.get("base_form") or {}
    bph = base.get("phases") or []
    sizes = ((r.get("realisations") or [{}])[0] or {}).get("particle_sizes") or {}

    def basis(i):
        return ((bph[i].get("fraction") or {}).get("basis") if i < len(bph) else None) or "vol"

    def name(i):
        return (phases[i].get("name") if i < len(phases) else None) or f"Phase {i + 1}"
    fill = [i for i, ph in enumerate(phases) if not ph.get("void") and ph.get("shape") != "network"]
    tg = comp.get("target") or {}
    tvf, twt = list(tg.get("vf") or []), list(tg.get("wt") or [])

    def target(i, b):
        t = tvf if b == "vol" else twt
        return float(t[i]) if i < len(t) and t[i] is not None else None
    out = {}
    if fill:
        b = basis(fill[0]) if len({basis(i) for i in fill}) == 1 else "vol"
        src = vf if b == "vol" else wt
        ts = [target(i, b) for i in fill]
        out["real_filler"] = ("Filler content, realised", f"{b} %", 100.0 * sum(src.get(i, 0.0) for i in fill),
                              100.0 * sum(ts) if None not in ts else None)
    paths = []
    for p in study.get("parameters") or []:
        paths.append(p.get("path") or "")
        paths += [ln.get("path") or "" for ln in (p.get("links") or [])]
    for path in paths:
        m = re.match(r"^phases\.(\d+)\.fraction\.value$", path)
        if m and int(m.group(1)) < len(phases):
            i = int(m.group(1))
            src = vf if basis(i) == "vol" else wt
            t = target(i, basis(i))
            out["real_" + path] = (f"{name(i)}: content, realised", f"{basis(i)} %", 100.0 * src.get(i, 0.0),
                                   100.0 * t if t is not None else None)
        m = re.match(r"^mix\.(\d+)$", path)
        if m and int(m.group(1)) in fill:
            i = int(m.group(1))
            src = vf if basis(i) == "vol" else wt
            tot = sum(src.get(k, 0.0) for k in fill)
            ts = [target(k, basis(i)) for k in fill]
            t = (100.0 * target(i, basis(i)) / sum(ts)) if (None not in ts and sum(ts) > 0) else None
            if tot > 0:
                out["real_" + path] = (f"{name(i)}: share of the filler, realised", "% of filler",
                                       100.0 * src.get(i, 0.0) / tot, t)
        m = SIZE_PATH.match(path)
        if m and int(m.group(1)) < len(phases):
            i = int(m.group(1))
            e = sizes.get(str(i))
            if not e:
                continue
            key = m.group(2) or m.group(4) or "d"
            if key == "d" and e.get("basis") == "eqd":
                key = "eqd"                      # the size entered is the volume-equivalent diameter
            dim = next((x for x in e.get("dimensions") or [] if x.get("key") == key), None)
            if not dim or not (dim.get("real") or {}).get("count"):
                continue
            stat = "std" if (m.group(3) or m.group(4)) else "number_mean"
            unit = ((bph[i].get("size") or {}) if i < len(bph) else {}).get("unit") or "um"
            length = dim.get("unit") == "um"
            f = 1.0 / LEN_UNIT.get(unit, 1.0) if length else 1.0
            real, inp = float(dim["real"].get(stat) or 0.0), (dim.get("input") or {}).get(stat)
            what = "standard deviation" if stat == "std" else (dim.get("label") or key).lower()
            out["real_" + path] = (f"{name(i)}: {what}, realised", ({"um": "µm"}.get(unit, unit) if length else "-"),
                                   real * f, float(inp) * f if inp is not None else None)
    return out


class DOEStore:
    def __init__(self, workspace):
        self.dir = os.path.join(workspace, "doe")
        os.makedirs(self.dir, exist_ok=True)

    # ------------------------------------------------------------- storage
    def _path(self, sid):
        return os.path.join(self.dir, f"{sid}.json")

    def save(self, study):
        with open(self._path(study["id"]), "w", encoding="utf-8") as fh:
            json.dump(study, fh, ensure_ascii=False, indent=1)
        # the copy beside the runs: a folder copied or opened elsewhere carries its study
        for d in study.get("dirs") or []:
            try:
                if os.path.isdir(d):
                    with open(os.path.join(d, MIRROR.format(study["id"])), "w", encoding="utf-8") as fh:
                        json.dump(study, fh, ensure_ascii=False, indent=1)
            except OSError:
                pass                                   # a read-only folder keeps the runs listed singly

    def get(self, sid):
        try:
            with open(self._path(sid), encoding="utf-8") as fh:
                return json.load(fh)
        except OSError:
            return None

    def all(self, owner=None):
        out = []
        for name in sorted(os.listdir(self.dir), reverse=True):
            if not name.endswith(".json"):
                continue
            s = self.get(name[:-5])
            if s and (owner is None or s.get("owner") in (owner, None)):
                out.append(s)
        return out

    def delete(self, sid):
        s = self.get(sid) or {}
        for d in s.get("dirs") or []:
            try:
                os.remove(os.path.join(d, MIRROR.format(sid)))
            except OSError:
                pass
        try:
            os.remove(self._path(sid))
            return True
        except OSError:
            return False

    @staticmethod
    def _add_dir(study, d):
        d = os.path.abspath(d)
        dirs = study.setdefault("dirs", [])
        if not any(os.path.normcase(x) == os.path.normcase(d) for x in dirs):
            dirs.append(d)

    # -------------------------------------------------------------- create
    def create(self, owner, payload, queue, runs_dir, normalize, folder=None, case_name="",
               estimate=None, plan=None):
        """Expand the design, validate every case and queue the runs.

        The cases take the Domain tab of the base as it is - thin film and
        thickness, accuracy level, arrangement, dispersion, realisations. The
        grid: the voxel size is the Domain's (a manual grid as entered, an
        automatic one as planned for the base) unless it is a parameter of the
        study; the RVE edge is the Domain's for every case ('fixed') or each
        case's own, planned for its particles at the Domain's level
        ('adaptive'), on that voxel size. 'own': edge and voxel both planned
        for each case's particles at the Domain's level, for studies that vary
        a particle size."""
        base = copy.deepcopy(payload.get("form") or {})
        # the machine's grid limits (Home -> Machine) go with every case; the
        # stored base form leaves them out, so a later change applies to
        # retries and optimum runs
        limits = base.pop("_limits", None)
        g = payload.get("grid") or {}
        edge = g.get("edge") or ("adaptive" if g.get("mode") == "adaptive" else "fixed")
        if plan is None:
            raise ValueError("The study needs the plan of its base case")
        try:
            bp = plan(base)
        except ValueError as e:
            raise ValueError(f"The base case cannot be planned: {e}") from None
        L0, h0 = float(bp["L_um"]), float(bp["h_um"])
        grid = {"edge": edge, "L_um": L0, "voxel_um": h0, "film": bool(bp.get("film")),
                "T_um": float(bp["T_um"]) if bp.get("film") else None, "quality": bp.get("quality")}
        parameters = [p for p in (payload.get("parameters") or []) if p.get("path")]
        # links: a parameter that also sets other entries (the same value or a
        # function of the parameters); checked once here, applied per case
        for p in parameters:
            p["links"] = [{"path": ln["path"], "expr": (str(ln.get("expr") or "x").strip() or "x")}
                          for ln in (p.get("links") or []) if (ln.get("path") or "").strip()]
        errs = DOE.check_links(parameters, base)
        if errs:
            raise ValueError("\n".join(errs))
        design = payload.get("design") if payload.get("design") in ("full", "oat", "lhs") else "full"
        cases = DOE.expand(parameters, design, int(payload.get("samples", 12)), int(payload.get("seed", 1)))
        if not cases:
            raise ValueError("No parameter with a range was given")
        varied = set()
        for p in parameters:
            varied.add(p["path"])
            varied.update(ln["path"] for ln in p.get("links") or [])
        forms = [self._case_form(base, c, i, parameters, grid, varied, plan, normalize) for i, c in enumerate(cases)]
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        study = {"id": sid, "name": payload.get("name") or (base.get("name") or "DOE study"),
                 "created": time.time(), "owner": owner, "design": design,
                 "parameters": parameters, "samples": payload.get("samples"), "seed": payload.get("seed"),
                 "grid": grid,
                 "analyses": [k for k, v in (base.get("analyses") or {}).items() if v],
                 "base_form": base, "cases": [], "dirs": []}
        self._add_dir(study, runs_dir)
        self._submit(study, forms, cases, queue, runs_dir, owner, folder, case_name, estimate, limits)
        self.save(study)
        return study

    @staticmethod
    def _varied(parameters):
        varied = set()
        for p in parameters:
            varied.add(p["path"])
            varied.update(ln["path"] for ln in p.get("links") or [])
        return varied

    @staticmethod
    def _case_form(base, c, i, parameters, grid, varied, plan, normalize):
        """The complete form of one case on the study's grid (see create)."""
        h0, L0 = float(grid["voxel_um"]), float(grid["L_um"])
        try:
            form = DOE.apply_case(base, c, i, parameters=parameters)
            rv = form.setdefault("rve", {})
            h = float(rv["voxel_um"]) if "rve.voxel_um" in varied else h0
            own = None
            if grid.get("edge") in ("adaptive", "own"):
                # the case's own plan at the Domain's level (its particles may differ)
                auto = copy.deepcopy(form)
                auto["rve"] = dict(auto.get("rve") or {}, auto=True, N=None, L_um=None, voxel_um=None)
                own = plan(auto)
                if grid["edge"] == "own" and "rve.voxel_um" not in varied:
                    # a particle size that varies: the voxel resolves this case's smallest particle
                    # (a voxel planned for the base would under-resolve the small ones and waste
                    # voxels on the large) with the same count in every case - not rounded to a nice
                    # value, which gives 6 to 8 voxels per diameter from case to case, and touching
                    # particles conduct through contacts that depend on it. A grid limit that moved
                    # the voxel keeps the limited one.
                    h = float(own["h_um"])
                    rec = own.get("recommended") or {}
                    if rec.get("h_exact") and abs(h - float(rec.get("h_um") or 0.0)) <= 1e-9 * h:
                        h = float(rec["h_exact"])
            if "rve.N" in varied and rv.get("N"):
                L = float(rv["N"]) * h                 # a grid-size study: the edge follows the count
            elif "rve.L_um" in varied:
                L = float(rv["L_um"])
            elif own is not None:
                L = float(own["L_um"])
            else:
                L = L0
            n = max(8, int(round(L / h)))
            n += n % 2
            rv.update(auto=False, voxel_um=h, L_um=n * h, N=None, auto_enlarge=False, L_unit="um", h_unit="um")
        except ValueError as e:
            raise ValueError(f"Case {i + 1} ({c['label']}): {e}") from None
        normalize(copy.deepcopy(form))          # raises ValueError on a bad case
        return form

    def _submit(self, study, forms, cases, queue, runs_dir, owner, folder, case_name, estimate, limits):
        sid = study["id"]
        for form, case in zip(forms, cases):
            meta = {"name": form["name"], "analyses": study["analyses"], "doe": sid,
                    "mem_gb": estimate(form) if estimate else 0.0,
                    # the structure phase's memory: structures are built side by
                    # side, the solves wait for their memory (jobs.JobQueue)
                    "mem_struct_gb": getattr(estimate, "last_struct", None) if estimate else None,
                    "gated": True,
                    # the variable values of this case, for the run list
                    "doe_label": case["label"], "kind": "analysis" if study["analyses"] else "structure"}
            # `case` here is the design point; the folder the run lands in is case_name
            if limits:
                form["_limits"] = limits
            job = queue.submit(owner, form, runs_dir, meta, folder=folder, case=case_name)
            study["cases"].append({"job_id": job.id, "values": case["values"], "label": case["label"],
                                   **({"added": case["added"]} if case.get("added") else {})})

    def add(self, study, points, queue, runs_dir, normalize, folder=None, case_name="", estimate=None, plan=None,
            limits=None, tag=""):
        """Further cases of a study at chosen points ({path: value} for every
        parameter): the points an optimisation wants run next, on the study's
        own grid, shown and fitted with the others."""
        parameters = study.get("parameters") or []
        paths = [p["path"] for p in parameters]
        cases = []
        for v in points:
            vals = {}
            for p in paths:
                if p not in v:
                    raise ValueError(f"A point needs a value for every parameter of the study ({p} is missing)")
                vals[p] = float(v[p])
            lab = " · ".join(f"{(q.get('label') or q['path'])} {DOE._fmt(vals[q['path']])}" for q in parameters)
            cases.append({"values": vals, "label": (f"{tag} · " if tag else "") + lab, "added": tag or True})
        if not cases:
            raise ValueError("No point was given")
        if any(p["path"].startswith("_label.") for p in parameters):
            raise ValueError("This study was rebuilt from its runs and one of its factors could not be traced to an "
                             "input entry; cases cannot be added to it")
        varied = self._varied(parameters)
        i0 = len(study["cases"])
        forms = [self._case_form(study["base_form"], c, i0 + k, parameters, study["grid"], varied, plan, normalize)
                 for k, c in enumerate(cases)]
        self._add_dir(study, runs_dir)
        self._submit(study, forms, cases, queue, runs_dir, study.get("owner"), folder, case_name, estimate, limits)
        self.save(study)
        return cases

    def place_copies(self, queue):
        """The copy beside the runs for studies made before 5.6.7, which had
        none: in the case folders of their runs as listed now. A study none of
        whose runs is listed is left for a later start."""
        for s in self.all():
            if "dirs" in s:
                continue
            jobs = [queue.get(c.get("job_id")) for c in s.get("cases") or []]
            jobs = [j for j in jobs if j is not None]
            if not jobs:
                continue
            s["dirs"] = []
            for j in jobs:
                self._add_dir(s, os.path.dirname(j.out_dir))
            try:
                self.save(s)
            except OSError:
                pass

    # ----------------------------------------------- studies of opened folders
    def take_in(self, folder, jobs, owner=None):
        """Studies of the runs of a folder opened in Runs (a work or case
        folder, made in another workspace or on another PC) into this
        workspace, so that the DOE tab shows them as they were. `jobs`: the
        runs listed from the folder. A study already here is left as it is.
        Each one comes, in this order, from the copy beside its runs
        (doe_study_<id>.json, written since 5.6.7), from the doe folder of the
        workspace the runs were made in (a work folder sits in its workspace
        unless it was put elsewhere), or rebuilt from the runs themselves
        (rebuild). -> [{id, name, cases, source}], the studies here already
        with source "here"."""
        folder = os.path.abspath(folder)
        out, groups = [], {}
        for j in jobs:
            sid = (j.meta or {}).get("doe")
            if sid:
                groups.setdefault(sid, []).append(j)

        def take(s, source, where=None):
            s["owner"] = owner if owner is not None else s.get("owner")
            # the folders the runs are in now (the paths of the PC they were made on mean nothing here)
            s["dirs"] = []
            for j in groups.get(s["id"]) or []:
                self._add_dir(s, os.path.dirname(j.out_dir))
            if where and not s["dirs"]:
                self._add_dir(s, where)
            self.save(s)
            out.append({"id": s["id"], "name": s.get("name"), "cases": len(s.get("cases") or []), "source": source})

        # 1. the copies beside the runs: in the folder and in its case folders
        seen = set()
        for d in [folder] + [os.path.join(folder, x) for x in sorted(os.listdir(folder))
                             if os.path.isdir(os.path.join(folder, x))]:
            try:
                names = os.listdir(d)
            except OSError:
                continue
            for name in names:
                m = MIRROR_RE.fullmatch(name)
                if not m or m.group(1) in seen:
                    continue
                seen.add(m.group(1))
                if self.get(m.group(1)) is not None:
                    continue
                try:
                    with open(os.path.join(d, name), encoding="utf-8") as fh:
                        s = json.load(fh)
                except (OSError, ValueError):
                    continue
                if s.get("id") == m.group(1) and s.get("cases"):
                    take(s, "file", d)
        # 2. the doe folder of the workspace above the folder; 3. rebuilt from the runs
        for sid, js in groups.items():
            if sid in seen or self.get(sid) is not None:
                continue
            s, anc = None, folder
            for _ in range(4):
                p = os.path.join(anc, "doe", f"{sid}.json")
                if os.path.isfile(p):
                    try:
                        with open(p, encoding="utf-8") as fh:
                            s = json.load(fh)
                    except (OSError, ValueError):
                        s = None
                    break
                up = os.path.dirname(anc)
                if up == anc:
                    break
                anc = up
            if s is not None and s.get("id") == sid:
                take(s, "workspace")
                continue
            try:
                s = rebuild(sid, js)
            except Exception:                                   # noqa: BLE001
                s = None                                       # its runs stay listed one by one
            if s is not None:
                take(s, "rebuilt")
        # studies of these runs that were here already: opened as well
        got = {x["id"] for x in out}
        for sid in groups:
            s = self.get(sid) if sid not in got else None
            if s is not None:
                out.append({"id": sid, "name": s.get("name"), "cases": len(s.get("cases") or []), "source": "here"})
        return out

    # -------------------------------------------------------------- status
    def rows(self, study, queue):
        """Per-case state and results, read from the finished runs."""
        rows = []
        metrics = {}
        for case in study["cases"]:
            job = queue.get(case["job_id"])
            try:
                lk = DOE.linked(study["parameters"], case["values"])
            except ValueError:
                lk = {}
            row = {"job_id": case["job_id"], "label": case["label"], "values": case["values"],
                   "linked": {k: (float(v) if not hasattr(v, "__len__") else None) for k, v in lk.items()},
                   "state": job.state if job else "missing",
                   "progress": (1.0 if job and job.state == "done" else (job.progress if job else 0.0)),
                   "elapsed_s": ((job.ended or time.time()) - job.started) if (job and job.started) else 0.0,
                   "metrics": {}}
            if job and job.state == "done":
                res = os.path.join(job.out_dir, "result.json")
                try:
                    with open(res, encoding="utf-8") as fh:
                        r = json.load(fh)
                    # a restored job has no start time any more; the run itself
                    # recorded how long it took
                    row["elapsed_s"] = float(r.get("elapsed_s") or row["elapsed_s"])
                    row["targets"] = {}
                    for k, (lab, unit, v, t) in realised(study, r).items():
                        row["metrics"][k] = v
                        row["targets"][k] = t
                        metrics.setdefault(k, {"key": k, "label": lab, "unit": unit, "realised": True})
                    for h in r.get("headline") or []:
                        row["metrics"][h["key"]] = h["value"]
                        metrics.setdefault(h["key"], {"key": h["key"], "label": h["label"], "unit": h["unit"]})
                except Exception:                                 # noqa: BLE001
                    pass
            rows.append(row)
        return rows, list(metrics.values())

    def summary(self, study, queue):
        rows, metrics = self.rows(study, queue)
        counts = {}
        for r in rows:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
        return {"id": study["id"], "name": study["name"], "created": study["created"],
                "design": study["design"], "n_cases": len(study["cases"]), "counts": counts,
                "grid": study.get("grid") or {"mode": "adaptive"},
                "parameters": [{"path": p["path"], "label": p.get("label") or p["path"], "unit": p.get("unit", ""),
                                "symbol": f"x{i + 1}", "links": p.get("links") or []}
                               for i, p in enumerate(study["parameters"])],
                "analyses": study.get("analyses") or [], "metrics": metrics, "rows": rows,
                "rebuilt": bool(study.get("rebuilt"))}

    def csv(self, study, queue):
        rows, metrics = self.rows(study, queue)
        params = [{"path": p["path"], "label": p.get("label") or p["path"]} for p in study["parameters"]]
        lpaths = [ln["path"] for p in study["parameters"] for ln in (p.get("links") or [])]
        buf = io.StringIO()
        head = (["case", "run_id", "state"] + [p["label"] for p in params] + [f"{lp} (linked)" for lp in lpaths]
                + [f"{m['label']} [{m['unit']}]" for m in metrics])
        buf.write(",".join(f'"{h}"' for h in head) + "\n")
        for i, r in enumerate(rows, start=1):
            cells = [str(i), r["job_id"], r["state"]]
            cells += [str(r["values"].get(p["path"], "")) for p in params]
            cells += [("" if (r.get("linked") or {}).get(lp) is None else repr(r["linked"][lp])) for lp in lpaths]
            cells += [("" if r["metrics"].get(m["key"]) is None else repr(r["metrics"][m["key"]])) for m in metrics]
            buf.write(",".join(f'"{c}"' for c in cells) + "\n")
        return buf.getvalue()


def _num_leaves(obj, prefix=""):
    """Paths of the numbers in a form (lists by index), for factors a study set outside the catalogue."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not str(k).startswith("_"):
                yield from _num_leaves(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _num_leaves(v, f"{prefix}{i}.")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix[:-1]


def _half_digit(txt, x):
    """Half the last digit DOE._fmt keeps of a number: 6 significant digits after rounding to 6 decimals, 3 for
    the very small and the very large (trailing zeros are dropped, so "30" stands for 30.0000)."""
    a = abs(x)
    if a == 0:
        return 0.5e-6
    if "e" in txt.lower() or a < 1e-3 or a >= 1e5:
        return 0.5 * 10.0 ** (math.floor(math.log10(a)) - 2)
    return max(0.5 * 10.0 ** (math.floor(math.log10(a)) - 5), 0.5e-6)


def _value(form, path):
    if path.startswith("mix."):
        return DOE.filler_share(form, int(path.split(".")[1]))
    try:
        v = DOE.get_path(form, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def rebuild(sid, jobs):
    """A study rebuilt from its runs alone, for runs whose study file is gone
    (made before 5.6.7, and copied without the doe folder of their
    workspace). Every case's name gives the factors as "<label> <value>"
    joined by " · "; each factor is traced to the input entry that holds
    that value in every case (the catalogue of the DOE tab first, with the
    same label first, then any number of the input). A factor that cannot be
    traced keeps its label and the values of the names, and the study then
    shows its cases but takes no new ones. The design (full factorial or
    sampled) is judged from the values; the grid is the first case's."""
    runs = []
    for j in jobs:
        form = j.spec if isinstance(j.spec, dict) else {}
        lab = str((j.meta or {}).get("doe_label") or "")
        if not form or not lab:
            continue
        tag, items = None, []
        for seg in lab.split(" · "):
            head, _, val = seg.rpartition(" ")
            try:
                items.append((head, float(val), val))
            except ValueError:
                if not items:
                    tag = seg                                  # a case added by an optimisation: its tag first
        runs.append({"job": j, "form": form, "label": lab, "tag": tag, "items": items})
    if not runs:
        return None
    runs.sort(key=lambda r: (r["form"].get("_doe_index", 10 ** 9), getattr(r["job"], "created", 0) or 0))
    n = max(set(len(r["items"]) for r in runs), key=lambda k: sum(len(r["items"]) == k for r in runs))
    runs = [r for r in runs if len(r["items"]) == n]
    if not runs or n == 0:
        return None
    first = runs[0]["form"]
    cat = DOE.parameter_catalogue(first)
    units = {c["path"]: c.get("unit", "") for c in cat}
    cands = [c["path"] for c in cat] + [p for p in _num_leaves(first) if p not in units]

    def fits(path, j):
        for r in runs:
            v = _value(r["form"], path)
            if v is None:
                return False
            _, x, txt = r["items"][j]
            if DOE._fmt(v) != txt and abs(v - x) > 1e-6 * max(1.0, abs(x)):
                return False
        return True

    fill = DOE._fillers(first)

    def total(form):
        return sum(float(DOE.get_path(form, f"phases.{k}.fraction.value") or 0.0) for k in fill)

    def fits_total(path, j):
        """A filler's content set before a share of the filler spread the total again (DOE.apply_case): the
        total in the run less the value of the name is the same in every case - what the other fillers held in
        the study's input. -> that value, or None."""
        m = re.fullmatch(r"phases\.(\d+)\.fraction\.value", path)
        if not m or int(m.group(1)) not in fill or len(fill) < 2:
            return None
        # a name holds 6 significant digits (DOE._fmt): the value lies within half its last digit; the
        # intersection over the cases holds the input value, taken with the fewest decimals
        lo, hi = -math.inf, math.inf
        for r in runs:
            _, x, txt = r["items"][j]
            u = _half_digit(txt, x) + 1e-9 * max(1.0, abs(x))
            dif = total(r["form"]) - x
            lo, hi = max(lo, dif - u), min(hi, dif + u)
        if lo > hi:
            return None
        for dec in range(10):
            c = round(0.5 * (lo + hi), dec)
            if lo <= c <= hi:
                return c
        return 0.5 * (lo + hi)

    params, used, before = [], set(), {}
    for j in range(n):
        label = runs[0]["items"][j][0]
        vals = [r["items"][j][1] for r in runs]
        same = [c["path"] for c in cat if c.get("label") == label]
        order = same + [p for p in cands if p not in same]
        path = next((p for p in order if p not in used and fits(p, j)), None)
        if path is None:
            for p in order:
                rest = fits_total(p, j) if p not in used else None
                if rest is not None:
                    path, before[p] = p, rest
                    break
        if path is None:
            path = f"_label.{j}"
        used.add(path)
        params.append({"path": path, "label": label, "unit": units.get(path, ""),
                       "min": min(vals), "max": max(vals), "links": []})
    cases = []
    for r in runs:
        values = {}
        for j, p in enumerate(params):
            if p["path"] in before:
                v = total(r["form"]) - before[p["path"]]
            else:
                v = None if p["path"].startswith("_label.") else _value(r["form"], p["path"])
            values[p["path"]] = float(v) if v is not None else r["items"][j][1]
        c = {"job_id": r["job"].id, "values": values, "label": r["label"]}
        if r["tag"]:
            c["added"] = r["tag"]
        cases.append(c)
    levels_n = 1
    for p in params:
        levels_n *= len({DOE._fmt(c["values"][p["path"]]) for c in cases})
    design = "full" if levels_n == len(cases) else "lhs"
    base = copy.deepcopy(first)
    name = str(base.get("name") or "")
    lab0 = runs[0]["label"]
    if name.endswith(" · " + lab0):
        name = name[:-len(lab0) - 3]
    base["name"] = name or "DOE study"
    for k in ("_doe_index", "_limits", "_reuse"):
        base.pop(k, None)
    for p, rest in before.items():
        # the other filler's content in the study's input, so that a case added later is built as these were
        others = [k for k in fill if k != int(p.split(".")[1])]
        if len(others) == 1:
            DOE.set_path(base, f"phases.{others[0]}.fraction.value", round(rest, 9))
    rv = first.get("rve") or {}
    Ls = {round(float((r["form"].get("rve") or {}).get("L_um") or 0), 9) for r in runs}
    hs = {round(float((r["form"].get("rve") or {}).get("voxel_um") or 0), 9) for r in runs}
    grid = {"edge": "fixed" if len(Ls) == 1 and len(hs) == 1 else "own",
            "L_um": float(rv.get("L_um") or 0) or None, "voxel_um": float(rv.get("voxel_um") or 0) or None,
            "film": bool(rv.get("film")), "T_um": rv.get("T_um"), "quality": rv.get("quality")}
    meta = runs[0]["job"].meta or {}
    created = min((getattr(r["job"], "created", 0) or time.time()) for r in runs)
    return {"id": sid, "name": base["name"], "created": created, "owner": getattr(runs[0]["job"], "owner", None),
            "design": design, "parameters": params, "samples": len(cases), "seed": None, "grid": grid,
            "analyses": list(meta.get("analyses") or []), "base_form": base, "cases": cases, "dirs": [],
            "rebuilt": True}
