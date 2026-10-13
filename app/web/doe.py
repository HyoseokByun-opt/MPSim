"""DOE studies: a set of runs created from one base form and a design.

A study is a small JSON file in `workspace/doe/`; the runs themselves are
ordinary jobs in the queue, so a study survives a server restart exactly as
single runs do and its cases can be opened, compared and deleted individually.
"""
from __future__ import annotations

import copy
import io
import json
import os
import re
import time
import uuid

from mpsim import doe as DOE


def realised(study, r):
    """What a finished case really holds, beside what was asked for: the
    filler content and, for every parameter (or linked entry) that sets a
    content or a share of the filler, the value of the generated structure.
    A dense packing can fall short of the target (BN stacks at half the
    filler: 51 of 55 vol%); a study is read against these, not the inputs.
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
        try:
            os.remove(self._path(sid))
            return True
        except OSError:
            return False

    # -------------------------------------------------------------- create
    def create(self, owner, payload, queue, runs_dir, normalize, folder=None, case_name="",
               estimate=None):
        """Expand the design, validate every case and queue the runs."""
        base = copy.deepcopy(payload.get("form") or {})
        base.pop("_limits", None)
        # the grid of the cases: adaptive (each case planned for its own
        # particles, as a single run) or one fixed grid for every case, so
        # that only the varied parameters differ between them
        grid = payload.get("grid") or {}
        if grid.get("mode") == "fixed":
            try:
                L_um, h_um = float(grid["L_um"]), float(grid["voxel_um"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("A fixed grid needs the RVE edge length and the voxel size")
            if not (L_um > 0 and h_um > 0 and L_um / h_um >= 8):
                raise ValueError("A fixed grid needs an RVE edge of at least 8 voxels")
            rv = base.setdefault("rve", {})
            rv.update(auto=False, L_um=L_um, voxel_um=h_um, auto_enlarge=False, N=None)
            if grid.get("seeds"):
                rv["n_seeds"] = int(grid["seeds"])
            grid = {"mode": "fixed", "L_um": L_um, "voxel_um": h_um, "N": int(round(L_um / h_um))}
        else:
            grid = {"mode": "adaptive"}
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
        forms = []
        for i, c in enumerate(cases):
            try:
                form = DOE.apply_case(base, c, i, parameters=parameters)
            except ValueError as e:
                raise ValueError(f"Case {i + 1} ({c['label']}): {e}") from None
            normalize(copy.deepcopy(form))          # raises ValueError on a bad case
            forms.append(form)
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        study = {"id": sid, "name": payload.get("name") or (base.get("name") or "DOE study"),
                 "created": time.time(), "owner": owner, "design": design,
                 "parameters": parameters, "samples": payload.get("samples"), "seed": payload.get("seed"),
                 "grid": grid,
                 "analyses": [k for k, v in (base.get("analyses") or {}).items() if v],
                 "base_form": base, "cases": []}
        for form, case in zip(forms, cases):
            meta = {"name": form["name"], "analyses": study["analyses"], "doe": sid,
                    "mem_gb": estimate(form) if estimate else 0.0,
                    # the variable values of this case, for the run list
                    "doe_label": case["label"], "kind": "analysis" if study["analyses"] else "structure"}
            # `case` here is the design point; the folder the run lands in is case_name
            job = queue.submit(owner, form, runs_dir, meta, folder=folder, case=case_name)
            study["cases"].append({"job_id": job.id, "values": case["values"], "label": case["label"]})
        self.save(study)
        return study

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
                "analyses": study.get("analyses") or [], "metrics": metrics, "rows": rows}

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
