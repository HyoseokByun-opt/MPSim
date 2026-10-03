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
import time
import uuid

from mpsim import doe as DOE


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
        parameters = payload.get("parameters") or []
        design = payload.get("design") if payload.get("design") in ("full", "oat", "lhs") else "full"
        cases = DOE.expand(parameters, design, int(payload.get("samples", 12)), int(payload.get("seed", 1)))
        if not cases:
            raise ValueError("No parameter with a range was given")
        forms = []
        for i, c in enumerate(cases):
            form = DOE.apply_case(base, c, i)
            normalize(copy.deepcopy(form))          # raises ValueError on a bad case
            forms.append(form)
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        study = {"id": sid, "name": payload.get("name") or (base.get("name") or "DOE study"),
                 "created": time.time(), "owner": owner, "design": design,
                 "parameters": parameters, "samples": payload.get("samples"), "seed": payload.get("seed"),
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
            row = {"job_id": case["job_id"], "label": case["label"], "values": case["values"],
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
                "parameters": [{"path": p["path"], "label": p.get("label") or p["path"], "unit": p.get("unit", "")}
                               for p in study["parameters"]],
                "analyses": study.get("analyses") or [], "metrics": metrics, "rows": rows}

    def csv(self, study, queue):
        rows, metrics = self.rows(study, queue)
        params = [{"path": p["path"], "label": p.get("label") or p["path"]} for p in study["parameters"]]
        buf = io.StringIO()
        head = ["case", "run_id", "state"] + [p["label"] for p in params] + [f"{m['label']} [{m['unit']}]" for m in metrics]
        buf.write(",".join(f'"{h}"' for h in head) + "\n")
        for i, r in enumerate(rows, start=1):
            cells = [str(i), r["job_id"], r["state"]]
            cells += [str(r["values"].get(p["path"], "")) for p in params]
            cells += [("" if r["metrics"].get(m["key"]) is None else repr(r["metrics"][m["key"]])) for m in metrics]
            buf.write(",".join(f'"{c}"' for c in cells) + "\n")
        return buf.getvalue()
