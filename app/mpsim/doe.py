"""Design of experiments over the input form.

A DOE varies numeric entries of the job form - filler content, particle size,
aspect ratio, interfacial resistance, voxel size, a material property - over
ranges and produces one complete form per case. The cases are ordinary runs,
so every result, figure and report of a DOE case is the same as for a case
entered by hand.

Designs:
    full    full factorial over all parameter levels
    oat     one factor at a time around the centre level (the reference case
            plus the levels of each parameter in turn)
    lhs     Latin hypercube sampling of the ranges (stratified, one sample per
            stratum per parameter)
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np

MAX_CASES = 64


def set_path(obj, path, value):
    keys = path.split(".")
    node = obj
    for k in keys[:-1]:
        if k.isdigit():
            node = node[int(k)]
        else:
            node = node.setdefault(k, {})
    last = keys[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


def get_path(obj, path):
    node = obj
    for k in path.split("."):
        if node is None:
            return None
        node = node[int(k)] if k.isdigit() else node.get(k)
    return node


def levels(p):
    """Explicit levels of one parameter."""
    if p.get("values"):
        return [float(v) for v in p["values"]]
    lo, hi = float(p["min"]), float(p["max"])
    n = max(int(p.get("steps", 3)), 1)
    if n == 1 or hi == lo:
        return [lo]
    if p.get("scale") == "log" and lo > 0 and hi > 0:
        return [float(x) for x in np.geomspace(lo, hi, n)]
    return [float(x) for x in np.linspace(lo, hi, n)]


def _from_unit(p, u):
    if p.get("values"):
        v = [float(x) for x in p["values"]]
        return v[min(int(u * len(v)), len(v) - 1)]
    lo, hi = float(p["min"]), float(p["max"])
    if p.get("scale") == "log" and lo > 0 and hi > 0:
        return float(10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo))))
    return float(lo + u * (hi - lo))


def _fmt(v):
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.3g}"
    return f"{round(v, 6):g}"


def expand(parameters, design="full", samples=12, seed=1):
    """-> list of {values: {path: value}, label: str}"""
    parameters = [p for p in parameters if p.get("path")]
    if not parameters:
        return []
    paths = [p["path"] for p in parameters]
    labels = [p.get("label") or p["path"] for p in parameters]
    combos = []
    if design == "lhs":
        n = max(int(samples), 2)
        rng = np.random.default_rng(int(seed))
        cols = []
        for p in parameters:
            u = (rng.permutation(n) + rng.random(n)) / n
            cols.append([_from_unit(p, x) for x in u])
        combos = [tuple(col[i] for col in cols) for i in range(n)]
    elif design == "oat":
        lv = [levels(p) for p in parameters]
        centre = tuple(v[len(v) // 2] for v in lv)
        combos = [centre]
        for i, v in enumerate(lv):
            for x in v:
                if x == centre[i]:
                    continue
                c = list(centre)
                c[i] = x
                combos.append(tuple(c))
    else:
        combos = list(itertools.product(*[levels(p) for p in parameters]))
    out = []
    for c in combos[:MAX_CASES]:
        values = {path: float(v) for path, v in zip(paths, c)}
        label = " · ".join(f"{lab} {_fmt(v)}" for lab, v in zip(labels, c))
        out.append({"values": values, "label": label})
    return out


def apply_case(base_form, case, index=None):
    """Complete form of one case (the base form with the parameters set)."""
    form = copy.deepcopy(base_form)
    for path, value in case["values"].items():
        set_path(form, path, value)
    base_name = (base_form.get("name") or "DOE case").strip()
    form["name"] = f"{base_name} · {case['label']}" if case.get("label") else base_name
    if index is not None:
        form["_doe_index"] = index
    return form


def parameter_catalogue(form):
    """Numeric entries of this form that a DOE can vary, for the browser."""
    out = [
        {"path": "rve.voxel_um", "label": "Voxel size", "unit": "µm", "group": "Domain"},
        {"path": "rve.L_um", "label": "RVE edge length", "unit": "µm", "group": "Domain"},
        {"path": "rve.N", "label": "Grid size", "unit": "voxels", "group": "Domain"},
        {"path": "rve.T_um", "label": "Film thickness (thin-film domain)", "unit": "µm", "group": "Domain"},
        {"path": "options.delta_T", "label": "ΔT", "unit": "K", "group": "Options"},
        {"path": "options.thickness_mm", "label": "Layer thickness", "unit": "mm", "group": "Options"},
        {"path": "options.filtration.face_velocity_m_s", "label": "Filtration face velocity", "unit": "m/s", "group": "Options"},
        {"path": "options.viscosity.gd_ref", "label": "Shear rate (viscosity)", "unit": "1/s", "group": "Options"},
        {"path": "matrix.props.k", "label": "Matrix thermal conductivity", "unit": "W/m·K", "group": "Matrix"},
        {"path": "matrix.props.E", "label": "Matrix Young's modulus", "unit": "GPa", "group": "Matrix"},
        {"path": "matrix.props.eps_r", "label": "Matrix permittivity", "unit": "-", "group": "Matrix"},
    ]
    for i, ph in enumerate(form.get("phases") or []):
        name = ph.get("name") or f"Phase {i+1}"
        g = f"{name}"
        out += [
            {"path": f"phases.{i}.fraction.value", "label": f"{name}: content", "unit": "%", "group": g},
            {"path": f"phases.{i}.size.d", "label": f"{name}: diameter", "unit": {"um": "µm"}.get(ph.get("size", {}).get("unit", "um"), ph.get("size", {}).get("unit", "um")), "group": g},
        ]
        if ph.get("shape") == "spheroid":
            out.append({"path": f"phases.{i}.size.aspect", "label": f"{name}: aspect ratio", "unit": "-", "group": g})
        un = ph.get("size", {}).get("unit", "um")
        if ph.get("shape") in ("cylinder", "spherocylinder"):
            out.append({"path": f"phases.{i}.size.length", "label": f"{name}: length", "unit": un, "group": g})
        if ph.get("shape") in ("cuboid", "superellipsoid"):
            out += [{"path": f"phases.{i}.size.ly", "label": f"{name}: edge b", "unit": un, "group": g},
                    {"path": f"phases.{i}.size.lz", "label": f"{name}: edge c", "unit": un, "group": g}]
        if ph.get("shape") == "superellipsoid":
            out.append({"path": f"phases.{i}.size.n", "label": f"{name}: blockiness n", "unit": "-", "group": g})
        if ph.get("shape") == "helix":
            out += [{"path": f"phases.{i}.size.coil_d", "label": f"{name}: coil diameter", "unit": un, "group": g},
                    {"path": f"phases.{i}.size.pitch", "label": f"{name}: pitch", "unit": un, "group": g},
                    {"path": f"phases.{i}.size.turns", "label": f"{name}: turns", "unit": "-", "group": g}]
        if (ph.get("dist") or {}).get("type") == "lognormal":
            out.append({"path": f"phases.{i}.dist.cv", "label": f"{name}: size CV", "unit": "-", "group": g})
        if (ph.get("orientation") or {}).get("mode") != "iso":
            out.append({"path": f"phases.{i}.orientation.spread_deg", "label": f"{name}: orientation spread", "unit": "°", "group": g})
        if (ph.get("shell") or {}).get("enabled"):
            out.append({"path": f"phases.{i}.shell.thickness", "label": f"{name}: shell thickness", "unit": ph.get("size", {}).get("unit", "um"), "group": g})
        out += [
            {"path": f"phases.{i}.r_int", "label": f"{name}: interfacial resistance R_int", "unit": "m²K/W", "group": g},
            {"path": f"phases.{i}.r_contact", "label": f"{name}: contact resistance R_c", "unit": "m²K/W", "group": g},
            {"path": f"phases.{i}.props.k", "label": f"{name}: thermal conductivity", "unit": "W/m·K", "group": g},
            {"path": f"phases.{i}.props.E", "label": f"{name}: Young's modulus", "unit": "GPa", "group": g},
        ]
    for p in out:
        p["current"] = get_path(form, p["path"])
    return out
