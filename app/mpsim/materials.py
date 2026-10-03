"""Material library, property validation and fraction bookkeeping.

A job never refers to the library by id alone: the browser sends the full
property set of every phase, so a value edited on screen is the value solved,
and a finished result stays reproducible even if the library changes later.
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "data", "materials.json")

# key -> (label, unit, minimum, maximum)
PROP_META = {
    "k":     ("thermal conductivity", "W/m·K", 0.0, 1e5),
    "sigma": ("electrical conductivity", "S/m", 0.0, 1e9),
    "eps_r": ("relative permittivity", "-", 1.0, 1e5),
    "tan_d": ("loss tangent", "-", 0.0, 10.0),
    "mu_r":  ("relative permeability", "-", 1.0, 1e6),
    "E":     ("Young's modulus", "GPa", 0.0, 5000.0),
    "nu":    ("Poisson's ratio", "-", 0.0, 0.4999),
    "alpha": ("coefficient of thermal expansion", "ppm/K", -100.0, 1000.0),
    "rho":   ("density", "g/cm³", 0.0, 30.0),
    "cp":    ("specific heat", "J/kg·K", 0.0, 1e4),
}
PROP_KEYS = list(PROP_META)
# Optional: only the analyses that use them ask for them, and a material
# without them is taken as not absorbing water and without a glass transition.
OPTIONAL_META = {
    "D_w":    ("moisture diffusivity", "m²/s", 0.0, 1e-3),
    "c_sat":  ("saturated moisture content", "wt%", 0.0, 100.0),
    "cme":    ("coefficient of moisture expansion", "1/(mass fraction)", 0.0, 10.0),
    "tg":     ("glass transition temperature", "°C", -150.0, 600.0),
    "E_r":    ("Young's modulus above Tg (rubbery)", "GPa", 0.0, 100.0),
    "alpha2": ("CTE above Tg", "ppm/K", -100.0, 2000.0),
}


def load_library():
    with open(LIB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def library_by_id():
    return {m["id"]: m for m in load_library()["materials"]}


def clean_props(props, where="material"):
    """Validated float copy of a property dict. Raises ValueError listing
    every problem at once, so the user fixes a form in one pass."""
    out, errors = {}, []
    for key, (label, unit, lo, hi) in PROP_META.items():
        if key not in props or props[key] is None or props[key] == "":
            errors.append(f"{where}: the {label} is missing")
            continue
        try:
            v = float(props[key])
        except (TypeError, ValueError):
            errors.append(f"{where}: the {label} is not a number ({props[key]!r})")
            continue
        if not math.isfinite(v) or v < lo or v > hi:
            errors.append(f"{where}: the {label} of {v:g} {unit} lies outside the "
                          f"permitted range [{lo:g}, {hi:g}]")
            continue
        out[key] = v
    for key, (label, unit, lo, hi) in OPTIONAL_META.items():
        if props.get(key) in (None, ""):
            continue
        try:
            v = float(props[key])
        except (TypeError, ValueError):
            errors.append(f"{where}: the {label} is not a number ({props[key]!r})")
            continue
        if not math.isfinite(v) or v < lo or v > hi:
            errors.append(f"{where}: the {label} of {v:g} {unit} lies outside the permitted range [{lo:g}, {hi:g}]")
            continue
        out[key] = v
    if errors:
        raise ValueError("\n".join(errors))
    return out


def resolve_fractions(matrix_props, phases):
    """Volume fraction of every phase and of the matrix.

    Each phase states its amount as vol % or wt % of the *whole composite*.
    Pores are naturally given in vol %, fillers often in wt %, and both may
    appear together, so the conversion is solved exactly rather than by the
    common two-phase shortcut:

        volume:  sum_vol v_i + M sum_wt w_j/rho_j + m_m/rho_m = 1
        mass:    m_m = M (1 - sum_wt w_j) - sum_vol rho_i v_i

    with M the composite mass per unit volume (= density). Returns a dict with
    volume and mass fractions and the composite density / specific heat.
    """
    rho_m = matrix_props["rho"]
    vol_v, wt_w = {}, {}
    for i, ph in enumerate(phases):
        val = float(ph["fraction"]["value"]) / 100.0
        if val < 0:
            raise ValueError(f"Phase {i+1}: the content is negative")
        if ph["fraction"].get("basis", "vol") == "wt":
            wt_w[i] = val
        else:
            vol_v[i] = val
    sum_v = sum(vol_v.values())
    sum_w = sum(wt_w.values())
    if sum_w >= 1.0:
        raise ValueError("The mass fractions add up to 100 % or more")
    rho = {i: ph["props"]["rho"] for i, ph in enumerate(phases)}
    if wt_w:
        if rho_m <= 0 or any(rho[j] <= 0 for j in wt_w):
            raise ValueError("A conversion from wt % requires the densities of the matrix and of the phase")
        denom = sum(w / rho[j] for j, w in wt_w.items()) + (1.0 - sum_w) / rho_m
        num = 1.0 - sum_v + sum(rho[i] * v for i, v in vol_v.items()) / rho_m
        M = num / denom
    else:
        M = None
    vf = [0.0] * len(phases)
    for i, v in vol_v.items():
        vf[i] = v
    for j, w in wt_w.items():
        vf[j] = w * M / rho[j]
    vf_matrix = 1.0 - sum(vf)
    if vf_matrix < 0.0 - 1e-12:
        raise ValueError(f"The phase volume fractions add up to {100*sum(vf):.1f} %, which exceeds 100 %")
    vf_matrix = max(vf_matrix, 0.0)

    mass = [vf[i] * rho[i] for i in range(len(phases))]
    m_matrix = vf_matrix * rho_m
    density = m_matrix + sum(mass)
    wt = ([m / density for m in mass] if density > 0 else [0.0] * len(phases))
    cp = ((m_matrix * matrix_props["cp"]
           + sum(mass[i] * phases[i]["props"]["cp"] for i in range(len(phases))))
          / density if density > 0 else 0.0)
    return {
        "vf": vf, "vf_matrix": vf_matrix,
        "wt": wt, "wt_matrix": (m_matrix / density if density > 0 else 0.0),
        "density": density, "cp": cp,
    }
