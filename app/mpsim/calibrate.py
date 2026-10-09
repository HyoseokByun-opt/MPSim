"""Calibration: unknown material parameters from measured effective properties.

The problem
-----------
A measured effective property - the thermal conductivity of an epoxy with
50 vol% alumina, say - depends on everything in the model. Most of it is
known: the resin and filler values from data sheets, the microstructure from
the recipe. A few things are not: the filler-matrix interfacial resistance,
the contact resistance between particles, the conductivity of one filler
grade. Calibration finds the unknowns that make the RVE reproduce the
measurements, and says how well the measurements determine them.

The method and why
------------------
1. Theory first. What theory gives is not fitted. The acoustic and diffuse
   mismatch models (Swartz and Pohl, Rev. Mod. Phys. 61 (1989) 605) give the
   phonon-limited interfacial resistance from the sound speeds and heat
   capacity of both materials - the value of a perfect interface and so a
   lower bound for a real one (adhesion, silane layers and voids add to it).
   A value from MD or DFT, or from the literature, can be entered instead and
   either fixes the parameter or centres its prior.
2. Identifiability. With one measurement and two unknowns the answer is a
   curve, not a point. In the dilute limit the conductivity depends on the
   filler conductivity k_p and the interfacial resistance R only through
   k_p / (1 + 2 R k_p / d) (Hasselman and Johnson, J. Compos. Mater. 21 (1987)
   508): loadings at one particle size cannot separate them; two particle
   sizes can (Every et al., Acta Metall. Mater. 40 (1992) 123). The sensitivity
   matrix of the measurements to the unknowns shows which combinations the
   data determine; the collinearity index of Brun, Reichert and Kuensch
   (Water Resour. Res. 37 (2001) 1015) flags a set that is not identifiable.
3. RVE surrogate. The structure of every measured sample is generated once;
   the unknowns change material values only, so one evaluation is a solve on
   a fixed structure. A design of solves on a log scale is fitted by a
   Gaussian process per measurement (Kennedy and O'Hagan, J. R. Stat. Soc. B
   63 (2001) 425), refined where the posterior lies, and the posterior is
   evaluated on a dense grid of the surrogate.
4. Bayesian answer. Posterior = prior x likelihood; the likelihood carries
   the measurement uncertainty, the RVE's own scatter and the surrogate's
   error. Reported: the maximum-posterior estimate, median and 95 % interval
   per unknown, their correlation, how much the data narrowed each one, a
   direct solve at the estimate, and the extra measurement that would best
   separate what is not yet separated.
5. Viscosity. The particle surface parameters of the particle dynamics -
   the bound resin layer, the work of adhesion, friction, roughness, the
   Hamaker constant - set how much finer fillers thicken a compound and are
   rarely known; a measured viscosity (relative, or of the compound at a
   shear rate) calibrates them. A sample is then a set of spheres, not a
   voxel structure: the same spheres for every parameter set (common random
   numbers, so the response is smooth), one particle-dynamics run per
   evaluation, its statistical error added to the measurement's.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import re
import time

import numpy as np

from . import analytical as AN
from . import doe as DOE
from . import generate as GEN
from . import materials as M
from . import parallel as PAR
from . import pipeline as PL
from . import rve as R
from . import spec as S
from .solvers import conduction as CD

MAX_UNKNOWNS = 3
# a viscosity sample whose effective content (filler + bound resin layer)
# reaches this share of the random close packing of its spheres is taken as
# jammed: frictional spheres jam a few per cent below the frictionless
# packing (Boyer et al. 2011: 0.585 against 0.64 for equal spheres)
JAM_SHARE = 0.95
N_INIT = {1: 5, 2: 13, 3: 24}
GRID = {1: 801, 2: 161, 3: 51}

PROPERTIES = {
    "k": {"label": "Thermal conductivity", "unit": "W/m·K", "solver": "cond", "key": "thermal"},
    "sigma": {"label": "Electrical conductivity", "unit": "S/m", "solver": "cond", "key": "electrical"},
    "eps_r": {"label": "Relative permittivity", "unit": "-", "solver": "cond", "key": "dielectric"},
    "E": {"label": "Young's modulus", "unit": "GPa", "solver": "elastic"},
    "alpha": {"label": "Thermal expansion (CTE)", "unit": "ppm/K", "solver": "elastic"},
    "mu_r": {"label": "Relative viscosity (compound / resin)", "unit": "-", "solver": "dem"},
    "mu": {"label": "Viscosity of the compound", "unit": "Pa·s", "solver": "dem"},
}

# what may be calibrated: material values and interface resistances, which
# leave the structure as it is (sizes and contents change the structure and
# are measurement conditions instead)
_UNKNOWN_RE = [
    (re.compile(r"^matrix\.props\.(k|sigma|eps_r|E|nu|alpha)$"), "matrix"),
    (re.compile(r"^phases\.(\d+)\.props\.(k|sigma|eps_r|E|nu|alpha)$"), "phase"),
    (re.compile(r"^phases\.(\d+)\.shell\.props\.(k|sigma|eps_r|E|nu|alpha)$"), "shell"),
    (re.compile(r"^phases\.(\d+)\.(r_int|r_contact)$"), "interface"),
    (re.compile(r"^contact_rc\.(\d+)-(\d+)$"), "pair"),
    # the particle surface of the particle dynamics (viscosity measurements)
    (re.compile(r"^options\.viscosity\.dem\.(bound_nm|adhesion_mJ_m2|mu_f|roughness_nm|hmin_nm|hamaker_J)$"), "dem"),
]
UNIT_TXT = {"nm": "nm", "um": "µm", "mm": "mm"}
PROP_UNITS = {"k": "W/m·K", "sigma": "S/m", "eps_r": "-", "E": "GPa", "nu": "-", "alpha": "ppm/K",
              "r_int": "m²K/W", "r_contact": "m²K/W", "bound_nm": "nm", "adhesion_mJ_m2": "mJ/m²", "mu_f": "-",
              "roughness_nm": "nm", "hmin_nm": "nm", "hamaker_J": "J"}


def unknown_kind(path):
    for rx, kind in _UNKNOWN_RE:
        if rx.match(path):
            return kind
    return None


def fill_props(form):
    """The form with every material's full property set (a form may name a
    material by id only), so one property can be changed without losing the
    others."""
    form = copy.deepcopy(form)
    # the analyses of the case do not matter here (each measurement says what
    # is solved), but a case must name one to be valid
    if not any((form.get("analyses") or {}).values()):
        form["analyses"] = {"thermal": True}
    lib = M.library_by_id()
    S._fill_from_library(form.get("matrix"), lib)
    for ph in form.get("phases") or []:
        S._fill_from_library(ph, lib)
        if (ph.get("shell") or {}).get("enabled"):
            S._fill_from_library(ph["shell"], lib)
    return form


# =========================================================================
# theory: phonon-limited interfacial resistance
# =========================================================================
def sound_speeds(E_gpa, nu, rho_gcc):
    """Longitudinal and transverse sound speeds [m/s] of an isotropic solid."""
    E, rho = float(E_gpa) * 1e9, float(rho_gcc) * 1e3
    nu = float(nu)
    if E <= 0 or rho <= 0:
        return None
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    return math.sqrt((K + 4.0 * G / 3.0) / rho), math.sqrt(G / rho)


def _amm_gamma(z1, z2, v1, v2, n=400):
    """Angle-averaged acoustic-mismatch transmission 2 int alpha cos sin."""
    th = (np.arange(n) + 0.5) * (0.5 * math.pi / n)
    s2 = v2 / v1 * np.sin(th)
    ok = s2 < 1.0
    c1 = np.cos(th)
    c2 = np.sqrt(np.clip(1.0 - s2 * s2, 0.0, 1.0))
    a = np.where(ok, 4.0 * z1 * z2 * c1 * c2 / (z1 * c1 + z2 * c2) ** 2, 0.0)
    return float(np.sum(a * 2.0 * np.sin(th) * c1) * (0.5 * math.pi / n))


def kapitza(p1, p2):
    """Phonon-limited interfacial resistance [m^2 K/W] between material 1 (the
    side the phonons come from, the matrix) and material 2, by the acoustic
    (AMM) and diffuse (DMM) mismatch models in the Debye, high-temperature
    form: h = 1/4 sum_j C_1j v_1j Gamma_j with the heat capacity shared
    equally by the three branches (Swartz and Pohl 1989). Order-of-magnitude
    values: a polymer's heat capacity includes modes that carry no heat, and
    a real interface adds adhesion defects, so a measured R is at or above
    these."""
    s1 = sound_speeds(p1.get("E", 0), p1.get("nu", 0.3), p1.get("rho", 0))
    s2 = sound_speeds(p2.get("E", 0), p2.get("nu", 0.3), p2.get("rho", 0))
    if s1 is None or s2 is None or not p1.get("cp"):
        return None
    C1 = float(p1["rho"]) * 1e3 * float(p1["cp"])
    v1 = [s1[0], s1[1], s1[1]]
    v2 = [s2[0], s2[1], s2[1]]
    z1 = [float(p1["rho"]) * 1e3 * v for v in v1]
    z2 = [float(p2["rho"]) * 1e3 * v for v in v2]
    h_amm = 0.25 * sum(C1 / 3.0 * v1[j] * _amm_gamma(z1[j], z2[j], v1[j], v2[j]) for j in range(3))
    a12 = sum(v ** -2 for v in v2) / (sum(v ** -2 for v in v1) + sum(v ** -2 for v in v2))
    h_dmm = 0.25 * a12 * sum(C1 / 3.0 * v for v in v1)
    return {"amm": 1.0 / h_amm if h_amm > 0 else None, "dmm": 1.0 / h_dmm if h_dmm > 0 else None,
            "v_matrix": [s1[0], s1[1]], "v_filler": [s2[0], s2[1]]}


def theory(form):
    """Theory values for the calibration editor: per filler the mismatch-model
    interfacial resistance with the matrix, the Kapitza radius and the
    particle diameter below which the filler stops helping."""
    form = fill_props(form)
    mp = (form.get("matrix") or {}).get("props") or {}
    out = []
    for i, ph in enumerate(form.get("phases") or []):
        pp = ph.get("props") or {}
        outer = (ph.get("shell") or {}).get("props") if (ph.get("shell") or {}).get("enabled") else pp
        kz = kapitza(mp, outer or pp)
        row = {"phase": i, "name": ph.get("name") or ph.get("material_id") or f"Phase {i+1}", "kapitza": kz}
        km, kp = float(mp.get("k") or 0), float(pp.get("k") or 0)
        if kz and kz.get("dmm") and km > 0:
            r = kz["dmm"]
            row["kapitza_radius_um"] = r * km * 1e6
            if kp > km:
                row["d_crit_um"] = 2.0 * r * km * kp / (kp - km) * 1e6
        out.append(row)
    return {"pairs": out}


# =========================================================================
# effective-medium model with interfacial resistance (quick preview and
# the design of further measurements)
# =========================================================================
def dem_conductivity(km, fillers):
    """Differential effective medium (asymmetric Bruggeman) for spheres with
    each filler as its Hasselman-Johnson equivalent k* = k / (1 + 2 R k / d);
    with one filler this is the model of Every et al. 1992.
    fillers: [(phi, k, R, d_m)]."""
    phi_t = sum(f[0] for f in fillers)
    if phi_t <= 0:
        return km
    kst = [(f[0] / phi_t, f[1] / (1.0 + 2.0 * f[2] * f[1] / max(f[3], 1e-30))) for f in fillers]
    n = 400
    lk = math.log(km)
    dt = phi_t / n
    t = 0.0

    def rate(t_, lk_):
        k = math.exp(lk_)
        return sum(w * 3.0 * (ks - k) / (ks + 2.0 * k) for w, ks in kst) / (1.0 - t_)
    for _ in range(n):
        a = rate(t, lk)
        b = rate(t + 0.5 * dt, lk + 0.5 * dt * a)
        c = rate(t + 0.5 * dt, lk + 0.5 * dt * b)
        d = rate(t + dt, lk + dt * c)
        lk += dt * (a + 2 * b + 2 * c + d) / 6.0
        t += dt
    return math.exp(lk)


def _theory_k(spec):
    """Thermal conductivity of a normalised spec by the DEM model, or None
    when it does not apply (pores, networks)."""
    km = float(spec["matrix"]["props"]["k"])
    fillers = []
    for ph in spec["phases"]:
        if ph["void"] or ph["shape"] == "network":
            return None
        d = float(ph["size_um"].get("d") or 1.0) * 1e-6
        k = float(ph["props"]["k"])
        if ph.get("shell"):
            d += 2.0 * ph["shell"]["thickness_um"] * 1e-6
        fillers.append((float(ph["vf"]), k, float(ph.get("r_int") or 0.0), d))
    return dem_conductivity(km, fillers)


# =========================================================================
# the job
# =========================================================================
def _validate(job):
    errors = []
    meas = job.get("measurements") or []
    unk = job.get("unknowns") or []
    if not meas:
        errors.append("Enter at least one measurement")
    if not unk:
        errors.append("Choose at least one unknown to calibrate")
    if len(unk) > MAX_UNKNOWNS:
        errors.append(f"At most {MAX_UNKNOWNS} unknowns at a time: more are not determined by a few "
                      f"effective-property measurements; fix the others at theory or literature values")
    seen = set()
    for u in unk:
        p = u.get("path", "")
        if unknown_kind(p) is None:
            errors.append(f"'{p}' cannot be calibrated: only material values, interface resistances and the "
                          f"particle surface parameters of the particle dynamics (sizes and contents are "
                          f"measurement conditions)")
        if p in seen:
            errors.append(f"'{p}' is listed twice")
        seen.add(p)
        lo, hi = float(u.get("lo", 0)), float(u.get("hi", 0))
        if not (hi > lo):
            errors.append(f"'{u.get('label') or p}': the upper bound must exceed the lower bound")
        if u.get("scale", "log") == "log" and lo <= 0:
            errors.append(f"'{u.get('label') or p}': a log scale needs a positive lower bound")
    for i, m in enumerate(meas):
        if m.get("property") not in PROPERTIES:
            errors.append(f"Measurement {i+1}: unknown property '{m.get('property')}'")
        if not (float(m.get("value") or 0) > 0):
            errors.append(f"Measurement {i+1}: the measured value must be positive")
        for path in (m.get("set") or {}):
            if path in seen:
                errors.append(f"Measurement {i+1} sets '{path}', which is being calibrated")
    if errors:
        raise ValueError("\n".join(errors))


class _Unknown:
    def __init__(self, u):
        self.path = u["path"]
        self.label = u.get("label") or u["path"]
        self.lo, self.hi = float(u["lo"]), float(u["hi"])
        self.log = u.get("scale", "log") == "log"
        self.prior = u.get("prior", "uniform")          # uniform | normal
        self.center = u.get("center")
        self.sigma_dec = float(u.get("sigma_dec") or 0.5)   # log scale: decades; linear: in units
        self.theory = u.get("theory")
        self.unit = u.get("unit") or PROP_UNITS.get(self.path.split(".")[-1], "")

    def value(self, uu):
        uu = np.asarray(uu, float)
        if self.log:
            return np.exp(math.log(self.lo) + uu * (math.log(self.hi) - math.log(self.lo)))
        return self.lo + uu * (self.hi - self.lo)

    def unit_of(self, v):
        v = np.asarray(v, float)
        if self.log:
            return (np.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def log_prior(self, uu):
        uu = np.asarray(uu, float)
        if self.prior != "normal" or self.center in (None, ""):
            return np.zeros_like(uu)
        c = float(self.center)
        v = self.value(uu)
        if self.log:
            z = (np.log10(v) - math.log10(c)) / self.sigma_dec
        else:
            z = (v - c) / self.sigma_dec
        return -0.5 * z * z

    def dlnv_du(self):
        """d ln(value) / du (log scale), for sensitivities per factor."""
        return math.log(self.hi) - math.log(self.lo) if self.log else None


def _design(p, n, seed=3):
    """Initial design in the unit cube: the ends and the middle for one
    unknown; a 3^p grid plus Latin-hypercube points for more."""
    if p == 1:
        return np.linspace(0.0, 1.0, n)[:, None]
    grid = np.array(list(itertools.product([0.0, 0.5, 1.0], repeat=p)))
    rng = np.random.default_rng(seed)
    extra = max(0, n - len(grid))
    if p == 3:
        grid = grid[np.sum(grid == 0.5, axis=1) != 1]          # corners, centre, face centres
        extra = max(0, n - len(grid))
    if extra:
        lhs = (np.argsort(rng.random((extra, p)), axis=0) + rng.random((extra, p))) / extra
        grid = np.vstack([grid, 0.05 + 0.9 * lhs])
    return grid


def _set_theta(form, unknowns, theta):
    f = copy.deepcopy(form)
    for u, v in zip(unknowns, theta):
        if u.path.startswith("contact_rc."):
            f.setdefault("contact_rc", {})[u.path.split(".", 1)[1]] = float(v)
        else:
            DOE.set_path(f, u.path, float(v))
    return f


def _prepare(job, base, unknowns, out_dir, limits, log, should_stop):
    """One sample per measurement: its spec, plan and the structure
    (generated once, kept on disk for a later calibration of the same
    samples)."""
    samples = []
    reuse_dirs = [os.path.join(out_dir)] + list(job.get("_structure_dirs") or [])
    any_contact = any(u.path.endswith("r_contact") or u.path.startswith("contact_rc.") for u in unknowns)
    mid = [u.value(0.5) for u in unknowns]
    for i, m in enumerate(job["measurements"]):
        form = copy.deepcopy(base)
        for path, v in (m.get("set") or {}).items():
            DOE.set_path(form, path, v)
        form = _set_theta(form, unknowns, mid)
        if any_contact:
            # the contact faces must be recorded even if the value set is 0
            for ph in form.get("phases") or []:
                ph["r_contact"] = ph.get("r_contact") or 1e-12
        spec = S.normalize(form)
        if PROPERTIES[m["property"]]["solver"] == "dem":
            samples.append(_dem_sample(i, m, form, spec, log))
            continue
        need = {PROPERTIES[m["property"]].get("key") or "cte"}
        if PROPERTIES[m["property"]]["solver"] == "elastic":
            need = {"cte"}
        plan = R.plan(PL._plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"]), need,
                      limits or {})
        film = bool(plan.get("film"))
        plan["skin"] = PL.film_skin(spec, plan) if film else None
        gen_phases = PL._generator_phases(spec)
        want_cf = PL.wants_contact_faces(spec)
        key = PL.structure_key(gen_phases, plan, spec["options"]["contacts"], spec["rve"]["seed"], want_cf)
        got = PL.load_structure(reuse_dirs, key)
        if got is not None:
            gen_labels, ginfo, _ = got
            log(f"   sample {i+1}: structure {plan['N']}³ reused")
        else:
            gen_labels, ginfo = GEN.generate(plan["N"], plan["h_um"], gen_phases, seed=spec["rve"]["seed"],
                                             log=log, should_stop=should_stop, contacts=spec["options"]["contacts"],
                                             want_contact_faces=want_cf, nz=int(plan["Nz"]) if film else None,
                                             skin=plan.get("skin"))
            try:
                PL.save_structure(out_dir, key, gen_labels, ginfo, ginfo.get("contact_mask"))
            except Exception as e:                                       # noqa: BLE001
                log(f"   (structure not kept: {e})")
        labels, area_ratio, _ = PL.build_labels(spec, gen_labels, ginfo, plan["h_um"])
        cmask = ginfo.pop("contact_mask", None)
        vf = np.bincount(labels.ravel(), minlength=len(spec["labels"])) / labels.size
        log(f"   sample {i+1} ({m.get('label') or 'measurement ' + str(i + 1)}): grid {plan['N']}³, voxel "
            f"{plan['h_um']:.4g} µm, " + ", ".join(f"{t['name']} {100 * v:.1f}%" for t, v in zip(spec["labels"], vf)))
        samples.append({"i": i, "m": m, "form": form, "spec": spec, "plan": plan, "labels": labels,
                        "lab_path": None, "cmask": cmask, "cm_path": None, "area_ratio": area_ratio,
                        "vf": vf.tolist(), "film": film, "theory_ok": _theory_k(spec) is not None})
    return samples


def _dem_sample(i, m, form, spec, log):
    """A viscosity sample: the spheres of the particle dynamics, drawn once
    and kept for every parameter set."""
    solid = [k for k, ph in enumerate(spec["phases"]) if not ph["void"] and ph["shape"] != "network"]
    name = m.get("label") or f"measurement {i + 1}"
    if not solid:
        raise ValueError(f"{name}: a viscosity measurement needs a filler")
    if any(spec["phases"][k]["shape"] != "sphere" for k in solid) or any(ph["void"] for ph in spec["phases"]):
        raise ValueError(f"{name}: the particle dynamics, which a viscosity calibration runs, handles spheres "
                         f"without bubbles")
    vo = spec["options"]["viscosity"]
    radii, owner = PL._dem_radii(spec, solid, vo["dem"]["n"])
    phi = float(sum(spec["phases"][k]["vf"] for k in solid))
    if not 0.0 < phi < 0.64:
        raise ValueError(f"{name}: the filler content {100 * phi:.1f} vol% is outside the particle dynamics' "
                         f"range (below random close packing, 64 %)")
    from .solvers import viscosity as VI
    phi_jam = JAM_SHARE * float(VI.rcp_farr_groot(2.0 * radii))
    log(f"   sample {i+1} ({name}): {len(radii)} spheres, φ {phi:.3f}, shear rate {vo['gd_ref']:g} 1/s "
        f"(particle dynamics; jammed from an effective content of {phi_jam:.3f})")
    # sums of the radius powers: the effective content with a bound layer b
    # is phi * sum((r + b)^3) / sum(r^3), for any number of b at once
    mom = [float(np.sum(radii ** k)) for k in range(4)]
    return {"i": i, "m": m, "form": form, "spec": spec, "plan": None, "labels": None, "lab_path": None, "cmask": None,
            "cm_path": None, "area_ratio": None, "vf": [1.0 - phi] + [float(spec["phases"][k]["vf"]) for k in solid],
            "film": False, "theory_ok": False, "dem": {"solid": solid, "radii": radii, "owner": owner, "phi": phi,
                                                       "phi_jam": phi_jam, "mom": mom},
            "dem_se": []}


def _dem_phi_eff(sample, bound_nm):
    """Effective content of a viscosity sample for bound layers (nm)."""
    d = sample["dem"]
    b = np.asarray(bound_nm, float) * 1e-9
    m0, m1, m2, m3 = d["mom"]
    return d["phi"] * (m3 + 3 * b * m2 + 3 * b * b * m1 + b ** 3 * m0) / m3


def _dem_jammed(sample, unknowns, Ug):
    """Grid points (unit cube) at which a viscosity sample would be jammed."""
    j = next((k for k, u in enumerate(unknowns) if u.path == "options.viscosity.dem.bound_nm"), None)
    if j is None:
        b = np.full(len(Ug), float(sample["spec"]["options"]["viscosity"]["dem"]["bound_nm"]))
    else:
        b = unknowns[j].value(Ug[:, j])
    return _dem_phi_eff(sample, b) >= sample["dem"]["phi_jam"]


def _dem_value(sample, theta_form, should_stop, cb=None):
    """The measured viscosity of a viscosity sample at one parameter set: one
    particle-dynamics run on the sample's spheres. Returns the value and its
    relative standard error."""
    from .solvers import suspension as SU
    from .solvers import viscosity as VI
    spec = S.normalize(theta_form)
    vo = spec["options"]["viscosity"]
    do = vo["dem"]
    d = sample["dem"]
    if float(_dem_phi_eff(sample, do["bound_nm"])) >= d["phi_jam"]:
        return float("nan"), float("nan")
    hks, _ = PL.dem_hamaker(spec, d["solid"], do)
    gd = vo["gd_ref"]
    mu_res = float(VI.matrix_flow(vo["model"], [gd])[0])
    out = SU.shear_viscosity(d["radii"], d["phi"], mu_res, gd, roughness_m=do["roughness_nm"] * 1e-9,
                             hmin_m=do["hmin_nm"] * 1e-9, hamaker_J=np.asarray(hks)[d["owner"]], mu_f=do["mu_f"],
                             strain=do["strain"], backend=do["backend"], progress=cb, should_stop=should_stop,
                             bound_m=do["bound_nm"] * 1e-9, adhesion_J_m2=do["adhesion_mJ_m2"] * 1e-3, frames=0.0)
    mu_r = float(out["eta_mean"])
    rel = float(out["eta_se"]) / max(mu_r, 1e-30)
    if sample["m"]["property"] == "mu":
        return float(VI.suspension_flow(vo["model"], mu_r, d["phi"], [gd], vo["yield_Pa"])[0][0]), rel
    return mu_r, rel


def _extra_unc(s):
    """The statistical error of a particle-dynamics sample (ln units): the
    median relative standard error of its runs."""
    return float(np.median(s["dem_se"])) if s.get("dem_se") else 0.0


def _dirs(m):
    d = m.get("direction") or "iso"
    return ["x", "y", "z"] if d == "iso" else [d]


def _cond_tasks(sample, theta_form, tol, pool, tag):
    """Pool tasks of one sample at one parameter set."""
    m = sample["m"]
    prop = m["property"]
    key = PROPERTIES[prop]["key"]
    spec = S.normalize(theta_form)
    h = sample["plan"]["h_um"]
    if prop == "k":
        vals = PL.thermal_values(spec)
        rpair, rcpair = PL.thermal_interfaces(spec, h, sample["area_ratio"])
    else:
        vals = np.array(S.label_values(spec["labels"], prop), float)
        rpair = rcpair = None
    tasks = []
    cache = {}
    for d in _dirs(m):
        di = "xyz".index(d)
        v_used, _ = CD.contrast_policy(sample["labels"], vals, di, spec["options"]["contrast_cap"], cache)
        tasks.append((PAR.cond_task, {"labels": sample["lab_path"], "vals": v_used, "d": d, "tol": tol,
                                      "backend": "fv", "rpair": rpair, "rcpair": rcpair,
                                      "cmask": sample["cm_path"] if rcpair is not None else None,
                                      "voxel_m": h * 1e-6, "fields": False, "tmp": pool.tmp, "key": key,
                                      "kind": PL.COND[key]["kind"], "film": sample["film"],
                                      "name": f"{tag} · sample {sample['i'] + 1} {d}"}))
    return tasks


def _elastic_value(sample, theta_form, tol, should_stop):
    from .solvers import elastic as EL
    from .solvers import fans as FA
    m = sample["m"]
    spec = S.normalize(theta_form)
    table = spec["labels"]
    E = S.label_values(table, "E")
    nu = S.label_values(table, "nu")
    al = S.label_values(table, "alpha")
    r = FA.homogenize(sample["labels"], E, nu, al, tol=tol, should_stop=should_stop)
    d = m.get("direction") or "iso"
    if m["property"] == "E":
        c = EL.engineering_constants(np.asarray(r["C"], float))
        return float(c["E_hill"]) if d == "iso" else float(c["E"]["xyz".index(d)])
    a = np.asarray(r["alpha"], float)
    return float(np.mean(a[:3])) if d == "iso" else float(a["xyz".index(d)])


def _evaluate(samples, form_of, U, unknowns, tol, pool, should_stop, log, event, tag, prog=(0.0, 1.0), rel=None):
    """The measured property of every sample at every row of U (unit cube):
    an (n_points, n_samples) array. rel, if given, is filled with the
    relative standard error of each particle-dynamics value."""
    Y = np.full((len(U), len(samples)), np.nan)
    jobs, where = [], []
    for a, uu in enumerate(U):
        theta = [u.value(x) for u, x in zip(unknowns, uu)]
        for s in samples:
            f = _set_theta(form_of(s), unknowns, theta)
            solver = PROPERTIES[s["m"]["property"]]["solver"]
            if solver == "cond":
                for t in _cond_tasks(s, f, tol, pool, f"{tag} {a + 1}"):
                    jobs.append(t)
                    where.append((a, s["i"], t[1]["d"]))
            elif solver == "dem":
                last = {"t": 0.0}

                def cb(strain, eta, its, a=a, s=s):
                    now = time.time()
                    if now - last["t"] >= 1.0:
                        last["t"] = now
                        event("progress", detail=f"{tag} {a + 1} of {len(U)}: particle dynamics, sample {s['i'] + 1}, "
                                                 f"strain {strain:.2f}, μr {eta:.3g}")
                t1 = time.time()
                Y[a, s["i"]], r_ = _dem_value(s, f, should_stop, cb)
                if math.isfinite(r_):
                    s["dem_se"].append(r_)
                    if rel is not None:
                        rel[a, s["i"]] = r_
                    log(f"   {tag} {a + 1}: sample {s['i'] + 1} particle dynamics -> {Y[a, s['i']]:.4g} "
                        f"(± {100 * r_:.1f} %, {time.time() - t1:.0f} s)")
                else:
                    log(f"   {tag} {a + 1}: sample {s['i'] + 1} jammed at these values (bound layer too thick for "
                        f"its spheres) - not run, ruled out")
            else:
                Y[a, s["i"]] = _elastic_value(s, f, tol, should_stop)
    if jobs:
        def on_done(res, n_done, n_all):
            event("progress", detail=f"{tag}: {n_done}/{n_all} solves",
                  progress=prog[0] + (prog[1] - prog[0]) * n_done / n_all)
        res = pool.run(jobs, should_stop=should_stop, on_done=on_done,
                       on_progress=lambda name, it, r, tg: event("solver", name=name, it=it, res=r, target=tg))
        acc = {}
        for (a, i, d), out in zip(where, res):
            col = out["r"]["column"]
            acc.setdefault((a, i), []).append(float(col["xyz".index(d)]))
        for (a, i), v in acc.items():
            Y[a, i] = float(np.mean(v))
    return Y


def _fit(U, y, noise=None):
    """Gaussian process for one measurement: log value over the unit cube.
    noise: the standard error of each value in ln units (particle dynamics):
    the surrogate then passes through the runs within their scatter instead
    of through every run, and its standard deviation is that of the mean."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    p = U.shape[1]
    kern = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(length_scale=np.full(p, 0.5), length_scale_bounds=(0.03, 30.0),
                                                     nu=2.5) + WhiteKernel(1e-8, (1e-12, 1e-4))
    alpha = 1e-10
    if noise is not None:
        # sklearn adds alpha to the kernel of the normalised values
        alpha = (np.asarray(noise, float) / max(float(np.std(np.log(y))), 1e-9)) ** 2 + 1e-10
    gp = GaussianProcessRegressor(kernel=kern, alpha=alpha, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        gp.fit(U, np.log(y))
    return gp


def _grid(p):
    n = GRID[p]
    ax = np.linspace(0.0, 1.0, n)
    return ax, np.array(list(itertools.product(ax, repeat=p))) if p > 1 else ax[:, None]


def _posterior(gps, samples, unknowns, Ug, model_unc):
    """Log posterior on the grid points Ug and the surrogate's mean and sd per
    measurement there."""
    lp = np.zeros(len(Ug))
    for j, u in enumerate(unknowns):
        lp += u.log_prior(Ug[:, j])
    # a measured, finite viscosity rules out the values that jam its sample
    for s in samples:
        if s.get("dem"):
            lp[_dem_jammed(s, unknowns, Ug)] = -np.inf
    if not np.isfinite(lp).any():
        raise ValueError("Every value in the ranges jams a viscosity sample (bound layer too thick for its spheres): "
                         "lower the upper bound of the bound layer")
    mus, sds = [], []
    for gp, s in zip(gps, samples):
        mu, sd = gp.predict(Ug, return_std=True)
        mus.append(mu)
        sds.append(sd)
        sm = math.log(1.0 + float(s["m"].get("rel_unc") or 0.05))
        var = sm * sm + model_unc * model_unc + sd * sd
        r = math.log(float(s["m"]["value"])) - mu
        lp += -0.5 * r * r / var - 0.5 * np.log(var)
    return lp, np.array(mus), np.array(sds)


def _summaries(post, Ug, ax, unknowns):
    """Marginals, MAP, median, 95 % interval, correlation (in the unit cube)."""
    p = len(unknowns)
    w = post / post.sum()
    i_map = int(np.argmax(post))
    out = []
    shape = (len(ax),) * p
    W = w.reshape(shape)
    for j, u in enumerate(unknowns):
        other = tuple(k for k in range(p) if k != j)
        marg = W.sum(axis=other) if other else W
        cdf = np.cumsum(marg)
        q = lambda f: float(np.interp(f, cdf, ax))
        lo, med, hi = q(0.025), q(0.5), q(0.975)
        mean_u = float(np.dot(marg, ax))
        sd_u = float(math.sqrt(max(np.dot(marg, (ax - mean_u) ** 2), 0.0)))
        # the prior's own 95 % width in the unit cube, for the narrowing
        pr = np.exp(u.log_prior(ax))
        pr /= pr.sum()
        pcdf = np.cumsum(pr)
        pw = float(np.interp(0.975, pcdf, ax) - np.interp(0.025, pcdf, ax))
        # a posterior still high at an end of the range does not fall off
        # there: the data bound the unknown on that side only. So does a long
        # low tail that reaches the end with a few per cent of the probability
        # in the last tenth of the range (an interfacial resistance far below
        # the particle's Kapitza scale is "no resistance" for any value): the
        # 95 % interval then runs to wherever the range was set to end.
        edge = None
        mm = float(marg.max()) or 1.0
        k10 = max(1, int(round(0.1 * len(marg))))
        tot = float(marg.sum()) or 1.0
        if marg[0] >= 0.2 * mm or (marg[0] >= 0.05 * mm and marg[:k10].sum() >= 0.025 * tot):
            edge = "low"
        if marg[-1] >= 0.2 * mm or (marg[-1] >= 0.05 * mm and marg[-k10:].sum() >= 0.025 * tot):
            edge = "high" if edge is None else "both"
        out.append({"path": u.path, "label": u.label, "unit": u.unit, "log": u.log,
                    "map": float(u.value(Ug[i_map, j])), "median": float(u.value(med)),
                    "ci95": [float(u.value(lo)), float(u.value(hi))], "lo": u.lo, "hi": u.hi,
                    "sd_unit": sd_u, "narrowing": pw / max(hi - lo, 1e-9), "edge": edge,
                    "marginal": {"x": [float(v) for v in u.value(ax)], "p": (marg / marg.max()).tolist()},
                    "prior": u.prior, "center": u.center, "sigma_dec": u.sigma_dec, "theory": u.theory})
    corr = np.eye(p)
    if p > 1:
        m = np.array([np.dot(w, Ug[:, j]) for j in range(p)])
        C = np.array([[np.dot(w, (Ug[:, a] - m[a]) * (Ug[:, b] - m[b])) for b in range(p)] for a in range(p)])
        dg = np.sqrt(np.maximum(np.diag(C), 1e-30))
        corr = C / np.outer(dg, dg)
    return out, i_map, corr


def _sensitivity(gps, u0, unknowns, h=0.02):
    """d ln(y_m) / d ln(theta_j) at u0 (per unit for linear unknowns)."""
    p = len(unknowns)
    J = np.zeros((len(gps), p))
    for j, u in enumerate(unknowns):
        up, dn = u0.copy(), u0.copy()
        a, b = min(1.0, u0[j] + h), max(0.0, u0[j] - h)
        up[j], dn[j] = a, b
        for m, gp in enumerate(gps):
            dy = float(gp.predict(up[None, :])[0] - gp.predict(dn[None, :])[0]) / (a - b)
            J[m, j] = dy / u.dlnv_du() if u.log else dy / (u.hi - u.lo)
    return J


def identifiability(J, sig, unknowns):
    """What the measurements determine at the estimate.

    Columns of the weighted sensitivity matrix W = J / sigma, scaled to unit
    length, give the collinearity index gamma = 1 / sqrt(min eig(W~^T W~))
    (Brun et al. 2001: above about 10-15 the set is not identifiable, a
    change of one unknown can be compensated by the others). The singular
    vectors of W are the combinations of ln(unknowns) the data fix, with the
    1-sigma change of each combination 1 / s_k."""
    W = J / sig[:, None]
    p = W.shape[1]
    norms = np.linalg.norm(W, axis=0)
    # an unknown whose e-fold change moves the predictions by less than a
    # tenth of their uncertainty is invisible to these measurements
    insensitive = [j for j in range(p) if norms[j] < 0.1]
    out = {"norms": norms.tolist(), "insensitive": [unknowns[j].label for j in insensitive]}
    sens = [j for j in range(p) if j not in insensitive]
    out["collinearity"] = None
    if len(sens) > 1:
        Wn = W[:, sens] / norms[sens]
        ev = np.linalg.eigvalsh(Wn.T @ Wn)
        out["collinearity"] = float(min(1e6, 1.0 / math.sqrt(max(ev.min(), 1e-12))))
    elif len(sens) == 1:
        out["collinearity"] = 1.0
    U_, s, Vt = np.linalg.svd(W, full_matrices=True)
    combos = []
    for k in range(p):
        sk = float(s[k]) if k < len(s) else 0.0
        v = Vt[k]
        v = v / (np.max(np.abs(v)) or 1.0)
        combos.append({"weights": v.tolist(), "names": [u.label for u in unknowns],
                       "sigma_ln": (1.0 / sk) if sk > 1e-12 else None,
                       "determined": bool(sk > 1.0)})
    out["combos"] = combos
    out["rank"] = int(np.sum(s > 1.0))
    # fewer measurements than unknowns: at most that many combinations exist
    out["underdetermined"] = bool(W.shape[0] < p)
    out["n_meas"] = int(W.shape[0])
    return out


def _notes(summ, ident):
    """What the posterior and the sensitivities say in words."""
    notes = []
    for r in summ:
        if r["edge"] in ("low", "high"):
            side, kind = ("lower", "an upper") if r["edge"] == "low" else ("upper", "a lower")
            lim = r["ci95"][1] if r["edge"] == "low" else r["ci95"][0]
            notes.append(f"{r['label']}: the posterior does not fall to zero toward the {side} end of the range; the "
                         f"data give {kind} bound ({'below' if r['edge'] == 'low' else 'above'} {lim:.3g} {r['unit']}, "
                         f"95 %; most probable {r['map']:.3g}), not a value - toward that end the property hardly "
                         f"depends on it")
        elif r["label"] in (ident.get("insensitive") or []) or r["edge"] == "both":
            notes.append(f"{r['label']}: these measurements do not respond to it (an e-fold change moves them by less "
                         f"than a tenth of their uncertainty); its value here is the prior's. Measure where it matters "
                         f"(see the suggestions) or fix it by theory")
        elif r["narrowing"] < 1.5:
            notes.append(f"{r['label']}: hardly narrowed by the measurements (×{r['narrowing']:.1f} against the "
                         f"prior): it is set by the prior, not by the data")
    if ident.get("underdetermined"):
        notes.append(f"Fewer measurements than unknowns: the data can fix at most {ident.get('n_meas')} "
                     f"combination(s) of the {len(summ)} unknowns, whatever their values - add a measurement of "
                     f"another sample (see the suggestions) or fix an unknown by theory")
    c = ident.get("collinearity")
    if c and c > 10:
        notes.append(f"Collinearity index {'>1000' if c > 1000 else f'{c:.0f}'} (above 10-15 a set is not "
                     f"identifiable, Brun et al. 2001): the unknowns compensate each other; only the determined "
                     f"combinations are meaningful unless a measurement is added or an unknown is fixed by theory")
    return notes


def _suggest(samples, unknowns, theta, sig_model, F0):
    """Extra measurements ranked by how much they would add (D-optimal on
    the theory model): each sample with another particle size (x0.3, x3) or
    another content (+-10 vol%), for thermal conductivity."""
    tk = [u for u in unknowns if re.match(r"^(matrix|phases\.\d+)\.props\.k$|^phases\.\d+\.r_int$", u.path)]
    if len(tk) != len(unknowns):
        return [], "Suggestions use the effective-medium model and cover the matrix and filler conductivities and R_int only"
    cands = []
    for s in samples:
        if s["m"]["property"] != "k" or not s["theory_ok"]:
            continue
        base = s["form"]
        rv = base.get("rve") or {}
        hand_sized = rv.get("auto", True) is False and bool(rv.get("L_um"))
        n_ph = len(base.get("phases") or [])
        for i, ph in enumerate(base.get("phases") or []):
            d = (ph.get("size") or {}).get("d")
            fr = (ph.get("fraction") or {}).get("value")
            for kind, f in (("size", 0.3), ("size", 3.0), ("content", -10.0), ("content", 10.0)):
                if kind == "size" and d:
                    sv = {f"phases.{i}.size.d": float(d) * f}
                    lab = f"{ph.get('name') or 'filler'}: d {float(d) * f:.3g} {UNIT_TXT.get((ph.get('size') or {}).get('unit', 'um'), 'µm')}"
                    if hand_sized:
                        # a hand-sized RVE kept as it was would draw a 0.3x
                        # particle with a few voxels, or hold only a couple of
                        # 3x particles: it is scaled with the particle, so the
                        # sample keeps its particle count and resolution. With
                        # several fillers no RVE fits one changed size without
                        # a far larger grid, and the size is not suggested.
                        if n_ph > 1:
                            continue
                        sv["rve.L_um"] = float(rv["L_um"]) * f
                        if rv.get("voxel_um"):
                            sv["rve.voxel_um"] = float(rv["voxel_um"]) * f
                        lab += f" (RVE {sv['rve.L_um']:.3g} µm)"
                    cands.append((s, sv, lab))
                if kind == "content" and fr and 2.0 < float(fr) + f < 70.0:
                    cands.append((s, {f"phases.{i}.fraction.value": float(fr) + f},
                                  f"{ph.get('name') or 'filler'}: {float(fr) + f:.3g} {(ph.get('fraction') or {}).get('basis', 'vol')}%"))
    # a condition already measured is not suggested again (sample 3 at the
    # size of sample 5 is sample 5); nor is one candidate twice
    def _key(st):
        return tuple(sorted((k, round(float(v), 9) if isinstance(v, (int, float)) else v) for k, v in st.items()))
    seen = {_key(s["m"].get("set") or {}) for s in samples}
    out = []
    d0 = np.linalg.slogdet(F0)[1]
    for s, setv, label in cands:
        merged = {**(s["m"].get("set") or {}), **setv}
        if _key(merged) in seen:
            continue
        seen.add(_key(merged))
        f = copy.deepcopy(s["form"])
        for path, v in setv.items():
            DOE.set_path(f, path, v)
        try:
            row = []
            for j, u in enumerate(unknowns):
                g = []
                for fac in (1.02, 1 / 1.02):
                    th = list(theta)
                    th[j] = th[j] * fac
                    g.append(_theory_k(S.normalize(_set_theta(f, unknowns, th))))
                row.append((math.log(g[0]) - math.log(g[1])) / (2 * math.log(1.02)))
            k0 = _theory_k(S.normalize(_set_theta(f, unknowns, theta)))
        except Exception:                                                # noqa: BLE001
            continue
        jrow = np.array(row) / sig_model
        F1 = F0 + np.outer(jrow, jrow)
        gain = float(np.linalg.slogdet(F1)[1] - d0)
        sd0 = np.sqrt(np.maximum(np.diag(np.linalg.pinv(F0)), 0))
        sd1 = np.sqrt(np.maximum(np.diag(np.linalg.pinv(F1)), 0))
        out.append({"label": f"sample {s['i'] + 1} with {label}", "set": merged,
                    "property": "k", "predicted": k0, "gain": gain,
                    "narrowing": [float(a / b) if b > 0 else None for a, b in zip(sd0, sd1)]})
    out.sort(key=lambda r: -r["gain"])
    return out[:5], None


def run(job, out_dir, log=print, event=None, should_stop=None, limits=None):
    """Calibrate the unknowns of `job` against its measurements; writes
    result.json (kind "calibration") and returns a short summary."""
    t0 = time.time()
    event = event or (lambda kind, **p: None)
    os.makedirs(out_dir, exist_ok=True)
    job = copy.deepcopy(job)
    _validate(job)
    opts = dict(job.get("options") or {})
    model_unc = float(opts.get("model_unc", 0.02))
    tol = float(opts.get("tol", 1e-6))
    rounds = int(opts.get("refine_rounds", 3))
    base = fill_props(job["base"])
    unknowns = [_Unknown(u) for u in job["unknowns"]]
    p = len(unknowns)
    with open(os.path.join(out_dir, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(job, fh, ensure_ascii=False, indent=1)
    log(f"== Calibration: {job.get('name') or 'unnamed'}")
    log(f"   {len(job['measurements'])} measurement(s), {p} unknown(s): " + ", ".join(u.label for u in unknowns))

    try:
        import numba
        threads = int(numba.get_num_threads())
    except Exception:                                                    # noqa: BLE001
        threads = os.cpu_count() or 4
    n_dir = sum(len(_dirs(m)) for m in job["measurements"] if PROPERTIES[m["property"]]["solver"] == "cond")
    event("stage", name="Calibration · samples", detail="structures of the measured samples", progress=0.02)
    samples = _prepare(job, base, unknowns, out_dir, limits, log, should_stop)
    # the solves of a round are independent: one per worker, side by side
    pool = None
    if n_dir:
        n_vox = max(s["labels"].size for s in samples if s["labels"] is not None)
        workers = PAR.plan_workers(threads, N_INIT[p] * n_dir, n_vox * 220e-9)
        pool = PAR.SolvePool(workers, max(1, threads // workers))
    try:
        for s in samples:
            if s["labels"] is None:
                continue
            s["lab_path"] = pool.share(s["labels"], name=f"lab{s['i']}")
            if s["cmask"] is not None:
                s["cm_path"] = os.path.join(pool.tmp, f"cm{s['i']}.npy")
                np.save(s["cm_path"], s["cmask"])
        if pool is not None:
            log(f"   solves run {workers} at a time ({max(1, threads // workers)} thread(s) each)")
        if any(s.get("dem") for s in samples):
            log("   viscosity samples: one particle-dynamics run per parameter set, one after the other")
        form_of = lambda s: s["form"]                                    # noqa: E731

        # ---- theory: the effective-medium preview -------------------------
        theory_rows = []
        for s in samples:
            if s["m"]["property"] == "k" and s["theory_ok"]:
                theory_rows.append({"sample": s["i"], "k_mid": _theory_k(s["spec"])})

        # ---- design, fit, refine -------------------------------------------
        U = _design(p, N_INIT[p])
        event("stage", name="Calibration · design", detail=f"{len(U)} parameter sets × {len(samples)} samples",
              progress=0.05)
        n_dem = sum(1 for s in samples if s.get("dem"))
        log(f"   initial design: {len(U)} parameter sets × "
            + " + ".join(([f"{n_dir} solves"] if n_dir else []) + ([f"{n_dem} particle-dynamics run(s)"] if n_dem else [])))
        Rel = np.full((len(U), len(samples)), np.nan)
        Y = _evaluate(samples, form_of, U, unknowns, tol, pool, should_stop, log, event, "design", (0.05, 0.45),
                      rel=Rel)
        ax, Ug = _grid(p)
        history = []
        for rnd in range(rounds + 1):
            ok = np.all(np.isfinite(Y) & (Y > 0), axis=1)
            if ok.sum() < 2:
                raise ValueError("Fewer than two parameter sets of the design could be solved (jammed or failed): "
                                 "narrow the ranges of the unknowns")
            gps = [_fit(U[ok], Y[ok, m], noise=Rel[ok, m] if samples[m].get("dem") else None)
                   for m in range(len(samples))]
            lp, mus, sds = _posterior(gps, samples, unknowns, Ug, model_unc)
            post = np.exp(lp - lp.max())
            summ, i_map, corr = _summaries(post, Ug, ax, unknowns)
            sig = np.array([math.hypot(math.log(1.0 + float(s["m"].get("rel_unc") or 0.05)), _extra_unc(s))
                            for s in samples])
            gp_sd_map = float(np.max(sds[:, i_map] / np.sqrt(sig ** 2 + model_unc ** 2)))
            history.append({"round": rnd, "points": int(len(U)), "gp_sd_rel": gp_sd_map,
                            "map": [r["map"] for r in summ]})
            log(f"   round {rnd}: {len(U)} points, estimate " +
                ", ".join(f"{r['label']} {r['map']:.4g}" for r in summ) +
                f"; surrogate error at the estimate {gp_sd_map:.2f} of the measurement uncertainty")
            if rnd == rounds or gp_sd_map < 0.2:
                break
            # new points: the estimate and the posterior points where the
            # surrogate is least certain
            w = post / post.sum()
            score = w * np.max(sds / sig[:, None], axis=0)
            new = [Ug[i_map]]
            for idx in np.argsort(-score):
                cand = Ug[idx]
                if all(np.max(np.abs(cand - q)) > 0.04 for q in list(U) + new):
                    new.append(cand)
                if len(new) >= p + 1:
                    break
            new = np.array([q for q in new if all(np.max(np.abs(q - r)) > 1e-3 for r in U)])
            if not len(new):
                break
            event("stage", name="Calibration · refinement", detail=f"round {rnd + 1}: {len(new)} parameter sets",
                  progress=0.1 + 0.7 * (rnd + 1) / (rounds + 1))
            Rn = np.full((len(new), len(samples)), np.nan)
            Yn = _evaluate(samples, form_of, new, unknowns, tol, pool, should_stop, log, event, f"round {rnd + 1}",
                           (0.45 + 0.4 * rnd / max(rounds, 1), 0.45 + 0.4 * (rnd + 1) / max(rounds, 1)), rel=Rn)
            U = np.vstack([U, new])
            Y = np.vstack([Y, Yn])
            Rel = np.vstack([Rel, Rn])

        # ---- a direct solve at the estimate --------------------------------
        u_map = Ug[i_map]
        event("stage", name="Calibration · check", detail="direct solve at the estimate", progress=0.9)
        Rv = np.full((1, len(samples)), np.nan)
        Yv = _evaluate(samples, form_of, u_map[None, :], unknowns, tol, pool, should_stop, log, event, "check",
                       (0.9, 0.97), rel=Rv)[0]
        theta_map = [u.value(x) for u, x in zip(unknowns, u_map)]
        pred = [float(np.exp(gps[m].predict(u_map[None, :])[0])) for m in range(len(samples))]
        rows = []
        try:
            cat = {c["path"]: c for c in DOE.parameter_catalogue(base)}
        except Exception:                                                # noqa: BLE001
            cat = {}
        for s, yv, yp in zip(samples, Yv, pred):
            m = s["m"]
            vo_s = s["spec"]["options"]["viscosity"]
            # the direct check is one run: its own standard error counts
            run_unc = float(Rv[0, s["i"]]) if (s.get("dem") and math.isfinite(Rv[0, s["i"]])) else 0.0
            rows.append({"model": "particle dynamics" if s.get("dem") else "RVE",
                         "gd": vo_s["gd_ref"] if s.get("dem") else None,
                         "n_spheres": int(len(s["dem"]["radii"])) if s.get("dem") else None,
                         "run_unc": run_unc if s.get("dem") else None,
                         "label": m.get("label") or f"measurement {s['i'] + 1}", "set": m.get("set") or {},
                         "conditions": [{"label": (cat.get(k) or {}).get("label") or k, "value": v,
                                         "unit": (cat.get(k) or {}).get("unit") or ""} for k, v in (m.get("set") or {}).items()],
                         "property": m["property"], "unit": PROPERTIES[m["property"]]["unit"],
                         "direction": m.get("direction") or "iso", "measured": float(m["value"]),
                         "rel_unc": float(m.get("rel_unc") or 0.05), "surrogate": yp, "direct": float(yv),
                         "residual_sigma": (math.log(float(yv)) - math.log(float(m["value"])))
                         / math.sqrt(math.log(1 + float(m.get("rel_unc") or 0.05)) ** 2 + model_unc ** 2
                                     + run_unc ** 2),
                         "grid": s["plan"]["N"] if s["plan"] else None,
                         "voxel_um": s["plan"]["h_um"] if s["plan"] else None, "vf": s["vf"],
                         "theory_k": _theory_k(S.normalize(_set_theta(s["form"], unknowns, theta_map)))
                         if (m["property"] == "k" and s["theory_ok"]) else None})
        # ---- identifiability and what to measure next ----------------------
        J = _sensitivity(gps, u_map.copy(), unknowns)
        sig_tot = np.sqrt(sig ** 2 + model_unc ** 2)          # sig holds the runs' own error already
        ident = identifiability(J, sig_tot, unknowns)
        ident["J"] = J.tolist()
        F0 = (J / sig_tot[:, None]).T @ (J / sig_tot[:, None])
        for j, u in enumerate(unknowns):
            # the prior as information: a normal prior adds 1/sigma^2 (ln units)
            if u.prior == "normal" and u.center not in (None, "") and u.log:
                F0[j, j] += 1.0 / (u.sigma_dec * math.log(10.0)) ** 2
            else:
                F0[j, j] += 1.0 / (0.5 * (math.log(u.hi) - math.log(u.lo)) if u.log else 1.0) ** 2
        sugg, sugg_note = _suggest(samples, unknowns, theta_map, float(np.median(sig_tot)), F0)
        # the 2-D posterior for the joint plot
        joint = None
        if p == 2:
            n = len(ax)
            joint = {"x": [float(v) for v in unknowns[0].value(ax[::2])], "y": [float(v) for v in unknowns[1].value(ax[::2])],
                     "p": (post / post.max()).reshape(n, n).T[::2, ::2].tolist()}
        elif p == 3:
            n = len(ax)
            P3 = (post / post.max()).reshape(n, n, n)
            joint = {"pairs": []}
            for a, b in ((0, 1), (0, 2), (1, 2)):
                other = 3 - a - b
                M2 = P3.sum(axis=other)
                M2 = M2 / M2.max()
                joint["pairs"].append({"a": a, "b": b, "p": M2.T.tolist(),
                                       "x": [float(v) for v in unknowns[a].value(ax)],
                                       "y": [float(v) for v in unknowns[b].value(ax)]})
        notes = _notes(summ, ident)
        bad = [r for r in rows if abs(r["residual_sigma"]) > 2.5]
        if bad:
            notes.append("No parameter set in the ranges reproduces " + ", ".join(r["label"] for r in bad)
                         + " within its uncertainty: another mechanism is missing from the model (agglomerates, "
                           "porosity, an interphase, a filler property) or a range is too narrow")
        from . import __version__
        result = {"kind": "calibration", "name": job.get("name") or "Calibration", "version": __version__,
                  "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
                  "unknowns": summ, "correlation": corr.tolist(), "measurements": rows,
                  "identifiability": ident, "suggestions": sugg, "suggestions_note": sugg_note,
                  "joint": joint, "history": history, "notes": notes, "model_unc": model_unc,
                  "design": {"points": [[float(u.value(x)) for u, x in zip(unknowns, uu)] for uu in U],
                             "values": Y.tolist()},
                  "theory": theory(base), "theory_preview": theory_rows,
                  "apply": {u.path: float(r["map"]) for u, r in zip(unknowns, summ)}}
        # what the run list shows, also after a restart (it reads this file)
        result["headline"] = [{"key": "cal_" + str(j), "label": r["label"], "value": r["map"], "unit": r["unit"]}
                              for j, r in enumerate(summ)]
        result["check_counts"] = {"pass": len(rows) - len(bad), "warn": len(notes), "fail": len(bad)}
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(PL._clean(result), fh, ensure_ascii=False, indent=1)
        log(f"== calibration finished in {time.time() - t0:.0f} s")
        for r in summ:
            log(f"   {r['label']}: {r['map']:.4g} {r['unit']} (95 % {r['ci95'][0]:.3g} – {r['ci95'][1]:.3g}, "
                f"narrowed ×{r['narrowing']:.1f})")
        for r in rows:
            log(f"   {r['label']}: measured {r['measured']:.4g}, "
                f"{'particle dynamics' if r['model'] == 'particle dynamics' else 'RVE'} at the estimate {r['direct']:.4g} "
                f"({100 * (r['direct'] / r['measured'] - 1):+.1f} %)")
        for n in notes:
            log("   · " + n)
        return PL._clean({"name": result["name"], "elapsed_s": result["elapsed_s"], "headline": result["headline"],
                          "check_counts": result["check_counts"], "kind": "calibration"})
    finally:
        if pool is not None:
            pool.close()


# =========================================================================
# instant preview on the effective-medium model
# =========================================================================
THEORY_UNC = 0.10          # relative error assumed for the effective-medium model


def _dem_vec(km, fillers, n=200):
    """dem_conductivity for arrays of parameter sets: km (P,), fillers
    [(phi, k (P,), R (P,), d_m)]."""
    km = np.asarray(km, float)
    phi_t = sum(f[0] for f in fillers)
    if phi_t <= 0:
        return km
    kst = [(f[0] / phi_t, np.asarray(f[1], float) / (1.0 + 2.0 * np.asarray(f[2], float) * np.asarray(f[1], float)
                                                    / max(f[3], 1e-30))) for f in fillers]
    lk = np.log(km)
    dt = phi_t / n
    t = 0.0

    def rate(t_, lk_):
        k = np.exp(lk_)
        return sum(w * 3.0 * (ks - k) / (ks + 2.0 * k) for w, ks in kst) / (1.0 - t_)
    for _ in range(n):
        a = rate(t, lk)
        b = rate(t + 0.5 * dt, lk + 0.5 * dt * a)
        c = rate(t + 0.5 * dt, lk + 0.5 * dt * b)
        d = rate(t + dt, lk + dt * c)
        lk = lk + dt * (a + 2 * b + 2 * c + d) / 6.0
        t += dt
    return np.exp(lk)


def _theory_model(spec, unknowns, thetas):
    """k of one sample for parameter sets thetas (P, p) on the DEM model."""
    P = len(thetas)
    km = np.full(P, float(spec["matrix"]["props"]["k"]))
    fillers = []
    for i, ph in enumerate(spec["phases"]):
        d = float(ph["size_um"].get("d") or 1.0) * 1e-6
        if ph.get("shell"):
            d += 2.0 * ph["shell"]["thickness_um"] * 1e-6
        k = np.full(P, float(ph["props"]["k"]))
        r = np.full(P, float(ph.get("r_int") or 0.0))
        for j, u in enumerate(unknowns):
            if u.path == f"phases.{i}.props.k":
                k = thetas[:, j]
            elif u.path == f"phases.{i}.r_int":
                r = thetas[:, j]
        fillers.append((float(ph["vf"]), k, r, d))
    for j, u in enumerate(unknowns):
        if u.path == "matrix.props.k":
            km = thetas[:, j]
    return _dem_vec(km, fillers)


def preview(job):
    """Calibration on the effective-medium model (differential Bruggeman with
    Hasselman-Johnson particles, Every et al. 1992), on the whole parameter
    grid in well under a second: the estimate theory alone gives, and above
    all whether the measurements can determine the unknowns at all, before
    any RVE is solved. Thermal conductivity with the matrix and filler
    conductivities and R_int as unknowns; spheres (other shapes are treated
    as spheres of their diameter)."""
    _validate(job)
    base = fill_props(job["base"])
    unknowns = [_Unknown(u) for u in job["unknowns"]]
    meas = job["measurements"]
    rx = re.compile(r"^(matrix|phases\.\d+)\.props\.k$|^phases\.\d+\.r_int$")
    if any(m.get("property") != "k" for m in meas) or not all(rx.match(u.path) for u in unknowns):
        return {"available": False,
                "reason": "The effective-medium preview covers thermal conductivity with the matrix and filler "
                          "conductivities and R_int as unknowns; the RVE calibration covers the rest"}
    p = len(unknowns)
    n = {1: 401, 2: 81, 3: 31}[p]
    ax = np.linspace(0.0, 1.0, n)
    Ug = np.array(list(itertools.product(ax, repeat=p))) if p > 1 else ax[:, None]
    thetas = np.column_stack([u.value(Ug[:, j]) for j, u in enumerate(unknowns)])
    lp = np.zeros(len(Ug))
    for j, u in enumerate(unknowns):
        lp += u.log_prior(Ug[:, j])
    specs, notes = [], []
    mid = [u.value(0.5) for u in unknowns]
    for m in meas:
        f = copy.deepcopy(base)
        for path, v in (m.get("set") or {}).items():
            DOE.set_path(f, path, v)
        sp = S.normalize(_set_theta(f, unknowns, mid))
        if any(ph["void"] or ph["shape"] == "network" for ph in sp["phases"]):
            return {"available": False, "reason": "The effective-medium preview does not cover pores or networks"}
        if any(ph["shape"] != "sphere" for ph in sp["phases"]):
            notes.append("Non-spherical fillers are treated as spheres of their diameter in this preview")
        specs.append(sp)
        sm = math.log(1.0 + float(m.get("rel_unc") or 0.05))
        var = sm * sm + THEORY_UNC ** 2
        k = _theory_model(sp, unknowns, thetas)
        r = math.log(float(m["value"])) - np.log(k)
        lp += -0.5 * r * r / var
    post = np.exp(lp - lp.max())
    summ, i_map, corr = _summaries(post, Ug, ax, unknowns)
    th0 = thetas[i_map]
    J = np.zeros((len(meas), p))
    for j in range(p):
        up, dn = th0.copy(), th0.copy()
        up[j] *= 1.01
        dn[j] /= 1.01
        for mi, sp in enumerate(specs):
            ku, kd = _theory_model(sp, unknowns, np.array([up, dn]))
            J[mi, j] = (math.log(ku) - math.log(kd)) / (2.0 * math.log(1.01))
    sig = np.array([math.sqrt(math.log(1.0 + float(m.get("rel_unc") or 0.05)) ** 2 + THEORY_UNC ** 2) for m in meas])
    ident = identifiability(J, sig, unknowns)
    ident["J"] = J.tolist()
    fit = [float(_theory_model(sp, unknowns, th0[None, :])[0]) for sp in specs]
    joint = None
    if p == 2:
        joint = {"x": [float(v) for v in unknowns[0].value(ax)], "y": [float(v) for v in unknowns[1].value(ax)],
                 "p": (post / post.max()).reshape(n, n).T.tolist()}
    notes += _notes(summ, ident)
    return PL._clean({"available": True, "model": "differential effective medium with Hasselman-Johnson particles "
                                                  "(Every et al. 1992)", "model_unc": THEORY_UNC,
                      "unknowns": summ, "correlation": corr.tolist(), "identifiability": ident, "joint": joint,
                      "fit": fit, "notes": notes})
