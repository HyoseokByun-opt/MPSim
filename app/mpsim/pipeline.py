"""One job, end to end: form -> structure -> solves -> analyses -> files.

    run(form, out_dir, log, event, should_stop) -> summary dict

Files written to out_dir:
    spec.normalized.json   the validated specification actually solved
    result.json            every number, check and reference model
    layer_card.json        the homogenised layer, ready for multilayer stacking
    structure.tif          full-resolution label volume (ImageJ / PuMA / any voxel code)
    view/                  viewer data (see visual.py)
    figures/*.png          publication figures (PyVista / matplotlib)
    report.html            self-contained report
    logs/                  NASA PuMA solver logs

Solver policy: every property is computed with open-source libraries - NASA
PuMA for conduction, elasticity, Stokes flow, diffusion, radiation and surface
measures, PoreSpy for image analysis, porosimetry and pore networks,
scikit-rf for the shielding network - unless PuMA is unavailable or the
built-in verification solvers are selected explicitly. The report states
which was used.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import time
import traceback

import numpy as np

from . import __version__
from . import acoustics as AC
from . import analytical as AN
from . import generate as GEN
from . import geometry as G
from . import morphology as MO
from . import parallel as PAR
from . import poro as PO
from . import rve as R
from . import spec as S
from . import live as LV
from . import materials as M
from . import mechtest as MT
from . import visual as V
from .solvers import complex_cond as CC
from .solvers import conduction as CD
from .solvers import elastic as EL
from .solvers import fans as FA
from .solvers import viscosity as VI
from . import figworker as FWK
from .solvers import emi as EMI
from .solvers import fullwave as FW
from .solvers import puma_backend as PB

EPS0 = 8.8541878128e-12
SIGMA_SB = 5.670374419e-8

COND = {
    "thermal":    {"prop": "k",     "title": "Thermal conductivity",    "unit": "W/m·K", "kind": "thermal",    "group": "Thermal"},
    "electrical": {"prop": "sigma", "title": "Electrical conductivity", "unit": "S/m",   "kind": "electrical", "group": "Electrical"},
    "dielectric": {"prop": "eps_r", "title": "Relative permittivity",   "unit": "-",     "kind": "electrical", "group": "Dielectric"},
    "magnetic":   {"prop": "mu_r",  "title": "Relative permeability",   "unit": "-",     "kind": "electrical", "group": "Magnetic"},
}


# =========================================================================
# small helpers
# =========================================================================
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _dump(path, obj):
    # aside and renamed: a stopped run's process ends at once, mid-write too.
    # Windows refuses the rename while a reader (the server serving the
    # partial result) has the file open: a few tries, then in place
    data = json.dumps(_clean(obj), ensure_ascii=False, indent=1)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    try:
        os.remove(tmp)
    except OSError:
        pass


def settle_worst(samples):
    """The widest relative 95 % confidence interval of a mean over the
    realisations so far, and what it belongs to: (value, name) or (None, None)
    with fewer than two realisations."""
    worst, which = None, None
    for key, rows in samples.items():
        a = np.asarray(rows, float)
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[0] < 2:
            continue
        mean, _std, ci = _mean_ci(a)
        for k, (m, c) in enumerate(zip(np.atleast_1d(mean), np.atleast_1d(ci))):
            if not np.isfinite(m) or abs(m) < 1e-300 or not np.isfinite(c):
                continue
            rel = float(c / abs(m))
            if worst is None or rel > worst:
                worst, which = rel, f"{key}" + (f" component {k + 1}" if a.shape[1] > 1 else "")
    return worst, which


def _mean_ci(samples):
    a = np.asarray(samples, float)
    n = a.shape[0]
    mean = a.mean(axis=0)
    if n < 2:
        return mean, None, None
    std = a.std(axis=0, ddof=1)
    tval = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 2.0)
    return mean, std, tval * std / math.sqrt(n)


def _label_stats(field, labels, table):
    """Mean, 99th percentile and maximum of a field in every region."""
    out = []
    lab = np.asarray(labels)
    fld = np.asarray(field)
    for li, t in enumerate(table):
        m = lab == li
        if not m.any():
            continue
        v = fld[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out.append({"label": t["name"], "mean": float(v.mean()), "p99": float(np.percentile(v, 99)),
                    "max": float(v.max()), "min": float(v.min())})
    return out


class Progress:
    def __init__(self, event, total_units):
        self.event = event
        self.total = max(float(total_units), 1e-9)
        self.done = 0.0
        self.base = 0.0
        self.units = 0.0

    def stage(self, name, detail="", units=1.0):
        self.base = self.done
        self.units = float(units)
        self.event("stage", name=name, detail=detail, progress=min(self.base / self.total, 0.999))

    def sub(self, frac):
        return min((self.base + self.units * max(0.0, min(frac or 0.0, 1.0))) / self.total, 0.999)

    def end(self):
        self.done = self.base + self.units


# =========================================================================
# structure
# =========================================================================
# inradius of the unit-circumradius polyhedra: a shell of thickness t moves
# every face out by t, i.e. the circumradius by t / inradius
_POLY_INR = {"tetrahedron": 1 / 3, "octahedron": 3 ** -0.5, "dodecahedron": 0.7947, "icosahedron": 0.7947,
             "irregular": 0.62}


def _outer_size(shape, size, t):
    """Size dict of the particle with its shell (thickness t) included."""
    size = dict(size)
    if not t:
        return size
    d0 = size["d"]
    if shape == "spheroid":
        a, b = 0.5 * d0, 0.5 * d0 * size["aspect"]
        size["aspect"] = (b + t) / (a + t)
    if shape in ("cylinder", "spherocylinder"):
        size["length"] = size["length"] + 2 * t
    if shape in ("cuboid", "superellipsoid"):
        size["ly"] = size["ly"] + 2 * t
        size["lz"] = size["lz"] + 2 * t
    if shape == "polyhedron":
        size["d"] = d0 + 2 * t / _POLY_INR.get(size.get("poly", "icosahedron"), 0.62)
        return size
    size["d"] = d0 + 2 * t
    return size


def _generator_phases(spec):
    """Outer particle sizes (shell included) for the generator."""
    out = []
    for ph in spec["phases"]:
        t = ph["shell"]["thickness_um"] if ph["shell"] and ph["shape"] != G.NETWORK else 0.0
        size = _outer_size(ph["shape"], ph["size_um"], t) if ph["shape"] != G.NETWORK else dict(ph["size_um"])
        out.append({"name": ph["name"], "shape": ph["shape"], "size_um": size, "dist": ph["dist"],
                    "vf": ph["vf"], "orientation": ph["orientation"], "overlap": ph["overlap"],
                    "gap_frac": ph["gap_frac"]})
        if ph.get("section"):
            out[-1]["section"] = dict(ph["section"])
    return out


# ----------------------------------------------------------- kept structures
# Raised whenever the generator would build a different structure from the
# same inputs, so that a structure kept by an older version is not reused.
# 2: packing budgets in steps instead of seconds (large boxes reached the
#    target size; before, they could stop short of it on the clock)
# 3: no particle overlaps or is trimmed (v5.0.1): the clump repair pushes and
#    turns, shrinks what it cannot free; sphere packings are separated exactly
# 4: flat and slender particles (platelets, rods) pack as a lattice of spheres
#    of half their thickness through their whole body (v5.2): the 40-sphere
#    template covered a thin platelet in patches, and the exact-shape repair
#    then shrank nearly every particle (alumina + BN at 55 vol% reached 31.5)
# 5: thick ribbons pack as two layers of spheres; the exact-shape repair grows
#    the particles it shrank back where they fit (v5.2.1)
STRUCTURE_VERSION = 5


def overlap_steps(opts):
    """The generator's progressive overlap setting, None when it is off."""
    o = opts.get("overlap_steps") or {}
    if not o.get("on", True):
        return None
    r = {"tries": int(o.get("tries", 2)), "max_vox": int(o.get("max_vox", 3))}
    if o.get("share", "auto") != "none" and o.get("max_pct") is not None:
        r["max_pct"] = float(o["max_pct"])      # the largest share in two particles, %
        if o.get("share", "auto") == "auto":
            r["auto"] = o.get("auto_text") or "Auto"
    return r


def resolve_overlap_share(spec, plan, log=None):
    """Auto for the largest share of the particle voxels in two particles
    (overlap_share.auto_share), from the analyses, the materials, the
    particles' size in voxels and the loading - set into the options once the
    grid is planned."""
    o = spec["options"].get("overlap_steps") or {}
    if not o.get("on", True) or o.get("share", "auto") != "auto":
        return None
    from . import overlap_share as OS
    a = OS.auto_share(spec, plan)
    o["max_pct"], o["auto_text"] = float(a["pct"]), a["text"]
    if log is not None:
        log(f"   progressive overlap, largest share in two particles (Auto): {a['text']}")
    return a


def kept_share_ok(ginfo, labels, h, gen_phases, film, opts):
    """Whether a kept structure may be used under the share in force: its
    voxels in two particles within it, and not built under a stricter one
    that held its overlap back (a fresh build could then go further)."""
    o = overlap_steps(opts)
    lim = None if o is None else o.get("max_pct")
    if lim is None and o is not None:
        lim = float("inf")
    oc = ginfo.get("overlap_check")
    if not oc or oc.get("error"):
        try:
            oc = ginfo["overlap_check"] = GEN.overlap_check(labels.shape, h, ginfo, gen_phases, film=film)
        except Exception:                                            # noqa: BLE001
            return True, None
    share = 100.0 * float(oc.get("of_particles") or 0.0)
    if o is None:
        return share <= 0.0 or not ginfo.get("overlap_vox"), share
    kl = ginfo.get("overlap_limit") or {}
    held = bool(kl.get("without_overlap") or int(kl.get("n_builds") or 1) > 1)
    return share <= lim + 1e-9 and not (held and float(kl.get("max_pct") or 0.0) < lim), share


def contacts_key(opts):
    """The contact handling as it identifies a structure: the mode, and for
    "apart" a gap other than the 2 voxels it always had (older keys stay)."""
    c = opts.get("contacts", "keep")
    g = opts.get("particle_gap_vox")
    if c == "apart" and g not in (None, "") and abs(float(g) - 2.0) > 1e-9:
        c = f"apart:{float(g):g}"
    # progressive overlap other than its default (on, 2 packings per step,
    # up to 3 voxels): the keys of the default stay those of 5.6.0. The
    # largest share in two particles is not in the key - a kept structure is
    # checked against the share in force when it is loaded (kept_share_ok)
    o = overlap_steps(opts)
    if o is None:
        c += "|ov:off"
    elif (o["tries"], o["max_vox"]) != (2, 3):
        c += f"|ov:{o['tries']}x{o['max_vox']}"
    return c


def structure_key(gen_phases, plan, contacts, seed, want_cf, sheared=None, dispersion=None, pressed=None):
    """What identifies one generated realisation: the generator's own inputs
    (phase names aside), the grid, the contact handling and the seed - and,
    for the arrangement after shear flow, what it depends on (shear_key), and
    an uneven dispersion (rve.dispersion)."""
    geo = [{k: v for k, v in ph.items() if k != "name"} for ph in gen_phases]
    key = {"v": STRUCTURE_VERSION, "phases": geo, "N": int(plan["N"]),
           "h": float(f"{plan['h_um']:.12g}"), "nz": int(plan["Nz"]) if plan.get("film") else None,
           "contacts": contacts, "seed": int(seed), "cf": bool(want_cf)}
    if plan.get("skin"):
        key["skin"] = int(plan["skin"])          # only a film with a skin: other keys stay as they were
    if any((ph.get("dist") or {}).get("type") in ("lognormal", "normal", "table") for ph in gen_phases):
        # v5.3: particles of a size distribution are drawn until the target
        # volume (no common rescaling); monodisperse structures are unchanged
        key["psd"] = 2
    if sheared:
        key["sheared"] = sheared                 # only a sheared one: random packings keep their keys
    if pressed:
        key["pressed"] = pressed
    if dispersion and dispersion.get("mode", "uniform") != "uniform":
        key["dispersion"] = {k: v for k, v in dispersion.items() if k != "peak_vf"}
    blob = json.dumps(key, sort_keys=True, default=float)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def shear_phases(spec):
    """The particle phases of a structure arranged by shear flow: spheres of
    a solid (no shells, no pores, no permitted overlap), at least one."""
    return [i for i, ph in enumerate(spec["phases"]) if ph["shape"] != G.NETWORK and ph["vf"] > 0]


def shear_key(spec):
    """What the arrangement after shear flow depends on (None for a random
    packing): the strain, and what sets the forces between the particles
    against the viscous ones - the resin at the reference rate, roughness,
    closest approach, friction, bound layer, adhesion and Hamaker constants."""
    rv = spec["rve"]
    if rv.get("arrangement") != "sheared":
        return None
    vo = spec["options"]["viscosity"]
    do = vo["dem"]
    solid = shear_phases(spec)
    hks, _ = dem_hamaker(spec, solid, do)
    mu = float(VI.matrix_flow(vo["model"], [vo["gd_ref"]])[0])
    return {"strain": int(rv["shear_strain"]), "stress": float(f"{mu * vo['gd_ref']:.10g}"),
            "dem": {k: do[k] for k in ("roughness_nm", "hmin_nm", "mu_f", "bound_nm", "adhesion_mJ_m2")},
            "A": [float(f"{a:.10g}") for a in hks]}


def press_key(spec):
    """What the arrangement after compression depends on (None for any
    other): the compression, its rate and lateral condition, and the resins'
    viscosities."""
    rv = spec["rve"]
    if rv.get("arrangement") != "pressed":
        return None
    vo = spec["options"]["viscosity"]
    cure = None
    if rv.get("press_cure"):
        cure = {k: rv.get(k) for k in ("press_mold_C", "press_control", "press_stress_Pa", "press_part_mm", "press_time_s")}
        cure["kin"] = [{k: t["props"].get(k) for k in CURE_KEYS} for t in spec["labels"] if t["kind"] == "matrix"]
    return {"v": 3, "compression": float(rv["press_compression"]), "rate": float(rv["press_rate"]),
            "lateral": rv.get("press_lateral", "xy"), "steps": rv.get("press_steps"), "cure": cure,
            "model": vo["model"], "mu_regions": [m.get("mu_pas") for m in spec.get("matrix_regions") or []]}


CURE_KEYS = ("cure_E1", "cure_logA1", "cure_E2", "cure_logA2", "cure_m", "cure_n", "cure_dH", "gel_alpha", "cm_A",
             "cm_B", "visc_Ea")
CURE_NEEDED = ("cure_E1", "cure_logA1", "cure_E2", "cure_logA2", "cure_m", "cure_n", "gel_alpha", "cm_A", "cm_B")


# every step of the compression in 3D (the 3D view's step player): the
# first realisation's, written beside the run while it compresses, marked
# with the key of its structure, copied into the run's view
PRESS_STEPS = "press_steps"


def press_steps_ok(d, skey):
    """The folder of the compression's steps in `d` if it belongs to the structure `skey`."""
    p = os.path.join(d, PRESS_STEPS)
    try:
        with open(os.path.join(p, "key.txt"), encoding="utf-8") as fh:
            return p if fh.read().strip() == skey else None
    except OSError:
        return None


def press_cure(spec, base, mu, log):
    """The Cure of a compression with cure (solvers/compress.Cure), or
    (None, reason): every resin whose material carries the cure constants,
    its viscosity at the start the one of the compression; the heat of
    reaction per volume of the RVE from the resin's share of the box; the
    exchange time with the mould from the part thickness and a Maxwell
    estimate of the compound's conductivity."""
    from .solvers import compress as CP
    rv = spec["rve"]
    table = spec["labels"]
    n = base.size
    kin, eta0, heat, skipped = {}, {}, {}, []
    for li, t in enumerate(table):
        if mu[li] is None or not (base == li).any():
            continue
        pr = t["props"]
        if any(pr.get(k) in (None, "") for k in CURE_NEEDED):
            skipped.append(t["name"])
            continue
        kin[li] = {"E1": float(pr["cure_E1"]) * 1e3, "A1": 10.0 ** float(pr["cure_logA1"]),
                   "E2": float(pr["cure_E2"]) * 1e3, "A2": 10.0 ** float(pr["cure_logA2"]),
                   "m": float(pr["cure_m"]), "n": float(pr["cure_n"]), "ag": float(pr["gel_alpha"]),
                   "A": float(pr["cm_A"]), "B": float(pr["cm_B"]), "Ea": float(pr.get("visc_Ea") or 0.0) * 1e3}
        eta0[li] = float(mu[li])
        phi = float(np.count_nonzero(base == li)) / n
        heat[li] = phi * float(pr.get("rho") or 1.2) * 1e3 * float(pr.get("cure_dH") or 0.0) * 1e3
    if skipped:
        log(f"  (no cure constants for {', '.join(skipped)}: its viscosity stays as it is)")
    if not kin:
        return None, "no resin has cure constants (Material tab)"
    # rho c of the RVE and its conductivity (Maxwell, the fillers as spheres)
    rhocp, kf, phif = 0.0, 0.0, 0.0
    for li, t in enumerate(table):
        f = float(np.count_nonzero(base == li)) / n
        pr = t["props"]
        rhocp += f * float(pr.get("rho") or 1.0) * 1e3 * float(pr.get("cp") or 1000.0)
    for ph in spec["phases"]:
        if ph["void"] or ph["shape"] == G.NETWORK:
            continue
        phif += float(ph["vf"])
        kf += float(ph["vf"]) * float(ph["props"].get("k") or 1.0)
    km = float(spec["matrix"]["props"].get("k") or 0.2)
    kfil = kf / phif if phif > 0 else km
    kc = km * (kfil + 2 * km + 2 * phif * (kfil - km)) / (kfil + 2 * km - phif * (kfil - km)) if phif > 0 else km
    L = float(rv.get("press_part_mm") or 0.0) * 1e-3
    tau = rhocp * L * L / (math.pi ** 2 * kc) if L > 0 else 0.0
    return CP.Cure(kin, eta0, rv["press_mold_C"], heat, rhocp, tau), {"tau_s": tau, "rhocp": rhocp, "k_compound": kc}


def press_viscosities(spec, rate):
    """Viscosity per label of the table for the compression (None where the
    label is no resin): the main resin by its flow model, a matrix region by
    its viscosity before cure (else the main resin's), both at the equivalent
    shear rate of the compression, sqrt(3) times its strain rate."""
    vo = spec["options"]["viscosity"]
    gd = math.sqrt(3.0) * float(rate)
    mu_main = float(VI.matrix_flow(vo["model"], [gd])[0])
    mregs = spec.get("matrix_regions") or []
    out = []
    for t in spec["labels"]:
        if t["kind"] != "matrix":
            out.append(None)
        elif t.get("region") is None:
            out.append(mu_main)
        else:
            m_ = mregs[t["region"]].get("mu_pas") if t["region"] < len(mregs) else None
            out.append(float(m_) if m_ else mu_main)
    return out, gd


def press_structure(spec, plan, gen_labels, ginfo, log, event, prog, should_stop, steps_dir=None):
    """The RVE compressed along z while its resin flows (solvers/compress.py):
    a periodic three-dimensional creeping-flow solve per step on the structure
    as it deforms - the resins with their viscosities, the spheres rigid, the
    pores compressible - and the sides free to expand or held. Returns
    (labels, info) of the compressed periodic box (nx x ny x nz voxels of the
    same size), the resin of every voxel (info['region_map']) and the
    compression's record (info['pressed'])."""
    from .solvers import compress as CP
    rv = spec["rve"]
    table = spec["labels"]
    N, h = int(plan["N"]), float(plan["h_um"])
    phases = spec["phases"]
    gph = ginfo_phases_spec(spec)
    # the resin of every voxel at the start: the main matrix, the matrix
    # regions in their slabs (also under the particles); the pores as they
    # were generated (they flow with the resin and close)
    base = np.zeros((N, N, N), np.uint8)
    for li, t in enumerate(table):
        if t["kind"] == "matrix" and t.get("section"):
            sec = t["section"]
            ax = "xyz".index(sec["axis"])
            k0, k1 = int(round(sec["from"] * N)), int(round(sec["to"] * N))
            sl = [slice(None)] * 3
            sl[ax] = slice(k0, k1)
            base[tuple(sl)] = li
    pore = np.zeros(len(table), bool)
    for i, ph in enumerate(phases):
        if ph["void"]:
            pore[i + 1] = True
            base[gen_labels == i + 1] = i + 1
    parts = ginfo["particles"]
    solid = np.array([not phases[int(p)]["void"] for p in parts["phase"]], bool)
    sub = {k: np.asarray(v)[solid] for k, v in parts.items() if k in ("codes", "P", "R", "phase", "axes", "centers")}
    rad = np.asarray(sub["P"], float)[:, 0] if len(sub["codes"]) else np.zeros(0)
    planes, nfaces = ginfo["planes"]
    overlap_of, net_of, _ = GEN._phase_arrays(gph, "keep")
    n_sp = len(rad)
    ext = (2.0 * float(rad.max()) if n_sp else h)
    buf = np.empty(int(math.ceil(ext / h) + 6) ** 3, np.int64)
    sph_label = (np.asarray(sub["phase"], np.int64) + 1) if n_sp else np.zeros(0, np.int64)
    slab_of = np.zeros(n_sp + 2, np.int64)
    slab_of[1:n_sp + 1] = sph_label

    def raster(shape, centres):
        owner = np.zeros(shape, np.int32)
        if n_sp:
            phase_of = np.full(n_sp + 2, -1, np.int32)
            GEN.raster_all(owner, phase_of, overlap_of, net_of, float(h), sub["codes"].astype(np.int64),
                           np.ascontiguousarray(centres), np.ascontiguousarray(sub["R"]),
                           np.ascontiguousarray(sub["P"]), planes, nfaces, np.zeros(n_sp),
                           sub["phase"].astype(np.int64), 1, buf, False)
        return owner, slab_of
    rate = float(rv["press_rate"])
    comp = float(rv["press_compression"]) / 100.0
    lateral = rv.get("press_lateral", "xy")
    mu, gd = press_viscosities(spec, rate)
    name = "Structure generation · compression"
    side = {"xy": "x and y free", "x": "x free, y held", "none": "x and y held"}[lateral]
    detail = f"{n_sp} spheres, compressed {100 * comp:.0f} % along z ({side}), 3D periodic flow"
    event("stage", name=name, detail=detail, progress=prog.sub(0.2))
    resins = [(table[li]["name"], m) for li, m in enumerate(mu) if m is not None and (base == li).any()]
    log(f"  compression: the RVE ({N * h:.4g} µm) shortened by {100 * comp:.0f} % along z at {rate:g} 1/s, {side}; "
        f"{n_sp} rigid spheres; resins " + ", ".join(f"{n} {m:.4g} Pa·s" for n, m in resins)
        + f" (at the equivalent shear rate {gd:.3g} 1/s)"
        + (f"; pores {100 * float(np.mean(pore[base])):.1f} vol%" if pore.any() else ""))

    def cb(step, n, c, szz):
        event("stage", name=name, detail=f"{detail} · step {step} of {n}, compression {100 * c:.1f} %",
              progress=prog.sub(0.2 + 0.8 * min(step / max(n, 1), 0.999)))
    cure, cure_info = (press_cure(spec, base, mu, log) if rv.get("press_cure") else (None, None))
    if rv.get("press_cure") and cure is None:
        log(f"  (the compression runs without cure: {cure_info})")
        cure_info = None
    if cure is not None:
        log(f"  cure at a mould temperature of {rv['press_mold_C']:g} °C: "
            + ", ".join(f"{table[li]['name']} (gel at {cure.kin[li]['ag']:g})" for li in cure.labels)
            + f"; heat exchange with the mould in {cure_info['tau_s']:.3g} s ({rv['press_part_mm']:g} mm part); "
            + ("closing at the strain rate" if rv["press_control"] == "rate" else
               f"closing at no more than {rv['press_stress_Pa']:g} Pa axial stress (at most {rate:g} 1/s)"))
    cpr = CP.Compress(base, h, mu, pore, np.ascontiguousarray(sub["centers"], float) if n_sp else np.zeros((0, 3)),
                      rad, sph_label, raster, lateral=lateral, compression=comp, rate=rate,
                      steps=rv.get("press_steps"), log=log, should_stop=should_stop, progress=cb, cure=cure,
                      stress_limit=(rv["press_stress_Pa"] if (cure is not None and rv["press_control"] == "stress")
                                    else None),
                      t_max=rv["press_time_s"] if cure is not None else None,
                      frame_dir=steps_dir)
    if steps_dir:
        shutil.rmtree(steps_dir, ignore_errors=True)
    out = cpr.run()
    shape = tuple(out["shape"])
    owner, fld = out["owner"], out["field"]
    # the generator's labels of the compressed box: the spheres, the pores
    lab = np.zeros(shape, np.uint8)
    for i, ph in enumerate(phases):
        if ph["void"]:
            lab[fld == i + 1] = i + 1
    inside = owner > 0
    lab[inside] = slab_of[owner[inside]].astype(np.uint8)
    region_map = np.where(np.isin(fld, [li for li, m in enumerate(mu) if m is not None]), fld, 0).astype(np.uint8)
    newp = {k: v.copy() for k, v in sub.items()}
    newp["centers"] = np.ascontiguousarray(out["centres_grid"]) if n_sp else np.zeros((0, 3))
    want_cf = ginfo.get("contact_mask") is not None
    cf = np.zeros(shape, np.uint8)
    phase_of = np.full(n_sp + 2, -1, np.int32)
    phase_of[1:n_sp + 1] = sub["phase"].astype(np.int32) if n_sp else phase_of[1:1]
    n_contact = int(GEN.contact_faces(owner, phase_of, net_of, cf, True)) if n_sp else 0
    n_touch = GEN.touching_pairs(owner, phase_of, net_of, True) if n_sp else 0
    nvox = int(np.prod(shape))
    counts = np.bincount(lab.ravel(), minlength=len(phases) + 1)
    pinfo = []
    for i, ph in enumerate(phases):
        m = lab == (i + 1)
        faces = sum(int(np.count_nonzero(m != np.roll(m, -1, axis=ax))) for ax in range(3))
        old = dict(ginfo["phases"][i])
        old.update({"vf_voxel": float(counts[i + 1] / nvox), "surface_voxel_um2": faces * h * h,
                    "n_particles": int(np.count_nonzero(newp["phase"] == i)) if n_sp else 0})
        if ph["void"]:
            old["surface_true_um2"] = None
        pinfo.append(old)
    shares = {}
    for li, rr in out["regions"].items():
        shares[table[li]["name"]] = {"volume0_um3": rr["volume0_um3"], "volume_um3": rr["volume_um3"][-1],
                                     "share0": rr["volume0_um3"] / float(N * h) ** 3,
                                     "share": rr["volume_um3"][-1] / float(np.prod(out["L_um"])),
                                     "strain": rr["strain"], "strain_all": rr["strain_all"],
                                     "share_all": rr["volume_all_um3"][-1] / float(np.prod(out["L_um"])),
                                     "vm_Pa": rr["vm_Pa"]}
    s = out["series"]
    eta_c = -s["sigma_zz_Pa"][-1] / rate
    info = {"particles": newp, "planes": ginfo["planes"], "poly_meta": ginfo.get("poly_meta"), "film": False,
            "shape": list(shape), "N": shape[0], "h_um": h, "L_um": shape[0] * h, "region_map": region_map,
            "phases": pinfo, "contact_faces": n_contact, "touching_pairs": n_touch,
            "contact_mask": cf if want_cf else None, "contacts_mode": "keep", "contacts_removed": 0,
            "voxel_conflicts": 0, "route": ginfo["route"] + f", then compressed {100 * comp:.0f} % along z ({side}; "
                                                            f"3D periodic resin flow, {out['steps']} steps)",
            "seconds": float(ginfo.get("seconds", 0.0)) + out["seconds"]}
    info["pressed"] = {
        "compression": comp, "rate_1_s": rate, "lateral": lateral, "steps": out["steps"], "seconds": out["seconds"],
        "box0_um": [N * h] * 3, "box_um": [n * h for n in shape], "shape": list(shape), "gd_eq": gd,
        "series": s, "regions": shares, "pore_fraction": out["pore_fraction"],
        "n_spheres": n_sp, "crossed_resin": out["crossed"], "max_overlap": out["max_overlap"],
        "eta_compression_Pa_s": eta_c, "resins": {n: m for n, m in resins},
        "cure": ({"mold_C": rv["press_mold_C"], "control": rv["press_control"], "stress_Pa": rv["press_stress_Pa"],
                  "part_mm": rv["press_part_mm"], "tau_s": cure_info["tau_s"], "stop": out.get("stop"),
                  "resins": {table[li]["name"]: {"alpha": s["alpha"][int(li)], "eta_Pa_s": s["eta_Pa_s"][int(li)],
                                                 "gel": cure.kin[li]["ag"], "gelled": cure.gelled(li)}
                             for li in cure.labels},
                  "time_s": s["time_s"], "T_C": s["T_C"], "rate_1_s": s["rate_1_s"]} if cure is not None else None),
        "frames": out["frames"], "von_mises": out["von_mises"], "pressure": out["pressure"]}
    log("  compression done: box " + " × ".join(f"{v:.4g}" for v in info["pressed"]["box_um"]) + " µm "
        f"({' × '.join(str(n) for n in shape)} voxels); axial stress {-s['sigma_zz_Pa'][-1]:.4g} Pa at the end "
        f"(compression viscosity {eta_c:.4g} Pa·s); "
        + "; ".join(f"{k}: {100 * v['share0']:.1f} -> {100 * v['share']:.1f} % of the box, strain with its particles "
                    f"z {v['strain_all'][2]:+.3f}, x {v['strain_all'][0]:+.3f}, y {v['strain_all'][1]:+.3f} (the resin "
                    f"alone z {v['strain'][2]:+.3f}), von Mises in the resin {v['vm_Pa'][-1]:.3g} Pa"
                    for k, v in shares.items())
        + (f"; pores {100 * out['pore_fraction'][0]:.1f} -> {100 * out['pore_fraction'][1]:.1f} vol%" if pore.any() else "")
        + f"; {out['crossed']} of {n_sp} spheres now surrounded by another resin ({out['seconds']:.0f} s)")
    return lab, info


def ginfo_phases_spec(spec):
    """The generator's phase dicts of a spec (the press redraws its film with them)."""
    return [dict(ph) for ph in _generator_phases(spec)]


def shear_packing(spec, log, event, prog, should_stop):
    """The function generate() calls to shear a sphere packing in the resin
    (None for a random packing): the particle dynamics of the viscosity
    analysis (its options: lubrication, contacts with friction, van der
    Waals, adhesion; on the GPU when there is one) starts from the packed
    spheres in the RVE's own box and shears them to a whole-number strain,
    where the sheared images line up with the periodic ones again. The flow
    runs along x with its velocity gradient through z - across the gap a
    compound flows in - and the vorticity along y."""
    if spec["rve"].get("arrangement") != "sheared":
        return None
    from .solvers import suspension as SU
    vo = spec["options"]["viscosity"]
    do = vo["dem"]
    solid = shear_phases(spec)
    hks, _ = dem_hamaker(spec, solid, do)
    hk_of = dict(zip(solid, hks))
    gd = vo["gd_ref"]
    mu = float(VI.matrix_flow(vo["model"], [gd])[0])
    S = float(int(spec["rve"]["shear_strain"]))

    def fn(pos, rad, ph, L):
        name = "Structure generation · shear flow"
        dev = SU.device_text(do["backend"], log)
        detail = (f"{len(rad)} spheres of the packing sheared in the resin to a strain of {S:g} on {dev} "
                  f"(particle dynamics, each sphere with half the packing's gap around it)")
        last = {"t": 0.0, "s": -1.0}

        def cb(strain, eta, its):
            now = time.time()
            if now - last["t"] >= 1.0:
                last["t"] = now
                event("stage", name=name, detail=f"{detail} · strain {strain:.2f} of {S:g}, μr {eta:.3g}",
                      progress=prog.sub(0.2 + 0.8 * min(strain / S, 0.999)))
            if strain >= last["s"] + 1.0:
                last["s"] = math.floor(strain)
                log(f"    strain {strain:.2f}: μr {eta:.4g}")
        event("stage", name=name, detail=detail, progress=prog.sub(0.2))
        log(f"  shear flow: {len(rad)} spheres sheared in the resin to a strain of {S:g} (flow along x, its gradient "
            f"through z), resin {mu:.4g} Pa·s at {gd:g} 1/s, friction {do['mu_f']:g}, roughness {do['roughness_nm']:g} nm")
        # the solver shears in its xy plane: x the flow, y the gradient, z the vorticity
        Xd = np.column_stack([pos[:, 0], pos[:, 2], np.mod(-pos[:, 1], L)])
        out = SU.shear_viscosity(np.asarray(rad, float) * 1e-6, 0.5, mu, gd, roughness_m=do["roughness_nm"] * 1e-9,
                                 hmin_m=do["hmin_nm"] * 1e-9,
                                 hamaker_J=np.array([hk_of.get(int(i), 1e-20) for i in ph]), mu_f=do["mu_f"],
                                 strain=S, backend=do["backend"], log=log, progress=cb, should_stop=should_stop,
                                 bound_m=do["bound_nm"] * 1e-9, adhesion_J_m2=do["adhesion_mJ_m2"] * 1e-3, frames=0,
                                 start_m=Xd * 1e-6, box_m=float(L) * 1e-6)
        F = np.asarray(out["final_m"], float) * 1e6
        SU.release()                       # the device memory goes back before the solves
        new = np.column_stack([F[:, 0], np.mod(-F[:, 2], L), F[:, 1]])
        # how far the flow rearranged them: the motion across the flow (y, z) -
        # along x each sphere is carried by the shear at the heights it passed
        d = (new - pos)[:, 1:]
        d -= L * np.round(d / L)
        st = np.asarray(out["strain"])
        m = st >= SU.settle_of(S)
        cpp = float(np.mean(np.asarray(out["contacts_per_particle"])[m])) if m.any() else None
        info = {"strain": float(out["final_strain"]), "mu_r": float(out["eta_mean"]), "mu_r_se": float(out["eta_se"]),
                "gd": gd, "mu_resin": mu, "n": int(len(rad)), "moved_um": float(np.median(np.linalg.norm(d, axis=1))),
                "contacts_per_particle": cpp, "backend": out["backend"], "seconds": float(out["seconds"]),
                "axes": "flow x, velocity gradient z, vorticity y"}
        log(f"  shear flow done: strain {info['strain']:.4g}; across the flow the spheres moved {info['moved_um']:.3g} µm "
            f"from the random packing (median, {info['moved_um'] / np.median(rad):.2f} radii); "
            f"{cpp if cpp is not None else 0:.2f} contacts per sphere ({out['backend']}, {info['seconds']:.0f} s)")
        return new, info
    return fn


def film_skin(spec, plan):
    """Voxels of matrix kept at each face of a film: rve.skin_um, or one voxel
    when it is not given (0 lets particles touch the faces)."""
    if not plan.get("film"):
        return 0
    s = spec["rve"].get("skin_um")
    return 1 if s is None else int(round(float(s) / plan["h_um"]))


def wants_contact_faces(spec):
    return (any((p.get("r_contact") or 0) > 0 for p in spec["phases"])
            or any(v > 0 for v in (spec.get("contact_rc") or {}).values()))


def contact_resistance(spec, i, j):
    """R_c [m^2 K/W] on the faces where a particle of phase i touches one of
    phase j: the phase's own R_c for i = j, the value set for the pair, or
    the mean of the two."""
    ph = spec["phases"]
    if i == j:
        return float(ph[i].get("r_contact") or 0.0)
    v = (spec.get("contact_rc") or {}).get(f"{min(i, j)}-{max(i, j)}")
    if v is not None:
        return float(v)
    return 0.5 * (float(ph[i].get("r_contact") or 0.0) + float(ph[j].get("r_contact") or 0.0))


def contact_pair_counts(gen_labels, contact_mask, nph):
    """Contact faces per pair of phases {"i-j": n} (i <= j, phase indices),
    from the generator's labels (phase + 1) and its contact-face bits."""
    out = {}
    if contact_mask is None:
        return out
    g = np.asarray(gen_labels).astype(np.int64)
    cm = np.asarray(contact_mask)
    for ax in range(3):
        m = (cm & (1 << ax)) != 0
        if not m.any():
            continue
        a = g[m]
        b = np.roll(g, -1, axis=ax)[m]
        ok = (a > 0) & (b > 0)
        lo, hi = np.minimum(a[ok], b[ok]) - 1, np.maximum(a[ok], b[ok]) - 1
        cnt = np.bincount(lo * nph + hi, minlength=nph * nph)
        for k in np.flatnonzero(cnt):
            key = f"{k // nph}-{k % nph}"
            out[key] = out.get(key, 0) + int(cnt[k])
    return out


def run_signature(spec):
    """Everything a result depends on except which analyses were chosen: two
    runs with the same signature give the same numbers for the analyses they
    share, so a later one may add to an earlier one."""
    s = {k: v for k, v in spec.items() if k not in ("analyses", "preview", "name")}
    return hashlib.sha1(json.dumps(s, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def save_structure(out_dir, key, gen_labels, ginfo, contact_mask):
    d = os.path.join(out_dir, "structure")
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, key + ".part.npz")
    np.savez_compressed(tmp, labels=gen_labels,
                        contact=(contact_mask if contact_mask is not None else np.zeros(0, bool)))
    os.replace(tmp, os.path.join(d, key + ".npz"))
    # written aside and renamed, as the labels: a stop ends the run's process
    # at once and must not leave half a file for the next run to reuse
    with open(os.path.join(d, key + ".part.pkl"), "wb") as fh:
        pickle.dump({k: v for k, v in ginfo.items() if k != "contact_mask"}, fh, protocol=4)
    os.replace(os.path.join(d, key + ".part.pkl"), os.path.join(d, key + ".pkl"))


def load_structure(dirs, key):
    """A kept realisation from the first of `dirs` that has it, or None."""
    for d in dirs:
        p, q = os.path.join(d, "structure", key + ".npz"), os.path.join(d, "structure", key + ".pkl")
        if not (os.path.exists(p) and os.path.exists(q)):
            continue
        try:
            with np.load(p) as z:
                gen_labels, contact = z["labels"], z["contact"]
            with open(q, "rb") as fh:
                ginfo = pickle.load(fh)
        except Exception:                                            # noqa: BLE001
            continue
        if contact.size:
            ginfo["contact_mask"] = contact
        return gen_labels, ginfo, d
    return None


def _dilute_eta(phases, log, should_stop, workers, vf=0.04):
    """[eta] of these particles from a dilute RVE (vf in total, the same mix):
    mu_r = 1 + [eta] phi to first order. The box is five times the largest
    particle and the smallest feature gets eight voxels."""
    parts = [dict(p) for p in phases]
    tot = sum(float(p["vf"]) for p in parts)
    ext, dmin = 0.0, math.inf
    for p in parts:
        p["vf"] = float(p["vf"]) * vf / tot
        st = G.phase_size_stats(p["shape"], p["size_um"], p.get("dist"))
        ext = max(ext, st["extent_max"])
        dmin = min(dmin, st["min_dim_p10"])
    L = 5.0 * ext
    N = int(min(128, max(32, round(8.0 * L / dmin))))
    h = L / N
    lab, info = GEN.generate(N, h, parts, seed=17, log=lambda *a: None, should_stop=should_stop, contacts="apart")
    phi = float((lab > 0).mean())
    mu = [1.0] + [VI.RIGID_RATIO] * len(parts)
    vr = VI.effective_viscosity(lab.astype(np.int32), mu, tol=1e-6, keep_field=False, extension=False,
                                workers=workers, should_stop=should_stop)
    eta = VI.intrinsic_from_rve(vr["mu_r"], phi)
    log(f"    intrinsic viscosity from a dilute RVE: [η] = {eta:.3f} (φ {phi:.4f}, μr {vr['mu_r']:.4f}, {N}³)")
    return eta, {"phi": phi, "mu_r": vr["mu_r"], "N": N, "L_um": L}


def sphere_distribution(phases, n=4000):
    """(diameters, number weights) of the solid particles when every one is a
    sphere, else None: the generator's phases (outer sizes, a coating
    included) and its truncated size distribution."""
    ds, ws = [], []
    for p in phases:
        if p.get("void") or p["shape"] == G.NETWORK or float(p.get("vf", 0)) <= 0:
            continue
        if p["shape"] != "sphere":
            return None
        sc = G.sample_scales(n, p.get("dist"), np.random.default_rng(7))
        d = float(p["size_um"]["d"]) * sc
        ds.append(d)
        ws.append(np.full(n, float(p["vf"]) / (n * float(np.mean(d ** 3)))))
    if not ds:
        return None
    return np.concatenate(ds), np.concatenate(ws)


def max_packing(phases, vo, log, should_stop):
    """phi_m for the viscosity analysis and how it was found."""
    vx = {}
    sd = sphere_distribution(phases)
    mode = vo["phi_m_mode"]
    if mode == "manual":
        vx["phi_m_rcp"], vx["phi_m_basis"] = vo["phi_m"], "entered"
    elif sd is not None and mode in ("auto", "farr_groot"):
        vx["phi_m_rcp"] = VI.rcp_farr_groot(*sd)
        vx["phi_m_basis"] = "random close packing of the size distribution (Farr-Groot)"
        log(f"    random close packing of these spheres (Farr-Groot): {vx['phi_m_rcp']:.4f}")
    else:
        solid = [p for p in phases if not p.get("void") and p["shape"] != G.NETWORK]
        try:
            vx["phi_m_rcp"], vx["jamming"] = GEN.jamming_fraction(solid, log=log, should_stop=should_stop)
            vx["phi_m_basis"] = "jammed packing of these particles"
        except InterruptedError:
            raise
        except Exception as e:                                       # noqa: BLE001
            log(f"    maximum packing fraction not computed ({e}); {vo['phi_m']} is used")
            vx["phi_m_rcp"], vx["phi_m_basis"] = vo["phi_m"], "entered (the jammed packing failed)"
    if sd is not None:
        dw, delta, S = VI.rcp_desmond_weeks(*sd)
        vx["desmond_weeks"] = {"phi": dw, "delta": delta, "skewness": S, "in_range": bool(delta <= 0.4)}
        vx["farr_groot"] = (vx["phi_m_rcp"] if "Farr-Groot" in vx["phi_m_basis"] else VI.rcp_farr_groot(*sd))
    fr = vo.get("friction", "none") == "frictional" and mode != "manual"
    vx["phi_m"] = vx["phi_m_rcp"] * (VI.FRICTION_FACTOR if fr else 1.0)
    vx["friction"] = bool(fr)
    return vx



# =========================================================================
# moisture: uptake, diffusion and hygroscopic swelling
# =========================================================================
def crank_plate(D, L, t):
    """Fractional uptake M(t)/M_inf of a plate of thickness L exposed on both
    faces (Crank, The Mathematics of Diffusion, eq. 4.18)."""
    t = np.asarray(t, float)
    out = np.ones_like(t)
    s = np.zeros_like(t)
    for n in range(200):
        k = (2 * n + 1) ** 2
        s += 8.0 / (k * math.pi ** 2) * np.exp(-k * math.pi ** 2 * D * t / (L * L))
    out -= s
    return np.clip(out, 0.0, 1.0)


def _phase_groups(table, void_labels):
    """(key, name, label ids) of the pore space and of every filler phase
    (core with its shells), as the size and percolation analyses take them."""
    groups = []
    if void_labels:
        groups.append(("pores", "Pores", list(void_labels)))
    for li, t in enumerate(table):
        if t["kind"] == "core" and li not in void_labels:
            groups.append((f"phase{li}", t["name"],
                           [li] + [j for j, u in enumerate(table) if u["kind"] == "shell" and u["phase"] == t["phase"]]))
    return groups


def _moisture_values(table):
    """Density [kg/m3], D_w [m2/s], saturated concentration [kg/m3], CME and
    the permeability P = D_w c_sat of every label."""
    rho = np.array(S.label_values(table, "rho"), float) * 1e3
    Dw = np.array([float(t["props"].get("D_w") or 0.0) for t in table])
    cs = np.array([float(t["props"].get("c_sat") or 0.0) for t in table]) / 100.0 * rho
    cme = np.array([float(t["props"].get("cme") or 0.0) for t in table])
    return rho, Dw, cs, cme, Dw * cs


def _moisture_solve(labels, table, spec, h_um, vf_labels, prog, log, event, should_stop, viewer, first,
                    C_known=None, pre=None):
    """Water in the composite moves by the gradient of its activity a = c/c_sat,
    which is continuous across an interface even where the two materials
    take up different amounts. The conduction analogue therefore carries the
    permeability P = D_w c_sat (a filler that takes up nothing is
    impermeable), and D_eff = P* / <c_sat> with <c_sat> the volume mean of the
    saturated concentration. Swelling: every region strains by CME x its own
    moisture mass fraction at saturation; the effective strain follows from
    the same thermo-elastic solve as the CTE with that eigenstrain."""
    from .solvers import fans as FA
    rho, Dw, cs, cme, P = _moisture_values(table)
    if not P.max() > 0:
        raise ValueError("no region takes up water (D_w and c_sat are not set)")
    opts = spec["options"]
    dirs = opts["directions"]
    Pd = np.full(3, np.nan)
    stats, cache = [], {}
    film = bool(spec["rve"].get("film"))
    for d in dirs:
        di = "xyz".index(d)
        v_used, policy = CD.contrast_policy(labels, P / P.max(), di, opts["contrast_cap"], cache, film=bool(spec["rve"].get("film")))
        prog.stage(f"Moisture diffusion · {d} direction", f"periodic FV · contrast policy {policy}",
                   units=0.0 if ("moisture", d) in (pre or {}) else 0.5)
        keep = first and viewer is not None and d == dirs[0]

        def cb(it, res, d=d):
            event("solver", name=f"Moisture {d}", it=it, res=res, target=opts["tol"], progress=None)
        r = (pre or {}).get(("moisture", d))
        if r is None:
            r = CD.solve_direction(labels, v_used, di, tol=opts["tol"], callback=cb, should_stop=should_stop,
                                   flux_h=(h_um * 1e-6) if keep else None, film=film)
        Pd[di] = float(r["column"][di]) * P.max()
        stats.append({"direction": d, "policy": policy, **{k: r.get(k) for k in ("iterations", "residual",
                                                                               "converged", "seconds")}})
        if keep and r.get("u") is not None:
            u = r.pop("u")
            if r.get("absolute_potential"):
                act = u
            else:
                n = labels.shape[di]
                shp = [1, 1, 1]
                shp[di] = n
                act = ((np.arange(n) + 0.5) / n).reshape(shp) + u / n
            act = np.where(cs[labels] > 0, act, np.nan)
            viewer.add_field(f"moist_{d}_a", act, f"Moisture activity c/c_sat ({d.upper()} gradient)", "-",
                             "Moisture")
            q = r.pop("q", None)
            if q is not None:
                jm = np.sqrt((q ** 2).sum(-1))
                jm = jm / max(float(np.mean(jm)), 1e-300)
                viewer.add_field(f"moist_{d}_j", jm, f"Moisture flux |j| / mean ({d.upper()} gradient)", "-",
                                 "Moisture")
                _vector_layer(viewer, f"moist_{d}_j", q, None, d, h_um, "Moisture flux j", "-", "Moisture",
                              f"moist_{d}_j")
            del u
        log(f"    moisture permeability {d}: {Pd[di]:.4g} kg/m·s per unit activity "
            f"({r.get('iterations')} iterations)")
        prog.end()
    c_mean = float(np.dot(vf_labels, cs))
    rho_c = float(np.dot(vf_labels, rho))
    wt = 100.0 * c_mean / rho_c
    out = {"P": Pd.tolist(), "D_eff": (Pd / c_mean).tolist(), "c_sat_kg_m3": c_mean, "wt_pct": wt,
           "rho_kg_m3": rho_c, "stats": stats, "D_label": Dw.tolist(), "c_label": cs.tolist(), "swelling": None}
    eps = cme * cs / np.maximum(rho, 1e-30)
    if eps.max() > 0:
        prog.stage("Moisture swelling", "eigenstrain at saturation" + (
            " · stiffness of the elastic solve reused" if C_known is not None else " · 7 load cases"), units=1.0)
        E = S.label_values(table, "E")
        nu = S.label_values(table, "nu")
        try:
            import numba
            workers = int(numba.get_num_threads())
        except Exception:                                                # noqa: BLE001
            workers = -1
        rf = FA.homogenize(labels, E, nu, eps * 1e6, tol=1e-6, workers=workers, should_stop=should_stop,
                           C_known=C_known,
                           progress=lambda tag, it, res: event("solver", name=f"Swelling {tag}", it=it, res=res,
                                                               target=1e-6, progress=None))
        e_h = np.asarray(rf["alpha"], float) * 1e-6
        out["swelling"] = {"strain": e_h.tolist(), "strain_vol": float(np.mean(e_h[:3])),
                           "cme_eff": (e_h[:3] / (wt / 100.0)).tolist() if wt > 0 else None,
                           "eps_label": eps.tolist(), "reused_stiffness": C_known is not None}
        log(f"    swelling at saturation {1e3 * np.mean(e_h[:3]):.3f} ‰ (CME "
            f"{np.mean(e_h[:3]) / (wt / 100.0):.3f} per mass fraction)")
        prog.end()
    log(f"    moisture: D_eff {np.nanmean(Pd / c_mean):.4g} m²/s, saturated uptake {wt:.3f} wt%")
    return out


def _shear_flow_layer(viewer, vel, h_um, n_lines=36):
    """The resin flow of simple shear in the xy plane for the viewer: speed
    over the wall speed (unit shear rate times half the RVE), arrows, and
    streamlines. The upper half moves in +x and the lower half in -x, so each
    half is traced on its own and the lower lines are reversed: every line,
    and every tracer the viewer moves along it, then runs the way the resin
    does."""
    from . import streamlines as SL
    ny = vel.shape[1]
    vmax = 0.5 * ny
    v = (vel / vmax).astype(np.float32)
    speed = np.sqrt((v ** 2).sum(-1))
    grp = "Viscosity"
    viewer.add_field("visc_v", speed, "Resin speed |v| / wall speed (simple shear, xy)", "-", grp,
                     note="Layers slide in x; the fillers deflect and squeeze the resin between them",
                     in_matrix=True)
    viewer.add_vector("visc_v", v, "Resin velocity (simple shear, xy)", "-", grp, field="visc_v",
                      note="Arrow length and colour follow the local speed")
    Y = np.arange(ny)[None, :, None]
    lines = []
    summ = {}
    for half, mask in (("upper", np.broadcast_to(Y >= ny // 2, vel.shape[:3])),
                       ("lower", np.broadcast_to(Y < ny // 2, vel.shape[:3]))):
        ls, sm = SL.trace(v, np.ascontiguousarray(mask), h_um, 0, n_lines=n_lines // 2, seed_quantile=0.15)
        if half == "lower":
            ls = [{**l, "points_um": l["points_um"][::-1], "speed": (l.get("speed") or [])[::-1]} for l in ls]
        lines += ls
        summ[half] = sm
    if lines:
        viewer.add_paths("visc_v", lines, "Resin streamlines (simple shear, xy)", grp, kind="streamline",
                         summary={"n_lines": len(lines)},
                         note="Animated: tracers move at the local resin speed",
                         speed_label="Resin speed / wall speed", speed_unit="-")
    return {"n_lines": len(lines)}


def _moisture_entry(spec, opts, table, realisations, results_first, checks):
    m0 = results_first["moisture"]
    mo = opts["moisture"]
    D = np.array(m0["D_eff"], float)
    Dz = float(D[2]) if np.isfinite(D[2]) else float(np.nanmean(D))
    vf = np.array(realisations[0]["vf_labels"], float)
    Dl = np.array(m0["D_label"])
    cl = np.array(m0["c_label"])
    i_m = 0
    Dm = float(Dl[i_m])
    # one face sealed: half of a plate twice as thick
    L = mo["thickness_mm"] * 1e-3 * (2.0 if mo["sides"] == "one" else 1.0)
    t95 = 1.0
    tt = np.logspace(-4, 1.5, 400) * L * L / Dz
    fr = crank_plate(Dz, L, tt)
    t50 = float(np.interp(0.5, fr, tt))
    t95 = float(np.interp(0.95, fr, tt))
    curve_t = np.logspace(math.log10(t50 / 100.0), math.log10(t95 * 3.0), 120)
    curve_f = crank_plate(Dz, L, curve_t)
    at = float(crank_plate(Dz, L, [mo["hours"] * 3600.0])[0])
    refs = {}
    others = [i for i in range(len(table)) if i != i_m and vf[i] > 0]
    phi = float(sum(vf[i] for i in others))
    if Dm > 0 and all(cl[i] == 0 for i in others):
        # impermeable, non-absorbing fillers (Maxwell, spheres): D*/D_m = 2/(2+phi)
        refs["Maxwell, impermeable spheres (D*/D_matrix)"] = 2.0 / (2.0 + phi)
        refs["Uptake if only the matrix absorbs (wt%)"] = 100.0 * (1 - phi) * cl[i_m] / m0["rho_kg_m3"]
    entry = {"title": "Moisture uptake, diffusion and swelling", "D_eff": D.tolist(), "D_z": Dz, "D_matrix": Dm,
             "ratio": (D / Dm).tolist() if Dm > 0 else None, "wt_pct": m0["wt_pct"],
             "c_sat_kg_m3": m0["c_sat_kg_m3"], "thickness_mm": mo["thickness_mm"], "sides": mo["sides"],
             "t50_h": t50 / 3600.0, "t95_h": t95 / 3600.0, "hours": mo["hours"], "fraction_at_hours": at,
             "wt_pct_at_hours": at * m0["wt_pct"],
             "curve": {"t_h": (curve_t / 3600.0).tolist(), "fraction": curve_f.tolist(),
                       "wt_pct": (curve_f * m0["wt_pct"]).tolist()},
             "swelling": m0.get("swelling"), "refs": refs, "stats": m0["stats"]}
    cte = results_first.get("cte")
    sw = m0.get("swelling")
    if cte is not None and sw:
        av = float(np.mean(cte["alpha"][:3]))
        if av > 0:
            # the temperature rise that strains the composite as much as saturation
            entry["equivalent_dT"] = sw["strain_vol"] / (av * 1e-6)
    if "Maxwell, impermeable spheres (D*/D_matrix)" in refs and Dm > 0:
        rat = float(np.nanmean(D)) / Dm
        ref = refs["Maxwell, impermeable spheres (D*/D_matrix)"]
        checks.append({"id": "moist_maxwell", "group": "Moisture", "label": "Diffusivity against Maxwell (impermeable spheres)",
                       "value": rat, "target": ref, "unit": "", "status": "pass" if rat <= 1.0 + 1e-6 else "warn",
                       "note": "Impermeable fillers can only slow the diffusion down (D*/D_matrix <= 1); Maxwell is "
                               "the dilute-sphere value, contacts and shape lower it further"})
    return entry

def _dem_radii(spec, solid, n_target, seed=11):
    """Sphere radii (m) and phase of each particle for the particle dynamics:
    every phase in proportion to its volume fraction, sizes drawn from its
    distribution; a particle of another shape has the radius of the sphere
    of its volume (_dem_bodies gives its shape). A size spread of at least
    CV 0.1 is used - equal spheres order into sliding layers under shear,
    which no real powder does. The largest phase gets at least 40
    particles, so the box is several of its diameters wide, and the box is
    at least twice as wide as the longest particle (a platelet or a rod
    would otherwise meet its own periodic image)."""
    rng = np.random.default_rng(seed)
    phases = [spec["phases"][i] for i in solid]
    dists, mean_v, d_eq, ext = [], [], [], []
    for p in phases:
        dd = dict(p.get("dist") or {})
        if dd.get("type") != "table":
            # a measured distribution is used as it is; otherwise at least CV 0.1
            # (equal spheres order into sliding layers under shear)
            cv = float(dd.get("cv", 0.0) or 0.0) if dd.get("type") in ("lognormal", "normal") else 0.0
            dd = (dict(dd, cv=max(cv, 0.1)) if dd.get("type") in ("lognormal", "normal")
                  else {"type": "lognormal", "cv": 0.1})
        dists.append(dd)
        if p["shape"] == "sphere":
            d_eq.append(float(p["size_um"]["d"]))
            # the 40 largest set the box; a broad (measured) distribution's
            # largest sphere must still fit twice
            ext.append(float(p["size_um"]["d"]) * G.scale_bounds(dd)[1])
        else:
            st = G.phase_size_stats(p["shape"], p["size_um"], dd)
            d_eq.append(float(st["eqd_median"]))
            ext.append(float(st["extent_max"]))
        sc = G.sample_scales(20000, dd, rng)
        mean_v.append(float(np.mean((d_eq[-1] * sc) ** 3)))
    vf = np.array([p["vf"] for p in phases], float)
    has = vf > 0                                   # a phase at 0 % gets no particle at all
    share = vf / np.array(mean_v)
    share = share / share.sum()
    counts = np.where(has, np.maximum(1, np.round(share * n_target)), 0).astype(int)
    big = int(np.argmax(np.where(has, d_eq, -1.0)))
    if counts[big] < 40:
        counts = np.where(has, np.maximum(1, np.round(counts * 40.0 / counts[big])), 0).astype(int)
    box = (float(np.sum(counts * np.pi / 6.0 * np.array(mean_v))) / float(vf.sum())) ** (1.0 / 3.0)
    if box < 2.0 * max(ext):
        counts = np.maximum(1, np.round(counts * (2.0 * max(ext) / box) ** 3)).astype(int)
    radii, owner = [], []
    for k, (p, dd) in enumerate(zip(phases, dists)):
        radii.append(0.5e-6 * d_eq[k] * G.sample_scales(int(counts[k]), dd, rng))
        owner.append(np.full(int(counts[k]), k))
    return np.concatenate(radii), np.concatenate(owner)


def _dem_bodies(spec, solid, radii, owner, max_spheres=256, seed=11):
    """The particles of the particle dynamics that are not spheres (None when
    all are): each a rigid clump of overlapping spheres (shapes.dem_template:
    the packing template for compact shapes, a lattice of spheres of half
    the thickness through flat and slender ones, at most max_spheres),
    scaled to its volume-equivalent radius from _dem_radii and turned as
    its phase's orientation says. The format of
    solvers.suspension.pack_clumps."""
    from . import shapes as SH
    phs = [spec["phases"][i] for i in solid]
    if all(p["shape"] == "sphere" for p in phs):
        return None
    rng = np.random.default_rng(seed + 1)
    nominal, planes, nfaces, meta = GEN.phase_templates(phs, rng)
    nb = len(radii)
    tpl = np.zeros(nb, np.int64)
    scale = np.ones(nb)
    vol, area, circ = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    rot = np.zeros((nb, 3, 3))
    rotmode = np.full(nb, -1, np.int64)
    templates, info = [], []
    for k, p in enumerate(phs):
        sel = np.flatnonzero(owner == k)
        if not len(sel):
            continue
        nom = nominal[k]
        code = int(nom["code"])
        ids = list(nom.get("poly_ids") or [0])
        var = (rng.choice(ids, len(sel)) if code == 6 and len(ids) > 1
               else np.full(len(sel), ids[0] if code == 6 else -1))
        nsph = []
        for t_id in np.unique(var):
            Pn = nom["P"].copy()
            if code == 6:
                Pn[1] = t_id
            q, r = SH.dem_template(code, Pn, planes, nfaces, max_spheres)
            v1 = float(SH.volume(code, Pn, meta))
            a1 = float(SH.surface(code, Pn, meta))
            req1 = (3.0 * v1 / (4.0 * math.pi)) ** (1.0 / 3.0) * 1e-6
            m = sel[var == t_id]
            tpl[m] = len(templates)
            templates.append((np.asarray(q, float) * 1e-6, np.asarray(r, float) * 1e-6))
            scale[m] = radii[m] / req1
            vol[m] = v1 * 1e-18 * scale[m] ** 3
            area[m] = a1 * 1e-12 * scale[m] ** 2
            circ[m] = float(SH.circumradius(code, Pn)) * 1e-6
            nsph.append(len(r))
        ori = p.get("orientation") or {}
        mode, spread = ori.get("mode", "iso"), float(ori.get("spread_deg", 0.0) or 0.0)
        rot[sel] = SH.sample_rotations(len(sel), mode, spread, rng)
        rotmode[sel] = -1 if code == 0 else GEN._rotmode(mode, spread)
        info.append({"phase": p["name"], "shape": p["shape"], "spheres": int(max(nsph)), "count": int(len(sel))})
    return {"templates": templates, "tpl": tpl, "scale": scale, "rot": rot, "rotmode": rotmode, "circ": circ,
            "vol": vol, "area": area, "phase": np.asarray(owner, np.int64), "info": info}


def _dem_from_rve(spec, solid, ginfo, max_spheres=256):
    """The particle dynamics' particles taken from a generated RVE - its
    particles where they are, with their sizes and orientations - so that
    particles of other shapes are not packed a second time (a 4 µm box of
    alumina and BN platelets is 150,000 member spheres to pack). (radii,
    owner, bodies) as _dem_radii and _dem_bodies give them, bodies with
    'X' and 'box_m'; None when the RVE cannot serve: spheres only (their own
    packing is quick and validated), a film, shells, or a box shorter than
    1.9 times the longest particle."""
    from . import shapes as SH
    if not ginfo or ginfo.get("film") or ginfo.get("shape") or not ginfo.get("particles"):
        return None
    phs = [spec["phases"][i] for i in solid]
    if all(p["shape"] == "sphere" for p in phs) or any(p.get("shell") for p in phs):
        return None
    parts = ginfo["particles"]
    pha = np.asarray(parts["phase"])
    sel = np.flatnonzero(np.isin(pha, solid))
    if not len(sel):
        return None
    codes, Pp, Rp, Cp = (np.asarray(parts[k])[sel] for k in ("codes", "P", "R", "centers"))
    L_um = float(ginfo["L_um"])
    ext = max(2.0 * SH.circumradius(int(c), P) for c, P in zip(codes, Pp))
    if L_um < 1.9 * ext:
        return None
    planes, nfaces = ginfo["planes"]
    meta = ginfo.get("poly_meta") or []
    kmap = {i: k for k, i in enumerate(solid)}
    owner = np.array([kmap[int(p)] for p in pha[sel]], np.int64)
    nominal, _, _, _ = GEN.phase_templates([dict(p, size_um=p["size_um"]) for p in phs], np.random.default_rng(1))
    nb = len(sel)
    tpl = np.zeros(nb, np.int64)
    scale, vol, area, circ = np.ones(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
    rotmode = np.full(nb, -1, np.int64)
    templates, tkey, info = [], {}, []
    for j in range(nb):
        k, code, P = int(owner[j]), int(codes[j]), Pp[j]
        Pn = nominal[k]["P"].copy()
        t_id = int(P[1]) if code == 6 else -1
        if code == 6:
            Pn[1] = t_id
        key = (k, t_id)
        if key not in tkey:
            q, r = SH.dem_template(code, Pn, planes, nfaces, max_spheres)
            tkey[key] = len(templates)
            templates.append((np.asarray(q, float) * 1e-6, np.asarray(r, float) * 1e-6))
        tpl[j] = tkey[key]
        scale[j] = float(P[0]) / float(Pn[0])
        vol[j] = float(SH.volume(code, P, meta)) * 1e-18
        area[j] = float(SH.surface(code, P, meta)) * 1e-12
        circ[j] = float(SH.circumradius(code, Pn)) * 1e-6
        if code != 0:
            ori = phs[k].get("orientation") or {}
            rotmode[j] = GEN._rotmode(ori.get("mode", "iso"), float(ori.get("spread_deg", 0.0) or 0.0))
    for k, p in enumerate(phs):
        m = owner == k
        if m.any():
            info.append({"phase": p["name"], "shape": p["shape"], "count": int(m.sum()),
                         "spheres": int(max(len(templates[t][1]) for t in np.unique(tpl[m])))})
    radii = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    bodies = {"templates": templates, "tpl": tpl, "scale": scale, "rot": np.asarray(Rp, float), "rotmode": rotmode,
              "circ": circ, "vol": vol, "area": area, "phase": owner, "info": info,
              "X": np.mod(np.asarray(Cp, float), L_um) * 1e-6, "box_m": L_um * 1e-6}
    return radii, owner, bodies


# [eta] of shapes without a closed form in viscosity.intrinsic_viscosity:
# cubes 3.1 (Mallavajula, Archer and Koch, Phys. Rev. E 88 (2013) 052302)
_ETA_SHAPE = {"cube": 3.1}


def dem_intrinsic(spec, solid, given=None):
    """[eta] of the filler for the particle dynamics' own stresslet of its
    particles (volume-weighted), and where it comes from: the closed forms
    (Einstein, Simha, cubes), else `given` (the analysis' dilute RVE of the
    same particles), else the sphere value. `given` may be (value, basis)."""
    w, etas = [], []
    for i in solid:
        p = spec["phases"][i]
        e = None
        if p["shape"] in ("sphere", "spheroid", "cylinder", "spherocylinder"):
            e = VI.intrinsic_viscosity(p["shape"], G.aspect_ratio(p["shape"], p["size_um"]))
        etas.append(e if e is not None else _ETA_SHAPE.get(p["shape"]))
        w.append(float(p["vf"]))
    w = np.asarray(w)
    if all(e is not None for e in etas):
        return float(np.dot(w, etas) / w.sum()), "closed form"
    if given is not None:
        v_, b_ = given if isinstance(given, (tuple, list)) else (given, "dilute RVE")
        return float(v_), str(b_)
    return 2.5, "sphere value (no closed form for this shape here)"


def _eta_closed(spec, solid):
    """Volume-weighted [eta] of the solid phases from the closed forms
    (VI.intrinsic_viscosity), or None when a phase has none."""
    etas = [VI.intrinsic_viscosity(spec["phases"][i]["shape"],
                                   G.aspect_ratio(spec["phases"][i]["shape"], spec["phases"][i]["size_um"]))
            for i in solid]
    if not solid or any(e is None for e in etas):
        return None
    w = np.array([spec["phases"][i]["vf"] for i in solid], float)
    return float(np.dot(w, etas) / w.sum())


def dem_hamaker(spec, solid, do):
    """Hamaker constant of each solid phase across the resin (J) and where it
    comes from: entered, Lifshitz from refractive indices, or a typical value."""
    from .solvers import suspension as SU
    mat = spec["matrix"]
    hks, hk_basis = [], []
    for i in solid:
        p = spec["phases"][i]
        if do["hamaker_J"] is not None:
            hks.append(float(do["hamaker_J"]))
            hk_basis.append("entered")
            continue
        a = SU.hamaker(p.get("material_id"), mat.get("material_id"), eps_f=p["props"].get("eps_r"),
                       eps_r=mat["props"].get("eps_r"), sigma_f=float(p["props"].get("sigma", 0.0) or 0.0))
        if a is None:
            a = 1.0e-20
            hk_basis.append("typical value (no optical data for this material)")
        else:
            hk_basis.append("Lifshitz, refractive indices" if a != SU.A_CONDUCTOR else "conductor across a polymer")
        hks.append(float(a))
    return hks, hk_basis


def _dem_viscosity(spec, solid, phi, vo, log, event, prog, should_stop, viewer=None, eta_int=None, start=None):
    """The relative viscosity of the particles from the particle dynamics at
    the reference shear rate (solvers/suspension.py): spheres, and rigid
    multi-sphere particles for the other shapes. eta_int: the intrinsic
    viscosity of the analysis' dilute RVE, for shapes without a closed form.
    start: the generated RVE's info - particles of other shapes then start
    where the RVE has them (_dem_from_rve) instead of being packed again."""
    from .solvers import suspension as SU
    do = vo["dem"]
    rve = None
    if start is not None:
        try:
            rve = _dem_from_rve(spec, solid, start)
        except Exception as e:                                           # noqa: BLE001
            log(f"    (the RVE's particles could not be taken over: {e}; the particle dynamics packs its own)")
    if rve is not None:
        radii, owner, bodies = rve
        phi = float(np.sum(bodies["vol"]) / bodies["box_m"] ** 3)
        log(f"    the particle dynamics starts from the RVE's own {len(radii)} particles (box {1e6 * bodies['box_m']:.4g} µm, "
            f"φ {phi:.3f}) - no second packing")
    else:
        radii, owner = _dem_radii(spec, solid, do["n"])
        bodies = _dem_bodies(spec, solid, radii, owner)
    eta_b = None
    if bodies is not None:
        bodies["intrinsic"], eta_b = dem_intrinsic(spec, solid, eta_int)
        for b_ in bodies["info"]:
            if b_["shape"] != "sphere":
                log(f"    {b_['phase']}: {b_['count']} rigid {b_['shape']} particles of {b_['spheres']} spheres each "
                    f"(multi-sphere)")
        log(f"    their own stresslet: [η] = {bodies['intrinsic']:.3g} ({eta_b})")
    hks, hk_basis = dem_hamaker(spec, solid, do)
    gd = vo["gd_ref"]
    mu_res = float(VI.matrix_flow(vo["model"], [gd])[0])
    tot = do["strain"]
    stage_name = "Viscosity · particle dynamics"
    dev = SU.device_text(do["backend"], log)
    what = (f"{len(radii)} spheres" if bodies is None else
            f"{len(radii)} particles ({sum(len(bodies['templates'][t][1]) for t in bodies['tpl'])} spheres in rigid "
            f"clumps)")
    detail = (f"{what} sheared in the resin at {gd:g} 1/s on {dev} - lubrication, contacts with "
              f"friction {do['mu_f']:g}, van der Waals")
    prog.stage(stage_name, detail, units=3.0)
    last = {"t": 0.0, "s": -1.0}

    def cb(strain, eta, its):
        now = time.time()
        if now - last["t"] >= 1.0:
            last["t"] = now
            event("stage", name=stage_name, detail=f"{detail} · strain {strain:.2f} of {tot:g}, μr {eta:.3g}",
                  progress=prog.sub(min(strain / tot, 0.999)))
        if strain >= last["s"] + 1.0:
            last["s"] = math.floor(strain)
            log(f"    strain {strain:.2f}: μr {eta:.4g} (CG {its} iterations)")
    live = _dem_slot("dem", f"Particle dynamics, {gd:g} 1/s", spec, solid, owner, viewer)
    try:
        early = _dem_rates_start(radii, owner, hks, phi, vo, do, gd, bodies, log)
    except Exception as e_:                                            # noqa: BLE001
        early = None
        log(f"    (the flow-curve runs could not be started beside it: {e_}; they follow it)")
    log(f"    particle dynamics: {what}, φ {phi:.3f}, resin {mu_res:.4g} Pa·s at {gd:g} 1/s, "
        f"roughness {do['roughness_nm']:g} nm, closest approach {do['hmin_nm']:g} nm, friction {do['mu_f']:g}, "
        f"bound layer {do['bound_nm']:g} nm, adhesion {do['adhesion_mJ_m2']:g} mJ/m², "
        f"Hamaker " + ", ".join(f"{spec['phases'][i]['name']} {a:.3g} J" for i, a in zip(solid, hks)))
    try:
        out = SU.shear_viscosity(radii, phi, mu_res, gd, roughness_m=do["roughness_nm"] * 1e-9,
                                 hmin_m=do["hmin_nm"] * 1e-9, hamaker_J=np.asarray(hks)[owner], mu_f=do["mu_f"],
                                 strain=tot, backend=do["backend"], log=log, progress=cb, should_stop=should_stop,
                                 bound_m=do["bound_nm"] * 1e-9, adhesion_J_m2=do["adhesion_mJ_m2"] * 1e-3, live=live,
                                 bodies=bodies)
    except BaseException:
        _dem_rates_kill(early)                 # Stop or an error: the side-by-side runs end with it
        raise
    if live is not None:
        live.done(eta=out["eta_mean"])
    prog.end()
    anim = False
    if viewer is not None and out.get("frames_um") is not None:
        # the solver's own box for the viewer: positions and speeds of the last
        # strain it ran, the phase colours of the structure
        labs = viewer.meta.get("labels") or []
        cols = []
        for k, i in enumerate(solid):
            nm = spec["phases"][i]["name"]
            lb = next((lb_ for lb_ in labs if lb_.get("name") == nm), None)
            cols.append(lb["color"] if lb else V.PHASE_COLORS[(k + 1) % len(V.PHASE_COLORS)])
        try:
            # every sphere drawn - the members of a rigid particle with its phase
            viewer.add_particles_anim(out["anim_radii_m"] * 1e6, owner[out["anim_body"]], out["frames_um"],
                                      out["frames_speed"], out["box_um"],
                                      out["frame_strain"], [spec["phases"][i]["name"] for i in solid], cols,
                                      mu_r=out["eta_mean"])
            anim = True
        except Exception as e_:                                          # noqa: BLE001
            log(f"    (the particle animation could not be written: {e_})")
    st = np.asarray(out["strain"])
    e = np.asarray(out["eta"])
    keep = max(1, len(st) // 200)
    s0 = SU.settle_of(tot)                    # the averaging starts here (as in the solver)
    res = {"mu_r": out["eta_mean"], "se": out["eta_se"], "n": out["n"], "phi": phi, "gd": gd, "mu_resin": mu_res,
           "backend": out["backend"], "gpu_fallback": out.get("gpu_fallback"),
           "seconds": out["seconds"], "steps": out["steps"], "box_um": out["box_um"],
           "strain": st[::keep].tolist(), "eta": e[::keep].tolist(),
           "eta_lub": float(np.mean(np.asarray(out["eta_lub"])[st >= s0])),
           "eta_contact": float(np.mean(np.asarray(out["eta_contact"])[st >= s0])),
           "contacts_per_particle": float(np.mean(np.asarray(out["contacts_per_particle"])[st >= s0])),
           "settle": s0,
           # over the strain the viscosity is averaged on: a start from the RVE
           # overlaps by up to a voxel (its particles are apart voxel by voxel),
           # which the contacts push apart before that
           "max_overlap": float(np.max(np.asarray(out["max_overlap"])[st >= s0])) if np.any(st >= s0)
           else float(max(out["max_overlap"])),
           "max_overlap_start": float(out["max_overlap"][0]),
           "roughness_nm": do["roughness_nm"], "hmin_nm": do["hmin_nm"], "mu_f": do["mu_f"],
           "bound_nm": do["bound_nm"], "adhesion_mJ_m2": do["adhesion_mJ_m2"], "phi_effective": out["phi_effective"],
           "hamaker": [{"phase": spec["phases"][i]["name"], "J": a, "basis": b} for i, a, b in zip(solid, hks, hk_basis)],
           "d_um": [float(2e6 * radii[owner == k].mean()) for k in range(len(solid))],
           "counts": [int((owner == k).sum()) for k in range(len(solid))], "anim": anim,
           "spheres": int(out.get("members") or len(radii)), "rigid": bodies["info"] if bodies is not None else None,
           "intrinsic": float(out.get("intrinsic", 2.5)), "intrinsic_basis": eta_b or "Einstein (spheres)"}
    # ---- the flow curve: the same particles at other shear rates ---------
    res["rate_curve"] = None
    if do["rates"] > 0:
        res["rate_curve"], res["rate_note"] = _dem_rates(radii, owner, hks, phi, vo, do, mu_res, gd, res, log, event,
                                                         prog, should_stop, bodies=bodies, early=early)
    log(f"    particle dynamics: μr {res['mu_r']:.4g} ± {res['se']:.2g} (strain {s0:g}-{tot:g}; lubrication "
        f"{res['eta_lub']:.3g}, contacts {res['eta_contact']:.3g}; {res['contacts_per_particle']:.2f} contacts per particle; "
        f"{out['backend']}, {out['seconds']:.0f} s)")
    return res


def dem_direct(spec, need):
    """Viscosity alone, of particles (no shells, no bubbles) above 35 vol%:
    the particle dynamics gives the answer and packs its own particles, so
    no voxel RVE is needed at all - unless its resin flow is asked for."""
    vo = spec["options"]["viscosity"]
    if set(need) != {"viscosity"} or not vo["dem"]["on"] or vo["rve_with_dem"]:
        return False
    # a phase set to 0 % takes no part (it stays in the form, but has no particles)
    solid = [ph for ph in spec["phases"] if not ph["void"] and ph["shape"] != G.NETWORK and ph["vf"] > 0]
    if not solid or any((ph["void"] or ph.get("shell")) and ph["vf"] > 0 for ph in spec["phases"]):
        return False
    return float(sum(ph["vf"] for ph in solid)) > 0.35


def _dem_direct_run(spec, opts, gen_phases, table, out_dir, log, event, prog, should_stop):
    """The viscosity without a voxel RVE: the composition from the recipe,
    the maximum packing of the size distribution, the particle dynamics with
    its animation written as the run's view. Returns the first-realisation
    results and the realisation record."""
    import types
    vo = opts["viscosity"]
    # the phases with particles (a phase set to 0 % has none; it would otherwise
    # get the one particle every phase is given, or forty when its are the largest)
    solid = [i for i, ph in enumerate(spec["phases"]) if not ph["void"] and ph["shape"] != G.NETWORK and ph["vf"] > 0]
    phi = float(sum(spec["phases"][i]["vf"] for i in solid))
    sph = all(spec["phases"][i]["shape"] == "sphere" for i in solid)
    why = f"{'spheres' if sph else 'particles'} at {100 * phi:.0f} vol%"
    prog.stage("Viscosity · no voxel RVE needed", f"the particle dynamics gives the answer ({why}) and packs its own "
               f"{'spheres' if sph else 'particles'}", units=0.5)
    prog.end()
    vf = [0.0] * len(table)
    for k, t in enumerate(table):
        if t["kind"] == "core":
            vf[k] = float(spec["phases"][t["phase"]]["vf"])
    vf[0] = max(0.0, 1.0 - sum(vf))
    rinfo = {"seed": spec["rve"]["seed"], "route": "particle dynamics only - no voxel RVE", "seconds": 0.0,
             "contact_pairs": {}, "contacts_mode": None, "contacts_removed": 0, "contact_faces": 0, "touching_pairs": 0,
             "voxel_conflicts": 0, "shrunk": None, "pack_scale": None, "phases": [], "vf_labels": vf,
             "interface_area_ratio": {}}
    res = {"viscosity": {"skipped": f"the particle dynamics gives the answer ({why}); no voxel RVE was generated",
                         "touching_pairs": 0}}
    prog.stage("Viscosity · maximum packing fraction", "random close packing of these particles", units=1.0)
    vx = dict(max_packing([dict(gen_phases[i], void=False) for i in solid], vo, log, should_stop))
    prog.end()
    vx["eta"], vx["eta_basis"] = dem_intrinsic(spec, solid)
    if vx["eta_basis"] != "closed form" and vo["dilute_rve"]:
        # another shape: [eta] from a dilute RVE of the same particles, as with an RVE
        prog.stage("Viscosity · intrinsic viscosity", "dilute RVE of the same particles", units=1.0)
        try:
            import numba
            vw = int(numba.get_num_threads())
        except Exception:                                                # noqa: BLE001
            vw = -1
        try:
            vx["eta"], vx["dilute"] = _dilute_eta([gen_phases[i] for i in solid], log, should_stop, vw)
            vx["eta_basis"] = "dilute RVE"
        except InterruptedError:
            raise
        except Exception as e:                                           # noqa: BLE001
            log(f"    dilute RVE failed ({e}); [η] = {vx['eta']:.3g} is used")
        prog.end()
    res["viscosity_extra"] = vx
    # the animation is the run's view: the player reads it from there
    vdir = os.path.join(out_dir, "view")
    os.makedirs(vdir, exist_ok=True)
    labs = [{"name": t["name"], "color": c} for t, c in zip(table, V.region_colors(table))]
    dv = types.SimpleNamespace(dir=vdir, meta={"labels": labs, "built": time.time()})
    dv.add_particles_anim = lambda *a, **k: V.ViewWriter.add_particles_anim(dv, *a, **k)
    try:
        res["viscosity_dem"] = _dem_viscosity(spec, solid, phi, vo, log, event, prog, should_stop, dv,
                                              eta_int=(vx["eta"], vx["eta_basis"]))
    except InterruptedError:
        raise
    except Exception as e:                                               # noqa: BLE001
        prog.end()
        res["viscosity_dem"] = {"error": f"the particle dynamics failed: {e}"}
        log(f"    particle dynamics not run: {e}")
    if dv.meta.get("dem"):
        _dump(os.path.join(vdir, "meta.json"), {"dem": dv.meta["dem"], "labels": labs, "built": dv.meta["built"]})
    return res, rinfo


def _dem_child(spec, solid, phi, vo, labels, want_anim, q, stop, live_dir=None, eta_int=None, start=None):
    """The particle dynamics in a process of its own (see _DemProcess): log
    lines, stage lines and the result go back through the queue; the live
    frames go to the run's live folder as here."""
    _LIVE["dir"] = live_dir
    try:
        def log(*a):
            q.put(("log", " ".join(str(x) for x in a)))

        def event(kind, **p):
            if kind == "stage":
                q.put(("stage", p.get("name"), p.get("detail"), p.get("progress")))

        class _Collect:                      # stands in for the viewer: keeps the animation
            def __init__(self):
                self.meta = {"labels": labels or []}
                self.anim = None

            def add_particles_anim(self, *a, **k):
                self.anim = (a, k)
        col = _Collect() if want_anim else None
        prog = Progress(event, 3.0 + float(vo["dem"]["rates"] or 0))
        res = _dem_viscosity(spec, solid, phi, vo, log, event, prog, stop.is_set, col, eta_int=eta_int, start=start)
        q.put(("done", res, col.anim if col is not None else None))
    except InterruptedError:
        q.put(("stopped",))
    except BaseException as e:                                          # noqa: BLE001
        q.put(("error", f"{type(e).__name__}: {e}"))


_LIVE_DEM = set()                       # particle-dynamics processes not yet collected
_LIVE = {"dir": None}                   # the live/ folder of the run going on in this process


class _DemProcess:
    """The particle dynamics running on the GPU in its own process while this
    process solves the RVE flow on the CPU cores."""

    def __init__(self, spec, solid, phi, vo, viewer, log, eta_int=None, start=None):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.stop = ctx.Event()
        self.log = log
        self.latest = None                   # (stage name, detail, fraction of its own work)
        self.result = None
        self.t_said = 0.0
        labels = viewer.meta.get("labels") if viewer is not None else None
        self.p = ctx.Process(target=_dem_child, args=(spec, solid, phi, vo, labels, viewer is not None, self.q,
                                                      self.stop, _LIVE["dir"], eta_int, start), daemon=True)
        self.p.start()
        _LIVE_DEM.add(self)

    def _take(self, m):
        if m[0] == "log":
            self.log(m[1])
        elif m[0] == "stage":
            self.latest = m[1:]
        else:
            self.result = m

    def poll(self):
        """Takes what the process has sent; True once it has finished."""
        import queue
        while self.result is None:
            try:
                self._take(self.q.get_nowait())
            except queue.Empty:
                break
        return self.result is not None

    def status(self):
        """One line on the particle dynamics for the flow solve's stage line, at
        most once a second (None in between)."""
        now = time.time()
        if self.latest is None or now - self.t_said < 1.0:
            return None
        self.t_said = now
        name, detail, _ = self.latest
        return f"{name.split('·', 1)[-1].strip()}: {detail}"

    def cancel(self):
        self.stop.set()
        self.p.join(timeout=15)
        if self.p.is_alive():
            self.p.terminate()
            self.p.join(timeout=5)
        _LIVE_DEM.discard(self)

    def wait(self, viewer, prog, event, should_stop, units):
        """The result, once the process has finished: a stage of its own for
        what is left after the flow solve."""
        import queue
        name = "Viscosity · particle dynamics (started with the flow solve)"
        self.poll()
        prog.stage(name, (self.latest[1] if self.latest else "finishing") if self.result is None
                   else "finished during the flow solve", units=units)
        while self.result is None:
            if should_stop is not None and should_stop():
                self.cancel()
                raise InterruptedError
            try:
                self._take(self.q.get(timeout=0.5))
            except queue.Empty:
                if not self.p.is_alive() and not self.poll():
                    self.result = ("error", f"its process ended unexpectedly (exit code {self.p.exitcode})")
            if self.result is None and self.latest and time.time() - self.t_said >= 1.0:
                self.t_said = time.time()
                event("stage", name=name, detail=f"{self.latest[0].split('·', 1)[-1].strip()}: {self.latest[1]}",
                      progress=prog.sub(self.latest[2] or 0.0))
        self.p.join(timeout=60)
        _LIVE_DEM.discard(self)
        prog.end()
        if self.result[0] == "stopped":
            raise InterruptedError
        if self.result[0] == "error":
            raise RuntimeError(self.result[1])
        res, anim = self.result[1], self.result[2]
        res["anim"] = False
        if anim is not None and viewer is not None:
            viewer.add_particles_anim(*anim[0], **anim[1])
            res["anim"] = True
        return res


def _dem_start(ginfo):
    """What the particle dynamics needs of a generated RVE (picklable for its
    own process): the particle list, the polyhedron planes, the box."""
    if not ginfo or not ginfo.get("particles"):
        return None
    return {k: ginfo.get(k) for k in ("particles", "planes", "poly_meta", "L_um", "film")}


def _dem_slot(slot, name, spec, solid, owner, viewer):
    """The live-view slot of a particle-dynamics run (None without a live
    folder): the phase names, colours and the phase of every sphere."""
    if not _LIVE["dir"]:
        return None
    nph = int(np.max(owner)) + 1 if len(owner) else 1
    names = [f"Filler {k + 1}" for k in range(nph)]
    if solid is not None:
        names = [spec["phases"][i]["name"] for i in solid]
    labs = (viewer.meta.get("labels") or []) if viewer is not None else []
    colors = []
    for k, nm in enumerate(names):
        lb = next((x for x in labs if x.get("name") == nm), None)
        colors.append(lb["color"] if lb else V.PHASE_COLORS[(k + 1) % len(V.PHASE_COLORS)])
    if solid is None and _LIVE.get("dem_info"):
        names, colors = _LIVE["dem_info"]                # the further rates: as the first run
    elif solid is not None:
        _LIVE["dem_info"] = (names, colors)
    s = LV.Slot(_LIVE["dir"], slot, name, kind="particles")
    s.static_info = {"names": names, "colors": colors, "phase": np.asarray(owner, np.uint8)}
    return s


def _dem_rate_plan(radii, hks, vo, do, gd_ref):
    """The further shear rates of the flow curve, or (None, note) when the
    attraction stays below 1 % of the viscous force over the whole range."""
    a = float(np.median(radii))
    rs = 0.5 * a
    f_att = max(hks) * rs / (6.0 * (do["hmin_nm"] * 1e-9) ** 2) + 2.0 * math.pi * do["adhesion_mJ_m2"] * 1e-3 * rs
    lo, hi = vo["gd_min"], vo["gd_max"]
    f_visc = 6.0 * math.pi * float(VI.matrix_flow(vo["model"], [lo])[0]) * a * a * lo
    if f_att < 0.01 * f_visc:
        return None, (f"the attraction stays below 1 % of the viscous force down to {lo:g} 1/s: the relative "
                      f"viscosity does not depend on the shear rate")
    rates = np.logspace(math.log10(lo), math.log10(hi), do["rates"])
    return [float(g) for g in rates if abs(math.log10(g / gd_ref)) > 0.15], None


def _dem_rate_worker(kw):
    """One flow-curve run in a worker process (the CPU, its share of the cores)."""
    os.environ["MPSIM_TI_THREADS"] = str(kw.pop("threads"))
    from .solvers import suspension as SU
    SU.init_backend("cpu")
    out = SU.shear_viscosity(**kw)
    return {"eta_mean": float(out["eta_mean"]), "eta_se": float(out["eta_se"]), "seconds": float(out["seconds"])}


def _dem_rates_start(radii, owner, hks, phi, vo, do, gd_ref, bodies, log):
    """Starts the flow-curve runs beside the reference run when the particle
    dynamics runs on the CPU: kernels on a few hundred particles keep only a
    few cores busy, so the runs at further rates go side by side in worker
    processes, each with its share of the cores. None on a GPU (one device:
    the runs follow each other) or without further rates."""
    if do["rates"] <= 0:
        return None
    from .solvers import suspension as SU
    if do["backend"] != "cpu" and SU.cuda_device_count() > 0:
        return None
    rates, _ = _dem_rate_plan(radii, hks, vo, do, gd_ref)
    if not rates:
        return None
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        cores = int(os.environ.get("MPSIM_JOB_CORES") or os.environ.get("NUMBA_NUM_THREADS") or 0) or (os.cpu_count() or 2)
    except ValueError:
        cores = os.cpu_count() or 2
    # as many at once as the free memory holds beside the reference run: a
    # process is Taichi's runtime and about 2 GB per 100 000 spheres
    n_sph = len(radii) if bodies is None else int(sum(len(bodies["templates"][t][1]) for t in bodies["tpl"]))
    per_gb = 0.4 + 2.0 * n_sph / 1e5
    try:
        import psutil
        fit = int(0.9 * psutil.virtual_memory().available / 2 ** 30 / per_gb) - 1
    except Exception:                                                  # noqa: BLE001
        fit = len(rates)
    workers = min(len(rates), fit)
    if workers < 1:
        log(f"    flow curve: the free memory holds no further run beside this one ({per_gb:.1f} GB each): "
            f"the runs follow it")
        return None
    share = max(1, cores // (workers + 1))
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    futs = {}
    for g in rates:
        mu_g = float(VI.matrix_flow(vo["model"], [g])[0])
        kw = dict(radii_m=np.asarray(radii, float), phi=phi, mu_resin=mu_g, gd=g, roughness_m=do["roughness_nm"] * 1e-9,
                  hmin_m=do["hmin_nm"] * 1e-9, hamaker_J=np.asarray(hks)[owner], mu_f=do["mu_f"], strain=do["strain"],
                  backend="cpu", bound_m=do["bound_nm"] * 1e-9, adhesion_J_m2=do["adhesion_mJ_m2"] * 1e-3,
                  frames=0.0, bodies=bodies, threads=share)
        futs[g] = ex.submit(_dem_rate_worker, kw)
    log(f"    flow curve: {len(rates)} further shear rates started beside it"
        + (f", {workers} at a time (memory, {per_gb:.1f} GB each)" if workers < len(rates) else "")
        + f", {share} core(s) each")
    return {"ex": ex, "futs": futs}


def _dem_rates_kill(early):
    """Ends the side-by-side flow-curve runs at once (Stop, an error)."""
    if early is None:
        return
    for p_ in list((getattr(early["ex"], "_processes", None) or {}).values()):
        try:
            p_.terminate()
        except Exception:                                              # noqa: BLE001
            pass
    early["ex"].shutdown(wait=False, cancel_futures=True)


def _dem_rates(radii, owner, hks, phi, vo, do, mu_ref, gd_ref, res_ref, log, event, prog, should_stop, bodies=None,
               early=None):
    """The particle dynamics at further shear rates, log-spaced over the
    flow curve's range, with the resin's viscosity at each rate. Returns the
    curve {gd, mu_r, se} (the reference rate included) and a note; None when
    the attraction stays below 1 % of the viscous force over the whole range
    - the relative viscosity then does not depend on the rate."""
    from .solvers import suspension as SU
    rates, note = _dem_rate_plan(radii, hks, vo, do, gd_ref)
    if rates is None:
        log(f"    flow curve: {note}")
        return None, note
    pts = [(gd_ref, res_ref["mu_r"], res_ref["se"])]
    stage = "Viscosity · particle dynamics, flow curve"
    if early is not None:
        # the runs started beside the reference run: wait for them
        prog.stage(stage, f"{len(rates)} further shear rates, run side by side", units=float(len(rates)))
        from concurrent.futures import wait, FIRST_COMPLETED
        ex, futs = early["ex"], early["futs"]
        try:
            pending = set(futs.values())
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if should_stop is not None and should_stop():
                    _dem_rates_kill(early)
                    raise InterruptedError
                event("stage", name=stage, detail=f"{len(futs) - len(pending)} of {len(futs)} shear rates done",
                      progress=prog.sub((len(futs) - len(pending)) / max(len(futs), 1)))
            for g, f_ in futs.items():
                out = f_.result()                     # a worker's error is raised here
                mu_g = float(VI.matrix_flow(vo["model"], [g])[0])
                pts.append((g, out["eta_mean"], out["eta_se"]))
                log(f"    flow curve: {g:.3g} 1/s, resin {mu_g:.4g} Pa·s -> μr {out['eta_mean']:.4g} ± "
                    f"{out['eta_se']:.2g} ({out['seconds']:.0f} s, side by side)")
        except BaseException:
            _dem_rates_kill(early)
            raise
        ex.shutdown(wait=False)
        prog.end()
        pts.sort()
        return ({"gd": [p[0] for p in pts], "mu_r": [p[1] for p in pts], "se": [p[2] for p in pts]},
                f"{len(pts)} shear rates from {pts[0][0]:.3g} to {pts[-1][0]:.3g} 1/s")
    prog.stage(stage, f"{len(rates)} further shear rates, {len(radii)} {'spheres' if bodies is None else 'particles'} "
               f"each, on {SU.device_text(do['backend'])}", units=float(len(rates)))
    for k, g in enumerate(rates):
        mu_g = float(VI.matrix_flow(vo["model"], [g])[0])
        last = {"t": 0.0}

        def cb(strain, eta, its, k=k, g=g):
            now = time.time()
            if now - last["t"] >= 1.0:
                last["t"] = now
                event("stage", name=stage, detail=f"{g:.3g} 1/s ({k + 1} of {len(rates)}) · strain {strain:.2f}, μr {eta:.3g}",
                      progress=prog.sub((k + min(strain / do["strain"], 0.999)) / len(rates)))
        live = _dem_slot(f"dem_rate{k + 1}", f"Particle dynamics, {g:.3g} 1/s", None, None, owner, None)
        out = SU.shear_viscosity(radii, phi, mu_g, g, roughness_m=do["roughness_nm"] * 1e-9,
                                 hmin_m=do["hmin_nm"] * 1e-9, hamaker_J=np.asarray(hks)[owner], mu_f=do["mu_f"],
                                 strain=do["strain"], backend=do["backend"], progress=cb, should_stop=should_stop,
                                 bound_m=do["bound_nm"] * 1e-9, adhesion_J_m2=do["adhesion_mJ_m2"] * 1e-3, frames=0.0,
                                 live=live, bodies=bodies)
        if live is not None:
            live.done(eta=out["eta_mean"])
        pts.append((g, out["eta_mean"], out["eta_se"]))
        log(f"    flow curve: {g:.3g} 1/s, resin {mu_g:.4g} Pa·s -> μr {out['eta_mean']:.4g} ± {out['eta_se']:.2g} "
            f"({out['seconds']:.0f} s)")
    prog.end()
    pts.sort()
    return ({"gd": [p[0] for p in pts], "mu_r": [p[1] for p in pts], "se": [p[2] for p in pts]},
            f"{len(pts)} shear rates from {pts[0][0]:.3g} to {pts[-1][0]:.3g} 1/s")


def _mu_r_at(curve, gd, fallback):
    """The relative viscosity of a particle-dynamics rate curve at the shear
    rates gd: linear in log-log between its points, flat beyond its ends."""
    if not curve:
        return fallback
    lg = np.log(np.asarray(curve["gd"], float))
    lm = np.log(np.asarray(curve["mu_r"], float))
    return np.exp(np.interp(np.log(np.maximum(np.asarray(gd, float), 1e-12)), lg, lm))


def _viscosity_entry(spec, opts, table, realisations, results_first, mus, checks):
    """The viscosity section of the result: the RVE value, the closed forms,
    Krieger-Dougherty with the jammed phi_m, the best estimate, the flow curve
    of the compound, underfill filling and settling."""
    vo = opts["viscosity"]
    v0 = results_first["viscosity"]
    rve_skipped = v0.get("skipped")          # the particle dynamics gave the answer
    vx = results_first.get("viscosity_extra") or {}
    vf = realisations[0]["vf_labels"]
    phi = float(sum(v for v, t in zip(vf, table) if t["kind"] in ("core", "shell") and float(t["props"]["E"]) > 0))
    bubbles = float(sum(v for v, t in zip(vf, table) if float(t["props"]["E"]) <= 0 and t["kind"] != "matrix"))
    mu_r = float(np.mean(mus)) if mus else v0.get("mu_r")
    ci = None
    if len(mus) >= 2:
        _, _, c = _mean_ci([[m] for m in mus])
        ci = float(c[0])
    phi_m = float(vx.get("phi_m", vo["phi_m"]))
    eta = float(vx.get("eta", 2.5))
    kd = float(VI.krieger_dougherty(phi, phi_m, eta))
    dem = results_first.get("viscosity_dem")
    dem_ok = bool(dem) and not dem.get("error")
    if phi >= 0.97 * phi_m:
        if dem_ok:
            # the RVE holds these particles at phi: a jammed packing estimate
            # below it is too low (platelets and other clumps creep toward
            # jamming and stop at the step limit), and the particle dynamics
            # has the value Krieger-Dougherty cannot give here
            lower = bool((vx.get("jamming") or {}).get("step_limited"))
            checks.append({"id": "visc_phim", "group": "Viscosity", "label": "Filler fraction below the maximum packing fraction",
                           "value": phi, "target": phi_m, "unit": "", "status": "warn",
                           "note": f"The maximum packing fraction of the jammed packing, {phi_m:.3f}"
                                   + (" (a lower estimate: its step limit was reached)" if lower else "")
                                   + f", is not above the filler fraction {phi:.3f} the RVE holds: Krieger-Dougherty is "
                                     "not defined here. The particle dynamics gives the viscosity."})
        else:
            checks.append({"id": "visc_phim", "group": "Viscosity", "label": "Filler fraction below the maximum packing fraction",
                           "value": phi, "target": phi_m, "unit": "", "status": "fail",
                           "note": "At or above φm the compound does not flow (jammed paste); Krieger-Dougherty diverges"})
    # the same curve for frictional (or frictionless) particles, as a range
    phi_m_alt = (float(vx.get("phi_m_rcp", phi_m)) if vx.get("friction") else phi_m * VI.FRICTION_FACTOR)
    kd_alt = float(VI.krieger_dougherty(phi, phi_m_alt, eta)) if phi < 0.995 * phi_m_alt else None
    mp = float(VI.maron_pierce(phi, phi_m))
    touching = int(v0.get("touching_pairs") or 0)
    resolved = phi <= 0.35 and touching == 0
    if resolved and mu_r is not None:
        best, basis = mu_r, "RVE flow solve"
    elif dem_ok:
        best, basis = float(dem["mu_r"]), "particle dynamics"
    else:
        best, basis = kd, "Krieger-Dougherty with the jammed packing fraction"
    phis = np.linspace(0.0, 0.97 * phi_m, 60)
    curve = {"phi": phis.tolist(), "kd": VI.krieger_dougherty(phis, phi_m, eta).tolist(),
             "kd_alt": [float(VI.krieger_dougherty(p, phi_m_alt, eta)) if p < 0.97 * phi_m_alt else None for p in phis],
             "mp": VI.maron_pierce(phis, phi_m).tolist(), "hs": [VI.hs_lower(p) for p in phis],
             "batchelor": [VI.batchelor(p) for p in phis], "einstein": [VI.einstein(p) for p in phis]}
    gd = np.logspace(math.log10(vo["gd_min"]), math.log10(vo["gd_max"]), 40)
    mu_m = VI.matrix_flow(vo["model"], gd)
    # where the particle dynamics is the best estimate and ran at several
    # rates, the relative viscosity follows the rate along the curve
    rc = dem.get("rate_curve") if (dem_ok and basis == "particle dynamics") else None
    mu_s, _ = VI.suspension_flow(vo["model"], _mu_r_at(rc, gd, best), phi, gd, vo["yield_Pa"])
    _, A = VI.suspension_flow(vo["model"], best, phi, [vo["gd_ref"]], vo["yield_Pa"])
    mu_m_ref = float(VI.matrix_flow(vo["model"], [vo["gd_ref"]])[0])
    mu_s_ref = float(VI.suspension_flow(vo["model"], best, phi, [vo["gd_ref"]], vo["yield_Pa"])[0][0])
    uf = vo["underfill"]
    t_resin = VI.underfill_time(mu_m_ref, uf["length_mm"] * 1e-3, uf["gap_um"] * 1e-6, uf["gamma_mN_m"] * 1e-3,
                                uf["theta_deg"])
    t_comp = VI.underfill_time(mu_s_ref, uf["length_mm"] * 1e-3, uf["gap_um"] * 1e-6, uf["gamma_mN_m"] * 1e-3,
                               uf["theta_deg"])
    settle = []
    for i, ph in enumerate(spec["phases"]):
        if ph["void"] or ph["shape"] == G.NETWORK:
            continue
        st = G.phase_size_stats(ph["shape"], ph["size_um"], ph.get("dist"))
        d = float(st.get("eqd_median") or st.get("eqd_max") or st.get("min_dim_p10") or 1.0)
        s_ = VI.settling(d * 1e-6, float(ph["props"]["rho"]) * 1000.0, vo["density_resin"], mu_m_ref, phi,
                         vo["settle_min"] * 60.0)
        settle.append({"name": ph["name"], "d_um": d, "rho": float(ph["props"]["rho"]), **s_})
    entry = {"title": "Viscosity and flowability", "phi": phi, "bubbles": bubbles, "mu_r_rve": mu_r, "ci95": ci,
             "seeds": mus, "shear": v0.get("shear"), "extension": v0.get("extension"), "stats": v0.get("stats") or {},
             "rve_skipped": rve_skipped,
             "touching_pairs": touching, "phi_m": phi_m, "phi_m_basis": vx.get("phi_m_basis", "entered"),
             "phi_m_rcp": vx.get("phi_m_rcp", phi_m), "friction": bool(vx.get("friction")),
             "phi_m_alt": phi_m_alt, "kd_alt": kd_alt, "farr_groot": vx.get("farr_groot"),
             "desmond_weeks": vx.get("desmond_weeks"),
             "jamming": vx.get("jamming"), "eta": eta, "eta_basis": vx.get("eta_basis", "sphere value"),
             "dilute": vx.get("dilute"),
             "refs": {"Einstein": VI.einstein(phi), "Batchelor": VI.batchelor(phi), "Hashin-Shtrikman lower bound": VI.hs_lower(phi),
                      "Krieger-Dougherty": kd, "Maron-Pierce": mp,
                      **({"Particle dynamics": float(dem["mu_r"])} if dem_ok else {})},
             "dem": dem,
             "kd_phi_m_fit": VI.kd_phi_m_from(mu_r, phi, eta) if mu_r is not None else None, "mu_r": best,
             "basis": basis, "curve": curve,
             "flow": {"gd": gd.tolist(), "mu_resin": mu_m.tolist(), "mu_compound": mu_s.tolist(),
                      "amplification": A, "model": vo["model"], "yield_Pa": vo["yield_Pa"],
                      "dem_rates": rc,
                      "dem_points": ({"gd": rc["gd"], "mu": [float(VI.suspension_flow(vo["model"], m_, phi, [g_],
                                                                                     vo["yield_Pa"])[0][0])
                                                            for g_, m_ in zip(rc["gd"], rc["mu_r"])]} if rc else None)},
             "gd_ref": vo["gd_ref"], "mu_resin_ref": mu_m_ref, "mu_compound_ref": mu_s_ref,
             "underfill": dict(uf, t_resin_s=t_resin, t_compound_s=t_comp),
             "settling": settle, "settle_min": vo["settle_min"]}
    hs = VI.hs_lower(phi)
    mu_chk = mu_r if mu_r is not None else best
    checks.append({"id": "visc_hs", "group": "Viscosity", "label": "Relative viscosity above the Hashin-Shtrikman lower bound",
                   "value": mu_chk, "target": hs, "unit": "", "status": "pass" if mu_chk >= 0.99 * hs else "warn",
                   "note": "Rigid particles always raise the viscosity at least this much (isotropic suspension)"})
    if not rve_skipped:
        checks.append({"id": "visc_contacts", "group": "Viscosity", "label": "Particles in voxel contact in the flow solve",
                       "value": touching, "target": 0, "unit": "pairs", "status": "pass" if touching == 0 else "warn",
                       "note": ("No resin film is left between touching voxels: they move as one rigid body and the RVE "
                                "value is too high; the best estimate uses Krieger-Dougherty" if touching else
                                "Every particle is surrounded by resin on the grid")})
    if phi > 0.35:
        checks.append({"id": "visc_dense", "group": "Viscosity", "label": "Filler fraction within the range the voxel flow solve resolves",
                       "value": phi, "target": 0.35, "unit": "",
                       "status": "pass" if dem_ok else "warn",
                       "note": "Above about 35 vol% the resin films between particles are thinner than a voxel or two; "
                               + ("the best estimate is the particle dynamics, which resolves them" if dem_ok else
                                  "the best estimate is Krieger-Dougherty with the jammed packing fraction")})
    if dem_ok:
        late = [e_ for s_, e_ in zip(dem["strain"], dem["eta"]) if s_ >= dem.get("settle", 1.0)]
        third = max(1, len(late) // 3)
        drift = abs(float(np.mean(late[-third:])) - float(np.mean(late[third:2 * third]))) / max(dem["mu_r"], 1e-9)
        checks.append({"id": "visc_dem_steady", "group": "Viscosity", "label": "Particle dynamics at steady state",
                       "value": drift, "target": 0.1, "unit": "rel",
                       "status": "pass" if drift <= 0.1 else "warn",
                       "note": "Change of the mean viscosity between the middle and the last third of the sheared strain; "
                               "a larger strain (Simulation tab) settles it further"})
        checks.append({"id": "visc_dem_overlap", "group": "Viscosity", "label": "Contact overlap in the particle dynamics",
                       "value": dem["max_overlap"], "target": 0.05, "unit": "radii",
                       "status": "pass" if dem["max_overlap"] <= 0.05 else "warn",
                       "note": "The stiff contact springs let surfaces overlap by this much of the largest radius "
                               "while the viscosity is averaged"
                               + (f" (at the start {dem['max_overlap_start']:.3f}: the RVE's particles are apart voxel by "
                                  "voxel, not exactly)" if dem.get("max_overlap_start", 0) > dem["max_overlap"] else "")})
    elif dem and dem.get("error"):
        checks.append({"id": "visc_dem", "group": "Viscosity", "label": "Particle dynamics", "value": None, "target": None,
                       "unit": "", "status": "warn", "note": dem["error"]})
    if bubbles > 0:
        checks.append({"id": "visc_bubbles", "group": "Viscosity", "label": "Bubbles treated as freely deforming",
                       "value": bubbles, "target": None, "unit": "", "status": "warn",
                       "note": "The flow solve has no surface tension: a bubble deforms freely (high capillary number) and "
                               "lowers the viscosity as 1 - 5/3 phi (dilute). Small bubbles kept spherical by surface "
                               "tension (low capillary number, Taylor) raise it as 1 + phi instead"})
    conv = [s.get("converged") for s in (v0.get("stats") or {}).values()]
    if conv:
        checks.append({"id": "visc_conv", "group": "Viscosity", "label": "Flow solves converged",
                       "value": sum(bool(c) for c in conv), "target": len(conv), "unit": "solves",
                       "status": "pass" if all(conv) else "fail", "note": ""})
    return entry


def view_feature_um(plan):
    """The smallest feature the 3D view must resolve: the P10 of the smallest
    particle dimension, or a coating or hollow wall where thinner. It sets
    the grid the particle surfaces are taken from, so a particle is drawn
    with the same detail in any size of RVE."""
    vals = []
    for f in plan.get("features") or []:
        if f.get("min_dim"):
            vals.append(float(f["min_dim"]))
        if f.get("shell"):
            vals.append(float(f["shell"]))
    return min(vals) if vals else None


def surface_owner(viewer, N, Nz, h, ginfo, gen_phases):
    """Particle numbers on the viewer's surface grid. A coarser grid is drawn
    directly, with the particles shifted so that it samples the very points
    the label pick samples - half a voxel apart otherwise, which left a skin
    of unnumbered voxels around the particles that was drawn as one extra
    surface. A grid that does not divide the box is drawn on the solver's
    grid and picked."""
    ss = viewer.surf_stride
    shape = ginfo.get("shape")
    if shape:
        # a compressed box: its own shape, periodic in all three directions
        if ss == 1 or any(n % ss for n in shape):
            own, _ = GEN.rasterize_owner(N, h, ginfo, gen_phases, shape=shape)
            return own
        own, _ = GEN.rasterize_owner(N, h * ss, ginfo, gen_phases, shape=[n // ss for n in shape],
                                     offset_um=V.pick_offset_um(ss, h))
        return own
    if ss == 1 or N % ss or (Nz and Nz % ss):
        own, _ = GEN.rasterize_owner(N, h, ginfo, gen_phases, Nz)
        return own
    own, _ = GEN.rasterize_owner(N // ss, h * ss, ginfo, gen_phases, (Nz // ss) if Nz else None,
                                 offset_um=V.pick_offset_um(ss, h))
    return own


def rebuild_view(run_dir, log=print):
    """Rebuild the 3D surfaces of a finished run from the structure it kept:
    the smooth per-particle skins, the voxel faces and the field values on
    them. The fields, arrows and lines already in view/ are kept; nothing is
    solved again. For runs made before a change in how surfaces are drawn."""
    with open(os.path.join(run_dir, "spec.normalized.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as fh:
        plan = json.load(fh)["plan"]
    gen_phases = _generator_phases(spec)
    plan.setdefault("skin", film_skin(spec, plan) if plan.get("film") else None)
    key = structure_key(gen_phases, plan, contacts_key(spec["options"]), spec["rve"]["seed"],
                        wants_contact_faces(spec), sheared=shear_key(spec), dispersion=spec["rve"].get("dispersion"))
    got = load_structure([run_dir], key)
    if got is None:
        raise FileNotFoundError("this run kept no structure (it was made before structures were kept)")
    gen_labels, ginfo, _ = got
    h = plan["h_um"]
    labels, _, _ = build_labels(spec, gen_labels, ginfo, h)
    viewer = V.ViewWriter(run_dir, labels, spec["labels"], h, feature_um=view_feature_um(plan),
                          n_particles=len(ginfo.get("particles", {}).get("codes", [])))
    viewer.adopt()
    Nz = int(plan["Nz"]) if plan.get("film") else None
    viewer.set_owner(surface_owner(viewer, plan["N"], Nz, h, ginfo, gen_phases))
    viewer.build_surfaces(log=log)
    viewer.sample_surfaces()
    return viewer.finish()


def _plan_phases(spec):
    return [{"name": ph["name"], "shape": ph["shape"], "size_um": ph["size_um"], "dist": ph["dist"],
             "vf": ph["vf"], "shell": ph["shell"], "overlap": ph["overlap"], "gap_frac": ph["gap_frac"]}
            for ph in spec["phases"]]


def build_labels(spec, gen_labels, info, h_um):
    """Generator phase map -> full label map with shells.

    A shell is what lies between a particle's outer surface and the same
    shape moved inward by the shell thickness (an exact normal offset for
    spheres, cylinders, boxes, polyhedra and wires). Returns the label map, the
    voxel-to-true interface area ratio per phase (for interface resistances)
    and notes.
    """
    from . import shapes as SH
    table = spec["labels"]
    lab = gen_labels.copy()
    parts = info["particles"]
    planes, nfaces = info["planes"]
    notes = []
    rmap = info.get("region_map")
    if rmap is not None:
        # compressed: each matrix voxel takes the resin that flowed there
        free = lab == 0
        lab[free] = rmap[free]
        notes.append("the matrix regions are where the resins flowed in the compression")
    # further matrices: the matrix voxels of their slabs (sections)
    for li, t in enumerate(table if rmap is None else []):
        if t["kind"] == "matrix" and t.get("section"):
            sec = t["section"]
            ax = "xyz".index(sec["axis"])
            n_ax = lab.shape[ax]
            k0, k1 = int(round(sec["from"] * n_ax)), int(round(sec["to"] * n_ax))
            sl = [slice(None)] * 3
            sl[ax] = slice(k0, k1)
            sub = lab[tuple(sl)]
            sub[sub == 0] = li
            notes.append(f"{t['name']} fills the matrix of {sec['axis']} {k0 * h_um:.4g}-{k1 * h_um:.4g} µm")
    for li, t in enumerate(table):
        if t["kind"] != "shell":
            continue
        i = t["phase"]
        ph = spec["phases"][i]
        sh = ph["shell"]
        th = sh["thickness_um"]
        sel = parts["phase"] == i
        if not sel.any():
            continue
        codes = parts["codes"][sel]
        Ps = parts["P"][sel]
        insets = np.full(len(codes), th)
        if sh.get("scale_with_size"):
            # the nominal particle carries `th`; every other one keeps its t/size
            P_nom = SH.params(ph["shape"], _outer_size(ph["shape"], ph["size_um"], th), [0])[1]
            insets = th * Ps[:, 0] / P_nom[0]
        tmp = np.zeros(lab.shape, np.int32)
        n = len(codes)
        phase_of = np.full(n + 4, -1, np.int32)
        zeros = np.zeros(2, np.uint8)
        ext = max(2.0 * SH.circumradius(int(c), P) for c, P in zip(codes, Ps)) + 2 * h_um
        buf = np.empty(int(math.ceil(ext / h_um) + 4) ** 3, np.int64)
        GEN.raster_all(tmp, phase_of, zeros, zeros, float(h_um), codes.astype(np.int64),
                       np.ascontiguousarray(parts["centers"][sel]), np.ascontiguousarray(parts["R"][sel]),
                       np.ascontiguousarray(Ps), planes, nfaces, insets, np.zeros(n, np.int64), 1, buf,
                       bool(info.get("film")))
        shell_mask = (gen_labels == i + 1) & (tmp == 0)
        lab[shell_mask] = li
        del tmp
    ratios = {}
    for i, ph in enumerate(spec["phases"]):
        pinfo = info["phases"][i]
        if pinfo.get("surface_true_um2"):
            ratios[i] = pinfo["surface_voxel_um2"] / pinfo["surface_true_um2"]
        elif ph.get("r_int", 0) > 0:
            ratios[i] = 1.5
            notes.append(f"'{ph['name']}': the true interface area is unknown; the voxel area was corrected by 1.5.")
    return lab, ratios, notes


def grey_levels(n):
    """Grey value per region: spread over 0..255 so every material is visible
    in an image viewer and separable by a threshold in another program."""
    if n <= 1:
        return [0]
    return [int(round(255.0 * i / (n - 1))) for i in range(n)]


def export_structure(out_dir, labels, table, h_um, vf_labels):
    """Write the structure for other programs (ImageJ, PuMA, any voxel code).

    structure.tif         8-bit, one grey value per region (0 = matrix ... 255),
                          ImageJ stack with the voxel size in um
    structure_labels.tif  8-bit region id (0, 1, 2 ...) - the material ID map
    structure_legend.csv  region id, grey value, kind, name, material, volume
                          fraction and the properties used, so a benchmark in
                          another code assigns exactly the same materials

    v3 wrote only the region ids; with values 0..3 of 255 the stack looked
    black in every viewer.
    """
    import csv
    import tifffile
    n = len(table)
    grey = np.array(grey_levels(n), np.uint8)
    zyx = np.ascontiguousarray(labels.transpose(2, 1, 0))
    meta = {"spacing": h_um, "unit": "um", "axes": "ZYX"}
    tifffile.imwrite(os.path.join(out_dir, "structure.tif"), grey[zyx], imagej=True,
                     resolution=(1.0 / h_um, 1.0 / h_um), metadata=meta)
    tifffile.imwrite(os.path.join(out_dir, "structure_labels.tif"), zyx.astype(np.uint8), imagej=True,
                     resolution=(1.0 / h_um, 1.0 / h_um), metadata=meta)
    keys = ["k", "sigma", "eps_r", "tan_d", "mu_r", "E", "nu", "alpha", "rho", "cp"]
    with open(os.path.join(out_dir, "structure_legend.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region_id", "grey_value", "kind", "name", "material_id", "volume_fraction", "void"]
                   + keys + ["voxel_size_um", "grid"])
        for li, t in enumerate(table):
            p = t["props"]
            w.writerow([li, int(grey[li]), t["kind"], t["name"], t.get("material_id") or "",
                        f"{float(vf_labels[li]):.6f}", int(p["E"] <= 0)]
                       + [p.get(k, "") for k in keys] + [h_um, "x".join(str(s) for s in labels.shape)])


def thermal_values(spec, gas_factor=None):
    """k per label, with the rarefied-gas (Knudsen) reduction applied to gas-filled regions."""
    table = spec["labels"]
    k = np.array(S.label_values(table, "k"), float)
    if gas_factor is not None:
        for li, t in enumerate(table):
            if t["props"]["E"] <= 0:
                k[li] *= gas_factor
    return k


def thermal_aniso(spec, labels, ginfo, gen_phases, h, log=None):
    """Anisotropic thermal conductivities for the FV solve (conduction.
    solve_direction's `aniso`), or None when every material is isotropic.
    A particle phase's k_x, k_y, k_z act along each particle's own axes,
    turned with it (the generator's rotation, rows = the particle axes:
    K = R^T diag(k) R); the matrix's along the box axes. A shell keeps its
    scalar k. A phase of stacked platelets (stack_layers n > 1) has, through
    each particle's thickness, its n layers and the n - 1 plate-plate
    contact resistances R_c in series: k_z = t / (t / k_z + (n - 1) R_c),
    t the particle's own thickness."""
    table = spec["labels"]
    label_axes, core_of = {}, {}
    for li, t in enumerate(table):
        ax = M.k_axes(t["props"])
        ph_ = spec["phases"][t["phase"]] if t["kind"] == "core" and t["phase"] is not None else None
        if ax is None and ph_ is not None and int(ph_.get("stack_layers", 1) or 1) > 1:
            ax = [float(t["props"]["k"])] * 3
        if ax is None:
            continue
        if t["kind"] == "matrix":
            label_axes[li] = ax
        elif t["kind"] == "core" and t["phase"] is not None:
            core_of[li] = (t["phase"], ax)
    if not label_axes and not core_of:
        return None
    out = {"label_axes": label_axes, "aid": None, "K": np.zeros((0, 3, 3))}
    parts = (ginfo or {}).get("particles")
    if core_of and parts is not None and len(parts["codes"]):
        own, _ = GEN.rasterize_owner(labels.shape[0], h, ginfo, gen_phases, labels.shape[2])
        ph_of = np.asarray(parts["phase"])
        Rp = np.asarray(parts["R"], float)
        aid = np.full(labels.shape, -1, np.int32)
        Ks, row = [], 0
        for li, (ip, ax) in core_of.items():
            sel = np.flatnonzero(ph_of == ip)
            if not len(sel):
                continue
            d = np.tile(np.asarray(ax, float), (len(sel), 1))
            nl = int(spec["phases"][ip].get("stack_layers", 1) or 1)
            if nl > 1:
                # each particle's thickness along its own z: spheroid / cylinder 2 P1, cuboid / superellipsoid 2 P2
                cds, Pp = np.asarray(parts["codes"])[sel], np.asarray(parts["P"], float)[sel]
                thick = np.where(np.isin(cds, (4, 5)), 2.0 * Pp[:, 2], 2.0 * Pp[:, 1]) * 1e-6
                rc = float(spec["phases"][ip].get("r_contact", 0.0) or 0.0)
                d[:, 2] = thick / (thick / ax[2] + (nl - 1) * rc)
                if log:
                    log(f"   stacked platelets: {spec['phases'][ip]['name']} as stacks of {nl}, through the thickness "
                        f"k {float(np.median(d[:, 2])):.3g} W/m·K ({nl} layers of {ax[2]:g} and {nl - 1} contacts of "
                        f"{rc:g} m²K/W in series), in the plane {ax[0]:g} / {ax[1]:g}")
            Ks.append(np.einsum("nj,nji,njl->nil", d, Rp[sel], Rp[sel]))
            lut = np.full(len(ph_of) + 1, -1, np.int32)
            lut[sel + 1] = row + np.arange(len(sel))
            m = labels == li
            aid[m] = lut[own[m]]
            row += len(sel)
            if log:
                cover = float(np.mean(aid[m] >= 0)) if m.any() else 0.0
                log(f"   anisotropic thermal conductivity: {spec['phases'][ip]['name']} k {ax[0]:g} / {ax[1]:g} / "
                    f"{ax[2]:g} W/m·K along the particle axes x / y / z, {len(sel)} particles turned with them "
                    f"({100 * cover:.1f} % of its voxels)")
        if Ks:
            out["aid"], out["K"] = aid, np.concatenate(Ks)
    if log:
        for li, ax in label_axes.items():
            log(f"   anisotropic thermal conductivity: {table[li]['name']} k {ax[0]:g} / {ax[1]:g} / {ax[2]:g} W/m·K "
                f"along the box axes x / y / z")
        if spec["rve"].get("film") and core_of:
            log("   (a film: the particles' own anisotropy is not applied - their k is used; a matrix anisotropic "
                "along the box axes is)")
    return out


def thermal_interfaces(spec, h_um, area_ratio):
    """Face resistances for the thermal solve, divided by the voxel size.

    R_int (filler-matrix) acts on every face between a filler's outer region
    (its shell if it has one) and the matrix. The voxelised surface is a
    staircase with more face area than the true surface (x1.5 for a sphere),
    so R is multiplied by that ratio to keep the total interface conductance
    A_true / R. R_c (contact) acts on the faces where two particles touch.
    Returns (rpair, rcpair) or (None, None) when no resistance is set.
    """
    table = spec["labels"]
    nlab = len(table)
    h_m = h_um * 1e-6
    rpair = np.zeros((nlab, nlab))
    rcpair = np.zeros((nlab, nlab))
    outer = {}
    for li, t in enumerate(table):
        if t["kind"] == "core" and t["phase"] is not None:
            outer.setdefault(t["phase"], li)
        if t["kind"] == "shell":
            outer[t["phase"]] = li
    regions = {i: [li for li, t in enumerate(table) if t["phase"] == i] for i in outer}
    any_r = any_c = False
    for i, ph in enumerate(spec["phases"]):
        if i not in outer:
            continue
        if ph.get("r_int", 0) > 0 and not ph["void"]:
            r = ph["r_int"] * area_ratio.get(i, 1.5) / h_m
            # against every matrix (the main one and the regions)
            for mi, tm in enumerate(table):
                if tm["kind"] == "matrix":
                    rpair[mi, outer[i]] = rpair[outer[i], mi] = r
            any_r = True
    if wants_contact_faces(spec):
        any_c = True
        for i in outer:
            for j in outer:
                v = contact_resistance(spec, i, j) / h_m
                for a in regions[i]:
                    for b in regions[j]:
                        rcpair[a, b] = v
    return (rpair if any_r else None), (rcpair if any_c else None)


# =========================================================================
# conduction-type solves
# =========================================================================
def emi_directions(opts):
    """The in-plane directions of the EMI admittivity: those among the load
    directions asked for, x alone when only z is (the field in a shielding
    layer lies in its plane)."""
    return tuple(i for i, d in enumerate("xy") if d in opts["directions"]) or (0,)


def cond_backend(spec, backend):
    if spec["rve"].get("film"):
        # PuMA's finite elements are periodic in all three directions; a film
        # needs insulated faces in-plane and plates through the thickness,
        # which the finite-volume solver carries
        return "fv"
    return _cond_backend(spec, backend)


def _cond_backend(spec, backend):
    """How the conduction-type properties are solved, as "primary[+check]".

    primary: "fv" the periodic finite volume (default with the PuMA backend:
    on voxel grids it came 2-3x closer to converged references than PuMA's
    finite elements, guide 7.2), "fe" PuMA's periodic finite elements
    (options.cond_method = "fe"), "puma_fv" PuMA's finite volume with fixed
    faces. Each problem is solved once; options.crosscheck adds a second solve
    with the other discretisation ("+fe" or "+fv") and reports the difference."""
    o = spec["options"]
    if backend == "puma":
        prim = "fe" if o.get("cond_method") == "fe" else "fv"
    elif backend == "puma_fv":
        prim = "puma_fv"
    else:
        prim = "fv"
    if o.get("crosscheck"):
        return prim + ("+fe" if prim == "fv" else "+fv")
    return prim


def _cond_solve(key, labels, vals, spec, h_um, backend, prog, log, event, should_stop,
                viewer, first, rpair=None, cmask=None, rcpair=None, pre=None, aniso=None):
    info = COND[key]
    iface = rpair is not None or rcpair is not None
    backend = cond_backend(spec, backend)
    if aniso is not None and not iface:
        log(f"    {info['title']}: anisotropic materials -> built-in periodic finite volume with their tensors")
        backend = "fv"
    if iface:
        # An interface resistance lives on voxel faces. The finite-volume
        # operator has faces; PuMA's finite elements share nodes across them
        # and cannot carry it, so these solves use the built-in periodic FV.
        log(f"    {info['title']}: interface resistances set -> built-in periodic finite volume with "
            f"face resistances")
        backend = "fv"
    primary, _, check = backend.partition("+")
    dirs = spec["options"]["directions"]
    cap = spec["options"]["contrast_cap"]
    tol = spec["options"]["tol"]
    cache = {}
    T = np.full((3, 3), np.nan)
    energy, stats = {}, []
    policies = {}
    field_stats = {}
    for d in dirs:
        di = "xyz".index(d)
        v_used, policy = CD.contrast_policy(labels, vals, di, cap, cache, film=bool(spec["rve"].get("film")))
        policies[d] = policy
        how = {"fv": "periodic FV", "fe": "PuMA periodic FE", "puma_fv": "PuMA FV"}[primary]
        if check:
            how += " + " + {"fv": "FV", "fe": "PuMA FE"}[check] + " check"
        prog.stage(f"{info['title']} · {d} direction", f"{how} · contrast policy {policy}",
                   units=0.0 if (key, d) in (pre or {}) else 1.0)

        def cb(tag, it, res, tgt, frac, d=d):
            event("solver", name=f"{info['title']} {d}", it=it, res=res, target=tgt,
                  progress=prog.sub(frac))
        done = (pre or {}).get((key, d))
        if primary in ("fe", "puma_fv"):
            if done is not None:
                r = done
            else:
                r = PB.conductivity(labels, v_used, d, h_um * 1e-6, kind=info["kind"], tol=tol,
                                    progress=cb, log=None, should_stop=should_stop, keep_fields=first,
                                    method="fe" if primary == "fe" else "fv")
                if check == "fv":
                    # an independent periodic FV discretisation of the same RVE
                    try:
                        xc = CD.solve_direction(labels, v_used, di, tol=1e-8, should_stop=should_stop)
                        r["crosscheck"], r["crosscheck_method"] = xc["k_eff"], "fv"
                        del xc
                    except InterruptedError:
                        raise
                    except Exception as e:                           # noqa: BLE001
                        log(f"    (the cross-check FV solve failed: {e})")
            T[:, di] = r["column"]
            pot, flux = r.get("T"), r.get("q")
        else:
            if done is not None:
                r = done
            else:
                def cb_d(it, res, d=d):
                    cb(d, it, res, tol, None)
                if first and _LIVE["dir"]:
                    cb_d.snapshot = LV.cond_snapshot(LV.Slot(_LIVE["dir"], f"{key}_{d}", f"{COND[key]['title']} {d}"),
                                                     di, bool(spec["rve"].get("film")), LV.LABEL.get(key, "Potential"))
                r = CD.solve_direction(labels, v_used, di, tol=tol, rpair=rpair, cmask=cmask, rcpair=rcpair,
                                       callback=cb_d,
                                       should_stop=should_stop, flux_h=(h_um * 1e-6) if first else None,
                                       film=bool(spec["rve"].get("film")), aniso=aniso)
                r["method"] = ("fv-film" if spec["rve"].get("film") else "fv-anisotropic" if aniso is not None
                               else ("fv-interface" if iface else "fv-periodic"))
                if check == "fe":
                    # PuMA's periodic finite elements on the same RVE, as the check
                    try:
                        xc = PB.conductivity(labels, v_used, d, h_um * 1e-6, kind=info["kind"], tol=tol,
                                             progress=None, log=None, should_stop=should_stop,
                                             keep_fields=False, method="fe")
                        r["crosscheck"], r["crosscheck_method"] = xc["k_eff"], "fe"
                        del xc
                    except InterruptedError:
                        raise
                    except Exception as e:                           # noqa: BLE001
                        log(f"    (the cross-check PuMA FE solve failed: {e})")
            T[:, di] = r["column"]
            pot = None
            flux = r.pop("q", None)
            if first and r.get("u") is not None:
                u = r.pop("u")
                if r.get("absolute_potential"):
                    pot = u                  # a film between plates: the full field
                else:
                    n = labels.shape[di]
                    shape = [1, 1, 1]
                    shape[di] = n
                    ramp = ((np.arange(n) + 0.5) / n).reshape(shape)
                    pot = ramp + u / n
                del u
        energy[d] = r["energy_frac"]
        stats.append({"direction": d, "policy": policy,
                      **{k: r.get(k) for k in ("iterations", "residual", "target", "converged", "seconds",
                                               "memory_mb", "method", "crosscheck", "crosscheck_method")}})
        log(f"    {info['title']} {d}: {T[di, di]:.6g} {info['unit']}  "
            f"({r.get('seconds', 0):.1f} s, {r.get('iterations')} iterations)")
        if first and viewer is not None and pot is not None:
            st_f = _cond_fields(key, viewer, labels, spec["labels"], vals, pot, flux, d, h_um,
                                with_vectors=(d == dirs[0]))
            if d == dirs[0]:
                field_stats = st_f
        del pot, flux, r
        prog.end()
    return {"tensor": T, "energy": energy, "stats": stats, "policies": policies,
            "percolation": cache.get("wraps"), "field_stats": field_stats}


def _grid(N, nz=None):
    """A grid as text: N³ for a cube, N × N × nz for a film."""
    return f"{N}³" if nz is None else f"{N} × {N} × {nz}"


EXTRAP_MAX = 0.10          # largest change on the finer grid that is extrapolated


def _fine_grid(spec, ginfo, gen_phases, N, h, n_max):
    """The same particles drawn on a 1.5x finer grid: {labels, N, h, nz}, or
    the reason it cannot be drawn."""
    if ginfo.get("shape"):
        return "the compressed structure is not redrawn on another grid (its resins and pores flowed there)"
    Nf = int(round(1.5 * N / 2.0)) * 2
    if n_max and Nf > n_max:
        return f"the finer grid ({Nf}³) would exceed the grid limit {n_max}³"
    hf = h * N / Nf
    film = bool(ginfo.get("film"))
    nzf = max(2, int(round(ginfo["T_um"] / hf))) if film else None
    gen_f = GEN.rasterize(Nf, hf, ginfo, gen_phases, nz=nzf)
    if gen_f is None:
        return "a network phase cannot be redrawn on another grid"
    labels_f, _, _ = build_labels(spec, gen_f, ginfo, hf)
    return {"labels": labels_f, "N": Nf, "h": hf, "nz": nzf}


def _resolution_check(spec, ginfo, gen_phases, N, h, need, gas_factor, area_ratio, results, backend, log,
                      should_stop, n_max, pre=None, fine=None):
    """The conduction-type properties once more, on the same particles drawn on
    a 1.5x finer grid (first direction, same method). Returns
    {key: {direction, N, N_fine, base, fine, change, extrapolated}} or a
    reason string when the check cannot run.

    extrapolated assumes an error proportional to the voxel size (first order,
    as a staircase interface gives): k_inf = k_fine + (k_fine - k_base) / (1.5 - 1).
    On one 50 vol% structure the FV values 0.891 (64^3) and 0.876 (128^3) give
    0.861, the value a third grid (192^3) also points to.

    It is given only while the change stays within EXTRAP_MAX and the value
    within the phases' own range. Particles kept half a voxel apart are
    further apart on the finer grid and lose contacts: 55 vol% of 250 nm
    spheres changed by -38 %, a different structure rather than a
    discretisation error, and the formula gave a negative conductivity."""
    fine = fine if fine is not None else _fine_grid(spec, ginfo, gen_phases, N, h, n_max)
    if isinstance(fine, str):
        return fine
    labels_f, Nf, hf, nzf = fine["labels"], fine["N"], fine["h"], fine["nz"]
    film = bool(ginfo.get("film"))
    nz = ginfo.get("nz") if film else None
    pre = pre or {}
    d = spec["options"]["directions"][0]
    di = "xyz".index(d)
    primary = cond_backend(spec, backend).partition("+")[0]
    out = {}
    for key in COND:
        if key not in need or key not in results:
            continue
        rpair = rcpair = None
        if key == "thermal":
            vals = thermal_values(spec, gas_factor)
            rpair, rcpair = thermal_interfaces(spec, hf, area_ratio)
            if rcpair is not None:
                log(f"    resolution check: {COND[key]['title']} skipped (contact resistances need the contact "
                    f"faces of the finer grid)")
                continue
            if any(M.k_axes(t["props"]) for t in spec["labels"]):
                log(f"    resolution check: {COND[key]['title']} skipped (anisotropic materials need the particles' "
                    f"cells of the finer grid)")
                continue
        else:
            vals = np.array(S.label_values(spec["labels"], COND[key]["prop"]), float)
        v_used, _ = CD.contrast_policy(labels_f, vals, di, spec["options"]["contrast_cap"], {}, film=bool(spec["rve"].get("film")))
        t0 = time.time()
        if ("res:" + key, d) in pre:
            kf = pre[("res:" + key, d)]["k_eff"]
        elif primary == "fv" or rpair is not None:
            kf = CD.solve_direction(labels_f, v_used, di, tol=spec["options"]["tol"], rpair=rpair,
                                    should_stop=should_stop, film=film)["k_eff"]
        else:
            kf = PB.conductivity(labels_f, v_used, d, hf * 1e-6, kind=COND[key]["kind"], tol=spec["options"]["tol"],
                                 should_stop=should_stop, keep_fields=False,
                                 method="fe" if primary == "fe" else "fv")["k_eff"]
        kb = float(results[key]["tensor"][di, di])
        change = kf / kb - 1.0 if kb else 0.0
        extrap = kf + (kf - kb) / 0.5
        fin = np.asarray(v_used, float)[np.isfinite(v_used)]
        ok = abs(change) <= EXTRAP_MAX and (fin.size == 0 or float(fin.min()) <= extrap <= float(fin.max()))
        why_not = None if ok else (
            f"the change exceeds {100 * EXTRAP_MAX:.0f} %: contacts or gaps differ between the two grids, "
            f"which a first-order correction does not describe")
        out[key] = {"direction": d, "N": N, "N_fine": Nf, "grid": _grid(N, nz), "grid_fine": _grid(Nf, nzf),
                    "base": kb, "fine": float(kf), "change": change,
                    "extrapolated": float(extrap) if ok else None, "extrapolation_note": why_not,
                    "seconds": time.time() - t0}
        log(f"    resolution check {COND[key]['title']} {d}: {_grid(N, nz)} {kb:.6g} → {_grid(Nf, nzf)} {kf:.6g} "
            f"({100*change:+.2f} %), " + (f"extrapolated {extrap:.6g}" if ok else f"not extrapolated ({why_not})"))
    return out


def _vector_layer(viewer, key, vec, mask, d, h_um, label, unit, group, field_key, n_lines=40):
    """Arrows and field lines for a solved vector field.

    The magnitude alone does not say where the transport goes; the arrows give
    the direction voxel by voxel and the field lines follow it through the
    structure, which is what makes a conduction or flow result readable at a
    glance.
    """
    if viewer is None or vec is None:
        return None
    from . import streamlines as SL
    D = d.upper()
    viewer.add_vector(key, vec, f"{label} ({D} gradient)", unit, group, field=field_key,
                      note="Arrow length and colour follow the local magnitude")
    try:
        # inside a pore the fast voxels are the middle of the channel, which is
        # where a line survives; across a whole RVE the median is a fine start
        lines, summary = SL.trace(vec, mask, h_um, "xyz".index(d), n_lines=n_lines,
                                  seed_quantile=0.7 if mask is not None else 0.5)
    except Exception:                                                # noqa: BLE001
        return None
    # Lines that all stall at a pore wall picture the discretisation, not the
    # transport: at a few voxels per pore the no-slip velocity near the wall is
    # too small to follow. Such a set is measured and reported, but not drawn.
    summary["published"] = bool(lines and summary.get("n_through", 0) >= max(3, 0.25 * len(lines)))
    if summary["published"]:
        viewer.add_paths(key, lines, f"{label} lines ({D} gradient)", group, kind="streamline",
                         summary=summary, note="Integrated from the solved field; colour is the local magnitude",
                         speed_label=f"{label} magnitude", speed_unit=unit)
    return summary


def _cond_fields(key, viewer, labels, table, vals, pot, flux, d, h_um, with_vectors=True):
    di = "xyz".index(d)
    L_m = labels.shape[di] * h_um * 1e-6
    D = d.upper()
    grp = COND[key]["group"]
    stats = {}
    # scalar fields are written for every solved direction so they can be
    # compared in the viewer; arrows and field lines are three times the size of
    # a scalar field, so they are kept for the first direction only
    vec_viewer = viewer if with_vectors else None
    if key == "thermal":
        viewer.add_field(f"thermal_{d}_T", pot, f"Temperature ({D} gradient, ΔT = 1 K)", "K", grp)
        if flux is not None:
            qm = np.sqrt((flux ** 2).sum(-1))
            viewer.add_field(f"thermal_{d}_q", qm, f"Heat flux magnitude |q| ({D} gradient)", "W/m²", grp)
            stats["flux"] = _label_stats(qm, labels, table)
            stats["field_lines"] = _vector_layer(vec_viewer, f"thermal_{d}_q", flux, None, d, h_um,
                                                 "Heat flux q", "W/m²", grp, f"thermal_{d}_q")
    elif key == "electrical":
        viewer.add_field(f"electrical_{d}_V", pot, f"Electric potential ({D} gradient, 1 V)", "V", grp)
        if flux is not None:
            jm = np.sqrt((flux ** 2).sum(-1))
            viewer.add_field(f"electrical_{d}_J", jm, f"Current density magnitude |J| ({D} gradient)", "A/m²", grp, log=True)
            stats["flux"] = _label_stats(jm, labels, table)
            stats["field_lines"] = _vector_layer(vec_viewer, f"electrical_{d}_J", flux, None, d, h_um,
                                                 "Current density J", "A/m²", grp, f"electrical_{d}_J")
    else:
        name = "Electric" if key == "dielectric" else "Magnetic"
        sym = "|E|/E₀" if key == "dielectric" else "|H|/H₀"
        viewer.add_field(f"{key}_{d}_phi", pot, f"{name} potential ({D} gradient)", "-", grp)
        if flux is not None:
            vv = np.asarray(vals, float)[labels]
            evec = flux / vv[..., None] * L_m
            enh = np.sqrt((evec ** 2).sum(-1))
            # The enhancement worth looking at is on the low-permittivity side of
            # each interface, where the field concentrates: the matrix around a
            # high-permittivity filler (measured on the 100^3 dielectric case,
            # 75.6 % of matrix voxels above 1 against 0.23 % of the particle's),
            # but the air inside a hollow particle (up to 1.57 x in the cores).
            viewer.add_field(f"{key}_{d}_E", enh, f"Local field enhancement {sym} ({D} field)", "-", grp,
                             low_side="eps_r" if key == "dielectric" else "mu_r")
            stats["enhancement"] = _label_stats(enh, labels, table)
            stats["field_lines"] = _vector_layer(vec_viewer, f"{key}_{d}_E", evec, None, d, h_um,
                                                 f"{name} field {sym}", "-", grp, f"{key}_{d}_E")
            del evec
    return stats


# =========================================================================
# thermo-elasticity
# =========================================================================
def _prefetch(box, need, labels, table, spec, h, backend, gas_factor, area_ratio, contact_mask, first,
              log, should_stop, log_dir, event=None, flow=None, prog=None, struct=None, rcheck=None, aniso=None):
    """Run the independent solves of this realisation side by side.

    Everything that does not depend on another solve goes into one pool: the
    conduction-type properties x directions, PuMA's seven elastic load cases,
    the moisture diffusion of each direction, the Stokes and the diffusion
    solve of each direction in the open pores, the EMI admittivity at each
    frequency and direction, and the ray casting. A heavy solve that keeps
    one core busy (a Stokes direction) and light ones (FV conduction) run at
    once on the cores of the job; each starts when its memory fits beside the
    ones running. The stages that follow then find their solves done.
    Returns ({key: result}, {load: (Ceff, s, t, stats)}); empty when parallel
    solves are off, when there is only one problem or when only one worker
    fits in memory - the stages then solve in turn.
    flow: {"labels", "void", "open"} of the open pore space, or None.
    struct: {"ginfo", "void"} for the structure analyses of the first
    realisation, or None. rcheck: {"fine": the finer grid} when the
    resolution check runs, or None.
    """
    from .solvers import complex_cond as CC
    opts = spec["options"]
    if opts.get("parallel_solves", "auto") == "off":
        return {}, {}
    try:
        # the cores the server's queue granted this run (MPSIM_JOB_CORES),
        # else those it was started with
        threads = int(os.environ.get("MPSIM_JOB_CORES") or os.environ.get("NUMBA_NUM_THREADS") or 0)
    except ValueError:
        threads = 0
    # without a job setting (the command line, a script): every core
    threads = threads or max(1, os.cpu_count() or 1)
    dirs = opts["directions"]
    n_vox = int(labels.size)
    mem_est = R.memory_estimate(labels.shape[0], set(need) | {"thermal"}, nz=labels.shape[2],
                                elastic=opts.get("elastic_method", "fans"))
    # memory of one solve of each kind, GB: rve.memory_estimate per voxel,
    # Stokes from a measurement (PuMA's matrix-free solve held 0.64 GB on an
    # 80^3 RVE, ~1.25 kB per voxel; the planner's 2.3 kB is its upper bound)
    mem_of = {"fv": mem_est.get("conduction", n_vox * 230 / 1e9), "fe": mem_est.get("conduction", n_vox * 230 / 1e9),
              "elastic": n_vox * R.elastic_bytes(opts.get("elastic_method", "fans")) / 1e9, "stokes": n_vox * 1300 / 1e9,
              "tort": n_vox * 230 / 1e9, "emi": n_vox * 2 * 230 / 1e9, "radiation": n_vox * 16 / 1e9}
    for k in ("morph", "perc", "poros", "network", "grains"):
        mem_of[k] = n_vox * 100 / 1e9
    # the progress units the stages after the batch would have used for the
    # same work (they find it done and take none)
    units_of = {"fv": 1.0, "fe": 1.0, "elastic": 0.0, "stokes": 2.0, "tort": 0.5, "emi": 0.0, "radiation": 0.5,
                "morph": 1.0, "perc": 1.0, "poros": 1.0, "network": 1.0, "grains": 1.0}
    tasks = []

    def add(fn, kind, t, units=None):
        t["_kind"] = kind
        t["mem_gb"] = mem_of[kind]
        if first and fn is PAR.cond_task and _LIVE["dir"]:
            t["live_dir"] = _LIVE["dir"]
        t["_units"] = units_of[kind] if units is None else units
        tasks.append((fn, t))
    for key in COND:
        if key not in need:
            continue
        rpair = rcpair = None
        if key == "thermal":
            vals = thermal_values(spec, gas_factor)
            rpair, rcpair = thermal_interfaces(spec, h, area_ratio)
        else:
            vals = np.array(S.label_values(table, COND[key]["prop"]), float)
        an = aniso if key == "thermal" else None
        be = "fv" if (rpair is not None or rcpair is not None or an is not None) else cond_backend(spec, backend)
        film_run = bool(spec["rve"].get("film"))
        cache = {}
        for d in dirs:
            v_used, _ = CD.contrast_policy(labels, vals, "xyz".index(d), opts["contrast_cap"], cache, film=bool(spec["rve"].get("film")))
            add(PAR.cond_task, "fe" if "fe" in be.split("+") else "fv",
                {"key": key, "d": d, "kind": COND[key]["kind"], "backend": be,
                 "name": f"{COND[key]['title']} {d}",
                 "vals": np.asarray(v_used, float), "voxel_m": h * 1e-6, "tol": opts["tol"],
                 "fields": bool(first), "rpair": rpair, "rcpair": rcpair,
                 "cmask": "use" if (rcpair is not None and contact_mask is not None) else None,
                 "film": film_run,
                 "aniso": None if an is None else {"label_axes": an["label_axes"], "K": an["K"],
                                                   "aid": "use" if an.get("aid") is not None else None}})
    if "cte" in need and backend in ("puma", "puma_fv") and opts.get("elastic_method", "fans") == "puma":
        keep = ("free" if opts.get("thermal_bc") == "free" else "th") if first else None
        tol = 1e-5 if spec["rve"]["quality"] == "fast" else 1e-6
        for load in ("x", "y", "z", "yz", "xz", "xy", "th"):
            add(PAR.elastic_task, "elastic",
                {"load": load, "E": S.label_values(table, "E"), "name": f"Elastic load {load}",
                 "nu": S.label_values(table, "nu"), "alpha": S.label_values(table, "alpha"),
                 "voxel_m": h * 1e-6, "tol": tol,
                 "fields": keep == "free" or (keep == "th" and load == "th")})
    # moisture: the activity of each direction, solved as conduction
    if "moisture" in need:
        P = _moisture_values(table)[4]
        if P.max() > 0:
            cache = {}
            for d in dirs:
                v_used, _ = CD.contrast_policy(labels, P / P.max(), "xyz".index(d), opts["contrast_cap"], cache, film=bool(spec["rve"].get("film")))
                add(PAR.cond_task, "fv",
                    {"key": "moisture", "d": d, "kind": "thermal", "backend": "fv", "name": f"Moisture {d}",
                     "vals": np.asarray(v_used, float), "voxel_m": h * 1e-6, "tol": opts["tol"],
                     "fields": bool(first) and d == dirs[0], "rpair": None, "rcpair": None, "cmask": None,
                     "film": bool(spec["rve"].get("film"))}, units=0.5)
    # the open pore space: Stokes and diffusion of each direction
    if flow is not None and flow.get("labels") is not None:
        wr = CD.wrapping_axes(flow["open"])
        for d in dirs:
            if not wr["xyz".index(d)]:
                continue
            if "permeability" in need:
                add(PAR.stokes_task, "stokes", {"d": d, "void": flow["void"], "voxel_m": h * 1e-6, "tol": opts["tol"],
                                                "name": f"Stokes flow {d}", "_flow": True})
            if "tortuosity" in need:
                add(PAR.tort_task, "tort", {"d": d, "void": flow["void"], "voxel_m": h * 1e-6, "tol": opts["tol"],
                                            "name": f"Diffusion {d}", "fields": bool(first), "_flow": True})
    # EMI: the complex admittivity at every frequency and in-plane direction
    emi_pol = {}
    if first and "emi" in need and opts["emi"].get("method", "complex") == "complex":
        eo = opts["emi"]
        freqs = np.logspace(math.log10(eo["f_min_hz"]), math.log10(eo["f_max_hz"]), int(eo["n_freq"]))
        ebc = "film" if spec["rve"].get("film") else "periodic"
        cache = {}
        edirs = emi_directions(opts)
        for i, fq in enumerate(freqs):
            for d in edirs:
                kap, pol = CC.cap_contrast(labels, CC.admittivity(table, fq), d, cache=cache)
                emi_pol[(i, d)] = pol
                add(PAR.emi_task, "emi", {"i": i, "d": d, "kap": kap, "bc": ebc, "tol": min(opts["tol"], 1e-7),
                                          "name": f"EMI {fq:.3g} Hz {'xyz'[d]}"},
                    units=1.0 / (len(edirs) * len(freqs)))
    # the structure analyses of the first realisation
    if first and struct is not None:
        void = list(struct.get("void") or [])
        groups = _phase_groups(table, void)
        if "morphology" in need:
            for gkey, gname, ids in groups:
                add(PAR.struct_task, "morph", {"what": "morph", "gkey": gkey, "ids": ids, "h": h, "fields": True,
                                               "name": f"Size distributions {gname}"})
        if "percolation" in need:
            for gkey, gname, ids in groups:
                add(PAR.struct_task, "perc", {"what": "perc", "gkey": gkey, "ids": ids, "h": h, "dirs": dirs,
                                              "full": gkey != "pores", "fields": True,
                                              "name": f"Percolation {gname}"})
        if "porosimetry" in need and void:
            po = opts["porosimetry"]
            add(PAR.struct_task, "poros", {"what": "poros", "void": void, "h": h, "gamma": po["surface_tension_N_m"],
                                           "theta": po["contact_angle_deg"], "steps": po["steps"], "dirs": dirs,
                                           "name": "Porosimetry"})
        if "pore_network" in need and void:
            add(PAR.struct_task, "network", {"what": "network", "void": void, "h": h, "name": "Pore network"})
        if "grains" in need:
            add(PAR.struct_task, "grains", {"what": "grains", "h": h, "spec": spec, "ginfo": struct["ginfo"],
                                            "table": table, "name": "Grain statistics"})
    # the resolution check: the first direction of each conduction property
    # on the finer grid
    if rcheck is not None and isinstance(rcheck.get("fine"), dict):
        fine = rcheck["fine"]
        d0 = dirs[0]
        primary = cond_backend(spec, backend).partition("+")[0]
        for key in COND:
            if key not in need:
                continue
            rpair = rcpair = None
            if key == "thermal":
                vals = thermal_values(spec, gas_factor)
                rpair, rcpair = thermal_interfaces(spec, fine["h"], area_ratio)
                if rcpair is not None or any(M.k_axes(t["props"]) for t in spec["labels"]):
                    continue
            else:
                vals = np.array(S.label_values(spec["labels"], COND[key]["prop"]), float)
            v_used, _ = CD.contrast_policy(fine["labels"], vals, "xyz".index(d0), opts["contrast_cap"], {}, film=bool(spec["rve"].get("film")))
            be = "fv" if (primary == "fv" or rpair is not None) else primary
            add(PAR.cond_task, "fe" if be == "fe" else "fv",
                {"key": "res:" + key, "d": d0, "kind": COND[key]["kind"], "backend": be,
                 "name": f"Resolution check {COND[key]['title']} {d0}",
                 "vals": np.asarray(v_used, float), "voxel_m": fine["h"] * 1e-6, "tol": opts["tol"],
                 "fields": False, "rpair": rpair, "rcpair": None, "cmask": None,
                 "film": bool(spec["rve"].get("film")), "_fine": True},
                units=1.0 / max(1, sum(1 for k in COND if k in need)))
            tasks[-1][1]["mem_gb"] = mem_of["fv"] * fine["labels"].size / max(n_vox, 1)
    if first and "radiation" in need and flow is not None and flow.get("void"):
        ro = opts["radiation"]
        add(PAR.radiation_task, "radiation", {"void": flow["void"], "voxel_m": h * 1e-6,
                                              "sources": ro["sources"], "rays": ro["rays"], "name": "Ray casting"})
    if len(tasks) < 2:
        return {}, {}
    # longest first: a Stokes direction or an elastic load case costs many
    # conduction solves, and a pool that takes them last finishes on one
    # busy worker
    tasks.sort(key=lambda t: -PAR.TASK_COST[t[1]["_kind"]])
    costs = [PAR.TASK_COST[t["_kind"]] for _, t in tasks]
    pars = [PAR.TASK_PAR[t["_kind"]] for _, t in tasks]
    # as many workers as the lightest solve allows; the admission below keeps
    # the heavy ones within the free memory
    w_mem = PAR.plan_workers(threads, len(tasks), min(t["mem_gb"] for _, t in tasks))
    workers, th_each = PAR.plan_split(threads, costs, w_mem, pars)
    if workers < 2:
        heavy = max(t["mem_gb"] for _, t in tasks)
        why = (f"only one worker fits in memory (~{heavy:.2f} GB for the largest solve)" if w_mem < 2 else
               f"one solve at a time on all {threads} cores finishes first")
        log(f"    parallel solves: {why}; solving in turn")
        return {}, {}
    # the solves running at once must fit in the memory free now; the idle
    # workers' own memory (~0.1-0.2 GB each, measured) is already counted in
    # what the system reports once the pool is up
    try:
        import psutil
        budget = 0.9 * psutil.virtual_memory().available / 2 ** 30
    except Exception:                                                  # noqa: BLE001
        budget = None
    pool = box.get("pool")
    if pool is None or pool.workers != workers or pool.threads_each != th_each:
        if pool is not None:
            pool.close()
        pool = box["pool"] = PAR.SolvePool(workers, th_each, log_dir, cores=threads)
    lab_path = pool.share(labels)
    flow_path = pool.share(flow["labels"], "flow") if any(t.get("_flow") for _, t in tasks) else None
    fine_path = pool.share(rcheck["fine"]["labels"], "fine") if any(t.get("_fine") for _, t in tasks) else None
    cm_path = pool.share(contact_mask.astype(np.uint8), "cmask") if contact_mask is not None else None
    aid_path = (pool.share(aniso["aid"], "aid") if aniso is not None and aniso.get("aid") is not None
                and any((t.get("aniso") or {}).get("aid") == "use" for _, t in tasks) else None)
    for _, t in tasks:
        t["labels"] = flow_path if t.get("_flow") else fine_path if t.get("_fine") else lab_path
        t["tmp"] = pool.tmp
        if "cmask" in t:
            t["cmask"] = cm_path if t.get("cmask") == "use" else None
        if (t.get("aniso") or {}).get("aid") == "use":
            t["aniso"] = dict(t["aniso"], aid=aid_path)
    kinds = {}
    for _, t in tasks:
        kinds[t["_kind"]] = kinds.get(t["_kind"], 0) + 1
    names = {"fv": "FV", "fe": "FE", "elastic": "elastic loads", "stokes": "Stokes", "tort": "diffusion",
             "emi": "EMI", "radiation": "ray casting", "morph": "size distributions", "perc": "percolation",
             "poros": "porosimetry", "network": "pore network", "grains": "grain statistics"}
    log(f"    parallel solves: {len(tasks)} independent problems ("
        + ", ".join(f"{n} {names[k]}" for k, n in kinds.items()) + f") on {workers} workers × {th_each} thread(s)"
        + (f" (memory allows {w_mem})" if w_mem < min(threads, len(tasks)) else ""))
    t0 = time.time()
    batch_units = sum(t["_units"] for _, t in tasks)
    units_done = [0.0]
    stage_name = "Independent solves side by side"
    stage_detail = f"{len(tasks)} problems on {workers} workers × {th_each} thread(s)"
    if prog is not None:
        prog.stage(stage_name, stage_detail, units=batch_units)

    def done(res, i, n):
        units_done[0] += res.pop("_units", 0.0)
        if prog is not None and event is not None:
            event("stage", name=stage_name, detail=f"{stage_detail} · {i}/{n} done",
                  progress=prog.sub(units_done[0] / batch_units if batch_units else 1.0))
        if "failed" in res:
            log(f"      {i}/{n} {res.get('name') or 'a solve'} failed in the batch ({res['failed']}); "
                f"its stage solves it again")
            return
        kind = res.get("kind")
        what = (f"{res['key']} {res['d']}" if "key" in res else f"elastic load {res['load']}" if "load" in res
                else f"Stokes {res['d']}" if kind == "stokes" else f"diffusion {res['d']}" if kind == "tort"
                else f"EMI {res['i']}/{'xyz'[res['d']]}" if kind == "emi"
                else f"{res['what']} {res.get('gkey') or ''}".strip() if kind == "struct" else str(kind))
        sec = (res.get("r", {}).get("worker_seconds") or res.get("r", {}).get("seconds")
               or (res.get("stats") or {}).get("seconds") or 0.0)
        log(f"      {i}/{n} {what} ({sec:.1f} s)")

    def progressed(name, it, res, tgt):
        if event is not None:
            event("solver", name=name, it=it, res=res, target=tgt)
    out = pool.run(tasks, should_stop=should_stop, on_done=done, on_progress=progressed, budget_gb=budget)
    log(f"    parallel solves finished in {time.time() - t0:.1f} s")
    if prog is not None:
        prog.end()

    def take(path):
        if not path:
            return None
        a = np.load(path)
        try:
            os.remove(path)
        except OSError:
            pass
        return a
    pre, pre_el = {}, {}
    for res in out:
        if res is None or "failed" in res:
            continue
        kind = res.get("kind")
        if "key" in res:
            r = res["r"]
            for k, pk in (("T", "T_path"), ("q", "q_path"), ("u", "u_path")):
                a = take(res.get(pk))
                if a is not None:
                    r[k] = a
            pre[(res["key"], res["d"])] = r
        elif "load" in res:
            pre_el[res["load"]] = (res["Ceff"], take(res.get("s_path")), take(res.get("t_path")), res["stats"])
        elif kind == "stokes":
            pre[("stokes", res["d"])] = {"K": res["K"], "stats": res["stats"],
                                         "u": [take(p) for p in res["u_paths"]]}
        elif kind == "tort":
            r = dict(res["r"])
            r["C"] = take(res.get("C_path"))
            pre[("tort", res["d"])] = r
        elif kind == "emi":
            pre[("emi", res["i"], res["d"])] = {**res["r"], "policy": emi_pol.get((res["i"], res["d"]))}
        elif kind == "radiation":
            pre[("radiation",)] = res["r"]
        elif kind == "struct":
            for line in res.get("lines") or []:
                log(line)
            r = {"res": res["res"]}
            for k in ("lt_path", "field_path"):
                if res.get(k):
                    r[k[:-5]] = take(res[k])
            if res.get("field_paths"):
                r["fields"] = {k: take(p) for k, p in res["field_paths"].items()}
            if "view" in res:
                r["view"] = res["view"]
            pre[(res["what"], res.get("gkey"))] = r
    # the workers keep their imports and allocator pools; the stages after the
    # batch (FANS elasticity on all the job's cores) need that memory back. A
    # later realisation starts a new pool (a few seconds).
    try:
        pool.close()
    finally:
        box.pop("pool", None)
    return pre, pre_el


def _elastic_solve(labels, table, spec, h_um, backend, prog, log, event, should_stop, viewer, first,
                   pre_loads=None):
    E = S.label_values(table, "E")
    nu = S.label_values(table, "nu")
    al = S.label_values(table, "alpha")
    # The tolerance is on the relative residual ||r|| / ||b||, which does not
    # depend on the grid or the units: 1e-5 gave C within 2e-6 of the
    # converged value on 48^3 and 96^3; PuMA's own test stopped after one or
    # two iterations there (C11 10x too high) once preconditioned.
    tol = 1e-5 if spec["rve"]["quality"] == "fast" else 1e-6
    method = spec["options"].get("elastic_method", "fans")
    prog.stage("Elasticity and thermal expansion",
               {"fans": "voxel FE · FFT-preconditioned CG", "puma": f"{backend} FE · MINRES",
                "fft": "FFT (Moulinec-Suquet)"}.get(method, method) + " · 7 load cases (6 strains + ΔT)", units=7.0)
    names = {"x": "εxx", "y": "εyy", "z": "εzz", "yz": "γyz", "xz": "γxz", "xy": "γxy", "th": "ΔT"}
    order = ["x", "y", "z", "yz", "xz", "xy", "th"]

    def cb(tag, it, res, tgt, frac):
        k = order.index(tag) if tag in order else 0
        event("solver", name=f"Elastic load {names.get(tag, tag)} ({k+1}/7)", it=it, res=res, target=tgt,
              progress=prog.sub((k + (frac or 0.0)) / 7.0))
    if method == "fans":
        free = spec["options"].get("thermal_bc") == "free"
        fnames = {"xx": "εxx", "yy": "εyy", "zz": "εzz", "yz": "γyz", "xz": "γxz", "xy": "γxy", "th": "ΔT",
                  "free": "free expansion"}
        forder = ["xx", "yy", "zz", "yz", "xz", "xy", "th", "free"]

        def cbf(tag, it, res):
            k = forder.index(tag)
            event("solver", name=f"Elastic load {fnames[tag]} ({min(k + 1, 7)}/7)", it=it, res=res, target=tol,
                  progress=prog.sub(min(k + 0.5, 6.99) / 7.0))
        try:
            import numba
            workers = int(numba.get_num_threads())
        except Exception:                                            # noqa: BLE001
            workers = -1
        if first and _LIVE["dir"]:
            cbf.snapshot = LV.Multi(_LIVE["dir"], "elastic", "Elasticity, load", "Displacement fluctuation |u| (voxels)")
        rf = FA.homogenize(labels, E, nu, al, tol=tol, workers=workers, progress=cbf, should_stop=should_stop,
                           keep_thermal_stress=bool(first), stress_bc="free" if free else "th")
        stats = [dict(st, load=fnames[k]) for k, st in rf["stats"].items()]
        its = [st["iterations"] for st in rf["stats"].values()]
        log(f"    FANS: {len(its)} solves, {min(its)}-{max(its)} iterations each "
            f"({sum(st['seconds'] for st in rf['stats'].values()):.1f} s)")
        fields = None
        if rf.get("thermal_stress") is not None:
            ts = rf.pop("thermal_stress")
            fields = (np.stack([ts[0], ts[1], ts[2]], axis=-1), np.stack([ts[3], ts[4], ts[5]], axis=-1))
            del ts
        r = {"C": rf["C"], "alpha": rf["alpha"], "sigma_th": rf["sigma_th"], "asymmetry": rf["asymmetry"],
             "stats": stats, "fields": fields, "void_phases": rf["void_phases"], "E_used": rf["E_used"],
             "nu_used": rf["nu_used"], "void_ratio": 1e-4, "method": "fans"}
    elif backend in ("puma", "puma_fv") and method == "puma":
        free = spec["options"].get("thermal_bc") == "free"
        r = PB.thermo_elastic(labels, E, nu, al, h_um * 1e-6, tol=tol, progress=cb, should_stop=should_stop,
                              keep_field=("free" if free else "th") if first else None, pre_loads=pre_loads)
    else:
        def cb2(name, k, it, res):
            event("solver", name=f"Elastic load (FFT) {name}", it=it, res=res, target=tol, progress=prog.sub(k / 7.0))
        r = EL.homogenize(labels, E, nu, al, tol=tol, progress=cb2, should_stop=should_stop)
        r["fields"] = None
        r["method"] = "fft"
    prog.end()
    if first and viewer is not None and r.get("fields"):
        s, t = r["fields"]
        dT = spec["options"]["delta_T"]
        bc = "free expansion" if spec["options"].get("thermal_bc") == "free" else "constrained"
        scale = dT * 1e-3          # GPa * ppm = kPa per K -> MPa for dT
        sx, sy, sz = s[..., 0] * scale, s[..., 1] * scale, s[..., 2] * scale
        tyz, txz, txy = t[..., 0] * scale, t[..., 1] * scale, t[..., 2] * scale
        vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (tyz ** 2 + txz ** 2 + txy ** 2))
        hyd = (sx + sy + sz) / 3.0
        grp = "Thermo-mechanical"
        viewer.add_field("cte_vm", vm, f"Thermal stress, von Mises (ΔT = {dT:g} K, {bc})", "MPa", grp)
        viewer.add_field("cte_hyd", hyd, f"Hydrostatic stress σm (ΔT = {dT:g} K, {bc})", "MPa", grp, diverging=True)
        viewer.add_field("cte_szz", sz, f"Normal stress σzz (ΔT = {dT:g} K, {bc})", "MPa", grp, diverging=True)
        viewer.add_field("cte_sxz", txz, f"Shear stress τxz (ΔT = {dT:g} K, {bc})", "MPa", grp, diverging=True)
        r["stress_stats"] = {"von_mises": _label_stats(vm, labels, table),
                             "hydrostatic": _label_stats(hyd, labels, table), "bc": bc}
        del s, t, sx, sy, sz, tyz, txz, txy, vm, hyd
    r.pop("fields", None)
    return r


def _mech_test(labels, table, spec, st, h_um, prog, log, event, should_stop, viewer):
    """The stress fields of the mechanical test (mechtest.py): one FANS solve
    at the test's macroscopic strain; their statistics per region, and the
    fields for the viewer (von Mises, axial stress and strain, axial
    displacement of the RVE)."""
    mt = spec["options"]["mechtest"]
    E = S.label_values(table, "E")
    nu = S.label_values(table, "nu")
    tol = 1e-5 if spec["rve"]["quality"] == "fast" else 1e-6
    word = {"tension": "Tension", "compression": "Compression"}[mt["mode"]]
    prog.stage("Mechanical test", f"{word} along {mt['axis']} · one FANS solve at the test strain", units=1.0)
    try:
        import numba
        workers = int(numba.get_num_threads())
    except Exception:                                                    # noqa: BLE001
        workers = -1

    def cb(it, res):
        event("solver", name=f"Mechanical test ({mt['mode']} along {mt['axis']})", it=it, res=res, target=tol,
              progress=prog.sub(0.5))
    sol = FA.FANS(labels, E, nu, None, 1e-4, workers)
    Ev = np.asarray(st["E_voigt"], float)
    sbar, info, s = sol.solve(Ev, 0.0, tol, 5000, cb, should_stop, keep_stress=True, keep_disp=True)
    u = sol.last_disp
    E_used, nu_used = np.asarray(sol.E_used, float), np.asarray(sol.nu_used, float)
    sol.last_disp = None
    del sol
    s *= 1e3                                                             # GPa -> MPa
    a = MT.AXES[mt["axis"]]
    b, c = [k for k in range(3) if k != a]
    vm = MT.von_mises(s).astype(np.float32)
    saa = s[a].astype(np.float32)
    # the local axial strain from each region's stress (isotropic regions)
    eaa = np.empty(labels.shape, np.float32)
    for li in np.unique(labels):
        m = labels == li
        eaa[m] = (s[a][m] - nu_used[li] * (s[b][m] + s[c][m])) / (E_used[li] * 1e3)
    del s
    # the displacement along the load: the mean strain over the RVE plus the
    # solved fluctuation (nodes at the lower voxel corners, voxel units)
    idx = np.arange(labels.shape[a], dtype=np.float32).reshape([-1 if k == a else 1 for k in range(3)])
    ua = ((Ev[a] * idx + u[a]) * h_um).astype(np.float32)
    del u
    mask_m = labels == 0
    q999 = float(np.percentile(vm[mask_m], 99.9)) if mask_m.any() else None
    stats = {"von_mises": _label_stats(vm, labels, table), "axial_stress": _label_stats(saa, labels, table),
             "axial_strain": _label_stats(eaa, labels, table)}
    if viewer is not None:
        grp = "Mechanical test"
        what = f"{mt['mode']} along {mt['axis']}, {abs(st['stress_MPa']):.3g} MPa"
        ax = mt["axis"]
        viewer.add_field("mt_vm", vm, f"von Mises stress ({what})", "MPa", grp)
        viewer.add_field("mt_saa", saa, f"Axial stress σ{ax}{ax} ({what})", "MPa", grp, diverging=True)
        # one sign through the RVE (the load's): a full colour scale, not a
        # symmetric one; the axial stress can change sign around particles
        viewer.add_field("mt_eaa", 100.0 * eaa, f"Axial strain ε{ax}{ax} ({what})", "%", grp)
        viewer.add_field("mt_ua", ua, f"Displacement along {ax} in the RVE ({what})", "µm", grp)
    del vm, saa, eaa, ua
    sb = (np.asarray(sbar) * 1e3).tolist()
    ref = np.asarray(st["sigma_voigt_MPa"], float)
    dev = float(np.max(np.abs(np.asarray(sb) - ref)) / max(np.max(np.abs(ref)), 1e-30))
    prog.end()
    log(f"    mechanical test: {mt['mode']} along {mt['axis']}, apparent modulus {st['modulus_GPa']:.4g} GPa, "
        f"{st['stress_MPa']:.4g} MPa at strain {100 * st['strain']:.4g} %; matrix von Mises 99.9 % "
        f"{q999 if q999 is None else round(q999, 4)} MPa ({info['iterations']} iterations)")
    return {"state": st, "stats": stats, "solve": info, "sbar_MPa": sb, "mean_stress_dev": dev,
            "matrix_vm_q999": q999}


def _thermomech_solve(labels, table, spec, h_um, gas_factor, area_ratio, contact_mask, prog, log, event,
                      should_stop, viewer, aniso=None):
    """Thermal stress and warpage of the film (solvers/thermomech.py): the
    temperature field of the heat flowing between its faces with the thermal
    analysis's conductivities and interface resistances, the thermal stress
    of every voxel from the stress-free temperature, and the laminate release
    of the film (free, on a substrate, or held flat)."""
    from .solvers import thermomech as TM
    tm = spec["options"]["thermomech"]
    tol = 1e-5 if spec["rve"]["quality"] == "fast" else 1e-6
    k_vals = thermal_values(spec, gas_factor)
    rpair, rcpair = thermal_interfaces(spec, h_um, area_ratio)
    sub_name = str((tm.get("substrate") or {}).get("name") or "a substrate").split(" (")[0]
    sup = {"free": "free film", "held": "held flat", "substrate": f"on {sub_name}"}[tm["support"]]
    prog.stage("Thermal stress and warpage",
               f"heat {tm['T_bottom']:g} °C (bottom) → {tm['T_top']:g} °C (top) · stress from {tm['T_ref']:g} °C "
               f"(voxel FE) · laminate release, {sup}", units=3.0)
    if aniso is not None:
        log("    (thermal stress: the anisotropic conductivities are taken at their mean k)")
    frac = {"f": 0.0}

    def pr(tag, f):
        frac["f"] = float(f)

    def cb(name, it, res, tgt):
        event("solver", name=name, it=it, res=res, target=tgt, progress=prog.sub(min(frac["f"] + 0.05, 0.99)))
    out = TM.run(labels, table, h_um, tm, k_vals, rpair=rpair, cmask=contact_mask if rcpair is not None else None,
                 rcpair=rcpair, log=log, progress=pr, should_stop=should_stop, tol=tol, keep_fields=viewer is not None,
                 compare_uniform=tm["compare_uniform"], callback=cb)
    f = out.pop("fields", None)
    T = out.pop("T", None)
    if viewer is not None and f is not None:
        grp = "Thermal stress and warpage"
        what = f"from {tm['T_ref']:g} to {tm['T_bottom']:g} / {tm['T_top']:g} °C, {sup}"
        viewer.add_field("tm_vm", f["vm"], f"von Mises stress ({what})", "MPa", grp)
        viewer.add_field("tm_T", T, f"Temperature ({tm['T_bottom']:g} °C bottom, {tm['T_top']:g} °C top)", "°C", grp)
        viewer.add_field("tm_p1", f["p1"], f"Largest principal stress ({what})", "MPa", grp, diverging=True)
        viewer.add_field("tm_sxx", f["sxx"], f"In-plane stress σxx ({what})", "MPa", grp, diverging=True)
        viewer.add_field("tm_szz", f["szz"], f"Through-thickness stress σzz, peeling ({what})", "MPa", grp,
                         diverging=True)
        if f["above"].any():
            viewer.add_field("tm_above", f["above"].astype(np.float32), "Above its Tg (1) or below (0)", "-", grp,
                             categorical=True)
    del f, T
    out["options"] = tm
    prog.end()
    w = out["warpage_um"]
    log(f"    thermal stress and warpage: T {out['T_min']:.4g} … {out['T_max']:.4g} °C, κ {out['kappa_1_m']:.4g} 1/m, "
        f"warpage {w:+.4g} µm over {out['span_mm']:g} mm ({'edges up' if w < 0 else 'edges down' if w > 0 else 'flat'})"
        + (f", uniform {out['uniform']['warpage_um']:+.4g} µm" if out.get("uniform") else "")
        + f" ({out['seconds']:.1f} s)")
    return out


def _joule_solve(labels, table, spec, h_um, gas_factor, area_ratio, contact_mask, prog, log, event,
                 should_stop, viewer):
    """Joule heating of the film (solvers/joule.py): the electric potential
    with every voxel's conductivity at its temperature, the heat of the
    current, the temperature it raises, repeated to a steady state at every
    step of the current or voltage."""
    from .solvers import joule as JH
    jo = spec["options"]["joule"]
    sig = np.array(S.label_values(table, "sigma"), float)
    sig_used, policy = CD.contrast_policy(labels, sig, 2, spec["options"]["contrast_cap"], {}, film=True)
    k_vals = thermal_values(spec, gas_factor)
    rpair, rcpair = thermal_interfaces(spec, h_um, area_ratio)
    n = len(jo["values"])
    word = "current density" if jo["control"] == "current" else "voltage"
    prog.stage("Joule heating", f"{n} steps of the {word} · potential and temperature solved in turn to 1 mK "
                                f"(σ with temperature) · contrast policy {policy}", units=1.0 + 0.25 * n)
    frac = {"f": 0.0}

    def pr(si, nn):
        frac["f"] = si / max(nn, 1)

    def cb(name, it, res, tgt):
        event("solver", name=name, it=it, res=res, target=tgt, progress=prog.sub(min(frac["f"] + 0.5 / max(n, 1), 0.99)))
    if policy == "dispersed":
        # the contrast limit has lowered the conductor to the matrix's level:
        # no current path between the electrodes, nothing to heat
        why = ("the most conductive phase does not connect the two faces of the film, so no current path joins the "
               "electrodes (a matrix skin at the faces - Domain tab - or too few touching particles)")
        log(f"    Joule heating not computed: {why}")
        prog.end()
        return {"error": why, "steps": [], "policy": policy, "options": jo, "control": jo["control"]}
    out = JH.run(labels, table, h_um, jo, sig_used, k_vals, rpair=rpair,
                 cmask=contact_mask if rcpair is not None else None, rcpair=rcpair, log=log, progress=pr,
                 should_stop=should_stop, keep_fields=viewer is not None, callback=cb)
    f = out.pop("fields", None)
    if viewer is not None and f is not None:
        grp = "Joule heating"
        st = out["steps"][f["step"]]
        what = (f"{st['J_A_m2']:.3g} A/m², {st['V']:.3g} V")
        viewer.add_field("jh_T", f["T"], f"Temperature ({what})", "°C", grp)
        viewer.add_field("jh_q", f["q"], f"Joule heat per volume ({what})", "W/m³", grp, log=True)
        viewer.add_field("jh_J", f["J"], f"Current density |J| ({what})", "A/m²", grp, log=True)
        viewer.add_field("jh_phi", f["phi"], f"Electric potential ({what})", "V", grp)
    del f
    out.update({"policy": policy, "sigma_used": sig_used.tolist(), "options": jo})
    prog.end()
    last = out["steps"][-1] if out["steps"] else None
    if last:
        log(f"    Joule heating: {len(out['steps'])} steps, last {last['J_A_m2']:.4g} A/m² at {last['V']:.4g} V, "
            f"T max {last['T_max']:.5g} °C ({last['status']}; {out['seconds']:.1f} s)")
    return out


NU_RUBBERY = 0.45   # Poisson's ratio of a cured resin above Tg (0.45-0.49); B-bar elements keep it lock-free


def _above_tg(labels, table, prog, log, event, should_stop):
    """Elasticity and CTE with every region that has a glass transition in its
    rubbery state (E above Tg, alpha2, nu 0.45): the alpha2 and the hot
    modulus of an EMC, which set warpage and stress at reflow."""
    from .solvers import fans as FA
    E = list(S.label_values(table, "E"))
    nu = list(S.label_values(table, "nu"))
    al = list(S.label_values(table, "alpha"))
    changed = []
    for i, t in enumerate(table):
        p = t["props"]
        if p.get("E_r") and p.get("alpha2") and E[i] > 0:
            E[i], nu[i], al[i] = float(p["E_r"]), NU_RUBBERY, float(p["alpha2"])
            changed.append(t["name"])
    if not changed:
        return None
    prog.stage("Elasticity above Tg", "rubbery " + ", ".join(changed) + " · 7 load cases (B-bar)", units=3.0)
    try:
        import numba
        workers = int(numba.get_num_threads())
    except Exception:                                                    # noqa: BLE001
        workers = -1
    rf = FA.homogenize(labels, E, nu, al, tol=1e-6, workers=workers, should_stop=should_stop, bbar=True,
                       progress=lambda tag, it, res: event("solver", name=f"Above Tg {tag}", it=it, res=res,
                                                           target=1e-6, progress=None))
    c = EL.engineering_constants(rf["C"])
    prog.end()
    log(f"    above Tg: α = {np.round(rf['alpha'][:3], 2)} ppm/K, E = {np.round(c['E'], 4)} GPa")
    return {"alpha": np.asarray(rf["alpha"]).tolist(), "C": np.asarray(rf["C"]).tolist(),
            "constants": {k: v for k, v in c.items() if k != "S"}, "rubbery": changed,
            "stats": {k: v for k, v in rf["stats"].items()}}


def _emi_se(sig, eps, mu, opts):
    t = opts["thickness_mm"] * 1e-3
    sp = EMI.spectrum(sig, eps, mu, t, opts["f_min_hz"], opts["f_max_hz"])
    f = np.asarray(sp["freq_hz"])
    se_tm = np.asarray(sp["se_db"])
    check = {"available": False}
    try:
        import skrf
        from skrf.media import DefinedGammaZ0
        fr = skrf.Frequency.from_f(f, unit="hz")
        gamma = EMI._gamma(f, eps, mu, sig)
        eta = EMI._eta(f, eps, mu, sig)
        media = DefinedGammaZ0(frequency=fr, gamma=gamma, z0_port=EMI.Z0, z0=eta)
        ntw = media.line(t, unit="m")
        s21 = np.abs(ntw.s[:, 1, 0])
        with np.errstate(divide="ignore"):
            se_rf = -20.0 * np.log10(s21)
        ok = np.isfinite(se_rf) & (se_rf < 250)
        diff = float(np.max(np.abs(se_rf[ok] - se_tm[ok]))) if ok.any() else None
        sp["se_db_skrf"] = [float(x) if np.isfinite(x) else None for x in se_rf]
        check = {"available": True, "version": skrf.__version__, "max_diff_db": diff,
                 "points_compared": int(ok.sum())}
    except Exception as e:                                      # noqa: BLE001
        check = {"available": False, "error": f"{type(e).__name__}: {e}"}
    return sp, check


# =========================================================================
# references and checks
# =========================================================================
def _isotropic_structure(spec):
    """Statistically isotropic: shapes turned at random, and an even dispersion
    (a region, a gradient or a few clusters make it directional on purpose)."""
    if ((spec.get("rve") or {}).get("dispersion") or {}).get("mode", "uniform") != "uniform":
        return False
    # sections and matrix regions layer the RVE on purpose
    if spec.get("matrix_regions") or any(ph.get("section") for ph in spec["phases"]):
        return False
    return all(ph["shape"] in ("sphere", G.NETWORK) or ph["orientation"]["mode"] == "iso"
               for ph in spec["phases"])


def _cond_refs(key, spec, vf_labels, vals):
    table = spec["labels"]
    phi = np.asarray(vf_labels, float)
    lo, hi = AN.wiener(phi, vals)
    refs = {"Wiener bounds": [lo, hi]}
    if key == "thermal":
        # anisotropic materials: bounds per direction - a material along the
        # box axes with its value along that axis, a turned particle with the
        # smallest and largest of its axes (rigorous whatever the turn)
        axes = [M.k_axes(t["props"]) for t in table]
        if any(a is not None for a in axes):
            per = []
            for d in range(3):
                lo_v, hi_v = [], []
                for t, a, v in zip(table, axes, vals):
                    if a is None:
                        lo_v.append(v); hi_v.append(v)
                    elif t["kind"] == "matrix":
                        lo_v.append(float(a[d])); hi_v.append(float(a[d]))
                    else:
                        lo_v.append(float(min(a))); hi_v.append(float(max(a)))
                per.append([AN.wiener(phi, lo_v)[0], AN.wiener(phi, hi_v)[1]])
            refs["_wiener_dir"] = per
    if _isotropic_structure(spec):
        refs["Hashin–Shtrikman"] = list(AN.hashin_shtrikman(phi, vals))
    # the mean-field models take one matrix with the fillers spread through it
    simple = (all(t["kind"] in ("matrix", "core") for t in table)
              and sum(1 for t in table if t["kind"] == "matrix") == 1 and _isotropic_structure(spec))
    if simple:
        km = vals[0]
        phs = []
        for li, t in enumerate(table[1:], start=1):
            ph = spec["phases"][t["phase"]]
            ar = 1.0 if ph["shape"] in ("sphere", G.NETWORK) else G.aspect_ratio(ph["shape"], ph["size_um"])
            phs.append({"phi": phi[li], "k": vals[li], "aspect": ar,
                        "orientation": ph["orientation"]["mode"] if ph["shape"] not in ("sphere", G.NETWORK) else "iso"})
        refs["Maxwell-Garnett / Mori–Tanaka"] = AN.mori_tanaka(km, phs)
        if all(spec["phases"][t["phase"]]["shape"] == "sphere" for t in table[1:]):
            refs["Bruggeman (EMA)"] = AN.bruggeman(phi, vals)
    return refs


def _interface_resistance(spec):
    """Whether the thermal solve carries interface or contact resistances."""
    return (any(float(ph.get("r_int") or 0) > 0 or float(ph.get("r_contact") or 0) > 0 for ph in spec["phases"])
            or any(float(v or 0) > 0 for v in (spec.get("contact_rc") or {}).values()))


def _cond_checks(key, tensor, refs, spec, stats):
    title = COND[key]["title"]
    out = []
    lo, hi = refs["Wiener bounds"]
    # interface and contact resistances act as further, thin phases of low
    # conductivity: the lower bounds of the phases alone no longer hold (Al/SiC
    # with R_int 1e-8 m2K/W came out below the series bound, as it should)
    upper_only = key == "thermal" and _interface_resistance(spec)
    for i, d in enumerate("xyz"):
        v = tensor[i, i]
        if not np.isfinite(v):
            continue
        if refs.get("_wiener_dir"):
            lo, hi = refs["_wiener_dir"][i]
        ok = (v <= hi * (1 + 1e-3)) if upper_only else (lo * (1 - 1e-3) <= v <= hi * (1 + 1e-3))
        out.append({"id": f"{key}_wiener_{d}", "group": title,
                    "label": f"{d}{d} component within the Wiener bounds" + (" (upper)" if upper_only else ""),
                    "value": v, "target": [lo, hi], "unit": COND[key]["unit"],
                    "status": "pass" if ok else "fail",
                    "note": "Rigorous bounds that every microstructure satisfies"
                            + ("; with interface resistances only the upper one (they add thin phases of low "
                               "conductivity, which can take the value below the lower bound of the phases alone)"
                               if upper_only else "")})
    if "Hashin–Shtrikman" in refs:
        hlo, hhi = refs["Hashin–Shtrikman"]
        diag = [tensor[i, i] for i in range(3) if np.isfinite(tensor[i, i])]
        if diag:
            m = float(np.mean(diag))
            ok = (m <= hhi * (1 + 0.01)) if upper_only else (hlo * (1 - 0.01) <= m <= hhi * (1 + 0.01))
            out.append({"id": f"{key}_hs", "group": title,
                        "label": "Isotropic mean within the Hashin–Shtrikman bounds" + (" (upper)" if upper_only else ""),
                        "value": m, "target": [hlo, hhi], "unit": COND[key]["unit"],
                        "status": "pass" if ok else "warn",
                        "note": "Bounds for statistically isotropic structures; a finite RVE may deviate slightly"})
        if len(diag) == 3:
            spread = (max(diag) - min(diag)) / max(np.mean(diag), 1e-300)
            out.append({"id": f"{key}_iso", "group": title, "label": "Directional scatter of an isotropic structure",
                        "value": spread, "target": 0.05, "unit": "rel",
                        "status": "pass" if spread <= 0.05 else ("warn" if spread <= 0.12 else "fail"),
                        "note": "A large scatter indicates an RVE that is too small or contains too few particles"})
    xs = [(s["direction"], s["crosscheck"], s.get("crosscheck_method")) for s in stats if s.get("crosscheck")]
    diffs = [abs(tensor["xyz".index(d), "xyz".index(d)] / v - 1.0) for d, v, _ in xs if v]
    if diffs:
        worst = max(diffs)
        other = "PuMA periodic finite elements" if xs[0][2] == "fe" else "an independent periodic finite volume"
        out.append({"id": f"{key}_xfv", "group": title, "label": f"Result against {other} on the same RVE",
                    "value": worst, "target": 0.05, "unit": "rel",
                    "status": "pass" if worst <= 0.05 else "warn",
                    "note": "Both discretisations converge to the same value as the voxels get finer, from "
                            "opposite sides (FE high, FV closer); the difference bounds the discretisation error. "
                            "A large one calls for a finer voxel size, or a resolution check."})
    conv = [s.get("converged") for s in stats if s.get("converged") is not None]
    if conv:
        out.append({"id": f"{key}_conv", "group": title, "label": "Linear solver convergence",
                    "value": sum(conv), "target": len(conv), "unit": "directions",
                    "status": "pass" if all(conv) else "fail", "note": ""})
    return out


# =========================================================================
# main entry
# =========================================================================
_GATE = {"fn": None, "done": False}


def set_gate(fn):
    """The server's gate (web/jobs.py): called once, after the structure of the
    first realisation is built and before its first solve; it returns when the
    queue has the memory for the solves, with the cores granted. None: no
    gate (the command line, the self-test)."""
    _GATE.update(fn=fn, done=False)


def _gate(log):
    if _GATE["fn"] is None or _GATE["done"]:
        return
    _GATE["done"] = True
    g = _GATE["fn"]()
    if g and g.get("waited_s", 0.0) > 1.0:
        log(f"   waited {g['waited_s']:.0f} s for the memory to solve; solving on {g['cores']} cores")
    elif g:
        log(f"   solving on {g['cores']} cores")


def run(form, out_dir, log=print, event=None, should_stop=None, limits=None):
    t_start = time.time()
    event = event or (lambda kind, **p: None)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    # the live frames of the solves going on (mpsim/live.py), for the browser
    _LIVE["dir"] = os.path.join(out_dir, "live")
    LV.clear(_LIVE["dir"])
    PB.set_log_dir(os.path.join(out_dir, "logs"))

    limits = limits or (form or {}).get("_limits") or {}
    reuse = dict((form or {}).get("_reuse") or {})
    spec = S.normalize(form)
    # A continued run: the analyses already in this folder, on the same inputs,
    # are kept, and only the ones not yet there are computed.
    prev = None
    if reuse.get("append"):
        try:
            with open(os.path.join(out_dir, "result.json"), encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = None
        if prev is not None and (prev.get("preview") or prev.get("error")
                                 or run_signature(prev.get("spec") or {}) != run_signature(spec)):
            prev = None
    _dump(os.path.join(out_dir, "spec.normalized.json"), spec)
    wanted = set(spec["analyses"])
    kept = set(prev.get("analyses") or []) if prev else set()
    need = S.required_analyses(wanted - kept)
    if "emi" in (wanted - kept) and any(t["props"]["mu_r"] != 1.0 for t in spec["labels"]):
        need.add("magnetic")
    if prev is not None and not need and not spec.get("preview"):
        log(f"== {spec['name']}")
        log("   Every selected analysis is already in this run with the same inputs; nothing is computed again.")
        return _clean({k: prev.get(k) for k in ("name", "elapsed_s", "headline", "check_counts", "backend")})
    opts = spec["options"]
    backend = opts["backend"]
    # PuMA is imported (1.4 s) only by a run that uses it
    uses_puma = (bool(need & {"permeability", "tortuosity", "acoustics", "filtration", "radiation", "morphology",
                              "grains", "porosimetry", "pore_network", "percolation"})
                 or (bool(need & set(COND)) and (backend == "puma_fv" or opts.get("cond_method") == "fe"
                                                  or opts.get("crosscheck")))
                 or ("cte" in need and opts.get("elastic_method") == "puma"))
    if backend in ("puma", "puma_fv") and uses_puma and not PB.available():
        log("!! NASA PuMA could not be imported; the built-in verification solvers are used instead: "
            + str(PB.unavailable_reason()))
        backend = "builtin"
    plan = R.plan(_plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"], particle_gap_vox=spec["options"].get("particle_gap_vox"), elastic_method=spec["options"]["elastic_method"]), need, limits)
    log(f"== {spec['name']}")
    resolve_overlap_share(spec, plan, log)
    if prev is not None:
        log(f"   Continuing this run: {', '.join(sorted(kept))} kept, {', '.join(sorted(need)) or 'nothing'} added.")
    direct = prev is None and not spec.get("preview") and dem_direct(spec, need)
    if direct:
        log("   Viscosity alone, of particles above 35 vol%: the particle dynamics gives the answer and packs its own "
            "particles, so no voxel RVE is generated (the option 'Also solve the resin flow on the RVE' builds one "
            "for its 3D resin flow)")
    elif plan.get("film"):
        log(f"   Film {plan['L_um']:.4g} × {plan['L_um']:.4g} µm (periodic in x and y) × {plan['T_um']:.4g} µm thick, "
            f"voxel {plan['h_um']:.4g} µm, grid {_grid(plan['N'], plan['Nz'])} "
            f"(level {plan['quality_label']}, {plan['seeds']} realisation(s))")
        # derived per-area quantities refer to the film itself
        opts["thickness_mm"] = plan["T_um"] * 1e-3
    else:
        log(f"   RVE {plan['L_um']:.4g} µm, voxel {plan['h_um']:.4g} µm, grid {plan['N']}³ "
            f"(level {plan['quality_label']}, {plan['seeds']} realisation(s))")
    for n in plan["notes"]:
        log("   · " + n)
    for c in plan["checks"]:
        if c["status"] != "pass":
            log(f"   [{c['status']}] {c['label']}: {c['value']:.4g} (criterion {c['target']})")
    event("partial", key="plan", value=_clean({k: plan[k] for k in ("N", "h_um", "L_um", "checks", "notes", "seeds",
                                                                     "film", "Nz", "T_um")}))

    seeds = plan["seeds"]
    settle = plan.get("settle")
    settled = (None, None)
    if settle:
        log(f"   Realisations until the means settle: at least {settle['n_min']}, at most {settle['n_max']}, until "
            f"every 95 % confidence interval is within {settle['target_pct']:g} % of its mean")
    dirs = opts["directions"]
    per_seed_units = 1.0 + sum(len(dirs) for k in COND if k in need) \
        + (7.0 if "cte" in need else 0.0) \
        + (2.0 * len(dirs) if "permeability" in need else 0.0) \
        + (0.5 * len(dirs) if "tortuosity" in need else 0.0) \
        + (0.5 * len(dirs) + 1.0 if "moisture" in need else 0.0) \
        + (3.0 / seeds if "thermomech" in need else 0.0) \
        + ((1.0 + 0.25 * len(opts["joule"]["values"])) / seeds if "joule" in need else 0.0) \
        + (3.0 if ("cte" in need and opts.get("above_tg")) else 0.0)         + (4.0 if "viscosity" in need else 0.0)         + (4.0 if spec["rve"].get("arrangement") == "sheared" else 0.0)
    vdem = opts["viscosity"]["dem"]
    once_units = (1.0 if "mechtest" in need else 0.0) + \
        (2.0 + (3.0 + vdem["rates"] if vdem["on"] else 0.0) if "viscosity" in need else 0.0) + sum(u for k, u in (("morphology", 1.0), ("porosimetry", 1.0), ("pore_network", 1.0),
                                    ("grains", 1.0), ("radiation", 0.5), ("acoustics", 0.3),
                                    ("percolation", 1.0), ("filtration", 0.7)) if k in need)
    prog = Progress(event, seeds * per_seed_units + once_units + 3.0)

    N, h = plan["N"], plan["h_um"]
    film = bool(plan.get("film"))
    Nz = int(plan["Nz"]) if film else None
    skin = film_skin(spec, plan)
    plan["skin"] = skin if film else None
    h_m = h * 1e-6
    gen_phases = _generator_phases(spec)
    table = spec["labels"]
    void_labels = [i for i, t in enumerate(table) if t["props"]["E"] <= 0]
    gas = opts["gas"]
    realisations = []
    samples = {}
    viewer = None
    results_first = {}
    visc_plan = {}                       # the viscosity's plan, set on the first realisation
    for j_ in list(_LIVE_DEM):           # left over from an earlier run in this process
        j_.cancel()
    checks = list(plan["checks"])
    labels = None

    pool_box = {}
    # ------------------------------------------------------ live results
    # After every analysis of the first realisation the view is written, a
    # provisional result is saved for the Result Viewer, and the figures of the
    # analysis go to a background process; the browser shows each as it lands.
    live_box = {"fw": None, "figs": {}}

    def figure_done(name, ok, err, sec, title, group):
        if ok:
            live_box["figs"][name] = title
            event("figure", name=name, title=title, group=group, seconds=round(sec, 1))
        else:
            log(f"   (figure {name} failed: {err})")

    def live(tag):
        if not first or viewer is None:
            return
        try:
            meta_l = viewer.finish()
        except Exception as e:                                           # noqa: BLE001
            log(f"   (view not written after {tag}: {e})")
            return
        if live_box["fw"] is None and opts.get("live_figures", True):
            try:
                live_box["fw"] = FWK.FigureWorker(on_done=figure_done)
            except Exception as e:                                       # noqa: BLE001
                log(f"   (figures will be drawn at the end: {e})")
                live_box["fw"] = False
        if live_box["fw"]:
            for item in figure_plan(meta_l):
                live_box["fw"].submit(item["name"], item["kind"], os.path.join(out_dir, "view"), item["req"],
                                      os.path.join(out_dir, "figures", item["name"]), item["title"], item["group"])
        try:
            chk = list(checks)
            p_, comp_, _, _ = aggregate(dict(res_r), realisations + [rinfo], chk)
            partial = {"schema": "mpsim.result/2", "version": __version__, "name": spec["name"], "partial": True,
                       "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t_start,
                       "backend": backend, "analyses": sorted(p_.keys()), "spec": spec, "plan": plan,
                       "versions": live_box.setdefault("versions", backend_versions()),
                       "realisations": realisations + [rinfo], "composition": comp_, "properties": p_,
                       "checks": chk, "headline": headline_numbers(p_, comp_), "figures": dict(live_box["figs"]),
                       "view": True, "preview": bool(spec.get("preview")),
                       "check_counts": {s_: sum(1 for c in chk if c["status"] == s_) for s_ in ("pass", "warn", "fail")}}
            _dump(os.path.join(out_dir, "result.partial.json"), _clean(partial))
            event("partial_result", tag=tag, analyses=sorted(p_.keys()))
        except Exception as e:                                           # noqa: BLE001
            log(f"   (provisional result after {tag} not written: {type(e).__name__}: {e})")

    # ------------------------------------------------- aggregation (callable)
    def aggregate(results_first, realisations, checks):
        """Effective properties, composition and checks from the results so
        far. Called once at the end, and after every analysis of the first
        realisation for the provisional result the Result Viewer shows while
        the run goes on (a copy of the checks is passed then)."""
        props = {}
        comp_vf = realisations[0]["vf_labels"]
        rho = sum(v * t["props"]["rho"] for v, t in zip(comp_vf, table))
        mass = [v * t["props"]["rho"] for v, t in zip(comp_vf, table)]
        cp = (sum(m * t["props"]["cp"] for m, t in zip(mass, table)) / rho) if rho > 0 else 0.0
        thick_m = opts["thickness_mm"] * 1e-3
        composition = {"labels": [{"name": t["name"], "kind": t["kind"], "vf": v,
                                   "wt": (m / rho if rho > 0 else 0.0)} for t, v, m in zip(table, comp_vf, mass)],
                       "density": rho, "cp": cp,
                       "porosity": float(sum(comp_vf[i] for i in void_labels)),
                       "target": spec.get("composition")}
        if "porosity" in results_first:
            props["porosity"] = {"title": "Porosity", **results_first["porosity"]}
        # each realisation's own values, beside their mean (two or more)
        seeds_used = [int(z.get("seed", 0)) for z in realisations]

        def by_real(title, rows, labels, unit):
            rows = [[float(v) for v in np.atleast_1d(np.asarray(r_, float))] for r_ in (rows or [])]
            if len(rows) < 2:
                return None
            return {"title": title, "seeds": seeds_used[:len(rows)], "labels": labels, "rows": rows, "unit": unit}

        for key in COND:
            if key not in need or key not in results_first or not samples.get(key):
                continue
            mean, std, ci = _mean_ci(samples[key])
            f0 = results_first[key]
            Tm = f0["tensor"].copy()
            for i in range(3):
                if np.isfinite(Tm[i, i]):
                    Tm[i, i] = mean[i]
            diag = [Tm[i, i] for i in range(3)]
            fin = [x for x in diag if np.isfinite(x)]
            iso = float(np.mean(fin)) if fin else None
            entry = {"title": COND[key]["title"], "unit": COND[key]["unit"],
                     "tensor": Tm, "diag": diag, "iso": iso,
                     "std": std, "ci95": ci, "seeds": samples[key], "refs": f0["refs"],
                     "energy_frac": f0["energy"], "stats": f0["stats"], "policies": f0["policies"],
                     "percolation": f0["percolation"], "values_used": f0["values_used"],
                     "field_stats": f0.get("field_stats") or {}}
            if f0.get("resolution"):
                entry["resolution"] = f0["resolution"]
            fv_by = {s["direction"]: s.get("crosscheck") for s in f0["stats"]}
            fv_diag = [fv_by.get(d) for d in "xyz"]
            fv_fin = [v for v in fv_diag if v]
            if fv_fin and iso:
                entry["xc_method"] = next((s.get("crosscheck_method") for s in f0["stats"] if s.get("crosscheck")), None)
                entry["fv_diag"] = fv_diag
                entry["fv_iso"] = float(np.mean(fv_fin))
                entry["disc_band"] = [min(iso, entry["fv_iso"]), max(iso, entry["fv_iso"])]
            derived = {}
            if all(np.isfinite(diag)) and diag[2] > 0:
                derived["anisotropy_inplane_over_z"] = 0.5 * (diag[0] + diag[1]) / diag[2]
            if key == "thermal" and wants_contact_faces(spec):
                # R_c of every pair of phases that touch, and how many voxel
                # faces carry it (the same filler and two different ones)
                pf = (realisations[0].get("contact_pairs") or {}) if realisations else {}
                rows_c = []
                nph = len(spec["phases"])
                for i_ in range(nph):
                    for j_ in range(i_, nph):
                        if spec["phases"][i_]["shape"] == G.NETWORK or spec["phases"][j_]["shape"] == G.NETWORK:
                            continue
                        rows_c.append({"pair": [i_, j_], "names": [spec["phases"][i_]["name"], spec["phases"][j_]["name"]],
                                       "r_c": contact_resistance(spec, i_, j_),
                                       "set": (i_ == j_) or f"{i_}-{j_}" in (spec.get("contact_rc") or {}),
                                       "faces": int(pf.get(f"{i_}-{j_}", 0))})
                entry["contacts"] = rows_c
            if key == "thermal" and iso:
                derived["resistivity_mK_W"] = 1.0 / iso
                derived["area_resistance_mm2K_W"] = thick_m / iso * 1e6
                if all(np.isfinite(diag)) and diag[2] > 0:
                    derived["area_resistance_z_mm2K_W"] = thick_m / diag[2] * 1e6
                if rho > 0 and cp > 0:
                    derived["diffusivity_mm2_s"] = iso / (rho * 1e3 * cp) * 1e6
                    entry["diffusivity_mm2s"] = derived["diffusivity_mm2_s"]
                derived["volumetric_heat_capacity_MJ_m3K"] = rho * 1e3 * cp * 1e-6
                if "knudsen_thermal" in results_first:
                    entry["knudsen"] = results_first["knudsen_thermal"]
            if key == "electrical" and iso:
                derived["resistivity_ohm_m"] = 1.0 / iso
                derived["sheet_resistance_ohm_sq"] = 1.0 / (iso * thick_m)
            if key == "dielectric" and iso:
                derived["capacitance_pF_mm2"] = EPS0 * iso / thick_m * 1e12 * 1e-6
                if all(np.isfinite(diag)):
                    derived["capacitance_z_pF_mm2"] = EPS0 * diag[2] / thick_m * 1e12 * 1e-6
            derived["thickness_mm"] = opts["thickness_mm"]
            entry["derived"] = derived
            if key == "dielectric":
                tand = np.array(S.label_values(table, "tan_d"))
                entry["tan_d"] = {d: float(np.dot(tand, w)) for d, w in f0["energy"].items()}
                entry["tan_d_iso"] = float(np.mean(list(entry["tan_d"].values())))
            entry["by_realisation"] = [b_ for b_ in [by_real(COND[key]["title"], samples[key], ["x", "y", "z"],
                                                               COND[key]["unit"])] if b_]
            props[key] = entry

        if "cte" in need and "cte" in results_first:
            el = results_first["cte"]
            a_mean, a_std, a_ci = _mean_ci(samples["cte_alpha"])
            e_mean, e_std, e_ci = _mean_ci(samples["cte_E"])
            consts = el["constants"]
            E_l = S.label_values(table, "E")
            nu_l = S.label_values(table, "nu")
            al_l = S.label_values(table, "alpha")
            refs = AN.cte_models(comp_vf, [max(e, 0.0) for e in E_l], nu_l, al_l, matrix=0)
            Kp, Gp = AN.bulk_shear(np.maximum(E_l, 1e-12), nu_l)
            hs = AN.hs_elastic(comp_vf, np.where(np.array(E_l) > 0, Kp, 0.0), np.where(np.array(E_l) > 0, Gp, 0.0))
            entry = {"title": "Elasticity and thermal expansion", "unit_alpha": "ppm/K", "unit_E": "GPa",
                     "alpha": el["alpha"], "alpha_diag_mean": a_mean, "alpha_ci95": a_ci,
                     "alpha_vol": float(np.mean(el["alpha"][:3])),
                     "C": el["C"], "S": consts["S"] * 1e3, "constants": {k: v for k, v in consts.items() if k != "S"},
                     "polar_E": EL.polar_moduli(consts["S"]),
                     "E_ci95": e_ci, "refs": refs, "hs": hs, "stats": el["stats"],
                     "asymmetry": el["asymmetry"], "void_ratio": el.get("void_ratio"),
                     "seeds_alpha": samples["cte_alpha"], "delta_T": opts["delta_T"],
                     "by_realisation": [b_ for b_ in (by_real("Thermal expansion", samples["cte_alpha"],
                                                              ["αx", "αy", "αz"], "ppm/K"),
                                                      by_real("Young's modulus", samples["cte_E"], ["Ex", "Ey", "Ez"],
                                                              "GPa")) if b_],
                     "stress_stats": el.get("stress_stats"), "above_tg": el.get("above_tg")}
            solid = [i for i in range(len(table)) if E_l[i] > 0 and comp_vf[i] > 0]
            if len(solid) == 2 and not any(comp_vf[i] > 0 for i in void_labels):
                i1, i2 = solid
                K1, K2 = Kp[i1], Kp[i2]
                lev = EL.levin_alpha(np.linalg.inv(el["C"]), al_l[i1], al_l[i2], K1, K2)
                err = float(np.max(np.abs(lev[:3] - el["alpha"][:3])) / max(np.max(np.abs(el["alpha"][:3])), 1e-12))
                entry["levin"] = lev
                checks.append({"id": "cte_levin", "group": "Thermo-mechanics", "label": "Levin theorem (exact two-phase relation)",
                               "value": err, "target": 0.01, "unit": "rel",
                               "status": "pass" if err <= 0.01 else ("warn" if err <= 0.05 else "fail"),
                               "note": "Mutual consistency of the thermal load case and the six mechanical load cases"})
            checks.append({"id": "cte_sym", "group": "Thermo-mechanics", "label": "Symmetry of the stiffness matrix",
                           "value": el["asymmetry"], "target": 1e-3, "unit": "rel",
                           "status": "pass" if el["asymmetry"] <= 1e-3 else "warn", "note": ""})
            if _isotropic_structure(spec) and hs["K"][1] > 0:
                Klo, Khi = hs["K"]
                Kh = consts["K_hill"]
                ok = Klo * 0.98 <= Kh <= Khi * 1.02
                checks.append({"id": "cte_hsK", "group": "Thermo-mechanics",
                               "label": "Hill bulk modulus within the Hashin–Shtrikman bounds",
                               "value": Kh, "target": [Klo, Khi], "unit": "GPa",
                               "status": "pass" if ok else "warn",
                               "note": "Bounds for statistically isotropic structures"})
                # The shear block needs its own check: the bulk modulus uses only
                # the normal entries, so shear columns wrong by a factor of two once
                # passed this test unnoticed.
                Glo, Ghi = hs["G"]
                Gh = consts["G_hill"]
                if Ghi > 0:
                    ok = Glo * 0.98 <= Gh <= Ghi * 1.02
                    checks.append({"id": "cte_hsG", "group": "Thermo-mechanics",
                                   "label": "Hill shear modulus within the Hashin–Shtrikman bounds",
                                   "value": Gh, "target": [Glo, Ghi], "unit": "GPa",
                                   "status": "pass" if ok else "warn",
                                   "note": "Bounds for statistically isotropic structures"})
            if "Schapery bounds" in refs:
                lo_, hi_ = refs["Schapery bounds"]
                av = entry["alpha_vol"]
                ok = lo_ - 0.02 * abs(lo_) <= av <= hi_ + 0.02 * abs(hi_)
                checks.append({"id": "cte_schapery", "group": "Thermo-mechanics", "label": "Volumetric mean α within the Schapery bounds",
                               "value": av, "target": [lo_, hi_], "unit": "ppm/K",
                               "status": "pass" if ok else ("warn" if not _isotropic_structure(spec) else "fail"),
                               "note": "Rigorous bounds for isotropic two-phase composites"})
            conv = [s.get("converged") for s in el["stats"] if s.get("converged") is not None]
            if conv:
                checks.append({"id": "cte_conv", "group": "Thermo-mechanics", "label": "Linear solver convergence (7 load cases)",
                               "value": sum(conv), "target": len(conv), "unit": "load cases",
                               "status": "pass" if all(conv) else "fail", "note": ""})
            props["cte"] = entry

        if "mechtest" in need and "mechtest" in results_first:
            m0 = results_first["mechtest"]
            st = m0["state"]
            mt = opts["mechtest"]
            m_mean, m_std, m_ci = _mean_ci([[v] for v in samples["mt_modulus"]])
            entry = {"title": "Mechanical test", **m0, "options": mt,
                     "modulus_mean_GPa": float(np.atleast_1d(m_mean)[0]),
                     "modulus_ci95": (float(np.atleast_1d(m_ci)[0]) if m_ci is not None else None),
                     "seeds_modulus": samples["mt_modulus"],
                     "onset": MT.onset(st, m0.get("matrix_vm_q999"), mt.get("strength_MPa")),
                     "curve": {"displacement_um": [0.0, st["displacement_um"]], "force_N": [0.0, st["force_N"]],
                               "strain": [0.0, st["strain"]], "stress_MPa": [0.0, st["stress_MPa"]]}}
            entry["by_realisation"] = [b_ for b_ in [by_real("Apparent modulus", [[v] for v in samples["mt_modulus"]],
                                                             [f"E{mt['axis']}"], "GPa")] if b_]
            checks.append({"id": "mt_conv", "group": "Mechanical test", "label": "Linear solver convergence (test load)",
                           "value": m0["solve"]["residual"], "target": 1e-5, "unit": "",
                           "status": "pass" if m0["solve"]["converged"] else "fail", "note": ""})
            checks.append({"id": "mt_mean", "group": "Mechanical test",
                           "label": "Mean stress of the fields against the macroscopic stress",
                           "value": m0["mean_stress_dev"], "target": 1e-3, "unit": "rel",
                           "status": "pass" if m0["mean_stress_dev"] <= 1e-3 else "warn",
                           "note": "The stress field averaged over the RVE reproduces the stress the effective "
                                   "stiffness gives for the test's strain"})
            on = entry["onset"]
            if on and on["fraction_of_load"] < 1.0:
                checks.append({"id": "mt_onset", "group": "Mechanical test",
                               "label": "Matrix strength reached below the test load",
                               "value": on["fraction_of_load"], "target": 1.0, "unit": "",
                               "status": "warn",
                               "note": f"0.1 % of the matrix reaches {on['strength_MPa']:.4g} MPa (von Mises) at "
                                       f"{100 * on['fraction_of_load']:.0f} % of the load: above it the matrix would "
                                       "yield or crack locally, which linear elasticity does not follow"})
            props["mechtest"] = entry

        if "emi" in wanted and "electrical" in props and "dielectric" in props:
            def inplane(key):
                e = props[key]
                vals_ = [e["diag"][i] for i in (0, 1) if e["diag"][i] is not None and np.isfinite(e["diag"][i])]
                return float(np.mean(vals_)) if vals_ else float(e["iso"])
            sig = inplane("electrical")
            eps = inplane("dielectric")
            mu = inplane("magnetic") if "magnetic" in props else 1.0
            sp, chk = _emi_se(sig, eps, mu, opts["emi"])
            cx = results_first.get("emi_complex")
            complex_info = None
            if cx:
                # kappa_eff(f) solved on the RVE, interpolated log-log onto the
                # spectrum grid: sigma' = Re kappa, eps' = Im kappa / (w eps0)
                f_sp = np.asarray(sp["freq_hz"])
                fk = np.asarray(cx["freq_hz"])
                kre = np.maximum(np.asarray(cx["kappa_re"]), 1e-300)
                kim = np.maximum(np.asarray(cx["kappa_im"]), 1e-300)
                lf, lk = np.log10(f_sp), np.log10(fk)
                sig_f = 10.0 ** np.interp(lf, lk, np.log10(kre))
                eps_f = 10.0 ** np.interp(lf, lk, np.log10(kim)) / (2.0 * math.pi * f_sp * EMI.EPS0)
                static_se = sp["se_db"]
                sp, chk = _emi_se(sig_f, eps_f, mu, opts["emi"])
                sp["se_db_static"] = static_se
                resolved = all(p_ != "network" for r_ in cx["records"] for p_ in r_["policy"])
                complex_info = {"freq_hz": cx["freq_hz"], "sigma_eff": list(kre),
                                "eps_eff": [float(k / (2.0 * math.pi * f_ * EMI.EPS0)) for k, f_ in zip(kim, fk)],
                                "eps_resolved": bool(resolved), "directions": cx["directions"], "bc": cx["bc"],
                                "policy": [r_["policy"] for r_ in cx["records"]],
                                "iterations": [r_["iterations"] for r_ in cx["records"]]}
            fillers = []
            for t in table:
                if t["kind"] == "core" and t["props"]["sigma"] > 1.0:
                    ph = spec["phases"][t["phase"]]
                    st = G.phase_size_stats(ph["shape"], ph["size_um"], ph["dist"]) if ph["shape"] != G.NETWORK else {"min_dim_p10": ph["size_um"]["d"]}
                    fillers.append((t["name"], t["props"]["sigma"], t["props"]["mu_r"], st["min_dim_p10"] * 1e-6))
            valid = EMI.validity(eps, mu, sig, N * h * 1e-6, fillers)
            f = np.asarray(sp["freq_hz"])
            marks = {}
            for fm in (1e6, 1e7, 1e8, 1e9, 1e10):
                if f[0] <= fm <= f[-1]:
                    marks[f"{fm:.0e}"] = float(np.interp(math.log10(fm), np.log10(f), sp["se_db"]))
            props["emi"] = {"title": "EMI shielding effectiveness", "sigma_inplane": sig, "eps_inplane": eps, "mu_inplane": mu,
                            "thickness_mm": opts["emi"]["thickness_mm"], "spectrum": sp,
                            "at": marks, "validity": valid, "skrf_check": chk,
                            "method": "complex" if complex_info else "static", "complex": complex_info}
            fwr = results_first.get("emi_fullwave")
            if fwr:
                if not fwr.get("error"):
                    # where the comparison cannot agree, say so instead of failing it
                    f_skin = min([x["f_skin_hz"] for x in valid["filler"]] or [math.inf])
                    notes_fw = []
                    if f_skin < max(fwr["freq_hz"]):
                        notes_fw.append(f"the skin depth of a conducting filler is smaller than the particle above "
                                        f"{f_skin:.3g} Hz; the homogenisation leaves out eddy currents there, the "
                                        f"full-wave run does not")
                    for ts_ in fwr.get("thin_skin") or []:
                        notes_fw.append(f"at 10 GHz the skin depth of {ts_['name']} ({ts_['skin_um']:.2g} µm) is below "
                                        f"the voxel ({ts_['voxel_um']:.3g} µm): the FDTD does not resolve its eddy "
                                        f"currents either, so neither result holds in this band")
                    if max(fwr["se_model_db"]) > 80.0:
                        notes_fw.append("the shielding exceeds about 80 dB, beyond what the FDTD run resolves "
                                        "(its signal ends at 1e-8 of the peak energy)")
                    fwr["notes"] = notes_fw
                    ok = fwr["max_diff_db"] <= 1.0
                    checks.append({"id": "emi_fullwave", "group": "EMI",
                                   "label": "Full-wave (openEMS) against the homogenised layer, 10-100 GHz",
                                   "value": fwr["max_diff_db"], "target": 1.0, "unit": "dB",
                                   "status": "pass" if ok else ("warn" if notes_fw or fwr["max_diff_db"] <= 3.0 else "fail"),
                                   "note": f"{fwr['thickness_um']:.3g} µm slab of the RVE, {fwr['cells']:,} cells, "
                                           f"{fwr['seconds']:.0f} s" + ("; " + "; ".join(notes_fw) if notes_fw else "")})
                props["emi"]["fullwave"] = fwr
            if chk.get("available") and chk.get("max_diff_db") is not None:
                checks.append({"id": "emi_skrf", "group": "EMI", "label": "scikit-rf transmission line against the transfer-matrix solution",
                               "value": chk["max_diff_db"], "target": 0.05, "unit": "dB",
                               "status": "pass" if chk["max_diff_db"] <= 0.05 else "warn",
                               "note": "Largest difference between two independent implementations"})

        if "permeability" in need and "permeability" in results_first:
            mean, std, ci = _mean_ci(samples["permeability"])
            p0 = results_first["permeability"]
            ptype = results_first.get("porosity", {})
            entry = {"title": "Permeability", "unit": "m²", "diag": mean, "ci95": ci,
                     "by_realisation": [b_ for b_ in [by_real("Permeability", samples["permeability"], ["Kx", "Ky", "Kz"],
                                                              "m²")] if b_],
                     "tensor": p0["tensor"], "percolation": p0["percolation"], "stats": p0["stats"],
                     "open_porosity": ptype.get("open"), "closed_porosity": ptype.get("closed"),
                     "darcy": [m / 9.869233e-13 for m in mean]}
            pos = [m for m in mean if m > 0]
            if pos:
                K_mean = float(np.mean(pos))
                entry["flow_resistivity_Pa_s_m2"] = gas["viscosity_Pa_s"] / K_mean
                sv = p0.get("specific_surface_open_1pm")
                phi_o = ptype.get("open")
                if sv and phi_o:
                    # Kozeny-Carman: K = phi^3 / (c S_v^2), S_v per unit bulk volume
                    entry["specific_surface_open_1pm"] = sv
                    entry["kozeny_constant"] = phi_o ** 3 / (K_mean * sv ** 2)
                ht = [s.get("hydraulic_tortuosity") for s in p0["stats"].values() if s.get("hydraulic_tortuosity")]
                if ht:
                    entry["hydraulic_tortuosity"] = float(np.mean(ht))
                # Two standard corrections on top of the computed K. Both are
                # correlations, not separate solves, and are reported as such:
                # Forchheimer says when the flow stops being linear in the velocity,
                # Klinkenberg how much a gas slips at the pore wall and so appears
                # to permeate faster than a liquid.
                rho_g = gas.get("density_kg_m3") or 1.204
                if phi_o and K_mean > 0:
                    beta_f = 1.75 / (phi_o ** 1.5 * math.sqrt(150.0 * K_mean))
                    u_nd = 0.1 * gas["viscosity_Pa_s"] / (rho_g * beta_f * K_mean)
                    entry["forchheimer_beta_1_m"] = beta_f
                    entry["non_darcy_velocity_m_s"] = u_nd
                    entry["non_darcy_reynolds"] = rho_g * u_nd * math.sqrt(K_mean) / gas["viscosity_Pa_s"]
                if sv and phi_o:
                    r_h = 2.0 * phi_o / sv                      # hydraulic radius [m]
                    lam_g = float(1.380649e-23 * opts["temperature_K"]
                                  / (math.sqrt(2.0) * math.pi * (gas["molecule_diameter_nm"] * 1e-9) ** 2
                                     * gas["pressure_Pa"]))
                    b_k = 4.0 * lam_g * gas["pressure_Pa"] / r_h
                    entry["klinkenberg_b_Pa"] = b_k
                    entry["apparent_gas_permeability_m2"] = K_mean * (1.0 + b_k / gas["pressure_Pa"])
                    entry["hydraulic_radius_um"] = r_h * 1e6
            props["permeability"] = entry

        if "tortuosity" in need and "tortuosity" in results_first:
            t0_ = results_first["tortuosity"]
            ptype = results_first.get("porosity", {})
            taus = [v["tortuosity"] for v in t0_.values() if v["tortuosity"]]
            deffs = [v["d_eff"] for v in t0_.values() if v["d_eff"]]
            entry = {"title": "Diffusion and tortuosity", "by_direction": t0_,
                     "by_realisation": [b_ for b_ in [by_real("Effective diffusivity / free diffusivity",
                                                              samples.get("tortuosity"), list(dirs), "-")] if b_],
                     "open_porosity": ptype.get("open"), "closed_porosity": ptype.get("closed")}
            if deffs:
                de = float(np.mean(deffs))
                F = 1.0 / de
                phi_o = ptype.get("open") or 0.0
                entry.update({"mean_tortuosity": float(np.mean(taus)) if taus else None, "mean_d_eff": de,
                              "formation_factor": F, "macmullin_number": F,
                              "archie_exponent": (-math.log(F) / math.log(phi_o)) if 0 < phi_o < 1 and F > 1 else None})
                dpo = results_first.get("open_pore_diameter_um")
                if dpo:
                    kn = PO.knudsen(dpo * 1e-6, gas, opts["temperature_K"])
                    kn["D_eff_molecular_m2_s"] = de * kn["D_molecular_m2_s"]
                    kn["D_eff_bosanquet_m2_s"] = de * kn["D_bosanquet_m2_s"]
                    entry["knudsen"] = kn
            props["tortuosity"] = entry

        for key, title in (("acoustics", "Acoustic absorption (JCA)"), ("radiation", "Radiative extinction"),
                           ("filtration", "Filtration efficiency"),
                           ("porosimetry", "Porosimetry and capillary pressure"), ("pore_network", "Pore network"),
                           ("grains", "Grain and filler analysis")):
            if key in results_first:
                props[key] = {"title": title, **results_first[key]}
        if "percolation" in results_first:
            props["percolation"] = results_first["percolation"]
            for gkey, g in props["percolation"].items():
                if "error" in g:
                    continue
                spans = [d for d, v in g["by_direction"].items() if v["percolates"]]
                checks.append({"id": f"perc_{gkey}", "group": "Percolation",
                               "label": f"{g['name']}: percolating directions ({g['connectivity']}-connectivity)",
                               "value": len(spans), "target": len(dirs), "unit": "directions",
                               "status": "pass" if len(spans) == len(dirs) else ("warn" if spans else "warn"),
                               "note": "Spanning along " + (", ".join(spans) if spans else "no direction")})
        if "moisture" in need and "moisture" in results_first:
            props["moisture"] = _moisture_entry(spec, opts, table, realisations, results_first, checks)
        if "thermomech" in need and "thermomech" in results_first:
            tmr = results_first["thermomech"]
            props["thermomech"] = {"title": "Thermal stress and warpage", **tmr}
            conv_ = bool(tmr["heat"]["converged"]) and all(tmr["converged"])
            checks.append({"id": "tm_conv", "group": "Thermal stress and warpage",
                           "label": "Solver convergence (heat flow, stiffness, thermal stress)",
                           "value": max([tmr["heat"]["residual"]] + list(tmr["residual"])), "target": 1e-5, "unit": "",
                           "status": "pass" if conv_ else "fail", "note": ""})
            sub_ = tmr.get("stress_stats") or {}
            hot_ = [v["name"] for v in sub_.values() if v.get("above_tg")]
            if hot_:
                checks.append({"id": "tm_tg", "group": "Thermal stress and warpage",
                               "label": "Regions above their glass transition", "value": tmr.get("above_tg") or 0.0,
                               "target": 0.0, "unit": "", "status": "warn",
                               "note": ", ".join(hot_) + " partly above Tg: the rubbery modulus and the CTE above Tg "
                                       "are used there (elastic; no relaxation)"})
        if "joule" in need and "joule" in results_first:
            jr = results_first["joule"]
            props["joule"] = {"title": "Joule heating", **jr}
            st_ = jr["steps"]
            if st_:
                bal = max(x["balance"] for x in st_)
                checks.append({"id": "jh_balance", "group": "Joule heating",
                               "label": "Energy balance: Joule heat against the heat leaving through the faces",
                               "value": bal, "target": 1e-3, "unit": "rel",
                               "status": "pass" if bal <= 1e-3 else ("warn" if bal <= 1e-2 else "fail"), "note": ""})
                bad = [x for x in st_ if x["status"] != "converged"]
                checks.append({"id": "jh_conv", "group": "Joule heating",
                               "label": "Steady state at every step (potential and temperature)",
                               "value": len(st_) - len(bad), "target": len(st_), "unit": "steps",
                               "status": "pass" if not bad else "warn",
                               "note": "" if not bad else
                               (f"thermal runaway at {bad[0]['value']:.4g} "
                                f"{'A/m²' if jr['control'] == 'current' else 'V'}: the temperature kept rising"
                                if bad[0]["status"] == "runaway" else
                                f"above 1000 °C at {bad[0]['value']:.4g} {'A/m²' if jr['control'] == 'current' else 'V'}")
                               + ("; the sweep stopped there" if bad else "")})
            if jr.get("error"):
                checks.append({"id": "jh_perc", "group": "Joule heating", "label": "A conducting path joins the faces",
                               "value": 0, "target": 1, "unit": "", "status": "fail", "note": jr["error"]})
        if "viscosity" in need and "viscosity" in results_first:
            props["viscosity"] = _viscosity_entry(spec, opts, table, realisations, results_first,
                                                  samples.get("viscosity") or [], checks)
            props["viscosity"]["by_realisation"] = [b_ for b_ in [by_real(
                "Relative viscosity, RVE flow solve", [[m_] for m_ in samples.get("viscosity") or []], ["μr"], "-")]
                if b_]
        if "morphology" in need and "morphology" in results_first:
            props["morphology"] = results_first.get("morphology", {})
            props["profiles"] = results_first.get("profiles")

        # statistics across realisations
        if seeds >= 2:
            for key, e in props.items():
                ci = e.get("ci95") if isinstance(e, dict) else None
                if ci is None or key not in COND:
                    continue
                rel = [c / abs(m) for c, m in zip(ci, e["diag"]) if m and np.isfinite(m) and c is not None]
                if rel:
                    worst = float(max(rel))
                    tgt = (settle["target_pct"] / 100.0) if settle else 0.02
                    # Kanit et al. (2003), n = 4 D_Z(V)^2 / (Z^2 e^2): the
                    # realisations of this size a 1 % and a 2 % interval need
                    covs = [float(s_) / abs(float(m)) for s_, m in zip(e.get("std") if e.get("std") is not None
                                                                         else [0.0], e["diag"])
                            if m and np.isfinite(m) and s_ is not None and np.isfinite(s_)]
                    cov = max(covs) if covs else 0.0
                    n_for = lambda eps: max(2, int(math.ceil(4.0 * cov ** 2 / eps ** 2)))  # noqa: E731
                    checks.append({"id": f"{key}_stat", "group": "Statistics", "label": f"{e['title']}: relative 95 % confidence interval",
                                   "value": worst, "target": tgt, "unit": "rel",
                                   "status": "pass" if worst <= tgt else ("warn" if worst <= 2.5 * tgt else "fail"),
                                   "note": f"{seeds} independent realisations; scatter between them {100 * cov:.2f} % "
                                           f"(standard deviation over the mean): about {n_for(0.01)} realisations of this "
                                           f"size give ±1 %, {n_for(0.02)} give ±2 % (Kanit et al. 2003, n = 4 D²/(Z² e²))"
                                           + ("; the realisations stopped at the target" if settle and worst <= tgt else
                                              "; the most realisations allowed were reached first" if settle else "")})
            if settle:
                checks.append({"id": "settle", "group": "Statistics",
                               "label": "Statistical RVE: a smaller volume, realisations until the means settle",
                               "value": seeds, "target": settle["n_max"], "unit": "realisations",
                               "status": "pass" if (settled[0] is not None and settled[0] <= settle["target_pct"] / 100.0)
                               else "warn",
                               "note": (f"widest interval {100 * settled[0]:.2f} % ({settled[1]}) after {seeds} realisations, "
                                        f"target {settle['target_pct']:g} %. " if settled[0] is not None else "")
                                       + "A mean over small volumes estimates the effective property only when the "
                                         "volume is large enough not to bias it (Kanit et al. 2003 found a bias for "
                                         "too small volumes with every boundary condition, least with periodic ones): "
                                         "compare once with a larger RVE (Standard or Accurate) before relying on it."})
        return props, composition, rho, cp

    if direct:
        _gate(log)
        prog = Progress(event, 1.5 + 3.0 + float(opts["viscosity"]["dem"]["rates"] or 0))
        results_first, rinfo_d = _dem_direct_run(spec, opts, gen_phases, table, out_dir, log, event, prog,
                                                 should_stop)
        realisations.append(rinfo_d)
    for r in range(0 if direct else seeds):
        first = r == 0
        seed = spec["rve"]["seed"] + 1000 * r
        sheared = shear_key(spec)
        pressed = press_key(spec)
        prog.stage("Structure generation", f"realisation {r+1}/{seeds} · {_grid(N, Nz)} voxels"
                   + (f" · sheared to a strain of {sheared['strain']}" if sheared else "")
                   + (f" · pressed by {pressed['compression']:g} %" if pressed else ""),
                   units=5.0 if (sheared or pressed) else 1.0)
        # the contact faces are kept whenever a contact resistance is set, not
        # only when this run solves heat: the structure stays the same one for
        # every selection of analyses
        want_cf = wants_contact_faces(spec)
        skey = structure_key(gen_phases, plan, contacts_key(opts), seed, want_cf, sheared=sheared,
                             dispersion=spec["rve"].get("dispersion"), pressed=pressed)
        got = load_structure([out_dir] + list(reuse.get("from") or []), skey) if opts.get("reuse_runs", True) else None
        if got is not None:
            ok, sh = kept_share_ok(got[1], got[0], h, gen_phases, bool(plan.get("film")), opts)
            if not ok:
                log(f"   realisation {r+1}: the structure generated earlier has {sh:.3g} % of its particle voxels in two "
                    f"particles, against the share in force now; it is generated again")
                got = None
        if got is not None:
            gen_labels, ginfo, src = got
            log(f"   realisation {r+1}: the structure generated earlier is used "
                f"({'this run' if os.path.abspath(src) == os.path.abspath(out_dir) else os.path.basename(src)}); "
                f"no new generation")
            if (overlap_steps(opts) and (ginfo.get("pack_scale") or 1.0) < 1.0 and not ginfo.get("overlap_steps")
                    and first):
                log(f"   !! this structure was packed before progressive overlap existed and fell short of the "
                    f"loading (size scale {ginfo['pack_scale']:.3f}); untick 'Reuse the structure of a matching "
                    f"run' (Simulation) to pack it again with overlap allowed")
            if os.path.abspath(src) != os.path.abspath(out_dir):
                save_structure(out_dir, skey, gen_labels, ginfo, ginfo.get("contact_mask"))
                # the compression's steps in 3D come with its structure
                ps = press_steps_ok(src, skey) if (pressed and first) else None
                if ps and not press_steps_ok(out_dir, skey):
                    try:
                        shutil.rmtree(os.path.join(out_dir, PRESS_STEPS), ignore_errors=True)
                        shutil.copytree(ps, os.path.join(out_dir, PRESS_STEPS))
                    except OSError:
                        pass
        else:
            gen_labels, ginfo = GEN.generate(N, h, gen_phases, seed=seed, log=log, should_stop=should_stop,
                                             contacts=opts["contacts"], gap_vox=opts.get("particle_gap_vox") or 2.0,
                                             overlap_steps=overlap_steps(opts),
                                             want_contact_faces=want_cf, nz=Nz,
                                             skin=skin, shear=shear_packing(spec, log, event, prog, should_stop),
                                             dispersion=spec["rve"].get("dispersion"))
            if pressed:
                gen_labels, ginfo = press_structure(spec, plan, gen_labels, ginfo, log, event, prog, should_stop,
                                                    steps_dir=os.path.join(out_dir, PRESS_STEPS) if first else None)
                if first and os.path.isdir(os.path.join(out_dir, PRESS_STEPS)):
                    with open(os.path.join(out_dir, PRESS_STEPS, "key.txt"), "w", encoding="utf-8") as fh:
                        fh.write(skey)
            try:
                save_structure(out_dir, skey, gen_labels, ginfo, ginfo.get("contact_mask"))
            except Exception as e:                                   # noqa: BLE001
                log(f"   (the structure could not be kept for later runs: {e})")
        if pressed:
            # the compressed structure is a periodic box of nx x ny x nz
            # voxels of the same size: every solve from here takes its shape
            shp = [int(v) for v in ginfo["shape"]]
            plan.update(N=shp[0], shape=shp, L_um=shp[0] * h, box_um=[v * h for v in shp])
        # whatever made the structure: every particle drawn once more and the
        # voxels two particles claim counted (the labels keep one owner per
        # voxel, so an overlap does not show in them)
        if "overlap_check" not in ginfo:
            try:
                ginfo["overlap_check"] = GEN.overlap_check(gen_labels.shape, h, ginfo, gen_phases,
                                                           film=bool(plan.get("film")))
            except Exception as e:                                   # noqa: BLE001
                ginfo["overlap_check"] = {"error": f"{type(e).__name__}: {e}"}
        oc = ginfo["overlap_check"]
        if oc.get("error"):
            log(f"   (the voxel overlap check could not run: {oc['error']})")
        elif oc.get("particles"):
            log(f"   voxel overlap check: {oc['shared']:,} voxels claimed by two particles "
                f"({100 * oc['of_particles']:.3f} % of the {oc['particle_voxels']:,} particle voxels, "
                f"{100 * oc['of_rve']:.3f} % of the RVE)"
                + (f"; {oc['permitted']:,} more where overlap is permitted" if oc.get("permitted") else "")
                + f" ({oc['seconds']:.1f} s)")
        labels, area_ratio, notes = build_labels(spec, gen_labels, ginfo, h)
        contact_mask = ginfo.pop("contact_mask", None)
        aniso_th = (thermal_aniso(spec, labels, ginfo, gen_phases, h, log if first else None)
                    if "thermal" in need else None)
        pair_faces = contact_pair_counts(gen_labels, contact_mask, len(spec["phases"]))
        del gen_labels
        for n in notes:
            log("   · " + n)
        counts = np.bincount(labels.ravel(), minlength=len(table))
        vf_labels = counts / labels.size
        rinfo = {"seed": seed, "route": ginfo["route"], "seconds": ginfo["seconds"],
                 "contact_pairs": pair_faces,
                 "contacts_mode": ginfo.get("contacts_mode"), "gap_vox": ginfo.get("gap_vox"),
                 "overlap_vox": ginfo.get("overlap_vox", 0),
                 "contacts_removed": ginfo["contacts_removed"],
                 "contact_faces": ginfo.get("contact_faces", 0), "touching_pairs": ginfo.get("touching_pairs", 0),
                 "voxel_conflicts": ginfo.get("voxel_conflicts", 0),
                 "overlap_check": ginfo.get("overlap_check"),
                 "overlap_limit": ginfo.get("overlap_limit"),
                 "shrunk": ginfo.get("shrunk"),
                 "pack_scale": ginfo.get("pack_scale"),
                 "sheared": ginfo.get("sheared"),
                 "pressed": ({k: v for k, v in ginfo["pressed"].items() if k not in ("frames", "von_mises", "pressure")}
                             if ginfo.get("pressed") else None),
                 "dispersion": ginfo.get("dispersion"),
                 "sections": ginfo.get("sections"),
                 "phases": ginfo["phases"], "vf_labels": vf_labels.tolist(),
                 "interface_area_ratio": {str(k): v for k, v in area_ratio.items()}}
        log(f"   realisation {r+1}: {ginfo['route']} · {ginfo['seconds']:.1f} s · "
            + ", ".join(f"{t['name']} {100*v:.2f}%" for t, v in zip(table, vf_labels)))
        if first:
            checks += R.post_checks(plan, ginfo, spec["rve"]["quality"])
            viewer = V.ViewWriter(out_dir, labels, table, h, feature_um=view_feature_um(plan),
                                  n_particles=len(ginfo.get("particles", {}).get("codes", [])),
                                  full_fields=opts.get("view_full_fields", True))
            # particle numbers on the surface grid, for one surface per particle
            try:
                viewer.set_owner(surface_owner(viewer, N, Nz, h, ginfo, gen_phases))
            except Exception as e:                                   # noqa: BLE001
                log(f"   (particle surfaces are drawn per phase: {e})")
            if prev is not None:
                # the fields of the kept analyses stay in the view
                viewer.adopt(keep=lambda key: True)
            if (ginfo.get("pressed") or {}).get("frames"):
                try:
                    viewer.add_press_anim(ginfo["pressed"], table, steps_dir=press_steps_ok(out_dir, skey))
                    pr_ = ginfo["pressed"]
                    viewer.add_field("press_vm", pr_["von_mises"], "von Mises stress in the resins at the end of the compression",
                                     "Pa", "Compression", in_matrix=True,
                                     note=f"viscous stress at {(pr_['series'].get('rate_1_s') or [pr_['rate_1_s']])[-1]:.3g} 1/s; "
                                          f"spheres and pores are left out")
                    viewer.add_field("press_p", pr_["pressure"], "Pressure at the end of the compression", "Pa",
                                     "Compression", diverging=True)
                except Exception as e:                               # noqa: BLE001
                    log(f"   (the compression could not be written for the viewer: {e})")
            # The surfaces are built before any solve, so that every field is
            # read onto them from the solver's grid when it is added (at the
            # end only the browser volume's block means were left).
            try:
                # the Voxels style is the full-resolution volume (vol_labels.u8.gz)
                viewer.build_surfaces(log=log)
            except Exception as e:                                   # noqa: BLE001
                log(f"   !! the particle surfaces could not be built: {e}")
                log(traceback.format_exc())
            viewer.set_owner(None)
            try:
                export_structure(out_dir, labels, table, h, vf_labels)
            except Exception as e:                                   # noqa: BLE001
                log(f"   (structure.tif could not be written: {e})")
        prog.end()
        res_r = {}

        live("structure")

        # ---- pore space: total, open and closed -------------------------
        void_mask = np.isin(labels, void_labels) if void_labels else None
        open_mask = None
        if void_labels:
            ptypes, open_mask = PO.porosity_types(void_mask)
            res_r["porosity"] = ptypes
            if first:
                log(f"    porosity {100*ptypes['total']:.2f} % (open {100*ptypes['open']:.2f} %, "
                    f"closed {100*ptypes['closed']:.2f} %)")

        # ---- rarefied gas in small pores ---------------------------------
        gas_factor = None
        if "thermal" in need and opts["knudsen"] and void_labels and void_mask.any():
            lt = MO.local_thickness(void_mask)
            d_pore = float(lt[void_mask].mean()) * h * 1e-6
            del lt
            kn = PO.knudsen(d_pore, gas, opts["temperature_K"])
            gas_factor, beta_k = PO.gas_conductivity_factor(kn["knudsen_number"], gas["gamma"], gas["prandtl"])
            res_r["knudsen_thermal"] = {**kn, "gas_factor": gas_factor, "kaganer_beta": beta_k}
            log(f"    Knudsen: pore diameter {d_pore*1e6:.4g} µm, Kn = {kn['knudsen_number']:.3g}, "
                f"gas conductivity × {gas_factor:.3f}")

        # ---- flow and diffusion use the open porosity only ---------------
        flow_labels = None
        if need & {"permeability", "tortuosity", "acoustics"}:
            flow_labels = labels.copy()
            flow_labels[void_mask & ~open_mask] = 0
        pore_open = open_mask

        # ---- the resolution check's finer grid, drawn once, before the batch
        rc_fine = None
        rc_mode = opts.get("resolution_check", "off")
        if first and any(k in need for k in COND) and (
                rc_mode == "on" or (rc_mode == "auto" and ginfo.get("touching_pairs", 0) > 0)):
            try:
                rc_fine = _fine_grid(spec, ginfo, gen_phases, N, h,
                                     int((limits or {}).get("N_max", R.DEFAULT_LIMITS["N_max"]) or 0))
            except Exception as e:                                       # noqa: BLE001
                rc_fine = f"the finer grid could not be drawn: {e}"

        # ---- independent solves, side by side ----------------------------
        # (the first one waits here until the server's queue has its memory)
        if not spec.get("preview"):
            _gate(log)
        pre_cond, pre_el = _prefetch(pool_box, need, labels, table, spec, h, backend, gas_factor, area_ratio,
                                     contact_mask, first, log, should_stop, os.path.join(out_dir, "logs"),
                                     event=event,
                                     flow={"labels": flow_labels, "void": void_labels, "open": pore_open}
                                     if void_labels else None,
                                     prog=prog, struct={"ginfo": ginfo, "void": void_labels} if first else None,
                                     rcheck={"fine": rc_fine} if isinstance(rc_fine, dict) else None,
                                     aniso=aniso_th)

        # ---- conduction-type properties ----------------------------------
        for key in COND:
            if key not in need:
                continue
            rpair = rcpair = None
            if key == "thermal":
                vals = thermal_values(spec, gas_factor)
                rpair, rcpair = thermal_interfaces(spec, h, area_ratio)
            else:
                vals = np.array(S.label_values(table, COND[key]["prop"]), float)
            out = _cond_solve(key, labels, vals, spec, h, backend, prog, log, event, should_stop,
                              viewer if first else None, first, rpair=rpair,
                              cmask=contact_mask if rcpair is not None else None, rcpair=rcpair, pre=pre_cond,
                              aniso=aniso_th if key == "thermal" else None)
            res_r[key] = out
            samples.setdefault(key, []).append(np.diag(out["tensor"]))
            if first:
                out["refs"] = _cond_refs(key, spec, vf_labels, vals)
                out["values_used"] = vals.tolist()
                checks += _cond_checks(key, out["tensor"], out["refs"], spec, out["stats"])
                # touching particles much more conductive than the matrix: the
                # area of each contact is a voxel face, so the value follows the
                # voxel size (AlN/epoxy, 7.5 voxels per diameter: 2.57 W/m.K,
                # 1.81 on a 1.5x finer grid, 0.64 with the particles apart)
                faces_ = int(ginfo.get("contact_faces", 0) or 0)
                fill_v = [float(v) for v, t in zip(vals, table) if t["kind"] in ("core", "shell") and float(v) > 0]
                mat_v = [float(v) for v, t in zip(vals, table) if t["kind"] == "matrix" and float(v) > 0]
                contrast = (max(fill_v) / min(mat_v)) if (fill_v and mat_v) else 0.0
                if faces_ > 0 and contrast >= 20.0 and opts.get("resolution_check", "off") != "on":
                    checks.append({"id": f"{key}_contacts", "group": COND[key]["title"],
                                   "label": "Touching particles much more conductive than the matrix",
                                   "value": faces_, "target": 0, "unit": "", "status": "warn",
                                   "note": f"{faces_:,} voxel faces of contact between particles up to {contrast:.0f}x "
                                           "as conductive as the matrix: each contact is a whole voxel face, so the "
                                           "value follows the voxel size and can fall by tens of per cent on a finer "
                                           "grid. Tick the resolution check to see by how much; a contact resistance "
                                           "R_c, a finer grid or a gap between particles bounds it (guide 5.5)."})
            event("partial", key=key, value=_clean(np.diag(out["tensor"])))
            live(key)

        # ---- frequency-resolved admittivity for EMI (first realisation) ----
        if first and "emi" in need and opts["emi"].get("method", "complex") == "complex":
            eo = opts["emi"]
            freqs = np.logspace(math.log10(eo["f_min_hz"]), math.log10(eo["f_max_hz"]), int(eo["n_freq"]))
            # the in-plane directions asked for (x when only z is): one
            # realisation of an "isotropic" structure can still differ 1.6x
            # between x and y near percolation, which x and y together show
            edirs = emi_directions(opts)
            ebc = "film" if film else "periodic"
            prog.stage("EMI · complex admittivity",
                       f"{len(freqs)} frequencies × {len(edirs)} in-plane direction(s), sigma + j w eps",
                       units=0.0 if any(k[0] == "emi" for k in pre_cond) else 1.0)
            log(f"    EMI: the RVE solved with complex admittivity at {len(freqs)} frequencies (x and y, {ebc})")
            kap, recs = CC.spectrum(labels, table, freqs, directions=edirs, bc=ebc, tol=min(opts["tol"], 1e-7),
                                    should_stop=should_stop, log=log,
                                    pre={k[1:]: v for k, v in pre_cond.items() if k[0] == "emi"})
            res_r["emi_complex"] = {"freq_hz": freqs.tolist(), "kappa_re": kap.real.tolist(),
                                    "kappa_im": kap.imag.tolist(), "records": recs,
                                    "directions": ["xyz"[d] for d in edirs], "bc": ebc}
            prog.end()
            if eo.get("fullwave"):
                prog.stage("EMI · full-wave simulation", "openEMS FDTD of the RVE slab, voxel by voxel, 10-100 GHz",
                           units=1.0)
                fw_last = {"t": 0.0}

                def fwcb(tag, step, n_exc, e_db):
                    # the energy against its maximum falls to the end
                    # criterion (1e-6) once the pulse has passed: drawn as
                    # the residual of the run. The narrow reference run is
                    # quick; the slab run is nearly all of the time, about
                    # 1.25 pulse lengths
                    now = time.time()
                    if now - fw_last["t"] < 1.0:
                        return
                    fw_last["t"] = now
                    frac = 0.02 if tag == "ref" else (0.03 + 0.96 * min(step / (1.25 * n_exc), 0.99) if n_exc else 0.03)
                    event("solver", name=f"openEMS {'incident wave' if tag == 'ref' else 'RVE slab'} (energy)",
                          it=int(step), res=10.0 ** (e_db / 10.0), target=1e-6, progress=prog.sub(frac))
                # a conducting filler whose skin depth at the lowest
                # full-wave frequency is below the voxel: the FDTD does not
                # resolve its eddy currents, the homogenisation leaves them
                # out, so the two can differ widely there
                thin_skin = []
                for t_ in table:
                    sg_ = float(t_["props"].get("sigma", 0.0) or 0.0)
                    if t_["kind"] != "matrix" and sg_ > 1e3:
                        mr_ = max(float(t_["props"].get("mu_r", 1.0) or 1.0), 1.0)
                        dl_ = 1.0 / math.sqrt(math.pi * 10e9 * 4e-7 * math.pi * mr_ * sg_)
                        if dl_ < h * 1e-6:
                            thin_skin.append((t_["name"], dl_ * 1e6))
                for nm_, dl_ in thin_skin:
                    log(f"    note: at 10 GHz the skin depth of {nm_} ({dl_:.2g} µm) is below the voxel ({h:.3g} µm); "
                        f"the FDTD does not resolve its eddy currents and the homogenisation leaves them out, "
                        f"so the two can differ widely")
                try:
                    if not FW.find_openems():
                        raise FileNotFoundError("openEMS is not installed (tools\\openEMS; 1_INSTALL.bat downloads it, "
                                                "the offline package contains it)")
                    res_r["emi_fullwave"] = FW.check(labels, table, h, os.path.join(out_dir, "fullwave"), log=log,
                                                     should_stop=should_stop, progress=fwcb)
                    res_r["emi_fullwave"]["thin_skin"] = [{"name": nm_, "skin_um": dl_, "voxel_um": h}
                                                          for nm_, dl_ in thin_skin]
                    log(f"    full-wave simulation: largest SE difference to the homogenised slab "
                        f"{res_r['emi_fullwave']['max_diff_db']:.2f} dB ({res_r['emi_fullwave']['seconds']:.0f} s)")
                except InterruptedError:
                    raise
                except Exception as e:                               # noqa: BLE001
                    res_r["emi_fullwave"] = {"error": str(e)}
                    log(f"    full-wave simulation skipped: {e}")
                prog.end()

        if "emi_complex" in res_r:
            live("emi")

        # ---- resolution check (first realisation) ------------------------
        rc_mode = opts.get("resolution_check", "off")
        cond_keys = [k for k in COND if k in need and k in res_r]
        if first and cond_keys and (rc_mode == "on" or (rc_mode == "auto" and ginfo.get("touching_pairs", 0) > 0)):
            why = "particles touch at voxel level" if rc_mode == "auto" else "requested"
            prog.stage("Resolution check", f"the same structure on a 1.5× finer grid ({why})",
                       units=0.0 if any(str(k[0]).startswith("res:") for k in pre_cond) else 1.0)
            rc = _resolution_check(spec, ginfo, gen_phases, N, h, need, gas_factor, area_ratio, res_r, backend,
                                   log, should_stop, int((limits or {}).get("N_max", R.DEFAULT_LIMITS["N_max"]) or 0),
                                   pre=pre_cond, fine=rc_fine)
            rc_fine = None
            prog.end()
            if isinstance(rc, str):
                log(f"    resolution check not run: {rc}")
            else:
                for key, v in rc.items():
                    res_r[key]["resolution"] = v
                    title = COND[key]["title"]
                    checks.append({"id": f"{key}_res", "group": title,
                                   "label": "Change on a 1.5× finer grid of the same structure",
                                   "value": abs(v["change"]), "target": 0.03, "unit": "rel",
                                   "status": "pass" if abs(v["change"]) <= 0.03 else "warn",
                                   "note": f"{v['direction']} direction: {v['grid']} {v['base']:.4g} → {v['grid_fine']} "
                                           f"{v['fine']:.4g}; "
                                           + (f"extrapolated to fine voxels {v['extrapolated']:.4g}. "
                                              if v.get("extrapolated") is not None
                                              else f"not extrapolated: {v['extrapolation_note']}. ")
                                           + f"Run because {why}."})

        # ---- elasticity and thermal expansion ----------------------------
        if "cte" in need:
            el = _elastic_solve(labels, table, spec, h, backend, prog, log, event, should_stop,
                                viewer if first else None, first, pre_loads=pre_el)
            consts = EL.engineering_constants(el["C"])
            el["constants"] = consts
            res_r["cte"] = el
            samples.setdefault("cte_alpha", []).append(el["alpha"][:3])
            samples.setdefault("cte_E", []).append(consts["E"])
            log(f"    α = {np.round(el['alpha'][:3], 3)} ppm/K, E = {np.round(consts['E'], 3)} GPa")
            if opts.get("above_tg"):
                el["above_tg"] = _above_tg(labels, table, prog, log, event, should_stop)
            event("partial", key="cte", value=_clean(el["alpha"][:3]))
            live("cte")

        # ---- mechanical test (tension or compression) --------------------
        if "mechtest" in need and "cte" in res_r:
            st = MT.macro_state(res_r["cte"]["C"], opts["mechtest"])
            samples.setdefault("mt_modulus", []).append(st["modulus_GPa"])
            if first:
                res_r["mechtest"] = _mech_test(labels, table, spec, st, h, prog, log, event, should_stop, viewer)
                event("partial", key="mechtest", value=_clean(st["modulus_GPa"]))
                live("mechtest")

        # ---- viscosity of the uncured compound ---------------------------
        if "viscosity" in need:
            vo = opts["viscosity"]
            vtol = 1e-5 if spec["rve"]["quality"] == "fast" else 1e-6
            # is the particle dynamics the answer here? spheres (no bubbles) above
            # 35 vol% or touching: the voxel flow solve cannot resolve their
            # films, so its value is not used - it is skipped unless its 3D
            # resin flow is asked for (option), and then solved at the same time
            # as the particle dynamics when that runs on a GPU
            from .solvers import suspension as SU
            solid_v = [i for i, ph in enumerate(spec["phases"])
                       if not ph["void"] and ph["shape"] != G.NETWORK and ph["vf"] > 0]
            dem_possible = (vo["dem"]["on"] and bool(solid_v)
                            and not any(ph["void"] and ph["vf"] > 0 for ph in spec["phases"]))
            if first:
                phi_v = float(sum(v for v, t in zip(vf_labels, table)
                                  if t["kind"] in ("core", "shell") and float(t["props"]["E"]) > 0))
                touch_v = int(rinfo.get("touching_pairs", 0) or 0)
                dem_answer = dem_possible and not (phi_v <= 0.35 and touch_v == 0)
                visc_plan.update(phi=phi_v, touching=touch_v, dem_answer=dem_answer,
                                 skip_rve=dem_answer and not vo["rve_with_dem"],
                                 concurrent=bool(vo["rve_with_dem"] and dem_possible and vo["dem"]["backend"] != "cpu"
                                                 and (SU.cuda_device_count() > 0
                                                      or os.environ.get("MPSIM_DEM_CONCURRENT") == "1")))
            try:
                import numba
                vworkers = int(numba.get_num_threads())
            except Exception:                                            # noqa: BLE001
                vworkers = -1
            dem_job = None
            if first and visc_plan["concurrent"]:
                # [eta] for the stresslet before the analysis has its own: the
                # closed forms of the particle dynamics, else those of the
                # analysis (a platelet stack of aspect 10 took 2.5 here)
                eta_c, eta_cb = dem_intrinsic(spec, solid_v)
                if eta_cb != "closed form" and _eta_closed(spec, solid_v) is not None:
                    eta_c, eta_cb = _eta_closed(spec, solid_v), "closed form"
                dem_job = _DemProcess(spec, solid_v, float(sum(spec["phases"][i]["vf"] for i in solid_v)), vo,
                                      viewer, log, eta_int=(eta_c, eta_cb) if eta_cb == "closed form" else None,
                                      start=_dem_start(ginfo))
                log("    the particle dynamics runs on the GPU in its own process, at the same time as the flow "
                    "solve below on the CPU cores")
            if visc_plan["skip_rve"]:
                sph_v = all(spec["phases"][i]["shape"] == "sphere" for i in solid_v)
                why = (f"{'spheres' if sph_v else 'particles'} at {100 * visc_plan['phi']:.0f} vol%"
                       + (f", {visc_plan['touching']} touching pairs" if visc_plan["touching"] else ""))
                prog.stage("Viscosity · RVE flow solve not needed",
                           f"the particle dynamics gives the answer ({why}); the option 'Also solve the resin flow on "
                           f"the RVE' adds it", units=4.0)
                if first:
                    log(f"    RVE flow solve skipped: the particle dynamics gives the answer here ({why}); the "
                        f"voxel flow solve cannot resolve the resin films between such particles")
                res_r["viscosity"] = {"skipped": f"the particle dynamics gives the answer ({why})",
                                      "touching_pairs": visc_plan["touching"]}
                prog.end()
            else:
                prog.stage("Viscosity · creeping flow of the resin around the fillers",
                           "voxel FE (B-bar) · FFT-preconditioned CG · 3 shear planes + extension · CPU cores", units=4.0)
                vnames = ["yz", "xz", "xy", "extension"]

                def cbv(plane, it, res):
                    k = vnames.index(plane)
                    event("solver", name=f"Viscosity, shear {plane}" if plane != "extension" else "Viscosity, extension",
                          it=it, res=res, target=vtol, progress=prog.sub(min(k + 0.5, 3.99) / 4.0))
                    if dem_job is not None:
                        dem_job.poll()
                        st_ = dem_job.status()
                        if st_:
                            event("stage", name="Viscosity · creeping flow of the resin around the fillers",
                                  detail=f"voxel FE on the CPU cores, {plane} · at the same time on the GPU, {st_}",
                                  progress=prog.sub(min(k + 0.5, 3.99) / 4.0))
                if first and _LIVE["dir"]:
                    cbv.snapshot = LV.Multi(_LIVE["dir"], "viscosity", "Viscosity, shear",
                                            "Resin velocity disturbance |v'| (wall speed per voxel)")
                try:
                    vr = VI.effective_viscosity(labels, VI.phase_viscosities(table), tol=vtol, progress=cbv,
                                                should_stop=should_stop, workers=vworkers, keep_field=first)
                except BaseException:
                    if dem_job is not None:              # stopped or failed: the GPU process ends too
                        dem_job.cancel()
                    raise
                vfield = vr.pop("field", None)
                vvel = vr.pop("velocity", None)
                if first and viewer is not None and vfield is not None:
                    # a quantity of the resin: the fillers are rigid and their
                    # inside is zero, so the particle surfaces read the resin
                    # beside them (read from inside, every particle came out the
                    # bottom colour of the scale)
                    viewer.add_field("visc_gd", vfield, "Local shear rate / applied shear rate (xy shear)", "-", "Viscosity",
                                     note="Where the resin is sheared hardest between the fillers", in_matrix=True)
                if first and viewer is not None and vvel is not None:
                    try:
                        vr["flow_lines"] = _shear_flow_layer(viewer, vvel, h)
                    except Exception as e:                                   # noqa: BLE001
                        log(f"    (the flow lines could not be drawn: {e})")
                del vfield, vvel
                vr["touching_pairs"] = rinfo.get("touching_pairs", 0)
                samples.setdefault("viscosity", []).append(vr["mu_r"])
                res_r["viscosity"] = vr
                sh_txt = " / ".join(f"{vr['shear'][p_]:.4g}" for p_ in ("yz", "xz", "xy"))
                log(f"    relative viscosity μr = {vr['mu_r']:.4g} (shear yz / xz / xy {sh_txt}, "
                    f"extension {vr['extension']:.4g})")
                event("partial", key="viscosity", value=_clean([vr["shear"][p] for p in ("yz", "xz", "xy")]))
                prog.end()
            if first:
                vx = {}
                solid = [i for i, ph in enumerate(spec["phases"])
                         if not ph["void"] and ph["shape"] != G.NETWORK and ph["vf"] > 0]
                if solid:
                    prog.stage("Viscosity · maximum packing fraction", "random close packing of these particles", units=1.0)
                    sp_solid = [dict(gen_phases[i], void=False) for i in solid]          # outer sizes
                    try:
                        vx.update(max_packing(sp_solid, vo, log, should_stop))
                    except BaseException:
                        if dem_job is not None:
                            dem_job.cancel()
                        raise
                    prog.end()
                else:
                    vx["phi_m"] = vo["phi_m"]
                eta_cf = _eta_closed(spec, solid)
                if eta_cf is not None:
                    vx["eta"], vx["eta_basis"] = eta_cf, "closed form"
                elif solid and vo["dilute_rve"]:
                    prog.stage("Viscosity · intrinsic viscosity", "dilute RVE of the same particles", units=1.0)
                    try:
                        vx["eta"], vx["dilute"] = _dilute_eta([gen_phases[i] for i in solid], log, should_stop,
                                                              vworkers)
                        vx["eta_basis"] = "dilute RVE"
                    except InterruptedError:
                        raise
                    except Exception as e:                               # noqa: BLE001
                        log(f"    dilute RVE failed ({e}); [η] = 2.5 is used")
                        vx["eta"], vx["eta_basis"] = 2.5, "sphere value (dilute RVE failed)"
                    prog.end()
                else:
                    vx["eta"], vx["eta_basis"] = 2.5, "sphere value"
                res_r["viscosity_extra"] = vx
                # particle dynamics of the spheres (all solid phases spheres, no bubbles)
                if dem_job is not None and not (vo["dem"]["on"] and solid):
                    dem_job.cancel()
                if vo["dem"]["on"] and solid:
                    bubbles_ = any(ph["void"] for ph in spec["phases"])
                    if bubbles_:
                        res_r["viscosity_dem"] = {"error": "bubbles are not in the particle dynamics; the RVE flow solve and "
                                                           "Krieger-Dougherty are used"}
                    else:
                        phi_s = float(sum(spec["phases"][i]["vf"] for i in solid))
                        vo_run = vo
                        if dem_job is not None:
                            try:
                                res_r["viscosity_dem"] = dem_job.wait(viewer if first else None, prog, event,
                                                                      should_stop, 3.0 + float(vo["dem"]["rates"] or 0))
                            except InterruptedError:
                                raise
                            except Exception as e:                       # noqa: BLE001
                                # the GPU process failed: once more here, on the CPU
                                log(f"    the particle dynamics process failed ({e}); it is run again here on the CPU")
                                vo_run = dict(vo, dem=dict(vo["dem"], backend="cpu"))
                                dem_job = None
                        if dem_job is None:
                            try:
                                res_r["viscosity_dem"] = _dem_viscosity(
                                    spec, solid, phi_s, vo_run, log, event, prog, should_stop, viewer if first else None,
                                    eta_int=(vx["eta"], vx.get("eta_basis", "dilute RVE")) if vx.get("eta") is not None
                                    else None, start=_dem_start(ginfo))
                            except InterruptedError:
                                raise
                            except Exception as e:                       # noqa: BLE001
                                prog.end()
                                res_r["viscosity_dem"] = {"error": f"the particle dynamics failed: {e}"}
                                log(f"    particle dynamics not run: {e}")

        # ---- moisture ------------------------------------------------------
        if "moisture" in need:
            res_r["moisture"] = _moisture_solve(labels, table, spec, h, vf_labels, prog, log, event, should_stop,
                                                viewer if first else None, first,
                                                C_known=(res_r.get("cte") or {}).get("C"), pre=pre_cond)
            event("partial", key="moisture", value=_clean([res_r["moisture"]["wt_pct"]]))
            live("moisture")

        # ---- coupled analyses of the film (first realisation) ---------------
        if "thermomech" in need and first:
            res_r["thermomech"] = _thermomech_solve(labels, table, spec, h, gas_factor, area_ratio, contact_mask,
                                                    prog, log, event, should_stop, viewer, aniso=aniso_th)
            event("partial", key="thermomech", value=_clean([res_r["thermomech"]["warpage_um"]]))
            live("thermomech")
        if "joule" in need and first:
            res_r["joule"] = _joule_solve(labels, table, spec, h, gas_factor, area_ratio, contact_mask, prog, log,
                                          event, should_stop, viewer)
            event("partial", key="joule", value=_clean([s_["T_max"] for s_ in res_r["joule"]["steps"]]))
            live("joule")

        if "permeability" in need:
            wr = CD.wrapping_axes(pore_open)
            Kt = np.zeros((3, 3))
            flow_stats = {}
            u_filt = None
            for d in dirs:
                di = "xyz".index(d)
                prog.stage(f"Permeability · {d} direction", "NASA PuMA Stokes FE",
                           units=0.0 if ("stokes", d) in pre_cond else 2.0)
                if not wr[di]:
                    log(f"    no connected pore path along {d}: permeability = 0")
                    prog.end()
                    continue

                def cbp(tag, it, res, tgt, frac, d=d):
                    event("solver", name=f"Stokes flow {d}", it=it, res=res, target=tgt, progress=prog.sub(frac))
                pre_s = pre_cond.get(("stokes", d))
                if pre_s is not None:
                    # solved in the pool, side by side with the other directions
                    Kmat, comps, st = pre_s["K"], [c for c in pre_s["u"] if c is not None], dict(pre_s["stats"])
                else:
                    pr = PB.permeability(flow_labels, void_labels, h_m, d, tol=opts["tol"],
                                         progress=cbp, should_stop=should_stop)
                    Kmat = pr["K"]
                    comps = [np.asarray(c, float) for c in pr["u"] if c is not None]
                    if comps and comps[0].ndim == 4:
                        comps = [comps[0][..., k] for k in range(comps[0].shape[-1])]
                    st = {"seconds": pr["seconds"], "iterations": pr["iterations"], "converged": pr["converged"]}
                    del pr
                Kt[:, di] = Kmat[:, di] if Kmat.ndim == 2 else Kmat
                if len(comps) == 3 and comps[0].shape == labels.shape:
                    speed = np.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2)
                    u_d = float(comps[di][pore_open].mean())
                    st["hydraulic_tortuosity"] = float(speed[pore_open].mean() / abs(u_d)) if u_d else None
                    if first and viewer is not None:
                        viewer.add_field(f"perm_{d}_u", speed, f"Velocity magnitude |u| ({d.upper()} pressure gradient)",
                                         "m/s", "Flow", note="Unit pressure gradient (1 Pa/m) and viscosity 1 Pa·s")
                        viewer.add_field(f"perm_{d}_ud", comps[di], f"Velocity component u{d} ({d.upper()} pressure gradient)",
                                         "m/s", "Flow", diverging=True)
                        if d == dirs[0]:
                            # the streamlines are the flow path through the pore
                            # space, and their mean length is a second, independent
                            # estimate of the hydraulic tortuosity
                            st["field_lines"] = _vector_layer(viewer, f"perm_{d}_u", np.stack(comps, axis=-1),
                                                              pore_open, d, h, "Velocity u", "m/s", "Flow",
                                                              f"perm_{d}_u")
                    if first and d == dirs[0] and "filtration" in need:
                        u_filt = np.stack(comps, axis=-1)   # carried into the filtration stage
                    del speed
                del comps
                flow_stats[d] = st
                log(f"    permeability {d}: {Kt[di, di]:.4g} m² ({st['seconds']:.1f} s)")
                prog.end()
            res_r["permeability"] = {"tensor": Kt, "percolation": wr, "stats": flow_stats}
            if first and pore_open.any():
                try:
                    _area, sv_open = PB.surface_area(pore_open.astype(np.uint8), [1], h_m)
                    res_r["permeability"]["specific_surface_open_1pm"] = sv_open
                except Exception:                                      # noqa: BLE001
                    pass
            samples.setdefault("permeability", []).append(np.diag(Kt))

            # ---- filtration: particles carried by that same flow ----------
            if first and "filtration" in need and u_filt is not None:
                from . import filtration as FI
                prog.stage("Filtration efficiency", "particle tracking through the Stokes field", units=0.7)
                di0 = "xyz".index(dirs[0])
                try:
                    lam = float(1.380649e-23 * opts["temperature_K"]
                                / (np.sqrt(2.0) * np.pi * (gas["molecule_diameter_nm"] * 1e-9) ** 2
                                   * gas["pressure_Pa"]))
                    fopts = {**opts["filtration"], "direction": dirs[0],
                             "temperature_K": opts["temperature_K"]}
                    fres, fpaths = FI.analyse(u_filt, pore_open, h, {**gas, "mean_free_path_m": lam},
                                              fopts, K_m2=float(Kt[di0, di0]), log=log)
                    fres["mean_free_path_nm"] = lam * 1e9
                    res_r["filtration"] = fres
                    if viewer is not None and fpaths:
                        by_d = {}
                        for pth in fpaths:
                            by_d.setdefault(pth["diameter_um"], []).append(pth)
                        for i, (d0, lines) in enumerate(sorted(by_d.items())):
                            n_cap = sum(1 for ln in lines if ln.get("captured"))
                            viewer.add_paths(f"path_filt_{i}", lines,
                                             f"Particle tracks, {d0:g} µm ({n_cap} of {len(lines)} captured)",
                                             "Filtration", kind="particle", diameter_um=d0,
                                             note="Convection, inertia, Brownian motion and interception in the solved flow")
                        # the smallest particles are caught within a voxel or two
                        # of where they start, so their tracks show almost
                        # nothing; the most penetrating size is the one that
                        # travels through the structure and is worth opening
                        if fres.get("mpps_um") is not None:
                            keys = {d0: f"path_filt_{i}" for i, (d0, _ln) in enumerate(sorted(by_d.items()))}
                            if keys:
                                fres["mpps_path_key"] = keys[min(keys, key=lambda d: abs(d - fres["mpps_um"]))]
                except Exception as e:                                # noqa: BLE001
                    res_r["filtration"] = {"error": f"{type(e).__name__}: {e}"}
                    log(f"    filtration failed: {e}")
                del u_filt
                u_filt = None
                prog.end()

        C_first = None
        # sound enters a layer through its thickness: JCA uses z when solved
        ac_dir = "z" if "z" in dirs else dirs[0]
        if "permeability" in need:
            live("permeability")
        if "tortuosity" in need:
            tt = {}
            for d in dirs:
                prog.stage(f"Diffusion · {d} direction", "NASA PuMA continuum tortuosity",
                           units=0.0 if ("tort", d) in pre_cond else 0.5)

                def cbt(tag, it, res, tgt, frac, d=d):
                    event("solver", name=f"Diffusion {d}", it=it, res=res, target=tgt, progress=prog.sub(frac))
                tr = pre_cond.get(("tort", d))
                if tr is None:
                    tr = PB.tortuosity(flow_labels, void_labels, h_m, d, tol=opts["tol"],
                                       progress=cbt, should_stop=should_stop)
                tt[d] = {k: tr[k] for k in ("tortuosity", "d_eff", "porosity", "seconds", "converged")}
                tt[d]["formation_factor"] = 1.0 / tr["d_eff"] if tr["d_eff"] > 0 else None
                if first and viewer is not None:
                    viewer.add_field(f"tort_{d}_C", tr["C"], f"Concentration ({d.upper()} gradient)", "-", "Diffusion")
                if first and "acoustics" in need and d == ac_dir:
                    C_first = np.asarray(tr["C"], float)
                log(f"    diffusion {d}: τ = {tr['tortuosity']:.4g}, D_eff/D₀ = {tr['d_eff']:.4g}")
                del tr
                prog.end()
            res_r["tortuosity"] = tt
            samples.setdefault("tortuosity", []).append([tt[d]["d_eff"] for d in dirs])
            if first and pore_open is not None and pore_open.any():
                lt = MO.local_thickness(pore_open)
                res_r["open_pore_diameter_um"] = float(lt[pore_open].mean()) * h
                del lt

        if "tortuosity" in need:
            live("tortuosity")

        # ---- first realisation only: analyses of the image ---------------
        if first and "acoustics" in need and C_first is not None:
            prog.stage("Acoustic absorption", "Johnson–Champoux–Allard from the RVE fields", units=0.3)
            try:
                phi_o = float(pore_open.mean())
                Kd = [res_r["permeability"]["tensor"][i, i] for i in range(3) if res_r["permeability"]["tensor"][i, i] > 0]
                taus = [v["tortuosity"] for v in res_r["tortuosity"].values() if v["tortuosity"]]
                area, _sv = PB.surface_area(pore_open.astype(np.uint8), [1], h_m)
                ratio = area / max(AC.voxel_face_area(pore_open, h_m), 1e-300)
                lam = AC.viscous_length(C_first, pore_open, h_m, ac_dir, ratio)
                lam_p = 2.0 * float(pore_open.sum()) * h_m ** 3 / area
                ai = "xyz".index(ac_dir)
                K_dir = float(res_r["permeability"]["tensor"][ai, ai])
                K_mean = K_dir if K_dir > 0 else float(np.mean(Kd))
                sigma_f = gas["viscosity_Pa_s"] / K_mean
                t_dir = (res_r["tortuosity"].get(ac_dir) or {}).get("tortuosity")
                a_inf = float(t_dir) if t_dir else float(np.mean(taus))
                ao = opts["acoustics"]
                f = np.geomspace(ao["f_min_hz"], ao["f_max_hz"], 140)
                ab = AC.absorption(f, ao["thickness_mm"] * 1e-3, phi_o, sigma_f, a_inf, lam, lam_p, gas)
                res_r["acoustics"] = {"direction": ac_dir, "porosity": phi_o, "flow_resistivity_Pa_s_m2": sigma_f, "tortuosity": a_inf,
                                      "viscous_length_um": lam * 1e6, "thermal_length_um": lam_p * 1e6,
                                      "thickness_mm": ao["thickness_mm"], "spectrum": ab,
                                      "nrc": AC.nrc(f, ab["alpha"]),
                                      "alpha_at": {str(int(b)): float(np.interp(math.log10(b), np.log10(f), ab["alpha"]))
                                                   for b in (125, 250, 500, 1000, 2000, 4000, 8000) if f[0] <= b <= f[-1]}}
                log(f"    JCA: σ = {sigma_f:.4g} Pa·s/m², α∞ = {a_inf:.3g}, Λ = {lam*1e6:.3g} µm, Λ' = {lam_p*1e6:.3g} µm")
            except Exception as e:                                        # noqa: BLE001
                res_r["acoustics"] = {"error": f"{type(e).__name__}: {e}"}
                log(f"    acoustic analysis failed: {e}")
            prog.end()
        del C_first

        if first and "acoustics" in need:
            live("acoustics")
        if first and "radiation" in need:
            ro = opts["radiation"]
            prog.stage("Radiative extinction", f"NASA PuMA ray casting · {ro['sources']} sources × {ro['rays']} rays",
                       units=0.0 if ("radiation",) in pre_cond else 0.5)
            try:
                rr = pre_cond.get(("radiation",)) or PB.radiation(labels, void_labels, h_m, ro["sources"], ro["rays"],
                                                                   should_stop)
                beta = float(np.mean(rr["beta_1pm"]))
                area, sv = PB.surface_area(labels, void_labels, h_m)
                phi_v = float(void_mask.mean())
                n2 = ro["refractive_index"] ** 2
                Ts = np.linspace(300.0, 2000.0, 35)
                rr.update({"beta_mean_1pm": beta, "temperature_K": ro["temperature_K"],
                           "refractive_index": ro["refractive_index"],
                           "k_rad_W_mK": 16.0 * n2 * SIGMA_SB * ro["temperature_K"] ** 3 / (3.0 * beta),
                           "k_rad_curve": {"T_K": Ts.tolist(), "k_W_mK": (16.0 * n2 * SIGMA_SB * Ts ** 3 / (3.0 * beta)).tolist()},
                           "mean_chord_beta_1pm": sv / (4.0 * phi_v) if phi_v > 0 else None,
                           "specific_surface_1pm": sv})
                # in an insulating material the two transport paths add: the
                # conduction solved above plus the Rosseland radiative term is
                # what a hot-plate measurement of this structure would read
                if res_r.get("thermal") is not None and res_r["thermal"].get("tensor") is not None:
                    k_cond = float(np.nanmean(np.diag(np.asarray(res_r["thermal"]["tensor"], float))))
                    if np.isfinite(k_cond):
                        rr["k_conduction_W_mK"] = k_cond
                        rr["k_total_W_mK"] = k_cond + rr["k_rad_W_mK"]
                        rr["radiation_share"] = rr["k_rad_W_mK"] / max(k_cond + rr["k_rad_W_mK"], 1e-300)
                res_r["radiation"] = rr
                log(f"    extinction coefficient β = {beta:.4g} 1/m, Rosseland k_rad({ro['temperature_K']:g} K) = {rr['k_rad_W_mK']:.4g} W/m·K"
                    + (f", conduction + radiation = {rr['k_total_W_mK']:.4g} W/m·K" if rr.get("k_total_W_mK") else ""))
            except Exception as e:                                        # noqa: BLE001
                res_r["radiation"] = {"error": f"{type(e).__name__}: {e}"}
                log(f"    radiation analysis failed: {e}")
            prog.end()

        if first and "radiation" in need:
            live("radiation")
        if first and "morphology" in need:
            groups = _phase_groups(table, void_labels)
            prog.stage("Porosity and size distributions", "PoreSpy · NASA PuMA",
                       units=0.0 if any(("morph", g[0]) in pre_cond for g in groups) else 1.0)
            morph = {}
            for gkey, gname, ids in groups:
                if should_stop is not None and should_stop():
                    raise InterruptedError
                try:
                    pm_ = pre_cond.get(("morph", gkey))
                    if pm_ is not None:
                        mres, lt = pm_["res"], pm_.get("lt")
                    else:
                        mres, lt = MO.analyse_phase(labels, ids, h)
                        try:
                            area, sv = PB.surface_area(labels, ids, h_m)
                            mres["surface_area_m2"], mres["specific_surface_1pm"] = area, sv
                            mres["mean_intercept_length_um"] = [v * 1e6 for v in
                                                                PB.mean_intercept_length(labels, ids, h_m)]
                        except Exception as e:                       # noqa: BLE001
                            mres["surface_error"] = str(e)
                    if lt is not None and viewer is not None:
                        viewer.add_field(f"lt_{gkey}", lt * h, f"Local thickness ({gname})", "µm", "Structure")
                    morph[gkey] = {"name": gname, **mres}
                    sd = mres.get("size_distribution") or {}
                    log(f"    {gname}: volume fraction {100*mres['volume_fraction']:.2f} %, D50 {sd.get('d50_um', float('nan')):.3g} µm, "
                        f"{mres['clusters']['n_clusters']} clusters")
                except Exception as e:                               # noqa: BLE001
                    morph[gkey] = {"name": gname, "error": f"{type(e).__name__}: {e}"}
                    log(f"    {gname}: structure analysis failed ({e})")
            res_r["morphology"] = morph
            res_r["profiles"] = PO.fraction_profiles(labels, len(table), h)
            prog.end()

        if first and "morphology" in need:
            live("morphology")
        if first and "percolation" in need:
            from . import percolation as PC
            groups = [(k, n, ids, k != "pores") for k, n, ids in _phase_groups(table, void_labels)]
            prog.stage("Percolation paths", "connected clusters · geodesic shortest paths",
                       units=0.0 if any(("perc", g[0]) in pre_cond for g in groups) else 1.0)
            perc = {}
            for gkey, gname, ids, full in groups:
                if should_stop is not None and should_stop():
                    raise InterruptedError
                try:
                    pp_ = pre_cond.get(("perc", gkey))
                    if pp_ is not None:
                        m, pres, pfields = None, pp_["res"], pp_.get("fields") or {}
                    else:
                        m = np.isin(labels, ids)
                        # which particle can pass is a question about the pore
                        # space; a solid phase conducts, it does not let spheres through
                        pres, pfields = PC.analyse(m, h, dirs, full_connectivity=full,
                                                   want_fields=viewer is not None, want_paths=not full)
                    perc[gkey] = {"name": gname, **pres}
                    d0 = dirs[0]
                    if viewer is not None and not full:
                        for d in dirs:
                            rows = pres["by_direction"][d].get("passable_spheres") or []
                            top = next((r for r in rows if r.get("polyline_um")), None)
                            if not top:
                                continue
                            viewer.add_paths(f"path_{gkey}_{d}", [{"points_um": top["polyline_um"]}],
                                             f"Largest passing sphere, {top['diameter_um']:.3g} µm ({d.upper()})",
                                             "Percolation", kind="particle", diameter_um=top["diameter_um"],
                                             summary={k: top[k] for k in ("path_length_um", "tortuosity", "n_paths")},
                                             note="Route of the centre of the largest sphere that crosses the sample")
                    if viewer is not None:
                        for dd in dirs:
                            if f"class_{dd}" in pfields:
                                viewer.add_field(f"perc_{gkey}_{dd}", pfields[f"class_{dd}"],
                                                 f"Percolation class ({gname}, {dd.upper()}): 3 spanning, 2 dead end, 1 isolated",
                                                 "-", "Percolation", categorical=True)
                            if f"geodesic_{dd}" in pfields:
                                viewer.add_field(f"geo_{gkey}_{dd}", pfields[f"geodesic_{dd}"],
                                                 f"Geodesic path length from the {dd.upper()} inlet ({gname})", "µm", "Percolation")
                    del m, pfields
                    def _largest(v):
                        rows = v.get("passable_spheres") or []
                        return f", largest sphere {rows[0]['diameter_um']:.3g} µm" if rows else ""
                    txt = ", ".join(f"{d}: " + ("spanning" if v["percolates"] else "blocked")
                                    + (f", τ_geo {v['geodesic_tortuosity']:.3g}" if v.get("geodesic_tortuosity") else "")
                                    + _largest(v)
                                    for d, v in pres["by_direction"].items())
                    log(f"    percolation {gname} ({pres['connectivity']}-connectivity): {txt}")
                except Exception as e:                                # noqa: BLE001
                    perc[gkey] = {"name": gname, "error": f"{type(e).__name__}: {e}"}
                    log(f"    percolation analysis of {gname} failed: {e}")
            res_r["percolation"] = perc
            prog.end()

        if first and "percolation" in need:
            live("percolation")
        if first and "porosimetry" in need:
            po = opts["porosimetry"]
            pp_ = pre_cond.get(("poros", None))
            prog.stage("Porosimetry and capillary pressure", f"PoreSpy morphological drainage · {po['fluid']}",
                       units=0.0 if pp_ is not None else 1.0)
            try:
                if pp_ is not None:
                    pres, pfield = pp_["res"], pp_.get("field")
                else:
                    pres, pfield = PO.porosimetry(void_mask, h, po["surface_tension_N_m"], po["contact_angle_deg"],
                                                  po["steps"], dirs, log=log)
                pres["fluid"] = po["fluid"]
                if pfield is not None and viewer is not None:
                    viewer.add_field("mip_d", pfield, f"Pore-entry diameter ({po['fluid']} intrusion)", "µm", "Porosimetry")
                res_r["porosimetry"] = pres
            except Exception as e:                                        # noqa: BLE001
                res_r["porosimetry"] = {"error": f"{type(e).__name__}: {e}"}
                log(f"    porosimetry failed: {e}")
            prog.end()

        if first and "porosimetry" in need:
            live("porosimetry")
        if first and "pore_network" in need:
            pn_ = pre_cond.get(("network", None))
            prog.stage("Pore network extraction", "PoreSpy SNOW2", units=0.0 if pn_ is not None else 1.0)
            try:
                if pn_ is not None:
                    nres, nview = pn_["res"], pn_["view"]
                else:
                    nres, nview = PO.pore_network(void_mask, h, log=log)
                if viewer is not None:
                    viewer.add_network(nview)
                res_r["pore_network"] = nres
            except Exception as e:                                        # noqa: BLE001
                res_r["pore_network"] = {"error": f"{type(e).__name__}: {e}"}
                log(f"    pore network extraction failed: {e}")
            prog.end()

        if first and "pore_network" in need:
            live("pore_network")
        if first and "grains" in need:
            pg_ = pre_cond.get(("grains", None))
            prog.stage("Grain and filler analysis", "particle list · PoreSpy · NASA PuMA",
                       units=0.0 if pg_ is not None else 1.0)
            try:
                from . import grains as GR
                if pg_ is not None:
                    gres, glt = pg_["res"], pg_.get("lt")
                else:
                    gres, glt = GR.analyse(spec, ginfo, labels, table, h, log=log)
                if glt is not None and viewer is not None:
                    viewer.add_field("grain_matrix_lt", glt * h, "Matrix ligament thickness between particles",
                                     "µm", "Grains", in_matrix=True)
                res_r["grains"] = gres
            except Exception as e:                                        # noqa: BLE001
                res_r["grains"] = {"error": f"{type(e).__name__}: {e}"}
                log(f"    grain analysis failed: {e}")
            prog.end()

        if first and "grains" in need:
            live("grains")
        realisations.append(rinfo)
        if first:
            results_first = res_r
        del void_mask, open_mask, flow_labels
        if settle and not direct:
            # a statistical RVE: realisations until the 95 % confidence
            # interval of every mean is within the target (Kanit et al. 2003,
            # e_rel = 2 D_Z(V) / (Z sqrt(n)); here with Student's t)
            worst, which = settle_worst(samples)
            if worst is not None:
                log(f"   after {r + 1} realisations: the widest relative 95 % confidence interval of a mean is "
                    f"{100 * worst:.2f} % ({which}); target {settle['target_pct']:g} %")
            if r + 1 >= settle["n_min"] and worst is not None and worst <= settle["target_pct"] / 100.0:
                log(f"   the means are settled after {r + 1} realisations")
                break
    if settle and not direct:
        seeds = len(realisations)
        settled = settle_worst(samples)

    # ------------------------------------------------------------------ view
    meta = None
    if not direct:
        prog.stage("Visualisation data", "field values on the surfaces", units=1.5)
        try:
            viewer.sample_surfaces()
            meta = viewer.finish()
        except Exception as e:                                       # noqa: BLE001
            log(f"   !! the visualisation data could not be written: {e}")
            log(traceback.format_exc())
        prog.end()

    # ---------------------------------------------------------- aggregation
    props, composition, rho, cp = aggregate(results_first, realisations, checks)

    # --------------------------------------------------------------- layer
    layer = {"schema": "mpsim.layer/1", "name": spec["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "density_g_cm3": rho, "cp_J_kgK": cp, "porosity": composition["porosity"]}
    if plan.get("film"):
        # the tensors belong to this film as it stands (its surface layers
        # included), not to a bulk material that could be cut to any thickness
        layer["film"] = {"thickness_um": plan["T_um"], "lateral_um": plan["L_um"],
                         "conduction": "z between plates on the two faces; x and y periodic with insulated faces"}
    for key, tag in (("thermal", "k_W_mK"), ("electrical", "sigma_S_m"), ("dielectric", "eps_r"), ("magnetic", "mu_r")):
        if key in props:
            layer[tag] = props[key]["tensor"]
    if "dielectric" in props:
        layer["tan_d"] = props["dielectric"]["tan_d"]
    if "cte" in props:
        layer["C_GPa"] = props["cte"]["C"]
        layer["alpha_ppm_K"] = props["cte"]["alpha"]
    if "permeability" in props:
        layer["permeability_m2"] = props["permeability"]["tensor"]
    if "tortuosity" in props and props["tortuosity"].get("mean_d_eff"):
        layer["diffusivity_ratio"] = props["tortuosity"]["mean_d_eff"]
    if "acoustics" in props and "flow_resistivity_Pa_s_m2" in props["acoustics"]:
        a = props["acoustics"]
        layer["jca"] = {k: a[k] for k in ("porosity", "flow_resistivity_Pa_s_m2", "tortuosity",
                                          "viscous_length_um", "thermal_length_um")}
    if "radiation" in props and props["radiation"].get("beta_mean_1pm"):
        layer["extinction_1pm"] = props["radiation"]["beta_1pm"]
    if "emi" in props and props["emi"].get("complex"):
        cxi = props["emi"]["complex"]
        layer["admittivity_inplane"] = {"freq_hz": cxi["freq_hz"], "sigma_S_m": cxi["sigma_eff"],
                                        "eps_r": cxi["eps_eff"], "eps_resolved": cxi["eps_resolved"]}
    _dump(os.path.join(out_dir, "layer_card.json"), layer)

    # ------------------------------------------------------------- figures
    figures = {}
    done_figs = {}
    if live_box["fw"]:
        # the figures of the analyses that finished early are ready; the
        # last ones are waited for here instead of drawn a second time
        if meta:
            for item in figure_plan(meta):
                live_box["fw"].submit(item["name"], item["kind"], os.path.join(out_dir, "view"), item["req"],
                                      os.path.join(out_dir, "figures", item["name"]), item["title"], item["group"])
        prog.stage("Figures", "background renderer", units=1.5)
        done_figs = {k: v[0] for k, v in live_box["fw"].finish(should_stop=should_stop).items()}
        prog.end()
    if meta:
        prog.stage("Figures", "PyVista 3D · matplotlib sections", units=1.5)
        figures = default_figures(out_dir, meta, log, done=done_figs)
        prog.end()
    try:
        os.remove(os.path.join(out_dir, "result.partial.json"))
    except OSError:
        pass

    if prev is not None:
        for k, v in (prev.get("properties") or {}).items():
            if k not in props:
                props[k] = v
        ids = {c.get("id") for c in checks}
        checks += [c for c in (prev.get("checks") or []) if c.get("id") not in ids]
        wanted = wanted | kept
    headline = headline_numbers(props, composition)
    versions = backend_versions()
    if pool_box.get("pool") is not None:
        pool_box.pop("pool").close()
    summary = {
        "schema": "mpsim.result/2", "version": __version__, "name": spec["name"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t_start,
        "backend": backend, "versions": versions, "analyses": sorted(wanted),
        "spec": spec, "plan": plan, "realisations": realisations,
        "composition": composition, "properties": props, "checks": checks,
        "headline": headline, "layer_card": layer, "figures": figures,
        "view": bool(meta), "preview": bool(spec.get("preview")), "dem_direct": direct,
        "check_counts": {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "warn", "fail")},
    }
    _dump(os.path.join(out_dir, "result.json"), summary)
    try:
        from . import report
        report.write(out_dir, json.loads(json.dumps(_clean(summary))))
    except Exception as e:                                           # noqa: BLE001
        log(f"   (the report could not be written: {e})")
        log(traceback.format_exc())
    log(f"== completed in {time.time() - t_start:.1f} s · checks: {summary['check_counts']['pass']} passed, "
        f"{summary['check_counts']['warn']} warnings, {summary['check_counts']['fail']} failed")
    return _clean({k: summary[k] for k in ("name", "elapsed_s", "headline", "check_counts", "backend")})


def headline_numbers(props, comp):
    out = [{"key": "density", "label": "Density", "value": comp["density"], "unit": "g/cm³"}]
    if comp["porosity"] > 0:
        item = {"key": "porosity", "label": "Porosity", "value": 100 * comp["porosity"], "unit": "%"}
        if "porosity" in props:
            item["sub"] = f"open {100*props['porosity']['open']:.3g} % · closed {100*props['porosity']['closed']:.3g} %"
        out.append(item)
    if "thermal" in props:
        out.append({"key": "k", "label": "Thermal conductivity", "value": props["thermal"]["iso"], "unit": "W/m·K",
                    "diag": props["thermal"]["diag"], "fv": props["thermal"].get("fv_iso")})
    if "electrical" in props:
        out.append({"key": "sigma", "label": "Electrical conductivity", "value": props["electrical"]["iso"], "unit": "S/m",
                    "diag": props["electrical"]["diag"], "fv": props["electrical"].get("fv_iso")})
    if "dielectric" in props:
        out.append({"key": "eps", "label": "Dielectric constant Dk", "value": props["dielectric"]["iso"], "unit": "-",
                    "diag": props["dielectric"]["diag"], "fv": props["dielectric"].get("fv_iso")})
        out.append({"key": "tand", "label": "Dissipation factor Df", "value": props["dielectric"]["tan_d_iso"], "unit": "-"})
    if "magnetic" in props:
        out.append({"key": "mu", "label": "Relative permeability μr", "value": props["magnetic"]["iso"], "unit": "-"})
    if "cte" in props:
        out.append({"key": "alpha", "label": "CTE (volumetric mean)", "value": props["cte"]["alpha_vol"], "unit": "ppm/K",
                    "diag": list(props["cte"]["alpha"][:3])})
        out.append({"key": "E", "label": "Young's modulus (Hill)", "value": props["cte"]["constants"]["E_hill"], "unit": "GPa",
                    "diag": props["cte"]["constants"]["E"]})
    if "mechtest" in props:
        m = props["mechtest"]
        st = m["state"]
        out.append({"key": "E_test", "label": f"Apparent modulus ({st['mode']} along {st['axis']}, "
                                              f"{'confined' if st['lateral'] == 'confined' else 'free sides'})",
                    "value": m["modulus_mean_GPa"], "unit": "GPa"})
        if m.get("onset"):
            out.append({"key": "F_onset", "label": "Load at the matrix strength (0.1 % of the matrix)",
                        "value": m["onset"]["force_N"], "unit": "N"})
    if "thermomech" in props:
        t_ = props["thermomech"]
        out.append({"key": "warp", "label": f"Warpage over {t_['span_mm']:g} mm (centre to edge)",
                    "value": t_["warpage_um"], "unit": "µm",
                    "sub": f"κ {t_['kappa_1_m']:.3g} 1/m · {'edges up' if t_['warpage_um'] < 0 else 'edges down'}"})
        vm_ = [v["vm_p999"] for v in (t_.get("stress_stats") or {}).values()]
        if vm_:
            out.append({"key": "vm_tm", "label": "Thermal stress, von Mises (99.9 %)", "value": max(vm_), "unit": "MPa"})
    if "joule" in props and props["joule"].get("steps"):
        j_ = props["joule"]
        ok_ = [x for x in j_["steps"] if x["status"] == "converged"] or j_["steps"]
        last = ok_[-1]
        out.append({"key": "T_joule", "label": f"Peak temperature at {last['J_A_m2']:.3g} A/m²", "value": last["T_max"],
                    "unit": "°C", "sub": f"{last['V']:.3g} V · {last['P_W_m2']:.3g} W/m²"
                                         + (" · the sweep stopped beyond" if j_.get("stopped") else "")})
        if last.get("sigma_eff"):
            out.append({"key": "rho_joule", "label": "Resistivity through the film at that step",
                        "value": 100.0 / last["sigma_eff"], "unit": "Ω·cm"})
    if "emi" in props:
        e = props["emi"]
        f = np.asarray(e["spectrum"]["freq_hz"])
        fm = 1e9 if f[0] <= 1e9 <= f[-1] else float(np.sqrt(f[0] * f[-1]))
        se = float(np.interp(math.log10(fm), np.log10(f), e["spectrum"]["se_db"]))
        out.append({"key": "se", "label": f"Shielding effectiveness at {fm/1e9:.3g} GHz", "value": se, "unit": "dB"})
    if "moisture" in props:
        m = props["moisture"]
        out.append({"key": "D_w", "label": "Moisture diffusivity (z)", "value": m["D_z"], "unit": "m²/s"})
        out.append({"key": "c_w", "label": "Saturated moisture uptake", "value": m["wt_pct"], "unit": "wt%"})
        if m.get("swelling") and m["swelling"].get("cme_eff"):
            out.append({"key": "cme", "label": "Moisture expansion CME", "value": float(np.mean(m["swelling"]["cme_eff"])),
                        "unit": "1/mass fr."})
    if "viscosity" in props:
        out.append({"key": "mu_r", "label": "Relative viscosity μ/μresin", "value": props["viscosity"]["mu_r"], "unit": "-"})
        out.append({"key": "mu_c", "label": f"Compound viscosity at {props['viscosity']['gd_ref']:g} 1/s",
                    "value": props["viscosity"]["mu_compound_ref"], "unit": "Pa·s"})
    if "permeability" in props:
        out.append({"key": "K", "label": "Permeability", "value": float(np.mean(props["permeability"]["diag"])), "unit": "m²",
                    "diag": list(props["permeability"]["diag"])})
    if "tortuosity" in props and props["tortuosity"].get("mean_tortuosity"):
        out.append({"key": "tau", "label": "Tortuosity factor", "value": props["tortuosity"]["mean_tortuosity"], "unit": "-"})
    if "acoustics" in props and props["acoustics"].get("nrc") is not None:
        out.append({"key": "nrc", "label": f"Noise reduction coefficient ({props['acoustics']['thickness_mm']:g} mm)",
                    "value": props["acoustics"]["nrc"], "unit": "-"})
    if "radiation" in props and props["radiation"].get("beta_mean_1pm"):
        rr = props["radiation"]
        out.append({"key": "beta", "label": "Extinction coefficient", "value": rr["beta_mean_1pm"], "unit": "1/m",
                    "sub": f"k_rad {rr['k_rad_W_mK']:.3g} W/m·K at {rr['temperature_K']:g} K"})
    if "porosimetry" in props and props["porosimetry"].get("d50_um"):
        out.append({"key": "mip", "label": "Median pore-entry diameter", "value": props["porosimetry"]["d50_um"], "unit": "µm"})
        tp = [v["diameter_um"] for v in (props["porosimetry"].get("through_pore") or {}).values() if v]
        if tp:
            out.append({"key": "bp", "label": "Largest through-pore diameter", "value": max(tp), "unit": "µm"})
    if "filtration" in props and props["filtration"].get("mpps_efficiency") is not None:
        fl = props["filtration"]
        # the efficiency at the most penetrating size is how a filter is
        # specified, so it is the headline a DOE or an optimisation targets
        out.append({"key": "eff_mpps", "label": "Efficiency at the most penetrating size",
                    "value": 100.0 * fl["mpps_efficiency"], "unit": "%",
                    "sub": f"{fl['mpps_um']:.3g} µm"
                           + (f" · Δp {fl['pressure_drop_Pa']:.3g} Pa" if fl.get("pressure_drop_Pa") else "")})
    if "percolation" in props:
        pores = props["percolation"].get("pores") or {}
        taus = [v.get("geodesic_tortuosity") for v in (pores.get("by_direction") or {}).values() if v.get("geodesic_tortuosity")]
        if taus:
            out.append({"key": "taugeo", "label": "Geodesic tortuosity (pores)", "value": float(np.mean(taus)), "unit": "-"})
    if "pore_network" in props and props["pore_network"].get("n_pores"):
        out.append({"key": "Z", "label": "Mean coordination number", "value": props["pore_network"]["coordination_mean"],
                    "unit": "-", "sub": f"{props['pore_network']['n_pores']} pores · {props['pore_network']['n_throats']} throats"})
    if "grains" in props and props["grains"].get("phases"):
        n = sum(p.get("n_particles", 0) for p in props["grains"]["phases"].values())
        out.append({"key": "np", "label": "Particles in the RVE", "value": n, "unit": ""})
    # Per-direction values, whenever more than one direction was solved. They
    # are what a DOE or an optimisation has to aim at for an anisotropic
    # structure: a substrate is specified by its through-thickness Dk, not by an
    # average over directions the signal never sees. Flagged `dir` so the
    # result tiles, which already show the average, do not repeat them.
    for key, prop, label, unit in (("k", "thermal", "Thermal conductivity", "W/m·K"),
                                   ("eps", "dielectric", "Dk", "-")):
        if prop not in props:
            continue
        # a direction that was not solved is NaN here (None only after the JSON
        # clean-up), so both are filtered
        solved = [(d, v) for d, v in zip("xyz", props[prop].get("diag") or [])
                  if v is not None and math.isfinite(v)]
        if len(solved) < 2:
            continue
        for d, v in solved:
            out.append({"key": f"{key}_{d}", "label": f"{label} ({d})", "value": v, "unit": unit, "dir": True})
        if prop == "dielectric":
            td = props[prop].get("tan_d") or {}
            for d, _ in solved:
                if td.get(d) is not None:
                    out.append({"key": f"tand_{d}", "label": f"Df ({d})", "value": td[d], "unit": "-", "dir": True})
    return out


def backend_versions():
    """The versions behind a result, from the installed packages' metadata:
    importing PoreSpy, OpenPNM, PyVista and PuMA only to read them took about
    4.5 s of every run (5.5)."""
    from importlib.metadata import version as _v
    out = {"mpsim": __version__}
    for key, dist in (("puma", "pumapy"), ("numpy", "numpy"), ("scipy", "scipy"), ("pyvista", "pyvista"),
                      ("vtk", "vtk"), ("porespy", "porespy"), ("openpnm", "openpnm"), ("skrf", "scikit-rf"),
                      ("numba", "numba")):
        try:
            out[key] = _v(dist)
        except Exception:                                            # noqa: BLE001
            out[key] = None
    return out


def figure_plan(meta):
    """The report figures a view calls for, in the order they are drawn:
    [{name, kind ("scene" | "slice"), req, title, group}]. The live figure
    worker takes the ones of each analysis as soon as it finishes; the end of
    the run draws whatever is still missing."""
    mid = {a: meta["shape"][i] // 2 for i, a in enumerate("xyz")}
    plan = [{"name": "structure_3d.png", "kind": "scene", "group": "Structure", "title": "Structure (3D)",
             "req": {"mode": "structure", "width": 1400, "height": 1100, "title": "Structure (3D)"}},
            {"name": "structure_slice_z.png", "kind": "slice", "group": "Structure", "title": "Structure section (z)",
             "req": {"axis": "z", "index": mid["z"], "title": "Structure section (z)"}}]
    if meta.get("network"):
        plan.append({"name": "pore_network_3d.png", "kind": "scene", "group": "Pore network",
                     "title": "Extracted pore network",
                     "req": {"mode": "network", "show_particles": False, "width": 1400, "height": 1100,
                             "title": "Extracted pore network"}})
    seen = set()
    for f in meta["fields"]:
        grp = f["group"]
        if grp in seen:
            continue
        seen.add(grp)
        k = f["key"]
        plan.append({"name": f"{k}_3d.png", "kind": "scene", "group": grp, "title": f["label"],
                     "req": {"mode": "field", "field": k, "slices": {}, "box_faces": True, "show_particles": False,
                             "log": f["log"], "width": 1400, "height": 1100, "title": f["label"]}})
        plan.append({"name": f"{k}_slice_z.png", "kind": "slice", "group": grp, "title": f["label"] + " — section",
                     "req": {"axis": "z", "index": mid["z"], "field": k, "log": f["log"],
                             "title": f["label"] + " — section"}})
    # one path figure per group: a field line or a passing particle says more
    # about the transport than another contour of the same quantity
    seen_paths = set()
    for pth in meta.get("paths", []):
        if pth["group"] in seen_paths:
            continue
        seen_paths.add(pth["group"])
        plan.append({"name": f"{pth['key']}_paths.png", "kind": "scene", "group": pth["group"], "title": pth["label"],
                     "req": {"mode": "structure", "show_particles": True, "particle_opacity": 0.16,
                             "paths": pth["key"], "width": 1400, "height": 1100, "title": pth["label"]}})
    return plan


def default_figures(out_dir, meta, log, done=None):
    """Draws the figures of figure_plan; `done` names the ones the live
    worker has already written."""
    view = os.path.join(out_dir, "view")
    figdir = os.path.join(out_dir, "figures")
    figs = {}
    done = done or {}
    for item in figure_plan(meta):
        name = item["name"]
        path = os.path.join(figdir, name)
        if done.get(name) and os.path.exists(path):
            figs[name] = item["title"]
            continue
        try:
            (V.slice_figure if item["kind"] == "slice" else V.render_scene)(view, item["req"], path)
            figs[name] = item["title"]
        except Exception as e:                                       # noqa: BLE001
            log(f"   (figure {name} failed: {e})")
    return figs
