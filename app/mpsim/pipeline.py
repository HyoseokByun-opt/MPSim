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
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(obj), fh, ensure_ascii=False, indent=1)


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
    return out


# ----------------------------------------------------------- kept structures
# Raised whenever the generator would build a different structure from the
# same inputs, so that a structure kept by an older version is not reused.
# 2: packing budgets in steps instead of seconds (large boxes reached the
#    target size; before, they could stop short of it on the clock)
# 3: no particle overlaps or is trimmed (v5.0.1): the clump repair pushes and
#    turns, shrinks what it cannot free; sphere packings are separated exactly
STRUCTURE_VERSION = 3


def structure_key(gen_phases, plan, contacts, seed, want_cf):
    """What identifies one generated realisation: the generator's own inputs
    (phase names aside), the grid, the contact handling and the seed."""
    geo = [{k: v for k, v in ph.items() if k != "name"} for ph in gen_phases]
    key = {"v": STRUCTURE_VERSION, "phases": geo, "N": int(plan["N"]),
           "h": float(f"{plan['h_um']:.12g}"), "nz": int(plan["Nz"]) if plan.get("film") else None,
           "contacts": contacts, "seed": int(seed), "cf": bool(want_cf)}
    if plan.get("skin"):
        key["skin"] = int(plan["skin"])          # only a film with a skin: other keys stay as they were
    blob = json.dumps(key, sort_keys=True, default=float)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


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
    with open(os.path.join(d, key + ".pkl"), "wb") as fh:
        pickle.dump({k: v for k, v in ginfo.items() if k != "contact_mask"}, fh, protocol=4)


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
        v_used, policy = CD.contrast_policy(labels, P / P.max(), di, opts["contrast_cap"], cache)
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

def _viscosity_entry(spec, opts, table, realisations, results_first, mus, checks):
    """The viscosity section of the result: the RVE value, the closed forms,
    Krieger-Dougherty with the jammed phi_m, the best estimate, the flow curve
    of the compound, underfill filling and settling."""
    vo = opts["viscosity"]
    v0 = results_first["viscosity"]
    vx = results_first.get("viscosity_extra") or {}
    vf = realisations[0]["vf_labels"]
    phi = float(sum(v for v, t in zip(vf, table) if t["kind"] in ("core", "shell") and float(t["props"]["E"]) > 0))
    bubbles = float(sum(v for v, t in zip(vf, table) if float(t["props"]["E"]) <= 0 and t["kind"] != "matrix"))
    mu_r = float(np.mean(mus)) if mus else v0["mu_r"]
    ci = None
    if len(mus) >= 2:
        _, _, c = _mean_ci([[m] for m in mus])
        ci = float(c[0])
    phi_m = float(vx.get("phi_m", vo["phi_m"]))
    eta = float(vx.get("eta", 2.5))
    kd = float(VI.krieger_dougherty(phi, phi_m, eta))
    if phi >= 0.97 * phi_m:
        checks.append({"id": "visc_phim", "group": "Viscosity", "label": "Filler fraction below the maximum packing fraction",
                       "value": phi, "target": phi_m, "unit": "", "status": "fail",
                       "note": "At or above φm the compound does not flow (jammed paste); Krieger-Dougherty diverges"})
    # the same curve for frictional (or frictionless) particles, as a range
    phi_m_alt = (float(vx.get("phi_m_rcp", phi_m)) if vx.get("friction") else phi_m * VI.FRICTION_FACTOR)
    kd_alt = float(VI.krieger_dougherty(phi, phi_m_alt, eta)) if phi < 0.995 * phi_m_alt else None
    mp = float(VI.maron_pierce(phi, phi_m))
    touching = int(v0.get("touching_pairs") or 0)
    resolved = phi <= 0.35 and touching == 0
    best, basis = (mu_r, "RVE flow solve") if resolved else (kd, "Krieger-Dougherty with the jammed packing fraction")
    phis = np.linspace(0.0, 0.97 * phi_m, 60)
    curve = {"phi": phis.tolist(), "kd": VI.krieger_dougherty(phis, phi_m, eta).tolist(),
             "kd_alt": [float(VI.krieger_dougherty(p, phi_m_alt, eta)) if p < 0.97 * phi_m_alt else None for p in phis],
             "mp": VI.maron_pierce(phis, phi_m).tolist(), "hs": [VI.hs_lower(p) for p in phis],
             "batchelor": [VI.batchelor(p) for p in phis], "einstein": [VI.einstein(p) for p in phis]}
    gd = np.logspace(math.log10(vo["gd_min"]), math.log10(vo["gd_max"]), 40)
    mu_m = VI.matrix_flow(vo["model"], gd)
    mu_s, A = VI.suspension_flow(vo["model"], best, phi, gd, vo["yield_Pa"])
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
             "seeds": mus, "shear": v0["shear"], "extension": v0.get("extension"), "stats": v0["stats"],
             "touching_pairs": touching, "phi_m": phi_m, "phi_m_basis": vx.get("phi_m_basis", "entered"),
             "phi_m_rcp": vx.get("phi_m_rcp", phi_m), "friction": bool(vx.get("friction")),
             "phi_m_alt": phi_m_alt, "kd_alt": kd_alt, "farr_groot": vx.get("farr_groot"),
             "desmond_weeks": vx.get("desmond_weeks"),
             "jamming": vx.get("jamming"), "eta": eta, "eta_basis": vx.get("eta_basis", "sphere value"),
             "dilute": vx.get("dilute"),
             "refs": {"Einstein": VI.einstein(phi), "Batchelor": VI.batchelor(phi), "Hashin-Shtrikman lower bound": VI.hs_lower(phi),
                      "Krieger-Dougherty": kd, "Maron-Pierce": mp},
             "kd_phi_m_fit": VI.kd_phi_m_from(mu_r, phi, eta), "mu_r": best, "basis": basis, "curve": curve,
             "flow": {"gd": gd.tolist(), "mu_resin": mu_m.tolist(), "mu_compound": mu_s.tolist(),
                      "amplification": A, "model": vo["model"], "yield_Pa": vo["yield_Pa"]},
             "gd_ref": vo["gd_ref"], "mu_resin_ref": mu_m_ref, "mu_compound_ref": mu_s_ref,
             "underfill": dict(uf, t_resin_s=t_resin, t_compound_s=t_comp),
             "settling": settle, "settle_min": vo["settle_min"]}
    hs = VI.hs_lower(phi)
    checks.append({"id": "visc_hs", "group": "Viscosity", "label": "Relative viscosity above the Hashin-Shtrikman lower bound",
                   "value": mu_r, "target": hs, "unit": "", "status": "pass" if mu_r >= 0.99 * hs else "warn",
                   "note": "Rigid particles always raise the viscosity at least this much (isotropic suspension)"})
    checks.append({"id": "visc_contacts", "group": "Viscosity", "label": "Particles in voxel contact in the flow solve",
                   "value": touching, "target": 0, "unit": "pairs", "status": "pass" if touching == 0 else "warn",
                   "note": ("No resin film is left between touching voxels: they move as one rigid body and the RVE "
                            "value is too high; the best estimate uses Krieger-Dougherty" if touching else
                            "Every particle is surrounded by resin on the grid")})
    if phi > 0.35:
        checks.append({"id": "visc_dense", "group": "Viscosity", "label": "Filler fraction within the range the voxel flow solve resolves",
                       "value": phi, "target": 0.35, "unit": "", "status": "warn",
                       "note": "Above about 35 vol% the resin films between particles are thinner than a voxel or two; "
                               "the best estimate is Krieger-Dougherty with the jammed packing fraction"})
    if bubbles > 0:
        checks.append({"id": "visc_bubbles", "group": "Viscosity", "label": "Bubbles treated as freely deforming",
                       "value": bubbles, "target": None, "unit": "", "status": "warn",
                       "note": "The flow solve has no surface tension: a bubble deforms freely (high capillary number) and "
                               "lowers the viscosity as 1 - 5/3 phi (dilute). Small bubbles kept spherical by surface "
                               "tension (low capillary number, Taylor) raise it as 1 + phi instead"})
    conv = [s.get("converged") for s in v0["stats"].values()]
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
    key = structure_key(gen_phases, plan, spec["options"]["contacts"], spec["rve"]["seed"],
                        wants_contact_faces(spec))
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
            rpair[0, outer[i]] = rpair[outer[i], 0] = r
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
                viewer, first, rpair=None, cmask=None, rcpair=None, pre=None):
    info = COND[key]
    iface = rpair is not None or rcpair is not None
    backend = cond_backend(spec, backend)
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
        v_used, policy = CD.contrast_policy(labels, vals, di, cap, cache)
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
                r = CD.solve_direction(labels, v_used, di, tol=tol, rpair=rpair, cmask=cmask, rcpair=rcpair,
                                       callback=lambda it, res: cb(d, it, res, tol, None),
                                       should_stop=should_stop, flux_h=(h_um * 1e-6) if first else None,
                                       film=bool(spec["rve"].get("film")))
                r["method"] = "fv-film" if spec["rve"].get("film") else ("fv-interface" if iface else "fv-periodic")
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
        else:
            vals = np.array(S.label_values(spec["labels"], COND[key]["prop"]), float)
        v_used, _ = CD.contrast_policy(labels_f, vals, di, spec["options"]["contrast_cap"], {})
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
              log, should_stop, log_dir, event=None, flow=None, prog=None, struct=None, rcheck=None):
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
        threads = int(os.environ.get("NUMBA_NUM_THREADS") or 0)
    except ValueError:
        threads = 0
    threads = threads or max(1, (os.cpu_count() or 4) // 4)
    dirs = opts["directions"]
    n_vox = int(labels.size)
    mem_est = R.memory_estimate(labels.shape[0], set(need) | {"thermal"}, nz=labels.shape[2])
    # memory of one solve of each kind, GB: rve.memory_estimate per voxel,
    # Stokes from a measurement (PuMA's matrix-free solve held 0.64 GB on an
    # 80^3 RVE, ~1.25 kB per voxel; the planner's 2.3 kB is its upper bound)
    mem_of = {"fv": mem_est.get("conduction", n_vox * 230 / 1e9), "fe": mem_est.get("conduction", n_vox * 230 / 1e9),
              "elastic": n_vox * (3 * 24 * 8 * 4 + 240) / 1e9, "stokes": n_vox * 1300 / 1e9,
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
        be = "fv" if (rpair is not None or rcpair is not None) else cond_backend(spec, backend)
        film_run = bool(spec["rve"].get("film"))
        cache = {}
        for d in dirs:
            v_used, _ = CD.contrast_policy(labels, vals, "xyz".index(d), opts["contrast_cap"], cache)
            add(PAR.cond_task, "fe" if "fe" in be.split("+") else "fv",
                {"key": key, "d": d, "kind": COND[key]["kind"], "backend": be,
                 "name": f"{COND[key]['title']} {d}",
                 "vals": np.asarray(v_used, float), "voxel_m": h * 1e-6, "tol": opts["tol"],
                 "fields": bool(first), "rpair": rpair, "rcpair": rcpair,
                 "cmask": "use" if (rcpair is not None and contact_mask is not None) else None,
                 "film": film_run})
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
                v_used, _ = CD.contrast_policy(labels, P / P.max(), "xyz".index(d), opts["contrast_cap"], cache)
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
        for i, fq in enumerate(freqs):
            for d in (0, 1):
                kap, pol = CC.cap_contrast(labels, CC.admittivity(table, fq), d, cache=cache)
                emi_pol[(i, d)] = pol
                add(PAR.emi_task, "emi", {"i": i, "d": d, "kap": kap, "bc": ebc, "tol": min(opts["tol"], 1e-7),
                                          "name": f"EMI {fq:.3g} Hz {'xyz'[d]}"}, units=1.0 / (2 * len(freqs)))
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
                if rcpair is not None:
                    continue
            else:
                vals = np.array(S.label_values(spec["labels"], COND[key]["prop"]), float)
            v_used, _ = CD.contrast_policy(fine["labels"], vals, "xyz".index(d0), opts["contrast_cap"], {})
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
        pool = box["pool"] = PAR.SolvePool(workers, th_each, log_dir)
    lab_path = pool.share(labels)
    flow_path = pool.share(flow["labels"], "flow") if any(t.get("_flow") for _, t in tasks) else None
    fine_path = pool.share(rcheck["fine"]["labels"], "fine") if any(t.get("_fine") for _, t in tasks) else None
    cm_path = pool.share(contact_mask.astype(np.uint8), "cmask") if contact_mask is not None else None
    for _, t in tasks:
        t["labels"] = flow_path if t.get("_flow") else fine_path if t.get("_fine") else lab_path
        t["tmp"] = pool.tmp
        if "cmask" in t:
            t["cmask"] = cm_path if t.get("cmask") == "use" else None
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
    return all(ph["shape"] in ("sphere", G.NETWORK) or ph["orientation"]["mode"] == "iso"
               for ph in spec["phases"])


def _cond_refs(key, spec, vf_labels, vals):
    table = spec["labels"]
    phi = np.asarray(vf_labels, float)
    lo, hi = AN.wiener(phi, vals)
    refs = {"Wiener bounds": [lo, hi]}
    if _isotropic_structure(spec):
        refs["Hashin–Shtrikman"] = list(AN.hashin_shtrikman(phi, vals))
    simple = all(t["kind"] in ("matrix", "core") for t in table)
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


def _cond_checks(key, tensor, refs, spec, stats):
    title = COND[key]["title"]
    out = []
    lo, hi = refs["Wiener bounds"]
    for i, d in enumerate("xyz"):
        v = tensor[i, i]
        if not np.isfinite(v):
            continue
        ok = lo * (1 - 1e-3) <= v <= hi * (1 + 1e-3)
        out.append({"id": f"{key}_wiener_{d}", "group": title, "label": f"{d}{d} component within the Wiener bounds",
                    "value": v, "target": [lo, hi], "unit": COND[key]["unit"],
                    "status": "pass" if ok else "fail",
                    "note": "Rigorous bounds that every microstructure satisfies"})
    if "Hashin–Shtrikman" in refs:
        hlo, hhi = refs["Hashin–Shtrikman"]
        diag = [tensor[i, i] for i in range(3) if np.isfinite(tensor[i, i])]
        if diag:
            m = float(np.mean(diag))
            ok = hlo * (1 - 0.01) <= m <= hhi * (1 + 0.01)
            out.append({"id": f"{key}_hs", "group": title, "label": "Isotropic mean within the Hashin–Shtrikman bounds",
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
def run(form, out_dir, log=print, event=None, should_stop=None, limits=None):
    t_start = time.time()
    event = event or (lambda kind, **p: None)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
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
    if backend in ("puma", "puma_fv") and not PB.available():
        log("!! NASA PuMA could not be imported; the built-in verification solvers are used instead: "
            + str(PB.unavailable_reason()))
        backend = "builtin"
    plan = R.plan(_plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"]), need, limits)
    log(f"== {spec['name']}")
    if prev is not None:
        log(f"   Continuing this run: {', '.join(sorted(kept))} kept, {', '.join(sorted(need)) or 'nothing'} added.")
    if plan.get("film"):
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
    dirs = opts["directions"]
    per_seed_units = 1.0 + sum(len(dirs) for k in COND if k in need) \
        + (7.0 if "cte" in need else 0.0) \
        + (2.0 * len(dirs) if "permeability" in need else 0.0) \
        + (0.5 * len(dirs) if "tortuosity" in need else 0.0) \
        + (0.5 * len(dirs) + 1.0 if "moisture" in need else 0.0) \
        + (3.0 if ("cte" in need and opts.get("above_tg")) else 0.0)
    once_units = sum(u for k, u in (("morphology", 1.0), ("porosimetry", 1.0), ("pore_network", 1.0),
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
        if "viscosity" in need and "viscosity" in results_first:
            props["viscosity"] = _viscosity_entry(spec, opts, table, realisations, results_first,
                                                  samples.get("viscosity") or [], checks)
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
                    checks.append({"id": f"{key}_stat", "group": "Statistics", "label": f"{e['title']}: relative 95 % confidence interval",
                                   "value": worst, "target": 0.02, "unit": "rel",
                                   "status": "pass" if worst <= 0.02 else ("warn" if worst <= 0.05 else "fail"),
                                   "note": f"{seeds} independent realisations"})
        return props, composition, rho, cp

    for r in range(seeds):
        first = r == 0
        seed = spec["rve"]["seed"] + 1000 * r
        prog.stage("Structure generation", f"realisation {r+1}/{seeds} · {_grid(N, Nz)} voxels", units=1.0)
        # the contact faces are kept whenever a contact resistance is set, not
        # only when this run solves heat: the structure stays the same one for
        # every selection of analyses
        want_cf = wants_contact_faces(spec)
        skey = structure_key(gen_phases, plan, opts["contacts"], seed, want_cf)
        got = load_structure([out_dir] + list(reuse.get("from") or []), skey) if opts.get("reuse_runs", True) else None
        if got is not None:
            gen_labels, ginfo, src = got
            log(f"   realisation {r+1}: the structure generated earlier is used "
                f"({'this run' if os.path.abspath(src) == os.path.abspath(out_dir) else os.path.basename(src)}); "
                f"no new generation")
            if os.path.abspath(src) != os.path.abspath(out_dir):
                save_structure(out_dir, skey, gen_labels, ginfo, ginfo.get("contact_mask"))
        else:
            gen_labels, ginfo = GEN.generate(N, h, gen_phases, seed=seed, log=log, should_stop=should_stop,
                                             contacts=opts["contacts"], want_contact_faces=want_cf, nz=Nz,
                                             skin=skin)
            try:
                save_structure(out_dir, skey, gen_labels, ginfo, ginfo.get("contact_mask"))
            except Exception as e:                                   # noqa: BLE001
                log(f"   (the structure could not be kept for later runs: {e})")
        labels, area_ratio, notes = build_labels(spec, gen_labels, ginfo, h)
        contact_mask = ginfo.pop("contact_mask", None)
        pair_faces = contact_pair_counts(gen_labels, contact_mask, len(spec["phases"]))
        del gen_labels
        for n in notes:
            log("   · " + n)
        counts = np.bincount(labels.ravel(), minlength=len(table))
        vf_labels = counts / labels.size
        rinfo = {"seed": seed, "route": ginfo["route"], "seconds": ginfo["seconds"],
                 "contact_pairs": pair_faces,
                 "contacts_mode": ginfo.get("contacts_mode"), "contacts_removed": ginfo["contacts_removed"],
                 "contact_faces": ginfo.get("contact_faces", 0), "touching_pairs": ginfo.get("touching_pairs", 0),
                 "voxel_conflicts": ginfo.get("voxel_conflicts", 0),
                 "shrunk": ginfo.get("shrunk"),
                 "pack_scale": ginfo.get("pack_scale"),
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
        rc_mode = opts.get("resolution_check", "auto")
        if first and any(k in need for k in COND) and (
                rc_mode == "on" or (rc_mode == "auto" and ginfo.get("touching_pairs", 0) > 0)):
            try:
                rc_fine = _fine_grid(spec, ginfo, gen_phases, N, h,
                                     int((limits or {}).get("N_max", R.DEFAULT_LIMITS["N_max"]) or 0))
            except Exception as e:                                       # noqa: BLE001
                rc_fine = f"the finer grid could not be drawn: {e}"

        # ---- independent solves, side by side ----------------------------
        pre_cond, pre_el = _prefetch(pool_box, need, labels, table, spec, h, backend, gas_factor, area_ratio,
                                     contact_mask, first, log, should_stop, os.path.join(out_dir, "logs"),
                                     event=event,
                                     flow={"labels": flow_labels, "void": void_labels, "open": pore_open}
                                     if void_labels else None,
                                     prog=prog, struct={"ginfo": ginfo, "void": void_labels} if first else None,
                                     rcheck={"fine": rc_fine} if isinstance(rc_fine, dict) else None)

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
                              cmask=contact_mask if rcpair is not None else None, rcpair=rcpair, pre=pre_cond)
            res_r[key] = out
            samples.setdefault(key, []).append(np.diag(out["tensor"]))
            if first:
                out["refs"] = _cond_refs(key, spec, vf_labels, vals)
                out["values_used"] = vals.tolist()
                checks += _cond_checks(key, out["tensor"], out["refs"], spec, out["stats"])
            event("partial", key=key, value=_clean(np.diag(out["tensor"])))
            live(key)

        # ---- frequency-resolved admittivity for EMI (first realisation) ----
        if first and "emi" in need and opts["emi"].get("method", "complex") == "complex":
            eo = opts["emi"]
            freqs = np.logspace(math.log10(eo["f_min_hz"]), math.log10(eo["f_max_hz"]), int(eo["n_freq"]))
            # both in-plane directions, as the static sigma and eps: one
            # realisation of an "isotropic" structure can still differ 1.6x
            # between x and y near percolation
            edirs = (0, 1)
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
                prog.stage("EMI · full-wave check", "openEMS FDTD of the RVE slab, 10-100 GHz", units=1.0)
                try:
                    res_r["emi_fullwave"] = FW.check(labels, table, h, os.path.join(out_dir, "fullwave"), log=log,
                                                     should_stop=should_stop)
                    log(f"    full-wave check: largest SE difference {res_r['emi_fullwave']['max_diff_db']:.2f} dB "
                        f"({res_r['emi_fullwave']['seconds']:.0f} s)")
                except InterruptedError:
                    raise
                except Exception as e:                               # noqa: BLE001
                    res_r["emi_fullwave"] = {"error": str(e)}
                    log(f"    full-wave check not run: {e}")
                prog.end()

        if "emi_complex" in res_r:
            live("emi")

        # ---- resolution check (first realisation) ------------------------
        rc_mode = opts.get("resolution_check", "auto")
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

        # ---- viscosity of the uncured compound ---------------------------
        if "viscosity" in need:
            vo = opts["viscosity"]
            vtol = 1e-5 if spec["rve"]["quality"] == "fast" else 1e-6
            prog.stage("Viscosity · creeping flow of the resin around the fillers",
                       "voxel FE (B-bar) · FFT-preconditioned CG · 3 shear planes + extension", units=4.0)
            vnames = ["yz", "xz", "xy", "extension"]

            def cbv(plane, it, res):
                k = vnames.index(plane)
                event("solver", name=f"Viscosity, shear {plane}" if plane != "extension" else "Viscosity, extension",
                      it=it, res=res, target=vtol, progress=prog.sub(min(k + 0.5, 3.99) / 4.0))
            try:
                import numba
                vworkers = int(numba.get_num_threads())
            except Exception:                                            # noqa: BLE001
                vworkers = -1
            vr = VI.effective_viscosity(labels, VI.phase_viscosities(table), tol=vtol, progress=cbv,
                                        should_stop=should_stop, workers=vworkers, keep_field=first)
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
                solid = [i for i, ph in enumerate(spec["phases"]) if not ph["void"] and ph["shape"] != G.NETWORK]
                if solid:
                    prog.stage("Viscosity · maximum packing fraction", "random close packing of these particles", units=1.0)
                    sp_solid = [dict(gen_phases[i], void=False) for i in solid]          # outer sizes
                    vx.update(max_packing(sp_solid, vo, log, should_stop))
                    prog.end()
                else:
                    vx["phi_m"] = vo["phi_m"]
                etas = [VI.intrinsic_viscosity(spec["phases"][i]["shape"],
                                               G.aspect_ratio(spec["phases"][i]["shape"], spec["phases"][i]["size_um"]))
                        for i in solid]
                if solid and all(e is not None for e in etas):
                    w = np.array([spec["phases"][i]["vf"] for i in solid], float)
                    vx["eta"], vx["eta_basis"] = float(np.dot(w, etas) / w.sum()), "closed form"
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

        # ---- moisture ------------------------------------------------------
        if "moisture" in need:
            res_r["moisture"] = _moisture_solve(labels, table, spec, h, vf_labels, prog, log, event, should_stop,
                                                viewer if first else None, first,
                                                C_known=(res_r.get("cte") or {}).get("C"), pre=pre_cond)
            event("partial", key="moisture", value=_clean([res_r["moisture"]["wt_pct"]]))
            live("moisture")

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

    # ------------------------------------------------------------------ view
    prog.stage("Visualisation data", "field values on the surfaces", units=1.5)
    meta = None
    try:
        viewer.sample_surfaces()
        meta = viewer.finish()
    except Exception as e:                                           # noqa: BLE001
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
        "view": bool(meta), "preview": bool(spec.get("preview")),
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
    out = {"mpsim": __version__, "puma": PB.version()}
    for mod in ("numpy", "scipy", "pyvista", "vtk", "porespy", "openpnm", "skrf", "numba"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if mod == "vtk":
                v = m.vtkVersion.GetVTKVersion()
            out[mod] = v
        except Exception:                                            # noqa: BLE001
            out[mod] = None
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
