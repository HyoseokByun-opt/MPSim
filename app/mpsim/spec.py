"""Job specification: browser form -> validated, solver-ready description.

The browser sends a loose JSON form. `normalize()` turns it into one strict
dict that every later stage reads, so validation lives in exactly one place:

* sizes converted to um, fractions resolved to volume fractions (vol % and
  wt % may be mixed),
* every property checked against its physical range,
* the **label table**: one integer per region that carries its own full set of
  properties. Besides matrix and phases it contains
    - shell labels for coated / hollow particles.
* interfacial thermal resistances (v4): R_int between a filler and the matrix
  and R_c between two touching fillers are not regions of the voxel map; the
  thermal solve puts them on the voxel faces where the regions meet.
* pores are not a shape: any phase - or the matrix - whose material is a gas
  (E = 0) is pore space, whatever its geometry.
"""
from __future__ import annotations

import copy

from . import geometry as G
from . import materials as M
from . import shapes as SH

# key -> (label, module)
ANALYSES = {
    "thermal":      ("Thermal conductivity", "Conduction"),
    "electrical":   ("Electrical conductivity", "Conduction"),
    "dielectric":   ("Permittivity and dielectric loss", "Electromagnetics"),
    "magnetic":     ("Magnetic permeability", "Electromagnetics"),
    "emi":          ("EMI shielding effectiveness", "Electromagnetics"),
    "cte":          ("Elasticity and thermal expansion", "Mechanics"),
    "mechtest":     ("Mechanical test (tension or compression)", "Mechanics"),
    "permeability": ("Permeability (Stokes flow)", "Flow"),
    "filtration":   ("Filtration efficiency", "Flow"),
    "tortuosity":   ("Diffusion and tortuosity", "Diffusion"),
    "acoustics":    ("Acoustic absorption (JCA)", "Acoustics"),
    "radiation":    ("Radiative extinction", "Radiation"),
    "morphology":   ("Porosity and size distributions", "Structure analysis"),
    "percolation":  ("Percolation path and geodesic tortuosity", "Structure analysis"),
    "porosimetry":  ("Porosimetry and capillary pressure", "Structure analysis"),
    "pore_network": ("Pore network extraction", "Structure analysis"),
    "grains":       ("Grain and filler analysis", "Structure analysis"),
    "viscosity":    ("Viscosity and flowability (uncured compound)", "Rheology"),
    "moisture":     ("Moisture uptake, diffusion and swelling", "Reliability"),
}
PORE_ANALYSES = {"permeability", "tortuosity", "acoustics", "radiation", "porosimetry", "pore_network",
                 "filtration"}

FLUIDS = {
    # name: (surface tension N/m, contact angle deg) - drainage of the pore space
    "mercury": (0.485, 140.0),
    "water": (0.0728, 0.0),
    "isopropanol": (0.0217, 0.0),
}


def required_analyses(chosen):
    """The analyses a selection needs, including the ones it is built on."""
    need = set(chosen)
    if "emi" in need:
        need |= {"electrical", "dielectric"}
    if "acoustics" in need:
        need |= {"permeability", "tortuosity"}
    if "filtration" in need:
        # particles are carried by the flow, so the Stokes solution is required
        need |= {"permeability"}
    if "mechtest" in need:
        # the test's macroscopic state comes from the effective stiffness
        need |= {"cte"}
    return need


def _num(x, what, lo=None, hi=None, allow_none=False):
    if x is None or x == "":
        if allow_none:
            return None
        raise ValueError(f"No value is given for the {what}")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"The {what} is not a number: {x!r}")
    if lo is not None and v < lo:
        raise ValueError(f"The {what} of {v:g} must not be smaller than {lo:g}")
    if hi is not None and v > hi:
        raise ValueError(f"The {what} of {v:g} must not be larger than {hi:g}")
    return v


_LINEAR = ("d", "length", "ly", "lz", "coil_d", "pitch")


def _rescale(size_um, d_new):
    """The size dict with its main size set to d_new and every other length
    scaled with it (the shape is kept)."""
    f = float(d_new) / float(size_um["d"])
    return {k: (v * f if k in _LINEAR else v) for k, v in size_um.items()}


# the separation an input of 5.3.1 or earlier meant by contacts = "apart"
LEGACY_GAP_VOX = 2.0


def _contacts(opts):
    """{contacts, particle_gap_vox} from the options.

    particle_gap_vox (Structure tab) is the gap, in voxels, kept between the
    surfaces of non-overlapping particles: 0 (default) lets them touch, as
    they do in a filled compound; more moves them apart (never trims them)
    wherever the loading allows. An input without it keeps its meaning:
    contacts = "apart" was a 2-voxel gap, "keep" and "separate" stay."""
    c = opts.get("contacts")
    if c == "separate":
        return {"contacts": "separate", "particle_gap_vox": 0.0}
    g = opts.get("particle_gap_vox")
    if g in (None, ""):
        g = LEGACY_GAP_VOX if c == "apart" else 0.0
    g = float(_num(g, "gap between particles (voxels)", 0.0, 50.0))
    return {"contacts": "apart" if g > 0.0 else "keep", "particle_gap_vox": g}


def _mechtest(mt, rve, errors):
    """A tension or compression test (mechtest.py): direction, lateral
    condition, the load as a displacement (µm), strain (%), force (N) or
    pressure (MPa), the specimen's length along the load (mm; a film's
    thickness by default when loaded through it) and loaded area (mm²), and
    an optional strength of the matrix (MPa)."""
    axis = mt.get("axis") if mt.get("axis") in ("x", "y", "z") else "z"
    film_z = bool(rve.get("film")) and axis == "z" and rve.get("T_um")
    out = {"mode": "tension" if mt.get("mode") == "tension" else "compression", "axis": axis,
           "lateral": "confined" if mt.get("lateral") == "confined" else "free",
           "control": mt.get("control") if mt.get("control") in ("displacement", "strain", "force", "pressure")
           else "force",
           "value": 100.0, "length_mm": float(rve["T_um"]) * 1e-3 if film_z else 10.0, "area_mm2": 10.0,
           "strength_MPa": None}
    try:
        out["value"] = _num(mt.get("value", 100.0), "load of the mechanical test", lo=0.0)
        if out["control"] == "strain" and out["value"] > 20.0:
            raise ValueError("The strain of the mechanical test is limited to 20 % (small strains, linear elasticity)")
        if mt.get("length_mm") not in (None, ""):
            out["length_mm"] = _num(mt.get("length_mm"), "specimen length along the load", lo=1e-9)
        out["area_mm2"] = _num(mt.get("area_mm2", 10.0), "loaded area of the specimen", lo=1e-12)
        out["strength_MPa"] = _num(mt.get("strength_MPa"), "strength of the matrix", lo=0.0, allow_none=True) or None
    except ValueError as e:
        errors.append(str(e))
    return out


def _dispersion(dp, phases, rve, errors):
    """How evenly the particles fill the box (rve.dispersion; percentages in
    the form, fractions here):
      uniform  - an even random dispersion (as always);
      region   - particles only in a slab across `axis`, from `from` to `to`
                 % of the box edge; matrix elsewhere;
      gradient - the loading changes linearly along `axis`, `ratio` times as
                 high at the end of the axis as at its start;
      clusters - `n_clusters` spherical clusters taking `cluster_vol` % of the
                 box hold `share` % of the particles; the rest spread evenly.
    The loading where the particles are densest must stay below random close
    packing (64 vol%)."""
    mode = dp.get("mode") if dp.get("mode") in ("uniform", "region", "gradient", "clusters") else "uniform"
    out = {"mode": mode}
    if mode == "uniform":
        return out
    axis = dp.get("axis") if dp.get("axis") in ("x", "y", "z") else "z"
    vf = sum(float(ph.get("vf") or 0.0) for ph in phases if ph.get("shape") != G.NETWORK)
    try:
        if mode == "region":
            a = _num(dp.get("from", 0), "start of the particle region", 0.0, 100.0) / 100.0
            b = _num(dp.get("to", 50), "end of the particle region", 0.0, 100.0) / 100.0
            if b - a < 0.05:
                raise ValueError("The particle region must span at least 5 % of the box edge (end above start)")
            out.update({"axis": axis, "from": a, "to": b})
            peak = vf / (b - a)
            if rve.get("film"):
                errors.append("A particle region is set in a periodic cube; in a thin film use a gradient or "
                              "clusters")
        elif mode == "gradient":
            r = _num(dp.get("ratio", 3), "loading ratio along the axis", 0.02, 50.0)
            out.update({"axis": axis, "ratio": r})
            peak = vf * 2.0 * max(r, 1.0) / (1.0 + r)
        else:
            n = int(_num(dp.get("n_clusters", 4), "number of clusters", 1, 64))
            fv = _num(dp.get("cluster_vol", 30), "box volume taken by the clusters", 2.0, 80.0) / 100.0
            sh = _num(dp.get("share", 45), "share of the particles in the clusters", 0.0, 100.0) / 100.0
            out.update({"n_clusters": n, "cluster_vol": fv, "share": sh})
            peak = vf * max(sh / fv, (1.0 - sh) / (1.0 - fv))
        out["peak_vf"] = peak
        if peak > 0.64:
            errors.append(f"With this dispersion the particles would fill {100 * peak:.0f} vol% where they are "
                          f"densest; no random packing exceeds about 64 vol%. Lower the contrast or widen the region")
    except ValueError as e:
        errors.append(str(e))
    if rve.get("arrangement") == "sheared":
        errors.append("The arrangement after shear flow starts from an even dispersion; set the dispersion to even")
    return out


def _size_distribution(dist, size_um, unit, where, particles=True):
    """(dist, size_um) of a phase. The generator draws size factors of the
    nominal size with a number mean of 1 (geometry.sample_scales), so the
    nominal size becomes the number mean of the distribution:

    log-normal, normal: the entered size is the number mean of the diameter
      and 'std' its standard deviation (number basis, the phase's length
      unit; a form of v5.3.0 and earlier gives 'cv', std / mean, instead);
      'd_min', 'd_max' cut the distribution (the phase's length unit), +-2
      standard deviations without them. The generator takes the coefficient
      of variation std / mean.
    table: a measured distribution - diameters 'd' (in 'unit', default µm)
      and values 'v' per channel or cumulative ('form'), by 'volume' or by
      'number' ('basis'); 'd_min', 'd_max' cut it. The nominal size becomes its
      number mean."""
    kind = dist.get("type")
    if not particles or kind not in ("lognormal", "normal", "table"):
        return {"type": "mono"}, size_um
    bound = lambda key, u: (G.to_um(_num(dist.get(key), f"{key.replace('_', ' ')} of the sizes of {where}", lo=1e-9), u)  # noqa: E731
                            if dist.get(key) not in (None, "") else None)
    if kind in ("lognormal", "normal"):
        d_um = float(size_um["d"])
        if dist.get("std") not in (None, ""):
            std_um = G.to_um(_num(dist.get("std"), f"standard deviation of the sizes of {where}", lo=0.0), unit)
        else:
            std_um = float(_num(dist.get("cv", 0.2), f"size coefficient of variation of {where}", 0.0)) * d_um
        cv = std_um / d_um if d_um > 0 else 0.0
        # a normal diameter beyond std = mean would be half below zero; a
        # log-normal one stays positive, but beyond twice the mean the few
        # largest particles set the whole RVE
        cv_max = 1.0 if kind == "normal" else 2.0
        if cv > cv_max + 1e-12:
            raise ValueError(f"{where}: a standard deviation of {std_um:.4g} µm exceeds "
                             f"{'the mean' if kind == 'normal' else 'twice the mean'} ({cv_max * d_um:.4g} µm)")
        out = {"type": kind, "cv": cv}
        lo, hi = bound("d_min", unit), bound("d_max", unit)
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"{where}: the upper bound of the sizes must exceed the lower bound")
        if (lo is not None or hi is not None) and cv > 0:
            if lo is not None:
                out.update(lo=lo / size_um["d"], d_min_um=lo)
            if hi is not None:
                out.update(hi=hi / size_um["d"], d_max_um=hi)
            g_lo, g_hi = G.scale_bounds(out)
            if g_hi <= g_lo * (1 + 1e-9):
                raise ValueError(f"{where}: no size lies between the bounds of the distribution")
        return out, size_um
    tu = dist.get("unit", "um")
    if tu not in G.UNIT_TO_UM:
        raise ValueError(f"{where}: unknown length unit '{tu}' of the size distribution")
    d = [G.to_um(x, tu) for x in (dist.get("d") or [])]
    v = list(dist.get("v") or [])
    if len(d) != len(v):
        raise ValueError(f"{where}: the size distribution needs as many values as diameters")
    if len(d) > 2000:
        raise ValueError(f"{where}: the size distribution has more than 2000 rows")
    basis = "number" if dist.get("basis") == "number" else "volume"
    form = dist.get("form") if dist.get("form") in ("differential", "cumulative") else "auto"
    try:
        edges, pn, info = G.psd_bins(d, v, form, basis, bound("d_min", tu), bound("d_max", tu))
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None
    mean = G.table_moments(edges, pn)
    size_um = _rescale(size_um, mean)
    out = {"type": "table", "edges": [float(x) for x in edges / mean], "p": [float(x) for x in pn],
           "basis": basis, "form": info["form"], "d_mean_um": mean,
           "file": str(dist.get("file") or "")[:120]}
    return out, size_um


def _fill_from_library(entry, lib):
    """A form may name a material by id only (presets, command line)."""
    if not isinstance(entry, dict) or entry.get("props") or not entry.get("material_id"):
        return
    m = lib.get(entry["material_id"])
    if m is None:
        raise ValueError(f"The material library contains no material '{entry['material_id']}'")
    entry["props"] = dict(m["props"])
    entry.setdefault("name", m["name"])


def normalize(form):
    form = copy.deepcopy(form or {})
    errors = []
    lib = M.library_by_id()
    try:
        _fill_from_library(form.get("matrix"), lib)
        for ph in form.get("phases") or []:
            _fill_from_library(ph, lib)
            if (ph.get("shell") or {}).get("enabled"):
                _fill_from_library(ph["shell"], lib)
    except ValueError as e:
        errors.append(str(e))
    out = {"name": str(form.get("name") or "Unnamed case")[:120],
           "preview": bool(form.get("preview"))}

    # ---- matrix -----------------------------------------------------------
    mx = form.get("matrix") or {}
    try:
        mprops = M.clean_props(mx.get("props") or {}, where="Matrix")
    except ValueError as e:
        errors.append(str(e))
        mprops = None
    out["matrix"] = {"name": mx.get("name") or "Matrix", "material_id": mx.get("material_id"),
                     "props": mprops, "void": bool(mprops and mprops["E"] <= 0)}

    # ---- phases -----------------------------------------------------------
    phases = []
    for i, ph in enumerate(form.get("phases") or []):
        where = f"Phase {i+1} ({ph.get('name') or ph.get('material_name') or ''})"
        try:
            props = M.clean_props(ph.get("props") or {}, where=where)
            shape = ph.get("shape", "sphere")
            if shape not in SH.USER_SHAPES and shape != G.NETWORK:
                raise ValueError(f"{where}: unknown shape '{shape}'")
            unit = (ph.get("size") or {}).get("unit", "um")
            if unit not in G.UNIT_TO_UM:
                raise ValueError(f"{where}: unknown length unit '{unit}'")
            sz = ph.get("size") or {}
            size_um = {"d": G.to_um(_num(sz.get("d"), f"size of {where}", lo=1e-9), unit)}
            if shape == "spheroid":
                size_um["aspect"] = _num(sz.get("aspect", 1.0), f"aspect ratio of {where}", 0.01, 100.0)
            if shape in ("cylinder", "spherocylinder"):
                size_um["length"] = G.to_um(_num(sz.get("length"), f"length of {where}", lo=1e-9), unit)
            if shape in ("cuboid", "superellipsoid"):
                size_um["ly"] = G.to_um(_num(sz.get("ly"), f"second edge of {where}", lo=1e-9), unit)
                size_um["lz"] = G.to_um(_num(sz.get("lz"), f"third edge of {where}", lo=1e-9), unit)
            if shape == "superellipsoid":
                size_um["n"] = _num(sz.get("n", 4.0), f"blockiness of {where}", 2.0, 40.0)
            if shape == "polyhedron":
                kind = sz.get("poly", "icosahedron")
                if kind not in SH.POLY_KINDS:
                    raise ValueError(f"{where}: unknown polyhedron '{kind}'")
                size_um["poly"] = kind
                size_um["n_vertices"] = int(_num(sz.get("n_vertices", 14), f"vertices of {where}", 5, 60))
                # 0: every grain its own shape (generate.phase_templates)
                size_um["variants"] = int(_num(sz.get("variants", 8), f"shape variants of {where}", 0, 20000))
            if shape == "helix":
                size_um["coil_d"] = G.to_um(_num(sz.get("coil_d"), f"coil diameter of {where}", lo=1e-9), unit)
                size_um["pitch"] = G.to_um(_num(sz.get("pitch", 0.0), f"pitch of {where}", 0.0), unit)
                size_um["turns"] = _num(sz.get("turns", 3.0), f"number of turns of {where}", 0.1, 200.0)
            if shape != G.NETWORK:
                SH.validate(shape, size_um)                  # raises on inconsistent sizes
            dist, size_um = _size_distribution(ph.get("dist") or {"type": "mono"}, size_um, unit, where,
                                               shape != G.NETWORK)
            fr = ph.get("fraction") or {}
            fraction = {"value": _num(fr.get("value"), f"content of {where}", 0.0, 99.0),
                        "basis": "wt" if fr.get("basis") == "wt" else "vol"}
            ori = ph.get("orientation") or {}
            mode = ori.get("mode", "iso")
            if mode not in ("iso", "x", "y", "z", "xy"):
                mode = "iso"
            orientation = {"mode": mode, "spread_deg": _num(ori.get("spread_deg", 0.0), f"orientation spread of {where}", 0.0, 90.0)}
            shell = None
            sh = ph.get("shell") or {}
            if sh.get("enabled"):
                shell = {"thickness_um": G.to_um(_num(sh.get("thickness"), f"shell thickness of {where}", lo=1e-9), unit),
                         "name": sh.get("name") or "shell",
                         "material_id": sh.get("material_id"),
                         "props": M.clean_props(sh.get("props") or {}, where=f"{where}, shell"),
                         # False: every particle gets the same wall. True: the wall is
                         # `thickness` at the nominal size and scales with each particle,
                         # so t/R - and with it the hollow fraction - is one number for
                         # the whole population, as in a density grade of glass bubbles.
                         "scale_with_size": bool(sh.get("scale_with_size"))}
            # interfacial thermal resistances [m^2 K/W] (Kapitza): filler-matrix and
            # filler-filler contact; typical 1e-9..1e-6
            r_int = _num(ph.get("r_int", 0.0) or 0.0, f"interfacial thermal resistance of {where}", 0.0, 1e-2)
            r_contact = _num(ph.get("r_contact", 0.0) or 0.0, f"contact resistance of {where}", 0.0, 1e-2)
            # stacked platelets (v5.2.1): each particle is a stack of this many
            # platelets bonded face to face along its own z axis (thickness)
            layers = int(round(_num(ph.get("stack_layers", 1) or 1, f"stacked layers of {where}", 1, 100)))
            if layers > 1 and shape not in ("spheroid", "cylinder", "cuboid", "superellipsoid"):
                raise ValueError(f"{where}: stacked layers need a platelet shape (spheroid, cylinder, cuboid or "
                                 f"superellipsoid - the stack lies along the particle's z axis)")
            phases.append({
                "name": ph.get("name") or ph.get("material_name") or f"Phase {i+1}",
                "material_id": ph.get("material_id"),
                "props": props, "void": props["E"] <= 0,
                "shape": shape, "size_um": size_um, "size_unit": unit, "dist": dist,
                "fraction": fraction, "orientation": orientation,
                "overlap": bool(ph.get("overlap")) and shape != G.NETWORK,
                "gap_frac": _num(ph.get("gap_frac", 0.0) or 0.0, f"minimum gap of {where}", 0.0, 1.0),
                "shell": shell, "r_int": r_int, "r_contact": r_contact, "stack_layers": layers,
            })
        except ValueError as e:
            errors.append(str(e))
    out["phases"] = phases

    # ---- contact resistance between two different fillers ----------------
    # {"i-j": R} for phase indices i < j. A pair not listed takes the mean of
    # the two fillers' own R_c (the diagonal), as before.
    rc = {}
    for key, v in (form.get("contact_rc") or {}).items():
        try:
            i, j = (int(x) for x in str(key).split("-"))
        except ValueError:
            continue
        if v in (None, "") or i == j or not (0 <= i < len(phases) and 0 <= j < len(phases)):
            continue
        try:
            rc[f"{min(i, j)}-{max(i, j)}"] = _num(v, f"contact resistance between {phases[i]['name']} and "
                                                     f"{phases[j]['name']}", 0.0, 1e-2)
        except ValueError as e:
            errors.append(str(e))
    out["contact_rc"] = rc

    # ---- fractions --------------------------------------------------------
    if mprops is not None and len(phases) == len(form.get("phases") or []):
        try:
            fr = M.resolve_fractions(mprops, phases)
            for ph, v in zip(phases, fr["vf"]):
                ph["vf"] = v
            out["composition"] = fr
        except ValueError as e:
            errors.append(str(e))

    # ---- RVE --------------------------------------------------------------
    rv = form.get("rve") or {}
    out["rve"] = {
        "auto": bool(rv.get("auto", True)),
        "quality": rv.get("quality") if rv.get("quality") in ("fast", "standard", "accurate") else "standard",
        "L_um": _num(rv.get("L_um"), "RVE edge length", lo=1e-9, allow_none=True) if not rv.get("auto", True) else None,
        "voxel_um": _num(rv.get("voxel_um"), "voxel size", lo=1e-9, allow_none=True) if not rv.get("auto", True) else None,
        "N": int(_num(rv.get("N"), "grid size", 8, 2048, allow_none=True) or 0) or None if not rv.get("auto", True) else None,
        "auto_enlarge": bool(rv.get("auto_enlarge", True)),
        "n_seeds": int(_num(rv.get("n_seeds", 0) or 0, "number of realisations", 0, 10)),
        "seed": int(_num(rv.get("seed", 1) or 1, "random seed", 0, 2**31 - 1)),
        "stat_enlarge": bool(rv.get("stat_enlarge", False)),
        # a thin film: x and y periodic, z the real thickness, every particle
        # wholly inside it (not cut by the faces, not wrapping through them)
        "film": bool(rv.get("film")),
        "T_um": (_num(rv.get("T_um"), "film thickness", lo=1e-9) if rv.get("film") else None),
        # matrix kept at each face of a film; None = one voxel, 0 = particles may touch the faces
        "skin_um": (_num(rv.get("skin_um"), "matrix skin at the faces", lo=0.0, allow_none=True)
                    if rv.get("film") else None),
        # how the particles are arranged: a random packing, or that packing
        # sheared in the resin by the particle dynamics (the arrangement a
        # dispensed or moulded compound keeps when it cures); a whole-number
        # strain, so that the sheared box is periodic again
        "arrangement": "sheared" if rv.get("arrangement") == "sheared" else "random",
        "shear_strain": int(round(_num(rv.get("shear_strain", 10) or 10, "strain of the shear flow", 1, 100))),
    }
    out["rve"]["dispersion"] = _dispersion(rv.get("dispersion") or {}, phases, out["rve"], errors)

    # ---- analyses -----------------------------------------------------------
    an = form.get("analyses") or {}
    chosen = [k for k in ANALYSES if an.get(k)]
    opts = form.get("options") or {}
    emi = opts.get("emi") or {}
    ac = opts.get("acoustics") or {}
    rad = opts.get("radiation") or {}
    por = opts.get("porosimetry") or {}
    gas = opts.get("gas") or {}
    filt = opts.get("filtration") or {}
    vis = opts.get("viscosity") or {}
    vmod = vis.get("model") or {}
    uf = vis.get("underfill") or {}
    dem = vis.get("dem") or {}
    f_dmin = _num(filt.get("d_min_um", 0.01), "smallest particle diameter", 1e-4, 1e3)
    f_dmax = _num(filt.get("d_max_um", 5.0), "largest particle diameter", 1e-4, 1e3)
    f_n = int(_num(filt.get("n_sizes", 8), "number of particle sizes", 2, 24))
    if f_dmax <= f_dmin:
        errors.append("The largest particle diameter must exceed the smallest")
        f_dmax = f_dmin * 10.0
    f_ratio = (f_dmax / f_dmin) ** (1.0 / (f_n - 1)) if f_n > 1 else 1.0
    thickness = _num(opts.get("thickness_mm", emi.get("thickness_mm", 1.0)), "layer thickness", 1e-6, 1e4)
    fluid = por.get("fluid") if por.get("fluid") in (*FLUIDS, "custom") else "mercury"
    gamma0, theta0 = FLUIDS.get(fluid, FLUIDS["mercury"])
    out["analyses"] = chosen
    out["options"] = {
        # z first: the fields, arrows, field lines and the resolution check
        # use the first direction, and through-thickness (z) is what a film or
        # a layer is most often asked for
        "directions": [d for d in "zxy" if d in (opts.get("directions") or "xyz")] or ["z", "x", "y"],
        # a run with the same geometry reuses the structure an earlier run
        # generated, and more analyses on the same inputs continue that run
        "reuse_runs": bool(opts.get("reuse_runs", True)),
        # every field also kept on the solver's grid (2 bytes per voxel), so
        # the viewer's sections and voxel volume show its full resolution
        "view_full_fields": bool(opts.get("view_full_fields", True)),
        # the figures of each analysis drawn by a background process as soon as
        # it finishes, shown in the browser while the run goes on
        "live_figures": bool(opts.get("live_figures", True)),
        "backend": opts.get("backend") if opts.get("backend") in ("puma", "puma_fv", "builtin") else "puma",
        "tol": _num(opts.get("tol", 1e-6), "solver tolerance", 1e-12, 1e-2),
        # conduction-type properties (k, sigma, eps_r, mu_r): "fv" solves them once
        # with the periodic finite-volume discretisation (default; on voxel grids
        # it came 2-3x closer to converged references than PuMA's finite
        # elements, guide 7.2), "fe" with PuMA's periodic finite elements
        "cond_method": "fe" if opts.get("cond_method") == "fe" else "fv",
        # elasticity and thermal expansion: "fans" voxel finite elements with an
        # FFT-preconditioned CG (default: PuMA's elements, iterations independent
        # of the grid), "puma" PuMA's own solve (MINRES; its isotropic material
        # doubles the shear modulus inside the solve, see solvers/fans.py),
        # "fft" the Moulinec-Suquet spectral scheme
        "elastic_method": opts.get("elastic_method") if opts.get("elastic_method") in ("fans", "puma", "fft") else "fans",
        # elasticity and CTE also with the regions that have a glass transition
        # in their rubbery state (E above Tg, alpha2): the alpha2 of an EMC
        "above_tg": bool(opts.get("above_tg", False)),
        "mechtest": _mechtest(opts.get("mechtest") or {}, out["rve"], errors),
        # moisture: plate thickness and exposure for the uptake curve
        "moisture": {"thickness_mm": _num((opts.get("moisture") or {}).get("thickness_mm", 1.0), "moisture plate thickness", 1e-4, 1e3),
                     "hours": _num((opts.get("moisture") or {}).get("hours", 168.0), "moisture exposure time", 0.0, 1e6),
                     "sides": "one" if (opts.get("moisture") or {}).get("sides") == "one" else "both"},
        # solve every conduction problem a second time with the other method and
        # report the difference (a discretisation check; doubles the cost)
        "crosscheck": bool(opts.get("crosscheck")) and opts.get("crosscheck") != "off",
        # re-solve the conduction-type properties (first direction) on the same
        # particles drawn 1.5x finer: "auto" only when particles touch at voxel
        # level (a gap the grid does not resolve), "on" always, "off" never
        "resolution_check": opts.get("resolution_check") if opts.get("resolution_check") in ("on", "off") else "auto",
        "delta_T": _num(opts.get("delta_T", 100.0), "temperature change ΔT", -1000.0, 1000.0),
        # thermal-stress maps: "constrained" holds the RVE at zero mean strain,
        # "free" lets it expand by alpha* dT so only the phase mismatch remains
        "thermal_bc": "free" if opts.get("thermal_bc") == "free" else "constrained",
        # touching particles: "keep" (default) lets them touch where the
        # packing puts them, never overlapping - the usual state of a filled
        # compound; "apart" moves them so that their surfaces stay
        # particle_gap_vox voxels apart wherever the loading allows;
        # "separate" trims touching voxels to matrix as v3 did
        **_contacts(opts),
        # independent solves (properties x directions, elastic load cases) side
        # by side on the run's cores; "off" solves them one after another
        "parallel_solves": "off" if opts.get("parallel_solves") == "off" else "auto",
        "thickness_mm": thickness,
        "temperature_K": _num(opts.get("temperature_K", 298.15), "temperature", 1.0, 5000.0),
        "knudsen": bool(opts.get("knudsen", False)),
        "gas": {"molar_mass_g_mol": _num(gas.get("molar_mass_g_mol", 28.97), "gas molar mass", 1.0, 1000.0),
                "viscosity_Pa_s": _num(gas.get("viscosity_Pa_s", 1.81e-5), "gas viscosity", 1e-7, 1.0),
                "diffusivity_m2_s": _num(gas.get("diffusivity_m2_s", 2.0e-5), "molecular diffusivity", 1e-12, 1.0),
                "pressure_Pa": _num(gas.get("pressure_Pa", 101325.0), "gas pressure", 1e-3, 1e8),
                "molecule_diameter_nm": _num(gas.get("molecule_diameter_nm", 0.37), "molecule diameter", 0.05, 5.0),
                "density_kg_m3": _num(gas.get("density_kg_m3", 1.204), "gas density", 1e-6, 2e4),
                "sound_speed_m_s": _num(gas.get("sound_speed_m_s", 343.2), "speed of sound", 1.0, 1e5),
                "gamma": _num(gas.get("gamma", 1.4), "ratio of specific heats", 1.0, 2.0),
                "prandtl": _num(gas.get("prandtl", 0.71), "Prandtl number", 0.01, 100.0)},
        "emi": {"thickness_mm": _num(emi.get("thickness_mm", thickness), "EMI layer thickness", 1e-6, 1e4),
                "f_min_hz": _num(emi.get("f_min_hz", 1e6), "minimum frequency", 1.0, 1e13),
                "f_max_hz": _num(emi.get("f_max_hz", 1e10), "maximum frequency", 1.0, 1e13),
                # "complex": the RVE solved with sigma + j w eps at n_freq
                # frequencies (default); "static": DC sigma and static eps
                # solved apart and combined (v1-v3)
                "method": emi.get("method") if emi.get("method") in ("complex", "static") else "complex",
                "n_freq": int(_num(emi.get("n_freq", 10), "number of EMI frequencies", 2, 40)),
                # a full-wave openEMS run of the RVE slab - Maxwell's
                # equations on the voxels, no homogenisation - against the
                # homogenised slab (10-100 GHz). On by default (user,
                # 2026-10-07): a shielding analysis is expected to include
                # the wave simulation; it is skipped with a note where
                # openEMS is not installed
                "fullwave": bool(emi.get("fullwave", True))},
        "acoustics": {"thickness_mm": _num(ac.get("thickness_mm", 20.0), "absorber thickness", 1e-3, 1e4),
                      "f_min_hz": _num(ac.get("f_min_hz", 100.0), "minimum acoustic frequency", 1.0, 1e6),
                      "f_max_hz": _num(ac.get("f_max_hz", 10000.0), "maximum acoustic frequency", 1.0, 1e6)},
        "radiation": {"temperature_K": _num(rad.get("temperature_K", 1000.0), "radiation temperature", 1.0, 5000.0),
                      "refractive_index": _num(rad.get("refractive_index", 1.0), "refractive index", 1.0, 10.0),
                      "sources": int(_num(rad.get("sources", 200), "number of ray sources", 10, 5000)),
                      "rays": int(_num(rad.get("rays", 500), "rays per source", 10, 20000))},
        "filtration": {"face_velocity_m_s": _num(filt.get("face_velocity_m_s", 0.05), "filtration face velocity", 1e-5, 100.0),
                       "n_particles": int(_num(filt.get("n_particles", 200), "number of tracked particles", 20, 5000)),
                       "particle_density_kg_m3": _num(filt.get("particle_density_kg_m3", 1000.0), "particle density", 1.0, 3e4),
                       "d_min_um": f_dmin, "d_max_um": f_dmax, "n_sizes": f_n,
                       "diameters_um": [f_dmin * f_ratio ** i for i in range(f_n)]},
        "porosimetry": {"fluid": fluid,
                        "surface_tension_N_m": _num(por.get("surface_tension_N_m", gamma0) if fluid == "custom" else gamma0,
                                                    "surface tension", 1e-4, 10.0),
                        "contact_angle_deg": _num(por.get("contact_angle_deg", theta0) if fluid == "custom" else theta0,
                                                  "contact angle", 0.0, 180.0),
                        "steps": int(_num(por.get("steps", 40), "number of pressure steps", 5, 200))},
        "contrast_cap": _num(opts.get("contrast_cap", 1e6), "property contrast limit", 10.0, 1e9),
        # the matrix as the uncured liquid resin, the fillers rigid
        "viscosity": {
            "model": {"type": vmod.get("type") if vmod.get("type") in ("newtonian", "power", "carreau", "cross") else "newtonian",
                      "mu": _num(vmod.get("mu", 1.0), "resin viscosity", 1e-6, 1e8),
                      "K": _num(vmod.get("K", 1.0), "power-law consistency", 1e-6, 1e8),
                      "n": _num(vmod.get("n", 0.7), "flow index", 0.05, 2.0),
                      "mu0": _num(vmod.get("mu0", 1.0), "zero-shear viscosity", 1e-6, 1e8),
                      "mu_inf": _num(vmod.get("mu_inf", 0.0), "infinite-shear viscosity", 0.0, 1e8),
                      "lam": _num(vmod.get("lam", 1.0), "time constant", 1e-9, 1e6)},
            "yield_Pa": _num(vis.get("yield_Pa", 0.0), "resin yield stress", 0.0, 1e6),
            "gd_min": _num(vis.get("gd_min", 0.01), "lowest shear rate", 1e-6, 1e6),
            # the RVE flow solve also where the particle dynamics gives the
            # answer: its 3D resin flow, at the same time as the particle
            # dynamics when that runs on a GPU (off: skipped there)
            "rve_with_dem": bool(vis.get("rve_with_dem", False)),
            "gd_max": _num(vis.get("gd_max", 1000.0), "highest shear rate", 1e-5, 1e7),
            "gd_ref": _num(vis.get("gd_ref", 10.0), "reference shear rate", 1e-6, 1e7),
            # maximum packing fraction phi_m: auto = Farr-Groot for spheres of
            # any size distribution, the jammed packing of the particles
            # otherwise; or one of them, or a number entered by the user
            "phi_m_mode": vis.get("phi_m_mode") if vis.get("phi_m_mode") in ("auto", "farr_groot", "jamming", "manual") else "auto",
            # frictional particles jam in shear below the random close packing
            "friction": vis.get("friction") if vis.get("friction") in ("none", "frictional") else "none",
            "phi_m": _num(vis.get("phi_m", 0.64), "maximum packing fraction", 0.2, 0.99),
            # intrinsic viscosity of non-spherical fillers from a dilute RVE
            "dilute_rve": bool(vis.get("dilute_rve", True)),
            "density_resin": _num(vis.get("density_resin", 1150.0), "resin density", 100.0, 3e4),
            "settle_min": _num(vis.get("settle_min", 60.0), "settling time", 0.0, 1e6),
            # particle dynamics (solvers/suspension.py): every filler particle
            # moving, turning and colliding in the sheared resin - the best
            # estimate for spheres where the voxel flow solve cannot resolve
            # the films between them, and the source of the particle-size
            # effect (roughness and van der Waals are lengths). Friction 0.25
            # reproduces the measured law of non-Brownian spheres (Boyer et al.
            # 2011: 20.7 at 50 vol%, 103 at 55 vol%; mu_f 0.2 gave 19.6 / 70,
            # 0.3 20.3 / 165); 0 reproduces Krieger-Dougherty with phi_m 0.64.
            "dem": {"on": bool(dem.get("on", True)),
                    "n": int(_num(dem.get("n", 500), "number of particles", 50, 10_000_000)),
                    "strain": _num(dem.get("strain", 10.0), "sheared strain", 1.0, 100.0),
                    "roughness_nm": _num(dem.get("roughness_nm", 5.0), "surface roughness", 0.01, 1e4),
                    "hmin_nm": _num(dem.get("hmin_nm", 1.0), "closest approach of the surfaces", 0.1, 1e3),
                    "mu_f": _num(dem.get("mu_f", 0.25), "friction coefficient", 0.0, 2.0),
                    # surface chemistry that bulk data cannot give - fitted to a
                    # measured viscosity (0: none): a resin layer bound to the
                    # surface, and the work of adhesion of touching surfaces
                    "bound_nm": _num(dem.get("bound_nm", 0.0), "bound resin layer", 0.0, 1e4),
                    "adhesion_mJ_m2": _num(dem.get("adhesion_mJ_m2", 0.0), "work of adhesion", 0.0, 1e4),
                    # further shear rates for the flow curve (0: only the reference
                    # rate); skipped where the attraction is negligible
                    "rates": int(_num(dem.get("rates", 4), "shear rates of the flow curve", 0, 12)),
                    "hamaker_J": (None if dem.get("hamaker_J") in (None, "", "auto") else
                                  _num(dem.get("hamaker_J"), "Hamaker constant", 0.0, 1e-17)),
                    "backend": dem.get("backend") if dem.get("backend") in ("auto", "cpu", "cuda") else "auto"},
            "underfill": {"gap_um": _num(uf.get("gap_um", 50.0), "underfill gap", 0.1, 1e5),
                          "length_mm": _num(uf.get("length_mm", 10.0), "underfill flow length", 1e-3, 1e4),
                          "gamma_mN_m": _num(uf.get("gamma_mN_m", 35.0), "surface tension", 0.1, 1000.0),
                          "theta_deg": _num(uf.get("theta_deg", 30.0), "contact angle", 0.0, 89.9)},
        },
    }
    if out["options"]["emi"]["f_max_hz"] <= out["options"]["emi"]["f_min_hz"]:
        errors.append("The maximum EMI frequency must exceed the minimum frequency")
    if out["options"]["acoustics"]["f_max_hz"] <= out["options"]["acoustics"]["f_min_hz"]:
        errors.append("The maximum acoustic frequency must exceed the minimum frequency")
    if abs(out["options"]["porosimetry"]["contact_angle_deg"] - 90.0) < 1e-6:
        errors.append("A contact angle of 90° gives no capillary pressure; porosimetry needs another angle")
    if not chosen and not out["preview"]:
        errors.append("At least one analysis must be selected")
    if out["preview"]:
        chosen = []
        out["analyses"] = []
    pores = [p for p in phases if p["void"]]
    need_pores = sorted(PORE_ANALYSES & set(chosen))
    if need_pores and not pores and not out["matrix"]["void"]:
        errors.append("The analyses " + ", ".join(ANALYSES[k][0] for k in need_pores)
                      + " require a pore (void) phase")
    if "grains" in chosen and not any(p["shape"] != G.NETWORK for p in phases):
        errors.append("Grain analysis requires at least one particle phase")
    if "moisture" in chosen:
        mp_ = out["matrix"]["props"]
        if not out["matrix"]["void"] and not (mp_.get("D_w") and mp_.get("c_sat")):
            errors.append("The moisture analysis needs the matrix's moisture diffusivity D_w and saturated uptake c_sat "
                          "(Material tab, optional properties)")
    if out["options"]["above_tg"] and "cte" in chosen:
        if not any(t.get("E_r") and t.get("alpha2") for t in [out["matrix"]["props"]] + [p["props"] for p in phases]):
            errors.append("Elasticity above Tg needs E above Tg and the CTE above Tg of at least one material "
                          "(Material tab, optional properties)")
    if out["rve"]["arrangement"] == "sheared":
        parts_ = [p for p in phases if p["shape"] != G.NETWORK and p["vf"] > 0]
        if out["rve"]["film"]:
            errors.append("The arrangement after shear flow is made in a periodic cube, not in a thin film")
        if out["matrix"]["void"]:
            errors.append("The arrangement after shear flow needs a liquid matrix (the resin); the matrix here is a gas")
        if not parts_:
            errors.append("The arrangement after shear flow needs particles")
        if any(p["shape"] != "sphere" for p in parts_):
            errors.append("The arrangement after shear flow is made for spheres; "
                          + ", ".join(p["name"] for p in parts_ if p["shape"] != "sphere") + " are not spheres")
        if any(p["shape"] == G.NETWORK and p["vf"] > 0 for p in phases):
            errors.append("The arrangement after shear flow is made of particles; a network phase cannot flow")
        if any(p.get("shell") or p["void"] or p.get("overlap") for p in parts_):
            errors.append("The arrangement after shear flow needs solid spheres apart from each other "
                          "(no shells, pores or permitted overlap)")
    if "viscosity" in chosen:
        if out["matrix"]["void"]:
            errors.append("Viscosity needs a liquid matrix (the resin); the matrix here is a gas")
        if any(p["shape"] == G.NETWORK and not p["void"] for p in phases):
            errors.append("Viscosity treats the fillers as particles in a liquid; a solid network phase is not a suspension")
        if out["options"]["viscosity"]["gd_max"] <= out["options"]["viscosity"]["gd_min"]:
            errors.append("The highest shear rate must exceed the lowest one")

    if errors:
        raise ValueError("\n".join(errors))
    out["labels"] = label_table(out)
    return out


def label_table(spec):
    """Region list; index in the list is the label value in the voxel map."""
    table = [{"kind": "matrix", "phase": None, "name": spec["matrix"]["name"],
              "material_id": spec["matrix"].get("material_id"), "props": dict(spec["matrix"]["props"])}]
    for i, ph in enumerate(spec["phases"]):
        table.append({"kind": "core", "phase": i, "name": ph["name"], "material_id": ph.get("material_id"),
                      "props": dict(ph["props"])})
    for i, ph in enumerate(spec["phases"]):
        if ph["shell"]:
            table.append({"kind": "shell", "phase": i, "name": f"{ph['name']} · {ph['shell']['name']}",
                          "material_id": ph["shell"].get("material_id"), "props": dict(ph["shell"]["props"])})
    return table


def label_values(table, key):
    return [float(t["props"][key]) for t in table]
