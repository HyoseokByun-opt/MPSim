"""RVE planning: voxel size, domain size and the checks that justify them.

An RVE result is only as good as three ratios, and this module makes each one
explicit instead of leaving it to habit:

    resolution   voxels across the smallest feature (particle diameter,
                 platelet thickness, shell thickness, pore size)
    size         RVE edge / largest particle (volume-equivalent diameter)
    length       RVE edge / longest particle end-to-end (fibres, platelets):
                 a fibre longer than half the box interacts with its own
                 periodic image
    count        particles per phase, so one unlucky placement cannot move
                 the answer

`plan()` recommends a grid that satisfies all of them, adopts the user's own
values when given, enlarges the box when those values are too small (and says
why), and degrades gracefully - resolution first, then box size - when the
server's voxel budget is hit. Nothing is solved here; it runs in milliseconds
on every keystroke of the form.
"""
from __future__ import annotations

import math

from . import geometry as G

# Where "size" (RVE edge / largest particle) comes from. Kanit et al. (2003,
# Int. J. Solids Struct. 40, 3647) define an RVE by the error it allows: for n
# realisations of volume V the 95 % error of the mean is 2 D(V) / (Z sqrt(n)),
# with D(V) the scatter between realisations, which falls as D^2 ~ V^-alpha;
# a box too small is also biased, not only noisy. Measured here with this
# generator and the FV / PuMA solvers (monodisperse spheres, periodic,
# contacts "apart", 10 realisations per size, 6 for E; guide 6.3):
#
#   vol%  property  contrast   L/d = 4    5       7       10     (95 % scatter
#   30    k         150        0.30 %   0.22 %  0.12 %  0.14 %   of a single
#   50    k         150       10.4 %    4.3 %   4.3 %   1.9 %    realisation)
#   30    E         127        0.82 %   0.88 %  0.92 %  0.38 %
#
# Bias against L/d = 10 was within the scatter from 5 d on (4 d: -0.9 % E;
# 2 d: +3 % k, +22 % E). alpha = 1.04 / 1.35 / 1.27. Anisotropy that only the finite box
# produces (spread of the three diagonal values): 4.3 % at 7 d, 1.9 % at 10 d
# for 50 vol%. So Standard (7 d, one realisation) is within 1 % for moderate
# loadings, and Accurate (10 d, three realisations: 2 x 1.9 % / sqrt 3 = 1.1 %
# at 50 vol%) is the level for dense, high-contrast fillers.
QUALITY = {
    "fast":     {"label": "Fast",     "res": 6,  "res_min": 4, "size": 4.0, "length": 1.5, "count": 20,  "seeds": 1, "vf_tol": 0.03},
    "standard": {"label": "Standard", "res": 10, "res_min": 6, "size": 7.0, "length": 2.0, "count": 50,  "seeds": 1, "vf_tol": 0.02},
    "accurate": {"label": "Accurate", "res": 14, "res_min": 8, "size": 10.0, "length": 2.5, "count": 100, "seeds": 3, "vf_tol": 0.01},
}

# Safety ceilings on the grid, not physics: they exist so that a mistyped voxel
# size cannot start a run that exhausts the machine. They are deliberately
# generous - a workstation handles them - and any of them can be switched off
# completely with 0 (run_web.py --max-voxels / --max-voxels-elastic /
# --max-voxels-flow), after which only the memory estimate warns. Elasticity
# keeps the lowest ceiling because it solves seven load cases, flow the next
# because Stokes carries four fields per node.
# 500^3 by default for every solver. Beyond this a solve is
# rarely practical on a workstation; the planner still warns with the memory
# estimate rather than refusing smaller machines.
DEFAULT_LIMITS = {"N_max": 500, "N_max_elastic": 500, "N_max_flow": 500}

_NICE = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def nice_floor(x):
    if x <= 0:
        return x
    e = math.floor(math.log10(x))
    base = x / 10 ** e
    best = _NICE[0]
    for v in _NICE:
        if v <= base * (1 + 1e-9):
            best = v
    return best * 10 ** e


def nice_ceil(x):
    if x <= 0:
        return x
    e = math.floor(math.log10(x))
    base = x / 10 ** e
    for v in _NICE + [10.0]:
        if v >= base * (1 - 1e-9):
            return v * 10 ** e
    return 10.0 ** (e + 1)


def even_up(n):
    n = int(math.ceil(n - 1e-9))
    return n + (n % 2)


def _status(value, target, warn_frac=0.7):
    if value >= target * (1 - 1e-9):
        return "pass"
    if value >= warn_frac * target:
        return "warn"
    return "fail"


def feature_stats(phases):
    """Per-phase numbers the planner needs, in um."""
    out = []
    for i, ph in enumerate(phases):
        vf = float(ph["vf"])
        name = ph.get("name") or f"Phase {i+1}"
        if ph["shape"] == G.NETWORK:
            pore = float(ph["size_um"]["d"])
            out.append({"index": i, "name": name, "network": True, "vf": vf,
                        "min_dim": pore, "min_dim_min": pore, "eqd_max": 2.0 * pore,
                        "extent_max": 2.0 * pore, "elongated": False, "mean_volume": None,
                        "shell": None})
            continue
        st = G.phase_size_stats(ph["shape"], ph["size_um"], ph.get("dist"))
        ar = G.aspect_ratio(ph["shape"], ph["size_um"])
        shell = ph.get("shell") or None
        t_shell = float(shell["thickness_um"]) if shell else None
        if t_shell and shell.get("scale_with_size"):
            # walls grow with the particle, so the thin ones sit on the small
            # particles: plan the grid for the wall of the 10th percentile
            d_nom = float(ph["size_um"].get("d") or 0.0)
            if d_nom > 0:
                t_shell *= min(1.0, st["min_dim_p10"] / d_nom)
        out.append({"index": i, "name": name, "network": False, "vf": vf,
                    "min_dim": st["min_dim_p10"], "min_dim_min": st["min_dim_min"],
                    "eqd_max": st["eqd_max"] + (2 * t_shell if t_shell else 0.0),
                    "extent_max": st["extent_max"] + (2 * t_shell if t_shell else 0.0),
                    "elongated": ar >= 2.0 or ar <= 0.5,
                    "mean_volume": st["mean_volume"], "shell": t_shell})
    return out


def shell_voxels(props):
    """Voxels to put across a coating or hollow-particle wall.

    Measured, not assumed: one hollow silica sphere (wall t/R = 0.2) in a
    periodic cell, refined from 2 to 10 voxels across the wall. Permittivity was
    within 0.01 % of the finest grid already at 2 voxels - the particle radius
    is tuned so the voxel volume fractions are exact, and a conduction-type
    property is set by those. Young's modulus was 0.97 % high at 2 voxels and
    0.29 % at 4: a wall carries load in bending, which goes as thickness cubed,
    so its shape matters and not only its volume.
    """
    return 4.0 if set(props or []) & {"cte", "elastic"} else 2.0


def plan(phases, rve=None, props=None, limits=None):
    """Grid recommendation plus a-priori checks.

    phases : normalised phase list (shape, size_um, dist, vf, shell)
    rve    : {auto: bool, quality, L_um, voxel_um, auto_enlarge: bool}
    props  : set of requested analysis keys (budgets differ per solver)
    """
    rve = dict(rve or {})
    props = set(props or [])
    lim = dict(DEFAULT_LIMITS)
    lim.update(limits or {})
    q = QUALITY.get(rve.get("quality", "standard"), QUALITY["standard"])
    active = [p for p in phases if float(p["vf"]) > 0]
    feats = feature_stats(active)
    notes = []

    # ---- recommended voxel size ------------------------------------------
    shell_vox = shell_voxels(props)
    if feats:
        h_res = min(f["min_dim"] / q["res"] for f in feats)
        for f in feats:
            if f["shell"]:
                h_res = min(h_res, f["shell"] / shell_vox)
        h_rec = nice_floor(h_res)
    else:
        h_rec = 1.0

    # ---- recommended domain size ----------------------------------------
    # A film fixes z at its real thickness T; the lateral edge L then has to
    # hold the material of the cube the level asks for, L² T >= (size d)³, so
    # that the film averages over as many particles as that cube (the
    # statistical error of Kanit et al. 2003 depends on the volume, D²(V) ~
    # V^-alpha, not on its shape), and never less than size × d laterally.
    film = bool(rve.get("film"))
    T_film = float(rve.get("T_um") or 0.0) if film else None
    L_req, why = 0.0, ""
    for f in feats:
        Ls = q["size"] * f["eqd_max"]
        cand = [(Ls, f"{q['size']:g} × the largest particle of '{f['name']}' ({f['eqd_max']:.3g} µm)")]
        if film and T_film and T_film < Ls:
            cand.append((math.sqrt(Ls ** 3 / T_film),
                         f"the material volume of a {Ls:.3g} µm cube ({q['size']:g} × the largest particle of "
                         f"'{f['name']}') in a {T_film:.4g} µm film"))
        if f["elongated"]:
            cand.append((q["length"] * f["extent_max"],
                         f"{q['length']:g} × the longest particle of '{f['name']}' ({f['extent_max']:.3g} µm)"))
        if not f["network"] and f["mean_volume"]:
            if film and T_film:
                Lc = math.sqrt(q["count"] * f["mean_volume"] / (f["vf"] * T_film))
            else:
                Lc = (q["count"] * f["mean_volume"] / f["vf"]) ** (1.0 / 3.0)
            cand.append((Lc, f"at least {q['count']} particles of '{f['name']}'"))
        for v, w in cand:
            if v > L_req:
                L_req, why = v, w
    if L_req <= 0:
        L_req, why = 10.0 * h_rec, "matrix only"

    # 0 or negative switches a ceiling off; math.inf then simply never binds, so
    # the degradation below is skipped and the grid the accuracy level asks for
    # is the grid that is used
    def _ceiling(key):
        v = int(lim.get(key) or 0)
        return math.inf if v <= 0 else v

    budget = _ceiling("N_max")
    if props & {"cte", "elastic"}:
        budget = min(budget, _ceiling("N_max_elastic"))
    if props & {"permeability", "acoustics"}:
        budget = min(budget, _ceiling("N_max_flow"))

    auto = rve.get("auto", True)
    h = h_rec
    L = L_req
    if not auto:
        if rve.get("voxel_um"):
            h = float(rve["voxel_um"])
        if rve.get("N"):
            # the grid size is given directly: the voxel size follows from it
            h = float(rve.get("L_um") or L_req) / max(even_up(float(rve["N"])), 8)
        if rve.get("L_um"):
            L_user = float(rve["L_um"])
            if L_user < L_req * (1 - 1e-9) and rve.get("auto_enlarge", True):
                notes.append(f"The RVE edge length was enlarged from {L_user:.4g} µm to {L_req:.4g} µm ({why}).")
                L = L_req
            else:
                L = L_user
    N = even_up(L / h)

    # ---- voxel budget ---------------------------------------------------
    capped = False
    if N > budget:
        h_min_res = (min(f["min_dim"] for f in feats) / q["res_min"]) if feats else h
        h_try = nice_ceil(L / budget)
        if h_try <= h_min_res * (1 + 1e-9):
            notes.append(f"The grid of {N}³ voxels exceeds the server limit of {budget}³; the voxel size was increased "
                         f"from {h:.4g} µm to {h_try:.4g} µm while keeping the minimum resolution.")
            h = h_try
            N = even_up(L / h)
        if N > budget:
            h_new = max(h, nice_floor(h_min_res)) if feats else h
            N = budget - (budget % 2)
            L_new = N * h_new
            notes.append(f"Because of the server limit of {budget}³ voxels, the RVE was reduced from {L:.4g} µm to "
                         f"{L_new:.4g} µm. The size criterion is not met, so the results are indicative only. "
                         f"Larger particles or the 'Fast' accuracy level resolve this.")
            h = h_new
            capped = True
    N = max(N, 8)
    L = N * h
    Nz = max(2, int(round(T_film / h))) if film else N
    T = Nz * h
    if film:
        names = {"cte": "elasticity and thermal expansion", "permeability": "permeability",
                 "tortuosity": "diffusion", "acoustics": "acoustics"}
        stack = [v for k, v in names.items() if k in props]
        if stack:
            notes.append("Film: conduction and permittivity are solved with the film's own faces (insulated along "
                         "the film, between two plates through it). " + ", ".join(stack).capitalize()
                         + " treat the film as a stack of identical films bonded face to face (periodic in z).")

    # ---- checks ---------------------------------------------------------
    checks = []
    if film:
        for f in feats:
            if f["network"]:
                continue
            # The thickness is the film's, not a choice of RVE size: two or
            # three particles across it are allowed (the size criterion is
            # applied laterally only). It fails only when no particle fits.
            skin = h if rve.get("skin_um") is None else float(rve["skin_um"])
            T_in = max(T - 2.0 * skin, 1e-12)                  # room left inside the matrix skins
            tr = T_in / f["eqd_max"]
            d_mean = (6.0 * f["mean_volume"] / math.pi) ** (1.0 / 3.0) if f.get("mean_volume") else f["eqd_max"]
            tm = T_in / max(d_mean, 1e-12)
            checks.append({"id": f"film_{f['index']}", "group": "Film",
                           "label": f"{f['name']}: film thickness / largest particle", "value": tr,
                           "target": 1.0, "unit": "×",
                           "status": "fail" if tm < 1.0 else "pass",
                           "note": (f"{T_in:.4g} µm inside the matrix skins ({skin:.3g} µm at each face) / "
                                    f"{f['eqd_max']:.3g} µm: about {tm:.1f} particle diameters across the "
                                    f"film. The thickness is fixed by the film; the RVE size criterion applies in x and y. "
                                    + ("Particles of the size distribution larger than the film are drawn as thick as "
                                       "the film allows. " if tr < 1.0 <= tm else "")
                                    + ("Fewer than two layers: the result is the film's own, strongly anisotropic "
                                       "property, not a bulk value." if tm < 2.0 else ""))})
        checks.append({"id": "film_vox", "group": "Film", "label": "Voxels across the film thickness",
                       "value": Nz, "target": 4, "unit": "voxel", "status": "pass" if Nz >= 4 else "warn",
                       "note": f"{T:.4g} µm at {h:.3g} µm per voxel"})
    for f in feats:
        nm = f["name"]
        res_v = f["min_dim"] / h
        checks.append({"id": f"res_{f['index']}", "group": "Resolution",
                       "label": f"{nm}: voxels across the smallest dimension",
                       "value": res_v, "target": q["res"], "unit": "voxel",
                       "status": ("pass" if res_v >= q["res"] - 1e-9 else
                                  "warn" if res_v >= q["res_min"] - 1e-9 else "fail"),
                       "note": f"smallest dimension (P10) {f['min_dim']:.3g} µm / voxel {h:.3g} µm"})
        if f["shell"]:
            sv = f["shell"] / h
            checks.append({"id": f"shell_{f['index']}", "group": "Resolution",
                           "label": f"{nm}: voxels across the shell thickness", "value": sv,
                           "target": shell_vox, "unit": "voxel",
                           "status": _status(sv, shell_vox, 0.5),
                           "note": ("elasticity needs more: a wall carries load in bending"
                                    if shell_vox > 2.0 else "")})
        sr = L / f["eqd_max"]
        checks.append({"id": f"size_{f['index']}", "group": "RVE size",
                       "label": f"{nm}: {'lateral edge' if film else 'RVE edge'} / largest particle", "value": sr,
                       "target": q["size"], "unit": "×", "status": _status(sr, q["size"]),
                       "note": f"{L:.4g} µm / {f['eqd_max']:.3g} µm"})
        if f["elongated"]:
            lr = L / f["extent_max"]
            checks.append({"id": f"len_{f['index']}", "group": "RVE size",
                           "label": f"{nm}: RVE edge / longest particle", "value": lr,
                           "target": q["length"], "unit": "×", "status": _status(lr, q["length"]),
                           "note": "A particle must not touch its own periodic image"})
        if not f["network"] and f["mean_volume"]:
            n_est = f["vf"] * L * L * T / f["mean_volume"]
            checks.append({"id": f"count_{f['index']}", "group": "Statistics",
                           "label": f"{nm}: expected number of particles", "value": n_est,
                           "target": q["count"], "unit": "", "status": _status(n_est, q["count"], 0.4),
                           "note": ""})
    checks.append({"id": "budget", "group": "Problem size", "label": "Grid size",
                   "value": N, "target": (None if budget == math.inf else budget),
                   "unit": "voxels per edge",
                   "status": "fail" if capped else "pass",
                   "note": (f"{N} × {N} × {Nz} = {N*N*Nz/1e6:.2f} M voxels (film)" if film else
                            f"{N}³ = {N**3/1e6:.2f} M voxels")
                           + ("" if budget == math.inf else f" · server limit {budget}³")})

    # feasibility of the loading
    # With contacts = "separate" (v3) non-overlapping particles are kept half a
    # voxel apart, and a minimum gap adds to that; either inflates the volume
    # they exclude above the volume they fill. contacts = "apart" only asks for
    # a separation where it fits and lets particles touch elsewhere, and
    # "keep" never separates, so for them only a gap the user asked for counts.
    sep = 0.5 * h if rve.get("contacts") == "separate" else 0.0
    excl = 0.0
    for f in feats:
        ph = active[feats.index(f)]
        if f["network"] or ph.get("overlap") or not f["min_dim"]:
            continue
        gap = sep + float(ph.get("gap_frac", 0.0) or 0.0) * f["min_dim"]
        excl += f["vf"] * ((f["min_dim"] + gap) / f["min_dim"]) ** 3
    if excl > 0.62 and excl > sum(f["vf"] for f in feats if not f["network"]) + 1e-9:
        checks.append({"id": "excl", "group": "Packing", "label": "Packing fraction including the separation gap",
                       "value": excl, "target": 0.62, "unit": "", "status": "fail" if excl > 0.68 else "warn",
                       "note": "The separation between particles inflates the packing demand above the random "
                               "close packing of spheres (≈ 0.64). A smaller gap, contacts = apart, larger particles "
                               "or permitted overlap resolves it."})

    hard = [f for f in feats if not f["network"]]
    total = sum(f["vf"] for f in active if not f.get("overlap"))
    # dense fillers scatter far more between realisations (see QUALITY)
    if (hard and total >= 0.40 and rve.get("quality", "standard") != "accurate" and not rve.get("n_seeds")
            and props & {"thermal", "electrical", "dielectric", "magnetic", "emi", "cte", "elastic"}):
        notes.append(f"At {100*total:.0f} vol% of particles, realisations of a 7-diameter RVE scattered by ±4 % "
                     f"(50 vol%, contrast 150); the Accurate level (10 diameters, 3 realisations) narrowed it "
                     f"to ±1.1 %. Accurate, or 2-3 realisations, is advisable for reported values.")
    if hard and total > 0.60:
        checks.append({"id": "loading", "group": "Packing", "label": "Non-overlapping packing fraction",
                       "value": total, "target": 0.60, "unit": "", "status": "warn",
                       "note": "Close to the random close packing of monodisperse spheres (≈ 0.64). "
                               "A broader (log-normal) size distribution facilitates packing."})
    # contacts = "apart" keeps particles SEP_APART voxels apart as far as the
    # loading allows. Where it cannot, particles touch at voxel level, and at a
    # high property contrast the FE and FV solutions then differ widely (guide
    # 5.5: +121 % at 50 vol%, 16 voxels per diameter; +19 % once the gap fits).
    # The generator cuts the gap so that the grown particles fill no more than
    # its fast-packing fraction; the same rule tells the voxel size needed.
    if rve.get("contacts", "apart") == "apart" and props & {"thermal", "electrical", "dielectric", "magnetic",
                                                            "emi", "cte", "elastic"}:
        from .generate import SEP_APART, _FAST_PACK
        sep_ph = [(f, active[feats.index(f)]) for f in feats
                  if not f["network"] and not active[feats.index(f)].get("overlap") and f["mean_volume"]]
        if sep_ph:
            lim = _FAST_PACK["sphere" if all(ph["shape"] == "sphere" for _, ph in sep_ph) else "other"]
            d_eq = [(6.0 * f["mean_volume"] / math.pi) ** (1.0 / 3.0) for f, _ in sep_ph]

            def grown(g):
                return sum(f["vf"] * ((d + g) / d) ** 3 for (f, _), d in zip(sep_ph, d_eq))

            if grown(0.0) >= lim:
                checks.append({"id": "apart", "group": "Packing", "label": "Particles kept apart (contacts = apart)",
                               "value": grown(0.0), "target": lim, "unit": "", "status": "warn",
                               "note": "The loading leaves no room for a gap between the particles: they will touch "
                                       "at voxel level. With a filler much more conductive or stiffer than the matrix "
                                       "the result then depends on how finely the contacts are resolved (a converged "
                                       "simple-cubic case with necks: +5 % at 64 voxels per cell); a contact "
                                       "resistance or a finer voxel narrows it (guide 5.5)."})
            elif grown(SEP_APART * h) > lim:
                lo, hi = 0.0, SEP_APART * h
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if grown(mid) <= lim else (lo, mid)
                h_need = lo / SEP_APART
                checks.append({"id": "apart", "group": "Packing", "label": "Voxel size that keeps every particle apart",
                               "value": h, "target": h_need, "unit": "µm", "status": "warn",
                               "note": f"A {SEP_APART:g}-voxel gap between the particles fits at a voxel size of "
                                       f"{h_need:.3g} µm or less ({min(d_eq) / h_need:.0f} voxels across the "
                                       f"smallest particle). At {h:.3g} µm some particles will touch at voxel "
                                       f"level; with a filler much more conductive or stiffer than the matrix a "
                                       f"one-voxel contact carries far more heat than the thin gap it replaces "
                                       f"(guide 5.5)."})

    for f in feats:
        ph = active[feats.index(f)]
        if f["elongated"] and not ph.get("overlap") and f["vf"] > 0.25:
            checks.append({"id": f"rsa_{f['index']}", "group": "Packing",
                           "label": f"{f['name']}: non-overlapping packing of non-spherical particles", "value": f["vf"],
                           "target": 0.25, "unit": "", "status": "warn",
                           "note": "Random sequential addition may not reach this fraction. "
                                   "Aligned orientation or permitted overlap should be considered."})

    mem = memory_estimate(N, props, nz=Nz)
    # Once the ceilings are raised or switched off, the grid is no longer what
    # stops a run - memory is. So it is reported in the same place and with the
    # actual number, which leaves the decision with whoever knows the machine
    # instead of refusing on their behalf.
    peak = max(mem.values()) if mem else 0.0
    # Judged against this machine's RAM rather than a fixed figure: 40 GB is
    # routine on a 512 GB workstation and impossible on a 16 GB laptop.
    ram = machine_ram_gb()
    warn_at, fail_at = (0.5 * ram, 0.9 * ram) if ram else (8.0, 32.0)
    checks.append({"id": "memory", "group": "Problem size", "label": "Estimated peak memory",
                   "value": peak, "target": warn_at, "unit": "GB",
                   "status": "pass" if peak <= warn_at else ("warn" if peak <= fail_at else "fail"),
                   "note": " · ".join(f"{k} {v:.1f} GB" for k, v in mem.items())
                           + (f". This machine has {ram:.0f} GB (warning above half of it, failure above 90 %)"
                              if ram else "")
                           + ". Every solve that runs side by side needs its own share. It is counted from the "
                             "arrays each solver allocates, not measured, so allow some margin."})
    return {
        "N": int(N), "h_um": float(h), "L_um": float(L),
        "film": film, "Nz": int(Nz), "T_um": float(T),
        "recommended": {"h_um": float(h_rec), "L_um": float(L_req), "reason": why},
        "quality": rve.get("quality", "standard"), "quality_label": q["label"],
        "seeds": int(rve.get("n_seeds") or q["seeds"]),
        "vf_tol": q["vf_tol"],
        "features": feats, "checks": checks, "notes": notes,
        # None rather than math.inf: Infinity is not valid JSON and would reach
        # the browser as a parse error
        "capped": capped, "budget": (None if budget == math.inf else int(budget)),
        "memory_gb": mem,
    }


def machine_ram_gb():
    """Installed RAM of this machine in GB, or None when it cannot be read."""
    try:
        import psutil
        return psutil.virtual_memory().total / 2 ** 30
    except Exception:                                                  # noqa: BLE001
        return None


def memory_estimate(N, props, nz=None):
    """Rough peak memory per solver, GB. From the array counts of each
    solver, not from a probe run. nz: a film's voxels in z (N otherwise)."""
    n = N * N * (N if nz is None else nz)
    out = {"structure": n * (4 + 1 + 8) / 1e9}
    # the multi-core FE path adds its DOF incidence tables: ~80 bytes per
    # voxel for conduction, ~240 for elasticity. Conduction counts the larger
    # of its two methods (PuMA FE ~230 bytes per voxel, FV ~120).
    if props & {"thermal", "electrical", "dielectric", "magnetic", "emi"}:
        out["conduction"] = n * 230 / 1e9
    if props & {"cte", "elastic"}:
        out["thermo-elasticity (PuMA FE)"] = n * (3 * 24 * 8 * 4 + 240) / 1e9
    if props & {"permeability", "acoustics"}:
        out["Stokes flow (PuMA FE)"] = n * 4 * 24 * 8 * 3 / 1e9
    if props & {"porosimetry", "pore_network", "morphology", "grains"}:
        out["structure analysis (PoreSpy)"] = n * 48 / 1e9
    return out


def post_checks(plan_, gen_info, quality="standard"):
    """Checks on the realised structure."""
    q = QUALITY.get(quality, QUALITY["standard"])
    checks = []
    for i, p in enumerate(gen_info["phases"]):
        tgt = p["vf_target"]
        if tgt <= 0:
            continue
        err = abs(p["vf_voxel"] - tgt) / tgt
        checks.append({"id": f"vf_{i}", "group": "Generated structure", "label": f"{p['name']}: volume fraction error",
                       "value": err, "target": q["vf_tol"], "unit": "rel",
                       "status": "pass" if err <= q["vf_tol"] else ("warn" if err <= 3 * q["vf_tol"] else "fail"),
                       "note": f"target {100*tgt:.2f} % → voxels {100*p['vf_voxel']:.2f} %"
                               + (" · packing limit reached (jammed)" if p.get("jammed") else "")})
        cov = p.get("octant_cov", 0.0)
        n = max(p.get("n_particles", 1), 1)
        expected = 1.0 / math.sqrt(max(n / 8.0, 1.0))
        checks.append({"id": f"oct_{i}", "group": "Generated structure", "label": f"{p['name']}: homogeneity over octants (CoV)",
                       "value": cov, "target": max(0.15, 2.0 * expected * math.sqrt(max(1 - tgt, 0.05))),
                       "unit": "", "status": "pass" if cov <= max(0.15, 2.0 * expected) else "warn",
                       "note": "Coefficient of variation of the volume fraction over the eight octants of the RVE"})
    if gen_info.get("contacts_mode") == "separate":
        checks.append({"id": "contacts", "group": "Generated structure", "label": "Particle contacts separated by a matrix voxel",
                       "value": gen_info.get("contacts_removed", 0), "target": 0, "unit": "", "status": "pass",
                       "note": "contacts = separate (v3 behaviour): touching particles lose a voxel at the contact"})
    else:
        touch = gen_info.get("touching_pairs", 0) or 0
        faces = gen_info.get("contact_faces", 0) or 0
        apart = gen_info.get("contacts_mode") == "apart"
        checks.append({"id": "contacts", "group": "Generated structure",
                       "label": "Touching particles (voxel neighbour pairs of two particles)",
                       "value": touch, "target": 0, "unit": "",
                       "status": "pass" if touch == 0 or not apart else "warn",
                       "note": ("No two particles touch at this resolution" if touch == 0 else
                                f"{faces:,} of them across a voxel face. "
                                + ("The loading is too dense to keep every particle apart. " if apart else "")
                                + "Heat crosses a contact face (the default finite-volume solver) or also an edge "
                                  "or corner (PuMA's finite elements); at a high conductivity contrast the result "
                                  "then depends on how finely the contacts are resolved. A contact resistance acts "
                                  "on the faces.")})
    conf = gen_info.get("voxel_conflicts", 0)
    if conf:
        nvox = gen_info.get("N", 1) ** 3
        rel = conf / nvox
        checks.append({"id": "overlap", "group": "Generated structure", "label": "Voxels claimed by two particles",
                       "value": rel, "target": 1e-3, "unit": "rel",
                       "status": "pass" if rel <= 1e-3 else ("warn" if rel <= 5e-3 else "fail"),
                       "note": f"{conf:,} voxels; kept by the first particle (clump packing does not cover sharp corners)"})
    shr = gen_info.get("shrunk") or {}
    if shr.get("n"):
        n_all = sum(int(p.get("n_particles", 0)) for p in gen_info.get("phases", [])) or 1
        checks.append({"id": "shrunk", "group": "Generated structure",
                       "label": "Particles shrunk to clear corner overlaps",
                       "value": shr["n"] / n_all, "target": 0.0, "unit": "rel",
                       "status": "warn" if shr["min"] >= 0.9 else "fail",
                       "note": f"{shr['n']} of {n_all} particles, to {100 * shr['min']:.0f} % of their size at most "
                               f"({100 * shr['mean']:.1f} % on average), shape and orientation kept: their corners "
                               f"were caught in each other and pushing and turning did not free them. The volume "
                               f"fraction table shows what the phases reached."})
    ps = gen_info.get("pack_scale")
    if ps is not None and ps < 1.0:
        checks.append({"id": "pack", "group": "Generated structure", "label": "Particle size reached by the dense packing",
                       "value": ps, "target": 1.0, "unit": "", "status": "warn" if ps > 0.95 else "fail",
                       "note": "Below 1 the particles were packed smaller than requested: the target is above what "
                               "these shapes can reach without overlap"})
    return checks
