"""Grain and filler analysis - particle statistics of the generated structure.

Two sources, reported side by side:

* the **particle list** of the generator (exact geometry of every particle):
  count, number density, equivalent-diameter distribution, sphericity,
  aspect ratio, orientation tensor and Hermans order parameters, nearest-
  neighbour spacing and the Clark-Evans dispersion index;
* the **voxel image** (what the solvers see): connected clusters, periodic
  percolation, PuMA structure-tensor orientation of elongated phases and the
  PoreSpy local thickness of the matrix between particles (the matrix
  ligament or film thickness that governs conduction in highly filled
  composites).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import geometry as G
from . import morphology as MO
from . import shapes as SH


def _quantiles(v, w=None):
    v = np.asarray(v, float)
    if v.size == 0:
        return {}
    order = np.argsort(v)
    v = v[order]
    w = np.ones_like(v) if w is None else np.asarray(w, float)[order]
    c = np.cumsum(w) / w.sum()
    q = lambda p: float(np.interp(p, c, v))
    return {"d10": q(0.1), "d50": q(0.5), "d90": q(0.9), "mean": float(np.average(v, weights=w))}


def _aspect(code, a, b):
    if code == 0:
        return np.ones_like(a)
    if code == 1:
        return b / a
    if code == 2:
        return b / a
    return (b + a) / a


def analyse(spec, info, labels, table, h_um, want_structure_tensor=True, log=None):
    from scipy.spatial import cKDTree
    from .solvers.conduction import wrapping_axes
    if info.get("image"):
        return analyse_image(spec, info, labels, table, h_um, log=log)
    t0 = time.time()
    parts = info["particles"]
    L = float(info["L_um"])
    box, film = particle_box(info)
    vol_box = float(np.prod(box))
    out = {"phases": {}}
    for li, t in enumerate(table):
        if t["kind"] != "core":
            continue
        i = t["phase"]
        ph = spec["phases"][i]
        if ph["shape"] in G.NON_PARTICLE:
            continue
        sel = parts["phase"] == i
        n = int(sel.sum())
        entry = {"name": t["name"], "shape": ph["shape"], "n_particles": n}
        if n == 0:
            out["phases"][str(li)] = entry
            continue
        codes = parts["codes"][sel]
        Ps = parts["P"][sel]
        cen = _wrapped(parts["centers"][sel], box, film)
        ax = parts["axes"][sel]
        code = int(codes[0])
        meta = info.get("poly_meta")
        V = np.array([SH.volume(int(c), P, meta) for c, P in zip(codes, Ps)])
        A = np.array([SH.surface(int(c), P, meta) for c, P in zip(codes, Ps)])
        a = np.array([SH.circumradius(int(c), P) for c, P in zip(codes, Ps)])
        eqd = (6.0 * V / math.pi) ** (1.0 / 3.0)
        psi = math.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0) / A
        ar = np.array([SH.aspect(int(c), P, meta) for c, P in zip(codes, Ps)])
        entry.update({
            "phase": int(i),
            "number_density_per_mm3": n / (vol_box * 1e-9),
            "eq_diameter_um": {"number": _quantiles(eqd), "volume": _quantiles(eqd, V)},
            # both diameters of the particles as built, by number and by volume (as the realised sizes give them)
            "diameters": {"circumscribed": size_stats(2.0 * a, V), "volume_equivalent": size_stats(eqd, V)},
            "sphericity": float(np.mean(psi)),
            "aspect_ratio": float(np.mean(ar)),
            "specific_surface_true_1pm": float(A.sum() / vol_box * 1e6),
        })
        hist, edges = np.histogram(eqd, bins=14)
        vhist, _ = np.histogram(eqd, bins=edges, weights=V)
        entry["size_histogram"] = {"center_um": (0.5 * (edges[1:] + edges[:-1])).tolist(),
                                   "number": (hist / hist.sum()).tolist(),
                                   "volume": (vhist / max(vhist.sum(), 1e-300)).tolist()}
        if code != 0:
            T = (ax.T @ ax) / n
            ev = np.sort(np.linalg.eigvalsh(T))[::-1]
            entry["orientation_tensor"] = T.tolist()
            entry["orientation_eigenvalues"] = ev.tolist()
            entry["hermans"] = {d: float(1.5 * T[k, k] - 0.5) for k, d in enumerate("xyz")}
        if n >= 2:
            # a film is periodic in x and y only: its z sits in the middle third of a box three times as tall
            zb = 3.0 * box[2] if film else box[2]
            q = cen + ([0.0, 0.0, box[2]] if film else 0.0)
            tree = cKDTree(q, boxsize=np.array([box[0], box[1], zb]) * (1 + 1e-12))
            dist, idx = tree.query(q, k=2)
            nn = dist[:, 1]
            rho = n / vol_box
            entry["nearest_neighbour_um"] = {"mean": float(nn.mean()), "cv": float(nn.std() / nn.mean())}
            entry["clark_evans_index"] = float(nn.mean() / (0.55396 * rho ** (-1.0 / 3.0)))
            if code == 0:
                gap = nn - a - a[idx[:, 1]]
                entry["surface_gap_um"] = {"mean": float(gap.mean()), "min": float(gap.min()),
                                           "d10": float(np.percentile(gap, 10))}
        ids = [li] + [j for j, u in enumerate(table) if u["kind"] == "shell" and u["phase"] == i]
        mask = np.isin(labels, ids)
        cl = MO.clusters(mask)
        entry["voxel"] = {"volume_fraction": float(mask.mean()), "clusters": cl,
                          "particles_per_cluster": n / max(cl["n_clusters"], 1),
                          "percolates": dict(zip("xyz", wrapping_axes(mask)))}
        if want_structure_tensor and code != 0 and (np.mean(ar) >= 2.0):
            try:
                from .solvers import puma_backend as PB
                if PB.available():
                    Ast = PB.orientation_tensor(labels, ids, h_um * 1e-6)
                    if Ast is not None:
                        entry["orientation_tensor_image"] = Ast.tolist()
            except Exception as e:                                  # noqa: BLE001
                entry["orientation_tensor_image_error"] = str(e)
        del mask
        out["phases"][str(li)] = entry
    matrix = np.asarray(labels) == 0
    lt = None
    if matrix.any() and not matrix.all():
        lt = MO.local_thickness(matrix)
        sd = MO.size_distribution(lt, matrix, h_um, bins=14)
        out["matrix_ligament"] = sd
    out["seconds"] = time.time() - t0
    if log:
        log(f"    grain analysis: {sum(p.get('n_particles', 0) for p in out['phases'].values())} particles "
            f"({time.time()-t0:.1f} s)")
    return out, lt


# =========================================================================
# realised sizes against the input, and the particle files (5.6.6)
# =========================================================================
QTY_LABEL = {"eqd": ("Volume-equivalent diameter", "um")}


def particle_box(info):
    """The box (x, y, z extents in um) the particles lie in and whether it is a film (z between walls)."""
    h = float(info.get("h_um") or 0.0)
    shp = info.get("shape")
    L = float(info["L_um"])
    if shp is not None and h > 0:
        box = np.array([float(v) * h for v in shp])
    else:
        box = np.array([L, L, float(info.get("T_um") or L)])
    return box, bool(info.get("film"))


def _wrapped(C, box, film):
    """Centres into the box (z kept for a film)."""
    C = np.array(C, float, copy=True)
    C[:, :2] = np.mod(C[:, :2], box[:2])
    C[:, 2] = np.clip(C[:, 2], 0.0, box[2]) if film else np.mod(C[:, 2], box[2])
    return C


def _ks(xr, xi):
    """Two-sample Kolmogorov-Smirnov test of the realised against the input sample: the largest gap D between
    their cumulative number distributions and its p-value (a small p: the realised particles are unlikely to be
    a sample of the input distribution)."""
    xr, xi = np.asarray(xr, float), np.asarray(xi, float)
    if len(xr) < 2 or np.ptp(xi) <= 1e-12 * max(abs(float(xi.mean())), 1e-300):
        return {"D": 0.0, "p": 1.0}
    # both on a grid of 0.1 % of the median: a size that takes a few values only (irregular grains of one size
    # and 8 shapes - 8 volume-equivalent diameters) came out 0.004 % above the input's from the factor that meets
    # the target volume, every step of the cumulative distribution just beside the input's, and the test read
    # it as p 4e-68
    q = 1e-3 * max(float(np.median(np.abs(xi))), 1e-300)
    xr, xi = np.round(xr / q) * q, np.round(xi / q) * q
    from scipy.stats import ks_2samp
    r = ks_2samp(xr, xi)
    return {"D": float(r.statistic), "p": float(r.pvalue)}


def size_checks(report):
    """One check per particle phase: does the structure hold the size distribution entered? Judged on the size
    entered (d, or the volume-equivalent diameter): the number mean within three of its standard errors
    (cv / sqrt(n)) or 2 %, and the two-sample test not rejecting at 1 %."""
    out = []
    for i, e in sorted((report or {}).items(), key=lambda kv: int(kv[0])):
        dims = {d["key"]: d for d in e.get("dimensions", [])}
        d = dims.get(e.get("basis") or "d") or dims.get("d")
        if not d or not d["real"].get("count"):
            continue
        r, s = d["real"], d["input"]
        err = r["number_mean"] / s["number_mean"] - 1.0 if s.get("number_mean") else 0.0
        se = (s.get("cv") or 0.0) / math.sqrt(max(r["count"], 1))
        tol = max(3.0 * se, 0.02)
        p = d.get("ks", {}).get("p", 1.0)
        ok = abs(err) <= tol and p >= 0.01
        others = [f"{x['label'].lower()} {x['real']['number_mean']:.4g} (input {x['input']['number_mean']:.4g})"
                  for x in e["dimensions"] if x is not d and x["key"] != "eqd" and x["input"].get("cv")]
        out.append({"id": f"psd_{i}", "group": "Particle sizes",
                    "label": f"{e['name']}: the size distribution entered is realised",
                    "value": abs(err), "target": tol, "unit": "rel", "status": "pass" if ok else "warn",
                    "note": f"{d['label']}: number mean {r['number_mean']:.4g} µm over {r['count']} particles against "
                            f"{s['number_mean']:.4g} µm drawn from the input ({100 * err:+.1f} %; three standard errors "
                            f"{100 * 3 * se:.1f} %), CV {r['cv']:.3f} against {s['cv']:.3f}, D50 by volume "
                            f"{r['d50v']:.4g} against {s['d50v']:.4g} µm; two-sample test D = {d.get('ks', {}).get('D', 0):.3f}, "
                            f"p = {p:.3g}" + ("; " + ", ".join(others) if others else "")
                            + ("" if ok else ". A small box holds few of the largest particles (some may not fit), "
                               "and particles drawn smaller to reach a high loading shift the sizes down; "
                               "Grain analysis shows the distributions side by side")})
    return out


def _poly_vertices(planes, nf):
    """The vertices of a convex polyhedron from its planes n.x <= d (structures kept before 5.6.6 hold no
    vertices)."""
    from scipy.spatial import HalfspaceIntersection
    pl = np.asarray(planes, float)[:int(nf)]
    hs = np.column_stack([pl[:, :3], -pl[:, 3]])
    v = HalfspaceIntersection(hs, np.zeros(3)).intersections
    keep = []
    for p in v:
        if not any(np.linalg.norm(p - q) < 1e-9 for q in keep):
            keep.append(p)
    return np.array(keep)


def size_stats(x, w):
    """Statistics of one dimension over particles: by number (mean, standard deviation, CV, D10/D50/D90,
    extremes) and by volume (w the particles' volumes: the volume-weighted mean - D[4,3] for spheres - and
    D10v/D50v/D90v)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if x.size == 0:
        return {"count": 0}
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    qv = lambda p: float(np.interp(p, cw, x))                                # noqa: E731
    mean = float(x.mean())
    sd = float(x.std())
    if sd <= 1e-9 * abs(mean):
        sd = 0.0                                     # a dimension the same for every particle (rounding only)
    return {"count": int(x.size), "number_mean": mean, "std": sd, "cv": sd / abs(mean) if mean else 0.0,
            "d10": float(np.quantile(x, 0.1)), "d50": float(np.quantile(x, 0.5)), "d90": float(np.quantile(x, 0.9)),
            "min": float(x[0]), "max": float(x[-1]),
            "volume_mean": float(np.sum(x * w) / w.sum()), "d10v": qv(0.1), "d50v": qv(0.5), "d90v": qv(0.9)}


def _hist(x, w, edges):
    """Number and volume shares (%) per bin and their cumulative sums."""
    hn, _ = np.histogram(x, bins=edges)
    hv, _ = np.histogram(x, bins=edges, weights=w)
    pn = 100.0 * hn / max(hn.sum(), 1)
    pv = 100.0 * hv / max(hv.sum(), 1e-300)
    return pn, pv, np.cumsum(pn), np.cumsum(pv)


def size_report(spec_phases, gen_phases, ginfo, n_input=100000):
    """The sizes the structure holds against those its input draws, per particle phase and per dimension
    (shapes.DIMENSIONS and the volume-equivalent diameter): size_stats of both and their number and volume
    shares in common bins (log bins for lengths, linear for ratios). The input is sampled as the generator
    samples it (shapes.draw_sample, with the structure's own polyhedron shapes), so the two compare the same
    quantity. 'basis' names the dimension the entered size is (d, or eqd for a volume-equivalent size).
    100,000 input draws (5.9.0; 20,000 up to 5.8.0): the 20,000 of the fixed seed put the mean of log-normal
    irregular grains 0.29 % above the distribution's, and every structure read 0.4 % small against them."""
    parts = ginfo.get("particles") or {}
    if not len(parts.get("codes", [])):
        return {}
    pha = np.asarray(parts["phase"])
    meta = ginfo.get("poly_meta")
    out = {}
    for i, gp in enumerate(gen_phases):
        if gp["shape"] in G.NON_PARTICLE:
            continue
        sel = np.flatnonzero(pha == i)
        if not len(sel):
            continue
        shape = gp["shape"]
        code = int(parts["codes"][sel[0]])
        Ps = np.asarray(parts["P"])[sel]
        real = SH.particle_dims(shape, code, Ps, meta)
        ids = sorted({int(p[1]) for p in Ps}) if code == 6 else None
        c_in, P_in, meta_in = SH.draw_sample(shape, gp["size_um"], gp.get("dist"), gp.get("dists"),
                                             gp.get("size_basis") or "shape", n=n_input, poly_meta=meta if ids else None,
                                             ids=ids)
        inp = SH.particle_dims(shape, c_in, P_in, meta_in)
        dims = [(k, lab, u) for k, lab, u in SH.DIMENSIONS.get(shape, [("d", "Size d", "um")]) if k in real]
        dims.append(("eqd", QTY_LABEL["eqd"][0], "um"))
        sp = spec_phases[i] if i < len(spec_phases) else {}
        entry = {"name": sp.get("name") or gp.get("name"), "shape": shape, "n_particles": int(len(sel)),
                 "basis": "eqd" if gp.get("size_basis") == "volume" else "d",
                 "shell_um": float(((sp.get("shell") or {}).get("thickness_um")) or 0.0),
                 "own": sorted((gp.get("dists") or {}).keys()), "dimensions": []}
        for key, lab, unit in dims:
            xr, xi = real[key], inp[key]
            lo, hi = float(min(xr.min(), xi.min())), float(max(xr.max(), xi.max()))
            # as many bins as the realised particles fill (Rice rule, 2 n^(1/3)), 8 to 30
            nb = int(min(30, max(8, round(2.0 * len(xr) ** (1.0 / 3.0)))))
            if unit == "um" and lo > 0 and hi > lo * 1.0001:
                edges = np.geomspace(lo / 1.0001, hi * 1.0001, nb + 1)
            elif hi > lo * (1 + 1e-9) + 1e-12:
                edges = np.linspace(lo - 1e-9 * abs(lo), hi + 1e-9 * abs(hi), nb + 1)
            else:
                edges = np.array([lo * 0.999 - 1e-12, hi * 1.001 + 1e-12])
            rn, rv, rcn, rcv = _hist(xr, real["volume"], edges)
            in_, iv, icn, icv = _hist(xi, inp["volume"], edges)
            entry["dimensions"].append({
                "key": key, "label": lab, "unit": unit,
                "real": size_stats(xr, real["volume"]), "input": size_stats(xi, inp["volume"]),
                "ks": _ks(xr, xi),
                "bins": {"edges": edges.tolist(), "mid": np.sqrt(edges[1:] * edges[:-1]).tolist() if edges[0] > 0
                         else (0.5 * (edges[1:] + edges[:-1])).tolist(),
                         "real_number": rn.tolist(), "real_volume": rv.tolist(),
                         "real_number_cum": rcn.tolist(), "real_volume_cum": rcv.tolist(),
                         "input_number": in_.tolist(), "input_volume": iv.tolist(),
                         "input_number_cum": icn.tolist(), "input_volume_cum": icv.tolist()}})
        out[str(i)] = entry
    return out


PARTICLE_COLUMNS = ["id", "phase", "phase_name", "shape", "x_um", "y_um", "z_um", "d_um", "eqd_um", "volume_um3",
                    "lx_um", "ly_um", "lz_um", "length_um", "aspect_c_a", "aspect_b_a", "n_exponent", "variant",
                    "shell_um", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]


def write_image_particles(out_dir, names, seps, h_um, shape, round_ratio=0.8):
    """particles.csv (with particles_README.txt) of a structure read from an image: every particle found in it
    (separate), in the conventions of the particle files of a generated structure, so that Structure - Particles
    (CSV) draws them again or another program reads them. A phase whose particles are round (the median of the
    shortest over the longest axis of their ellipsoids of inertia at least round_ratio) is written as spheres of
    their volume-equivalent diameter; any other as the ellipsoid of each particle's second moments - the same
    volume, its three axes and their orientation (a triaxial spheroid). names: {phase index: name}; seps:
    {phase index: separate()}. Returns the files written."""
    import csv
    import os
    nx, ny, nz = (int(v) for v in shape)
    box = [nx * h_um, ny * h_um, nz * h_um]
    n_all = 0
    with open(os.path.join(out_dir, "particles.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(PARTICLE_COLUMNS)
        for i in sorted(seps):
            sp = seps[i]
            if not sp or not sp.get("n"):
                continue
            semi, axes = np.asarray(sp["semi_um"], float), np.asarray(sp["axes"], float)
            ratio = semi[:, 2] / np.maximum(semi[:, 0], 1e-12)
            spheres = float(np.median(ratio)) >= round_ratio
            for k in range(int(sp["n"])):
                n_all += 1
                c = np.mod(np.asarray(sp["centers_um"][k], float), box)
                eqd, V = float(sp["eqd_um"][k]), float(sp["vol_um3"][k])
                if spheres:
                    row = [n_all, i + 1, names.get(i, f"Phase {i + 1}"), "sphere", c[0], c[1], c[2], eqd, eqd, V,
                           eqd, eqd, eqd, None, None, None, None, None, None] + [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                else:
                    a, b, cc = semi[k]
                    row = [n_all, i + 1, names.get(i, f"Phase {i + 1}"), "spheroid", c[0], c[1], c[2], 2 * a, eqd, V,
                           2 * a, 2 * b, 2 * cc, None, cc / a, b / a, None, None, None] + [float(v) for v in axes[k].ravel()]
                wr.writerow(["" if v is None else (f"{v:.10g}" if isinstance(v, float) else v) for v in row])
    with open(os.path.join(out_dir, "particles_README.txt"), "w", encoding="utf-8") as fh:
        fh.write(PARTICLES_README.format(X=box[0], Y=box[1], Z=box[2], n=n_all, h=h_um,
                                         film="periodic in x, y and z (a particle may cross a face and continue on the "
                                              "opposite one)")
                 .replace("as MPSim built them.", "as MPSim found them in the image (touching particles separated by a "
                                                  "periodic watershed; a round phase as spheres of the volume-equivalent "
                                                  "diameter, any other as each particle's ellipsoid of inertia of the "
                                                  "same volume).")
                 .replace("The voxel structure (structure.tif) samples these particles", "Drawn again, these particles "
                          "approximate the image (structure.tif), which holds the particles' own shapes; drawn particles sample them"))
    return ["particles.csv", "particles_README.txt"]


def write_particle_files(out_dir, spec_phases, gen_phases, ginfo, report=None):
    """particles.csv: every particle as built - centre, size, shape dimensions, volume and orientation - so
    that another program can draw the same structure; polyhedra.csv: the vertices of every polyhedron shape
    used; particles_README.txt: the box and the conventions; psa.csv and psa_summary.csv: the size
    distributions realised and input (size_report). Returns the files written."""
    import csv
    import os
    parts = ginfo.get("particles") or {}
    written = []
    if not len(parts.get("codes", [])):
        return written
    meta = ginfo.get("poly_meta") or []
    box, film = particle_box(ginfo)
    codes = np.asarray(parts["codes"])
    Ps = np.asarray(parts["P"], float)
    Rs = np.asarray(parts["R"], float)
    C = _wrapped(parts["centers"], box, film)
    pha = np.asarray(parts["phase"])
    shapes_of = [gp["shape"] for gp in gen_phases]
    p_csv = os.path.join(out_dir, "particles.csv")
    with open(p_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(PARTICLE_COLUMNS)
        for j in range(len(codes)):
            code, P, i = int(codes[j]), Ps[j], int(pha[j])
            shape = shapes_of[i] if i < len(shapes_of) else "?"
            sp = spec_phases[i] if i < len(spec_phases) else {}
            hx, hy, hz = SH.local_half_extents(code, P, *(ginfo.get("planes") or SH.EMPTY_PLANES))
            if code == 6:
                hx = hy = hz = float(P[0])
            V = SH.volume(code, P, meta)
            length = 2.0 * P[1] if code == 2 else (2.0 * (P[1] + P[0]) if code == 3 else None)
            asp = (P[1] / P[0], (P[2] if P[2] > 0 else P[0]) / P[0]) if code == 1 else (None, None)
            row = [j + 1, i + 1, sp.get("name") or gen_phases[i].get("name"), shape,
                   C[j, 0], C[j, 1], C[j, 2], 2.0 * P[0], (6.0 * V / math.pi) ** (1.0 / 3.0), V,
                   2.0 * hx, 2.0 * hy, 2.0 * hz, length, asp[0], asp[1], P[3] if code == 5 else None,
                   int(P[1]) if code == 6 else None, ((sp.get("shell") or {}).get("thickness_um")) or None]
            row += [float(v) for v in Rs[j].ravel()]
            wr.writerow(["" if v is None else (f"{v:.10g}" if isinstance(v, float) else v) for v in row])
    written.append("particles.csv")
    used = sorted({int(P[1]) for c, P in zip(codes, Ps) if int(c) == 6})
    if used:
        with open(os.path.join(out_dir, "polyhedra.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["variant", "vertex", "x", "y", "z"])
            for t in used:
                vs = (meta[t] or {}).get("vertices") if t < len(meta) else None
                if not vs and ginfo.get("planes") is not None:
                    vs = _poly_vertices(ginfo["planes"][0][t], ginfo["planes"][1][t]).tolist()
                for k, v in enumerate(vs or []):
                    wr.writerow([t, k + 1] + [f"{float(x):.8g}" for x in v])
        written.append("polyhedra.csv")
    with open(os.path.join(out_dir, "particles_README.txt"), "w", encoding="utf-8") as fh:
        fh.write(PARTICLES_README.format(X=box[0], Y=box[1], Z=box[2], n=len(codes),
                                         h=float(ginfo.get("h_um") or (box[0] / max(int(ginfo.get("N") or 1), 1))),
                                         film="a film: x and y periodic, z from 0 to the thickness (walls)"
                                         if film else "periodic in x, y and z (a particle may cross a face and "
                                                      "continue on the opposite one)"))
    written.append("particles_README.txt")
    if report is None:
        report = size_report(spec_phases, gen_phases, ginfo)
    return written + write_psa_files(out_dir, report)


def write_psa_files(out_dir, report):
    """psa.csv (the distributions in bins, realised and input) and psa_summary.csv (their statistics); without an
    input (a structure read from an image) its columns stay empty."""
    import csv
    import os
    written = []
    with open(os.path.join(out_dir, "psa.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["phase", "phase_name", "dimension", "unit", "bin_low", "bin_high", "bin_mid",
                     "realised_number_pct", "realised_number_cum_pct", "realised_volume_pct", "realised_volume_cum_pct",
                     "input_number_pct", "input_number_cum_pct", "input_volume_pct", "input_volume_cum_pct"])
        for i, e in sorted(report.items(), key=lambda kv: int(kv[0])):
            for dmn in e["dimensions"]:
                b = dmn["bins"]
                inp = b.get("input_number") is not None
                for k in range(len(b["mid"])):
                    vals = [b["edges"][k], b["edges"][k + 1], b["mid"][k], b["real_number"][k], b["real_number_cum"][k],
                            b["real_volume"][k], b["real_volume_cum"][k]]
                    vals += ([b["input_number"][k], b["input_number_cum"][k], b["input_volume"][k],
                              b["input_volume_cum"][k]] if inp else [None] * 4)
                    wr.writerow([int(i) + 1, e["name"], dmn["label"], dmn["unit"].replace("um", "µm")] +
                                ["" if v is None else f"{float(v):.6g}" for v in vals])
    written.append("psa.csv")
    keys = ["count", "number_mean", "std", "cv", "d10", "d50", "d90", "min", "max", "volume_mean", "d10v", "d50v", "d90v"]
    with open(os.path.join(out_dir, "psa_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["phase", "phase_name", "dimension", "unit", "source"] + keys)
        for i, e in sorted(report.items(), key=lambda kv: int(kv[0])):
            for dmn in e["dimensions"]:
                for src in ("input", "real"):
                    st = dmn.get(src) or {}
                    wr.writerow([int(i) + 1, e["name"], dmn["label"], dmn["unit"].replace("um", "µm"),
                                 "realised" if src == "real" else "input"] +
                                [f"{float(st.get(k)):.6g}" if st.get(k) is not None else "" for k in keys])
    written.append("psa_summary.csv")
    return written


def _periodic_components(mask):
    """Connected regions of a mask (faces shared, 6-connectivity), joined across the faces of the periodic box.
    Returns (labels int32 0 = none, number of regions)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ax in range(3):
        a = np.take(lab, 0, axis=ax).ravel()
        b = np.take(lab, -1, axis=ax).ravel()
        both = (a > 0) & (b > 0)
        for p, q in set(zip(a[both].tolist(), b[both].tolist())):
            rp, rq = find(p), find(q)
            if rp != rq:
                parent[max(rp, rq)] = min(rp, rq)
    roots = np.array([find(k) for k in range(n + 1)])
    uniq, new = np.unique(roots, return_inverse=True)
    return new.reshape(-1)[lab].astype(np.int32), int(len(uniq) - 1)


def separate(mask, h_um, log=None):
    """The particles of one phase of a voxel image, touching ones apart: a marker watershed of the distance to the
    phase's surface, its markers the maxima of that distance (smoothed, sigma 0.4) and of them only one where two lie
    nearer to each other than to the surface (the steps of SNOW, J. T. Gostick, Phys. Rev. E 96 (2017) 023307, by
    porespy.filters, without its saddle-point trimming, which merged a small sphere into a larger neighbour: 371 of
    385 touching spheres of 8 voxels, CV 0.3, found with it, all 385 without), on the image padded by its own
    periodic copies, so that a particle cut by a face of the box is found whole; a particle is kept where its centre
    lies inside the box (each once). Checked on generated packings of touching particles: spheres of one size and
    of CV 0.3 and 0.5 and polyhedra exactly, oblate spheroids (2 : 1) +0.2 %, prolate ones (3 : 1) +7 % (a long
    particle can split in two). Returns {"n", "vol_um3", "eqd_um" (volume-equivalent diameters), "circ_um"
    (circumscribed diameters about the centre), "centers_um", "pad", "seconds", "specks" (regions of fewer than 4
    voxels or 2 % of the median particle's, noise of a segmentation, left out), "speck_voxels"}."""
    from scipy import ndimage
    import porespy as ps
    t0 = time.time()
    mask = np.ascontiguousarray(np.asarray(mask, bool))
    n = np.array(mask.shape, np.int64)
    if not mask.any():
        return {"n": 0, "vol_um3": np.zeros(0), "eqd_um": np.zeros(0), "circ_um": np.zeros(0),
                "centers_um": np.zeros((0, 3)), "pad": 0, "seconds": 0.0, "specks": 0, "speck_voxels": 0.0,
                "speck_min_vox": 1.0, "semi_um": np.zeros((0, 3)), "axes": np.zeros((0, 3, 3))}
    # the largest inscribed radius sets how far a particle reaches across a face
    p0 = int(min(n.min() // 2, 16))
    rmax = float(ndimage.distance_transform_edt(np.pad(mask, p0, mode="wrap")).max())
    pad = int(min(n.min() // 2, max(8, math.ceil(3.0 * rmax))))
    big = np.pad(mask, pad, mode="wrap")
    if log and big.size > 2e7:
        # measured: about 50 bytes per voxel of the padded image at the peak (the distance map and the watershed)
        log(f"    separating the particles of a {' × '.join(str(x) for x in mask.shape)} image (padded by {pad} voxels "
            f"to {big.size / 1e6:.0f} M voxels) takes about {50 * big.size / 2**30:.1f} GB")
    from skimage.segmentation import watershed
    dt = ndimage.distance_transform_edt(big).astype(np.float32)
    dt = ndimage.gaussian_filter(dt, 0.4, output=dt)
    # a maximum within 3 voxels (or a fifth of the largest radius) of a higher one is not a particle of its own
    r_peak = int(max(3, round(0.2 * rmax)))
    peaks = ps.filters.trim_nearby_peaks(ps.filters.find_peaks(dt, r_max=r_peak), dt)
    reg = watershed(-dt, markers=ndimage.label(peaks > 0)[0], mask=big).astype(np.int32)
    del dt, peaks
    reg[~big] = 0
    nl = int(reg.max())
    # voxel counts and centres, summed slice by slice (a full-size weight array would double the memory)
    cnt = np.zeros(nl + 1)
    sums = np.zeros((nl + 1, 3))
    # and the second moments (xx, yy, zz, xy, xz, yz) for each particle's ellipsoid of inertia
    mom = np.zeros((nl + 1, 6))
    jj, kk = np.meshgrid(np.arange(big.shape[1], dtype=np.float64) + 0.5, np.arange(big.shape[2], dtype=np.float64) + 0.5,
                         indexing="ij")
    jj, kk = jj.ravel(), kk.ravel()
    jk = jj * kk
    for i in range(big.shape[0]):
        sl = reg[i].ravel()
        c = np.bincount(sl, minlength=nl + 1)
        cnt += c
        x_ = i + 0.5
        sj = np.bincount(sl, weights=jj, minlength=nl + 1)
        sk = np.bincount(sl, weights=kk, minlength=nl + 1)
        sums[:, 0] += c * x_
        sums[:, 1] += sj
        sums[:, 2] += sk
        mom[:, 0] += c * x_ * x_
        mom[:, 1] += np.bincount(sl, weights=jj * jj, minlength=nl + 1)
        mom[:, 2] += np.bincount(sl, weights=kk * kk, minlength=nl + 1)
        mom[:, 3] += x_ * sj
        mom[:, 4] += x_ * sk
        mom[:, 5] += np.bincount(sl, weights=jk, minlength=nl + 1)
    cen = sums / np.maximum(cnt, 1.0)[:, None]
    keep = np.flatnonzero((cnt > 0) & np.all((cen >= pad) & (cen < pad + n), axis=1))
    keep = keep[keep > 0]
    # specks of a segmentation (a few noisy voxels over a threshold) are no particles: fewer than 4 voxels, or than
    # 2 % of the median particle's (a scan-like image of 90 spheres gave 95 particles, the five extra single voxels)
    speck_min = max(4.0, 0.02 * float(np.median(cnt[keep]))) if len(keep) else 4.0
    speck = cnt[keep] < speck_min
    n_speck, v_speck = int(speck.sum()), float(cnt[keep][speck].sum())
    keep = keep[~speck]
    # the circumscribed radius: the farthest voxel centre from the particle's centre, plus half a voxel diagonal;
    # the farthest voxel lies on the particle's boundary, so only the boundary voxels are measured
    bnd = np.zeros(big.shape, bool)
    for ax in range(3):
        for sh in (1, -1):
            bnd |= reg != np.roll(reg, sh, axis=ax)
    bnd &= reg > 0
    idx = np.nonzero(bnd)
    del bnd
    rb = reg[idx]
    del reg
    d2b = np.zeros(len(rb))
    for ax in range(3):
        d2b += (idx[ax] + 0.5 - cen[rb, ax]) ** 2
    far = np.zeros(nl + 1)
    np.maximum.at(far, rb, d2b)
    del idx, rb, d2b
    rmx = np.sqrt(far[keep]) + 0.5 * math.sqrt(3.0)
    vol = cnt[keep] * h_um ** 3
    eqd = (6.0 * vol / math.pi) ** (1.0 / 3.0)
    # the ellipsoid of the same second moments (a uniform ellipsoid has the variance a^2 / 5 along a semi-axis
    # a; a voxel adds h^2 / 12), scaled to the particle's volume: its semi-axes a >= b >= c and its axes
    m_ = cnt[keep][:, None]
    ex = mom[keep] / m_
    c0 = cen[keep]
    cov = np.empty((len(keep), 3, 3))
    cov[:, 0, 0] = ex[:, 0] - c0[:, 0] ** 2 - 1.0 / 12.0
    cov[:, 1, 1] = ex[:, 1] - c0[:, 1] ** 2 - 1.0 / 12.0
    cov[:, 2, 2] = ex[:, 2] - c0[:, 2] ** 2 - 1.0 / 12.0
    cov[:, 0, 1] = cov[:, 1, 0] = ex[:, 3] - c0[:, 0] * c0[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = ex[:, 4] - c0[:, 0] * c0[:, 2]
    cov[:, 1, 2] = cov[:, 2, 1] = ex[:, 5] - c0[:, 1] * c0[:, 2]
    lam, vec = np.linalg.eigh(cov)                     # ascending
    semi = np.sqrt(np.maximum(5.0 * lam[:, ::-1], 1e-12))
    semi *= ((3.0 * vol / (4.0 * math.pi * h_um ** 3)) / np.prod(semi, axis=1))[:, None] ** (1.0 / 3.0)
    rows = np.transpose(vec[:, :, ::-1], (0, 2, 1))  # row k: the axis of semi-axis k
    rows[:, 2] = np.cross(rows[:, 0], rows[:, 1])      # right-handed
    out = {"n": int(len(keep)), "vol_um3": vol, "eqd_um": eqd, "circ_um": np.maximum(2.0 * rmx * h_um, eqd),
           "semi_um": semi * h_um, "axes": rows,
           "centers_um": np.mod(cen[keep] - pad, n) * h_um, "pad": pad, "seconds": time.time() - t0,
           "specks": n_speck, "speck_voxels": v_speck, "speck_min_vox": speck_min}
    if log:
        log(f"    {out['n']} particles separated (periodic watershed, {out['seconds']:.1f} s)"
            + (f"; {n_speck} specks of fewer voxels left out" if n_speck else ""))
    return out


def analyse_image(spec, info, labels, table, h_um, log=None):
    """Grain analysis of a structure read from an image: every phase's particles, touching ones separated by a
    periodic watershed (separate) - their number, volume-equivalent and circumscribed diameters by number and by
    volume - beside its connected regions (joined across the periodic faces: the clusters and percolation), and
    the matrix between them. Up to 5.10.0 the connected regions were the grains, so a packed powder read as fewer,
    larger grains than it holds."""
    from .solvers.conduction import wrapping_axes
    t0 = time.time()
    box, _ = particle_box(info)
    vol_box = float(np.prod(box))
    out = {"phases": {}, "image": True, "sizes": {}, "_components": []}
    for li, t in enumerate(table):
        if t["kind"] != "core":
            continue
        i = t["phase"]
        mask = np.asarray(labels) == li
        comp, n = _periodic_components(mask)
        entry = {"name": t["name"], "shape": "image", "n_particles": int(n)}
        if n == 0:
            out["phases"][str(li)] = entry
            continue
        nv_reg = np.bincount(comp.ravel(), minlength=n + 1)[1:].astype(float)
        # the particles: touching ones separated (periodic watershed); the connected regions are the clusters
        try:
            # the viscosity analysis may have separated them already (pipeline: info["_separated"])
            sep = (info.get("_separated") or {}).get(li)
            if sep is None:
                sep = separate(mask, h_um, log=log)
                if isinstance(info, dict):
                    info.setdefault("_separated", {})[li] = sep     # the particle files take them (pipeline)
        except Exception as e:                                       # noqa: BLE001
            sep = None
            if log:
                log(f"    {t['name']}: the particles could not be separated ({e}); the connected regions are taken")
        if sep is not None and sep["n"] > 0:
            V, eqd, circ = sep["vol_um3"], sep["eqd_um"], sep["circ_um"]
            n_p = int(sep["n"])
        else:
            V = nv_reg * h_um ** 3
            eqd = (6.0 * V / math.pi) ** (1.0 / 3.0)
            circ = eqd.copy()
            n_p = int(n)
        nv = V / h_um ** 3
        n_cl = int(np.count_nonzero(nv_reg >= (sep or {}).get("speck_min_vox", 1.0))) if sep is not None else int(n)
        faces = sum(int(np.count_nonzero(mask != np.roll(mask, -1, axis=ax))) for ax in range(3))
        entry.update({
            "phase": int(i),
            "n_particles": n_p, "separated": sep is not None and sep["n"] > 0,
            "number_density_per_mm3": n_p / (vol_box * 1e-9),
            "eq_diameter_um": {"number": _quantiles(eqd), "volume": _quantiles(eqd, V)},
            # both diameters of the particles, by number and by volume (as the realised sizes give them)
            "diameters": {"circumscribed": size_stats(circ, V), "volume_equivalent": size_stats(eqd, V)},
            "specific_surface_voxel_1pm": faces * h_um * h_um / vol_box * 1e6,
            "single_voxel_regions": int(np.count_nonzero(nv_reg == 1)),
            # the clusters: connected regions of at least a particle's least size (the specks are no clusters)
            "voxel": {"volume_fraction": float(mask.mean()), "clusters": {"n_clusters": n_cl, "n_regions": int(n)},
                      "particles_per_cluster": n_p / max(n_cl, 1), "percolates": dict(zip("xyz", wrapping_axes(mask)))}})
        out["phases"][str(li)] = entry
        st = size_stats(eqd, V)
        lo, hi = float(eqd.min()), float(eqd.max())
        nb = int(min(30, max(8, round(2.0 * n ** (1.0 / 3.0)))))
        edges = (np.geomspace(lo / 1.0001, hi * 1.0001, nb + 1) if hi > lo * 1.0001
                 else np.array([lo * 0.999, hi * 1.001]))
        rn, rv, rcn, rcv = _hist(eqd, V, edges)
        out["sizes"][str(i)] = {
            "name": t["name"], "shape": "image", "n_particles": n_p, "basis": "eqd", "shell_um": 0.0, "own": [],
            "regions": True,
            "dimensions": [{"key": "eqd", "label": QTY_LABEL["eqd"][0], "unit": "um", "real": st, "input": None,
                            "bins": {"edges": edges.tolist(), "mid": np.sqrt(edges[1:] * edges[:-1]).tolist(),
                                     "real_number": rn.tolist(), "real_volume": rv.tolist(),
                                     "real_number_cum": rcn.tolist(), "real_volume_cum": rcv.tolist(),
                                     "input_number": None, "input_volume": None,
                                     "input_number_cum": None, "input_volume_cum": None}}]}
        out["_components"].append((i, t["name"], nv, V, eqd, circ,
                                   sep["centers_um"] if sep is not None and sep["n"] > 0 else None))
        if sep is not None and sep["n"] > 0:
            out.setdefault("_particles", {})[int(i)] = sep
        del mask, comp
    matrix = np.asarray(labels) == 0
    lt = None
    if matrix.any() and not matrix.all():
        lt = MO.local_thickness(matrix)
        out["matrix_ligament"] = MO.size_distribution(lt, matrix, h_um, bins=14)
    out["seconds"] = time.time() - t0
    if log:
        log(f"    grain analysis of the image: " + ", ".join(
            f"{e['name']} {e['n_particles']} particles in {e.get('voxel', {}).get('clusters', {}).get('n_clusters', 0)} clusters"
            for e in out["phases"].values()) + f" ({out['seconds']:.1f} s)")
    return out, lt


def write_components(out_dir, comps):
    """grains.csv: every particle of every phase of an image, touching ones separated (5.10.2) - its voxels, volume,
    volume-equivalent and circumscribed diameters and centre; where a phase's particles could not be separated, its
    connected regions (no centre)."""
    import csv
    import os
    with open(os.path.join(out_dir, "grains.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["phase", "phase_name", "particle", "voxels", "volume_um3", "eqd_um", "circ_um", "x_um", "y_um", "z_um"])
        for c in comps:
            i, name, nv, V, eqd = c[:5]
            circ = c[5] if len(c) > 5 else None
            cen = c[6] if len(c) > 6 else None
            for k in range(len(nv)):
                row = [int(i) + 1, name, k + 1, int(round(float(nv[k]))), f"{float(V[k]):.6g}", f"{float(eqd[k]):.6g}",
                       f"{float(circ[k]):.6g}" if circ is not None else ""]
                row += [f"{float(x):.6g}" for x in cen[k]] if cen is not None else ["", "", ""]
                wr.writerow(row)
    return ["grains.csv"]


PARTICLES_README = """Particles of the structure ({n} particles), as MPSim built them.

Box: {X:.8g} x {Y:.8g} x {Z:.8g} um (x, y, z), from the origin; {film}. Lengths in micrometres.
Voxel: {h:.8g} um, the grid the run drew them on (Structure - List (CSV) reads this file and draws them again).
The voxel structure (structure.tif) samples these particles at the voxel centres ((i + 0.5) h, (j + 0.5) h, (k + 0.5) h):
a voxel belongs to a particle when its centre lies inside it.

particles.csv - one row per particle
  x_um, y_um, z_um   the centre
  d_um               the size d of the shape (sphere: diameter; spheroid: 2a along its local x; cylinder and
                     capsule: diameter; cube and cuboid: edge along local x; superellipsoid: 2a along local x;
                     polyhedron: circumscribed diameter)
  eqd_um, volume_um3 the volume-equivalent diameter and the volume
  lx_um, ly_um, lz_um  the extent along the particle's own axes x, y, z (spheroid and superellipsoid: the axes
                     2a, 2b, 2c; cuboid: the edges; cylinder: diameter, diameter, length; capsule: the same,
                     with the hemispherical ends in the length; polyhedron: its circumscribed diameter)
  length_um          cylinder and capsule: the length along local z (capsule: total, ends included)
  aspect_c_a, aspect_b_a  spheroid: c/a (local z) and b/a (local y)
  n_exponent         superellipsoid: |x/a|^n + |y/b|^n + |z/c|^n <= 1
  variant            polyhedron: its shape in polyhedra.csv
  shell_um           a coating's thickness (the sizes above include it)
  r11 ... r33        the rotation: row 1 (r11 r12 r13) is the particle's local x axis in the box, row 2 its y
                     axis, row 3 its z axis. A point (u, v, w) of the particle's own frame lies at
                     centre + u * row1 + v * row2 + w * row3.

polyhedra.csv - the vertices of every polyhedron shape in its own frame at circumradius 1: multiply by d_um / 2
and place them with the rotation and the centre above; the particle is their convex hull.

psa.csv - the size distributions per phase and dimension in bins: the realised (the particles above) and the
input (what the entered distribution draws, 20,000 particles sampled as the generator samples them), as number %
and volume % per bin and cumulative.
psa_summary.csv - their statistics: number mean, standard deviation, CV, D10/D50/D90 by number, the volume-weighted
mean (D[4,3] for spheres) and D10/D50/D90 by volume.
"""
