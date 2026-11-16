"""A structure from a particle list (5.10.2): the particles of a CSV file drawn on a voxel grid by the generator's own
raster - the structure the list describes, without packing it again.

Two kinds of file are read:
- MPSim's own particles.csv (written beside every run with particles; its conventions in particles_README.txt):
  every shape - spheres, spheroids (also triaxial), cylinders, capsules, cubes and cuboids, superellipsoids and
  polyhedra (their vertices in polyhedra.csv) - with the centre, the dimensions and the rotation of each particle.
  particles_README.txt beside it gives the box (and whether it is a film); polyhedra.csv the polyhedron shapes.
- A plain table of spheres: a centre and a diameter per row (x, y, z, d; or r for the radius), and optionally a
  phase (a number or a name) - the arrangement of any other program.

The particle list read back is the generator's own (codes, sizes P, rotations R, centres, phase), so a structure
written by MPSim is drawn again voxel for voxel, and every analysis that needs the particles (grain analysis, the
particle dynamics, the contacts) has them as for a generated RVE."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import time
import uuid

import numpy as np

from . import geometry as G
from . import shapes as SH

META_FILE = "meta.json"
LIST_FILE = "particles.npz"
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "particles")

_ALIASES = {"x": ("x_um", "x", "cx", "x_center", "xc"), "y": ("y_um", "y", "cy", "y_center", "yc"),
            "z": ("z_um", "z", "cz", "z_center", "zc"), "d": ("d_um", "d", "diameter", "diameter_um", "dia"),
            "r": ("r_um", "r", "radius", "radius_um")}
_SHAPES = {"sphere": 0, "spheroid": 1, "ellipsoid": 1, "cylinder": 2, "fibre": 2, "fiber": 2, "spherocylinder": 3,
           "capsule": 3, "cube": 4, "cuboid": 4, "superellipsoid": 5, "polyhedron": 6}


def _col(fields, key):
    low = {f.strip().lower(): f for f in fields}
    for a in _ALIASES[key]:
        if a in low:
            return low[a]
    return None


def _num(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _planes_of(vertices):
    """The outward planes (n, d) of a polyhedron from its vertices at circumradius 1, as the generator keeps them
    (shapes.hull_planes without centring or scaling again: the vertices are already its own frame)."""
    from scipy.spatial import ConvexHull
    v = np.asarray(vertices, float)
    hull = ConvexHull(v)
    eq = hull.equations
    _, first = np.unique(np.round(eq, 7), axis=0, return_index=True)
    eq = eq[np.sort(first)]
    return np.column_stack([eq[:, :3], -eq[:, 3]]), float(hull.volume), float(hull.area)


def read_readme(path):
    """The box (x, y, z), film and voxel size of an MPSim particles_README.txt (None where it says nothing)."""
    out = {"box": None, "film": False, "voxel": None}
    try:
        txt = open(path, encoding="utf-8").read()
    except OSError:
        return out
    m = re.search(r"Box:\s*([0-9.eE+-]+)\s*x\s*([0-9.eE+-]+)\s*x\s*([0-9.eE+-]+)\s*um", txt)
    if m:
        out["box"] = [float(m.group(k)) for k in (1, 2, 3)]
    out["film"] = "a film:" in txt
    m = re.search(r"Voxel(?: size)?:\s*([0-9.eE+-]+)\s*um", txt)
    if m:
        out["voxel"] = float(m.group(1))
    return out


def read(folder):
    """The particle list of the CSV files in a folder: particles.csv (or the one CSV there), polyhedra.csv and
    particles_README.txt when they are there. Returns (arrays, description): arrays the generator's particle list
    in the file's own length unit - codes, P, R, centers, phase, and the polyhedron planes - and the description:
    the phases (key, name, shape, count, total volume), the box, film and voxel size from the README, the format."""
    names = sorted(os.listdir(folder))
    csvs = [n for n in names if n.lower().endswith(".csv") and n.lower() not in ("polyhedra.csv", "psa.csv",
                                                                                  "psa_summary.csv", "grains.csv")]
    main = "particles.csv" if "particles.csv" in names else (csvs[0] if len(csvs) == 1 else None)
    if main is None:
        if not csvs:
            raise ValueError("no particle CSV file (particles.csv, or a table with x, y, z and d per particle)")
        raise ValueError(f"several CSV files ({', '.join(csvs)}): name the particle list particles.csv")
    with open(os.path.join(folder, main), encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{main} has no rows")
    fields = list(rows[0].keys())
    cx, cy, cz = _col(fields, "x"), _col(fields, "y"), _col(fields, "z")
    cd, cr = _col(fields, "d"), _col(fields, "r")
    if not (cx and cy and cz and (cd or cr)):
        raise ValueError(f"{main}: columns x, y, z and d (or r) are needed; found {', '.join(fields)}")
    low = {f.strip().lower(): f for f in fields}
    c_shape = low.get("shape")
    c_phase = low.get("phase")
    c_pname = low.get("phase_name")
    mpsim = all(k in low for k in ("lx_um", "ly_um", "lz_um", "r11", "r33"))
    # polyhedra: the vertices of every variant, as MPSim writes them
    verts = {}
    if os.path.isfile(os.path.join(folder, "polyhedra.csv")):
        with open(os.path.join(folder, "polyhedra.csv"), encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                verts.setdefault(int(float(r["variant"])), []).append([float(r["x"]), float(r["y"]), float(r["z"])])
    n = len(rows)
    codes = np.zeros(n, np.int64)
    P = np.zeros((n, 6))
    R = np.tile(np.eye(3), (n, 1, 1))
    C = np.zeros((n, 3))
    keys, phase = [], np.zeros(n, np.int64)
    shape_names = []
    pname = {}
    for j, r in enumerate(rows):
        f = lambda k: _num(r.get(low.get(k, k), "")) if k else None          # noqa: E731
        xyz = [_num(r[cx]), _num(r[cy]), _num(r[cz])]
        if any(v is None for v in xyz):
            raise ValueError(f"{main}, row {j + 2}: a coordinate is not a number")
        C[j] = xyz
        shape = (str(r.get(c_shape) or "sphere").strip().lower() if c_shape else "sphere") or "sphere"
        if shape not in _SHAPES:
            raise ValueError(f"{main}, row {j + 2}: unknown shape '{shape}' ({', '.join(SH.USER_SHAPES)})")
        code = _SHAPES[shape]
        codes[j] = code
        d = _num(r[cd]) if cd else None
        if d is None and cr:
            rr = _num(r[cr])
            d = 2.0 * rr if rr is not None else None
        if d is None or d <= 0:
            raise ValueError(f"{main}, row {j + 2}: no size (d > 0)")
        if code != 0 and not mpsim:
            raise ValueError(f"{main}, row {j + 2}: a {shape} needs MPSim's columns (lx_um, ly_um, lz_um, the "
                             f"rotation r11 ... r33); a plain table holds spheres")
        if code == 0:
            P[j, 0] = 0.5 * d
        else:
            lx, ly, lz = f("lx_um"), f("ly_um"), f("lz_um")
            if code == 1:
                P[j, 0], P[j, 1], P[j, 2] = 0.5 * lx, 0.5 * lz, 0.5 * ly
            elif code == 2:
                P[j, 0], P[j, 1] = 0.5 * lx, 0.5 * (f("length_um") or lz)
            elif code == 3:
                P[j, 0] = 0.5 * lx
                P[j, 1] = 0.5 * (f("length_um") or lz) - P[j, 0]
            elif code == 4:
                P[j, 0], P[j, 1], P[j, 2] = 0.5 * lx, 0.5 * ly, 0.5 * lz
            elif code == 5:
                P[j, 0], P[j, 1], P[j, 2] = 0.5 * lx, 0.5 * ly, 0.5 * lz
                P[j, 3] = f("n_exponent") or 2.0
            else:
                t = int(f("variant") if f("variant") is not None else -1)
                if t not in verts:
                    raise ValueError(f"{main}, row {j + 2}: polyhedron variant {t} is not in polyhedra.csv")
                P[j, 0] = 0.5 * d
                P[j, 1] = float(t)
        if mpsim:
            rot = [f(f"r{a}{b}") for a in (1, 2, 3) for b in (1, 2, 3)]
            if all(v is not None for v in rot):
                R[j] = np.asarray(rot, float).reshape(3, 3)
        key = str(r.get(c_phase)).strip() if c_phase and str(r.get(c_phase) or "").strip() else "1"
        if c_pname and str(r.get(c_pname) or "").strip():
            pname.setdefault(key, str(r[c_pname]).strip())
        # one shape per phase: a phase holding two shapes is split by shape
        full = f"{key}|{shape}"
        if full not in keys:
            keys.append(full)
            shape_names.append(shape)
        phase[j] = keys.index(full)
    # the polyhedron planes, indexed by the variant numbers of the file
    T = max(verts) + 1 if verts else 1
    planes = np.zeros((T, SH.MAX_FACES, 4))
    nfaces = np.zeros(T, np.int64)
    poly_meta = [None] * T
    for t, v in verts.items():
        pl, vol, area = _planes_of(v)
        if len(pl) > SH.MAX_FACES:
            raise ValueError(f"polyhedron variant {t} has {len(pl)} faces (at most {SH.MAX_FACES})")
        planes[t, :len(pl)] = pl
        nfaces[t] = len(pl)
        poly_meta[t] = {"kind": "irregular", "volume": vol, "area": area, "inradius": float(pl[:, 3].min()),
                        "faces": len(pl), "vertices": [list(map(float, x)) for x in v]}
    rd = read_readme(os.path.join(folder, "particles_README.txt"))
    # the phases, in the order of the file's phase numbers where they are numbers
    def order_key(k):
        a = k.split("|")[0]
        return (0, float(a), k) if _num(a) is not None else (1, 0.0, k)
    order = sorted(range(len(keys)), key=lambda i: order_key(keys[i]))
    remap = np.empty(len(keys), np.int64)
    remap[order] = np.arange(len(keys))
    phase = remap[phase]
    keys = [keys[i] for i in order]
    shape_names = [shape_names[i] for i in order]
    split = {k.split("|")[0] for k in keys}
    phases = []
    for i, k in enumerate(keys):
        base = k.split("|")[0]
        sel = phase == i
        nm = pname.get(base) or (f"Phase {base}" if c_phase else "Particles")
        if sum(1 for q in keys if q.split("|")[0] == base) > 1:
            nm = f"{nm} ({shape_names[i]})"
        vols = np.array([SH.volume(int(c), p, poly_meta) for c, p in zip(codes[sel], P[sel])])
        phases.append({"key": k, "name": nm, "shape": shape_names[i], "count": int(sel.sum()),
                       "volume": float(vols.sum()), "d_mean": float(2.0 * P[sel, 0].mean())})
    arrays = {"codes": codes, "P": P, "R": R, "centers": C, "phase": phase, "planes": planes, "nfaces": nfaces}
    ext = C.max(axis=0) + P[:, 0].max()
    desc = {"file": main, "n": n, "phases": phases, "format": "mpsim" if mpsim else "spheres",
            "box": rd["box"], "film": bool(rd["film"]), "voxel": rd["voxel"], "poly_meta": poly_meta,
            "box_suggested": [float(math.ceil(v)) for v in ext], "readme": rd["box"] is not None,
            "split_phases": len(split) != len(keys)}
    return arrays, desc


def store(root, files, name=None):
    """Keep particle files (a list of (path, file name)) under root/<id>: the files themselves, the particle list
    read from them (particles.npz) and their description (meta.json). Returns the meta."""
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = os.path.join(root, pid)
    os.makedirs(d, exist_ok=True)
    try:
        for path, fname in files:
            base = os.path.basename(str(fname))
            if not base.lower().endswith((".csv", ".txt")):
                continue
            # MPSim's own names are kept; a single other CSV is the particle list
            shutil.copyfile(path, os.path.join(d, base))
        arrays, desc = read(d)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    np.savez_compressed(os.path.join(d, LIST_FILE), **arrays)
    meta = dict(desc, id=pid, name=name or desc["file"], dir=os.path.abspath(d))
    with open(os.path.join(d, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return meta


def store_folder(root, folder, name=None):
    """The particle files of a run folder (particles.csv, polyhedra.csv, particles_README.txt) kept as an upload is."""
    files = [(os.path.join(folder, f), f) for f in ("particles.csv", "polyhedra.csv", "particles_README.txt")
             if os.path.isfile(os.path.join(folder, f))]
    if not files:
        raise ValueError("the run has no particles.csv")
    return store(root, files, name=name)


def example_dir(name):
    """The folder of an example particle list shipped with MPSim (data/particles/<name>), read once into the
    user's temporary folder (the program folder may be read-only)."""
    import tempfile
    name = os.path.basename(str(name))
    src = os.path.join(EXAMPLES, name)
    if not os.path.isdir(src):
        raise ValueError(f"no example particle list '{name}'")
    stamp = int(max(os.path.getmtime(os.path.join(src, f)) for f in os.listdir(src)))
    root = os.path.join(tempfile.gettempdir(), "mpsim_example_particles")
    d = os.path.join(root, f"{name}-{stamp}")
    if not os.path.isfile(os.path.join(d, META_FILE)):
        meta = store_folder(root, src, name=name)
        try:
            os.replace(meta["dir"], d)
        except OSError:
            return meta["dir"]
        meta["dir"] = os.path.abspath(d)
        with open(os.path.join(d, META_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    return os.path.abspath(d)


def load_meta(d):
    with open(os.path.join(d, META_FILE), encoding="utf-8") as fh:
        meta = json.load(fh)
    if "phases" not in meta or "n" not in meta:
        raise ValueError(f"{d} is not an MPSim particle list")
    return meta


def load_list(d):
    with np.load(os.path.join(d, LIST_FILE)) as z:
        return {k: z[k] for k in z.files}


def size_table(eqd):
    """Measured diameters (µm) as the table a form takes for a measured size distribution: logarithmic bins (6 to
    30, about 2 n^(1/3)), their mid diameters and counts, by number; None where they hardly differ or are fewer
    than 3 (one size)."""
    eqd = np.asarray(eqd, float)
    if len(eqd) < 3 or eqd.max() <= 1.02 * eqd.min():
        return None
    nb = int(min(30, max(6, round(2.0 * len(eqd) ** (1.0 / 3.0)))))
    edges = np.geomspace(eqd.min() / 1.0001, eqd.max() * 1.0001, nb + 1)
    cnt = np.histogram(eqd, edges)[0].astype(float)
    return {"type": "table", "d": np.sqrt(edges[1:] * edges[:-1]).tolist(), "v": cnt.tolist(), "basis": "number",
            "form": "differential"}


def phase_sizes(arrays, i, scale):
    """The size entry of phase i of a particle list for a form (shape, size dict in µm, measured distribution of
    d): the nominal proportions are the means of its particles, d is distributed as they have it."""
    sel = arrays["phase"] == i
    codes, P = arrays["codes"][sel], arrays["P"][sel] * scale
    code = int(codes[0])
    d = 2.0 * P[:, 0]
    dm = float(d.mean())
    if code == 0:
        size = {"d": dm}
    elif code == 1:
        size = {"d": dm, "aspect": float(np.mean(P[:, 1] / P[:, 0])),
                "aspect_y": float(np.mean(np.where(P[:, 2] > 0, P[:, 2], P[:, 0]) / P[:, 0]))}
    elif code == 2:
        size = {"d": dm, "length": float(np.mean(2.0 * P[:, 1]))}
    elif code == 3:
        size = {"d": dm, "length": float(np.mean(2.0 * (P[:, 1] + P[:, 0])))}
    elif code == 4:
        if np.allclose(P[:, 1], P[:, 0]) and np.allclose(P[:, 2], P[:, 0]):
            size = {"d": dm}
        else:
            size = {"d": dm, "ly": float(np.mean(2.0 * P[:, 1])), "lz": float(np.mean(2.0 * P[:, 2]))}
    elif code == 5:
        size = {"d": dm, "ly": float(np.mean(2.0 * P[:, 1])), "lz": float(np.mean(2.0 * P[:, 2])),
                "n": float(np.mean(P[:, 3]))}
    else:
        nv = len(np.unique(P[:, 1]))
        size = {"d": dm, "poly": "irregular", "variants": int(max(1, nv)), "n_vertices": 14}
    return size, size_table(d)


def structure(pin, phases, want_contact_faces=False, log=None):
    """The structure of a particle list on its grid: (labels, info) as generate() returns them - the phases'
    fractions and surfaces, the contacts, and the particle list itself (in µm), so that every later step sees a
    structure with its particles. pin: spec['particles_in'] (dir, scale to µm, h_um, shape, film)."""
    from . import generate as GEN
    log = log or (lambda *a: None)
    t0 = time.time()
    a = load_list(pin["dir"])
    s = float(pin.get("scale", 1.0))
    h = float(pin["h_um"])
    nx, ny, nz = (int(v) for v in pin["shape"])
    film = bool(pin.get("film"))
    codes = a["codes"].astype(np.int64)
    P = a["P"].copy()
    for j, c in enumerate(codes):
        for k in GEN._linear_slots(int(c)):
            P[j, k] *= s
    parts = {"codes": codes, "P": P, "R": a["R"].astype(float), "centers": a["centers"] * s,
             "phase": a["phase"].astype(np.int64)}
    parts["axes"] = parts["R"][:, 2, :] if len(parts["R"]) else np.zeros((0, 3))
    planes = (a["planes"], a["nfaces"])
    poly_meta = pin.get("poly_meta") or []
    info = {"particles": parts, "planes": planes, "film": film, "T_um": nz * h}
    cubic = nx == ny == nz and not film
    owner, phase_of = GEN.rasterize_owner(nx, h, info, phases, nz, shape=None if (cubic or film) else (nx, ny, nz))
    overlap_of, net_of, _ = GEN._phase_arrays(phases, "keep")
    cf = np.zeros(owner.shape, np.uint8)
    n_contact = int(GEN.contact_faces(owner, phase_of, net_of, cf, not film))
    n_touch = GEN.touching_pairs(owner, phase_of, net_of, not film)
    lut = np.zeros(len(phase_of), np.uint8)
    valid = phase_of >= 0
    lut[valid] = (phase_of[valid] + 1).astype(np.uint8)
    lut[0] = 0
    labels = lut[owner]
    del owner
    nvox = labels.size
    counts = np.bincount(labels.ravel(), minlength=len(phases) + 1)
    box_vol = nx * ny * nz * h ** 3
    pinfo = []
    hx, hy, hz = nx // 2, ny // 2, nz // 2
    for i, ph in enumerate(phases):
        sel = parts["phase"] == i
        m = labels == (i + 1)
        faces = sum(int(np.count_nonzero(m != np.roll(m, -1, axis=ax))) for ax in range(3))
        vols = [SH.volume(int(c), p, poly_meta) for c, p in zip(codes[sel], P[sel])]
        octs = [float(m[x0:x1, y0:y1, z0:z1].mean()) for x0, x1 in ((0, hx), (hx, nx)) for y0, y1 in ((0, hy), (hy, ny))
                for z0, z1 in ((0, hz), (hz, nz))]
        om = float(np.mean(octs))
        pinfo.append({"name": ph.get("name", f"phase {i + 1}"),
                      # the content the file's particles have (their volumes, overlaps counted twice)
                      "vf_target": float(sum(vols)) / box_vol,
                      "vf_voxel": float(counts[i + 1] / nvox), "n_particles": int(sel.sum()), "jammed": False,
                      "surface_true_um2": (float(sum(SH.surface(int(c), p, poly_meta) for c, p in zip(codes[sel], P[sel])))
                                           if sel.any() else None),
                      "surface_voxel_um2": faces * h * h, "octant_cov": float(np.std(octs) / om) if om > 0 else 0.0})
    info.update({"route": f"particle list {pin.get('name') or ''} ({len(codes):,} particles drawn on {nx} × {ny} × "
                          f"{nz} voxels of {h:.4g} µm)".replace("  ", " "),
                 "phases": pinfo, "contact_faces": n_contact, "touching_pairs": n_touch,
                 "contact_mask": cf if want_contact_faces else None, "contacts_mode": "keep", "contacts_removed": 0,
                 "voxel_conflicts": 0, "N": nx, "h_um": h, "L_um": nx * h, "nz": nz, "poly_meta": poly_meta,
                 "from_csv": pin.get("name") or "particles.csv", "seconds": time.time() - t0})
    if not cubic and not film:
        info["shape"] = [nx, ny, nz]
    log(f"  structure drawn from the particle list {pin.get('name') or ''}: {len(codes):,} particles on "
        f"{nx} × {ny} × {nz} voxels of {h:.4g} µm ({info['seconds']:.1f} s)"
        + (f"; particles touch at {n_touch:,} voxel neighbour pairs" if n_touch else ""))
    return labels, info
