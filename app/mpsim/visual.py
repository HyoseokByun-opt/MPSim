"""Rendering: interactive viewer data and publication figures.

One set of fields feeds two outputs.

* ``view/`` - what the browser draws with vtk.js (VTK's web counterpart):
    labels.u8            phase map, downsampled to <= VIEW_MAX per edge
    field_<key>.f32      one scalar field each (float32, x-fastest order),
                         block means on the same grid
    faces_labels.u8      the six outer faces of the RVE at the solver's
    faces_<key>.f32      resolution (labels, and each field)
    surf_<label>.bin     smoothed, box-capped particle surfaces from VTK
                         SurfaceNets (PyVista ``contour_labels``), one skin
                         per particle, the same detail per particle for any
                         RVE size (compact format, see write_surface)
    surf_<label>_<key>.f32  the field just inside each surface point, read
                         from the solver's grid
    vox_<label>.bin      the voxel faces of each label (the "Voxels" style):
                         shared uint16 corners, quads, and the view voxel
                         behind each quad
    vox_<label>_<key>.f32   the field value of each quad's own voxel
    network.json         extracted pore network (pores and throats), optional
    meta.json            shapes, spacing, colours, ranges
  Sections at the solver's resolution are read from structure_labels.tif
  next to view/ (see label_slice).
  Rotation, zoom, sections and colour ranges are then all client-side, so
  moving a section never waits for the server.

* ``render_scene`` / ``slice_figure`` - PyVista off-screen and matplotlib
  images for reports and papers, drawn from the same ``view/`` files with the
  camera and colour range chosen on screen. Fields are drawn as voxel cells
  on the six outer faces of the RVE (and on optional section planes), so the
  contour shows the true voxel values without interpolation across phases.

Array convention everywhere: index [x, y, z]; files are written in Fortran
order (x fastest), which is VTK's point order.
"""
from __future__ import annotations

import json
import math
import os
import struct
import time

import numpy as np

VIEW_MAX = 160
# The particles in the 3D view keep the same detail whatever the size of the
# RVE. Every limit used to be a total - a 400-voxel surface grid, 1.5 million
# triangles, 6 million voxel faces - so a larger box drew each particle with
# less: 250 nm spheres on 15.6 nm voxels looked round in a 5 µm box and broke
# into shards of ~22 triangles in a 10 µm box, although both structures were
# made of the same spheres. Now the surface grid follows the particle size,
# every particle keeps a triangle allowance, and where a box holds more
# particles than the browser can draw at that detail the view shows a corner
# region of it in full detail and says so.
SURF_VOX_PER_FEATURE = 6      # surface grid: voxels across the smallest particle feature
SURF_GRID_MAX = 640           # largest edge of that grid (memory of the extraction)
TRI_MAX = 5_000_000           # triangles the browser is given, all phases together
TRI_REGION = 100              # triangles per particle in a region drawn instead of the whole box
# Below about 60 triangles (quadric decimation) a sphere stops looking like
# one; a region is drawn instead of thinner particles.
TRI_MIN_PER_PARTICLE = 60
TRI_NETWORK = 1_500_000       # a network phase (one connected skin)
VOX_QUADS_MAX = 6_000_000     # voxel faces of the "Voxels" style
FACE_MAX = 1024               # RVE faces at the solver's resolution up to this edge
SURF_MAGIC = b"MPS2"

PHASE_COLORS = ["#d9dde3", "#2f6fdd", "#f28e2b", "#3aa655", "#d64545", "#8e6bbf",
                "#1bb3c8", "#c9a227", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"]
VOID_COLOR = "#7fd3f7"


def _stride(n, max_n):
    return max(1, int(math.ceil(n / max_n)))


def block_mean(a, s):
    a = np.asarray(a)
    if s == 1:
        return a.astype(np.float32)
    nx, ny, nz = (d // s for d in a.shape[:3])
    a = a[:nx * s, :ny * s, :nz * s]
    return a.reshape(nx, s, ny, s, nz, s).mean(axis=(1, 3, 5)).astype(np.float32)


def block_pick(a, s):
    if s == 1:
        return np.asarray(a)
    nx, ny, nz = (d // s for d in a.shape[:3])
    o = s // 2
    return np.ascontiguousarray(a[o::s, o::s, o::s][:nx, :ny, :nz])


def pick_offset_um(s, h):
    """Where block_pick samples, relative to the centre of a coarse voxel of
    s x h: the generator draws particles shifted by this much so that its
    coarse grid tests the same points (see pipeline.surface_owner)."""
    return (s // 2 + 0.5 - 0.5 * s) * h


def _stats(ds):
    fin = ds[np.isfinite(ds)]
    if fin.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    p = np.percentile(fin, [1, 99])
    return float(fin.min()), float(fin.max()), float(p[0]), float(p[1])


def _region(shape, frac_vol, floor=8):
    """Edges of a corner region holding about frac_vol of the box: a cube cut
    from a block, only the in-plane edges cut for a flat film."""
    shape = np.asarray(shape, int)
    if frac_vol >= 1.0:
        return shape.copy()
    flat = shape[2] < 0.5 * min(shape[0], shape[1])
    k = frac_vol ** (0.5 if flat else 1.0 / 3.0)
    out = shape.copy()
    axes = (0, 1) if flat else (0, 1, 2)
    for a in axes:
        out[a] = max(min(floor, shape[a]), int(shape[a] * k))
    return out


def _touch_groups(ids):
    """A group number for every particle number in `ids` (0 = none) such that
    no two particles of one group have voxels next to each other (26
    neighbours): a greedy colouring of the contact graph. A dense packing of
    spheres needs a handful of groups."""
    n = int(ids.max()) if ids.size else 0
    groups = np.zeros(n + 1, np.int64)
    if n == 0:
        return groups

    def sl(d, size):
        return slice(0, size - 1) if d == 1 else (slice(1, size) if d == -1 else slice(None))

    def sl2(d, size):
        return slice(1, size) if d == 1 else (slice(0, size - 1) if d == -1 else slice(None))
    keys = []
    nx, ny, nz = ids.shape
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                a = ids[sl(dx, nx), sl(dy, ny), sl(dz, nz)]
                b = ids[sl2(dx, nx), sl2(dy, ny), sl2(dz, nz)]
                m = (a != b) & (a > 0) & (b > 0)
                if m.any():
                    lo = np.minimum(a[m], b[m]).astype(np.int64)
                    hi_ = np.maximum(a[m], b[m]).astype(np.int64)
                    keys.append(np.unique(lo * (n + 1) + hi_))
                del a, b, m
    if not keys:
        return groups
    k = np.unique(np.concatenate(keys))
    a, b = k // (n + 1), k % (n + 1)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.argsort(src, kind="stable")
    dst = dst[order]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n + 1))])
    for i in range(1, n + 1):
        nb = dst[ptr[i]:ptr[i + 1]]
        nb = nb[nb < i]
        if nb.size == 0:
            continue
        used = np.unique(groups[nb])
        free = np.flatnonzero(np.isin(np.arange(used.size + 1), used, invert=True))
        groups[i] = int(free[0])
    return groups


def write_surface(path, pts, nrm, faces):
    """Compact surface file: points as uint16 on a grid of `step` from
    `origin`, normals as int8, triangles as uint32. A third of the float32
    file, which matters once every particle keeps its own triangles."""
    pts = np.asarray(pts, np.float64)
    org = pts.min(axis=0) if len(pts) else np.zeros(3)
    span = float((pts.max(axis=0) - org).max()) if len(pts) else 1.0
    step = max(span, 1e-12) / 65535.0
    q = np.clip(np.round((pts - org) / step), 0, 65535).astype("<u2")
    n8 = np.clip(np.round(np.asarray(nrm, np.float64) * 127.0), -127, 127).astype("i1")
    with open(path, "wb") as fh:
        fh.write(SURF_MAGIC)
        fh.write(struct.pack("<IIf3f", len(pts), len(faces), step, *org.astype(float)))
        fh.write(q.tobytes())
        fh.write(n8.tobytes())
        pad = (-(28 + 9 * len(pts))) % 4
        fh.write(b"\0" * pad)
        fh.write(np.asarray(faces, "<u4").tobytes())


class ViewWriter:
    """Accumulates viewer files for one realisation.

    feature_um: the smallest particle feature (diameter, thickness, wall) the
    view must show; it sets the grid the smooth surfaces are taken from.
    n_particles: how many particles there are; where so many share the
    triangle budget that a finer grid would only be decimated away, the grid
    is coarser (down to four voxels across the feature).
    """

    def __init__(self, out_dir, labels, table, voxel_um, max_n=VIEW_MAX, feature_um=None, n_particles=None,
                 full_fields=True):
        self.dir = os.path.join(out_dir, "view")
        os.makedirs(self.dir, exist_ok=True)
        self.stride = _stride(max(labels.shape), max_n)
        self.labels = block_pick(labels, self.stride).astype(np.uint8)
        self.shape = list(self.labels.shape)
        self.spacing = float(voxel_um * self.stride)
        self.fields = {}
        self.table = table
        self.voxel_um = float(voxel_um)
        self.full = labels                       # the solver's own grid (not copied)
        self._full_shape = tuple(labels.shape)
        if feature_um:
            # SurfaceNets gives about 9 d^2 triangles to a sphere d voxels
            # across; extract at about three times what each particle keeps
            want = float(SURF_VOX_PER_FEATURE)
            if n_particles:
                want = min(want, max(4.0, 0.56 * math.sqrt(TRI_MAX / float(n_particles))))
            self.surf_stride = max(1, int(float(feature_um) / float(voxel_um) // want))
        else:
            self.surf_stride = _stride(max(labels.shape), 400)
        self.surf_owner = None
        self.surf_labels = labels if self.surf_stride == 1 else block_pick(labels, self.surf_stride)
        self._vox_cells = {}                     # label -> voxel behind each quad, solver grid
        self._vox_out = {}                       # label -> voxel in front of each quad
        self._sides = {}                         # label -> voxels inside/outside each surface point
        self._built = False
        self._old = None                         # sampled files of an adopted view
        colors = []
        ci = 1
        for t in table:
            if t["kind"] == "matrix":
                colors.append(PHASE_COLORS[0])
            elif t["kind"] == "interphase":
                colors.append("#9a9a9a")
            elif t["props"].get("E", 1.0) <= 0:
                colors.append(VOID_COLOR)
            else:
                colors.append(PHASE_COLORS[ci % len(PHASE_COLORS)])
                ci += 1
        counts = np.bincount(labels.ravel(), minlength=len(table))
        self.meta = {
            "shape": self.shape, "spacing_um": self.spacing, "stride": self.stride,
            "full_shape": list(labels.shape), "voxel_um": float(voxel_um),
            "length_um": [float(n * voxel_um) for n in labels.shape],
            "labels": [{"value": i, "name": t["name"], "kind": t["kind"],
                        "color": colors[i], "void": bool(t["props"].get("E", 1.0) <= 0),
                        "vf": float(counts[i] / labels.size)} for i, t in enumerate(table)],
            "fields": [], "vectors": [], "paths": [], "surfaces": [], "network": None,
            "surface_region": None, "faces": None,
            # changes whenever the view is written, so the browser does not
            # reuse files it cached from an earlier build of the same run
            "built": int(time.time() * 1000),
        }
        self.labels.ravel(order="F").tofile(os.path.join(self.dir, "labels.u8"))
        self.full_fields = bool(full_fields)
        self._write_faces()
        self._write_volume()

    # ------------------------------------------------------------ RVE faces
    def _face_stack(self, data, dtype):
        """The six outer faces x=0, x=end, y=0, y=end, z=0, z=end of a voxel
        array on the solver's grid (every fs-th voxel above FACE_MAX), each in
        VTK point order of its 2D image, one after the other."""
        fs = self._fs
        o = fs // 2

        def pick2(a):
            return a if fs == 1 else a[o::fs, o::fs]
        parts = [pick2(data[0, :, :]), pick2(data[-1, :, :]), pick2(data[:, 0, :]), pick2(data[:, -1, :]),
                 pick2(data[:, :, 0]), pick2(data[:, :, -1])]
        return np.concatenate([np.asarray(p, dtype).ravel(order="F") for p in parts])

    def _write_faces(self):
        """The faces of the RVE at the solver's resolution: the 3D view paints
        the box with them, so a particle cut by a face keeps its true outline
        instead of the browser volume's blocks (a quarter of the resolution
        on a 640-voxel grid)."""
        nx, ny, nz = self._full_shape
        self._fs = _stride(max(nx, ny, nz), FACE_MAX)
        fs, o = self._fs, self._fs // 2
        dims = [len(range(o, n, fs)) for n in (nx, ny, nz)]
        self._face_stack(self.full, "<u1").tofile(os.path.join(self.dir, "faces_labels.u8"))
        h = self.voxel_um
        self.meta["faces"] = {"stride": fs, "dims": dims, "spacing_um": h * fs, "first_um": (o + 0.5) * h,
                              "planes_um": [[0.5 * h, (nx - 0.5) * h], [0.5 * h, (ny - 0.5) * h],
                                            [0.5 * h, (nz - 0.5) * h]],
                              "labels": "faces_labels.u8", "fields": {}}

    def _write_volume(self):
        """Every voxel of the structure, for the Voxels style: the browser
        draws it by GPU volume rendering, so a 640-cubed RVE shows all of its
        262 million voxels instead of a reduced copy. Labels in VTK point
        order (x fastest), gzip-compressed and served as such."""
        import gzip
        fn = "vol_labels.u8.gz"
        nz = self._full_shape[2]
        with gzip.open(os.path.join(self.dir, fn), "wb", compresslevel=1) as fh:
            for k0 in range(0, nz, 64):
                fh.write(np.asarray(self.full[:, :, k0:k0 + 64], np.uint8).ravel(order="F").tobytes())
        self.meta["voxel_volume"] = {"labels": fn, "dims": [int(n) for n in self._full_shape],
                                     "spacing_um": self.voxel_um}

    def _write_full(self, key, data, log):
        """The field on the solver's grid, 16 bits per voxel: 65,534 levels
        between its minimum and maximum (of log10 for a log field), 0 where it
        is undefined. The 2D sections, the section planes and the voxel volume
        read it, so they show the solver's resolution, not the block means of
        the browser volume. Written in z slabs to bound the memory."""
        nz = data.shape[2]

        def tr(k0):
            v = np.asarray(data[:, :, k0:k0 + 32], np.float64)
            if log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = np.log10(np.where(v > 0, v, np.nan))
            return v
        lo, hi = np.inf, -np.inf
        for k0 in range(0, nz, 32):
            v = tr(k0)
            f = v[np.isfinite(v)]
            if f.size:
                lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + max(1e-12, 1e-9 * abs(lo))
        fn = f"full_{key}.u16"
        with open(os.path.join(self.dir, fn), "wb") as fh:
            for k0 in range(0, nz, 32):
                v = tr(k0)
                m = np.isfinite(v)
                q = np.zeros(v.shape, np.uint16)
                q[m] = (1.0 + np.round((v[m] - lo) / (hi - lo) * 65534.0)).astype(np.uint16)
                fh.write(q.ravel(order="F").astype("<u2").tobytes())
        return {"file": fn, "lo": lo, "hi": hi, "log": bool(log)}

    # ------------------------------------------------------------ fields
    def add_field(self, key, data, label, unit, group, log=False, diverging=False,
                  note="", categorical=False, in_matrix=False, low_side=None):
        """`categorical` marks a field whose values are class numbers, not a
        measurement. Such a field must be coloured over its whole range: the
        usual 1-99 percentile window collapses when one class fills almost
        everything, and the view comes out a single flat colour with the few
        interesting voxels invisible.

        `in_matrix` reads the field on a surface from the outside - for a
        quantity that only exists in the matrix. `low_side` names a material
        property ("eps_r", "mu_r") and reads the field on whichever side of the
        surface has the lower value of it; see _sample.

        The particle surfaces, the voxel faces and the RVE faces take their
        values from `data` itself, on the solver's grid, while it is at hand;
        the browser volume keeps a block mean.
        """
        ds = block_mean(data, self.stride)
        mn, mx, p1, p99 = _stats(ds)
        fn = f"field_{key}.f32"
        ds.ravel(order="F").astype("<f4").tofile(os.path.join(self.dir, fn))
        self.fields[key] = ds
        if log:
            pos = ds[ds > 0]
            lp1 = float(np.percentile(pos, 1)) if pos.size else 1e-12
        else:
            lp1 = None
        self.meta["fields"] = [f for f in self.meta["fields"] if f["key"] != key]
        self.meta["fields"].append({
            "key": key, "file": fn, "label": label, "unit": unit,
            "group": group, "min": mn, "max": mx, "p01": p1, "p99": p99,
            "log": bool(log), "log_min": lp1, "diverging": bool(diverging),
            "categorical": bool(categorical), "in_matrix": bool(in_matrix),
            "low_side": low_side, "note": note})
        if tuple(np.shape(data)) == self._full_shape:
            data = np.asarray(data)
            ff = f"faces_{key}.f32"
            self._face_stack(data, "<f4").tofile(os.path.join(self.dir, ff))
            self.meta["faces"]["fields"][key] = ff
            if self.full_fields:
                self.meta["fields"][-1]["full"] = self._write_full(key, data, log)
            if self._built:
                for sf in self.meta["surfaces"]:
                    self._sample(sf, key, data, full=True)
                vs = self.meta.get("voxel_style") or {}
                for lab, cells in self._vox_cells.items():
                    if in_matrix and lab in self._vox_out:
                        # a quantity of the matrix (the resin's shear rate)
                        # is zero inside a rigid filler: the face shows the
                        # voxel in front of it, as the smooth surface does
                        cells = self._vox_out[lab]
                    vf = f"vox_{lab}_{key}.f32"
                    data[cells[:, 0], cells[:, 1], cells[:, 2]].astype("<f4").tofile(os.path.join(self.dir, vf))
                    vs["files"][str(lab)]["fields"][key] = vf

    def add_vector(self, key, vec, label, unit, group, field=None, note=""):
        """A vector field, written as x, y, z per voxel for arrow glyphs.

        The scalar magnitude is already stored as a field; the arrows add the
        direction, which is what makes a flux or a velocity map readable.
        """
        v = np.stack([block_mean(np.asarray(vec)[..., k], self.stride) for k in range(3)], axis=-1)
        mag = np.sqrt((v ** 2).sum(-1))
        fin = mag[np.isfinite(mag)]
        fn = f"vec_{key}.f32"
        flat = np.stack([v[..., k].ravel(order="F") for k in range(3)], axis=1)
        np.ascontiguousarray(flat, "<f4").tofile(os.path.join(self.dir, fn))
        self.meta["vectors"] = [x for x in self.meta["vectors"] if x["key"] != key]
        self.meta["vectors"].append({
            "key": key, "file": fn, "label": label, "unit": unit, "group": group,
            "field": field, "note": note,
            "max": float(fin.max()) if fin.size else 1.0,
            "p99": float(np.percentile(fin, 99)) if fin.size else 1.0,
            "p50": float(np.percentile(fin, 50)) if fin.size else 0.0})

    def add_paths(self, key, lines, label, group, kind="streamline", unit="µm",
                  diameter_um=None, note="", summary=None, speed_label=None, speed_unit="-"):
        """Polylines through the structure: field lines, or the route of a
        particle that passes through the pore space. A line's "speed" (the
        magnitude of the field along it) colours it; speed_label and
        speed_unit title that colour bar."""
        lines = [dict(l) for l in (lines or []) if l.get("points_um")]
        if not lines:
            return
        fn = f"paths_{key}.json"
        with open(os.path.join(self.dir, fn), "w", encoding="utf-8") as fh:
            json.dump({"lines": lines, "kind": kind, "diameter_um": diameter_um},
                      fh, separators=(",", ":"))
        sp = [v for l in lines for v in (l.get("speed") or [])]
        self.meta["paths"] = [x for x in self.meta["paths"] if x["key"] != key]
        self.meta["paths"].append({
            "key": key, "file": fn, "label": label, "group": group, "kind": kind,
            "unit": unit, "diameter_um": diameter_um, "note": note,
            "n_lines": len(lines), "summary": summary or {},
            "speed_min": float(min(sp)) if sp else None, "speed_max": float(max(sp)) if sp else None,
            "speed_label": speed_label, "speed_unit": speed_unit})

    def add_particles_anim(self, radii_um, phase_of, frames_um, speeds, box_um, strain_step, names, colors,
                           mu_r=None):
        """The particle dynamics of the viscosity analysis for the viewer: its
        own sheared box (not the RVE), the sphere radii and phase of every
        particle, and frames of the positions (um) and of the speed relative
        to the shear flow (in units of the shear rate times the largest
        radius), float32, frame after frame."""
        fr = np.asarray(frames_um, np.float32)
        sp = np.asarray(speeds, np.float32)
        fr.tofile(os.path.join(self.dir, "dem_pos.f32"))
        sp.tofile(os.path.join(self.dir, "dem_speed.f32"))
        np.asarray(radii_um, np.float32).tofile(os.path.join(self.dir, "dem_radii.f32"))
        np.asarray(phase_of, np.uint8).tofile(os.path.join(self.dir, "dem_phase.u8"))
        self.meta["dem"] = {"n": int(fr.shape[1]), "frames": int(fr.shape[0]), "box_um": float(box_um),
                            "strain_step": float(strain_step), "pos": "dem_pos.f32", "speed": "dem_speed.f32",
                            "radii": "dem_radii.f32", "phase": "dem_phase.u8", "names": list(names),
                            "colors": list(colors), "speed_p99": float(np.percentile(sp, 99)) if sp.size else 1.0,
                            "mu_r": mu_r}

    def add_network(self, view):
        if not view:
            return
        with open(os.path.join(self.dir, "network.json"), "w", encoding="utf-8") as fh:
            json.dump(view, fh, separators=(",", ":"))
        d = np.asarray(view["pores"]["diameter_um"], float)
        td = np.asarray(view["throats"]["diameter_um"], float)
        self.meta["network"] = {"file": "network.json", "n_pores": int(d.size), "n_throats": int(td.size),
                                "d_min": float(d.min()) if d.size else 0.0,
                                "d_max": float(d.max()) if d.size else 1.0}

    def adopt(self, keep=lambda key: True):
        """Carries over the fields, vectors and paths already written to this
        view folder (a continued run): the structure is the same, so they
        still belong to it. Fields computed again replace their old entry.
        Their samples on the surfaces, voxel faces and RVE faces are reused
        where the rebuilt surfaces match the old ones point for point."""
        try:
            with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            return
        if list(old.get("shape") or []) != self.shape:
            return
        kept = []
        for f in old.get("fields") or []:
            p = os.path.join(self.dir, f["file"])
            if not keep(f["key"]) or not os.path.exists(p):
                continue
            self.fields[f["key"]] = np.fromfile(p, "<f4").reshape(tuple(self.shape), order="F")
            self.meta["fields"].append(f)
            kept.append(f["key"])
        for kind in ("vectors", "paths"):
            self.meta[kind] += [x for x in old.get(kind) or []
                                if keep(x["key"]) and os.path.exists(os.path.join(self.dir, x["file"]))]
        if old.get("network") and os.path.exists(os.path.join(self.dir, "network.json")):
            self.meta["network"] = old["network"]
        if old.get("dem") and os.path.exists(os.path.join(self.dir, old["dem"].get("pos", ""))):
            self.meta["dem"] = old["dem"]
        self._old = {"keys": kept,
                     "surf": {int(sf["label"]): (int(sf.get("n_points", -1)), dict(sf.get("fields") or {}))
                              for sf in old.get("surfaces") or []},
                     "vox": {k: (int(v.get("n_quads", -1)), dict(v.get("fields") or {}))
                             for k, v in ((old.get("voxel_style") or {}).get("files") or {}).items()},
                     "faces": dict((old.get("faces") or {}).get("fields") or {}),
                     "face_dims": (old.get("faces") or {}).get("dims")}
        if self._old["face_dims"] == self.meta["faces"]["dims"]:
            for key in kept:
                fn = self._old["faces"].get(key)
                if fn and os.path.exists(os.path.join(self.dir, fn)):
                    self.meta["faces"]["fields"][key] = fn
        if list(old.get("full_shape") or []) == list(self._full_shape):
            for f in self.meta["fields"]:
                full = f.get("full")
                if full and not os.path.exists(os.path.join(self.dir, full.get("file", ""))):
                    f.pop("full", None)

    # ------------------------------------------------------------------
    def set_owner(self, owner):
        """The particle number of every voxel (0 = none), on the surface grid
        (or the solver's grid, then picked). With it every particle gets a
        surface of its own: touching particles of one phase are no longer
        merged into one skin, smoothed into each other or bridged where they
        meet only at a voxel edge."""
        if owner is not None and tuple(owner.shape) == tuple(self.surf_labels.shape):
            self.surf_owner = owner                   # already on the surface grid
        elif owner is not None and tuple(owner.shape) == tuple(self._full_shape):
            self.surf_owner = block_pick(owner, self.surf_stride)
        else:
            self.surf_owner = None

    def _drawn(self):
        return [lab for lab in self.meta["labels"] if lab["kind"] not in ("matrix", "interphase")]

    def _phase_masks(self, grid):
        """(label, mask) of every drawn surface: a hollow or coated particle is
        drawn as two nested skins - its outer surface, in the wall's colour,
        and the core inside it - rather than as the wall region itself, whose
        inner and outer faces lie a couple of voxels apart and would be pulled
        into each other by mesh decimation."""
        core_of = self._core_of()
        for lab in self._drawn():
            v = lab["value"]
            mask = (grid == v)
            if v in core_of:
                mask |= (grid == core_of[v])          # outer skin of the whole particle
            if mask.any():
                yield lab, v, mask, core_of

    def _core_of(self):
        core_of = {}
        for i, t in enumerate(self.table):
            if t["kind"] == "shell":
                for j, u in enumerate(self.table):
                    if u["kind"] == "core" and u.get("phase") == t.get("phase"):
                        core_of[i] = j
        return core_of

    def build_surfaces(self, log=None):
        """Smooth particle surfaces with the same detail per particle for any
        size of RVE.

        The grid they are taken from follows the particle size (about six
        voxels across the smallest feature), not the box, and every particle
        keeps a share of the triangle budget (a mesh within the budget is not
        reduced at all, and no particle gets fewer than 60). A box with more
        particles than the browser can draw at that detail is shown as a
        corner region, and meta.surface_region says which.

        VTK SurfaceNets runs on particle numbers when they are known
        (set_owner), so each particle is its own label and keeps its own skin.
        The faces are then oriented per particle: a face whose normal points
        into its own particle is turned over (VTK 9.5 winds about half the
        faces of a label map inward). The mesh is reduced by quadric
        decimation, which keeps a particle round down to a few dozen
        triangles; the decimation used before cut the 10 µm box's spheres
        into shards.
        """
        import pyvista as pv
        grid, own, ss = self.surf_labels, self.surf_owner, self.surf_stride
        s = self.voxel_um * ss
        drawn = [lab["value"] for lab in self._drawn()]
        shp = np.array(grid.shape)
        # how many particles are drawn: one per particle number, one per phase
        # without numbers (a network)
        m_all = np.isin(grid, drawn)

        def count(g, o, m):
            n = 0
            if o is not None:
                ids = o[m]
                n += int(np.count_nonzero(np.bincount(ids)[1:])) if ids.size else 0
                n += sum(1 for v in drawn if np.any((g == v) & (o == 0)))
            else:
                n += sum(1 for v in drawn if np.any(g == v))
            return n
        n_all = count(grid, own, m_all)
        del m_all
        frac = 1.0
        if n_all * TRI_MIN_PER_PARTICLE > TRI_MAX:
            frac = TRI_MAX / (n_all * float(TRI_REGION))
        frac = min(frac, float(SURF_GRID_MAX) ** 3 / float(np.prod(shp)))
        reg = _region(shp, frac)
        n_reg = n_all
        while frac < 1.0:
            g = grid[:reg[0], :reg[1], :reg[2]]
            o = None if own is None else own[:reg[0], :reg[1], :reg[2]]
            n_reg = count(g, o, np.isin(g, drawn))
            if n_reg * TRI_MIN_PER_PARTICLE <= TRI_MAX and np.all(reg <= SURF_GRID_MAX):
                break
            new = np.maximum((reg * 0.9).astype(int), 4)
            if np.all(new == reg):
                break
            reg = new
        if frac < 1.0:
            grid = grid[:reg[0], :reg[1], :reg[2]]
            own = None if own is None else own[:reg[0], :reg[1], :reg[2]]
            self.meta["surface_region"] = {"um": [float(r * s) for r in reg], "n_particles": int(n_reg),
                                           "n_total": int(n_all)}
            if log:
                log(f"    smooth surfaces: {n_all:,} particles are more than the 3D view draws in full detail; "
                    f"it shows the corner region {' x '.join(f'{r * s:.3g}' for r in reg)} µm "
                    f"({n_reg:,} particles)")
        # each phase may use the budget in proportion to its particles; a mesh
        # within it is not decimated at all (small RVEs keep every triangle)
        allow = TRI_MAX / max(n_reg, 1)
        img = pv.ImageData(dimensions=grid.shape, spacing=(s, s, s), origin=(0.5 * s,) * 3)
        hi = np.array(grid.shape) - 1
        self.meta["surfaces"] = []
        self._sides = {}
        for lab, v, mask, core_of in self._phase_masks(grid):
            # One skin per particle, built in groups of particles none of
            # which touch another of its group (a colouring of the contact
            # graph): extracted together, touching particles shared their rim
            # vertices and the smoothing pulled them into fins and necks, so
            # two spheres in contact looked fused.
            passes = []                           # (label grid, number of particles, network)
            n_ids = 0
            if own is not None:
                ids = np.where(mask, own, 0)
                stray = mask & (ids == 0)             # network phases, trimmed voxels
                n_stray = int(np.count_nonzero(stray))
                network = n_stray > 0.5 * int(np.count_nonzero(mask))
                if network:
                    passes.append((mask.astype(np.int32), 1, True))
                    n_ids = 1
                else:
                    groups = _touch_groups(ids)
                    present = np.flatnonzero(np.bincount(ids.ravel()))[1:]
                    n_ids = len(present)
                    for gk in range(int(groups[present].max()) + 1 if n_ids else 0):
                        members = present[groups[present] == gk]
                        lut = np.zeros(int(present[-1]) + 1, np.int32)
                        lut[members] = np.arange(1, len(members) + 1, dtype=np.int32)
                        passes.append((lut[ids], len(members), False))
                    if n_stray:
                        passes.append((stray.astype(np.int32), 1, True))
                del ids, stray
            else:
                passes.append((mask.astype(np.int32), 1, True))
                n_ids = 1
            P, Nn, F, off = [], [], [], 0
            for scal, k, net in passes:
                target = TRI_NETWORK if net else int(allow * k)
                got = self._skin(img, scal, s, hi, target)
                if got is None:
                    continue
                pts, nrm, faces = got
                P.append(pts)
                Nn.append(nrm)
                F.append(faces + off)
                off += len(pts)
            del passes
            if not P:
                continue
            pts, nrm, faces = np.concatenate(P), np.concatenate(Nn), np.concatenate(F).astype(np.uint32)
            del P, Nn, F
            if v in core_of.values():
                # Where a particle is cut by the box, the core's cap lies in the
                # same plane as the particle's outer cap and the two fight for
                # the same pixels - the cut showed solid wall. The core's cap is
                # moved a tenth of a voxel outwards so the cut shows the core.
                box = np.array(grid.shape, float) * s
                lo = pts <= 0.3 * s
                hib = pts >= box - 0.3 * s
                pts = np.where(lo, pts - 0.1 * s, np.where(hib, pts + 0.1 * s, pts)).astype(np.float32)
            fname = f"surf_{v}.bin"
            write_surface(os.path.join(self.dir, fname), pts, nrm, faces)
            self.meta["surfaces"].append({"label": v, "file": fname, "n_points": int(len(pts)),
                                          "n_tris": int(len(faces)), "n_particles": int(n_ids),
                                          "fields": {}, "_pts": pts, "_nrm": nrm})
            if log:
                log(f"    surface '{lab['name']}': {len(faces):,} triangles"
                    + (f" ({n_ids:,} particles, {len(faces) / max(n_ids, 1):.0f} each)"
                       if own is not None and n_ids > 1 else ""))
        self._built = True

    @staticmethod
    def _skin(img, scal, s, hi, target):
        """SurfaceNets of a label grid, faces turned outwards per label, reduced
        to `target` triangles by quadric decimation. (points, normals, faces)
        or None."""
        import pyvista as pv
        img.point_data["m"] = scal.ravel(order="F")
        surf = img.contour_labels(boundary_style="external", background_value=0,
                                  output_mesh_type="triangles", smoothing=True,
                                  smoothing_iterations=12, smoothing_relaxation=0.4,
                                  scalars="m", pad_background=True)
        if surf.n_cells == 0:
            return None
        lbl = np.asarray(surf.cell_data["boundary_labels"]).reshape(surf.n_cells, -1).max(axis=1)
        pts = np.asarray(surf.points, np.float32)
        tri = np.asarray(surf.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
        p0, p1, p2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
        c = (p0 + p1 + p2) / 3.0
        del p0, p1, p2, surf

        def owner_at(q):
            idx = np.floor(q / s).astype(np.int64)
            inside = np.all((idx >= 0) & (idx <= hi), axis=1)
            idx = np.clip(idx, 0, hi)
            return np.where(inside, scal[idx[:, 0], idx[:, 1], idx[:, 2]], 0)
        score = np.zeros(len(tri), np.int16)
        for d in (0.35, 0.7, 1.05):
            score += (owner_at(c - d * s * fn) == lbl).astype(np.int16)
            score -= (owner_at(c + d * s * fn) == lbl).astype(np.int16)
        flip = score < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        del c, fn, score, flip, lbl
        raw = len(tri)
        surf = pv.PolyData(pts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
        del tri
        if target > 0 and raw > 1.05 * target:
            surf = surf.decimate(1.0 - target / raw, volume_preservation=True)
        surf = surf.compute_normals(point_normals=True, cell_normals=False, consistent_normals=False,
                                    auto_orient_normals=False, split_vertices=False)
        return (np.asarray(surf.points, np.float32), np.array(surf.point_data["Normals"], np.float32),
                np.asarray(surf.faces).reshape(-1, 4)[:, 1:].astype(np.int64))

    def build_voxel_surfaces(self, log=None):
        """The voxels themselves: every face between a drawn label and the rest,
        exactly as the solver sees the structure (the "Voxels" style of the 3D
        view, beside the smooth surfaces). Always the solver's own voxels: a
        box with more faces than the browser can draw is shown as a corner
        region (voxel_style.region_um), never as a coarser pick that would
        no longer be the structure that was solved.

        Written per label as corner points (uint16 grid coordinates, shared
        between faces), quads, and for each quad the view voxel behind it;
        the field value of each quad's own voxel is written per field."""
        grid = self.full
        vs = self.stride
        vshape = np.array(self.shape)
        h = self.voxel_um

        def faces_in(g):
            n = 0
            for lab, v, mask, _ in self._phase_masks(g):
                p = np.pad(mask, 1)
                n += sum(int(np.count_nonzero(np.diff(p, axis=a))) for a in range(3))
            return n
        total = faces_in(grid)
        shp = np.array(grid.shape)
        reg = shp.copy()
        if total > VOX_QUADS_MAX:
            reg = _region(shp, VOX_QUADS_MAX / total)
            while True:
                n = faces_in(grid[:reg[0], :reg[1], :reg[2]])
                new = np.maximum((reg * 0.9).astype(int), 4)
                if n <= VOX_QUADS_MAX or np.all(new == reg):
                    break
                reg = new
            grid = grid[:reg[0], :reg[1], :reg[2]]
        region = None if np.all(reg == shp) else [float(r * h) for r in reg]
        self.meta["voxel_style"] = {"spacing_um": h, "stride": 1, "files": {}, "region_um": region,
                                    "n_faces_total": int(total)}
        self._vox_cells = {}
        self._vox_out = {}
        drawn_quads = 0
        for lab, v, mask, _ in self._phase_masks(grid):
            p = np.pad(mask, 1)
            corners, inside, front = [], [], []
            for ax in range(3):
                a = np.moveaxis(p, ax, 0)
                d = a[1:] != a[:-1]
                i, j, k = np.nonzero(d)
                ph = a[i + 1, j, k]
                ins = np.where(ph, i, i - 1)                        # voxel of the phase behind the face
                out = np.clip(np.where(ph, i - 1, i), 0, a.shape[0] - 3)   # and the one in front of it
                base = np.stack([i, j - 1, k - 1], axis=1)
                q = np.stack([base, base + [0, 1, 0], base + [0, 1, 1], base + [0, 0, 1]], axis=1)
                vox = np.stack([ins, j - 1, k - 1], axis=1)
                vof = np.stack([out, j - 1, k - 1], axis=1)
                inv_ax = np.argsort([ax] + [b for b in range(3) if b != ax])
                corners.append(q[:, :, inv_ax])
                inside.append(vox[:, inv_ax])
                front.append(vof[:, inv_ax])
                del d, i, j, k, ph, ins, out, base, q, vox, vof
            q = np.concatenate(corners)
            vox = np.concatenate(inside).astype(np.int32)
            vof = np.concatenate(front).astype(np.int32)
            del corners, inside, front, p
            if not len(q):
                continue
            n1 = np.array(grid.shape) + 1
            flat = q.reshape(-1, 3).astype(np.int64)
            key = (flat[:, 0] * n1[1] + flat[:, 1]) * n1[2] + flat[:, 2]
            uk, inv = np.unique(key, return_inverse=True)
            pts = np.stack([uk // (n1[1] * n1[2]), (uk // n1[2]) % n1[1], uk % n1[2]], axis=1).astype("<u2")
            quads = inv.reshape(-1, 4).astype("<u4")
            vv = np.minimum(vox.astype(np.int64) // vs, vshape - 1)
            vidx = (vv[:, 0] + vshape[0] * (vv[:, 1] + vshape[1] * vv[:, 2])).astype("<u4")
            fname = f"vox_{v}.bin"
            with open(os.path.join(self.dir, fname), "wb") as fh:
                fh.write(struct.pack("<II", len(pts), len(quads)))
                fh.write(pts.tobytes())
                fh.write(quads.tobytes())
                fh.write(vidx.tobytes())
            self.meta["voxel_style"]["files"][str(v)] = {"file": fname, "n_points": int(len(pts)),
                                                        "n_quads": int(len(quads)), "fields": {}}
            self._vox_cells[v] = vox
            self._vox_out[v] = vof
            drawn_quads += len(quads)
        if log:
            log(f"    voxel faces: {drawn_quads:,}" + (f" in the corner region {' x '.join(f'{r:.3g}' for r in region)}"
                                                       f" µm (the whole RVE has {total:,})" if region else ""))

    # ------------------------------------------------------------ sampling
    def _side_index(self, sf, grid, h):
        """Voxel just inside and just outside every point of a surface, on a
        grid of spacing h. The smoothed skin lies within about a voxel of the
        voxel boundary, so half a voxel along the normal sometimes still lands
        on the wrong side (speckles of the neighbour's value): the first of
        0.5, 1 and 1.5 voxels that is on the wanted side."""
        pts, nrm = sf["_pts"], sf["_nrm"]
        hi = np.array(grid.shape) - 1
        core_of = self._core_of()
        own = [sf["label"]] + ([core_of[sf["label"]]] if sf["label"] in core_of else [])

        def side(sign):
            best, found = None, None
            for d in (0.5, 1.0, 1.5):
                idx = np.clip(np.floor((pts + sign * d * h * nrm) / h).astype(np.int32), 0, hi)
                ins = np.isin(grid[idx[:, 0], idx[:, 1], idx[:, 2]], own)
                ok = ins if sign < 0 else ~ins
                if best is None:
                    best, found = idx, ok
                else:
                    take = ok & ~found
                    best[take] = idx[take]
                    found |= ok
            return best
        inside = side(-1)
        outside = side(+1)
        return (inside, outside, grid[inside[:, 0], inside[:, 1], inside[:, 2]],
                grid[outside[:, 0], outside[:, 1], outside[:, 2]])

    def _sample(self, sf, key, data, full):
        """The field on a surface: just inside it, or outside for a quantity
        of the matrix, or on the side of the lower permittivity/permeability
        for a field concentration (the normal flux is continuous, so the field
        is larger where the material property is smaller; a fixed "outside"
        rule painted hollow-particle cores with the wall's depressed value).
        full: from the solver's grid (data has its shape), otherwise from the
        browser volume."""
        if sf.get("_pts") is None:
            return
        tag = (sf["label"], full)
        if tag not in self._sides:
            self._sides[tag] = (self._side_index(sf, self.full, self.voxel_um) if full
                                else self._side_index(sf, self.labels, self.spacing))
        inside, outside, lab_in, lab_out = self._sides[tag]
        fm = next((f for f in self.meta["fields"] if f["key"] == key), {})
        if fm.get("low_side"):
            pv_ = np.array([float(t["props"].get(fm["low_side"], 1.0)) for t in self.table])
            use_out = pv_[lab_out] < pv_[lab_in]
            idx = np.where(use_out[:, None], outside, inside)
        else:
            idx = outside if fm.get("in_matrix") else inside
        vals = np.asarray(data)[idx[:, 0], idx[:, 1], idx[:, 2]].astype("<f4")
        fn = f"surf_{sf['label']}_{key}.f32"
        vals.tofile(os.path.join(self.dir, fn))
        sf["fields"][key] = fn

    def sample_surfaces(self):
        """Fields that reached the view without their solver-grid values - an
        adopted run's, or any added before the surfaces were built - get their
        old samples back where the surfaces are unchanged, otherwise samples
        from the browser volume."""
        old = self._old or {"surf": {}, "vox": {}}
        for sf in self.meta["surfaces"]:
            n_old, f_old = old["surf"].get(int(sf["label"]), (-1, {}))
            for key in self.fields:
                if key in sf["fields"]:
                    continue
                fn = f_old.get(key)
                if (n_old == sf["n_points"] and fn and os.path.exists(os.path.join(self.dir, fn))
                        and os.path.getsize(os.path.join(self.dir, fn)) == 4 * sf["n_points"]):
                    sf["fields"][key] = fn
                else:
                    self._sample(sf, key, self.fields[key], full=False)
        vst = self.meta.get("voxel_style") or {}
        for lab, vf in (vst.get("files") or {}).items():
            n_old, f_old = old["vox"].get(lab, (-1, {}))
            for key in self.fields:
                fn = f_old.get(key)
                if (key not in vf["fields"] and n_old == vf["n_quads"] and fn
                        and os.path.exists(os.path.join(self.dir, fn))):
                    vf["fields"][key] = fn
        # without its own values a voxel face takes the browser volume's
        # value behind it (vox_<label>.bin carries that index)

    def finish(self):
        meta = json.loads(json.dumps(self.meta, default=lambda o: None))
        for sf in meta["surfaces"]:
            sf.pop("_pts", None)
            sf.pop("_nrm", None)
        # written beside and renamed: the figure worker and the browser read it
        # while the run writes it again after every analysis
        p = os.path.join(self.dir, "meta.json")
        with open(p + ".part", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        for _ in range(20):
            try:
                os.replace(p + ".part", p)
                break
            except PermissionError:             # a reader holds it open (Windows)
                time.sleep(0.05)
        return meta


# =========================================================================
# loading view files (renderer side)
# =========================================================================
def load_view(view_dir):
    with open(os.path.join(view_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    shape = tuple(meta["shape"])
    labels = np.fromfile(os.path.join(view_dir, "labels.u8"), np.uint8).reshape(shape, order="F")
    return meta, labels


def load_field(view_dir, meta, key):
    f = next(f for f in meta["fields"] if f["key"] == key)
    arr = np.fromfile(os.path.join(view_dir, f["file"]), "<f4").reshape(tuple(meta["shape"]), order="F")
    return f, arr


def load_surface(view_dir, sf):
    with open(os.path.join(view_dir, sf["file"]), "rb") as fh:
        head = fh.read(4)
        if head == SURF_MAGIC:
            n, m, step, ox, oy, oz = struct.unpack("<IIf3f", fh.read(24))
            q = np.frombuffer(fh.read(6 * n), "<u2").reshape(n, 3)
            pts = (q.astype(np.float32) * np.float32(step) + np.array([ox, oy, oz], np.float32))
            nrm = np.frombuffer(fh.read(3 * n), "i1").reshape(n, 3).astype(np.float32) / 127.0
            fh.read((-(28 + 9 * n)) % 4)
            tri = np.frombuffer(fh.read(12 * m), "<u4").reshape(m, 3)
        else:                                           # the float32 file of earlier runs
            n, m = struct.unpack("<II", head + fh.read(4))
            pts = np.frombuffer(fh.read(12 * n), "<f4").reshape(n, 3)
            nrm = np.frombuffer(fh.read(12 * n), "<f4").reshape(n, 3)
            tri = np.frombuffer(fh.read(12 * m), "<u4").reshape(m, 3)
    return pts, nrm, tri


# fields of the resin alone, zero inside the fillers; earlier runs sampled
# them on the particle surfaces from inside, and every particle came out
# the bottom colour of the scale
MATRIX_FIELDS = ("visc_gd", "visc_v")


def upgrade_matrix_fields(view_dir):
    """Samples the resin fields of an older run on the particle surfaces again,
    from the resin side, out of the solver-grid copy the run kept (nothing is
    solved again), and marks them as matrix fields. Returns True when
    meta.json was rewritten; False when there was nothing to do."""
    p = os.path.join(view_dir, "meta.json")
    with open(p, encoding="utf-8") as fh:
        meta = json.load(fh)
    vv = meta.get("voxel_volume") or {}
    todo = [f for f in meta.get("fields") or []
            if f["key"] in MATRIX_FIELDS and not f.get("in_matrix") and f.get("full")
            and os.path.exists(os.path.join(view_dir, f["full"]["file"]))]
    if not todo or not vv.get("labels") or not meta.get("surfaces"):
        return False
    import gzip
    nx, ny, nz = (int(n) for n in vv["dims"])
    h = float(vv["spacing_um"])
    with gzip.open(os.path.join(view_dir, vv["labels"]), "rb") as fh:
        grid = np.frombuffer(fh.read(), np.uint8).reshape((nx, ny, nz), order="F")
    hi = np.array(grid.shape) - 1
    fronts = {}
    for sf in meta["surfaces"]:
        pts, nrm, _ = load_surface(view_dir, sf)
        pts, nrm = np.asarray(pts, np.float64), np.asarray(nrm, np.float64)
        best = found = None
        for d in (0.5, 1.0, 1.5):           # the first point off the particle
            idx = np.clip(np.floor((pts + d * h * nrm) / h).astype(np.int64), 0, hi)
            ok = grid[idx[:, 0], idx[:, 1], idx[:, 2]] != sf["label"]
            if best is None:
                best, found = idx, ok
            else:
                take = ok & ~found
                best[take] = idx[take]
                found |= ok
        fronts[sf["label"]] = best
    for f in todo:
        q = _full_memmap(view_dir, meta, f["full"])      # [z, y, x]
        for sf in meta["surfaces"]:
            idx = fronts[sf["label"]]
            vals = _decode(np.asarray(q[idx[:, 2], idx[:, 1], idx[:, 0]]), f["full"])
            vals = np.nan_to_num(vals, nan=float(f.get("min") or 0.0)).astype("<f4")
            fn = f"surf_{sf['label']}_{f['key']}.f32"
            vals.tofile(os.path.join(view_dir, fn))
            sf.setdefault("fields", {})[f["key"]] = fn
        del q
        f["in_matrix"] = True
    # new file URLs, so the browser does not keep the old samples
    meta["built"] = int(time.time() * 1000)
    with open(p + ".part", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    os.replace(p + ".part", p)
    return True


def _full_memmap(view_dir, meta, full):
    nx, ny, nz = meta["full_shape"]
    return np.memmap(os.path.join(view_dir, full["file"]), "<u2", "r", shape=(nz, ny, nx))


def _decode(q, full):
    v = full["lo"] + (q.astype(np.float64) - 1.0) / 65534.0 * (full["hi"] - full["lo"])
    if full.get("log"):
        v = 10.0 ** v
    return np.where(q == 0, np.nan, v)


def field_slice(view_dir, meta, key, axis, index):
    """One section of a field at the solver's resolution, [u, v] with u, v
    the remaining axes in x, y, z order, float32; None when the run kept no
    full-resolution copy of it."""
    f = next((x for x in meta.get("fields") or [] if x["key"] == key), None)
    if not f or not f.get("full") or not os.path.exists(os.path.join(view_dir, f["full"]["file"])):
        return None
    q = _full_memmap(view_dir, meta, f["full"])
    a = "xyz".index(axis)
    k = int(min(max(int(index), 0), q.shape[2 - a] - 1))
    s = q[k] if a == 2 else (q[:, k, :] if a == 1 else q[:, :, k])
    return np.ascontiguousarray(_decode(np.asarray(s).T, f["full"]).astype(np.float32))


def field_volume8(view_dir, meta, key):
    """The field on the solver's grid at 8 bits (code 1..255 between the
    16-bit range, 0 undefined), gzip-compressed, for the voxel volume of the
    browser; written once and kept. Returns the file name or None."""
    import gzip
    f = next((x for x in meta.get("fields") or [] if x["key"] == key), None)
    if not f or not f.get("full") or not os.path.exists(os.path.join(view_dir, f["full"]["file"])):
        return None
    resin = None
    if f.get("in_matrix"):
        # a quantity of the matrix alone (the resin's shear rate) is zero in
        # the fillers, and the voxels drawn are the fillers': each filler
        # voxel beside the matrix shows the matrix beside it, as the smooth
        # surfaces do; deeper voxels (a cut at the box) keep their own value
        mat = [int(lb["value"]) for lb in meta.get("labels") or [] if lb.get("kind") == "matrix"]
        vv = meta.get("voxel_volume") or {}
        if mat and vv.get("labels") and os.path.exists(os.path.join(view_dir, vv["labels"])):
            nx, ny, nz = (int(n) for n in vv["dims"])
            with gzip.open(os.path.join(view_dir, vv["labels"]), "rb") as fh:
                resin = np.isin(np.frombuffer(fh.read(), np.uint8).reshape((nz, ny, nx)), mat)
    fn = f"vol8{'m' if resin is not None else ''}_{key}.u8.gz"
    p = os.path.join(view_dir, fn)
    if not os.path.exists(p):
        q = _full_memmap(view_dir, meta, f["full"])
        with gzip.open(p + ".part", "wb", compresslevel=1) as fh:
            for k0 in range(0, q.shape[0], 32):
                s = np.asarray(q[k0:k0 + 32]).astype(np.uint32)
                if resin is not None:
                    s = _from_matrix(q, resin, k0, min(k0 + 32, q.shape[0]), s)
                c = np.where(s == 0, 0, 1 + (s - 1) * 254 // 65534).astype(np.uint8)
                fh.write(c.tobytes())
        os.replace(p + ".part", p)
    return fn


def _from_matrix(q, resin, k0, k1, s):
    """Slab k0:k1 of the 16-bit field [z, y, x] with every non-matrix voxel
    that touches the matrix (6 neighbours) set to the mean of those matrix
    neighbours."""
    a0, a1 = max(k0 - 1, 0), min(k1 + 1, q.shape[0])
    qv = np.asarray(q[a0:a1]).astype(np.uint32)
    rm = resin[a0:a1]
    v = np.where(rm, qv, 0)
    tot = np.zeros(v.shape, np.uint32)
    cnt = np.zeros(v.shape, np.uint8)
    for ax in range(3):
        for sh in (1, -1):
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if sh == 1:
                src[ax], dst[ax] = slice(0, -1), slice(1, None)
            else:
                src[ax], dst[ax] = slice(1, None), slice(0, -1)
            tot[tuple(dst)] += v[tuple(src)]
            cnt[tuple(dst)] += rm[tuple(src)]
    lo = k0 - a0
    tot, cnt, rm = tot[lo:lo + (k1 - k0)], cnt[lo:lo + (k1 - k0)], rm[lo:lo + (k1 - k0)]
    fill = ~rm & (cnt > 0)
    out = s.copy()
    out[fill] = tot[fill] // cnt[fill]
    return out


def label_slice(view_dir, axis, index):
    """One section of the structure at the solver's resolution, from
    structure_labels.tif (an uncompressed ZYX stack beside view/), as a 2D
    array [u, v] with u, v the remaining axes in x, y, z order. None when
    the run wrote no stack."""
    p = os.path.join(os.path.dirname(os.path.abspath(view_dir)), "structure_labels.tif")
    if not os.path.exists(p):
        return None
    import tifffile
    try:
        zyx = tifffile.memmap(p, mode="r")
    except Exception:                                   # noqa: BLE001
        zyx = tifffile.imread(p)
    a = "xyz".index(axis)
    k = int(min(max(int(index), 0), zyx.shape[2 - a] - 1))
    if a == 2:
        out = np.asarray(zyx[k]).T                     # (y, x) -> [x, y]
    elif a == 1:
        out = np.asarray(zyx[:, k, :]).T               # (z, x) -> [x, z]
    else:
        out = np.asarray(zyx[:, :, k]).T               # (z, y) -> [y, z]
    return np.ascontiguousarray(out)


# =========================================================================
# publication renders
# =========================================================================
def _clim(fmeta, req):
    lo = req.get("vmin")
    hi = req.get("vmax")
    if lo is None or hi is None:
        lo, hi = fmeta["p01"], fmeta["p99"]
        if fmeta.get("diverging"):
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        if req.get("log") and fmeta.get("log_min"):
            lo = max(lo, fmeta["log_min"])
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


_GREEK = {"Δ": "d", "α": "alpha", "σ": "sigma", "τ": "tau", "ε": "eps", "μ": "mu", "φ": "phi", "Λ": "Lambda",
          "₀": "0", "|": "|"}


def _latin(t):
    """VTK's built-in font has Latin glyphs only (µ, ², ³ are fine); Greek
    letters are spelled out rather than dropped."""
    t = "".join(_GREEK.get(ch, ch) for ch in str(t))
    t = "".join(ch for ch in t if ord(ch) < 0x0370)
    return " ".join(t.replace("()", "").split())


def load_paths(view_dir, entry):
    with open(os.path.join(view_dir, entry["file"]), encoding="utf-8") as fh:
        return json.load(fh)


def _draw_paths(p, view_dir, meta, key, sbar=None, show_bar=False):
    """Field lines, or the route of a particle, drawn as tubes.

    Lines carrying the magnitude along them are coloured on one scale for the
    whole set (each tube added on its own would otherwise stretch the colour
    map over its own range, and every line would run blue to red); with
    show_bar the scale gets the bar, and the bar's title is returned.

    A percolation path also carries the particle itself: spheres of the
    diameter that passes, placed on the route, put the result of the analysis
    into the picture instead of leaving it in a table.
    """
    import pyvista as pv
    entry = next((x for x in meta.get("paths", []) if x["key"] == key), None)
    if entry is None:
        return None
    data = load_paths(view_dir, entry)
    s = meta["spacing_um"]
    radius = max(0.3 * s, 0.004 * max(meta["length_um"]))
    lo, hi = entry.get("speed_min"), entry.get("speed_max")
    clim = [float(lo), float(hi)] if lo is not None and hi is not None and hi > lo else None
    title = None
    for line in data["lines"]:
        pts = np.asarray(line["points_um"], float)
        if len(pts) < 2:
            continue
        poly = pv.lines_from_points(pts)
        sp = line.get("speed")
        coloured = bool(sp) and len(sp) == len(pts)
        if coloured:
            poly["speed"] = np.asarray(sp, float)
        tube = poly.tube(radius=radius, n_sides=12)
        if coloured:
            bar = bool(show_bar and sbar is not None and clim is not None and title is None)
            p.add_mesh(tube, scalars="speed", cmap="turbo", clim=clim, show_scalar_bar=bar,
                       scalar_bar_args=sbar if bar else None, smooth_shading=True)
            if bar:
                su = entry.get("speed_unit") or "-"
                title = (entry.get("speed_label") or "Magnitude along the lines") + (f" [{su}]" if su not in ("-", "") else "")
        else:
            p.add_mesh(tube, color="#e36209", show_scalar_bar=False, smooth_shading=True)
    d = entry.get("diameter_um")
    if d and data["lines"]:
        pts = np.asarray(data["lines"][0]["points_um"], float)
        for t in (0.15, 0.5, 0.85):
            p.add_mesh(pv.Sphere(radius=0.5 * d, center=pts[int(t * (len(pts) - 1))], theta_resolution=28,
                                 phi_resolution=28), color="#0b62c4", opacity=0.6, smooth_shading=True,
                       show_scalar_bar=False)
    return title


def _draw_vectors(p, view_dir, meta, key, every=None, scale=1.0, uniform=False):
    """Arrow glyphs of a vector field on a regular subsample of the grid."""
    import pyvista as pv
    entry = next((x for x in meta.get("vectors", []) if x["key"] == key), None)
    if entry is None:
        return
    nx, ny, nz = meta["shape"]
    s = meta["spacing_um"]
    raw = np.fromfile(os.path.join(view_dir, entry["file"]), "<f4").reshape(-1, 3)
    comp = [raw[:, k].reshape((nx, ny, nz), order="F") for k in range(3)]
    if not every:
        every = max(1, int(round(max(nx, ny, nz) / 20)))
    o = every // 2
    sub = np.stack([c[o::every, o::every, o::every] for c in comp], axis=-1)
    axes = [(np.arange(n)[o::every] + 0.5) * s for n in (nx, ny, nz)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    vv = sub.reshape(-1, 3)
    mag = np.sqrt((vv ** 2).sum(1))
    ref = entry.get("p99") or (mag.max() or 1.0)
    keep = mag > 0.02 * ref
    if not keep.any():
        return
    cloud = pv.PolyData(np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])[keep])
    cloud["vec"] = vv[keep]
    cloud["mag"] = mag[keep]
    # as in the browser: length ~ sqrt(magnitude), or one length for all
    cloud["len"] = (np.ones(int(keep.sum())) if uniform
                    else np.minimum(1.35, np.sqrt(mag[keep] / max(ref, 1e-30))))
    arrows = cloud.glyph(orient="vec", scale="len", factor=0.9 * scale * every * s,
                         geom=pv.Arrow(tip_length=0.32, tip_radius=0.13, shaft_radius=0.045))
    p.add_mesh(arrows, scalars="mag", cmap="turbo", show_scalar_bar=False, smooth_shading=True)


def _volume_voxels(p, view_dir, meta, vvol, fmeta, clim, log, cmap, op):
    """Every voxel of the structure as a volume (nearest neighbour, shaded):
    coloured by phase, or by the field (RGBA, from the 16-bit copy) with the
    matrix transparent. One empty voxel is added around the box so that the
    particles cut by its faces are shaded like the rest."""
    import gzip
    import pyvista as pv
    import vtk
    nx, ny, nz = vvol["dims"]
    h = float(vvol["spacing_um"])
    with gzip.open(os.path.join(view_dir, vvol["labels"]), "rb") as fh:
        lab = np.frombuffer(fh.read(), np.uint8).reshape((nx, ny, nz), order="F")
    lab = np.pad(lab, 1)
    drawn = np.array([0.0 if (l["kind"] in ("matrix", "interphase")) else 1.0 for l in meta["labels"]])
    img = pv.ImageData(dimensions=lab.shape, spacing=(h, h, h), origin=(-0.5 * h,) * 3)
    full = fmeta.get("full") if fmeta else None
    if full and os.path.exists(os.path.join(view_dir, full["file"])):
        import matplotlib
        cm = matplotlib.colormaps[cmap]
        q = _full_memmap(view_dir, meta, full)
        rgba = np.zeros(lab.shape + (4,), np.uint8)
        lo, hi = clim
        for k0 in range(0, nz, 32):
            v = _decode(np.asarray(q[k0:k0 + 32]), full).transpose(2, 1, 0)       # -> x, y, z
            if log:
                t = (np.log10(np.maximum(v, 1e-300)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            else:
                t = (v - lo) / (hi - lo)
            c = (cm(np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)) * 255).astype(np.uint8)
            rgba[1:-1, 1:-1, 1 + k0:1 + k0 + c.shape[2], :3] = c[..., :3]
        rgba[..., 3] = (255 * drawn[lab] * op).astype(np.uint8)
        arr = rgba.reshape(-1, 4, order="F")
        img.point_data.set_array(arr, "rgba")
        img.point_data.active_scalars_name = "rgba"
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(img)
        prop = vtk.vtkVolumeProperty()
        prop.IndependentComponentsOff()
        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(0, 0.0)
        otf.AddPoint(1, 1.0)
        otf.AddPoint(255, 1.0)
        prop.SetScalarOpacity(otf)
    else:
        img.point_data["l"] = lab.ravel(order="F")
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(img)
        prop = vtk.vtkVolumeProperty()
        ctf = vtk.vtkColorTransferFunction()
        otf = vtk.vtkPiecewiseFunction()
        for l in meta["labels"]:
            c = [int(l["color"][i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
            ctf.AddRGBPoint(l["value"], *c)
            otf.AddPoint(l["value"], float(drawn[l["value"]]) * op)
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
    prop.SetInterpolationTypeToNearest()
    prop.ShadeOn()
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.75)
    prop.SetSpecular(0.2)
    prop.SetScalarOpacityUnitDistance(0.1 * h)
    mapper.SetSampleDistance(0.5 * h)
    vol = vtk.vtkVolume()
    vol.SetMapper(mapper)
    vol.SetProperty(prop)
    p.renderer.AddVolume(vol)


def render_scene(view_dir, req, out_png):
    """3D figure with PyVista, off-screen.

    req: mode ('structure' | 'field' | 'network'), field, box_faces,
         slices {x, y, z: index}, show_particles, particle_opacity,
         color_particles, show_network, colormap, vmin/vmax, log,
         camera {position, focal_point, view_up, parallel_scale, parallel},
         width, height, show_scalar_bar, show_axes, background
    """
    import pyvista as pv
    pv.OFF_SCREEN = True
    meta, labels = load_view(view_dir)
    s = meta["spacing_um"]
    nx, ny, nz = meta["shape"]
    W = int(req.get("width", 1800))
    H = int(req.get("height", 1400))
    bg = req.get("background", "white")
    ink = "black" if bg in ("white", "#ffffff", "#fff") else "white"
    p = pv.Plotter(off_screen=True, window_size=(W, H))
    p.set_background(bg)
    cmap = req.get("colormap", "jet")
    mode = req.get("mode", "structure")
    fmeta = arr = None
    if mode == "field" and req.get("field"):
        fmeta, arr = load_field(view_dir, meta, req["field"])
    clim = _clim(fmeta, req) if fmeta else None
    log = bool(req.get("log")) and fmeta is not None
    # the bar carries only tick labels; its title is a separate centred text
    # above it, so a long title never collides with the numbers
    bar_title = ""
    if fmeta:
        bar_title = _latin(fmeta["label"] + (f" [{fmeta['unit']}]" if fmeta["unit"] not in ("-", "") else ""))
    elif mode == "network":
        bar_title = "Pore diameter [µm]"
    # vertical bar on the right, as in the interactive viewer: it never runs
    # into the RVE, which the default camera shifts to the left
    sbar = {"title": " ", "vertical": True, "position_x": 0.87, "position_y": 0.2, "height": 0.55,
            "width": 0.04, "title_font_size": 1, "label_font_size": 18, "fmt": "%.3g",
            "color": ink, "n_labels": 6}
    bar_done = False

    # voxel cells: a face shows the true value of the voxel behind it
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(s, s, s), origin=(0.0, 0.0, 0.0))
    grid.cell_data["labels"] = labels.ravel(order="F")
    if arr is not None:
        a = np.maximum(arr, clim[0] * 1e-3) if log else arr
        grid.cell_data["f"] = a.ravel(order="F")

    show_particles = req.get("show_particles", mode != "field")
    op = float(req.get("particle_opacity", 1.0))
    vvol = meta.get("voxel_volume") if req.get("surface_style") == "voxels" else None
    if vvol and show_particles:
        _volume_voxels(p, view_dir, meta, vvol, fmeta if (mode == "field" and req.get("color_particles", True))
                       else None, clim, log, cmap, op)
    vox = meta.get("voxel_style") if (req.get("surface_style") == "voxels" and not vvol) else None
    for key_, vf in ((vox or {}).get("files") or {}).items():
        # the voxel faces, flat shaded, each coloured by its own voxel
        if not show_particles:
            break
        lab = meta["labels"][int(key_)]
        with open(os.path.join(view_dir, vf["file"]), "rb") as fh:
            n, m = struct.unpack("<II", fh.read(8))
            vp = np.frombuffer(fh.read(6 * n), "<u2").reshape(n, 3).astype(np.float64) * vox["spacing_um"]
            vq = np.frombuffer(fh.read(16 * m), "<u4").reshape(m, 4).astype(np.int64)
            vi = np.frombuffer(fh.read(4 * m), "<u4")
        mesh = pv.PolyData(vp, np.hstack([np.full((m, 1), 4, np.int64), vq]).ravel())
        if mode == "field" and fmeta and req.get("color_particles", True):
            own_f = (vf.get("fields") or {}).get(fmeta["key"])
            flat = (np.fromfile(os.path.join(view_dir, own_f), "<f4") if own_f
                    else arr.ravel(order="F")[vi])
            mesh.cell_data["f"] = np.maximum(flat, clim[0] * 1e-3) if log else flat
            p.add_mesh(mesh, scalars="f", preference="cell", cmap=cmap, clim=clim, log_scale=log,
                       smooth_shading=False, opacity=op, show_scalar_bar=not bar_done, scalar_bar_args=sbar)
            bar_done = True
        else:
            p.add_mesh(mesh, color=lab["color"], smooth_shading=False, opacity=op, specular=0.2)
    for sf in meta["surfaces"]:
        if not show_particles or vox or vvol:
            break
        lab = meta["labels"][sf["label"]]
        pts, nrm, tri = load_surface(view_dir, sf)
        faces = np.hstack([np.full((len(tri), 1), 3, np.int64), tri.astype(np.int64)]).ravel()
        mesh = pv.PolyData(pts.astype(np.float64), faces)
        mesh.point_data["Normals"] = nrm
        if mode == "field" and fmeta and sf["fields"].get(fmeta["key"]) and req.get("color_particles", True):
            vals = np.fromfile(os.path.join(view_dir, sf["fields"][fmeta["key"]]), "<f4")
            mesh.point_data["f"] = np.maximum(vals, clim[0] * 1e-3) if log else vals
            p.add_mesh(mesh, scalars="f", cmap=cmap, clim=clim, log_scale=log, smooth_shading=True,
                       opacity=op, specular=0.25, show_scalar_bar=not bar_done, scalar_bar_args=sbar)
            bar_done = True
        else:
            p.add_mesh(mesh, color=lab["color"], smooth_shading=True, opacity=op, specular=0.35,
                       specular_power=20)
    if mode == "field" and fmeta:
        pieces = []
        fc = meta.get("faces") or {}
        face_file = (fc.get("fields") or {}).get(fmeta["key"])
        if req.get("box_faces", True) and face_file:
            # the faces as the solver resolved them, not the browser volume's blocks
            vals = np.fromfile(os.path.join(view_dir, face_file), "<f4")
            if log:
                vals = np.maximum(vals, clim[0] * 1e-3)
            fx, fy, fz = fc["dims"]
            fsp = fc["spacing_um"]
            L = meta["length_um"]
            off = 0
            for ax, (d1, d2) in enumerate(((fy, fz), (fx, fz), (fx, fy))):
                for side in (0, 1):
                    dims = [d1 + 1, d2 + 1]
                    dims.insert(ax, 1)
                    org = [0.0, 0.0, 0.0]
                    org[ax] = 0.0 if side == 0 else L[ax]
                    spc = [fsp, fsp, fsp]
                    spc[ax] = 1.0
                    face = pv.ImageData(dimensions=dims, spacing=spc, origin=org)
                    face.cell_data["f"] = vals[off:off + d1 * d2]
                    off += d1 * d2
                    pieces.append(face)
        elif req.get("box_faces", True):
            pieces.append(grid.extract_surface(algorithm="dataset_surface"))
        for ax, idx in (req.get("slices") or {}).items():
            if idx is None:
                continue
            k = "xyz".index(ax)
            origin = [0.5 * n_ * s for n_ in (nx, ny, nz)]
            origin[k] = (int(idx) + 0.5) * s
            normal = [0.0, 0.0, 0.0]
            normal[k] = 1.0
            pieces.append(grid.slice(normal=normal, origin=origin))
        for pc in pieces:
            if pc.n_cells == 0:
                continue
            p.add_mesh(pc, scalars="f", preference="cell", cmap=cmap, clim=clim, log_scale=log,
                       lighting=False, show_scalar_bar=(req.get("show_scalar_bar", True) and not bar_done),
                       scalar_bar_args=sbar)
            bar_done = True
    if req.get("vectors"):
        _draw_vectors(p, view_dir, meta, req["vectors"], every=req.get("vector_every"),
                      scale=float(req.get("vector_scale", 1.0)), uniform=bool(req.get("vector_uniform")))
    if req.get("paths"):
        pt = _draw_paths(p, view_dir, meta, req["paths"], sbar=sbar, show_bar=not bar_done)
        if pt:
            bar_done = True
            bar_title = _latin(pt)
    net = meta.get("network")
    if net and (mode == "network" or req.get("show_network")):
        with open(os.path.join(view_dir, net["file"]), encoding="utf-8") as fh:
            nw = json.load(fh)
        P = np.asarray(nw["pores"]["xyz_um"], float)
        D = np.asarray(nw["pores"]["diameter_um"], float)
        if P.size:
            cloud = pv.PolyData(P)
            cloud["d"] = D
            balls = cloud.glyph(scale="d", orient=False, factor=1.0,
                                geom=pv.Sphere(radius=0.5, theta_resolution=18, phi_resolution=12))
            p.add_mesh(balls, scalars="d", cmap=cmap, smooth_shading=True, specular=0.3,
                       show_scalar_bar=not bar_done, scalar_bar_args=sbar)
            bar_done = True
            C = np.asarray(nw["throats"]["conns"], np.int64).reshape(-1, 2)
            if C.size:
                seg = P[C]
                # periodic images: skip throats longer than half the box
                keep = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1) < 0.5 * min(nx, ny, nz) * s
                C = C[keep]
                lines = np.hstack([np.full((len(C), 1), 2, np.int64), C]).ravel()
                p.add_mesh(pv.PolyData(P, lines=lines), color="#5b6570", line_width=2.0)
        if mode == "network" and req.get("show_particles") is None:
            pass
    p.add_mesh(grid.outline(), color=ink, line_width=2)
    if bar_done and bar_title:
        import textwrap
        wrapped = "\n".join(textwrap.wrap(bar_title, 24)) or bar_title
        ta = p.add_text(wrapped, position=(int(W - 18), int(0.77 * H)), font_size=12, color=ink)
        try:
            tp = ta.GetTextProperty()
            tp.SetJustificationToRight()
            tp.SetVerticalJustificationToBottom()
        except Exception:                                      # noqa: BLE001
            pass
    if req.get("show_axes", True):
        p.show_axes()
    cam = req.get("camera")
    if cam:
        p.camera.position = cam["position"]
        p.camera.focal_point = cam["focal_point"]
        p.camera.up = cam["view_up"]
        if cam.get("parallel"):
            p.camera.enable_parallel_projection()
            if cam.get("parallel_scale"):
                p.camera.parallel_scale = cam["parallel_scale"]
        if cam.get("view_angle"):
            p.camera.view_angle = cam["view_angle"]
    else:
        p.camera_position = "iso"
        p.camera.azimuth = -12
        p.camera.zoom(0.9 if bar_done else 1.0)
        if bar_done:
            # move the view to the right so the RVE sits left of the colour bar
            pos = np.array(p.camera.position, float)
            foc = np.array(p.camera.focal_point, float)
            up = np.array(p.camera.up, float)
            right = np.cross(foc - pos, up)
            nr = np.linalg.norm(right)
            if nr > 0:
                shift = right / nr * 0.13 * max(nx, ny, nz) * s
                p.camera.position = (pos + shift).tolist()
                p.camera.focal_point = (foc + shift).tolist()
    try:
        p.enable_anti_aliasing("fxaa")
    except Exception:                                          # noqa: BLE001
        pass
    p.screenshot(out_png)
    p.close()
    return out_png


def slice_figure(view_dir, req, out_png):
    """2D section with phase outlines, µm axes and a colour bar (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    meta, labels = load_view(view_dir)
    s = meta["spacing_um"]
    ax_name = req.get("axis", "z")
    a = "xyz".index(ax_name)
    idx = int(req.get("index", meta["shape"][a] // 2))
    idx = max(0, min(meta["shape"][a] - 1, idx))
    lab2 = np.take(labels, idx, axis=a)
    other = [i for i in range(3) if i != a]
    ext = [0, meta["shape"][other[0]] * s, 0, meta["shape"][other[1]] * s]
    # the structure and its outlines at the solver's resolution (the field
    # stays on the browser volume's grid); the same extent as the view
    st = int(meta.get("stride", 1))
    full = label_slice(view_dir, ax_name, idx * st + st // 2) if st > 1 else None
    if full is not None:
        lab2 = full[:meta["shape"][other[0]] * st, :meta["shape"][other[1]] * st]
    s_lab = s / st if full is not None else s
    # a section across a film is a strip (100 x 12 µm): a wide figure with the
    # colour bar underneath, instead of a thin band beside a full-height bar
    wide = ext[3] / ext[1] < 0.4
    size = (9.6, max(2.6, 9.6 * ext[3] / ext[1] + 2.0)) if wide else (6.4, 5.2)
    fig, axp = plt.subplots(figsize=size, dpi=int(req.get("dpi", 220)))
    if req.get("field"):
        fmeta, arr = load_field(view_dir, meta, req["field"])
        f2 = np.take(arr, idx, axis=a)
        lo, hi = _clim(fmeta, req)
        norm = LogNorm(vmin=max(lo, 1e-30), vmax=hi) if req.get("log") else Normalize(vmin=lo, vmax=hi)
        im = axp.imshow(np.ma.masked_invalid(f2.T), origin="lower", extent=ext, cmap=req.get("colormap", "jet"),
                        norm=norm, interpolation="nearest")
        cb = (fig.colorbar(im, ax=axp, orientation="horizontal", fraction=0.12, pad=0.32, aspect=45) if wide
              else fig.colorbar(im, ax=axp, fraction=0.046, pad=0.03))
        cb.set_label(f"{fmeta['label']} [{fmeta['unit']}]")
        title = fmeta["label"]
    else:
        cols = [l["color"] for l in meta["labels"]]
        axp.imshow(lab2.T, origin="lower", extent=ext, cmap=ListedColormap(cols), vmin=-0.5,
                   vmax=len(cols) - 0.5, interpolation="nearest")
        title = "Structure"
    if req.get("outlines", True):
        xs = (np.arange(lab2.shape[0]) + 0.5) * s_lab
        ys = (np.arange(lab2.shape[1]) + 0.5) * s_lab
        for lab in meta["labels"]:
            if lab["kind"] in ("matrix", "interphase"):
                continue
            m = (lab2 == lab["value"]).astype(float)
            if 0 < m.sum() < m.size:
                axp.contour(xs, ys, m.T, levels=[0.5], colors="k", linewidths=0.6)
    # the in-plane part of the vector field, as in the browser's section view:
    # length ~ sqrt(magnitude) (or one length), direction as solved
    ventry = next((x for x in meta.get("vectors", []) if x["key"] == req.get("vectors")), None)
    if ventry is not None:
        raw = np.fromfile(os.path.join(view_dir, ventry["file"]), "<f4").reshape(-1, 3)
        comp = [raw[:, k].reshape(tuple(meta["shape"]), order="F") for k in range(3)]
        u2, v2 = np.take(comp[other[0]], idx, axis=a), np.take(comp[other[1]], idx, axis=a)
        every = int(req.get("vector_every") or max(1, round(max(u2.shape) / 18)))
        o = every // 2
        uu, vv = u2[o::every, o::every], v2[o::every, o::every]
        mag = np.hypot(uu, vv)
        ref = float(ventry.get("p99") or (mag.max() or 1.0))
        keep = mag > 0.02 * ref
        ln = np.ones_like(mag) if req.get("vector_uniform") else np.minimum(1.0, np.sqrt(mag / max(ref, 1e-30)))
        with np.errstate(invalid="ignore", divide="ignore"):
            du, dv = np.where(keep, uu / mag * ln, 0.0), np.where(keep, vv / mag * ln, 0.0)
        gx = (np.arange(u2.shape[0])[o::every] + 0.5) * s
        gy = (np.arange(u2.shape[1])[o::every] + 0.5) * s
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        L = 0.95 * every * s * float(req.get("vector_scale", 1.0))
        axp.quiver(GX[keep], GY[keep], du[keep] * L, dv[keep] * L, angles="xy", scale_units="xy", scale=1,
                   color="#111418", width=0.0032, headwidth=4.2, headlength=4.6, pivot="middle", alpha=0.85)
    names = ["x", "y", "z"]
    axp.set_xlabel(f"{names[other[0]]} (µm)")
    axp.set_ylabel(f"{names[other[1]]} (µm)")
    axp.set_title(f"{title} — section {ax_name} = {(idx + 0.5) * s:.3g} µm", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
