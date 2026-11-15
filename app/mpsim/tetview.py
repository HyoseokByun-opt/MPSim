"""A field of the body-fitted tetrahedral mesh drawn in 3D as the mesh holds it (5.10.0).

The viewer's fields live on voxels; a finite-element result sampled there loses what the mesh was made for - the
particles' own surfaces. Here the three faces of the RVE that the report's camera sees are cut through the
tetrahedra: every tetrahedron the face passes through leaves a triangle or a quadrilateral, coloured by its own
value, so that a particle's outline on the face is the exact section of its surface. The same quantity of a voxel
solver (the cross-check) is drawn beside it with the same camera and colour range, as the voxels hold it.

save() writes what a figure needs into the view folder while the mesh is at hand; render() draws it later (the
figure worker, the report).
"""
from __future__ import annotations

import os

import numpy as np


def _unwrap(mesh):
    """Corner coordinates per tetrahedron (n, 4, 3), around its first corner, its centre inside the box."""
    P, T, per, L = mesh["points"], mesh["tets"], mesh["per"], mesh["L"]
    X = P[T]
    d = X - X[:, :1, :]
    for a in range(3):
        if per[a]:
            d[:, :, a] -= L[a] * np.floor(d[:, :, a] / L[a] + 0.5)
    X = X[:, :1, :] + d
    c = X.mean(axis=1)
    for a in range(3):
        if per[a]:
            X[:, :, a] -= (L[a] * np.floor(c[:, a] / L[a]))[:, None]
    return X


def _section(X, val, a, c):
    """The polygons the plane x_a = c leaves in the tetrahedra X: (points (k, 4, 3), corners per polygon, values)."""
    z = X[:, :, a] - c
    hit = np.flatnonzero((z.min(axis=1) < 0) & (z.max(axis=1) > 0))
    if not len(hit):
        return np.zeros((0, 4, 3)), np.zeros(0, np.int64), np.zeros(0)
    x, z = X[hit], z[hit]
    above = z > 0
    na = above.sum(axis=1)
    order = np.argsort(above, axis=1, kind="stable")              # below first
    r = np.arange(len(hit))

    def cut(i, j):
        with np.errstate(divide="ignore", invalid="ignore"):      # rows of the other kind are not used
            t = z[r, i] / (z[r, i] - z[r, j])
            return x[r, i] + t[:, None] * (x[r, j] - x[r, i])
    out = np.zeros((len(hit), 4, 3))
    nc = np.where(na == 2, 4, 3)
    lone = np.where(na == 1, order[:, 3], order[:, 0])
    oth = np.where((na == 1)[:, None], order[:, 0:3], order[:, 1:4])
    tri = na != 2
    for q in range(3):
        out[tri, q] = cut(lone, oth[:, q])[tri]
    out[tri, 3] = out[tri, 2]
    quad = ~tri
    b0, b1, a0, a1 = order[:, 0], order[:, 1], order[:, 2], order[:, 3]
    for q, (i, j) in enumerate(((a0, b0), (a0, b1), (a1, b1), (a1, b0))):
        out[quad, q] = cut(i, j)[quad]
    return out, nc, np.asarray(val)[hit]


def box_faces(mesh, val, eps=1e-6):
    """The faces x = L, y = L, z = L of the RVE as the polygons the tetrahedra leave on them: (points (m, 3),
    VTK face array, value per polygon)."""
    import pyvista as pv
    X = _unwrap(mesh)
    L, per = mesh["L"], mesh["per"]
    pts, cells, vals = [], [], []
    n0 = 0
    for a in range(3):
        c = L[a] * (1.0 - eps)
        parts = [_section(X, val, a, c)]
        if per[a]:
            Xs = X.copy()
            Xs[:, :, a] += L[a]
            parts.append(_section(Xs, val, a, c))
        P4 = np.concatenate([p[0] for p in parts])
        nc = np.concatenate([p[1] for p in parts])
        v = np.concatenate([p[2] for p in parts])
        if not len(P4):
            continue
        # copies across the in-plane periods, then cut to the box
        reps = [(P4, nc, v)]
        for b in range(3):
            if b == a or not per[b]:
                continue
            more = []
            for Pq, nq, vq in reps:
                lo, hi = Pq[:, :, b].min(axis=1), Pq[:, :, b].max(axis=1)
                for sh, m in ((L[b], lo < 0), (-L[b], hi > L[b])):
                    if m.any():
                        Pm = Pq[m].copy()
                        Pm[:, :, b] += sh
                        more.append((Pm, nq[m], vq[m]))
            reps += more
        P4 = np.concatenate([p[0] for p in reps])
        nc = np.concatenate([p[1] for p in reps])
        v = np.concatenate([p[2] for p in reps])
        P4[:, :, a] = L[a]
        k = len(P4)
        idx = np.arange(4 * k).reshape(k, 4)
        fa = np.empty((k, 5), np.int64)
        fa[:, 0] = nc
        fa[:, 1:] = idx
        tri = nc == 3
        # triangles as their own cells (a padded quadrilateral would repeat a corner)
        faces = np.concatenate([np.column_stack([np.full(int(tri.sum()), 3), idx[tri, :3]]).ravel(),
                                fa[~tri].ravel()])
        poly = pv.PolyData(P4.reshape(-1, 3), faces=faces)
        poly.cell_data["f"] = np.concatenate([v[tri], v[~tri]])
        for b in range(3):
            if b == a:
                continue
            nrm = [0.0, 0.0, 0.0]
            nrm[b] = 1.0
            poly = poly.clip(normal=nrm, origin=[0.0, 0.0, 0.0], invert=False)
            org = [0.0, 0.0, 0.0]
            org[b] = float(L[b])
            poly = poly.clip(normal=nrm, origin=org, invert=True)
        poly = poly.triangulate() if poly.n_cells else poly
        if not poly.n_cells:
            continue
        f = np.asarray(poly.faces).reshape(-1, 4)[:, 1:] + n0
        pts.append(np.asarray(poly.points))
        cells.append(f)
        vals.append(np.asarray(poly.cell_data["f"]))
        n0 += poly.n_points
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
    return (np.concatenate(pts).astype(np.float32), np.concatenate(cells).astype(np.int32),
            np.concatenate(vals).astype(np.float32))


def voxel_faces(field):
    """The outer voxel layers of a field on the same three faces: x = L, y = L, z = L."""
    f = np.asarray(field)
    return (np.ascontiguousarray(f[-1, :, :], np.float32), np.ascontiguousarray(f[:, -1, :], np.float32),
            np.ascontiguousarray(f[:, :, -1], np.float32))


def save(view_dir, key, mesh, val, voxel=None):
    """The faces of field `val` (per tetrahedron) for figure `key`; voxel: the voxel solver's field beside it."""
    pts, tri, v = box_faces(mesh, val)
    d = {"pts": pts, "tri": tri, "val": v, "L": np.asarray(mesh["L"], float)}
    if voxel is not None:
        d["vx"], d["vy"], d["vz"] = voxel_faces(voxel)
    path = os.path.join(view_dir, f"tet_{key}.npz")
    np.savez_compressed(path, **d)
    return path


def _voxel_poly(d):
    import pyvista as pv
    L = d["L"]
    out = []
    for a, name in enumerate(("vx", "vy", "vz")):
        f = d[name]
        b, c = [q for q in range(3) if q != a]
        n1, n2 = f.shape
        g1, g2 = np.linspace(0, L[b], n1 + 1), np.linspace(0, L[c], n2 + 1)
        G1, G2 = np.meshgrid(g1, g2, indexing="ij")
        P = np.zeros((G1.size, 3))
        P[:, a], P[:, b], P[:, c] = L[a], G1.ravel(), G2.ravel()
        i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        v0 = (i * (n2 + 1) + j).ravel()
        quads = np.column_stack([np.full(v0.size, 4), v0, v0 + n2 + 1, v0 + n2 + 2, v0 + 1]).ravel()
        poly = pv.PolyData(P, faces=quads)
        poly.cell_data["f"] = f.ravel()
        out.append(poly)
    return out[0].merge(out[1:], merge_points=False)


def render(view_dir, req, out_png):
    """The figure of save(): req {"key", "label", "unit", "log", "title", "width", "height", "colormap", "clim",
    "edges", "voxel_title", "mesh_title"}. With a voxel field in the file, two panels with one colour range."""
    import pyvista as pv
    d = np.load(os.path.join(view_dir, f"tet_{req['key']}.npz"))
    L = d["L"]
    both = "vx" in d.files and req.get("compare", True)
    W, H = int(req.get("width", 1400)), int(req.get("height", 1100))
    cmap = req.get("colormap", "jet")
    log = bool(req.get("log"))
    tri = d["tri"]
    mesh = pv.PolyData(d["pts"].astype(float), faces=np.column_stack([np.full(len(tri), 3), tri]).ravel())
    mesh.cell_data["f"] = d["val"]
    vox = _voxel_poly(d) if both else None
    clim = req.get("clim")
    if not clim:
        allv = np.concatenate([d["val"]] + ([np.asarray(vox.cell_data["f"])] if both else []))
        allv = allv[np.isfinite(allv)]
        if log:
            pos = allv[allv > 0]
            clim = [float(np.percentile(pos, 1)), float(np.percentile(pos, 99.5))] if len(pos) else [1e-12, 1.0]
        else:
            clim = [float(np.percentile(allv, 1)), float(np.percentile(allv, 99))]
        if clim[1] <= clim[0]:
            clim[1] = clim[0] + 1e-12
    n = 2 if both else 1
    p = pv.Plotter(off_screen=True, shape=(1, n), window_size=(W * n, H), border=False)
    p.set_background("white")
    sbar = {"title": " ", "vertical": True, "position_x": 0.88, "position_y": 0.2, "height": 0.55, "width": 0.05,
            "title_font_size": 1, "label_font_size": 18, "fmt": "%.3g", "color": "black", "n_labels": 6}
    box = pv.Box(bounds=(0, L[0], 0, L[1], 0, L[2]))
    panels = ([(vox, req.get("voxel_title", "On the voxels"))] if both else []) + \
        [(mesh, req.get("mesh_title", "On the tetrahedral mesh fitted to the particles"))]
    for q, (poly, title) in enumerate(panels):
        p.subplot(0, q)
        last = q == n - 1
        p.add_mesh(poly, scalars="f", preference="cell", cmap=cmap, clim=clim, log_scale=log, lighting=False,
                   show_edges=bool(req.get("edges")) and poly is mesh, edge_color="#1b1f24", line_width=0.4,
                   show_scalar_bar=last, scalar_bar_args=sbar if last else None)
        p.add_mesh(box.outline(), color="black", line_width=2)
        p.add_text(str(title).replace("|", ""), position="upper_edge", font_size=13, color="black")
        p.camera_position = "iso"
        p.camera.azimuth = -12
        p.camera.zoom(0.92)
        if last:
            import textwrap
            import re
            from . import visual as V
            # VTK reads "|" in a text as a column break
            lab = V._latin(re.sub(r"\|([^|]*)\|", r"\1", req.get("label", ""))
                           + (f" [{req['unit']}]" if req.get("unit") not in (None, "", "-") else ""))
            ta = p.add_text("\n".join(textwrap.wrap(lab, 26)) or lab, position=(int(W - 12), int(0.78 * H)),
                            font_size=11, color="black")
            try:
                tp = ta.GetTextProperty()
                tp.SetJustificationToRight()
                tp.SetVerticalJustificationToBottom()
            except Exception:                                      # noqa: BLE001
                pass
        if req.get("show_axes", True):
            p.show_axes()
    try:
        p.enable_anti_aliasing("fxaa")
    except Exception:                                              # noqa: BLE001
        pass
    p.screenshot(out_png)
    p.close()
    return out_png
