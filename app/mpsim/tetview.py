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


def _unwrap(mesh, sel=None, shifts=False):
    """Corner coordinates per tetrahedron (n, 4, 3), around its first corner, its centre inside the box.
    sel: only these tetrahedra (a whole mesh of millions does not fit in memory as corners). shifts: also the
    periods (n, 3) it was moved by from where the solver placed it (around its first corner) - a potential, which
    rises by one across the RVE, is one higher in a tetrahedron moved up by one period."""
    P, T, per, L = mesh["points"], mesh["tets"], mesh["per"], mesh["L"]
    X = P[T if sel is None else T[sel]]
    d = X - X[:, :1, :]
    for a in range(3):
        if per[a]:
            d[:, :, a] -= L[a] * np.floor(d[:, :, a] / L[a] + 0.5)
    X = X[:, :1, :] + d
    c = X.mean(axis=1)
    k = np.zeros((len(X), 3), np.int8)
    for a in range(3):
        if per[a]:
            f = np.floor(c[:, a] / L[a])
            X[:, :, a] -= (L[a] * f)[:, None]
            k[:, a] = -f.astype(np.int8)
    return (X, k) if shifts else X


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


# =========================================================================
# the mesh in the 3D viewer (5.11.0)
# =========================================================================
# face l of a tetrahedron is the one opposite its corner l
FACE_LOCAL = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], np.int64)
FEM_MAGIC = b"FEM1"
FEM_TRI_MAX = 2_500_000          # triangles the browser is given; above, a corner region of the RVE


def _partners(T):
    """For every face (tetrahedron * 4 + corner opposite), the index of the same face in its neighbour, or -1 (the
    mesh's outer boundary, or a film whose two sides do not share vertices). A periodic mesh shares the vertices
    of opposite box faces, so its box faces have partners."""
    n = len(T)
    F = np.sort(T[:, FACE_LOCAL].reshape(-1, 3), axis=1).astype(np.int64)
    nv = int(T.max()) + 1 if n else 1
    if nv < 2_000_000:
        k = (F[:, 0] * nv + F[:, 1]) * nv + F[:, 2]
        o = np.argsort(k, kind="stable")
        ks = k[o]
        same = ks[1:] == ks[:-1]
        del k, ks
    else:
        o = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
        ks = F[o]
        same = np.all(ks[1:] == ks[:-1], axis=1)
        del ks
    del F
    i = np.flatnonzero(same)
    partner = np.full(4 * n, -1, np.int64)
    partner[o[i]] = o[i + 1]
    partner[o[i + 1]] = o[i]
    return partner


def _clip_box(P, tet, L, lo=None, sh=None):
    """Triangles P (m, 3, 3) cut to the box lo..L (default from 0); their tetrahedra (and periods sh (m, 3))
    carried along. -> (P, tet) or (P, tet, sh)."""
    lo = np.zeros(3) if lo is None else np.asarray(lo, float)
    L = np.asarray(L, float)
    sh = np.zeros((len(P), 3), np.int8) if sh is None else sh
    if not len(P):
        return P, tet, sh
    a, b = P.min(axis=1), P.max(axis=1)
    tol = 1e-9 * float(L.max())
    inside = np.all(a >= lo - tol, axis=1) & np.all(b <= L + tol, axis=1)
    gone = np.any(b <= lo + tol, axis=1) | np.any(a >= L - tol, axis=1)
    cross = ~inside & ~gone
    if not cross.any():
        return P[inside], tet[inside], sh[inside]
    import pyvista as pv
    Q = P[cross]
    k = len(Q)
    poly = pv.PolyData(Q.reshape(-1, 3).astype(np.float64),
                       faces=np.column_stack([np.full(k, 3), np.arange(3 * k).reshape(k, 3)]).ravel())
    poly.cell_data["t"] = tet[cross].astype(np.int64)
    for q in range(3):
        poly.cell_data[f"s{q}"] = sh[cross, q].astype(np.int32)
    for ax in range(3):
        nrm = [0.0, 0.0, 0.0]
        nrm[ax] = 1.0
        org = [0.0, 0.0, 0.0]
        org[ax] = float(lo[ax])
        poly = poly.clip(normal=nrm, origin=org, invert=False)
        if poly.n_cells == 0:
            break
        org[ax] = float(L[ax])
        poly = poly.clip(normal=nrm, origin=org, invert=True)
        if poly.n_cells == 0:
            break
    if poly.n_cells == 0:
        return P[inside], tet[inside], sh[inside]
    poly = poly.triangulate()
    tri = np.asarray(poly.faces).reshape(-1, 4)[:, 1:]
    Pc = np.asarray(poly.points)[tri]
    shc = np.stack([np.asarray(poly.cell_data[f"s{q}"]) for q in range(3)], axis=1).astype(np.int8)
    return (np.concatenate([P[inside], Pc]), np.concatenate([tet[inside], np.asarray(poly.cell_data["t"]).astype(np.int64)]),
            np.concatenate([sh[inside], shc]))


def _orient(P, inner):
    """Triangles turned so that their normal points away from the point `inner` (per triangle)."""
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    flip = (nrm * (inner - P[:, 0])).sum(axis=1) > 0
    P[flip] = P[flip][:, [0, 2, 1]]
    return P


def _caps(mesh, eps=1e-6):
    """The six faces of the RVE as the tetrahedra leave them: triangles (m, 3, 3) on the box faces and the
    tetrahedron each belongs to."""
    P, T, per = mesh["points"], mesh["tets"], mesh["per"]
    L = np.asarray(mesh["L"], float)
    w = 1.5 * float(mesh.get("coarsest_um") or mesh.get("h") or (L.max() / 50.0))
    near = np.zeros(len(T), np.bool_)
    for a in range(3):
        c = P[:, a]
        nn = (c < w) | (c > L[a] - w)
        near |= nn[T].any(axis=1)
    sel = np.flatnonzero(near)
    X, k0 = _unwrap(mesh, sel, shifts=True)
    # the row of each tetrahedron in sel rides along as the "value" of the sections
    rows = np.arange(len(sel)).astype(np.float64)
    outP, outT, outS = [], [], []
    for a in range(3):
        for side in (0, 1):
            c = L[a] * (eps if side == 0 else 1.0 - eps)
            parts = [(_section(X, rows, a, c), 0)]
            if per[a]:
                Xs = X.copy()
                step = -1 if side == 0 else 1
                Xs[:, :, a] += step * L[a]
                parts.append((_section(Xs, rows, a, c), step))
            P4 = np.concatenate([p[0][0] for p in parts])
            nc = np.concatenate([p[0][1] for p in parts])
            rr = np.concatenate([p[0][2] for p in parts]).astype(np.int64)
            sa = np.concatenate([np.full(len(p[0][2]), p[1], np.int8) for p in parts])
            if not len(P4):
                continue
            P4[:, :, a] = L[a] * side
            sh = k0[rr].copy()
            sh[:, a] += sa
            q4 = nc == 4
            tri = np.concatenate([P4[:, [0, 1, 2]], P4[q4][:, [0, 2, 3]]])
            tt = np.concatenate([sel[rr], sel[rr[q4]]])
            ss = np.concatenate([sh, sh[q4]])
            # copies across the in-plane periods, then cut to the face
            for b in range(3):
                if b == a or not per[b]:
                    continue
                lo_b, hi_b = tri[:, :, b].min(axis=1), tri[:, :, b].max(axis=1)
                adds, addt, adds_s = [tri], [tt], [ss]
                for stp, m in ((1, lo_b < 0), (-1, hi_b > L[b])):
                    if m.any():
                        q = tri[m].copy()
                        q[:, :, b] += stp * L[b]
                        adds.append(q)
                        addt.append(tt[m])
                        s2 = ss[m].copy()
                        s2[:, b] += stp
                        adds_s.append(s2)
                tri, tt, ss = np.concatenate(adds), np.concatenate(addt), np.concatenate(adds_s)
            tri, tt, ss = _clip_box(tri, tt, L, sh=ss)
            inner = np.zeros((len(tri), 3)) + 0.5 * L
            outP.append(_orient(tri, inner))
            outT.append(tt)
            outS.append(ss)
    if not outP:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int64), np.zeros((0, 3), np.int8)
    return np.concatenate(outP), np.concatenate(outT), np.concatenate(outS)


def fem_geometry(mesh, drawn, max_tri=FEM_TRI_MAX, log=None):
    """What the 3D viewer draws of the body-fitted mesh: for every drawn label (a filler, a pore phase) the faces
    of its tetrahedra that border another label - the particles' surfaces as the mesh fits them, every face an
    element face - and the cut faces of its tetrahedra on the six faces of the RVE; and the cut faces of the
    others (the matrix) for the field on the RVE faces. -> {"surf": {label: (P (m, 3, 3), tet (m,), periods
    (m, 3))}, "faces": (P, tet, periods), "region": None or [x, y, z] µm of the corner region kept}. periods: by
    how many RVE lengths each triangle was moved from where the solver had its tetrahedron (a potential is
    raised by as many units)."""
    T = mesh["tets"]
    mat = np.asarray(mesh["mat"]).astype(np.int64)
    L = np.asarray(mesh["L"], float)
    partner = _partners(T)
    occ = np.arange(4 * len(T))
    mine = mat[occ // 4]
    other = np.where(partner >= 0, mat[np.maximum(partner, 0) // 4], -1)
    keep = (other != mine) & np.isin(mine, np.asarray(list(drawn), np.int64))
    del other
    occ = occ[keep]
    del keep, partner, mine
    t, l = occ // 4, occ % 4
    sel, inv = np.unique(t, return_inverse=True)
    X, ks = _unwrap(mesh, sel, shifts=True)
    Xt = X[inv]
    P = np.take_along_axis(Xt, FACE_LOCAL[l][:, :, None].repeat(3, axis=2), axis=1)
    opp = Xt[np.arange(len(l)), l]
    sh = ks[inv]
    del Xt, X
    P = _orient(P, opp)
    P, t, sh = _clip_box(P, t, L, sh=sh)
    capP, capT, capS = _caps(mesh)
    surf = {}
    capmat = mat[capT] if len(capT) else np.zeros(0, np.int64)
    for lab in drawn:
        m1, m2 = mat[t] == lab, capmat == lab
        if m1.any() or m2.any():
            surf[int(lab)] = (np.concatenate([P[m1], capP[m2]]), np.concatenate([t[m1], capT[m2]]),
                              np.concatenate([sh[m1], capS[m2]]))
    rest = ~np.isin(capmat, np.asarray(list(drawn), np.int64))
    faces = (capP[rest], capT[rest], capS[rest])
    total = sum(len(v[1]) for v in surf.values()) + len(faces[1])
    region = None
    if total > max_tri:
        # a corner region of the RVE with as many triangles as the browser is given, as the voxel view does
        f = (max_tri / float(total)) ** (1.0 / 3.0)
        R = L * f
        region = R.tolist()
        cut = lambda PT: _clip_box(PT[0], PT[1], R, sh=PT[2])       # noqa: E731
        surf = {k: cut(v) for k, v in surf.items()}
        faces = cut(faces)
        # the region's own faces inside the RVE are not drawn: only what lies on the RVE's faces stays
        if log:
            log(f"    finite-element view: {total:,} triangles, more than the 3D view draws: the corner region "
                f"{' x '.join(f'{v:.3g}' for v in R)} µm")
    return {"surf": surf, "faces": faces, "region": region, "total": int(total)}


def tet_volumes(mesh, chunk=400_000):
    """The volume of every tetrahedron (µm³), in chunks so that a mesh of millions stays small in memory."""
    n = len(mesh["tets"])
    out = np.empty(n)
    for a in range(0, n, chunk):
        X = _unwrap(mesh, np.arange(a, min(n, a + chunk)))
        e1, e2, e3 = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]
        out[a:a + len(X)] = np.abs(np.einsum("ij,ij->i", e1, np.cross(e2, e3))) / 6.0
    return out


def weighted_percentiles(v, w, qs):
    """Percentiles of v weighted by w (a field's range over the volume, not over the element count: the small
    elements at a contact are many and carry the extremes)."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw /= cw[-1] if cw[-1] > 0 else 1.0
    return [float(v[o][min(len(o) - 1, int(np.searchsorted(cw, q / 100.0)))]) for q in qs]


def write_fem(path, P):
    """Triangles (m, 3, 3) µm, every one with its own three corners (so that each element face is shaded flat):
    "FEM1", m, origin (3 float32), step (float32), then uint16 corners - 18 bytes per triangle."""
    import struct
    P = np.asarray(P, np.float64).reshape(-1, 3)
    org = P.min(axis=0) if len(P) else np.zeros(3)
    span = float((P.max(axis=0) - org).max()) if len(P) else 1.0
    step = max(span, 1e-12) / 65535.0
    q = np.clip(np.round((P - org) / step), 0, 65535).astype("<u2")
    with open(path, "wb") as fh:
        fh.write(FEM_MAGIC)
        fh.write(struct.pack("<I4f", len(P) // 3, *org.astype(float), step))
        fh.write(q.tobytes())


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
