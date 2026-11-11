"""FEniCSx (DOLFINx) version of MPSim's via thermal stress (solvers/via_stress.py), for verification - not imported
by MPSim. The same axisymmetric linear elastic model on P2 triangles meshed by gmsh; MPSim's self-test compares its
own 9-node quadrilateral solution with values computed by this script (agreement within 0.3 %).

Runs where FEniCSx runs: Linux or WSL (conda-forge: fenics-dolfinx mpich python-gmsh scipy). On Windows the
conda-forge build has no PETSc (the SciPy path below is used) and compiles its forms with a C compiler, so it needs
the Microsoft C++ Build Tools.

    python fenicsx_tgv.py cases.json out.json [maps]

cases.json: [{"name", "t_end" (um), "throw", "core": "paste"|"cu"|"none", "dT" (K),
              "mats": {"cu"|"glass"|"paste": {"E" (MPa), "nu", "alpha" (1/K)}}}]
"""
import json
import math
import os
import sys
import time

import gmsh
import numpy as np
import ufl
from dolfinx import fem, geometry
from mpi4py import MPI

try:                         # PETSc where available (Linux, macOS); dolfinx's own assembly and SciPy otherwise (Windows)
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc
    HAVE_PETSC = True
except ImportError:
    HAVE_PETSC = False

    class PETSc:  # noqa: N801
        ScalarType = np.float64

try:
    from dolfinx.io import gmsh as gmshio
except ImportError:  # older dolfinx
    from dolfinx.io import gmshio

COMM = MPI.COMM_WORLD
A_UM = 50.0          # hole radius
H_UM = 900.0         # half the glass thickness (1.8 mm)


def wall_t(t_end, throw, z):
    s = np.asarray(z, float) / H_UM
    return t_end * (throw + (1.0 - throw) * s ** 2)


def build_mesh(t_end, throw, core, R, h_wall=None, h_corner=0.15, h_far=25.0):
    """gmsh: core (if any), wall and glass surfaces; returns mesh, cell tags (1 core, 2 wall, 3 glass), facet tags
    (11 axis, 12 bottom, 13 top, 14 rim, 15 inner wall surface of a hollow via)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("tgv")
    geo = gmsh.model.geo
    nz = 61
    zs = np.linspace(0.0, H_UM, nz)
    b = A_UM - wall_t(t_end, throw, zs)
    pb = [geo.addPoint(float(bb), float(z), 0.0) for bb, z in zip(b, zs)]
    p_a0, p_aH = geo.addPoint(A_UM, 0, 0), geo.addPoint(A_UM, H_UM, 0)
    p_R0, p_RH = geo.addPoint(R, 0, 0), geo.addPoint(R, H_UM, 0)
    spl = geo.addSpline(pb)                       # wall inner surface, bottom -> top
    l_a = geo.addLine(p_a0, p_aH)                 # wall / glass interface
    l_wb, l_wt = geo.addLine(pb[0], p_a0), geo.addLine(p_aH, pb[-1])
    l_gb, l_gt = geo.addLine(p_a0, p_R0), geo.addLine(p_RH, p_aH)
    l_rim = geo.addLine(p_R0, p_RH)
    s_wall = geo.addPlaneSurface([geo.addCurveLoop([l_wb, l_a, l_wt, -spl])])
    s_glass = geo.addPlaneSurface([geo.addCurveLoop([l_gb, l_rim, l_gt, -l_a])])
    surfs = [s_wall, s_glass]
    if core:
        p_00, p_0H = geo.addPoint(0, 0, 0), geo.addPoint(0, H_UM, 0)
        l_cb, l_ax, l_ct = geo.addLine(p_00, pb[0]), geo.addLine(p_0H, p_00), geo.addLine(pb[-1], p_0H)
        s_core = geo.addPlaneSurface([geo.addCurveLoop([l_cb, spl, l_ct, l_ax])])
        surfs.append(s_core)
    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [s_wall], 2)
    gmsh.model.addPhysicalGroup(2, [s_glass], 3)
    gmsh.model.addPhysicalGroup(1, [l_wb, l_gb] + ([l_cb] if core else []), 12)
    gmsh.model.addPhysicalGroup(1, [l_wt, l_gt] + ([l_ct] if core else []), 13)
    gmsh.model.addPhysicalGroup(1, [l_rim], 14)
    if core:
        gmsh.model.addPhysicalGroup(2, [s_core], 1)
        gmsh.model.addPhysicalGroup(1, [l_ax], 11)
    else:
        gmsh.model.addPhysicalGroup(1, [spl], 15)
    # mesh size: fine along the wall's two surfaces, finer at the top corners, coarse far away
    t_mid = throw * t_end
    hw = h_wall or min(1.0, t_mid / 4.0)
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "CurvesList", [spl, l_a])
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 600)
    f2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    for k, v in (("SizeMin", hw), ("SizeMax", h_far), ("DistMin", 1.0), ("DistMax", 220.0)):
        gmsh.model.mesh.field.setNumber(f2, k, v)
    f3 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f3, "PointsList", [p_aH, pb[-1]])
    f4 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f4, "InField", f3)
    for k, v in (("SizeMin", h_corner), ("SizeMax", h_far), ("DistMin", 0.3), ("DistMax", 80.0)):
        gmsh.model.mesh.field.setNumber(f4, k, v)
    fm = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(fm, "FieldsList", [f2, f4])
    gmsh.model.mesh.field.setAsBackgroundMesh(fm)
    for k in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(k, 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    md = gmshio.model_to_mesh(gmsh.model, COMM, 0, gdim=2)
    gmsh.finalize()
    if hasattr(md, "mesh"):
        return md.mesh, md.cell_tags, md.facet_tags
    return md[0], md[1], md[2]


def solve(case, R=500.0, fields=None, **mesh_kw):
    """case: {t_end, throw, core: 'paste'|'cu'|'none', mats: {cu, glass, paste}: {E (MPa), nu, alpha (1/K)}, dT}"""
    t0 = time.time()
    core = case["core"] != "none"
    msh, ct, ft = build_mesh(case["t_end"], case.get("throw", 0.8), core, R, **mesh_kw)
    mats = case["mats"]
    region = {2: mats["cu"], 3: mats["glass"]}
    if core:
        region[1] = mats["cu"] if case["core"] == "cu" else mats["paste"]
    Q = fem.functionspace(msh, ("DG", 0))
    lam, mu, beta = fem.Function(Q), fem.Function(Q), fem.Function(Q)
    for tag, m in region.items():
        cells = ct.find(tag)
        E, nu, al = m["E"], m["nu"], m["alpha"]
        lam.x.array[cells] = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu.x.array[cells] = E / (2 * (1 + nu))
        beta.x.array[cells] = al
    dT = case["dT"]
    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    r = x[0]

    def eps(w):
        return ufl.as_tensor([[w[0].dx(0), 0.5 * (w[0].dx(1) + w[1].dx(0)), 0],
                              [0.5 * (w[0].dx(1) + w[1].dx(0)), w[1].dx(1), 0],
                              [0, 0, w[0] / r]])
    I3 = ufl.Identity(3)

    def sig(w):
        e = eps(w) - beta * dT * I3
        return lam * ufl.tr(e) * I3 + 2 * mu * e
    a = ufl.inner(lam * ufl.tr(eps(u)) * I3 + 2 * mu * eps(u), eps(v)) * r * ufl.dx
    L = ufl.inner((3 * lam + 2 * mu) * beta * dT * I3, eps(v)) * r * ufl.dx
    fdim = 1
    bcs = [fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V.sub(1), fdim, ft.find(12)), V.sub(1))]
    if core:
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V.sub(0), fdim, ft.find(11)), V.sub(0)))
    if HAVE_PETSC:
        prob = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="tgv_",
                             petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
        uh = prob.solve()
        if isinstance(uh, tuple):
            uh = uh[0]
    else:
        from scipy.sparse.linalg import spsolve
        A = fem.assemble_matrix(fem.form(a), bcs=bcs)
        A.scatter_reverse()
        b = fem.assemble_vector(fem.form(L))
        b.scatter_reverse(__import__("dolfinx").la.InsertMode.add)
        bc_dofs = np.unique(np.concatenate([bc.dof_indices()[0] for bc in bcs]))
        b.array[bc_dofs] = 0.0
        uh = fem.Function(V)
        uh.x.array[:] = spsolve(A.to_scipy().tocsc(), b.array)
        uh.x.scatter_forward()
    # stress (DG1 tensor, components rr, zz, tt, rz) at points
    W = fem.functionspace(msh, ("DG", 1, (4,)))
    s = sig(uh)
    sh = fem.Function(W)
    sh.interpolate(fem.Expression(ufl.as_vector([s[0, 0], s[1, 1], s[2, 2], s[0, 1]]), W.element.interpolation_points))
    tree = geometry.bb_tree(msh, 2)

    def at(fun, pts):
        p3 = np.c_[pts, np.zeros(len(pts))]
        cand = geometry.compute_collisions_points(tree, p3)
        cells = geometry.compute_colliding_cells(msh, cand, p3)
        out = np.full((len(pts), fun.function_space.value_size if hasattr(fun.function_space, "value_size") else 4), np.nan)
        keep, cc = [], []
        for i in range(len(pts)):
            c = cells.links(i)
            if len(c):
                keep.append(i)
                cc.append(c[0])
        if keep:
            out[keep] = fun.eval(p3[keep], np.array(cc, np.int32))
        return out

    def principal(sv):
        rr, zz, tt, rz = sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]
        c, d = 0.5 * (rr + zz), np.sqrt((0.5 * (rr - zz)) ** 2 + rz ** 2)
        return np.max(np.c_[c + d, c - d, tt], axis=1)

    def mises(sv):
        rr, zz, tt, rz = sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]
        return np.sqrt(0.5 * ((rr - zz) ** 2 + (zz - tt) ** 2 + (tt - rr) ** 2) + 3 * rz ** 2)

    t_mid = case.get("throw", 0.8) * case["t_end"]
    zline = np.r_[np.linspace(0, H_UM - 30, 120), np.linspace(H_UM - 30, H_UM - 0.05, 160)]
    g_in = np.c_[np.full_like(zline, A_UM + 0.05), zline]                       # glass side of the interface
    w_mid = np.c_[A_UM - 0.5 * wall_t(case["t_end"], case.get("throw", 0.8), zline), zline]   # wall mid-thickness
    sg, sw = at(sh, g_in), at(sh, w_mid)
    res = {"case": case, "R": R, "n_cells": int(msh.topology.index_map(2).size_global),
           "n_dofs": int(V.dofmap.index_map.size_global * 2), "z": zline.tolist(),
           "glass_if": {"rr": sg[:, 0].tolist(), "zz": sg[:, 1].tolist(), "tt": sg[:, 2].tolist(), "rz": sg[:, 3].tolist(),
                        "p1": principal(sg).tolist()},
           "wall_mid": {"mises": mises(sw).tolist(), "zz": sw[:, 1].tolist(), "tt": sw[:, 2].tolist()}}
    # mid-plane radial profile (for the comparison with the analytic solution)
    rr_ = np.r_[np.linspace(0.05, A_UM - 0.05, 200), np.linspace(A_UM + 0.05, R - 0.05, 300)]
    if not core:
        rr_ = rr_[rr_ > A_UM - t_mid]
    smid = at(sh, np.c_[rr_, np.full_like(rr_, 0.5)])
    res["mid"] = {"r": rr_.tolist(), "rr": smid[:, 0].tolist(), "tt": smid[:, 2].tolist(), "zz": smid[:, 1].tolist()}
    # displacement of the top surface: Cu protrusion relative to the glass far away
    xs = np.r_[np.linspace(0.05, A_UM - 0.05, 60), [A_UM + 1, A_UM + 20, R - 1]]
    if not core:
        xs = xs[xs > A_UM - case["t_end"]]
    ut = at(uh, np.c_[xs, np.full_like(xs, H_UM - 1e-3)])
    res["top"] = {"r": xs.tolist(), "uz": ut[:, 1].tolist(), "ur": ut[:, 0].tolist()}
    # fixed points near the top (the bimaterial corner at the hole edge is singular in elasticity: values at a fixed
    # distance from it converge with the mesh, the value at the corner itself does not)
    dd = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    s_top = at(sh, np.c_[A_UM + dd, np.full_like(dd, H_UM - 0.02)])            # glass top surface, d from the hole edge
    s_dep = at(sh, np.c_[np.full_like(dd, A_UM + 0.05), H_UM - dd])           # glass at the interface, d below the top
    res["fixed"] = {"d": dd.tolist(), "top_p1": principal(s_top).tolist(), "top_rr": s_top[:, 0].tolist(),
                    "top_tt": s_top[:, 2].tolist(), "dep_p1": principal(s_dep).tolist(), "dep_rr": s_dep[:, 0].tolist(),
                    "dep_rz": s_dep[:, 3].tolist(), "dep_zz": s_dep[:, 1].tolist()}
    # the largest principal stress in the glass away from the singular top corner (> 2 um from it)
    Wg = fem.functionspace(msh, ("DG", 1))
    p1 = fem.Function(Wg)
    c_, d_ = 0.5 * (s[0, 0] + s[1, 1]), ufl.sqrt((0.5 * (s[0, 0] - s[1, 1])) ** 2 + s[0, 1] ** 2)
    p1.interpolate(fem.Expression(ufl.max_value(c_ + d_, s[2, 2]), Wg.element.interpolation_points))
    gcells = ct.find(3)
    xy = Wg.tabulate_dof_coordinates()[:, :2]
    dofs = np.unique(Wg.dofmap.list[gcells].ravel())
    far = np.hypot(xy[dofs, 0] - A_UM, xy[dofs, 1] - H_UM) > 2.0
    vals = p1.x.array[dofs]
    k = int(np.argmax(np.where(far, vals, -np.inf)))
    res["glass_p1_max"] = {"value": float(vals[k]), "r": float(xy[dofs[k], 0]), "z": float(xy[dofs[k], 1])}
    kc = int(np.argmax(vals))
    res["glass_p1_corner"] = {"value": float(vals[kc]), "r": float(xy[dofs[kc], 0]), "z": float(xy[dofs[kc], 1])}
    if fields:
        # a grid near the top of the via for maps
        gr, gz = np.meshgrid(np.linspace(0.1, 130, 261), np.linspace(H_UM - 130, H_UM - 0.05, 261))
        pts = np.c_[gr.ravel(), gz.ravel()]
        sv = at(sh, pts)
        res["map"] = {"r": gr[0].tolist(), "z": gz[:, 0].tolist(), "p1": principal(sv).reshape(gr.shape).tolist(),
                      "rr": sv[:, 0].reshape(gr.shape).tolist(), "mises": mises(sv).reshape(gr.shape).tolist()}
    if case.get("save_mesh"):
        np.savez_compressed(case["save_mesh"], x=msh.geometry.x[:, :2], cells=msh.geometry.dofmap[:, :3],
                            tags=np.array([ct.values[np.searchsorted(ct.indices, c)] if c in set(ct.indices) else 0
                                           for c in range(msh.topology.index_map(2).size_local)]))
    res["time_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    fields = len(sys.argv) > 3
    done = json.load(open(out_path)) if os.path.exists(out_path) else []
    keys = {d["case"]["name"] for d in done}
    for c in cases:
        if c["name"] in keys:
            continue
        r = solve(c, R=c.get("R", 500.0), fields=fields and c.get("map", False),
                  **{k: c[k] for k in ("h_wall", "h_corner") if k in c})
        done.append(r)
        json.dump(done, open(out_path, "w"))
        g = r["glass_if"]
        i_mid, i_top = 0, len(r["z"]) - 1
        print(f"{c['name']}: cells {r['n_cells']}, glass at the interface mid sr {g['rr'][i_mid]:.1f} st {g['tt'][i_mid]:.1f} MPa; "
              f"glass p1 max {r['glass_p1_max']['value']:.1f} MPa at z {r['glass_p1_max']['z']:.1f}; {r['time_s']:.1f} s", flush=True)
