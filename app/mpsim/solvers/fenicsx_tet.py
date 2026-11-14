"""The body-fitted tetrahedral mesh of MPSim solved in FEniCSx (DOLFINx), as a check of solvers/tetfem.py.

Not imported by MPSim: FEniCSx runs on Linux (or WSL). MPSim writes its mesh with tetfem.write_fenics(mesh, path)
- made without periodic wrap, so that the problem needs no multi-point constraints - and this script solves the
same problem on it: plates at z = 0 and z = L (T = 0 and T = L), the other faces insulated, linear elements,
one conductivity per material. It prints the conductivity along z, the volume average of k dT/dz.

    python fenicsx_tet.py mesh.npz k0 k1 [k2 ...]          (k_i: the conductivity of material label i)

Both programs solve the same P1 Galerkin system on the same mesh, so the two values agree to the solvers'
tolerance (self-test: 1e-8 relative).
"""
import sys

import numpy as np


def solve(path, kvals, degree=1):
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    d = np.load(path)
    pts, tets, mat, L = d["points"], d["tets"].astype(np.int64), d["mat"], d["L"]
    dom = mesh.create_mesh(MPI.COMM_WORLD, tets, basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)), pts)
    V = fem.functionspace(dom, ("Lagrange", degree))
    Q = fem.functionspace(dom, ("DG", 0))
    k = fem.Function(Q)
    orig = dom.topology.original_cell_index
    k.x.array[:] = np.asarray(kvals, float)[mat[orig]]
    fdim = dom.topology.dim - 1
    bot = mesh.locate_entities_boundary(dom, fdim, lambda x: np.isclose(x[2], 0.0, atol=1e-9 * L[2]))
    top = mesh.locate_entities_boundary(dom, fdim, lambda x: np.isclose(x[2], L[2], atol=1e-9 * L[2]))
    bcs = [fem.dirichletbc(fem.Constant(dom, 0.0), fem.locate_dofs_topological(V, fdim, bot), V),
           fem.dirichletbc(fem.Constant(dom, float(L[2])), fem.locate_dofs_topological(V, fdim, top), V)]
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    rhs = fem.Constant(dom, 0.0) * v * ufl.dx
    opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-12, "ksp_atol": 0.0}
    try:
        prob = LinearProblem(a, rhs, bcs=bcs, petsc_options=opts, petsc_options_prefix="mpsim_tet_")
    except TypeError:
        prob = LinearProblem(a, rhs, bcs=bcs, petsc_options=opts)
    T = prob.solve()
    T = T[0] if isinstance(T, (tuple, list)) else T
    vol = float(np.prod(L))
    kzz = fem.assemble_scalar(fem.form(k * ufl.grad(T)[2] * ufl.dx)) / vol
    kxz = fem.assemble_scalar(fem.form(k * ufl.grad(T)[0] * ufl.dx)) / vol
    return float(kzz), float(kxz), int(V.dofmap.index_map.size_global), int(dom.topology.index_map(3).size_global)


if __name__ == "__main__":
    kz, kx, ndof, ncell = solve(sys.argv[1], [float(x) for x in sys.argv[2:]])
    print(f"FEniCSx: k_zz {kz:.12g}  k_xz {kx:.6g}  ({ndof} dofs, {ncell} tetrahedra)")
