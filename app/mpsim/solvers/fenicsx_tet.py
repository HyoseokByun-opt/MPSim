"""The body-fitted tetrahedral mesh of MPSim solved in FEniCSx (DOLFINx), as a check of solvers/tetfem.py.

Not imported by MPSim: FEniCSx runs on Linux (or WSL). MPSim writes its mesh with tetfem.write_fenics(mesh, path)
- made without periodic wrap, so that the problem needs no multi-point constraints - and this script solves the
same problem on it: plates at z = 0 and z = L (T = 0 and T = L), the other faces insulated, linear elements,
one conductivity per material. It prints the conductivity along z, the volume average of k dT/dz.

    python fenicsx_tet.py mesh.npz k0 k1 [k2 ...]          (k_i: the conductivity of material label i)

    python fenicsx_tet.py mesh.npz --elastic E0,nu0,alpha0 E1,nu1,alpha1 ...
                                                           (elasticity: the faces held at u = E.x, six strains and dT)

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


def solve_elastic(path, E, nu, alpha):
    """Linear elasticity on the same mesh (P1 vector elements, the faces held at u = E.x): the six unit strains
    and the thermal strain alpha (dT = 1, faces fixed). Returns (C 6 x 6 from the mean stresses, sigma_th 6), Voigt
    order xx yy zz yz xz xy with engineering shear - what solvers/tetelast.homogenize returns on that mesh."""
    import basix.ufl
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    d = np.load(path)
    pts, tets, mat, L = d["points"], d["tets"].astype(np.int64), d["mat"], d["L"]
    dom = mesh.create_mesh(MPI.COMM_WORLD, tets, basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)), pts)
    V = fem.functionspace(dom, ("Lagrange", 1, (3,)))
    Q = fem.functionspace(dom, ("DG", 0))
    orig = dom.topology.original_cell_index
    E, nu, alpha = np.asarray(E, float), np.asarray(nu, float), np.asarray(alpha, float)
    lam_v, mu_v = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    lam, mu, bt = fem.Function(Q), fem.Function(Q), fem.Function(Q)
    lam.x.array[:] = lam_v[mat[orig]]
    mu.x.array[:] = mu_v[mat[orig]]
    bt.x.array[:] = ((3 * lam_v + 2 * mu_v) * alpha)[mat[orig]]
    fdim = dom.topology.dim - 1
    dom.topology.create_connectivity(fdim, dom.topology.dim)
    dofs = fem.locate_dofs_topological(V, fdim, mesh.exterior_facet_indices(dom.topology))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(w):
        return lam * ufl.tr(eps(w)) * ufl.Identity(3) + 2 * mu * eps(w)
    a = ufl.inner(sig(u), eps(v)) * ufl.dx
    zero = fem.Constant(dom, np.zeros(3))
    opts = {"ksp_type": "preonly", "pc_type": "lu"}
    vol = float(np.prod(L))
    comp = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def run(Em, dT, tag):
        uD = fem.Function(V)
        uD.interpolate(lambda x: Em @ x)
        dTc = fem.Constant(dom, float(dT))
        rhs = ufl.inner(zero, v) * ufl.dx + dTc * bt * ufl.div(v) * ufl.dx
        try:
            prob = LinearProblem(a, rhs, bcs=[fem.dirichletbc(uD, dofs)], petsc_options=opts, petsc_options_prefix=tag)
        except TypeError:
            prob = LinearProblem(a, rhs, bcs=[fem.dirichletbc(uD, dofs)], petsc_options=opts)
        w = prob.solve()
        w = w[0] if isinstance(w, (tuple, list)) else w
        st = sig(w) - dTc * bt * ufl.Identity(3)
        return np.array([fem.assemble_scalar(fem.form(st[i, j] * ufl.dx)) / vol for i, j in comp])
    C = np.zeros((6, 6))
    for k, (i, j) in enumerate(comp):
        Em = np.zeros((3, 3))
        Em[i, j] = Em[j, i] = 1.0 if i == j else 0.5
        C[:, k] = run(Em, 0.0, f"mpsim_el{k}_")
    return C, run(np.zeros((3, 3)), 1.0, "mpsim_elth_")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--elastic":
        # python fenicsx_tet.py mesh.npz --elastic E0,nu0,alpha0 E1,nu1,alpha1 ...
        vals = np.array([[float(x) for x in a.split(",")] for a in sys.argv[3:]])
        C, sth = solve_elastic(sys.argv[1], vals[:, 0], vals[:, 1], vals[:, 2])
        np.set_printoptions(precision=12, linewidth=200)
        print("FEniCSx elasticity: C diag", np.diag(C), "C12 C13 C23", np.array([C[0, 1], C[0, 2], C[1, 2]]))
        print("FEniCSx elasticity: sigma_th", sth)
        sys.exit(0)
    kz, kx, ndof, ncell = solve(sys.argv[1], [float(x) for x in sys.argv[2:]])
    print(f"FEniCSx: k_zz {kz:.12g}  k_xz {kx:.6g}  ({ndof} dofs, {ncell} tetrahedra)")
