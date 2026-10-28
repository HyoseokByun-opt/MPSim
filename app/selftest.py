"""Self-test: the engine against problems whose answers are known.

    python selftest.py            # full set (a few minutes)
    python selftest.py --quick    # the short set used by the installer

Each check states what the answer must be *before* running, so a pass means
the number is right, not just that the code ran. Checks of different kinds
are included on purpose - physics against closed forms, invariance under a
rigid translation (a coordinate or boundary fault cannot pass it), two
independent implementations of the same quantity, and the browser API.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:                                              # noqa: BLE001
    pass

import numpy as np

RESULTS = []
_TMP_LOGS = tempfile.mkdtemp(prefix="mpsim_selftest_")


def check(name, quick=True):
    def deco(fn):
        fn._name, fn._quick = name, quick
        return fn
    return deco


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ------------------------------------------------------------------ helpers
def sc_sphere(N, phi, shift):
    r = (phi * 3 / (4 * math.pi)) ** (1 / 3) * N
    c = N / 2 + (N / 2 if shift else 0)
    i = np.arange(N) + 0.5
    dx = (i - c + N / 2) % N - N / 2
    D2 = dx[:, None, None] ** 2 + dx[None, :, None] ** 2 + dx[None, None, :] ** 2
    return (D2 <= r * r).astype(np.uint8)


def channel(NX=48, NY=40, R=8.0):
    """Straight cylindrical pore (label 1) of radius R voxels along x."""
    y = np.arange(NY) + 0.5 - NY / 2
    Y, Z = np.meshgrid(y, y, indexing="ij")
    lab = np.zeros((NX, NY, NY), np.uint8)
    lab[:, (Y ** 2 + Z ** 2) <= R * R] = 1
    return lab


def _puma():
    from mpsim.solvers import puma_backend as PB
    PB.set_log_dir(_TMP_LOGS)
    return PB


# ------------------------------------------------------------------ checks
@check("NASA PuMA installation (conda-forge)")
def t_puma():
    PB = _puma()
    expect(PB.available(), f"PuMA could not be imported: {PB.unavailable_reason()}")
    return f"PuMA {PB.version()}"


@check("Conduction: exact series and parallel laminate (PuMA periodic FE and PuMA FV)")
def t_laminate():
    PB = _puma()
    n = 16
    lab = np.zeros((n, n, n), np.uint8)
    lab[:, :, n // 2:] = 1
    ser, par = 2.0 / (1.0 + 0.1), 5.5
    out = []
    for method in ("fe", "fv"):
        s = PB.conductivity(lab, [1.0, 10.0], "z", 1e-6, tol=1e-10, method=method)["k_eff"]
        p = PB.conductivity(lab, [1.0, 10.0], "x", 1e-6, tol=1e-10, method=method)["k_eff"]
        expect(abs(s / ser - 1) < 2e-3 and abs(p / par - 1) < 2e-3,
               f"{method}: series {s:.5f} (exact {ser:.5f}), parallel {p:.5f} (exact {par})")
        out.append(f"{method}: series {s:.4f} / parallel {p:.4f}")
    return "; ".join(out)


@check("Multi-core FE: index tables, product and MINRES reproduce PuMA's single-threaded solve")
def t_fe_parallel():
    import contextlib
    import io as _io
    PB = _puma()
    pm = PB.require()
    from pumapy.physics_models.finite_element.fe_conductivity import ConductivityFE
    from pumapy.physics_models.finite_element.fe_elasticity import ElasticityFE
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 3, size=(5, 7, 6)).astype(np.uint16)       # odd, unequal sides
    cmap = pm.AnisotropicConductivityMap()
    emap = PB.elasticity_map()
    for i in range(3):
        cmap.add_isotropic_material((i, i), 1.0 + i)
        emap.add_isotropic_material((i, i), [3e9, 70e9, 200e9][i], [0.35, 0.22, 0.3][i])
    with contextlib.redirect_stdout(_io.StringIO()):
        a = ConductivityFE(PB._workspace(lab, 1e-6), cmap, "x", 1e-6, 100, "minres", False, True)
        a.error_check()
        ConductivityFE.initialize(a)
        b = PB._fe_cond_class()(PB._workspace(lab, 1e-6), cmap, "x", 1e-6, 100, "minres", False, True)
        b.error_check()
        b.initialize()
        c = ElasticityFE(PB._workspace(lab, 1e-6), emap, "x", 1e-6, 100, "minres", False, True)
        c.error_check()
        ElasticityFE.initialize(c)
        d = PB._thermo_class()(PB._workspace(lab, 1e-6), emap, "x", {0: 6e-5, 1: 5e-6, 2: 1e-5}, 1e-6, 100,
                               "minres", False)
        d.error_check()
        d.initialize()
    expect(np.array_equal(a.pElemDOFNum, b.pElemDOFNum) and np.array_equal(a.elemMatMap, b.elemMatMap),
           "conduction DOF tables differ from PuMA's")
    expect(np.array_equal(c.pElemDOFNum, d.pElemDOFNum) and np.array_equal(c.elemMatMap, d.elemMatMap),
           "elasticity DOF tables differ from PuMA's")
    lab = (rng.random((20, 20, 20)) < 0.35).astype(np.uint8)
    res = {}
    # both paths driven to full convergence: PuMA's own stopping test is lenient
    # on larger grids (see t_fe_stop), so a default-tolerance comparison would
    # compare an unconverged answer with a converged one
    for on, tol in ((False, 1e-13), (True, 1e-10)):
        PB.set_fe_parallel(on)
        try:
            k = PB.conductivity(lab, [0.2, 30.0], "y", 1e-6, tol=tol, keep_fields=False, method="fe")["k_eff"]
            C = PB.elastic_load(lab, [3e9, 370e9], [0.35, 0.22], [6e-5, 7e-6], 1e-6, "xy", tol=tol)[0]
        finally:
            PB.set_fe_parallel(True)
        res[on] = (k, np.asarray(C, float))
    ek = abs(res[True][0] / res[False][0] - 1)
    eC = np.abs(res[True][1] - res[False][1]).max() / np.abs(res[False][1]).max()
    expect(ek < 1e-7 and eC < 1e-7, f"k {res[True][0]} vs {res[False][0]}, C column rel. diff {eC:.1e}")
    return f"tables identical; converged k and C(xy) differ by {ek:.0e} and {eC:.0e}"


@check("Conduction discretisations: FV and PuMA FE bracket a converged simple-cubic reference; FV the closer")
def t_fv_fe_reference():
    # One sphere per periodic cell, surface gap 5 % of the cell (45 vol%),
    # contrast 150. Reference 3.90: FE and FV on 192^3 and 256^3 grids,
    # extrapolated, agree to 0.1 % (guide 7.2).
    PB = _puma()
    from mpsim.solvers import conduction as CD
    ref, N, R = 3.90, 64, 0.475
    x = (np.arange(N) + 0.5) / N - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    lab = (X * X + Y * Y + Z * Z <= R * R).astype(np.uint8)
    fe = PB.conductivity(lab, [1.0, 150.0], "x", 1e-6, keep_fields=False, method="fe")["k_eff"]
    fv = CD.solve_direction(lab, np.array([1.0, 150.0]), 0, tol=1e-8)["k_eff"]
    efe, efv = fe / ref - 1, fv / ref - 1
    expect(abs(efv) < 0.02 and abs(efe) < 0.06, f"FE {fe:.4f} ({100*efe:+.1f} %), FV {fv:.4f} ({100*efv:+.1f} %)")
    expect(abs(efv) < abs(efe), f"FV {100*efv:+.1f} % is not closer than FE {100*efe:+.1f} %")
    return f"64³: FE {fe:.3f} ({100*efe:+.1f} %), FV {fv:.3f} ({100*efv:+.1f} %) against {ref}"


@check("PuMA FE stopping test: the same k on a coarse and a fine grid of one structure")
def t_fe_stop():
    # PuMA's MINRES test ||r|| / (||A|| ||x||) loosened with the grid: on a
    # 50 vol% structure it stopped at +32 % on 192^3. The relative residual
    # does not; here a 2x finer rasterisation of one sphere must give k within
    # the discretisation change, not a jump.
    PB = _puma()
    out = []
    for N in (32, 64):
        x = (np.arange(N) + 0.5) / N - 0.5
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        lab = (X * X + Y * Y + Z * Z <= 0.475 ** 2).astype(np.uint8)
        k = PB.conductivity(lab, [1.0, 150.0], "x", 1e-6, keep_fields=False, method="fe")["k_eff"]
        kt = PB.conductivity(lab, [1.0, 150.0], "x", 1e-6, tol=1e-10, keep_fields=False, method="fe")["k_eff"]
        expect(abs(k / kt - 1) < 1e-4, f"{N}³: default tolerance {k:.5f} vs converged {kt:.5f}")
        out.append(f"{N}³ {k:.4f} (converged {kt:.4f})")
    return "; ".join(out)


@check("Conduction: invariance under a rigid translation of the periodic RVE by half a cell")
def t_translation():
    PB = _puma()
    from mpsim.solvers import conduction as CD
    vals = [1.0, 150.0]
    a = PB.conductivity(sc_sphere(24, 0.3, False), vals, "x", 1e-6, tol=1e-9, method="fe")["k_eff"]
    b = PB.conductivity(sc_sphere(24, 0.3, True), vals, "x", 1e-6, tol=1e-9, method="fe")["k_eff"]
    c = CD.solve_direction(sc_sphere(24, 0.3, False), vals, 0, tol=1e-9)["k_eff"]
    d = CD.solve_direction(sc_sphere(24, 0.3, True), vals, 0, tol=1e-9)["k_eff"]
    expect(abs(a / b - 1) < 1e-4, f"PuMA FE: {a:.5f} vs shifted {b:.5f}")
    expect(abs(c / d - 1) < 1e-6, f"built-in FV: {c:.5f} vs shifted {d:.5f}")
    return f"PuMA FE {a:.4f} = {b:.4f}, built-in FV {c:.4f} = {d:.4f}"


@check("Conduction: simple cubic sphere array against Rayleigh's solution (±8 %)")
def t_rayleigh():
    PB = _puma()
    kap, phi = 150.0, 0.3
    bb = (kap + 2) / (kap - 1)
    ray = 1 + 3 * phi / (bb - phi - 0.525 * ((kap - 1) / (kap + 4 / 3)) * phi ** (10 / 3))
    k = PB.conductivity(sc_sphere(32, phi, False), [1.0, kap], "x", 1e-6, tol=1e-9, method="fe")["k_eff"]
    expect(abs(k / ray - 1) < 0.08, f"PuMA FE {k:.4f}, Rayleigh {ray:.4f}")
    return f"PuMA FE {k:.4f} vs Rayleigh {ray:.4f} ({100*(k/ray-1):+.1f} %, 32³ voxelisation)"


@check("Electrical conduction: convergence for absolute values of σ ≈ 1e-13 (scale invariance)")
def t_scale():
    PB = _puma()
    rng = np.random.default_rng(1)
    lab = (rng.random((20, 20, 20)) < 0.3).astype(np.uint8)
    a = PB.conductivity(lab, [1e-13, 1e-12], "x", 1e-6, tol=1e-8, method="fv")["k_eff"]
    b = PB.conductivity(lab, [1.0, 10.0], "x", 1e-6, tol=1e-8, method="fv")["k_eff"] * 1e-13
    expect(abs(a / b - 1) < 1e-4, f"{a:.6e} vs {b:.6e}")
    return f"{a:.5e} = {b:.5e}"


@check("Thermal expansion: PuMA FE extension (thermal load) reproduces the unmodified PuMA stiffness")
def t_elastic_identity():
    import contextlib
    import io
    import warnings
    PB = _puma()
    pm = PB.require()
    rng = np.random.default_rng(0)
    lab = (rng.random((10, 10, 10)) < 0.3).astype(np.uint8)
    E, nu, al = [3.0, 70.0], [0.35, 0.2], [60.0, 5.0]
    C0 = np.zeros((6, 6))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, d in enumerate(["x", "y", "z", "yz", "xz", "xy"]):
            ws = pm.Workspace.from_array(lab.astype(np.uint16))
            ws.voxel_length = 1e-6
            em = PB.elasticity_map()
            # the materials as thermo_elastic enters them (mu on the shear
            # diagonal; PuMA's add_isotropic_material doubles it, see
            # puma_backend._iso_material)
            PB._iso_material(em, 0, E[0], nu[0])
            PB._iso_material(em, 1, E[1], nu[1])
            C0[:, k], _, _, _ = pm.experimental.compute_elasticity(ws, em, d, solver_type="minres", tolerance=1e-9,
                                                                   method="fe", display_iter=False)
    # thermo_elastic maps PuMA's mirrored shear axes onto ours
    P, sg = [0, 1, 2, 5, 4, 3], np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    C0 = (sg[:, None] * sg[None, :]) * C0[np.ix_(P, P)]
    r = PB.thermo_elastic(lab, E, nu, al, 1e-6, tol=1e-9)
    err = np.abs(r["C"] - 0.5 * (C0 + C0.T)).max() / np.abs(C0).max()
    expect(err < 1e-8, f"max rel diff {err:.2e}")
    return f"largest relative difference {err:.1e}"


@check("Elasticity: a homogeneous RVE returns the exact isotropic stiffness, shear included")
def t_elastic_homogeneous():
    PB = _puma()
    from mpsim.solvers import elastic as EL
    rng = np.random.default_rng(4)
    lab = (rng.random((10, 10, 10)) < 0.4).astype(np.uint8)
    E, nu = 3.2, 0.35
    r = PB.thermo_elastic(lab, [E, E], [nu, nu], [55.0, 55.0], 1e-6, tol=1e-10)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    exact = np.zeros((6, 6))
    exact[:3, :3] = lam
    exact[np.arange(3), np.arange(3)] = lam + 2 * mu
    exact[np.arange(3, 6), np.arange(3, 6)] = mu
    err = np.abs(r["C"] - exact).max() / (lam + 2 * mu)
    # This caught shear entries twice too large: PuMA works in Mandel notation,
    # and nothing checked the shear block until now.
    expect(err < 1e-6, f"C44 {r['C'][3, 3]:.4f} vs {mu:.4f}, C11 {r['C'][0, 0]:.4f} vs {lam + 2 * mu:.4f}")
    G = EL.engineering_constants(r["C"])["G"][0]
    expect(abs(G / mu - 1) < 1e-6, f"G {G:.4f} vs {mu:.4f}")
    return f"C44 {r['C'][3, 3]:.4f} = E/2(1+ν) {mu:.4f} · relative error {err:.1e}"


@check("Thermal stress, free expansion: zero stress in a homogeneous RVE, zero mean stress in a composite")
def t_thermal_free():
    PB = _puma()
    n = 12
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) + 0.5
    sphere = (np.linalg.norm(g - n / 2, axis=-1) < 3.8).astype(np.uint8)
    r = PB.thermo_elastic(sphere, [3.2, 3.2], [0.35, 0.35], [55.0, 55.0], 1e-6, tol=1e-10, keep_field="free")
    peak = max(np.abs(np.asarray(f)).max() for f in r["fields"])
    ref = 3.2 * 55.0 / (1 - 2 * 0.35)        # the constrained stress it must cancel
    expect(peak < 1e-6 * ref, f"homogeneous free expansion left {peak:.3g} (scale {ref:.1f})")
    r = PB.thermo_elastic(sphere, [3.2, 72.0], [0.35, 0.17], [55.0, 0.5], 1e-6, tol=1e-10, keep_field="free")
    s, t = (np.asarray(f).reshape(-1, 3) for f in r["fields"])
    mean = max(np.abs(s.mean(0)).max(), np.abs(t.mean(0)).max())
    expect(mean < 1e-6 * np.abs(s).max(), f"mean stress {mean:.3g} of a free composite")
    return f"homogeneous peak {peak:.1e} · composite mean {mean:.1e} (peak {np.abs(s).max():.0f})"


@check("Thermal expansion: Levin theorem (exact two-phase relation) between thermal and mechanical loads")
def t_levin():
    PB = _puma()
    from mpsim.solvers import elastic as EL
    rng = np.random.default_rng(2)
    lab = (rng.random((16, 16, 16)) < 0.3).astype(np.uint8)
    E, nu, al = [3.0, 380.0], [0.35, 0.22], [60.0, 7.0]
    r = PB.thermo_elastic(lab, E, nu, al, 1e-6, tol=1e-9)
    K = [E[i] / (3 * (1 - 2 * nu[i])) for i in range(2)]
    lev = EL.levin_alpha(np.linalg.inv(r["C"]), al[0], al[1], K[0], K[1])
    err = np.max(np.abs(lev[:3] - r["alpha"][:3])) / np.max(np.abs(r["alpha"][:3]))
    expect(err < 1e-4, f"alpha {r['alpha'][:3]} vs Levin {lev[:3]}")
    c = EL.engineering_constants(r["C"])
    expect(abs(EL.directional_modulus(c["S"], [1, 0, 0]) / c["E"][0] - 1) < 1e-10, "directional modulus along x")
    return f"α {np.round(r['alpha'][:3], 3)} · relative error {err:.1e} · A_U {c['anisotropy_index']:.3f}"


@check("Thermal expansion: single solid with pores gives α of the solid")
def t_porous_cte():
    PB = _puma()
    rng = np.random.default_rng(3)
    lab = (rng.random((12, 12, 12)) < 0.3).astype(np.uint8)
    r = PB.thermo_elastic(lab, [3.0, 0.0], [0.35, 0.0], [60.0, 0.0], 1e-6, tol=1e-9)
    err = np.max(np.abs(r["alpha"][:3] - 60.0)) / 60.0
    expect(err < 1e-3, f"alpha {r['alpha'][:3]}")
    return f"α {np.round(r['alpha'][:3], 4)}"


def _sphere_lab(n, r, c=None):
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) + 0.5
    c = np.full(3, n / 2) if c is None else np.asarray(c, float)
    return (np.linalg.norm(g - c, axis=-1) < r).astype(np.int32)


@check("Elasticity FANS (v5 default): a homogeneous RVE and a laminate exactly, free expansion stress-free")
def t_fans_exact():
    from mpsim.solvers import fans as FA
    from mpsim.solvers import elastic as EL
    E, nu, al = [3.2, 370.0], [0.35, 0.22], [60.0, 7.0]
    r = FA.homogenize(np.zeros((8, 8, 8), np.int32), E, nu, al, tol=1e-10)
    lam, mu = EL.lame(3.2, 0.35)
    ex = FA.iso_C(lam, mu)
    e1 = float(np.abs(r["C"] - ex).max() / ex.max())
    expect(e1 < 1e-10 and abs(r["alpha"][0] - 60.0) < 1e-8, f"homogeneous C error {e1:.1e}")
    # a laminate along z: the Backus average is exact for voxel-aligned layers
    lab = np.zeros((6, 6, 12), np.int32)
    lab[:, :, 6:] = 1
    r = FA.homogenize(lab, E, nu, al, tol=1e-12)
    lam_p, mu_p = EL.lame(np.array(E), np.array(nu))
    C33 = 1.0 / np.mean(1.0 / (lam_p + 2 * mu_p))
    G44 = 1.0 / np.mean(1.0 / mu_p)
    e2 = max(abs(r["C"][2, 2] / C33 - 1), abs(r["C"][3, 3] / G44 - 1), abs(r["C"][4, 4] / G44 - 1))
    expect(e2 < 1e-9, f"laminate C33 {r['C'][2, 2]:.5f} vs {C33:.5f}, C44 {r['C'][3, 3]:.5f} vs {G44:.5f}")
    sph = _sphere_lab(12, 3.8)
    rf = FA.homogenize(sph, [3.2, 72.0], [0.35, 0.17], [55.0, 0.5], tol=1e-10, keep_thermal_stress=True,
                       stress_bc="free")
    s = rf["thermal_stress"].reshape(6, -1)
    mean = float(np.abs(s.mean(1)).max())
    expect(mean < 1e-6 * np.abs(s).max(), f"mean stress {mean:.3g} of a free composite")
    return f"homogeneous {e1:.0e} · laminate C33, C44 {e2:.0e} · free expansion mean stress {mean:.0e}"


@check("Elasticity FANS = PuMA's voxel FE (same elements, shear modulus as entered, shear axes mapped)")
def t_fans_puma():
    PB = _puma()
    from mpsim.solvers import fans as FA
    rng = np.random.default_rng(5)
    lab = np.zeros((20, 20, 20), np.int32)
    for c in rng.random((14, 3)) * 20:
        g = np.indices(lab.shape).transpose(1, 2, 3, 0) + 0.5
        d = (g - c + 10) % 20 - 10
        lab[np.linalg.norm(d, axis=-1) < 3.2] = 1
    E, nu, al = [3.2, 370.0], [0.35, 0.22], [60.0, 7.0]
    rf = FA.homogenize(lab, E, nu, al, tol=1e-10)
    rp = PB.thermo_elastic(lab, E, nu, al, 1e-6, tol=1e-10)
    eC = float(np.abs(rf["C"] - rp["C"]).max() / np.abs(rp["C"]).max())
    ea = float(np.abs(rf["alpha"] - rp["alpha"]).max() / np.abs(rp["alpha"]).max())
    expect(eC < 1e-6 and ea < 1e-5, f"C differs by {eC:.1e}, alpha by {ea:.1e}")
    return f"C {eC:.0e}, α {ea:.0e} apart · G_yz {rf['C'][3, 3]:.4f}, G_xy {rf['C'][5, 5]:.4f}"


@check("Elasticity FANS: dilute sphere bulk modulus within 1 % of the Hashin-Shtrikman lower bound (shear entered once)")
def t_fans_dilute():
    from mpsim.solvers import fans as FA
    from mpsim.solvers import elastic as EL
    lab = _sphere_lab(40, 8.0)
    f = float(lab.mean())
    E, nu = [3.2, 370.0], [0.35, 0.22]
    r = FA.homogenize(lab, E, nu, [0.0, 0.0], tol=1e-9, loads=("xx", "yy", "zz"))
    C = r["C"]
    K = (C[:3, :3].sum()) / 9.0
    lam, mu = EL.lame(np.array(E), np.array(nu))
    Km, Kp, Gm = lam[0] + 2 * mu[0] / 3, lam[1] + 2 * mu[1] / 3, mu[0]
    hs = Km + f / (1.0 / (Kp - Km) + 3 * (1 - f) / (3 * Km + 4 * Gm))
    # with the matrix shear modulus doubled (PuMA's isotropic material) the
    # same RVE gives HS(2 Gm): the check sits between the two
    hs2 = Km + f / (1.0 / (Kp - Km) + 3 * (1 - f) / (3 * Km + 8 * Gm))
    err = K / hs - 1
    # 1 %: the voxelised sphere sits +0.4 % above; a doubled shear modulus
    # moves the result to about +1.8 %
    expect(abs(err) < 0.01, f"K {K:.4f} vs HS {hs:.4f} (doubled shear would give {hs2:.4f})")
    return f"φ {f:.4f}: K {K:.4f}, HS lower bound {hs:.4f} ({100 * err:+.2f} %); doubled shear {hs2:.4f}"


@check("Elasticity FANS: the iteration count does not grow with the grid (same spheres on 24³ and 48³)")
def t_fans_scaling():
    from mpsim.solvers import fans as FA
    its = []
    for n, r in ((24, 4.5), (48, 9.0)):
        lab = np.zeros((n, n, n), np.int32)
        for c in np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75]]) * n:
            lab[_sphere_lab(n, r, c) == 1] = 1
        sol = FA.FANS(lab, [3.2, 370.0], [0.35, 0.22], [0.0, 0.0])
        _, st, _ = sol.solve([1, 0, 0, 0, 0, 0], 0.0, 1e-6)
        its.append(st["iterations"])
    expect(its[1] <= 1.25 * its[0] + 2, f"iterations {its}")
    return f"iterations {its[0]} on 24³, {its[1]} on 48³"


@check("Viscosity: a dilute rigid sphere gives Einstein's 2.5 and moves with the resin (Jeffery spin), a free bubble -5/3")
def t_visc_dilute():
    from mpsim.solvers import viscosity as VI
    lab = _sphere_lab(32, 6.0)
    phi = float(lab.mean())
    r = VI.effective_viscosity(lab, [1.0, VI.RIGID_RATIO], tol=1e-7, keep_field=False, extension=False)
    r4 = VI.effective_viscosity(lab, [1.0, 10 * VI.RIGID_RATIO], tol=1e-7, keep_field=False, extension=False)
    b = VI.effective_viscosity(lab, [1.0, VI.BUBBLE_RATIO], tol=1e-7, keep_field=False, extension=False)
    eta, eta4, eta_b = (r["mu_r"] - 1) / phi, (r4["mu_r"] - 1) / phi, (b["mu_r"] - 1) / phi
    # a voxelised sphere and the images of the periodic array add a few
    # per cent at this fraction; a doubled or missing shear term would be
    # off by tens of per cent
    expect(abs(eta / 2.5 - 1) < 0.15, f"rigid [eta] {eta:.3f} (Einstein 2.5)")
    # a bubble with no surface tension (capillary number -> infinity) is an
    # incompressible inclusion without shear stiffness: [eta] = -5/3 (Eshelby);
    # Taylor's +1 needs surface tension to keep it spherical
    expect(abs(eta_b / (-5.0 / 3.0) - 1) < 0.15, f"bubble [eta] {eta_b:.3f} (free bubble -5/3)")
    expect(abs(r4["mu_r"] / r["mu_r"] - 1) < 0.01, f"contrast 1e3 {r['mu_r']:.5f} vs 1e4 {r4['mu_r']:.5f}")
    expect(r["mu_r"] >= 0.995 * VI.hs_lower(phi), f"mu_r {r['mu_r']:.5f} below HS {VI.hs_lower(phi):.5f}")
    # the filler is not held: a fluid 1000x the resin's viscosity with no
    # force or torque on it. Off the centre line of simple shear it must
    # translate with the resin at its centre and turn at half the shear rate
    # (Jeffery 1922); a fixed filler would not move at all
    n, c = 40, np.array([20.0, 30.0, 20.0])
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) + 0.5
    ins = ((g - c) ** 2).sum(-1) <= 25.0
    m = VI.effective_viscosity(ins.astype(np.int32), [1.0, VI.RIGID_RATIO], tol=1e-7, keep_field=True,
                               extension=False)
    p, v = g[ins] - c, m["velocity"][ins]
    a = np.zeros((3 * len(p), 6))
    for k in range(3):
        a[k::3, k] = 1.0
    a[0::3, 4], a[0::3, 5] = p[:, 2], -p[:, 1]
    a[1::3, 3], a[1::3, 5] = -p[:, 2], p[:, 0]
    a[2::3, 3], a[2::3, 4] = p[:, 1], -p[:, 0]
    rb, *_ = np.linalg.lstsq(a, v.reshape(-1), rcond=None)
    u_res = c[1] - n / 2.0                      # the resin's velocity at the centre (unit shear rate)
    expect(abs(rb[0] / u_res - 1) < 0.01, f"filler moves {rb[0]:.3f}, the resin there {u_res:.3f}")
    expect(abs(rb[5] + 0.5) < 0.01, f"filler spins {rb[5]:.3f} (Jeffery -0.5)")
    return (f"φ {phi:.4f}: rigid [η] {eta:.3f} (contrast ×10: {eta4:.3f}), bubble [η] {eta_b:.3f}; "
            f"a filler off the centre moves {rb[0]:.3f} with the resin's {u_res:.3f} and spins {rb[5]:.3f}")


@check("Particle dynamics: frictionless spheres at 50 vol% follow Krieger-Dougherty (phi_m 0.64); GPU = CPU where an NVIDIA GPU is present", quick=False)
def t_dem_kd():
    # every filler moving and colliding in the sheared resin; without
    # friction the jamming point is the random close packing, so the
    # viscosity follows Krieger-Dougherty with phi_m 0.64 (11.4 at 50 vol%;
    # 400 spheres to strain 4 gave 11.65 +- 0.33). Here 200 spheres to strain 2.
    from mpsim.solvers import suspension as SU
    n = 200
    na = int(round(n / (1 + 1.4 ** 3)))
    r = np.concatenate([np.full(na, 1.0), np.full(n - na, 1 / 1.4)])
    s = SU.Suspension(r, 0.50, seed=3, backend="cpu")
    s.pack()
    out = s.run(2.0, every=0.05)
    st, e = np.asarray(out["strain"]), np.asarray(out["eta"])
    eta = float(e[st >= 0.5].mean())
    kd = (1 - 0.5 / 0.64) ** (-2.5 * 0.64)
    expect(abs(eta / kd - 1) < 0.25, f"eta {eta:.2f} against Krieger-Dougherty {kd:.2f}")
    msg = f"μr {eta:.2f} against Krieger-Dougherty {kd:.2f} ({out['steps']} steps, CPU {out['seconds']:.0f} s)"
    # the GPU path runs the same kernels: a short shear from the same start
    # must agree (chaotic dynamics separate the two only slowly)
    import subprocess, sys as _s, json as _j
    code = ("import sys,json,numpy as np;sys.path.insert(0,%r);from mpsim.solvers import suspension as SU;"
            "r=np.concatenate([np.full(%d,1.0),np.full(%d,1/1.4)]);out={}\n"
            "for b in ('cuda','cpu'):\n"
            "  SU._TI['arch']=None\n  import taichi as ti\n  ti.reset()\n"
            "  try:\n    s=SU.Suspension(r,0.5,seed=3,backend=b)\n  except Exception as ex:\n    out[b]=str(ex);continue\n"
            "  if b=='cuda' and s.backend!='cuda':\n    out[b]='no CUDA';continue\n"
            "  s.pack();o=s.run(0.05,every=0.01);out[b]=o['eta'][-1]\n"
            "print(json.dumps(out))") % (os.path.dirname(os.path.abspath(__file__)), na, n - na)
    try:
        p = subprocess.run([_s.executable, "-c", code], capture_output=True, text=True, timeout=600)
        got = _j.loads(p.stdout.strip().splitlines()[-1])
    except Exception as ex:                                            # noqa: BLE001
        return msg + f"; GPU comparison not run ({ex})"
    if isinstance(got.get("cuda"), float):
        rel = abs(got["cuda"] / got["cpu"] - 1)
        expect(rel < 1e-3, f"GPU {got['cuda']:.6g} against CPU {got['cpu']:.6g}")
        return msg + f"; GPU (CUDA) agrees with the CPU to {rel:.1e}"
    return msg + "; no NVIDIA GPU here - the GPU comparison was skipped"


@check("Particle dynamics of other shapes: one-sphere rigid particles reproduce the spheres; for cubes the "
       "dissipated power equals the stress work (rigid multi-sphere particles)", quick=False)
def t_dem_rigid():
    # (1) the rigid-particle kernels with every particle one sphere at its
    # centre are the sphere kernels: the same start sheared the same way
    # gives the same viscosity (to the CG tolerance). (2) Cubes as rigid
    # clumps of spheres, lubrication only: the power the particles dissipate
    # (pair films and drag) must equal the shear stress times the rate -
    # this holds only if the forces, the torques about the particle centres
    # and the stress from the centre-to-centre vectors are consistent.
    from mpsim.solvers import suspension as SU
    from mpsim import shapes as SH
    SU.init_backend("cpu")
    rng = np.random.default_rng(3)
    r = np.clip(rng.lognormal(0, 0.15, 120), 0.6, 1.5)
    r /= r.max()
    s1 = SU.Suspension(r, 0.45, seed=4, backend="cpu", mu_f=0.5)
    s1.pack()
    x0 = s1.K["x"].to_numpy()
    e1 = np.asarray(s1.run(0.2, every=0.02)["eta"])
    SU.release()
    n = len(r)
    s2 = SU.RigidSuspension(np.zeros((n, 3)), r, np.arange(n), x0, np.tile(np.eye(3), (n, 1, 1)), r, s1.L, 0.45,
                            backend="cpu", mu_f=0.5)
    s2.pack()
    e2 = np.asarray(s2.run(0.2, every=0.02)["eta"])
    SU.release()
    m = min(len(e1), len(e2))
    d12 = float(np.max(np.abs(e1[:m] - e2[:m]) / e1[:m]))
    expect(d12 < 1e-3, f"one-sphere rigid particles differ from spheres by {d12:.1e}")
    q, rq = SH.clump_template(4, np.array([1.0, 1.0, 1.0, 0, 0, 0]), *SH.EMPTY_PLANES, max_spheres=40, coverage=0.98)
    nb = 60
    bodies = {"templates": [(q * 1e-6, rq * 1e-6)], "tpl": np.zeros(nb, int), "scale": np.ones(nb),
              "rot": SH.sample_rotations(nb, "iso", 0, rng), "rotmode": np.zeros(nb, int),
              "circ": np.full(nb, math.sqrt(3) * 1e-6), "vol": np.full(nb, 8e-18), "area": np.full(nb, 24e-12),
              "phase": np.zeros(nb, int)}
    # packed with a 20 nm layer that is then taken off: no contact, films only
    rig = SU.pack_clumps(bodies, 0.40, bound_m=2e-8, seed=2)
    s3 = SU.RigidSuspension(rig["mq"], rig["mr"] - 2e-8 / rig["a0"], rig["mb"], rig["X"], rig["Rot"], rig["a_eq"],
                            rig["L"], 0.40, intrinsic=3.1, backend="cpu", mu_f=0.0)
    s3.neighbours()
    s3.K["geometry"](1e-3)
    s3.solve(tol=1e-10, maxit=20000)
    s3.K["stress"]()
    sig = s3.K["sig"].to_numpy()
    diss = s3.K["dissipation"]()
    SU.release()
    gap = abs(sig[0] - diss) / diss
    expect(sig[2] == 0, f"{int(sig[2])} contacts in a contact-free start")
    expect(gap < 5e-3, f"stress work {sig[0]:.5g} against dissipation {diss:.5g} ({gap:.1e})")
    return (f"one-sphere particles = spheres to {d12:.1e}; 60 cubes of {len(rq)} spheres: dissipation = stress work "
            f"to {gap:.1e} (the films' gaps the only difference)")


@check("Viscosity closed forms: Krieger-Dougherty and Maron-Pierce limits, intrinsic viscosity of rods and discs; the particle-dynamics flow curve")
def t_visc_forms():
    from mpsim.solvers import viscosity as VI
    slope = (float(VI.krieger_dougherty(1e-4, 0.64, 2.5)) - 1.0) / 1e-4
    expect(abs(slope - 2.5) < 1e-3, f"KD slope at φ → 0 is {slope:.5f}, not Einstein's 2.5")
    expect(abs(float(VI.krieger_dougherty(0.63, 0.64, 2.5)) - 64.0 ** 1.6) < 1e-6 * 64.0 ** 1.6, "KD at 0.63 of 0.64 is not 64^1.6")
    expect(abs(float(VI.maron_pierce(0.32, 0.64)) - 4.0) < 1e-9, "Maron-Pierce at φm/2 is not 4")
    phi_m = VI.kd_phi_m_from(float(VI.krieger_dougherty(0.4, 0.61, 2.5)), 0.4, 2.5)
    expect(abs(phi_m - 0.61) < 1e-6, f"KD inversion {phi_m}")
    rod, disc = VI.intrinsic_viscosity("cylinder", 20.0), VI.intrinsic_viscosity("cylinder", 0.05)
    expect(VI.intrinsic_viscosity("sphere", 1.0) == 2.5, "sphere")
    expect(rod is not None and rod > 5 and disc is not None and disc > 5, f"rod {rod}, disc {disc}")
    # the flow curve from the particle dynamics at several rates: mu_r
    # log-log between its points, flat beyond; for a Newtonian resin the
    # compound is mu_r(gd) times the resin at every rate
    from mpsim import pipeline as PL
    cv = {"gd": [0.1, 10.0, 1000.0], "mu_r": [400.0, 40.0, 20.0], "se": [0, 0, 0]}
    m = PL._mu_r_at(cv, [0.01, 1.0, 100.0, 1e5], 7.0)
    expect(np.allclose(m, [400.0, 126.4911, 28.2843, 20.0], rtol=1e-4), f"interpolated {m}")
    expect(PL._mu_r_at(None, [1.0], 7.0) == 7.0, "no curve: the single value")
    mu_s, _ = VI.suspension_flow({"type": "newtonian", "mu": 2.0}, m, 0.5, [0.01, 1.0, 100.0, 1e5])
    expect(np.allclose(mu_s, 2.0 * m, rtol=1e-9), f"compound {mu_s} for mu_r {m}")
    return (f"KD(0.63) {float(VI.krieger_dougherty(0.63, 0.64, 2.5)):.0f}, rod (20) [η] {rod:.2f}, disc (1/20) [η] {disc:.2f}; "
            f"flow curve μr {m[1]:.1f} at 1 1/s between 400 and 40")


@check("Live view: a conduction solve writes its converging section, the last frame marked done")
def t_live_frames():
    from mpsim import live as LV
    from mpsim.solvers import conduction as CD
    lab = _sphere_lab(24, 6.0)
    with tempfile.TemporaryDirectory() as d:
        frames = []

        def cb(it, res):
            pass
        slot = LV.Slot(d, "thermal_x", "Thermal conductivity x")
        inner = LV.cond_snapshot(slot, 0, False, "Temperature")

        def snap(u, it, res, final=False):
            inner(u, it, res, final)
            frames.append(slot.seq)
        cb.snapshot = snap
        slot.every = 0.0                        # every call writes, for the test
        r = CD.solve_direction(lab, [1.0, 20.0], 0, tol=1e-7, callback=cb)
        idx = LV.index(d)
        expect(len(idx) == 1 and idx[0]["state"] == "done", f"slots {idx}")
        h = idx[0]
        img = np.fromfile(os.path.join(d, "thermal_x.f32"), np.float32).reshape(h["shape"])
        # the applied gradient is in the section: the temperature rises along x
        expect(img[:, -1].mean() - img[:, 0].mean() > 0.5 * img.shape[1], f"no gradient along x: {img.mean(axis=0)}")
    return (f"{len(frames)} frames over {r['iterations']} iterations, last {h['shape'][0]}x{h['shape'][1]} "
            f"section marked done, k {r['k_eff']:.4g}")


@check("Contact faces per pair of fillers: counted across faces and the periodic boundary")
def t_contact_pairs():
    from mpsim import pipeline
    g = np.zeros((4, 4, 4), np.int32)
    g[0:2] = 1
    g[2:4] = 2
    g[:, :, 3] = 0
    cm = np.zeros((4, 4, 4), np.uint8)
    cm[1, :, :3] |= 1          # x-faces between x = 1 (phase 0) and x = 2 (phase 1)
    cm[3, :, :3] |= 1          # x = 3 (phase 1) and x = 0 (phase 0) across the boundary
    cm[0, 0, 0] |= 2           # a y-face inside phase 0
    pf = pipeline.contact_pair_counts(g, cm, 2)
    expect(pf == {"0-1": 24, "0-0": 1}, f"pairs {pf}")
    spec = {"phases": [{"r_contact": 1e-8}, {"r_contact": 3e-8}], "contact_rc": {"0-1": 5e-8}}
    expect(pipeline.contact_resistance(spec, 0, 1) == 5e-8, "set pair value not used")
    spec.pop("contact_rc")
    expect(abs(pipeline.contact_resistance(spec, 1, 0) - 2e-8) < 1e-20, "unset pair is not the mean")
    return f"pairs {pf}; set and default pair resistances"


@check("Monte-Carlo compression: ends overlap-free and never above full size", quick=False)
def t_mc_compress():
    from scipy.spatial import cKDTree
    from mpsim import generate as GEN
    rng = np.random.default_rng(1)
    n = 200
    d = np.full(n, 5.0)
    L = (n * np.pi / 6 * 125 / 0.70) ** (1 / 3)
    pos = rng.random((n, 3)) * L
    p2, s2 = GEN.mc_compress(pos, d, L, 0.05, rng, max_iter=200)
    box = np.full(3, L)
    t = cKDTree(np.mod(p2, box), boxsize=box)
    c = t.query_pairs(r=float(d.max()) * s2 * 1.5, output_type="ndarray")
    dv = p2[c[:, 1]] - p2[c[:, 0]]
    dv -= box * np.round(dv / box)
    ratio = float(np.min(np.linalg.norm(dv, axis=1) / (s2 * d[c[:, 0]])))
    expect(s2 <= 1.0 + 1e-12, f"scale {s2}")
    expect(ratio >= 1 - 1e-6, f"overlap: closest pair at {ratio:.4f} of the contact distance")
    return f"φ {0.70 * s2 ** 3:.3f} of the 0.70 target, closest pair {ratio:.6f} of contact"


@check("Maximum packing fraction: equal spheres jam at the random close packing 0.64", quick=False)
def t_jamming():
    from mpsim import generate as GEN
    ph = [dict(name="s", shape="sphere", size_um={"d": 5.0}, dist={"type": "mono"}, vf=0.3,
               orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)]
    phi, info = GEN.jamming_fraction(ph)
    expect(abs(phi - 0.64) < 0.015, f"φm {phi:.4f}")
    return f"φm {phi:.4f} (raw {info['phi_m_raw']:.4f}, {info['n_particles']} spheres, {info['seconds']:.0f} s)"


@check("Calibration theory: mismatch models for identical materials (full transmission, DMM twice AMM) and detailed balance across a mismatch")
def t_cal_kapitza():
    from mpsim import calibrate as CA
    p = {"E": 70.0, "nu": 0.22, "rho": 2.2, "cp": 740.0}
    r = CA.kapitza(p, p)
    vl, vt = CA.sound_speeds(70.0, 0.22, 2.2)
    C = 2.2e3 * 740.0
    r_amm = 12.0 / (C * (vl + 2.0 * vt))
    expect(abs(r["amm"] / r_amm - 1) < 2e-3, f"AMM {r['amm']:.4g} vs {r_amm:.4g}")
    expect(abs(r["dmm"] / (2.0 * r_amm) - 1) < 1e-9, f"DMM {r['dmm']:.4g} vs {2 * r_amm:.4g}")
    # epoxy into alumina and back: equal energy fluxes at one temperature
    # need Gamma_12 / v1^2 = Gamma_21 / v2^2 (acoustic reciprocity)
    z1, z2, v1, v2 = 2.4e6, 4.1e7, 2003.0, 10480.0
    g12, g21 = CA._amm_gamma(z1, z2, v1, v2, n=4000), CA._amm_gamma(z2, z1, v2, v1, n=4000)
    expect(abs(g12 / v1 ** 2 / (g21 / v2 ** 2) - 1) < 1e-3, f"AMM balance {g12 / v1 ** 2:.4g} vs {g21 / v2 ** 2:.4g}")
    return (f"AMM {r['amm']:.3e}, DMM {r['dmm']:.3e} m²K/W (vL {vl:.0f}, vT {vt:.0f} m/s); "
            f"epoxy→alumina transmission {g12:.4f}, back {g21:.4f}")


def _cal_base(d_um, r_int, k_p):
    from mpsim import calibrate as CA
    base = {"name": "cal", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "F", "shape": "sphere", "size": {"d": d_um, "unit": "um"},
                        "fraction": {"value": 30, "basis": "vol"}, "r_int": r_int}],
            "rve": {"auto": False, "L_um": 8 * d_um, "voxel_um": d_um / 6, "quality": "fast"},
            "analyses": {"thermal": True}}
    base = CA.fill_props(base)
    base["phases"][0]["props"]["k"] = k_p
    return base


@check("Calibration preview: one unknown is recovered; two from one sample are flagged, two sizes separate them")
def t_cal_preview():
    import copy
    from mpsim import calibrate as CA
    from mpsim import spec as S
    base = _cal_base(0.5, 2e-7, 2.0)
    meas = []
    for d in (0.5, 4.0):
        f = copy.deepcopy(base)
        f["phases"][0]["size"]["d"] = d
        meas.append({"property": "k", "value": CA._theory_k(S.normalize(f)), "rel_unc": 0.03,
                     "set": {"phases.0.size.d": d}})
    U_R = {"path": "phases.0.r_int", "label": "R", "lo": 1e-9, "hi": 1e-5}
    U_K = {"path": "phases.0.props.k", "label": "kp", "lo": 0.3, "hi": 30.0}
    a = CA.preview({"base": base, "measurements": meas[:1], "unknowns": [U_R]})
    ra = a["unknowns"][0]
    expect(ra["ci95"][0] < 2e-7 < ra["ci95"][1] and abs(math.log(ra["map"] / 2e-7)) < 0.15, f"R {ra['map']:.3g} {ra['ci95']}")
    b = CA.preview({"base": base, "measurements": meas[:1], "unknowns": [U_R, U_K]})
    rb = b["identifiability"]["rank"]
    expect(rb < 2 and b["identifiability"]["underdetermined"], f"one sample, two unknowns: {rb} combinations determined")
    c = CA.preview({"base": base, "measurements": meas, "unknowns": [U_R, U_K]})
    cc = c["identifiability"]["collinearity"]
    expect(c["identifiability"]["rank"] == 2 and cc is not None and cc < 10, f"two sizes: collinearity {cc}, rank {c['identifiability']['rank']}")
    rc, kc = c["unknowns"]
    expect(rc["ci95"][0] < 2e-7 < rc["ci95"][1] and kc["ci95"][0] < 2.0 < kc["ci95"][1], f"{rc['ci95']} {kc['ci95']}")
    return (f"R alone {ra['map']:.3g} (true 2e-7); one sample: {rb} of 2 combinations; "
            f"two sizes γ {cc:.1f}: R {rc['map']:.3g}, k_p {kc['map']:.3g} (true 2.0)")


@check("Calibration on the RVE: R_int and the filler conductivity from two particle sizes (synthetic data)", quick=False)
def t_cal_rve():
    import json
    from mpsim import calibrate as CA
    from mpsim import parallel as PAR
    base = _cal_base(0.5, 2e-7, 2.0)
    conds = [{"label": "d 0.5", "set": {}}, {"label": "d 4", "set": {"phases.0.size.d": 4.0, "rve.L_um": 32.0, "rve.voxel_um": 4.0 / 6}}]
    with tempfile.TemporaryDirectory() as d:
        unk = [CA._Unknown({"path": "phases.0.r_int", "lo": 1e-9, "hi": 1e-5})]
        samples = CA._prepare({"measurements": [dict(c, property="k") for c in conds]}, base, unk, d, {}, lambda *a: None, None)
        pool = PAR.SolvePool(2, 2)
        try:
            for s_ in samples:
                s_["lab_path"] = pool.share(s_["labels"], name=f"lab{s_['i']}")
            Y = CA._evaluate(samples, lambda s_: s_["form"], np.array([[float(unk[0].unit_of(2e-7))]]), unk, 1e-7,
                             pool, None, lambda *a: None, lambda *a, **k: None, "truth")[0]
        finally:
            pool.close()
        meas = [dict(c, property="k", direction="iso", value=float(y), rel_unc=0.03) for c, y in zip(conds, Y)]
        CA.run({"name": "t", "base": base, "measurements": meas, "_structure_dirs": [d],
                "unknowns": [{"path": "phases.0.r_int", "label": "R", "lo": 1e-9, "hi": 1e-5},
                             {"path": "phases.0.props.k", "label": "kp", "lo": 0.3, "hi": 30.0}]},
               os.path.join(d, "cal"), log=lambda *a: None)
        with open(os.path.join(d, "cal", "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
    R_, K_ = r["unknowns"]
    expect(R_["ci95"][0] < 2e-7 < R_["ci95"][1], f"R {R_['map']:.3g} {R_['ci95']}")
    expect(K_["ci95"][0] < 2.0 < K_["ci95"][1], f"k_p {K_['map']:.3g} {K_['ci95']}")
    expect(abs(math.log(K_["map"] / 2.0)) < 0.25, f"k_p {K_['map']:.3g}")
    dev = max(abs(m["direct"] / m["measured"] - 1) for m in r["measurements"])
    expect(dev < 0.02, f"direct solve at the estimate off by {100 * dev:.1f} %")
    # a suggested particle size comes with the hand-sized RVE scaled to it
    # (8 diameters, 6 voxels per diameter, as in both samples)
    for sg in r["suggestions"]:
        st = sg["set"]
        if "phases.0.size.d" in st:
            dd = st["phases.0.size.d"]
            expect(abs(st.get("rve.L_um", 0) / dd - 8) < 1e-6 and abs(st.get("rve.voxel_um", 0) * 6 / dd - 1) < 1e-6,
                   f"suggestion {sg['label']}: RVE {st.get('rve.L_um')} µm, voxel {st.get('rve.voxel_um')} µm for d {dd} µm")
    return (f"R {R_['map']:.3g} (95 % {R_['ci95'][0]:.2g}–{R_['ci95'][1]:.2g}), k_p {K_['map']:.3g} "
            f"({K_['ci95'][0]:.2g}–{K_['ci95'][1]:.2g}); true 2e-7, 2.0; γ {r['identifiability']['collinearity']:.1f}")


@check("Calibration with touching particles: R_c = 2·R_int recovers R_int; R_int alone is flagged when no value reaches "
       "the measurement; R_int with R_c from one sample is a combination; every parameter set is recorded", quick=False)
def t_cal_contacts():
    import json
    from mpsim import calibrate as CA
    from mpsim import parallel as PAR
    base = _cal_base(0.5, 0.0, 30.0)
    base["phases"][0]["fraction"]["value"] = 50
    link = [{"path": "phases.0.r_contact", "expr": "2*x"}]
    U_L = {"path": "phases.0.r_int", "label": "R", "lo": 1e-10, "hi": 1e-6, "links": link}
    with tempfile.TemporaryDirectory() as d:
        # the 'measurement': R_int 5e-9 with R_c 1e-8 on the contact faces
        unk = [CA._Unknown(U_L)]
        samples = CA._prepare({"measurements": [{"property": "k", "set": {}}]}, base, unk, d, {}, lambda *a: None, None)
        pool = PAR.SolvePool(2, 2)
        try:
            samples[0]["lab_path"] = pool.share(samples[0]["labels"], name="lab0")
            if samples[0]["cmask"] is not None:
                samples[0]["cm_path"] = os.path.join(pool.tmp, "cm0.npy")
                np.save(samples[0]["cm_path"], samples[0]["cmask"])
            y_true = float(CA._evaluate(samples, lambda s_: s_["form"], np.array([[float(unk[0].unit_of(5e-9))]]), unk,
                                        1e-7, pool, None, lambda *a: None, lambda *a, **k: None, "truth")[0][0])
        finally:
            pool.close()
        meas = [{"label": "m", "property": "k", "direction": "iso", "value": y_true, "rel_unc": 0.03}]

        def run(tag, unknowns, m=meas):
            CA.run({"name": tag, "base": base, "measurements": m, "_structure_dirs": [d], "unknowns": unknowns},
                   os.path.join(d, tag), log=lambda *a: None)
            with open(os.path.join(d, tag, "result.json"), encoding="utf-8") as fh:
                return json.load(fh)
        a = run("linked", [U_L])
        ra = a["unknowns"][0]
        expect(ra["ci95"][0] < 5e-9 < ra["ci95"][1], f"linked: R {ra['map']:.3g} {ra['ci95']}")
        ev = a["design"]["evaluations"]
        expect(len(ev) == len(a["design"]["points"]) + 1 and ev[-1]["stage"] == "check", f"{len(ev)} evaluations")
        expect(all(abs(e["linked"][0] / (2 * e["values"][0]) - 1) < 1e-9 and e["results"][0] > 0 for e in ev),
               "every set records R_c = 2 R_int and its value")
        # R_int alone with free contacts, against a value below the matrix
        b = run("alone", [{"path": "phases.0.r_int", "label": "R", "lo": 1e-10, "hi": 1e-6}],
                [dict(meas[0], value=0.05)])
        expect(b["unknowns"][0].get("unreached") and b["contact_bypass"] == ["F"], "R_int alone: not flagged as unreached")
        expect(any("lies below every value" in n and "touch" in n for n in b["notes"]), "R_int alone: no reason given")
        # R_int and R_c from one sample: a combination, said in one note
        c = run("both", [{"path": "phases.0.r_int", "label": "R", "lo": 1e-10, "hi": 1e-6},
                         {"path": "phases.0.r_contact", "label": "Rc", "lo": 1e-10, "hi": 1e-6}])
        expect(any(n.startswith("R_int and R_c of F act together") for n in c["notes"]), "R_int + R_c: no combination note")
    lo_b = min(e["results"][0] for e in b["design"]["evaluations"])
    return (f"k {y_true:.3f} at R_int 5e-9, R_c 1e-8; linked: R_int {ra['map']:.3g} (95 % {ra['ci95'][0]:.2g}–"
            f"{ra['ci95'][1]:.2g}); R_int alone: lowest k {lo_b:.3f}, flagged; both: R_c {c['unknowns'][1]['map']:.3g}, "
            f"γ {c['identifiability']['collinearity']:.3g}")


@check("Calibration from a viscosity: the bound resin layer of 1 µm silica from a particle-dynamics measurement (synthetic data)", quick=False)
def t_cal_visc():
    # a 'measured' relative viscosity made by the particle dynamics with a
    # 10 nm bound layer, then the layer calibrated back (0-20 nm): the same
    # spheres for every parameter set, the run's own scatter in the likelihood
    import json
    from mpsim import calibrate as CA
    from mpsim import spec as S
    base = {"name": "t", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "sio2", "name": "Silica", "shape": "sphere", "size": {"d": 1.0, "unit": "um"},
                        "dist": {"type": "lognormal", "cv": 0.15}, "fraction": {"value": 50, "basis": "vol"}}],
            "analyses": {"viscosity": True},
            "options": {"viscosity": {"gd_ref": 10, "dem": {"n": 150, "strain": 2.0, "backend": "cpu"}}}}
    f = CA.fill_props(base)
    f["options"]["viscosity"]["dem"]["bound_nm"] = 10.0
    smp = CA._dem_sample(0, {"property": "mu_r"}, f, S.normalize(f), lambda *a: None)
    y, rel = CA._dem_value(smp, f, None)
    expect(math.isfinite(y) and y > 1, f"synthetic measurement {y}")
    # a layer that jams these spheres is ruled out, not run
    expect(not math.isfinite(CA._dem_value(smp, CA._set_theta(f, [CA._Unknown(
        {"path": "options.viscosity.dem.bound_nm", "lo": 0, "hi": 100, "scale": "lin"})], [100.0]), None)[0]),
        "a 100 nm layer on 1 µm spheres was run")
    with tempfile.TemporaryDirectory() as d:
        CA.run({"name": "t", "base": base,
                "measurements": [{"label": "1 µm", "property": "mu_r", "value": y, "rel_unc": 0.05}],
                "unknowns": [{"path": "options.viscosity.dem.bound_nm", "label": "b", "unit": "nm", "lo": 0, "hi": 20,
                              "scale": "lin"}],
                "options": {"refine_rounds": 1}}, d, log=lambda *a: None)
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
    u = r["unknowns"][0]
    # 150 spheres to a strain of 2 scatter by about 20 % between packings:
    # the interval is wide, but it holds the truth and is narrower than 0-20
    expect(u["ci95"][0] < 10.0 < u["ci95"][1], f"bound layer {u['map']:.2f} nm, 95 % {u['ci95']}")
    expect(u["narrowing"] > 1.3, f"95 % interval {u['ci95']} hardly narrower than the range")
    return (f"μr {y:.1f} (± {100 * rel:.1f} %) with 10 nm; calibrated {u['map']:.1f} nm "
            f"(95 % {u['ci95'][0]:.1f}–{u['ci95'][1]:.1f}) from {len(r['design']['points'])} runs")


@check("Moisture: Crank's half-saturation time, an impermeable sphere against Maxwell, swelling with stiff dry fillers")
def t_moisture():
    from mpsim import pipeline
    from mpsim.solvers import conduction as CD
    from mpsim.solvers import fans as FA
    # Crank: a plate of thickness L reaches half saturation at t = 0.04919 L^2/D
    D, L = 1e-12, 1e-3
    t = np.linspace(1, 0.2 * L * L / D, 4000)
    f = pipeline.crank_plate(D, L, t)
    t50 = float(np.interp(0.5, f, t)) * D / (L * L)
    expect(abs(t50 / 0.04919 - 1) < 2e-3, f"t50 {t50:.5f} L^2/D (0.04919)")
    # impermeable sphere: P*/P_m = 2(1-phi)/(2+phi) (Maxwell), with the
    # filler's c_sat = 0 its permeability is zero
    lab = _sphere_lab(40, 9.0)
    phi = float(lab.mean())
    v, _ = CD.contrast_policy(lab, np.array([1.0, 0.0]), 0, 1e6, {})
    r = CD.solve_direction(lab, v, 0, tol=1e-9)
    ratio = float(r["column"][0]) / (1.0 - phi)          # D*/D_m = P*/P_m / (1 - phi)
    mx = 2.0 / (2.0 + phi)
    expect(abs(ratio / mx - 1) < 0.02, f"D*/D_m {ratio:.4f} vs Maxwell {mx:.4f}")
    # swelling: resin eigenstrain e, rigid dry filler -> strain below e (1 - phi)
    e = 1e-3
    rf = FA.homogenize(lab, [3.0, 70.0], [0.35, 0.2], [e * 1e6, 0.0], tol=1e-8)
    sw = float(np.mean(rf["alpha"][:3])) * 1e-6
    expect(0 < sw < e * (1 - phi) * 1.01, f"swelling {sw:.3e} vs resin {e:.1e}, (1-phi) e {e * (1 - phi):.3e}")
    return f"t50 = {t50:.5f} L²/D; φ {phi:.3f}: D*/D_m {ratio:.4f} (Maxwell {mx:.4f}); swelling {sw / e:.3f} of the resin's"


@check("Structure generation: volume fraction, periodic rasterisation, particles kept apart without trimming")
def t_generator():
    from mpsim import generate as GEN
    from mpsim import shapes as SH
    lab, info = GEN.generate(40, 1.0, [dict(name="s", shape="sphere", size_um={"d": 8.0}, dist={"type": "mono"},
                                            vf=0.30, orientation={"mode": "iso"}, overlap=False, gap_frac=0.0)],
                             seed=5, log=lambda *a: None)
    vf = info["phases"][0]["vf_voxel"]
    expect(abs(vf / 0.30 - 1) < 0.02, f"vf {vf}")
    expect(info["contacts_removed"] == 0, "voxels were removed at contacts although contacts = apart")
    expect(info["touching_pairs"] == 0, f"{info['touching_pairs']} voxel pairs of two particles touch at 30 vol %")
    expect(info["voxel_conflicts"] == 0, f"{info['voxel_conflicts']} voxels claimed by two particles")
    own = np.zeros((32, 32, 32), np.int32)
    counts = []
    planes, nf = SH.EMPTY_PLANES
    P = np.zeros(6)
    P[0] = 6.3
    for c in ((16.0, 16.0, 16.0), (0.0, 0.0, 0.0), (31.0, 1.0, 16.0)):
        own[:] = 0
        ph = np.full(4, -1, np.int32)
        ph[1] = 0
        GEN._place(own, ph, np.zeros(2, np.uint8), np.zeros(2, np.uint8), 1.0, 0, c[0], c[1], c[2],
                   np.eye(3), P, planes, nf, 0.0, 0.0, 0, 1, False, False, np.zeros(1, np.int64),
                   np.empty(40 ** 3, np.int64))
        counts.append(int((own == 1).sum()))
    expect(len(set(counts)) == 1, f"voxel counts differ under whole-voxel translation: {counts}")
    return (f"vf {100*vf:.2f} %, {info['phases'][0]['n_particles']} particles, no two touching, "
            f"translation-invariant {counts[0]} voxels")


@check("Redrawing: the stored particles reproduce the generated structure exactly, and a finer grid keeps the fraction")
def t_rasterize():
    from mpsim import generate as GEN
    out = []
    for shape, size, vf in (("sphere", {"d": 8.0}, 0.40), ("cube", {"d": 7.0}, 0.30)):
        ph = [dict(name=shape, shape=shape, size_um=size, dist={"type": "mono"}, vf=vf,
                   orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)]
        lab, info = GEN.generate(40, 1.0, ph, seed=4, log=lambda *a: None)
        same = GEN.rasterize(40, 1.0, info, ph)
        diff = int(np.count_nonzero(same != lab))
        expect(diff == 0, f"{shape}: {diff} voxels differ when the particles are redrawn on the same grid")
        fine = GEN.rasterize(60, 40.0 / 60, info, ph)
        f0, f1 = float((lab == 1).mean()), float((fine == 1).mean())
        expect(abs(f1 / f0 - 1) < 0.02, f"{shape}: fraction {f0:.4f} on 40³, {f1:.4f} on 60³")
        out.append(f"{shape} identical, {100*f0:.2f} % → {100*f1:.2f} % on 60³")
    return "; ".join(out)


@check("Thin film: every particle lies wholly inside the thickness (spheres and rigid clumps), x and y stay periodic")
def t_film_generate():
    import copy
    from mpsim import generate as GEN
    out = []
    for shape, size, vf in (("sphere", {"d": 5.0}, 0.35), ("cube", {"d": 4.0}, 0.25)):
        ph = [dict(name=shape, shape=shape, size_um=size, dist={"type": "mono"}, vf=vf,
                   orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)]
        N, nz, m = 36, 14, 6
        lab, info = GEN.generate(N, 1.0, ph, seed=7, log=lambda *a: None, nz=nz)
        expect(lab.shape == (N, N, nz), f"{shape}: shape {lab.shape}")
        f = float((lab == 1).mean())
        expect(abs(f / vf - 1) < 0.05, f"{shape}: fraction {f:.4f} for {vf}")
        # the same particles on a grid m voxels taller at each face: nothing
        # may be drawn outside the film
        big = copy.deepcopy(info)
        big["particles"]["centers"] = big["particles"]["centers"] + np.array([0.0, 0.0, m * 1.0])
        big["T_um"] = (nz + 2 * m) * 1.0
        tall = GEN.rasterize(N, 1.0, big, ph, nz=nz + 2 * m)
        outside = int(np.count_nonzero(tall[:, :, :m])) + int(np.count_nonzero(tall[:, :, nz + m:]))
        expect(outside == 0, f"{shape}: {outside} particle voxels beyond the film faces")
        same = int(np.count_nonzero(tall[:, :, m:nz + m] != lab))
        expect(same == 0, f"{shape}: {same} voxels differ inside the film")
        # x, y periodic: particles do cross the lateral faces
        wrap = bool(lab[0].any() and lab[-1].any())
        out.append(f"{shape} {100*f:.1f} %, 0 voxels outside, lateral wrap {'yes' if wrap else 'no'}")
    # a matrix skin at the faces: no particle voxel in the face layers, the
    # fraction still that of the whole film
    ph = [dict(name="s", shape="sphere", size_um={"d": 5.0}, dist={"type": "mono"}, vf=0.35,
               orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)]
    lab, info = GEN.generate(36, 1.0, ph, seed=7, log=lambda *a: None, nz=14, skin=1)
    faces = int(np.count_nonzero(lab[:, :, 0])) + int(np.count_nonzero(lab[:, :, -1]))
    f = float((lab == 1).mean())
    expect(faces == 0 and lab.shape == (36, 36, 14), f"skin: {faces} particle voxels in the face layers, shape {lab.shape}")
    expect(abs(f / 0.35 - 1) < 0.05, f"skin: fraction {f:.4f} for 0.35")
    out.append(f"skin 1 voxel: faces clear, {100*f:.1f} %")
    return "; ".join(out)


@check("Thin film FV: exact series and parallel values for layers across and along the film")
def t_film_fv():
    from mpsim.solvers import conduction as CD
    k = np.array([1.0, 10.0])
    ser, par = 2.0 / (1.0 / 1.0 + 1.0 / 10.0), 5.5
    stack = np.zeros((16, 16, 12), np.uint8)
    stack[:, :, 6:] = 1                         # layers stacked through the thickness
    side = np.zeros((16, 16, 12), np.uint8)
    side[8:] = 1                                # layers side by side along x
    got = {}
    for name, lab, want in (("stacked", stack, (par, par, ser)), ("side by side", side, (ser, par, par))):
        vals = []
        for di in range(3):
            r = CD.solve_direction(lab, k, di, tol=1e-10, film=True)
            vals.append(r["k_eff"])
            if di == 2:
                expect(r.get("absolute_potential"), "the z solve must return the full potential between plates")
        for d, v, w in zip("xyz", vals, want):
            expect(abs(v / w - 1) < 1e-5, f"{name} k{d}{d} {v:.6f} against exact {w:.6f}")
        got[name] = "/".join(f"{v:.4f}" for v in vals)
    return f"stacked {got['stacked']}, side by side {got['side by side']} (exact {par} and {ser:.4f})"


@check("Shapes: rasterised volume of every v4 shape = analytic volume (±3 %), exact under rotation")
def t_shapes():
    from mpsim import generate as GEN
    from mpsim import shapes as SH
    rng = np.random.default_rng(3)
    tab = SH.PolyTable()
    ico = tab.add("icosahedron")
    irr = tab.add("irregular", n_vertices=12, rng=rng)
    planes, nf = tab.arrays()
    cases = [("sphere", {"d": 20}, None), ("spheroid", {"d": 24, "aspect": 0.4}, None),
             ("cylinder", {"d": 12, "length": 40}, None), ("spherocylinder", {"d": 12, "length": 40}, None),
             ("cube", {"d": 18}, None), ("cuboid", {"d": 30, "ly": 16, "lz": 8}, None),
             ("superellipsoid", {"d": 26, "ly": 20, "lz": 14, "n": 5}, None),
             ("polyhedron", {"d": 30}, ico), ("polyhedron", {"d": 30}, irr),
             ("helix", {"d": 5, "coil_d": 20, "pitch": 10, "turns": 2.5}, None)]
    worst = 0.0
    for shape, size, ids in cases:
        code, P = SH.params(shape, size, ids)
        R = SH.sample_rotations(1, "iso", 0, rng)[0]
        own = np.zeros((64, 64, 64), np.int32)
        ph = np.full(4, -1, np.int32)
        ph[1] = 0
        GEN._place(own, ph, np.zeros(2, np.uint8), np.zeros(2, np.uint8), 1.0, code, 32.0, 32.0, 32.0,
                   np.ascontiguousarray(R), P, planes, nf, 0.0, 0.0, 0, 1, False, False, np.zeros(1, np.int64),
                   np.empty(70 ** 3, np.int64))
        v = float((own == 1).sum())
        va = SH.volume(code, P, tab.meta)
        e = abs(v / va - 1)
        worst = max(worst, e)
        expect(e < 0.03, f"{shape}: voxel volume {v:.0f} vs analytic {va:.0f}")
    return f"10 shapes, largest volume error {100*worst:.2f} %"


@check("Blends never overlap: two sphere sizes at 55 vol%, cubes with flakes at 45 vol% - no voxel in two particles, no particle trimmed", quick=False)
def t_blend_overlap():
    from scipy.spatial import cKDTree
    from mpsim import generate as GEN
    from mpsim import shapes as SH

    def ph(name, shape, size, vf):
        return dict(name=name, shape=shape, size_um=size, dist={"type": "mono"}, vf=vf,
                    orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)
    out = []
    for name, N, h, phases in (
            ("10 + 2.5 µm spheres", 64, 0.5, [ph("S10", "sphere", {"d": 10.0}, 0.385),
                                              ph("S2.5", "sphere", {"d": 2.5}, 0.165)]),
            ("cubes + flakes", 48, 0.5, [ph("C", "cube", {"d": 5.0}, 0.25),
                                         ph("F", "cuboid", {"d": 8.0, "ly": 8.0, "lz": 1.5}, 0.20)])):
        lab, info = GEN.generate(N, h, phases, seed=3, log=lambda *a: None)
        p = info["particles"]
        planes, nf = info["planes"]
        n = len(p["codes"])
        # every particle drawn again on an empty grid: a voxel inside two counts
        owner = np.zeros(lab.shape, np.int32)
        phase_of = np.full(n + 4, -1, np.int32)
        cs, cc, fs, ts = np.zeros((n, 3)), np.zeros(n, np.int64), np.zeros((n, 3)), np.zeros((n, 3))
        ext = max(2.0 * SH.circumradius(int(c), P) for c, P in zip(p["codes"], p["P"]))
        shared = int(GEN._raster_track(owner, phase_of, np.zeros(len(phases) + 1, np.uint8), float(h),
                                       np.ascontiguousarray(p["codes"]), np.ascontiguousarray(p["centers"]),
                                       np.ascontiguousarray(p["R"]), np.ascontiguousarray(p["P"]), planes, nf,
                                       np.ascontiguousarray(p["phase"]), 1, cs, cc, fs, ts, np.ones(n),
                                       np.ones(n), 0.9, np.zeros(n, np.int64), np.empty(int(math.ceil(ext / h) + 4) ** 3, np.int64),
                                       False))
        expect(shared == 0, f"{name}: {shared} voxels inside two particles")
        expect(info["voxel_conflicts"] == 0, f"{name}: {info['voxel_conflicts']} voxels kept by the first particle")
        pairs = 0
        if all(int(c) == 0 for c in p["codes"]):
            L = N * h
            C, r = np.mod(p["centers"], L), p["P"][:, 0]
            cand = cKDTree(C, boxsize=L).query_pairs(2.0 * float(r.max()), output_type="ndarray")
            dv = C[cand[:, 1]] - C[cand[:, 0]]
            dv -= L * np.round(dv / L)
            pairs = int(np.count_nonzero(np.linalg.norm(dv, axis=1) < r[cand[:, 0]] + r[cand[:, 1]]))
            expect(pairs == 0, f"{name}: {pairs} sphere pairs overlap")
        vf = [pi["vf_voxel"] / pi["vf_target"] for pi in info["phases"]]
        expect(min(vf) > 0.98, f"{name}: volume fraction reached {min(vf):.3f} of the target")
        shr = (info.get("shrunk") or {}).get("n", 0)
        out.append(f"{name}: {n} particles, 0 shared voxels, {shr} shrunk, fraction ≥ {100 * min(vf):.1f} % of target")
    return "; ".join(out)


@check("Dense packing: spheres to 60 vol% and cubes to 40 vol% without overlap (growth relaxation / clump relaxation)", quick=False)
def t_dense():
    from mpsim import generate as GEN
    out = []
    for shape, size, vf in (("sphere", {"d": 10.0}, 0.60), ("cube", {"d": 8.0}, 0.40)):
        lab, info = GEN.generate(64, 0.5, [dict(name=shape, shape=shape, size_um=size, dist={"type": "mono"}, vf=vf,
                                                orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)],
                                 seed=2, log=lambda *a: None)
        got = info["phases"][0]["vf_voxel"]
        rel = info["voxel_conflicts"] / 64 ** 3
        expect(got > 0.97 * vf, f"{shape}: {100*got:.1f} % of {100*vf:.0f} %")
        expect(info["voxel_conflicts"] == 0, f"{shape}: {info['voxel_conflicts']} shared voxels")
        out.append(f"{shape} {100*got:.1f} % ({info['route'].split('(')[0].strip()}, shared voxels {100*rel:.3f} %)")
    return "; ".join(out)


@check("Interfacial resistance: dilute spheres match Hasselman-Johnson; a contact plane matches the series law")
def t_kapitza():
    from mpsim.solvers import conduction as CD
    n, d = 48, 20.0
    g = np.indices((n, n, n)).transpose(1, 2, 3, 0) + 0.5
    lab = (np.linalg.norm(g - n / 2, axis=-1) < d / 2).astype(np.uint8)
    phi = float(lab.mean())
    a = d / 2                                    # voxel units, h = 1
    faces = sum(int(np.count_nonzero(lab != np.roll(lab, -1, axis=ax))) for ax in range(3))
    ratio = faces / (4 * math.pi * a * a)
    km, kp = 1.0, 20.0
    worst = 0.0
    for alpha in (0.0, 0.5, 2.0):
        R = alpha * a / km                       # R_int in voxel-length units
        rp = np.zeros((2, 2))
        rp[0, 1] = rp[1, 0] = R * ratio
        k = CD.solve_direction(lab, np.array([km, kp]), 0, rpair=rp, tol=1e-9)["k_eff"]
        b = kp * (1 + 2 * alpha) + 2 * km
        c = kp * (1 - alpha) - km
        hj = km * (b + 2 * phi * c) / (b - phi * c)
        e = abs(k / hj - 1)
        worst = max(worst, e)
        expect(e < 0.01, f"alpha {alpha}: FV {k:.5f} vs Hasselman-Johnson {hj:.5f}")
    lab2 = np.zeros((32, 8, 8), np.uint8)
    cm = np.zeros_like(lab2)
    cm[15] = 1                                   # +x faces of the plane x = 15
    rc = np.full((1, 1), 4.0)
    k2 = CD.solve_direction(lab2, np.array([2.0]), 0, cmask=cm, rcpair=rc, tol=1e-12)["k_eff"]
    exact = 32.0 / (32.0 / 2.0 + 4.0)
    expect(abs(k2 / exact - 1) < 1e-8, f"contact plane {k2} vs {exact}")
    return f"largest deviation from Hasselman-Johnson {100*worst:.2f} % (φ {100*phi:.1f} %, voxel/true area {ratio:.2f}); contact plane exact"


@check("Structure export: one grey value per material, a label map and a legend")
def t_export():
    import csv
    import tempfile
    import tifffile
    from mpsim import pipeline as PL
    lab = np.zeros((6, 7, 8), np.uint8)
    lab[1:3] = 1
    lab[4:] = 2
    table = [{"kind": "matrix", "name": "m", "props": {"E": 3.0, "k": 0.2}},
             {"kind": "core", "name": "f", "props": {"E": 70.0, "k": 1.4}},
             {"kind": "core", "name": "air", "props": {"E": 0.0, "k": 0.026}}]
    with tempfile.TemporaryDirectory() as d:
        PL.export_structure(d, lab, table, 0.5, [0.3, 0.3, 0.4])
        grey = tifffile.imread(os.path.join(d, "structure.tif"))
        ids = tifffile.imread(os.path.join(d, "structure_labels.tif"))
        rows = list(csv.DictReader(open(os.path.join(d, "structure_legend.csv"), encoding="utf-8")))
    expect(sorted(set(grey.ravel().tolist())) == [0, 128, 255], f"grey values {sorted(set(grey.ravel().tolist()))}")
    expect(ids.shape == (8, 7, 6) and int(ids.max()) == 2, "label map shape or values")
    expect(len(rows) == 3 and rows[2]["void"] == "1", "legend rows")
    return "grey 0 / 128 / 255, ZYX label map, 3 legend rows"


@check("Structure analysis: local thickness of a 20-voxel sphere ≈ 20; a sphere cut by the faces = 1 cluster")
def t_morph():
    from mpsim import morphology as MO
    i = np.arange(48) + 0.5
    D2 = (i[:, None, None] - 24) ** 2 + (i[None, :, None] - 24) ** 2 + (i[None, None, :] - 24) ** 2
    sph = D2 <= 100.0
    sd = MO.size_distribution(MO.local_thickness(sph), sph, 1.0)
    expect(abs(sd["d50_um"] / 20.0 - 1) < 0.05, f"D50 {sd['d50_um']}")
    corner = np.roll(np.roll(np.roll(sph, 24, 0), 24, 1), 24, 2)
    cl = MO.clusters(corner)
    expect(cl["n_clusters"] == 1, f"clusters {cl}")
    box = np.zeros((40, 40, 40), bool)
    box[:, 10:20, 10:20] = True
    box[5:15, 25:35, 25:35] = True
    ch = MO.chord_lengths(box, 1.0)
    expect(all(abs(ch[a]["mean_um"] - 10.0) < 1e-9 for a in "xyz"), f"chords {ch}")
    from mpsim.solvers import conduction as CD
    core = D2 <= 64.0
    openp = np.zeros_like(core)
    openp[:, 0:3, 0:3] = True                      # a channel that wraps along x
    keep = CD.percolating_mask(core | openp)
    expect(keep[openp].all() and not keep[core].any(), "a closed core must not count as open porosity")
    return f"D50 {sd['d50_um']:.2f}, corner sphere {cl['n_clusters']} cluster, chord length 10.0 µm, closed pore excluded"


@check("Pore space: Poiseuille permeability and tortuosity of a straight cylindrical channel", quick=False)
def t_channel():
    PB = _puma()
    lab = channel()
    phi = float((lab == 1).mean())
    R = 8.0e-6
    K = np.asarray(PB.permeability(lab, [1], 1e-6, "x", tol=1e-8)["K"])
    Kxx = float(K[0, 0] if K.ndim == 2 else K[0])
    ratio = Kxx / (phi * R * R / 8.0)
    tr = PB.tortuosity(lab, [1], 1e-6, "x", tol=1e-8)
    expect(abs(ratio - 1) < 0.10, f"K / (φ r²/8) = {ratio:.3f}")
    expect(abs(tr["tortuosity"] - 1) < 1e-3, f"τ = {tr['tortuosity']}")
    return f"K/(φr²/8) = {ratio:.3f}, τ = {tr['tortuosity']:.4f}"


@check("Acoustics: viscous and thermal lengths of straight pores equal the pore radius (Λ = Λ' = r)", quick=False)
def t_jca_lengths():
    PB = _puma()
    from mpsim import acoustics as AC
    lab = channel()
    void = lab == 1
    tr = PB.tortuosity(lab, [1], 1e-6, "x", tol=1e-8)
    area, _ = PB.surface_area(void.astype(np.uint8), [1], 1e-6)
    lam = AC.viscous_length(tr["C"], void, 1e-6, "x", area / AC.voxel_face_area(void, 1e-6))
    lam_p = 2 * void.sum() * 1e-18 / area
    expect(abs(lam / 8e-6 - 1) < 0.05 and abs(lam_p / 8e-6 - 1) < 0.05, f"Λ {lam*1e6:.3f} µm, Λ' {lam_p*1e6:.3f} µm")
    from mpsim import spec as S
    gas = S.normalize({"matrix": {"material_id": "al2o3"}, "phases": [{"material_id": "air", "shape": "network",
                       "size": {"d": 2, "unit": "um"}, "fraction": {"value": 30}}], "analyses": {"acoustics": True}})["options"]["gas"]
    f = np.geomspace(100, 10000, 60)
    a = np.asarray(AC.absorption(f, 0.02, 0.9, 20000.0, 1.2, 100e-6, 200e-6, gas)["alpha"])
    expect(np.all((a >= 0) & (a <= 1)) and a[-1] > a[0], "absorption outside [0, 1] or without a rising trend")
    return f"Λ = {lam*1e6:.3f} µm, Λ' = {lam_p*1e6:.3f} µm (r = 8 µm); α(1 kHz) = {np.interp(1000, f, a):.3f}"


@check("Porosimetry: largest through-pore of a channel of 16 µm diameter (±1.5 voxels)", quick=False)
def t_porosimetry():
    from mpsim import poro as PO
    lab = channel()
    res, field = PO.porosimetry(lab == 1, 1.0, steps=25, directions="xy")
    d = res["through_pore"]["x"]["diameter_um"]
    expect(abs(d - 16.0) <= 1.5, f"through-pore {d:.2f} µm")
    expect(res["through_pore"]["y"] is None, "a pore without a path along y was reported")
    expect(res["max_intrusion"] > 0.99, f"max intrusion {res['max_intrusion']}")
    kn = PO.knudsen(1e-6, {"pressure_Pa": 101325.0, "molecule_diameter_nm": 0.37, "molar_mass_g_mol": 28.97,
                           "diffusivity_m2_s": 2e-5}, 298.15)
    expect(abs(kn["mean_free_path_nm"] / 67.0 - 1) < 0.08, f"mean free path {kn['mean_free_path_nm']:.1f} nm")
    return f"through-pore {d:.2f} µm, maximum intrusion {res['max_intrusion']:.3f}, air mean free path {kn['mean_free_path_nm']:.1f} nm"


@check("Pore network and ray casting: SNOW2 extraction and isotropic extinction of a random structure", quick=False)
def t_network_radiation():
    from mpsim import generate as GEN
    from mpsim import poro as PO
    PB = _puma()
    rng = np.random.default_rng(1)
    fld = GEN.random_field((56, 56, 56), 1.0, 6.0, rng)
    pores = fld <= np.quantile(fld, 0.4)
    net, view = PO.pore_network(pores, 1.0)
    expect(net["n_pores"] > 20 and 2.0 < net["coordination_mean"] < 10.0, f"network {net['n_pores']} pores, Z {net['coordination_mean']}")
    rad = PB.radiation(pores.astype(np.uint8), [1], 1e-6, 150, 400)
    b = np.asarray(rad["beta_1pm"])
    spread = (b.max() - b.min()) / b.mean()
    expect(spread < 0.10, f"extinction coefficients {b}")
    return f"{net['n_pores']} pores, Z = {net['coordination_mean']:.2f}; β = {b.mean():.3g} 1/m, directional spread {100*spread:.1f} %"


@check("Grain analysis: orientation of z-aligned rods from the particle list and from the PuMA structure tensor", quick=False)
def t_grains():
    from mpsim import generate as GEN
    from mpsim import grains as GR
    from mpsim import spec as S
    _puma()
    form = {"matrix": {"material_id": "epoxy"}, "phases": [{"material_id": "al2o3", "name": "rod", "shape": "spheroid",
            "size": {"d": 3, "aspect": 4, "unit": "um"}, "fraction": {"value": 12}, "orientation": {"mode": "z", "spread_deg": 5}}],
            "analyses": {"grains": True}}
    spec = S.normalize(form)
    labs, info = GEN.generate(56, 0.5, [dict(name="rod", shape="spheroid", size_um={"d": 3.0, "aspect": 4.0}, dist={"type": "mono"},
                                            vf=0.12, orientation={"mode": "z", "spread_deg": 5.0}, overlap=False, gap_frac=0.0)],
                              seed=2, log=lambda *a: None)
    g, _ = GR.analyse(spec, info, labs, spec["labels"], 0.5)
    p = g["phases"]["1"]
    Azz = np.asarray(p["orientation_tensor_image"])[2, 2]
    expect(p["hermans"]["z"] > 0.9 and Azz > 0.85, f"Hermans z {p['hermans']['z']:.3f}, image A_zz {Azz:.3f}")
    return f"Hermans z {p['hermans']['z']:.3f}, structure tensor A_zz {Azz:.3f}, sphericity {p['sphericity']:.3f}"


@check("Percolation: straight channel spans x with geodesic tortuosity 1 and is blocked along y", quick=False)
def t_percolation():
    from mpsim import percolation as PC
    lab = channel()
    res, fields = PC.analyse(lab == 1, 1.0, "xy")
    x, y = res["by_direction"]["x"], res["by_direction"]["y"]
    expect(x["percolates"] and not y["percolates"], f"x {x['percolates']}, y {y['percolates']}")
    expect(abs(x["geodesic_tortuosity"] - 1.0) < 0.02, f"geodesic tortuosity {x['geodesic_tortuosity']}")
    # a sphere cut off from the faces is isolated, not spanning
    i = np.arange(40) + 0.5
    D2 = (i[:, None, None] - 20) ** 2 + (i[None, :, None] - 20) ** 2 + (i[None, None, :] - 20) ** 2
    blob = D2 <= 36.0
    r2, _ = PC.analyse(blob, 1.0, "x", want_fields=False)
    expect(not r2["by_direction"]["x"]["percolates"] and r2["by_direction"]["x"]["isolated_fraction"] > 0.99,
           f"isolated sphere: {r2['by_direction']['x']}")
    # the largest sphere through a pore of radius 8 is the pore itself
    r3, _ = PC.analyse(lab == 1, 1.0, "x", want_fields=False, want_paths=True)
    rows = r3["by_direction"]["x"]["passable_spheres"]
    d_max = rows[0]["diameter_um"]
    expect(abs(d_max - 16.0) <= 1.5, f"largest passing sphere {d_max:.2f} µm")
    expect(all(rows[i]["diameter_um"] > rows[i + 1]["diameter_um"] for i in range(len(rows) - 1)),
           "the passing sizes are not ordered from large to small")
    expect(rows[0]["polyline_um"] and abs(rows[0]["tortuosity"] - 1.0) < 0.05,
           f"path of the largest sphere: {rows[0]['tortuosity']}")
    return (f"τ_geo(x) = {x['geodesic_tortuosity']:.4f}, y blocked, largest passing sphere "
            f"{d_max:.2f} µm over {len(rows)} sizes, isolated sphere "
            f"{100*r2['by_direction']['x']['isolated_fraction']:.0f} % isolated")


@check("Field lines: a uniform field in a straight channel gives straight lines that cross the sample")
def t_streamlines():
    from mpsim import streamlines as SL
    mask = channel() == 1
    vec = np.zeros(mask.shape + (3,))
    vec[..., 0] = np.where(mask, 1.0, 0.0)
    lines, summary = SL.trace(vec, mask, 1.0, 0, n_lines=12)
    expect(summary["n_lines"] >= 6, f"only {summary['n_lines']} lines were traced")
    expect(summary["n_through"] == summary["n_lines"], f"{summary['n_through']} of {summary['n_lines']} reached the far face")
    expect(abs(summary["tortuosity"] - 1.0) < 0.05, f"tortuosity of a straight channel {summary['tortuosity']}")
    # the same field pointing the other way (the finite-volume flux of a unit
    # gradient does): the lines must still cross; they once all left the box
    _, sneg = SL.trace(-vec, mask, 1.0, 0, n_lines=12)
    expect(sneg["n_lines"] >= 6 and sneg["n_through"] == sneg["n_lines"],
           f"reversed field: {sneg['n_through']} of {sneg['n_lines']} lines crossed")
    # a swirl of the same magnitude as the axial field puts every line at 45°
    # to the axis without pushing it out of the channel, so each one must come
    # out exactly sqrt(2) longer than the straight distance
    c = 1.0
    yy = (np.arange(mask.shape[1]) + 0.5 - mask.shape[1] / 2)[None, :, None]
    zz = (np.arange(mask.shape[2]) + 0.5 - mask.shape[2] / 2)[None, None, :]
    rr = np.where(np.sqrt(yy ** 2 + zz ** 2) == 0, 1.0, np.sqrt(yy ** 2 + zz ** 2))
    vec[..., 1] = np.where(mask, -c * zz / rr, 0.0)
    vec[..., 2] = np.where(mask, c * yy / rr, 0.0)
    # a line seeded next to the wall drifts out of the channel and stops; the
    # ones that do cross must come out exactly sqrt(2) longer
    l2, s2 = SL.trace(vec, mask, 1.0, 0, n_lines=12)
    crossed = [l["length_um"] for l in l2 if l["through"]]
    expect(len(crossed) >= 3, f"only {len(crossed)} of {s2['n_lines']} lines crossed the swirling channel")
    ratio = float(np.mean(crossed)) / s2["straight_um"]
    expect(abs(ratio - math.sqrt(1 + c * c)) < 0.05, f"swirling field ratio {ratio}")
    return (f"{summary['n_lines']} lines, all crossing, τ = {summary['tortuosity']:.4f}; "
            f"45° swirl {len(crossed)} crossing lines at {ratio:.3f} × the straight distance (√2 = 1.414); "
            f"reversed field {sneg['n_through']} crossing")


@check("Filtration: large particles are intercepted, the smallest diffuse to the wall, and a minimum lies between")
def t_filtration():
    from mpsim import filtration as FI
    mask = channel() == 1                      # straight pore, radius 8 voxels, along x
    u = np.zeros(mask.shape + (3,))
    u[..., 0] = np.where(mask, 1.0, 0.0)
    gas = {"viscosity_Pa_s": 1.81e-5, "mean_free_path_m": 6.8e-8}
    opts = {"face_velocity_m_s": 0.05, "diameters_um": [0.01, 0.05, 0.3, 2.0, 12.0],
            "n_particles": 60, "direction": "x", "temperature_K": 298.15}
    res, paths = FI.analyse(u, mask, 1.0, gas, opts, K_m2=1e-12)
    eff = {r["diameter_um"]: r["efficiency"] for r in res["by_size"]}
    # a 12 um sphere cannot pass a 16 um pore without touching the wall
    expect(eff[12.0] > 0.9, f"interception of a 12 µm particle in a 16 µm pore: {eff[12.0]:.2f}")
    expect(eff[0.01] > eff[0.3], f"Brownian capture {eff[0.01]:.3f} should exceed the middle size {eff[0.3]:.3f}")
    expect(0.01 < res["mpps_um"] < 12.0, f"the most penetrating size {res['mpps_um']} is at an end of the range")
    expect(res["pressure_drop_Pa"] > 0 and paths, "no pressure drop or no trajectories were produced")
    return (f"efficiency 0.01 µm {100*eff[0.01]:.0f} %, 12 µm {100*eff[12.0]:.0f} %, "
            f"most penetrating size {res['mpps_um']:g} µm at {100*res['mpps_efficiency']:.0f} %")


@check("Optimisation: Sobol indices of the Ishigami function and the optimum of a known surface")
def t_optimise():
    from mpsim import optimise as OPT
    # Ishigami is the standard test of a Sobol implementation: its indices are
    # known in closed form, and the third input acts only through an
    # interaction, so S1 = 0 while ST > 0
    a, b = 7.0, 0.1
    pi = math.pi

    def ishigami(X):
        X = np.atleast_2d(X)
        return (np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2
                + b * X[:, 2] ** 4 * np.sin(X[:, 0]))
    V = a * a / 8 + b * pi ** 4 / 5 + b * b * pi ** 8 / 18 + 0.5
    V1 = 0.5 * (1 + b * pi ** 4 / 5) ** 2
    V2 = a * a / 8
    V13 = 8 * b * b * pi ** 8 / 225
    S1x = [V1 / V, V2 / V, 0.0]
    STx = [(V1 + V13) / V, V2 / V, V13 / V]
    bounds = [(-pi, pi)] * 3
    res = OPT.sobol({"predict": lambda X, std=False: ishigami(X)}, bounds, n=8192, seed=1)
    for i in range(3):
        expect(abs(res["s1"][i] - S1x[i]) < 0.05, f"S1[{i}] = {res['s1'][i]:.3f}, expected {S1x[i]:.3f}")
        expect(abs(res["total"][i] - STx[i]) < 0.05, f"ST[{i}] = {res['total'][i]:.3f}, expected {STx[i]:.3f}")

    # a surrogate fitted to a known bowl must find its peak
    g = np.linspace(-5, 5, 6)
    X = np.array([[x, y] for x in g for y in g])
    y = -((X[:, 0] - 2.0) ** 2 + (X[:, 1] + 1.0) ** 2)
    model = OPT.fit_surrogate(X, y, [(-5, 5), (-5, 5)], seed=0)
    opt = OPT.optimum(model, [(-5, 5), (-5, 5)], "max", seed=0)
    expect(abs(opt["x"][0] - 2.0) < 0.35 and abs(opt["x"][1] + 1.0) < 0.35,
           f"optimum at {opt['x']}, expected (2, -1)")
    cv = OPT.cross_validate(X, y, [(-5, 5), (-5, 5)], seed=0)
    expect(cv["r2"] > 0.95, f"cross-validated R² {cv['r2']:.3f}")
    return (f"Ishigami S1 {', '.join(f'{v:.3f}' for v in res['s1'])} (exact "
            f"{', '.join(f'{v:.3f}' for v in S1x)}); optimum ({opt['x'][0]:.2f}, {opt['x'][1]:.2f}) "
            f"of (2, -1), cross-validated R² {cv['r2']:.3f}")


@check("DOE: designs expand to the expected cases and apply to the form")
def t_doe():
    from mpsim import doe as DOE
    params = [{"path": "phases.0.fraction.value", "label": "content", "min": 10, "max": 50, "steps": 5},
              {"path": "phases.0.size.d", "label": "d", "values": [1, 5]}]
    full = DOE.expand(params, "full")
    oat = DOE.expand(params, "oat")
    lhs = DOE.expand(params, "lhs", samples=7, seed=3)
    expect(len(full) == 10, f"full factorial {len(full)}")
    expect(len(oat) == 1 + 4 + 1, f"one at a time {len(oat)}")
    expect(len(lhs) == 7, f"latin hypercube {len(lhs)}")
    vals = sorted({c["values"]["phases.0.fraction.value"] for c in full})
    expect(vals == [10, 20, 30, 40, 50], f"levels {vals}")
    base = {"name": "base", "phases": [{"fraction": {"value": 20}, "size": {"d": 3}}]}
    form = DOE.apply_case(base, full[-1])
    expect(form["phases"][0]["fraction"]["value"] == 50 and form["phases"][0]["size"]["d"] == 5, f"applied {form}")
    expect(base["phases"][0]["fraction"]["value"] == 20, "the base form was modified")
    return f"full {len(full)}, OAT {len(oat)}, LHS {len(lhs)} cases; last case '{full[-1]['label']}'"


@check("Anisotropic conduction: a turned tensor is returned exactly, a tilted laminate converges to the laminate "
       "formula, energy balance exact, an isotropic tensor = the plain solve, stacked platelets in series")
def t_cond_aniso():
    from mpsim.solvers import conduction as CD

    def rot(ax, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return (np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) if ax == "y"
                else np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    R = rot("z", 30) @ rot("y", 40)
    Kw = R @ np.diag([300.0, 300.0, 30.0]) @ R.T
    lab = np.ones((10, 10, 10), np.uint8)
    aid = np.zeros(lab.shape, np.int32)
    cols = np.column_stack([CD.solve_direction(lab, [1.0, 100.0], d, tol=1e-12, aniso={"aid": aid, "K": Kw[None]})["column"]
                            for d in range(3)])
    e_hom = float(np.abs(cols - Kw).max() / np.abs(Kw).max())
    expect(e_hom < 1e-10, f"homogeneous turned tensor: error {e_hom:.1e}")
    # laminate normal to z: half the turned tensor, half resin 0.2
    Kr = 0.2 * np.eye(3)
    kzz = 1.0 / (0.5 / Kw[2, 2] + 0.5 / Kr[2, 2])
    ref = np.zeros((3, 3))
    ref[2, 2] = kzz
    for a in range(2):
        ref[2, a] = ref[a, 2] = kzz * 0.5 * (Kw[a, 2] / Kw[2, 2] + Kr[a, 2] / Kr[2, 2])
        for b in range(2):
            ref[a, b] = (0.5 * (Kw[a, b] - Kw[a, 2] * Kw[2, b] / Kw[2, 2]) + 0.5 * (Kr[a, b] - Kr[a, 2] * Kr[2, b] / Kr[2, 2])
                         + kzz * 0.5 * (Kw[a, 2] / Kw[2, 2] + Kr[a, 2] / Kr[2, 2]) * 0.5 * (Kw[2, b] / Kw[2, 2] + Kr[2, b] / Kr[2, 2]))
    errs, gapmax = [], 0.0
    for t in (8, 16):
        lab = np.zeros((4, 4, 2 * t), np.uint8)
        lab[:, :, :t] = 1
        aid = np.full(lab.shape, -1, np.int32)
        aid[:, :, :t] = 0
        rs = [CD.solve_direction(lab, [0.2, 100.0], d, tol=1e-12, aniso={"aid": aid, "K": Kw[None]}) for d in range(3)]
        cols = np.column_stack([r_["column"] for r_ in rs])
        errs.append(float(np.abs(cols - ref).max() / np.abs(ref).max()))
        gapmax = max(gapmax, max(r_["hill_mandel_gap"] for r_ in rs))
        expect(abs(cols[0, 2] - cols[2, 0]) < 1e-9 * abs(ref).max(), "the effective tensor is not symmetric")
    expect(errs[1] < 0.035 and errs[1] < 0.6 * errs[0], f"laminate errors {errs} (first order expected)")
    expect(gapmax < 1e-9, f"energy balance gap {gapmax:.1e}")
    lab = _sphere_lab(24, 6.0)
    aid = np.where(lab > 0, 0, -1).astype(np.int32)
    rp = np.array([[0, 0.3], [0.3, 0]])
    plain = CD.solve_direction(lab, [0.2, 30.0], 0, tol=1e-10, rpair=rp)["k_eff"]
    iso = CD.solve_direction(lab, [0.2, 30.0], 0, tol=1e-10, rpair=rp, aniso={"aid": aid, "K": (30.0 * np.eye(3))[None]})["k_eff"]
    expect(abs(iso / plain - 1) < 1e-9, f"isotropic tensor {iso} against the plain solve {plain}")
    # stacked platelets: through a stack of n layers its n - 1 contacts are in
    # series with the layers, k_z = t / (t / k_z + (n - 1) R_c); in the plane
    # the layers conduct side by side
    from mpsim import pipeline as PL, spec as S
    import mpsim.generate as GEN_
    spec = S.normalize({"name": "stack", "matrix": {"material_id": "epoxy"},
                        "phases": [{"material_id": "hbn_platelet", "name": "BN", "shape": "cuboid",
                                    "size": {"d": 2.0, "ly": 0.3, "lz": 0.2, "unit": "um"}, "fraction": {"value": 10, "basis": "vol"},
                                    "stack_layers": 4, "r_contact": 1.2e-8}],
                        "rve": {"auto": False, "L_um": 4, "voxel_um": 0.25}, "analyses": {"thermal": True}})
    lab = np.zeros((4, 4, 4), np.uint8)
    lab[1:3, 1:3, 1:3] = 1
    ginfo = {"particles": {"codes": np.array([4]), "P": np.array([[1.0, 0.15, 0.1, 0, 0, 0]]), "R": np.eye(3)[None],
                           "phase": np.array([0]), "centers": np.array([[2.0, 2.0, 2.0]])}}
    real = GEN_.rasterize_owner
    GEN_.rasterize_owner = lambda *a, **k: ((lab > 0).astype(np.int32), None)
    try:
        an = PL.thermal_aniso(spec, lab, ginfo, None, 1.0)
    finally:
        GEN_.rasterize_owner = real
    kz = 0.2e-6 / (0.2e-6 / 30.0 + 3 * 1.2e-8)
    got = np.diag(an["K"][0])
    expect(np.allclose(got, [300.0, 300.0, kz], rtol=1e-12), f"stack tensor diagonal {got} (k_z {kz:.4f} expected)")
    # a film with a laminate matrix (0.81 / 0.81 / 0.29, FR-4) and copper rods
    # through it: k_z is the parallel mean of the copper and the matrix's k_z
    # exactly (films took the mean k of the matrix up to v5.2.2)
    labf = np.zeros((20, 20, 8), np.uint8)
    labf[4:8, 4:8, :] = 1
    f_cu = float(np.mean(labf == 1))
    anf = {"label_axes": {0: (0.81, 0.81, 0.29)}, "aid": None, "K": np.zeros((0, 3, 3))}
    kzf = CD.solve_direction(labf, [0.64, 401.0], 2, tol=1e-12, film=True, aniso=anf)["column"][2]
    kz_ref = f_cu * 401.0 + (1 - f_cu) * 0.29
    expect(abs(kzf / kz_ref - 1) < 1e-6, f"film k_z {kzf:.6g}, parallel {kz_ref:.6g}")
    return (f"turned tensor exact ({e_hom:.0e}); laminate error {100 * errs[0]:.1f} % at 8 cells, {100 * errs[1]:.1f} % at 16 "
            f"(first order); energy balance {gapmax:.0e}; isotropic tensor = plain solve; a 4-layer stack conducts "
            f"{kz:.2f} W/m·K through its thickness; a laminate film with rods = the parallel mean")


@check("DOE: a filler's share with the total kept; the Domain's grid on every case, or each case's own edge on its voxel size")
def t_doe_mix_grid():
    from mpsim import doe as DOE
    from web.doe import DOEStore
    base = {"name": "mix", "phases": [
        {"name": "A", "shape": "sphere", "props": {"E": 380}, "fraction": {"value": 41.25, "basis": "vol"}, "size": {"d": 0.25}},
        {"name": "B", "shape": "cuboid", "props": {"E": 100}, "fraction": {"value": 13.75, "basis": "vol"}, "size": {"d": 2}},
        {"name": "pores", "shape": "sphere", "props": {"E": 0}, "fraction": {"value": 2.0, "basis": "vol"}, "size": {"d": 1}}],
        "rve": {"auto": True}}
    cat = {c["path"]: c for c in DOE.parameter_catalogue(base)}
    expect("mix.1" in cat and abs(cat["mix.1"]["current"] - 25.0) < 1e-9, f"catalogue share {cat.get('mix.1')}")
    cases = DOE.expand([{"path": "mix.1", "label": "B share", "min": 5, "max": 50, "steps": 10}], "full")
    shares = []
    for c in cases:
        f = DOE.apply_case(base, c)
        a, b = f["phases"][0]["fraction"]["value"], f["phases"][1]["fraction"]["value"]
        expect(abs(a + b - 55.0) < 1e-9 and f["phases"][2]["fraction"]["value"] == 2.0, f"total {a + b}, pores changed")
        shares.append(round(100 * b / (a + b), 6))
    expect(shares == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0], f"shares {shares}")

    class _Job:
        def __init__(self, i):
            self.id = f"j{i}"

    class _Queue:
        def __init__(self):
            self.forms = []

        def submit(self, owner, form, runs_dir, meta, folder=None, case=""):
            self.forms.append(form)
            return _Job(len(self.forms))
    def plan(f):
        # a stand-in for the Domain's planner: a manual grid as entered; an
        # automatic one with an edge that grows with the BN share
        rv = f.get("rve") or {}
        if rv.get("auto") is False:
            return {"L_um": float(rv["L_um"]), "h_um": float(rv["voxel_um"]), "film": False}
        a_, b_ = f["phases"][0]["fraction"]["value"], f["phases"][1]["fraction"]["value"]
        return {"L_um": 3.0 + 0.04 * 100 * b_ / (a_ + b_), "h_um": 0.03, "film": False}
    with tempfile.TemporaryDirectory() as d:
        st, q = DOEStore(d), _Queue()
        man = dict(base, rve={"auto": False, "L_um": 4.0, "voxel_um": 0.025})
        study = st.create(None, {"form": man, "parameters": [{"path": "mix.1", "min": 5, "max": 50, "steps": 3}],
                                 "grid": {"edge": "fixed"}}, q, d, lambda f: f, plan=plan)
        rv = [f["rve"] for f in q.forms]
        expect(all(r_["auto"] is False and abs(r_["L_um"] - 4.0) < 1e-9 and r_["voxel_um"] == 0.025 for r_ in rv),
               f"fixed: the Domain's grid on every case {rv}")
        expect(abs(study["grid"]["L_um"] - 4.0) < 1e-9 and study["grid"]["edge"] == "fixed", f"grid {study['grid']}")
        # adaptive: each case its own edge, on the Domain's (automatic) voxel size
        q2 = _Queue()
        st.create(None, {"form": base, "parameters": [{"path": "mix.1", "min": 5, "max": 50, "steps": 2}],
                         "grid": {"edge": "adaptive"}}, q2, d, lambda f: f, plan=plan)
        Ls = [f["rve"]["L_um"] for f in q2.forms]
        hs = {f["rve"]["voxel_um"] for f in q2.forms}
        expect(hs == {0.03} and Ls[0] < Ls[-1] and all(abs(L / 0.03 - round(L / 0.03)) < 1e-9 and round(L / 0.03) % 2 == 0
                                                      for L in Ls), f"adaptive edges {Ls}, voxels {hs}")
        # the voxel size as a parameter: its values; otherwise the Domain's
        q3 = _Queue()
        st.create(None, {"form": man, "parameters": [{"path": "rve.voxel_um", "min": 0.02, "max": 0.04, "steps": 3}],
                         "grid": {"edge": "fixed"}}, q3, d, lambda f: f, plan=plan)
        expect([f["rve"]["voxel_um"] for f in q3.forms] == [0.02, 0.03, 0.04], f"voxel parameter {[f['rve'] for f in q3.forms]}")
    # what a finished case really holds: 51.1 of 55 vol%, a BN share of 48 of 50 %
    from web.doe import realised
    res = {"spec": {"labels": [{"kind": "matrix", "phase": None}, {"kind": "core", "phase": 0}, {"kind": "core", "phase": 1}],
                    "phases": [{"name": "Al2O3"}, {"name": "BN"}]},
           "composition": {"labels": [{"vf": 0.489}, {"vf": 0.266}, {"vf": 0.245}], "target": {"vf": [0.275, 0.275]}}}
    rz = realised({"base_form": base, "parameters": [{"path": "mix.1"}]}, res)
    expect(abs(rz["real_filler"][2] - 51.1) < 1e-9 and abs(rz["real_filler"][3] - 55.0) < 1e-9, f"filler {rz['real_filler']}")
    expect(abs(rz["real_mix.1"][2] - 100 * 0.245 / 0.511) < 1e-9 and abs(rz["real_mix.1"][3] - 50.0) < 1e-9, f"share {rz['real_mix.1']}")
    return (f"shares {shares[0]:g}-{shares[-1]:g} % of 55 vol% with the total kept; the Domain's grid on every case, "
            f"edges {Ls[0]:.3g}-{Ls[-1]:.3g} µm adapted on its voxel size; "
            f"realised 51.1 vol%, share {rz['real_mix.1'][2]:.1f} %")


@check("EMI complex homogenisation: complex laminates exact, real values = real solver, dilute spheres = Maxwell-Garnett")
def t_complex_cond():
    from mpsim.solvers import complex_cond as CC
    from mpsim.solvers import conduction as CD
    ka, kb = 1.0 + 2.0j, 30.0 - 0.5j
    ser, par = 2 / (1 / ka + 1 / kb), (ka + kb) / 2
    lab = np.zeros((16, 16, 12), np.uint8)
    lab[:, :, 6:] = 1
    worst = 0.0
    for bc, d, want in (("periodic", 0, par), ("periodic", 2, ser), ("film", 0, par), ("plates", 0, par), ("plates", 1, par)):
        k = CC.solve(lab, [ka, kb], d, bc=bc, tol=1e-11)["kappa"]
        worst = max(worst, abs(k / want - 1))
    expect(worst < 1e-9, f"laminates: worst relative error {worst:.1e}")
    rng = np.random.default_rng(1)
    lab3 = (rng.random((20, 20, 20)) < 0.3).astype(np.uint8)
    kr = CD.solve_direction(lab3, np.array([1.0, 20.0]), 0, tol=1e-11)["k_eff"]
    kc = CC.solve(lab3, [1.0 + 0j, 20.0 + 0j], 0, tol=1e-11)["kappa"]
    expect(abs(kc / kr - 1) < 1e-8, f"real values: {kc} vs {kr}")
    N, R = 48, 7.0
    x = np.arange(N) + 0.5 - N / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    lab4 = (X ** 2 + Y ** 2 + Z ** 2 <= R ** 2).astype(np.uint8)
    phi = lab4.mean()
    km, kp = 1.0 + 1.0j, 50.0 - 20.0j
    beta = (kp - km) / (kp + 2 * km)
    mg = km * (1 + 2 * phi * beta) / (1 - phi * beta)
    k = CC.solve(lab4, [km, kp], 0, tol=1e-9)["kappa"]
    expect(abs(k / mg - 1) < 0.005, f"Maxwell-Garnett: {k} vs {mg}")
    return (f"laminates within {worst:.0e}, real = real solver, dilute spheres {100*abs(k/mg-1):.2f} % "
            f"from Maxwell-Garnett (phi {phi:.3f})")


@check("EMI full-wave: openEMS homogeneous slab = exact slab transmission (skipped without openEMS)", quick=False)
def t_fullwave_slab():
    import tempfile
    from mpsim.solvers import emi as EMI
    from mpsim.solvers import fullwave as FW
    exe = FW.find_openems()
    if not exe:
        return "openEMS not installed - skipped"
    lab = np.zeros((10, 10, 10), np.uint8)
    table = [{"name": "slab", "props": {"sigma": 2000.0, "eps_r": 4.0, "mu_r": 1.0, "tan_d": 0.0}}]
    with tempfile.TemporaryDirectory() as d:
        r = FW.check(lab, table, 2.0, d, n_f=7, log=lambda *a: None)
    f = np.asarray(r["freq_hz"])
    _, l21 = EMI.slab_transmission(f, [(20e-6, 4.0, 1.0, 2000.0)])
    exact = -20 * l21
    dev = float(np.max(np.abs(np.asarray(r["se_fullwave_db"]) - exact)))
    expect(dev < 0.05, f"openEMS vs exact: {dev:.3f} dB")
    expect(r["max_diff_db"] < 0.05, f"openEMS vs homogenised: {r['max_diff_db']:.3f} dB")
    return f"20 µm slab, 10-100 GHz: SE {exact[0]:.2f} dB, openEMS within {dev:.3f} dB of exact ({r['seconds']:.0f} s)"


@check("EMI: scikit-rf transmission line = transfer-matrix solution, SE_R + SE_A + SE_M = SE")
def t_emi():
    from mpsim.pipeline import _emi_se
    sp, chk = _emi_se(1e3, 5.0, 1.0, {"thickness_mm": 1.0, "f_min_hz": 1e6, "f_max_hz": 1e10})
    expect(chk.get("available"), f"scikit-rf: {chk}")
    expect(chk["max_diff_db"] < 0.01, f"diff {chk['max_diff_db']}")
    tot = np.array(sp["se_r_db"]) + np.array(sp["se_a_db"]) + np.array(sp["se_m_db"])
    err = np.max(np.abs(tot - np.array(sp["se_db"])))
    expect(err < 1e-6, f"split error {err}")
    return f"scikit-rf difference {chk['max_diff_db']:.2e} dB, decomposition error {err:.1e} dB"


@check("Linked inputs: expressions are parsed safely; a link sets its entries in DOE cases and one unknown of a calibration")
def t_links():
    import copy
    from mpsim import calibrate as CAL
    from mpsim import doe as DOE
    from mpsim import links as LK
    from mpsim import spec as S
    expect(abs(LK.evaluate("sqrt(x1*x2) + 2^3 - pi*0", {"x": 1.0, "x1": 4.0, "x2": 9.0}) - 14.0) < 1e-12, "arithmetic")
    for bad in ("__import__('os')", "x.real", "[x]", "open(1)", "y+1", "1/0*x", "lambda: 1"):
        try:
            LK.evaluate(bad, {"x": 1.0})
            raise AssertionError(f"'{bad}' was evaluated")
        except LK.LinkError:
            pass
    # a blend of two grades of one alumina: one R_int for both
    form = {"name": "links", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3 10 um", "shape": "sphere", "size": {"d": 10, "unit": "um"},
                        "fraction": {"value": 25, "basis": "vol"}, "r_int": 2e-8},
                       {"material_id": "al2o3", "name": "Al2O3 3 um", "shape": "sphere", "size": {"d": 3, "unit": "um"},
                        "fraction": {"value": 15, "basis": "vol"}, "r_int": 2e-8}],
            "rve": {"auto": True, "quality": "fast"}, "analyses": {"thermal": True}}
    params = [{"path": "phases.0.r_int", "min": 1e-9, "max": 1e-7, "steps": 3, "scale": "log",
               "links": [{"path": "phases.1.r_int", "expr": "x"}, {"path": "contact_rc.0-1", "expr": "0.5*x"}]}]
    expect(not DOE.check_links(params, form), f"links rejected: {DOE.check_links(params, form)}")
    expect(DOE.check_links([dict(params[0], links=[{"path": "phases.0.r_int"}])], form), "a doubly set entry passed")
    cases = DOE.expand(params, "full")
    f = DOE.apply_case(form, cases[2], parameters=params)
    expect(f["phases"][1]["r_int"] == f["phases"][0]["r_int"] == 1e-7 and abs(f["contact_rc"]["0-1"] - 5e-8) < 1e-20,
           f"case values {f['phases'][0]['r_int']}, {f['phases'][1]['r_int']}, {f.get('contact_rc')}")
    # calibration (effective-medium preview): synthetic k at two contents, truth
    # R_int = 3e-8 for both grades; linked, one unknown recovers it
    f0 = copy.deepcopy(form)
    for ph in f0["phases"]:
        ph["r_int"] = 3e-8
    k1 = CAL._theory_k(S.normalize(copy.deepcopy(f0)))
    f0["phases"][0]["fraction"]["value"] = 35
    k2 = CAL._theory_k(S.normalize(f0))
    meas = [{"property": "k", "value": k1, "rel_unc": 0.03},
            {"property": "k", "value": k2, "rel_unc": 0.03, "set": {"phases.0.fraction.value": 35}}]
    unk = {"path": "phases.0.r_int", "lo": 1e-9, "hi": 1e-6, "scale": "log"}
    linked = CAL.preview({"base": form, "measurements": meas, "unknowns": [dict(unk, links=[{"path": "phases.1.r_int"}])]})
    alone = CAL.preview({"base": form, "measurements": meas, "unknowns": [unk]})
    m1, m0 = linked["unknowns"][0]["map"], alone["unknowns"][0]["map"]
    expect(abs(m1 / 3e-8 - 1) < 0.05, f"linked calibration {m1:.3g} (truth 3e-8)")
    expect(abs(m0 / 3e-8 - 1) > 0.2, f"unlinked calibration {m0:.3g} should be biased")
    return (f"unsafe expressions rejected; DOE case sets both grades and the pair; calibration linked {m1:.3g} "
            f"(truth 3e-8), the 10 µm grade alone {m0:.3g}")


@check("Web API: materials, presets and plans are JSON the browser can read")
def t_api():
    import json
    from web.server import create_app
    with tempfile.TemporaryDirectory() as d:
        app = create_app(workspace=d)
        c = app.test_client()
        mats = json.loads(c.get("/api/materials").data)
        pres = json.loads(c.get("/api/presets").data)
        expect(len(mats["materials"]) > 30, "materials")
        n_ok = 0
        for p in pres["presets"]:
            r = json.loads(c.post("/api/plan", json=p["form"]).data)
            expect(r["ok"], f"{p['id']}: {r.get('errors')}")
            n_ok += 1
        return f"{len(mats['materials'])} materials, plans of {n_ok} presets succeeded"


@check("Stop: a running analysis ends within seconds with every process it started, and stays stopped", quick=False)
def t_stop_now():
    import psutil
    from web.jobs import JobQueue, STOPPED_FILE
    # a run that is busy for a long while: thermal and viscosity of a dense
    # filled RVE (packing, three solves, particle dynamics)
    form = {"name": "selftest stop", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 6, "unit": "um"},
                        "fraction": {"value": 50, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 96, "voxel_um": 0.75, "auto_enlarge": False, "quality": "fast"},
            "analyses": {"thermal": True, "viscosity": True}, "options": {"directions": "xyz"}}
    with tempfile.TemporaryDirectory() as d:
        q = JobQueue(concurrency=1, threads=4)
        job = q.submit(None, form, d, {"name": "selftest stop", "analyses": ["thermal", "viscosity"]})
        # stopped while its three thermal solves run in their worker
        # processes - the part a stop used to wait for
        t0 = time.time()
        kids = []
        while time.time() - t0 < 300 and job.state in ("queued", "running"):
            if job.state == "running" and "Thermal" in (job.stage or ""):
                kids = [p.pid for p in psutil.Process(job.proc.pid).children(recursive=True)]
                if kids:
                    break
            time.sleep(0.2)
        expect(job.state == "running", f"the run did not reach its solves ({job.state})")
        time.sleep(1.0)
        pid = job.proc.pid
        kids = [p.pid for p in psutil.Process(pid).children(recursive=True)]
        stage = job.stage
        t1 = time.time()
        q.stop(job)
        while time.time() - t1 < 20 and job.state == "running":
            time.sleep(0.1)
        dt = time.time() - t1
        expect(job.state == "stopped", f"state {job.state} after the stop")
        expect(dt < 6.0, f"the stop took {dt:.1f} s")
        time.sleep(1.0)
        alive = [x for x in [pid] + kids if psutil.pid_exists(x) and psutil.Process(x).status() != psutil.STATUS_ZOMBIE]
        expect(not alive, f"processes still alive after the stop: {alive}")
        expect(os.path.exists(os.path.join(job.out_dir, STOPPED_FILE)), "no stop marker in the run directory")
        q2 = JobQueue(concurrency=1)
        q2.restore(d)
        back = q2.get(job.id)
        expect(back is not None and back.state == "stopped", f"after a restart the run is {back and back.state}")
        return f"stopped in {dt:.1f} s during '{stage or 'start-up'}', {len(kids)} child process(es) ended with it"


@check("Statistical RVE: realisations of a small box until the 95 % interval of the mean is within the target; "
       "the seed is a DOE variable with whole-number levels", quick=False)
def t_settle():
    import json
    from mpsim import doe as DOE
    from mpsim import pipeline as PL
    lv = DOE.expand([{"path": "rve.seed", "min": 1, "max": 4, "steps": 4}])
    expect([c["values"]["rve.seed"] for c in lv] == [1.0, 2.0, 3.0, 4.0], f"full: {lv}")
    lh = DOE.expand([{"path": "rve.seed", "min": 1, "max": 50}], design="lhs", samples=6)
    expect(all(float(c["values"]["rve.seed"]).is_integer() for c in lh), f"lhs: {lh}")
    form = {"name": "settle", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 4, "unit": "um"},
                        "fraction": {"value": 40, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 16, "voxel_um": 0.5, "quality": "fast",
                    "realisations": {"mode": "settle", "target_pct": 3.0, "n_min": 3, "n_max": 8}},
            "analyses": {"thermal": True}, "options": {"directions": "z"}}
    lines = []
    with tempfile.TemporaryDirectory() as d:
        PL.run(form, d, log=lines.append)
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
    n = len(r["realisations"])
    expect(3 <= n <= 8, f"{n} realisations")
    ck = {c["id"]: c for c in r["checks"]}
    expect("settle" in ck and "thermal_stat" in ck, "no statistics checks")
    rel = ck["thermal_stat"]["value"]
    expect(rel <= 0.03 or n == 8, f"stopped at {n} with a {100 * rel:.2f} % interval")
    seeds = [x["seed"] for x in r["realisations"]]
    expect(len(set(seeds)) == n, f"seeds {seeds}")
    k = r["properties"]["thermal"]["diag"][2]
    return (f"{n} realisations of a 16 µm box, k_z {k:.3f} ± {100 * rel:.2f} % (95 %); "
            f"{ck['thermal_stat']['note'].split(';')[1].strip()}")


@check("Progressive overlap in the random placement: log-normal irregular grains at 60 vol% whose placement jams "
       "short reach the loading in seconds (not minutes of clump relaxation); the voxel overlap check finds no overlap in "
       "a placement that reached it, and the overlap it allowed in one that did not", quick=False)
def t_rsa_overlap():
    from mpsim import generate as GEN
    from mpsim import pipeline as PL
    from mpsim import spec as S
    form = {"name": "g", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "sio2", "name": "grains", "shape": "polyhedron",
                        "size": {"d": 4, "unit": "um", "poly": "irregular", "variants": 0},
                        "dist": {"type": "lognormal", "std": 1.6}, "fraction": {"value": 60, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 30, "voxel_um": 0.5, "quality": "fast"}, "analyses": {"thermal": True},
            "options": {"directions": "z"}}
    spec = S.normalize(form)
    gp = PL._generator_phases(spec)
    out = {}
    for seed in (1, 2):
        lines = []
        t0 = time.time()
        lab, info = GEN.generate(60, 0.5, gp, seed=seed, log=lines.append, contacts="keep", gap_vox=0.0,
                                 overlap_steps=PL.overlap_steps(spec["options"]))
        dt = time.time() - t0
        oc = GEN.overlap_check(lab.shape, 0.5, info, gp)
        vf = float((lab == 1).mean())
        out[seed] = (dt, info.get("overlap_vox", 0), oc, vf, any("clump packing" in l for l in lines))
    d1, k1, oc1, v1, c1 = out[1]
    d2, k2, oc2, v2, c2 = out[2]
    expect(k1 == 0 and oc1["shared"] == 0 and abs(v1 - 0.6) < 0.002, f"seed 1: overlap {k1}, shared {oc1['shared']}, vf {v1}")
    expect(1 <= k2 <= 3 and not c2 and v2 > 0.595, f"seed 2: overlap {k2}, clump {c2}, vf {v2}")
    expect(oc2["shared"] > 0 and oc2["of_particles"] < 0.02, f"seed 2 shared {oc2['shared']} ({oc2['of_particles']})")
    expect(d2 < 60, f"seed 2 took {d2:.0f} s")
    return (f"seed 1 placed without overlap ({d1:.1f} s, {100 * v1:.2f} vol%); seed 2 jammed short and was finished with "
            f"up to {k2} voxel(s) of overlap in {d2:.1f} s ({100 * v2:.2f} vol%), {100 * oc2['of_particles']:.2f} % of its "
            f"particle voxels in two particles")


@check("Progressive overlap: spheres at 66 vol% (above random close packing) reach the loading with an overlap of "
       "at most the voxels allowed; off, they are drawn smaller and fall short; with a largest share of the particle "
       "voxels in two particles, a structure above it is built again with less overlap; Auto allows what changes "
       "the result by at most 1 % (contrast, particle size in voxels, loading)", quick=False)
def t_overlap_steps():
    from mpsim import generate as GEN
    ph = [{"shape": "sphere", "size_um": {"d": 4.0}, "dist": {"type": "mono"}, "vf": 0.66,
           "orientation": {"mode": "iso"}, "overlap": False, "gap_frac": 0.0, "name": "spheres"}]
    lab0, i0 = GEN.generate(60, 0.5, ph, seed=3, log=lambda *a: None, contacts="keep", gap_vox=0.0, overlap_steps=None)
    lab1, i1 = GEN.generate(60, 0.5, ph, seed=3, log=lambda *a: None, contacts="keep", gap_vox=0.0,
                            overlap_steps={"tries": 2, "max_vox": 3})
    v0, v1 = float(np.mean(lab0 == 1)), float(np.mean(lab1 == 1))
    expect(i0["pack_scale"] < 1.0 and i0["overlap_vox"] == 0 and v0 < 0.645, f"off: scale {i0['pack_scale']}, vf {v0}")
    expect(1 <= i1["overlap_vox"] <= 3 and abs(v1 / 0.66 - 1.0) < 0.01, f"on: overlap {i1['overlap_vox']}, vf {v1}")
    # the overlap is bounded: no two sphere centres closer than the sum of
    # the drawn radii less the overlap allowed
    p = i1["particles"]
    r = np.asarray(p["P"])[:, 0]
    i, j, d = GEN._sphere_pairs(np.asarray(p["centers"]), r, np.array([30.0, 30.0, 30.0]))
    depth = float(np.max(r[i] + r[j] - d)) if d.size else 0.0
    expect(depth <= i1["overlap_vox"] * 0.5 * (1 + 1e-6), f"deepest overlap {depth:.3f} µm")
    # a largest share: the 2-voxel structure shares more than 2.5 % of its
    # particle voxels, so it is built again with 1 voxel, which keeps within
    sh1 = 100 * GEN.overlap_check(lab1.shape, 0.5, i1, ph)["of_particles"]
    lab2, i2 = GEN.generate(60, 0.5, ph, seed=3, log=lambda *a: None, contacts="keep", gap_vox=0.0,
                            overlap_steps={"tries": 2, "max_vox": 3, "max_pct": 2.5})
    sh2 = 100 * GEN.overlap_check(lab2.shape, 0.5, i2, ph)["of_particles"]
    ol = i2.get("overlap_limit") or {}
    expect(sh1 > 2.5 and i2["overlap_vox"] < i1["overlap_vox"] and sh2 <= 2.5 and len(ol.get("builds") or []) >= 2,
           f"share limit 2.5 %: {sh1:.2f} % at {i1['overlap_vox']} voxels, then {sh2:.2f} % at {i2['overlap_vox']}, {ol}")
    # Auto: from the analysis, the contrast, the particle size in voxels and
    # the loading; a value as entered (0: no overlap); No limit
    from mpsim import pipeline as PL
    from mpsim import rve as R
    from mpsim import spec as S

    def share(mat, vf, an, o):
        sp = S.normalize({"name": "s", "matrix": {"material_id": "epoxy"}, "phases": [
            {"material_id": mat, "shape": "sphere", "size": {"d": 4, "unit": "um"}, "fraction": {"value": vf, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 30, "voxel_um": 0.5}, "analyses": {an: True}, "options": {"overlap_steps": o}})
        pl = R.plan(PL._plan_phases(sp), dict(sp["rve"], contacts="keep", elastic_method="fans"), {an}, {})
        PL.resolve_overlap_share(sp, pl)
        return (PL.overlap_steps(sp["options"]) or {}).get("max_pct")
    a_si, a_al = share("sio2", 60, "thermal", {}), share("al2o3", 60, "thermal", {})
    a_si50, a_die = share("sio2", 50, "thermal", {}), share("sio2", 60, "dielectric", {})
    expect(0.5 < a_si < 1.5 and a_al < 0.1 and a_si50 < 0.05 and a_die >= a_si,
           f"Auto: silica 60 vol% {a_si}, alumina {a_al}, silica 50 vol% {a_si50}, silica dielectric {a_die}")
    expect(share("sio2", 60, "thermal", {"share": "value", "max_pct": 0.3}) == 0.3
           and share("sio2", 60, "thermal", {"share": "value", "max_pct": 0}) == 0.0
           and share("sio2", 60, "thermal", {"share": "none"}) is None, "a value is not used as entered, or No limit limits")
    return (f"off: {100 * v0:.2f} vol% (size scale {i0['pack_scale']:.3f}); on: {100 * v1:.2f} vol% with up to "
            f"{i1['overlap_vox']} voxel(s) of overlap (deepest {depth / 0.5:.2f} voxel), {sh1:.2f} % of the particle "
            f"voxels shared; at most 2.5 %: built again with {i2['overlap_vox']} voxel(s), {sh2:.2f} %, "
            f"{100 * float(np.mean(lab2 == 1)):.2f} vol%; Auto (k, 8 voxels): silica 60 vol% {a_si:.2f} %, 50 vol% "
            f"{a_si50:.3f} %, alumina {a_al:.3f} %, silica permittivity {a_die:.2f} %")


@check("Runs: the Terminal sees the run's processes; a stopped run starts again in its own folder and uses the "
       "structure it had finished; a run folder copied elsewhere opens into the list; a run folder ZIP imports with its "
       "3D view", quick=False)
def t_restart_open():
    import io
    import json
    import shutil
    from web.server import create_app
    form = {"name": "selftest restart", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 4, "unit": "um"},
                        "fraction": {"value": 40, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 40, "voxel_um": 0.4, "auto_enlarge": False, "quality": "fast"},
            "analyses": {"thermal": True, "electrical": True, "dielectric": True}, "options": {"directions": "xyz"}}
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as elsewhere:
        app = create_app(workspace=d, concurrency=1)
        c = app.test_client()
        jid = json.loads(c.post("/api/jobs", json=form).data)["id"]
        job = app.queue.get(jid)
        t0 = time.time()
        # stopped once its structure is kept, while it solves
        while time.time() - t0 < 300 and job.state in ("queued", "running"):
            sd = os.path.join(job.out_dir, "structure")
            if os.path.isdir(sd) and any(n.endswith(".pkl") for n in os.listdir(sd)):
                break
            time.sleep(0.2)
        expect(job.state == "running", f"the run ended before it could be stopped ({job.state})")
        # the Terminal tab's view of the process: alive, its tree, its memory
        pr = json.loads(c.get(f"/api/jobs/{jid}/proc").data)
        expect(pr.get("alive") and pr.get("n_procs", 0) >= 1 and pr.get("rss_gb", 0) > 0 and pr.get("since_s") is not None,
               f"proc: {pr}")
        c.post(f"/api/jobs/{jid}/stop")
        while time.time() - t0 < 320 and job.state == "running":
            time.sleep(0.2)
        expect(job.state == "stopped", f"state {job.state} after the stop")
        r = json.loads(c.post(f"/api/jobs/{jid}/restart", json={}).data)
        expect(r.get("ok"), f"restart: {r}")
        new = app.queue.get(r["id"])
        expect(new.out_dir == job.out_dir and app.queue.get(jid) is None, "the restart is not in the run's place")
        while time.time() - t0 < 900 and new.state in ("queued", "running"):
            time.sleep(0.5)
        expect(new.state == "done", f"the restarted run is {new.state}")
        log = open(os.path.join(new.out_dir, "run.log"), encoding="utf-8").read()
        expect("Started again" in log and "generated earlier is used" in log.split("Started again")[1],
               "the restart generated the structure again")
        res = json.loads(c.get(f"/api/jobs/{new.id}/result").data)
        k = res["properties"]["thermal"]["diag"]
        # a copy of the folder elsewhere opens into the list with an id of its own
        cp = os.path.join(elsewhere, "copied_run")
        shutil.copytree(new.out_dir, cp)
        o = json.loads(c.post("/api/runs/open", json={"path": cp}).data)
        expect(o.get("ok") and o.get("ids") and o["ids"][0] != new.id, f"open: {o}")
        expect(app.queue.get(o["ids"][0]).state == "done", "the opened run is not done")
        listed = {row["id"] for row in json.loads(c.get("/api/jobs").data)}
        expect(o["ids"][0] in listed, "the opened run is not listed")
        app2 = create_app(workspace=d, concurrency=1)
        listed2 = {row["id"] for row in json.loads(app2.test_client().get("/api/jobs").data)}
        expect(o["ids"][0] in listed2, "the opened run is not listed after a restart of the server")
        # the run folder ZIP: everything, imported as a run of its own
        z = c.get(f"/api/jobs/{new.id}/download/run").data
        imp = json.loads(c.post("/api/runs/import", data={"file": (io.BytesIO(z), "run.zip")},
                                content_type="multipart/form-data").data)
        expect(imp.get("ok") and imp["state"] == "done" and imp["view"] and imp["id"] != new.id, f"import: {imp}")
        res2 = json.loads(c.get(f"/api/jobs/{imp['id']}/result").data)
        expect(res2["properties"]["thermal"]["diag"] == k, "the imported result differs")
        return (f"stopped while solving, started again with its structure ({new.ended - new.started:.0f} s), k "
                f"{k[0]:.3f}; the copy opened and kept after a server restart; ZIP of {len(z) / 1e6:.1f} MB imported")


@check("Compression of a periodic RVE: a resin alone gives Trouton's 3 eta (sides free) and 4 eta (plane strain) with "
       "their von Mises stress, and keeps its volume")
def t_compress_resin():
    from mpsim.solvers import compress as CP
    base = np.zeros((16, 16, 16), np.uint8)
    no = (lambda shape, c: (np.zeros(shape, np.int32), np.zeros(2, np.int64)))
    out = []
    for lat, s_zz, s_l, vm in (("xy", -3.0, (0.0, 0.0), 3.0), ("x", -4.0, (0.0, -2.0), 2.0 * math.sqrt(3.0))):
        cp = CP.Compress(base, 1.0, [2.0], [False], np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64), no,
                         lateral=lat, compression=0.2, rate=0.5, steps=2, frames=False)
        o = cp.run()
        s = o["series"]
        eta, rate = 2.0, 0.5
        expect(abs(s["sigma_zz_Pa"][-1] / (s_zz * eta * rate) - 1) < 1e-6, f"{lat}: σzz {s['sigma_zz_Pa'][-1]:.6g}")
        expect(abs(s["sigma_xx_Pa"][-1] - s_l[0] * eta * rate) < 1e-6 and abs(s["sigma_yy_Pa"][-1] - s_l[1] * eta * rate) < 1e-6,
               f"{lat}: lateral stresses {s['sigma_xx_Pa'][-1]:.4g}, {s['sigma_yy_Pa'][-1]:.4g}")
        expect(abs(float(o["von_mises"].mean()) / (vm * eta * rate) - 1) < 1e-5, f"{lat}: von Mises {o['von_mises'].mean():.6g}")
        r0 = o["regions"][0]
        expect(abs(r0["volume_um3"][-1] / r0["volume0_um3"] - 1) < 2e-3, f"{lat}: resin volume {r0['volume_um3'][-1]:.1f} of {r0['volume0_um3']:.1f}")
        expect(abs(s["strain_z"][-1] - math.log(0.8)) < 1e-9, f"{lat}: final strain {s['strain_z'][-1]}")
        out.append(f"{lat}: σzz {s['sigma_zz_Pa'][-1] / (eta * rate):.4f} η ε̇, von Mises {o['von_mises'].mean() / (eta * rate):.4f}")
    return "; ".join(out) + " (exact 3, 3 and 4, 2√3)"


@check("Compression with both sides held: the pores close by the volume lost, the resin keeps its own (markers)",
       quick=False)
def t_compress_pores():
    from mpsim.solvers import compress as CP
    N = 24
    base = np.zeros((N, N, N), np.uint8)
    idx = np.indices(base.shape).reshape(3, -1).T + 0.5
    d = idx - N / 2.0
    R = (0.3 * N ** 3 * 3 / (4 * np.pi)) ** (1 / 3)
    base.reshape(-1)[np.sum(d * d, axis=1) <= R * R] = 1
    no = (lambda shape, c: (np.zeros(shape, np.int32), np.zeros(2, np.int64)))
    cp = CP.Compress(base, 1.0, [1.0, None], [False, True], np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64), no,
                     lateral="none", compression=0.2, rate=1.0, steps=8, frames=False)
    o = cp.run()
    p0, p1 = o["pore_fraction"]
    r0 = o["regions"][0]
    expected = (p0 - 0.2) / 0.8
    expect(abs(p1 - expected) < 0.01, f"pores {100 * p0:.2f} -> {100 * p1:.2f} vol%, expected {100 * expected:.2f}")
    expect(abs(r0["volume_um3"][-1] / r0["volume0_um3"] - 1) < 0.005,
           f"resin volume {r0['volume_um3'][-1]:.1f} of {r0['volume0_um3']:.1f}")
    s = o["series"]
    expect(s["sigma_zz_Pa"][-1] < s["sigma_zz_Pa"][0] < 0, "the axial stress does not rise as the pores close")
    return (f"pores {100 * p0:.1f} -> {100 * p1:.2f} vol% (expected {100 * expected:.2f}), resin volume "
            f"{100 * (r0['volume_um3'][-1] / r0['volume0_um3'] - 1):+.2f} %, axial stress {-s['sigma_zz_Pa'][0]:.2f} -> "
            f"{-s['sigma_zz_Pa'][-1]:.2f} Pa")


@check("Arrangement after compression: resin A | resin B with silica compressed 30 % in 3D, sides free; each resin keeps "
       "its volume, the filled one widens less, the box is solved as it is, the 3D record written; every step's whole "
       "box kept for the 3D step player and brought along when the structure is reused", quick=False)
def t_pressed():
    import json
    from mpsim import pipeline as PL
    f = {"name": "pressed", "matrix": {"material_id": "epoxy", "name": "Resin A"},
         "matrix_regions": [{"name": "Resin B", "material_id": "epoxy", "mu_pas": 2.0,
                             "section": {"mode": "section", "axis": "x", "from": 50, "to": 100}}],
         "phases": [{"material_id": "sio2", "name": "Silica", "shape": "sphere", "size": {"d": 5, "unit": "um"},
                     "fraction": {"value": 40, "basis": "vol"}, "section": {"mode": "section", "axis": "x", "from": 50, "to": 100}}],
         "rve": {"auto": False, "L_um": 25, "voxel_um": 0.625, "quality": "fast", "arrangement": "pressed",
                 "press_compression": 30, "press_rate": 1.0, "press_lateral": "xy"},
         "analyses": {"thermal": True}, "options": {"directions": "z"}}
    import gzip
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as d2:
        PL.run(f, d, log=lambda *a: None)
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        with open(os.path.join(d, "view", "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        files = os.listdir(os.path.join(d, "view"))
        st3 = (meta.get("press") or {}).get("steps3d") or []
        expect(st3 and len(st3) == len(meta["press"]["frames"]), f"{len(st3)} steps in 3D for {len(meta['press']['frames'])} steps")
        for st in (st3[0], st3[-1]):
            with gzip.open(os.path.join(d, "view", st["lab"])) as fh:
                lab = np.frombuffer(fh.read(), np.uint8)
            with gzip.open(os.path.join(d, "view", st["vm"])) as fh:
                vmq = np.frombuffer(fh.read(), np.uint8)
            expect(lab.size == vmq.size == int(np.prod(st["dims"])), f"step volume {lab.size} for {st['dims']}")
            expect(vmq.max() > 200 and (vmq == 0).any(), f"stress codes {vmq.min()}..{vmq.max()}")
        expect(st3[-1]["dims"] == r["realisations"][0]["pressed"]["shape"], "the last step is not the compressed box")
        # the same structure in another folder: no new compression, its steps come along
        f2 = dict(f, _reuse={"from": [d]})
        logs = []
        PL.run(f2, d2, log=logs.append)
        with open(os.path.join(d2, "view", "meta.json"), encoding="utf-8") as fh:
            meta2 = json.load(fh)
        expect(any("generated earlier is used" in x for x in logs), "the structure was not reused")
        expect(len((meta2.get("press") or {}).get("steps3d") or []) == len(st3)
               and os.path.exists(os.path.join(d2, "view", st3[-1]["lab"])), "the steps did not come with the reused structure")
    pr = r["realisations"][0].get("pressed") or {}
    expect(pr, "no compression record")
    shp = pr["shape"]
    expect(abs(shp[2] / 40 - 0.7) < 0.03 and shp[0] > 40 and shp[1] > 40, f"box {shp} voxels from 40³")
    expect(r["plan"].get("shape") == shp, f"the analyses ran on {r['plan'].get('shape')}, not the compressed box {shp}")
    A, B = pr["regions"]["Resin A"], pr["regions"]["Resin B"]
    expect(abs(A["share"] - A["share0"]) < 0.01 and abs(B["share"] - B["share0"]) < 0.01,
           f"volume shares A {A['share0']:.3f} -> {A['share']:.3f}, B {B['share0']:.3f} -> {B['share']:.3f}")
    expect(A["strain_all"][0] > B["strain_all"][0], f"x strain A {A['strain_all'][0]:.3f}, B {B['strain_all'][0]:.3f}")
    expect(B["strain"][2] < B["strain_all"][2], "the resin between the rigid spheres does not strain more than its region")
    kz = r["properties"]["thermal"]["diag"][2]
    expect(0.2 < kz < 1.5, f"k_z {kz}")
    expect("press" in meta and "press_lab.u8" in files and "field_press_vm.f32" in files, "the 3D record was not written")
    return (f"box 40³ -> {' × '.join(str(v) for v in shp)}; strain x of A {A['strain_all'][0]:+.3f}, of B with its "
            f"spheres {B['strain_all'][0]:+.3f}; resin B alone z {B['strain'][2]:+.3f} (region {B['strain_all'][2]:+.3f}); "
            f"{pr['crossed_resin']} of {pr['n_spheres']} spheres in another resin; k_z {kz:.3f} W/m·K; {pr['seconds']:.0f} s; "
            f"{len(st3)} steps kept in 3D")


@check("Arrangement after shear flow: contents kept, periodic, close pairs tilted by the flow; spheres only", quick=False)
def t_sheared():
    import json
    import numpy as np
    from mpsim import generate as GEN
    from mpsim import pipeline as PL
    from mpsim import rve as R
    from mpsim import spec as S
    form = {"name": "sheared", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "sio2", "name": "Silica 4 um", "shape": "sphere", "size": {"d": 4, "unit": "um"},
                        "fraction": {"value": 32, "basis": "vol"}},
                       {"material_id": "sio2", "name": "Silica 1.6 um", "shape": "sphere", "size": {"d": 1.6, "unit": "um"},
                        "fraction": {"value": 13, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 16, "voxel_um": 0.4, "auto_enlarge": False, "quality": "fast",
                    "arrangement": "sheared", "shear_strain": 2},
            "analyses": {"thermal": True}, "options": {"viscosity": {"dem": {"backend": "cpu"}}}}
    for bad, word in (({"shape": "spheroid", "size": {"d": 4, "aspect": 0.3, "unit": "um"}}, "spheres"), (None, "thin film")):
        g = json.loads(json.dumps(form))
        if bad:
            g["phases"][0].update(bad)
        else:
            g["rve"].update(film=True, T_um=12)
        try:
            S.normalize(g)
            raise AssertionError(f"a sheared arrangement of {word} passed")
        except ValueError as e:
            expect(word in str(e), str(e))
    spec = S.normalize(form)
    need = S.required_analyses(set(spec["analyses"]))
    plan = R.plan(PL._plan_phases(spec), dict(spec["rve"], contacts=spec["options"]["contacts"]), need, {})
    gp = PL._generator_phases(spec)
    k0 = PL.structure_key(gp, plan, "apart", 1, False)
    k1 = PL.structure_key(gp, plan, "apart", 1, False, sheared=PL.shear_key(spec))
    expect(k0 != k1, "the sheared structure has the random packing's key")
    lab, info = GEN.generate(plan["N"], plan["h_um"], gp, seed=1, log=lambda *a: None, contacts="apart",
                             shear=PL.shear_packing(spec, lambda *a: None, lambda *a, **k: None,
                                                    PL.Progress(lambda *a, **k: None, 1.0), None))
    sh = info["sheared"]
    vf = [float(np.mean(lab == i + 1)) for i in range(2)]
    expect(all(abs(v / ph["vf"] - 1.0) < 0.01 for v, ph in zip(vf, spec["phases"])), f"contents {vf}")
    expect(sh["strain"] == 2.0, f"ended at strain {sh['strain']}")
    expect(sh["separation_um"] < 0.25 * plan["h_um"], f"pushed {sh['separation_um']:.3g} µm apart after the shear")
    expect(sh["moved_um"] > 0.2, f"the flow moved the spheres {sh['moved_um']:.3g} µm across it")
    parts = info["particles"]
    rad = np.asarray(parts["P"])[:, 0]
    cr = GEN.contact_ratio(np.asarray(parts["centers"]), rad, plan["N"] * plan["h_um"])
    expect(cr >= 1.0 - 1e-9, f"spheres overlap across the periodic faces (ratio {cr:.4f})")
    fr, fs = sh["fabric"]["random"][0][2], sh["fabric"]["sheared"][0][2]
    n_r, _, n_s = sh["fabric"]["pairs_per_sphere"]
    expect(fs < fr and n_s > n_r, f"close pairs: tilt {fr:+.3f} -> {fs:+.3f}, {n_r:.2f} -> {n_s:.2f} per sphere")
    return (f"{len(rad)} spheres sheared to 2 in {sh['seconds']:.0f} s: moved {sh['moved_um']:.2f} µm across the flow, "
            f"pushed {sh['separation_um']:.3f} µm after; close pairs {n_r:.2f} -> {n_s:.2f} per sphere, tilt "
            f"{fr:+.3f} -> {fs:+.3f}; contents {100 * vf[0]:.2f} / {100 * vf[1]:.2f} %")


@check("Complete pipeline: small filled RVE → result, layer card, viewer data, report", quick=False)
def t_pipeline():
    from mpsim import pipeline
    form = {"name": "selftest", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 8, "unit": "um"},
                        "fraction": {"value": 25, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 32, "voxel_um": 1.0, "auto_enlarge": False, "quality": "fast"},
            "analyses": {"thermal": True, "dielectric": True, "grains": True}, "options": {"directions": "x"}}
    with tempfile.TemporaryDirectory() as d:
        s = pipeline.run(form, d, log=lambda *a: None)
        for f in ("result.json", "layer_card.json", "report.html", "structure.tif", "view/meta.json"):
            expect(os.path.exists(os.path.join(d, f)), f"missing {f}")
        k = next(h["value"] for h in s["headline"] if h["key"] == "k")
        expect(0.2 < k < 30, f"k {k}")
        return f"k = {k:.4f} W/m·K, checks {s['check_counts']}, {s['elapsed_s']:.0f} s"


@check("Complete pipeline: viscosity, live figures and the provisional result after each analysis", quick=False)
def t_pipeline_live():
    import json
    from mpsim import pipeline
    form = {"name": "selftest live", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 8, "unit": "um"},
                        "fraction": {"value": 20, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 32, "voxel_um": 1.0, "auto_enlarge": False, "quality": "fast"},
            "analyses": {"thermal": True, "viscosity": True},
            # particles kept apart: touching ones would make the particle
            # dynamics the answer and skip the RVE flow solve checked here
            "options": {"directions": "x", "live_figures": True, "particle_gap_vox": 2,
                        "viscosity": {"phi_m_mode": "manual", "phi_m": 0.64, "dem": {"n": 100, "strain": 1.5}}}}
    seen = {"figure": [], "partial_result": 0}
    lines = []

    def event(kind, **p):
        if kind == "figure":
            seen["figure"].append(p.get("name") or p.get("file"))
        elif kind == "partial_result":
            seen["partial_result"] += 1
    with tempfile.TemporaryDirectory() as d:
        s = pipeline.run(form, d, log=lambda *a: lines.append(" ".join(str(x) for x in a)), event=event)
        expect(not os.path.exists(os.path.join(d, "result.partial.json")), "result.partial.json left behind")
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        # the 3D view: the resin's shear rate is listed and read on the
        # particle surfaces from the resin side (inside a filler it is zero)
        with open(os.path.join(d, "view", "meta.json"), encoding="utf-8") as fh:
            vm = json.load(fh)
        vg = next((x for x in vm["fields"] if x["key"] == "visc_gd"), None)
        expect(vg is not None and vg.get("in_matrix"), "visc_gd missing from the view or not a matrix field")
        import numpy as np
        med = [float(np.median(np.fromfile(os.path.join(d, "view", sf["fields"]["visc_gd"]), "<f4")))
               for sf in vm["surfaces"] if "visc_gd" in sf["fields"]]
        expect(med and min(med) > 0.3, f"shear rate on the particle surfaces {med} (read inside the fillers?)")
    v = r["properties"]["viscosity"]
    expect(seen["partial_result"] >= 2, f"{seen['partial_result']} provisional results")
    expect(len(seen["figure"]) >= 2, f"figures {seen['figure']}")
    expect(v["mu_r_rve"] >= 0.99 * v["refs"]["Hashin-Shtrikman lower bound"], f"mu_r {v['mu_r_rve']}")
    expect(v["mu_compound_ref"] > v["mu_resin_ref"], "compound not more viscous than the resin")
    # the particle dynamics ran at further shear rates (alumina attracts):
    # the relative viscosity does not rise with the rate
    rc = (v.get("dem") or {}).get("rate_curve")
    expect(rc and len(rc["gd"]) >= 4, f"particle-dynamics flow curve {rc} ({(v.get('dem') or {}).get('rate_note')})")
    expect(rc["mu_r"][0] >= 0.9 * rc["mu_r"][-1], f"μr {rc['mu_r']} at {rc['gd']} 1/s")
    # on the CPU the further rates run side by side with the reference run,
    # in worker processes that end with it
    import multiprocessing as mp
    import time as _t
    from mpsim.solvers import suspension as SU
    side = any("side by side" in ln for ln in lines)
    if SU.cuda_device_count() == 0:
        expect(side, "the flow-curve runs did not run side by side on the CPU")
    t_end = _t.time() + 10.0
    while mp.active_children() and _t.time() < t_end:
        _t.sleep(0.2)
    expect(not mp.active_children(), f"worker processes left running: {mp.active_children()}")
    return (f"μr {v['mu_r_rve']:.3f} (HS {v['refs']['Hashin-Shtrikman lower bound']:.3f}, KD {v['refs']['Krieger-Dougherty']:.3f}), "
            f"{seen['partial_result']} provisional results, {len(seen['figure'])} live figures, "
            f"surface shear rate {min(med):.2f}; particle dynamics μr {rc['mu_r'][0]:.3g} → {rc['mu_r'][-1]:.3g} "
            f"from {rc['gd'][0]:g} to {rc['gd'][-1]:g} 1/s{' (side by side)' if side else ''}; {s['elapsed_s']:.0f} s")


@check("Complete pipeline: dense spheres - the particle dynamics gives the viscosity without a voxel RVE (default) "
       "or with the RVE flow solved at the same time (option)", quick=False)
def t_pipeline_dense_visc():
    import json
    from mpsim import pipeline

    def form(both):
        return {"name": "selftest dense", "matrix": {"material_id": "epoxy"},
                "phases": [{"material_id": "sio2", "name": "Silica", "shape": "sphere", "size": {"d": 6, "unit": "um"},
                            "dist": {"type": "lognormal", "cv": 0.15}, "fraction": {"value": 45, "basis": "vol"}}],
                "rve": {"auto": False, "L_um": 24, "voxel_um": 1.0, "auto_enlarge": False, "quality": "fast"},
                "analyses": {"viscosity": True},
                "options": {"viscosity": {"rve_with_dem": both, "dem": {"n": 100, "strain": 1.5, "rates": 0}}}}
    out = {}
    old = os.environ.get("MPSIM_DEM_CONCURRENT")
    try:
        for both in (False, True):
            if both:
                os.environ["MPSIM_DEM_CONCURRENT"] = "1"      # the process path also without a GPU
            with tempfile.TemporaryDirectory() as d:
                pipeline.run(form(both), d, log=lambda *a: None)
                with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
                    res = json.load(fh)
                v = res["properties"]["viscosity"]
                with open(os.path.join(d, "view", "meta.json"), encoding="utf-8") as fh:
                    vm = json.load(fh)
                expect(os.path.exists(os.path.join(d, "report.html")), "no report")
                no_struct = not os.path.exists(os.path.join(d, "structure.tif"))
            keys = [x["key"] for x in vm.get("fields") or []]
            expect(res.get("dem_direct") == (not both) and no_struct == (not both),
                   f"voxel RVE built: {not no_struct}, direct: {res.get('dem_direct')} (option {both})")
            expect(v["basis"] == "particle dynamics", f"best estimate from {v['basis']}")
            expect(bool(vm.get("dem")), "no particle animation")
            if both:
                expect(v["mu_r_rve"] is not None and "visc_gd" in keys, "the RVE flow solve did not run with the option")
            else:
                expect(v.get("rve_skipped") and v["mu_r_rve"] is None and "visc_gd" not in keys,
                       f"the RVE flow solve was not skipped ({v.get('mu_r_rve')})")
            out[both] = v
    finally:
        if old is None:
            os.environ.pop("MPSIM_DEM_CONCURRENT", None)
        else:
            os.environ["MPSIM_DEM_CONCURRENT"] = old
    expect(abs(out[True]["mu_r"] / out[False]["mu_r"] - 1) < 1e-9,
           f"the particle dynamics in its own process {out[True]['mu_r']} against here {out[False]['mu_r']}")
    expect(not pipeline._LIVE_DEM, "a particle-dynamics process left running")
    return (f"45 vol%: particle dynamics μr {out[False]['mu_r']:.4g} without a voxel RVE; with the option the same "
            f"{out[True]['mu_r']:.4g} from its own process, beside the RVE value {out[True]['mu_r_rve']:.4g}")


@check("Complete pipeline: thin film (x, y periodic, z the real thickness), z solved first", quick=False)
def t_pipeline_film():
    import json
    from mpsim import pipeline
    form = {"name": "selftest film", "matrix": {"material_id": "epoxy"},
            "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 5, "unit": "um"},
                        "fraction": {"value": 30, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 32, "voxel_um": 1.0, "auto_enlarge": False, "quality": "fast",
                    "film": True, "T_um": 14},
            "analyses": {"thermal": True}, "options": {"directions": "xyz"}}
    with tempfile.TemporaryDirectory() as d:
        pipeline.run(form, d, log=lambda *a: None)
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        with open(os.path.join(d, "view", "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    th = r["properties"]["thermal"]
    expect(list(meta["full_shape"]) == [32, 32, 14], f"grid {meta['full_shape']}")
    expect([s["direction"] for s in th["stats"]][0] == "z", "z is not solved first")
    expect(all(s["method"] == "fv-film" for s in th["stats"]), f"methods {[s['method'] for s in th['stats']]}")
    expect(abs(th["derived"]["thickness_mm"] - 0.014) < 1e-9, f"thickness {th['derived']['thickness_mm']}")
    kx, ky, kz = th["diag"]
    expect(0.2 < kz < min(kx, ky) * 1.05, f"k {kx:.4f} {ky:.4f} {kz:.4f}")
    return f"32 × 32 × 14: kxx {kx:.4f}, kyy {ky:.4f}, kzz {kz:.4f} W/m·K"


@check("Complete pipeline: porous RVE with flow, diffusion, acoustics, radiation, porosimetry and pore network", quick=False)
def t_pipeline_porous():
    import json
    from mpsim import pipeline
    form = {"name": "selftest porous", "matrix": {"material_id": "al2o3"},
            "phases": [{"material_id": "air", "name": "Pores", "shape": "network", "size": {"d": 4, "unit": "um"},
                        "fraction": {"value": 40, "basis": "vol"}}],
            "rve": {"auto": False, "L_um": 40, "voxel_um": 1.0, "auto_enlarge": False, "quality": "fast"},
            "analyses": {"permeability": True, "tortuosity": True, "acoustics": True, "radiation": True,
                         "porosimetry": True, "pore_network": True, "morphology": True},
            "options": {"directions": "x"}}
    with tempfile.TemporaryDirectory() as d:
        s = pipeline.run(form, d, log=lambda *a: None)
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
        P = r["properties"]
        for key in ("permeability", "tortuosity", "acoustics", "radiation", "porosimetry", "pore_network", "morphology"):
            expect(key in P and "error" not in P[key], f"{key}: {P.get(key, {}).get('error') if isinstance(P.get(key), dict) else 'missing'}")
        a = P["acoustics"]
        expect(a["viscous_length_um"] <= a["thermal_length_um"] * 1.05, f"Λ {a['viscous_length_um']} > Λ' {a['thermal_length_um']}")
        expect(os.path.exists(os.path.join(d, "view", "network.json")), "network view file missing")
        return (f"K {P['permeability']['diag'][0]:.3g} m², τ {P['tortuosity']['mean_tortuosity']:.3g}, "
                f"Λ/Λ' {a['viscous_length_um']:.3g}/{a['thermal_length_um']:.3g} µm, NRC {a.get('nrc')}, "
                f"{P['pore_network']['n_pores']} pores, {s['elapsed_s']:.0f} s")


@check("Size distributions: mean and standard deviation, a measured table gives back its number distribution, bounds")
def t_psd():
    from mpsim import geometry as G
    rng = np.random.default_rng(3)
    # a number distribution given by channel: the drawn sizes must fall in the
    # channels in the proportions given
    d = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    v = np.array([10.0, 30.0, 40.0, 20.0, 0.0])
    edges, pn, _ = G.psd_bins(d, v, "differential", "number")
    mean = G.table_moments(edges, pn)
    dist = {"type": "table", "edges": list(edges / mean), "p": list(pn)}
    sz = G.sample_scales(200000, dist, rng) * mean
    got = np.histogram(sz, bins=edges)[0] / len(sz)
    expect(np.max(np.abs(got - pn / pn.sum())) < 0.01, f"channel shares {np.round(got, 3)} against {np.round(pn / pn.sum(), 3)}")
    out = []
    for kind in ("lognormal", "normal"):
        dd = {"type": kind, "cv": 0.4, "lo": 0.7, "hi": 1.5}
        s_ = G.sample_scales(100000, dd, rng)
        expect(s_.min() >= 0.7 - 1e-12 and s_.max() <= 1.5 + 1e-12, f"{kind}: {s_.min():.4f}..{s_.max():.4f} outside the bounds")
        lo, hi = G.scale_bounds(dd)
        expect(abs(lo - 0.7) < 1e-9 and abs(hi - 1.5) < 1e-9, f"{kind}: bounds {lo}, {hi}")
        out.append(f"{kind} {s_.min():.3f}..{s_.max():.3f}")
    # mean and standard deviation entered (in the phase's unit) or, from a form
    # of v5.3.0, the coefficient of variation: the same distribution
    from mpsim import spec as S
    for kind in ("normal", "lognormal"):
        a1, _ = S._size_distribution({"type": kind, "std": 3.0}, {"d": 12.0}, "um", "test")
        a2, _ = S._size_distribution({"type": kind, "cv": 0.25}, {"d": 12.0}, "um", "test")
        a3, _ = S._size_distribution({"type": kind, "std": 3000.0}, {"d": 12.0}, "nm", "test")
        expect(abs(a1["cv"] - 0.25) < 1e-12 and a1 == a2 and abs(a3["cv"] - 0.25) < 1e-12, f"{kind}: {a1} {a2} {a3}")
    try:
        S._size_distribution({"type": "normal", "std": 13.0}, {"d": 12.0}, "um", "test")
        expect(False, "a normal standard deviation above the mean was accepted")
    except ValueError:
        pass
    # a normal distribution without bounds: mean 1, cut at +-2 sigma (std x 0.88)
    s_ = G.sample_scales(200000, {"type": "normal", "cv": 0.2}, rng)
    expect(abs(s_.mean() - 1) < 0.003 and abs(s_.std() / 0.2 - 0.8796) < 0.01, f"normal mean {s_.mean():.4f} std {s_.std():.4f}")
    return f"channel shares within 1 %; bounded draws {', '.join(out)}; normal ±2σ std ratio {s_.std() / 0.2:.4f} (0.8796)"


@check("Uneven dispersion: a region holds every particle, a gradient and clusters reach the loading asked", quick=False)
def t_dispersion():
    from mpsim import generate as GEN
    ph = [dict(name="s", shape="sphere", size_um={"d": 8.0}, dist={"type": "mono"}, vf=0.25,
               orientation={"mode": "iso"}, overlap=False, gap_frac=0.0)]
    lab, info = GEN.generate(60, 1.0, ph, seed=2, log=lambda *a: None,
                             dispersion={"mode": "region", "axis": "y", "from": 0.2, "to": 0.7})
    d = info["dispersion"]
    c = info["particles"]["centers"][:, 1]
    expect(d["share_in_region"] > 0.999, f"{d['share_in_region']} of the particles in the region")
    expect(c.min() >= 12.0 + 4.0 - 1e-6 and c.max() <= 42.0 - 4.0 + 1e-6, f"centres along y {c.min()}..{c.max()}")
    lab, info = GEN.generate(60, 1.0, ph, seed=2, log=lambda *a: None,
                             dispersion={"mode": "gradient", "axis": "z", "ratio": 3.0})
    r = info["dispersion"]["ratio"]
    expect(2.0 < r < 4.0, f"gradient ratio {r:.3f} (asked 3)")
    lab, info = GEN.generate(60, 1.0, ph, seed=2, log=lambda *a: None,
                             dispersion={"mode": "clusters", "n_clusters": 2, "cluster_vol": 0.3, "share": 0.5})
    sh = info["dispersion"]["share_realised"]
    expect(0.4 < sh < 0.6, f"cluster share {sh:.3f} (asked 0.5)")
    return f"region exact, gradient {r:.2f} (asked 3), clusters {100 * sh:.1f} % (asked 50)"


@check("Mechanical test: modulus, Poisson's ratio and forces against closed forms; the fields of a laminate")
def t_mechtest():
    from mpsim import mechtest as MT
    from mpsim.solvers import fans as FA
    from mpsim.solvers.elastic import lame
    E, nu = 3.0, 0.35
    lam, mu = lame(np.array([E]), np.array([nu]))
    C = FA.iso_C(float(lam[0]), float(mu[0]))
    st = MT.macro_state(C, {"axis": "x", "mode": "tension", "lateral": "free", "control": "force", "value": 50.0,
                            "area_mm2": 5.0, "length_mm": 20.0})
    expect(abs(st["modulus_GPa"] - E) < 1e-9 and abs(st["poisson"]["xy"] - nu) < 1e-9, f"E {st['modulus_GPa']}, ν {st['poisson']}")
    expect(abs(st["stress_MPa"] - 10.0) < 1e-9 and abs(st["displacement_um"] - 10.0 / 3000.0 * 20e3) < 1e-6,
           f"{st['stress_MPa']} MPa, {st['displacement_um']} µm")
    cf = MT.macro_state(C, {"axis": "z", "mode": "compression", "lateral": "confined", "control": "strain", "value": 1.0,
                            "area_mm2": 1.0, "length_mm": 1.0})
    m_c = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    expect(abs(cf["modulus_GPa"] - m_c) < 1e-9 and abs(cf["lateral_stress_MPa"]["x"] - (-0.01 * lam[0] * 1e3)) < 1e-6,
           f"constrained modulus {cf['modulus_GPa']} vs {m_c}, lateral {cf['lateral_stress_MPa']}")
    # a laminate loaded across its layers under uniaxial stress: the stress is
    # the applied one in both layers (series)
    lab = np.zeros((8, 8, 16), np.int32)
    lab[:, :, 8:] = 1
    Es, nus = [3.0, 70.0], [0.35, 0.22]
    C2 = FA.homogenize(lab, Es, nus, [0.0, 0.0], tol=1e-8, loads=["xx", "yy", "zz", "yz", "xz", "xy"])["C"]
    st2 = MT.macro_state(C2, {"axis": "z", "mode": "compression", "lateral": "free", "control": "pressure",
                              "value": 10.0, "area_mm2": 1.0, "length_mm": 1.0})
    sol = FA.FANS(lab, Es, nus)
    _, info, s = sol.solve(np.asarray(st2["E_voigt"]), 0.0, 1e-9, 2000, keep_stress=True)
    szz = s[2] * 1e3
    expect(np.max(np.abs(szz + 10.0)) < 1e-3, f"σzz {szz.min():.5f}..{szz.max():.5f} MPa, not −10 everywhere")
    return (f"E, ν exact; constrained modulus {cf['modulus_GPa']:.4f} GPa; laminate σzz −10 MPa in both layers "
            f"({info['iterations']} iterations)")


@check("Irregular grains: variants 0 gives every grain a shape of its own", quick=False)
def t_unique_grains():
    from mpsim import generate as GEN
    ph = [dict(name="g", shape="polyhedron", size_um={"d": 10.0, "poly": "irregular", "n_vertices": 12, "variants": 0},
               dist={"type": "mono"}, vf=0.15, orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)]
    lab, info = GEN.generate(48, 1.0, ph, seed=3, log=lambda *a: None)
    ids = np.asarray(info["particles"]["P"])[:, 1].astype(int)
    expect(len(set(ids.tolist())) == len(ids), f"{len(ids)} grains share {len(set(ids.tolist()))} shapes")
    return f"{len(ids)} grains, {len(set(ids.tolist()))} distinct shapes"


@check("Gap between particles: 0 lets them touch (never overlap), N voxels keeps their surfaces N voxels apart")
def t_particle_gap():
    from mpsim import spec as S, generate as GEN
    # the default touches; an input of 5.3.1 with contacts = "apart" keeps its 2 voxels
    expect(S._contacts({}) == {"contacts": "keep", "particle_gap_vox": 0.0}, f"default {S._contacts({})}")
    expect(S._contacts({"contacts": "apart"})["particle_gap_vox"] == 2.0, "legacy 'apart' is not 2 voxels")
    expect(S._contacts({"particle_gap_vox": 3})["contacts"] == "apart", "a gap of 3 is not 'apart'")
    N, h = 48, 0.25
    L = N * h
    ph = [dict(name="s", shape="sphere", size_um={"d": 3.0}, dist=None, vf=0.30,
               orientation={"mode": "iso", "spread_deg": 0}, overlap=False, gap_frac=0.0)]
    out = []
    for contacts, g in (("keep", 0.0), ("apart", 2.0), ("apart", 3.0)):
        lab, info = GEN.generate(N, h, ph, seed=2, log=lambda *a: None, contacts=contacts, gap_vox=g)
        C = np.asarray(info["particles"]["centers"])
        r = np.asarray(info["particles"]["P"])[:, 0]
        gmin = np.inf
        for i in range(len(C) - 1):
            d = C[i + 1:] - C[i]
            d -= L * np.round(d / L)
            gmin = min(gmin, float((np.linalg.norm(d, axis=1) - r[i + 1:] - r[i]).min()))
        gmin /= h
        expect(abs(float((lab > 0).mean()) - 0.30) < 0.01, f"gap {g}: fraction {float((lab > 0).mean()):.4f}")
        expect(gmin > -1e-6, f"gap {g}: particles overlap ({gmin:.3f} voxels)")
        if g > 0:
            expect(gmin >= g - 0.05, f"gap {g}: surfaces {gmin:.2f} voxels apart")
            expect(info["touching_pairs"] == 0, f"gap {g}: {info['touching_pairs']} touching voxel pairs")
        else:
            expect(info["contact_faces"] > 0, "gap 0: no particle touches another")
        out.append(f"gap {g:g}: closest surfaces {gmin:.2f} voxels, {info['contact_faces']} contact faces")
    return "; ".join(out)


@check("Built-in FV: PuMA's finite-volume operator to rounding, periodic instead of fixed end slices")
def t_fv_vs_puma():
    import contextlib
    import io as _io
    from mpsim.solvers import conduction as CD
    PB = _puma()
    pm = PB.require()
    from pumapy.physics_models.finite_volume.isotropic_conductivity import IsotropicConductivity
    rng = np.random.default_rng(4)
    lab = rng.integers(0, 3, size=(14, 11, 9)).astype(np.uint8)        # unequal sides
    k = np.array([0.2, 30.0, 3.0])
    out = []
    for d in "xyz":
        cmap = pm.IsotropicConductivityMap()
        for i, v in enumerate(k):
            cmap.add_material((i, i), float(v))
        with contextlib.redirect_stdout(_io.StringIO()):
            s_ = IsotropicConductivity(PB._workspace(lab, 1e-6), cmap, d, "p", None, 1e-6, 10, "cg", False, True)
            s_.error_check()
            s_.initialize()
            s_.assemble_bvector()
            s_.assemble_Amatrix()
        # PuMA's solve frame has the direction first
        perm = {"x": (0, 1, 2), "y": (1, 0, 2), "z": (2, 1, 0)}[d]
        lf = np.ascontiguousarray(np.transpose(lab, perm))
        x = rng.standard_normal(lf.shape)
        yp = s_.Amat.matvec(x.flatten("F")).reshape(lf.shape, order="F")
        kf = [np.empty(lf.shape) for _ in range(3)]
        z = np.zeros((3, 3))
        for ax in range(3):
            CD._faces(lf.astype(np.int32), k, z, np.zeros((1, 1, 1), np.uint8), z, False, ax, kf[ax])
        yo = np.empty_like(x)
        CD._apply(np.ascontiguousarray(x), kf[0], kf[1], kf[2], yo)
        inner = (slice(1, lf.shape[0] - 1), slice(None), slice(None))
        rel = float(np.abs(-2.0 * yp[inner] - yo[inner]).max() / np.abs(yo[inner]).max())
        expect(rel < 1e-13, f"{d}: the built-in FV operator differs from PuMA's by {rel:.1e}")
        out.append(f"{d} {rel:.0e}")
    # the boundary condition: PuMA's fixed slices move with the cut, the periodic FV does not
    lab2 = (rng.random((24, 24, 24)) < 0.4).astype(np.uint8)
    kk = np.array([0.2, 30.0])
    vp, vo = [], []
    for sh in (0, 12):
        l2 = np.roll(lab2, sh, axis=0)
        vp.append(PB.conductivity(l2, kk, "x", 1e-6, tol=1e-9, method="fv", keep_fields=False)["k_eff"])
        vo.append(CD.solve_direction(l2, kk, 0, tol=1e-10)["k_eff"])
    expect(abs(vo[1] - vo[0]) / vo[0] < 1e-8, f"periodic FV moved with a shift: {vo}")
    return (f"operator ×(−2) vs PuMA FV, interior cells: {', '.join(out)}; a half-cell shift: PuMA FV "
            f"{vp[0]:.4f} → {vp[1]:.4f}, built-in FV {vo[0]:.6f} → {vo[1]:.6f}")


@check("Multi-core tortuosity and Stokes flow: PuMA's single-threaded results, any thread count")
def t_porous_parallel():
    import numba
    PB = _puma()
    rng = np.random.default_rng(5)
    # connected pores: a smoothed random field, 45 % void
    from scipy import ndimage
    f = ndimage.gaussian_filter(rng.standard_normal((20, 22, 18)), 2.0, mode="wrap")
    lab = (f > np.quantile(f, 0.45)).astype(np.uint8)              # 0 pore, 1 solid
    res = {}
    for on in (False, True):
        PB.set_fe_parallel(on)
        try:
            t = PB.tortuosity(lab, [0], 1e-6, "y", tol=1e-8)
            p = PB.permeability(lab, [0], 1e-6, "z", tol=1e-8)
        finally:
            PB.set_fe_parallel(True)
        res[on] = (t["tortuosity"], np.asarray(t["C"]), float(p["K"][2, 2]), np.asarray(p["u"][2]), p["iterations"])
    (t0, C0, K0, u0, i0), (t1, C1, K1, u1, i1) = res[False], res[True]
    dt, dC = abs(t1 - t0) / t0, float(np.abs(C1 - C0).max())
    dK, du = abs(K1 - K0) / K0, float(np.abs(u1 - u0).max() / np.abs(u0).max())
    expect(dt < 1e-10 and dC < 1e-10, f"tortuosity {t0} vs {t1} (field {dC:.1e})")
    expect(dK < 1e-7 and du < 1e-6, f"permeability {K0:.6e} vs {K1:.6e} (field {du:.1e})")
    # the threaded sums are blocked: one thread or all give the same numbers
    n0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        t_1 = PB.tortuosity(lab, [0], 1e-6, "y", tol=1e-8)["tortuosity"]
    finally:
        numba.set_num_threads(n0)
    expect(t_1 == t1, f"tortuosity on 1 thread {t_1!r} and on {n0} {t1!r}")
    return (f"τ {t1:.6f} (PuMA {dt:.0e}, field {dC:.0e}); K {K1:.4e} m² (PuMA {dK:.0e}, field {du:.0e}, "
            f"{i1}/{i0} MINRES iterations); the same τ on 1 and {n0} threads")


@check("Threaded CG: the same conduction and EMI results on one thread and on all")
def t_thread_invariance():
    import numba
    from mpsim.solvers import conduction as CD, complex_cond as CC
    rng = np.random.default_rng(6)
    lab = (rng.random((28, 26, 24)) < 0.35).astype(np.uint8)
    kap = np.array([1e-12 + 2.2e-4j, 1e2 + 5.6e-5j])
    n0 = numba.get_num_threads()
    out = {}
    try:
        for n in (1, n0):
            numba.set_num_threads(n)
            out[n] = (CD.solve_direction(lab, np.array([0.2, 30.0]), 2, tol=1e-9, workers=n)["k_eff"],
                      CD.solve_direction(lab, np.array([0.2, 30.0]), 2, tol=1e-9, workers=n, film=True)["k_eff"],
                      CC.solve(lab, kap, 0, tol=1e-9, workers=n)["kappa"])
    finally:
        numba.set_num_threads(n0)
    expect(out[1] == out[n0], f"1 thread {out[1]} against {n0} threads {out[n0]}")
    return f"k {out[1][0]:.8f}, film {out[1][1]:.8f}, κ {out[1][2]:.6e}: identical on 1 and {n0} threads"


@check("Structure generation on all cores: random placement, overlap repair and rasterisation give the serial "
       "loops' voxels and sums to the last bit")
def t_parallel_structure():
    from mpsim import generate as GEN, shapes as SH
    tab = SH.PolyTable()
    tab.add("octahedron")
    planes, nfaces = tab.arrays()
    h, N = 0.3, 48
    L = N * h
    rng = np.random.default_rng(21)
    n = 260
    codes = rng.choice(np.array([0, 1, 4, 6]), n).astype(np.int64)
    Ps = np.zeros((n, 6))
    Ps[:, 0] = rng.uniform(0.8, 2.0, n)
    Ps[:, 1] = np.where(codes == 1, 0.5 * Ps[:, 0], 0.0)
    Ps[codes == 4, 1] = Ps[codes == 4, 0]
    Ps[codes == 4, 2] = Ps[codes == 4, 0]
    Ps[codes == 6, 1] = 0.0
    Rs = np.ascontiguousarray(SH.sample_rotations(n, "iso", 0.0, rng))
    ph = rng.integers(0, 2, n).astype(np.int64)
    ov = np.array([1, 0], np.int32)
    notes = []
    # overlap repair: one round on overlapping particles, a film and a box
    C = rng.random((n, 3)) * L
    for zwall in (False, True):
        res = []
        for fn in (GEN._raster_track, GEN.raster_track_par):
            own = np.zeros((N, N, N - 6 if zwall else N), np.int32)
            pof = np.full(n + 8, -1, np.int32)
            arr = [np.zeros((n, 3)), np.zeros(n, np.int64), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, np.int64)]
            size = Ps[:, 0] ** 3
            fac = np.where(np.arange(n) % 5 == 0, 0.93, 1.0)
            tot = fn(own, pof, ov, h, codes, C, Rs, Ps, planes, nfaces, ph, 2, arr[0], arr[1], arr[2], arr[3], size,
                     fac, GEN.SHRINK_FLOOR, arr[4], np.empty(100000, np.int64), zwall)
            res.append((int(tot), own, *arr))
        same = res[0][0] == res[1][0] and all(np.array_equal(a.view(np.uint8), b.view(np.uint8))
                                              for a, b in zip(res[0][1:], res[1][1:]))
        expect(same, f"overlap repair (film {zwall}): the parallel sums differ from the serial loop's")
        notes.append(f"repair {res[0][0]:,} shared voxels")
    # rasterisation onto a grid that already holds voxels
    res = []
    for fn in (GEN.raster_all_serial, GEN.raster_all_par):
        own = np.zeros((N, N, N), np.int32)
        own[:N // 4] = 1
        pof = np.full(n + 8, -1, np.int32)
        pof[1] = 1
        r = fn(own, pof, ov, np.zeros(2, np.int32), h, codes, C, Rs, Ps, planes, nfaces, np.zeros(n), ph, 2,
               np.empty(100000, np.int64), False)
        res.append((r, own))
    expect(res[0][0] == res[1][0] and np.array_equal(res[0][1], res[1][1]),
           f"rasterisation: {res[0][0]} against {res[1][0]}")
    notes.append(f"rasterisation {res[0][0][1]:,} conflicts")
    # random sequential addition with a gap and the voxel-neighbour rule
    res = []
    for fn in (GEN._rsa_serial, GEN._rsa):
        own = np.zeros((N, N, N), np.int32)
        pof = np.full(n + 8, -1, np.int32)
        cnt, npl = np.zeros(2), np.zeros(2, np.int64)
        jam, cen, pid_ = np.zeros(2, np.bool_), np.zeros((n, 3)), np.zeros(n, np.int64)
        last = fn(own, pof, np.zeros(2, np.int32), np.zeros(2, np.int32), np.array([1, 0], np.int32), h, L, codes, Rs,
                  Ps, planes, nfaces, np.full(n, 0.15), ph, np.array([0.2, 0.15]) * N ** 3, cnt, npl, jam, cen, pid_,
                  1, 400, 25, 77, np.empty(100000, np.int64), False, np.ones((1, 1, 1)), 0.0)
        res.append((last, own, cen, cnt))
    expect(res[0][0] == res[1][0] and np.array_equal(res[0][1], res[1][1]) and np.array_equal(res[0][2], res[1][2])
           and np.array_equal(res[0][3], res[1][3]), f"placement: {res[0][0] - 1} against {res[1][0] - 1} particles")
    import numba
    return (f"{', '.join(notes)}, placement of {res[0][0] - 1} particles: identical to the serial loops on "
            f"{numba.get_num_threads()} threads")


@check("Sections: a second matrix and a filler in one half; the slabs, the contents and the layered conduction")
def t_sections():
    from mpsim import spec as S, pipeline as PL, generate as GEN
    form = {"name": "sections", "matrix": {"material_id": "epoxy", "name": "A"},
            "matrix_regions": [{"material_id": "polyimide", "name": "B", "section": {"axis": "z", "from": 50, "to": 100}}],
            "phases": [{"material_id": "al2o3", "name": "filler", "shape": "sphere", "size": {"d": 3, "unit": "um"},
                        "fraction": {"value": 30, "basis": "vol"}, "section": {"mode": "section", "axis": "z", "from": 50, "to": 100}}],
            "rve": {"auto": False, "L_um": 24, "voxel_um": 0.4, "quality": "fast"}, "analyses": {"thermal": True}}
    sp = S.normalize(form)
    c = sp["composition"]
    expect(abs(c["vf"][0] - 0.15) < 1e-12 and abs(c["matrix_regions"][0]["vf"] - 0.35) < 1e-12
           and abs(c["vf_matrix"] - 0.5) < 1e-12, f"composition {c}")
    gp = PL._generator_phases(sp)
    N, h = 60, 0.4
    lab, info = GEN.generate(N, h, gp, seed=3, log=lambda *a: None, contacts="keep")
    lower, upper = lab[:, :, :N // 2], lab[:, :, N // 2:]
    expect(not lower.any(), f"{int(lower.astype(bool).sum())} filler voxels in the lower half")
    vf_up = float((upper > 0).mean())
    expect(abs(vf_up - 0.30) < 0.02, f"filler in its half {vf_up:.4f}")
    labels, _, _ = PL.build_labels(sp, lab, info, h)
    mreg = len(sp["labels"]) - 1
    expect(set(np.unique(labels[:, :, :N // 2]).tolist()) == {0}, "the lower half is not all matrix A")
    up = labels[:, :, N // 2:]
    expect(set(np.unique(up).tolist()) == {1, mreg}, f"upper half labels {np.unique(up)}")
    # layers in series along z, side by side along x: k_z below k_x
    from mpsim.solvers import conduction as CD
    vals = np.array([float(t["props"]["k"]) for t in sp["labels"]])
    kx = CD.solve_direction(labels, vals, 0, tol=1e-8)["k_eff"]
    kz = CD.solve_direction(labels, vals, 2, tol=1e-8)["k_eff"]
    expect(kz < kx, f"k_z {kz} not below k_x {kx}")
    for bad, why in (({"phases": form["phases"] + [{"material_id": "aln", "shape": "sphere", "size": {"d": 2, "unit": "um"},
                                                     "fraction": {"value": 5, "basis": "vol"}}]}, "mixed"),
                     ({"matrix_regions": form["matrix_regions"] + [{"material_id": "silicone", "section": {"axis": "z", "from": 70, "to": 90}}]}, "overlap")):
        try:
            S.normalize({**form, **bad})
            raise AssertionError(f"{why}: not refused")
        except ValueError:
            pass
    return (f"filler 30 % of its half ({vf_up:.4f} drawn), matrix A 50 %, B 35 %, filler 15 %; k_x {kx:.4f}, "
            f"k_z {kz:.4f} W/m·K; a filler over the whole RVE beside it and overlapping regions are refused")


# ------------------------------------------------------- coupled analyses (5.6)
@check("Thermal stress and warpage: a free film with a linear temperature bends stress-free (κ = α ΔT / t); a film on "
       "a substrate bends as Timoshenko's bimetal; a held film cooled through Tg carries E/(1-ν) [α1 (Tg - T) + "
       "α2 E_r/E (T_ref - Tg)]")
def t_thermomech_film():
    from mpsim.solvers import thermomech as TM
    mat = {"E": 3.0, "nu": 0.35, "alpha": 60.0, "k": 0.2}
    table = [{"name": "resin", "props": mat}]
    nz, n, h_um = 20, 6, 1.0
    lab = np.zeros((n, n, nz), np.uint8)
    t = nz * h_um * 1e-6
    o = TM.run(lab, table, h_um, {"T_ref": 75.0, "T_bottom": 100.0, "T_top": 50.0, "support": "free", "span_mm": 10},
               [0.2], tol=1e-9, compare_uniform=True)
    k_exact = 60e-6 * (50.0 - 100.0) / t
    expect(abs(o["kappa_1_m"] / k_exact - 1) < 2e-3, f"free film κ {o['kappa_1_m']:.5g}, exact {k_exact:.5g}")
    expect(o["stress_stats"][0]["vm_p999"] < 0.01, f"free film stress {o['stress_stats'][0]['vm_p999']:.3g} MPa")
    expect(abs(o["uniform"]["kappa_1_m"]) < 1e-6, "the uniform film bends")
    sub = {"E": 160.0, "nu": 0.22, "alpha": 2.6, "thickness_um": 40.0}
    o2 = TM.run(lab, table, h_um, {"T_ref": 175.0, "T_bottom": 25.0, "T_top": 25.0, "support": "substrate",
                                   "substrate": sub, "span_mm": 10}, [0.2], tol=1e-9, compare_uniform=False)
    E1, h1, a1 = 3e9 / (1 - 0.35), t, 60e-6
    E2, h2, a2 = 160e9 / (1 - 0.22), 40e-6, 2.6e-6
    dT = 25.0 - 175.0
    num = 6 * E1 * E2 * (h1 + h2) * h1 * h2 * (a1 - a2) * dT
    den = (E1 ** 2 * h1 ** 4 + 4 * E1 * E2 * h1 ** 3 * h2 + 6 * E1 * E2 * h1 ** 2 * h2 ** 2 + 4 * E1 * E2 * h1 * h2 ** 3
           + E2 ** 2 * h2 ** 4)
    # Timoshenko's curvature, negative here: the film on top shrinks more and
    # the edges rise (the solver's sign: κ < 0 lifts the edges)
    kt = num / den
    expect(abs(o2["kappa_1_m"] / kt - 1) < 2e-3, f"bimetal κ {o2['kappa_1_m']:.5g}, Timoshenko {kt:.5g}")
    tg = {"E": 3.0, "nu": 0.35, "alpha": 60.0, "tg": 150.0, "E_r": 0.03, "alpha2": 180.0, "k": 0.2}
    o3 = TM.run(np.zeros((8, 8, 8), np.uint8), [{"name": "epoxy", "props": tg}], 1.0,
                {"T_ref": 175.0, "T_bottom": 25.0, "T_top": 25.0, "support": "held", "span_mm": 10}, [0.2],
                compare_uniform=False)
    s_held = float(np.mean(o3["profile"]["s_MPa"]))
    s_ex = 3e9 / 0.65 * (60e-6 * 125.0 + 180e-6 * 0.01 * 25.0) * 1e-6
    expect(abs(s_held / s_ex - 1) < 1e-4, f"held film through Tg {s_held:.5f} MPa, exact {s_ex:.5f}")
    return (f"free κ {o['kappa_1_m']:.5g} (exact {k_exact:.5g}), bimetal κ {o2['kappa_1_m']:.5g} (Timoshenko {kt:.5g}), "
            f"held through Tg {s_held:.4f} MPa (exact {s_ex:.4f})")


@check("Joule heating: a uniform conductor at a set current against the 1D parabola, with a TCR against the 1D "
       "finite-volume solve; two layers in series; the Joule heat leaves through the faces")
def t_joule_film():
    from mpsim.solvers import joule as JO
    n, nz, h_um = 4, 20, 1.0
    h = h_um * 1e-6
    lab = np.zeros((n, n, nz), np.uint8)

    def ref_1d(sig0, tcr, J, Tsink):
        T = np.full(nz, Tsink)
        A = 2.0 * np.eye(nz) - np.eye(nz, k=1) - np.eye(nz, k=-1)
        A[0, 0] += 1.0
        A[-1, -1] += 1.0
        for _ in range(200):
            q = J * J / (sig0 / (1 + tcr * (T - 20.0)))
            b = q * h * h / 1.0
            b[0] += 2 * Tsink
            b[-1] += 2 * Tsink
            Tn = np.linalg.solve(A, b)
            done = np.abs(Tn - T).max() < 1e-9
            T = Tn
            if done:
                break
        return T
    worst, out = 0.0, []
    for tcr in (0.0, 0.004):
        table = [{"name": "conductor", "props": {"sigma": 1e5, "k": 1.0, "tcr": tcr}}]
        o = JO.run(lab, table, h_um, {"control": "current", "values": [1e8, 4e8], "T_amb": 25.0, "R_th": 0.0},
                   [1e5], [1.0], keep_fields=False)
        for st in o["steps"]:
            Tr = ref_1d(1e5, tcr, st["J_A_m2"], 25.0)
            worst = max(worst, abs(st["T_max"] - Tr.max()))
            expect(st["balance"] < 1e-6, f"energy balance {st['balance']:.2e}")
            if tcr == 0.0:
                par = 25 + (st["J_A_m2"] ** 2 / 1e5) * (nz * h) ** 2 / 8
                expect(abs(st["T_max"] - par) < 1e-6, f"T max {st['T_max']:.6f}, parabola {par:.6f}")
        out.append(f"TCR {tcr}: T max {o['steps'][-1]['T_max']:.4f} °C at 4e8 A/m²")
    expect(worst < 1e-3, f"1D difference {worst:.2e} K")
    lab2 = np.zeros((n, n, nz), np.uint8)
    lab2[:, :, nz // 2:] = 1
    table2 = [{"name": "a", "props": {"sigma": 1e5, "k": 1.0}}, {"name": "b", "props": {"sigma": 2e4, "k": 1.0}}]
    o2 = JO.run(lab2, table2, h_um, {"control": "voltage", "values": [1e-3], "T_amb": 25.0, "R_th": 0.0},
                [1e5, 2e4], [1.0, 1.0], keep_fields=False)
    R_ex = (nz // 2) * h / 1e5 + (nz - nz // 2) * h / 2e4
    expect(abs(o2["steps"][0]["R_ohm_m2"] / R_ex - 1) < 1e-6, f"series R {o2['steps'][0]['R_ohm_m2']:.6g}, exact {R_ex:.6g}")
    return "; ".join(out) + f"; largest difference to the 1D solve {worst:.1e} K; series R exact to 1e-6"


@check("Cure: n-th order kinetics against the closed form; a resin pressed at a constant stress creeps by "
       "∫ P / 3η dt until it gels (Castro-Macosko)")
def t_cure_kinetics():
    from mpsim.solvers import compress as CP
    R = CP.R_GAS
    Tm, k1, E1 = 175.0, 0.02, 60e3
    A1 = k1 * math.exp(E1 / (R * (Tm + 273.15)))
    kin = {0: {"E1": E1, "A1": A1, "E2": 50e3, "A2": 0.0, "m": 1.0, "n": 1.5, "ag": 0.6, "A": 0.5, "B": 3.0, "Ea": 0.0}}
    for t in (10.0, 60.0):
        c = CP.Cure(kin, {0: 2.0}, Tm, {0: 0.0}, 2e6, 0.0)
        c.advance(t)
        ex = 1.0 - (1.0 + 0.5 * k1 * t) ** (1.0 / (1.0 - 1.5))
        expect(abs(c.alpha(0) - ex) < 1e-6, f"α({t:g} s) {c.alpha(0):.7f}, exact {ex:.7f}")
    P = 3.0 * 2.0 * 0.05
    cure = CP.Cure(kin, {0: 2.0}, Tm, {0: 0.0}, 2e6, 0.0)
    no = (lambda shape, cc: (np.zeros(shape, np.int32), np.zeros(2, np.int64)))
    cp = CP.Compress(np.zeros((12, 12, 12), np.uint8), 1.0, [2.0], [False], np.zeros((0, 3)), np.zeros(0),
                     np.zeros(0, np.int64), no, lateral="xy", compression=0.6, rate=10.0, frames=False, cure=cure,
                     stress_limit=P, t_max=200.0)
    o = cp.run()
    s = o["series"]
    t_end = s["time_s"][-1]

    def eta(t):
        a = 1.0 - (1.0 + 0.5 * k1 * t) ** (1.0 / (1.0 - 1.5))
        return 2.0 * (0.6 / (0.6 - a)) ** (0.5 + 3.0 * a)
    tt = np.linspace(0.0, t_end, 100001)
    eps_ex = float(np.trapz(P / (3.0 * np.array([eta(x) for x in tt])), tt))
    expect(o["stop"] == "gelled", f"stopped: {o['stop']}")
    expect(abs(-s["strain_z"][-1] / eps_ex - 1) < 1e-3, f"creep strain {-s['strain_z'][-1]:.5f}, exact {eps_ex:.5f}")
    return (f"α exact to 1e-6; creep {-s['strain_z'][-1]:.5f} (exact {eps_ex:.5f}), gelled at {t_end:.1f} s "
            f"in {o['steps']} steps")


@check("Film percolation: a conductor that joins the two faces of a film spans it (not a periodic wrap); the "
       "contrast limit keeps it conducting through the film")
def t_film_spans():
    from mpsim.solvers import conduction as CD
    lab = np.zeros((12, 12, 16), np.uint8)
    # a staircase from the bottom face to the top: no voxel column, no wrap in z
    for z in range(16):
        x = 2 + z // 3
        lab[x:x + 2, 5:7, z] = 1
    expect(CD.spans_z(lab == 1), "the staircase does not span the film")
    expect(not CD.wrapping_axes(lab == 1)[2], "the staircase wraps periodically in z")
    lab2 = lab.copy()
    lab2[:, :, 8] = 0
    expect(not CD.spans_z(lab2 == 1), "a broken path spans the film")
    vals = np.array([1e-12, 1e6])
    v1, p1 = CD.contrast_policy(lab, vals, 2, 1e6, {}, film=True)
    v0, p0 = CD.contrast_policy(lab, vals, 2, 1e6, {}, film=False)
    expect(p1 == "network" and p0 == "dispersed", f"policies film {p1}, periodic {p0}")
    k = CD.solve_direction(lab, v1, 2, film=True, tol=1e-9)["k_eff"]
    expect(k > 1e3, f"film σ {k:.3g} through a spanning conductor")
    return f"spans the film, does not wrap; policy {p1} (periodic test: {p0}); σ_zz {k:.4g} S/m"


@check("Queue: runs build their structures side by side and wait before solving until their memory fits; a light "
       "solve passes a heavy one only while the heavy one still fits beside it; a run alone keeps the cores set in "
       "Cores each (it took every core in 5.6.1) and takes them all when automatic")
def t_queue_gate():
    import threading
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))
    from jobs import JobQueue, Job

    class V:
        value = 0
    q = JobQueue(concurrency=4, mem_budget_gb=10.0)
    q.MEM_SCALE, q.MEM_BASE = 1.0, 0.0
    q.check_free = False                   # the budget alone, not this PC's free memory

    def job(jid, peak, struct=0.2, threads=4):
        j = Job(jid, None, {"name": jid}, ".", {"mem_gb": peak, "mem_struct_gb": struct, "gated": True}, q)
        j.state, j.phase, j.threads = "running", "structure", threads
        j.grant = (threading.Event(), V())
        with q.lock:
            q.jobs[jid] = j
            q.order.append(jid)
        return j
    a, h, l1, l2 = job("A", 6.0), job("H", 6.0), job("L1", 1.0), job("L2", 3.5)
    with q.lock:
        expect(abs(sum(q._expected(x) for x in q.running()) - 0.8) < 1e-9, "structures do not reserve their own memory")
        for j, t in ((a, 1.0), (h, 2.0), (l1, 3.0), (l2, 4.0)):
            j.phase, j.gate_t = "gate", t
        q._open_gates()
    expect(a.phase == "solve" and a.grant[0].is_set(), "the first at the gate does not solve")
    expect(h.phase == "gate", "the heavy second solves beside the first")
    expect(l1.phase == "solve", "the light one does not pass the heavy one")
    expect(l2.phase == "gate", "a light one passes that would keep the heavy one out")
    with q.lock:
        a.state = "done"
        q._open_gates()
    expect(h.phase == "solve", "the heavy one does not solve once the first is done")
    # the cores of a run alone: the number set (Cores each), or every core when automatic
    cores = os.cpu_count() or 2
    got = {}
    for auto in (False, True):
        with q.lock:
            q.jobs.clear()
            q.order.clear()
        q.auto_threads, q.threads = auto, 3
        s = job("S%d" % auto, 1.0, threads=3)
        s.auto_cores = auto
        with q.lock:
            s.phase, s.gate_t = "gate", 1.0
            q._open_gates()
        got[auto] = (s.phase, int(s.threads), int(s.grant[1].value))
    expect(got[False] == ("solve", 3, 3), f"a run set to 3 cores: {got[False]}")
    expect(got[True] == ("solve", cores, cores), f"an automatic run alone: {got[True]} of {cores} cores")
    with q.lock:
        q.jobs.clear()
        q.order.clear()
    return (f"A solves, H (6 GB) waits, L1 (1 GB) passes it, L2 (3.5 GB) does not; H solves when A is done; "
            f"a run alone solves on the 3 cores set, or on all {cores} when automatic")


def _flag_waiter(flag):
    while not flag.wait(0.3):
        pass


@check("Stop at the memory gate: a run stopped while it waits for the memory of its solves is not granted, and the "
       "queue goes on answering (it hung in 5.6.0)", quick=False)
def t_stop_at_gate():
    import multiprocessing as mp
    import threading
    from web.jobs import Flag, JobQueue
    # the cause: a multiprocessing Event's set() waits for a process killed
    # in its wait() - for ever; the Flag that replaces it does not
    ctx = mp.get_context("spawn")
    hung = {}
    for name, make in (("Event", lambda: ctx.Event()), ("Flag", lambda: Flag(ctx))):
        f = make()
        pr = ctx.Process(target=_flag_waiter, args=(f,))
        pr.start()
        time.sleep(3.0)
        pr.kill()
        pr.join()
        th = threading.Thread(target=f.set, daemon=True)
        th.start()
        th.join(5.0)
        hung[name] = th.is_alive()
    expect(not hung["Flag"], "the Flag's set() blocks after its waiter was killed")
    big = {"name": "selftest gate A", "matrix": {"material_id": "epoxy"},
           "phases": [{"material_id": "al2o3", "name": "Al2O3", "shape": "sphere", "size": {"d": 4, "unit": "um"},
                       "fraction": {"value": 40, "basis": "vol"}}],
           "rve": {"auto": False, "L_um": 64, "voxel_um": 0.4, "auto_enlarge": False, "quality": "fast"},
           "analyses": {"thermal": True, "electrical": True, "dielectric": True, "cte": True}, "options": {"directions": "xyz"}}
    small = dict(big, name="selftest gate B", rve=dict(big["rve"], L_um=16), analyses={"thermal": True},
                 options={"directions": "z"})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        # 1 GB to solve each, 1.5 GB for both: B builds its structure beside
        # A's solves, then waits at the gate
        q = JobQueue(concurrency=2, mem_budget_gb=1.5)
        q.MEM_SCALE, q.MEM_BASE = 1.0, 0.0
        q.check_free = False
        meta = lambda n, a: {"name": n, "analyses": a, "mem_gb": 1.0, "mem_struct_gb": 0.01, "gated": True}  # noqa: E731
        ja = q.submit(None, big, d, meta("A", ["thermal", "electrical", "dielectric", "cte"]))
        t0 = time.time()
        while time.time() - t0 < 120 and ja.phase != "solve":
            time.sleep(0.2)
        jb = q.submit(None, small, d, meta("B", ["thermal"]))
        while time.time() - t0 < 240 and not (jb.state == "running" and jb.phase == "gate"):
            time.sleep(0.2)
        if not (jb.phase == "gate" and ja.state == "running"):
            for j in (ja, jb):
                q.stop(j)
            time.sleep(3)
        expect(jb.phase == "gate" and ja.state == "running", f"B is not waiting at the gate ({jb.state}, {jb.phase}; A {ja.state})")
        time.sleep(1.5)                      # B sleeps in its wait for the grant
        q.stop(jb)
        while time.time() - t0 < 600 and ja.state == "running":
            time.sleep(0.5)
        time.sleep(3.0)                      # the queue's loop opens the gates again
        got = q.lock.acquire(timeout=10)
        if got:
            q.lock.release()
        expect(got, "the queue's lock is held: the loop hung")
        expect(jb.state == "stopped" and jb.phase == "gate", f"B ended {jb.state} in phase {jb.phase}")
        expect(ja.state == "done", f"A ended {ja.state}")
    return (f"a killed waiter {'hangs' if hung['Event'] else 'does not hang'} a multiprocessing Event's set(), not the "
            f"Flag's; A solved and finished ({ja.ended - ja.started:.0f} s); B, stopped at the gate, was not granted; "
            f"the queue answers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="run the checks whose name contains this text")
    a = ap.parse_args()
    tests = [v for v in globals().values() if callable(v) and hasattr(v, "_name")]
    if a.quick:
        tests = [t for t in tests if t._quick]
    if a.only:
        tests = [t for t in tests if a.only.lower() in t._name.lower()]
    print("=" * 72)
    print("  MPSim (Material Property Simulation) · self-test")
    print("=" * 72)
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            msg = t()
            print(f"  [ OK ] {t._name}\n         {msg}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t._name}\n         {type(e).__name__}: {e}", flush=True)
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    print("-" * 72)
    print(f"  {len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
