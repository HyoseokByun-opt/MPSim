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


@check("Viscosity: a dilute rigid sphere gives Einstein's 2.5, a free bubble -5/3, independent of the rigid contrast")
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
    return (f"φ {phi:.4f}: rigid [η] {eta:.3f} (contrast ×10: {eta4:.3f}), bubble [η] {eta_b:.3f}, "
            f"iterations {[st['iterations'] for st in r['stats'].values()]}")


@check("Viscosity closed forms: Krieger-Dougherty and Maron-Pierce limits, intrinsic viscosity of rods and discs")
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
    return f"KD(0.63) {float(VI.krieger_dougherty(0.63, 0.64, 2.5)):.0f}, rod (20) [η] {rod:.2f}, disc (1/20) [η] {disc:.2f}"


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


@check("Calibration theory: mismatch models for identical materials (full transmission, DMM twice AMM)")
def t_cal_kapitza():
    from mpsim import calibrate as CA
    p = {"E": 70.0, "nu": 0.22, "rho": 2.2, "cp": 740.0}
    r = CA.kapitza(p, p)
    vl, vt = CA.sound_speeds(70.0, 0.22, 2.2)
    C = 2.2e3 * 740.0
    r_amm = 12.0 / (C * (vl + 2.0 * vt))
    expect(abs(r["amm"] / r_amm - 1) < 2e-3, f"AMM {r['amm']:.4g} vs {r_amm:.4g}")
    expect(abs(r["dmm"] / (2.0 * r_amm) - 1) < 1e-9, f"DMM {r['dmm']:.4g} vs {2 * r_amm:.4g}")
    return f"AMM {r['amm']:.3e}, DMM {r['dmm']:.3e} m²K/W (vL {vl:.0f}, vT {vt:.0f} m/s)"


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
                                       np.ascontiguousarray(p["phase"]), 1, cs, cc, fs, ts,
                                       np.empty(int(math.ceil(ext / h) + 4) ** 3, np.int64), False))
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
            "options": {"directions": "x", "live_figures": True, "viscosity": {"phi_m_mode": "manual", "phi_m": 0.64}}}
    seen = {"figure": [], "partial_result": 0}

    def event(kind, **p):
        if kind == "figure":
            seen["figure"].append(p.get("name") or p.get("file"))
        elif kind == "partial_result":
            seen["partial_result"] += 1
    with tempfile.TemporaryDirectory() as d:
        s = pipeline.run(form, d, log=lambda *a: None, event=event)
        expect(not os.path.exists(os.path.join(d, "result.partial.json")), "result.partial.json left behind")
        with open(os.path.join(d, "result.json"), encoding="utf-8") as fh:
            r = json.load(fh)
    v = r["properties"]["viscosity"]
    expect(seen["partial_result"] >= 2, f"{seen['partial_result']} provisional results")
    expect(len(seen["figure"]) >= 2, f"figures {seen['figure']}")
    expect(v["mu_r_rve"] >= 0.99 * v["refs"]["Hashin-Shtrikman lower bound"], f"mu_r {v['mu_r_rve']}")
    expect(v["mu_compound_ref"] > v["mu_resin_ref"], "compound not more viscous than the resin")
    return (f"μr {v['mu_r_rve']:.3f} (HS {v['refs']['Hashin-Shtrikman lower bound']:.3f}, KD {v['refs']['Krieger-Dougherty']:.3f}), "
            f"{seen['partial_result']} provisional results, {len(seen['figure'])} live figures, {s['elapsed_s']:.0f} s")


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
