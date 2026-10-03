"""Full-wave check of the homogenised layer with openEMS (FDTD).

The RVE is simulated as a slab of its own thickness in a plane wave at
normal incidence, voxel for voxel, with Maxwell's equations in the time
domain - no homogenisation. The same slab is then described by the
homogenised admittivity kappa_eff(f) and permeability mu_eff and put through
the exact slab transmission; the two S21 (shielding effectiveness) and S11
must agree while the homogenisation holds.

Set-up (openEMS, GPL, https://openems.de, run as a separate program):
* unit cell of the RVE's lateral size with PEC walls at x = 0, L (E along x)
  and PMC walls at y = 0, L; the wave travels along z. These walls are
  mirror planes, so the cell stands for the RVE repeated by mirroring. The
  homogenised values for the comparison are therefore solved with the same
  walls - fixed potentials on the x faces for kappa, on the y faces for mu,
  the other faces insulated (complex_cond, bc="plates") - not periodic;
* one voxel = one FDTD cell inside the slab; the mesh grows by 1.3 per cell
  outside it, up to 4 voxels;
* first-order Mur absorbing boundaries on both z ends (exact for this
  normal-incidence TEM wave);
* a soft Gaussian E_x source on a plane before the slab; voltage probes
  across the cell before and after it; a second run without the slab gives
  the incident wave, so S21 = u_out / u_out,ref and S11 = (u_in - u_in,ref)
  / u_in,ref.
Loss tangents are left out on both sides (openEMS takes a frequency-
independent conductivity), so the comparison is like for like.

Verified on a homogeneous slab (eps_r 4, sigma 2000 S/m, 20 um): openEMS
and the exact transmission agree to 0.001 dB in SE and 1e-4 in |S11| from
10 to 100 GHz.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import time

import numpy as np

from . import complex_cond as CC
from . import emi as EMI


def find_openems():
    """Path of openEMS.exe, or None: MPSIM_OPENEMS (the exe or its folder),
    then tools\\openEMS and openEMS next to the program, then C:\\openEMS."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cands = []
    env = os.environ.get("MPSIM_OPENEMS")
    if env:
        cands += [env, os.path.join(env, "openEMS.exe")]
    cands += [os.path.join(here, "tools", "openEMS", "openEMS.exe"), os.path.join(here, "openEMS", "openEMS.exe"),
              r"C:\openEMS\openEMS.exe"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return shutil.which("openEMS")


def _vec(v):
    return ",".join(f"{x:.10g}" for x in np.atleast_1d(v))


def _box(p0, p1, prio=0):
    return (f'<Box Priority="{prio}"><P1 X="{p0[0]:.10g}" Y="{p0[1]:.10g}" Z="{p0[2]:.10g}"/>'
            f'<P2 X="{p1[0]:.10g}" Y="{p1[1]:.10g}" Z="{p1[2]:.10g}"/></Box>')


def _runs(mask):
    """Runs of True along x for every (y, z) row: (i0, i1, j, k), half-open."""
    n0 = mask.shape[0]
    pad = np.zeros((n0 + 2,) + mask.shape[1:], bool)
    pad[1:-1] = mask
    d = np.diff(pad.astype(np.int8), axis=0)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    # both sorted by (j, k) then i after a stable sort on the row index
    so = np.lexsort((starts[:, 0], starts[:, 2], starts[:, 1]))
    eo = np.lexsort((ends[:, 0], ends[:, 2], ends[:, 1]))
    s, e = starts[so], ends[eo]
    return np.column_stack([s[:, 0], e[:, 0], s[:, 1], s[:, 2]])


def write_model(path, labels, table, h_um, with_slab, f0, fc, gap_um, end_crit=1e-6):
    """model.xml for one run. Coordinates in um; the slab is 0 <= z <= T."""
    n0, n1, n2 = labels.shape
    L, W, T = n0 * h_um, n1 * h_um, n2 * h_um
    xs = np.arange(n0 + 1) * h_um
    ys = np.arange(n1 + 1) * h_um
    zs_src, zp_in, zp_out = -2.0 * gap_um, -gap_um, T + gap_um
    z_lo, z_hi = -3.0 * gap_um, T + 2.0 * gap_um
    up, dz = [T], h_um
    while up[-1] < z_hi:
        dz = min(dz * 1.3, 4.0 * h_um)
        up.append(up[-1] + dz)
    dn, dz = [0.0], h_um
    while dn[-1] > z_lo:
        dz = min(dz * 1.3, 4.0 * h_um)
        dn.append(dn[-1] - dz)
    zl = np.unique(np.round(np.concatenate([dn[::-1], np.arange(n2 + 1) * h_um, up,
                                            [zs_src, zp_in, zp_out]]), 7))
    props = []
    if with_slab:
        mat = int(np.bincount(labels.ravel()).argmax())             # the continuous region
        for li, t in enumerate(table):
            p = t["props"]
            eps = max(float(p.get("eps_r", 1.0)), 1.0)
            kap = max(float(p.get("sigma", 0.0)), 0.0)
            mu = max(float(p.get("mu_r", 1.0)), 1.0)
            prims = []
            if li == mat:
                prims.append(_box((0, 0, 0), (L, W, T), 1))
            else:
                for i0, i1, j, k in _runs(labels == li):
                    prims.append(_box((i0 * h_um, j * h_um, k * h_um), (i1 * h_um, (j + 1) * h_um, (k + 1) * h_um), 2))
            if prims:
                props.append(f'<Material Name="m{li}"><Property Epsilon="{eps:.10g}" Kappa="{kap:.10g}" Mue="{mu:.10g}"/>'
                             f'<Primitives>{"".join(prims)}</Primitives></Material>')
    props.append(f'<Excitation Name="src" Type="0" Excite="1,0,0"><Primitives>'
                 f'{_box((0, 0, zs_src), (L, W, zs_src))}</Primitives></Excitation>')
    for name, z in (("u_in", zp_in), ("u_out", zp_out)):
        props.append(f'<ProbeBox Name="{name}" Type="0" Weight="-1"><Primitives>'
                     f'{_box((0, W / 2, z), (L, W / 2, z))}</Primitives></ProbeBox>')
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<openEMS>'
           f'<FDTD NumberOfTimesteps="3000000" endCriteria="{end_crit:g}" f_max="{f0 + fc:.6g}">'
           f'<Excitation Type="0" f0="{f0:.6g}" fc="{fc:.6g}"/>'
           f'<BoundaryCond xmin="0" xmax="0" ymin="1" ymax="1" zmin="2" zmax="2"/></FDTD>'
           f'<ContinuousStructure CoordSystem="0"><Properties>{"".join(props)}</Properties>'
           f'<RectilinearGrid DeltaUnit="1e-6" CoordSystem="0"><XLines>{_vec(xs)}</XLines>'
           f'<YLines>{_vec(ys)}</YLines><ZLines>{_vec(zl)}</ZLines></RectilinearGrid>'
           f'</ContinuousStructure></openEMS>\n')
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.xml"), "w", encoding="utf-8") as fh:
        fh.write(xml)
    return len(xs) * len(ys) * len(zl)


def _run(exe, path, should_stop=None):
    t0 = time.time()
    with open(os.path.join(path, "openEMS.log"), "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen([exe, "model.xml", "--engine=fastest"], cwd=path, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while p.poll() is None:
            if should_stop is not None and should_stop():
                p.kill()
                raise InterruptedError
            time.sleep(0.5)
    if p.returncode != 0:
        with open(os.path.join(path, "openEMS.log"), encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-1500:]
        raise RuntimeError(f"openEMS stopped with code {p.returncode}: {tail}")
    return time.time() - t0


def _dft(path, name, f):
    d = np.loadtxt(os.path.join(path, name), comments="%")
    t, v = d[:, 0], d[:, 1]
    dt = t[1] - t[0]
    return np.exp(-2j * np.pi * np.outer(f, t)) @ v * dt


def check(labels, table, h_um, workdir, f_lo=10e9, f_hi=100e9, n_f=19, tol=1e-7, log=print, should_stop=None,
          exe=None, max_cells=4_000_000, end_crit=1e-6):
    """Full-wave S21/S11 of the RVE slab against the homogenised slab.

    Returns freq_hz, se_fullwave_db, se_model_db, s11 both, the homogenised
    kappa and mu used, the largest SE difference (dB) and run times."""
    exe = exe or find_openems()
    if not exe:
        raise FileNotFoundError("openEMS was not found (MPSIM_OPENEMS, tools\\openEMS or C:\\openEMS)")
    labels = np.ascontiguousarray(labels, np.uint8)
    n0, n1, n2 = labels.shape
    f = np.linspace(f_lo, f_hi, n_f)
    f0, fc = 0.5 * (f_lo + f_hi), 0.5 * (f_hi - f_lo) + 0.1 * f_lo
    gap = max(1.5 * max(n0, n1) * h_um, 30.0 * h_um)           # higher modes decay as exp(-pi z / L)
    res = {}
    cells = 0
    for tag, slab in (("ref", False), ("slab", True)):
        d = os.path.join(workdir, tag)
        cells = write_model(d, labels, table, h_um, slab, f0, fc, gap, end_crit)
        if cells > max_cells:
            raise ValueError(f"the full-wave model would have {cells:,} cells (limit {max_cells:,}); "
                             f"a smaller RVE or a coarser voxel keeps the check affordable")
        log(f"    openEMS {tag}: {cells:,} cells ...")
        sec = _run(exe, d, should_stop)
        res[tag] = {"u_in": _dft(d, "u_in", f), "u_out": _dft(d, "u_out", f), "seconds": sec}
        log(f"    openEMS {tag}: {sec:.0f} s")
    s21 = res["slab"]["u_out"] / res["ref"]["u_out"]
    s11 = (res["slab"]["u_in"] - res["ref"]["u_in"]) / res["ref"]["u_in"]
    se_fw = -20.0 * np.log10(np.abs(s21))

    # the homogenised slab with the same walls (no loss tangents, as openEMS)
    tab0 = [dict(t, props=dict(t["props"], tan_d=0.0)) for t in table]
    mu_tab = np.array([max(float(t["props"].get("mu_r", 1.0)), 1.0) for t in table], np.complex128)
    mu_eff = 1.0
    if np.any(mu_tab.real != 1.0):
        mu_eff = float(CC.solve(labels, mu_tab, 1, bc="plates", tol=tol, should_stop=should_stop)["kappa"].real)
    kap = []
    for fk in f:
        k, _ = CC.cap_contrast(labels, CC.admittivity(tab0, fk), 0)
        kap.append(CC.solve(labels, k, 0, bc="plates", tol=tol, should_stop=should_stop)["kappa"])
    kap = np.array(kap)
    w = 2.0 * math.pi * f
    eps_f = kap.imag / (w * EMI.EPS0)
    sig_f = kap.real
    t_m = n2 * h_um * 1e-6
    s11_m, l21 = EMI.slab_transmission(f, [(t_m, eps_f, mu_eff, sig_f)])
    se_m = -20.0 * l21
    diff = np.abs(se_fw - se_m)
    return {"freq_hz": f.tolist(), "se_fullwave_db": se_fw.tolist(), "se_model_db": se_m.tolist(),
            "s11_fullwave": np.abs(s11).tolist(), "s11_model": np.abs(s11_m).tolist(),
            "max_diff_db": float(diff.max()), "mean_diff_db": float(diff.mean()),
            "kappa_re": sig_f.tolist(), "eps_eff": eps_f.tolist(), "mu_eff": mu_eff,
            "thickness_um": n2 * h_um, "cells": int(cells),
            "seconds": float(res["ref"]["seconds"] + res["slab"]["seconds"]), "openems": exe}
