"""Through via: a plated wall around a conductive fill (5.7.0).

A plated-through hole whose wall is electroplated copper and whose core is
filled with a conductive paste - the composite of the RVE, or a value -
through a board of thickness l, with return vias at the pitch. Computed
here, for the via as a transmission line along its axis (quasi-TEM):

* DC resistance: the wall and the fill in parallel at every height, the wall
  thinner towards the middle of the hole by the throwing power of the
  plating (hole-centre over surface thickness; a parabola between them),
  integrated along the via.
* Internal impedance per length Z_i(f) of the two-layer round conductor:
  the Bessel solution of a wall (radius a, thickness t, sigma_w) around a
  core (radius b = a - t, sigma_c) - S. A. Schelkunoff, Bell Syst. Tech. J.
  13 (1934) 532, doi:10.1002/j.1538-7305.1934.tb00679.x. Its limits: the DC
  resistance of the two in parallel, and above the frequency at which the
  skin depth equals the wall the surface resistance of the wall alone; the
  current share of the core follows from the same solution.
* The cross-section: the capacitance per length C' between the signal via
  and its return vias in a homogeneous dielectric, by the multipole method
  (line charge and multipoles on every via, the potentials of the via
  surfaces matched); L'_ext = mu0 eps0 / C'_0 (TEM), G' = omega C' tan d.
* The line: Z' = Z_i,signal + Z_i,return / N + j omega L'_ext, Y' = G' +
  j omega C'; Z0 = sqrt(Z'/Y'), gamma = sqrt(Z' Y'); the via cut in slices
  (the wall varies along it) cascaded as ABCD matrices; S11 and S21 between
  ports of Z_ref. The via operated as a transmission line between its ends
  is the model of Yildiz and Schuster (EDAPS 2019, doi:10.1109/
  EDAPS47854.2019.9011681), in which the characteristic impedance depends on
  the distance to and the number of ground vias.

Left out (stated with the results): the pads, the antipads and the planes
of the stack-up and the transitions to traces (a 3D full-wave problem), the
proximity effect between the vias (each via's current is taken
axisymmetric), the frequency dependence of the fill's conductivity (its DC
value: in the band where the fill carries current, the skin depth of the
fill is far larger than its particles), and the dispersion of the
laminate's permittivity (one eps_r and tan d).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special as SP

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
C0 = 299792458.0

LAYOUTS = {
    # return vias around the signal via at the pitch p: (x, y) / p
    "coax4": ("4 return vias around the signal (square, at the pitch)", [(1, 0), (-1, 0), (0, 1), (0, -1)]),
    "coax8": ("8 return vias around the signal (every neighbour at the pitch grid)",
              [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]),
    "gsg": ("ground-signal-ground in a row", [(1, 0), (-1, 0)]),
    "sg": ("one return via beside the signal", [(1, 0)]),
}


# ------------------------------------------------------------------ DC
def wall_profile(t_end, throw, z):
    """Wall thickness along the via (z from 0 to 1): t_end at the two ends, throw * t_end in the middle, a
    parabola between them."""
    z = np.asarray(z, float)
    return t_end * (throw + (1.0 - throw) * (2.0 * z - 1.0) ** 2)


def dc_resistance(a, t_end, throw, length, s_wall, s_core, n=401):
    """Resistance [ohm] of one via: wall and fill in parallel at every height, integrated along it."""
    z = np.linspace(0.0, 1.0, n)
    t = np.minimum(wall_profile(t_end, throw, z), a)
    b = a - t
    g = s_wall * math.pi * (a * a - b * b) + s_core * math.pi * b * b      # conductance per length x length
    return float(np.trapz(1.0 / g, z) * length)


# ------------------------------------------------------------------ internal impedance
def wire_impedance(f, a, t, s_wall, s_core):
    """Internal impedance per length [ohm/m] of a round conductor of radius a - a wall of thickness t
    (s_wall) around a core of radius a - t (s_core; 0 for a hollow wall) - and the share of the current the
    core carries (|I_core / I|). Bessel solution in scaled functions (scipy ive, kve), valid from the DC
    limit to walls hundreds of skin depths thick."""
    f = np.atleast_1d(np.asarray(f, float))
    b = max(a - t, 0.0)
    rdc = 1.0 / (math.pi * (s_wall * (a * a - b * b) + s_core * b * b))
    Z = np.empty(f.shape, complex)
    share = np.empty(f.shape)
    for i, fi in enumerate(f):
        w = 2.0 * math.pi * fi
        ks = np.sqrt(1j * w * MU0 * s_wall)
        if abs(ks) * a < 1e-3:
            # below the Bessel range the current is uniform: the DC values
            Z[i] = rdc
            share[i] = (s_core * b * b) / (s_wall * (a * a - b * b) + s_core * b * b)
            continue
        xa = ks * a
        if b <= 0.0:
            Z[i] = 1j * w * MU0 / (2.0 * math.pi * a) * SP.ive(0, xa) / (ks * SP.ive(1, xa))
            share[i] = 0.0
            continue
        xb = ks * b
        if s_core > 0.0:
            kc = np.sqrt(1j * w * MU0 * s_core)
            xc = kc * b
            if abs(xc) < 1e-8:
                zeta = np.inf
            else:
                zeta = SP.ive(0, xc) / (kc * SP.ive(1, xc))      # E / E' of the core at r = b
        else:
            zeta = np.inf                                        # a hollow wall: no current inside
        i0b, i1b, k0b, k1b = SP.ive(0, xb), SP.ive(1, xb), SP.kve(0, xb), SP.kve(1, xb)
        rho = i1b / k1b if np.isinf(zeta) else (zeta * ks * i1b - i0b) / (k0b + zeta * ks * k1b)
        F = np.exp(-ks.real * (a - b) - ks * (a - b))
        i0a, i1a, k0a, k1a = SP.ive(0, xa), SP.ive(1, xa), SP.kve(0, xa), SP.kve(1, xa)
        ratio = (i0a + rho * k0a * F) / (ks * (i1a - rho * k1a * F))  # E / E' at r = a
        Z[i] = 1j * w * MU0 / (2.0 * math.pi * a) * ratio
        # the core's share: explicit amplitudes where the wall is not too many skin depths thick
        if s_core <= 0.0:
            share[i] = 0.0
        elif ks.real * (a - b) > 30.0:
            share[i] = 0.0
        else:
            c0, c1 = SP.ive(0, xc), SP.ive(1, xc)
            M = np.array([[i0b, k0b], [ks * i1b, -ks * k1b]], complex)
            beta, gam = np.linalg.solve(M, np.array([c0, kc * c1], complex))
            dEa = beta * ks * i1a * math.exp(ks.real * (a - b)) - gam * ks * k1a * np.exp(-ks * (a - b))
            share[i] = float(abs(b * kc * c1 / (a * dEa)))
    return Z, share, rdc


def skin_depth(f, sigma):
    return np.sqrt(1.0 / (math.pi * np.asarray(f, float) * MU0 * sigma))


# ------------------------------------------------------------------ the cross-section
def cross_section(layout, d, pitch, order=10, m=64):
    """Capacitance per length between the signal via and its return vias in vacuum, C'_0 [F/m], by the
    multipole method: on every via a line charge q_k and multipoles up to `order` about its centre; the
    potential of the signal via surface is 1 V, of the returns 0 V (plus a common constant), the charges add
    to zero. Collocation at m points on every surface, solved in least squares; the residual of the surface
    potentials is returned as a check."""
    pts = [(0.0, 0.0)] + [(x * pitch, y * pitch) for x, y in LAYOUTS[layout][1]]
    r = 0.5 * d
    nv = len(pts)
    th = 2.0 * math.pi * (np.arange(m) + 0.5) / m
    # unknowns: per via q, a_1..a_K, b_1..b_K; then the constant
    nu = nv * (1 + 2 * order) + 1
    rows, rhs = [], []
    for j, (xj, yj) in enumerate(pts):
        X = xj + r * np.cos(th)
        Y = yj + r * np.sin(th)
        A = np.zeros((m, nu))
        for k, (xk, yk) in enumerate(pts):
            dx, dy = X - xk, Y - yk
            rr = np.hypot(dx, dy) / r
            ph = np.arctan2(dy, dx)
            c = k * (1 + 2 * order)
            A[:, c] = -np.log(rr)
            for n in range(1, order + 1):
                A[:, c + n] = np.cos(n * ph) / rr ** n
                A[:, c + order + n] = np.sin(n * ph) / rr ** n
        A[:, -1] = 1.0
        rows.append(A)
        rhs.append(np.full(m, 1.0 if j == 0 else 0.0))
    neutral = np.zeros((1, nu))
    for k in range(nv):
        neutral[0, k * (1 + 2 * order)] = 1.0
    A = np.vstack(rows + [neutral * 1e3])
    rhs = np.concatenate(rhs + [np.zeros(1)])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    res = float(np.max(np.abs(A[:-1] @ sol - rhs[:-1])))
    # the potential of a line charge q is -q / (2 pi eps0) ln r: q_k / (2 pi eps0) is the coefficient
    q_sig = sol[0] * 2.0 * math.pi * EPS0
    return float(q_sig), res


def two_wire_capacitance(d, s):
    """Exact C'_0 of two parallel round wires of diameter d at centre distance s (check of cross_section)."""
    return math.pi * EPS0 / math.acosh(s / d)


# ------------------------------------------------------------------ the line
def _abcd_slice(Zp, Yp, dl):
    g = np.sqrt(Zp * Yp)
    z0 = np.sqrt(Zp / Yp)
    ch, sh = np.cosh(g * dl), np.sinh(g * dl)
    return ch, z0 * sh, sh / z0, ch


def line(f, Zi_sig, Zi_ret, n_ret, Lext, C, tan_d, length, z_ref=50.0):
    """S11, S21, Z0 and the series R, L of a via cut in slices: Zi_sig, Zi_ret are (n_slices, n_f) internal
    impedances per length of each slice."""
    f = np.asarray(f, float)
    w = 2.0 * math.pi * f
    ns = Zi_sig.shape[0]
    dl = length / ns
    A = np.ones(f.shape, complex)
    B = np.zeros(f.shape, complex)
    Cm = np.zeros(f.shape, complex)
    D = np.ones(f.shape, complex)
    Zsum = np.zeros(f.shape, complex)
    Yp = w * C * tan_d + 1j * w * C
    for s_ in range(ns):
        Zp = Zi_sig[s_] + Zi_ret[s_] / n_ret + 1j * w * Lext
        Zsum += Zp * dl
        a, b, c, d = _abcd_slice(Zp, Yp, dl)
        A, B, Cm, D = A * a + B * c, A * b + B * d, Cm * a + D * c, Cm * b + D * d
    den = A + B / z_ref + Cm * z_ref + D
    S21 = 2.0 / den
    S11 = (A + B / z_ref - Cm * z_ref - D) / den
    Zp_mean = Zsum / length
    Z0 = np.sqrt(Zp_mean / Yp)
    return {"S11": S11, "S21": S21, "Z0": Z0, "R": Zsum.real, "L": Zsum.imag / w, "Zseries": Zsum}


# ------------------------------------------------------------------ the whole analysis
def analyse(o, s_core, s_wall, log=print):
    """The via analysis for the options o (spec "via") with the fill conductivity s_core [S/m] and the wall's
    s_wall [S/m]. Returns the result entry (lists for JSON)."""
    a = 0.5 * o["hole_d_um"] * 1e-6
    length = o["length_mm"] * 1e-3
    p = o["pitch_um"] * 1e-6
    t_end = o["plating_um"] * 1e-6
    throw = o["throw"]
    n_ret = len(LAYOUTS[o["returns"]][1])
    f = np.unique(np.append(np.logspace(math.log10(o["f_min"]), math.log10(o["f_max"]), int(o["n_f"])), o["f_ref"]))
    C0p, res = cross_section(o["returns"], 2.0 * a, p)
    C = C0p * o["eps_r"]
    Lext = MU0 * EPS0 / C0p
    zs = (np.arange(o["slices"]) + 0.5) / o["slices"]

    def one(t_e, keep=False):
        ts = np.minimum(wall_profile(t_e, throw, zs), a)
        Zi = []
        sh = []
        for t in ts:
            z_, s_, _ = wire_impedance(f, a, t, s_wall, s_core)
            Zi.append(z_)
            sh.append(s_)
        Zi = np.array(Zi)
        ln = line(f, Zi, Zi, n_ret, Lext, C, o["tan_d"], length, o["z_ref"])
        r_dc = dc_resistance(a, t_e, throw, length, s_wall, s_core)
        r_dc_wall = dc_resistance(a, t_e, throw, length, s_wall, 0.0)
        out = {"t_um": round(t_e * 1e6, 9), "t_mid_um": round(throw * t_e * 1e6, 9), "R_dc_ohm": r_dc, "R_dc_wall_only_ohm": r_dc_wall,
               "fill_share_dc": float(1.0 - r_dc / r_dc_wall),
               "R_via_ohm": (np.mean(Zi, axis=0).real * length),
               "fill_share": np.mean(np.array(sh), axis=0)}
        if keep:
            out.update(ln)
        else:
            out.update({"S21": ln["S21"], "Z0": ln["Z0"], "R": ln["R"], "L": ln["L"]})
        return out

    main = one(t_end, keep=True)
    sweep = [one(float(x) * 1e-6) for x in o["sweep_um"]]
    # where the skin depth of the wall equals its thickness: the fill drops out of the current above it
    f_wall = 1.0 / (math.pi * MU0 * s_wall * (throw * t_end) ** 2)
    fr = o["f_ref"]
    iref = int(np.argmin(np.abs(np.log(f / fr))))
    db = lambda x: 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))  # noqa: E731
    out = {
        "title": "Through via", "layout": o["returns"], "layout_text": LAYOUTS[o["returns"]][0], "n_returns": n_ret,
        "hole_d_um": o["hole_d_um"], "length_mm": o["length_mm"], "aspect_ratio": o["length_mm"] * 1e3 / o["hole_d_um"],
        "pitch_um": o["pitch_um"], "plating_um": o["plating_um"], "throw": throw, "eps_r": o["eps_r"], "tan_d": o["tan_d"],
        "z_ref": o["z_ref"], "sigma_fill_S_m": s_core, "sigma_wall_S_m": s_wall,
        "fill_from": o["fill"], "C_pF_m": C * 1e12, "L_ext_nH_m": Lext * 1e9, "multipole_residual_V": res,
        "Z0_lossless_ohm": math.sqrt(Lext / C), "delay_ps": length * math.sqrt(Lext * C) * 1e12,
        "f_wall_Hz": f_wall, "f_ref": float(f[iref]),
        "R_dc_mohm": main["R_dc_ohm"] * 1e3, "R_dc_wall_only_mohm": main["R_dc_wall_only_ohm"] * 1e3,
        "fill_share_dc": main["fill_share_dc"],
        "loop_R_dc_mohm": main["R_dc_ohm"] * (1.0 + 1.0 / n_ret) * 1e3,
        "at_ref": {"R_via_mohm": float(main["R_via_ohm"][iref] * 1e3), "Z0_ohm": float(abs(main["Z0"][iref])),
                   "S21_dB": float(db(main["S21"][iref])), "S11_dB": float(db(main["S11"][iref])),
                   "L_loop_pH": float(main["L"][iref] * 1e12), "fill_share": float(main["fill_share"][iref])},
        "freq_hz": f.tolist(),
        "R_via_ohm": main["R_via_ohm"].tolist(), "fill_share": main["fill_share"].tolist(),
        "R_loop_ohm": main["R"].tolist(), "L_loop_H": main["L"].tolist(),
        "Z0_abs_ohm": np.abs(main["Z0"]).tolist(), "Z0_re_ohm": main["Z0"].real.tolist(),
        "S21_dB": db(main["S21"]).tolist(), "S11_dB": db(main["S11"]).tolist(),
        "S21_deg": np.degrees(np.angle(main["S21"])).tolist(),
        "sweep": [{"t_um": s["t_um"], "t_mid_um": s["t_mid_um"], "R_dc_mohm": s["R_dc_ohm"] * 1e3,
                   "R_dc_wall_only_mohm": s["R_dc_wall_only_ohm"] * 1e3, "fill_share_dc": s["fill_share_dc"],
                   "R_via_mohm": (s["R_via_ohm"] * 1e3).tolist(), "fill_share": s["fill_share"].tolist(),
                   "S21_dB": db(s["S21"]).tolist(), "Z0_abs_ohm": np.abs(s["Z0"]).tolist(),
                   "R_ref_mohm": float(s["R_via_ohm"][iref] * 1e3), "S21_ref_dB": float(db(s["S21"][iref]))}
                  for s in sweep],
    }
    log(f"   through via: R_dc {out['R_dc_mohm']:.3f} mOhm per via ({100 * out['fill_share_dc']:.1f} % through the fill), "
        f"Z0 {out['Z0_lossless_ohm']:.1f} Ohm ({o['returns']}, eps_r {o['eps_r']:g}), at {fr / 1e9:g} GHz S21 "
        f"{out['at_ref']['S21_dB']:.3f} dB; the fill drops out of the current above {f_wall / 1e6:.3g} MHz")
    return out
