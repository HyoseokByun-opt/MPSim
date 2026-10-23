"""Electro-thermal coupling of a film: Joule heating with temperature-
dependent conduction.

The film (x and y periodic, z its real thickness) carries a current between
its two faces (two pads, the electrodes). For every current density (or
voltage) of the sweep:

1. Electric potential. Steady conduction between the faces at the
   conductivity of every voxel at its temperature,
   sigma(T) = sigma_20 / (1 + TCR (T - 20 C)) (solvers/conduction.solve_film_z,
   per-voxel values); for a set current density the voltage is scaled to it.
2. Joule heat. Every face between two voxels dissipates G dphi^2 (its
   conductance times the potential difference squared), half to each voxel,
   and the half cell next to each pad likewise: the heat per volume of
   every voxel, concentrated where the current crowds (necks between
   particles).
3. Temperature. Steady conduction of that heat to the two faces (with the
   interface and contact resistances of the thermal analysis), the faces at
   the sink temperature T_amb + R_th P (P the heat per area of the film; R_th
   the thermal resistance from the joint to ambient, per area, which carries
   the heating of the whole joint).
4. Repeat 1-3 until the temperature changes by less than 1 mK. A current
   set by the source heats the film more as its resistance rises; when the
   temperature keeps climbing (beyond 1000 C or 60 iterations without
   settling) the step is marked as thermal runaway and the sweep ends.

Direct current, steady state. The thermal conductivity is taken constant;
an electrical contact resistance between touching particles is not
included (touching voxels conduct fully).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import conduction as CD

T_SIGMA_REF = 20.0           # C, the temperature the library's conductivities refer to
T_RUNAWAY = 1000.0           # C


def dissipation(phi, faces, gb, gt, h, v_bottom, v_top, sv=None):
    """Joule heat per volume (W/m^3) of every voxel from the potential and
    the face conductivities (S/m) of solve_film_z(keep_faces=True). The heat
    of a face goes to its two voxels in proportion to their halves of its
    resistance (h / 2 sigma each; equal halves without sv), which a series of
    two half cells dissipates exactly."""
    q = np.zeros(phi.shape)
    inv = None if sv is None else 1.0 / np.asarray(sv, float)
    for ax in range(3):
        d = np.roll(phi, -1, axis=ax) - phi
        p = faces[ax] * h * d * d                        # W per face
        if ax == 2:
            p[:, :, -1] = 0.0                            # no face across the plates
        if inv is None:
            w = 0.5
        else:
            nb = np.roll(inv, -1, axis=ax)
            w = inv / (inv + nb)                          # this voxel's share of the face
        q += w * p
        q += np.roll((1.0 - w) * p, 1, axis=ax)
    q[:, :, 0] += gb * h * (v_bottom - phi[:, :, 0]) ** 2
    q[:, :, -1] += gt * h * (phi[:, :, -1] - v_top) ** 2
    return q / h ** 3


def current_density(phi, faces, gb, gt, h, v_bottom, v_top):
    """|J| (A/m^2) at the voxel centres."""
    J = np.zeros((3,) + phi.shape)
    for ax in range(3):
        jf = -faces[ax] * (np.roll(phi, -1, axis=ax) - phi) / h
        if ax == 2:
            below = np.concatenate([(gb * (v_bottom - phi[:, :, 0]) / h)[:, :, None], jf[:, :, :-1]], axis=2)
            above = np.concatenate([jf[:, :, :-1], (gt * (phi[:, :, -1] - v_top) / h)[:, :, None]], axis=2)
            J[2] = 0.5 * (below + above)
        else:
            J[ax] = 0.5 * (jf + np.roll(jf, 1, axis=ax))
    return np.sqrt((J * J).sum(axis=0))


def lumped(steps, Rth, control):
    """The film as one resistor R(T) = R20 (1 + a (T - 20 C)) at its mean
    temperature, from the first and the last settled step: its effective
    temperature coefficient a and resistance at 20 C, and - for a set current
    through a conductor whose resistance rises (a > 0) with a resistance to
    ambient R_th - the current density at which the heating no longer
    settles: R_th J^2 R rises faster with temperature than the temperature
    itself, J_crit = 1 / sqrt(a R20 R_th). A set voltage heats less as R rises
    and does not run away this way."""
    ok = [x for x in steps if x["status"] == "converged" and x.get("R_ohm_m2")]
    if len(ok) < 2 or abs(ok[-1]["T_mean"] - ok[0]["T_mean"]) < 0.5:
        return None
    R1, R2 = ok[0]["R_ohm_m2"], ok[-1]["R_ohm_m2"]
    d1, d2 = ok[0]["T_mean"] - T_SIGMA_REF, ok[-1]["T_mean"] - T_SIGMA_REF
    den = R1 * d2 - R2 * d1
    if abs(den) < 1e-300:
        return None
    a = (R2 - R1) / den
    R20 = R1 / (1.0 + a * d1)
    out = {"tcr_eff": a, "R20_ohm_m2": R20, "J_crit_A_m2": None}
    if control == "current" and a > 0 and Rth > 0 and R20 > 0:
        out["J_crit_A_m2"] = 1.0 / math.sqrt(a * R20 * Rth)
    return out


def run(labels, table, h_um, opts, sig_vals, k_vals, rpair=None, cmask=None, rcpair=None, log=None,
        progress=None, should_stop=None, keep_fields=True, callback=None):
    """sig_vals: electrical conductivity per label (S/m, with the insulating
    phases raised to the solver's contrast limit); k_vals, rpair, cmask,
    rcpair: the thermal analysis's conduction inputs. opts: control
    ("current", "voltage"), values (A/m^2 or V), T_amb (C), R_th (K m^2/W),
    T_limit (C, optional: the temperature a step is judged against)."""
    log = log or (lambda *a: None)
    progress = progress or (lambda *a: None)
    t0 = time.time()
    h = float(h_um) * 1e-6
    nx, ny, nz = labels.shape
    L = nz * h
    props = [t["props"] for t in table]
    tcr = np.array([float(p.get("tcr") or 0.0) for p in props])
    sig = np.asarray(sig_vals, float)
    Tamb = float(opts["T_amb"])
    Rth = float(opts.get("R_th") or 0.0)
    control = opts.get("control", "current")
    values = [float(v) for v in opts["values"]]
    tg_labels = [i for i, p in enumerate(props) if p.get("tg") not in (None, "")]
    steps, fields = [], None
    T = np.full(labels.shape, Tamb)
    runaway = None
    for si, val in enumerate(values):
        if should_stop is not None and should_stop():
            raise InterruptedError
        progress(si, len(values))
        Tsink = Tamb
        it, conv = 0, False
        u_prev = th_prev = d_prev = None
        predicted = None                  # the extrapolated steady state, when above T_RUNAWAY
        for it in range(1, 61):
            fac = 1.0 + tcr[labels] * (T - T_SIGMA_REF)
            sv = sig[labels] / np.maximum(fac, 0.05)
            if should_stop is not None and should_stop():
                raise InterruptedError
            cbe = (lambda i_, r_: callback(f"Joule step {si + 1}: electric potential", i_, r_, 1e-9)) if callback else None
            cbt = (lambda i_, r_: callback(f"Joule step {si + 1}: temperature", i_, r_, 1e-9)) if callback else None
            # each pass starts from the last one's fields (warm start)
            e = CD.solve_film_z(labels, sig, kvox=sv, h_m=h, t_bottom=1.0, t_top=0.0, tol=1e-9, keep_faces=True,
                                callback=cbe, should_stop=should_stop, x0=u_prev)
            u_prev = e["u"]
            J1 = float(e["q_top"])                                    # A/m^2 per volt
            V = val / J1 if control == "current" else val
            phi = e["u"] * V
            qv = dissipation(phi, e["faces"], e["gb"], e["gt"], h, V, 0.0, sv)
            P_area = float(qv.sum()) * h ** 3 / (nx * ny * h * h)     # W/m^2
            Tsink = Tamb + Rth * P_area
            th = CD.solve_film_z(labels, np.asarray(k_vals, float), rpair=rpair, cmask=cmask, rcpair=rcpair,
                                 source=qv, h_m=h, t_bottom=Tsink, t_top=Tsink, tol=1e-9, callback=cbt,
                                 should_stop=should_stop, x0=th_prev)
            Tn = th["u"]
            th_prev = Tn
            q_out = float(th["q_bottom"]) + float(th["q_top"])                 # W/m^2 leaving through the faces
            d = Tn - T
            dT = float(np.abs(d).max())
            # Near runaway the passes converge slowly, nearly all of the change
            # in the uniform rise of the faces: Aitken's extrapolation of that
            # mode (its contraction rho from two passes) jumps to its limit.
            # The test below still asks one plain pass to move < 1 mK.
            m1 = float(d.mean())
            if d_prev is not None and dT >= 1e-3:
                m0 = float(d_prev.mean())
                rho = m1 / m0 if m0 != 0.0 else 0.0
                if 0.2 < rho < 0.95:
                    shift = m1 * rho / (1.0 - rho)
                    if float(Tn.max()) + shift > T_RUNAWAY:
                        # its steady state lies above 1000 C: stop at this pass
                        predicted = float(Tn.max()) + shift
                        T = Tn
                        break
                    Tn = Tn + shift
            d_prev = d
            T = Tn
            if not np.isfinite(dT) or float(T.max()) > T_RUNAWAY:
                break
            if dT < 1e-3:
                conv = True
                break
        J = J1 * V
        Tmax = float(T.max())
        # beyond 1000 C: too hot to mean anything; not settling: runaway (a set
        # current heating a conductor whose resistance rises with temperature)
        status = ("too hot" if (not np.isfinite(Tmax) or Tmax > T_RUNAWAY or predicted is not None) else
                  "converged" if conv else "runaway")
        run_away = status != "converged"
        hot = np.unravel_index(int(np.argmax(T)), T.shape)
        st = {"value": val, "V": V, "J_A_m2": J, "R_ohm_m2": V / J if J else None, "sigma_eff": J * L / V if V else None,
              "P_W_m2": P_area, "T_sink": Tsink, "T_mean": float(T.mean()), "T_max": Tmax,
              "dT_micro": Tmax - Tsink, "hot_um": [(c + 0.5) * h * 1e6 for c in hot], "iterations": it,
              "converged": conv, "status": status, "T_predicted": predicted,
              "q_max_W_m3": float(qv.max()), "q_mean_W_m3": float(qv.mean()),
              "heat_out_W_m2": q_out, "balance": abs(q_out - P_area) / max(abs(P_area), 1e-300),
              "above_tg": (float(np.mean(np.concatenate([(T[labels == i] > float(props[i]["tg"])) for i in tg_labels])))
                           if tg_labels else None)}
        steps.append(st)
        log(f"    {'J' if control == 'current' else 'V'} = {val:.4g}: V {V:.4g} V, J {J:.4g} A/m², sink {Tsink:.4g} C, "
            f"T max {Tmax:.5g} C (+{Tmax - Tsink:.3g} K in the film), {it} iterations"
            + ("" if status == "converged" else f" - {'thermal runaway' if status == 'runaway' else 'above 1000 C'}")
            + (f" (its steady state extrapolated to {predicted:.0f} C)" if predicted is not None else ""))
        # the fields of the last step that settled (or of the first, when none
        # does): a step that runs away has no steady field to show
        if keep_fields and (status == "converged" or fields is None):
            fields = {"T": T.astype(np.float32), "q": qv.astype(np.float32),
                      "J": current_density(phi, e["faces"], e["gb"], e["gt"], h, V, 0.0).astype(np.float32),
                      "phi": phi.astype(np.float32), "step": si}
        if run_away:
            runaway = status
            break
    out = {"steps": steps, "stopped": runaway or None, "control": control, "T_amb": Tamb, "R_th": Rth,
           "thickness_um": L * 1e6, "seconds": time.time() - t0, "lumped": lumped(steps, Rth, control)}
    if fields is not None:
        out["fields"] = fields
    return out
