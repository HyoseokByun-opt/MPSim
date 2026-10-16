"""A tension or compression test of the composite, as on a universal testing
machine: the force-displacement line of a specimen of the composite and the
stresses inside the RVE at the test load.

The specimen is a uniform bar (or a film between platens) of the composite:
the RVE is a material point inside it, away from the grips and the free
surfaces. The macroscopic state follows from the effective stiffness C* of
the elasticity analysis, which this test builds on (no solve is repeated):

  free lateral faces (uniaxial stress) - every mean stress but the axial one
      is zero: the axial strain e gives the strains S* (e / S*_aa) and the
      apparent modulus is 1 / S*_aa (Young's modulus along the axis); the
      Poisson ratios are -e_b / e_a;
  confined (uniaxial strain) - every mean strain but the axial one is zero:
      a film bonded to rigid platens much wider than it is thick, or a die;
      the apparent modulus is C*_aa (the constrained modulus) and the lateral
      stresses C*_ba e are what the platens or the die hold.

The load is set as a displacement of the specimen's length, a strain, a force
on its loaded area or a pressure. The material is linear elastic with small
strains, so the line is straight, tension and compression differ only in
sign, and every field scales with the load: one more solve on the RVE at the
test's macroscopic strain gives the stress fields (FANS, solvers/fans.py).
A strength of the matrix gives the load at which its most stressed 0.1 % of
volume reaches it (von Mises), an estimate of the onset of yielding or
cracking - not of the ultimate strength, which needs damage and plasticity.
"""
from __future__ import annotations

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}
VOIGT = ["xx", "yy", "zz", "yz", "xz", "xy"]


def macro_state(C, mt):
    """The macroscopic strain and stress (Voigt; engineering shear strain;
    GPa) at the test load, the apparent modulus and what a testing machine
    reads: force (N), displacement (µm), engineering stress (MPa) and
    strain."""
    C = np.asarray(C, float)
    S = np.linalg.inv(C)
    a = AXES[mt["axis"]]
    sign = -1.0 if mt["mode"] == "compression" else 1.0
    if mt["lateral"] == "confined":
        M = float(C[a, a])
        e_unit = np.zeros(6)
        e_unit[a] = 1.0
    else:
        M = float(1.0 / S[a, a])
        e_unit = S[:, a] * M
    s_unit = C @ e_unit
    ctl, val = mt["control"], float(mt["value"])
    if ctl == "strain":
        eps = val / 100.0
    elif ctl == "displacement":
        eps = val * 1e-3 / float(mt["length_mm"])            # µm over mm
    elif ctl == "pressure":
        eps = val * 1e-3 / M                                 # MPa over GPa
    else:
        eps = val / float(mt["area_mm2"]) * 1e-3 / M         # N/mm² = MPa
    eps *= sign
    E_v = e_unit * eps
    sig_v = s_unit * eps
    stress = float(sig_v[a] * 1e3)                           # MPa
    out = {"axis": mt["axis"], "mode": mt["mode"], "lateral": mt["lateral"], "control": ctl, "value": val,
           "modulus_GPa": M, "strain": float(eps), "stress_MPa": stress,
           "force_N": stress * float(mt["area_mm2"]), "displacement_um": float(eps) * float(mt["length_mm"]) * 1e3,
           "length_mm": float(mt["length_mm"]), "area_mm2": float(mt["area_mm2"]),
           "E_voigt": E_v.tolist(), "sigma_voigt_MPa": (sig_v * 1e3).tolist(),
           "energy_kJ_m3": float(0.5 * np.dot(sig_v, E_v) * 1e6)}  # GPa x strain = GJ/m³
    lat = [b for b in range(3) if b != a]
    if mt["lateral"] == "confined":
        out["lateral_stress_MPa"] = {"xyz"[b]: float(sig_v[b] * 1e3) for b in lat}
    else:
        out["poisson"] = {"xyz"[a] + "xyz"[b]: float(-E_v[b] / E_v[a]) for b in lat}
        out["lateral_strain"] = {"xyz"[b]: float(E_v[b]) for b in lat}
        # the cross-section the specimen ends with (small strains)
        out["area_change"] = float(E_v[lat[0]] + E_v[lat[1]])
    return out


def von_mises(s):
    """von Mises stress of a Voigt stress field (6, ...)."""
    return np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
                   + 3.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))


def onset(state, vm_matrix_q, strength_MPa):
    """The test load at which the matrix's most stressed 0.1 % reaches its
    strength: the fields scale with the load (linear elasticity)."""
    if not strength_MPa or not vm_matrix_q or vm_matrix_q <= 0:
        return None
    f = float(strength_MPa) / float(vm_matrix_q)
    return {"fraction_of_load": f, "force_N": abs(state["force_N"]) * f, "stress_MPa": abs(state["stress_MPa"]) * f,
            "strain": abs(state["strain"]) * f, "displacement_um": abs(state["displacement_um"]) * f,
            "strength_MPa": float(strength_MPa), "matrix_vm_q999_MPa": float(vm_matrix_q)}
