"""Thermo-elastic homogenisation by the FFT (Lippmann-Schwinger) method.

Same scheme as the NTE study (Moulinec-Suquet 1994, conjugate-gradient form of
Zeman et al. 2010): the compatible strain fluctuation solves

    G : C : e~ = - G : C : (E - alpha(x) dT I)

where G is the orthogonal projection onto compatible, zero-mean strain fields.
Six unit strains give the stiffness C*, one thermal load gives alpha*:

    alpha* = - C*^-1 . <sigma_th> / dT

Differences from the NTE code: any number of phases, memory-lean 6-component
storage, voids handled as a phase of vanishing stiffness, and the stress
average reported for every load so C* symmetry can be checked.

Voigt order: xx, yy, zz, yz, xz, xy, with engineering shear strain.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import fft as sfft

WEIGHT = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def lame(E, nu):
    E = np.asarray(E, float)
    nu = np.asarray(nu, float)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


class ElasticFFT:
    def __init__(self, labels, E, nu, void_ratio=1e-4, workers=-1):
        E = np.asarray(E, float).copy()
        nu = np.asarray(nu, float).copy()
        solid = E > 0
        if not solid.any():
            raise ValueError("No phase has a non-zero stiffness")
        self.void_phases = np.nonzero(~solid)[0].tolist()
        E[~solid] = void_ratio * E[solid].min()
        nu[~solid] = 0.2
        lam_p, mu_p = lame(E, nu)
        self.E_used = E
        self.nu_used = nu
        labels = np.asarray(labels)
        self.lam = lam_p[labels]
        self.mu = mu_p[labels]
        self.shape = labels.shape
        self.workers = workers
        n0, n1, n2 = self.shape
        f0 = np.fft.fftfreq(n0)
        f1 = np.fft.fftfreq(n1)
        f2 = np.fft.rfftfreq(n2)
        norm = np.sqrt(f0[:, None, None] ** 2 + f1[None, :, None] ** 2 + f2[None, None, :] ** 2)
        norm[0, 0, 0] = 1.0
        self.n = [(f0[:, None, None] / norm), (f1[None, :, None] / norm),
                  (f2[None, None, :] / norm)]

    # ------------------------------------------------------------------
    def C(self, e):
        tr = e[0] + e[1] + e[2]
        out = np.empty_like(e)
        lt = self.lam * tr
        for c in range(3):
            out[c] = 2.0 * self.mu * e[c] + lt
        for c in range(3, 6):
            out[c] = 2.0 * self.mu * e[c]
        return out

    def G(self, t):
        w = self.workers
        T = [sfft.rfftn(t[c], workers=w) for c in range(6)]
        nx, ny, nz = self.n
        vx = T[0] * nx + T[5] * ny + T[4] * nz
        vy = T[5] * nx + T[1] * ny + T[3] * nz
        vz = T[4] * nx + T[3] * ny + T[2] * nz
        del T
        nv = nx * vx + ny * vy + nz * vz
        comps = [
            2.0 * nx * vx - nx * nx * nv,
            2.0 * ny * vy - ny * ny * nv,
            2.0 * nz * vz - nz * nz * nv,
            ny * vz + nz * vy - ny * nz * nv,
            nx * vz + nz * vx - nx * nz * nv,
            nx * vy + ny * vx - nx * ny * nv,
        ]
        out = np.empty((6,) + self.shape)
        for c in range(6):
            comps[c][0, 0, 0] = 0.0
            out[c] = sfft.irfftn(comps[c], s=self.shape, workers=w)
        return out

    @staticmethod
    def dot(a, b):
        return float(sum(WEIGHT[c] * np.vdot(a[c], b[c]) for c in range(6)))

    # ------------------------------------------------------------------
    def solve(self, E_voigt, alpha_field=None, tol=1e-5, maxiter=3000,
              callback=None, should_stop=None):
        """Mean stress for macroscopic strain E (Voigt, engineering shear) and
        an optional isotropic eigenstrain field alpha(x) (dT = 1)."""
        eps0 = np.empty((6,) + self.shape)
        for c in range(6):
            val = E_voigt[c] if c < 3 else 0.5 * E_voigt[c]
            eps0[c].fill(val)
        if alpha_field is not None:
            for c in range(3):
                eps0[c] -= alpha_field
        b = -self.G(self.C(eps0))
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        rs = self.dot(r, r)
        bn = rs ** 0.5
        it, res = 0, 0.0
        t0 = time.time()
        if bn > 0:
            for it in range(1, maxiter + 1):
                Ap = self.G(self.C(p))
                pAp = self.dot(p, Ap)
                if pAp <= 0:
                    break
                a = rs / pAp
                x += a * p
                r -= a * Ap
                del Ap
                rs_new = self.dot(r, r)
                res = rs_new ** 0.5 / bn
                if res < tol:
                    rs = rs_new
                    break
                if callback is not None and it % 5 == 0:
                    callback(it, res)
                if should_stop is not None and it % 10 == 0 and should_stop():
                    raise InterruptedError
                p = r + (rs_new / rs) * p
                rs = rs_new
        eps0 += x
        sig = self.C(eps0)
        sbar = np.array([float(sig[c].mean()) for c in range(6)])
        return sbar, {"iterations": it, "residual": float(res),
                      "converged": bool(bn == 0 or res < tol),
                      "seconds": time.time() - t0}


def homogenize(labels, E, nu, alpha, tol=1e-5, maxiter=3000, void_ratio=1e-4,
               progress=None, should_stop=None, workers=-1):
    """Stiffness C* [GPa] and CTE alpha* [ppm/K] of a periodic voxel RVE."""
    sol = ElasticFFT(labels, E, nu, void_ratio=void_ratio, workers=workers)
    Cstar = np.zeros((6, 6))
    stats = []
    names = ["εxx", "εyy", "εzz", "γyz", "γxz", "γxy"]
    for k in range(6):
        Ev = np.zeros(6)
        Ev[k] = 1.0

        def cb(it, res, k=k):
            if progress:
                progress(f"load {names[k]}", k, it, res)
        sbar, st = sol.solve(Ev, None, tol, maxiter, cb, should_stop)
        Cstar[:, k] = sbar
        st["load"] = names[k]
        stats.append(st)
    alpha = np.asarray(alpha, float)
    afield = alpha[np.asarray(labels)]

    def cbt(it, res):
        if progress:
            progress("thermal", 6, it, res)
    sth, st = sol.solve(np.zeros(6), afield, tol, maxiter, cbt, should_stop)
    st["load"] = "ΔT"
    stats.append(st)
    asym = float(np.abs(Cstar - Cstar.T).max() / max(np.abs(Cstar).max(), 1e-300))
    Csym = 0.5 * (Cstar + Cstar.T)
    alpha_star = -np.linalg.solve(Csym, sth)
    return {"C": Csym, "alpha": alpha_star, "asymmetry": asym, "stats": stats,
            "E_used": sol.E_used.tolist(), "nu_used": sol.nu_used.tolist(),
            "void_phases": sol.void_phases}


def engineering_constants(C):
    """Directional moduli, all six Poisson ratios, shear moduli, the Voigt,
    Reuss and Hill averages and the universal anisotropy index
    A_U = 5 G_V/G_R + K_V/K_R - 6 (Ranganathan & Ostoja-Starzewski 2008;
    zero for an isotropic material)."""
    C = np.asarray(C, float)
    S = np.linalg.inv(C)
    E = [1.0 / S[i, i] for i in range(3)]
    Gm = [1.0 / S[i, i] for i in range(3, 6)]
    # nu_ij = -eps_j / eps_i under uniaxial stress along i
    nu = {"nu_xy": -S[1, 0] / S[0, 0], "nu_xz": -S[2, 0] / S[0, 0],
          "nu_yx": -S[0, 1] / S[1, 1], "nu_yz": -S[2, 1] / S[1, 1],
          "nu_zx": -S[0, 2] / S[2, 2], "nu_zy": -S[1, 2] / S[2, 2]}
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2])
          + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    KR = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                 + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    K = 0.5 * (KV + KR)
    Gh = 0.5 * (GV + GR)
    Eh = 9 * K * Gh / (3 * K + Gh)
    nuh = (3 * K - 2 * Gh) / (2 * (3 * K + Gh))
    au = 5.0 * GV / GR + KV / KR - 6.0 if GR > 0 and KR > 0 else None
    return {"E": E, "G": Gm, **nu, "K_voigt": KV, "K_reuss": KR, "G_voigt": GV, "G_reuss": GR,
            "K_hill": K, "G_hill": Gh, "E_hill": Eh, "nu_hill": nuh,
            "anisotropy_index": au, "S": S}


def directional_modulus(S, n):
    """Young's modulus along unit vector n from the Voigt compliance
    (engineering shear): 1/E(n) = m^T S m, m = (n1², n2², n3², n2n3, n1n3, n1n2)."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    m = np.array([n[0] ** 2, n[1] ** 2, n[2] ** 2, n[1] * n[2], n[0] * n[2], n[0] * n[1]])
    inv = float(m @ np.asarray(S, float) @ m)
    return 1.0 / inv if inv > 0 else float("nan")


def polar_moduli(S, n_angles=73):
    """E(theta) in the xy, xz and yz planes, for polar plots."""
    th = np.linspace(0.0, 2.0 * np.pi, n_angles)
    out = {"theta_deg": np.degrees(th).tolist()}
    for plane, (i, j) in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
        vals = []
        for t in th:
            v = np.zeros(3)
            v[i], v[j] = np.cos(t), np.sin(t)
            vals.append(directional_modulus(S, v))
        out[plane] = vals
    return out


def levin_alpha(S, alpha1, alpha2, K1, K2):
    """Levin's exact identity for two phases: alpha* from S* alone (Voigt,
    engineering shear). Holds for ANY two-phase microstructure with isotropic
    phases, so it checks the thermal load against the six mechanical ones."""
    S = np.asarray(S, float)
    factor = (alpha2 - alpha1) / (1.0 / (3.0 * K2) - 1.0 / (3.0 * K1))
    out = np.zeros(6)
    for i in range(6):
        sk = S[i, 0] + S[i, 1] + S[i, 2]
        out[i] = factor * (sk - (1.0 / (3.0 * K1) if i < 3 else 0.0))
        if i < 3:
            out[i] += alpha1
    return out
