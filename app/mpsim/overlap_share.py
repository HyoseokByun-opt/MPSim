"""Auto for the largest share of the particle voxels that may lie in two
particles (progressive overlap, Structure - Gap between particles).

An overlap between two particles is a neck between them. In conduction
through particles in contact the flux between two particles is carried by
the contact spot and grows with its radius once alpha rho / a >> 1 (alpha
the particle/matrix conductivity ratio, rho the spot radius, a the particle
radius), while at a point contact it grows only as ln alpha: G. K. Batchelor,
R. W. O'Brien, "Thermal or electrical conduction through a granular
material", Proc. R. Soc. Lond. A 355 (1682), 313-333 (1977),
doi:10.1098/rspa.1977.0100. An overlap therefore matters little for a filler
close to the matrix and a great deal for a conductive one. In particle
simulations an overlap below 1 % of the diameter is taken to leave bulk
results unchanged (D. M. Hanes, O. R. Walton, Powder Technol. 109 (1-3),
133-144 (2000), doi:10.1016/S0032-5910(99)00232-6; M. Paulick, M. Morgeneyer,
A. Kwade, Powder Technol. 283, 66-76 (2015),
doi:10.1016/j.powtec.2015.03.040) - far below the one voxel a voxel RVE can
represent - and an overlap allowed for numerical reasons raises contact
conduction (A. B. Morris, S. Pannala, Z. Ma, C. M. Hrenya, AIChE J. 62 (12),
4526-4535 (2016), doi:10.1002/aic.15331).

How much a share changes the result was measured with this program (2026-10,
study ov_sens): touching spheres, a packing without overlap against packings
at a lower content whose spheres were grown in place to the same content
(as progressive overlap packs them at a smaller size and draws them whole),
so that neighbours overlap; 4-6 seeds each, the finite-volume conductivity
along z (and FANS C_zzzz for the stiffness). The change of the mean in %
against the share (% of the particle voxels in two particles):

- at 50 vol% (0.78 of random close packing) the packing without overlap has
  few contacts and the overlap makes them: +1.2 % for a contrast of 7 at
  0.012 % shared, +19 % for 150;
- at 60 vol% (0.94 of it) the packing without overlap is near jamming with a
  contact network of its own, and the overlap changes far less: +1.2 % for 7
  at 1.2 % shared; for high contrasts the arrangement itself moves the result
  by up to 10 % either way.

Progressive overlap acts near jamming, where the second applies; the tables
are blended by the loading's fraction of random close packing (0.64).

Auto = the largest share whose change stays within EPS (1 %) for every
analysis selected and every particle phase, at its contrast, its size in
voxels and the loading; at least 0 (no overlap), at most the largest share
measured.
"""
from __future__ import annotations

import math

EPS = 1.0                       # % - the change of a result the overlap may make
RCP = 0.64                      # random close packing of equal spheres
# 50 vol%: change of k in % (6 seeds at D = 8, 4 at D = 12); the column at
# share 0 is the 0.49 -> 0.50 growth (no voxel shared), about a fifth of the
# change at 0.012 % for every contrast
SHARES_50 = [0.0, 0.012, 0.18, 0.68, 1.39]
CHANGE_50 = {
    8: {1.5: [0.004, 0.02, 0.04, 0.06, 0.10],
        3.0: [0.05, 0.24, 0.53, 0.76, 1.11],
        7.0: [0.25, 1.22, 2.77, 3.95, 5.36],
        30.0: [1.48, 6.61, 14.91, 20.76, 26.99],
        150.0: [4.08, 19.22, 42.33, 58.39, 75.18],
        1000.0: [7.52, 37.01, 78.63, 108.51, 138.03]},
    12: {1.5: [0.004, 0.02, 0.04, 0.07, 0.09],
         3.0: [0.05, 0.22, 0.56, 0.91, 1.16],
         7.0: [0.25, 1.22, 3.12, 4.95, 6.10],
         30.0: [1.5, 7.07, 18.17, 28.74, 34.22],
         150.0: [4.5, 21.36, 56.53, 95.29, 110.98],
         1000.0: [10.0, 48.63, 131.24, 231.53, 266.52]},
}
# 60 vol%, D = 8, 4 seeds: the largest |change| up to each share (the
# arrangement moves high contrasts both ways)
SHARES_60 = [0.0, 0.04, 0.377, 1.204, 2.051]
CHANGE_60 = {1.5: [0.0, 0.02, 0.03, 0.06, 0.08],
             3.0: [0.0, 0.11, 0.11, 0.44, 0.57],
             7.0: [0.0, 0.30, 0.38, 1.16, 1.62],
             30.0: [0.0, 0.60, 4.24, 4.24, 4.24],
             150.0: [0.0, 0.57, 10.38, 10.38, 10.38],
             1000.0: [0.0, 0.47, 13.37, 13.37, 13.37]}
# the analyses whose result goes through the particles, and the property
# whose ratio (particle / matrix) sets the contrast
CONDUCTION = {"thermal": "k", "electrical": "sigma", "joule": "sigma", "emi": "sigma",
              "dielectric": "eps_r", "magnetic": "mu_r", "radiation": "k", "moisture": "D_w"}
STIFFNESS = {"cte": "E", "mechtest": "E", "thermomech": "E"}
# flow and connectivity of the pore space: an overlap opens a throat between
# two pores - no share is small enough
PORE_SPACE = {"permeability", "filtration", "tortuosity", "acoustics", "pore_network", "porosimetry", "percolation"}


def effective_contrast(ratio, stiffness=False):
    """The contrast at which the conduction tables are read. Insulating
    particles (ratio < 1) change little: at 1.4 % shared, -1.5 % for a ratio
    of 0.001 and +0.1 % for 0.1 (measured, 50 vol%), as a conductive ratio of
    3 and 1.5. Stiffness changes as conduction at about the square root of
    the modulus ratio (measured, 50 vol%: a ratio of 23 gave +1.4 % / +2.2 %
    at 0.18 / 1.4 % shared, 127 gave +5.7 % / +7.7 %)."""
    r = float(ratio)
    if math.isinf(r):
        return 1e3
    if not r > 0.0:
        return 3.0
    if r < 1.0:
        return 1.0 + (2.0 / 3.0) * min(math.log10(1.0 / r), 3.0)
    return math.sqrt(r) if stiffness else r


def _by_contrast(tab, a):
    """A table's row for contrast a (>= 1): log-interpolated between the
    measured contrasts; towards no contrast as (a - 1)^2 (3 -> 1.5 measured
    about x0.07)."""
    al = sorted(tab)
    if a <= al[0]:
        f = ((a - 1.0) / (al[0] - 1.0)) ** 2
        return [f * v for v in tab[al[0]]]
    if a >= al[-1]:
        return list(tab[al[-1]])
    for lo, hi in zip(al, al[1:]):
        if lo <= a <= hi:
            t = math.log(a / lo) / math.log(hi / lo)
            return [math.exp((1 - t) * math.log(max(x, 1e-6)) + t * math.log(max(y, 1e-6)))
                    for x, y in zip(tab[lo], tab[hi])]
    return list(tab[al[-1]])


def _at(shares, change, s):
    """A curve read at share s (linear, held at its last point)."""
    if s >= shares[-1]:
        return change[-1]
    for (s0, c0), (s1, c1) in zip(zip(shares, change), zip(shares[1:], change[1:])):
        if s0 <= s <= s1:
            return c0 + (c1 - c0) * (s - s0) / (s1 - s0)
    return change[0]


def curve(alpha, d_vox, phi):
    """(shares, change in %) for a contrast alpha (>= 1), particles d_vox
    voxels across and a loading phi: the 50 vol% table (between D = 8 and
    12, held at the nearer outside) and the 60 vol% one blended by phi / RCP
    from 0.78 to 0.94."""
    a = max(1.0, float(alpha))
    d = min(max(float(d_vox), 8.0), 12.0)
    td = math.log(d / 8.0) / math.log(12.0 / 8.0)
    c50 = [(1 - td) * x + td * y for x, y in zip(_by_contrast(CHANGE_50[8], a), _by_contrast(CHANGE_50[12], a))]
    c60 = _by_contrast(CHANGE_60, a)
    t = min(max((float(phi) / RCP - 0.78) / (0.94 - 0.78), 0.0), 1.0)
    top = SHARES_60[-1] if t >= 1.0 else SHARES_50[-1]
    grid = sorted({x for x in SHARES_50 + SHARES_60 if x <= top})
    return grid, [(1 - t) * _at(SHARES_50, c50, x) + t * _at(SHARES_60, c60, x) for x in grid]


def allowed_share(alpha, d_vox, phi=0.6, eps=EPS):
    """The largest share (% of the particle voxels) whose change stays within
    eps %: 0 when even closing the gaps below a voxel changes more."""
    sh, ch = curve(alpha, d_vox, phi)
    if ch[0] >= eps:
        return 0.0
    for (s0, c0), (s1, c1) in zip(zip(sh, ch), zip(sh[1:], ch[1:])):
        if c1 >= eps:
            return s0 + (s1 - s0) * (eps - c0) / (c1 - c0)
    return sh[-1]


def auto_share(spec, plan):
    """Auto for a run: {"pct", "governs" (what sets it), "rows" (every
    analysis x phase considered), "text"}."""
    from .rve import feature_stats
    a0 = spec.get("analyses") or []
    an = {k for k, v in a0.items() if v} if isinstance(a0, dict) else set(a0)
    h = float(plan.get("h_um") or 0.0) or 1.0
    mprops = (spec.get("matrix") or {}).get("props") or {}
    feats = feature_stats([{"name": ph["name"], "shape": ph["shape"], "size_um": ph["size_um"], "dist": ph["dist"],
                            "vf": ph["vf"], "shell": ph["shell"], "overlap": ph["overlap"], "gap_frac": ph["gap_frac"]}
                           for ph in spec["phases"]])
    # the loading the packing has to reach: the particles that may not overlap
    phi = sum(float(ph["vf"]) for ph in spec["phases"] if not ph.get("overlap"))
    rows = []
    for ph, f in zip(spec["phases"], feats):
        if ph.get("overlap") or f.get("network") or not f.get("mean_volume"):
            continue                       # a phase whose particles may overlap anyway
        d_vox = (6.0 * f["mean_volume"] / math.pi) ** (1.0 / 3.0) / h
        props = ph.get("props") or {}
        for a in sorted(an):
            if a in PORE_SPACE:
                rows.append({"analysis": a, "phase": ph["name"], "ratio": None, "alpha": None, "d_vox": d_vox,
                             "phi": phi, "pct": 0.0, "why": "flow through the pore space: an overlap opens a throat"})
                continue
            key = CONDUCTION.get(a) or STIFFNESS.get(a)
            if key is None:
                continue
            pm, pp = mprops.get(key), props.get(key)
            if pm in (None, 0) or pp is None:
                continue
            ratio = float(pp) / float(pm)
            al = effective_contrast(ratio, stiffness=a in STIFFNESS)
            rows.append({"analysis": a, "phase": ph["name"], "ratio": ratio, "alpha": al, "d_vox": d_vox,
                         "phi": phi, "pct": round(allowed_share(al, d_vox, phi), 4)})
    if not rows:
        return {"pct": SHARES_50[-1], "governs": None, "rows": [],
                "text": f"{SHARES_50[-1]:g} % - no analysis selected goes through the particles' contacts "
                        f"(the largest share measured)"}
    g = min(rows, key=lambda r: r["pct"])
    what = (f"{g['analysis']}, {g['phase']}" + (f" at {g['ratio']:.3g}x the matrix" if g.get("ratio") is not None else "")
            + f", {g['d_vox']:.1f} voxels across, {100 * phi:.0f} vol% of particles")
    top = max(SHARES_50[-1], SHARES_60[-1]) if g["pct"] >= SHARES_60[-1] - 1e-9 else SHARES_50[-1]
    text = ((f"{g['pct']:.2g} % - {what}: the largest share measured, which changes the result by less than {EPS:g} %"
             if g["pct"] >= top - 1e-9 else
             f"{g['pct']:.2g} % - {what}: a larger share changes the result by more than {EPS:g} %")
            if g["pct"] > 0 else f"no overlap - {what}: "
            + (g.get("why") or f"any overlap changes the result by more than {EPS:g} %"))
    return {"pct": g["pct"], "governs": g, "rows": rows, "text": text}
