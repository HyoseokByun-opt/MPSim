"""Self-contained HTML report (figures embedded, no external files).

Written for a reader who was not at the keyboard: what was modelled, how
large the RVE was and why, what came out, which checks passed, and which
open-source code produced each number.
"""
from __future__ import annotations

import base64
import html
import json
import math
import os

REFS = [
    ("NASA PuMA", "J. C. Ferguson et al., “Update 3.0 to PuMA: Porous Microstructure Analysis software”, SoftwareX 15 (2021) 100775. "
                  "Finite-element elasticity: P. C. Lopes et al., Comput. Mater. Sci. 219 (2023) 112021."),
    ("PoreSpy", "J. T. Gostick et al., “PoreSpy: A Python toolkit for quantitative analysis of porous media images”, JOSS 4 (2019) 1296. "
                "SNOW network extraction: J. T. Gostick, Phys. Rev. E 96 (2017) 023307. "
                "Morphological drainage: M. Hilpert, C. T. Miller, Adv. Water Resour. 24 (2001) 243."),
    ("OpenPNM", "J. Gostick et al., “OpenPNM: A pore network modeling package”, Comput. Sci. Eng. 18 (2016) 60."),
    ("PyVista / VTK", "C. B. Sullivan, A. A. Kaszynski, “PyVista: 3D plotting and mesh analysis through a streamlined interface for VTK”, JOSS 4 (2019) 1450."),
    ("scikit-rf", "A. Arsenovic et al., “scikit-rf: An open source Python package for microwave network creation, analysis, and calibration”, IEEE Microwave Magazine 23 (2022) 98."),
    ("SciPy / NumPy / Numba", "P. Virtanen et al., Nature Methods 17 (2020) 261; C. R. Harris et al., Nature 585 (2020) 357; S. K. Lam et al., LLVM-HPC (2015)."),
    ("Homogenisation theory", "Z. Hashin, S. Shtrikman, J. Appl. Phys. 33 (1962) 3125; V. M. Levin, Mech. Solids 2 (1967) 58; "
                              "R. A. Schapery, J. Compos. Mater. 2 (1968) 380; T. Mori, K. Tanaka, Acta Metall. 21 (1973) 571; "
                              "S. I. Ranganathan, M. Ostoja-Starzewski, Phys. Rev. Lett. 101 (2008) 055504."),
    ("Porous-media acoustics", "D. L. Johnson, J. Koplik, R. Dashen, J. Fluid Mech. 176 (1987) 379; Y. Champoux, J.-F. Allard, "
                               "J. Appl. Phys. 70 (1991) 1975; J.-F. Allard, N. Atalla, Propagation of Sound in Porous Media, 2nd ed., Wiley (2009)."),
    ("Rarefied gas transport", "M. G. Kaganer, Thermal Insulation in Cryogenic Engineering, IPST (1969); "
                               "W. G. Pollard, R. D. Present, Phys. Rev. 73 (1948) 762 (Bosanquet interpolation)."),
]

CSS = """
body{font:14px/1.6 "Segoe UI",Roboto,Arial,sans-serif;color:#1f2328;margin:0;background:#ffffff}
.wrap{max-width:1120px;margin:0 auto;padding:30px 28px 70px}
h1{font-size:25px;margin:0 0 4px;color:#0d1117}h2{font-size:18px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid #0b62c4;color:#0d1117}
h3{font-size:15px;margin:22px 0 6px;color:#0d1117}.sub{color:#59636e;font-size:12.5px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin:16px 0}
.card{background:#f6f8fa;border:1px solid #d8dee4;border-radius:8px;padding:10px 12px}
.card .v{font:600 20px Consolas,monospace;color:#0d1117}.card .l{color:#0b62c4;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em}
.card .d{color:#59636e;font-size:11.5px}
table{border-collapse:collapse;width:100%;background:#fff;margin:6px 0 14px;font-size:13px}
th,td{border:1px solid #d8dee4;padding:5px 8px;text-align:left;vertical-align:top}th{background:#f6f8fa;font-weight:600}
td.n{text-align:right;font-family:Consolas,monospace;white-space:nowrap}
.pass{color:#1a7f37;font-weight:700}.warn{color:#9a6700;font-weight:700}.fail{color:#cf222e;font-weight:700}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.figs figure{margin:0;background:#fff;border:1px solid #d8dee4;border-radius:8px;padding:6px}
.figs img{width:100%;display:block}.figs figcaption{font-size:12px;color:#424a53;padding:4px}
pre{background:#f6f8fa;color:#1f2328;border:1px solid #d8dee4;padding:10px;border-radius:6px;overflow:auto;font-size:12px}
.note{background:#f0f6ff;border-left:3px solid #0b62c4;padding:8px 12px;margin:10px 0;font-size:13px}
svg{max-width:100%}
"""


def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return " / ".join(_fmt(x) for x in v) + (f" {unit}" if unit else "")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if not math.isfinite(f):
        return "—"
    a = abs(f)
    s = f"{f:.4g}" if (a == 0 or 1e-3 <= a < 1e5) else f"{f:.3e}"
    return s + (f" {unit}" if unit else "")


def _img(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _svg_lines(series, xlabel, ylabel, xlog=False, ylog=False, ymin=None, ymax=None, W=760, H=320):
    """Minimal SVG line chart: series = [(label, xs, ys, colour, dash)]."""
    L, B, T, Rm = 64, 44, 26, 18
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    pts = [(tx(x), ty(y)) for _, xs, ys, _, _ in series for x, y in zip(xs, ys)
           if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
           and (not xlog or x > 0) and (not ylog or y > 0)]
    if not pts:
        return ""
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0 = ty(ymin) if ymin is not None else min(p[1] for p in pts)
    y1 = ty(ymax) if ymax is not None else max(p[1] for p in pts)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - Rm)
    Y = lambda v: H - B - (v - y0) / (y1 - y0) * (H - B - T)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="background:#fff;border:1px solid #d8dee4;border-radius:8px" font-family="Segoe UI" font-size="11">']
    for k in range(6):
        gx = x0 + (x1 - x0) * k / 5
        gy = y0 + (y1 - y0) * k / 5
        lx = f"1e{gx:.1f}" if xlog else _fmt(gx)
        ly = f"1e{gy:.1f}" if ylog else _fmt(gy)
        out.append(f'<line x1="{X(gx):.1f}" x2="{X(gx):.1f}" y1="{T}" y2="{H-B}" stroke="#eef1f4"/>'
                   f'<text x="{X(gx):.1f}" y="{H-B+15}" text-anchor="middle" fill="#59636e">{lx}</text>'
                   f'<line x1="{L}" x2="{W-Rm}" y1="{Y(gy):.1f}" y2="{Y(gy):.1f}" stroke="#eef1f4"/>'
                   f'<text x="{L-6}" y="{Y(gy)+4:.1f}" text-anchor="end" fill="#59636e">{ly}</text>')
    lx0 = L + 8
    for label, xs, ys, col, dash in series:
        d = []
        for x, y in zip(xs, ys):
            if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)) or (xlog and x <= 0) or (ylog and y <= 0):
                continue
            d.append(("M" if not d else "L") + f"{X(tx(x)):.1f},{max(T, min(H-B, Y(ty(y)))):.1f}")
        out.append(f'<path d="{" ".join(d)}" fill="none" stroke="{col}" stroke-width="2.2" stroke-dasharray="{dash or ""}"/>')
        out.append(f'<rect x="{lx0}" y="9" width="14" height="3" fill="{col}"/><text x="{lx0+18}" y="14" fill="#1f2328">{html.escape(label)}</text>')
        lx0 += 30 + 6.2 * len(label)
    out.append(f'<text x="{L+(W-L-Rm)/2}" y="{H-8}" text-anchor="middle" fill="#1f2328">{html.escape(xlabel)}</text>'
               f'<text x="14" y="{T+(H-B-T)/2}" transform="rotate(-90 14 {T+(H-B-T)/2})" text-anchor="middle" fill="#1f2328">{html.escape(ylabel)}</text></svg>')
    return "".join(out)


def _flow_model(m):
    """The resin's flow model in words."""
    m = m or {}
    t = m.get("type", "newtonian")
    if t == "power":
        return f"power law, K {_fmt(m.get('K'))} Pa·sⁿ, n {_fmt(m.get('n'))}"
    if t in ("carreau", "cross"):
        return (f"{t.capitalize()}, μ₀ {_fmt(m.get('mu0'))} Pa·s, μ∞ {_fmt(m.get('mu_inf'))} Pa·s, "
                f"λ {_fmt(m.get('lam'))} s, n {_fmt(m.get('n'))}")
    return f"Newtonian, {_fmt(m.get('mu', 1.0))} Pa·s"


def _table(head, rows):
    h = "<table><tr>" + "".join(f"<th{' class=n' if i else ''}>{c}</th>" for i, c in enumerate(head)) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td{' class=n' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return h + "</table>"


SHAPES = {"sphere": "sphere", "spheroid": "spheroid", "cylinder": "cylinder", "spherocylinder": "spherocylinder",
          "cube": "cube", "cuboid": "cuboid", "superellipsoid": "rounded box (superellipsoid)", "polyhedron": "polyhedron",
          "helix": "helix", "network": "bicontinuous network"}
BACKENDS = {"puma": "NASA PuMA (periodic finite elements)", "puma_fv": "NASA PuMA (finite volumes)"}


def write(out_dir, r):
    e = html.escape
    spec, plan, props = r["spec"], r["plan"], r["properties"]
    P = []
    P.append(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{e(r['name'])} — analysis report</title>"
             f"<style>{CSS}</style></head><body><div class='wrap'>")
    P.append(f"<h1>{e(r['name'])}</h1><div class='sub'>MPSim {e(r['version'])} · {e(r['created'])} · "
             f"elapsed {r['elapsed_s']:.0f} s · solver {e(BACKENDS.get(r['backend'], 'built-in verification solver'))}</div>")
    cc = r["check_counts"]
    P.append(f"<div class='note'>Verification: <span class='pass'>{cc['pass']} passed</span> · "
             f"<span class='warn'>{cc['warn']} warnings</span> · <span class='fail'>{cc['fail']} failed</span></div>")
    P.append("<div class='cards'>")
    for h in r["headline"]:
        d = ("xx/yy/zz " + " / ".join(_fmt(x) for x in h["diag"])) if h.get("diag") else (h.get("sub") or "")
        P.append(f"<div class='card'><div class='l'>{e(h['label'])}</div><div class='v'>{_fmt(h['value'])}</div>"
                 f"<div class='d'>{e(h['unit'])} {e(d)}</div></div>")
    P.append("</div>")

    # ---------------------------------------------------------------- input
    P.append("<h2>1. Input</h2><table><tr><th>Role</th><th>Name</th><th>Shape and size</th><th>Content</th><th>Principal properties</th></tr>")
    mp = spec["matrix"]["props"]
    P.append(f"<tr><td>Matrix</td><td>{e(spec['matrix']['name'])}</td><td>—</td><td class='n'>{_fmt(100*r['composition']['labels'][0]['vf'])} vol%</td>"
             f"<td>k {_fmt(mp['k'])} W/m·K, σ {_fmt(mp['sigma'])} S/m, εr {_fmt(mp['eps_r'])}, E {_fmt(mp['E'])} GPa, α {_fmt(mp['alpha'])} ppm/K</td></tr>")
    for ph in spec["phases"]:
        sz = ph["size_um"]
        desc = f"{SHAPES[ph['shape']]}, d {_fmt(sz['d'])} µm" + (f", aspect ratio {_fmt(sz.get('aspect'))}" if "aspect" in sz else "") + \
               (f", length {_fmt(sz.get('length'))} µm" if "length" in sz else "") + \
               (f", log-normal CV {ph['dist']['cv']}" if ph["dist"]["type"] == "lognormal" else "") + \
               (f", orientation {ph['orientation']['mode']}" if ph["shape"] not in ("sphere", "network") else "") + \
               (f", shell {_fmt(ph['shell']['thickness_um'])} µm" if ph.get("shell") else "") + \
               (f" × {_fmt(sz['ly'])} × {_fmt(sz['lz'])} µm" if "ly" in sz else "") +                (f", n {_fmt(sz['n'])}" if ph["shape"] == "superellipsoid" else "") +                (f", {sz.get('poly')}" if ph["shape"] == "polyhedron" else "") +                (f", coil {_fmt(sz['coil_d'])} µm, pitch {_fmt(sz['pitch'])} µm, {_fmt(sz['turns'])} turns" if ph["shape"] == "helix" else "") +                (f", R<sub>int</sub> {_fmt(ph['r_int'])} m²K/W" if ph.get("r_int") else "") +                (f", R<sub>c</sub> {_fmt(ph['r_contact'])} m²K/W" if ph.get("r_contact") else "")
        p = ph["props"]
        P.append(f"<tr><td>{'Pore phase' if ph['void'] else ('Network' if ph['shape'] == 'network' else 'Filler')}</td><td>{e(ph['name'])}</td><td>{desc}</td>"
                 f"<td class='n'>{_fmt(100*ph['vf'])} vol%</td><td>k {_fmt(p['k'])} W/m·K, σ {_fmt(p['sigma'])} S/m, εr {_fmt(p['eps_r'])}, "
                 f"E {_fmt(p['E'])} GPa, α {_fmt(p['alpha'])} ppm/K</td></tr>")
    P.append("</table>")

    # ------------------------------------------------------------------ RVE
    P.append(f"<h2>2. RVE model</h2><p>Grid <b>{plan['N']}³</b>, voxel size <b>{_fmt(plan['h_um'])} µm</b>, RVE edge length <b>{_fmt(plan['L_um'])} µm</b> "
             f"(accuracy level {e(plan['quality_label'])}, {plan['seeds']} independent realisation(s)). Recommended voxel size {_fmt(plan['recommended']['h_um'])} µm; "
             f"recommended edge length {_fmt(plan['recommended']['L_um'])} µm, governed by {e(plan['recommended']['reason'])}.</p>")
    for n in plan["notes"]:
        P.append(f"<div class='note'>{e(n)}</div>")
    rz = r["realisations"][0]
    mode = rz.get("contacts_mode") or "apart"
    if mode == "separate":
        ctxt = f"{rz['contacts_removed']} particle contacts were separated by clearing voxels to matrix (contacts = separate)."
    else:
        tp = rz.get("touching_pairs", 0) or 0
        ctxt = (("Particles were kept apart where the loading allows (contacts = apart). " if mode == "apart" else
                 "Particles may touch (contacts = keep). ")
                + (f"{tp:,} voxel neighbour pairs of two particles remain, {rz.get('contact_faces', 0):,} across a face."
                   if tp else "No two particles touch at this resolution."))
    P.append(f"<p>Placement method: {e(rz['route'] or '-')}. {e(ctxt)}</p>")
    P.append(_table(["Region", "Kind", "Volume fraction", "Mass fraction"],
                    [[e(l["name"]), e(l["kind"]), _fmt(100 * l["vf"], "%"), _fmt(100 * l["wt"], "%")] for l in r["composition"]["labels"]]))
    P.append(f"<p>Density {_fmt(r['composition']['density'])} g/cm³ · specific heat {_fmt(r['composition']['cp'])} J/kg·K"
             f" · porosity {_fmt(100*r['composition']['porosity'])} %</p>")

    # -------------------------------------------------------------- results
    P.append("<h2>3. Results</h2>")
    if "porosity" in props:
        p = props["porosity"]
        P.append("<h3>Porosity</h3>" + _table(["Quantity", "Value"], [
            ["Total porosity", _fmt(100 * p["total"], "%")], ["Open porosity (spanning the periodic cell)", _fmt(100 * p["open"], "%")],
            ["Closed porosity", _fmt(100 * p["closed"], "%")],
            ["Connected along x / y / z", " ".join("●" if p["connected_axes"].get(a) else "○" for a in "xyz")]]))
    for key in ("thermal", "electrical", "dielectric", "magnetic"):
        if key not in props:
            continue
        p = props[key]
        P.append(f"<h3>{e(p['title'])} [{e(p['unit'])}]</h3>")
        rows = []
        for i, d in enumerate("xyz"):
            v = p["diag"][i]
            if v is None:
                continue
            ci = p["ci95"][i] if p.get("ci95") else None
            fv = (p.get("fv_diag") or [None] * 3)[i]
            rows.append([f"{d}{d}", _fmt(v), ("±" + _fmt(ci)) if ci else "—", _fmt(fv)])
        rows.append(["<b>isotropic mean</b>", f"<b>{_fmt(p['iso'])}</b>", "", _fmt(p.get("fv_iso"))])
        xc_fe = p.get("xc_method") == "fe"
        meth = ((p.get("stats") or [{}])[0] or {}).get("method")
        prim = {"fe": "PuMA FE", "fv": "PuMA FV", "fv-film": "Film FV"}.get(meth, "Periodic FV")
        other = "PuMA periodic FE" if xc_fe else "Independent periodic FV"
        P.append(_table(["Component", prim, "95 % CI", other], rows))
        if p.get("fv_iso"):
            P.append(f"<p>The same RVE solved with {'PuMA periodic finite elements' if xc_fe else 'an independent periodic finite-volume discretisation'} "
                     f"gives {_fmt(p['fv_iso'])} ({100*(p['fv_iso']/p['iso']-1):+.1f} %). The interval between both values bounds the "
                     f"discretisation error at this resolution and narrows as the resolution increases; on converged references the "
                     f"finite-volume value was the closer one.</p>")
        rs = p.get("resolution")
        if rs:
            P.append(f"<p>Resolution check: the same particles drawn on a {rs.get('grid_fine') or str(rs['N_fine']) + '³'} grid "
                     f"instead of {rs.get('grid') or str(rs['N']) + '³'} give "
                     f"{_fmt(rs['fine'])} in {rs['direction']} ({100*rs['change']:+.2f} % against {_fmt(rs['base'])}); "
                     + (f"extrapolated to fine voxels {_fmt(rs['extrapolated'])}.</p>" if rs.get("extrapolated") is not None
                        else f"not extrapolated: {e(rs.get('extrapolation_note') or 'the change is too large')}.</p>"))
        der = p.get("derived") or {}
        names = {"resistivity_mK_W": ("Thermal resistivity", "m·K/W"), "area_resistance_mm2K_W": ("Area-specific thermal resistance (isotropic)", "mm²·K/W"),
                 "area_resistance_z_mm2K_W": ("Area-specific thermal resistance (through-plane z)", "mm²·K/W"),
                 "diffusivity_mm2_s": ("Thermal diffusivity", "mm²/s"), "volumetric_heat_capacity_MJ_m3K": ("Volumetric heat capacity", "MJ/m³·K"),
                 "resistivity_ohm_m": ("Electrical resistivity", "Ω·m"), "sheet_resistance_ohm_sq": ("Sheet resistance", "Ω/sq"),
                 "capacitance_pF_mm2": ("Capacitance per area (isotropic)", "pF/mm²"), "capacitance_z_pF_mm2": ("Capacitance per area (z)", "pF/mm²"),
                 "anisotropy_inplane_over_z": ("Anisotropy ratio (xx+yy)/2 : zz", "-")}
        drows = [[names[k][0], _fmt(v, names[k][1])] for k, v in der.items() if k in names]
        if drows:
            P.append(f"<p>Layer-based quantities refer to a thickness of {_fmt(der.get('thickness_mm'))} mm.</p>" + _table(["Derived quantity", "Value"], drows))
        if key == "dielectric":
            P.append(f"<p>Dissipation factor tan δ (energy-weighted, first order): {_fmt(p['tan_d_iso'])}</p>")
        if p.get("knudsen"):
            kn = p["knudsen"]
            P.append(f"<p>Rarefied gas in the pores: mean pore diameter {_fmt(kn['d_pore_um'])} µm, mean free path {_fmt(kn['mean_free_path_nm'])} nm, "
                     f"Kn = {_fmt(kn['knudsen_number'])} ({e(kn['regime'])} regime); the gas conductivity was multiplied by {_fmt(kn['gas_factor'])} (Kaganer).</p>")
        fs = (p.get("field_stats") or {})
        if fs.get("enhancement"):
            P.append(_table(["Region", "Mean |E|/E₀", "99th percentile", "Maximum"],
                            [[e(s["label"]), _fmt(s["mean"]), _fmt(s["p99"]), _fmt(s["max"])] for s in fs["enhancement"]]))
        P.append(_table(["Reference model", "Value"], [[e(n), _fmt(v)] for n, v in p["refs"].items()]))
        if p.get("contacts"):
            P.append("<p>Contact resistance between touching particles (every pair of fillers; a pair that was not set "
                     "takes the mean of both fillers' own values)</p>" +
                     _table(["Pair", "R<sub>c</sub> [m²K/W]", "Set for the pair", "Contact voxel faces"],
                            [[e(" – ".join(c["names"])), _fmt(c["r_c"]), "●" if c["set"] else "○", f"{c['faces']:,}"]
                             for c in p["contacts"]]))
    if "cte" in props:
        p = props["cte"]
        c = p["constants"]
        P.append("<h3>Elasticity and thermal expansion</h3>")
        P.append(_table(["Quantity", "x", "y", "z"], [
            ["α [ppm/K]"] + [_fmt(v) for v in p["alpha"][:3]], ["E [GPa]"] + [_fmt(v) for v in c["E"]],
            ["G (yz / xz / xy) [GPa]"] + [_fmt(v) for v in c["G"]],
            ["ν (xy / yz / zx)", _fmt(c.get("nu_xy")), _fmt(c.get("nu_yz")), _fmt(c.get("nu_zx"))]]))
        P.append(_table(["Average", "K [GPa]", "G [GPa]"], [["Voigt", _fmt(c.get("K_voigt")), _fmt(c.get("G_voigt"))],
                                                            ["Reuss", _fmt(c.get("K_reuss")), _fmt(c.get("G_reuss"))],
                                                            ["Hill", _fmt(c["K_hill"]), _fmt(c["G_hill"])]]))
        P.append(f"<p>Volumetric mean α {_fmt(p['alpha_vol'])} ppm/K · Hill Young's modulus {_fmt(c['E_hill'])} GPa, Poisson's ratio {_fmt(c['nu_hill'])} · "
                 f"universal anisotropy index A<sup>U</sup> = {_fmt(c.get('anisotropy_index'))} (zero for isotropy).</p>")
        P.append(_table(["Reference model (α, ppm/K)", "Value"], [[e(n), _fmt(v)] for n, v in p["refs"].items()]))
        at = p.get("above_tg")
        if at:
            ca = at["constants"]
            av = sum(at["alpha"][:3]) / 3.0
            P.append(f"<p>Above Tg (rubbery: {e(', '.join(at['rubbery']))}; ν 0.45, B-bar elements)</p>" +
                     _table(["Quantity", "Below Tg", "Above Tg"],
                            [["α volumetric mean [ppm/K]", _fmt(p["alpha_vol"]), _fmt(av)],
                             ["α xx / yy / zz", " / ".join(_fmt(v) for v in p["alpha"][:3]), " / ".join(_fmt(v) for v in at["alpha"][:3])],
                             ["Hill E [GPa]", _fmt(c["E_hill"]), _fmt(ca["E_hill"])]]))
        P.append("<p>Stiffness matrix C* [GPa] (Voigt order xx, yy, zz, yz, xz, xy)</p><table>" +
                 "".join("<tr>" + "".join(f"<td class='n'>{_fmt(v)}</td>" for v in row) + "</tr>" for row in p["C"]) + "</table>")
        ss = p.get("stress_stats") or {}
        if ss.get("von_mises"):
            hyd = {s["label"]: s for s in ss.get("hydrostatic", [])}
            P.append(f"<p>Thermal stresses for ΔT = {_fmt(p['delta_T'])} K under zero macroscopic strain [MPa]</p>" +
                     _table(["Region", "Mean von Mises", "99th percentile", "Maximum", "Mean hydrostatic"],
                            [[e(s["label"]), _fmt(s["mean"]), _fmt(s["p99"]), _fmt(s["max"]), _fmt(hyd.get(s["label"], {}).get("mean"))]
                             for s in ss["von_mises"]]))
    if "viscosity" in props:
        p = props["viscosity"]
        P.append("<h3>Viscosity and flowability of the uncured compound</h3>")
        P.append(f"<p>Rigid filler fraction φ = {_fmt(100 * p['phi'])} vol%"
                 + (f", bubbles {_fmt(100 * p['bubbles'])} vol%" if p.get("bubbles") else "")
                 + f". Best estimate of the relative viscosity μ/μ<sub>resin</sub> = <b>{_fmt(p['mu_r'])}</b> ({e(p['basis'])}).</p>")
        sh = p.get("shear") or {}
        rows = [[f"Simple shear {k}", _fmt(sh.get(k))] for k in ("yz", "xz", "xy") if sh.get(k) is not None]
        if p.get("extension") is not None:
            rows.append(["Uniaxial extension (as a shear viscosity)", _fmt(p["extension"])])
        rows.append(["<b>RVE mean</b>" + (f" ± {_fmt(p['ci95'])} (95 % CI)" if p.get("ci95") else ""), f"<b>{_fmt(p['mu_r_rve'])}</b>"])
        P.append(_table(["Stokes flow solve on the RVE", "μ/μ<sub>resin</sub>"], rows))
        P.append(_table(["Closed-form model", "μ/μ<sub>resin</sub>"], [[e(n), _fmt(v)] for n, v in p["refs"].items()]))
        jm = p.get("jamming") or {}
        P.append(f"<p>Maximum packing fraction φ<sub>m</sub> = {_fmt(p['phi_m'])} ({e(p['phi_m_basis'])}"
                 + (f"; {jm.get('n_particles')} particles of the same mix jammed in {_fmt(jm.get('seconds'))} s, "
                    f"raw {_fmt(jm.get('phi_m_raw'))} × calibration {_fmt(jm.get('calibration'))}" if jm else "")
                 + f"). Intrinsic viscosity [η] = {_fmt(p['eta'])} ({e(p['eta_basis'])}). The φ<sub>m</sub> that makes Krieger-Dougherty "
                 f"reproduce the RVE value: {_fmt(p.get('kd_phi_m_fit'))}.</p>")
        mrows = []
        if p.get("farr_groot") is not None:
            mrows.append(["Farr-Groot random close packing (spheres)", _fmt(p["farr_groot"])])
        dw = p.get("desmond_weeks")
        if dw:
            mrows.append([f"Desmond-Weeks (δ {_fmt(dw['delta'])}, S {_fmt(dw['skewness'])}"
                          + ("" if dw["in_range"] else ", outside its fitted range δ ≤ 0.4") + ")", _fmt(dw["phi"])])
        if p.get("kd_alt") is not None:
            mrows.append([f"Krieger-Dougherty for {'frictionless' if p.get('friction') else 'frictional'} particles "
                          f"(φm {_fmt(p['phi_m_alt'])})", _fmt(p["kd_alt"])])
        if mrows:
            P.append(_table(["Maximum packing and its effect", "Value"], mrows))
        c = p["curve"]
        P.append(_svg_lines([("Krieger-Dougherty", c["phi"], c["kd"], "#0b62c4", "")]
                            + ([("KD, other friction case", c["phi"], c["kd_alt"], "#0b62c4", "2 3")] if c.get("kd_alt") else [])
                            + [
                             ("Maron-Pierce", c["phi"], c["mp"], "#e36209", "6 4"),
                             ("Hashin-Shtrikman lower bound", c["phi"], c["hs"], "#1a7f37", "2 3"),
                             ("Batchelor", c["phi"], c["batchelor"], "#8c959f", "4 3")],
                            "Filler volume fraction φ", "μ/μresin", ylog=True, ymin=1))
        fl = p["flow"]
        P.append(f"<p>Flow curve (resin: {e(_flow_model(fl['model']))}" + (f", yield stress {_fmt(fl['yield_Pa'])} Pa" if fl.get("yield_Pa") else "")
                 + f"); at {_fmt(p['gd_ref'])} 1/s the resin has {_fmt(p['mu_resin_ref'])} Pa·s and the compound "
                 f"<b>{_fmt(p['mu_compound_ref'])} Pa·s</b>.</p>")
        P.append(_svg_lines([("Compound", fl["gd"], fl["mu_compound"], "#0b62c4", ""),
                             ("Resin", fl["gd"], fl["mu_resin"], "#8c959f", "6 4")],
                            "Shear rate (1/s)", "Viscosity (Pa·s)", xlog=True, ylog=True))
        uf = p["underfill"]
        P.append(_table(["Capillary underfill (Washburn, parallel plates)", "Value"], [
            ["Gap / flow length", f"{_fmt(uf['gap_um'])} µm / {_fmt(uf['length_mm'])} mm"],
            ["Surface tension / contact angle", f"{_fmt(uf['gamma_mN_m'])} mN/m / {_fmt(uf['theta_deg'])}°"],
            ["Filling time, resin alone", _fmt(uf["t_resin_s"], "s")], ["Filling time, compound", _fmt(uf["t_compound_s"], "s")]]))
        if p.get("settling"):
            P.append(_table(["Filler", "Diameter", "Stokes velocity", "Hindered velocity (Richardson-Zaki)", f"Settled in {_fmt(p['settle_min'])} min"],
                            [[e(s_["name"]), _fmt(s_["d_um"], "µm"), _fmt(1e6 * s_["v0_m_s"], "µm/s"),
                              _fmt(1e6 * s_["v_m_s"], "µm/s"), _fmt(1e6 * s_["distance_m"], "µm")] for s_ in p["settling"]]))
    if "moisture" in props:
        p = props["moisture"]
        P.append("<h3>Moisture uptake, diffusion and swelling</h3>")
        rows = [["Effective diffusivity D_eff (x / y / z)", " / ".join(_fmt(v) for v in p["D_eff"]) + " m²/s"],
                ["Saturated uptake", f"{_fmt(p['wt_pct'])} wt% ({_fmt(p['c_sat_kg_m3'])} kg/m³)"],
                [f"Half / 95 % saturation, {_fmt(p['thickness_mm'])} mm plate ({'one face' if p['sides'] == 'one' else 'both faces'})",
                 f"{_fmt(p['t50_h'])} h / {_fmt(p['t95_h'])} h"],
                [f"Uptake after {_fmt(p['hours'])} h", f"{_fmt(p['wt_pct_at_hours'])} wt% ({_fmt(100 * p['fraction_at_hours'])} % of saturation)"]]
        if p.get("ratio"):
            rows.insert(1, ["D_eff / D_matrix", " / ".join(_fmt(v) for v in p["ratio"])])
        sw = p.get("swelling")
        if sw:
            rows += [["Swelling strain at saturation (x / y / z)", " / ".join(_fmt(1e3 * v) for v in sw["strain"][:3]) + " ‰"],
                     ["Effective CME", " / ".join(_fmt(v) for v in (sw.get("cme_eff") or [])) + " per mass fraction"]]
            if p.get("equivalent_dT"):
                rows.append(["Same strain as a temperature rise of", f"{_fmt(p['equivalent_dT'])} K"])
        P.append(_table(["Quantity", "Value"], rows))
        P.append(_svg_lines([("Uptake", p["curve"]["t_h"], p["curve"]["wt_pct"], "#0b62c4", "")],
                            "Exposure time (h)", "Moisture uptake (wt%)", xlog=True, ymin=0))
        if p.get("refs"):
            P.append(_table(["Reference", "Value"], [[e(k), _fmt(v)] for k, v in p["refs"].items()]))
        P.append("<p>The water moves by the gradient of its activity c/c<sub>sat</sub>; the solve carries the permeability "
                 "D·c<sub>sat</sub> and D<sub>eff</sub> = P<sub>eff</sub>/⟨c<sub>sat</sub>⟩. The uptake curve is Crank's solution "
                 "for a plate; swelling is the RVE's response to every material straining by CME × its moisture mass fraction.</p>")
    if "emi" in props:
        p = props["emi"]
        sp = p["spectrum"]
        P.append(f"<h3>EMI shielding effectiveness (thickness {_fmt(p['thickness_mm'])} mm, normal plane-wave incidence)</h3>")
        cx = p.get("complex")
        if cx:
            P.append(f"<p>Frequency-resolved: the RVE was solved with the complex admittivity σ + jωε of every region at "
                     f"{len(cx['freq_hz'])} frequencies ({', '.join(cx['directions'])}); the layer uses the resulting "
                     f"σ′(f) and ε′(f). Static values (DC σ and static ε solved apart): σ {_fmt(p['sigma_inplane'])} S/m, "
                     f"εr {_fmt(p['eps_inplane'])}, μr {_fmt(p['mu_inplane'])}.</p>")
            P.append(_table(["Frequency", "σ′ (S/m)", "ε′ (-)"],
                            [[f"{_fmt(f)} Hz", _fmt(s), _fmt(e_) if cx["eps_resolved"] else "—"]
                             for f, s, e_ in zip(cx["freq_hz"], cx["sigma_eff"], cx["eps_eff"])]))
        else:
            P.append(f"<p>In-plane effective σ {_fmt(p['sigma_inplane'])} S/m, εr {_fmt(p['eps_inplane'])}, μr {_fmt(p['mu_inplane'])}</p>")
        P.append(_svg_lines([("Total SE", sp["freq_hz"], sp["se_db"], "#0b62c4", ""),
                             ("Reflection SE_R", sp["freq_hz"], sp["se_r_db"], "#e36209", "6 4"),
                             ("Absorption SE_A", sp["freq_hz"], sp["se_a_db"], "#1a7f37", "2 3")]
                            + ([("Static σ, ε (for comparison)", sp["freq_hz"], sp["se_db_static"], "#8c959f", "4 3")]
                               if sp.get("se_db_static") else []),
                            "Frequency (Hz)", "SE (dB)", xlog=True, ymin=0))
        P.append(_table(["Frequency", "SE (dB)"], [[f"{k} Hz", _fmt(v)] for k, v in p["at"].items()]))
        fw = p.get("fullwave")
        if fw and not fw.get("error"):
            P.append(f"<h4>Full-wave check (openEMS, {_fmt(fw['thickness_um'])} µm slab of the RVE)</h4>")
            P.append(_svg_lines([("openEMS (voxel structure)", fw["freq_hz"], fw["se_fullwave_db"], "#d1242f", ""),
                                 ("Homogenised slab", fw["freq_hz"], fw["se_model_db"], "#0b62c4", "6 4")],
                                "Frequency (Hz)", "SE (dB)", xlog=False))
            P.append(f"<p>Largest difference {_fmt(fw['max_diff_db'])} dB (mean {_fmt(fw['mean_diff_db'])} dB), "
                     f"{fw['cells']:,} FDTD cells, {_fmt(fw['seconds'])} s.</p>"
                     + "".join(f"<div class='note'>{e(n)}</div>" for n in fw.get("notes", [])))
        elif fw:
            P.append(f"<p>Full-wave check not run: {e(fw['error'])}</p>")
        v = p["validity"]
        P.append(f"<div class='note'>Validity of the quasi-static homogenisation: wavelength condition f &lt; {_fmt(v['f_wavelength_hz'])} Hz" +
                 "".join(f"; skin-depth condition of {e(x['name'])} f &lt; {_fmt(x['f_skin_hz'])} Hz" for x in v["filler"]) + "</div>")
    if "permeability" in props:
        p = props["permeability"]
        P.append("<h3>Permeability (Stokes flow)</h3>" + _table(["Component", "m²", "darcy", "Connected"],
                 [[f"{d}{d}", _fmt(p["diag"][i]), _fmt(p["darcy"][i]), "●" if p["percolation"][i] else "○"] for i, d in enumerate("xyz")]))
        P.append(_table(["Derived quantity", "Value"], [
            ["Static flow resistivity (gas viscosity)", _fmt(p.get("flow_resistivity_Pa_s_m2"), "Pa·s/m²")],
            ["Kozeny–Carman constant φ³/(K·S_v²)", _fmt(p.get("kozeny_constant"))],
            ["Hydraulic tortuosity ⟨|u|⟩/⟨u∥⟩", _fmt(p.get("hydraulic_tortuosity"))],
            ["Open-pore specific surface", _fmt(p.get("specific_surface_open_1pm"), "1/m")]]))
    if "tortuosity" in props:
        p = props["tortuosity"]
        P.append("<h3>Diffusion and tortuosity</h3>" + _table(["Direction", "Tortuosity factor τ", "D_eff/D₀", "Formation factor F"],
                 [[d, _fmt(v["tortuosity"]), _fmt(v["d_eff"]), _fmt(v.get("formation_factor"))] for d, v in p["by_direction"].items()]))
        rows = [["MacMullin number", _fmt(p.get("macmullin_number"))], ["Archie cementation exponent m", _fmt(p.get("archie_exponent"))]]
        kn = p.get("knudsen")
        if kn:
            rows += [["Mean open-pore diameter", _fmt(kn["d_pore_um"], "µm")], ["Knudsen number", f"{_fmt(kn['knudsen_number'])} ({e(kn['regime'])})"],
                     ["Effective diffusivity, molecular", _fmt(kn["D_eff_molecular_m2_s"], "m²/s")],
                     ["Effective diffusivity, Bosanquet (Knudsen included)", _fmt(kn["D_eff_bosanquet_m2_s"], "m²/s")]]
        P.append(_table(["Quantity", "Value"], rows))
    if "acoustics" in props and "spectrum" in props["acoustics"]:
        p = props["acoustics"]
        sp = p["spectrum"]
        P.append(f"<h3>Acoustic absorption (hard-backed layer, {_fmt(p['thickness_mm'])} mm, normal incidence)</h3>")
        P.append(_table(["JCA parameter", "Value"], [["Open porosity φ", _fmt(p["porosity"])], ["Static flow resistivity σ", _fmt(p["flow_resistivity_Pa_s_m2"], "Pa·s/m²")],
                                                    ["Tortuosity α∞", _fmt(p["tortuosity"])], ["Viscous characteristic length Λ", _fmt(p["viscous_length_um"], "µm")],
                                                    ["Thermal characteristic length Λ'", _fmt(p["thermal_length_um"], "µm")],
                                                    ["Noise reduction coefficient", _fmt(p.get("nrc"))]]))
        P.append(_svg_lines([("Absorption coefficient", sp["freq_hz"], sp["alpha"], "#0b62c4", "")], "Frequency (Hz)", "α", xlog=True, ymin=0, ymax=1))
    if "radiation" in props and props["radiation"].get("beta_1pm"):
        p = props["radiation"]
        P.append("<h3>Radiative extinction (ray casting)</h3>" + _table(["Quantity", "Value"], [
            ["Extinction coefficient β (x / y / z)", _fmt(p["beta_1pm"], "1/m")], ["Mean β", _fmt(p["beta_mean_1pm"], "1/m")],
            ["Geometric reference S_v/(4φ) (mean chord)", _fmt(p.get("mean_chord_beta_1pm"), "1/m")],
            [f"Rosseland radiative conductivity at {_fmt(p['temperature_K'])} K (n = {_fmt(p['refractive_index'])})", _fmt(p["k_rad_W_mK"], "W/m·K")]]))
    if props.get("morphology"):
        P.append("<h3>Size distributions and structure measures</h3><table><tr><th>Phase</th><th class='n'>Volume fraction</th><th class='n'>D10 / D50 / D90 (µm)</th>"
                 "<th class='n'>Specific surface (1/m)</th><th class='n'>Mean intercept length x/y/z (µm)</th><th class='n'>Clusters</th><th>Connected (x, y, z)</th></tr>")
        for m in props["morphology"].values():
            if "error" in m:
                P.append(f"<tr><td>{e(m['name'])}</td><td colspan='6'>{e(m['error'])}</td></tr>")
                continue
            sd = m.get("size_distribution") or {}
            perc = m.get("percolates", {})
            P.append(f"<tr><td>{e(m['name'])}</td><td class='n'>{_fmt(100*m['volume_fraction'])} %</td>"
                     f"<td class='n'>{_fmt(sd.get('d10_um'))} / {_fmt(sd.get('d50_um'))} / {_fmt(sd.get('d90_um'))}</td>"
                     f"<td class='n'>{_fmt(m.get('specific_surface_1pm'))}</td><td class='n'>{_fmt(m.get('mean_intercept_length_um'))}</td>"
                     f"<td class='n'>{m['clusters']['n_clusters']}</td>"
                     f"<td>{' '.join('●' if perc.get(a) else '○' for a in 'xyz')}</td></tr>")
        P.append("</table>")
    if props.get("percolation"):
        P.append("<h3>Percolation paths</h3><table><tr><th>Phase</th><th>Connectivity</th><th>Direction</th><th class='n'>Spanning</th>"
                 "<th class='n'>Geodesic tortuosity</th><th class='n'>Spanning share</th><th class='n'>Dead end</th><th class='n'>Isolated</th></tr>")
        for g in props["percolation"].values():
            if "error" in g:
                P.append(f"<tr><td>{e(g['name'])}</td><td colspan='7'>{e(g['error'])}</td></tr>")
                continue
            for d, v in g["by_direction"].items():
                P.append(f"<tr><td>{e(g['name'])}</td><td class='n'>{g['connectivity']}</td><td class='n'>{d}</td>"
                         f"<td class='n'>{'yes' if v['percolates'] else 'no'}</td><td class='n'>{_fmt(v.get('geodesic_tortuosity'))}</td>"
                         f"<td class='n'>{_fmt(100 * v['spanning_fraction'], '%') if v.get('spanning_fraction') else '—'}</td>"
                         f"<td class='n'>{_fmt(100 * v['dead_end_fraction'], '%')}</td><td class='n'>{_fmt(100 * v['isolated_fraction'], '%')}</td></tr>")
        P.append("</table><p class='hint'>The geodesic tortuosity is the mean shortest path through the spanning cluster divided by the "
                 "straight distance; it is a lower bound for the diffusive tortuosity factor of the same structure.</p>")
    if "porosimetry" in props and props["porosimetry"].get("intrusion"):
        p = props["porosimetry"]
        it = p["intrusion"]
        P.append(f"<h3>Porosimetry ({e(p.get('fluid', ''))}, γ = {_fmt(p['fluid_gamma_N_m'])} N/m, θ = {_fmt(p['contact_angle_deg'])}°)</h3>")
        ser = [("Intrusion", it["pressure_Pa"], it["saturation"], "#0b62c4", "")]
        if p.get("extrusion"):
            ser.append(("Extrusion", p["extrusion"]["pressure_Pa"], p["extrusion"]["saturation"], "#e36209", "6 4"))
        P.append(_svg_lines(ser, "Pressure (Pa)", "Intruded fraction of the pore volume", xlog=True, ymin=0, ymax=1))
        P.append(_table(["Quantity", "Value"], [["Pore-entry diameter at 10 / 50 / 90 % intrusion", _fmt([p.get("d10_um"), p.get("d50_um"), p.get("d90_um")], "µm")],
                                                ["Threshold (breakthrough) pressure", _fmt(p.get("threshold_pressure_Pa"), "Pa")],
                                                ["Maximum intruded fraction of the pore volume", _fmt(p.get("max_intrusion"))]]))
        tp = p.get("through_pore") or {}
        P.append(_table(["Direction", "Largest through-pore diameter", "Bubble point, water", "Bubble point, isopropanol"],
                        [[d, _fmt(v["diameter_um"], "µm"), _fmt(v["bubble_point_water_Pa"], "Pa"), _fmt(v["bubble_point_ipa_Pa"], "Pa")] if v else [d, "no through-pore", "—", "—"]
                         for d, v in tp.items()]))
    if "pore_network" in props and props["pore_network"].get("n_pores") is not None:
        p = props["pore_network"]
        g = lambda k: (p.get(k) or {})
        P.append("<h3>Pore network (SNOW2)</h3>" + _table(["Quantity", "Value"], [
            ["Pores / throats", f"{p['n_pores']} / {p['n_throats']}"], ["Mean / maximum coordination number", f"{_fmt(p['coordination_mean'])} / {p['coordination_max']}"],
            ["Isolated pores", str(p["isolated_pores"])], ["Pore density", _fmt(p["pore_density_per_mm3"], "1/mm³")],
            ["Pore inscribed diameter, mean / median", _fmt([g("pore_inscribed_diameter_um").get("mean"), g("pore_inscribed_diameter_um").get("d50")], "µm")],
            ["Throat inscribed diameter, mean / median", _fmt([g("throat_inscribed_diameter_um").get("mean"), g("throat_inscribed_diameter_um").get("d50")], "µm")],
            ["Throat length, mean", _fmt(g("throat_length_um").get("mean"), "µm")]]))
    if "grains" in props and props["grains"].get("phases"):
        p = props["grains"]
        rows = []
        for gph in p["phases"].values():
            if not gph.get("n_particles"):
                continue
            her = gph.get("hermans") or {}
            rows.append([e(gph["name"]), str(gph["n_particles"]), _fmt(gph.get("number_density_per_mm3")),
                         _fmt([gph["eq_diameter_um"]["number"].get("d50"), gph["eq_diameter_um"]["volume"].get("d50")]),
                         _fmt(gph.get("sphericity")), _fmt(gph.get("aspect_ratio")),
                         _fmt([her.get("x"), her.get("y"), her.get("z")]) if her else "—",
                         _fmt((gph.get("nearest_neighbour_um") or {}).get("mean")), _fmt(gph.get("clark_evans_index")),
                         str(gph["voxel"]["clusters"]["n_clusters"])])
        P.append("<h3>Grain and filler analysis</h3>" + _table(["Phase", "Particles", "Number density (1/mm³)", "D50 number / volume (µm)", "Sphericity",
                                                               "Aspect ratio", "Hermans x / y / z", "Nearest neighbour (µm)", "Clark–Evans index", "Clusters"], rows))
        ml = p.get("matrix_ligament")
        if ml:
            P.append(f"<p>Matrix ligament thickness between particles (local thickness of the matrix): D10 {_fmt(ml['d10_um'])} µm, "
                     f"D50 {_fmt(ml['d50_um'])} µm, D90 {_fmt(ml['d90_um'])} µm.</p>")

    # -------------------------------------------------------------- figures
    figs = r.get("figures") or {}
    if figs:
        P.append("<h2>4. Figures</h2><div class='figs'>")
        for name, title in figs.items():
            path = os.path.join(out_dir, "figures", name)
            if os.path.exists(path):
                P.append(f"<figure><img src='{_img(path)}'><figcaption>{e(title)}</figcaption></figure>")
        P.append("</div>")

    # --------------------------------------------------------------- checks
    lab = {"pass": "passed", "warn": "warning", "fail": "failed"}
    P.append("<h2>5. Verification</h2><table><tr><th>Group</th><th>Check</th><th class='n'>Value</th><th class='n'>Criterion</th><th>Status</th><th>Note</th></tr>")
    for c in r["checks"]:
        P.append(f"<tr><td>{e(c['group'])}</td><td>{e(c['label'])}</td><td class='n'>{_fmt(c['value'])}</td>"
                 f"<td class='n'>{_fmt(c['target'])}</td><td class='{c['status']}'>{lab[c['status']]}</td><td>{e(c.get('note', ''))}</td></tr>")
    P.append("</table>")

    # -------------------------------------------------------------- methods
    v = r["versions"]
    P.append("<h2>6. Methods and open-source software</h2><ul>"
             "<li>Microstructure: periodic voxel RVE. Non-overlapping spheres are packed by a force-biased algorithm followed by Monte-Carlo equilibration; "
             "other shapes are placed by voxel random sequential addition; connected networks are thresholded band-limited Gaussian random fields.</li>"
             "<li>Thermal and electrical conductivity, permittivity and magnetic permeability: NASA PuMA periodic Q1 finite elements (fully periodic boundaries). "
             "The same RVE is additionally solved with an independent periodic finite-volume discretisation for cross-checking.</li>"
             "<li>Elasticity and thermal expansion: NASA PuMA periodic Q1 finite-element elasticity with six unit strains and an isotropic thermal eigenstrain "
             "load assembled from PuMA's element matrices.</li>"
             "<li>Permeability: NASA PuMA Stokes finite elements on the open porosity. Diffusion: PuMA continuum tortuosity. Surface areas: PuMA marching-cubes isosurface. "
             "Mean intercept length and structure-tensor orientation: PuMA.</li>"
             "<li>Acoustic absorption: Johnson–Champoux–Allard model with parameters obtained from the RVE fields (flow resistivity from the permeability, "
             "tortuosity from the formation factor, viscous length from the field-weighted surface integral, thermal length from the pore volume and surface).</li>"
             "<li>Radiative extinction: NASA PuMA ray casting in the pore space; Rosseland diffusion approximation for the radiative conductivity.</li>"
             "<li>Pore- and particle-size distributions, chord lengths, two-point correlation: PoreSpy (periodicity by wrap padding). "
             "Porosimetry and capillary pressure: PoreSpy morphological drainage with Washburn pressures. Pore network: PoreSpy SNOW2.</li>"
             "<li>EMI shielding: homogenised σ, εr and μr in a scikit-rf transmission-line model, cross-checked with an independent transfer-matrix solution.</li>"
             "<li>3D rendering: PyVista (VTK) SurfaceNets surfaces and voxel-cell face contours; vtk.js in the browser viewer.</li></ul>")
    P.append(_table(["Package", "Version"], [[e(k), e(str(val))] for k, val in v.items()]))
    P.append("<h3>References</h3><ul>" + "".join(f"<li><b>{e(a)}</b> — {e(b)}</li>" for a, b in REFS) + "</ul>")
    P.append("<h3>Scope and limitations</h3><ul><li>Interfaces are perfectly bonded (with an optional interfacial thermal resistance for conduction). "
             "Contact resistance, tunnelling and debonding are not modelled.</li>"
             "<li>Properties are linear room-temperature representative values; temperature and frequency dependence and anisotropic crystal properties are not included.</li>"
             "<li>EMI results assume quasi-static homogenisation and an infinite slab under normal incidence. Enclosures with apertures and seams require a full electromagnetic analysis.</li>"
             "<li>Acoustic results assume a rigid frame; porosimetry assumes quasi-static invasion from the sample faces; ray casting assumes an opaque, diffusely absorbing solid.</li></ul>")
    P.append("<h3>Layer card (input for multilayer analyses)</h3><pre>" + e(json.dumps(r["layer_card"], ensure_ascii=False, indent=1)) + "</pre>")
    P.append("</div></body></html>")
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as fh:
        fh.write("".join(P))
