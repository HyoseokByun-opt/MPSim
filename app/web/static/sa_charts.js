/* Sensitivity-analysis figures (5.11.0): SVG built in the browser, without a library (the page must render on an
   isolated network), in the manner of a journal figure - thin axes on two sides, Arial, a panel letter, direct
   labels where they fit, values on hover, export as SVG or PNG.

   Every function takes the analysis of mpsim/sensitivity.py and returns an SVG string drawn in a viewBox, so
   the figure scales with its card. Inputs keep one colour across every figure (fixed order, validated for
   colour-vision deficiency: dataviz validate_palette.js, light surface); magnitudes use the chosen colour map. */
(function () {
  "use strict";
  const SA = (window.MPSSA = {});
  const fmt = (v) => (window.MPSCharts ? window.MPSCharts.fmt(v) : String(v));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const FONT = "Arial, Helvetica, sans-serif";
  const INK = "#1b1b1b", INK2 = "#52514e", MUTED = "#898781", RULE = "#c3c2b7", HAIR = "#e7e6e0";
  // categorical identity of the inputs (validated: CVD separation and normal-vision floor pass)
  // vivid colours drawn from the jet map, cool and warm in turn (validated: lightness band, chroma, colour-vision
  // separation of neighbours and the normal-vision floor all pass); the first four are jet's blue, red, cyan, orange
  const JB = "#0050f0", JR = "#f02000", JC = "#00a8e0", JO = "#ff8c00", JI = "#3a3ad0", JG = "#00a03c", JK = "#c0002a", JY = "#d9a400";
  SA.SERIES = [JB, JR, JC, JO, JI, JG, JK, JY];
  const OTHER = "#9a9890";
  SA.inputColor = (i) => (i < SA.SERIES.length ? SA.SERIES[i] : OTHER);
  // the families of methods, in the same fixed order
  SA.FAMILIES = ["Correlation", "Regression", "Dependence", "Machine learning", "Variance-based", "Screening", "Moment-independent"];
  SA.familyColor = (f) => SA.SERIES[Math.max(0, SA.FAMILIES.indexOf(f))];
  const KIND_COL = { main: JB, quad: JG, inter: JR };

  /* ------------------------------------------------------------ colour maps */
  let CMAP = "jet";
  SA.setCmap = (c) => { CMAP = c; };
  const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);
  function lerpTab(tab) {
    return (t) => {
      t = clamp01(t) * (tab.length - 1);
      const i = Math.min(Math.floor(t), tab.length - 2), f = t - i;
      return [0, 1, 2].map((c) => Math.round(tab[i][c] + (tab[i + 1][c] - tab[i][c]) * f));
    };
  }
  const MAPS = {
    // the user's vivid jet (the viewer's), its perceptual cousin, a single hue, and viridis
    jet: (t) => { t = clamp01(t); return [1.5 - Math.abs(4 * t - 3), 1.5 - Math.abs(4 * t - 2), 1.5 - Math.abs(4 * t - 1)].map((v) => Math.round(255 * clamp01(v))); },
    turbo: (t) => (window.MPSViewer && window.MPSViewer.CMAPS ? window.MPSViewer.CMAPS.turbo(t).map((v) => Math.round(255 * v)) : MAPS.jet(t)),
    blues: lerpTab([[247, 251, 255], [205, 226, 251], [158, 197, 244], [109, 167, 236], [57, 135, 229], [37, 106, 191], [24, 79, 149], [13, 54, 107]]),
    viridis: lerpTab([[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]]),
  };
  const DIV = lerpTab([[24, 79, 149], [57, 135, 229], [158, 197, 244], [240, 239, 236], [244, 176, 160], [227, 73, 72], [150, 30, 30]]);
  const rgb = (a) => `rgb(${a[0]},${a[1]},${a[2]})`;
  SA.seq = (t) => rgb((MAPS[CMAP] || MAPS.jet)(t));
  SA.div = (t) => rgb(DIV(0.5 + 0.5 * Math.max(-1, Math.min(1, t))));
  // dark text on a light cell, white on a dark one
  function inkOn(fill) {
    const m = /rgb\((\d+),(\d+),(\d+)\)/.exec(fill);
    if (!m) return INK;
    const L = (0.2126 * m[1] + 0.7152 * m[2] + 0.0722 * m[3]) / 255;
    return L > 0.55 ? INK : "#ffffff";
  }

  /* ------------------------------------------------------------ primitives */
  const r2 = (v) => Math.round(v * 100) / 100;
  function svg(W, H, body, defs = "") {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" font-family="${FONT}" style="display:block;max-width:${Math.round(W * 1.6)}px;margin:0 auto">` +
      `<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#fff" fill-opacity="0"/><line x1="0" y1="0" x2="0" y2="5" stroke="currentColor" stroke-width="2.2"/></pattern>${defs}</defs>` +
      `<rect width="${W}" height="${H}" fill="#ffffff"/>${body}</svg>`;
  }
  const T = (x, y, s, o = {}) => `<text x="${r2(x)}" y="${r2(y)}" font-size="${o.size || 10.5}" fill="${o.fill || INK2}" text-anchor="${o.anchor || "start"}"${o.weight ? ` font-weight="${o.weight}"` : ""}${o.rot ? ` transform="rotate(${o.rot} ${r2(x)} ${r2(y)})"` : ""}${o.base ? ` dominant-baseline="${o.base}"` : ""}${o.style ? ` font-style="${o.style}"` : ""}>${o.raw ? s : esc(s)}</text>`;
  const L = (x1, y1, x2, y2, c = INK, w = 0.9, dash) => `<line x1="${r2(x1)}" y1="${r2(y1)}" x2="${r2(x2)}" y2="${r2(y2)}" stroke="${c}" stroke-width="${w}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
  const R = (x, y, w, h, fill, o = {}) => `<rect x="${r2(x)}" y="${r2(y)}" width="${r2(Math.max(0, w))}" height="${r2(Math.max(0, h))}" fill="${fill}"${o.rx ? ` rx="${o.rx}"` : ""}${o.stroke ? ` stroke="${o.stroke}" stroke-width="${o.sw || 1}"` : ""}${o.op !== undefined ? ` fill-opacity="${o.op}"` : ""}${o.tip ? ` data-tip="${esc(o.tip)}"` : ""}${o.style ? ` style="${o.style}"` : ""}/>`;
  const C = (x, y, r, fill, o = {}) => `<circle cx="${r2(x)}" cy="${r2(y)}" r="${r}" fill="${fill}"${o.stroke !== false ? ` stroke="${o.stroke || "#fff"}" stroke-width="${o.sw || 0.8}"` : ""}${o.op !== undefined ? ` fill-opacity="${o.op}"` : ""}${o.tip ? ` data-tip="${esc(o.tip)}"` : ""}/>`;
  const P = (d, stroke, w = 1.6, o = {}) => `<path d="${d}" fill="${o.fill || "none"}" stroke="${stroke}" stroke-width="${w}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}${o.op !== undefined ? ` stroke-opacity="${o.op}"` : ""}${o.fop !== undefined ? ` fill-opacity="${o.fop}"` : ""} stroke-linejoin="round" stroke-linecap="round"${o.tip ? ` data-tip="${esc(o.tip)}"` : ""}/>`;
  const line = (xs, ys, sx, sy) => xs.map((x, i) => `${i ? "L" : "M"}${r2(sx(x))},${r2(sy(ys[i]))}`).join("");
  function scale(d0, d1, r0, r1, log) {
    if (log) {
      const a = Math.log10(d0), b = Math.log10(d1);
      return (v) => r0 + (Math.log10(Math.max(v, 1e-300)) - a) / (b - a || 1) * (r1 - r0);
    }
    return (v) => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);
  }
  function nice(v) { const e = Math.floor(Math.log10(Math.abs(v) || 1)), f = v / 10 ** e; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * 10 ** e; }
  function ticks(a, b, n = 5) {
    if (!(b > a)) return [a];
    const st = nice((b - a) / n), out = [];
    for (let v = Math.ceil(a / st - 1e-9) * st; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : +v.toPrecision(12));
    return out;
  }
  function extent(arr, pad = 0.06) {
    const v = arr.filter((x) => x !== null && x !== undefined && isFinite(x));
    if (!v.length) return [0, 1];
    let a = Math.min(...v), b = Math.max(...v);
    if (a === b) { a -= Math.abs(a) * 0.1 || 1; b += Math.abs(b) * 0.1 || 1; }
    const p = (b - a) * pad;
    return [a - p, b + p];
  }
  // axes on two sides, ticks outward
  function axes(x0, y0, w, h, xr, yr, o = {}) {
    let s = "";
    const sx = scale(xr[0], xr[1], x0, x0 + w), sy = scale(yr[0], yr[1], y0 + h, y0);
    if (o.grid) ticks(yr[0], yr[1], o.ny || 5).forEach((v) => { s += L(x0, sy(v), x0 + w, sy(v), HAIR, 0.7); });
    s += L(x0, y0 + h, x0 + w, y0 + h, INK, 0.9) + L(x0, y0, x0, y0 + h, INK, 0.9);
    if (!o.noX) ticks(xr[0], xr[1], o.nx || 5).forEach((v) => {
      if (v < xr[0] - 1e-12 || v > xr[1] + 1e-12) return;
      s += L(sx(v), y0 + h, sx(v), y0 + h + 3.5, INK, 0.8) + T(sx(v), y0 + h + 13.5, (o.xfmt || fmt)(v), { anchor: "middle", size: o.ts || 9.5 });
    });
    if (!o.noY) ticks(yr[0], yr[1], o.ny || 5).forEach((v) => {
      if (v < yr[0] - 1e-12 || v > yr[1] + 1e-12) return;
      s += L(x0 - 3.5, sy(v), x0, sy(v), INK, 0.8) + T(x0 - 6, sy(v) + 3.3, (o.yfmt || fmt)(v), { anchor: "end", size: o.ts || 9.5 });
    });
    if (o.xlabel) s += T(x0 + w / 2, y0 + h + (o.xlo || 29), o.xlabel, { anchor: "middle", size: o.ls || 10.5, fill: INK });
    if (o.ylabel) s += T(x0 - (o.ylo || 36), y0 + h / 2, o.ylabel, { anchor: "middle", size: o.ls || 10.5, fill: INK, rot: -90 });
    return { s, sx, sy };
  }
  function colorbar(x, y, w, h, lo, hi, label, f = SA.seq, o = {}) {
    let s = "";
    const n = 40;
    for (let k = 0; k < n; k++) {
      if (o.vertical) s += R(x, y + h - (k + 1) * h / n, w, h / n + 0.4, f(k / (n - 1)));
      else s += R(x + k * w / n, y, w / n + 0.4, h, f(k / (n - 1)));
    }
    s += R(x, y, w, h, "none", { stroke: RULE, sw: 0.6 });
    if (o.vertical) {
      s += T(x + w + 4, y + 8, fmt(hi), { size: 9 }) + T(x + w + 4, y + h, fmt(lo), { size: 9 });
      if (label) s += T(x + w / 2, y - 6, label, { anchor: "middle", size: 9.5, fill: INK });
    } else {
      s += T(x, y + h + 11, o.lolab || fmt(lo), { size: 9 }) + T(x + w, y + h + 11, o.hilab || fmt(hi), { size: 9, anchor: "end" });
      if (label) s += T(x + w / 2, y - 4, label, { anchor: "middle", size: 9.5, fill: INK });
    }
    return s;
  }
  const short = (s, n = 22) => { s = String(s ?? ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; };
  const nameOf = (res, i) => res.inputs[i].name;
  const stars = (p) => (p === null || p === undefined ? "" : p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "");
  const num = (a) => (a || []).map((v) => (v === null || v === undefined ? NaN : +v));
  const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
  function legend(items, x, y, o = {}) {
    // items: [{label, color, kind: "dot"|"bar"|"line"|"dash"|"hatch"}], laid out in a row (wrapping at o.w)
    let s = "", cx = x, cy = y;
    const W = o.w || 9999;
    items.forEach((it) => {
      const tw = 14 + 5.6 * String(it.label).length + 12;
      if (cx + tw > x + W && cx > x) { cx = x; cy += 14; }
      if (it.kind === "line") s += L(cx, cy - 3.5, cx + 12, cy - 3.5, it.color, 2);
      else if (it.kind === "dash") s += L(cx, cy - 3.5, cx + 12, cy - 3.5, it.color, 1.6, "3,2");
      else if (it.kind === "dot") s += C(cx + 6, cy - 3.5, 4, it.color);
      else if (it.kind === "hatch") s += `<g color="${it.color}">${R(cx, cy - 8, 12, 9, "url(#hatch)", { stroke: it.color, sw: 0.8 })}</g>`;
      else if (it.kind === "hollow") s += C(cx + 6, cy - 3.5, 3.8, "#fff", { stroke: it.color, sw: 1.4 });
      else s += R(cx, cy - 8, 12, 9, it.color, { rx: 1.5 });
      s += T(cx + 16, cy, it.label, { size: 9.5, fill: INK2 });
      cx += tw;
    });
    return { s, h: cy - y + 14 };
  }

  /* ------------------------------------------------------------ summary */
  // the order of the method rows: family by family, as in the module's list
  function methodRows(res) {
    const info = {};
    (res.method_info || []).forEach((m) => (info[m.id] = m));
    const ids = res.consensus.methods.slice();
    ids.sort((a, b) => SA.FAMILIES.indexOf((info[a] || {}).family) - SA.FAMILIES.indexOf((info[b] || {}).family) ||
      (res.method_info.findIndex((m) => m.id === a) - res.method_info.findIndex((m) => m.id === b)));
    return ids.map((id) => ({ id, k: res.consensus.methods.indexOf(id), ...(info[id] || { label: id, family: "" }) }));
  }

  SA.consensusHeatmap = function (res) {
    const c = res.consensus, d = res.d, order = c.order, rows = methodRows(res);
    const left = 176, top = 46, cw = Math.max(30, Math.min(70, 470 / d)), ch = 15.5, W = left + d * cw + 86;
    const H = top + rows.length * ch + 4 * 2 + 70;
    let s = "", y = top, fam = null;
    // column heads: inputs by consensus rank
    order.forEach((i, k) => {
      const x = left + k * cw + cw / 2;
      s += C(x, top - 30, 4.2, SA.inputColor(i), { tip: nameOf(res, i) });
      s += T(x, top - 8, short(nameOf(res, i), Math.max(6, Math.round(cw / 5.6))), { anchor: "middle", size: 9.5, fill: INK, rot: d > 6 ? -28 : 0 });
    });
    const famStart = {};
    rows.forEach((r) => {
      if (r.family !== fam) { if (fam !== null) y += 4; fam = r.family; famStart[fam] = y; }
      s += T(left - 8, y + ch / 2 + 3.4, r.label, { anchor: "end", size: 9.5, fill: INK });
      order.forEach((i, k) => {
        const v = c.matrix[r.k][i];
        const f = SA.seq(v);
        s += R(left + k * cw + 0.6, y + 0.6, cw - 1.2, ch - 1.2, f, { tip: `${r.label} · ${nameOf(res, i)}\nnormalised importance ${fmt(v)} (rank ${c.ranks[r.k][i]} of ${d})\nreads: ${r.reads}${r.model ? `\nmodel: ${r.model}, weight ${fmt(r.weight)}` : ""}` });
        if (cw >= 34) s += T(left + k * cw + cw / 2, y + ch / 2 + 3.3, v >= 0.995 ? "1" : v.toFixed(2).replace(/^0/, ""), { anchor: "middle", size: 8.5, fill: inkOn(f) });
      });
      // the weight of the method in the consensus (Q2 of its model, shared in its family)
      const wv = c.weights[r.k];
      s += R(left + d * cw + 8, y + 4, 40 * Math.min(1, wv * 5), ch - 8, "#b9b7ad", { tip: `weight in the consensus ${fmt(wv)}${r.model ? ` (model ${r.model})` : ""}` });
      y += ch;
    });
    // family brackets
    Object.entries(famStart).forEach(([f, y0]) => {
      const n = rows.filter((r) => r.family === f).length;
      const y1 = y0 + n * ch;
      s += R(left - 4, y0 + 1, 2.6, y1 - y0 - 2, SA.familyColor(f));
    });
    s += T(left + d * cw + 8, top - 8, "weight", { size: 9, fill: MUTED });
    // consensus row
    y += 8;
    s += L(left, y - 3, left + d * cw, y - 3, INK, 0.9);
    s += T(left - 8, y + ch / 2 + 3.6, "Consensus", { anchor: "end", size: 10, fill: INK, weight: 700 });
    order.forEach((i, k) => {
      const v = c.score[i];
      const f = SA.seq(v);
      s += R(left + k * cw + 0.6, y + 0.6, cw - 1.2, ch + 2, f, { tip: `${nameOf(res, i)}: consensus ${fmt(v)}, weighted mean rank ${fmt(c.mean_rank[i])}` });
      s += T(left + k * cw + cw / 2, y + ch / 2 + 4.2, v.toFixed(2), { anchor: "middle", size: 9, fill: inkOn(f), weight: 700 });
    });
    y += ch + 6;
    s += T(left - 8, y + 10, "Rank", { anchor: "end", size: 9.5, fill: INK2 });
    order.forEach((i, k) => { s += T(left + k * cw + cw / 2, y + 10, `#${k + 1}`, { anchor: "middle", size: 9.5, fill: INK, weight: 700 }); });
    s += colorbar(left, y + 26, Math.min(200, d * cw), 8, 0, 1, "normalised importance (1 = the method's strongest input)");
    // family legend
    const lg = legend(SA.FAMILIES.filter((f) => famStart[f] !== undefined).map((f) => ({ label: f, color: SA.familyColor(f) })), left + Math.min(200, d * cw) + 20, y + 34, { w: Math.max(160, W - left - Math.min(200, d * cw) - 30) });
    s += lg.s;
    return svg(W, Math.max(H, y + 34 + lg.h + 4), s);
  };

  // every method's normalised importance as a dot, the consensus as a bar
  SA.consensusStrip = function (res) {
    const c = res.consensus, d = res.d, order = c.order, rows = methodRows(res);
    const left = 160, right = 150, rh = 30, top = 20, pw = 300, W = left + pw + right, H = top + d * rh + 70;
    let s = "";
    const sx = scale(0, 1, left, left + pw);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * rh, HAIR, 0.7); s += T(sx(v), top + d * rh + 13, String(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * rh, left + pw, top + d * rh, INK, 0.9);
    s += T(left + pw / 2, top + d * rh + 29, "normalised importance (each method scaled to its strongest input)", { anchor: "middle", size: 10.5, fill: INK });
    order.forEach((i, k) => {
      const y = top + k * rh + rh / 2;
      s += C(left - 150, y, 4.2, SA.inputColor(i));
      s += T(left - 142, y + 3.5, short(nameOf(res, i), 22), { size: 10.5, fill: INK });
      s += R(left, y - 7, sx(c.score[i]) - left, 14, SA.inputColor(i), { op: 0.42, rx: 2, tip: `${nameOf(res, i)}: consensus ${fmt(c.score[i])}` });
      rows.forEach((r, q) => {
        const v = c.matrix[r.k][i];
        const jy = y + ((q * 7.3) % 13) - 6.5;
        s += C(sx(v), jy, 3, SA.familyColor(r.family), { sw: 0.6, tip: `${r.label}: ${fmt(v)}` });
      });
      s += `<path d="M${r2(sx(c.score[i]))},${r2(y - 7)}l5,7l-5,7l-5,-7z" fill="${INK}" data-tip="consensus ${fmt(c.score[i])}"/>`;
      const dir = (res.direction || [])[i], shp = (res.shapes || [])[i];
      const arrow = shp === "peaked" ? "∩" : shp === "valley" ? "∪" : dir > 0 ? "↑" : dir < 0 ? "↓" : "·";
      s += T(left + pw + 12, y + 3.5, `${c.score[i].toFixed(2)}`, { size: 10.5, fill: INK, weight: 700 });
      s += T(left + pw + 48, y + 3.5, `rank ${(+c.mean_rank[i]).toFixed(1)}`, { size: 9.5 });
      s += T(left + pw + 108, y + 4, arrow, { size: 13, fill: INK, anchor: "middle" });
    });
    s += T(left + pw + 12, top - 6, "score", { size: 9, fill: MUTED }) + T(left + pw + 48, top - 6, "mean rank", { size: 9, fill: MUTED }) + T(left + pw + 108, top - 6, "effect", { size: 9, fill: MUTED, anchor: "middle" });
    const lg = legend(SA.FAMILIES.filter((f) => rows.some((r) => r.family === f)).map((f) => ({ label: f, color: SA.familyColor(f), kind: "dot" })), left, top + d * rh + 48, { w: pw + right - 10 });
    return svg(W, H + lg.h - 6, s + lg.s);
  };

  // how each family of methods ranks the inputs: lines that stay flat mean agreement
  SA.bump = function (res) {
    const c = res.consensus, d = res.d, rows = methodRows(res);
    const fams = SA.FAMILIES.filter((f) => rows.some((r) => r.family === f));
    const left = 56, right = 150, top = 26, ph = Math.max(150, d * 26), pw = Math.max(340, fams.length * 74), W = left + pw + right, H = top + ph + 66;
    const fx = (k) => left + (fams.length > 1 ? (k * pw) / (fams.length - 1) : pw / 2);
    const famImp = fams.map((f) => {
      const rs = rows.filter((r) => r.family === f);
      return Array.from({ length: d }, (_, i) => mean(rs.map((r) => c.matrix[r.k][i])));
    });
    const famRank = famImp.map((imp) => { const o = imp.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]); const rk = new Array(d); o.forEach(([, i], k) => (rk[i] = k + 1)); return rk; });
    const sy = scale(1, d, top, top + ph);
    let s = "";
    for (let k = 1; k <= d; k++) s += L(left, sy(k), left + pw, sy(k), HAIR, 0.7) + T(left - 8, sy(k) + 3.4, `#${k}`, { anchor: "end", size: 9.5 });
    // the family names alternate between two lines, so that long neighbours never run into each other
    fams.forEach((f, k) => {
      const yl = top + ph + 18 + (k % 2) * 14;
      s += L(fx(k), top + ph + 4, fx(k), yl - 9, RULE, 0.6);
      s += R(fx(k) - 8, yl - 9, 16, 3, SA.familyColor(f));
      s += T(fx(k), yl + 4, f, { anchor: "middle", size: 9.5, fill: INK });
    });
    c.order.forEach((i) => {
      const xs = fams.map((_, k) => fx(k)), ys = famRank.map((rk) => sy(rk[i]));
      const col = SA.inputColor(i);
      s += `<path d="${xs.map((x, k) => `${k ? "L" : "M"}${r2(x)},${r2(ys[k])}`).join("")}" fill="none" stroke="${col}" stroke-width="2.2" stroke-opacity="0.9"/>`;
      xs.forEach((x, k) => { s += C(x, ys[k], 4.6, col, { sw: 1.4, tip: `${nameOf(res, i)} · ${fams[k]}: rank ${famRank[k][i]} (mean normalised importance ${fmt(famImp[k][i])})` }); });
      s += T(left + pw + 10, ys[ys.length - 1] + 3.5, short(nameOf(res, i), 22), { size: 10, fill: INK });
    });
    s += T(left + pw / 2, top - 10, `rank given by each family of methods · Kendall's W = ${fmt(c.kendall_w)}`, { anchor: "middle", size: 10, fill: INK2 });
    return svg(W, H, s);
  };

  // cross-validated accuracy of every model behind the model-based measures
  SA.modelQuality = function (res) {
    const ks = Object.keys(res.models || {});
    const left = 190, pw = 300, rh = 24, top = 22, W = left + pw + 120, H = top + ks.length * rh + 46;
    const sx = scale(-0.5, 1, left, left + pw);
    let s = "";
    [-0.5, 0, 0.5, 1].forEach((v) => { s += L(sx(v), top - 4, sx(v), top + ks.length * rh, v === 0.5 ? JO : HAIR, v === 0.5 ? 1.2 : 0.7, v === 0.5 ? "4,3" : null); s += T(sx(v), top + ks.length * rh + 13, String(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + ks.length * rh, left + pw, top + ks.length * rh, INK, 0.9);
    s += T(sx(0.5) + 4, top - 8, "trusted above 0.5", { size: 9, fill: "#b45f00" });
    ks.forEach((k, j) => {
      const m = res.models[k], y = top + j * rh + rh / 2, q = m.q2;
      const role = [res.surrogate.global === k ? "global surrogate" : "", res.surrogate.explainer === k ? "explained (SHAP, PD)" : ""].filter(Boolean).join(" · ");
      s += T(left - 10, y + 3.5, m.label, { anchor: "end", size: 10, fill: INK, weight: role ? 700 : 400 });
      if (q === null || q === undefined) { s += T(left + 4, y + 3.5, m.error ? "failed" : "—", { size: 9.5 }); return; }
      const v = Math.max(-0.5, Math.min(1, q));
      // the bar in the colour map of its Q2 (0 .. 1), grey below zero
      const col = q >= 0 ? SA.seq(Math.min(1, q)) : "#c8c6be";
      s += R(sx(0), y - 6, sx(v) - sx(0), 12, col, { rx: 2, tip: `${m.label}: Q² = ${fmt(q)} (cross-validated on ${m.folds} folds), RMSE ${fmt(m.rmse)}, R² on the training runs ${fmt(m.r2_train)}` });
      s += T(left + pw + 8, y + 3.5, `Q² ${q.toFixed(3)}`, { size: 10, fill: INK, weight: 700 });
      if (role) s += T(left + pw + 64, y + 3.5, role.includes("global") ? "S" + (role.includes("SHAP") ? "+E" : "") : "E", { size: 9, fill: MUTED });
    });
    s += T(left + pw / 2, top + ks.length * rh + 30, "Q² — share of the variance predicted on runs the model never saw", { anchor: "middle", size: 10.5, fill: INK });
    return svg(W, H, s);
  };

  /* ------------------------------------------------------------ statistics */
  SA.forest = function (res) {
    const M = res.methods, d = res.d;
    const keys = [["pearson", "Pearson r"], ["spearman", "Spearman ρ"], ["kendall", "Kendall τ"], ["pcc", "PCC"], ["prcc", "PRCC"]].filter(([k]) => M[k] && !M[k].skipped);
    const cols = [JB, JR, JG, JI, JO];
    const left = 168, pw = 340, gh = 14 + keys.length * 9, top = 30, W = left + pw + 70, H = top + d * gh + 58;
    const sx = scale(-1, 1, left, left + pw);
    let s = "";
    [-1, -0.5, 0, 0.5, 1].forEach((v) => { s += L(sx(v), top - 6, sx(v), top + d * gh, v === 0 ? RULE : HAIR, v === 0 ? 1 : 0.7); s += T(sx(v), top + d * gh + 13, String(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    res.consensus.order.forEach((i, row) => {
      const y0 = top + row * gh;
      if (row % 2 === 0) s += R(left, y0, pw, gh, "#f6f5f1");
      s += C(14, y0 + gh / 2, 4, SA.inputColor(i));
      s += T(24, y0 + gh / 2 + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      keys.forEach(([k, lab], q) => {
        const v = M[k].value[i], ci = (M[k].ci || [])[i] || [], p = (M[k].p || [])[i];
        if (v === null || v === undefined) return;
        const y = y0 + 9 + q * 9;
        if (ci[0] !== null && ci[0] !== undefined) s += L(sx(Math.max(-1, ci[0])), y, sx(Math.min(1, ci[1])), y, cols[q], 1.3);
        const sig = p !== null && p !== undefined && p < 0.05;
        s += `<circle cx="${r2(sx(v))}" cy="${r2(y)}" r="3.4" fill="${sig ? cols[q] : "#fff"}" stroke="${cols[q]}" stroke-width="1.4" data-tip="${esc(`${nameOf(res, i)} · ${lab} = ${fmt(v)}  95 % CI [${fmt(ci[0])}, ${fmt(ci[1])}]  p = ${fmt(p)}`)}"/>`;
        if (sig) s += T(left + pw + 6, y + 3.3, stars(p), { size: 9.5, fill: cols[q] });
      });
    });
    s += T(left + pw / 2, top + d * gh + 30, "correlation coefficient (bootstrap 95 % interval)", { anchor: "middle", size: 10.5, fill: INK });
    const lg = legend(keys.map(([, lab], q) => ({ label: lab, color: cols[q], kind: "dot" })).concat([{ label: "p ≥ 0.05", color: INK2, kind: "hollow" }]), left - 150, 14, { w: W });
    return svg(W, H, lg.s + s);
  };

  SA.paretoEffects = function (res) {
    const an = res.methods.anova;
    if (!an) return null;
    const terms = an.terms.filter((t) => t.t !== null && t.t !== undefined).sort((a, b) => Math.abs(b.t) - Math.abs(a.t)).slice(0, 22);
    const left = 200, pw = 330, rh = 17, top = 34, W = left + pw + 80, H = top + terms.length * rh + 50;
    const tmax = Math.max(an.t_crit * 1.15, ...terms.map((t) => Math.abs(t.t)));
    const sx = scale(0, tmax, left, left + pw);
    let s = "";
    ticks(0, tmax, 5).forEach((v) => { s += L(sx(v), top - 4, sx(v), top + terms.length * rh, HAIR, 0.7); s += T(sx(v), top + terms.length * rh + 13, fmt(v), { anchor: "middle", size: 9.5 }); });
    terms.forEach((t, k) => {
      const y = top + k * rh;
      const sig = Math.abs(t.t) >= an.t_crit;
      s += T(left - 8, y + rh / 2 + 3.5, short(`${t.symbol}  ${t.name}`, 34), { anchor: "end", size: 9.5, fill: sig ? INK : MUTED });
      s += R(left, y + 2.5, sx(Math.abs(t.t)) - left, rh - 5, KIND_COL[t.kind], { op: sig ? 1 : 0.35, rx: 1.5, tip: `${t.name}\ncoefficient ${fmt(t.coef)} (coded −1…1), t = ${fmt(t.t)}, p = ${fmt(t.p)}\nη² = ${fmt(t.eta2)}${t.coef > 0 ? " · raises" : " · lowers"} the response` });
      s += T(sx(Math.abs(t.t)) + 4, y + rh / 2 + 3.3, `${t.coef > 0 ? "+" : "−"}${stars(t.p)}`, { size: 9, fill: INK2 });
    });
    s += L(left, top + terms.length * rh, left + pw, top + terms.length * rh, INK, 0.9) + L(left, top - 4, left, top + terms.length * rh, INK, 0.9);
    s += L(sx(an.t_crit), top - 8, sx(an.t_crit), top + terms.length * rh, JR, 1.4, "4,3");
    s += T(sx(an.t_crit) + 4, top - 10, an.lenth ? `Lenth margin (α = 0.05)` : `t crit = ${fmt(an.t_crit)} (α = 0.05)`, { size: 9, fill: JK });
    s += T(left + pw / 2, top + terms.length * rh + 30, "|t| — standardised effect", { anchor: "middle", size: 10.5, fill: INK });
    const lg = legend([{ label: "main", color: KIND_COL.main }, { label: "quadratic", color: KIND_COL.quad }, { label: "two-factor interaction", color: KIND_COL.inter }], 10, 14);
    return svg(W, H, lg.s + s);
  };

  // where the variation of the response surface comes from
  SA.varianceDonut = function (res) {
    const an = res.methods.anova;
    if (!an) return null;
    const d = res.d, seg = [];
    for (let i = 0; i < d; i++) {
      const e = an.terms.filter((t) => t.kind !== "inter" && t.inputs[0] === i).reduce((a, t) => a + (t.eta2 || 0), 0);
      if (e > 0) seg.push({ label: nameOf(res, i), v: e, color: SA.inputColor(i) });
    }
    const ei = an.terms.filter((t) => t.kind === "inter").reduce((a, t) => a + (t.eta2 || 0), 0);
    if (ei > 0) seg.push({ label: "interactions", v: ei, color: "#3b3a36", hatch: true });
    const rv = Math.max(0, an.residual_share || 0);
    if (rv > 0) seg.push({ label: "residual", v: rv, color: "#d9d8d1" });
    const tot = seg.reduce((a, b) => a + b.v, 0) || 1;
    const W = 520, H = Math.max(230, 40 + seg.length * 20), cx = 120, cy = H / 2, ro = 92, ri = 58;
    let s = "", a0 = -Math.PI / 2;
    seg.forEach((g) => {
      const a1 = a0 + (2 * Math.PI * g.v) / tot, large = a1 - a0 > Math.PI ? 1 : 0;
      const p = (r, a) => `${r2(cx + r * Math.cos(a))},${r2(cy + r * Math.sin(a))}`;
      const dpath = `M${p(ro, a0)}A${ro},${ro} 0 ${large} 1 ${p(ro, a1)}L${p(ri, a1)}A${ri},${ri} 0 ${large} 0 ${p(ri, a0)}Z`;
      s += `<path d="${dpath}" fill="${g.color}" stroke="#fff" stroke-width="2" data-tip="${esc(`${g.label}: ${(100 * g.v / tot).toFixed(1)} % of the total variation`)}"/>`;
      if (g.hatch) s += `<g color="#ffffff" pointer-events="none"><path d="${dpath}" fill="url(#hatch)" fill-opacity="0.35"/></g>`;
      a0 = a1;
    });
    s += T(cx, cy - 4, `R² ${fmt(an.r2)}`, { anchor: "middle", size: 13, fill: INK, weight: 700 });
    s += T(cx, cy + 13, an.q2 !== null && an.q2 !== undefined ? `Q² ${fmt(an.q2)}` : "", { anchor: "middle", size: 10, fill: INK2 });
    seg.forEach((g, k) => {
      const y = cy - (seg.length * 20) / 2 + k * 20 + 12;
      s += R(250, y - 9, 12, 12, g.color, { rx: 2 });
      s += T(270, y + 1, short(g.label, 26), { size: 10.5, fill: INK });
      s += T(W - 16, y + 1, `${(100 * g.v / tot).toFixed(1)} %`, { size: 10.5, fill: INK, anchor: "end", weight: 700 });
    });
    return svg(W, H, s);
  };

  SA.srcBars = function (res) {
    const M = res.methods, d = res.d;
    const ks = [["src", "SRC", JB], ["srrc", "SRRC", JR]].filter(([k]) => M[k] && !M[k].skipped);
    if (!ks.length) return null;
    const left = 168, pw = 330, gh = 26, top = 30, W = left + pw + 60, H = top + d * gh + 50;
    const all = ks.flatMap(([k]) => M[k].ci.flat().concat(M[k].value)).map(Number).filter(isFinite);
    const m = Math.max(0.2, ...all.map(Math.abs)) * 1.05;
    const sx = scale(-m, m, left, left + pw);
    let s = "";
    ticks(-m, m, 6).forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * gh, v === 0 ? RULE : HAIR, v === 0 ? 1 : 0.7); s += T(sx(v), top + d * gh + 13, fmt(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    res.consensus.order.forEach((i, row) => {
      const y0 = top + row * gh;
      s += C(14, y0 + gh / 2, 4, SA.inputColor(i)) + T(24, y0 + gh / 2 + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      ks.forEach(([k, lab, col], q) => {
        const v = M[k].value[i], ci = M[k].ci[i] || [], y = y0 + 6 + q * 9;
        s += R(Math.min(sx(0), sx(v)), y - 3.5, Math.abs(sx(v) - sx(0)), 7, col, { rx: 1.5, tip: `${nameOf(res, i)} · ${lab} = ${fmt(v)} [${fmt(ci[0])}, ${fmt(ci[1])}], p = ${fmt((M[k].p || [])[i])}` });
        if (ci[0] !== null && ci[0] !== undefined) s += L(sx(ci[0]), y, sx(ci[1]), y, INK, 0.9) + L(sx(ci[0]), y - 2.5, sx(ci[0]), y + 2.5, INK, 0.9) + L(sx(ci[1]), y - 2.5, sx(ci[1]), y + 2.5, INK, 0.9);
      });
    });
    s += T(left + pw / 2, top + d * gh + 30, "standardised coefficient (95 % interval)", { anchor: "middle", size: 10.5, fill: INK });
    const lg = legend(ks.map(([k, lab, col]) => ({ label: `${lab} (R² ${fmt(M[k].r2)})`, color: col })), left - 150, 14);
    return svg(W, H, lg.s + s);
  };

  SA.dependenceBars = function (res) {
    const M = res.methods, d = res.d;
    const ks = [["corr_ratio", "Correlation ratio η²", JG, null], ["mi", "Mutual information", JI, "p"], ["dcor", "Distance correlation", JO, "p"], ["hsic", "HSIC (normalised)", JB, "p"]].filter(([k]) => M[k]);
    const left = 168, pw = 320, gh = 12 + ks.length * 8, top = 34, W = left + pw + 70, H = top + d * gh + 48;
    let s = "";
    const sx = scale(0, 1, left, left + pw);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * gh, HAIR, 0.7); s += T(sx(v), top + d * gh + 13, String(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    const mx = {};
    ks.forEach(([k]) => (mx[k] = Math.max(1e-12, ...num(M[k].value).filter(isFinite))));
    res.consensus.order.forEach((i, row) => {
      const y0 = top + row * gh;
      s += C(14, y0 + gh / 2, 4, SA.inputColor(i)) + T(24, y0 + gh / 2 + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      ks.forEach(([k, lab, col], q) => {
        const raw = +M[k].value[i], v = k === "mi" ? raw / mx[k] : raw, p = M[k].p ? M[k].p[i] : null;
        const y = y0 + 6 + q * 8;
        s += R(left, y - 3, sx(Math.max(0, Math.min(1, v))) - left, 6, col, { rx: 1.5, op: p !== null && p >= 0.05 ? 0.35 : 1, tip: `${nameOf(res, i)} · ${lab} = ${fmt(raw)}${k === "mi" ? " nats" : ""}${p !== null ? `, permutation p = ${fmt(p)}` : ""}` });
        if (p !== null && p < 0.05) s += T(sx(Math.max(0, Math.min(1, v))) + 3, y + 3, stars(p), { size: 8.5, fill: col });
      });
    });
    s += T(left + pw / 2, top + d * gh + 30, "dependence (mutual information scaled to its largest)", { anchor: "middle", size: 10.5, fill: INK });
    const lg = legend(ks.map(([, lab, col]) => ({ label: lab, color: col })).concat([{ label: "faded: p ≥ 0.05", color: "#c9c7bf" }]), 10, 14, { w: W - 20 });
    return svg(W, H + lg.h - 14, `<g transform="translate(0,${lg.h - 14})">${s}</g>` + lg.s);
  };

  SA.lassoPath = function (res) {
    const la = res.methods.lasso;
    if (!la) return null;
    const d = res.d, al = la.alphas.map(Number), lx = al.map((a) => Math.log10(a));
    const left = 56, top = 20, pw = 400, ph = 230, W = left + pw + 150, H = top + ph + 50;
    const xs = [Math.max(...lx), Math.min(...lx)];
    const ys = extent(la.coefs.flat().map(Number), 0.08);
    const ax = axes(left, top, pw, ph, xs, ys, { xlabel: "log₁₀ penalty α (weaker →)", ylabel: "standardised coefficient", grid: true });
    let s = ax.s + L(left, ax.sy(0), left + pw, ax.sy(0), RULE, 1);
    const ends = [];
    for (let i = 0; i < d; i++) {
      const col = SA.inputColor(i);
      s += P(line(lx, la.coefs[i].map(Number), ax.sx, ax.sy), col, 2, { tip: `${nameOf(res, i)} enters at α = ${fmt(la.enter[i])}` });
      ends.push([ax.sy(+la.coefs[i][la.coefs[i].length - 1]), i]);
    }
    ends.sort((a, b) => a[0] - b[0]);
    let last = -1e9;
    ends.forEach(([y, i]) => { const yy = Math.max(y, last + 11); last = yy; s += T(left + pw + 6, yy + 3.5, short(nameOf(res, i), 20), { size: 9.5, fill: INK }); s += L(left + pw, y, left + pw + 4, yy, RULE, 0.6); });
    if (la.cv && la.cv.alpha) {
      const x = ax.sx(Math.log10(la.cv.alpha));
      s += L(x, top, x, top + ph, JR, 1.4, "4,3") + T(x + 4, top + 10, "α by cross-validation", { size: 9, fill: JK });
    }
    return svg(W, H, s);
  };

  // the mean response at each level (data, ± SE) beside the model's partial dependence and its ICE band
  SA.effects = function (res) {
    const d = res.d, pd = res.pdp || [], me = res.main_effects || [], al = res.ale || [];
    const cols = Math.min(4, d), rows = Math.ceil(d / cols), cw = 170, chh = 150, W = cols * cw + 24, H = rows * (chh + 40) + 30;
    const yAll = [].concat(...pd.map((p) => p.lo.concat(p.hi)), ...me.map((l) => l.map((v) => v.mean)), res.data.y);
    const yr = extent(yAll.map(Number), 0.06);
    let s = "";
    for (let i = 0; i < d; i++) {
      const cx = 46 + (i % cols) * cw, cy = 14 + Math.floor(i / cols) * (chh + 40), pw = cw - 54, ph = chh - 30;
      const inp = res.inputs[i];
      const ax = axes(cx, cy, pw, ph, [inp.lo, inp.hi], yr, { nx: 3, ny: 4, ts: 8.5, xlabel: short(inp.name + (inp.unit ? ` [${inp.unit}]` : ""), 28), xlo: 25, ls: 9.5, noY: i % cols !== 0 });
      s += ax.s;
      const col = SA.inputColor(i), p = pd[i];
      if (p) {
        (p.ice || []).forEach((c) => { s += P(line(p.x, c.map(Number), ax.sx, ax.sy), col, 0.7, { op: 0.3 }); });
        const band = line(p.x, p.hi.map(Number), ax.sx, ax.sy) + p.x.slice().reverse().map((x, k) => `L${r2(ax.sx(x))},${r2(ax.sy(+p.lo[p.lo.length - 1 - k]))}`).join("") + "Z";
        s += `<path d="${band}" fill="${col}" fill-opacity="0.2" stroke="none"/>`;
        s += P(line(p.x, p.pd.map(Number), ax.sx, ax.sy), col, 2.4, { tip: `${inp.name}: partial dependence from ${fmt(Math.min(...p.pd))} to ${fmt(Math.max(...p.pd))}` });
        const a = al[i];
        if (a && a.x && a.x.length > 1) {
          const m = mean(p.pd.map(Number));
          s += P(line(a.x.map(Number), a.ale.map((v) => m + +v), ax.sx, ax.sy), INK, 1.3, { dash: "4,3", tip: `${inp.name}: accumulated local effect (ALE), centred on the partial dependence's mean` });
        }
      }
      (me[i] || []).forEach((v) => {
        if (v.se) s += L(ax.sx(v.x), ax.sy(v.mean - v.se), ax.sx(v.x), ax.sy(v.mean + v.se), INK, 1);
        s += C(ax.sx(v.x), ax.sy(v.mean), 3.6, INK, { sw: 1, tip: `${inp.name} = ${fmt(v.x)}: mean ${fmt(v.mean)} ± ${fmt(v.se)} (${v.n} run${v.n === 1 ? "" : "s"})` });
      });
      res.data.X.forEach((x) => { s += L(ax.sx(+x[i]), cy + ph, ax.sx(+x[i]), cy + ph - 4, MUTED, 0.6); });
      s += C(cx + 6, cy - 4, 3.6, col) + T(cx + 13, cy - 0.5, `x${i + 1}`, { size: 9.5, fill: INK, weight: 700 });
    }
    const lg = legend([{ label: "partial dependence (model)", color: INK2, kind: "line" }, { label: "ICE curves and 10–90 % band (thin, in the input's colour)", color: "#9aa1aa", kind: "line" }, { label: "ALE", color: INK, kind: "dash" }, { label: "mean of the runs ± SE", color: INK, kind: "dot" }], 46, H - 6, { w: W - 50 });
    return svg(W, H + lg.h - 8, s + lg.s);
  };

  /* ------------------------------------------------------------ machine learning */
  SA.parity = function (res) {
    const ks = Object.keys(res.models || {}).filter((k) => res.models[k].pred);
    const cols = Math.min(3, ks.length), rows = Math.ceil(ks.length / cols), cw = 190, chh = 172, W = cols * cw + 20, H = rows * chh + 10;
    const y = res.data.y.map(Number);
    let s = "";
    ks.forEach((k, j) => {
      const m = res.models[k], pr = m.pred.map(Number);
      const lim = extent(y.concat(pr.filter(isFinite)), 0.06);
      const cx = 46 + (j % cols) * cw, cy = 22 + Math.floor(j / cols) * chh, pw = cw - 60, ph = chh - 60;
      const ax = axes(cx, cy, pw, ph, lim, lim, { nx: 3, ny: 3, ts: 8.5, xlabel: j >= ks.length - cols ? "run" : "", ylabel: j % cols === 0 ? "predicted (held out)" : "", ylo: 34, xlo: 24, ls: 9.5 });
      s += ax.s + L(ax.sx(lim[0]), ax.sy(lim[0]), ax.sx(lim[1]), ax.sy(lim[1]), RULE, 1, "4,3");
      const good = m.q2 !== null && m.q2 >= 0.5;
      pr.forEach((p, i) => { if (isFinite(p)) s += C(ax.sx(y[i]), ax.sy(p), 3, good ? JB : "#9a9890", { op: 0.9, tip: `${res.data.labels[i]}: run ${fmt(y[i])}, predicted ${fmt(p)}` }); });
      const roles = [res.surrogate.global === k ? "surrogate" : "", res.surrogate.explainer === k ? "explained" : ""].filter(Boolean);
      s += T(cx, cy - 8, short(m.label, roles.length ? 22 : 30) + (roles.length ? ` · ${roles.join(", ")}` : ""), { size: 9.5, fill: INK, weight: 700 });
      s += T(cx + 4, cy + 11, `Q² ${fmt(m.q2)}`, { size: 9.5, fill: good ? JB : INK2, weight: 700 });
      s += T(cx + 4, cy + 23, `RMSE ${fmt(m.rmse)}`, { size: 8.5, fill: INK2 });
    });
    return svg(W, H, s);
  };

  SA.permBox = function (res) {
    const M = res.methods, d = res.d;
    const ks = [["rf_perm", "Random forest", JG], ["gbm_perm", "Gradient boosting", JI]].filter(([k]) => M[k] && M[k].draws);
    if (!ks.length) return null;
    const left = 168, pw = 320, gh = 14 + ks.length * 12, top = 30, W = left + pw + 110, H = top + d * gh + 50;
    const all = ks.flatMap(([k]) => M[k].draws.flat().map(Number)).filter(isFinite);
    const q = (a, p) => { const b = a.slice().sort((x, y) => x - y); const h = (b.length - 1) * p, i = Math.floor(h); return b[i] + (b[Math.min(i + 1, b.length - 1)] - b[i]) * (h - i); };
    const xr = [Math.min(0, q(all, 0.02)), Math.max(0.05, q(all, 0.98)) * 1.05];
    const sx = scale(xr[0], xr[1], left, left + pw);
    let s = "";
    ticks(xr[0], xr[1], 5).forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * gh, v === 0 ? RULE : HAIR, v === 0 ? 1 : 0.7); s += T(sx(v), top + d * gh + 13, fmt(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    const mdi = M.rf_mdi ? num(M.rf_mdi.value) : null;
    res.consensus.order.forEach((i, row) => {
      const y0 = top + row * gh;
      s += C(14, y0 + gh / 2, 4, SA.inputColor(i)) + T(24, y0 + gh / 2 + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      ks.forEach(([k, lab, col], j) => {
        const v = M[k].draws.map((r) => +r[i]).filter(isFinite);
        if (!v.length) return;
        const y = y0 + 8 + j * 12, cl = (x) => Math.max(xr[0], Math.min(xr[1], x));
        const [p05, p25, p50, p75, p95] = [0.05, 0.25, 0.5, 0.75, 0.95].map((p) => cl(q(v, p)));
        s += L(sx(p05), y, sx(p25), y, col, 1) + L(sx(p75), y, sx(p95), y, col, 1);
        s += R(sx(p25), y - 4, sx(p75) - sx(p25), 8, col, { op: 0.5, stroke: col, sw: 1.2, tip: `${nameOf(res, i)} · ${lab}: median ${fmt(q(v, 0.5))}, 25–75 % ${fmt(q(v, 0.25))}…${fmt(q(v, 0.75))} (${v.length} draws)` });
        s += L(sx(p50), y - 4, sx(p50), y + 4, col, 2);
      });
      if (mdi) s += T(left + pw + 14, y0 + gh / 2 + 3.5, `${(100 * mdi[i]).toFixed(1)} %`, { size: 9.5, fill: INK2 });
    });
    if (mdi) s += T(left + pw + 14, top - 8, "RF impurity", { size: 9, fill: MUTED });
    s += T(left + pw / 2, top + d * gh + 30, "rise of the held-out squared error when shuffled (share of Var y)", { anchor: "middle", size: 10.5, fill: INK });
    const lg = legend(ks.map(([, lab, col]) => ({ label: lab, color: col })), left - 150, 14);
    return svg(W, H, lg.s + s);
  };

  SA.beeswarm = function (res) {
    const sh = res.methods.shap;
    if (!sh) return null;
    const d = res.d, X = res.data.X, phi = sh.phi;
    const order = d ? Array.from({ length: d }, (_, i) => i).sort((a, b) => sh.value[b] - sh.value[a]) : [];
    const left = 168, pw = 330, rh = 30, top = 18, W = left + pw + 90, H = top + d * rh + 70;
    const all = phi.flat().map(Number);
    const m = Math.max(1e-12, ...all.map(Math.abs)) * 1.06;
    const sx = scale(-m, m, left, left + pw);
    let s = "";
    ticks(-m, m, 6).forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * rh, v === 0 ? RULE : HAIR, v === 0 ? 1 : 0.7); s += T(sx(v), top + d * rh + 13, fmt(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * rh, left + pw, top + d * rh, INK, 0.9);
    order.forEach((i, row) => {
      const yc = top + row * rh + rh / 2;
      s += C(14, yc, 4, SA.inputColor(i)) + T(24, yc + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      s += T(left + pw + 8, yc + 3.5, fmt(sh.value[i]), { size: 9.5, fill: INK2 });
      const xs = X.map((r) => +r[i]), lo = Math.min(...xs), hi = Math.max(...xs);
      // stack points that would overlap
      const pts = phi.map((r, k) => ({ v: +r[i], k })).sort((a, b) => a.v - b.v);
      const bins = {};
      pts.forEach((p) => {
        const b = Math.round(sx(p.v) / 6.2);
        const n = (bins[b] = (bins[b] || 0) + 1) - 1;
        const off = (n % 2 ? -1 : 1) * Math.ceil(n / 2) * 4.2;
        const t = hi > lo ? (xs[p.k] - lo) / (hi - lo) : 0.5;
        s += C(sx(p.v), yc + Math.max(-rh / 2 + 3, Math.min(rh / 2 - 3, off)), 3.1, SA.seq(t), { sw: 0.5, tip: `${res.data.labels[p.k]}: ${nameOf(res, i)} = ${fmt(xs[p.k])}, SHAP ${fmt(p.v)}` });
      });
    });
    s += T(left + pw + 8, top - 6, "mean |φ|", { size: 9, fill: MUTED });
    s += T(left + pw / 2, top + d * rh + 30, `SHAP value — contribution to the response (base ${fmt(sh.base)})`, { anchor: "middle", size: 10.5, fill: INK });
    s += colorbar(left + pw / 2 - 80, top + d * rh + 46, 160, 7, 0, 1, "", SA.seq, { lolab: "input low", hilab: "input high" });
    return svg(W, H, s);
  };

  // the contribution of the leading input against its value, coloured by its strongest partner
  SA.shapDependence = function (res, which) {
    const sh = res.methods.shap;
    if (!sh) return null;
    const d = res.d, order = Array.from({ length: d }, (_, i) => i).sort((a, b) => sh.value[b] - sh.value[a]);
    const i = which === undefined ? order[0] : which;
    let j = null;
    if (sh.inter_mean_abs) { let best = -1; for (let k = 0; k < d; k++) if (k !== i && sh.inter_mean_abs[i][k] > best) { best = sh.inter_mean_abs[i][k]; j = k; } }
    else if (d > 1) j = order.find((k) => k !== i);
    const X = res.data.X, xs = X.map((r) => +r[i]), ys = sh.phi.map((r) => +r[i]);
    const left = 58, top = 20, pw = 360, ph = 220, W = left + pw + 110, H = top + ph + 50;
    const ax = axes(left, top, pw, ph, extent(xs), extent(ys, 0.1), { xlabel: `${res.inputs[i].name}${res.inputs[i].unit ? ` [${res.inputs[i].unit}]` : ""}`, ylabel: `SHAP value of x${i + 1}`, grid: true });
    let s = ax.s + L(left, ax.sy(0), left + pw, ax.sy(0), RULE, 1);
    const cs = j !== null ? X.map((r) => +r[j]) : null, clo = cs ? Math.min(...cs) : 0, chi = cs ? Math.max(...cs) : 1;
    xs.forEach((x, k) => { const t = cs && chi > clo ? (cs[k] - clo) / (chi - clo) : 0.5; s += C(ax.sx(x), ax.sy(ys[k]), 4.2, cs ? SA.seq(t) : SA.inputColor(i), { sw: 0.9, tip: `${res.data.labels[k]}: x${i + 1} = ${fmt(x)}${cs ? `, x${j + 1} = ${fmt(cs[k])}` : ""}, SHAP ${fmt(ys[k])}` }); });
    if (cs) s += colorbar(left + pw + 26, top + 20, 9, 140, clo, chi, short(res.inputs[j].name, 16), SA.seq, { vertical: true });
    return svg(W, H, s);
  };

  /* ------------------------------------------------------------ variance-based and screening */
  SA.sobolBars = function (res) {
    const M = res.methods, d = res.d;
    const est = [["sobol", "Sobol (Saltelli, MC)"], ["pce", "PCE (closed form)"], ["efast", "eFAST"]].filter(([k]) => M[k] && M[k].S1);
    if (!est.length) return null;
    const left = 168, pw = 330, gh = 10 + est.length * 11, top = 34, W = left + pw + 70, H = top + d * gh + 50;
    const sx = scale(0, 1, left, left + pw);
    let s = "";
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * gh, HAIR, 0.7); s += T(sx(v), top + d * gh + 13, String(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    const order = Array.from({ length: d }, (_, i) => i).sort((a, b) => (M[est[0][0]].ST[b] || 0) - (M[est[0][0]].ST[a] || 0));
    order.forEach((i, row) => {
      const y0 = top + row * gh;
      s += C(14, y0 + gh / 2, 4, SA.inputColor(i)) + T(24, y0 + gh / 2 + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      est.forEach(([k, lab], q) => {
        const e = M[k], s1 = Math.max(0, +e.S1[i]), st = Math.max(s1, +e.ST[i]), y = y0 + 6 + q * 11, col = SA.inputColor(i);
        s += R(left, y - 4, sx(Math.min(1, s1)) - left, 8, col, { op: [1, 0.8, 0.6][q], tip: `${nameOf(res, i)} · ${lab}: S₁ = ${fmt(e.S1[i])}, S_T = ${fmt(e.ST[i])}${e.S1_ci ? `\nS₁ 95 % [${fmt(e.S1_ci[i][0])}, ${fmt(e.S1_ci[i][1])}], S_T 95 % [${fmt(e.ST_ci[i][0])}, ${fmt(e.ST_ci[i][1])}]` : ""}` });
        s += `<g color="${col}">${R(sx(Math.min(1, s1)), y - 4, sx(Math.min(1, st)) - sx(Math.min(1, s1)), 8, "url(#hatch)", { stroke: col, sw: 0.7 })}</g>`;
        if (e.ST_ci && e.ST_ci[i] && e.ST_ci[i][0] !== null) s += L(sx(Math.max(0, e.ST_ci[i][0])), y, sx(Math.min(1, e.ST_ci[i][1])), y, INK, 0.9);
        s += T(left + pw + 6, y + 3, ["MC", "PCE", "FAST"][q], { size: 8, fill: MUTED });
      });
    });
    s += T(left + pw / 2, top + d * gh + 30, "share of the response's variance", { anchor: "middle", size: 10.5, fill: INK });
    const lg = legend([{ label: "first order S₁ (alone)", color: "#5f5e58" }, { label: "S_T − S₁ (through interactions)", color: "#5f5e58", kind: "hatch" }, { label: "rows: Sobol MC · PCE · eFAST", color: "#fff" }], 10, 14, { w: W - 20 });
    return svg(W, H, lg.s + s);
  };

  SA.morris = function (res) {
    const mo = res.methods.morris;
    if (!mo) return null;
    const d = res.d, mu = num(mo.mu_star), sg = num(mo.sigma);
    const left = 60, top = 22, pw = 380, ph = 250, W = left + pw + 140, H = top + ph + 50;
    const mx = Math.max(1e-12, ...mu, ...num((mo.mu_star_ci || []).map((c) => c[1]))) * 1.08, my = Math.max(1e-12, ...sg) * 1.12;
    const ax = axes(left, top, pw, ph, [0, mx], [0, my], { xlabel: "μ* — mean |elementary effect| over the full range", ylabel: "σ — spread of the effects", grid: false });
    let s = ax.s;
    [[0.1, "linear"], [0.5, "monotonic"], [1, "non-linear or interacting"]].forEach(([k, lab]) => {
      const xe = Math.min(mx, my / k), ye = k * xe;
      s += L(ax.sx(0), ax.sy(0), ax.sx(xe), ax.sy(ye), "#b9b7ad", 1, "4,3");
      s += T(ax.sx(xe) + 3, ax.sy(ye) + (xe < mx ? 10 : -3), `σ/μ* = ${k} · ${lab}`, { size: 8.5, fill: MUTED, anchor: xe < mx ? "start" : "end" });
    });
    const lab = [];
    for (let i = 0; i < d; i++) {
      const ci = (mo.mu_star_ci || [])[i];
      if (ci && ci[0] !== null) s += L(ax.sx(ci[0]), ax.sy(sg[i]), ax.sx(ci[1]), ax.sy(sg[i]), SA.inputColor(i), 1.2);
      s += C(ax.sx(mu[i]), ax.sy(sg[i]), 6.5, SA.inputColor(i), { sw: 1.6, tip: `${nameOf(res, i)}: μ* = ${fmt(mu[i])}, σ = ${fmt(sg[i])}, μ = ${fmt(mo.mu[i])} (σ/μ* ${fmt(sg[i] / (mu[i] || 1e-30))})` });
      lab.push([ax.sy(sg[i]), ax.sx(mu[i]), i]);
    }
    lab.sort((a, b) => a[0] - b[0]);
    let last = -1e9;
    lab.forEach(([y, x, i]) => { const yy = Math.max(y - 7, last + 11); last = yy; s += T(x + 8, yy + 3, short(nameOf(res, i), 18), { size: 9.5, fill: INK }); });
    s += T(left + pw, top + ph + 42, `${mo.r} trajectories · ${mo.levels} levels`, { size: 9, fill: MUTED, anchor: "end" });
    return svg(W, H, s);
  };

  SA.dgsm = function (res) {
    const g = res.methods.dgsm, sb = res.methods.sobol;
    if (!g) return null;
    const d = res.d, left = 168, pw = 330, gh = 26, top = 30, W = left + pw + 80, H = top + d * gh + 50;
    const b = num(g.bound), st = sb ? num(sb.ST) : null;
    const mx = Math.min(1.5, Math.max(0.05, ...b.filter(isFinite), ...(st || [0])) * 1.05);
    const sx = scale(0, mx, left, left + pw);
    let s = "";
    ticks(0, mx, 5).forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * gh, HAIR, 0.7); s += T(sx(v), top + d * gh + 13, fmt(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    const order = Array.from({ length: d }, (_, i) => i).sort((a, c) => b[c] - b[a]);
    order.forEach((i, row) => {
      const y = top + row * gh + gh / 2;
      s += C(14, y, 4, SA.inputColor(i)) + T(24, y + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      s += R(left, y - 6, sx(Math.min(mx, b[i])) - left, 12, SA.inputColor(i), { op: 0.75, rx: 1.5, tip: `${nameOf(res, i)}: ν = ${fmt(g.nu[i])}, bound on S_T ${fmt(b[i])}` });
      if (b[i] > mx) s += T(left + pw + 2, y + 3.5, "›", { size: 12, fill: INK });
      if (st) s += `<path d="M${r2(sx(Math.max(0, st[i])))},${r2(y - 6)}l4,6l-4,6l-4,-6z" fill="${INK}" data-tip="${esc(`Sobol S_T ${fmt(st[i])}`)}"/>`;
    });
    s += T(left + pw / 2, top + d * gh + 30, "DGSM upper bound ν/(π²V) on S_T (bar) · Sobol S_T (◆)", { anchor: "middle", size: 10.5, fill: INK });
    return svg(W, H, s);
  };

  SA.pawn = function (res) {
    const p = res.methods.pawn;
    if (!p) return null;
    const d = res.d, left = 168, pw = 330, gh = 26, top = 30, W = left + pw + 60, H = top + d * gh + 50;
    const med = num(p.median), mx = num(p.max);
    const xm = Math.max(0.1, ...mx, ...med) * 1.08;
    const sx = scale(0, xm, left, left + pw);
    let s = "";
    ticks(0, xm, 5).forEach((v) => { s += L(sx(v), top - 4, sx(v), top + d * gh, HAIR, 0.7); s += T(sx(v), top + d * gh + 13, fmt(v), { anchor: "middle", size: 9.5 }); });
    s += L(left, top + d * gh, left + pw, top + d * gh, INK, 0.9);
    s += `<rect x="${r2(left)}" y="${r2(top - 4)}" width="${r2(sx(p.ks_crit) - left)}" height="${r2(d * gh + 4)}" fill="#efeee8"/>`;
    s += L(sx(p.ks_crit), top - 8, sx(p.ks_crit), top + d * gh, JR, 1.3, "4,3") + T(sx(p.ks_crit) + 4, top - 10, "sampling noise (5 %)", { size: 9, fill: JK });
    const order = Array.from({ length: d }, (_, i) => i).sort((a, b) => med[b] - med[a]);
    order.forEach((i, row) => {
      const y = top + row * gh + gh / 2, ci = (p.median_ci || [])[i];
      s += C(14, y, 4, SA.inputColor(i)) + T(24, y + 3.5, short(nameOf(res, i), 23), { size: 10.5, fill: INK });
      s += P(`M${r2(sx(med[i]))},${r2(y)}H${r2(sx(mx[i]))}`, SA.inputColor(i), 1.4, { op: 0.55 });
      if (ci && ci[0] !== null) s += L(sx(ci[0]), y, sx(ci[1]), y, SA.inputColor(i), 2.4);
      s += C(sx(med[i]), y, 5, SA.inputColor(i), { sw: 1.3, tip: `${nameOf(res, i)}: PAWN median KS ${fmt(med[i])}, max ${fmt(mx[i])}` });
      s += `<path d="M${r2(sx(mx[i]))},${r2(y - 5)}l4.5,8.5h-9z" fill="#fff" stroke="${INK}" stroke-width="1" data-tip="${esc(`max over the slices ${fmt(mx[i])}`)}"/>`;
    });
    s += T(left + pw / 2, top + d * gh + 30, "Kolmogorov–Smirnov distance: median (●) and maximum (△) over 10 slices", { anchor: "middle", size: 10.5, fill: INK });
    return svg(W, H, s);
  };

  /* ------------------------------------------------------------ interactions */
  SA.interHeatmaps = function (res) {
    const d = res.d, pairs = res.interactions || [];
    if (d < 2) return null;
    const ms = [["sobol_s2", "Sobol S₂"], ["pce_s2", "PCE S₂"], ["shap_int", "SHAP interaction"], ["h2", "Friedman H²"], ["anova_eta2", "ANOVA η²"]].filter(([k]) => pairs.some((p) => p[k] !== undefined && p[k] !== null));
    const cell = Math.max(14, Math.min(30, 150 / d)), side = cell * d, gap = 40, labW = d > 5 ? 0 : 0;
    const cols = Math.min(ms.length, 3), rowsN = Math.ceil(ms.length / cols);
    const W = 40 + cols * (side + gap) + 10, H = rowsN * (side + 62) + 40;
    let s = "";
    ms.forEach(([k, lab], q) => {
      const ox = 40 + (q % cols) * (side + gap) + labW, oy = 30 + Math.floor(q / cols) * (side + 62);
      const vals = pairs.map((p) => Math.max(0, +p[k] || 0)), mx = Math.max(1e-12, ...vals);
      s += T(ox + side / 2, oy - 12, lab, { anchor: "middle", size: 10, fill: INK, weight: 700 });
      for (let i = 0; i < d; i++) {
        s += T(ox - 4, oy + i * cell + cell / 2 + 3, `x${i + 1}`, { anchor: "end", size: 8.5 });
        s += T(ox + i * cell + cell / 2, oy + side + 11, `x${i + 1}`, { anchor: "middle", size: 8.5 });
        for (let j = 0; j < d; j++) {
          if (j <= i) { if (j === i) s += R(ox + j * cell + 0.5, oy + i * cell + 0.5, cell - 1, cell - 1, "#f3f2ed"); continue; }
          const p = pairs.find((e) => e.i === i && e.j === j);
          const v = p ? Math.max(0, +p[k] || 0) : 0, t = v / mx, f = SA.seq(t);
          // mirrored into the lower triangle so a row reads every partner of one input
          [[i, j], [j, i]].forEach(([a, b2]) => {
            s += R(ox + b2 * cell + 0.5, oy + a * cell + 0.5, cell - 1, cell - 1, f, { tip: `${p ? p.name : ""} · ${lab} = ${fmt(p ? p[k] : 0)}${k === "anova_eta2" && p && p.anova_p !== undefined ? `, p = ${fmt(p.anova_p)}` : ""}${k === "sobol_s2" && p && p.sobol_s2_ci ? `, 95 % [${fmt(p.sobol_s2_ci[0])}, ${fmt(p.sobol_s2_ci[1])}]` : ""}` });
          });
          if (cell >= 26) s += T(ox + j * cell + cell / 2, oy + i * cell + cell / 2 + 3, v >= 0.1 ? v.toFixed(2).replace(/^0/, "") : v.toFixed(3).replace(/^0/, ""), { anchor: "middle", size: 7.5, fill: inkOn(f) });
        }
      }
      s += T(ox + side / 2, oy + side + 26, `max ${fmt(mx)}`, { anchor: "middle", size: 8.5, fill: MUTED });
    });
    s += colorbar(40, H - 20, 160, 7, 0, 1, "", SA.seq, { lolab: "0", hilab: "the measure's largest pair" });
    return svg(W, H, s);
  };

  SA.network = function (res) {
    const d = res.d, pairs = res.interactions || [];
    if (d < 2) return null;
    const W = 540, H = 440, cx = 270, cy = 222, Rr = 138;
    const st = res.methods.sobol ? num(res.methods.sobol.ST) : res.consensus.score.map(Number);
    const smax = Math.max(1e-12, ...st.filter(isFinite));
    const pos = Array.from({ length: d }, (_, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / d; return [cx + Rr * Math.cos(a), cy + Rr * Math.sin(a), a]; });
    let s = "";
    // the number of measures that found an interaction, along the jet map: blue 1, green 2, orange 3, red 4 and more
    const ecol = ["#d6d4cc", JB, JG, JO, JR, JK];
    pairs.slice().sort((a, b) => a.score - b.score).forEach((p) => {
      const [x1, y1] = pos[p.i], [x2, y2] = pos[p.j], n = p.detected_by.length;
      if (p.score < 0.04 && !n) return;
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, qx = mx + (cx - mx) * 0.35, qy = my + (cy - my) * 0.35;
      s += `<path d="M${r2(x1)},${r2(y1)}Q${r2(qx)},${r2(qy)} ${r2(x2)},${r2(y2)}" fill="none" stroke="${ecol[Math.min(5, n)]}" stroke-width="${r2(1 + 9 * p.score)}" stroke-opacity="0.9" stroke-linecap="round" data-tip="${esc(`${p.name}\nconsensus ${fmt(p.score)} · found by ${n} of 5 measures${n ? ` (${p.detected_by.join(", ")})` : ""}`)}"/>`;
    });
    pos.forEach(([x, y, a], i) => {
      const r = 9 + 20 * Math.sqrt(Math.max(0, st[i]) / smax);
      s += C(x, y, r, SA.inputColor(i), { sw: 2, tip: `${nameOf(res, i)}: ${res.methods.sobol ? `Sobol S_T ${fmt(st[i])}` : `consensus ${fmt(st[i])}`}` });
      s += T(x, y + 3.5, `x${i + 1}`, { anchor: "middle", size: 9.5, fill: "#fff", weight: 700 });
      const lx = cx + (Rr + r + 12) * Math.cos(a), ly = cy + (Rr + r + 12) * Math.sin(a);
      s += T(lx, ly + 3, short(nameOf(res, i), 20), { anchor: Math.cos(a) > 0.3 ? "start" : Math.cos(a) < -0.3 ? "end" : "middle", size: 9.5, fill: INK });
    });
    const lg = legend([1, 2, 3, 4].map((n) => ({ label: `found by ${n}${n === 4 ? "+" : ""}`, color: ecol[n], kind: "line" })), 10, 16);
    s += T(W - 10, H - 10, `node area ∝ ${res.methods.sobol ? "Sobol S_T" : "consensus"} · edge width ∝ interaction consensus`, { size: 9, fill: MUTED, anchor: "end" });
    return svg(W, H, s + lg.s);
  };

  // marching squares: the iso-lines of a grid z[i][j] at value v, in grid coordinates
  function isolines(z, v) {
    const ni = z.length, nj = z[0].length, segs = [];
    const lerp = (a, b) => (v - a) / ((b - a) || 1e-30);
    for (let i = 0; i < ni - 1; i++) for (let j = 0; j < nj - 1; j++) {
      const a = z[i][j], b = z[i + 1][j], c = z[i + 1][j + 1], e = z[i][j + 1];
      const pts = [];
      if ((a < v) !== (b < v)) pts.push([i + lerp(a, b), j]);
      if ((b < v) !== (c < v)) pts.push([i + 1, j + lerp(b, c)]);
      if ((e < v) !== (c < v)) pts.push([i + lerp(e, c), j + 1]);
      if ((a < v) !== (e < v)) pts.push([i, j + lerp(a, e)]);
      if (pts.length === 2) segs.push(pts);
      else if (pts.length === 4) { segs.push([pts[0], pts[1]]); segs.push([pts[2], pts[3]]); }
    }
    return segs;
  }
  SA.pdp2 = function (res, k = 0) {
    const p = (res.pdp2 || [])[k];
    if (!p) return null;
    const i = p.i, j = p.j, z = p.z.map((r) => r.map(Number)), xi = p.xi.map(Number), xj = p.xj.map(Number);
    const left = 58, top = 34, pw = 330, ph = 260, W = left + pw + 120, H = top + ph + 50;
    const ax = axes(left, top, pw, ph, [xi[0], xi[xi.length - 1]], [xj[0], xj[xj.length - 1]], { xlabel: `${res.inputs[i].name}${res.inputs[i].unit ? ` [${res.inputs[i].unit}]` : ""}`, ylabel: `${res.inputs[j].name}${res.inputs[j].unit ? ` [${res.inputs[j].unit}]` : ""}` });
    const flat = z.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
    let s = "";
    const dx = pw / (xi.length - 1), dy = ph / (xj.length - 1);
    for (let a = 0; a < xi.length; a++) for (let b = 0; b < xj.length; b++) {
      const t = hi > lo ? (z[a][b] - lo) / (hi - lo) : 0.5;
      s += R(ax.sx(xi[a]) - dx / 2, ax.sy(xj[b]) - dy / 2, dx + 0.6, dy + 0.6, SA.seq(t), { tip: `x${i + 1} = ${fmt(xi[a])}, x${j + 1} = ${fmt(xj[b])}: ${fmt(z[a][b])}` });
    }
    s = `<g clip-path="url(#pdclip${k})">${s}</g>`;
    const gx = (g) => ax.sx(xi[0] + (g / (xi.length - 1)) * (xi[xi.length - 1] - xi[0]));
    const gy = (g) => ax.sy(xj[0] + (g / (xj.length - 1)) * (xj[xj.length - 1] - xj[0]));
    ticks(lo, hi, 8).forEach((v) => {
      if (v <= lo || v >= hi) return;
      isolines(z, v).forEach(([u, w]) => { s += L(gx(u[0]), gy(u[1]), gx(w[0]), gy(w[1]), "#ffffff", 0.9); });
    });
    res.data.X.forEach((x, r) => { s += C(ax.sx(+x[i]), ax.sy(+x[j]), 3.2, "#1b1b1b", { sw: 1, stroke: "#fff", tip: `${res.data.labels[r]}: response ${fmt(res.data.y[r])}` }); });
    s += ax.s;
    s += colorbar(left + pw + 30, top + 20, 10, 170, lo, hi, "", SA.seq, { vertical: true });
    s += T(left + pw + 26, top - 12, short(res.response.label, 24) + (res.response.unit && res.response.unit !== "-" ? ` [${res.response.unit}]` : ""), { size: 9.5, fill: INK, anchor: "middle" });
    return svg(W, H, s, `<clipPath id="pdclip${k}"><rect x="${left}" y="${top}" width="${pw}" height="${ph}"/></clipPath>`);
  };

  // the classic interaction plot of the runs: the response against one input at a low and a high level of the other
  SA.interactionPlot = function (res, pair) {
    const pr = pair || (res.interactions || [])[0];
    if (!pr) return null;
    const i = pr.i, j = pr.j, X = res.data.X.map((r) => r.map(Number)), y = res.data.y.map(Number);
    const xj = X.map((r) => r[j]), uj = [...new Set(xj)].sort((a, b) => a - b);
    let groups;
    if (uj.length <= 3) groups = uj.map((v) => ({ label: `x${j + 1} = ${fmt(v)}`, m: xj.map((x) => x === v) }));
    else { const med = xj.slice().sort((a, b) => a - b)[Math.floor(xj.length / 2)]; groups = [{ label: `x${j + 1} low (< ${fmt(med)})`, m: xj.map((x) => x < med) }, { label: `x${j + 1} high (≥ ${fmt(med)})`, m: xj.map((x) => x >= med) }]; }
    const xi = X.map((r) => r[i]), ui = [...new Set(xi)].sort((a, b) => a - b);
    const bins = ui.length <= 6 ? ui.map((v) => [v, v]) : (() => { const q = xi.slice().sort((a, b) => a - b), n = 4; return Array.from({ length: n }, (_, k) => [q[Math.floor((k * q.length) / n)], q[Math.min(q.length - 1, Math.floor(((k + 1) * q.length) / n) - 1)]]); })();
    const left = 58, top = 20, pw = 360, ph = 210, W = left + pw + 150, H = top + ph + 50;
    const ax = axes(left, top, pw, ph, extent(xi), extent(y, 0.1), { xlabel: `${res.inputs[i].name}${res.inputs[i].unit ? ` [${res.inputs[i].unit}]` : ""}`, ylabel: `${short(res.response.label, 28)}${res.response.unit && res.response.unit !== "-" ? ` [${res.response.unit}]` : ""}`, grid: true });
    let s = ax.s;
    const cols = [JB, JR, JG];
    groups.forEach((g, q) => {
      const pts = bins.map(([a, b]) => { const sel = y.filter((_, r) => g.m[r] && xi[r] >= a && xi[r] <= b); return sel.length ? [xi.filter((_, r) => g.m[r] && xi[r] >= a && xi[r] <= b).reduce((s2, v) => s2 + v, 0) / sel.length, mean(sel), sel.length] : null; }).filter(Boolean);
      y.forEach((v, r) => { if (g.m[r]) s += C(ax.sx(xi[r]), ax.sy(v), 2.4, cols[q], { op: 0.35, stroke: false }); });
      s += P(pts.map(([a, b], k) => `${k ? "L" : "M"}${r2(ax.sx(a))},${r2(ax.sy(b))}`).join(""), cols[q], 2.2);
      pts.forEach(([a, b, n]) => { s += C(ax.sx(a), ax.sy(b), 4.6, cols[q], { sw: 1.3, tip: `${g.label}, x${i + 1} ≈ ${fmt(a)}: mean ${fmt(b)} (${n} runs)` }); });
      if (pts.length) s += T(left + pw + 8, ax.sy(pts[pts.length - 1][1]) + 3.5, g.label, { size: 9.5, fill: INK });
    });
    return svg(W, H, s);
  };

  /* ------------------------------------------------------------ every response */
  SA.overview = function (ov, cmapSigned) {
    const rs = (ov.responses || []).filter((r) => !r.error);
    if (!rs.length || !ov.inputs) return null;
    const d = ov.inputs.length, left = 220, top = 70, cw = Math.max(46, Math.min(80, 460 / d)), ch = 22, W = left + d * cw + 80, H = top + rs.length * ch + 60;
    let s = "";
    ov.inputs.forEach((it, i) => {
      const x = left + i * cw + cw / 2;
      s += C(x, top - 48, 4.2, SA.inputColor(i));
      s += T(x, top - 10, short(it.name, Math.round(cw / 5.2)), { anchor: "middle", size: 9.5, fill: INK, rot: d > 6 ? -28 : 0 });
    });
    rs.forEach((r, k) => {
      const y = top + k * ch;
      s += T(left - 8, y + ch / 2 + 3.5, short(r.label + (r.unit && r.unit !== "-" ? ` [${r.unit}]` : ""), 36), { anchor: "end", size: 10, fill: r.realised_metric ? MUTED : INK });
      for (let i = 0; i < d; i++) {
        const st = r.sobol_st ? +r.sobol_st[i] : null, rho = +r.spearman[i];
        const v = st !== null ? st : Math.abs(rho), f = SA.seq(Math.max(0, Math.min(1, v)));
        s += R(left + i * cw + 0.6, y + 0.6, cw - 1.2, ch - 1.2, f, { tip: `${r.label} · ${ov.inputs[i].name}\nSobol S_T ${fmt(st)} · S₁ ${fmt(r.sobol_s1 ? r.sobol_s1[i] : null)}\nSpearman ρ ${fmt(rho)} (${r.n} runs)` });
        const arrow = rho > 0.2 ? "▲" : rho < -0.2 ? "▼" : "";
        s += T(left + i * cw + cw / 2, y + ch / 2 + 3.5, `${v.toFixed(2)}${arrow ? " " + arrow : ""}`, { anchor: "middle", size: 8.5, fill: inkOn(f) });
      }
      s += T(left + d * cw + 10, y + ch / 2 + 3.5, r.q2 !== undefined && r.q2 !== null ? `Q² ${r.q2.toFixed(2)}` : "—", { size: 9.5, fill: r.q2 >= 0.5 ? INK : MUTED });
    });
    s += T(left + d * cw + 10, top - 10, "GP", { size: 9, fill: MUTED });
    s += colorbar(left, top + rs.length * ch + 22, Math.min(220, d * cw), 8, 0, 1, "Sobol S_T of a Gaussian process per response (▲ ▼ sign of Spearman ρ)");
    return svg(W, H, s);
  };

  /* ------------------------------------------------------------ export */
  SA.exportSvg = function (svgEl, name) {
    const s = new XMLSerializer().serializeToString(svgEl).replace(/ data-tip="[^"]*"/g, "");
    const b = new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${s}`], { type: "image/svg+xml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = name + ".svg"; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  };
  SA.exportPng = function (svgEl, name, scale = 4) {
    const vb = svgEl.viewBox.baseVal, w = vb.width, h = vb.height;
    const s = new XMLSerializer().serializeToString(svgEl).replace(/ data-tip="[^"]*"/g, "").replace(/width="100%"/, `width="${w}" height="${h}"`);
    const img = new Image();
    img.onload = () => {
      const cv = document.createElement("canvas");
      cv.width = Math.round(w * scale); cv.height = Math.round(h * scale);
      const g = cv.getContext("2d");
      g.fillStyle = "#fff"; g.fillRect(0, 0, cv.width, cv.height);
      g.drawImage(img, 0, 0, cv.width, cv.height);
      cv.toBlob((b) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(b); a.download = name + ".png"; document.body.appendChild(a); a.click();
        setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
      }, "image/png");
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(s);
  };
  // one tooltip for every figure: any element with data-tip
  SA.bindTips = function (root) {
    let tip = document.getElementById("saTip");
    if (!tip) { tip = document.createElement("div"); tip.id = "saTip"; tip.className = "satip"; tip.hidden = true; document.body.appendChild(tip); }
    root.onmousemove = (e) => {
      const t = e.target.closest ? e.target.closest("[data-tip]") : null;
      if (!t || !root.contains(t)) { tip.hidden = true; return; }
      tip.textContent = t.getAttribute("data-tip");
      tip.hidden = false;
      const x = e.clientX + 14, y = e.clientY + 14, bw = tip.offsetWidth, bh = tip.offsetHeight;
      tip.style.left = Math.min(x, window.innerWidth - bw - 8) + "px";
      tip.style.top = Math.min(y, window.innerHeight - bh - 8) + "px";
    };
    root.onmouseleave = () => { tip.hidden = true; };
  };
})();
