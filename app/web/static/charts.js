/* Canvas and SVG charts without a library (the page must render on an
   isolated network): line, histogram, horizontal bars, polar and the
   bounds-and-models chart. */
(function () {
  "use strict";
  const C = (window.MPSCharts = {});
  const FONT = '11px "Segoe UI", system-ui, sans-serif';
  const INK = "#1f2328", MUTED = "#6e7781", GRID = "#eef1f4", AXIS = "#d0d7de";
  C.PALETTE = ["#0b62c4", "#e36209", "#1a7f37", "#8250df", "#cf222e", "#0a8fa8", "#9a6700", "#bf3989"];

  function nice(v) {
    const e = Math.floor(Math.log10(Math.abs(v) || 1));
    const f = v / 10 ** e;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * 10 ** e;
  }
  C.fmt = function (v) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2).replace("e+", "e");
    return String(Number(v.toPrecision(4)));
  };
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const decade = (e) => (Math.abs(e - Math.round(e)) > 1e-9 ? C.fmt(10 ** e)
    : (e >= -2 && e <= 3 ? String(10 ** Math.round(e)) : "10" + String(Math.round(e)).split("").map((c) => SUP[c]).join("")));

  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const h = canvas.clientHeight || +canvas.getAttribute("height") || 280;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.font = FONT;
    return { g, w, h };
  }
  function noData(g, w, h) { g.fillStyle = MUTED; g.textAlign = "center"; g.fillText("No data", w / 2, h / 2); }

  function axes(g, w, h, box, o, xr, yr) {
    const { L, R, T, B } = box;
    const ticks = (a, b, log) => {
      const out = [];
      if (log) {
        for (let e = Math.ceil(a - 1e-9); e <= Math.floor(b + 1e-9); e++) out.push(e);
        // a range inside one or two decades also gets 2 and 5 (and 1-9 when narrower still)
        if (out.length < 3) {
          const steps = b - a < 0.6 ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : [1, 2, 5];
          const more = [];
          for (let e = Math.floor(a) - 1; e <= Math.ceil(b); e++) steps.forEach((m) => {
            const v = e + Math.log10(m);
            if (v >= a - 1e-9 && v <= b + 1e-9) more.push(v);
          });
          return more.length >= 2 ? more : out;
        }
        return out;
      }
      const st = nice((b - a) / 5 || 1);
      for (let v = Math.ceil(a / st - 1e-9) * st; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
      return out;
    };
    g.lineWidth = 1;
    ticks(xr[0], xr[1], o.xlog).forEach((t) => {
      const px = L + ((t - xr[0]) / (xr[1] - xr[0])) * (w - L - R);
      g.strokeStyle = GRID; g.beginPath(); g.moveTo(px, T); g.lineTo(px, h - B); g.stroke();
      g.fillStyle = MUTED; g.textAlign = "center"; g.fillText(o.xlog ? decade(t) : C.fmt(t), px, h - B + 14);
    });
    ticks(yr[0], yr[1], o.ylog).forEach((t) => {
      const py = h - B - ((t - yr[0]) / (yr[1] - yr[0])) * (h - T - B);
      g.strokeStyle = GRID; g.beginPath(); g.moveTo(L, py); g.lineTo(w - R, py); g.stroke();
      g.fillStyle = MUTED; g.textAlign = "right"; g.fillText(o.ylog ? decade(t) : C.fmt(t), L - 6, py + 4);
    });
    g.strokeStyle = AXIS; g.strokeRect(L + 0.5, T + 0.5, w - L - R - 1, h - T - B - 1);
    g.fillStyle = INK; g.textAlign = "center";
    if (o.xlabel) g.fillText(o.xlabel, L + (w - L - R) / 2, h - 6);
    if (o.ylabel) { g.save(); g.translate(13, T + (h - T - B) / 2); g.rotate(-Math.PI / 2); g.fillText(o.ylabel, 0, 0); g.restore(); }
  }

  function legend(g, series, L, y) {
    let lx = L + 6;
    series.forEach((s) => {
      if (!s.label) return;
      g.strokeStyle = s.color || C.PALETTE[0]; g.lineWidth = 2.5; g.setLineDash(s.dash || []);
      if (s.marker && s.width === 0) {
        // points only: a point in the legend too
        g.fillStyle = "#fff"; g.lineWidth = 2;
        g.beginPath(); g.arc(lx + 8, y - 3, 4, 0, 2 * Math.PI); g.fill(); g.stroke();
      } else { g.beginPath(); g.moveTo(lx, y - 3); g.lineTo(lx + 16, y - 3); g.stroke(); }
      g.setLineDash([]);
      g.fillStyle = INK; g.textAlign = "left"; g.fillText(s.label, lx + 21, y + 1);
      lx += g.measureText(s.label).width + 40;
    });
  }

  /* line chart: {series:[{x,y,color,width,dash,label,marker}], xlog, ylog, xlabel, ylabel, xmin, xmax, ymin, ymax} */
  C.line = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const box = { L: 62, R: 16, T: 26, B: 38 };
    const xs = [], ys = [];
    o.series.forEach((s) => s.x.forEach((x, i) => {
      const y = s.y[i];
      if (x === null || y === null || !isFinite(x) || !isFinite(y) || (o.xlog && x <= 0) || (o.ylog && y <= 0)) return;
      xs.push(x); ys.push(y);
    }));
    if (!xs.length) { noData(g, w, h); return; }
    const tx = (v) => (o.xlog ? Math.log10(v) : v), ty = (v) => (o.ylog ? Math.log10(v) : v);
    let x0 = o.xmin !== undefined ? tx(o.xmin) : Math.min(...xs.map(tx)), x1 = o.xmax !== undefined ? tx(o.xmax) : Math.max(...xs.map(tx));
    let y0 = o.ymin !== undefined ? ty(o.ymin) : Math.min(...ys.map(ty)), y1 = o.ymax !== undefined ? ty(o.ymax) : Math.max(...ys.map(ty));
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) { y1 = y0 + 1; y0 -= o.ylog ? 1 : 0; }
    if (!o.ylog && o.ymin === undefined) { const p = 0.06 * (y1 - y0); y0 -= p; y1 += p; }
    // a log axis also keeps a margin, so a flat line is not drawn on the frame
    if (o.ylog && o.ymin === undefined && o.ymax === undefined) { const p = Math.max(0.06 * (y1 - y0), 0.05); y0 -= p; y1 += p; }
    if (!o.ylog && o.ymax === undefined && o.ymin !== undefined) y1 += 0.06 * (y1 - y0);
    axes(g, w, h, box, o, [x0, x1], [y0, y1]);
    const X = (v) => box.L + ((tx(v) - x0) / (x1 - x0)) * (w - box.L - box.R);
    const Y = (v) => h - box.B - ((ty(v) - y0) / (y1 - y0)) * (h - box.T - box.B);
    g.save(); g.beginPath(); g.rect(box.L, box.T, w - box.L - box.R, h - box.T - box.B); g.clip();
    o.series.forEach((s, k) => {
      g.strokeStyle = s.color || C.PALETTE[k % C.PALETTE.length]; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []);
      g.lineJoin = "round";
      g.beginPath(); let on = false;
      s.x.forEach((x, i) => {
        const y = s.y[i];
        if (x === null || y === null || !isFinite(x) || !isFinite(y) || (o.xlog && x <= 0) || (o.ylog && y <= 0)) { on = false; return; }
        const px = X(x), py = Y(y);
        if (!on) { g.moveTo(px, py); on = true; } else g.lineTo(px, py);
      });
      // width 0: the points alone, no line through them
      if (s.width !== 0) g.stroke();
      g.setLineDash([]);
      if (s.marker) {
        g.fillStyle = "#fff";
        if (s.width === 0) g.lineWidth = 2;
        s.x.forEach((x, i) => {
          const y = s.y[i];
          if (x === null || y === null || !isFinite(x) || !isFinite(y)) return;
          g.beginPath(); g.arc(X(x), Y(y), s.width === 0 ? 4 : 3, 0, 2 * Math.PI); g.fill(); g.stroke();
        });
      }
    });
    g.restore();
    legend(g, o.series, box.L, 14);
  };

  /* probability map (a joint posterior): {x:[nx], y:[ny], p:[ny][nx] in 0..1,
     xlog, ylog, xlabel, ylabel, marks:[{x, y, label, color}]}. One hue from
     the surface colour to dark blue: more probable is darker. */
  C.heat = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const box = { L: 66, R: 16, T: 26, B: 40 };
    const nx = (o.x || []).length, ny = (o.y || []).length;
    if (!nx || !ny) { noData(g, w, h); return; }
    const tx = (v) => (o.xlog ? Math.log10(v) : v), ty = (v) => (o.ylog ? Math.log10(v) : v);
    const x0 = tx(o.x[0]), x1 = tx(o.x[nx - 1]), y0 = ty(o.y[0]), y1 = ty(o.y[ny - 1]);
    const pw = w - box.L - box.R, ph = h - box.T - box.B;
    const lo = [246, 248, 250], hi = [8, 48, 120];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = o.p[j][i];
      if (!(v > 0.003)) continue;
      const t = Math.pow(Math.min(1, v), 0.6);
      g.fillStyle = `rgb(${lo.map((a, k) => Math.round(a + (hi[k] - a) * t)).join(",")})`;
      // cell edges halfway between grid points
      const xa = i === 0 ? x0 : (tx(o.x[i - 1]) + tx(o.x[i])) / 2, xb = i === nx - 1 ? x1 : (tx(o.x[i]) + tx(o.x[i + 1])) / 2;
      const ya = j === 0 ? y0 : (ty(o.y[j - 1]) + ty(o.y[j])) / 2, yb = j === ny - 1 ? y1 : (ty(o.y[j]) + ty(o.y[j + 1])) / 2;
      const px = box.L + ((xa - x0) / (x1 - x0)) * pw, qx = box.L + ((xb - x0) / (x1 - x0)) * pw;
      const py = h - box.B - ((yb - y0) / (y1 - y0)) * ph, qy = h - box.B - ((ya - y0) / (y1 - y0)) * ph;
      g.fillRect(px, py, qx - px + 0.6, qy - py + 0.6);
    }
    axes(g, w, h, box, o, [x0, x1], [y0, y1]);
    (o.marks || []).forEach((m, k) => {
      if (!isFinite(m.x) || !isFinite(m.y)) return;
      const X = box.L + ((tx(m.x) - x0) / (x1 - x0)) * pw, Y = h - box.B - ((ty(m.y) - y0) / (y1 - y0)) * ph;
      g.strokeStyle = "#fff"; g.lineWidth = 3; g.beginPath(); g.arc(X, Y, 5, 0, 2 * Math.PI); g.stroke();
      g.strokeStyle = m.color || C.PALETTE[(k + 1) % C.PALETTE.length]; g.lineWidth = 2; g.beginPath(); g.arc(X, Y, 5, 0, 2 * Math.PI); g.stroke();
      if (m.label) { g.fillStyle = INK; g.textAlign = "left"; g.fillText(m.label, X + 8, Y - 6); }
    });
  };

  /* histogram: {x (bin centres), y, color, xlabel, ylabel, line:{y, color, label}, labels} */
  C.hist = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const box = { L: 58, R: 16, T: 24, B: 38 };
    const tx = (v) => (o.xlog ? Math.log10(v) : v);
    const idx = (o.x || []).map((x, i) => i).filter((i) => o.x[i] !== null && isFinite(o.x[i]) && (!o.xlog || o.x[i] > 0))
      .sort((a, b) => tx(o.x[a]) - tx(o.x[b]));
    const n = idx.length;
    if (!n) { noData(g, w, h); return; }
    const t = idx.map((i) => tx(o.x[i]));
    // bar edges halfway between neighbouring centres, so uneven spacing
    // (logarithmic pressure steps, for example) still tiles the axis
    const step = n > 1 ? (t[n - 1] - t[0]) / (n - 1) : 1;
    const left = t.map((v, k) => (k ? 0.5 * (v + t[k - 1]) : v - 0.5 * (n > 1 ? t[1] - t[0] : step)));
    const right = t.map((v, k) => (k < n - 1 ? 0.5 * (v + t[k + 1]) : v + 0.5 * (n > 1 ? t[n - 1] - t[n - 2] : step)));
    const x0 = left[0], x1 = right[n - 1] > x0 ? right[n - 1] : x0 + 1;
    const vmax = Math.max(...idx.map((i) => o.y[i] || 0), 1e-300) * 1.08;
    axes(g, w, h, box, { xlabel: o.xlabel, ylabel: o.ylabel, xlog: o.xlog }, [x0, x1], [0, vmax]);
    const X = (v) => box.L + ((v - x0) / (x1 - x0)) * (w - box.L - box.R);
    const Y = (v) => h - box.B - (v / vmax) * (h - box.T - box.B);
    idx.forEach((i, k) => {
      const v = o.y[i] || 0;
      const a = X(left[k]), b = X(right[k]), gap = Math.min(3, 0.09 * (b - a));
      const grad = g.createLinearGradient(0, Y(v), 0, h - box.B);
      grad.addColorStop(0, o.color || C.PALETTE[0]); grad.addColorStop(1, (o.color2 || "#9cc3f0"));
      g.fillStyle = grad;
      g.fillRect(a + gap, Y(v), Math.max(1, b - a - 2 * gap), h - box.B - Y(v));
    });
    if (o.line) {
      const lm = Math.max(...o.line.y.map((v) => v || 0), 1e-300);
      g.strokeStyle = o.line.color || "#e36209"; g.lineWidth = 2; g.beginPath();
      idx.forEach((i, k) => { const py = h - box.B - ((o.line.y[i] || 0) / lm) * (h - box.T - box.B) * 0.92; k ? g.lineTo(X(t[k]), py) : g.moveTo(X(t[k]), py); });
      g.stroke();
      legend(g, [{ label: o.label || "Distribution", color: o.color || C.PALETTE[0] }, { label: o.line.label, color: o.line.color || "#e36209" }], box.L, 14);
    } else if (o.label) legend(g, [{ label: o.label, color: o.color || C.PALETTE[0] }], box.L, 14);
  };

  /* horizontal bars: {labels, values, colors, unit, max} */
  C.bars = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const n = o.labels.length, L = 170, R = 66, rowh = Math.min(24, (h - 10) / Math.max(n, 1));
    const vmax = o.max || Math.max(...o.values.map((v) => Math.abs(v) || 0), 1e-12);
    o.labels.forEach((lab, i) => {
      const y = 5 + i * rowh, v = o.values[i] || 0;
      g.fillStyle = INK; g.textAlign = "right"; g.fillText(lab.length > 28 ? lab.slice(0, 27) + "…" : lab, L - 8, y + rowh * 0.66);
      g.fillStyle = "#eef1f4"; g.fillRect(L, y + 4, w - L - R, rowh - 8);
      g.fillStyle = (o.colors && o.colors[i]) || C.PALETTE[0];
      g.fillRect(L, y + 4, (Math.abs(v) / vmax) * (w - L - R), rowh - 8);
      g.fillStyle = INK; g.textAlign = "left";
      g.fillText(C.fmt(v) + (o.unit || ""), L + (w - L - R) + 6, y + rowh * 0.66);
    });
  };

  /* polar plot: {series:[{theta_deg, r, color, label}], title} */
  C.polar = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const all = o.series.flatMap((s) => s.r.filter((v) => v !== null && isFinite(v)));
    if (!all.length) { noData(g, w, h); return; }
    const rmax = nice(Math.max(...all) * 1.05);
    const cx = w / 2, cy = h / 2 + 10, rad = Math.min(w, h - 30) / 2 - 22;
    g.strokeStyle = GRID; g.fillStyle = MUTED; g.textAlign = "left";
    for (let k = 1; k <= 4; k++) {
      g.beginPath(); g.arc(cx, cy, (rad * k) / 4, 0, 2 * Math.PI); g.stroke();
      g.fillText(C.fmt((rmax * k) / 4), cx + 3, cy - (rad * k) / 4 - 2);
    }
    for (let a = 0; a < 360; a += 30) {
      const t = (a * Math.PI) / 180;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + rad * Math.cos(t), cy - rad * Math.sin(t)); g.stroke();
      g.textAlign = "center"; g.fillText(a + "°", cx + (rad + 12) * Math.cos(t), cy - (rad + 12) * Math.sin(t) + 4);
    }
    o.series.forEach((s, k) => {
      g.strokeStyle = s.color || C.PALETTE[k]; g.lineWidth = 2.2; g.beginPath();
      s.theta_deg.forEach((d, i) => {
        const v = s.r[i]; if (v === null || !isFinite(v)) return;
        const t = (d * Math.PI) / 180, px = cx + (rad * v / rmax) * Math.cos(t), py = cy - (rad * v / rmax) * Math.sin(t);
        i ? g.lineTo(px, py) : g.moveTo(px, py);
      });
      g.closePath(); g.stroke();
      g.globalAlpha = 0.07; g.fillStyle = s.color || C.PALETTE[k]; g.fill(); g.globalAlpha = 1;
    });
    legend(g, o.series, 0, 14);
    if (o.unit) { g.fillStyle = MUTED; g.textAlign = "right"; g.fillText(o.unit, w - 6, h - 6); }
  };

  /* scatter, with an optional 1:1 line for a predicted-against-measured plot:
     {points:[{x,y}], xlabel, ylabel, unity, color, label} */
  C.scatter = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const box = { L: 66, R: 18, T: 26, B: 40 };
    const pts = (o.points || []).filter((p) => p && isFinite(p.x) && isFinite(p.y));
    if (!pts.length) { noData(g, w, h); return; }
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (o.unity) { const lo = Math.min(x0, y0), hi = Math.max(x1, y1); x0 = y0 = lo; x1 = y1 = hi; }
    const mx = (x1 - x0) * 0.08 || Math.abs(x0) * 0.05 || 1;
    const my = (y1 - y0) * 0.08 || Math.abs(y0) * 0.05 || 1;
    x0 -= mx; x1 += mx; y0 -= my; y1 += my;
    axes(g, w, h, box, { xlabel: o.xlabel, ylabel: o.ylabel }, [x0, x1], [y0, y1]);
    const X = (v) => box.L + ((v - x0) / (x1 - x0)) * (w - box.L - box.R);
    const Y = (v) => h - box.B - ((v - y0) / (y1 - y0)) * (h - box.T - box.B);
    g.save(); g.beginPath(); g.rect(box.L, box.T, w - box.L - box.R, h - box.T - box.B); g.clip();
    if (o.unity) {
      g.strokeStyle = "#afb8c1"; g.lineWidth = 1.2; g.setLineDash([6, 4]);
      g.beginPath(); g.moveTo(X(x0), Y(x0)); g.lineTo(X(x1), Y(x1)); g.stroke(); g.setLineDash([]);
    }
    g.fillStyle = o.color || C.PALETTE[0]; g.strokeStyle = "#fff"; g.lineWidth = 1.4;
    pts.forEach((p) => { g.beginPath(); g.arc(X(p.x), Y(p.y), 4.5, 0, 2 * Math.PI); g.fill(); g.stroke(); });
    g.restore();
    if (o.label) legend(g, [{ label: o.label, color: o.color || C.PALETTE[0] }], box.L, 14);
  };

  /* bubble scatter: any quantity on x, y, colour and size, the way a design
     exploration tool lets a study be interrogated. Low values run light blue,
     the median grey, high values dark red, which reads the same way in print.
     {points:[{x,y,c,s,label}], xlabel, ylabel, clabel, slabel} */
  C.ramp = function (t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [[0.04, 0.42, 0.72], [0.62, 0.66, 0.70], [0.70, 0.11, 0.11]];
    const k = t < 0.5 ? 0 : 1, f = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
    const a = stops[k], b = stops[k + 1];
    return `rgb(${a.map((v, i) => Math.round(255 * (v + (b[i] - v) * f))).join(",")})`;
  };
  C.bubble = function (canvas, o) {
    const { g, w, h } = setup(canvas);
    const hasC = (o.points || []).some((p) => isFinite(p.c));
    const box = { L: 66, R: hasC ? 74 : 18, T: 26, B: 40 };
    const pts = (o.points || []).filter((p) => p && isFinite(p.x) && isFinite(p.y));
    if (!pts.length) { noData(g, w, h); return; }
    const ext = (f) => { const v = pts.map(f).filter(isFinite); return [Math.min(...v), Math.max(...v)]; };
    let [x0, x1] = ext((p) => p.x), [y0, y1] = ext((p) => p.y);
    const mx = (x1 - x0) * 0.08 || Math.abs(x0) * 0.05 || 1;
    const my = (y1 - y0) * 0.08 || Math.abs(y0) * 0.05 || 1;
    x0 -= mx; x1 += mx; y0 -= my; y1 += my;
    axes(g, w, h, box, { xlabel: o.xlabel, ylabel: o.ylabel }, [x0, x1], [y0, y1]);
    const X = (v) => box.L + ((v - x0) / (x1 - x0)) * (w - box.L - box.R);
    const Y = (v) => h - box.B - ((v - y0) / (y1 - y0)) * (h - box.T - box.B);
    const [c0, c1] = hasC ? ext((p) => p.c) : [0, 1];
    const hasS = pts.some((p) => isFinite(p.s));
    const [s0, s1] = hasS ? ext((p) => p.s) : [0, 1];
    g.save(); g.beginPath(); g.rect(box.L, box.T, w - box.L - box.R, h - box.T - box.B); g.clip();
    pts.forEach((p) => {
      const t = hasC && c1 > c0 ? (p.c - c0) / (c1 - c0) : 0.5;
      const u = hasS && s1 > s0 ? (p.s - s0) / (s1 - s0) : 0.5;
      g.fillStyle = hasC ? C.ramp(t) : (o.color || C.PALETTE[0]);
      g.strokeStyle = "rgba(255,255,255,.9)"; g.lineWidth = 1.4;
      g.beginPath(); g.arc(X(p.x), Y(p.y), hasS ? 3.5 + 8 * u : 5, 0, 2 * Math.PI);
      g.fill(); g.stroke();
    });
    g.restore();
    if (hasC) {
      const bx = w - box.R + 16, by = box.T + 6, bh = h - box.T - box.B - 26;
      for (let i = 0; i < bh; i++) { g.fillStyle = C.ramp(1 - i / bh); g.fillRect(bx, by + i, 13, 1); }
      g.strokeStyle = "#d0d7de"; g.lineWidth = 1; g.strokeRect(bx, by, 13, bh);
      g.fillStyle = MUTED; g.font = "10px Consolas, monospace"; g.textAlign = "left";
      g.fillText(C.fmt(c1), bx + 17, by + 8);
      g.fillText(C.fmt(c0), bx + 17, by + bh);
      if (o.clabel) { g.save(); g.translate(w - 6, by + bh / 2); g.rotate(-Math.PI / 2); g.textAlign = "center"; g.fillText(o.clabel, 0, 0); g.restore(); }
    }
  };

  /* bounds and models (SVG): bands [{label,lo,hi,color}], points [{label,value,color,shape,bold}] */
  C.refs = function (o) {
    const W = 760, rowh = 26, rows = o.bands.length + o.points.length, H = 46 + rows * rowh;
    const vals = [];
    o.bands.forEach((b) => vals.push(b.lo, b.hi));
    o.points.forEach((p) => vals.push(p.value));
    const fin = vals.filter((v) => v !== null && isFinite(v) && (!o.log || v > 0));
    if (!fin.length) return "";
    const a = Math.min(...fin), b = Math.max(...fin);
    const log = o.log && a > 0 && b / a > 30;
    const t = (v) => (log ? Math.log10(v) : v);
    let ta = t(a), tb = t(b);
    const pad = (tb - ta) * 0.08 || Math.abs(ta) * 0.05 || 1; ta -= pad; tb += pad;
    const L = 220, R = 96;
    const X = (v) => L + ((t(v) - ta) / (tb - ta)) * (W - L - R);
    let s = `<svg class="refsvg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, system-ui, sans-serif" font-size="12">`;
    let y = 22;
    o.bands.forEach((bd) => {
      if (bd.lo === null || bd.hi === null) return;
      s += `<text x="${L - 10}" y="${y + 4}" text-anchor="end" fill="#424a53">${bd.label}</text>`;
      s += `<rect x="${X(bd.lo)}" y="${y - 7}" width="${Math.max(2, X(bd.hi) - X(bd.lo))}" height="14" rx="4" fill="${bd.color}" opacity=".6"/>`;
      s += `<text x="${Math.min(X(bd.hi) + 6, W - R + 4)}" y="${y + 4}" fill="#6e7781" font-size="10.5">${C.fmt(bd.lo)} – ${C.fmt(bd.hi)}</text>`;
      y += rowh;
    });
    o.points.forEach((p) => {
      if (p.value === null || !isFinite(p.value)) return;
      const px = X(p.value);
      s += `<text x="${L - 10}" y="${y + 4}" text-anchor="end" fill="${p.bold ? "#0d1117" : "#424a53"}" font-weight="${p.bold ? 700 : 400}">${p.label}</text>`;
      s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#eef1f4"/>`;
      if (p.shape === "diamond") s += `<path d="M${px} ${y - 7} L${px + 7} ${y} L${px} ${y + 7} L${px - 7} ${y} Z" fill="${p.color}"/>`;
      else s += `<circle cx="${px}" cy="${y}" r="6.5" fill="${p.color}" stroke="#fff" stroke-width="1.5"/>`;
      s += `<text x="${px + 10}" y="${y + 4}" fill="#0d1117" font-size="11">${C.fmt(p.value)}</text>`;
      y += rowh;
    });
    s += `<text x="${W - 8}" y="${H - 6}" text-anchor="end" fill="#6e7781" font-size="11">${log ? "logarithmic scale · " : ""}${o.unit || ""}</text></svg>`;
    return s;
  };
})();
