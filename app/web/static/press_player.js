/* The compression in 3D (Result Viewer): the periodic box at every step of
   the compression, drawn in an isometric view as its three outer faces (top
   z, front y = 0, side x = end) or as its three middle sections, coloured by
   material or by the von Mises stress (view/press_lab.u8, press_vm.f32;
   meta.press from visual.add_press_anim).
   PressPlayer(host, jobId, meta) -> {stop()} */
(function () {
  "use strict";
  async function bin(job, name, Type) {
    const r = await fetch(`/api/jobs/${job}/view/${name}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`${name}: ${r.status}`);
    return new Type(await r.arrayBuffer());
  }
  const hex = (h) => { const v = parseInt((h || "#999999").slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
  // jet, as the colour bars of the program
  function jet(t) {
    t = Math.min(1, Math.max(0, t));
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    return [Math.round(255 * r), Math.round(255 * g), Math.round(255 * b)];
  }
  const R2 = Math.SQRT2, R6 = Math.sqrt(6);

  window.PressPlayer = async function (host, job, meta) {
    const F = meta.frames.length;
    const [lab, vm] = await Promise.all([bin(job, meta.lab, Uint8Array), bin(job, meta.vm, Float32Array)]);
    const col = {};
    Object.entries(meta.colors || {}).forEach(([k, v]) => { col[+k] = hex(v); });
    host.innerHTML = `<div class="row wrap" style="gap:8px;margin:4px 0 6px;align-items:center">
        <button class="btn sm" data-p="play">Pause</button>
        <input type="range" data-p="frame" min="0" max="${F - 1}" value="0" style="flex:1;min-width:160px">
        <select data-p="view" class="sm"><option value="box">3D box</option><option value="sec">Middle sections</option></select>
        <select data-p="what" class="sm"><option value="mat">Materials</option><option value="vm">von Mises stress</option></select></div>
      <div class="hint mono" data-p="read" style="margin-bottom:4px"></div>
      <canvas data-p="cv" style="width:100%;border:1px solid var(--line, #d0d7de);border-radius:6px;background:#fff"></canvas>
      <div class="row wrap" data-p="leg" style="gap:12px;margin-top:6px"></div>`;
    const q = (k) => host.querySelector(`[data-p="${k}"]`);
    const cv = q("cv"), read = q("read"), slider = q("frame"), playB = q("play"), viewS = q("view"), whatS = q("what"), leg = q("leg");
    const ctx = cv.getContext("2d");
    const st = { k: 0, playing: true, id: null, last: 0 };
    const cache = new Map();
    // one face or section as an image (columns along its first axis)
    function image(k, part, what) {
      const key = `${k}:${part}:${what}`;
      if (cache.has(key)) return cache.get(key);
      const [o, n0, n1] = meta.frames[k].parts[part];
      const c = document.createElement("canvas");
      c.width = n0; c.height = n1;
      const g = c.getContext("2d"), im = g.createImageData(n0, n1);
      for (let a = 0; a < n0; a++) {
        for (let b = 0; b < n1; b++) {
          const v = o + a * n1 + b;
          // von Mises: the resins only; a sphere or a pore in its own colour, faded
          const own = col[lab[v]] || [153, 153, 153];
          const rgb = what !== "vm" ? own : Number.isFinite(vm[v]) ? jet(vm[v] / (meta.vm_max || 1))
            : own.map((c) => Math.round(0.35 * c + 0.65 * 235));
          const p = 4 * (b * n0 + a);
          im.data[p] = rgb[0]; im.data[p + 1] = rgb[1]; im.data[p + 2] = rgb[2]; im.data[p + 3] = 255;
        }
      }
      g.putImageData(im, 0, 0);
      cache.set(key, c);
      return c;
    }
    // isometric view from (+x, -y, +z): u = (x + y) / sqrt2, v = (-x + y + 2 z) / sqrt6
    const proj = (x, y, z) => [(x + y) / R2, -(-x + y + 2 * z) / R6];
    let bounds = null;
    meta.frames.forEach((f) => {
      const [Lx, Ly, Lz] = f.L_um;
      [[0, 0, 0], [Lx, 0, 0], [0, Ly, 0], [0, 0, Lz], [Lx, Ly, 0], [Lx, 0, Lz], [0, Ly, Lz], [Lx, Ly, Lz]].forEach(([x, y, z]) => {
        const [u, v] = proj(x, y, z);
        bounds = bounds ? [Math.min(bounds[0], u), Math.min(bounds[1], v), Math.max(bounds[2], u), Math.max(bounds[3], v)] : [u, v, u, v];
      });
    });
    function size() {
      const w = Math.max(320, host.clientWidth || 640);
      const sec = viewS.value === "sec";
      const h = sec ? Math.min(420, 0.42 * w) : Math.min(520, w * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]) + 30);
      cv.width = Math.round(w * devicePixelRatio);
      cv.height = Math.round(h * devicePixelRatio);
      cv.style.height = `${h}px`;
    }
    function legend() {
      if (whatS.value === "vm") {
        const stops = Array.from({ length: 11 }, (_, i) => `rgb(${jet(i / 10).join(",")}) ${i * 10}%`).join(",");
        leg.innerHTML = `<span class="hint">0</span><span style="display:inline-block;width:180px;height:11px;border-radius:2px;background:linear-gradient(90deg,${stops})"></span><span class="hint">${fmtN(meta.vm_max)} Pa (von Mises in the resins, the 99th percentile of all steps; spheres and pores faded)</span>`;
      } else {
        leg.innerHTML = meta.legend.map((l) => `<span class="hint"><i style="display:inline-block;width:11px;height:11px;border-radius:${l.kind === "particle" ? "50%" : "2px"};background:${l.color};vertical-align:-1px;margin-right:4px"></i>${l.name}</span>`).join("");
      }
    }
    const fmtN = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3));
    function draw(k) {
      const dpr = devicePixelRatio, W = cv.width / dpr, H = cv.height / dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.imageSmoothingEnabled = false;
      const f = meta.frames[k], [Lx, Ly, Lz] = f.L_um, [nx, ny, nz] = f.shape, what = whatS.value;
      const hx = Lx / nx, hy = Ly / ny, hz = Lz / nz;
      if (viewS.value === "box") {
        const S = Math.min((W - 20) / (bounds[2] - bounds[0]), (H - 20) / (bounds[3] - bounds[1]));
        const ox = 10 - bounds[0] * S + (W - 20 - (bounds[2] - bounds[0]) * S) / 2, oy = 10 - bounds[1] * S;
        const face = (part, a, b, c, d, x0, y0, z0) => {
          const [e, g] = proj(x0, y0, z0);
          ctx.setTransform(dpr * a * S, dpr * b * S, dpr * c * S, dpr * d * S, dpr * (ox + e * S), dpr * (oy + g * S));
          ctx.drawImage(image(k, part, what), 0, 0);
        };
        // top z = Lz: columns along x, rows along y
        face("top", hx / R2, hx / R6, hy / R2, -hy / R6, 0, 0, Lz);
        // front y = 0: columns along x, rows along z
        face("front", hx / R2, hx / R6, 0, -2 * hz / R6, 0, 0, 0);
        // side x = Lx: columns along y, rows along z
        face("side", hy / R2, -hy / R6, 0, -2 * hz / R6, Lx, 0, 0);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // the edges of the box
        ctx.strokeStyle = "rgba(36,41,47,0.85)"; ctx.lineWidth = 1;
        const P = (x, y, z) => { const [u, v] = proj(x, y, z); return [ox + u * S, oy + v * S]; };
        const edges = [[[0, 0, 0], [Lx, 0, 0]], [[Lx, 0, 0], [Lx, Ly, 0]], [[0, 0, 0], [0, 0, Lz]], [[Lx, 0, 0], [Lx, 0, Lz]], [[Lx, Ly, 0], [Lx, Ly, Lz]],
          [[0, 0, Lz], [Lx, 0, Lz]], [[Lx, 0, Lz], [Lx, Ly, Lz]], [[0, 0, Lz], [0, Ly, Lz]], [[0, Ly, Lz], [Lx, Ly, Lz]]];
        ctx.beginPath();
        edges.forEach(([a, b]) => { const p = P(...a), r = P(...b); ctx.moveTo(p[0], p[1]); ctx.lineTo(r[0], r[1]); });
        ctx.stroke();
        // the axes
        ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif";
        const ax = P(Lx * 0.5, -0.06 * Ly, 0), ay = P(Lx * 1.04, Ly * 0.5, 0), az = P(-0.04 * Lx, 0, Lz * 0.5);
        ctx.fillText("x", ax[0], ax[1] + 12); ctx.fillText("y", ay[0] + 4, ay[1] + 10); ctx.fillText("z", az[0] - 12, az[1]);
      } else {
        // the three middle sections side by side, at the same scale
        const parts = [["mxz", Lx, Lz, "x–z (middle of y)"], ["myz", Ly, Lz, "y–z (middle of x)"], ["mxy", Lx, Ly, "x–y (middle of z)"]];
        const gap = 16, wsum = parts.reduce((a, p) => a + p[1], 0);
        const maxH = Math.max(...meta.frames.map((g) => Math.max(g.L_um[2], g.L_um[1])));
        const maxW = Math.max(...meta.frames.map((g) => g.L_um[0] + g.L_um[1] + g.L_um[0]));
        const S = Math.min((W - 2 * gap - 20) / Math.max(maxW, wsum), (H - 34) / maxH);
        let x0 = 10;
        parts.forEach(([part, a, b, name]) => {
          const [, n0, n1] = f.parts[part];
          // columns along the first axis, rows along the second drawn upwards
          ctx.setTransform(dpr * (a / n0) * S, 0, 0, -dpr * (b / n1) * S, dpr * x0, dpr * (H - 22));
          ctx.drawImage(image(k, part, what), 0, 0);
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.strokeStyle = "rgba(36,41,47,0.85)"; ctx.strokeRect(x0, H - 22 - b * S, a * S, b * S);
          ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif"; ctx.fillText(name, x0, H - 6);
          x0 += a * S + gap;
        });
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const comp = meta.compression[k] || 0, sz = meta.sigma_zz_Pa[k] || 0;
      read.textContent = `step ${k} of ${F - 1} · compression ${(100 * comp).toFixed(1)} % · box ${f.L_um.map((v) => v.toFixed(1)).join(" × ")} µm · axial stress ${fmtN(-sz)} Pa`;
      slider.value = String(k);
    }
    function tick(t) {
      if (st.playing && t - st.last > 450) {
        st.last = t;
        st.k = (st.k + 1) % F;
        draw(st.k);
      }
      st.id = requestAnimationFrame(tick);
    }
    playB.addEventListener("click", () => { st.playing = !st.playing; playB.textContent = st.playing ? "Pause" : "Play"; });
    slider.addEventListener("input", () => { st.playing = false; playB.textContent = "Play"; st.k = +slider.value; draw(st.k); });
    viewS.addEventListener("change", () => { size(); draw(st.k); });
    whatS.addEventListener("change", () => { legend(); draw(st.k); });
    const ro = new ResizeObserver(() => { size(); draw(st.k); });
    ro.observe(host);
    size(); legend(); draw(0);
    st.id = requestAnimationFrame(tick);
    return { stop() { cancelAnimationFrame(st.id); ro.disconnect(); } };
  };
})();
