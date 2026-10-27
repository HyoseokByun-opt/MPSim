/* The compression step by step in 3D: the whole periodic box of every step
   the solver took (view/press_steps, meta.press.steps3d from
   visual.add_press_anim), every voxel drawn crisp as in the 3D view, the
   box at its own size in an iso perspective that stays where the first step
   put it - the box is seen closing. Shown: the whole box, the particles and
   pores alone, or the box cut at its middle (y); coloured by material or by
   the von Mises stress of the resins. Play, step, scrub, GIF.
   Press3D(host, jobId, meta, {fill, name}) -> {stop()} or null (no steps). */
(function () {
  "use strict";
  const ns = (path) => path.split(".").reduce((o, k) => (o ? o[k] : undefined), window.vtk);
  const hex = (h) => { const v = parseInt((h || "#999999").slice(1), 16); return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]; };
  const jet = (t) => { t = Math.min(1, Math.max(0, t)); const c = (x) => Math.min(1, Math.max(0, x)); return [c(1.5 - Math.abs(4 * t - 3)), c(1.5 - Math.abs(4 * t - 2)), c(1.5 - Math.abs(4 * t - 1))]; };
  const fmt = (x) => (x === 0 ? "0" : Math.abs(x) >= 1e4 || Math.abs(x) < 0.01 ? x.toExponential(2) : x.toPrecision(3));
  const ABSENT = 255;                      // the label of a voxel not drawn: the pad, the cut-away half
  const DELAY = { "0.5": 600, "1": 300, "2": 150, "4": 75 };

  window.Press3D = async function (host, jobId, meta, opts) {
    opts = opts || {};
    const P = meta && meta.press, steps = P && P.steps3d;
    const MV = window.MPSViewer || {};
    if (!window.vtk || !host || !steps || !steps.length || !MV.crispMix) return null;
    const fill = !!opts.fill, F = steps.length, built = meta.built || 0;
    // every step on the solver's own voxels: nx x ny x nz of the voxel size
    // the analyses use (the box the step's grid holds, which the solves of
    // the last step ran on; the press's continuous box differs from it by
    // less than a voxel)
    const h0 = meta.spacing_um || Math.min(...steps[0].L_um.map((v, a) => v / steps[0].dims[a]));
    const boxOf = (s) => s.dims.map((n) => n * h0);
    // the camera holds every step: the box shortens in z and widens across
    const U = [0, 1, 2].map((a) => Math.max(...steps.map((s) => s.dims[a] * h0)));
    const url = (f) => `/api/jobs/${jobId}/view/${f}?v=${built}`;
    const lab = {};
    (meta.labels || []).forEach((l) => { lab[l.value] = l; });
    const colours = P.colors || {};
    host.innerHTML = `<div style="position:relative;${fill ? "flex:1;min-height:0;display:flex;flex-direction:column" : ""}">
        <div class="p3view" style="position:relative;${fill ? "flex:1;min-height:0" : "height:420px"};border:1px solid var(--line,#d0d7de);border-radius:6px;overflow:hidden;background:#fff"></div>
        <div data-p3="legend" style="position:absolute;right:10px;top:10px;background:rgba(255,255,255,.9);border:1px solid #d0d7de;border-radius:4px;padding:6px 8px;font:11px system-ui,sans-serif;color:#1f2328;pointer-events:none;max-width:46%"></div>
        <div data-p3="read" class="mono" style="position:absolute;left:10px;top:10px;background:rgba(255,255,255,.9);border:1px solid #d0d7de;border-radius:4px;padding:4px 8px;font-size:11px;color:#1f2328;pointer-events:none;max-width:52%"></div>
        <div data-p3="wait" class="hint" style="position:absolute;left:10px;bottom:8px;pointer-events:none"></div></div>
      <div class="row gap" style="align-items:center;margin-top:6px;flex-wrap:wrap">
        <button class="btn sm" data-p3="prev" title="The step before">&#9664;</button>
        <button class="btn sm" data-p3="play" style="min-width:58px">Play</button>
        <button class="btn sm" data-p3="next" title="The next step">&#9654;</button>
        <input type="range" data-p3="step" min="0" max="${F - 1}" value="${F - 1}" style="flex:1;min-width:140px" title="Step of the compression">
        <select data-p3="speed" class="sm" style="width:auto" title="Playing speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
      </div>
      <div class="row gap" style="align-items:center;margin-top:4px;flex-wrap:wrap">
        <select data-p3="show" class="sm" style="width:auto" title="What of the box is drawn"><option value="box">Whole box</option><option value="cut">Box cut at the middle (y)</option><option value="particles">Particles and pores only</option></select>
        <select data-p3="colour" class="sm" style="width:auto"><option value="mat">Colour: materials</option><option value="vm">Colour: von Mises stress in the resins</option></select>
        <select data-p3="scale" class="sm" style="width:auto" title="The colour range of the stress"><option value="step">Range: each step its own</option><option value="all">Range: the largest of all steps</option></select>
        <button class="btn sm" data-p3="iso" title="Back to the iso view">Iso view</button>
        <button class="btn sm" data-p3="gif" title="Every step as an animated GIF (the view as it is turned now)">Save GIF</button>
      </div>`;
    const q = (k) => host.querySelector(`[data-p3="${k}"]`);
    const view = host.querySelector(".p3view");
    const GRW = ns("Rendering.Misc.vtkGenericRenderWindow"), PD = ns("Common.DataModel.vtkPolyData");
    const ImageData = ns("Common.DataModel.vtkImageData"), DataArray = ns("Common.Core.vtkDataArray");
    const Volume = ns("Rendering.Core.vtkVolume"), VolumeMapper = ns("Rendering.Core.vtkVolumeMapper");
    const PWF = ns("Common.DataModel.vtkPiecewiseFunction"), CTF = ns("Rendering.Core.vtkColorTransferFunction");
    const Mapper = ns("Rendering.Core.vtkMapper"), Actor = ns("Rendering.Core.vtkActor");
    const grw = GRW.newInstance({ background: [1, 1, 1] });
    grw.setContainer(view);
    grw.resize();
    const ren = grw.getRenderer(), rw = grw.getRenderWindow();
    // the lights of the 3D view: a key light above and to the left, a weak
    // fill from the other side, both moving with the camera, so that every
    // face orientation of a voxel reads differently
    if (MV.cameraLights) MV.cameraLights(ren);
    const nmax = Math.max(...steps.map((s) => Math.max(...s.dims)));
    const mapper = VolumeMapper.newInstance();
    // crisp, as the Voxels style of the 3D view: the full screen resolution
    // also while the view moves, a fine step along the ray
    if (mapper.setAutoAdjustSampleDistances) mapper.setAutoAdjustSampleDistances(false);
    if (mapper.setImageSampleDistance) mapper.setImageSampleDistance(1.0);
    if (mapper.setInteractionSampleDistanceFactor) mapper.setInteractionSampleDistanceFactor(1.0);
    if (mapper.setMaximumSamplesPerRay) mapper.setMaximumSamplesPerRay(8 * (nmax + 2) + 64);
    const actor = Volume.newInstance();
    actor.setMapper(mapper);
    ren.addVolume(actor);
    const pr = actor.getProperty();
    pr.setIndependentComponents(true);
    pr.setColorMixPreset(3);
    pr.setInterpolationTypeToNearest();
    pr.setShade(true);
    pr.setDiffuse(0.7); pr.setSpecular(0.15); pr.setSpecularPower(20);
    // the box of the first step (faint) and of the step shown
    function outline(L, rgb, w) {
      const pts = [];
      for (const z of [0, L[2]]) for (const y of [0, L[1]]) for (const x of [0, L[0]]) pts.push(x, y, z);
      const ed = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
      const lines = new Uint32Array(ed.length * 3);
      ed.forEach(([a, b], k) => { lines[3 * k] = 2; lines[3 * k + 1] = a; lines[3 * k + 2] = b; });
      const pd = PD.newInstance();
      pd.getPoints().setData(new Float32Array(pts), 3);
      pd.getLines().setData(lines);
      const m = Mapper.newInstance();
      m.setInputData(pd);
      const a = Actor.newInstance();
      a.setMapper(m);
      a.getProperty().setColor(...rgb);
      a.getProperty().setLineWidth(w);
      ren.addActor(a);
      return { pd, set(L2) { const p = [];
        for (const z of [0, L2[2]]) for (const y of [0, L2[1]]) for (const x of [0, L2[0]]) p.push(x, y, z);
        pd.getPoints().setData(new Float32Array(p), 3); pd.modified(); } };
    }
    outline(boxOf(steps[0]), [0.75, 0.78, 0.81], 1);
    const box = outline(boxOf(steps[0]), [0.25, 0.28, 0.32], 1.5);

    // the steps' volumes, fetched as they are needed (gzip on the wire), the
    // most recent ones kept up to a budget
    const cache = new Map(), BUDGET = 320e6;
    let held = 0;
    async function load(k) {
      if (cache.has(k)) { const c = cache.get(k); cache.delete(k); cache.set(k, c); return c; }
      const s = steps[k];
      const [a, b] = await Promise.all([s.lab, s.vm].map(async (f) => {
        const r = await fetch(url(f));
        if (!r.ok) throw new Error(`${r.status} ${f}`);
        return new Uint8Array(await r.arrayBuffer());
      }));
      const c = { lab: a, vm: b };
      cache.set(k, c);
      held += a.length + b.length;
      for (const [kk, cc] of cache) {
        if (held <= BUDGET || kk === k) break;
        held -= cc.lab.length + cc.vm.length;
        cache.delete(kk);
      }
      return c;
    }
    // two components per voxel: the stress code (colour of a resin) and the
    // label (what is drawn, the colour of a material); one empty voxel on
    // every side so that the faces of the box are shaded as the rest
    function assemble(s, d, cut) {
      const [nx, ny, nz] = s.dims, X = nx + 2, Y = ny + 2, Z = nz + 2;
      const out = new Uint8Array(2 * X * Y * Z);
      for (let i = 1; i < out.length; i += 2) out[i] = ABSENT;
      const j0 = cut ? Math.floor(ny / 2) : 0;
      for (let k = 0; k < nz; k++) {
        for (let j = j0; j < ny; j++) {
          const src = (k * ny + j) * nx, dst = 2 * (((k + 1) * Y + (j + 1)) * X + 1);
          for (let i = 0; i < nx; i++) { out[dst + 2 * i] = d.vm[src + i]; out[dst + 2 * i + 1] = d.lab[src + i]; }
        }
      }
      return out;
    }
    const st = { k: F - 1, playing: false, token: 0, seq: 0, show: "box", colour: "mat", scale: "step", speed: "1", shader: null };
    const drawn = (l) => {
      const L = lab[l] || {};
      const resin = L.kind === "matrix" || L.kind === "interphase";
      if (st.show === "particles") return !resin;
      return !L.void;
    };
    function transfer(s) {
      const h = h0;
      const op = PWF.newInstance(), lc = CTF.newInstance();
      const vals = Object.keys(colours).map(Number).sort((a, b) => a - b);
      vals.forEach((v) => {
        const o = drawn(v) ? 1.0 : 0.0, c = hex(colours[v]);
        op.addPoint(v - 0.45, o); op.addPoint(v + 0.45, o);
        lc.addRGBPoint(v - 0.45, ...c); lc.addRGBPoint(v + 0.45, ...c);
      });
      op.addPoint(ABSENT - 0.45, 0); op.addPoint(ABSENT + 0.45, 0);
      const vmax = (st.scale === "all" ? P.vm_max3d : s.vm_max) || 1;
      const fc = CTF.newInstance();
      fc.addRGBPoint(0, 0.80, 0.80, 0.80);              // no resin here: a sphere or a pore, grey
      for (let c = 1; c <= 255; c += 2) fc.addRGBPoint(c, ...jet(((c - 1) / 254) * s.vm_max / vmax));
      const zero = PWF.newInstance(); zero.addPoint(0, 0); zero.addPoint(255, 0);
      pr.setRGBTransferFunction(0, fc); pr.setScalarOpacity(0, zero);
      pr.setRGBTransferFunction(1, lc); pr.setScalarOpacity(1, op);
      pr.setScalarOpacityUnitDistance(0, 0.1 * h);
      pr.setScalarOpacityUnitDistance(1, 0.1 * h);
      pr.setAmbient(st.colour === "vm" ? 0.4 : 0.3);
      mapper.setSampleDistance(0.25 * h);
      const want = st.colour === "vm" ? "vm" : "mat";
      if (st.shader !== want) {
        mapper.getViewSpecificProperties().OpenGL = { ShaderReplacements: [{ shaderType: "Fragment", originalValue: "//VTK::CustomColorMix",
          replaceFirst: false, replaceAll: false, replacementValue: MV.crispMix(1, want === "vm" ? 0 : 1, false) }] };
        mapper.modified();
        st.shader = want;
      }
      return vmax;
    }
    const legend = q("legend"), read = q("read"), slider = q("step"), wait = q("wait");
    const START = `<div style="display:flex;align-items:center;gap:6px;margin-top:4px;color:#57606a"><i style="display:inline-block;width:14px;height:0;border-top:1px solid #bfc6ce"></i>the box at the start</div>`;
    function legendHTML(vmax) {
      if (st.colour === "vm") {
        const stops = Array.from({ length: 13 }, (_, i) => `rgb(${jet(i / 12).map((c) => Math.round(255 * c))}) ${(100 * i) / 12}%`).join(",");
        return `<b>von Mises stress in the resins (Pa)</b><div style="display:flex;align-items:center;gap:6px;margin-top:4px"><span>0</span><div style="width:120px;height:10px;border:1px solid #afb8c1;background:linear-gradient(to right,${stops})"></div><span>${fmt(vmax)}</span></div>
          <div style="color:#57606a;margin-top:3px">${st.scale === "all" ? "one range for every step" : "the range of this step"}; spheres and pores grey</div>${START}`;
      }
      return (P.legend || []).filter((l) => {
        const v = Object.keys(lab).find((k) => lab[k].name === l.name);
        return v === undefined || drawn(+v);
      }).map((l) => `<div style="display:flex;align-items:center;gap:6px"><i style="display:inline-block;width:10px;height:10px;background:${l.color};border:1px solid #8c959f"></i>${l.name}</div>`).join("") + START;
    }
    function readout(k) {
      const s = steps[k];
      let t = `Step ${k + 1}/${F}`;
      if (s.compression !== null && s.compression !== undefined) t += ` · compression ${(100 * s.compression).toFixed(1)} %`;
      t += `<br>box ${boxOf(s).map((v) => v.toPrecision(3)).join(" × ")} µm (${s.dims.join(" × ")} voxels)`;
      if (s.sigma_zz_Pa !== null && s.sigma_zz_Pa !== undefined) t += ` · σzz ${fmt(s.sigma_zz_Pa)} Pa`;
      if (s.time_s !== undefined) {
        t += `<br>t ${fmt(s.time_s)} s · ${s.T_C.toFixed(1)} °C`;
        const al = Object.entries(s.alpha || {});
        if (al.length) t += " · conversion " + al.map(([n, a]) => `${n.split(" (")[0]} ${a.toFixed(3)}`).join(", ");
      }
      return t;
    }
    async function show(k) {
      const my = ++st.seq;
      wait.textContent = cache.has(k) ? "" : "loading the step …";
      let d;
      try { d = await load(k); } catch (e) { wait.textContent = `Step ${k + 1} could not be loaded: ${e.message}`; return false; }
      if (my !== st.seq) return false;
      wait.textContent = "";
      const s = steps[k];
      const sp = [h0, h0, h0];
      const img = ImageData.newInstance();
      img.setDimensions(s.dims[0] + 2, s.dims[1] + 2, s.dims[2] + 2);
      img.setSpacing(...sp);
      img.setOrigin(...sp.map((v) => -0.5 * v));
      img.getPointData().setScalars(DataArray.newInstance({ name: "v", numberOfComponents: 2, values: assemble(s, d, st.show === "cut") }));
      mapper.setInputData(img);
      const vmax = transfer(s);
      box.set(boxOf(s));
      st.k = k;
      slider.value = String(k);
      read.innerHTML = readout(k);
      legend.innerHTML = legendHTML(vmax);
      ren.resetCameraClippingRange();
      rw.render();
      return true;
    }
    // the camera from the iso direction of the 3D view, on the first box
    function iso() {
      const L = U, c = L.map((v) => v / 2), dir = [1, -1.25, 0.9], n = Math.hypot(...dir), d = Math.max(...L) * 2.6;
      const cam = ren.getActiveCamera();
      cam.setParallelProjection(false);
      cam.setFocalPoint(...c);
      cam.setPosition(...c.map((v, a) => v + (dir[a] / n) * d));
      cam.setViewUp(0, 0, 1);
      ren.resetCamera([0, L[0], 0, L[1], 0, L[2]]);
      cam.zoom(0.92);
      ren.resetCameraClippingRange();
      rw.render();
    }
    function setPlaying(on) {
      st.playing = on;
      q("play").textContent = on ? "Pause" : "Play";
      st.token++;
      if (on) loop(st.token);
    }
    async function loop(token) {
      while (st.playing && token === st.token) {
        const t0 = performance.now();
        const k = (st.k + 1) % F;
        await show(k);
        if (k + 1 < F) load(k + 1).catch(() => {});
        // the last step held a little longer before it starts again
        const dt = DELAY[st.speed] * (k === F - 1 ? 4 : 1) - (performance.now() - t0);
        if (dt > 0) await new Promise((r) => setTimeout(r, dt));
      }
    }
    q("play").addEventListener("click", () => { if (!st.playing && st.k === F - 1) st.k = -1; setPlaying(!st.playing); });
    q("prev").addEventListener("click", () => { setPlaying(false); show(Math.max(0, st.k - 1)); });
    q("next").addEventListener("click", () => { setPlaying(false); show(Math.min(F - 1, st.k + 1)); });
    slider.addEventListener("input", () => { setPlaying(false); show(+slider.value); });
    q("speed").addEventListener("change", (e) => { st.speed = e.target.value; });
    q("show").addEventListener("change", (e) => { st.show = e.target.value; show(st.k); });
    q("colour").addEventListener("change", (e) => { st.colour = e.target.value; q("scale").style.display = st.colour === "vm" ? "" : "none"; show(st.k); });
    q("scale").addEventListener("change", (e) => { st.scale = e.target.value; show(st.k); });
    q("scale").style.display = "none";
    q("iso").addEventListener("click", iso);
    q("gif").addEventListener("click", async (e) => {
      if (!window.GifRec) return;
      const b = e.target, was = st.playing, keep = st.k;
      setPlaying(false);
      b.disabled = true;
      try {
        await window.GifRec.record({
          count: F, delay: Math.max(80, DELAY[st.speed]), maxWidth: 900,
          filename: `${(opts.name || "compression").replace(/[^\w.-]+/g, "_")}_compression.gif`,
          frame: async (k) => { await show(k); const imgs = await rw.captureImages(); return imgs[0]; },
          overlay: (g, w, h, k) => {
            const s = steps[k];
            g.font = "13px system-ui, sans-serif"; g.fillStyle = "#1f2328"; g.textAlign = "left";
            g.fillText(`Compression · step ${k + 1}/${F}` + (s.compression !== null ? ` · ${(100 * s.compression).toFixed(1)} %` : "")
              + (s.sigma_zz_Pa !== null ? ` · σzz ${fmt(s.sigma_zz_Pa)} Pa` : ""), 12, h - 12);
            if (st.colour === "vm") {
              window.GifRec.colourBar(g, w - 90, 46, Math.round(h * 0.38), "von Mises (Pa)", 0, (st.scale === "all" ? P.vm_max3d : s.vm_max) || 1, jet, fmt);
            } else {
              (P.legend || []).forEach((l, j) => {
                g.fillStyle = l.color; g.fillRect(w - 170, 22 + 18 * j, 11, 11);
                g.fillStyle = "#1f2328"; g.fillText(l.name.split(" (")[0], w - 154, 32 + 18 * j);
              });
            }
          },
          onProgress: (i, n) => { b.textContent = `GIF ${Math.round((100 * i) / n)} %`; },
        });
      } catch (err) { console.warn("GIF", err); }
      finally { b.disabled = false; b.textContent = "Save GIF"; await show(keep); if (was) setPlaying(true); }
    });
    const ro = new ResizeObserver(() => { try { grw.resize(); rw.render(); } catch (e) { } });
    ro.observe(view);
    // the last step first (the box as it ended), seen from the iso direction
    // on the first box; Play runs from the start
    await show(F - 1);
    iso();
    load(0).catch(() => {});
    return {
      stop() { st.playing = false; st.token++; ro.disconnect(); try { grw.delete(); } catch (e) { } cache.clear(); },
    };
  };
})();
