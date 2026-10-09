/* The particle dynamics of the viscosity analysis, played in a small 3D view
   of its own: the periodic box sheared in the xy plane (the top moves +x,
   the bottom -x), every filler sphere at its size, moving and turning with
   the resin, frame after frame of the last strain the solver ran. Spheres are
   coloured by their phase, or by their speed relative to the shear flow -
   the particles the shear pushes past their neighbours light up. The box is
   the solver's own sample, not the RVE of the 3D view. */
(function () {
  "use strict";
  const ns = (path) => path.split(".").reduce((o, k) => (o ? o[k] : undefined), window.vtk);
  const JET = [[0, 0, 0, 0.5], [0.11, 0, 0, 1], [0.36, 0, 1, 1], [0.5, 0.5, 1, 0.5], [0.64, 1, 1, 0], [0.89, 1, 0, 0], [1, 0.5, 0, 0]];
  const hex = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255);

  async function bin(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.arrayBuffer();
  }

  window.DemPlayer = async function (host, jobId, dem, built) {
    if (!window.vtk || !host) return null;
    const v = built || 0;
    const base = `/api/jobs/${jobId}/view/`;
    const [pos, spd, rad, ph] = await Promise.all([dem.pos, dem.speed, dem.radii, dem.phase]
      .map((f) => bin(`${base}${f}?v=${v}`)));
    const P = new Float32Array(pos), Sp = new Float32Array(spd), Rd = new Float32Array(rad), Ph = new Uint8Array(ph);
    const n = dem.n, F = dem.frames, L = dem.box_um;
    host.innerHTML = `<div style="position:relative"><div class="demview" style="position:relative;height:360px;border:1px solid var(--line,#d0d7de);border-radius:6px;overflow:hidden"></div>
      <div data-dem="legend" style="position:absolute;right:10px;top:10px;background:rgba(255,255,255,.88);border:1px solid #d0d7de;border-radius:4px;padding:6px 8px;font:11px system-ui,sans-serif;color:#1f2328;pointer-events:none"></div></div>
      <div class="row gap" style="align-items:center;margin-top:6px;flex-wrap:wrap">
        <button class="btn sm" data-dem="play">Pause</button>
        <input type="range" data-dem="frame" min="0" max="${F - 1}" value="0" style="flex:1;min-width:160px">
        <span class="mono" data-dem="read" style="min-width:150px;text-align:right"></span>
        <select data-dem="color"><option value="speed">Colour: speed relative to the shear</option><option value="phase">Colour: filler</option></select>
      </div>`;
    const view = host.querySelector(".demview");
    const GRW = ns("Rendering.Misc.vtkGenericRenderWindow"), PD = ns("Common.DataModel.vtkPolyData");
    const DA = ns("Common.Core.vtkDataArray"), Glyph = ns("Rendering.Core.vtkGlyph3DMapper");
    const Sphere = ns("Filters.Sources.vtkSphereSource"), Actor = ns("Rendering.Core.vtkActor");
    const Mapper = ns("Rendering.Core.vtkMapper"), CTF = ns("Rendering.Core.vtkColorTransferFunction");
    const grw = GRW.newInstance({ background: [1, 1, 1] });
    grw.setContainer(view);
    grw.resize();
    const ren = grw.getRenderer(), rw = grw.getRenderWindow();
    // the spheres: points of frame 0, scaled by diameter, coloured by an array
    const pts = new Float32Array(3 * n);
    const diam = new Float32Array(n), col = new Float32Array(n);
    for (let i = 0; i < n; i++) diam[i] = 2 * Rd[i];
    const pd = PD.newInstance();
    pd.getPoints().setData(pts, 3);
    const dArr = DA.newInstance({ name: "diam", numberOfComponents: 1, values: diam });
    const cArr = DA.newInstance({ name: "c", numberOfComponents: 1, values: col });
    pd.getPointData().addArray(dArr);
    pd.getPointData().addArray(cArr);
    pd.getPointData().setScalars(cArr);
    const gm = Glyph.newInstance();
    gm.setInputData(pd, 0);
    gm.setInputConnection(Sphere.newInstance({ radius: 0.5, thetaResolution: 16, phiResolution: 12 }).getOutputPort(), 1);
    gm.setScaleArray("diam");
    gm.setScaleMode(Glyph.ScaleModes.SCALE_BY_MAGNITUDE);
    gm.setScaleFactor(1.0);
    gm.setScalarVisibility(true);
    gm.setUseLookupTableScalarRange(true);
    const actor = Actor.newInstance();
    actor.setMapper(gm);
    actor.getProperty().setSpecular(0.35);
    actor.getProperty().setSpecularPower(30);
    ren.addActor(actor);
    // the box edges
    const corners = [];
    for (const z of [0, L]) for (const y of [0, L]) for (const x of [0, L]) corners.push(x, y, z);
    const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    const lines = new Uint32Array(edges.length * 3);
    edges.forEach(([a, b], k) => { lines[3 * k] = 2; lines[3 * k + 1] = a; lines[3 * k + 2] = b; });
    const box = PD.newInstance();
    box.getPoints().setData(new Float32Array(corners), 3);
    box.getLines().setData(lines);
    const bm = Mapper.newInstance();
    bm.setInputData(box);
    const ba = Actor.newInstance();
    ba.setMapper(bm);
    ba.getProperty().setColor(0.35, 0.38, 0.42);
    ren.addActor(ba);

    const vmax = Math.max(dem.speed_p99 || 1, 1e-9);
    const lutSpeed = () => { const c = CTF.newInstance(); JET.forEach(([t, r, g, b]) => c.addRGBPoint(t * vmax, r, g, b)); return c; };
    const lutPhase = () => {
      const c = CTF.newInstance();
      (dem.colors || []).forEach((h, k) => { const [r, g, b] = hex(h); c.addRGBPoint(k, r, g, b); c.addRGBPoint(k + 0.999, r, g, b); });
      return c;
    };
    const state = { frame: 0, playing: true, mode: "speed", last: 0, id: null };
    const legend = host.querySelector('[data-dem="legend"]');
    const fmtv = (x) => (Math.abs(x) >= 100 || Math.abs(x) < 0.01 ? x.toExponential(1) : x.toPrecision(2));
    function setMode(m) {
      state.mode = m;
      gm.setLookupTable(m === "speed" ? lutSpeed() : lutPhase());
      if (m === "speed") {
        const stops = JET.map(([t, r, g, b]) => `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${t * 100}%`).join(",");
        legend.innerHTML = `<b>Speed relative to the shear</b><br><span style="color:#57606a">in units of shear rate × largest radius</span>
          <div style="display:flex;align-items:center;gap:6px;margin-top:4px"><span>0</span><div style="width:110px;height:10px;border:1px solid #afb8c1;background:linear-gradient(to right,${stops})"></div><span>${fmtv(vmax)}</span></div>`;
      } else {
        legend.innerHTML = (dem.names || []).map((nm, k) => `<div style="display:flex;align-items:center;gap:6px"><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${(dem.colors || [])[k] || "#888"}"></i>${nm}</div>`).join("");
      }
      show(state.frame);
    }
    const read = host.querySelector('[data-dem="read"]'), slider = host.querySelector('[data-dem="frame"]');
    function show(k) {
      const o = 3 * n * k;
      for (let i = 0; i < 3 * n; i++) pts[i] = P[o + i];
      for (let i = 0; i < n; i++) col[i] = state.mode === "speed" ? Sp[n * k + i] : Ph[i] + 0.5;
      pd.getPoints().modified();
      cArr.modified();
      pd.modified();
      rw.render();
      slider.value = String(k);
      read.textContent = `strain + ${(k * dem.strain_step).toFixed(2)} · frame ${k + 1}/${F}`;
    }
    function tick(ts) {
      state.id = requestAnimationFrame(tick);
      if (!state.playing || ts - state.last < 60) return;
      state.last = ts;
      state.frame = (state.frame + 1) % F;
      show(state.frame);
    }
    host.querySelector('[data-dem="play"]').addEventListener("click", (e) => {
      state.playing = !state.playing;
      e.target.textContent = state.playing ? "Pause" : "Play";
    });
    slider.addEventListener("input", () => { state.playing = false; host.querySelector('[data-dem="play"]').textContent = "Play"; state.frame = +slider.value; show(state.frame); });
    host.querySelector('[data-dem="color"]').addEventListener("change", (e) => setMode(e.target.value));
    // the camera from a corner, the shear plane facing the viewer
    const cam = ren.getActiveCamera();
    cam.setFocalPoint(L / 2, L / 2, L / 2);
    cam.setPosition(L / 2 + 0.9 * L, L / 2 + 0.7 * L, L / 2 + 2.3 * L);
    cam.setViewUp(0, 1, 0);
    ren.resetCameraClippingRange();
    new ResizeObserver(() => { try { grw.resize(); rw.render(); } catch (e) { } }).observe(view);
    setMode("speed");
    state.id = requestAnimationFrame(tick);
    return {
      stop() { if (state.id) cancelAnimationFrame(state.id); state.id = null; try { grw.delete(); } catch (e) { } },
    };
  };
})();
