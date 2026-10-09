/* 3D and section viewer.

   3D:  vtk.js (the web build of VTK, the family of the PyVista renderer on
        the server). A field is shown as voxel cells on the six outer faces of
        the RVE; particle and pore surfaces (VTK SurfaceNets, computed on the
        server) can be coloured by the field; orthogonal section planes, volume
        rendering and the extracted pore network are optional layers.
   2D:  canvas sections with nearest-neighbour pixels, phase boundaries, µm
        axes, value read-out, wheel zoom and drag pan. A click in one section
        moves the other two sections through the selected point.
   Every colour map is evaluated by the same function in all views, so a value
   has the same colour everywhere. */
(function () {
  "use strict";
  const V = (window.MPSViewer = {});
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  const fmt = (v) => window.MPSCharts.fmt(v);

  /* ------------------------------------------------------------ colour maps */
  function lerpTable(tab) {
    return (t) => {
      t = clamp01(t) * (tab.length - 1);
      const i = Math.min(Math.floor(t), tab.length - 2), f = t - i;
      return [0, 1, 2].map((c) => (tab[i][c] + (tab[i + 1][c] - tab[i][c]) * f) / 255);
    };
  }
  const CMAPS = {
    jet: (t) => { t = clamp01(t); return [clamp01(1.5 - Math.abs(4 * t - 3)), clamp01(1.5 - Math.abs(4 * t - 2)), clamp01(1.5 - Math.abs(4 * t - 1))]; },
    turbo: (t) => {
      t = clamp01(t);
      const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
      const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
      const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
      return [clamp01(r), clamp01(g), clamp01(b)];
    },
    viridis: lerpTable([[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]]),
    coolwarm: lerpTable([[59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77], [180, 4, 38]]),
    gray: (t) => { t = clamp01(t); return [t, t, t]; },
  };
  V.CMAPS = CMAPS;
  const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.substr(i, 2), 16) / 255);

  /* ------------------------------------------------------------ state */
  const S = {
    jobId: null, meta: null, labels: null, fields: {}, surfaces: {}, surfLoaded: false, network: null,
    // "smooth": SurfaceNets skins, one per particle; "voxels": the voxel faces
    // the solver sees, flat shaded and coloured voxel by voxel
    surfStyle: "smooth", voxels: {}, voxLoaded: false,
    // the RVE faces and the structure sections at the solver's resolution
    // (the browser volume is a quarter of it on a 640-voxel grid)
    faceData: {}, lslices: {}, fslices: {},
    // the voxel volume: every voxel of the structure (padded by one empty
    // voxel), and the field on every voxel at 8 bits when one is shown
    vvolLab: null, vvolField: null,
    field: "", cmap: "jet", range: "auto", vmin: null, vmax: null, log: false,
    slices: { x: 0, y: 0, z: 0 }, show: { x: false, y: false, z: false }, faces: false,
    particles: true, colorParticles: true, opacity: 1, clip: false, outline: true, outlines2d: true,
    // a phase keeps its colour everywhere it is drawn, so an override is held
    // by label index and read by the 3D surfaces, the 2D pixels and the legends
    colours: {},
    // Phong terms, as a digital-material suite exposes them: the defaults are
    // flat-looking on a pale structure, and what reads well depends on the model
    light: { ambient: 0.18, diffuse: 0.82, specular: 0.28, power: 30, shade: true, edges: false },
    hidden: {}, mode: "3d", parallel: false, axis2d: "z", volume: false, networkOn: false, axes: true,
    vectors: false, vecScale: 1, vecDensity: 1, vecUniform: false, vecData: {}, pathKey: "", pathData: {}, pathAnim: true,
  };
  let vtkReady = false, R = null;
  const views2d = {};
  const ns = (path) => path.split(".").reduce((o, k) => (o ? o[k] : undefined), window.vtk);
  V.can = { volume: () => !!(window.vtk && ns("Rendering.Core.vtkVolume") && ns("Rendering.Core.vtkVolumeMapper")),
            network: () => !!(window.vtk && ns("Rendering.Core.vtkGlyph3DMapper") && ns("Filters.Sources.vtkSphereSource")),
            arrows: () => !!(window.vtk && ns("Rendering.Core.vtkGlyph3DMapper") && ns("Filters.Sources.vtkArrowSource")) };

  /* ------------------------------------------------------------ data */
  async function fetchBin(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + " " + r.status);
    return r.arrayBuffer();
  }
  // a view written again (a continued run, a rebuilt view) gets new URLs, so
  // the browser never pairs files it cached with a newer meta.json
  const vurl = (file, meta) => `/api/jobs/${S.jobId}/view/${file}?v=${(meta || S.meta || {}).built || 0}`;
  async function loadField(key) {
    if (!key) return null;
    if (S.fields[key]) return S.fields[key];
    const f = S.meta.fields.find((x) => x.key === key);
    S.fields[key] = new Float32Array(await fetchBin(vurl(f.file)));
    return S.fields[key];
  }
  // surf_<label>.bin: "MPS2", n, m, step, origin, uint16 points, int8 normals,
  // uint32 triangles (runs before it: float32 points and normals)
  function parseSurface(buf) {
    const dv = new DataView(buf);
    if (buf.byteLength >= 28 && dv.getUint32(0, true) === 0x3253504d) {
      const n = dv.getUint32(4, true), m = dv.getUint32(8, true), st = dv.getFloat32(12, true);
      const o0 = dv.getFloat32(16, true), o1 = dv.getFloat32(20, true), o2 = dv.getFloat32(24, true);
      const q = new Uint16Array(buf, 28, 3 * n), n8 = new Int8Array(buf, 28 + 6 * n, 3 * n);
      const pts = new Float32Array(3 * n), nrm = new Float32Array(3 * n);
      for (let i = 0; i < 3 * n; i += 3) {
        pts[i] = o0 + q[i] * st; pts[i + 1] = o1 + q[i + 1] * st; pts[i + 2] = o2 + q[i + 2] * st;
        nrm[i] = n8[i] / 127; nrm[i + 1] = n8[i + 1] / 127; nrm[i + 2] = n8[i + 2] / 127;
      }
      const off = 28 + 9 * n + ((4 - ((28 + 9 * n) % 4)) % 4);
      return { n, m, pts, nrm, tri: new Uint32Array(buf, off, 3 * m) };
    }
    const n = dv.getUint32(0, true), m = dv.getUint32(4, true);
    return { n, m, pts: new Float32Array(buf, 8, 3 * n), nrm: new Float32Array(buf, 8 + 12 * n, 3 * n),
             tri: new Uint32Array(buf.slice(8 + 24 * n, 8 + 24 * n + 12 * m)) };
  }
  async function loadSurfaces() {
    if (S.surfLoaded) return;
    for (const sf of S.meta.surfaces) {
      const p = parseSurface(await fetchBin(vurl(sf.file)));
      S.surfaces[sf.label] = { meta: sf, ...p, fields: {} };
    }
    S.surfLoaded = true;
  }
  // the field value of every voxel face, read from the solver's grid when
  // the run wrote it (otherwise the browser volume's value behind the face)
  async function loadVoxField(label, key) {
    const vd = S.voxels[label], vf = ((S.meta.voxel_style || {}).files || {})[label];
    const fn = vf && (vf.fields || {})[key];
    if (!vd || !fn) return null;
    vd.fields = vd.fields || {};
    if (!vd.fields[key]) vd.fields[key] = new Float32Array(await fetchBin(vurl(fn)));
    return vd.fields[key];
  }
  // the six RVE faces at the solver's resolution: labels, or the field shown
  async function loadFaces(key) {
    const fc = S.meta.faces;
    if (!fc) return null;
    const k = key || "_labels";
    if (S.faceData[k]) return S.faceData[k];
    if (key) {
      const fn = (fc.fields || {})[key];
      if (!fn) return null;
      S.faceData[k] = new Float32Array(await fetchBin(vurl(fn)));
    } else {
      const u = new Uint8Array(await fetchBin(vurl(fc.labels)));
      const f = new Float32Array(u.length);
      for (let i = 0; i < u.length; i++) f[i] = u[i];
      S.faceData[k] = f;
    }
    return S.faceData[k];
  }
  // One section of the structure at the solver's resolution (server reads
  // it from structure_labels.tif), cropped to what the browser volume
  // covers: { w, h, st, data } with data[a + w * b], or null.
  async function labelSlice(axis, idx) {
    const st = S.meta.stride || 1;
    if (st <= 1 || !S.jobId) return null;
    const key = `${axis}:${idx}`;
    if (key in S.lslices) return S.lslices[key];
    const i = "xyz".indexOf(axis), fs = S.meta.full_shape, ws = S.meta.shape;
    const k = Math.min(idx * st + (st >> 1), fs[i] - 1);
    const uv = [0, 1, 2].filter((a) => a !== i);
    let out = null;
    try {
      const r = await fetch(`/api/jobs/${S.jobId}/lslice/${axis}/${k}?v=${S.meta.built || 0}`);
      if (r.ok) {
        const raw = new Uint8Array(await r.arrayBuffer());
        const W = fs[uv[0]], H = fs[uv[1]], w = Math.min(W, ws[uv[0]] * st), h = Math.min(H, ws[uv[1]] * st);
        let data = raw;
        if (w !== W || h !== H) {
          data = new Uint8Array(w * h);
          for (let b = 0; b < h; b++) data.set(raw.subarray(b * W, b * W + w), b * w);
        }
        out = { w, h, st, data };
      }
    } catch (e) { out = null; }
    const keys = Object.keys(S.lslices);
    if (keys.length > 24) delete S.lslices[keys[0]];
    S.lslices[key] = out;
    return out;
  }
  // One section of a field at the solver's resolution (server decodes the
  // run's 16-bit copy), cropped like labelSlice: { w, h, st, data } or null.
  async function fieldSlice(key, axis, idx) {
    const st = S.meta.stride || 1;
    const f = S.meta.fields.find((x) => x.key === key);
    if (st <= 1 || !S.jobId || !f || !f.full) return null;
    const ck = `${key}|${axis}:${idx}`;
    if (ck in S.fslices) return S.fslices[ck];
    const i = "xyz".indexOf(axis), fs = S.meta.full_shape, ws = S.meta.shape;
    const k = Math.min(idx * st + (st >> 1), fs[i] - 1);
    const uv = [0, 1, 2].filter((a) => a !== i);
    let out = null;
    try {
      const r = await fetch(`/api/jobs/${S.jobId}/fslice/${encodeURIComponent(key)}/${axis}/${k}?v=${S.meta.built || 0}`);
      if (r.ok) {
        const raw = new Float32Array(await r.arrayBuffer());
        const W = fs[uv[0]], H = fs[uv[1]], w = Math.min(W, ws[uv[0]] * st), h = Math.min(H, ws[uv[1]] * st);
        let data = raw;
        if (w !== W || h !== H) {
          data = new Float32Array(w * h);
          for (let b = 0; b < h; b++) data.set(raw.subarray(b * W, b * W + w), b * w);
        }
        out = { w, h, st, data };
      }
    } catch (e) { out = null; }
    const keys = Object.keys(S.fslices);
    if (keys.length > 24) delete S.fslices[keys[0]];
    S.fslices[ck] = out;
    return out;
  }
  // a volume in VTK point order, one empty voxel added on every side so that
  // the particles cut by the box faces are shaded like the rest
  function padVolume(raw, dims) {
    const [nx, ny, nz] = dims, X = nx + 2, Y = ny + 2;
    const out = new Uint8Array(X * Y * (nz + 2));
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        const s = (k * ny + j) * nx;
        out.set(raw.subarray(s, s + nx), ((k + 1) * Y + (j + 1)) * X + 1);
      }
    }
    return out;
  }
  async function loadVoxVolume() {
    const vv = S.meta.voxel_volume;
    if (!vv) return null;
    if (!S.vvolLab) S.vvolLab = padVolume(new Uint8Array(await fetchBin(vurl(vv.labels))), vv.dims);
    return S.vvolLab;
  }
  async function loadVoxFieldVolume(key) {
    const f = S.meta.fields.find((x) => x.key === key);
    if (!f || !f.full || !S.meta.voxel_volume) return null;
    if (S.vvolField && S.vvolField.key === key) return S.vvolField;
    S.vvolField = null;
    try {
      const raw = new Uint8Array(await fetchBin(`/api/jobs/${S.jobId}/vol8/${encodeURIComponent(key)}?v=${S.meta.built || 0}`));
      S.vvolField = { key, full: f.full, data: padVolume(raw, S.meta.voxel_volume.dims) };
    } catch (e) { S.vvolField = null; }
    return S.vvolField;
  }

  async function loadVoxels() {
    const vs = S.meta.voxel_style;
    if (S.voxLoaded || !vs) return;
    for (const [label, vf] of Object.entries(vs.files || {})) {
      const buf = await fetchBin(vurl(vf.file));
      const dv = new DataView(buf);
      const n = dv.getUint32(0, true), m = dv.getUint32(4, true);
      const corner = new Uint16Array(buf.slice(8, 8 + 6 * n));
      const pts = new Float32Array(3 * n), sp = vs.spacing_um;
      for (let i = 0; i < 3 * n; i++) pts[i] = corner[i] * sp;
      const q = new Uint32Array(buf.slice(8 + 6 * n, 8 + 6 * n + 16 * m));
      const polys = new Uint32Array(5 * m);
      for (let i = 0, o = 0; i < m; i++) { polys[o++] = 4; polys[o++] = q[4 * i]; polys[o++] = q[4 * i + 1]; polys[o++] = q[4 * i + 2]; polys[o++] = q[4 * i + 3]; }
      S.voxels[label] = { n, m, pts, polys, vidx: new Uint32Array(buf.slice(8 + 6 * n + 16 * m, 8 + 6 * n + 20 * m)) };
    }
    S.voxLoaded = true;
  }
  async function loadSurfaceField(label, key) {
    const s = S.surfaces[label];
    if (!s || !key || !s.meta.fields[key]) return null;
    if (!s.fields[key]) s.fields[key] = new Float32Array(await fetchBin(vurl(s.meta.fields[key])));
    return s.fields[key];
  }
  async function loadNetwork() {
    if (S.network || !S.meta.network) return S.network;
    const r = await fetch(vurl(S.meta.network.file));
    if (r.ok) S.network = await r.json();
    return S.network;
  }

  async function loadVector(key) {
    if (S.vecData[key]) return S.vecData[key];
    const v = (S.meta.vectors || []).find((x) => x.key === key);
    if (!v) return null;
    S.vecData[key] = new Float32Array(await fetchBin(vurl(v.file)));
    return S.vecData[key];
  }
  async function loadPathData(key) {
    if (S.pathData[key]) return S.pathData[key];
    const p = (S.meta.paths || []).find((x) => x.key === key);
    if (!p) return null;
    const r = await fetch(vurl(p.file));
    if (!r.ok) return null;
    S.pathData[key] = await r.json();
    return S.pathData[key];
  }

  function fieldMeta() { return S.field ? S.meta.fields.find((f) => f.key === S.field) : null; }
  // the arrows belong to the displayed quantity: a magnitude and its direction
  function vectorFor(field) {
    const vs = (S.meta && S.meta.vectors) || [];
    return vs.find((v) => v.field === field) || vs.find((v) => v.key === field) || null;
  }
  function pathMeta() { return S.pathKey ? ((S.meta.paths || []).find((p) => p.key === S.pathKey) || null) : null; }
  function vecEvery(n) {
    return Math.max(1, Math.round(n / (18 * (S.vecDensity || 1))));
  }
  /* Arrow length relative to the spacing between arrows. Proportional to the
     magnitude, the arrows in a low-conductivity matrix - where most of them
     stand - came out a few pixels long beside the p99 flux inside the filler,
     and the length and density sliders seemed to do nothing. The square root
     keeps the order visible over two decades; "same length" shows the
     direction only and leaves the magnitude to the colour. */
  function vecLen(m, ref) {
    if (S.vecUniform) return 1;
    return Math.min(1.35, Math.sqrt(Math.max(m, 0) / Math.max(ref, 1e-30)));
  }
  V.vecLen = vecLen;
  V.vectorFor = vectorFor;
  V.pathMeta = pathMeta;
  const escHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function currentRange() {
    const f = fieldMeta();
    if (!f) return [0, S.meta.labels.length - 1];
    let lo, hi;
    if (S.range === "manual" && S.vmin !== null && S.vmax !== null && S.vmax > S.vmin) { lo = S.vmin; hi = S.vmax; }
    else if (S.range === "full") { lo = f.min; hi = f.max; }
    // class numbers are not a measurement: the percentile window collapses onto
    // whichever class fills the volume, and every other class disappears into
    // one flat colour. The whole range is the only range that shows them.
    else if (f.categorical) { lo = f.min; hi = f.max; }
    else { lo = f.p01; hi = f.p99; }
    // A diverging map is centred on zero so that the two signs read apart, but
    // only when the data actually has two signs. A stress that is compressive
    // everywhere (-100 .. -56 MPa) was being given a -78 .. +78 bar: half the
    // colours went unused and every voxel landed in the bottom quarter, leaving
    // a flat blue block.
    if (f.diverging && S.range === "auto" && lo < 0 && hi > 0) {
      const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m;
    }
    if (S.log) { lo = Math.max(lo, f.log_min || hi * 1e-6, 1e-30); if (hi <= lo) hi = lo * 10; }
    if (hi <= lo) hi = lo + (Math.abs(lo) || 1) * 1e-6;
    return [lo, hi];
  }
  V.currentRange = () => (S.meta ? currentRange() : [0, 1]);
  function mapValue(v, lo, hi) {
    if (S.log) return (Math.log10(Math.max(v, 1e-300)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
    return (v - lo) / (hi - lo);
  }

  /* ------------------------------------------------------------ 3D (vtk.js) */
  function initVtk(container) {
    if (vtkReady) return true;
    if (!window.vtk) { V.showEmpty("vtk.js could not be loaded."); return false; }
    const GRW = ns("Rendering.Misc.vtkGenericRenderWindow");
    const grw = GRW.newInstance({ background: [1, 1, 1] });
    grw.setContainer(container);
    grw.resize();
    const renderer = grw.getRenderer(), rw = grw.getRenderWindow(), interactor = grw.getInteractor();
    const AxesActor = ns("Rendering.Core.vtkAxesActor"), OMW = ns("Interaction.Widgets.vtkOrientationMarkerWidget");
    let omw = null;
    try {
      omw = OMW.newInstance({ actor: AxesActor.newInstance(), interactor });
      omw.setEnabled(true);
      omw.setViewportCorner(OMW.Corners.BOTTOM_LEFT);
      omw.setViewportSize(0.13);
      omw.setMinPixelSize(70);
      omw.setMaxPixelSize(130);
    } catch (e) { console.warn("axes widget", e); }
    // A light on the camera axis gives the three faces of a box seen from a
    // corner the same brightness, so flat-shaded voxels read as one solid
    // block. A key light above and to the left of the view, with a weak fill
    // from the other side, lets every face orientation read differently; both
    // move with the camera.
    try {
      const Light = ns("Rendering.Core.vtkLight");
      if (Light && renderer.addLight) {
        if (renderer.setAutomaticLightCreation) renderer.setAutomaticLightCreation(false);
        if (renderer.removeAllLights) renderer.removeAllLights();
        const key = Light.newInstance(), fill = Light.newInstance();
        key.setLightTypeToCameraLight(); key.setPosition(-0.5, 0.75, 1.0); key.setFocalPoint(0, 0, 0); key.setIntensity(0.85);
        fill.setLightTypeToCameraLight(); fill.setPosition(0.7, -0.35, 1.0); fill.setFocalPoint(0, 0, 0); fill.setIntensity(0.35);
        renderer.addLight(key); renderer.addLight(fill);
      }
    } catch (e) { console.warn("lights", e); }
    R = { grw, renderer, rw, interactor, omw, image: null, slices: {}, faces: null, surf: {}, vox: {}, outline: null,
          ctf: null, net: null, volume: null, vec: null, path: null, animId: null };
    new ResizeObserver(() => { try { grw.resize(); rw.render(); } catch (e) { } }).observe(container);
    vtkReady = true;
    return true;
  }

  function makeCTF(lo, hi, cmapName, logScale) {
    const ctf = ns("Rendering.Core.vtkColorTransferFunction").newInstance();
    const cm = CMAPS[cmapName || S.cmap] || CMAPS.jet, n = 64;
    const lg = logScale === undefined ? S.log : logScale;
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const x = lg ? Math.log10(lo) + t * (Math.log10(hi) - Math.log10(lo)) : lo + t * (hi - lo);
      ctf.addRGBPoint(x, ...cm(t));
    }
    return ctf;
  }
  function labelCTF() {
    const ctf = ns("Rendering.Core.vtkColorTransferFunction").newInstance();
    S.meta.labels.forEach((l) => {
      const c = hex2rgb(l.color);
      ctf.addRGBPoint(l.value - 0.49, ...c);
      ctf.addRGBPoint(l.value + 0.49, ...c);
    });
    return ctf;
  }

  function scalarsFor(arr) {
    if (!arr) { const out = new Float32Array(S.labels.length); for (let i = 0; i < out.length; i++) out[i] = S.labels[i]; return out; }
    if (!S.log) return arr;
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = Math.log10(Math.max(arr[i], 1e-300));
    return out;
  }

  function styleSlice(actor, ctf, visible) {
    const p = actor.getProperty();
    p.setRGBTransferFunction(0, ctf);
    p.setUseLookupTableScalarRange(true);
    p.setInterpolationTypeToNearest();
    actor.setVisibility(visible);
  }

  async function build3D() {
    if (!R || !S.meta) return;
    const { renderer, rw } = R;
    const [nx, ny, nz] = S.meta.shape, s = S.meta.spacing_um;
    const ImageData = ns("Common.DataModel.vtkImageData"), DataArray = ns("Common.Core.vtkDataArray");
    const ImageMapper = ns("Rendering.Core.vtkImageMapper"), ImageSlice = ns("Rendering.Core.vtkImageSlice");
    const [lo, hi] = currentRange();
    const arr = S.field ? await loadField(S.field) : null;
    const ctf = S.field ? makeCTF(lo, hi) : labelCTF();
    R.ctf = ctf;
    if (!R.image) {
      R.image = ImageData.newInstance();
      R.image.setDimensions(nx, ny, nz);
      R.image.setSpacing(s, s, s);
      R.image.setOrigin(s / 2, s / 2, s / 2);
    }
    R.image.getPointData().setScalars(DataArray.newInstance({ name: "v", numberOfComponents: 1, values: scalarsFor(arr) }));
    R.image.modified();
    const modes = { x: ImageMapper.SlicingMode.I, y: ImageMapper.SlicingMode.J, z: ImageMapper.SlicingMode.K };
    const newSlice = (ax, idx) => {
      const mapper = ImageMapper.newInstance();
      mapper.setInputData(R.image);
      mapper.setSlicingMode(modes[ax]);
      mapper.setSlice(idx);
      const actor = ImageSlice.newInstance();
      actor.setMapper(mapper);
      renderer.addActor(actor);
      return { mapper, actor };
    };
    // six outer faces: at the solver's resolution where the run wrote them,
    // otherwise as slices of the browser volume
    if (!R.faces) R.faces = [["x", 0], ["x", nx - 1], ["y", 0], ["y", ny - 1], ["z", 0], ["z", nz - 1]].map(([ax, idx]) => newSlice(ax, idx));
    const fine = S.faces ? await buildFaces(ctf) : (hideFaces(), false);
    R.faces.forEach((f) => styleSlice(f.actor, ctf, !!S.faces && !fine));
    // optional section planes
    ["x", "y", "z"].forEach((ax) => {
      if (!R.slices[ax]) R.slices[ax] = newSlice(ax, S.slices[ax]);
      R.slices[ax].mapper.setSlice(S.slices[ax]);
      styleSlice(R.slices[ax].actor, ctf, !!S.show[ax]);
    });
    await fineSections(ctf);
    // bounding box
    const PD = ns("Common.DataModel.vtkPolyData"), Mapper = ns("Rendering.Core.vtkMapper"), Actor = ns("Rendering.Core.vtkActor");
    if (!R.outline) {
      const X = nx * s, Y = ny * s, Z = nz * s;
      const P = new Float32Array([0, 0, 0, X, 0, 0, X, Y, 0, 0, Y, 0, 0, 0, Z, X, 0, Z, X, Y, Z, 0, Y, Z]);
      const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
      const lines = new Uint32Array(E.length * 3);
      E.forEach((e, i) => { lines[3 * i] = 2; lines[3 * i + 1] = e[0]; lines[3 * i + 2] = e[1]; });
      const box = PD.newInstance();
      box.getPoints().setData(P, 3);
      box.getLines().setData(lines);
      const m = Mapper.newInstance();
      m.setInputData(box);
      const a = Actor.newInstance();
      a.setMapper(m);
      a.getProperty().setLineWidth(1.6);
      a.getProperty().setColor(0.25, 0.28, 0.32);
      renderer.addActor(a);
      R.outline = a;
    }
    R.outline.setVisibility(S.outline);
    // µm scale on the box edges, as digital-material suites annotate the RVE
    try {
      const CubeAxes = ns("Rendering.Core.vtkCubeAxesActor");
      if (CubeAxes && !R.cube) {
        const ca = CubeAxes.newInstance();
        ca.setCamera(renderer.getActiveCamera());
        ca.setDataBounds([0, nx * s, 0, ny * s, 0, nz * s]);
        ca.setAxisLabels(["x (µm)", "y (µm)", "z (µm)"]);
        if (ca.setGridLines) ca.setGridLines(false);
        if (ca.setTickTextStyle) ca.setTickTextStyle({ fontColor: "#424a53", fontSize: 15, fontFamily: "Segoe UI, Arial" });
        if (ca.setAxisTextStyle) ca.setAxisTextStyle({ fontColor: "#1f2328", fontSize: 17, fontFamily: "Segoe UI, Arial" });
        ca.getProperty().setColor(0.45, 0.49, 0.54);
        renderer.addActor(ca);
        R.cube = ca;
      }
      if (R.cube) R.cube.setVisibility(!!S.axes);
    } catch (e) { console.warn("cube axes", e); R.cube = null; }
    // particle and pore surfaces
    await loadSurfaces();
    const Plane = ns("Common.DataModel.vtkPlane");
    for (const key of Object.keys(S.surfaces)) {
      const sd = S.surfaces[key];
      if (!R.surf[key]) {
        const pd = PD.newInstance();
        pd.getPoints().setData(sd.pts, 3);
        const cells = new Uint32Array(sd.m * 4);
        for (let i = 0, o = 0; i < sd.m; i++) { cells[o++] = 3; cells[o++] = sd.tri[3 * i]; cells[o++] = sd.tri[3 * i + 1]; cells[o++] = sd.tri[3 * i + 2]; }
        pd.getPolys().setData(cells);
        pd.getPointData().setNormals(DataArray.newInstance({ name: "Normals", numberOfComponents: 3, values: sd.nrm }));
        const m = Mapper.newInstance();
        m.setInputData(pd);
        const a = Actor.newInstance();
        a.setMapper(m);
        const pr = a.getProperty();
        const L = S.light;
        pr.setAmbient(L.ambient); pr.setDiffuse(L.diffuse); pr.setSpecular(L.specular); pr.setSpecularPower(L.power);
        renderer.addActor(a);
        R.surf[key] = { pd, m, a };
      }
      const { pd, m, a } = R.surf[key];
      const lab = S.meta.labels[+key];
      const vals = S.field && S.colorParticles ? await loadSurfaceField(+key, S.field) : null;
      if (vals) {
        pd.getPointData().setScalars(DataArray.newInstance({ name: "f", numberOfComponents: 1, values: S.log ? scalarsFor(vals) : vals }));
        m.setLookupTable(ctf);
        m.setUseLookupTableScalarRange(true);
        m.setScalarVisibility(true);
      } else {
        m.setScalarVisibility(false);
        a.getProperty().setColor(...hex2rgb(S.colours[key] || lab.color));
      }
      // re-applied on every refresh, not only when the actor is built, so the
      // lighting sliders take effect on the structure already on screen
      const LP = a.getProperty(), L = S.light;
      LP.setAmbient(L.ambient); LP.setDiffuse(L.diffuse);
      LP.setSpecular(L.specular); LP.setSpecularPower(L.power);
      // an outline around each phase separates touching particles, which a
      // smooth shaded surface alone leaves ambiguous
      LP.setEdgeVisibility(!!L.edges);
      LP.setEdgeColor(0.16, 0.19, 0.22);
      LP.setLineWidth(1);
      a.getProperty().setOpacity(S.opacity);
      a.setVisibility(S.particles && !S.hidden[key] && !(S.surfStyle === "voxels" && (S.meta.voxel_style || S.meta.voxel_volume)));
      m.removeAllClippingPlanes();
      if (S.clip) m.addClippingPlane(Plane.newInstance({ origin: [0, 0, (S.slices.z + 0.5) * s], normal: [0, 0, -1] }));
      pd.modified();
    }
    // the voxels themselves: every face between a label and the rest, flat
    // shaded (vtk.js takes the normal from the face when none is given), each
    // face in the colour of its own voxel's value
    const vs = S.meta.voxel_style;
    if (vs && S.surfStyle === "voxels") await loadVoxels();
    for (const key of Object.keys(S.voxels)) {
      const vd = S.voxels[key];
      if (!R.vox[key]) {
        const pd = PD.newInstance();
        pd.getPoints().setData(vd.pts, 3);
        pd.getPolys().setData(vd.polys);
        const m = Mapper.newInstance();
        m.setInputData(pd);
        const a = Actor.newInstance();
        a.setMapper(m);
        renderer.addActor(a);
        R.vox[key] = { pd, m, a };
      }
      const { pd, m, a } = R.vox[key];
      const lab = S.meta.labels[+key];
      const own = S.field && S.colorParticles ? await loadVoxField(key, S.field) : null;
      const arrF = S.field && S.colorParticles && !own ? await loadField(S.field) : null;
      if (own || arrF) {
        let vals = own;
        if (!vals) { vals = new Float32Array(vd.m); for (let i = 0; i < vd.m; i++) vals[i] = arrF[vd.vidx[i]]; }
        pd.getCellData().setScalars(DataArray.newInstance({ name: "f", numberOfComponents: 1, values: S.log ? scalarsFor(vals) : vals }));
        m.setScalarModeToUseCellData();
        m.setLookupTable(ctf);
        m.setUseLookupTableScalarRange(true);
        m.setScalarVisibility(true);
      } else {
        m.setScalarVisibility(false);
        a.getProperty().setColor(...hex2rgb(S.colours[key] || lab.color));
      }
      const LP = a.getProperty(), L = S.light;
      LP.setInterpolationToFlat();
      LP.setAmbient(Math.max(L.ambient, 0.25)); LP.setDiffuse(L.diffuse);
      LP.setSpecular(Math.min(L.specular, 0.15)); LP.setSpecularPower(L.power);
      LP.setEdgeVisibility(!!L.edges);
      LP.setEdgeColor(0.16, 0.19, 0.22);
      LP.setOpacity(S.opacity);
      a.setVisibility(S.particles && !S.hidden[key] && S.surfStyle === "voxels");
      m.removeAllClippingPlanes();
      if (S.clip) m.addClippingPlane(Plane.newInstance({ origin: [0, 0, (S.slices.z + 0.5) * s], normal: [0, 0, -1] }));
      pd.modified();
    }
    await buildVoxVolume();
    buildRegion();
    await buildNetwork();
    await buildVectors(ctf);
    await buildPaths();
    buildVolume(ctf, lo, hi);
    V.updateOverlays();
    renderer.getActiveCamera().setParallelProjection(S.parallel);
    rw.render();
  }

  // The six faces as 2D images at the solver's resolution. Returns false
  // when the run has no such faces for what is shown (the caller then uses
  // slices of the browser volume).
  function hideFaces() { (R.ffaces || []).forEach((f) => f && f.actor.setVisibility(false)); }
  async function buildFaces(ctf) {
    const fc = S.meta.faces;
    const data = fc ? await loadFaces(S.field || null) : null;
    if (!data) { hideFaces(); return false; }
    const ImageData = ns("Common.DataModel.vtkImageData"), DataArray = ns("Common.Core.vtkDataArray");
    const ImageMapper = ns("Rendering.Core.vtkImageMapper"), ImageSlice = ns("Rendering.Core.vtkImageSlice");
    const [fx, fy, fz] = fc.dims, sp = fc.spacing_um, f0 = fc.first_um, P = fc.planes_um;
    const defs = [[0, 0, [1, fy, fz]], [0, 1, [1, fy, fz]], [1, 0, [fx, 1, fz]], [1, 1, [fx, 1, fz]], [2, 0, [fx, fy, 1]], [2, 1, [fx, fy, 1]]];
    const modes = [ImageMapper.SlicingMode.I, ImageMapper.SlicingMode.J, ImageMapper.SlicingMode.K];
    R.ffaces = R.ffaces || [];
    let off = 0;
    defs.forEach(([ax, side, dims], k) => {
      const n = dims[0] * dims[1] * dims[2];
      if (!R.ffaces[k]) {
        const img = ImageData.newInstance();
        img.setDimensions(...dims);
        img.setSpacing(sp, sp, sp);
        const org = [f0, f0, f0];
        org[ax] = P[ax][side];
        img.setOrigin(...org);
        const mapper = ImageMapper.newInstance();
        mapper.setInputData(img);
        mapper.setSlicingMode(modes[ax]);
        mapper.setSlice(0);
        const actor = ImageSlice.newInstance();
        actor.setMapper(mapper);
        R.renderer.addActor(actor);
        R.ffaces[k] = { img, mapper, actor };
      }
      let vals = data.slice(off, off + n);
      off += n;
      if (S.field && S.log) for (let i = 0; i < n; i++) vals[i] = Math.log10(Math.max(vals[i], 1e-300));
      const f = R.ffaces[k];
      f.img.getPointData().setScalars(DataArray.newInstance({ name: "v", numberOfComponents: 1, values: vals }));
      f.img.modified();
      styleSlice(f.actor, ctf, true);
    });
    return true;
  }
  // The section planes of the structure (no field shown) at the solver's
  // resolution; with a field, the browser volume's slices stay.
  // The Voxels style: GPU volume rendering of every voxel the solver used,
  // nearest neighbour and shaded. Coloured by phase, or by the field (its
  // 8-bit copy as a second component; a shader mix takes the colour from the
  // field and the opacity and the shading normal from the structure).
  async function buildVoxVolume() {
    const vv = S.meta.voxel_volume;
    const want = !!vv && S.particles && S.surfStyle === "voxels";
    if (!want) { if (R.vvol) R.vvol.actor.setVisibility(false); return; }
    const lab = await loadVoxVolume();
    if (!lab) return;
    const fv = S.field && S.colorParticles ? await loadVoxFieldVolume(S.field) : null;
    const ImageData = ns("Common.DataModel.vtkImageData"), DataArray = ns("Common.Core.vtkDataArray");
    const Volume = ns("Rendering.Core.vtkVolume"), VolumeMapper = ns("Rendering.Core.vtkVolumeMapper");
    const PWF = ns("Common.DataModel.vtkPiecewiseFunction"), CTF = ns("Rendering.Core.vtkColorTransferFunction");
    const [nx, ny, nz] = vv.dims, h = vv.spacing_um;
    if (!R.vvol) {
      const img = ImageData.newInstance();
      img.setDimensions(nx + 2, ny + 2, nz + 2);
      img.setSpacing(h, h, h);
      img.setOrigin(-0.5 * h, -0.5 * h, -0.5 * h);
      const mapper = VolumeMapper.newInstance();
      mapper.setInputData(img);
      mapper.setSampleDistance(0.5 * h);
      if (mapper.setMaximumSamplesPerRay) mapper.setMaximumSamplesPerRay(4 * Math.max(nx, ny, nz) + 64);
      const actor = Volume.newInstance();
      actor.setMapper(mapper);
      R.renderer.addVolume(actor);
      R.vvol = { img, mapper, actor, key: null };
    }
    const { img, mapper, actor } = R.vvol;
    const key = fv ? fv.key : "";
    const pr = actor.getProperty();
    if (R.vvol.key !== key) {
      if (fv) {
        const two = new Uint8Array(2 * lab.length);
        for (let i = 0; i < lab.length; i++) { two[2 * i] = fv.data[i]; two[2 * i + 1] = lab[i]; }
        img.getPointData().setScalars(DataArray.newInstance({ name: "v", numberOfComponents: 2, values: two }));
        pr.setIndependentComponents(true);
        pr.setColorMixPreset(3);
        mapper.getViewSpecificProperties().OpenGL = { ShaderReplacements: [{ shaderType: "Fragment", originalValue: "//VTK::CustomColorMix",
          replaceFirst: false, replaceAll: false, replacementValue: `
    mat4 normalMat = computeMat4Normal(posIS, tValue);
    float op = getOpacityFromTexture(tValue[1], 1, volume.transferFunctionsSampleHeight[1]);
    vec3 col = getColorFromTexture(tValue[0], 0, volume.transferFunctionsSampleHeight[0]);
    #if vtkNumberOfLights > 0
      col = applyLighting(col, normalMat[1]);
    #endif
    return vec4(col, op);` }] };
      } else {
        img.getPointData().setScalars(DataArray.newInstance({ name: "v", numberOfComponents: 1, values: lab }));
        pr.setIndependentComponents(true);
        pr.setColorMixPreset(0);
        mapper.getViewSpecificProperties().OpenGL = { ShaderReplacements: [] };
      }
      img.modified();
      R.vvol.key = key;
    }
    // opacity by phase: the matrix and hidden phases transparent
    const op = PWF.newInstance(), lc = CTF.newInstance();
    S.meta.labels.forEach((l) => {
      const drawn = !(l.kind === "matrix" || l.kind === "interphase") && !S.hidden[l.value];
      op.addPoint(l.value - 0.45, drawn ? S.opacity : 0); op.addPoint(l.value + 0.45, drawn ? S.opacity : 0);
      const c = hex2rgb(S.colours[l.value] || l.color);
      lc.addRGBPoint(l.value - 0.45, ...c); lc.addRGBPoint(l.value + 0.45, ...c);
    });
    if (fv) {
      // field codes 1..255 span the run's 16-bit range; colour them through
      // the current map and display range
      const f = fieldMeta(), [lo, hi] = currentRange(), cm = CMAPS[S.cmap] || CMAPS.jet, fq = fv.full;
      const fc = CTF.newInstance();
      fc.addRGBPoint(0, 0.78, 0.78, 0.78);
      for (let c = 1; c <= 255; c += 2) {
        let v = fq.lo + ((c - 1) / 254) * (fq.hi - fq.lo);
        if (fq.log) v = 10 ** v;
        fc.addRGBPoint(c, ...cm(mapValue(v, lo, hi)));
      }
      const zero = PWF.newInstance(); zero.addPoint(0, 0); zero.addPoint(255, 0);
      pr.setRGBTransferFunction(0, fc); pr.setScalarOpacity(0, zero);
      pr.setRGBTransferFunction(1, lc); pr.setScalarOpacity(1, op);
      pr.setScalarOpacityUnitDistance(1, 0.1 * h);
    } else {
      pr.setRGBTransferFunction(0, lc); pr.setScalarOpacity(0, op);
    }
    pr.setScalarOpacityUnitDistance(0, 0.1 * h);
    pr.setInterpolationTypeToNearest();
    pr.setShade(true);
    const L = S.light;
    pr.setAmbient(Math.max(L.ambient, fv ? 0.4 : 0.25)); pr.setDiffuse(fv ? Math.min(L.diffuse, 0.7) : L.diffuse); pr.setSpecular(Math.min(L.specular, 0.2)); pr.setSpecularPower(L.power);
    mapper.removeAllClippingPlanes();
    if (S.clip) mapper.addClippingPlane(ns("Common.DataModel.vtkPlane").newInstance({ origin: [0, 0, (S.slices.z + 0.5) * S.meta.spacing_um], normal: [0, 0, -1] }));
    actor.setVisibility(true);
  }

  async function fineSections(ctf) {
    const ImageData = ns("Common.DataModel.vtkImageData"), DataArray = ns("Common.Core.vtkDataArray");
    const ImageMapper = ns("Rendering.Core.vtkImageMapper"), ImageSlice = ns("Rendering.Core.vtkImageSlice");
    R.fsec = R.fsec || {};
    const h = S.meta.voxel_um || S.meta.spacing_um;
    for (const ax of ["x", "y", "z"]) {
      const want = !!S.show[ax];
      const sl = !want ? null : (S.field ? await fieldSlice(S.field, ax, S.slices[ax]) : await labelSlice(ax, S.slices[ax]));
      const cur = R.fsec[ax];
      if (!sl) { if (cur) cur.actor.setVisibility(false); continue; }
      const i = "xyz".indexOf(ax), k = Math.min(S.slices[ax] * sl.st + (sl.st >> 1), S.meta.full_shape[i] - 1);
      const dims = [sl.w, sl.h];
      dims.splice(i, 0, 1);
      if (!cur || cur.dims.join() !== dims.join()) {
        if (cur) R.renderer.removeActor(cur.actor);
        const img = ImageData.newInstance();
        img.setDimensions(...dims);
        img.setSpacing(h, h, h);
        const mapper = ImageMapper.newInstance();
        mapper.setInputData(img);
        mapper.setSlicingMode([ImageMapper.SlicingMode.I, ImageMapper.SlicingMode.J, ImageMapper.SlicingMode.K][i]);
        mapper.setSlice(0);
        const actor = ImageSlice.newInstance();
        actor.setMapper(mapper);
        R.renderer.addActor(actor);
        R.fsec[ax] = { img, mapper, actor, dims };
      }
      const f = R.fsec[ax], org = [0.5 * h, 0.5 * h, 0.5 * h];
      org[i] = (k + 0.5) * h;
      f.img.setOrigin(...org);
      const vals = new Float32Array(sl.data.length);
      if (S.field && S.log) for (let j = 0; j < vals.length; j++) vals[j] = Math.log10(Math.max(sl.data[j], 1e-300));
      else for (let j = 0; j < vals.length; j++) vals[j] = sl.data[j];
      f.img.getPointData().setScalars(DataArray.newInstance({ name: "v", numberOfComponents: 1, values: vals }));
      f.img.modified();
      styleSlice(f.actor, ctf, true);
      if (R.slices[ax]) R.slices[ax].actor.setVisibility(false);
    }
  }
  // Where the particle surfaces (or voxel faces) show only a corner of a
  // large RVE, that region is outlined in orange.
  function buildRegion() {
    const PD = ns("Common.DataModel.vtkPolyData"), Mapper = ns("Rendering.Core.vtkMapper"), Actor = ns("Rendering.Core.vtkActor");
    const vox = S.surfStyle === "voxels" && (S.meta.voxel_style || S.meta.voxel_volume);
    const reg = vox ? (S.meta.voxel_volume ? null : S.meta.voxel_style.region_um) : (S.meta.surface_region || {}).um;
    if (R.region) { R.renderer.removeActor(R.region); R.region = null; }
    if (!reg || !S.particles) return;
    const [X, Y, Z] = reg;
    const P = new Float32Array([0, 0, 0, X, 0, 0, X, Y, 0, 0, Y, 0, 0, 0, Z, X, 0, Z, X, Y, Z, 0, Y, Z]);
    const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const lines = new Uint32Array(E.length * 3);
    E.forEach((e, i) => { lines[3 * i] = 2; lines[3 * i + 1] = e[0]; lines[3 * i + 2] = e[1]; });
    const box = PD.newInstance();
    box.getPoints().setData(P, 3);
    box.getLines().setData(lines);
    const m = Mapper.newInstance();
    m.setInputData(box);
    const a = Actor.newInstance();
    a.setMapper(m);
    a.getProperty().setLineWidth(2.2);
    a.getProperty().setColor(0.89, 0.38, 0.04);
    R.renderer.addActor(a);
    R.region = a;
  }
  V.regionNote = function () {
    if (!S.meta) return "";
    const fmtN = (n) => Number(n).toLocaleString("en-US");
    const dims = (r) => r.map((v) => fmtLen(v)).join(" × ");
    if (S.surfStyle === "voxels" && !S.meta.voxel_volume && S.meta.voxel_style && S.meta.voxel_style.region_um) {
      const vs = S.meta.voxel_style;
      return `The whole RVE has ${fmtN(vs.n_faces_total)} voxel faces, more than the 3D view draws: it shows the solver's own voxels in the corner region ${dims(vs.region_um)} µm (outlined). Smooth draws the whole RVE; 2D sections, the TIFF stack and every result use all of it.`;
    }
    const sr = S.meta.surface_region;
    if (S.surfStyle !== "voxels" && sr) {
      return `The RVE holds ${fmtN(sr.n_total)} particles, more than the smooth surfaces draw in full detail: they show the corner region ${dims(sr.um)} µm (${fmtN(sr.n_particles)} particles, outlined).` + (S.meta.voxel_volume ? " Voxels draws every voxel of the whole RVE." : " 2D sections, the TIFF stack and every result use the whole RVE.");
    }
    return "";
  };

  async function buildNetwork() {
    if (!R) return;
    if (!S.meta.network || !V.can.network()) { if (R.net) { R.net.balls.setVisibility(false); R.net.sticks.setVisibility(false); } return; }
    if (!S.networkOn && !R.net) return;
    const nw = await loadNetwork();
    if (!nw) return;
    const { renderer } = R;
    if (!R.net) {
      const PD = ns("Common.DataModel.vtkPolyData"), DataArray = ns("Common.Core.vtkDataArray");
      const Glyph = ns("Rendering.Core.vtkGlyph3DMapper"), Sphere = ns("Filters.Sources.vtkSphereSource");
      const Mapper = ns("Rendering.Core.vtkMapper"), Actor = ns("Rendering.Core.vtkActor");
      const n = nw.pores.xyz_um.length;
      const pts = new Float32Array(3 * n), dia = new Float32Array(n);
      nw.pores.xyz_um.forEach((p, i) => { pts[3 * i] = p[0]; pts[3 * i + 1] = p[1]; pts[3 * i + 2] = p[2]; dia[i] = nw.pores.diameter_um[i]; });
      const pd = PD.newInstance();
      pd.getPoints().setData(pts, 3);
      const darr = DataArray.newInstance({ name: "diam", numberOfComponents: 1, values: dia });
      pd.getPointData().addArray(darr);
      pd.getPointData().setScalars(darr);
      const sphere = Sphere.newInstance({ radius: 0.5, thetaResolution: 18, phiResolution: 12 });
      const gm = Glyph.newInstance();
      gm.setInputData(pd, 0);
      gm.setInputConnection(sphere.getOutputPort(), 1);
      gm.setScaleArray("diam");
      gm.setScaleMode(Glyph.ScaleModes.SCALE_BY_MAGNITUDE);
      gm.setScaleFactor(1.0);
      const lo = S.meta.network.d_min, hi = Math.max(S.meta.network.d_max, lo * 1.001 + 1e-9);
      gm.setLookupTable(makeCTF(lo, hi, "jet", false));
      gm.setUseLookupTableScalarRange(true);
      gm.setScalarVisibility(true);
      const balls = Actor.newInstance();
      balls.setMapper(gm);
      balls.getProperty().setSpecular(0.3);
      balls.getProperty().setSpecularPower(25);
      renderer.addActor(balls);
      const [nx, ny, nz] = S.meta.shape, s = S.meta.spacing_um, lim = 0.5 * Math.min(nx, ny, nz) * s;
      const conns = nw.throats.conns.filter(([a, b]) => {
        const pa = nw.pores.xyz_um[a], pb = nw.pores.xyz_um[b];
        return Math.hypot(pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]) < lim;
      });
      const lines = new Uint32Array(conns.length * 3);
      conns.forEach(([a, b], i) => { lines[3 * i] = 2; lines[3 * i + 1] = a; lines[3 * i + 2] = b; });
      const lp = PD.newInstance();
      lp.getPoints().setData(pts, 3);
      lp.getLines().setData(lines);
      const lm = Mapper.newInstance();
      lm.setInputData(lp);
      lm.setScalarVisibility(false);
      const sticks = Actor.newInstance();
      sticks.setMapper(lm);
      sticks.getProperty().setColor(0.36, 0.4, 0.45);
      sticks.getProperty().setLineWidth(2);
      renderer.addActor(sticks);
      R.net = { balls, sticks, lo, hi };
    }
    R.net.balls.setVisibility(S.networkOn);
    R.net.sticks.setVisibility(S.networkOn);
  }

  /* Arrows. A contour says how strong the field is; the arrows say where the
     heat, the current or the flow goes, which is the half a colour map cannot
     show. Length follows the magnitude, so weak regions stay quiet. */
  async function buildVectors(ctf) {
    if (!R) return;
    const vm = S.vectors ? vectorFor(S.field) : null;
    if (!vm || !V.can.arrows()) { if (R.vec) R.vec.actor.setVisibility(false); return; }
    const raw = await loadVector(vm.key);
    if (!raw) { if (R.vec) R.vec.actor.setVisibility(false); return; }
    const [nx, ny, nz] = S.meta.shape, s = S.meta.spacing_um;
    const every = vecEvery(Math.max(nx, ny, nz)), o = every >> 1;
    const ref = vm.p99 || vm.max || 1, cut = 0.02 * ref;
    const px = [], pv = [], pm = [];
    for (let k = o; k < nz; k += every) for (let j = o; j < ny; j += every) for (let i = o; i < nx; i += every) {
      const idx = i + nx * (j + ny * k), a = raw[3 * idx], b = raw[3 * idx + 1], c = raw[3 * idx + 2];
      const m = Math.sqrt(a * a + b * b + c * c);
      if (!(m > cut)) continue;
      px.push((i + 0.5) * s, (j + 0.5) * s, (k + 0.5) * s); pv.push(a, b, c); pm.push(m);
    }
    if (!px.length) { if (R.vec) R.vec.actor.setVisibility(false); return; }
    const PD = ns("Common.DataModel.vtkPolyData"), DataArray = ns("Common.Core.vtkDataArray");
    const Glyph = ns("Rendering.Core.vtkGlyph3DMapper"), Arrow = ns("Filters.Sources.vtkArrowSource"), Actor = ns("Rendering.Core.vtkActor");
    const same = vm.field === S.field;
    const pd = PD.newInstance();
    pd.getPoints().setData(Float32Array.from(px), 3);
    pd.getPointData().addArray(DataArray.newInstance({ name: "vec", numberOfComponents: 3, values: Float32Array.from(pv) }));
    // scale always on the true magnitude, colour on the scale of the view, so
    // a logarithmic field keeps one colour meaning everywhere
    pd.getPointData().addArray(DataArray.newInstance({ name: "mag", numberOfComponents: 1, values: Float32Array.from(pm) }));
    pd.getPointData().addArray(DataArray.newInstance({ name: "len", numberOfComponents: 1, values: Float32Array.from(pm.map((v) => vecLen(v, ref))) }));
    const col = DataArray.newInstance({ name: "col", numberOfComponents: 1,
      values: Float32Array.from(same && S.log ? pm.map((v) => Math.log10(Math.max(v, 1e-300))) : pm) });
    pd.getPointData().addArray(col);
    pd.getPointData().setScalars(col);
    if (!R.vec) {
      const src = Arrow.newInstance({ tipResolution: 12, tipRadius: 0.13, tipLength: 0.32, shaftResolution: 12, shaftRadius: 0.042 });
      const gm = Glyph.newInstance();
      gm.setInputConnection(src.getOutputPort(), 1);
      gm.setOrientationArray("vec");
      if (Glyph.OrientationModes) gm.setOrientationMode(Glyph.OrientationModes.DIRECTION);
      gm.setScaleArray("len");
      gm.setScaleMode(Glyph.ScaleModes.SCALE_BY_MAGNITUDE);
      const actor = Actor.newInstance();
      actor.setMapper(gm);
      actor.getProperty().setSpecular(0.25);
      R.renderer.addActor(actor);
      R.vec = { gm, actor, src };
    }
    R.vec.gm.setInputData(pd, 0);
    R.vec.gm.setScaleFactor(0.9 * every * s * (S.vecScale || 1));
    R.vec.gm.setLookupTable(same ? ctf : makeCTF(0, Math.max(ref, 1e-30), S.cmap, false));
    R.vec.gm.setUseLookupTableScalarRange(true);
    R.vec.gm.setScalarVisibility(true);
    R.vec.actor.setVisibility(true);
  }

  function stopAnim() { if (R && R.animId) { cancelAnimationFrame(R.animId); R.animId = null; } }
  /* The particle is drawn at the size the analysis reports and walks the route
     it found, so the answer to "what passes through, and where" is visible
     without reading a table. */
  function startAnim() {
    if (!R || !R.path || !(R.path.particle || R.path.tracer) || !S.pathAnim || S.mode === "2d" || R.animId) return;
    let last = -1e9;
    if (R.path.tracer) R.path.tracer.actor.setVisibility(true);
    const tick = (ts) => {
      if (!R || !R.path || !(R.path.particle || R.path.tracer) || !S.pathAnim || S.mode === "2d") { if (R) R.animId = null; return; }
      R.animId = requestAnimationFrame(tick);
      if (ts - last < 40) return;                       // ~25 fps is plenty for moving spheres
      last = ts;
      if (R.path.particle) {
        const pts = R.path.first, n = pts.length / 3;
        if (n >= 2) {
          const f = ((ts / 7000) % 1) * (n - 1), i = Math.floor(f), a = f - i, j = Math.min(i + 1, n - 1);
          R.path.particle.setPosition(pts[3 * i] + (pts[3 * j] - pts[3 * i]) * a,
                                      pts[3 * i + 1] + (pts[3 * j + 1] - pts[3 * i + 1]) * a,
                                      pts[3 * i + 2] + (pts[3 * j + 2] - pts[3 * i + 2]) * a);
        }
      }
      if (R.path.tracer) moveTracers(R.path.tracer, ts / 1000);
      R.rw.render();
    };
    R.animId = requestAnimationFrame(tick);
  }
  /* Tracers on field lines: each line carries a few dots that travel along it
     at the local speed of the solved field (the time to cross a segment is its
     length over the mean speed of its ends), so slow layers creep, the resin
     squeezed between two fillers hurries, and a heat-flux line shows where the
     heat moves fast. The time scale is chosen so that a line of median speed
     is crossed in about eight seconds. */
  function makeTracer(lines, radius, ns_) {
    const PD = ns_("Common.DataModel.vtkPolyData"), Glyph = ns_("Rendering.Core.vtkGlyph3DMapper");
    const Sphere = ns_("Filters.Sources.vtkSphereSource"), Actor = ns_("Rendering.Core.vtkActor");
    if (!PD || !Glyph || !Sphere) return null;
    const L = [];
    let vmax = 0;
    lines.forEach((l) => (l.speed || []).forEach((v) => { if (v > vmax) vmax = v; }));
    const floor = Math.max(vmax * 2e-3, 1e-30);
    lines.forEach((l) => {
      const p = l.points_um, sp = l.speed || [];
      if (!p || p.length < 2) return;
      const n = p.length, pts = new Float32Array(3 * n), t = new Float64Array(n);
      for (let k = 0; k < n; k++) { pts[3 * k] = p[k][0]; pts[3 * k + 1] = p[k][1]; pts[3 * k + 2] = p[k][2]; }
      for (let k = 1; k < n; k++) {
        const ds = Math.hypot(p[k][0] - p[k - 1][0], p[k][1] - p[k - 1][1], p[k][2] - p[k - 1][2]);
        const v = sp.length === n ? Math.max(0.5 * (sp[k] + sp[k - 1]), floor) : 1;
        t[k] = t[k - 1] + ds / v;
      }
      if (t[n - 1] > 0) L.push({ pts, t, T: t[n - 1] });
    });
    if (!L.length) return null;
    const Ts = L.map((l) => l.T).sort((a, b) => a - b);
    const rate = Ts[Math.floor(Ts.length / 2)] / 8.0;            // model time per second of animation
    const K = L.length > 60 ? 2 : 3;
    const nd = L.length * K;
    const tp = new Float32Array(3 * nd);
    const pd = PD.newInstance();
    pd.getPoints().setData(tp, 3);
    const sphere = Sphere.newInstance({ radius, thetaResolution: 12, phiResolution: 10 });
    const gm = Glyph.newInstance();
    gm.setInputData(pd, 0);
    gm.setInputConnection(sphere.getOutputPort(), 1);
    gm.setScalarVisibility(false);
    const actor = Actor.newInstance();
    actor.setMapper(gm);
    actor.getProperty().setColor(0.13, 0.15, 0.18);
    actor.getProperty().setSpecular(0.5);
    actor.getProperty().setSpecularPower(40);
    actor.getProperty().setAmbient(0.25);
    return { actor, pd, tp, L, K, rate };
  }
  function moveTracers(tr, sec) {
    const { L, K, tp, rate } = tr;
    let o = 0;
    for (let i = 0; i < L.length; i++) {
      const l = L[i];
      for (let k = 0; k < K; k++) {
        let tau = (sec * rate + (k / K + 0.37 * i) * l.T) % l.T;
        // the segment the dot is on: bisection on the arrival times
        let lo = 0, hi = l.t.length - 1;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; if (l.t[m] <= tau) lo = m; else hi = m; }
        const a = (tau - l.t[lo]) / Math.max(l.t[hi] - l.t[lo], 1e-30);
        const p = l.pts;
        tp[o++] = p[3 * lo] + (p[3 * hi] - p[3 * lo]) * a;
        tp[o++] = p[3 * lo + 1] + (p[3 * hi + 1] - p[3 * lo + 1]) * a;
        tp[o++] = p[3 * lo + 2] + (p[3 * hi + 2] - p[3 * lo + 2]) * a;
      }
    }
    tr.pd.getPoints().modified();
    tr.pd.modified();
  }

  async function buildPaths() {
    if (!R) return;
    const pm = pathMeta();
    if (R.path) { R.path.tubes.setVisibility(false); if (R.path.particle) R.path.particle.setVisibility(false); if (R.path.tracer) R.path.tracer.actor.setVisibility(false); }
    if (!pm) { stopAnim(); return; }
    const data = await loadPathData(pm.key);
    if (!data || !data.lines || !data.lines.length) { stopAnim(); return; }
    const s = S.meta.spacing_um, span = Math.max(...S.meta.length_um);
    if (!R.path || R.path.key !== pm.key) {
      stopAnim();
      if (R.path) { R.renderer.removeActor(R.path.tubes); if (R.path.particle) R.renderer.removeActor(R.path.particle); if (R.path.tracer) R.renderer.removeActor(R.path.tracer.actor); }
      const PD = ns("Common.DataModel.vtkPolyData"), DataArray = ns("Common.Core.vtkDataArray");
      const Mapper = ns("Rendering.Core.vtkMapper"), Actor = ns("Rendering.Core.vtkActor");
      const Tube = ns("Filters.General.vtkTubeFilter"), Sphere = ns("Filters.Sources.vtkSphereSource");
      let n = 0;
      data.lines.forEach((l) => { n += l.points_um.length; });
      const pts = new Float32Array(3 * n), spd = new Float32Array(n);
      const cells = new Uint32Array(n + data.lines.length);
      let pi = 0, ci = 0, hasSpeed = false;
      data.lines.forEach((l) => {
        cells[ci++] = l.points_um.length;
        l.points_um.forEach((p, k) => {
          pts[3 * pi] = p[0]; pts[3 * pi + 1] = p[1]; pts[3 * pi + 2] = p[2];
          if (l.speed && l.speed.length === l.points_um.length) { spd[pi] = l.speed[k]; hasSpeed = true; }
          cells[ci++] = pi; pi++;
        });
      });
      const pd = PD.newInstance();
      pd.getPoints().setData(pts, 3);
      pd.getLines().setData(cells);
      if (hasSpeed) pd.getPointData().setScalars(DataArray.newInstance({ name: "speed", numberOfComponents: 1, values: spd }));
      const mapper = Mapper.newInstance();
      let tubed = false;
      if (Tube) {
        try {
          const tubeR = Math.max(0.35 * s, 0.004 * span) * (pm.kind === "particle" ? 1.4 : 1.0);
          const tf = Tube.newInstance({ radius: tubeR, numberOfSides: 12, capping: true });
          tf.setInputData(pd);
          mapper.setInputConnection(tf.getOutputPort());
          tubed = true;
        } catch (e) { console.warn("tube filter", e); }
      }
      if (!tubed) mapper.setInputData(pd);
      const tubes = Actor.newInstance();
      tubes.setMapper(mapper);
      tubes.getProperty().setLineWidth(3);
      if (hasSpeed) {
        const lo = pm.speed_min || 0, hi = Math.max(pm.speed_max || 1, (pm.speed_min || 0) * 1.001 + 1e-30);
        mapper.setLookupTable(makeCTF(lo, hi, "turbo", false));
        mapper.setUseLookupTableScalarRange(true);
        mapper.setScalarVisibility(true);
        // the tube filter carries "speed" over as a plain array, not as the
        // active scalars: colour by it by name, or the tubes come out white
        mapper.setScalarModeToUsePointFieldData();
        mapper.setColorByArrayName("speed");
      } else {
        mapper.setScalarVisibility(false);
        tubes.getProperty().setColor(0.89, 0.38, 0.04);
      }
      R.renderer.addActor(tubes);
      let particle = null;
      if (pm.diameter_um && Sphere) {
        // a 10 nm particle beside a 15 µm RVE would be a single pixel: it is
        // drawn at a minimum visible size instead, and the read-out says so
        const drawR = Math.max(0.5 * pm.diameter_um, 0.012 * span);
        const sp = Sphere.newInstance({ radius: drawR, thetaResolution: 28, phiResolution: 20 });
        const m2 = Mapper.newInstance();
        m2.setInputConnection(sp.getOutputPort());
        m2.setScalarVisibility(false);
        particle = Actor.newInstance();
        particle.setMapper(m2);
        particle.getProperty().setColor(0.04, 0.38, 0.77);
        particle.getProperty().setOpacity(0.78);
        particle.getProperty().setSpecular(0.4);
        particle.getProperty().setSpecularPower(30);
        R.renderer.addActor(particle);
      }
      const f0 = data.lines[0].points_um;
      const first = new Float32Array(3 * f0.length);
      f0.forEach((p, k) => { first[3 * k] = p[0]; first[3 * k + 1] = p[1]; first[3 * k + 2] = p[2]; });
      let tracer = null;
      if (pm.kind !== "particle") {
        try {
          tracer = makeTracer(data.lines, 2.8 * Math.max(0.35 * s, 0.004 * span), ns);
          if (tracer) { R.renderer.addActor(tracer.actor); tracer.actor.setVisibility(false); moveTracers(tracer, 0); }
        } catch (e) { console.warn("tracers", e); tracer = null; }
      }
      R.path = { key: pm.key, tubes, particle, first, tracer };
    }
    R.path.tubes.setVisibility(true);
    // the dots are wider than the tubes and dark, so they read against both the
    // coloured lines and the white background without fading the lines
    if (R.path.tracer) R.path.tracer.actor.setVisibility(!!S.pathAnim && S.mode !== "2d");
    if (R.path.particle) {
      R.path.particle.setVisibility(true);
      if (!S.pathAnim) R.path.particle.setPosition(R.path.first[0], R.path.first[1], R.path.first[2]);
    }
    if (S.pathAnim) startAnim(); else stopAnim();
  }

  function buildVolume(ctf, lo, hi) {
    if (!R || !V.can.volume()) return;
    const want = S.volume && !!S.field;
    if (!want && !R.volume) return;
    const s = S.meta.spacing_um;
    if (!R.volume) {
      const Volume = ns("Rendering.Core.vtkVolume"), VolumeMapper = ns("Rendering.Core.vtkVolumeMapper");
      const mapper = VolumeMapper.newInstance();
      mapper.setInputData(R.image);
      mapper.setSampleDistance(0.7 * s);
      const actor = Volume.newInstance();
      actor.setMapper(mapper);
      R.renderer.addVolume(actor);
      R.volume = { actor, mapper };
    }
    const PWF = ns("Common.DataModel.vtkPiecewiseFunction");
    const pwf = PWF.newInstance();
    const a = S.log ? Math.log10(lo) : lo, b = S.log ? Math.log10(hi) : hi;
    const k = Math.max(0.05, S.opacity);
    pwf.addPoint(a, 0.0);
    pwf.addPoint(a + 0.35 * (b - a), 0.05 * k);
    pwf.addPoint(a + 0.75 * (b - a), 0.35 * k);
    pwf.addPoint(b, 0.8 * k);
    const pr = R.volume.actor.getProperty();
    pr.setRGBTransferFunction(0, ctf);
    pr.setScalarOpacity(0, pwf);
    pr.setScalarOpacityUnitDistance(0, 3 * s);
    pr.setInterpolationTypeToLinear();
    // shading the volume is what stops it looking like flat fog
    pr.setShade(!!S.light.shade);
    if (S.light.shade) {
      pr.setAmbient(S.light.ambient); pr.setDiffuse(S.light.diffuse);
      pr.setSpecular(S.light.specular);
    }
    R.volume.actor.setVisibility(want);
  }

  function camera(which) {
    if (!R || !S.meta) return;
    const { renderer, rw } = R;
    const [nx, ny, nz] = S.meta.shape, s = S.meta.spacing_um;
    const c = [nx * s / 2, ny * s / 2, nz * s / 2], d = Math.max(nx, ny, nz) * s * 2.6;
    const cam = renderer.getActiveCamera();
    const dirs = { iso: [1, -1.25, 0.9], x: [1, 0, 0], y: [0, -1, 0], z: [0, 0, 1] };
    const dir = dirs[which] || dirs.iso, len = Math.hypot(...dir);
    cam.setFocalPoint(...c);
    cam.setPosition(c[0] + dir[0] / len * d, c[1] + dir[1] / len * d, c[2] + dir[2] / len * d);
    cam.setViewUp(...(which === "z" ? [0, 1, 0] : [0, 0, 1]));
    renderer.resetCamera();
    cam.zoom(which === "iso" || which === "reset" ? 1.0 : 1.1);
    rw.render();
  }

  function cameraState() {
    if (!R) return null;
    const cam = R.renderer.getActiveCamera();
    return { position: cam.getPosition(), focal_point: cam.getFocalPoint(), view_up: cam.getViewUp(),
      parallel: cam.getParallelProjection(), parallel_scale: cam.getParallelScale(), view_angle: cam.getViewAngle() };
  }

  /* ------------------------------------------------------------ overlays (DOM) */
  function gradient(cm, dir) {
    const stops = [];
    for (let i = 0; i <= 24; i++) { const c = cm(dir === "down" ? 1 - i / 24 : i / 24).map((v) => Math.round(v * 255)); stops.push(`rgb(${c}) ${(i * 100) / 24}%`); }
    return stops.join(",");
  }
  function colorbarHTML(title, unit, lo, hi, cmapName, logScale, height = 210) {
    const cm = CMAPS[cmapName] || CMAPS.jet;
    const ticks = [];
    for (let i = 0; i <= 5; i++) {
      const t = i / 5;
      const v = logScale ? 10 ** (Math.log10(lo) + t * (Math.log10(hi) - Math.log10(lo))) : lo + t * (hi - lo);
      ticks.push(`<div style="position:absolute;left:24px;bottom:calc(${t * 100}% - 7px);font:10.5px Consolas,monospace;color:#1f2328;white-space:nowrap"><span style="display:inline-block;width:5px;height:1px;background:#6e7781;vertical-align:middle;margin-right:3px"></span>${fmt(v)}</div>`);
    }
    return `<div class="cbox"><div class="cbar-title">${title}${unit && unit !== "-" ? ` [${unit}]` : ""}</div>
      <div style="position:relative;height:${height}px;width:96px"><div style="position:absolute;left:0;top:0;bottom:0;width:17px;border:1px solid #afb8c1;border-radius:2px;background:linear-gradient(to bottom,${gradient(cm, "down")})"></div>${ticks.join("")}</div>
      ${logScale ? '<div class="hint" style="margin-top:3px">log scale</div>' : ""}</div>`;
  }
  V.updateOverlays = function () {
    const el = document.getElementById("vCbar"), hud = document.getElementById("vHud"), foot = document.getElementById("vFoot");
    if (!el || !S.meta) return;
    // in the quad layout every 2D section carries its own colour bar
    el.style.display = S.mode === "3d" ? "" : "none";
    hud.style.display = S.mode === "3d" ? "" : "none";
    const barH = S.mode === "quad" ? 120 : 210;
    const f = fieldMeta();
    const m = S.meta;
    const cube = m.length_um.every((v) => v === m.length_um[0]);
    foot.textContent = `${m.full_shape.join(" × ")} voxels · ${fmt(m.voxel_um)} µm · ${cube ? `RVE ${fmt(m.length_um[0])} µm` : `${m.length_um.map(fmt).join(" × ")} µm`}`;
    if (f) {
      const [lo, hi] = currentRange();
      el.innerHTML = colorbarHTML(f.label, f.unit, lo, hi, S.cmap, S.log, barH);
      hud.innerHTML = `<b>${f.label}</b>${f.group} · min ${fmt(f.min)} · max ${fmt(f.max)} ${f.unit !== "-" ? f.unit : ""}`;
    } else if (S.networkOn && S.meta.network) {
      el.innerHTML = colorbarHTML("Pore diameter", "µm", S.meta.network.d_min, Math.max(S.meta.network.d_max, S.meta.network.d_min * 1.001), "jet", false, barH);
      hud.innerHTML = `<b>Pore network</b>${S.meta.network.n_pores} pores · ${S.meta.network.n_throats} throats`;
    } else {
      // mapped before filtering so the label keeps its own index, which is what
      // a colour override is keyed by
      el.innerHTML = `<div class="cbox">` + m.labels.map((l, i) => ({ l, i }))
        .filter(({ l }) => l.kind !== "interphase" && (l.vf > 0 || l.kind === "matrix"))
        .map(({ l, i }) => `<div class="lg"><i style="background:${S.colours[i] || l.color}"></i>${l.name}<span class="hint" style="margin-left:auto;padding-left:10px">${fmt(100 * l.vf)} %</span></div>`).join("") + `</div>`;
      hud.innerHTML = `<b>Structure</b>${m.labels.length} regions`;
    }
    const pm = pathMeta();
    // lines coloured by the magnitude along them get their own bar beside the
    // structure legend or the field's bar
    if (pm && pm.kind !== "particle" && pm.speed_max != null) {
      const lo = pm.speed_min || 0, hi = Math.max(pm.speed_max, lo * 1.001 + 1e-30);
      el.innerHTML += colorbarHTML(escHtml(pm.speed_label || "Magnitude along the lines"), pm.speed_unit || "-", lo, hi, "turbo", false, Math.round(0.62 * barH));
    }
    if (pm) {
      const sm = pm.summary || {};
      const bits = [`${pm.n_lines} ${pm.kind === "particle" ? "route" : "line"}${pm.n_lines === 1 ? "" : "s"}`];
      if (pm.diameter_um) {
        const enlarged = pm.diameter_um < 0.024 * Math.max(...S.meta.length_um);
        bits.push(`sphere ⌀ ${fmt(pm.diameter_um)} µm${enlarged ? " (drawn enlarged to stay visible)" : ""}`);
      }
      if (sm.tortuosity) bits.push(`τ ${fmt(sm.tortuosity)}`);
      if (sm.n_through !== undefined) bits.push(`${sm.n_through} of ${sm.n_lines} cross the sample`);
      hud.innerHTML += `<br><b>${escHtml(pm.label)}</b>${escHtml(bits.join(" · "))}`;
    }
  };
  function arrow2d(g, x0, y0, x1, y1, head) {
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0);
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - head * Math.cos(a - 0.42), y1 - head * Math.sin(a - 0.42));
    g.lineTo(x1 - head * Math.cos(a + 0.42), y1 - head * Math.sin(a + 0.42));
    g.closePath(); g.fill();
  }

  /* ------------------------------------------------------------ 2D sections */
  function Slice2D(pane, axis) {
    this.pane = pane; this.axis = axis; this.canvas = pane.querySelector("canvas");
    this.zoom = 1; this.panX = 0; this.panY = 0;
    const cv = this.canvas;
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const k = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      this.panX = mx - (mx - this.panX) * k; this.panY = my - (my - this.panY) * k; this.zoom *= k;
      this.draw();
    }, { passive: false });
    let drag = null;
    cv.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, px: this.panX, py: this.panY, moved: false }; });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      this.panX = drag.px + dx; this.panY = drag.py + dy;
      if (drag.moved) this.draw();
    });
    window.addEventListener("mouseup", (e) => {
      if (drag && !drag.moved && e.target === cv) this.click(e);
      drag = null;
    });
    cv.addEventListener("mousemove", (e) => this.hover(e));
    cv.addEventListener("mouseleave", () => { document.getElementById("vTip").hidden = true; });
    cv.addEventListener("dblclick", () => { this.zoom = 1; this.panX = 0; this.panY = 0; this.draw(); });
  }
  Slice2D.prototype.dims = function () {
    const [nx, ny, nz] = S.meta.shape;
    if (this.axis === "z") return { w: nx, h: ny, u: "x", v: "y" };
    if (this.axis === "y") return { w: nx, h: nz, u: "x", v: "z" };
    return { w: ny, h: nz, u: "y", v: "z" };
  };
  Slice2D.prototype.index = function (a, b) {
    const [nx, ny] = S.meta.shape, k = S.slices[this.axis];
    if (this.axis === "z") return a + nx * (b + ny * k);
    if (this.axis === "y") return a + nx * (k + ny * b);
    return k + nx * (a + ny * b);
  };
  Slice2D.prototype.geom = function () {
    const cv = this.canvas, W = cv.clientWidth, H = cv.clientHeight, d = this.dims();
    const big = S.mode === "2d";
    // 2D mode leaves room at the bottom for the section slider bar
    const margin = big ? 56 : 44, bar = 84, bottom = big ? 108 : 44;
    const base = Math.min((W - 2 * margin - bar) / d.w, (H - margin - bottom) / d.h);
    const px = base * this.zoom;
    const ox = margin + (W - bar - 2 * margin - d.w * base) / 2 + this.panX, oy = margin * 0.8 + (H - margin - bottom - d.h * base) / 2 + this.panY;
    return { W, H, d, px, ox, oy, bar };
  };
  Slice2D.prototype.draw = async function () {
    if (!S.meta || this.pane.offsetParent === null) return;
    const cv = this.canvas, dpr = window.devicePixelRatio || 1;
    const { W, H, d, px, ox, oy } = this.geom();
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#fff"; g.fillRect(0, 0, W, H);
    const arr = S.field ? await loadField(S.field) : null;
    const [lo, hi] = currentRange();
    const cm = CMAPS[S.cmap] || CMAPS.jet;
    const colors = S.meta.labels.map((l, i) => hex2rgb(S.colours[i] || l.color).map((c) => Math.round(c * 255)));
    // the structure at the solver's resolution where the browser volume is
    // coarser; the field stays on the volume's grid
    const fine = await labelSlice(this.axis, S.slices[this.axis]);
    this.fine = fine;
    const ffine = arr ? await fieldSlice(S.field, this.axis, S.slices[this.axis]) : null;
    this.ffine = ffine;
    const off = document.createElement("canvas");
    if (ffine) {
      off.width = ffine.w; off.height = ffine.h;
      const og = off.getContext("2d"), im = og.createImageData(ffine.w, ffine.h);
      for (let b = 0; b < ffine.h; b++) {
        for (let a = 0; a < ffine.w; a++) {
          const v = ffine.data[a + ffine.w * b], o = 4 * (a + (ffine.h - 1 - b) * ffine.w);
          const c = isFinite(v) ? cm(mapValue(v, lo, hi)).map((x) => Math.round(x * 255)) : [200, 200, 200];
          im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
        }
      }
      og.putImageData(im, 0, 0);
    } else if (!arr && fine) {
      off.width = fine.w; off.height = fine.h;
      const og = off.getContext("2d"), im = og.createImageData(fine.w, fine.h);
      for (let b = 0; b < fine.h; b++) {
        for (let a = 0; a < fine.w; a++) {
          const c = colors[fine.data[a + fine.w * b]] || [0, 0, 0], o = 4 * (a + (fine.h - 1 - b) * fine.w);
          im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
        }
      }
      og.putImageData(im, 0, 0);
    } else {
      off.width = d.w; off.height = d.h;
      const og = off.getContext("2d"), im = og.createImageData(d.w, d.h);
      for (let b = 0; b < d.h; b++) {
        for (let a = 0; a < d.w; a++) {
          const idx = this.index(a, b), o = 4 * (a + (d.h - 1 - b) * d.w);
          let c;
          if (arr) { const v = arr[idx]; c = isFinite(v) ? cm(mapValue(v, lo, hi)).map((x) => Math.round(x * 255)) : [200, 200, 200]; }
          else c = colors[S.labels[idx]] || [0, 0, 0];
          im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
        }
      }
      og.putImageData(im, 0, 0);
    }
    g.imageSmoothingEnabled = false;
    g.drawImage(off, ox, oy, d.w * px, d.h * px);
    if (S.outlines2d) {
      g.strokeStyle = "rgba(20,24,28,.85)";
      g.beginPath();
      if (fine) {
        const q = px / fine.st, L = fine.data, W = fine.w, H = fine.h;
        g.lineWidth = Math.max(0.5, Math.min(1.4, q / 6));
        for (let b = 0; b < H; b++) {
          for (let a = 0; a < W; a++) {
            const l = L[a + W * b];
            if (a + 1 < W && L[a + 1 + W * b] !== l) { const x = ox + (a + 1) * q, y = oy + (H - 1 - b) * q; g.moveTo(x, y); g.lineTo(x, y + q); }
            if (b + 1 < H && L[a + W * (b + 1)] !== l) { const x = ox + a * q, y = oy + (H - 1 - b) * q; g.moveTo(x, y); g.lineTo(x + q, y); }
          }
        }
      } else {
        g.lineWidth = Math.max(0.6, Math.min(1.4, px / 6));
        for (let b = 0; b < d.h; b++) {
          for (let a = 0; a < d.w; a++) {
            const l = S.labels[this.index(a, b)];
            if (a + 1 < d.w && S.labels[this.index(a + 1, b)] !== l) { const x = ox + (a + 1) * px, y = oy + (d.h - 1 - b) * px; g.moveTo(x, y); g.lineTo(x, y + px); }
            if (b + 1 < d.h && S.labels[this.index(a, b + 1)] !== l) { const x = ox + a * px, y = oy + (d.h - 1 - b) * px; g.moveTo(x, y); g.lineTo(x + px, y); }
          }
        }
      }
      g.stroke();
    }
    // the in-plane part of the vector field, on the same grid as the contour
    if (S.vectors) {
      const vm = vectorFor(S.field);
      const raw = vm ? await loadVector(vm.key) : null;
      if (raw) {
        const ci = { x: 0, y: 1, z: 2 }, cu = ci[d.u], cv2 = ci[d.v];
        const every = vecEvery(Math.max(d.w, d.h)), ref = vm.p99 || vm.max || 1;
        const len = every * px * 0.95 * (S.vecScale || 1);
        g.strokeStyle = "rgba(17,20,24,.85)"; g.fillStyle = "rgba(17,20,24,.85)";
        g.lineWidth = Math.max(0.7, Math.min(1.8, px / 9));
        for (let b = every >> 1; b < d.h; b += every) {
          for (let a = every >> 1; a < d.w; a += every) {
            const idx = this.index(a, b), vu = raw[3 * idx + cu], vv2 = raw[3 * idx + cv2];
            const m = Math.hypot(vu, vv2);
            if (!(m > 0.02 * ref)) continue;
            const L = len * Math.min(1, vecLen(m, ref));
            const x = ox + (a + 0.5) * px, y = oy + (d.h - 0.5 - b) * px;
            const ux = (vu / m) * L, uy = (-vv2 / m) * L;
            arrow2d(g, x - ux / 2, y - uy / 2, x + ux / 2, y + uy / 2, Math.max(2.5, 0.32 * L));
          }
        }
      }
    }
    if (S.mode !== "2d") {
      const other = { z: ["x", "y"], y: ["x", "z"], x: ["y", "z"] }[this.axis];
      g.setLineDash([5, 4]); g.lineWidth = 1.2; g.strokeStyle = "#e36209";
      const cu = ox + (S.slices[other[0]] + 0.5) * px, cvv = oy + (d.h - 0.5 - S.slices[other[1]]) * px;
      g.beginPath(); g.moveTo(cu, oy); g.lineTo(cu, oy + d.h * px); g.moveTo(ox, cvv); g.lineTo(ox + d.w * px, cvv); g.stroke();
      g.setLineDash([]);
    }
    g.strokeStyle = "#57606a"; g.lineWidth = 1; g.strokeRect(ox, oy, d.w * px, d.h * px);
    g.fillStyle = "#424a53"; g.font = "10.5px Segoe UI";
    const s = S.meta.spacing_um, Lw = d.w * s, Lh = d.h * s;
    const step = niceStep(Lw / 5);
    g.textAlign = "center";
    for (let t = 0; t <= Lw + 1e-9; t += step) { const x = ox + (t / s) * px; g.fillRect(x, oy + d.h * px, 1, 4); g.fillText(fmtLen(t), x, oy + d.h * px + 15); }
    g.textAlign = "right";
    for (let t = 0; t <= Lh + 1e-9; t += step) { const y = oy + d.h * px - (t / s) * px; g.fillRect(ox - 4, y, 4, 1); g.fillText(fmtLen(t), ox - 6, y + 4); }
    g.textAlign = "center"; g.font = "600 11px Segoe UI";
    g.fillText(`${d.u} (µm)`, ox + (d.w * px) / 2, oy + d.h * px + 31);
    g.save(); g.translate(Math.max(11, ox - 40), oy + (d.h * px) / 2); g.rotate(-Math.PI / 2); g.fillText(`${d.v} (µm)`, 0, 0); g.restore();
    const bx = W - 70, by = Math.max(oy, 34), bh = Math.min(d.h * px, H - by - 40);
    if (arr && bh > 40) {
      for (let i = 0; i < bh; i++) { const c = cm(1 - i / bh).map((x) => Math.round(x * 255)); g.fillStyle = `rgb(${c})`; g.fillRect(bx, by + i, 14, 1); }
      g.strokeStyle = "#afb8c1"; g.strokeRect(bx, by, 14, bh);
      g.fillStyle = "#1f2328"; g.textAlign = "left"; g.font = "10.5px Consolas";
      for (let i = 0; i <= 4; i++) {
        const t = i / 4, v = S.log ? 10 ** (Math.log10(lo) + t * (Math.log10(hi) - Math.log10(lo))) : lo + t * (hi - lo);
        g.fillText(fmt(v), bx + 18, by + bh - t * bh + 4);
      }
    } else if (!arr && S.mode === "2d") {
      let y = by + 4;
      g.font = "11px Segoe UI";
      S.meta.labels.map((l, i) => ({ l, i })).filter(({ l }) => l.kind !== "interphase").forEach(({ l, i }) => {
        g.fillStyle = S.colours[i] || l.color; g.fillRect(bx - 40, y, 12, 12); g.strokeStyle = "rgba(0,0,0,.25)"; g.strokeRect(bx - 40, y, 12, 12);
        g.fillStyle = "#1f2328"; g.fillText(l.name.length > 14 ? l.name.slice(0, 13) + "…" : l.name, bx - 23, y + 10); y += 18;
      });
    }
  };
  Slice2D.prototype.pick = function (e) {
    const r = this.canvas.getBoundingClientRect();
    const { d, px, ox, oy } = this.geom();
    const a = Math.floor((e.clientX - r.left - ox) / px), bb = Math.floor((e.clientY - r.top - oy) / px);
    const b = d.h - 1 - bb;
    if (a < 0 || b < 0 || a >= d.w || b >= d.h) return null;
    return { a, b, d, rx: e.clientX - r.left, ry: e.clientY - r.top };
  };
  Slice2D.prototype.hover = function (e) {
    const tip = document.getElementById("vTip");
    const p = this.pick(e);
    if (!p || !S.meta) { tip.hidden = true; return; }
    const idx = this.index(p.a, p.b), s = S.meta.spacing_um;
    let lab = S.meta.labels[S.labels[idx]];
    const fine = this.fine;
    if (fine) {
      const r = this.canvas.getBoundingClientRect(), g = this.geom(), q = g.px / fine.st;
      const fa = Math.floor((p.rx - g.ox) / q), fb = fine.h - 1 - Math.floor((p.ry - g.oy) / q);
      if (fa >= 0 && fb >= 0 && fa < fine.w && fb < fine.h) lab = S.meta.labels[fine.data[fa + fine.w * fb]] || lab;
    }
    let fval = null;
    const ff = this.ffine;
    if (ff) {
      const g = this.geom(), q = g.px / ff.st;
      const fa = Math.floor((p.rx - g.ox) / q), fb = ff.h - 1 - Math.floor((p.ry - g.oy) / q);
      if (fa >= 0 && fb >= 0 && fa < ff.w && fb < ff.h) fval = ff.data[fa + ff.w * fb];
    }
    const arr = S.field ? S.fields[S.field] : null, f = fieldMeta();
    const pos = { [p.d.u]: (p.a + 0.5) * s, [p.d.v]: (p.b + 0.5) * s, [this.axis]: (S.slices[this.axis] + 0.5) * s };
    tip.textContent = `x ${fmtLen(pos.x)}  y ${fmtLen(pos.y)}  z ${fmtLen(pos.z)} µm\n${lab ? lab.name : ""}` +
      (arr ? `\n${f.label}: ${fmt(fval != null ? fval : arr[idx])} ${f.unit}` : "");
    const stage = document.getElementById("vStage").getBoundingClientRect(), cr = this.canvas.getBoundingClientRect();
    tip.style.left = (cr.left - stage.left + p.rx + 14) + "px";
    tip.style.top = (cr.top - stage.top + p.ry + 10) + "px";
    tip.hidden = false;
  };
  Slice2D.prototype.click = function (e) {
    const p = this.pick(e);
    if (!p) return;
    S.slices[p.d.u] = p.a; S.slices[p.d.v] = p.b;
    V.syncSliders(); V.refresh(false);
  };
  function niceStep(v) { const e = Math.floor(Math.log10(v || 1)), f = v / 10 ** e; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * 10 ** e; }
  function fmtLen(v) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2); }
  V.fmtLen = fmtLen;

  /* ------------------------------------------------------------ public */
  V.state = S;
  V.hasData = () => !!S.meta;

  V.showEmpty = function (html) {
    const el = document.getElementById("vEmpty");
    if (!el) return;
    el.innerHTML = html;
    el.hidden = false;
    el.style.display = "flex";
    el.style.background = "#fff";
    el.style.zIndex = 15;
  };
  V.hideEmpty = function () { const el = document.getElementById("vEmpty"); if (el) { el.hidden = true; el.style.display = "none"; } };

  // what a view offers; it changes while a run adds analyses to the same view
  V.metaSig = (m) => (m ? [m.built || 0, (m.fields || []).map((f) => f.key).join(","),
    (m.vectors || []).map((v) => v.key).join(","), (m.paths || []).map((p) => p.key).join(","),
    (m.surfaces || []).length, m.network ? 1 : 0].join("|") : "");

  V.open = async function (jobId) {
    // meta.json is read every time. A run opened here before it finished - a
    // structure generated first and then continued with analyses, or a view
    // opened while the run went on - writes it again with every analysis; the
    // copy kept from the first opening left the later fields out of the list
    // (the viscosity fields, the last analysis of a run, never appeared).
    const r = await fetch(`/api/jobs/${jobId}/view/meta.json`, { cache: "no-store" });
    if (!r.ok) throw new Error("This run has no visualisation data");
    const meta = await r.json();
    if (S.jobId === jobId && S.meta && V.metaSig(S.meta) === V.metaSig(meta)) return S.meta;
    const keep = S.jobId === jobId ? S.field : null;
    // load the labels before publishing the new meta: a refresh running in
    // between would otherwise pair the new grid with old (or no) labels
    const labels = new Uint8Array(await fetchBin(`/api/jobs/${jobId}/view/labels.u8?v=${meta.built || 0}`));
    Object.assign(S, { jobId, meta, labels, fields: {}, surfaces: {}, surfLoaded: false, voxels: {}, voxLoaded: false,
                       faceData: {}, lslices: {}, fslices: {}, vvolLab: null, vvolField: null,
                       hidden: {}, network: null,
                       vecData: {}, pathData: {}, pathKey: "", vectors: false });
    meta.vectors = meta.vectors || [];
    meta.paths = meta.paths || [];
    S.slices = { x: Math.floor(meta.shape[0] / 2), y: Math.floor(meta.shape[1] / 2), z: Math.floor(meta.shape[2] / 2) };
    S.show = { x: false, y: false, z: false };
    // the same run read again keeps what was on display (a field, or the
    // phases alone)
    const kf = keep ? meta.fields.find((f) => f.key === keep) : null;
    const f0 = keep === "" ? null : (kf || meta.fields[0] || null);
    S.field = f0 ? f0.key : "";
    // the structure reads better than a box of coloured faces: surfaces are on
    // by default and carry the field, and the faces stay off unless there is
    // no surface to show the field on at all
    S.faces = !(meta.surfaces || []).length && !!S.field;
    S.log = f0 ? !!f0.log : false;
    S.range = "auto";
    S.volume = false;
    S.networkOn = !S.field && !!meta.network;
    S.particles = true;
    if (R) {
      stopAnim();
      if (R.vec) R.renderer.removeActor(R.vec.actor);
      if (R.path) { R.renderer.removeActor(R.path.tubes); if (R.path.particle) R.renderer.removeActor(R.path.particle); if (R.path.tracer) R.renderer.removeActor(R.path.tracer.actor); }
      Object.values(R.slices).forEach((o) => R.renderer.removeActor(o.actor));
      (R.faces || []).forEach((o) => R.renderer.removeActor(o.actor));
      (R.ffaces || []).forEach((o) => o && R.renderer.removeActor(o.actor));
      Object.values(R.fsec || {}).forEach((o) => R.renderer.removeActor(o.actor));
      if (R.region) R.renderer.removeActor(R.region);
      if (R.vvol) R.renderer.removeVolume(R.vvol.actor);
      Object.values(R.surf).forEach((o) => R.renderer.removeActor(o.a));
      Object.values(R.vox || {}).forEach((o) => R.renderer.removeActor(o.a));
      if (R.outline) R.renderer.removeActor(R.outline);
      if (R.net) { R.renderer.removeActor(R.net.balls); R.renderer.removeActor(R.net.sticks); }
      if (R.volume) R.renderer.removeVolume(R.volume.actor);
      if (R.cube) R.renderer.removeActor(R.cube);
      Object.assign(R, { image: null, slices: {}, faces: null, ffaces: [], fsec: {}, region: null, vvol: null,
                         surf: {}, vox: {}, outline: null, net: null, volume: null,
                         cube: null, vec: null, path: null, animId: null });
    }
    return meta;
  };

  V.mount = function () {
    const stage = document.getElementById("vStage");
    if (!views2d.z) document.querySelectorAll(".v2d").forEach((p) => { views2d[p.dataset.axis] = new Slice2D(p, p.dataset.axis); });
    stage.className = "vstage m" + S.mode;
    if (S.mode === "2d") stopAnim();
    document.querySelectorAll(".v2d").forEach((p) => p.classList.toggle("on2d", S.mode === "2d" && p.dataset.axis === (S.axis2d || "z")));
    if (S.meta && S.mode !== "2d") {
      if (initVtk(document.getElementById("v3d"))) R.grw.resize();
    }
  };

  V.refresh = async function (rebuild3d = true) {
    if (!S.meta || !S.labels || S.labels.length !== S.meta.shape[0] * S.meta.shape[1] * S.meta.shape[2]) return;
    V.hideEmpty();
    V.mount();
    if (S.mode !== "2d" && R) {
      if (rebuild3d || S.clip || !R.image) await build3D();
      else {
        ["x", "y", "z"].forEach((ax) => { if (R.slices[ax]) { R.slices[ax].mapper.setSlice(S.slices[ax]); R.slices[ax].actor.setVisibility(!!S.show[ax]); } });
        if (R.ctf) await fineSections(R.ctf);
        R.rw.render();
      }
    }
    Object.values(views2d).forEach((v) => v.draw());
    V.updateOverlays();
  };

  V.syncSliders = function () {
    if (!S.meta) return;
    ["x", "y", "z"].forEach((ax, i) => {
      const el = document.getElementById("vS" + ax);
      if (!el) return;
      el.min = 0; el.max = S.meta.shape[i] - 1; el.value = S.slices[ax];
      document.getElementById("vS" + ax + "v").textContent = fmtLen((S.slices[ax] + 0.5) * S.meta.spacing_um) + " µm";
    });
    const a = S.axis2d || "z", i = "xyz".indexOf(a), sl = document.getElementById("v2dSlice");
    if (sl) {
      sl.min = 0; sl.max = S.meta.shape[i] - 1; sl.value = S.slices[a];
      document.getElementById("v2dPos").textContent = `${a} = ${fmtLen((S.slices[a] + 0.5) * S.meta.spacing_um)} µm`;
    }
  };

  V.camera = camera;
  V.cameraState = cameraState;
  V.screenshot = async function () {
    if (!R) return null;
    const imgs = await R.rw.captureImages();
    return imgs && imgs[0];
  };
  /* An animated GIF of the 3D view as it is shown: one loop of the moving
     field lines or route when that animation is on, otherwise one turn around
     the RVE. The colour bar of the field is drawn into every frame. */
  V.recordGif = async function ({ filename, onProgress } = {}) {
    if (!R || !window.GifRec || S.mode === "2d") return null;
    const anim = !!(R.path && (R.path.tracer || R.path.particle) && S.pathAnim);
    stopAnim();
    const cam = R.renderer.getActiveCamera();
    const saved = { pos: cam.getPosition(), fp: cam.getFocalPoint(), up: cam.getViewUp() };
    const f = fieldMeta();
    const [lo, hi] = f ? currentRange() : [0, 1];
    const cm = CMAPS[S.cmap] || CMAPS.jet;
    const n = anim ? 80 : 60;
    // lines coloured along their length get their own bar, below the field's
    const pm = pathMeta();
    const lineBar = !!(pm && pm.kind !== "particle" && pm.speed_max != null);
    const overlay = (f || lineBar) ? (g, w, h) => {
      const bh = Math.round(h * (f && lineBar ? 0.3 : 0.42));
      let y = 52;
      if (f) {
        window.GifRec.colourBar(g, w - 96, y, bh, f.label || "", lo, hi, cm, fmt);
        y += bh + 48;
      }
      if (lineBar) {
        const llo = pm.speed_min || 0, lhi = Math.max(pm.speed_max, llo * 1.001 + 1e-30);
        window.GifRec.colourBar(g, w - 96, y, bh, pm.speed_label || "Along the lines", llo, lhi, CMAPS.turbo, fmt);
      }
    } : null;
    try {
      return await window.GifRec.record({
        count: n, delay: anim ? 100 : 70, filename: filename || "view.gif",
        frame: async (k) => {
          if (anim) {
            const t = k / n;
            if (R.path.particle) {
              const pts = R.path.first, np = pts.length / 3;
              if (np >= 2) {
                const fpos = t * (np - 1), i = Math.floor(fpos), a = fpos - i, j = Math.min(i + 1, np - 1);
                R.path.particle.setPosition(pts[3 * i] + (pts[3 * j] - pts[3 * i]) * a,
                                            pts[3 * i + 1] + (pts[3 * j + 1] - pts[3 * i + 1]) * a,
                                            pts[3 * i + 2] + (pts[3 * j + 2] - pts[3 * i + 2]) * a);
              }
            }
            if (R.path.tracer) moveTracers(R.path.tracer, 8.0 * t);
          } else {
            cam.azimuth(360 / n);
            R.renderer.resetCameraClippingRange();
          }
          const imgs = await R.rw.captureImages();
          return imgs && imgs[0];
        },
        overlay,
        onProgress,
      });
    } finally {
      if (!anim) {
        cam.setPosition(...saved.pos); cam.setFocalPoint(...saved.fp); cam.setViewUp(...saved.up);
        R.renderer.resetCameraClippingRange();
        R.rw.render();
      }
      startAnim();
    }
  };
  V.renderRequest = function () {
    const f = fieldMeta();
    const [lo, hi] = currentRange();
    const slices = {};
    ["x", "y", "z"].forEach((ax) => { if (S.show[ax]) slices[ax] = S.slices[ax]; });
    const vm = S.vectors ? vectorFor(S.field) : null;
    return {
      mode: f ? "field" : (S.networkOn && S.meta.network ? "network" : "structure"), field: f ? f.key : null, colormap: S.cmap,
      vectors: vm ? vm.key : null, vector_scale: S.vecScale, vector_every: vm ? vecEvery(Math.max(...S.meta.shape)) : null,
      vector_uniform: !!S.vecUniform,
      paths: S.pathKey || null,
      vmin: f ? lo : null, vmax: f ? hi : null, log: S.log, slices, box_faces: S.faces,
      show_particles: S.particles, particle_opacity: S.opacity, color_particles: S.colorParticles, show_network: S.networkOn,
      surface_style: S.surfStyle,
      camera: cameraState(), width: 1800, height: 1400, background: "white",
      axis: S.axis2d || "z", index: S.slices[S.axis2d || "z"], outlines: S.outlines2d,
    };
  };
})();
