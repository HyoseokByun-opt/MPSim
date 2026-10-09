/* MPSim (Material Property Simulation) - browser front-end.

   The browser holds only the form. Everything that decides a number - the
   validation, the RVE plan, the solves - happens on the server, so this file
   never re-implements engine logic; it requests /api/plan whenever the form
   changes and displays the response.

   Layout: ribbon tabs select a command panel on the left; the centre holds
   the 3D/2D viewport, the Result Viewer and the report; the right inspector
   holds display settings and the run list; the dock shows the console,
   solver convergence and the RVE checks. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const CH = window.MPSCharts;
  const fmt = (v) => CH.fmt(v);
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const V = window.MPSViewer;
  const I = window.ICON;

  /* ================================================================ catalogues */
  const PROPS = [
    ["k", "Thermal conductivity", "W/m·K"], ["sigma", "Electrical conductivity", "S/m"], ["eps_r", "Relative permittivity", "-"], ["tan_d", "Loss tangent", "-"],
    ["mu_r", "Relative permeability", "-"], ["E", "Young's modulus", "GPa"], ["nu", "Poisson's ratio", "-"], ["alpha", "CTE", "ppm/K"],
    ["rho", "Density", "g/cm³"], ["cp", "Specific heat", "J/kg·K"],
  ];
  // optional: used by the moisture analysis and the above-Tg elasticity; empty = not set
  const OPT_PROPS = [
    ["D_w", "Moisture diffusivity", "m²/s"], ["c_sat", "Saturated moisture uptake", "wt%"], ["cme", "Moisture expansion CME", "1/mass fr."],
    ["tg", "Glass transition Tg", "°C"], ["E_r", "Young's modulus above Tg", "GPa"], ["alpha2", "CTE above Tg", "ppm/K"],
  ];
  const SHAPES = {
    sphere: { label: "Sphere", icon: "sphere", hint: "Spherical particle." },
    spheroid: { label: "Spheroid", icon: "spheroid", hint: "Aspect ratio c/a (length along the symmetry axis / equatorial diameter): above 1 needle-like, below 1 platelet-like." },
    cylinder: { label: "Cylinder", icon: "cylinder", hint: "Rod or fibre; a length below the diameter gives a disc." },
    spherocylinder: { label: "Capsule", icon: "spherocylinder", hint: "Rod or fibre with hemispherical ends; the total length is not smaller than the diameter." },
    cube: { label: "Cube", icon: "shape_cube", hint: "Cube with edge length d (cubic fillers, crystals). Orientation applies to its edges." },
    cuboid: { label: "Cuboid", icon: "cuboid", hint: "Box with three edge lengths (bricks, flakes with square edges, rectangular platelets)." },
    superellipsoid: { label: "Rounded box", icon: "superellipsoid", hint: "Superellipsoid |x/a|ⁿ+|y/b|ⁿ+|z/c|ⁿ ≤ 1: n = 2 is an ellipsoid, larger n a box with rounded edges (the 'meta-particle' of discrete-element codes)." },
    polyhedron: { label: "Polyhedron", icon: "polyhedron", hint: "Convex polyhedron: a regular solid, or irregular grains (crushed particles) drawn from a set of random convex shapes. d is the circumscribed diameter." },
    helix: { label: "Helix", icon: "helix", hint: "A wire of diameter d wound on a coil (curled fibre, spring-shaped filler); coil diameter is measured at the wire centre." },
    network: { label: "Network", icon: "network", hint: "Bicontinuous random structure of any material: open porosity when the material is a gas, a sintered skeleton or foam solid when it is solid. The size is the characteristic ligament width." },
  };
  const PARTICLE_SHAPES = ["sphere", "spheroid", "cylinder", "spherocylinder", "cube", "cuboid", "superellipsoid", "polyhedron", "helix"];
  const POLY_KINDS = { tetrahedron: "Tetrahedron", octahedron: "Octahedron", dodecahedron: "Dodecahedron", icosahedron: "Icosahedron", irregular: "Irregular grains" };
  const AN = {
    thermal: { title: "Thermal conductivity", icon: "thermal", color: "var(--c-thermal)", solver: "Periodic finite volume (default) · NASA PuMA FE on request", out: "k tensor · diffusivity · area-specific resistance · temperature and heat-flux fields" },
    electrical: { title: "Electrical conductivity", icon: "electrical", color: "var(--c-electrical)", solver: "Periodic finite volume (default) · NASA PuMA FE on request", out: "σ tensor · percolation · sheet resistance · current-density field" },
    dielectric: { title: "Permittivity and loss", icon: "dielectric", color: "var(--c-diel)", solver: "Periodic finite volume (default) · NASA PuMA FE on request", out: "Dk tensor · Df · capacitance · local field enhancement" },
    magnetic: { title: "Magnetic permeability", icon: "magnetic", color: "var(--c-mag)", solver: "Periodic finite volume (default) · NASA PuMA FE on request", out: "μr tensor · field enhancement" },
    emi: { title: "EMI shielding", icon: "emi", color: "var(--c-mag)", solver: "PuMA σ, ε, μ → scikit-rf", out: "SE(f) with reflection, absorption and multiple-reflection terms" },
    cte: { title: "Elasticity and CTE", icon: "cte", color: "var(--c-mech)", solver: "Voxel FE with an FFT preconditioner (FANS) · 6 strains + ΔT", out: "C and S matrices · E, G, ν · α tensor · thermal stress fields" },
    permeability: { title: "Permeability", icon: "permeability", color: "var(--c-flow)", solver: "NASA PuMA · Stokes finite elements", out: "K tensor · flow resistivity · Kozeny constant · velocity field", pores: true },
    moisture: { title: "Moisture uptake and swelling", icon: "tortuosity", color: "var(--c-flow)", solver: "Periodic FV on the activity c/c_sat (permeability D·c_sat) · FANS eigenstrain for swelling", out: "Effective moisture diffusivity · saturated uptake · uptake curve of the part · hygroscopic swelling and CME" },
    viscosity: { title: "Viscosity and flowability", icon: "permeability", color: "var(--c-flow)", solver: "Voxel FE creeping flow (B-bar, FFT-preconditioned CG) · particle dynamics of the fillers (lubrication, contacts, friction, van der Waals; CPU or CUDA GPU) · Krieger–Dougherty with the jammed packing fraction", out: "Relative viscosity · flow curve of the compound · maximum packing fraction · underfill filling time · settling" },
    filtration: { title: "Filtration efficiency", icon: "pores", color: "var(--c-flow)", solver: "Particle tracking in the PuMA Stokes field", out: "Efficiency per particle size · most penetrating size · pressure drop · quality factor", pores: true },
    tortuosity: { title: "Diffusion", icon: "tortuosity", color: "var(--c-diff)", solver: "NASA PuMA · continuum tortuosity", out: "τ · D_eff/D₀ · formation factor · Knudsen diffusion", pores: true },
    acoustics: { title: "Acoustic absorption", icon: "acoustics", color: "var(--c-ac)", solver: "JCA model from PuMA flow and diffusion fields", out: "φ, σ, α∞, Λ, Λ' · absorption coefficient · NRC", pores: true },
    radiation: { title: "Radiation", icon: "radiation", color: "var(--c-rad)", solver: "NASA PuMA · ray casting", out: "Extinction coefficient · Rosseland radiative conductivity", pores: true },
    morphology: { title: "Porosity and sizes", icon: "morphology", color: "var(--c-struct)", solver: "PoreSpy · NASA PuMA", out: "Size distributions · specific surface · chords · mean intercept length · connectivity" },
    percolation: { title: "Percolation path", icon: "tortuosity", color: "var(--c-struct)", solver: "connected clusters · geodesic shortest paths", out: "Spanning, dead-end and isolated fractions · geodesic tortuosity · path fields in 3D" },
    porosimetry: { title: "Porosimetry", icon: "porosimetry", color: "var(--c-struct)", solver: "PoreSpy morphological drainage", out: "Intrusion curve · pore-entry sizes · largest through-pore · bubble point", pores: true },
    pore_network: { title: "Pore network", icon: "pore_network", color: "var(--c-struct)", solver: "PoreSpy SNOW2", out: "Pores and throats · coordination number · size distributions", pores: true },
    grains: { title: "Grain analysis", icon: "grains", color: "var(--c-struct)", solver: "Particle list · PoreSpy · PuMA", out: "Size, sphericity, orientation tensor, spacing, clusters, matrix ligament", particles: true },
  };
  // short names for the run list
  const SHORT_AN = { thermal: "Thermal", electrical: "Electrical", dielectric: "Dielectric", magnetic: "Magnetic", emi: "EMI",
    cte: "Elastic/CTE", permeability: "Permeability", filtration: "Filtration", tortuosity: "Diffusion", acoustics: "Acoustics",
    radiation: "Radiation", morphology: "Pore sizes", percolation: "Percolation", porosimetry: "Porosimetry",
    pore_network: "Pore network", grains: "Grains", viscosity: "Viscosity", moisture: "Moisture" };
  const ST = { pass: "Passed", warn: "Warning", fail: "Failed" };
  const STATE = { queued: "Queued", running: "Running", done: "Completed", failed: "Failed", stopped: "Stopped" };
  const QUALITY_HINT = {
    fast: "6 voxels across the smallest feature, RVE ≥ 4 × the largest particle, ≥ 20 particles. Suitable for trend screening.",
    standard: "10 voxels across the smallest feature, RVE ≥ 7 × the largest particle, ≥ 50 particles. Suitable for routine analysis.",
    accurate: "14 voxels across the smallest feature, RVE ≥ 10 × the largest particle, ≥ 100 particles and 3 realisations. Suitable for reporting.",
  };
  const BACKEND = { puma: "NASA PuMA FE + periodic FV", puma_fv: "NASA PuMA FV", builtin: "built-in verification solver" };
  const backendTxt = (r) => (BACKEND[r.backend] || "") + (r.plan && r.plan.film ? " · thin film" : "");
  const PHASE_COLORS = ["#d9dde3", "#2f6fdd", "#f28e2b", "#3aa655", "#d64545", "#8e6bbf", "#1bb3c8", "#c9a227", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"];
  const TABS = {
    home: ["home", "Home", "Presets and the analysis workflow"],
    material: ["matrix", "Material", "Matrix material and properties"],
    structure: ["cube", "Structure", "Fillers, fibres, pores and their geometry"],
    domain: ["auto", "Domain", "Voxel size, RVE size and the criteria behind them"],
    analysis: ["morphology", "Structure analysis", "Image-based analyses of the generated structure"],
    simulation: ["solver", "Simulation", "Effective properties from full-field solves"],
    doe: ["preset", "DOE", "Parameter studies: ranges, queued runs, collected results"],
    optimization: ["optimize", "Optimization", "Key design factors, a surrogate model and the optimum"],
    calibration: ["calibrate", "Calibration", "Unknown material values from measured effective properties"],
    results: ["table", "Results", "Runs, downloads and figures"],
  };

  const state = {
    lib: null, byId: {}, custom: {}, presets: [], form: null, plan: null, planErrors: [], wf: null, case: null,
    jobs: [], active: null, seq: 0, history: {}, partial: {}, activeState: null, activePreview: false,
    result: null, resultJob: null, health: null, viewerJob: null, viewerSig: "",
    tab: "home", center: "vis", rtab: "display", dtab: "console", sel: 0,
    doeCat: [], doeList: [], doeId: null, doeData: null,
    doe: { name: "", design: "full", samples: 12, seed: 1, parameters: [] },
    // any input or response on either axis, a third on the colour, a fourth on
    // the size - the way a design study is actually interrogated
    scatter: { x: "", y: "", c: "", s: "" },
    opt: {
      studyId: null, response: "", goal: "max", target: null, data: null, busy: false,
      // building a trustworthy model and using it are two different jobs, so
      // they are two steps rather than one button
      stage: "model",
      options: { kernel: "matern52", noise: "", restarts: 4 },
      multi: { responses: [], goals: {}, pop: 64, generations: 60, data: null, busy: false },
    },
    settings: null,
    openDoe: {},
  };

  /* ================================================================ utils */
  async function api(method, url, body) {
    const r = await fetch(url, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    const ct = r.headers.get("content-type") || "";
    const data = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) {
      const msg = (data && data.errors && data.errors.join("\n")) || (data && data.error) || r.statusText;
      throw Object.assign(new Error(msg), { data, status: r.status });
    }
    return data;
  }
  let toastTimer = null;
  function toast(msg, ms = 3800) {
    const t = $("#toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
  }
  function getPath(o, p) { return p.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o); }
  function setPath(o, p, v) {
    const ks = p.split(".");
    let a = o;
    for (let i = 0; i < ks.length - 1; i++) { if (a[ks[i]] == null) a[ks[i]] = /^\d+$/.test(ks[i + 1]) ? [] : {}; a = a[ks[i]]; }
    a[ks[ks.length - 1]] = v;
  }
  function deepMerge(base, over) {
    if (over === undefined) return base;
    if (Array.isArray(over) || typeof over !== "object" || over === null) return over;
    const out = Array.isArray(base) ? [] : { ...(base || {}) };
    for (const [k, v] of Object.entries(over)) out[k] = typeof v === "object" && v !== null && !Array.isArray(v) ? deepMerge(base ? base[k] : undefined, v) : v;
    return out;
  }
  function download(name, blob) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  const pill = (st) => `<span class="pill ${st}">${ST[st] || st}</span>`;
  function secs(s) { if (!s) return "0 s"; if (s < 90) return `${s.toFixed(0)} s`; if (s < 5400) return `${(s / 60).toFixed(1)} min`; return `${(s / 3600).toFixed(1)} h`; }
  const u = (unit) => ({ nm: "nm", um: "µm", mm: "mm" }[unit] || "µm");
  function tbl(head, rows, cls = "") {
    return `<div class="tablewrap"><table class="grid ${cls}"><tr>${head.map((h, i) => `<th class="${i ? "n" : ""}">${h}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr class="${r._cls || ""}">${r.map((c, i) => `<td class="${i ? "n" : ""}">${c ?? "—"}</td>`).join("")}</tr>`).join("") + "</table></div>";
  }
  const kv = (rows) => tbl(["Quantity", "Value"], rows.filter((r) => r));
  const withUnit = (v, unit) => (v === null || v === undefined || (typeof v === "number" && !isFinite(v)) ? "—" : `${fmt(v)}${unit ? " " + unit : ""}`);

  /* ================================================================ materials and form */
  function mat(id) { return state.custom[id] || state.byId[id] || null; }
  function loadCustom() { try { state.custom = JSON.parse(localStorage.getItem("mpsim.custom") || "{}"); } catch (e) { state.custom = {}; } }
  function saveCustom() { try { localStorage.setItem("mpsim.custom", JSON.stringify(state.custom)); } catch (e) { } }
  /* The material library: the built-in entries, edits of them (kept under the
     same id in this browser) and materials the user adds. The Material tab
     edits it; the Structure tab assigns a library material to the matrix and
     to every phase, and a library edit reaches every use in the open case. */
  const isOverride = (id) => !!(state.byId[id] && state.custom[id]);
  const isUserMat = (id) => !state.byId[id] && !!state.custom[id];
  function allMaterials() {
    return state.lib.materials.map((m) => mat(m.id)).concat(Object.values(state.custom).filter((m) => !state.byId[m.id]));
  }
  function groupOf(m) { return state.byId[m.id] ? state.byId[m.id].group : (m.group || "custom"); }
  function groupLabel(g) { const x = state.lib.groups.find((q) => q.id === g); return x ? x.label : "User materials"; }
  function materialOptions(sel) {
    let h = "";
    const all = allMaterials();
    for (const g of state.lib.groups.map((x) => x.id).concat(["custom"])) {
      const ms = all.filter((m) => groupOf(m) === g);
      if (!ms.length) continue;
      h += `<optgroup label="${esc(groupLabel(g))}">` + ms.map((m) => `<option value="${m.id}" ${m.id === sel ? "selected" : ""}>${esc(m.name)}${isOverride(m.id) ? " (edited)" : ""}</option>`).join("") + "</optgroup>";
    }
    return h;
  }
  function usesOf(id) {
    const f = state.form, out = [];
    if (f.matrix.material_id === id) out.push({ path: "matrix", label: "Matrix" });
    f.phases.forEach((p, i) => {
      if (p.material_id === id) out.push({ path: `phases.${i}`, label: p.name });
      if (p.shell && p.shell.enabled && p.shell.material_id === id) out.push({ path: `phases.${i}.shell`, label: `${p.name} (shell)` });
    });
    return out;
  }
  const differs = (entry) => { const m = mat(entry.material_id); return !!m && PROPS.some(([k]) => Number(entry.props[k]) !== Number(m.props[k])); };
  function libChanged(id) {
    const m = mat(id);
    if (!m) return 0;
    const uses = usesOf(id);
    uses.forEach((x) => { getPath(state.form, x.path).props = clone(m.props); });
    saveCustom();
    if (uses.length) { saveForm(); schedulePlan(); }
    return uses.length;
  }
  function propsEditor(path, props, ref) {
    const row = ([k, lab, unit]) => {
      const mod = ref && props && Number(props[k]) !== Number(ref[k]);
      return `<label class="prop"><span>${lab}<em>${unit}</em></span><input type="number" step="any" class="${mod ? "mod" : ""}" data-k="${path}.${k}" data-num data-ref="${ref ? (ref[k] ?? "") : ""}"></label>`;
    };
    const anyOpt = OPT_PROPS.some(([k]) => props && props[k] !== undefined && props[k] !== null && props[k] !== "");
    return `<div class="props">` + PROPS.map(row).join("") + "</div>" +
      `<details class="optprops" ${anyOpt ? "" : ""}><summary>Moisture and glass transition <span class="hint">(optional${anyOpt ? ", set" : ""})</span></summary><div class="props">${OPT_PROPS.map(row).join("")}</div></details>`;
  }
  function matEntry(id) { const m = mat(id); return { material_id: id, name: m.name, props: clone(m.props) }; }
  function phaseEntry(id, shape) {
    const m = mat(id);
    return {
      material_id: id, name: m.name, props: clone(m.props), shape,
      size: { d: shape === "network" ? 2 : shape === "helix" ? 1 : 10, aspect: 0.2, length: 60, ly: 6, lz: 3, n: 4,
              poly: "icosahedron", n_vertices: 14, variants: 8, coil_d: 6, pitch: 4, turns: 3, unit: "um" },
      dist: { type: "mono", cv: 0.25 }, fraction: { value: 20, basis: "vol" },
      orientation: { mode: "iso", spread_deg: 0 }, overlap: false, gap_frac: 0,
      shell: { enabled: false, thickness: 0.5, material_id: "sio2", name: "shell", props: clone(mat("sio2").props) },
      r_int: 0, r_contact: 0,
    };
  }
  function defaultForm() {
    return {
      name: "New case", matrix: matEntry("epoxy"), phases: [phaseEntry("al2o3", "sphere")],
      rve: { auto: true, quality: "standard", L_um: null, voxel_um: null, auto_enlarge: true, n_seeds: 0, seed: 1, film: false, T_um: null, L_unit: "um", h_unit: "um", T_unit: "um" },
      analyses: { thermal: true },
      options: {
        directions: "xyz", tol: 1e-6, delta_T: 100, thermal_bc: "constrained", contacts: "apart", parallel_solves: "auto", cond_method: "fv", elastic_method: "fans", crosscheck: false, resolution_check: "auto", reuse_runs: true, backend: "puma", contrast_cap: 1e6, thickness_mm: 1, temperature_K: 298.15, knudsen: false,
        emi: { thickness_mm: 1, f_min_hz: 1e6, f_max_hz: 1e10, method: "complex", n_freq: 10, fullwave: true },
        acoustics: { thickness_mm: 20, f_min_hz: 100, f_max_hz: 10000 },
        radiation: { temperature_K: 1000, refractive_index: 1, sources: 200, rays: 500 },
        porosimetry: { fluid: "mercury", surface_tension_N_m: 0.485, contact_angle_deg: 140, steps: 40 },
        filtration: { face_velocity_m_s: 0.05, n_particles: 200, particle_density_kg_m3: 1000, d_min_um: 0.01, d_max_um: 5, n_sizes: 8 },
        viscosity: { model: { type: "newtonian", mu: 1, K: 1, n: 0.7, mu0: 1, mu_inf: 0, lam: 1 }, yield_Pa: 0, gd_min: 0.01, gd_max: 1000, gd_ref: 10,
          phi_m_mode: "auto", friction: "none", phi_m: 0.64, dilute_rve: true, density_resin: 1150, settle_min: 60,
          underfill: { gap_um: 50, length_mm: 10, gamma_mN_m: 35, theta_deg: 30 },
          dem: { on: true, n: 500, strain: 5, roughness_nm: 5, hmin_nm: 1, mu_f: 0.25, hamaker_J: null, bound_nm: 0, adhesion_mJ_m2: 0, rates: 4, backend: "auto" } },
        moisture: { thickness_mm: 1, hours: 168, sides: "both" }, above_tg: false,
        gas: { molar_mass_g_mol: 28.97, viscosity_Pa_s: 1.81e-5, diffusivity_m2_s: 2.0e-5, pressure_Pa: 101325, molecule_diameter_nm: 0.37, density_kg_m3: 1.204, sound_speed_m_s: 343.2, gamma: 1.4, prandtl: 0.71 },
      },
    };
  }
  // switching to manual sizing starts from the grid the planner had chosen
  function manualFromPlan() {
    const pl = state.plan && state.plan.plan, rv = state.form.rve;
    if (!pl) return;
    if (!rv.L_um) rv.L_um = +pl.L_um.toPrecision(6);
    if (!rv.voxel_um) rv.voxel_um = +pl.h_um.toPrecision(6);
    rv.N = null;
  }
  function completeForm(f) {
    const base = defaultForm();
    const out = deepMerge(base, clone(f));
    out.analyses = clone(f.analyses || base.analyses);
    if (out.matrix && out.matrix.material_id && !(f.matrix && f.matrix.props) && mat(out.matrix.material_id)) {
      out.matrix.props = clone(mat(out.matrix.material_id).props);
      if (!f.matrix.name) out.matrix.name = mat(out.matrix.material_id).name;
    }
    out.phases = (f.phases || []).map((p) => {
      const d = phaseEntry(p.material_id && mat(p.material_id) ? p.material_id : "al2o3", p.shape || "sphere");
      const q = deepMerge(d, clone(p));
      if (!p.props && mat(q.material_id)) q.props = clone(mat(q.material_id).props);
      if (p.shell && p.shell.material_id && !p.shell.props && mat(p.shell.material_id)) q.shell.props = clone(mat(p.shell.material_id).props);
      if (!p.name) q.name = mat(q.material_id).name;
      return q;
    });
    return out;
  }
  let saveTimer = null;
  function saveForm() { clearTimeout(saveTimer); saveTimer = setTimeout(() => { try { localStorage.setItem("mpsim.form", JSON.stringify(state.form)); } catch (e) { } }, 300); }
  const isVoid = (p) => (+((p.props || {}).E) || 0) <= 0;
  function hasPores() { return isVoid(state.form.matrix) || state.form.phases.some(isVoid); }
  function hasParticles() { return state.form.phases.some((p) => p.shape !== "network"); }
  function disabledReason(key) {
    const a = AN[key];
    if (a.pores && !hasPores()) return "Requires a pore phase or a gas matrix.";
    if (a.particles && !hasParticles()) return "Requires a particle phase.";
    return "";
  }
  function phaseColors() {
    let ci = 1;
    return state.form.phases.map((p) => (isVoid(p) ? "#7fd3f7" : PHASE_COLORS[ci++ % PHASE_COLORS.length]));
  }

  /* ================================================================ ribbon */
  const RB = (id, icon, label, extra = {}) => ({ id, icon, label, ...extra });
  function ribbonGroups() {
    const f = state.form, an = f.analyses || {};
    const toggle = (k) => RB("an:" + k, AN[k].icon, AN[k].title, { on: !!an[k], dis: !!disabledReason(k), color: AN[k].color, tip: `${AN[k].title} — ${AN[k].out}` });
    const running = ["running", "queued"].includes(state.activeState);
    // building belongs to the two stages that build something - the geometry on
    // Structure, the voxel grid on Domain - so Execute only runs and stops
    // full means every parallel slot is taken; the button says so instead of
    // silently adding another run to the pile
    // one source of truth with the guard in submit(), and it counts everyone's
    // runs rather than only this browser's
    const full = queueRoom().free <= 0;
    const execute = { group: "Execute", items: [
      RB("run", "play", "Run analyses", { primary: true, dis: full, tip: full ? "The machine is already running its maximum number of analyses" : "Run analyses" }),
      RB("stop", "stop", "Stop", { dis: !running })] };
    const G = {
      home: [
        { group: "Case", items: [RB("new", "new", "New case"), RB("import", "open", "Open input", { file: true }), RB("export", "save", "Save input")] },
        { group: "Library", items: [RB("presets", "preset", "Presets"), RB("database", "database", "Material database")] },
        execute,
      ],
      material: [
        { group: "Library", items: [RB("libNew", "new", "New material"), RB("libDup", "duplicate", "Duplicate"), RB("libDel", "delete", "Delete", { dis: !isUserMat(state.libSel), tip: "Deletes a material you added; a built-in entry is restored instead" }), RB("libReset", "reset", "Restore library values", { dis: !isOverride(state.libSel) })] },
        { group: "Exchange", items: [RB("libExport", "save", "Export library"), RB("database", "database", "Material table")] },
      ],
      structure: [
        { group: "Add a phase (any material; Air makes it pore space)", items: PARTICLE_SHAPES.map((s) => RB("add:" + s, SHAPES[s].icon, SHAPES[s].label, { color: "var(--c-gen)" })) },
        { group: "Bicontinuous", items: [RB("add:network", "network", "Network", { color: "var(--c-flow)" })] },
        { group: "Selected phase", items: [RB("dup", "duplicate", "Duplicate", { dis: !f.phases.length }), RB("del", "delete", "Delete", { dis: !f.phases.length })] },
        { group: "Build", items: [RB("generate", "cube", "Generate structure", { primary: true })] },
      ],
      domain: [
        { group: "Accuracy level", items: ["fast", "standard", "accurate"].map((q) => RB("q:" + q, q, q[0].toUpperCase() + q.slice(1), { on: f.rve.quality === q })) },
        { group: "Grid", items: [RB("autoGrid", "auto", "Automatic grid", { on: !!f.rve.auto }), RB("checks", "checks", "RVE checks")] },
        { group: "Build", items: [RB("generate", "auto", "Generate voxel grid", { primary: true })] },
      ],
      analysis: [
        { group: "Pore space", items: ["morphology", "porosimetry", "pore_network", "percolation"].map(toggle) },
        { group: "Particles", items: [toggle("grains")] },
        execute,
      ],
      doe: [
        { group: "Design", items: [RB("doeAdd", "plus", "Add parameter"), RB("doeClear", "delete", "Clear parameters")] },
        { group: "Study", items: [RB("doeCreate", "play", "Create and queue", { primary: true }), RB("c:doe", "preset", "DOE results")] },
      ],
      optimization: [
        { group: "Study", items: [RB("c:opt", "optimize", "Optimization", { on: state.center === "opt" }), RB("c:doe", "preset", "DOE results")] },
        { group: "Model", items: [RB("optRun", "surrogate", "Fit and optimise", { primary: true }),
                                  RB("optQueue", "play", "Queue the optimum", { dis: !(state.opt.data && state.opt.data.optimum) })] },
      ],
      calibration: [
        { group: "Problem", items: [RB("calAddM", "plus", "Add measurement"), RB("calAddU", "plus", "Add unknown"), RB("calClear", "delete", "Clear")] },
        { group: "Check", items: [RB("calPreview", "checks", "Check identifiability", { tip: "Instant: an effective-medium model says whether these measurements can determine these unknowns" })] },
        { group: "Calibrate", items: [RB("calRun", "play", "Calibrate on the RVE", { primary: true }), RB("c:cal", "calibrate", "Calibration results", { on: state.center === "cal" })] },
        { group: "Result", items: [RB("calApply", "save", "Apply to the case", { dis: !(state.cal && state.cal.result), tip: "Set the calibrated values in the case (only those the measurements determine)" })] },
      ],
      simulation: [
        { group: "Conduction", items: ["thermal", "electrical"].map(toggle) },
        { group: "Electromagnetics", items: ["dielectric", "magnetic", "emi"].map(toggle) },
        { group: "Mechanics", items: [toggle("cte")] },
        { group: "Porous media", items: ["permeability", "filtration", "tortuosity", "acoustics"].map(toggle) },
        { group: "Rheology", items: [toggle("viscosity")] },
        { group: "Reliability", items: [toggle("moisture")] },
        { group: "Radiation", items: [toggle("radiation")] },
        execute,
      ],
      results: [
        { group: "Views", items: [RB("c:vis", "view3d", "3D / 2D view", { on: state.center === "vis" }), RB("c:res", "table", "Result Viewer", { on: state.center === "res" }), RB("c:rep", "report", "Report", { on: state.center === "rep" })] },
        { group: "Export", items: [RB("dl:result", "json", "Result JSON", { dis: !state.resultJob }), RB("dl:layer_card", "layers", "Layer card", { dis: !state.resultJob }), RB("dl:structure_set", "image", "Structure TIFF set", { dis: !state.resultJob, tip: "structure.tif (one grey value per material), structure_labels.tif (material IDs) and structure_legend.csv (IDs, grey values, materials and properties) - for ImageJ, PuMA or any other voxel code" }), RB("dl:bundle", "zip", "Complete ZIP", { dis: !state.resultJob })] },
        { group: "Figures", items: [RB("shot", "camera", "Screen capture"), RB("render", "image", "High-resolution figure")] },
      ],
    };
    return G[state.tab];
  }
  function renderRibbon() {
    $("#rbar").innerHTML = ribbonGroups().map((g) => `<div class="rgroup"><div class="ritems">${g.items.map((it) =>
      `<button class="rbtn ${it.on ? "on" : ""} ${it.dis ? "dis" : ""} ${it.primary ? "primary" : ""} ${it.file ? "file" : ""}" data-rb="${it.id}" title="${esc(it.tip || it.label)}" style="--ic:${it.color || "var(--c-gen)"}">${I(it.icon)}<span>${it.label}</span>${it.file ? '<input type="file" accept=".json" data-rbfile>' : ""}</button>`).join("")}</div><div class="rcap">${g.group}</div></div>`).join("");
    $$("#rtabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === state.tab));
  }
  async function ribbonAction(id, el) {
    const f = state.form;
    if (el && el.classList.contains("dis")) return;
    const [kind, arg] = id.split(":");
    if (kind === "an") {
      if (disabledReason(arg)) { toast(disabledReason(arg)); return; }
      f.analyses[arg] = !f.analyses[arg];
      saveForm(); schedulePlan(); renderRibbon(); renderLeft();
      const card = $(`[data-card="${arg}"]`); if (card) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
      return;
    }
    if (kind === "add") {
      let p;
      if (arg === "pores") { p = phaseEntry("air", "sphere"); p.name = "Pores"; p.fraction.value = 10; }
      else if (arg === "network") { p = phaseEntry("air", "network"); p.name = "Network"; p.fraction.value = 20; }
      else {
        p = phaseEntry("al2o3", arg);
        if (arg === "spheroid") p.size.aspect = 0.2;
        if (arg === "helix") { p.name = "Curled fibre"; p.fraction.value = 5; }
      }
      f.phases.push(p); state.sel = f.phases.length - 1;
      if (state.tab !== "structure") setTab("structure");
      renderLeft(); renderRibbon(); schedulePlan(0); saveForm();
      return;
    }
    if (kind === "q") { f.rve.quality = arg; renderRibbon(); renderLeft(); schedulePlan(0); saveForm(); return; }
    if (kind === "c") { setCenter(arg); renderRibbon(); return; }
    if (kind === "dl") { if (state.resultJob) window.location.href = `/api/jobs/${state.resultJob}/download/${arg}`; return; }
    switch (id) {
      case "new": state.form = defaultForm(); state.sel = 0; renderAll(); schedulePlan(0); saveForm(); toast("A new case has been created."); break;
      case "export": download(`${f.name || "input"}.json`, new Blob([JSON.stringify(f, null, 1)], { type: "application/json" })); break;
      case "presets": setTab("home"); setTimeout(() => { const s = $("#presetSec"); if (s) s.scrollIntoView({ behavior: "smooth" }); }, 50); break;
      case "database": openDatabase(); break;
      case "libNew": case "libDup": {
        const src = mat(state.libSel) || mat(f.matrix.material_id);
        const nid = "user_" + Date.now().toString(36);
        state.custom[nid] = { id: nid, name: id === "libNew" ? "New material" : `${src.name} (copy)`, group: "custom",
                              props: clone(src.props), note: id === "libNew" ? `Start values from ${src.name}` : src.note || "" };
        state.libSel = nid; saveCustom(); renderLeft(); renderRibbon();
        toast(id === "libNew" ? `A new material starts from the values of ${src.name}.` : "The material has been duplicated.");
        setTimeout(() => { const el = $('#lbody input[data-lk="name"]'); if (el) { el.focus(); el.select(); } }, 30);
        break;
      }
      case "libDel": {
        const sid = state.libSel;
        if (!isUserMat(sid)) break;
        const n = usesOf(sid).length;
        if (n && !confirm(`${mat(sid).name} is used by ${n} region(s) of this case. They keep its values. Delete it from the library?`)) break;
        delete state.custom[sid]; state.libSel = f.matrix.material_id; saveCustom(); renderLeft(); renderRibbon(); toast("The material has been deleted from the library.");
        break;
      }
      case "libReset": {
        const sid = state.libSel;
        if (!isOverride(sid)) break;
        delete state.custom[sid];
        const n = libChanged(sid);
        renderLeft(); renderRibbon(); toast(`The library values have been restored${n ? ` (applied to ${n} region(s) of this case)` : ""}.`);
        break;
      }
      case "libExport": {
        const out = { schema: "mpsim.materials/1", exported: new Date().toISOString(), materials: Object.values(state.custom) };
        download("materials_library.json", new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
        break;
      }
      case "makePore": {
        const p = f.phases[state.sel];
        if (p && mat("air")) { p.material_id = "air"; p.props = clone(mat("air").props); if (!/pore/i.test(p.name)) p.name = "Pores (" + SHAPES[p.shape].label.toLowerCase() + ")"; p.r_int = 0; p.r_contact = 0; renderLeft(); renderRibbon(); schedulePlan(0); saveForm(); toast("This phase is now pore space; its geometry is the pore shape."); }
        break;
      }
      case "dup": if (f.phases.length && state.sel >= 0) { f.phases.splice(state.sel + 1, 0, clone(f.phases[state.sel])); state.sel++; renderLeft(); renderRibbon(); schedulePlan(0); saveForm(); } break;
      case "del": if (f.phases.length && state.sel >= 0) {
        const gone = state.sel, rc = {};
        Object.entries(f.contact_rc || {}).forEach(([k, v]) => {
          let [a, b] = k.split("-").map(Number);
          if (a === gone || b === gone) return;
          if (a > gone) a--; if (b > gone) b--;
          rc[`${Math.min(a, b)}-${Math.max(a, b)}`] = v;
        });
        f.contact_rc = rc;
        f.phases.splice(state.sel, 1); state.sel = Math.max(0, state.sel - 1); renderLeft(); renderRibbon(); schedulePlan(0); saveForm(); } break;
      case "autoGrid": f.rve.auto = !f.rve.auto; if (!f.rve.auto) manualFromPlan(); renderRibbon(); renderLeft(); schedulePlan(0); saveForm(); break;
      case "checks": setDock("checks", true); break;
      case "doeAdd": doeAddParam(); break;
      case "doeClear": state.doe.parameters = []; renderLeft(); break;
      case "calAddM": calAddMeasurement(); break;
      case "calAddU": calAddUnknown(); break;
      case "calClear": { const c = CAL(); c.measurements = []; c.unknowns = []; c.preview = null; calSave(); renderLeft(); renderCal(); break; }
      case "calPreview": calPreview(); break;
      case "calRun": calRun(); break;
      case "calApply": calApply(); break;
      case "doeCreate": doeCreate(); break;
      case "optRun": setCenter("opt"); runOptimize(); break;
      case "optQueue": queueOptPoint("optimum"); break;
      case "generate": submit(true); break;
      case "run": submit(false); break;
      case "stop": stopActive(); break;
      case "shot": setCenter("vis"); screenshot(); break;
      case "render": setCenter("vis"); hiresRender(); break;
    }
  }

  /* ================================================================ navigation */
  function setTab(tab) {
    state.tab = tab;
    if (tab === "doe") loadDoeCatalogue().then(() => { if (state.tab === "doe") renderLeft(); });
    if (tab === "calibration") { CAL(); loadDoeCatalogue().then(() => { if (state.tab === "calibration") renderLeft(); }); calLoadTheory(); }
    renderRibbon();
    renderLeft();
  }
  function setCenter(c) {
    state.center = c;
    $$("#ctabs button").forEach((b) => b.classList.toggle("on", b.dataset.c === c));
    $$(".cpage").forEach((p) => p.classList.toggle("on", p.id === "c-" + c));
    if (c === "vis") openViewer();
    if (c === "res") renderResults();
    if (c === "doe") { renderDoe(); if (state.doeId && !state.doeData) openDoe(state.doeId); }
    if (c === "opt") renderOpt();
    if (c === "cal") renderCal();
    if (c === "rep") renderReport();
    if (state.tab === "results") renderRibbon();
  }
  function setRight(r) {
    state.rtab = r;
    $$("#rptabs button").forEach((b) => b.classList.toggle("on", b.dataset.r === r));
    $$(".rpage").forEach((p) => p.classList.toggle("on", p.id === "r-" + r));
  }
  function setDock(d, open) {
    state.dtab = d;
    $$("#dtabs button[data-d]").forEach((b) => b.classList.toggle("on", b.dataset.d === d));
    $$(".dpage").forEach((p) => p.classList.toggle("on", p.id === "d-" + d));
    if (open) toggleDock(false);
    if (d === "conv") setTimeout(() => drawConv(), 30);
  }
  function toggleDock(collapse) {
    const dock = $("#dock");
    const c = collapse === undefined ? !dock.classList.contains("collapsed") : collapse;
    dock.classList.toggle("collapsed", c);
    $("#dockToggle").innerHTML = I("collapse") ;
    $("#dockToggle").style.transform = c ? "rotate(180deg)" : "";
    try { localStorage.setItem("mpsim.dock", c ? "1" : "0"); } catch (e) { }
    setTimeout(() => { if (state.center === "vis") V.refresh(false); drawConv(); }, 60);
  }

  /* ================================================================ left panel */
  function renderLeft() {
    const [icon, title, sub] = TABS[state.tab];
    $("#lhead").innerHTML = `${I(icon)}<div><b>${title}</b><small>${sub}</small></div>`;
    const body = $("#lbody");
    const top = body.scrollTop;
    body.innerHTML = { home: leftHome, material: leftMaterial, structure: leftStructure, domain: leftDomain, analysis: leftAnalysis, simulation: leftSimulation, doe: leftDoe, optimization: leftOptimization, calibration: leftCalibration, results: leftResults }[state.tab]();
    fillInputs(body);
    body.scrollTop = top;
    if (state.tab === "structure") renderComposition();
  }

  async function loadSettings() {
    try { state.settings = await api("GET", "/api/settings"); } catch (e) { return; }
    if (state.tab === "home") renderLeft();
    renderRibbon();
  }
  async function onSetting(e) {
    const el = e.target, body = {};
    body[el.dataset.set] = el.value === "" ? null : Number(el.value);
    try {
      state.settings = await api("POST", "/api/settings", body);
      toast(`${state.settings.concurrency} analysis(es) at a time · `
        + `${state.settings.threads} core${state.settings.threads === 1 ? "" : "s"} each`);
    } catch (err) { toast("The setting could not be applied: " + err.message); }
    renderLeft(); renderRibbon(); pollRuns();
  }

  function leftHome() {
    const steps = [["material", "Material", "Matrix material and properties"], ["structure", "Structure", "Fillers, fibres and pores"], ["domain", "Domain", "Voxel size, RVE size and criteria"],
      ["analysis", "Structure analysis", "Porosity, porosimetry, pore network, grains"], ["simulation", "Simulation", "Conduction, mechanics, flow, acoustics, radiation"], ["run", "Run", "Results appear in the viewer and the Result Viewer"]];
    const presets = state.presets.map((p) => {
      const chips = Object.entries(p.form.analyses || {}).filter(([, v]) => v).map(([k]) => AN[k] ? `<span class="chip" style="--ic:${AN[k].color}">${I(AN[k].icon)}${AN[k].title}</span>` : "").join("");
      return `<div class="preset" data-preset="${p.id}"><div class="pn">${esc(p.name)}</div><div class="pd">${esc(p.description)}</div><div class="chips">${chips}</div></div>`;
    }).join("");
    return `<div class="sec"><div class="sh">${I("home")}Workflow</div><div class="sb"><ol class="flow">${steps.map(([k, t, s], i) => `<li data-go="${k}"><b>${i + 1}</b><span>${t}<small>${s}</small></span></li>`).join("")}</ol>
        <div class="hint">Every analysis uses validated open-source codes: NASA PuMA, PoreSpy, OpenPNM, scikit-rf and PyVista. The report lists the method and version behind each number.</div></div></div>
      ${workAreaHTML()}
      ${machineHTML()}
      <div class="sec" id="presetSec"><div class="sh">${I("preset")}Presets<small>typical semiconductor material problems</small></div><div class="sb">${presets}</div></div>`;
  }

  /* Where the work goes. A run is written into the open work folder, so a study
     is a directory you can sort, zip or move as one thing, rather than a pile of
     cases mixed in with everyone else's. It sits above the machine settings
     because it is chosen first - before anything is run, not after. */
  function workspaceHTML() {
    const ws = state.ws;
    if (!ws) return "";
    const opts = (ws.recent || []).map((p) => `<option value="${esc(p)}" ${p === ws.workspace ? "selected" : ""}>${esc(p)}${p === ws.default ? " (program folder)" : ""}</option>`).join("");
    return `<div class="sec"><div class="sh">${I("open")}Workspace<small>root of every work folder</small></div><div class="sb">
        <select data-wspick>${opts}</select>
        <div class="g2" style="margin-top:6px">
          <div><label class="f">Another workspace <span class="u">full path, any disk</span></label>
            <input type="text" data-wsnew placeholder="D:\\SimData\\MPSim"></div>
          <div style="align-self:end"><button class="btn sm" data-wsmake>${I("open")}Use this workspace</button></div>
        </div>
        <div class="hint">The root of all work, like the work directory of a simulation project: work folders given by name, their cases and runs, and the DOE studies are all created under it. The choice is kept for the next start; runs in progress finish where they started.</div>
      </div></div>`;
  }

  async function useWorkspace(path) {
    if (!path) return;
    try {
      const r = await api("POST", "/api/workspace", { path });
      state.ws = r;
      state.wf = r.workfolder;
      try { state.case = await api("GET", "/api/case"); } catch (e) { state.case = null; }
      toast(`Workspace: ${r.workspace}`);
    } catch (err) { toast("The workspace could not be used: " + err.message, 9000); return; }
    renderLeft(); renderRibbon(); pollRuns();
  }

  function workFolderHTML() {
    const w = state.wf;
    if (!w) return "";
    const opts = (w.folders || []).map((f) =>
      `<option value="${esc(f.path)}" ${f.current ? "selected" : ""}>${esc(f.name)} · ${f.runs} run${f.runs === 1 ? "" : "s"}</option>`).join("");
    return `<div class="sec"><div class="sh">${I("open")}Work folder<small>${w.runs} run${w.runs === 1 ? "" : "s"} here</small></div><div class="sb">
        <select data-wfpick>${opts}</select>
        <div class="g2" style="margin-top:6px">
          <div><label class="f">New folder <span class="u">name, or a full path</span></label>
            <input type="text" data-wfnew placeholder="${esc(todayName())}"></div>
          <div style="align-self:end"><button class="btn sm" data-wfmake>${I("open")}Create and open</button></div>
        </div>
        <div class="hint" style="margin-top:6px;word-break:break-all">Open: <b class="code">${esc(w.current)}</b></div>
        <div class="hint">Only this folder's runs are listed. Anything already running finishes into the folder it started in, so switching is safe at any time. A name with no path goes under the workspace; a full path may be on another disk.</div>
      </div></div>
      ${caseHTML()}`;
  }
  function workAreaHTML() {
    return workspaceHTML() + workFolderHTML();
  }

  /* A case is a named study inside the work folder, and every run of it lands
     in that case's directory. The tree on disk then is the tree in the head:
     one folder per session, one directory per case, runs stacked inside in the
     order they were made. */
  function caseHTML() {
    const c = state.case;
    if (!c) return "";
    const opts = (c.cases || []).map((x) =>
      `<option value="${esc(x.name)}" ${x.current ? "selected" : ""}>${esc(x.name)} · ${x.runs} run${x.runs === 1 ? "" : "s"}</option>`).join("");
    const here = (c.cases || []).find((x) => x.current);
    return `<div class="sec"><div class="sh">${I("runs")}Case<small>${here ? here.runs : 0} run${here && here.runs === 1 ? "" : "s"} in this case</small></div><div class="sb">
        <select data-casepick>${opts}</select>
        <div class="g2" style="margin-top:6px">
          <div><label class="f">New case <span class="u">a folder under the work folder</span></label>
            <input type="text" data-casenew placeholder="case-02"></div>
          <div style="align-self:end"><button class="btn sm" data-casemake>${I("runs")}Create and open</button></div>
        </div>
        <div class="hint" style="margin-top:6px">Every analysis you run is written under <b class="code">${esc(c.current)}</b>, so repeat runs and variations of one study stay together.</div>
      </div></div>`;
  }

  async function useCase(name) {
    if (!name) return;
    try {
      const r = await api("POST", "/api/case", { name });
      state.case = r;
      toast(`Case: ${r.current}`);
    } catch (err) { toast("The case could not be opened: " + err.message, 8000); return; }
    renderLeft(); pollRuns();
  }

  function todayName() {
    const d = new Date(), p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  }

  async function useWorkFolder(path) {
    if (!path) return;
    try {
      const r = await api("POST", "/api/workfolder", { path });
      state.wf = r;
      try { state.case = await api("GET", "/api/case"); } catch (e) { state.case = null; }
      toast(`Work folder: ${r.name} · ${r.runs} run${r.runs === 1 ? "" : "s"}`);
    } catch (err) { toast("The work folder could not be opened: " + err.message, 8000); return; }
    renderLeft(); renderRibbon(); pollRuns();
  }

  /* How the machine is shared out. Many small cases finish sooner side by side;
     one large case wants every core. Only the person at the keyboard knows
     which of the two they are doing, so it is set here rather than at start-up. */
  function machineHTML() {
    const s = state.settings || {};
    const cores = s.cores || (state.health && state.health.cores) || 1;
    const conc = s.concurrency || 1, th = s.threads || 1;
    const used = conc * th;
    return `<div class="sec"><div class="sh">${I("runs")}Machine<small>${cores} cores available</small></div><div class="sb">
        <div class="g2">
          <div><label class="f">Analyses at once <span class="u">parallel</span></label>
            <input type="number" min="1" max="${Math.max(cores, 1)}" step="1" data-set="concurrency" value="${conc}"></div>
          <div><label class="f">Cores each <span class="u">0 = auto (¼)</span></label>
            <input type="number" min="0" max="${4 * cores}" step="1" data-set="threads" value="${s.auto_threads ? 0 : th}"></div>
        </div>
        <div class="hint" style="margin-top:6px">Now: <b>${conc}</b> at a time × <b>${th}</b> core${th === 1 ? "" : "s"} = ${used} of ${cores}${used > cores ? " — oversubscribed, which slows every run down" : ""}.
          ${s.auto_threads ? "Automatic: a quarter of the PC per analysis, and never more than an equal share when several run at once." : "A fixed number of cores per analysis."}</div>
        <div class="hint">A change applies to the next analysis to start; nothing running is disturbed. An analysis splits its cores between its independent solves (properties, directions, elastic load cases) and the threads inside each solve, and takes the split that finishes first: a PuMA FE solve runs about 4× faster on 14 threads, several solves side by side do more work per core.</div>
      </div></div>`;
  }

  function leftMaterial() {
    const f = state.form;
    if (!state.libSel || !mat(state.libSel)) state.libSel = f.matrix.material_id;
    const id = state.libSel, m = mat(id), base = state.byId[id], uses = usesOf(id);
    const q = (state.libFilter || "").trim().toLowerCase();
    const inCase = new Set([f.matrix.material_id, ...f.phases.flatMap((p) => [p.material_id, p.shell && p.shell.enabled ? p.shell.material_id : null])].filter(Boolean));
    const all = allMaterials().filter((x) => !q || `${x.name} ${groupLabel(groupOf(x))} ${x.note || ""}`.toLowerCase().includes(q));
    let list = "";
    for (const g of state.lib.groups.map((x) => x.id).concat(["custom"])) {
      const ms = all.filter((x) => groupOf(x) === g);
      if (!ms.length) continue;
      list += `<div class="mgrp">${esc(groupLabel(g))}</div>` + ms.map((x) => `<div class="prow m ${x.id === id ? "on" : ""}" data-libsel="${x.id}"><div class="nm">${esc(x.name)}${isOverride(x.id) ? ' <span class="chip">edited</span>' : ""}${isUserMat(x.id) ? ' <span class="chip">user</span>' : ""}<small>k ${fmt(x.props.k)} · E ${fmt(x.props.E)} GPa · α ${fmt(x.props.alpha)} ppm/K · εr ${fmt(x.props.eps_r)}</small></div><span class="vf">${inCase.has(x.id) ? "in case" : ""}</span></div>`).join("");
    }
    const lrow = ([k, lab, unit]) => {
      const mod = base && Number(m.props[k]) !== Number(base.props[k]);
      return `<label class="prop"><span>${lab}<em>${unit}</em></span><input type="number" step="any" class="${mod ? "mod" : ""}" data-lk="props.${k}" value="${m.props[k] ?? ""}" title="${base ? `library value ${base.props[k] ?? "not set"}` : ""}"></label>`;
    };
    const props = `<div class="props">` + PROPS.map(lrow).join("") + "</div>" +
      `<details class="optprops" open><summary>Moisture and glass transition <span class="hint">(optional; empty = not absorbing, no Tg)</span></summary><div class="props">${OPT_PROPS.map(lrow).join("")}</div></details>`;
    // keep the selected material in view inside the list
    setTimeout(() => {
      const box = $("#lbody .mlist"), on = box && $(".prow.on", box);
      if (on && (on.offsetTop < box.scrollTop || on.offsetTop + on.offsetHeight > box.scrollTop + box.clientHeight)) box.scrollTop = on.offsetTop - 60;
    }, 0);
    const caseRows = [{ label: "Matrix", e: f.matrix }].concat(f.phases.flatMap((p) => [{ label: p.name, e: p }].concat(p.shell && p.shell.enabled ? [{ label: `${p.name} (shell)`, e: p.shell }] : [])));
    return `<div class="sec"><div class="sh">${I("database")}Material library<div class="grow"></div><span class="hint">${allMaterials().length} materials</span></div><div class="sb">
        <input type="search" placeholder="Search by name, group or note" data-libfilter value="${esc(state.libFilter || "")}">
        <div class="plist mlist">${list || '<div class="hint" style="padding:8px">No material matches.</div>'}</div>
        <div class="row wrap" style="gap:6px"><button class="btn sm" data-act="libNew">${I("new")}New material</button><button class="btn sm" data-act="libDup">${I("duplicate")}Duplicate</button>
          <button class="btn sm" data-act="libExport">${I("save")}Export</button><label class="btn sm file">${I("open")}Import<input type="file" accept=".json" data-libimport></label></div>
        <div class="hint" style="margin-top:6px">Every material can be edited here, built-in or added. Edits are kept in this browser; Export writes them to a file that Import reads on another PC.</div></div></div>
      <div class="sec"><div class="sh">${I("matrix")}<span>${esc(m.name)}</span><div class="grow"></div>${isOverride(id) ? '<span class="chip">edited library entry</span>' : isUserMat(id) ? '<span class="chip">user material</span>' : '<span class="chip">library</span>'}</div><div class="sb">
        <label class="f">Name</label><input data-lk="name" value="${esc(m.name)}">
        ${isUserMat(id) ? `<label class="f">Group</label><select data-lk="group">${state.lib.groups.map((g) => `<option value="${g.id}" ${g.id === m.group ? "selected" : ""}>${esc(g.label)}</option>`).join("")}<option value="custom" ${!state.lib.groups.some((g) => g.id === m.group) ? "selected" : ""}>User materials</option></select>` : ""}
        <label class="f">Note or source</label><input data-lk="note" value="${esc(m.note || "")}">
        ${props}
        <div class="row wrap" style="gap:6px;margin-top:9px">${isOverride(id) ? `<button class="btn sm" data-act="libReset">${I("reset")}Restore library values</button>` : ""}${isUserMat(id) ? `<button class="btn sm danger" data-act="libDel">${I("delete")}Delete</button>` : ""}</div>
        <div class="hint" style="margin-top:6px">${base ? "Values that differ from the built-in library are highlighted. " : ""}${uses.length ? `Used in this case by <b>${uses.map((x) => esc(x.label)).join(", ")}</b>: a change applies there at once.` : "Not used in this case. The Structure tab assigns a material to the matrix and to every phase."}</div></div></div>
      <div class="sec"><div class="sh">${I("grains")}Materials in this case</div><div class="sb">
        <table class="grid"><tr><th>Region</th><th>Material</th><th class="n">k</th><th class="n">E</th><th class="n">α</th></tr>${caseRows.map((r) => `<tr class="click" data-libsel="${r.e.material_id}"><td>${esc(r.label)}</td><td>${esc((mat(r.e.material_id) || {}).name || r.e.material_id)}${differs(r.e) ? ' <span class="chip" title="This case carries its own values (from a preset or an older input)">own values</span>' : ""}</td><td class="n">${fmt(r.e.props.k)}</td><td class="n">${fmt(r.e.props.E)}</td><td class="n">${fmt(r.e.props.alpha)}</td></tr>`).join("")}</table>
        <div class="hint" style="margin-top:6px">Which material each region uses is set on the Structure tab. A click on a row opens its material above.</div></div></div>
      <div class="note">${esc(state.lib.note)}</div>`;
  }

  function sizeLabel(shape) {
    return { network: "Ligament width", cube: "Edge length d", cuboid: "Edge a (d)", superellipsoid: "Length a (d)",
             polyhedron: "Circumscribed diameter d", helix: "Wire diameter d" }[shape] || "Diameter d";
  }
  function shapePreview(ph) {
    const s = ph.size, W = 300, H = 84;
    let len = +s.d, dia = +s.d;
    if (ph.shape === "spheroid") len = +s.d * +s.aspect;
    if (ph.shape === "cylinder" || ph.shape === "spherocylinder") len = +s.length;
    if (ph.shape === "cuboid" || ph.shape === "superellipsoid") { len = +s.d; dia = +s.ly; }
    if (ph.shape === "helix") { len = +s.pitch * +s.turns + +s.d; dia = +s.coil_d + +s.d; }
    if (!(len > 0 && dia > 0)) return "";
    const k = Math.min((W - 40) / Math.max(len, 1e-9), (H - 16) / Math.max(dia, 1e-9));
    const w = len * k, h = dia * k, x = (W - w) / 2, y = (H - h) / 2;
    const st = 'fill="#cfe0f7" stroke="#0b62c4" stroke-width="1.5"';
    let body = "";
    if (ph.shape === "sphere") body = `<circle cx="${W / 2}" cy="${H / 2}" r="${h / 2}" ${st}/>`;
    else if (ph.shape === "spheroid") body = `<ellipse cx="${W / 2}" cy="${H / 2}" rx="${w / 2}" ry="${h / 2}" ${st}/>`;
    else if (ph.shape === "cylinder") body = `<rect x="${x}" y="${y}" width="${w}" height="${h}" ${st}/>`;
    else if (ph.shape === "spherocylinder") body = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${h / 2}" ${st}/>`;
    else if (ph.shape === "cube" || ph.shape === "cuboid") body = `<rect x="${x}" y="${y}" width="${w}" height="${h}" ${st}/>`;
    else if (ph.shape === "superellipsoid") body = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${Math.min(w, h) / 2 * Math.max(0.05, 2 / Math.max(+s.n || 2, 2))}" ${st}/>`;
    else if (ph.shape === "polyhedron") { const r = h / 2, cx = W / 2, cy = H / 2; const n = { tetrahedron: 3, octahedron: 4, dodecahedron: 5, icosahedron: 6, irregular: 7 }[s.poly] || 6;
      const pts = Array.from({ length: n }, (_, k) => { const a = 2 * Math.PI * k / n - Math.PI / 2; const rr = s.poly === "irregular" ? r * (0.75 + 0.25 * ((k * 37) % 7) / 6) : r; return `${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`; }).join(" ");
      body = `<polygon points="${pts}" ${st}/>`; }
    else if (ph.shape === "helix") { const turns = Math.max(0.5, +s.turns || 1), pts = [];
      for (let t = 0; t <= 200; t++) { const u = t / 200; pts.push(`${x + u * (w - +s.d * k) + +s.d * k / 2},${H / 2 + (h - +s.d * k) / 2 * Math.sin(2 * Math.PI * turns * u)}`); }
      body = `<polyline points="${pts.join(" ")}" fill="none" stroke="#0b62c4" stroke-width="${Math.max(1.5, +s.d * k)}" stroke-linecap="round" opacity=".85"/>`; }
    else body = `<path d="M30 56 C70 10 110 76 150 38 S230 10 270 48 C230 78 190 38 150 70 S60 82 30 56Z" ${st}/>`;
    let shell = "";
    // The engine takes d as the particle inside the shell and adds the shell
    // outside it (outer diameter d + 2t). The sketch shows the complete particle
    // with the core at its true share, rather than cutting the shell out of d.
    if (ph.shell && ph.shell.enabled && ph.shape === "sphere") {
      const t = Math.max(+ph.shell.thickness || 0, 0), d = Math.max(+s.d || 0, 1e-9);
      shell = `<circle cx="${W / 2}" cy="${H / 2}" r="${h / 2 * d / (d + 2 * t)}" fill="#fff" stroke="#6e7781" stroke-dasharray="3 2"/>`;
    }
    const shT = ph.shell && ph.shell.enabled ? +ph.shell.thickness || 0 : 0;
    const txt = ph.shape === "network" ? `ligament width ${s.d} ${u(s.unit)}` : `d ${s.d} ${u(s.unit)}` + (ph.shape === "spheroid" ? ` · c/a ${s.aspect}` : "") + (["cylinder", "spherocylinder"].includes(ph.shape) ? ` · L ${s.length} ${u(s.unit)}` : "")
      + (["cuboid", "superellipsoid"].includes(ph.shape) ? ` × ${s.ly} × ${s.lz} ${u(s.unit)}` : "") + (ph.shape === "superellipsoid" ? ` · n ${s.n}` : "")
      + (ph.shape === "polyhedron" ? ` · ${(POLY_KINDS[s.poly] || "").toLowerCase()}` : "") + (ph.shape === "helix" ? ` · coil ${s.coil_d} · pitch ${s.pitch} · ${s.turns} turns` : "")
      + (shT > 0 && ph.shape !== "network" ? ` · outer ${fmt(+s.d + 2 * shT)} ${u(s.unit)}` : "");
    return `<svg viewBox="0 0 ${W} ${H + 16}" width="100%">${body}${shell}<text x="${W / 2}" y="${H + 12}" font-size="11" text-anchor="middle" fill="#424a53">${esc(txt)}</text></svg>`;
  }
  function phaseEditor(i) {
    const ph = state.form.phases[i];
    if (!ph) return "";
    const isNet = ph.shape === "network", m = mat(ph.material_id), vd = isVoid(ph), un = u(ph.size.unit);
    const orientable = !isNet && ph.shape !== "sphere", sh = ph.shell || {};
    const col = phaseColors()[i];
    return `<div class="sec"><div class="sh" style="--ic:${col}">${I(SHAPES[ph.shape].icon)}<span>${esc(ph.name)}</span><div class="grow"></div>
        <button class="btn sm" data-act="dup" title="Duplicate">${I("duplicate")}</button><button class="btn sm danger" data-act="del" title="Delete">${I("delete")}</button></div><div class="sb">
      <label class="f">Name</label><input data-k="phases.${i}.name">
      <label class="f">Material ${vd ? '<span class="u">pore space (gas)</span>' : ""}</label><select data-mat="phases.${i}">${materialOptions(ph.material_id)}</select>
      <div class="row wrap" style="gap:6px;margin-top:4px">${vd ? `<span class="chip" style="--ic:var(--c-flow)">${I("pores")}Pore phase — its geometry is the pore shape</span>` : `<button class="btn sm" data-act="makePore" title="Set the material of this phase to Air: its geometry becomes pore space">${I("pores")}Make it pore space (Air)</button>`}</div>
      <label class="f">Shape</label><div class="shapes">${Object.entries(SHAPES).map(([k, s]) => `<button class="shape ${ph.shape === k ? "on" : ""}" data-shape="${k}" title="${s.hint}">${I(s.icon)}<span>${s.label}</span></button>`).join("")}</div>
      <div class="hint" style="margin-top:3px">${SHAPES[ph.shape].hint}</div>
      <div class="g2">
        <div><label class="f">Content</label><div class="row" style="gap:4px"><input type="number" step="any" data-k="phases.${i}.fraction.value" data-num><select data-k="phases.${i}.fraction.basis" style="width:74px"><option value="vol">vol %</option><option value="wt">wt %</option></select></div></div>
        <div><label class="f">Length unit</label><select data-k="phases.${i}.size.unit"><option value="nm">nm</option><option value="um">µm</option><option value="mm">mm</option></select></div>
        <div><label class="f">${sizeLabel(ph.shape)} <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.size.d" data-num></div>
        ${ph.shape === "spheroid" ? `<div><label class="f">Aspect ratio c/a</label><input type="number" step="any" data-k="phases.${i}.size.aspect" data-num></div>` : ""}
        ${["cylinder", "spherocylinder"].includes(ph.shape) ? `<div><label class="f">Total length <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.size.length" data-num></div>` : ""}
        ${["cuboid", "superellipsoid"].includes(ph.shape) ? `<div><label class="f">Edge b <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.size.ly" data-num></div>
          <div><label class="f">Edge c <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.size.lz" data-num></div>` : ""}
        ${ph.shape === "superellipsoid" ? `<div><label class="f">Blockiness n <span class="u">2 … 40</span></label><input type="number" step="any" data-k="phases.${i}.size.n" data-num></div>` : ""}
        ${ph.shape === "polyhedron" ? `<div><label class="f">Polyhedron</label><select data-k="phases.${i}.size.poly">${Object.entries(POLY_KINDS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></div>` : ""}
        ${ph.shape === "polyhedron" && ph.size.poly === "irregular" ? `<div><label class="f">Vertices per grain</label><input type="number" step="1" data-k="phases.${i}.size.n_vertices" data-num></div>
          <div><label class="f">Shape variants</label><input type="number" step="1" data-k="phases.${i}.size.variants" data-num></div>` : ""}
        ${ph.shape === "helix" ? `<div><label class="f">Coil diameter <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.size.coil_d" data-num></div>
          <div><label class="f">Pitch (rise per turn) <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.size.pitch" data-num></div>
          <div><label class="f">Number of turns</label><input type="number" step="any" data-k="phases.${i}.size.turns" data-num></div>` : ""}
        ${!isNet ? `<div><label class="f">Size distribution</label><select data-k="phases.${i}.dist.type"><option value="mono">Monodisperse</option><option value="lognormal">Log-normal</option></select></div>` : ""}
        ${!isNet && ph.dist.type === "lognormal" ? `<div><label class="f">Coefficient of variation</label><input type="number" step="0.05" data-k="phases.${i}.dist.cv" data-num></div>` : ""}
        ${orientable ? `<div><label class="f">Orientation</label><select data-k="phases.${i}.orientation.mode"><option value="iso">Random 3D</option><option value="x">Aligned with x</option><option value="y">Aligned with y</option><option value="z">Aligned with z</option><option value="xy">Random in the xy plane</option></select></div>
          <div><label class="f">Orientation spread <span class="u">°</span></label><input type="number" step="any" data-k="phases.${i}.orientation.spread_deg" data-num></div>` : ""}
        ${!isNet && !ph.overlap ? `<div><label class="f">Minimum gap <span class="u">× d</span></label><input type="number" step="0.01" data-k="phases.${i}.gap_frac" data-num></div>` : ""}
      </div>
      ${!vd ? `<div class="subh">Interfacial thermal resistance</div><div class="g2">
          <div><label class="f">R<sub>int</sub> filler–matrix <span class="u">m²K/W</span></label><input type="number" step="any" data-k="phases.${i}.r_int" data-num></div>
          ${!isNet ? `<div><label class="f">R<sub>c</sub> filler–filler <span class="u">m²K/W</span></label><input type="number" step="any" data-k="phases.${i}.r_contact" data-num></div>` : ""}</div>
        <div class="hint">Kapitza resistance on the interface, applied on the voxel faces where the regions meet (thermal conductivity only). Typical values 1e-9 … 1e-7 m²K/W (boundary conductance 1e9 … 1e7 W/m²K). A filler smaller than the critical radius R<sub>int</sub>·k<sub>matrix</sub>${ph.r_int > 0 ? ` = <b>${fmt(ph.r_int * (+state.form.matrix.props.k || 0) * 1e6)} µm</b>` : ""} no longer raises k. R<sub>c</sub> acts where two particles touch.</div>` : ""}
      ${orientable ? '<div class="hint">A platelet (c/a &lt; 1) aligned with z lies in the xy plane; a fibre randomly oriented in the xy plane lies in plane.</div>' : ""}
      ${!isNet ? `<label class="chk"><input type="checkbox" data-k="phases.${i}.overlap"> Overlap permitted (agglomerates, fibre networks)</label>` : ""}
      ${!isNet ? `<label class="chk"><input type="checkbox" data-k="phases.${i}.shell.enabled"> Coating or hollow shell</label>` : ""}
      ${!isNet && sh.enabled ? `<div class="g2"><div><label class="f">Shell thickness <span class="u">${un}</span></label><input type="number" step="any" data-k="phases.${i}.shell.thickness" data-num></div>
          <div><label class="f">Shell material</label><select data-mat="phases.${i}.shell">${materialOptions(sh.material_id)}</select></div></div>
          ${(ph.dist || {}).type && (ph.dist || {}).type !== "mono" ? `<label class="chk"><input type="checkbox" data-k="phases.${i}.shell.scale_with_size"> Wall scales with particle size (constant t/R)</label>
          <div class="hint">${sh.scale_with_size ? `The thickness above belongs to a particle of the nominal diameter; every other particle keeps the same wall-to-radius ratio, so all share one hollow fraction — as in one density grade of glass bubbles.` : `Every particle gets the same wall, so small particles are relatively thick-walled and large ones mostly air.`}</div>` : ""}
          <div class="hint">The sizes above are the particle <b>inside</b> the shell; the shell is added outside it${ph.shape === "polyhedron" ? " (every face moves out by the thickness)" : `, so the outer ${ph.shape === "helix" ? "wire diameter" : "size"} is d + 2 × thickness = <b>${fmt(+(ph.size || {}).d + 2 * (+sh.thickness || 0))} ${un}</b>`}. A hollow particle quoted by its outer size needs d = outer − 2 × wall. Hollow particles: phase material Air, shell material SiO₂. The content refers to the complete particle, shell included.</div>
          <details class="more"><summary>Shell properties · ${esc((mat(sh.material_id) || {}).name || "")}</summary>${matSummary(`phases.${i}.shell`, sh)}</details>` : ""}
      <details class="more"><summary>Properties · ${esc(m ? m.name : "")}</summary>${m && m.note ? `<div class="hint">${esc(m.note)}</div>` : ""}${matSummary(`phases.${i}`, ph)}</details>
      <div class="preview">${shapePreview(ph)}</div>
    </div></div>`;
  }
  // the properties a region takes from its material, read-only here: they are
  // edited once, in the library on the Material tab
  function matSummary(path, entry) {
    const own = differs(entry);
    return `<div class="props">${PROPS.map(([k, lab, unit]) => `<div class="prop"><span>${lab}<em>${unit}</em></span><b class="mono pv">${fmt(entry.props[k])}</b></div>`).join("")}</div>
      <div class="row wrap" style="gap:6px;margin-top:7px"><button class="btn sm" data-editmat="${entry.material_id}">${I("matrix")}Edit this material</button>${own ? `<button class="btn sm" data-uselib="${path}">${I("reset")}Use library values</button>` : ""}</div>
      <div class="hint">${own ? "This case carries its own values for this material (from a preset or an older input); they differ from the library. " : ""}Properties are edited on the Material tab and apply to every region that uses the material.</div>`;
  }
  function matrixEditor() {
    const mx = state.form.matrix, m = mat(mx.material_id), vd = isVoid(mx);
    return `<div class="sec"><div class="sh" style="--ic:${vd ? "#7fd3f7" : PHASE_COLORS[0]}">${I("matrix")}<span>${esc(mx.name)}</span><div class="grow"></div><span class="chip">matrix</span></div><div class="sb">
      <label class="f">Name</label><input data-k="matrix.name">
      <label class="f">Material ${vd ? '<span class="u">gas: the phases form a packed bed or a porous solid</span>' : ""}</label><select data-mat="matrix">${materialOptions(mx.material_id)}</select>
      ${m && m.note ? `<div class="hint" style="margin-top:4px">${esc(m.note)}</div>` : ""}
      <div class="hint" style="margin-top:4px">The matrix fills everything the phases leave free.</div>
      <details class="more" open><summary>Properties · ${esc(m ? m.name : "")}</summary>${matSummary("matrix", mx)}</details>
    </div></div>`;
  }
  function leftStructure() {
    const f = state.form, cols = phaseColors();
    if (state.sel >= f.phases.length) state.sel = Math.max(0, f.phases.length - 1);
    const rows = [`<div class="prow ${state.sel < 0 ? "on" : ""}" data-sel="-1"><i class="sw2" style="background:${isVoid(f.matrix) ? "#7fd3f7" : PHASE_COLORS[0]}"></i>${I("matrix")}<div class="nm">${esc(f.matrix.name)}<small>matrix${(mat(f.matrix.material_id) || {}).name && mat(f.matrix.material_id).name !== f.matrix.name ? " · " + esc(mat(f.matrix.material_id).name) : ""}${isVoid(f.matrix) ? " · gas" : ""}</small></div><span class="vf" id="vfMatrix"></span></div>`]
      .concat(f.phases.map((p, i) => `<div class="prow ${i === state.sel ? "on" : ""}" data-sel="${i}"><i class="sw2" style="background:${cols[i]}"></i>${I(SHAPES[p.shape].icon)}<div class="nm">${esc(p.name)}<small>${SHAPES[p.shape].label.toLowerCase()} · ${p.shape === "network" ? "width" : "d"} ${p.size.d} ${u(p.size.unit)}${isVoid(p) ? " · void" : ""}</small></div><span class="vf">${fmt(p.fraction.value)} ${p.fraction.basis === "wt" ? "wt%" : "vol%"}</span></div>`));
    const add = PARTICLE_SHAPES.map((s) => `<button class="btn sm" data-add="${s}" title="Add ${SHAPES[s].label.toLowerCase()} phase">${I(SHAPES[s].icon)}</button>`).join("") +
      `<button class="btn sm" data-add="network" title="Add a bicontinuous network (any material)">${I("network")}</button>`;
    return `<div class="row wrap" style="margin-bottom:8px"><span class="hint">Add</span>${add}</div>
      <div class="hint" style="margin:-4px 0 8px">A phase is pore space when its material is a gas (Air) — any shape can be a pore. The matrix can be Air too: the phases then form a packed bed or a porous solid.</div>
      <div class="plist">${rows.join("")}</div>
      ${state.sel < 0 ? matrixEditor() : f.phases.length ? phaseEditor(state.sel) : '<div class="note">No phases are defined; only the matrix is analysed. Phases are added with the buttons above or in the ribbon.</div>'}
      ${contactTableHTML()}
      <div class="sec"><div class="sh">${I("grains")}Touching particles</div><div class="sb">
        <select data-k="options.contacts"><option value="apart">Keep apart where the loading allows</option><option value="keep">Allow contacts</option><option value="separate">Trim contacts to matrix (v3)</option></select>
        <div class="hint" style="margin-top:4px">${{
          separate: "Every voxel where two non-overlapping particles touch is cleared to matrix. No contacts remain, but the particles are trimmed.",
          keep: "Particles are placed as the packing puts them and may touch. Shapes are exact; the touching voxel faces are counted and can carry a contact resistance R<sub>c</sub>.",
        }[(f.options || {}).contacts] || "Particles are moved (never trimmed) until no voxel of one is next to a voxel of another, with a 2-voxel gap where it fits. A dense loading that cannot keep every particle apart lets them touch only where needed; the remaining contacts are counted and can carry R<sub>c</sub>. Near-contacts one voxel wide are what a voxel solver resolves worst."}</div></div></div>
      <div class="sec"><div class="sh">${I("morphology")}Composition</div><div class="sb" id="compBody"></div></div>`;
  }
  /* R_c for every pair of fillers. The diagonal is each filler's own R_c
     (the same field as on its card); between two different fillers the value
     is set here, or left empty to take the mean of the two. */
  function contactTableHTML() {
    const f = state.form;
    const idx = f.phases.map((p, i) => i).filter((i) => f.phases[i].shape !== "network" && !isVoid(f.phases[i]));
    if (idx.length < 2) return "";
    f.contact_rc = f.contact_rc || {};
    const cell = (i, j) => {
      if (i === j) return `<td><input type="number" step="any" data-k="phases.${i}.r_contact" data-num title="R_c between two particles of ${esc(f.phases[i].name)}"></td>`;
      const a = Math.min(i, j), b = Math.max(i, j);
      const mean = 0.5 * ((+f.phases[a].r_contact || 0) + (+f.phases[b].r_contact || 0));
      if (i > j) { const v = f.contact_rc[`${a}-${b}`]; return `<td class="n hint">${v !== undefined && v !== "" && v !== null ? fmt(+v) : fmt(mean)}</td>`; }
      return `<td><input type="number" step="any" data-k="contact_rc.${a}-${b}" data-num placeholder="${fmt(mean)} (mean)" title="R_c between ${esc(f.phases[a].name)} and ${esc(f.phases[b].name)}"></td>`;
    };
    return `<div class="sec"><div class="sh">${I("grains")}Contact resistance between fillers<small>R<sub>c</sub>, m²K/W</small></div><div class="sb">
      <table class="grid rcgrid"><tr><th></th>${idx.map((j) => `<th>${esc(f.phases[j].name)}</th>`).join("")}</tr>
        ${idx.map((i) => `<tr><th>${esc(f.phases[i].name)}</th>${idx.map((j) => cell(i, j)).join("")}</tr>`).join("")}</table>
      <div class="hint">Acts on the voxel faces where two particles touch (thermal conductivity). The diagonal is a filler against itself; an empty cell between two fillers takes the mean of their own values. The result lists how many contact faces each pair has.</div></div></div>`;
  }

  function renderComposition() {
    const el = $("#compBody");
    if (!el) return;
    if (state.planErrors.length) { el.innerHTML = `<div class="errs">${esc(state.planErrors.join("\n"))}</div>`; return; }
    if (!state.plan) { el.innerHTML = '<div class="hint">Computing…</div>'; return; }
    const c = state.plan.composition;
    const vm = $("#vfMatrix"); if (vm) vm.textContent = `${fmt(100 * c.vf_matrix)} vol%`;
    let h = `<table class="grid"><tr><th>Region</th><th class="n">vol %</th><th class="n">wt %</th></tr>
      <tr><td>${esc(state.form.matrix.name)}</td><td class="n">${fmt(100 * c.vf_matrix)}</td><td class="n">${fmt(100 * c.wt_matrix)}</td></tr>`;
    state.form.phases.forEach((p, i) => { h += `<tr><td>${esc(p.name)}</td><td class="n">${fmt(100 * c.vf[i])}</td><td class="n">${fmt(100 * c.wt[i])}</td></tr>`; });
    h += `</table><div class="hint" style="margin-top:5px">Density ${fmt(c.density)} g/cm³ · specific heat ${fmt(c.cp)} J/kg·K</div>`;
    el.innerHTML = h;
  }

  function leftDomain() {
    const f = state.form, pl = state.plan && state.plan.plan;
    let tiles = "", notes = "", mem = "", counts = "";
    if (pl) {
      const vol = pl.L_um * pl.L_um * (pl.film ? pl.T_um : pl.L_um);
      const nPart = pl.features.filter((x) => x.mean_volume).reduce((s, x) => s + x.vf * vol / x.mean_volume, 0);
      const tile = (v, l, d) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div><div class="d" title="${esc(d)}">${esc(d || "")}</div></div>`;
      tiles = `<div class="tiles">${tile(gridTxt(pl), "Grid", `${fmt(nVox(pl) / 1e6)} M voxels · ${pl.budget ? `limit ${pl.budget}³` : "no grid limit"}`)}${tile(`${fmt(pl.h_um)} µm`, "Voxel size", `recommended ${fmt(pl.recommended.h_um)} µm`)}${tile(`${fmt(pl.L_um)} µm`, pl.film ? `Lateral edge (film ${fmt(pl.T_um)} µm)` : "RVE edge length", pl.recommended.reason)}${tile(pl.features.some((x) => x.mean_volume) ? fmt(Math.round(nPart)) : "—", "Expected particles", `${pl.seeds} realisation(s)`)}</div>`;
      notes = pl.notes.map((n) => `<div class="note warn">${esc(n)}</div>`).join("");
      mem = `<div class="hint">Estimated peak memory: ${Object.entries(pl.memory_gb).map(([k, v]) => `${k} ${fmt(v)} GB`).join(" · ")}</div>`;
      const cc = { pass: 0, warn: 0, fail: 0 }; pl.checks.forEach((c) => cc[c.status]++);
      counts = `<div class="row wrap">${pill("pass")} ${cc.pass} ${pill("warn")} ${cc.warn} ${pill("fail")} ${cc.fail}<div class="grow"></div><button class="btn sm" data-act="checks">${I("checks")}Show all checks</button></div>
        ${pl.checks.filter((c) => c.status !== "pass").map((c) => `<div class="hint" style="margin-top:4px">${pill(c.status)} ${esc(c.label)}: ${fmt(c.value)} (criterion ${Array.isArray(c.target) ? c.target.map(fmt).join("–") : fmt(c.target)})</div>`).join("")}`;
    } else if (state.planErrors.length) notes = `<div class="errs">${esc(state.planErrors.join("\n"))}</div>`;
    return `<div class="sec"><div class="sh">${I(f.rve.quality)}Accuracy level</div><div class="sb">
        <div class="seg" id="qualitySeg">${["fast", "standard", "accurate"].map((q) => `<button data-q="${q}" class="${f.rve.quality === q ? "on" : ""}">${q[0].toUpperCase() + q.slice(1)}</button>`).join("")}</div>
        <div class="hint" style="margin-top:6px">${QUALITY_HINT[f.rve.quality]}</div></div></div>
      <div class="sec"><div class="sh">${I("cube")}Domain</div><div class="sb">
        <div class="seg" id="domainSeg"><button data-film="0" class="${f.rve.film ? "" : "on"}">Periodic bulk</button><button data-film="1" class="${f.rve.film ? "on" : ""}">Thin film</button></div>
        ${f.rve.film ? `<div class="g2" style="margin-top:6px"><div><label class="f">Film thickness (z)</label><div class="row"><input type="number" step="any" data-k="rve.T_um" data-uk="rve.T_unit" data-num style="flex:1"><select data-k="rve.T_unit" style="width:70px">${unitOpts()}</select></div></div>
            <div><label class="f">Grid through the thickness</label><div class="hint" style="padding-top:6px"><b>${pl && pl.film ? `${pl.Nz} voxels` : "—"}</b>${pl && pl.film ? ` (${fmt(pl.T_um / LEN_UNITS[f.rve.T_unit || "um"])} ${LEN_LABEL[f.rve.T_unit || "um"]})` : ""}</div></div></div>
          <div class="g2" style="margin-top:4px"><div><label class="f">Matrix skin at each face <span class="u">empty = 1 voxel</span></label><div class="row"><input type="number" step="any" min="0" data-k="rve.skin_um" data-uk="rve.T_unit" data-num placeholder="1 voxel" style="flex:1"><span class="hint" style="padding-top:6px">${LEN_LABEL[f.rve.T_unit || "um"]}</span></div></div><div></div></div>
          <div class="hint">x and y repeat periodically; z is the real film thickness. Every particle lies wholly inside the film: none is cut by the faces or continues through them, and a thin matrix skin stays at each face (0 lets particles touch the faces - the electrode plates of the through-thickness solve, through contacts whose area depends on the voxel size). Conduction and permittivity: along the film (x, y) the two faces are insulated; through it (z) two plates on the faces drive the flux, as in a measurement across the film. z is solved and displayed first.</div>`
          : `<div class="hint" style="margin-top:6px">The RVE repeats in x, y and z: a bulk material, or a layer much thicker than its particles.</div>`}</div></div>
      <div class="sec"><div class="sh">${I("auto")}Grid</div><div class="sb">
        <label class="chk"><span class="switch"><input type="checkbox" data-k="rve.auto"><span></span></span> Automatic voxel size and RVE size</label>
        ${!f.rve.auto ? `<div class="g2"><div><label class="f">${f.rve.film ? "Lateral edge length (x, y)" : "RVE edge length"}</label><div class="row"><input type="number" step="any" data-k="rve.L_um" data-uk="rve.L_unit" data-num style="flex:1"><select data-k="rve.L_unit" style="width:70px">${unitOpts()}</select></div></div>
            <div><label class="f">Voxel size</label><div class="row"><input type="number" step="any" data-k="rve.voxel_um" data-uk="rve.h_unit" data-num style="flex:1"><select data-k="rve.h_unit" style="width:70px">${unitOpts()}</select></div></div></div>
            <div class="hint" style="margin-top:4px">Grid size (computed): <b>${pl ? gridTxt(pl) : "—"}</b>${pl ? ` · ${fmt(nVox(pl) / 1e6)} M voxels · ${fmt(pl.h_um / LEN_UNITS[f.rve.h_unit || "um"])} ${LEN_LABEL[f.rve.h_unit || "um"]}/voxel` : ""}</div>
          <div class="hint">The grid size follows from the two: edge length ÷ voxel size, rounded to an even number of voxels. A grid-size variable remains in the DOE tab for convergence studies.</div>` : ""}
        <label class="chk"><input type="checkbox" data-k="rve.auto_enlarge"> Enlarge the RVE when the criteria are not met</label>
        <div class="g2"><div><label class="f">Realisations <span class="u">0 = level default</span></label><input type="number" min="0" max="10" data-k="rve.n_seeds" data-num></div><div><label class="f">Random seed</label><input type="number" min="0" data-k="rve.seed" data-num></div></div>
        <div class="hint">With two or more realisations, results carry a 95 % confidence interval.</div></div></div>
      <div class="sec"><div class="sh">${I("cube")}Recommended RVE</div><div class="sb">${tiles}${notes}${mem}</div></div>
      <div class="sec"><div class="sh">${I("checks")}Criteria before solving</div><div class="sb">${counts || '<div class="hint">Computing…</div>'}</div></div>`;
  }

  function anCard(key, options = "") {
    const a = AN[key], on = !!state.form.analyses[key], dis = disabledReason(key);
    return `<div class="sec acard ${on ? "on" : ""} ${dis ? "dis" : ""}" data-card="${key}"><div class="sh" style="--ic:${a.color}">${I(a.icon)}<span>${a.title}</span><div class="grow"></div>
        <label class="switch" title="${esc(dis || "Include in the run")}"><input type="checkbox" data-an="${key}" ${on ? "checked" : ""} ${dis ? "disabled" : ""}><span></span></label></div>
      <div class="sb"><div class="solver">${a.solver}</div><div class="outs">${a.out}</div>${dis ? `<div class="note warn">${dis}</div>` : ""}
        ${on && options ? `<div class="opts"><div class="cap">Options</div>${options}</div>` : ""}</div></div>`;
  }
  const num = (k, label, unit, step = "any") => `<div><label class="f">${label}${unit ? ` <span class="u">${unit}</span>` : ""}</label><input type="number" step="${step}" data-k="${k}" data-num></div>`;
  function gasDetails() {
    return `<details class="more"><summary>Gas properties (pores)</summary><div class="g2">${num("options.gas.viscosity_Pa_s", "Viscosity", "Pa·s")}${num("options.gas.density_kg_m3", "Density", "kg/m³")}
      ${num("options.gas.pressure_Pa", "Pressure", "Pa")}${num("options.gas.sound_speed_m_s", "Speed of sound", "m/s")}${num("options.gas.molar_mass_g_mol", "Molar mass", "g/mol")}${num("options.gas.diffusivity_m2_s", "Molecular diffusivity", "m²/s")}
      ${num("options.gas.molecule_diameter_nm", "Molecule diameter", "nm")}${num("options.gas.gamma", "Heat capacity ratio γ", "")}${num("options.gas.prandtl", "Prandtl number", "")}${num("options.temperature_K", "Temperature", "K")}</div></details>`;
  }
  function selectBar(keys) {
    const n = keys.filter((k) => state.form.analyses[k]).length;
    return `<div class="selbar"><span class="hint">${n} of ${keys.length} selected — several analyses run in one job</span>
      <button class="btn sm" data-anall="${keys.join(",")}">Select all</button><button class="btn sm" data-annone="${keys.join(",")}">Clear</button></div>`;
  }
  function leftAnalysis() {
    const po = state.form.options.porosimetry;
    const poro = `<div class="g2"><div><label class="f">Invading fluid</label><select data-k="options.porosimetry.fluid"><option value="mercury">Mercury (MIP)</option><option value="water">Water</option><option value="isopropanol">Isopropanol</option><option value="custom">Custom</option></select></div>${num("options.porosimetry.steps", "Pressure steps", "", "1")}
      ${po.fluid === "custom" ? num("options.porosimetry.surface_tension_N_m", "Surface tension", "N/m") + num("options.porosimetry.contact_angle_deg", "Contact angle", "°") : ""}</div>
      <div class="hint">Bubble points for water and isopropanol are reported for every fluid.</div>`;
    return selectBar(["morphology", "porosimetry", "pore_network", "percolation", "grains"]) +
      anCard("morphology") + anCard("porosimetry", poro) + anCard("pore_network") +
      anCard("percolation", `<div class="hint">Pore space uses face connectivity (a fluid cannot pass through a corner); solid phases use full connectivity, as a voxel finite-element model conducts through edge and corner contacts.</div>`) +
      anCard("grains") +
      `<div class="note">Structure analyses run on the first realisation. Local-thickness, pore-entry, ligament, percolation-class and geodesic-path fields become available in the 3D and 2D views, and the percolation analysis adds the route of the largest sphere that passes, with the sphere drawn at its true size.</div>`;
  }
  function leftSimulation() {
    const o = state.form.options;
    const dirs = o.directions || "xyz";
    const general = `<div class="sec"><div class="sh">${I("solver")}Solver settings</div><div class="sb">
        <label class="f">Load directions</label><div class="row">${["z", "x", "y"].map((d) => `<label class="chk inline"><input type="checkbox" data-dir="${d}" ${dirs.includes(d) ? "checked" : ""}> ${d}${d === "z" ? " (through-thickness)" : ""}</label>`).join("")}</div>
        <div class="hint">z is solved first; its fields, arrows and field lines are the ones kept for the 3D and 2D views.</div>
        <label class="f">Solver</label><select data-k="options.backend"><option value="puma">NASA PuMA (default)</option><option value="puma_fv">NASA PuMA · FV with fixed faces (fast)</option><option value="builtin">Built-in periodic FV only</option></select>
        ${o.backend === "puma" || !o.backend ? `<label class="f">Conduction (k, σ, Dk, μ<sub>r</sub>)</label><select data-k="options.cond_method"><option value="fv">Periodic finite volume (default, more accurate)</option><option value="fe">PuMA periodic finite elements</option></select>
        <label class="chk"><span class="switch"><input type="checkbox" data-k="options.crosscheck"><span></span></span> Also solve with the other method and compare</label>
        <div class="hint">Each problem is solved once. On voxel grids the finite-volume result came 2–3× closer to converged reference values than the finite-element one, which conducts through voxels that touch only at an edge or corner (guide 7.2). The comparison doubles the conduction time and shows the discretisation error. Flow uses PuMA's finite elements.</div>` : ""}
        <label class="f">Elasticity and CTE</label><select data-k="options.elastic_method"><option value="fans">Voxel FE, FFT-preconditioned (FANS, default)</option><option value="puma">NASA PuMA FE (MINRES)</option><option value="fft">FFT, Moulinec–Suquet</option></select>
        <div class="hint">FANS solves the same voxel finite elements as PuMA (identical results) with conjugate gradients preconditioned by the exact inverse of a homogeneous reference medium. Its iteration count does not grow with the grid (about 60 for 10<sup>-6</sup>), where PuMA's MINRES needs more iterations the larger the grid.</div>
        <label class="f">Resolution check</label><select data-k="options.resolution_check"><option value="auto">When particles touch at voxel level (default)</option><option value="on">Always</option><option value="off">Off</option></select>
        <div class="hint">The same particles are drawn on a 1.5× finer grid and the conduction-type properties solved once more in the first direction; the result shows the change and a value extrapolated to fine voxels.</div>
        <label class="chk"><span class="switch"><input type="checkbox" data-k="options.reuse_runs"><span></span></span> Reuse the structure of a matching run</label>
        <div class="hint">A run keeps the structure it generated. Another run with the same geometry, grid and seed uses it instead of generating it again, and when nothing but the choice of analyses differs it continues that run: the analyses already there are kept and only the new ones are computed.</div>
        <label class="f">Independent solves</label><select data-k="options.parallel_solves"><option value="auto">Side by side on the run's cores (default)</option><option value="off">One after another</option></select>
        <div class="hint">Each property, each load direction and each elastic load case is its own problem. A run splits its ${((state.settings || {}).threads) || 4} cores between these problems and the threads inside each one, whichever finishes first (13 problems on 16 cores: 8 side by side × 2 threads), within the free memory and longest first.</div>
        ${state.form.rve.film ? `${num("options.tol", "Tolerance", "")}<div class="hint">Area-specific resistance, sheet resistance and capacitance refer to the film thickness set on the Domain tab (${fmt(state.form.rve.T_um)} µm).</div></div></div>`
          : `<div class="g2">${num("options.tol", "Tolerance", "")}${num("options.thickness_mm", "Layer thickness", "mm")}</div>
        <div class="hint">The layer thickness converts effective properties into area-specific resistance, sheet resistance and capacitance.</div></div></div>`}`;
    return general + selectBar(["thermal", "electrical", "dielectric", "magnetic", "emi", "cte", "permeability", "filtration", "tortuosity", "acoustics", "radiation", "viscosity"]) +
      anCard("thermal", `<label class="chk"><span class="switch"><input type="checkbox" data-k="options.knudsen"><span></span></span> Rarefied gas in small pores (Knudsen)</label>${num("options.temperature_K", "Gas temperature", "K")}
        <div class="hint">${state.form.phases.some((p) => (+p.r_int || 0) > 0 || (+p.r_contact || 0) > 0) ? `Interfacial resistances are set (${state.form.phases.filter((p) => (+p.r_int || 0) > 0 || (+p.r_contact || 0) > 0).map((p) => esc(p.name)).join(", ")}): the thermal solve runs on the built-in periodic finite-volume solver, which carries a resistance on every interface face.` : "Interfacial thermal resistances (filler–matrix R<sub>int</sub>, filler–filler contact R<sub>c</sub>) are set per phase on the Structure tab."}</div>`) +
      anCard("electrical") + anCard("dielectric") + anCard("magnetic") +
      anCard("emi", `<div class="g3">${num("options.emi.thickness_mm", "Thickness", "mm")}${num("options.emi.f_min_hz", "f min", "Hz")}${num("options.emi.f_max_hz", "f max", "Hz")}</div>
        <div class="g2"><div><label class="f">Method</label><select data-k="options.emi.method"><option value="complex">Frequency-resolved (σ + jωε on the RVE)</option><option value="static">Static σ and ε (earlier method)</option></select></div>${num("options.emi.n_freq", "Solved frequencies", "", "1")}</div>
        <div class="hint">Frequency-resolved: the RVE is solved with the complex admittivity of every region at each frequency, which captures the interfacial polarisation of conducting fillers and metal particles acting as conductors. Each frequency costs about one conduction solve per in-plane direction.</div>
        <label class="chk"><span class="switch"><input type="checkbox" data-k="options.emi.fullwave"><span></span></span> Full-wave simulation with openEMS (10-100 GHz)</label>
        <div class="hint">On by default. Simulates the RVE itself as a slab with Maxwell's equations (FDTD, voxel by voxel, no homogenisation) and compares its shielding with the homogenised slab. Takes minutes. The two can only agree while conducting fillers are thinner than their skin depth (see Validity) and the shielding stays below about 80 dB.</div>`) +
      anCard("cte", `${num("options.delta_T", "ΔT for thermal stresses", "K")}<label class="f">Stress maps</label><select data-k="options.thermal_bc"><option value="constrained">Constrained — zero mean strain (layer held rigidly)</option><option value="free">Free expansion — zero mean stress (phase mismatch only)</option></select><div class="hint">Constrained adds the stress of holding the whole composite still; free expansion leaves only what the phases do to each other, which is what a filler feels inside a free film. α and the stiffness do not depend on this choice.</div>
        <label class="chk"><span class="switch"><input type="checkbox" data-k="options.above_tg"><span></span></span> Also above Tg (rubbery resin: E above Tg, α2)</label>
        <div class="hint">Solves elasticity and CTE a second time with every material that has a glass transition in its rubbery state (optional properties E above Tg and CTE above Tg, ν 0.45, B-bar elements): the α2 and hot modulus of an EMC or underfill, which set warpage at reflow.</div>`) +
      anCard("permeability", gasDetails()) +
      anCard("viscosity", viscosityOptions()) +
      anCard("moisture", `<div class="g3">${num("options.moisture.thickness_mm", "Thickness", "mm")}${num("options.moisture.hours", "Exposure", "h")}<div><label class="f">Exposed faces</label><select data-k="options.moisture.sides"><option value="both">Both</option><option value="one">One (the other sealed)</option></select></div></div>
        <div class="hint">Uses the optional moisture properties of each material (Material tab): diffusivity D_w and saturated uptake c_sat at the test condition (e.g. 85 °C / 85 % RH) and the swelling coefficient CME. A material without them takes up no water. The uptake curve is Crank's solution for a plate of the given thickness; 168 h is the JEDEC level-1 soak.</div>`) +
      anCard("filtration", `<div class="g3">${num("options.filtration.face_velocity_m_s", "Face velocity", "m/s")}${num("options.filtration.n_particles", "Particles per size", "", "1")}${num("options.filtration.particle_density_kg_m3", "Particle density", "kg/m³")}</div>
        <div class="g3">${num("options.filtration.d_min_um", "Smallest particle", "µm")}${num("options.filtration.d_max_um", "Largest particle", "µm")}${num("options.filtration.n_sizes", "Sizes", "", "1")}</div>
        <div class="hint">Includes permeability automatically. Particles are followed through the solved flow with inertia, Brownian motion and interception; the pressure drop and the quality factor come from the computed K.</div>`) +
      anCard("tortuosity") +
      anCard("acoustics", `<div class="g3">${num("options.acoustics.thickness_mm", "Thickness", "mm")}${num("options.acoustics.f_min_hz", "f min", "Hz")}${num("options.acoustics.f_max_hz", "f max", "Hz")}</div><div class="hint">Includes permeability and diffusion automatically.</div>`) +
      anCard("radiation", `<div class="g2">${num("options.radiation.temperature_K", "Temperature", "K")}${num("options.radiation.refractive_index", "Refractive index", "")}${num("options.radiation.sources", "Ray sources", "", "1")}${num("options.radiation.rays", "Rays per source", "", "1")}</div>`) +
      `<div class="note">Problem size: ${state.plan ? `${gridTxt(state.plan.plan)} voxels, ${state.plan.plan.seeds} realisation(s)` : "—"}. Elasticity requires seven load cases and dominates the run time; server grid limits apply per solver. Flux, current, field and velocity solutions are stored as vectors, so the 3D and 2D views can show arrows and field lines beside the contour.</div>`;
  }
  function runsHTML() {
    if (!state.jobs.length) return '<div class="empty">' + I("runs") + '<div>No runs yet.</div></div>';
    // what each run did: a structure only, or which analyses (one chip each),
    // its grid and its phases; a DOE case shows its variable values instead
    const chips = (j) => j.kind === "calibration" ? `<span class="rtag" style="--c:#8250df">Calibration</span>`
      : (j.kind === "structure" || !(j.analyses || []).length)
      ? `<span class="rtag struct">Structure</span>`
      : (j.analyses || []).map((a) => `<span class="rtag" style="--c:${(AN[a] || {}).color || "var(--accent)"}">${esc(SHORT_AN[a] || (AN[a] || {}).title || a)}</span>`).join("");
    const one = (j) => `<div class="run ${j.id === state.resultJob ? "on" : ""}" data-open="${j.id}">
      <i class="dot ${j.state}"></i><div class="nm">${esc(j.name || j.id)}
        <div class="rtags">${chips(j)}${j.grid ? `<span class="rgrid">${esc(j.grid)}</span>` : ""}</div>
        ${j.doe_label ? `<small class="rvars">${esc(j.doe_label)}</small>` : (j.structure ? `<small class="rstruct" title="${esc(j.structure)}">${esc(j.structure)}</small>` : "")}
        <small>${STATE[j.state]}${j.state === "running" ? " · " + Math.round(100 * j.progress) + " %" : ""}${j.state === "queued" ? " · position " + j.queue_position : ""} · ${secs(j.elapsed_s)}</small></div>
      <div class="acts">${["running", "queued"].includes(j.state) ? `<button class="btn sm danger" data-stop="${j.id}" title="Stop">${I("stop")}</button>` : `<button class="btn sm" data-del="${j.id}" title="Delete">${I("delete")}</button>`}</div></div>`;
    // A study queues up to 64 cases. Listed one by one they bury every earlier
    // result, so a study collapses into a single row that opens on a click.
    const seen = new Set(), out = [];
    state.jobs.forEach((j) => {
      if (!j.doe) { out.push(one(j)); return; }
      if (seen.has(j.doe)) return;
      seen.add(j.doe);
      const cases = state.jobs.filter((x) => x.doe === j.doe);
      const n = (st) => cases.filter((x) => x.state === st).length;
      const open = !!state.openDoe[j.doe];
      const study = (state.doeList || []).find((s) => s.id === j.doe);
      const live = n("running") + n("queued");
      const dot = live ? "running" : (n("failed") + n("stopped") ? "failed" : "done");
      const el = cases.reduce((s, x) => s + (x.elapsed_s || 0), 0);
      out.push(`<div class="run study ${open ? "open" : ""}" data-doegroup="${j.doe}">
        <i class="dot ${dot}"></i><div class="nm">${esc((study && study.name) || "DOE study")}
          <small>DOE · ${cases.length} case${cases.length === 1 ? "" : "s"} · ${n("done")} done${live ? ` · ${live} in flight` : ""}${n("failed") + n("stopped") ? ` · ${n("failed") + n("stopped")} failed` : ""} · ${secs(el)}</small></div>
        <div class="acts"><button class="btn sm" data-doeopen2="${j.doe}" title="Open the study">${I("preset")}</button>
          <button class="btn sm" data-doetoggle="${j.doe}" title="${open ? "Collapse" : "Show the cases"}">${open ? "−" : "+"}</button></div></div>`
        + (open ? `<div class="substack">${cases.map(one).join("")}</div>` : ""));
    });
    return out.join("");
  }
  function leftResults() {
    const r = state.result;
    let sel = '<div class="hint">No completed run is selected.</div>';
    if (r) {
      sel = `<div style="font-weight:700">${esc(r.name)}</div><div class="hint">${esc(r.created)} · ${secs(r.elapsed_s)} · ${backendTxt(r)}</div>
        <div class="tiles">${(r.headline || []).filter((h) => !h.dir).slice(0, 8).map((h) => `<div class="tile"><div class="v">${fmt(h.value)}</div><div class="l">${esc(h.label)}</div><div class="d">${esc(h.unit)}</div></div>`).join("")}</div>
        <div class="row wrap">${pill("pass")} ${r.check_counts.pass} ${pill("warn")} ${r.check_counts.warn} ${pill("fail")} ${r.check_counts.fail}</div>
        <div class="row wrap" style="margin-top:8px"><button class="btn sm primary" data-center="res">${I("table")}Result Viewer</button><button class="btn sm" data-center="vis">${I("view3d")}3D view</button><button class="btn sm" data-center="rep">${I("report")}Report</button><button class="btn sm" data-editspec>${I("open")}Edit this input</button></div>`;
    }
    return `<div class="sec"><div class="sh">${I("table")}Selected run</div><div class="sb">${sel}</div></div><div class="sec"><div class="sh">${I("runs")}Runs</div><div class="sb">${runsHTML()}</div></div>`;
  }

  /* ================================================================ inputs */
  // lengths stored in µm and shown in the unit chosen next to the input
  const LEN_UNITS = { nm: 1e-3, um: 1, mm: 1e3, cm: 1e4 };
  const LEN_LABEL = { nm: "nm", um: "µm", mm: "mm", cm: "cm" };
  const unitOpts = () => Object.keys(LEN_UNITS).map((u) => `<option value="${u}">${LEN_LABEL[u]}</option>`).join("");
  // a cube N³, a film N × N × Nz
  const gridTxt = (pl) => (pl.film ? `${pl.N} × ${pl.N} × ${pl.Nz}` : `${pl.N}³`);
  const nVox = (pl) => pl.N * pl.N * (pl.film ? pl.Nz : pl.N);
  const unitScale = (el) => (el.dataset.uk ? LEN_UNITS[getPath(state.form, el.dataset.uk) || "um"] || 1 : 1);
  function fillInputs(root) {
    $$("[data-k]", root).forEach((el) => {
      let v = getPath(state.form, el.dataset.k);
      if (el.type === "checkbox") el.checked = !!v;
      else if (document.activeElement !== el) {
        if (el.dataset.uk && typeof v === "number") v = parseFloat((v / unitScale(el)).toPrecision(10));
        el.value = v === null || v === undefined ? "" : v;
      }
    });
  }
  const RERENDER = /(\.shape|\.overlap|shell\.enabled|size\.unit|dist\.type|fraction\.basis|orientation\.mode|rve\.auto|rve\.L_unit|rve\.h_unit|rve\.T_unit|rve\.film|porosimetry\.fluid|\.props\.E|\.name)$/;
  function onField(e) {
    const el = e.target;
    if (!el.dataset || !el.dataset.k) return;
    let v;
    if (el.type === "checkbox") v = el.checked;
    else if ("num" in el.dataset) v = el.value === "" ? null : Number(el.value) * unitScale(el);
    else v = el.value;
    setPath(state.form, el.dataset.k, v);
    // the grid size follows from the edge length and the voxel size; an old
    // form's fixed grid size would otherwise override them silently
    if (/^rve\.(L_um|voxel_um)$/.test(el.dataset.k)) state.form.rve.N = null;
    if (el.dataset.k === "rve.auto" && !v) manualFromPlan();
    if (el.dataset.ref !== undefined && el.dataset.ref !== "") el.classList.toggle("mod", Number(el.value) !== Number(el.dataset.ref));
    saveForm();
    if (e.type === "change" && RERENDER.test(el.dataset.k)) { renderLeft(); renderRibbon(); }
    if (el.id === "caseName") $("#cTitle").textContent = state.form.name;
    schedulePlan();
  }
  function onDoeField(e) {
    const t = e.target;
    if (t.dataset.doe) {
      state.doe[t.dataset.doe] = t.type === "number" ? Number(t.value) : t.value;
      if (t.dataset.doe === "design") renderLeft();
      else if (t.dataset.doe !== "name") { const el = $(".note", $("#lbody")); if (el) el.textContent = `${doeCaseCount()} cases will be queued.`; }
      return;
    }
    const i = +t.dataset.pi, f = t.dataset.pf, p = state.doe.parameters[i];
    if (!p) return;
    if (f === "path") {
      const c = state.doeCat.find((x) => x.path === t.value);
      if (c) state.doe.parameters[i] = paramFrom(c);
      renderLeft();
      return;
    }
    p[f] = f === "scale" ? t.value : Number(t.value);
    if (f === "steps") renderLeft();
  }
  function onLibField(e) {
    const el = e.target, key = el.dataset.lk, id = state.libSel;
    if (!mat(id)) return;
    let m = state.custom[id];
    if (!m) m = state.custom[id] = Object.assign(clone(state.byId[id]), { override: true });
    if (key.startsWith("props.")) {
      const k0 = key.slice(6);
      // an optional value cleared is no longer set (not absorbing, no Tg)
      if (el.value === "" && OPT_PROPS.some(([o]) => o === k0)) { delete m.props[k0]; libChanged(id); return; }
      if (el.value === "" || !isFinite(Number(el.value))) return;
      const k = key.slice(6), v = Number(el.value);
      m.props[k] = v;
      const base = state.byId[id];
      el.classList.toggle("mod", !!base && Number(base.props[k]) !== v);
    } else m[key] = el.value;
    const n = libChanged(id);
    // re-drawn on change only: re-drawing on every keystroke would take the focus
    if (e.type === "change") { renderLeft(); renderRibbon(); if (n && key.startsWith("props.")) toast(`Applied to ${n} region(s) of this case.`); }
  }
  async function importLibrary(input) {
    const file = input.files && input.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      const list = Array.isArray(data) ? data : data.materials || [];
      let n = 0;
      for (const m of list) {
        if (!m || !m.id || !m.props || !m.name) continue;
        state.custom[m.id] = { id: m.id, name: m.name, group: m.group || "custom", note: m.note || "", props: clone(m.props), override: !!state.byId[m.id] };
        libChanged(m.id); n++;
      }
      saveCustom(); renderLeft(); renderRibbon(); toast(`${n} material(s) imported.`);
    } catch (err) { toast("The file could not be read: " + err.message); }
    input.value = "";
  }
  function onMaterial(e) {
    const el = e.target;
    const path = el.dataset.mat, id = el.value, m = mat(id);
    if (!m) return;
    const entry = getPath(state.form, path);
    const prevMat = mat(entry.material_id);
    const prevName = entry.name;
    entry.material_id = id; entry.props = clone(m.props);
    if (!prevName || (prevMat && prevName === prevMat.name) || path === "matrix") entry.name = m.name;
    saveForm(); renderLeft(); renderRibbon(); schedulePlan(0);
  }

  let planTimer = null, planSeq = 0;
  function schedulePlan(delay = 450) {
    clearTimeout(planTimer);
    planTimer = setTimeout(async () => {
      const my = ++planSeq;
      try {
        const res = await api("POST", "/api/plan", state.form);
        if (my !== planSeq) return;
        if (res.ok) { state.plan = res; state.planErrors = []; } else { state.plan = null; state.planErrors = res.errors || ["Input error"]; }
      } catch (err) { state.plan = null; state.planErrors = [String(err.message || err)]; }
      renderComposition();
      if (state.tab === "domain" && !document.activeElement.closest("#lbody")) renderLeft();
      renderPlanInfo();
    }, delay);
  }
  function renderPlanInfo() {
    const pl = state.plan && state.plan.plan;
    $("#sbGrid").textContent = pl ? `Grid ${gridTxt(pl)} · voxel ${fmt(pl.h_um)} µm · ${pl.film ? `film ${fmt(pl.L_um)} × ${fmt(pl.L_um)} × ${fmt(pl.T_um)} µm` : `RVE ${fmt(pl.L_um)} µm`} · ${pl.quality_label}` : (state.planErrors.length ? "Input error: " + state.planErrors[0] : "");
    $("#sbGrid").style.color = state.planErrors.length ? "var(--red)" : "";
    if (!pl) { $("#rveChecks").innerHTML = ""; $("#chkBadge").innerHTML = ""; return; }
    $("#rveChecks").innerHTML = checksRows(pl.checks);
    const bad = pl.checks.filter((c) => c.status !== "pass").length;
    $("#chkBadge").innerHTML = bad ? `<span class="pill warn">${bad}</span>` : `<span class="pill pass">${pl.checks.length}</span>`;
  }
  function checksRows(checks) {
    return `<tr><th>Group</th><th>Check</th><th class="n">Value</th><th class="n">Criterion</th><th>Status</th><th>Note</th></tr>` +
      checks.map((c) => `<tr><td>${esc(c.group)}</td><td>${esc(c.label)}</td><td class="n">${Array.isArray(c.value) ? c.value.map(fmt).join("–") : fmt(c.value)} ${c.unit && c.unit !== "rel" ? esc(c.unit) : ""}</td>
        <td class="n">${Array.isArray(c.target) ? c.target.map(fmt).join(" – ") : fmt(c.target)}</td><td>${pill(c.status)}</td><td class="hint">${esc(c.note || "")}</td></tr>`).join("");
  }

  /* ================================================================ run */
  function queueRoom() {
    const st = state.settings;
    const cap = (st && st.concurrency) || (state.health && state.health.concurrency) || 1;
    // mine, for naming one of them in the message
    const busy = state.jobs.filter((j) => ["running", "queued"].includes(j.state));
    // everyone's, because the limit belongs to the machine: on a shared server
    // another person's run fills a slot that will not appear in my list
    const total = st && st.running !== undefined ? st.running + (st.queued || 0) : busy.length;
    return { cap, busy, total, free: cap - total };
  }
  async function submit(preview) {
    // One machine, one set of cores. Starting another run while the machine is
    // already full only makes both slower, and it happened on a stray click
    // because nothing said no. Up to the chosen number of parallel jobs is
    // still allowed, since that is what the setting is for.
    const q = queueRoom();
    if (q.free <= 0) {
      const j = q.busy[0];
      toast(`${q.busy.length} analysis${q.busy.length === 1 ? " is" : "es are"} already under way `
        + `(limit ${q.cap}). Stop "${esc(j.name || j.id)}" or wait, `
        + `or raise the number of parallel analyses on the Home tab.`, 7000);
      return;
    }
    const body = clone(state.form);
    body.preview = !!preview;
    // the run on screen is the one to continue when it holds the same structure
    if (state.resultJob) body._continue = state.resultJob;
    try {
      const res = await api("POST", "/api/jobs", body);
      if (res.continued && state.resultJob === res.continued) { state.result = null; state.resultJob = null; state.viewerJob = null; }
      attach(res.id, preview);
      toast(res.continued
        ? (res.kept && res.kept.length
          ? `Continuing the run with this structure: ${res.kept.join(", ")} kept, only the new analyses are computed.`
          : "Continuing the run with this structure: it is not generated again.")
        : res.structure_from ? "A new run on the structure an earlier run generated: it is not generated again."
        : preview ? "Structure generation has started." : "The analysis has been queued.", 6000);
      pollRuns();
    } catch (err) {
      toast("The run could not be started:\n" + err.message, 7000);
    }
  }
  async function stopActive() { if (state.active) { await api("POST", `/api/jobs/${state.active}/stop`); toast("A stop has been requested."); } }
  function attach(id, preview) {
    state.active = id; state.seq = 0; state.history = {}; state.partial = {}; state.activeState = null; state.activePreview = !!preview;
    state.liveFigs = []; state.livePick = null;
    // the run being followed takes over the Result Viewer with its provisional result
    if (state.resultJob && state.resultJob !== id) { state.result = null; state.resultJob = null; }
    try { localStorage.setItem("mpsim.active", id); } catch (e) { }
    $("#console").textContent = "";
    setDock("console", true);
    pollActive();
  }
  let polling = false;
  async function pollActive() {
    if (!state.active || polling) return;
    polling = true;
    try {
      const ev = await api("GET", `/api/jobs/${state.active}/events?after=${state.seq}`);
      const con = $("#console"), pane = $("#d-console");
      const nearBottom = pane.scrollTop + pane.clientHeight > pane.scrollHeight - 40;
      let text = "", liveDirty = false, partialDirty = false;
      for (const e of ev.events) {
        if (e.kind === "line") text += esc(e.data.text) + "\n";
        else if (e.kind === "stage") text += `<span class="st">▶ ${esc(e.data.name)}</span> — ${esc(e.data.detail || "")}\n`;
        else if (e.kind === "partial") state.partial[e.data.key] = e.data.value;
        else if (e.kind === "figure") { if (!state.liveFigs.some((f) => f.name === e.data.name)) state.liveFigs.push(e.data); liveDirty = true; }
        else if (e.kind === "partial_result") partialDirty = true;
      }
      if (text) { con.insertAdjacentHTML("beforeend", text); if (con.innerHTML.length > 500000) con.innerHTML = con.innerHTML.slice(-300000); if (nearBottom) pane.scrollTop = pane.scrollHeight; }
      state.seq = ev.seq;
      state.history = ev.history || {};
      const prev = state.activeState;
      state.activeState = ev.state;
      const pct = Math.round(100 * (ev.state === "done" ? 1 : ev.progress || 0));
      $("#progBar").style.width = `${pct}%`;
      $("#sbState").innerHTML = `<i class="dot ${ev.state}"></i>${STATE[ev.state] || ev.state}${["running", "queued"].includes(ev.state) ? ` · ${pct} %` : ""}`;
      const sv = ev.solver;
      $("#sbStage").textContent = ev.state === "queued" ? `Queue position ${ev.queue_position}` : `${ev.stage || ""}${ev.stage_detail ? " · " + ev.stage_detail : ""}${sv ? ` · ${sv.name}: iteration ${sv.it}, residual ${fmt(sv.res)}` : ""} · ${secs(ev.elapsed_s)}`;
      const running = ["running", "queued"].includes(ev.state);
      $("#btnStop").disabled = !running;
      if (prev !== ev.state && state.tab && ["home", "analysis", "simulation"].includes(state.tab)) renderRibbon();
      if (state.dtab === "conv") drawConv(sv);
      renderPartial();
      if (liveDirty || prev !== ev.state) renderLive(ev.state);
      if (partialDirty && running) loadPartialResult(state.active);
      if (["done", "failed", "stopped"].includes(ev.state) && prev !== ev.state && prev !== null) {
        if (ev.state === "done") {
          await loadResult(state.active);
          if (state.cal && state.cal.resultJob === state.active) toast("The calibration is complete.");
          else if (state.activePreview) { toast("The structure has been generated."); setCenter("vis"); }
          else { toast("The analysis is complete. The Result Viewer shows all results."); setCenter("res"); }
        } else toast(ev.state === "failed" ? "The run failed; the console contains the details." : "The run was stopped.", 6000);
        pollRuns();
      }
    } catch (err) {
      console.warn(err);
    } finally { polling = false; }
  }
  // one curve per solve: the load cases of an elastic run one after the
  // other, or the problems a run solves side by side at the same time
  const CONV_COLORS = ["#0b62c4", "#d1242f", "#1a7f37", "#8250df", "#bf8700", "#0a8ca6", "#cf222e", "#57606a",
    "#e36209", "#116329", "#953800", "#6639ba"];
  /* While a run goes on, each analysis' figures are drawn by a background
     process as soon as it finishes and appear here in the 3D/2D View, newest
     first; a click shows one large. The Result Viewer meanwhile shows the
     analyses finished so far (result.partial.json). */
  function renderLive(st) {
    const box = $("#liveBox");
    if (!box) return;
    const running = ["running", "queued"].includes(st || state.activeState);
    const figs = state.liveFigs || [];
    const showing = running && state.active && (state.viewerJob !== state.active);
    if (!showing) { box.hidden = true; return; }
    const url = (f) => `/api/jobs/${state.active}/figures/${encodeURIComponent(f.name)}?t=${f.seconds || 0}`;
    const pick = figs.find((f) => f.name === state.livePick) || figs[figs.length - 1];
    box.hidden = false;
    box.innerHTML = !figs.length
      ? `<div class="livewait">${I("view3d")}<div><b>Results so far</b><br>The figures of each analysis appear here as soon as it finishes; the interactive 3D view opens when the run is complete.</div></div>`
      : `<div class="livemain"><div class="livehead"><b>Results so far</b> · ${figs.length} figure${figs.length === 1 ? "" : "s"} · <span class="hint">${esc(pick.title || pick.name)}</span></div>
          <img src="${url(pick)}" alt="${esc(pick.title || "")}"></div>
        <div class="livestrip">${figs.slice().reverse().map((f) => `<figure class="${f === pick ? "on" : ""}" data-livepick="${esc(f.name)}"><img src="${url(f)}" loading="lazy"><figcaption>${esc(f.group || "")}<br><small>${esc(f.title || f.name)}</small></figcaption></figure>`).join("")}</div>`;
  }
  async function loadPartialResult(id) {
    if (!id || (state.resultJob && state.resultJob !== id && state.result && !state.result.partial)) return;
    try {
      const r = await api("GET", `/api/jobs/${id}/result?partial=1`);
      if (state.activeState === "done") return;
      state.result = r; state.resultJob = id;
      if (state.center === "res") renderResults();
    } catch (e) { /* not written yet */ }
  }

  function drawConv(sv) {
    const cv = $("#convChart");
    if (!cv || !cv.offsetParent) return;
    const hist = Array.isArray(state.history) ? { Residual: state.history } : (state.history || {});
    const names = Object.keys(hist);
    const series = [];
    let t0 = Infinity, t1 = -Infinity;
    names.forEach((n, i) => {
      const h = (hist[n] || []).filter((p) => p[1] > 0);
      if (!h.length) return;
      t0 = Math.min(t0, h[0][0]); t1 = Math.max(t1, h[h.length - 1][0]);
      series.push({ x: h.map((p) => p[0]), y: h.map((p) => p[1]), color: CONV_COLORS[i % CONV_COLORS.length], label: n,
                    width: names.length > 6 ? 1.4 : 2 });
    });
    const tgt = sv && sv.target;
    if (tgt && series.length) series.push({ x: [t0, t1], y: [tgt, tgt], color: "#e36209", dash: [5, 4], width: 1.4, label: "Target" });
    CH.line(cv, { series, ylog: true, xlabel: "Elapsed time (s)", ylabel: "Residual" });
  }
  function renderPartial() {
    const p = state.partial, rows = [];
    const lab = { thermal: ["Thermal conductivity", "W/m·K"], electrical: ["Electrical conductivity", "S/m"], dielectric: ["Relative permittivity", "-"], magnetic: ["Relative permeability", "-"], cte: ["CTE α", "ppm/K"] };
    for (const [k, [l, un]] of Object.entries(lab)) if (p[k]) rows.push(`<tr><td>${l}</td><td class="n">${p[k].map(fmt).join(" / ")}</td><td>${un}</td></tr>`);
    $("#partialBox").innerHTML = rows.length ? `<table class="grid"><tr><th>Property (xx / yy / zz)</th><th class="n">Value</th><th>Unit</th></tr>${rows.join("")}</table>` : '<div class="hint">Intermediate values appear here while a run is in progress.</div>';
  }
  async function pollRuns() {
    try { state.jobs = await api("GET", "/api/jobs"); } catch (e) { return; }
    // /api/jobs is filtered to this user, but the concurrency limit is the
    // machine's and counts everyone. On a shared server that difference matters:
    // another person's run would not appear here, the Run button would stay
    // enabled, and the click would silently queue behind them. The settings
    // endpoint carries the authoritative counts, and it is a few bytes.
    try { state.settings = await api("GET", "/api/settings"); } catch (e) { /* keep the last */ }
    $("#runsList").innerHTML = runsHTML();
    if (state.tab === "results") renderLeft();
    const running = state.jobs.filter((j) => j.state === "running").length, queued = state.jobs.filter((j) => j.state === "queued").length;
    // The Run button is disabled once the machine is full, but that only shows
    // if the ribbon is redrawn when the count changes. Polling the run list
    // alone left the button looking clickable for the whole of a run, which is
    // what made a second click seem to do something. Redrawn on change only, so
    // the ribbon is not rebuilt under the pointer every four seconds.
    const inflight = running + queued;
    if (inflight !== state._inflight) { state._inflight = inflight; renderRibbon(); }
    // the limit is read from the live setting, not from the value the server
    // happened to start with, or the status bar contradicts the Home tab
    const st = state.settings;
    if (state.health) {
      $("#sbServer").textContent = `${running} running · ${queued} queued · `
        + (st ? `${st.concurrency} at a time × ${st.threads} of ${st.cores} cores`
              : `${state.health.cores} cores`);
    }
  }
  async function openJob(id) {
    const j = state.jobs.find((x) => x.id === id);
    if (j && j.state === "done") { await loadResult(id); setCenter(state.center === "rep" ? "rep" : state.center); if (state.center === "vis") openViewer(); }
    else attach(id, j && !(j.analyses || []).length);
  }
  async function loadResult(id) {
    try {
      const got = await api("GET", `/api/jobs/${id}/result`);
      if (got && got.kind === "calibration") {
        // a calibration has its own page; the Result Viewer keeps its run
        const c = CAL();
        c.result = got; c.resultJob = id;
        $("#runsList").innerHTML = runsHTML();
        setCenter("cal");
        if (state.tab === "calibration") { renderLeft(); renderRibbon(); }
        return;
      }
      state.result = got;
      state.resultJob = id;
      $("#cTitle").textContent = state.result.name;
      $("#runsList").innerHTML = runsHTML();
      if (state.center === "res") renderResults();
      if (state.center === "rep") renderReport();
      if (state.tab === "results") { renderLeft(); renderRibbon(); }
    } catch (err) { toast("The result could not be loaded: " + err.message); }
  }

  /* ================================================================ result viewer */
  const jobUrl = (p) => `/api/jobs/${state.resultJob}/${p}`;
  let draws = [];
  const canvas = (id, h = 260) => `<canvas class="chart" id="${id}" style="height:${h}px"></canvas>`;
  function renderResults() {
    const r = state.result, nav = $("#rvNav"), body = $("#rvBody");
    if (!r) {
      nav.innerHTML = "";
      body.innerHTML = `<div class="empty">${I("table")}<div><b>No result is selected.</b><br>Completed runs are opened from the Runs list on the right.</div></div>`;
      return;
    }
    draws = [];
    const P = r.properties || {};
    const secs_ = [];
    const add = (id, title, icon, color, group, html) => secs_.push({ id, title, icon, color, group, html });
    add("overview", "Summary", "info", "var(--c-gen)", "Overview", secOverview(r));
    if (P.thermal) add("thermal", "Thermal conductivity", "thermal", "var(--c-thermal)", "Simulation", secCond("thermal"));
    if (P.electrical) add("electrical", "Electrical conductivity", "electrical", "var(--c-electrical)", "Simulation", secCond("electrical"));
    if (P.dielectric) add("dielectric", "Permittivity and loss", "dielectric", "var(--c-diel)", "Simulation", secCond("dielectric"));
    if (P.magnetic) add("magnetic", "Magnetic permeability", "magnetic", "var(--c-mag)", "Simulation", secCond("magnetic"));
    if (P.emi) add("emi", "EMI shielding", "emi", "var(--c-mag)", "Simulation", secEmi(P.emi));
    if (P.cte) add("cte", "Elasticity and thermal expansion", "cte", "var(--c-mech)", "Simulation", secCte(P.cte));
    if (P.permeability) add("permeability", "Permeability", "permeability", "var(--c-flow)", "Simulation", secPerm(P.permeability));
    if (P.viscosity) add("viscosity", "Viscosity and flowability", "permeability", "var(--c-flow)", "Simulation", secVisc(P.viscosity));
    if (P.moisture) add("moisture", "Moisture uptake and swelling", "tortuosity", "var(--c-flow)", "Simulation", secMoist(P.moisture));
    if (P.filtration) add("filtration", "Filtration efficiency", "pores", "var(--c-flow)", "Simulation", secFiltration(P.filtration));
    if (P.tortuosity) add("tortuosity", "Diffusion and tortuosity", "tortuosity", "var(--c-diff)", "Simulation", secTort(P.tortuosity));
    if (P.acoustics) add("acoustics", "Acoustic absorption", "acoustics", "var(--c-ac)", "Simulation", secAcoustics(P.acoustics));
    if (P.radiation) add("radiation", "Radiative extinction", "radiation", "var(--c-rad)", "Simulation", secRadiation(P.radiation));
    if (P.porosity || P.morphology) add("morphology", "Porosity and size distributions", "morphology", "var(--c-struct)", "Structure analysis", secMorph(P));
    if (P.percolation) add("percolation", "Percolation paths", "tortuosity", "var(--c-struct)", "Structure analysis", secPercolation(P.percolation));
    if (P.porosimetry) add("porosimetry", "Porosimetry", "porosimetry", "var(--c-struct)", "Structure analysis", secPorosimetry(P.porosimetry));
    if (P.pore_network) add("pore_network", "Pore network", "pore_network", "var(--c-struct)", "Structure analysis", secNetwork(P.pore_network));
    if (P.grains) add("grains", "Grain analysis", "grains", "var(--c-struct)", "Structure analysis", secGrains(P.grains));
    add("checks", `Verification (${r.check_counts.fail ? r.check_counts.fail + " failed" : r.check_counts.pass + " passed"})`, "checks", "var(--c-diff)", "Verification", `<div class="box">${`<div class="tablewrap"><table class="grid">${checksRows(r.checks)}</table></div>`}</div>`);
    add("input", "Input and RVE", "cube", "var(--c-gen)", "Verification", secInput(r));
    let lastGroup = "";
    nav.innerHTML = secs_.map((s) => {
      const cap = s.group !== lastGroup ? `<div class="cap">${s.group}</div>` : "";
      lastGroup = s.group;
      return `${cap}<button data-sec="${s.id}" style="--ic:${s.color}">${I(s.icon)}<span>${esc(s.title)}</span></button>`;
    }).join("");
    body.innerHTML = `<div class="rvhead">${I("logo")}<div class="grow"><h2>${esc(r.name)}</h2><div class="hint">${esc(r.created)} · ${secs(r.elapsed_s)} · ${backendTxt(r)} · PuMA ${esc((r.versions || {}).puma || "—")}</div></div>
        <div class="row wrap"><button class="btn sm" data-center="vis">${I("view3d")}3D view</button><a class="btn sm primary" href="${jobUrl("report")}" target="_blank">${I("report")}Report</a><a class="btn sm" href="${jobUrl("download/bundle")}">${I("zip")}ZIP</a></div></div>` +
      secs_.map((s) => `<section class="rsec" id="sec-${s.id}"><h3 style="--ic:${s.color}">${I(s.icon)}${esc(s.title)}</h3>${s.html}</section>`).join("");
    requestAnimationFrame(() => { draws.forEach((d) => { try { d(); } catch (e) { console.warn(e); } }); });
    $$("button", nav)[0] && $$("button", nav)[0].classList.add("on");
  }
  function tiles(headline) {
    return `<div class="big">${headline.map((h) => `<div class="tile"><div class="l">${esc(h.label)}</div><div class="v">${fmt(h.value)} <span style="font-size:12px;color:var(--muted)">${esc(h.unit)}</span></div>
      <div class="d">${h.diag ? "xx / yy / zz " + h.diag.map(fmt).join(" / ") : esc(h.sub || "")}</div>${h.fv ? `<div class="d" title="Same RVE, independent periodic finite-volume discretisation">periodic FV ${fmt(h.fv)} (${h.value / h.fv >= 1 ? "+" : ""}${fmt(100 * (h.value / h.fv - 1))} %)</div>` : ""}</div>`).join("")}</div>`;
  }
  function secOverview(r) {
    const cc = r.check_counts;
    const prov = r.partial ? `<div class="note warn"><b>Provisional result</b> — the run is still going. Shown are the analyses finished so far (${(r.analyses || []).map((a) => esc(SHORT_AN[a] || a)).join(", ") || "none yet"}), from the first realisation; the figures, the statistics over realisations and the report are completed when the run ends.</div>` : "";
    const figs = Object.entries(r.figures || {}).map(([n, t]) => `<figure><a href="${jobUrl("figures/" + n)}" target="_blank"><img loading="lazy" src="${jobUrl("figures/" + n)}"></a><figcaption>${esc(t)}</figcaption></figure>`).join("");
    const comp = r.composition.labels.map((l) => [esc(l.name), esc(l.kind), fmt(100 * l.vf), fmt(100 * l.wt)]);
    // per-direction entries exist for DOE targets; the tiles show the average
    return prov + tiles((r.headline || []).filter((h) => !h.dir)) +
      `<div class="rgrid"><div class="box"><div class="bt">Verification</div><p>${pill("pass")} ${cc.pass} passed · ${pill("warn")} ${cc.warn} warnings · ${pill("fail")} ${cc.fail} failed</p>
        <div class="row wrap"><a class="btn sm" href="${jobUrl("download/layer_card")}">${I("layers")}Layer card</a><a class="btn sm" href="${jobUrl("download/result")}">${I("json")}Result JSON</a><a class="btn sm" href="${jobUrl("download/structure")}">${I("image")}Structure TIFF</a><button class="btn sm" data-editspec>${I("open")}Edit this input</button></div>
        <p class="hint">Grid ${gridTxt(r.plan)} · voxel ${fmt(r.plan.h_um)} µm · ${r.plan.film ? `film ${fmt(r.plan.L_um)} × ${fmt(r.plan.L_um)} µm × ${fmt(r.plan.T_um)} µm thick` : `edge ${fmt(r.plan.L_um)} µm`} · ${r.plan.seeds} realisation(s) · ${esc(r.realisations[0].route || "")}</p>${r.plan.notes.map((n) => `<div class="note warn">${esc(n)}</div>`).join("")}</div>
      <div class="box"><div class="bt">Realised composition (voxels)</div>${tbl(["Region", "Kind", "vol %", "wt %"], comp)}<p class="hint">Density ${fmt(r.composition.density)} g/cm³ · specific heat ${fmt(r.composition.cp)} J/kg·K · porosity ${fmt(100 * r.composition.porosity)} %</p></div></div>
      ${figs ? `<div class="box" style="margin-top:12px"><div class="bt">Figures (PyVista · matplotlib)</div><div class="figs">${figs}</div></div>` : ""}`;
  }
  function tensorTable(p, unit) {
    const T = p.tensor, ax = ["x", "y", "z"];
    const rows = ax.map((a, i) => [a, ...ax.map((b, j) => (T[i][j] === null ? "—" : fmt(T[i][j])))]);
    return tbl(["", "x", "y", "z"], rows) + `<p>Isotropic mean <b class="mono">${fmt(p.iso)}</b> ${unit && unit !== "-" ? esc(unit) : ""}${p.ci95 ? ` · 95 % CI ± ${p.ci95.map(fmt).join(" / ")}` : ""}${p.fv_iso ? ` · ${p.xc_method === "fe" ? "check with PuMA periodic FE" : "independent periodic FV"} <b class="mono">${fmt(p.fv_iso)}</b>` : ""}</p>${p.resolution ? `<p class="hint">Resolution check (${p.resolution.direction}): ${p.resolution.grid || p.resolution.N + "³"} ${fmt(p.resolution.base)} → ${p.resolution.grid_fine || p.resolution.N_fine + "³"} <b class="mono">${fmt(p.resolution.fine)}</b> (${p.resolution.change >= 0 ? "+" : ""}${fmt(100 * p.resolution.change)} %) · ${p.resolution.extrapolated != null ? `extrapolated to fine voxels <b class="mono">${fmt(p.resolution.extrapolated)}</b>` : `not extrapolated: ${esc(p.resolution.extrapolation_note || "the change is too large for a first-order correction")}`}</p>` : ""}`;
  }
  function refsChart(p) {
    const bands = [], points = [], refs = p.refs || {};
    if (refs["Wiener bounds"]) bands.push({ label: "Wiener bounds", lo: refs["Wiener bounds"][0], hi: refs["Wiener bounds"][1], color: "#d0d7de" });
    if (refs["Hashin–Shtrikman"]) bands.push({ label: "Hashin–Shtrikman bounds", lo: refs["Hashin–Shtrikman"][0], hi: refs["Hashin–Shtrikman"][1], color: "#9cc3f0" });
    const how = ({ fe: "PuMA FE", fv: "PuMA FV", "fv-periodic": "periodic FV", "fv-interface": "periodic FV", "fv-film": "film FV" })[((p.stats || [])[0] || {}).method] || "solved";
    ["x", "y", "z"].forEach((a, i) => { if (p.diag[i] !== null) points.push({ label: `RVE ${a}${a} (${how})`, value: p.diag[i], color: "#0b62c4", bold: true }); });
    if (p.fv_iso) points.push({ label: p.xc_method === "fe" ? "RVE mean (PuMA FE check)" : "RVE mean (periodic FV)", value: p.fv_iso, color: "#5aa0e8" });
    if (refs["Maxwell-Garnett / Mori–Tanaka"]) points.push({ label: "Maxwell-Garnett / Mori–Tanaka", value: refs["Maxwell-Garnett / Mori–Tanaka"].reduce((s, v) => s + v, 0) / 3, color: "#e36209", shape: "diamond" });
    if (refs["Bruggeman (EMA)"] !== undefined) points.push({ label: "Bruggeman (EMA)", value: refs["Bruggeman (EMA)"], color: "#1a7f37", shape: "diamond" });
    return CH.refs({ bands, points, unit: p.unit, log: true });
  }
  function polarFromTensor(T) {
    const th = Array.from({ length: 73 }, (_, i) => i * 5);
    const out = [];
    [["xy", 0, 1, "#0b62c4"], ["xz", 0, 2, "#e36209"], ["yz", 1, 2, "#1a7f37"]].forEach(([nm, i, j, col]) => {
      const vals = [T[i][i], T[j][j], T[i][j], T[j][i]];
      if (vals.some((v) => v === null || !isFinite(v))) return;
      out.push({ label: nm + " plane", color: col, theta_deg: th, r: th.map((d) => { const c = Math.cos((d * Math.PI) / 180), s = Math.sin((d * Math.PI) / 180); return T[i][i] * c * c + (T[i][j] + T[j][i]) * c * s + T[j][j] * s * s; }) });
    });
    return out;
  }
  function viewBtn(prefix, label = "Show field") { return state.result.view ? `<button class="btn sm" data-goviewer="${prefix}">${I("view3d")}${label}</button>` : ""; }
  function linesBtn(prefix, label = "Show the field lines") { return state.result.view ? `<button class="btn sm" data-goviewer="lines:${prefix}">${I("tortuosity")}${label}</button>` : ""; }
  function lineRows(fl) {
    if (!fl) return [];
    // with no line crossing, a mean length would read as a tortuosity below
    // one, so only the count is reported
    const rows = [["Field lines that cross the sample", `${fl.n_through} of ${fl.n_lines}`]];
    if (fl.tortuosity) rows.unshift(["Field-line tortuosity (mean line length / straight distance)", fmt(fl.tortuosity)]);
    return rows;
  }
  function secCond(key) {
    const r = state.result, p = r.properties[key], d = p.derived || {};
    const id = "c_" + key;
    const names = {
      resistivity_mK_W: ["Thermal resistivity", "m·K/W"], area_resistance_mm2K_W: ["Area-specific thermal resistance", "mm²·K/W"], area_resistance_z_mm2K_W: ["Area-specific thermal resistance (z)", "mm²·K/W"],
      diffusivity_mm2_s: ["Thermal diffusivity", "mm²/s"], volumetric_heat_capacity_MJ_m3K: ["Volumetric heat capacity", "MJ/m³·K"], resistivity_ohm_m: ["Electrical resistivity", "Ω·m"],
      sheet_resistance_ohm_sq: ["Sheet resistance", "Ω/sq"], capacitance_pF_mm2: ["Capacitance per area", "pF/mm²"], capacitance_z_pF_mm2: ["Capacitance per area (z)", "pF/mm²"], anisotropy_inplane_over_z: ["Anisotropy (xx+yy)/2 : zz", ""],
    };
    const drows = Object.entries(d).filter(([k]) => names[k]).map(([k, v]) => [names[k][0], withUnit(v, names[k][1])]);
    if (key === "dielectric") drows.unshift(["Dissipation factor Df (energy-weighted)", fmt(p.tan_d_iso)]);
    if (p.percolation) drows.push(["Connected high-value phase (x / y / z)", p.percolation.map((v) => (v ? "●" : "○")).join(" ")]);
    const fs = p.field_stats || {}, stats = fs.enhancement || fs.flux;
    drows.push(...lineRows(fs.field_lines));
    const statsTbl = stats ? tbl(["Region", key === "dielectric" || key === "magnetic" ? "Mean enhancement" : "Mean", "99th percentile", "Maximum"], stats.map((s) => [esc(s.label), fmt(s.mean), fmt(s.p99), fmt(s.max)])) : "";
    const labels = r.spec.labels.map((l) => l.name);
    const d0 = Object.keys(p.energy_frac || {})[0];
    const colors = labels.map((_, i) => PHASE_COLORS[i % PHASE_COLORS.length]);
    const pol = polarFromTensor(p.tensor);
    if (d0) draws.push(() => CH.bars($("#" + id + "_e"), { labels, values: p.energy_frac[d0].map((v) => 100 * v), unit: " %", max: 100, colors }));
    if (pol.length) draws.push(() => CH.polar($("#" + id + "_p"), { series: pol, unit: p.unit }));
    const kn = p.knudsen;
    const contacts = (p.contacts || []).length ? `<div class="box" style="grid-column:1/-1"><div class="bt">Contact resistance between particles <span class="hint">on the voxel faces where two particles touch</span></div>${tbl(["Particles in contact", "R<sub>c</sub> (m²K/W)", "Source", "Contact faces"], p.contacts.map((c) => [esc(c.names[0] === c.names[1] ? `${c.names[0]} – ${c.names[1]} (same filler)` : `${c.names[0]} – ${c.names[1]}`), fmt(c.r_c), c.pair[0] === c.pair[1] ? "filler card" : (c.set ? "set for the pair" : "mean of the two"), (c.faces || 0).toLocaleString()]))}</div>` : "";
    return contacts + `<div class="rgrid">
      <div class="box"><div class="bt">${esc(p.title)} tensor ${p.unit !== "-" ? `<span class="hint">${esc(p.unit)}</span>` : ""}<div class="grow"></div>${viewBtn(key)}${fs.field_lines && fs.field_lines.published ? linesBtn(key + "_") : ""}</div>${tensorTable(p, p.unit)}
        ${drows.length ? `<div class="bt" style="margin-top:8px">Derived quantities <span class="hint">layer thickness ${fmt(d.thickness_mm)} mm</span></div>${kv(drows)}` : ""}
        ${kn ? `<div class="note">Rarefied gas: pore diameter ${fmt(kn.d_pore_um)} µm, Kn = ${fmt(kn.knudsen_number)} (${esc(kn.regime)}); gas conductivity × ${fmt(kn.gas_factor)}.</div>` : ""}</div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Bounds and mean-field models</div><div class="hint">A value outside the rigorous bounds indicates an analysis error; distance from the models reflects the microstructure.</div>${refsChart(p)}</div>
      ${pol.length ? `<div class="box"><div class="bt">Directional value n·K·n</div>${canvas(id + "_p", 300)}</div>` : ""}
      ${d0 ? `<div class="box"><div class="bt">Energy share by region (${d0} gradient)</div>${canvas(id + "_e", 24 * labels.length + 16)}${statsTbl ? `<div class="bt" style="margin-top:10px">Field statistics by region</div>${statsTbl}` : ""}</div>` : ""}
      <div class="box" style="grid-column:1/-1"><div class="bt">Solver record</div>${tbl(["Direction", "Method", "Contrast policy", "Iterations", "Residual / target", "Converged", "Time", "Cross-check"], p.stats.map((s) => [s.direction, ({ fe: "PuMA periodic FE", fv: "PuMA FV (fixed faces)", "fv-periodic": "Periodic FV", "fv-interface": "Periodic FV + interface resistances", "fv-film": "Film FV (plates in z, insulated sides)" })[s.method] || "Periodic FV", s.policy, s.iterations ?? "—", `${fmt(s.residual)} / ${fmt(s.target)}`, s.converged === null || s.converged === undefined ? "—" : s.converged ? pill("pass") : pill("fail"), `${fmt(s.seconds)} s`, s.crosscheck ? `${fmt(s.crosscheck)} (${s.crosscheck_method === "fe" ? "PuMA FE" : "FV"})` : "—"]))}</div></div>`;
  }
  function secEmi(p) {
    const v = p.validity;
    // no floor at 0 dB: the multiple-reflection term is negative for thin layers
    draws.push(() => { const sp = p.spectrum; CH.line($("#emiChart"), { xlog: true, xlabel: "Frequency (Hz)", ylabel: "SE (dB)", series: [
      { x: sp.freq_hz, y: sp.se_db, color: "#0b62c4", width: 2.8, label: "Total SE" }, { x: sp.freq_hz, y: sp.se_r_db, color: "#e36209", dash: [6, 4], width: 1.6, label: "Reflection" },
      { x: sp.freq_hz, y: sp.se_a_db, color: "#1a7f37", dash: [2, 3], width: 1.6, label: "Absorption" }, { x: sp.freq_hz, y: sp.se_m_db, color: "#8250df", dash: [8, 3, 2, 3], width: 1.2, label: "Multiple reflection" }]
      .concat(sp.se_db_static ? [{ x: sp.freq_hz, y: sp.se_db_static, color: "#8c959f", dash: [4, 3], width: 1.4, label: "Static σ, ε (comparison)" }] : []) }); });
    const fw = p.fullwave;
    if (fw && !fw.error) draws.push(() => CH.line($("#emiFw"), { xlabel: "Frequency (Hz)", ylabel: "SE (dB)", series: [
      { x: fw.freq_hz, y: fw.se_fullwave_db, color: "#d1242f", width: 2.4, label: "openEMS, voxel structure" },
      { x: fw.freq_hz, y: fw.se_model_db, color: "#0b62c4", dash: [6, 4], width: 2.0, label: "Homogenised slab" }] }));
    const fwBox = !fw ? "" : fw.error ? `<div class="box" style="grid-column:1/-1"><div class="bt">Full-wave simulation (openEMS)</div><p class="hint">Skipped: ${esc(fw.error)}</p></div>`
      : `<div class="box" style="grid-column:1/-1"><div class="bt">Full-wave simulation <span class="hint">openEMS FDTD of the ${fmt(fw.thickness_um)} µm RVE slab, voxel by voxel, against the same slab homogenised · ${fw.cells.toLocaleString()} cells · ${fmt(fw.seconds)} s</span></div>${canvas("emiFw", 260)}
        <p>Largest difference <b class="mono">${fmt(fw.max_diff_db)} dB</b> (mean ${fmt(fw.mean_diff_db)} dB). |S11| at ${fmt(fw.freq_hz[0])} Hz: ${fmt(fw.s11_fullwave[0])} full-wave, ${fmt(fw.s11_model[0])} homogenised.</p>
        ${(fw.notes || []).map((n) => `<div class="note warn">${esc(n)}</div>`).join("")}</div>`;
    const cx = p.complex;
    const cxBox = cx ? `<div class="box" style="grid-column:1/-1"><div class="bt">Frequency-resolved effective properties <span class="hint">σ + jωε solved on the RVE at each frequency (${esc(cx.directions.join(", "))}, ${esc(cx.bc)})</span></div>
      ${tbl(["Frequency", "σ′ (S/m)", "ε′", "Contrast limit"], cx.freq_hz.map((f, i) => [`${fmt(f)} Hz`, fmt(cx.sigma_eff[i]), cx.eps_resolved ? fmt(cx.eps_eff[i]) : "—", esc([...new Set(cx.policy[i])].join(", "))]))}
      <p class="hint">The grey dashed line is the earlier static method: DC σ and static ε solved separately and combined. ${cx.eps_resolved ? "" : "A conducting network carries the current, so ε′ (below 1/10⁶ of σ′) is not resolved and not shown."}</p></div>` : "";
    return `<div class="rgrid"><div class="box" style="grid-column:1/-1"><div class="bt">Shielding effectiveness <span class="hint">thickness ${fmt(p.thickness_mm)} mm · σ ${fmt(p.sigma_inplane)} S/m · εr ${fmt(p.eps_inplane)} · μr ${fmt(p.mu_inplane)}</span></div>${canvas("emiChart", 320)}</div>
      ${cxBox}${fwBox}
      <div class="box"><div class="bt">SE by frequency</div>${tbl(["Frequency", "SE (dB)"], Object.entries(p.at).map(([f, s]) => [`${f} Hz`, fmt(s)]))}</div>
      <div class="box"><div class="bt">Validity</div><p>Quasi-static homogenisation holds when the RVE is much smaller than the wavelength and conductive fillers are thinner than their skin depth.</p>
        ${kv([["Wavelength condition", `f < ${fmt(v.f_wavelength_hz)} Hz`], ...v.filler.map((x) => [`Skin depth of ${esc(x.name)}`, `f < ${fmt(x.f_skin_hz)} Hz`])])}
        <p class="hint">scikit-rf: ${p.skrf_check.available ? `v${p.skrf_check.version}, largest difference ${fmt(p.skrf_check.max_diff_db)} dB` : esc(p.skrf_check.error || "unavailable")}</p></div></div>`;
  }
  function secCte(p) {
    const c = p.constants, vo = ["xx", "yy", "zz", "yz", "xz", "xy"];
    const bands = [], points = [{ label: "RVE volumetric mean", value: p.alpha_vol, color: "#0b62c4", bold: true }];
    ["xx", "yy", "zz"].forEach((a, i) => points.push({ label: `RVE α${a}`, value: p.alpha[i], color: "#5aa0e8" }));
    for (const [k, v] of Object.entries(p.refs)) {
      if (Array.isArray(v)) bands.push({ label: k, lo: v[0], hi: v[1], color: "#9cc3f0" });
      else points.push({ label: k, value: v, color: "#e36209", shape: "diamond" });
    }
    if (p.polar_E) draws.push(() => CH.polar($("#cteP"), { unit: "GPa", series: [["xy", "#0b62c4"], ["xz", "#e36209"], ["yz", "#1a7f37"]].map(([pl, col]) => ({ label: pl + " plane", color: col, theta_deg: p.polar_E.theta_deg, r: p.polar_E[pl] })) }));
    const ss = p.stress_stats || {};
    const hyd = Object.fromEntries((ss.hydrostatic || []).map((s) => [s.label, s]));
    const mat6 = (M, dig) => tbl(["", ...vo], M.map((row, i) => [vo[i], ...row.map(fmt)]));
    return `<div class="rgrid">
      <div class="box"><div class="bt">Thermal expansion α <span class="hint">ppm/K</span><div class="grow"></div>${viewBtn("cte", "Stress fields")}</div>${tbl(vo, [p.alpha.map((v, i) => fmt(i < 3 ? v : v / 2))])}
        <p class="hint">Tensor components α<sub>ij</sub>. The layer card stores the Voigt vector that pairs with C*, whose shear entries are 2α<sub>ij</sub>.</p>
        <p>Volumetric mean α = <b class="mono">${fmt(p.alpha_vol)}</b> ppm/K${p.alpha_ci95 ? " · 95 % CI ± " + p.alpha_ci95.map(fmt).join(" / ") : ""}</p>
        <div class="bt" style="margin-top:6px">Engineering constants</div>
        ${tbl(["", "x", "y", "z"], [["E [GPa]", ...c.E.map(fmt)], ["G (yz, xz, xy) [GPa]", ...c.G.map(fmt)], ["ν (xy, yz, zx)", fmt(c.nu_xy), fmt(c.nu_yz), fmt(c.nu_zx)], ["ν (yx, zy, xz)", fmt(c.nu_yx), fmt(c.nu_zy), fmt(c.nu_xz)]])}
        ${tbl(["Average", "K [GPa]", "G [GPa]"], [["Voigt", fmt(c.K_voigt), fmt(c.G_voigt)], ["Reuss", fmt(c.K_reuss), fmt(c.G_reuss)], ["Hill", fmt(c.K_hill), fmt(c.G_hill)]])}
        <p>Hill E ${fmt(c.E_hill)} GPa · ν ${fmt(c.nu_hill)} · universal anisotropy index A<sup>U</sup> = ${fmt(c.anisotropy_index)}</p></div>
      <div class="box"><div class="bt">Directional Young's modulus E(n)</div>${canvas("cteP", 320)}</div>
      <div class="box" style="grid-column:1/-1"><div class="bt">α compared with analytical models</div>${CH.refs({ bands, points, unit: "ppm/K" })}</div>
      <div class="box"><div class="bt">Stiffness matrix C* <span class="hint">GPa</span></div>${mat6(p.C)}<div class="bt" style="margin-top:8px">Compliance matrix S* <span class="hint">1/TPa</span></div>${mat6(p.S)}</div>
      ${(ss.von_mises || []).length ? `<div class="box"><div class="bt">Thermal stresses by region <span class="hint">ΔT = ${fmt(p.delta_T)} K, ${ss.bc === "free expansion" ? "free expansion (zero mean stress)" : "zero macroscopic strain"}, MPa</span></div>${tbl(["Region", "Mean von Mises", "99th percentile", "Maximum", "Mean hydrostatic"], ss.von_mises.map((s) => [esc(s.label), fmt(s.mean), fmt(s.p99), fmt(s.max), fmt((hyd[s.label] || {}).mean)]))}
        ${p.void_ratio ? `<p class="hint">The stiffness of pores is the smallest solid E × ${p.void_ratio}.</p>` : ""}</div>` : ""}
      ${p.above_tg ? (() => { const a = p.above_tg, ca = a.constants;
        return `<div class="box" style="grid-column:1/-1"><div class="bt">Above Tg <span class="hint">rubbery: ${esc(a.rubbery.join(", "))}</span></div>
          ${tbl(["", "Below Tg", "Above Tg", "Ratio"], [["α volumetric mean [ppm/K]", fmt(p.alpha_vol), fmt((a.alpha[0] + a.alpha[1] + a.alpha[2]) / 3), fmt((a.alpha[0] + a.alpha[1] + a.alpha[2]) / 3 / p.alpha_vol)],
            ["α xx / yy / zz", p.alpha.slice(0, 3).map(fmt).join(" / "), a.alpha.slice(0, 3).map(fmt).join(" / "), ""],
            ["Hill E [GPa]", fmt(c.E_hill), fmt(ca.E_hill), fmt(ca.E_hill / c.E_hill)], ["Hill ν", fmt(c.nu_hill), fmt(ca.nu_hill), ""],
            ["E xx / yy / zz [GPa]", c.E.map(fmt).join(" / "), ca.E.map(fmt).join(" / "), ""]])}
          <p class="hint">α1 and α2 of the composite, as on an EMC data sheet. Above Tg the stiff fillers carry the load and hold the rubbery resin back, so α2 rises far less than the resin's own.</p></div>`; })() : ""}
      <div class="box" style="grid-column:1/-1"><div class="bt">Solver record by load case</div>${tbl(["Load case", "Iterations", "Residual / target", "Converged", "Time"], p.stats.map((s) => [s.load, s.iterations ?? "—", `${fmt(s.residual)} / ${fmt(s.target)}`, s.converged === null || s.converged === undefined ? "—" : s.converged ? pill("pass") : pill("fail"), `${fmt(s.seconds)} s`]))}</div></div>`;
  }
  function secMoist(p) {
    const fmtT = (h) => !isFinite(h) ? "—" : h < 1 ? `${fmt(h * 60)} min` : h < 72 ? `${fmt(h)} h` : `${fmt(h / 24)} days`;
    const c = p.curve;
    draws.push(() => CH.line($("#moistCurve"), { xlog: true, xlabel: "Exposure time (h)", ylabel: "Moisture uptake (wt%)", ymin: 0, series: [
      { x: c.t_h, y: c.wt_pct, color: "#0b62c4", width: 2.4, label: `${fmt(p.thickness_mm)} mm, ${p.sides === "one" ? "one face" : "both faces"} exposed` },
      { x: [p.hours, p.hours], y: [0, p.wt_pct], color: "#8c959f", dash: [4, 4], label: `${fmt(p.hours)} h` }] }));
    const sw = p.swelling;
    const ax = ["x", "y", "z"];
    return `<div class="rgrid">
      <div class="box"><div class="bt">Diffusion and uptake</div>
        ${kv([["Effective diffusivity D_eff (x / y / z)", p.D_eff.map((v) => (v === null || !isFinite(v) ? "—" : fmt(v))).join(" / ") + " m²/s"],
              p.ratio ? ["D_eff / D_matrix", p.ratio.map((v) => (v === null || !isFinite(v) ? "—" : fmt(v))).join(" / ")] : null,
              ["Saturated uptake", `<b class="mono">${fmt(p.wt_pct)}</b> wt% (${fmt(p.c_sat_kg_m3)} kg/m³)`],
              ["Half saturation t₅₀", fmtT(p.t50_h)], ["95 % saturation", fmtT(p.t95_h)],
              [`Uptake after ${fmt(p.hours)} h`, `${fmt(p.wt_pct_at_hours)} wt% (${fmt(100 * p.fraction_at_hours)} % of saturation)`]])}
        ${Object.keys(p.refs || {}).length ? `<div class="bt" style="margin-top:8px">Reference</div>${kv(Object.entries(p.refs).map(([k, v]) => [esc(k), fmt(v)]))}` : ""}
        <p class="hint">Water moves by the gradient of its activity c/c_sat, which is continuous across an interface where the materials take up different amounts; the solve carries the permeability D·c_sat and D_eff = P_eff / ⟨c_sat⟩. Fillers that take up nothing are impermeable obstacles.</p></div>
      <div class="box"><div class="bt">Uptake of the part <span class="hint">Crank, plate of ${fmt(p.thickness_mm)} mm</span><div class="grow"></div>${viewBtn("moist", "Activity and flux")}</div>${canvas("moistCurve", 280)}</div>
      ${sw ? `<div class="box"><div class="bt">Hygroscopic swelling at saturation</div>
        ${kv([["Swelling strain (x / y / z)", sw.strain.slice(0, 3).map((v) => fmt(1e3 * v)).join(" / ") + " ‰"],
              ["Volumetric mean", `<b class="mono">${fmt(1e3 * sw.strain_vol)}</b> ‰`],
              sw.cme_eff ? ["Effective CME (x / y / z)", sw.cme_eff.map(fmt).join(" / ") + " per mass fraction"] : null,
              p.equivalent_dT ? ["Same strain as a temperature rise of", `${fmt(p.equivalent_dT)} K`] : null])}
        <p class="hint">Every material swells by its CME times its own moisture mass fraction at saturation; the RVE gives what the composite does with stiff, dry fillers holding the resin back.${sw.reused_stiffness ? " The stiffness of the elastic solve was reused: one extra load case." : ""}</p></div>` : ""}
      <div class="box"><div class="bt">Solver record</div>${tbl(["Direction", "Iterations", "Residual", "Converged", "Time"], (p.stats || []).map((s) => [s.direction, s.iterations, fmt(s.residual), s.converged ? pill("pass") : pill("fail"), secs(s.seconds)]))}</div>
    </div>`;
  }
  const poroNote = (o) => (o && o.open_porosity !== undefined && o.open_porosity !== null ? `<p class="hint">Open porosity ${fmt(100 * o.open_porosity)} % · closed porosity ${fmt(100 * o.closed_porosity)} %. Flow and diffusion use the open porosity only.</p>` : "");
  function viscosityOptions() {
    const v = ((state.form.options || {}).viscosity) || {};
    const t = ((v.model || {}).type) || "newtonian";
    const mrow = {
      newtonian: num("options.viscosity.model.mu", "Resin viscosity μ", "Pa·s"),
      power: `<div class="g2">${num("options.viscosity.model.K", "Consistency K", "Pa·sⁿ")}${num("options.viscosity.model.n", "Flow index n", "")}</div>`,
      carreau: `<div class="g2">${num("options.viscosity.model.mu0", "μ₀", "Pa·s")}${num("options.viscosity.model.mu_inf", "μ∞", "Pa·s")}${num("options.viscosity.model.lam", "Time constant λ", "s")}${num("options.viscosity.model.n", "Index n", "")}</div>`,
      cross: `<div class="g2">${num("options.viscosity.model.mu0", "μ₀", "Pa·s")}${num("options.viscosity.model.mu_inf", "μ∞", "Pa·s")}${num("options.viscosity.model.lam", "Time constant λ", "s")}${num("options.viscosity.model.n", "Index n", "")}</div>`,
    }[t];
    return `<div class="hint">The matrix is taken as the uncured liquid resin at the process temperature and the fillers as rigid particles. The relative viscosity μr = μ/μ<sub>resin</sub> does not depend on the resin's viscosity; the absolute values and the flow curve use the resin model below.</div>
      <label class="f">Resin rheology</label><select data-k="options.viscosity.model.type"><option value="newtonian">Newtonian</option><option value="power">Power law</option><option value="carreau">Carreau</option><option value="cross">Cross</option></select>
      ${mrow}${num("options.viscosity.yield_Pa", "Resin yield stress", "Pa")}
      <div class="g3">${num("options.viscosity.gd_min", "Shear rate from", "1/s")}${num("options.viscosity.gd_max", "to", "1/s")}${num("options.viscosity.gd_ref", "Reference shear rate", "1/s")}</div>
      <label class="f">Maximum packing fraction φ<sub>m</sub></label><select data-k="options.viscosity.phi_m_mode"><option value="auto">Automatic: Farr–Groot for spheres, jammed packing otherwise (default)</option><option value="farr_groot">Farr–Groot random close packing (spheres)</option><option value="jamming">Jammed packing of these particles (simulated)</option><option value="manual">Entered</option></select>
      ${v.phi_m_mode === "manual" ? num("options.viscosity.phi_m", "φm", "") : `<label class="f">Particle friction</label><select data-k="options.viscosity.friction"><option value="none">Frictionless or lubricated: φm = random close packing (default)</option><option value="frictional">Frictional: φm × 0.585/0.64 (Boyer et al. 2011)</option></select>`}
      <div class="hint">Farr–Groot maps the sphere size distribution (several fillers, log-normal spreads, coatings included) onto rods on a line; it reproduced 3D packings of bidisperse spheres to within 0.01 up to a 1:10 size ratio, and gives 0.6435 for equal spheres. Other shapes are packed by the structure generator itself until they jam. Frictional particles jam in shear earlier; the result shows the other case as a range.</div>
      <label class="chk"><span class="switch"><input type="checkbox" data-k="options.viscosity.dilute_rve"><span></span></span> Intrinsic viscosity of non-spherical fillers from a dilute RVE</label>
      <div class="subh">Particle dynamics (spherical fillers)</div>
      <label class="chk"><span class="switch"><input type="checkbox" data-k="options.viscosity.dem.on"><span></span></span> Shear the fillers as moving particles (on by default)</label>
      <div class="g3">${num("options.viscosity.dem.n", "Particles", "", "1")}${num("options.viscosity.dem.strain", "Sheared strain", "")}${num("options.viscosity.dem.mu_f", "Friction coefficient", "")}</div>
      <div class="g3">${num("options.viscosity.dem.roughness_nm", "Surface roughness", "nm")}${num("options.viscosity.dem.hmin_nm", "Closest approach", "nm")}${num("options.viscosity.dem.hamaker_J", "Hamaker constant (empty: auto)", "J")}</div>
      <div class="g2">${num("options.viscosity.dem.bound_nm", "Bound resin layer", "nm")}${num("options.viscosity.dem.adhesion_mJ_m2", "Work of adhesion", "mJ/m²")}</div>
      <div class="hint">Surface chemistry that bulk data cannot give: a resin layer that moves with the particle (adsorbed resin, coupling agent) adds (1 + b/a)³ to the filler volume, and adhesion of touching surfaces (hydrogen bonding of untreated silica) pulls them together with 2πW·R*. Both weigh most on a fine filler. Leave them at 0, or fit them to one measured viscosity and predict other sizes and loadings.</div>
      <div class="g2">${num("options.viscosity.dem.rates", "Further shear rates (flow curve)", "", "1")}<div></div></div>
      <div class="hint">The attraction weighs against the viscous force as 1/(μγ̇): with van der Waals or adhesion a compound flows more easily the faster it is sheared. The particles are sheared again at this many rates, log-spaced over the flow curve's range, and the curve follows them (0: only the reference rate; skipped where the attraction is negligible).</div>
      <label class="f">Computed on</label><select data-k="options.viscosity.dem.backend"><option value="auto">Automatic: the NVIDIA GPU (CUDA) first, all CPU cores when there is none or a GPU run fails</option><option value="cuda">NVIDIA GPU (CUDA), the CPU if it fails</option><option value="cpu">CPU only</option></select>
      <div class="hint">Every filler particle moves, turns and collides in the sheared resin: lubrication between close surfaces, contacts with friction, and the van der Waals attraction of the filler across the resin (Hamaker constant from the refractive indices of the library, or entered). Roughness and the closest approach are lengths, so a fine filler meets them at a larger share of its size than a coarse one — this is where the particle-size effect comes from. Friction 0.25 reproduces the measured law of non-Brownian spheres (Boyer et al. 2011: μr 20.7 at 50 vol% and 103 at 55 vol%), friction 0 the frictionless Krieger–Dougherty curve. It is the best estimate above about 35 vol% or where particles touch.</div>
      <div class="subh">Capillary underfill (parallel plates)</div>
      <div class="g2">${num("options.viscosity.underfill.gap_um", "Gap", "µm")}${num("options.viscosity.underfill.length_mm", "Flow length", "mm")}${num("options.viscosity.underfill.gamma_mN_m", "Surface tension", "mN/m")}${num("options.viscosity.underfill.theta_deg", "Contact angle", "°")}</div>
      <div class="subh">Filler settling</div>
      <div class="g2">${num("options.viscosity.density_resin", "Resin density", "kg/m³")}${num("options.viscosity.settle_min", "Time before gelation", "min")}</div>
      <div class="hint">The RVE flow solve is reliable while every particle stays surrounded by resin on the grid (up to about 35 vol%); above that the particle dynamics (spheres) or, for other shapes, Krieger–Dougherty with the maximum packing fraction carries the result.</div>`;
  }
  function secVisc(p) {
    const fmtT = (s) => !isFinite(s) ? "—" : s < 60 ? `${fmt(s)} s` : s < 3600 ? `${fmt(s / 60)} min` : `${fmt(s / 3600)} h`;
    const c = p.curve;
    draws.push(() => CH.line($("#viscCurve"), { ylog: true, xlabel: "Filler volume fraction φ", ylabel: "Relative viscosity μr", series: [
      { x: c.phi, y: c.kd, color: "#0b62c4", width: 2.4, label: `Krieger–Dougherty (φm ${fmt(p.phi_m)}, [η] ${fmt(p.eta)})` },
      ...(c.kd_alt ? [{ x: c.phi, y: c.kd_alt, color: "#0b62c4", dash: [2, 3], label: `Krieger–Dougherty, ${p.friction ? "frictionless" : "frictional"} (φm ${fmt(p.phi_m_alt)})` }] : []),
      { x: c.phi, y: c.mp, color: "#8250df", dash: [6, 4], label: "Maron–Pierce" },
      { x: c.phi, y: c.batchelor, color: "#1a7f37", dash: [3, 3], label: "Batchelor" },
      { x: c.phi, y: c.hs, color: "#57606a", dash: [2, 3], label: "Hashin–Shtrikman lower bound" },
      { x: [p.phi], y: [p.mu_r_rve], color: "#d1242f", marker: true, width: 0, label: "RVE flow solve" },
      ...(p.dem && !p.dem.error ? [{ x: [p.phi], y: [p.dem.mu_r], color: "#bf8700", marker: true, width: 0, label: "Particle dynamics" }] : [])] }));
    const dm = p.dem;
    if (dm && !dm.error) draws.push(() => CH.line($("#viscDem"), { xlabel: "Sheared strain", ylabel: "Relative viscosity μr", series: [
      { x: dm.strain, y: dm.eta, color: "#bf8700", width: 1.8, label: "Instantaneous" },
      { x: [1, dm.strain[dm.strain.length - 1]], y: [dm.mu_r, dm.mu_r], color: "#0b62c4", dash: [6, 4], width: 1.6, label: "Mean from strain 1" }] }));
    if (dm && !dm.error && dm.anim) draws.push(async () => {
      // the solver's own sheared box, played in a small 3D view
      try {
        if (state.demPlayer) { state.demPlayer.stop(); state.demPlayer = null; }
        const id = state.resultJob, host = $("#demPlayer");
        if (!id || !host || !window.DemPlayer) return;
        const m = await (await fetch(`/api/jobs/${id}/view/meta.json`, { cache: "no-store" })).json();
        if (m.dem) state.demPlayer = await window.DemPlayer(host, id, m.dem, m.built, state.result && state.result.name);
      } catch (e) { console.warn("particle dynamics player", e); }
    });
    const demBox = !dm ? "" : dm.error ? `<div class="box" style="grid-column:1/-1"><div class="bt">Particle dynamics</div><p class="hint">Not run: ${esc(dm.error)}</p></div>`
      : `<div class="box" style="grid-column:1/-1"><div class="bt">Particle dynamics <span class="hint">${dm.n} spheres moving, turning and colliding in the resin sheared at ${fmt(dm.gd)} 1/s · ${dm.backend === "cuda" ? "NVIDIA GPU" : dm.gpu_fallback ? "CPU (the GPU run failed)" : "CPU"} · ${secs(dm.seconds)}</span></div>
        <div class="rgrid"><div>${canvas("viscDem", 230)}</div><div>
        ${kv([["Relative viscosity", `<b class="mono">${fmt(dm.mu_r)}</b> ± ${fmt(dm.se)} <span class="hint">mean over the strain after 1; standard error including the scatter between runs from other random packings</span>`],
              ["From lubrication / contacts", `${fmt(dm.eta_lub)} / ${fmt(dm.eta_contact)} <span class="hint">plus 1 + 2.5φ (resin and single-sphere stresslet)</span>`],
              ["Contacts per particle", fmt(dm.contacts_per_particle)],
              ["Particles", dm.counts.map((c, i) => `${c} × ${fmt(dm.d_um[i])} µm`).join(" + ") + ` <span class="hint">box ${fmt(dm.box_um)} µm</span>`],
              ["Friction · roughness · closest approach", `${fmt(dm.mu_f)} · ${fmt(dm.roughness_nm)} nm · ${fmt(dm.hmin_nm)} nm`],
              ["Bound layer · adhesion", `${fmt(dm.bound_nm)} nm · ${fmt(dm.adhesion_mJ_m2)} mJ/m²${dm.bound_nm ? ` <span class="hint">effective φ ${fmt(dm.phi_effective)}</span>` : ""}`],
              ["Hamaker constant", dm.hamaker.map((h) => `${esc(h.phase)} ${fmt(h.J)} J <span class="hint">${esc(h.basis)}</span>`).join("<br>")]])}
        </div></div>
        ${dm.anim ? `<div class="bt" style="margin-top:10px">The sheared particles <span class="hint">the solver's own box (not the RVE above), the top moving +x and the bottom −x; drag to turn it</span></div><div id="demPlayer"></div>` : ""}
        <p class="hint">The fillers are not held: each one moves and turns with the resin, the films between close surfaces resist squeezing and sliding (lubrication), touching surfaces push and rub (friction), and the van der Waals attraction pulls them together. Roughness, the closest approach and the attraction carry lengths of their own, so the same loading of a finer filler comes out stiffer. Above about 35 vol%, or where the particles of the RVE touch, this is the best estimate.</p></div>`;
    const f = p.flow;
    draws.push(() => CH.line($("#viscFlow"), { xlog: true, ylog: true, xlabel: "Shear rate (1/s)", ylabel: "Viscosity (Pa·s)", series: [
      { x: f.gd, y: f.mu_compound, color: "#d1242f", width: 2.4, label: "Compound" },
      { x: f.gd, y: f.mu_resin, color: "#0b62c4", dash: [6, 4], label: "Resin" },
      ...(f.dem_points ? [{ x: f.dem_points.gd, y: f.dem_points.mu, color: "#bf8700", marker: true, width: 0, label: "Particle dynamics" }] : [])] }));
    const uf = p.underfill;
    draws.push(() => setupUfAnim(p));
    const refs = Object.entries(p.refs).map(([k, v]) => [esc(k), fmt(v)]);
    return `<div class="rgrid">
      <div class="box"><div class="bt">Relative viscosity <span class="hint">μ / μ<sub>resin</sub>, φ = ${fmt(100 * p.phi)} vol% filler${p.bubbles ? ` · ${fmt(100 * p.bubbles)} vol% bubbles` : ""}</span></div>
        ${kv([["Best estimate", `<b class="mono">${fmt(p.mu_r)}</b> <span class="hint">${esc(p.basis)}</span>`],
              ["RVE flow solve (isotropic mean)", fmt(p.mu_r_rve) + (p.ci95 ? ` ± ${fmt(p.ci95)}` : "")],
              p.dem && !p.dem.error ? ["Particle dynamics", `${fmt(p.dem.mu_r)} ± ${fmt(p.dem.se)}`] : null,
              ["Shear yz / xz / xy", ["yz", "xz", "xy"].map((k) => fmt(p.shear[k])).join(" / ")],
              ["Extension (η<sub>E</sub> / 3μ)", fmt(p.extension)],
              ["Maximum packing fraction φm", `${fmt(p.phi_m)} <span class="hint">${esc(p.phi_m_basis || "")}${p.jamming ? ` · ${p.jamming.n_particles} particles` : ""}${p.friction ? ` · frictional (random close packing ${fmt(p.phi_m_rcp)} × 0.914)` : ""}</span>`],
              p.farr_groot != null && !/Farr/.test(p.phi_m_basis || "") ? ["Farr–Groot random close packing", fmt(p.farr_groot)] : null,
              p.desmond_weeks ? ["Desmond–Weeks φ<sub>RCP</sub>", `${fmt(p.desmond_weeks.phi)} <span class="hint">δ ${fmt(p.desmond_weeks.delta)}, skewness ${fmt(p.desmond_weeks.skewness)}${p.desmond_weeks.in_range ? "" : " · outside its fitted range δ ≤ 0.4"}</span>`] : null,
              p.kd_alt != null ? [`Krieger–Dougherty, ${p.friction ? "frictionless" : "frictional"} particles`, `${fmt(p.kd_alt)} <span class="hint">φm ${fmt(p.phi_m_alt)}</span>`] : null,
              ["Intrinsic viscosity [η]", `${fmt(p.eta)} <span class="hint">${esc(p.eta_basis)}</span>`],
              p.kd_phi_m_fit ? ["φm that puts Krieger–Dougherty through the RVE value", fmt(p.kd_phi_m_fit)] : null])}
        <div class="bt" style="margin-top:8px">Closed forms at this φ</div>${kv(refs)}</div>
      <div class="box"><div class="bt">Relative viscosity against filler loading</div>${canvas("viscCurve", 300)}</div>
      ${demBox}
      <div class="box"><div class="bt">Flow curve <span class="hint">resin ${esc(f.model.type)}${f.yield_Pa ? `, yield stress ${fmt(f.yield_Pa)} Pa` : ""}; shear-rate amplification in the resin ${fmt(f.amplification)}</span></div>${canvas("viscFlow", 280)}
        ${kv([[`Compound at ${fmt(p.gd_ref)} 1/s`, `<b class="mono">${fmt(p.mu_compound_ref)}</b> Pa·s`], [`Resin at ${fmt(p.gd_ref)} 1/s`, `${fmt(p.mu_resin_ref)} Pa·s`]])}
        ${f.dem_rates ? `<div class="hint">The compound follows the particle dynamics at ${f.dem_rates.gd.length} shear rates (dots; relative viscosity ${f.dem_rates.mu_r.map(fmt).join(" → ")} from ${fmt(f.dem_rates.gd[0])} to ${fmt(f.dem_rates.gd[f.dem_rates.gd.length - 1])} 1/s), linear in log-log between them.</div>` : (p.dem && p.dem.rate_note ? `<div class="hint">Particle dynamics: ${esc(p.dem.rate_note)}.</div>` : "")}</div>
      <div class="box"><div class="bt">Capillary underfill <span class="hint">parallel plates, gap ${fmt(uf.gap_um)} µm, flow length ${fmt(uf.length_mm)} mm, γ ${fmt(uf.gamma_mN_m)} mN/m, θ ${fmt(uf.theta_deg)}°</span></div>
        ${kv([["Filling time, compound", `<b class="mono">${fmtT(uf.t_compound_s)}</b>`], ["Filling time, resin alone", fmtT(uf.t_resin_s)]])}
        <div class="hint">Washburn's law t = 3μL² / (hγ cos θ) with the compound's viscosity at the reference shear rate; bumps, the dispensing pattern and curing during the flow are not included.</div></div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Flow front under the die <span class="hint">top view, the same gap for the resin alone and the compound; the front advances as x = L·√(t / t_fill)</span>
          <div class="grow"></div><button class="btn sm" id="ufPlay">${I("play")}Play</button><button class="btn sm" id="ufGif" title="Save the filling as an animated GIF">${I("download")}GIF</button></div>
        ${canvas("ufAnim", 230)}
        <div class="row gap" style="align-items:center;margin-top:4px"><input type="range" id="ufTime" min="0" max="1000" value="0" style="flex:1"><span class="mono" id="ufRead" style="min-width:330px;text-align:right"></span></div>
        <div class="hint">The dots are the bumps (schematic). The compound's front starts as fast as it can and slows as the wetted length grows; the resin alone fills ${fmt(uf.t_compound_s / uf.t_resin_s)}× faster.</div></div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Resin flow inside the RVE <span class="hint">simple shear in the xy plane at the reference shear rate</span><div class="grow"></div>
          ${viewBtn("visc_gd", "Local shear rate")}${linesBtn("visc_v", "Animated resin flow")}</div>
        <p class="hint">The animated view moves tracers along the resin streamlines at the local speed: the layers slide past each other, the resin swerves around the fillers and hurries through the narrow gaps between them — the places where the local shear rate, and so the extra dissipation that makes the compound viscous, is highest.</p></div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Filler settling before gelation <span class="hint">${fmt(p.settle_min)} min, Stokes velocity × hindered settling (1 − φ)^4.65</span></div>
        ${tbl(["Filler", "Diameter (µm)", "Density (g/cm³)", "Velocity (µm/min)", "Distance (µm)"], p.settling.map((s) => [esc(s.name), fmt(s.d_um), fmt(s.rho), fmt(s.v_m_s * 6e7), fmt(s.distance_m * 1e6)]))}</div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Solver record</div>${tbl(["Load", "Iterations", "Residual", "Converged", "Time"], Object.entries(p.stats).map(([k, s]) => [k, s.iterations, fmt(s.residual), s.converged ? "yes" : "no", secs(s.seconds)]))}</div>
    </div>`;
  }
  /* Underfill flow front: two lanes under the same die, the resin alone and
     the compound, each front at x = L sqrt(t / t_fill) (Washburn). Played over
     eight seconds from 0 to the compound's filling time, or scrubbed with the
     slider. */
  let ufAnimState = null;
  function setupUfAnim(p) {
    const cv = $("#ufAnim"), sl = $("#ufTime"), rd = $("#ufRead"), btn = $("#ufPlay");
    if (!cv || !sl) return;
    if (ufAnimState && ufAnimState.raf) cancelAnimationFrame(ufAnimState.raf);
    const uf = p.underfill, tR = uf.t_resin_s, tC = uf.t_compound_s, tEnd = 1.05 * Math.max(tR, tC);
    const fmtT = (s) => s < 60 ? `${s.toFixed(0)} s` : s < 3600 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(2)} h`;
    const st = ufAnimState = { raf: null, playing: false, t: 0 };
    const draw = () => {
      const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 600, h = cv.clientHeight || 230;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      const g = cv.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      g.font = '11px "Segoe UI", system-ui, sans-serif';
      const L0 = 118, R0 = 14, laneH = (h - 46) / 2, W = w - L0 - R0;
      const lanes = [["Resin alone", tR, "#0b62c4", "rgba(11,98,196,0.16)"], ["Compound", tC, "#d1242f", "rgba(209,36,47,0.16)"]];
      lanes.forEach(([name, tf, col, fill], k) => {
        const y0 = 14 + k * (laneH + 14);
        const x = Math.min(1, Math.sqrt(st.t / tf));
        g.fillStyle = "#f6f8fa"; g.fillRect(L0, y0, W, laneH);
        g.fillStyle = fill; g.fillRect(L0, y0, W * x, laneH);
        // bumps: a staggered array, schematic
        g.fillStyle = "#afb8c1";
        const nb = 22, pitch = W / nb;
        for (let i = 0; i < nb; i++) for (let j = 0; j < 4; j++) {
          const bx = L0 + (i + 0.5 + (j % 2) * 0.5) * pitch, by = y0 + (j + 0.5) * laneH / 4;
          if (bx < L0 + W) { g.beginPath(); g.arc(bx, by, Math.min(pitch, laneH / 4) * 0.22, 0, 2 * Math.PI); g.fill(); }
        }
        g.strokeStyle = col; g.lineWidth = 2.4; g.beginPath();
        // a slightly wavy front: the bumps hold it back where it meets them
        for (let yy = 0; yy <= laneH; yy += 2) {
          const xf = L0 + W * x - (x > 0 && x < 1 ? 3 * Math.abs(Math.sin(yy / laneH * 4 * Math.PI)) : 0);
          if (yy === 0) g.moveTo(xf, y0 + yy); else g.lineTo(xf, y0 + yy);
        }
        g.stroke();
        g.strokeStyle = "#d0d7de"; g.lineWidth = 1; g.strokeRect(L0 + 0.5, y0 + 0.5, W - 1, laneH - 1);
        g.fillStyle = "#1f2328"; g.textAlign = "right";
        g.fillText(name, L0 - 8, y0 + laneH / 2 - 2);
        g.fillStyle = "#57606a";
        g.fillText(x >= 1 ? `filled at ${fmtT(tf)}` : `${(x * uf.length_mm).toFixed(1)} mm`, L0 - 8, y0 + laneH / 2 + 13);
      });
      g.fillStyle = "#57606a"; g.textAlign = "left";
      g.fillText("dispensed edge", L0, h - 8);
      g.textAlign = "right"; g.fillText(`far edge, ${fmt(uf.length_mm)} mm`, L0 + W, h - 8);
      const xr = Math.min(1, Math.sqrt(st.t / tR)), xc = Math.min(1, Math.sqrt(st.t / tC));
      rd.textContent = `t = ${fmtT(st.t)} · resin ${(xr * uf.length_mm).toFixed(1)} mm · compound ${(xc * uf.length_mm).toFixed(1)} mm`;
    };
    const setT = (t) => { st.t = Math.max(0, Math.min(tEnd, t)); sl.value = Math.round(1000 * st.t / tEnd); draw(); };
    sl.oninput = () => { st.playing = false; btn.innerHTML = `${I("play")}Play`; setT(tEnd * sl.value / 1000); };
    btn.onclick = () => {
      st.playing = !st.playing;
      btn.innerHTML = st.playing ? `${I("stop")}Pause` : `${I("play")}Play`;
      if (st.playing) {
        if (st.t >= tEnd) st.t = 0;
        let last = performance.now();
        const step = (now) => {
          if (!st.playing || ufAnimState !== st || !document.body.contains(cv)) return;
          setT(st.t + (now - last) / 8000 * tEnd);
          last = now;
          if (st.t >= tEnd) { st.playing = false; btn.innerHTML = `${I("play")}Play`; return; }
          st.raf = requestAnimationFrame(step);
        };
        st.raf = requestAnimationFrame(step);
      }
    };
    const gb = $("#ufGif");
    if (gb) gb.onclick = async () => {
      if (!window.GifRec) return;
      st.playing = false; btn.innerHTML = `${I("play")}Play`;
      const keep = st.t, n = 60;
      gb.disabled = true;
      try {
        await window.GifRec.record({ count: n + 1, delay: 90, maxWidth: 900, top: 22,
          filename: `${(state.result ? state.result.name : "underfill").replace(/[^\w.-]+/g, "_")}_underfill.gif`,
          frame: async (k) => { setT(tEnd * k / n); return cv; },
          overlay: (g, w) => { g.font = "13px system-ui, sans-serif"; g.fillStyle = "#1f2328"; g.textAlign = "left"; g.fillText(`Capillary underfill · ${rd.textContent}`, 10, 16); },
          onProgress: (i, m) => { gb.textContent = `${Math.round(100 * i / m)} %`; } });
      } catch (e) { console.warn("GIF", e); }
      finally { gb.disabled = false; gb.innerHTML = `${I("download")}GIF`; setT(keep); }
    };
    setT(0.35 * tR);
  }
  function secPerm(p) {
    const fl = Object.values(p.stats || {}).map((s) => s.field_lines).find(Boolean);
    return `<div class="rgrid"><div class="box"><div class="bt">Permeability tensor (diagonal)<div class="grow"></div>${viewBtn("perm", "Velocity field")}${fl && fl.published ? linesBtn("perm_", "Streamlines") : ""}</div>
        ${tbl(["Component", "m²", "darcy", "Connected"], ["x", "y", "z"].map((a, i) => [`${a}${a}`, fmt(p.diag[i]), fmt(p.darcy[i]), p.percolation[i] ? "●" : "○"]))}${poroNote(p)}</div>
      <div class="box"><div class="bt">Derived quantities</div>${kv([["Static flow resistivity (gas viscosity)", withUnit(p.flow_resistivity_Pa_s_m2, "Pa·s/m²")], ["Kozeny–Carman constant φ³/(K·S_v²)", fmt(p.kozeny_constant)],
        ["Hydraulic tortuosity ⟨|u|⟩ / ⟨u∥⟩", fmt(p.hydraulic_tortuosity)], ["Open-pore specific surface S_v", withUnit(p.specific_surface_open_1pm, "1/m")],
        ["Forchheimer coefficient β", withUnit(p.forchheimer_beta_1_m, "1/m")], ["Velocity at which inertia adds 10 %", withUnit(p.non_darcy_velocity_m_s, "m/s")],
        ["Klinkenberg slip factor b", withUnit(p.klinkenberg_b_Pa, "Pa")], ["Apparent gas permeability (1 atm)", withUnit(p.apparent_gas_permeability_m2, "m²")],
        ...lineRows(fl)])}
        ${p.forchheimer_beta_1_m ? '<p class="hint">Darcy\'s law is linear in the velocity; the Forchheimer coefficient (Ergun correlation on the computed K) says where that stops holding. The Klinkenberg factor is the gas-slip correction for the same pore size, so a gas measurement reads higher than the intrinsic permeability. Both are correlations on top of the solve, not separate solves.</p>' : ""}
        ${tbl(["Direction", "Iterations", "Converged", "Time"], Object.entries(p.stats || {}).map(([d, s]) => [d, s.iterations ?? "—", s.converged === null || s.converged === undefined ? "—" : s.converged ? pill("pass") : pill("fail"), `${fmt(s.seconds)} s`]))}</div></div>`;
  }
  function secFiltration(p) {
    if (p.error) return `<div class="box"><div class="errs">${esc(p.error)}</div></div>`;
    const rows = p.by_size || [];
    const hasQ = rows.some((r) => r.quality_factor_1_Pa);
    draws.push(() => CH.line($("#filtE"), { xlog: true, xlabel: "Particle diameter (µm)", ylabel: "Single-pass efficiency (%)", ymin: 0, ymax: 100,
      series: [{ x: rows.map((r) => r.diameter_um), y: rows.map((r) => 100 * r.efficiency), color: "#0a8fa8", width: 2.8, marker: true, label: "Efficiency" }] }));
    if (hasQ) draws.push(() => CH.line($("#filtQ"), { xlog: true, xlabel: "Particle diameter (µm)", ylabel: "Quality factor (1/Pa)", ymin: 0,
      series: [{ x: rows.map((r) => r.diameter_um), y: rows.map((r) => r.quality_factor_1_Pa), color: "#8250df", width: 2.4, marker: true, label: "−ln(P) / Δp" }] }));
    const tiles = [["Most penetrating size", p.mpps_um, "µm"], ["Efficiency at that size", 100 * (p.mpps_efficiency || 0), "%"],
      ["Pressure drop", p.pressure_drop_Pa, "Pa"], ["Face velocity", p.face_velocity_m_s, "m/s"]];
    return `<div class="big">${tiles.map(([l, v, un]) => `<div class="tile"><div class="l">${l}</div><div class="v">${fmt(v)} <span style="font-size:12px;color:var(--muted)">${un}</span></div></div>`).join("")}</div>
      <div class="rgrid"><div class="box"><div class="bt">Efficiency against particle size<div class="grow"></div>${viewBtn(p.mpps_path_key || "path_filt_", "Show the particle tracks")}</div>${canvas("filtE", 300)}
        <p class="hint">The smallest particles are caught by Brownian motion and the largest by interception and impaction, so the efficiency passes through a minimum — the most penetrating particle size, which is what a filter is specified by.</p></div>
      ${hasQ ? `<div class="box"><div class="bt">Quality factor</div>${canvas("filtQ", 300)}<p class="hint">−ln(penetration) divided by the pressure drop: the efficiency weighed against the energy it costs. The pressure drop follows from the computed permeability.</p></div>` : ""}
      <div class="box" style="grid-column:1/-1"><div class="bt">Per particle size <span class="hint">${p.n_particles} particles per size · ${fmt(p.thickness_um)} µm thick · ${esc(p.direction)} direction · mean free path ${fmt(p.mean_free_path_nm)} nm</span></div>
        ${tbl(["Diameter", "Efficiency", "Penetration", "Captured", "Passed through", "Returned to the inlet", "Stokes number", "Péclet number", "Quality factor"],
          rows.map((r) => [withUnit(r.diameter_um, "µm"), fmt(100 * r.efficiency) + " %", fmt(100 * r.penetration) + " %",
            `${r.captured} of ${r.captured + r.passed}`, String(r.passed), String(r.returned_to_inlet ?? 0),
            fmt(r.stokes_number), fmt(r.peclet), withUnit(r.quality_factor_1_Pa, "1/Pa")]))}
        <p class="hint">The efficiency counts the particles that were resolved: captured, or carried out of the far face. A particle that diffuses back out of the inlet never entered the filter and is excluded rather than counted as having passed through it, which is why the captured count can be smaller than the number started at the smallest sizes.</p>
        <p class="hint">A Stokes number well below one means the particle follows the streamlines; a Péclet number well below one means Brownian motion dominates over convection. Electrostatic capture, rebound and loading as particles deposit are not modelled.</p></div></div>`;
  }
  function secTort(p) {
    const kn = p.knudsen;
    return `<div class="rgrid"><div class="box"><div class="bt">Tortuosity and effective diffusivity<div class="grow"></div>${viewBtn("tort", "Concentration field")}</div>
        ${tbl(["Direction", "τ", "D_eff/D₀", "Formation factor F"], Object.entries(p.by_direction).map(([d, v]) => [d, fmt(v.tortuosity), fmt(v.d_eff), fmt(v.formation_factor)]))}${poroNote(p)}
        ${kv([["Mean tortuosity factor", fmt(p.mean_tortuosity)], ["MacMullin number", fmt(p.macmullin_number)], ["Archie cementation exponent m", fmt(p.archie_exponent)]])}</div>
      ${kn ? `<div class="box"><div class="bt">Gas diffusion with Knudsen effects <span class="hint">${fmt(state.result.spec.options.temperature_K)} K</span></div>${kv([["Mean open-pore diameter", withUnit(kn.d_pore_um, "µm")], ["Mean free path", withUnit(kn.mean_free_path_nm, "nm")],
        ["Knudsen number", `${fmt(kn.knudsen_number)} (${esc(kn.regime)} regime)`], ["Knudsen diffusivity D_K", withUnit(kn.D_knudsen_m2_s, "m²/s")], ["Bosanquet pore diffusivity", withUnit(kn.D_bosanquet_m2_s, "m²/s")],
        ["Effective diffusivity, molecular", withUnit(kn.D_eff_molecular_m2_s, "m²/s")], ["Effective diffusivity, with Knudsen", withUnit(kn.D_eff_bosanquet_m2_s, "m²/s")]])}</div>` : ""}</div>`;
  }
  function secAcoustics(p) {
    if (p.error) return `<div class="box"><div class="errs">${esc(p.error)}</div></div>`;
    const sp = p.spectrum;
    draws.push(() => CH.line($("#acA"), { xlog: true, xlabel: "Frequency (Hz)", ylabel: "Absorption coefficient α", ymin: 0, ymax: 1, series: [{ x: sp.freq_hz, y: sp.alpha, color: "#bf3989", width: 2.8, label: `Hard-backed layer, ${fmt(p.thickness_mm)} mm` }] }));
    draws.push(() => CH.line($("#acZ"), { xlog: true, xlabel: "Frequency (Hz)", ylabel: "Z_s / ρ₀c₀", series: [{ x: sp.freq_hz, y: sp.z_real, color: "#0b62c4", label: "Real part" }, { x: sp.freq_hz, y: sp.z_imag, color: "#e36209", dash: [6, 4], label: "Imaginary part" }] }));
    return `<div class="big">${[["NRC", p.nrc, ""], ["Flow resistivity", p.flow_resistivity_Pa_s_m2, "Pa·s/m²"], ["Tortuosity α∞", p.tortuosity, ""], ["Viscous length Λ", p.viscous_length_um, "µm"], ["Thermal length Λ'", p.thermal_length_um, "µm"], ["Open porosity", p.porosity, ""]]
        .map(([l, v, un]) => `<div class="tile"><div class="l">${l}</div><div class="v">${fmt(v)} <span style="font-size:12px;color:var(--muted)">${un}</span></div></div>`).join("")}</div>
      <div class="rgrid"><div class="box"><div class="bt">Normal-incidence absorption</div>${canvas("acA", 300)}</div><div class="box"><div class="bt">Surface impedance</div>${canvas("acZ", 300)}</div>
      <div class="box"><div class="bt">Octave bands</div>${tbl(["Frequency", "α"], Object.entries(p.alpha_at || {}).map(([f, a]) => [`${f} Hz`, fmt(a)]))}<p class="hint">Johnson–Champoux–Allard model; all five parameters are computed from the RVE fields, none is fitted.</p></div></div>`;
  }
  function secRadiation(p) {
    if (p.error) return `<div class="box"><div class="errs">${esc(p.error)}</div></div>`;
    draws.push(() => CH.line($("#radK"), { xlabel: "Temperature (K)", ylabel: "k_rad (W/m·K)", ymin: 0, series: [{ x: p.k_rad_curve.T_K, y: p.k_rad_curve.k_W_mK, color: "#d15704", width: 2.6, label: "Rosseland radiative conductivity" }] }));
    if (p.free_path) draws.push(() => CH.hist($("#radF"), { x: p.free_path.length_um, y: p.free_path.probability, color: "#d4a72c", color2: "#ffe8a3", xlabel: "Ray free path (µm)", ylabel: "Probability", label: "Ray free-path distribution" }));
    return `<div class="rgrid"><div class="box"><div class="bt">Extinction coefficient</div>${kv([["β along x / y / z", withUnit(null) === "—" ? p.beta_1pm.map(fmt).join(" / ") + " 1/m" : ""], ["Mean β", withUnit(p.beta_mean_1pm, "1/m")], ["Standard deviation x / y / z", p.beta_std_1pm.map(fmt).join(" / ") + " 1/m"],
        ["Geometric reference S_v/(4φ)", withUnit(p.mean_chord_beta_1pm, "1/m")], [`Radiative conductivity at ${fmt(p.temperature_K)} K`, withUnit(p.k_rad_W_mK, "W/m·K")],
        ...(p.k_total_W_mK ? [["Conduction alone (solved)", withUnit(p.k_conduction_W_mK, "W/m·K")],
                              ["Conduction + radiation", withUnit(p.k_total_W_mK, "W/m·K")],
                              ["Radiative share of the total", fmt(100 * p.radiation_share) + " %"]] : []),
        ["Rays", `${p.sources} sources × ${p.rays} rays`]])}
        <p class="hint">Ray casting in the pore space with an opaque solid (PuMA). The Rosseland conductivity is 16 n² σ_SB T³ / (3β).</p></div>
      <div class="box"><div class="bt">Radiative conductivity</div>${canvas("radK", 260)}</div>${p.free_path ? `<div class="box"><div class="bt">Free path of the rays</div>${canvas("radF", 260)}</div>` : ""}</div>`;
  }
  function secMorph(P) {
    let h = "";
    if (P.porosity) {
      const q = P.porosity;
      h += `<div class="big">${[["Total porosity", 100 * q.total, "%"], ["Open porosity", 100 * q.open, "%"], ["Closed porosity", 100 * q.closed, "%"]].map(([l, v, un]) => `<div class="tile"><div class="l">${l}</div><div class="v">${fmt(v)} <span style="font-size:12px;color:var(--muted)">${un}</span></div></div>`).join("")}
        <div class="tile"><div class="l">Connected along</div><div class="v">${["x", "y", "z"].map((a) => (q.connected_axes[a] ? a : "·")).join(" ")}</div></div></div>`;
    }
    const M = P.morphology || {};
    h += `<div class="rgrid">` + Object.entries(M).map(([k, m]) => {
      if (m.error) return `<div class="box"><div class="bt">${esc(m.name)}</div><div class="errs">${esc(m.error)}</div></div>`;
      const sd = m.size_distribution || {}, ch = m.chords || {}, tp = m.two_point || {};
      if (sd.diameter_um) draws.push(() => CH.hist($("#psd_" + k), { x: sd.diameter_um, y: sd.volume_fraction, xlabel: "Local thickness (µm)", ylabel: "Volume fraction", label: "Distribution", line: { y: sd.cdf, label: "Cumulative", color: "#e36209" } }));
      if (tp.r_um) draws.push(() => CH.line($("#tp_" + k), { xlabel: "Distance r (µm)", ylabel: "S₂(r)", series: [{ x: tp.r_um, y: tp.s2, color: "#8250df", label: "Two-point correlation" }] }));
      return `<div class="box"><div class="bt">${esc(m.name)}<div class="grow"></div>${viewBtn("lt_" + k, "Local thickness")}</div>
        ${kv([["Volume fraction", withUnit(100 * m.volume_fraction, "%")], ["D10 / D50 / D90 (local thickness)", `${fmt(sd.d10_um)} / ${fmt(sd.d50_um)} / ${fmt(sd.d90_um)} µm`], ["Specific surface area", withUnit(m.specific_surface_1pm, "1/m")],
          ["Mean intercept length x / y / z", m.mean_intercept_length_um ? m.mean_intercept_length_um.map(fmt).join(" / ") + " µm" : "—"], ["Mean chord length x / y / z", ["x", "y", "z"].map((a) => fmt(ch[a] && ch[a].mean_um)).join(" / ") + " µm"],
          ["Clusters (periodic) / largest share", `${m.clusters.n_clusters} / ${fmt(100 * m.clusters.largest_fraction)} %`], ["Connected x / y / z", ["x", "y", "z"].map((a) => (m.percolates[a] ? "●" : "○")).join(" ")], ["Correlation length", withUnit(tp.correlation_length_um, "µm")]])}
        ${sd.diameter_um ? canvas("psd_" + k, 220) : ""}${tp.r_um ? canvas("tp_" + k, 200) : ""}</div>`;
    }).join("");
    const pr = P.profiles;
    if (pr) {
      const labels = state.result.spec.labels;
      draws.push(() => CH.line($("#profZ"), { xlabel: "z (µm)", ylabel: "Volume fraction", ymin: 0, series: labels.map((l, i) => ({ x: pr.position_um.z, y: pr.profiles.z[i], color: PHASE_COLORS[i % PHASE_COLORS.length] === "#d9dde3" ? "#8c959f" : PHASE_COLORS[i % PHASE_COLORS.length], label: l.name })) }));
      h += `<div class="box"><div class="bt">Volume fraction profile along z</div>${canvas("profZ", 240)}<p class="hint">Slice averages over the xy plane; a flat profile indicates a homogeneous structure.</p></div>`;
    }
    return h + "</div>";
  }
  function secPercolation(pc) {
    return `<div class="rgrid">` + Object.entries(pc).map(([gkey, g]) => {
      if (g.error) return `<div class="box"><div class="bt">${esc(g.name)}</div><div class="errs">${esc(g.error)}</div></div>`;
      const rows = Object.entries(g.by_direction).map(([d, v]) => [d, v.percolates ? "● spanning" : "○ blocked", fmt(v.geodesic_tortuosity),
        v.spanning_fraction ? fmt(100 * v.spanning_fraction) : "—", fmt(100 * v.dead_end_fraction), fmt(100 * v.isolated_fraction)]);
      const box = `<div class="box"><div class="bt">${esc(g.name)} <span class="hint">${g.connectivity}-connectivity</span><div class="grow"></div>
          ${viewBtn("perc_" + gkey, "Path classes")}${viewBtn("geo_" + gkey, "Geodesic length")}</div>
        ${tbl(["Direction", "Percolation", "Geodesic tortuosity", "Spanning %", "Dead end %", "Isolated %"], rows)}
        <p class="hint">The geodesic tortuosity is the mean shortest path through the spanning cluster divided by the straight distance, a lower bound for the diffusive tortuosity factor. The path field shows 3 = spanning, 2 = dead end, 1 = isolated.</p></div>`;
      // which particle actually gets through, largest first
      const withSph = Object.entries(g.by_direction).filter(([, v]) => (v.passable_spheres || []).length);
      if (!withSph.length) return box;
      const cid = "sph_" + gkey;
      draws.push(() => CH.line($("#" + cid), { xlabel: "Sphere diameter (µm)", ylabel: "Detour of the route (path / straight)", series: withSph.map(([d, v], i) => ({
        x: v.passable_spheres.map((r) => r.diameter_um).reverse(), y: v.passable_spheres.map((r) => r.tortuosity).reverse(),
        color: ["#0b62c4", "#e36209", "#1a7f37"][i % 3], width: 2.4, marker: true, label: `${d} direction` })) }));
      const tiles = `<div class="big">${withSph.map(([d, v]) => `<div class="tile"><div class="l">Largest sphere through ${d}</div>
        <div class="v">${fmt(v.passable_spheres[0].diameter_um)} <span style="font-size:12px;color:var(--muted)">µm</span></div>
        <div class="d">${v.passable_spheres[0].n_paths} route${v.passable_spheres[0].n_paths === 1 ? "" : "s"}</div></div>`).join("")}</div>`;
      const tables = withSph.map(([d, v]) => `<div class="bt" style="margin-top:10px">Sizes that pass along ${d}<div class="grow"></div>${viewBtn(`path_${gkey}_${d}`, "Show the particle and its path")}</div>
        ${tbl(["Sphere diameter", "Independent routes", "Shortest route", "Detour"], v.passable_spheres.map((r) => [withUnit(r.diameter_um, "µm"), String(r.n_paths), withUnit(r.path_length_um, "µm"), fmt(r.tortuosity)]))}`).join("");
      return box + `<div class="box" style="grid-column:1/-1"><div class="bt">Which particle passes through</div>
        <div class="hint">A sphere of diameter d passes when the points that stay at least d/2 away from the solid form a connected set from one face to the opposite one. The list starts with the largest size that still passes — the bubble-point size of that direction — and the 3D view shows the sphere travelling along the route of its centre.</div>
        ${tiles}${tables}${canvas(cid, 250)}</div>`;
    }).join("") + "</div>";
  }
  function secPorosimetry(p) {
    if (p.error) return `<div class="box"><div class="errs">${esc(p.error)}</div></div>`;
    const it = p.intrusion, sd = p.entry_size_distribution;
    const ex = p.extrusion;
    if (it) draws.push(() => CH.line($("#mipC"), { xlog: true, xlabel: "Pressure (Pa)", ylabel: "Intruded pore volume fraction", ymin: 0, ymax: 1, series: [
      { x: it.pressure_Pa, y: it.saturation, color: "#0a8fa8", width: 2.6, marker: true, label: `Intrusion (${p.fluid})` },
      ...(ex ? [{ x: ex.pressure_Pa, y: ex.saturation, color: "#e36209", width: 2, dash: [6, 4], label: "Extrusion" }] : [])] }));
    if (sd) draws.push(() => CH.hist($("#mipD"), { xlog: true, x: sd.diameter_um, y: sd.dS, xlabel: "Pore-entry diameter (µm)", ylabel: "Intruded fraction per step", color: "#8250df", color2: "#d8c7f7", label: "Pore volume entered through this diameter" }));
    const tp = p.through_pore || {};
    return `<div class="big">${[["Median entry diameter", p.d50_um, "µm"], ["Threshold pressure", p.threshold_pressure_Pa, "Pa"], ["Maximum intrusion", p.max_intrusion, ""]].map(([l, v, un]) => `<div class="tile"><div class="l">${l}</div><div class="v">${fmt(v)} <span style="font-size:12px;color:var(--muted)">${un}</span></div></div>`).join("")}</div>
      <div class="rgrid"><div class="box"><div class="bt">Intrusion curve<div class="grow"></div>${viewBtn("mip_d", "Entry-diameter field")}</div>${canvas("mipC", 280)}</div>
      <div class="box"><div class="bt">Pore-entry size distribution</div>${canvas("mipD", 280)}<p class="hint">Entry diameters at 10 / 50 / 90 % intrusion: ${fmt(p.d10_um)} / ${fmt(p.d50_um)} / ${fmt(p.d90_um)} µm.</p></div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Largest through-pore and bubble point</div>${tbl(["Direction", "Diameter", `Pressure (${p.fluid})`, "Bubble point, water", "Bubble point, isopropanol"], Object.entries(tp).map(([d, v]) => (v ? [d, withUnit(v.diameter_um, "µm"), withUnit(v.pressure_fluid_Pa, "Pa"), withUnit(v.bubble_point_water_Pa, "Pa"), withUnit(v.bubble_point_ipa_Pa, "Pa")] : [d, "no through-pore", "—", "—", "—"])))}
        <p class="hint">The largest sphere that passes from one face to the opposite face; its Young–Laplace pressure is the bubble point of a fully wetting liquid.</p></div></div>`;
  }
  function secNetwork(p) {
    if (p.error) return `<div class="box"><div class="errs">${esc(p.error)}</div></div>`;
    const h1 = p.pore_inscribed_diameter_um, h2 = p.throat_inscribed_diameter_um, cd = p.coordination_distribution;
    if (h1) draws.push(() => CH.hist($("#pnP"), { x: h1.center, y: h1.fraction, xlabel: "Pore inscribed diameter (µm)", ylabel: "Number fraction", color: "#0b62c4", label: "Pores" }));
    if (h2) draws.push(() => CH.hist($("#pnT"), { x: h2.center, y: h2.fraction, xlabel: "Throat inscribed diameter (µm)", ylabel: "Number fraction", color: "#e36209", color2: "#ffd8b5", label: "Throats" }));
    if (cd && cd.center.length) draws.push(() => CH.hist($("#pnZ"), { x: cd.center, y: cd.fraction, xlabel: "Coordination number", ylabel: "Fraction of pores", color: "#1a7f37", color2: "#b4e5c2", label: "Coordination" }));
    return `<div class="big">${[["Pores", p.n_pores, ""], ["Throats", p.n_throats, ""], ["Mean coordination", p.coordination_mean, ""], ["Isolated pores", p.isolated_pores, ""], ["Pore density", p.pore_density_per_mm3, "1/mm³"], ["Mean throat length", (p.throat_length_um || {}).mean, "µm"]]
        .map(([l, v, un]) => `<div class="tile"><div class="l">${l}</div><div class="v">${fmt(v)} <span style="font-size:12px;color:var(--muted)">${un}</span></div></div>`).join("")}</div>
      <div class="rgrid"><div class="box"><div class="bt">Pore sizes<div class="grow"></div>${state.result.view ? `<button class="btn sm" data-network>${I("pore_network")}Show network</button>` : ""}</div>${canvas("pnP", 240)}</div><div class="box"><div class="bt">Throat sizes</div>${canvas("pnT", 240)}</div><div class="box"><div class="bt">Coordination numbers</div>${canvas("pnZ", 240)}</div></div>`;
  }
  function secGrains(g) {
    if (g.error) return `<div class="box"><div class="errs">${esc(g.error)}</div></div>`;
    let h = `<div class="rgrid">`;
    Object.entries(g.phases || {}).forEach(([li, p]) => {
      if (!p.n_particles) return;
      const hs = p.size_histogram;
      if (hs) draws.push(() => CH.hist($("#gr_" + li), { x: hs.center_um, y: hs.number, xlabel: "Equivalent diameter (µm)", ylabel: "Fraction", label: "Number-weighted", line: { y: hs.volume, label: "Volume-weighted", color: "#e36209" } }));
      const T = p.orientation_tensor, Ti = p.orientation_tensor_image;
      h += `<div class="box"><div class="bt">${esc(p.name)} <span class="hint">${esc(p.shape)}</span></div>
        ${kv([["Particles in the RVE", String(p.n_particles)], ["Number density", withUnit(p.number_density_per_mm3, "1/mm³")], ["D10 / D50 / D90 (number)", `${fmt(p.eq_diameter_um.number.d10)} / ${fmt(p.eq_diameter_um.number.d50)} / ${fmt(p.eq_diameter_um.number.d90)} µm`],
          ["D50 (volume)", withUnit(p.eq_diameter_um.volume.d50, "µm")], ["Sphericity (Wadell) / aspect ratio", `${fmt(p.sphericity)} / ${fmt(p.aspect_ratio)}`], p.hermans ? ["Hermans order parameter x / y / z", `${fmt(p.hermans.x)} / ${fmt(p.hermans.y)} / ${fmt(p.hermans.z)}`] : null,
          p.nearest_neighbour_um ? ["Nearest-neighbour distance (CV)", `${fmt(p.nearest_neighbour_um.mean)} µm (${fmt(p.nearest_neighbour_um.cv)})`] : null, p.surface_gap_um ? ["Mean / minimum surface gap", `${fmt(p.surface_gap_um.mean)} / ${fmt(p.surface_gap_um.min)} µm`] : null,
          ["Clark–Evans index (1 = random, > 1 regular)", fmt(p.clark_evans_index)], ["Clusters / particles per cluster", `${p.voxel.clusters.n_clusters} / ${fmt(p.voxel.particles_per_cluster)}`], ["Percolating x / y / z", ["x", "y", "z"].map((a) => (p.voxel.percolates[a] ? "●" : "○")).join(" ")]])}
        ${T ? `<div class="bt" style="margin-top:6px">Orientation tensor ⟨u u⟩ <span class="hint">particle list</span></div>${tbl(["", "x", "y", "z"], ["x", "y", "z"].map((a, i) => [a, ...T[i].map(fmt)]))}` : ""}
        ${Ti ? `<div class="bt" style="margin-top:6px">Orientation tensor <span class="hint">PuMA structure tensor of the voxel image</span></div>${tbl(["", "x", "y", "z"], ["x", "y", "z"].map((a, i) => [a, ...Ti[i].map(fmt)]))}` : ""}
        ${hs ? canvas("gr_" + li, 220) : ""}</div>`;
    });
    const ml = g.matrix_ligament;
    if (ml) {
      draws.push(() => CH.hist($("#grML"), { x: ml.diameter_um, y: ml.volume_fraction, xlabel: "Matrix ligament thickness (µm)", ylabel: "Volume fraction", color: "#8c959f", color2: "#e1e4e8", label: "Distribution", line: { y: ml.cdf, label: "Cumulative", color: "#0b62c4" } }));
      h += `<div class="box"><div class="bt">Matrix between particles<div class="grow"></div>${viewBtn("grain_matrix_lt", "Ligament field")}</div>${kv([["D10 / D50 / D90", `${fmt(ml.d10_um)} / ${fmt(ml.d50_um)} / ${fmt(ml.d90_um)} µm`], ["Mean", withUnit(ml.mean_um, "µm")]])}${canvas("grML", 220)}
        <p class="hint">Local thickness of the matrix: the width of the matrix films and ligaments that separate the particles.</p></div>`;
    }
    return h + "</div>";
  }
  function secInput(r) {
    const pl = r.plan;
    const rz = r.realisations.map((z) => [String(z.seed), esc(z.route || ""), `${fmt(z.seconds)} s`, z.contacts_mode === "separate" ? `${z.contacts_removed} trimmed` : `${(z.touching_pairs || 0).toLocaleString()} pairs · ${(z.contact_faces || 0).toLocaleString()} faces`, r.spec.labels.map((l, i) => `${esc(l.name)} ${fmt(100 * z.vf_labels[i])} %`).join(" · ")]);
    const phases = r.spec.phases.map((p) => [esc(p.name), SHAPES[p.shape].label, `${fmt(p.size_um.d)} µm`, `${fmt(100 * p.vf)} %`, p.orientation.mode, p.overlap ? "overlapping" : "separated", p.shell ? fmt(p.shell.thickness_um) + " µm" : "—", p.r_int ? fmt(p.r_int) : "—"]);
    return `<div class="rgrid"><div class="box"><div class="bt">Input</div><p>Matrix: ${esc(r.spec.matrix.name)}</p>${tbl(["Phase", "Shape", "Size", "vol %", "Orientation", "Placement", "Shell", "R_int"], phases)}</div>
      <div class="box"><div class="bt">RVE plan</div><p>Grid ${gridTxt(pl)}${pl.film ? ` (film ${fmt(pl.T_um)} µm thick)` : ""} · voxel ${fmt(pl.h_um)} µm (recommended ${fmt(pl.recommended.h_um)}) · edge ${fmt(pl.L_um)} µm (recommended ${fmt(pl.recommended.L_um)}, ${esc(pl.recommended.reason)})</p>
        ${pl.notes.map((n) => `<div class="note warn">${esc(n)}</div>`).join("")}<div class="tablewrap"><table class="grid">${checksRows(pl.checks)}</table></div></div>
      <div class="box" style="grid-column:1/-1"><div class="bt">Generated structures</div>${tbl(["Seed", "Placement method", "Time", "Touching particles", "Voxel volume fractions"], rz)}
        <p class="hint">Software: ${Object.entries(r.versions || {}).map(([k, v]) => `${k} ${v || "—"}`).join(" · ")}</p></div></div>`;
  }

  /* ================================================================ DOE */
  /* ================================================================ calibration */
  // A calibration finds unknown material values - an interfacial resistance,
  // the conductivity of a filler grade - from measured effective properties.
  // The case open in the editor is the base; each measurement is that case with
  // a few entries changed (content, particle size) and the value measured on it.
  const CAL_PROPS = { k: ["Thermal conductivity", "W/m·K"], sigma: ["Electrical conductivity", "S/m"], eps_r: ["Relative permittivity", "-"], E: ["Young's modulus", "GPa"], alpha: ["CTE", "ppm/K"],
    mu_r: ["Relative viscosity (compound / resin)", "-"], mu: ["Viscosity of the compound", "Pa·s"] };
  const calVisc = (p) => p === "mu_r" || p === "mu";
  let calDraws = [];
  function CAL() {
    if (!state.cal) {
      state.cal = { name: "", measurements: [], unknowns: [], options: { model_unc: 2, refine_rounds: 3 }, preview: null, theory: null, result: null, resultJob: null, busy: false };
      try {
        const s = JSON.parse(localStorage.getItem("mpsim.cal") || "null");
        if (s) ["name", "measurements", "unknowns", "options"].forEach((k) => { if (s[k] !== undefined) state.cal[k] = s[k]; });
      } catch (e) { }
    }
    return state.cal;
  }
  function calSave() {
    const c = CAL();
    try { localStorage.setItem("mpsim.cal", JSON.stringify({ name: c.name, measurements: c.measurements, unknowns: c.unknowns, options: c.options })); } catch (e) { }
  }
  function calCandidates() {
    const f = state.form, out = [];
    const r3 = (v) => +(+v).toPrecision(3);
    const add = (path, label, unit, cur, lo, hi, scale) => {
      cur = Number(cur);
      if (scale === "log" && !(lo > 0)) { lo = 1e-3; hi = 1e3; }
      out.push({ path, label, unit, current: isFinite(cur) ? cur : null, lo, hi, scale });
    };
    const mp = (f.matrix || {}).props || {}, mn = (f.matrix || {}).name || "Matrix";
    add("matrix.props.k", `${mn}: thermal conductivity`, "W/m·K", mp.k, r3(mp.k * 0.5), r3(mp.k * 2), "log");
    add("matrix.props.E", `${mn}: Young's modulus`, "GPa", mp.E, r3(mp.E * 0.3), r3(mp.E * 2), "log");
    add("matrix.props.alpha", `${mn}: CTE`, "ppm/K", mp.alpha, r3(mp.alpha * 0.5), r3(mp.alpha * 1.5), "lin");
    add("matrix.props.eps_r", `${mn}: permittivity`, "-", mp.eps_r, r3(mp.eps_r * 0.7), r3(mp.eps_r * 1.5), "lin");
    (f.phases || []).forEach((ph, i) => {
      const n = ph.name || `Phase ${i + 1}`, p = ph.props || {};
      add(`phases.${i}.r_int`, `${n}: interfacial resistance R_int`, "m²K/W", ph.r_int || 0, 1e-9, 1e-5, "log");
      add(`phases.${i}.r_contact`, `${n}: contact resistance R_c`, "m²K/W", ph.r_contact || 0, 1e-9, 1e-4, "log");
      add(`phases.${i}.props.k`, `${n}: thermal conductivity`, "W/m·K", p.k, r3(p.k * 0.1), r3(p.k * 2), "log");
      add(`phases.${i}.props.E`, `${n}: Young's modulus`, "GPa", p.E, r3(p.E * 0.1), r3(p.E * 1.5), "log");
      add(`phases.${i}.props.alpha`, `${n}: CTE`, "ppm/K", p.alpha, r3(Math.max(p.alpha * 0.3, 0.1)), r3(p.alpha * 2 + 0.5), "lin");
      add(`phases.${i}.props.eps_r`, `${n}: permittivity`, "-", p.eps_r, r3(p.eps_r * 0.5), r3(p.eps_r * 2), "lin");
      add(`phases.${i}.props.sigma`, `${n}: electrical conductivity`, "S/m", p.sigma, r3(Math.max(p.sigma * 1e-3, 1e-12)), r3(Math.max(p.sigma * 10, 1e-6)), "log");
    });
    const ph = f.phases || [];
    for (let i = 0; i < ph.length; i++) for (let j = i + 1; j < ph.length; j++)
      add(`contact_rc.${i}-${j}`, `${ph[i].name || "Phase " + (i + 1)} – ${ph[j].name || "Phase " + (j + 1)}: contact resistance`, "m²K/W", (f.contact_rc || {})[`${i}-${j}`] || 0, 1e-9, 1e-4, "log");
    // the particle surface of the particle dynamics, from a measured viscosity
    if (ph.some((p) => p.shape === "sphere")) {
      const dm = ((f.options || {}).viscosity || {}).dem || {}, P = "options.viscosity.dem.";
      add(P + "bound_nm", "Particle dynamics: bound resin layer", "nm", dm.bound_nm ?? 0, 0, 20, "lin");
      add(P + "adhesion_mJ_m2", "Particle dynamics: work of adhesion", "mJ/m²", dm.adhesion_mJ_m2 ?? 0, 0, 0.5, "lin");
      add(P + "mu_f", "Particle dynamics: friction coefficient", "-", dm.mu_f ?? 0.25, 0.05, 1, "lin");
      add(P + "roughness_nm", "Particle dynamics: surface roughness", "nm", dm.roughness_nm ?? 5, 1, 50, "log");
      add(P + "hamaker_J", "Particle dynamics: Hamaker constant (all fillers)", "J", dm.hamaker_J || undefined, 1e-21, 1e-19, "log");
    }
    return out;
  }
  function calTheoryFor(path) {
    const m = /^phases\.(\d+)\.r_int$/.exec(path || "");
    const t = CAL().theory;
    if (!m || !t || !t.pairs) return null;
    return t.pairs.find((p) => p.phase === +m[1]) || null;
  }
  function calAddMeasurement() {
    const c = CAL();
    c.measurements.push({ label: c.measurements.length ? `Sample ${c.measurements.length + 1}` : "As in the case", set: [], property: "k", direction: "iso", value: "", unc: 5 });
    calSave(); renderLeft();
  }
  function calAddUnknown() {
    const c = CAL(), used = new Set(c.unknowns.map((u) => u.path));
    const pref = c.measurements.some((m) => calVisc(m.property)) ? /dem\.bound_nm$/ : /r_int$/;
    const cand = calCandidates().find((x) => !used.has(x.path) && pref.test(x.path)) || calCandidates().find((x) => !used.has(x.path));
    if (!cand) { toast("Every candidate is already an unknown."); return; }
    if (c.unknowns.length >= 3) { toast("At most three unknowns: a few effective-property measurements cannot determine more. Fix the others at theory or literature values.", 7000); return; }
    c.unknowns.push(calUnknownFrom(cand));
    calSave(); renderLeft();
  }
  function calUnknownFrom(cand) {
    return { path: cand.path, label: cand.label, unit: cand.unit, lo: cand.lo, hi: cand.hi, scale: cand.scale, prior: "uniform", center: "", sigma_dec: 1 };
  }
  async function calLoadTheory() {
    try { const r = await api("POST", "/api/calibration/theory", { form: state.form }); CAL().theory = r.theory || null; } catch (e) { CAL().theory = null; }
    if (state.tab === "calibration") renderLeft();
  }
  function calJob() {
    const c = CAL();
    return {
      name: c.name || `${state.form.name} · calibration`, base: clone(state.form),
      measurements: c.measurements.map((m) => ({ label: m.label, property: m.property, direction: calVisc(m.property) ? "iso" : m.direction, value: Number(m.value), rel_unc: Number(m.unc || 5) / 100,
        set: Object.assign(Object.fromEntries((m.set || []).filter((s) => s.path && s.value !== "" && s.value !== null).map((s) => [s.path, Number(s.value)])),
          calVisc(m.property) && Number(m.gd) > 0 ? { "options.viscosity.gd_ref": Number(m.gd) } : {}) })),
      unknowns: c.unknowns.map((u) => ({ path: u.path, label: u.label, unit: u.unit, lo: Number(u.lo), hi: Number(u.hi), scale: u.scale, prior: u.prior,
        center: u.prior === "normal" && u.center !== "" ? Number(u.center) : null, sigma_dec: Number(u.sigma_dec || 1),
        theory: (calTheoryFor(u.path) || {}).kapitza || null })),
      options: { model_unc: Number(c.options.model_unc || 2) / 100, refine_rounds: Number(c.options.refine_rounds || 3) },
    };
  }
  async function calPreview() {
    const c = CAL();
    if (!c.measurements.length || !c.unknowns.length) { toast("Add at least one measurement and one unknown first."); return; }
    c.busy = true; renderCal();
    try {
      const r = await api("POST", "/api/calibration/theory", calJob());
      c.theory = r.theory || c.theory;
      c.preview = r.preview || { available: false, reason: r.preview_error || "No preview" };
    } catch (e) { c.preview = { available: false, reason: e.message }; }
    c.busy = false;
    setCenter("cal");
    if (state.tab === "calibration") renderLeft();
  }
  async function calRun() {
    const c = CAL();
    if (!c.measurements.length || !c.unknowns.length) { toast("Add at least one measurement and one unknown first."); return; }
    try {
      const r = await api("POST", "/api/calibration", calJob());
      toast("The calibration has been queued. The console follows it; the result opens here when it is done.", 6000);
      attach(r.id, false);
      pollRuns();
    } catch (err) { toast("The calibration could not be started:\n" + err.message, 9000); }
  }
  function calApply() {
    const c = CAL(), r = c.result;
    if (!r) return;
    const done = [], skipped = [];
    r.unknowns.forEach((u) => {
      const st = calStatus(u, r.identifiability);
      if (st.kind !== "ok") { skipped.push(`${u.label} (${st.tag.toLowerCase()})`); return; }
      if (u.path.startsWith("contact_rc.")) { state.form.contact_rc = state.form.contact_rc || {}; state.form.contact_rc[u.path.split(".")[1]] = u.map; }
      else setPath(state.form, u.path, u.map);
      done.push(`${u.label} = ${fmt(u.map)}`);
    });
    saveForm(); schedulePlan(0);
    toast((done.length ? "Set in the case: " + done.join(", ") : "Nothing was set.") + (skipped.length ? `\nNot set (not determined by the measurements): ${skipped.join(", ")}` : ""), 9000);
  }
  // what the data say about one unknown, in one word and one line
  function calStatus(u, ident) {
    const ins = ((ident || {}).insensitive || []).includes(u.label);
    if (u.edge === "low") return { kind: "bound", tag: "Upper bound", text: `below ${fmt(u.ci95[1])} (95 %) · most probable ${fmt(u.map)}; smaller values still fit` };
    if (u.edge === "high") return { kind: "bound", tag: "Lower bound", text: `above ${fmt(u.ci95[0])} (95 %) · most probable ${fmt(u.map)}; larger values still fit` };
    if (ins || u.edge === "both" || u.narrowing < 1.5) return { kind: "none", tag: "Not determined", text: `the measurements do not fix it; range ${fmt(u.lo)} … ${fmt(u.hi)}` };
    return { kind: "ok", tag: "Determined", text: `95 % interval ${fmt(u.ci95[0])} … ${fmt(u.ci95[1])} · narrowed ×${fmt(u.narrowing)}` };
  }
  function leftCalibration() {
    const c = CAL(), cands = calCandidates();
    const cat = (state.doeCat || []).filter((x) => !/props\.|r_int|r_contact|gd_ref/.test(x.path));
    const gdRef = (((state.form.options || {}).viscosity || {}).gd_ref) ?? 10;
    const catOpts = (sel) => `<option value="">— choose —</option>` + cat.map((x) => `<option value="${esc(x.path)}" ${x.path === sel ? "selected" : ""}>${esc(x.label)}${x.unit ? ` [${esc(x.unit)}]` : ""}</option>`).join("");
    const meas = c.measurements.map((m, i) => {
      const unit = (CAL_PROPS[m.property] || ["", ""])[1];
      const sets = (m.set || []).map((s, k) => `<div class="calset"><select data-cm="${i}" data-cs="${k}">${catOpts(s.path)}</select><input type="number" step="any" data-cm="${i}" data-csv="${k}" value="${esc(s.value ?? "")}" placeholder="value"><button class="btn sm" data-csdel="${i}:${k}" title="Remove">×</button></div>`).join("");
      return `<div class="param">
        <div class="prow1"><input data-cm="${i}" data-cf="label" value="${esc(m.label || "")}" placeholder="Sample name"><button class="btn sm danger" data-cmdel="${i}" title="Remove this measurement">${I("delete")}</button></div>
        <div class="g2 calg">
          <div><label>Property</label><select data-cm="${i}" data-cf="property">${Object.entries(CAL_PROPS).map(([k, v]) => `<option value="${k}" ${m.property === k ? "selected" : ""}>${v[0]}</option>`).join("")}</select></div>
          ${calVisc(m.property) ? `<div><label>Shear rate [1/s]</label><input type="number" step="any" min="0" data-cm="${i}" data-cf="gd" value="${esc(m.gd ?? "")}" placeholder="${esc(gdRef)} (the case)"></div>`
            : `<div><label>Direction</label><select data-cm="${i}" data-cf="direction">${[["iso", "Mean of x, y, z"], ["x", "x"], ["y", "y"], ["z", "z (through-plane)"]].map(([k, v]) => `<option value="${k}" ${m.direction === k ? "selected" : ""}>${v}</option>`).join("")}</select></div>`}
        </div><div class="g2 calg">
          <div><label>Measured [${esc(unit)}]</label><input type="number" step="any" data-cm="${i}" data-cf="value" value="${esc(m.value ?? "")}"></div>
          <div><label>Uncertainty ± %</label><input type="number" step="any" min="0.1" data-cm="${i}" data-cf="unc" value="${esc(m.unc ?? 5)}"></div>
        </div>
        ${calVisc(m.property) ? '<div class="hint">Solved by the particle dynamics (spheres): one run per parameter set, a few minutes each on a CPU. The relative viscosity needs no resin data; the compound viscosity uses the resin flow model of the case.</div>' : ""}
        <div class="hint">Differs from the case in:</div>${sets || '<div class="hint">nothing — the case as it is.</div>'}
        <button class="btn sm" data-csadd="${i}">${I("plus")}Condition</button></div>`;
    }).join("");
    const unk = c.unknowns.map((u, i) => {
      const th = calTheoryFor(u.path), kz = th && th.kapitza;
      const cur = (cands.find((x) => x.path === u.path) || {}).current;
      return `<div class="param">
        <div class="prow1"><select data-cu="${i}" data-uf="path">${cands.map((x) => `<option value="${esc(x.path)}" ${x.path === u.path ? "selected" : ""}>${esc(x.label)}</option>`).join("")}</select><button class="btn sm danger" data-cudel="${i}" title="Remove this unknown">${I("delete")}</button></div>
        <div class="g2 calg">
          <div><label>From [${esc(u.unit || "")}]</label><input type="number" step="any" data-cu="${i}" data-uf="lo" value="${u.lo}"></div>
          <div><label>To</label><input type="number" step="any" data-cu="${i}" data-uf="hi" value="${u.hi}"></div>
        </div><div class="g2 calg">
          <div><label>Scale</label><select data-cu="${i}" data-uf="scale"><option value="log" ${u.scale === "log" ? "selected" : ""}>Logarithmic</option><option value="lin" ${u.scale !== "log" ? "selected" : ""}>Linear</option></select></div>
          <div><label>Prior</label><select data-cu="${i}" data-uf="prior"><option value="uniform" ${u.prior !== "normal" ? "selected" : ""}>Uniform over the range</option><option value="normal" ${u.prior === "normal" ? "selected" : ""}>Centred on a value</option></select></div>
        </div>
        ${u.prior === "normal" ? `<div class="g2"><div><label class="f">Centre</label><input type="number" step="any" data-cu="${i}" data-uf="center" value="${esc(u.center ?? "")}"></div><div><label class="f">Spread (${u.scale === "log" ? "decades" : esc(u.unit || "units")}, 1σ)</label><input type="number" step="any" data-cu="${i}" data-uf="sigma_dec" value="${esc(u.sigma_dec ?? 1)}"></div></div>` : ""}
        <div class="hint">Value in the case: ${cur === null || cur === undefined ? "—" : fmt(cur)} ${esc(u.unit || "")}${kz && kz.dmm ? ` · theory for a perfect interface (phonon mismatch): DMM ${fmt(kz.dmm)}, AMM ${fmt(kz.amm)} m²K/W — a real interface lies at or above <button class="btn xs" data-cutheory="${i}" title="Centre the prior one decade wide on the DMM value">use as prior</button>` : ""}</div></div>`;
    }).join("");
    const runs = state.jobs.filter((j) => j.kind === "calibration").map((j) => `<div class="run ${j.id === c.resultJob ? "on" : ""}" data-open="${j.id}"><i class="dot ${j.state}"></i><div class="nm">${esc(j.name)}<small>${STATE[j.state] || j.state} · ${secs(j.elapsed_s)}</small></div><div class="acts"></div></div>`).join("") || '<div class="hint">No calibration yet.</div>';
    const theoryRows = ((c.theory || {}).pairs || []).filter((p) => p.kapitza).map((p) => `<tr><td>${esc(p.name)}</td><td class="n">${fmt(p.kapitza.dmm)}</td><td class="n">${fmt(p.kapitza.amm)}</td><td class="n">${fmt(p.kapitza_radius_um)}</td><td class="n">${fmt(p.d_crit_um)}</td></tr>`).join("");
    return `<div class="sec"><div class="sh">${I("calibrate")}Problem</div><div class="sb">
        <div class="hint">The case in the editor is the base: matrix, fillers, structure and RVE. Each measurement is a sample of it — the case itself or with its content, particle size or another entry changed — and the value measured on it. The unknowns are found so that the RVE reproduces every measurement.</div>
        <label class="f">Name</label><input data-cal="name" value="${esc(c.name || "")}" placeholder="${esc(state.form.name)} · calibration">
      </div></div>
      <div class="sec"><div class="sh">${I("table")}Measurements</div><div class="sb">${meas || '<div class="hint">No measurement yet.</div>'}
        <button class="btn sm" data-act="calAddM">${I("plus")}Add measurement</button></div></div>
      <div class="sec"><div class="sh">${I("surrogate")}Unknowns (up to 3)</div><div class="sb">${unk || '<div class="hint">No unknown yet.</div>'}
        <button class="btn sm" data-act="calAddU">${I("plus")}Add unknown</button>
        <div class="hint">Whatever theory, the data sheet or an MD/DFT result already gives is better entered in the case than calibrated: every extra unknown needs more measurements. Two unknowns from one measurement have a curve of solutions, not one — the identifiability check shows it before any solve.</div></div></div>
      ${theoryRows ? `<div class="sec"><div class="sh">${I("info")}Theory: phonon-limited interfacial resistance</div><div class="sb">
        <table class="tbl"><tr><th>Filler</th><th>DMM [m²K/W]</th><th>AMM</th><th>Kapitza radius [µm]</th><th>Critical d [µm]</th></tr>${theoryRows}</table>
        <div class="hint">Acoustic and diffuse mismatch models from the sound speeds (E, ν, ρ) and the matrix heat capacity (Swartz and Pohl 1989): the resistance of a perfect interface. Particles smaller than the critical diameter conduct worse than the matrix around them (Hasselman and Johnson 1987).</div></div></div>` : ""}
      <div class="sec"><div class="sh">${I("solver")}Settings</div><div class="sb">
        <div class="g2"><div><label class="f">RVE model uncertainty ± %</label><input type="number" step="any" min="0" data-calopt="model_unc" value="${esc(c.options.model_unc)}"></div>
        <div><label class="f">Refinement rounds</label><input type="number" min="0" max="6" data-calopt="refine_rounds" value="${esc(c.options.refine_rounds)}"></div></div>
        <div class="hint">The model uncertainty covers what one RVE realisation and its voxel size leave open (about 2 % for a converged RVE). Each refinement round solves the RVE at a few more parameter sets where the answer lies. A viscosity measurement adds the statistical error of its particle-dynamics runs.</div>
        <div class="row gap" style="margin-top:8px"><button class="btn" data-act="calPreview">${I("checks")}Check identifiability</button><button class="btn primary" data-act="calRun">${I("play")}Calibrate on the RVE</button></div>
      </div></div>
      <div class="sec"><div class="sh">${I("runs")}Calibrations</div><div class="sb">${runs}</div></div>`;
  }
  function onCalField(e) {
    const t = e.target, c = CAL();
    if (t.dataset.cal) { c[t.dataset.cal] = t.value; calSave(); return; }
    if (t.dataset.calopt) { c.options[t.dataset.calopt] = t.value; calSave(); return; }
    if (t.dataset.cm !== undefined) {
      const m = c.measurements[+t.dataset.cm];
      if (!m) return;
      if (t.dataset.cf) { m[t.dataset.cf] = t.value; if (t.dataset.cf === "property" && e.type === "change") renderLeft(); }
      else if (t.dataset.cs !== undefined) {
        const s = m.set[+t.dataset.cs];
        s.path = t.value;
        const x = (state.doeCat || []).find((q) => q.path === t.value);
        if (x && (s.value === "" || s.value === null || s.value === undefined)) s.value = x.current;
        if (e.type === "change") renderLeft();
      } else if (t.dataset.csv !== undefined) m.set[+t.dataset.csv].value = t.value;
      calSave();
      return;
    }
    if (t.dataset.cu !== undefined) {
      const u = c.unknowns[+t.dataset.cu];
      if (!u) return;
      const f = t.dataset.uf;
      if (f === "path") {
        const x = calCandidates().find((q) => q.path === t.value);
        if (x) c.unknowns[+t.dataset.cu] = calUnknownFrom(x);
        calSave(); renderLeft(); return;
      }
      u[f] = ["scale", "prior"].includes(f) ? t.value : t.value;
      calSave();
      if (["scale", "prior"].includes(f) && e.type === "change") renderLeft();
    }
  }
  function onCalClick(t) {
    const c = CAL();
    if (t.dataset.cmdel !== undefined) { c.measurements.splice(+t.dataset.cmdel, 1); calSave(); renderLeft(); return true; }
    if (t.dataset.cudel !== undefined) { c.unknowns.splice(+t.dataset.cudel, 1); calSave(); renderLeft(); return true; }
    if (t.dataset.csadd !== undefined) { c.measurements[+t.dataset.csadd].set.push({ path: "", value: "" }); calSave(); renderLeft(); return true; }
    if (t.dataset.csdel !== undefined) { const [i, k] = t.dataset.csdel.split(":").map(Number); c.measurements[i].set.splice(k, 1); calSave(); renderLeft(); return true; }
    if (t.dataset.cutheory !== undefined) {
      const u = c.unknowns[+t.dataset.cutheory], th = calTheoryFor(u.path);
      if (th && th.kapitza && th.kapitza.dmm) {
        u.prior = "normal"; u.center = +th.kapitza.dmm.toPrecision(3); u.sigma_dec = 1; u.scale = "log";
        if (!(u.lo < u.center && u.center < u.hi)) { u.lo = +(u.center / 100).toPrecision(2); u.hi = +(u.center * 1e3).toPrecision(2); }
        calSave(); renderLeft();
      }
      return true;
    }
    return false;
  }
  /* ---- the calibration page (centre) ---- */
  function calUnknownTiles(us, ident) {
    return `<div class="tiles">` + us.map((u) => {
      const st = calStatus(u, ident);
      return `<div class="tile cal-${st.kind}"><div class="l">${esc(u.label)}</div>
        <div class="v">${st.kind === "none" ? "—" : st.kind === "bound" ? (u.edge === "low" ? "≤ " + fmt(u.ci95[1]) : "≥ " + fmt(u.ci95[0])) : fmt(u.map)} <small>${esc(u.unit || "")}</small></div>
        <div class="d"><span class="calchip ${st.kind}">${st.tag}</span> ${st.text}</div></div>`;
    }).join("") + "</div>";
  }
  function calIdentHTML(ident, us) {
    if (!ident) return "";
    const lab = us.map((u) => u.label.split(": ").pop());
    const combo = (cb) => {
      const first = cb.weights.find((w) => Math.abs(w) >= 0.02) || 1;
      if (first < 0) cb = Object.assign({}, cb, { weights: cb.weights.map((w) => -w) });
      const terms = cb.weights.map((w, k) => Math.abs(w) < 0.02 ? "" : `${w < 0 ? "−" : "+"} ${Math.abs(w) === 1 ? "" : fmt(Math.abs(w)) + "·"}ln ${esc(lab[k])}`).filter(Boolean).join(" ").replace(/^\+ /, "");
      const f = cb.sigma_ln ? Math.exp(cb.sigma_ln) : null;
      return `<tr><td>${terms}</td><td>${cb.determined ? `<span class="calchip ok">to ×/÷ ${fmt(f)}</span>` : `<span class="calchip none">not determined</span>`}</td></tr>`;
    };
    const ci = ident.collinearity;
    return `<div class="box"><div class="bt">Identifiability <span class="hint">what these measurements can determine at the estimate</span></div>
      ${kv([["Collinearity index γ", ci === null || ci === undefined ? "—" : (ci > 1000 ? "> 1000" : fmt(ci))]])}
      <p class="hint">${ci === null || ci === undefined ? "" : ci > 15 ? "Not identifiable: a change of one unknown is compensated by the others." : ci > 10 ? "Borderline: the unknowns partly compensate each other." : "The unknowns act differently on the measurements: as a set they are identifiable."} Brun et al. (2001): above 10–15 a set is not identifiable.</p>
      <div class="tablewrap"><table class="grid"><tr><th>Combination of the unknowns</th><th>Fixed by the data (1σ)</th></tr>${(ident.combos || []).map(combo).join("")}</table></div>
      <div class="hint">The singular vectors of the measurement sensitivities: a combination the data fix to within a factor of a few is determined; the others are carried by the prior. Fixing one unknown by theory, or a measurement that responds to it differently (another particle size, another content), turns a combination into separate values.</div></div>`;
  }
  function calJointDraw(id, joint, us, marks) {
    if (!joint || !us || us.length < 2) return "";
    const pairs = joint.pairs || [{ a: 0, b: 1, x: joint.x, y: joint.y, p: joint.p }];
    return pairs.map((pr, k) => {
      const cid = `${id}${k}`;
      calDraws.push(() => CH.heat($("#" + cid), { x: pr.x, y: pr.y, p: pr.p, xlog: !!us[pr.a].log, ylog: !!us[pr.b].log,
        xlabel: `${us[pr.a].label} [${us[pr.a].unit || ""}]`, ylabel: `${us[pr.b].label} [${us[pr.b].unit || ""}]`,
        marks: (marks || []).map((m) => ({ x: m.v[pr.a], y: m.v[pr.b], label: m.label, color: m.color })) }));
      return `<div class="box"><div class="bt">Joint posterior <span class="hint">darker is more probable; a long valley is a trade-off the data cannot resolve</span></div>${canvas(cid, 300)}</div>`;
    }).join("");
  }
  // a table whose listed columns are numbers (right-aligned, monospace); the others are text
  function ctbl(head, rows, numCols) {
    const n = new Set(numCols);
    return `<div class="tablewrap"><table class="grid"><tr>${head.map((h, i) => `<th class="${n.has(i) ? "n" : ""}">${h}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr>${r.map((c, i) => `<td class="${n.has(i) ? "n" : ""}">${c ?? "—"}</td>`).join("")}</tr>`).join("") + "</table></div>";
  }
  function calCondText(set) {
    const lab = (path) => ((state.doeCat || []).find((x) => x.path === path) || {}).label || path;
    return Object.entries(set || {}).map(([k, v]) => `${lab(k)} ${fmt(v)}`).join(", ") || "as the case";
  }
  function calResultHTML(r) {
    const us = r.unknowns, ident = r.identifiability;
    const meas = ctbl(["Sample", "Conditions", "Property", "Measured", "Solved at the estimate", "Deviation", "Surrogate", "Effective medium"],
      r.measurements.map((m) => [esc(m.label), esc(m.conditions ? (m.conditions.map((c) => `${c.label} ${fmt(c.value)}${c.unit && c.unit !== "-" ? " " + c.unit : ""}`).join(", ") || "as the case") : calCondText(m.set)),
        `${esc((CAL_PROPS[m.property] || [m.property])[0])} ${m.direction === "iso" ? "" : "(" + m.direction + ")"}`, `${fmt(m.measured)} ± ${fmt(100 * m.rel_unc)} %`,
        `<b>${fmt(m.direct)}</b> ${esc(m.unit)}${m.model === "particle dynamics" ? ` <span class="hint">particle dynamics${m.run_unc ? `, ± ${fmt(100 * m.run_unc)} %` : ""}</span>` : ""}`, `<span class="${Math.abs(m.residual_sigma) > 2 ? "bad" : ""}">${(100 * (m.direct / m.measured - 1)).toFixed(1)} % (${m.residual_sigma.toFixed(1)}σ)</span>`,
        fmt(m.surrogate), m.theory_k ? fmt(m.theory_k) : "—"]), [3, 4, 5, 6, 7]);
    us.forEach((u, j) => {
      const th = (u.theory || {});
      const ser = [{ x: u.marginal.x, y: u.marginal.p, color: "#0b62c4", width: 2.4, label: "Posterior" }];
      if (th.dmm) ser.push({ x: [th.dmm, th.dmm], y: [0, 1], color: "#8250df", dash: [5, 4], label: "DMM (perfect interface)" });
      if (th.amm) ser.push({ x: [th.amm, th.amm], y: [0, 1], color: "#bf8700", dash: [2, 3], label: "AMM" });
      if (u.prior === "normal" && u.center) ser.push({ x: [u.center, u.center], y: [0, 1], color: "#57606a", dash: [1, 3], label: "Prior centre" });
      calDraws.push(() => CH.line($("#calMarg" + j), { series: ser, xlog: !!u.log, xlabel: `${u.label} [${u.unit || ""}]`, ylabel: "Probability (relative)", ymin: 0, ymax: 1.05 }));
    });
    const marg = us.map((u, j) => `<div class="box"><div class="bt">${esc(u.label)} <span class="hint">MAP ${fmt(u.map)} · median ${fmt(u.median)} · 95 % ${fmt(u.ci95[0])} … ${fmt(u.ci95[1])} ${esc(u.unit || "")}</span></div>${canvas("calMarg" + j, 220)}</div>`).join("");
    const sug = (r.suggestions || []).length ? `<div class="box" style="grid-column:1/-1"><div class="bt">What to measure next <span class="hint">ranked by the information it adds (effective-medium model at the estimate)</span></div>
      ${ctbl(["Measurement", "Predicted k [W/m·K]", "Information gain"].concat(us.map((u) => `${esc(u.label.split(": ").pop())}: uncertainty ÷`)),
        r.suggestions.map((s) => [esc(s.label.replace(/^sample (\d+) with /, "Sample $1 with ")), fmt(s.predicted), fmt(s.gain)].concat(s.narrowing.map((v) => v ? fmt(v) : "—"))),
        [1, 2].concat(us.map((_, j) => 3 + j)))}
      <div class="hint">A gain near zero adds nothing; a factor above 2 in a column halves that unknown's uncertainty. Sizes and contents that change how much the interface matters separate it from the bulk values.</div></div>`
      : (r.suggestions_note ? `<div class="note" style="grid-column:1/-1">${esc(r.suggestions_note)}</div>` : "");
    const marks = [{ v: us.map((u) => u.map), label: "estimate", color: "#d1242f" }];
    return `<div class="calhead"><b>${esc(r.name)}</b> <span class="hint">RVE calibration · ${r.design.points.length} parameter sets solved · ${secs(r.elapsed_s)} · ${esc(r.created || "")}</span>
        <div class="grow"></div><button class="btn sm primary" data-act="calApply">${I("save")}Apply the estimates to the case</button></div>
      ${calUnknownTiles(us, ident)}
      ${(r.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("")}
      <div class="rgrid">
        <div class="box" style="grid-column:1/-1"><div class="bt">Measurements against the RVE at the estimate</div>${meas}
          <div class="hint">"Solved at the estimate" is a direct RVE solve (a particle-dynamics run for a viscosity) with the calibrated values, not the surrogate. A deviation within about 2σ is a fit; larger means no value in the ranges explains that sample.</div></div>
        ${marg}
        ${calJointDraw("calJoint", r.joint, us, marks)}
        ${calIdentHTML(ident, us)}
        ${sug}
        <div class="box"><div class="bt">Surrogate refinement</div>${tbl(["Round", "Parameter sets", "Surrogate error at the estimate (÷ measurement σ)", "Estimate"],
          (r.history || []).map((h) => [h.round, h.points, fmt(h.gp_sd_rel), h.map.map((v) => fmt(v)).join(", ")]))}</div>
      </div>`;
  }
  function calPreviewHTML(p) {
    if (!p.available) return `<div class="note">Effective-medium preview: ${esc(p.reason || "not available")}</div>`;
    const us = p.unknowns;
    return `<div class="calhead"><b>Identifiability check</b> <span class="hint">instant, on the ${esc(p.model)}; model uncertainty ±${fmt(100 * p.model_unc)} %</span></div>
      ${calUnknownTiles(us, p.identifiability)}
      ${(p.notes || []).map((n) => `<div class="note">${esc(n)}</div>`).join("")}
      <div class="rgrid">${calIdentHTML(p.identifiability, us)}${calJointDraw("calPJoint", p.joint, us, [{ v: us.map((u) => u.map), label: "estimate", color: "#d1242f" }])}</div>
      <div class="hint">This uses a mean-field model, not the RVE: it is quick and says whether the problem is well posed. The RVE calibration then resolves the particle arrangement, contacts and the real interface area.</div>`;
  }
  function renderCal() {
    const body = $("#calBody");
    if (!body) return;
    const c = CAL();
    calDraws = [];
    let html = "";
    if (c.busy) html += `<div class="note">Checking…</div>`;
    if (c.result) html += calResultHTML(c.result);
    if (c.preview) html += (c.result ? '<hr class="sep">' : "") + calPreviewHTML(c.preview);
    if (!html) html = `<div class="empty">${I("calibrate")}<div><b>Calibration of unknown material values</b><br>
      1 · In the Calibration tab, enter the measured samples (the case, or the case with another content or particle size) and their measured values.<br>
      2 · Choose up to three unknowns — an interfacial resistance, a contact resistance, a filler's conductivity — with their ranges. Theory values are shown where theory has them.<br>
      3 · "Check identifiability" says in a second whether the measurements can determine the unknowns at all; "Calibrate on the RVE" then finds them with full RVE solves.</div></div>`;
    body.innerHTML = html;
    setTimeout(() => calDraws.forEach((f) => { try { f(); } catch (e) { console.warn(e); } }), 20);
  }

  async function loadDoeCatalogue() {
    try { state.doeCat = await api("POST", "/api/doe/parameters", state.form); } catch (e) { state.doeCat = []; }
  }
  function doeAddParam() {
    const used = new Set(state.doe.parameters.map((p) => p.path));
    const cand = state.doeCat.find((c) => !used.has(c.path) && Number(c.current) > 0) || state.doeCat.find((c) => !used.has(c.path));
    if (!cand) { toast("No parameter is available for this case."); return; }
    state.doe.parameters.push(paramFrom(cand));
    renderLeft();
  }
  function paramFrom(c) {
    const v = Number(c.current);
    const base = isFinite(v) && v !== 0 ? Math.abs(v) : 1;
    return { path: c.path, label: c.label, unit: c.unit, min: +(base * 0.5).toPrecision(3), max: +(base * 1.5).toPrecision(3), steps: 3, scale: "lin" };
  }
  function doeCaseCount() {
    const ps = state.doe.parameters.filter((p) => p.path);
    if (!ps.length) return 0;
    if (state.doe.design === "lhs") return Math.max(2, +state.doe.samples || 2);
    const lv = ps.map((p) => Math.max(1, Math.min(+p.steps || 1, 24)));
    if (state.doe.design === "oat") return 1 + lv.reduce((s, n) => s + Math.max(n - 1, 0), 0);
    return lv.reduce((s, n) => s * n, 1);
  }
  function leftDoe() {
    const d = state.doe;
    const groups = {};
    state.doeCat.forEach((c) => { (groups[c.group] = groups[c.group] || []).push(c); });
    const opts = (sel) => Object.entries(groups).map(([g, items]) => `<optgroup label="${esc(g)}">` +
      items.map((c) => `<option value="${esc(c.path)}" ${c.path === sel ? "selected" : ""}>${esc(c.label)}${c.unit ? ` [${esc(c.unit)}]` : ""}</option>`).join("") + "</optgroup>").join("");
    const params = d.parameters.map((p, i) => `<div class="param">
        <div class="prow1"><select data-pi="${i}" data-pf="path">${opts(p.path)}</select><button class="btn sm danger" data-pdel="${i}" title="Remove this parameter">${I("delete")}</button></div>
        <div class="prow2">
          <div><label>Minimum</label><input type="number" step="any" data-pi="${i}" data-pf="min" value="${p.min}"></div>
          <div><label>Maximum</label><input type="number" step="any" data-pi="${i}" data-pf="max" value="${p.max}"></div>
          <div><label>Levels</label><input type="number" min="1" max="24" data-pi="${i}" data-pf="steps" value="${p.steps}"></div>
          <div><label>Spacing</label><select data-pi="${i}" data-pf="scale"><option value="lin" ${p.scale !== "log" ? "selected" : ""}>Linear</option><option value="log" ${p.scale === "log" ? "selected" : ""}>Logarithmic</option></select></div>
        </div></div>`).join("");
    const n = doeCaseCount();
    const an = Object.keys(state.form.analyses || {}).filter((k) => state.form.analyses[k] && AN[k]);
    const studies = state.doeList.map((s) => {
      const c = s.counts || {};
      const st = c.running ? "running" : (c.done === s.n_cases ? "done" : (c.failed ? "failed" : "queued"));
      return `<div class="run ${s.id === state.doeId ? "on" : ""}" data-doeopen="${s.id}"><i class="dot ${st}"></i>
        <div class="nm">${esc(s.name)}<small>${s.n_cases} cases · ${Object.entries(c).map(([k, v]) => `${v} ${STATE[k] || k}`).join(" · ") || "queued"}</small></div><div class="acts"></div></div>`;
    }).join("") || '<div class="hint">No study yet.</div>';
    return `<div class="sec"><div class="sh">${I("preset")}New study</div><div class="sb">
        <div class="hint">The current case is the base: its material, structure, domain and the ${an.length} selected analyses (${an.map((k) => AN[k].title).join(", ") || "none"}) apply to every case.</div>
        <label class="f">Study name</label><input data-doe="name" value="${esc(d.name || (state.form.name + " · DOE"))}">
        <label class="f">Parameters</label>${params || '<div class="hint">No parameter yet. "Add parameter" takes the current value of the case as the centre of its range.</div>'}
        <div class="row gap" style="margin-top:6px"><button class="btn sm" data-act="doeAdd">${I("plus")}Add parameter</button>${d.parameters.length ? '<button class="btn sm" data-act="doeClear">Clear</button>' : ""}</div>
        <label class="f">Design</label><select data-doe="design">
          <option value="full" ${d.design === "full" ? "selected" : ""}>Full factorial (every combination)</option>
          <option value="oat" ${d.design === "oat" ? "selected" : ""}>One factor at a time</option>
          <option value="lhs" ${d.design === "lhs" ? "selected" : ""}>Latin hypercube</option></select>
        ${d.design === "lhs" ? `<div class="g2"><div><label class="f">Samples</label><input type="number" min="2" max="64" data-doe="samples" value="${d.samples}"></div>
          <div><label class="f">Random seed</label><input type="number" min="0" data-doe="seed" value="${d.seed}"></div></div>` : ""}
        <div class="note ${n > 64 ? "bad" : ""}">${n} case${n === 1 ? "" : "s"}${n > 64 ? " — at most 64 are accepted" : " will be queued"}. Each case is a complete run with its own result, figures and report.</div>
        <button class="btn primary" data-act="doeCreate" ${n < 1 || n > 64 ? "disabled" : ""}>${I("play")}Create study and queue runs</button>
      </div></div>
      <div class="sec"><div class="sh">${I("runs")}Studies</div><div class="sb">${studies}</div></div>`;
  }
  async function doeCreate() {
    const d = state.doe;
    if (!d.parameters.length) { toast("At least one parameter is required."); return; }
    const body = {
      name: d.name || (state.form.name + " · DOE"), design: d.design, samples: +d.samples || 12, seed: +d.seed || 1,
      parameters: d.parameters.map((p) => ({ path: p.path, label: p.label, unit: p.unit, min: +p.min, max: +p.max, steps: +p.steps, scale: p.scale })),
      form: clone(state.form),
    };
    try {
      const res = await api("POST", "/api/doe", body);
      toast(`The study has been created: ${res.n_cases} runs queued.`);
      await loadDoeList();
      await openDoe(res.id);
      setCenter("doe");
      pollRuns();
    } catch (err) { toast("The study could not be created:\n" + err.message, 8000); }
  }
  async function loadDoeList() {
    try { state.doeList = await api("GET", "/api/doe"); } catch (e) { return; }
    if (!state.doeId && state.doeList.length) state.doeId = state.doeList[0].id;
    if (state.tab === "doe" || state.tab === "optimization") renderLeft();
    if (state.center === "opt") renderOpt();
  }
  async function openDoe(id) {
    if (!id) return;
    try { state.doeData = await api("GET", "/api/doe/" + id); state.doeId = id; }
    catch (err) { toast("The study could not be loaded: " + err.message); return; }
    if (state.center === "doe") renderDoe();
    if (state.center === "opt") renderOpt();
    if (state.tab === "doe" || state.tab === "optimization") renderLeft();
  }
  async function pollDoe() {
    if (!state.doeId || !state.doeData) return;
    const busy = state.doeData.rows.some((r) => ["running", "queued"].includes(r.state));
    if (!busy && state.center !== "doe") return;
    if (busy) await openDoe(state.doeId);
  }
  /* Every input and every response of a study, either of them on either axis,
     with a third on the colour and a fourth on the size. A fixed set of charts
     answers the questions it was built for; this answers the ones that come up. */
  function scatterFields(d) {
    return [
      ...d.parameters.map((p) => ({ key: "p:" + p.path, label: p.label + (p.unit ? ` [${p.unit}]` : ""),
                                    get: (r) => r.values[p.path] })),
      ...d.metrics.map((m) => ({ key: "m:" + m.key, label: m.label + (m.unit && m.unit !== "-" ? ` [${m.unit}]` : ""),
                                 get: (r) => r.metrics[m.key] })),
    ];
  }
  function drawScatter(d) {
    if (!d) return;
    const fields = scatterFields(d), sc = state.scatter;
    if (!fields.length) return;
    const has = (k) => fields.some((f) => f.key === k);
    if (!has(sc.x)) sc.x = fields[0].key;
    if (!has(sc.y)) sc.y = (fields.find((f) => f.key.startsWith("m:")) || fields[fields.length - 1]).key;
    if (sc.c && !has(sc.c)) sc.c = "";
    if (sc.s && !has(sc.s)) sc.s = "";
    const fill = (id, cur, none) => {
      const el = $(id);
      if (!el) return;
      el.innerHTML = (none ? `<option value="">${none}</option>` : "")
        + fields.map((f) => `<option value="${esc(f.key)}" ${f.key === cur ? "selected" : ""}>${esc(f.label)}</option>`).join("");
      // assigned rather than added, so re-rendering cannot stack listeners
      el.onchange = () => { state.scatter[id.slice(3).toLowerCase()] = el.value; drawScatter(state.doeData); };
    };
    fill("#scX", sc.x); fill("#scY", sc.y);
    fill("#scC", sc.c, "— none —"); fill("#scS", sc.s, "— none —");
    const F = (k) => fields.find((f) => f.key === k);
    const fx = F(sc.x), fy = F(sc.y), fc = F(sc.c), fs = F(sc.s);
    const cv = $("#doeScatter");
    if (!cv || !fx || !fy) return;
    const pts = d.rows.filter((r) => r.state === "done").map((r) => ({
      x: +fx.get(r), y: +fy.get(r),
      c: fc ? +fc.get(r) : NaN, s: fs ? +fs.get(r) : NaN, label: r.label }));
    CH.bubble(cv, { points: pts, xlabel: fx.label, ylabel: fy.label, clabel: fc ? fc.label : "" });
  }

  function renderDoe() {
    const el = $("#doeBody"), sel = $("#doeSel");
    sel.innerHTML = state.doeList.map((s) => `<option value="${s.id}" ${s.id === state.doeId ? "selected" : ""}>${esc(s.name)} (${s.n_cases})</option>`).join("") || '<option value="">No study</option>';
    const d = state.doeData;
    if (!d) {
      el.innerHTML = `<div class="empty">${I("preset")}<div><b>No DOE study is selected.</b><br>The DOE tab in the ribbon creates a study from the current case.</div></div>`;
      $("#doeInfo").textContent = "";
      return;
    }
    const counts = {};
    d.rows.forEach((r) => (counts[r.state] = (counts[r.state] || 0) + 1));
    const design = { full: "full factorial", oat: "one factor at a time", lhs: "Latin hypercube" }[d.design] || d.design;
    $("#doeInfo").textContent = `${esc(d.name)} · ${d.n_cases} cases · ${design}`;
    const tile = (l, v, dd) => `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${dd || ""}</div></div>`;
    const elapsed = d.rows.reduce((s, r) => s + (r.elapsed_s || 0), 0);
    const nbad = (counts.failed || 0) + (counts.stopped || 0) + (counts.missing || 0);
    const head = ["#", "Case", "State", ...d.parameters.map((p) => `${p.label}${p.unit ? ` [${p.unit}]` : ""}`),
                  ...d.metrics.map((m) => `${m.label}${m.unit && m.unit !== "-" ? ` [${m.unit}]` : ""}`)];
    const rows = d.rows.map((r, i) => {
      const cells = [String(i + 1), esc(r.label),
        r.state === "running" ? `<div class="bar"><div style="width:${Math.round(100 * r.progress)}%"></div></div>` : (STATE[r.state] || r.state)];
      d.parameters.forEach((p) => cells.push(fmt(r.values[p.path])));
      d.metrics.forEach((m) => cells.push(fmt(r.metrics[m.key])));
      return `<tr class="doerow ${r.state}" data-doerun="${r.job_id}">${cells.map((c, k) => `<td class="${k > 1 ? "n" : ""}">${c}</td>`).join("")}</tr>`;
    }).join("");
    const scatterBox = `<div class="box" style="grid-column:1/-1"><div class="bt">Scatter<div class="grow"></div>
        <span class="hint">the selectors above the page choose what each axis, the colour and the size show</span></div>
      ${canvas("doeScatter", 360)}</div>`;
    const charts = scatterBox + d.metrics.slice(0, 6).map((m) => `<div class="box"><div class="bt">${esc(m.label)}${m.unit && m.unit !== "-" ? ` <span class="hint">${esc(m.unit)}</span>` : ""}</div>${canvas("doe_" + m.key, 250)}</div>`).join("");
    el.innerHTML = `<div class="big">${tile("Cases", d.n_cases, design)}${tile("Completed", counts.done || 0, "")}${tile("Running / queued", (counts.running || 0) + (counts.queued || 0), "")}
        ${tile("Failed", (counts.failed || 0) + (counts.stopped || 0), "")}${tile("Total solver time", secs(elapsed), "")}</div>
      <div class="box"><div class="bt">Cases<div class="grow"></div>${nbad ? `<button class="btn sm" data-doeretry title="Queue these cases again with their own values; the study then points at the new runs">${I("run")}Re-run ${nbad} failed case${nbad === 1 ? "" : "s"}</button>` : ""}<span class="hint">a row opens that run in the Result Viewer</span></div>
        <div class="tablewrap" style="max-height:52vh">${`<table class="grid"><tr>${head.map((h, i) => `<th class="${i > 1 ? "n" : ""}">${esc(h)}</th>`).join("")}</tr>${rows}</table>`}</div></div>
      ${d.metrics.length ? `<div class="rgrid" style="margin-top:12px">${charts}</div>` : ""}`;
    drawScatter(d);
    const p0 = d.parameters[0], p1 = d.parameters[1];
    if (!p0) return;
    d.metrics.slice(0, 6).forEach((m) => {
      const cv = $("#doe_" + m.key);
      if (!cv) return;
      const pts = d.rows.filter((r) => r.metrics[m.key] !== undefined && r.metrics[m.key] !== null);
      if (!pts.length) { CH.line(cv, { series: [{ x: [], y: [] }], xlabel: p0.label, ylabel: "" }); return; }
      const series = [];
      if (p1) {
        const lv = [...new Set(pts.map((r) => r.values[p1.path]))].sort((a, b) => a - b);
        lv.forEach((v, k) => {
          const sub = pts.filter((r) => r.values[p1.path] === v).sort((a, b) => a.values[p0.path] - b.values[p0.path]);
          series.push({ x: sub.map((r) => r.values[p0.path]), y: sub.map((r) => r.metrics[m.key]), color: CH.PALETTE[k % CH.PALETTE.length], marker: true, label: `${p1.label} ${fmt(v)}` });
        });
      } else {
        const sub = [...pts].sort((a, b) => a.values[p0.path] - b.values[p0.path]);
        series.push({ x: sub.map((r) => r.values[p0.path]), y: sub.map((r) => r.metrics[m.key]), color: CH.PALETTE[0], marker: true, label: m.label });
      }
      CH.line(cv, { series, xlabel: `${p0.label}${p0.unit ? ` [${p0.unit}]` : ""}`, ylabel: m.unit && m.unit !== "-" ? m.unit : "" });
    });
  }

  /* ================================================================ optimization */
  function leftOptimization() {
    const o = state.opt, d = state.doeData;
    const studies = state.doeList.map((s) => {
      const c = s.counts || {};
      const st = c.running ? "running" : (c.done === s.n_cases ? "done" : (c.failed ? "failed" : "queued"));
      return `<div class="run ${s.id === state.doeId ? "on" : ""}" data-optopen="${s.id}"><i class="dot ${st}"></i>
        <div class="nm">${esc(s.name)}<small>${s.n_cases} cases · ${c.done || 0} finished</small></div><div class="acts"></div></div>`;
    }).join("") || '<div class="hint">No study yet. A study is created on the DOE tab from the current case.</div>';
    const r = o.data;
    const ready = d ? d.rows.filter((x) => x.state === "done").length : 0;
    const summary = !r ? '<div class="hint">Nothing has been fitted yet.</div>'
      : (r.error ? `<div class="note warn">${esc(r.error)}</div>` : `
        <div class="tiles">
          <div class="tile"><div class="v">${r.optimum ? fmt(r.optimum.predicted) : "—"}</div><div class="l">Predicted optimum</div><div class="d">${r.optimum ? "± " + fmt(r.optimum.std) : ""}</div></div>
          <div class="tile"><div class="v">${r.surrogate && r.surrogate.cv ? fmt(r.surrogate.cv.r2) : "—"}</div><div class="l">Cross-validated R²</div><div class="d">${r.n_points} runs</div></div>
        </div>
        <div class="hint">${esc(r.surrogate ? r.surrogate.kind : "")}</div>`)
      + (r.error ? "" : `
        <div class="row gap" style="margin-top:8px"><button class="btn sm primary" data-optqueue="optimum" ${r.optimum ? "" : "disabled"}>${I("play")}Run the optimum</button>
          <button class="btn sm" data-optqueue="next" ${r.next_run ? "" : "disabled"}>${I("plus")}Run the suggestion</button></div>
        <div class="hint" style="margin-top:6px">The optimum is a prediction of the surrogate; queueing a real run there is how it is confirmed.</div>`);
    const oo = state.opt.options;
    const KERNEL_OPTS = [["matern52", "Matérn 5/2 — smooth, the usual choice"], ["matern32", "Matérn 3/2 — less smooth"],
      ["matern12", "Matérn 1/2 — rough responses"], ["rbf", "Squared exponential — very smooth"],
      ["quadratic", "Quadratic surface — few points, no tuning"]];
    return `<div class="sec"><div class="sh">${I("optimize")}Two steps</div><div class="sb">
        <div class="hint"><b>1 · Surrogate model.</b> A model is fitted to the finished runs and scored by cross-validation, so its accuracy is known before anything is decided from it.
          <b>2 · Optimisation.</b> That model is then searched — one response for a single optimum, or several at once for the trade-off front.</div>
        <div class="note ${ready < 3 ? "warn" : ""}">${ready} finished run${ready === 1 ? "" : "s"} available${ready < 3 ? " — at least three are needed" : ""}.</div>
      </div></div>
      <div class="sec"><div class="sh">${I("surrogate")}Surrogate tuning</div><div class="sb">
        <label class="f">Model</label>
        <select data-opt="kernel">${KERNEL_OPTS.map(([k, l]) => `<option value="${k}" ${oo.kernel === k ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
        <div class="g2" style="margin-top:6px">
          <div><label class="f">Noise <span class="u">blank = fitted</span></label><input type="number" step="any" min="0" data-opt="noise" value="${oo.noise}"></div>
          <div><label class="f">Restarts</label><input type="number" min="0" max="20" step="1" data-opt="restarts" value="${oo.restarts}"></div>
        </div>
        <div class="hint" style="margin-top:6px">Which model fits best is a property of the response, not a setting that can be decided once. Change it, refit, and compare the cross-validated R² — that number is measured on points the model never saw.</div>
        <div class="hint">A fixed noise stops the model from interpolating scatter, which matters when every case is a fresh random structure.</div>
      </div></div>
      <div class="sec"><div class="sh">${I("table")}Result</div><div class="sb">${summary}</div></div>
      <div class="sec"><div class="sh">${I("runs")}Studies</div><div class="sb">${studies}</div></div>`;
  }

  function onOptOption(e) {
    const el = e.target, k = el.dataset.opt;
    state.opt.options[k] = k === "kernel" ? el.value : (el.value === "" ? "" : Number(el.value));
    // the fit is not redone on its own: changing a setting and seeing the score
    // change should be a deliberate step, not a surprise
    renderLeft();
  }
  async function runOptimize() {
    const o = state.opt;
    if (!state.doeId) { toast("No DOE study is selected."); return; }
    if (!o.response) { toast("No response quantity is available yet; a study needs finished runs."); return; }
    o.busy = true; renderOpt();
    try {
      o.data = await api("POST", `/api/doe/${state.doeId}/optimize`,
        { response: o.response, goal: o.goal, target: o.goal === "target" ? o.target : null,
          options: o.options });
    } catch (err) {
      toast("The optimisation failed: " + err.message, 7000);
      o.data = null;
    } finally {
      o.busy = false;
      renderOpt();
      renderRibbon();
      if (state.tab === "optimization") renderLeft();
    }
  }

  async function queueOptPoint(which) {
    const r = state.opt.data;
    const src = r && (which === "next" ? r.next_run : r.optimum);
    if (!src || !src.values) { toast("No point is available yet."); return; }
    try {
      const res = await api("POST", `/api/doe/${state.doeId}/point`,
        { values: src.values, label: which === "next" ? "suggested run" : "predicted optimum" });
      toast(`Queued: ${res.name}`);
      pollRuns();
    } catch (err) { toast("The run could not be queued: " + err.message, 7000); }
  }

  function bindOptStage() {
    const seg = $("#optStage");
    if (!seg) return;
    $$("#optStage button").forEach((b) => b.classList.toggle("on", b.dataset.stage === state.opt.stage));
    seg.onclick = (e) => {
      const b = e.target.closest("button[data-stage]");
      if (!b) return;
      state.opt.stage = b.dataset.stage;
      renderOpt();
    };
  }

  /* Step 2. One response has a single best point; several have a set of points
     where nothing can be improved without giving something else up. NSGA-II
     returns that set, so the trade-off can be seen rather than guessed. */
  async function runPareto() {
    const o = state.opt, mu = o.multi;
    if (!state.doeId) { toast("No DOE study is selected."); return; }
    if (mu.responses.length < 2) { toast("Choose two or more responses for a trade-off."); return; }
    mu.busy = true; renderOpt();
    try {
      mu.data = await api("POST", `/api/doe/${state.doeId}/pareto`, {
        responses: mu.responses, goals: mu.responses.map((k) => mu.goals[k] || "max"),
        options: o.options, pop: mu.pop, generations: mu.generations, seed: 0 });
      if (mu.data && mu.data.error) toast(mu.data.error, 7000);
    } catch (err) { toast("The trade-off search failed: " + err.message, 7000); mu.data = null; }
    finally { mu.busy = false; renderOpt(); }
  }

  function renderOptStage2(body, r, metrics) {
    const o = state.opt, mu = o.multi, m = metrics.find((x) => x.key === r.response) || { label: r.response, unit: "" };
    const u = m.unit && m.unit !== "-" ? ` ${m.unit}` : "";
    const cv = r.surrogate && r.surrogate.cv;
    const tile = (l, v, d) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div><div class="d">${d || ""}</div></div>`;
    const optRows = r.optimum ? r.parameters.map((n, i) => [esc(n), `<b>${fmt(r.optimum.x[i])}</b>`,
      `${fmt(r.bounds[i][0])} … ${fmt(r.bounds[i][1])}`, r.next_run ? fmt(r.next_run.x[i]) : "—"]) : [];
    const picks = metrics.map((x) => `<label class="chk" style="display:inline-flex;margin-right:12px">
        <input type="checkbox" data-mresp="${esc(x.key)}" ${mu.responses.includes(x.key) ? "checked" : ""}> ${esc(x.label)}
        <select data-mgoal="${esc(x.key)}" style="margin-left:6px;width:auto">
          <option value="max" ${(mu.goals[x.key] || "max") === "max" ? "selected" : ""}>max</option>
          <option value="min" ${mu.goals[x.key] === "min" ? "selected" : ""}>min</option></select></label>`).join("");
    const d = mu.data;
    const front = d && !d.error && d.y ? d.y : null;
    body.innerHTML = `<div class="big">
        ${r.optimum ? tile("Predicted optimum", fmt(r.optimum.predicted) + u, `± ${fmt(r.optimum.std)} (1σ)`) : ""}
        ${cv ? tile("Model accuracy", "R² " + fmt(cv.r2), `cross-validated · RMSE ${fmt(cv.rmse)}${u}`) : ""}
        ${front ? tile("Trade-off points", String(d.n_front), `${d.generations} generations of ${d.pop}`) : ""}</div>
      <div class="rgrid">
        <div class="box"><div class="bt">Single objective <span class="hint">${esc(m.label)}</span></div>
          ${tbl(["Parameter", "Optimum", "Range searched", "Suggested next run"], optRows)}
          <div class="row gap"><button class="btn sm primary" data-optqueue="optimum" ${r.optimum ? "" : "disabled"}>${I("play")}Queue a run at the optimum</button>
            <button class="btn sm" data-optqueue="next" ${r.next_run ? "" : "disabled"}>${I("plus")}Queue the suggested run</button></div>
          <p class="hint">An optimum sitting on the edge of the range searched almost always means the real one lies outside it — widen the study and run it again.</p></div>
        <div class="box"><div class="bt">Several objectives at once <span class="hint">NSGA-II</span></div>
          <div style="margin:4px 0 8px">${picks || '<div class="hint">No finished response yet.</div>'}</div>
          <button class="btn sm primary" id="optPareto" ${mu.busy || mu.responses.length < 2 ? "disabled" : ""}>${I("optimize")}${mu.busy ? "Searching…" : "Find the trade-off front"}</button>
          ${front ? canvas("optPareto2", 300) : ""}
          ${d && d.error ? `<div class="note warn" style="margin-top:8px">${esc(d.error)}</div>` : ""}
          ${front && d.knee_y ? `<p class="hint">Each point is a design where nothing can be improved without giving something else up. The most balanced of them is at ${d.responses.map((k, i) => `${esc((metrics.find((x) => x.key === k) || {}).label || k)} ${fmt(d.knee_y[i])}`).join(", ")}.</p>` : ""}
          ${front ? `<p class="hint">Every front is only as good as the models under it: ${d.surrogates.map((s) => `${esc((metrics.find((x) => x.key === s.response) || {}).label || s.response)} R² ${fmt(s.r2)}`).join(" · ")}.</p>` : ""}
        </div>
      </div>`;
    $$("[data-mresp]").forEach((el) => { el.onchange = () => {
      const k = el.dataset.mresp;
      mu.responses = el.checked ? [...new Set([...mu.responses, k])] : mu.responses.filter((x) => x !== k);
      renderOpt();
    }; });
    $$("[data-mgoal]").forEach((el) => { el.onchange = () => { mu.goals[el.dataset.mgoal] = el.value; }; });
    const pb = $("#optPareto");
    if (pb) pb.onclick = runPareto;
    if (front) {
      const ix = 0, iy = Math.min(1, d.responses.length - 1);
      const lab = (i) => { const mm = metrics.find((x) => x.key === d.responses[i]) || {}; return (mm.label || d.responses[i]) + (mm.unit && mm.unit !== "-" ? ` [${mm.unit}]` : ""); };
      CH.bubble($("#optPareto2"), {
        points: front.map((row) => ({ x: row[ix], y: row[iy], c: NaN, s: NaN })),
        xlabel: lab(ix), ylabel: lab(iy) });
    }
  }

  function renderOpt() {
    const sel = $("#optSel"), resp = $("#optResp"), goalSel = $("#optGoal"), body = $("#optBody"), info = $("#optInfo");
    if (!sel || !body || !resp) return;
    const o = state.opt, d = state.doeData;
    sel.innerHTML = state.doeList.map((s) => `<option value="${s.id}" ${s.id === state.doeId ? "selected" : ""}>${esc(s.name)} (${s.n_cases})</option>`).join("") || '<option value="">No study</option>';
    const metrics = (d && d.metrics) || [];
    if (metrics.length && !metrics.some((m) => m.key === o.response)) o.response = metrics[0].key;
    resp.innerHTML = metrics.map((m) => `<option value="${esc(m.key)}" ${m.key === o.response ? "selected" : ""}>${esc(m.label)}${m.unit && m.unit !== "-" ? ` [${esc(m.unit)}]` : ""}</option>`).join("") || '<option value="">No finished run yet</option>';
    if (goalSel) goalSel.value = o.goal;
    const tgt = $("#optTarget");
    if (tgt) { tgt.hidden = o.goal !== "target"; if (o.target !== null && o.target !== undefined) tgt.value = o.target; }
    const btn = $("#optRun");
    if (btn) { btn.innerHTML = `${I("surrogate")}${o.busy ? "Working…" : "Run optimisation"}`; btn.disabled = !!o.busy || !metrics.length; }
    const done = d ? d.rows.filter((r) => r.state === "done").length : 0;
    if (info) info.textContent = d ? `${d.name} · ${done} of ${d.n_cases} runs finished` : "No study is selected";
    // bound before the early return below: on a tab that has not been run yet
    // the stage buttons exist but would carry no handler, so the second step
    // could not be reached at all
    bindOptStage();
    const r = o.data;
    if (!r || r.error) {
      body.innerHTML = `<div class="empty">${I("optimize")}<div><b>${r && r.error ? esc(r.error) : "No optimisation has been run yet."}</b><br>
        Choose a study and a response, then <i>Run optimisation</i>: the key design factors, a surrogate model of the study and its optimum are computed from the finished runs.</div></div>`;
      return;
    }
    bindOptStage();
    // step 1 answers "is this model any good", step 2 "what does it tell me to
    // build" - mixing them hid whether the answer rested on anything
    if (o.stage === "optimise") { renderOptStage2(body, r, metrics); return; }
    const m = metrics.find((x) => x.key === r.response) || { label: r.response, unit: "" };
    const u = m.unit && m.unit !== "-" ? ` ${m.unit}` : "";
    const cv = r.surrogate && r.surrogate.cv;
    const eff = r.effects || [];
    const hasSobol = eff.some((e) => e.sobol_total !== undefined && e.sobol_total !== null);
    const goalTxt = { max: "maximise", min: "minimise", target: `reach ${fmt(r.target)}` }[r.goal] || r.goal;
    const tile = (l, v, dd) => `<div class="tile"><div class="l">${l}</div><div class="v">${v}</div><div class="d">${dd || ""}</div></div>`;
    const ob = r.observed_best || {};
    const rows = eff.map((e) => [esc(e.name),
      hasSobol ? fmt(100 * (e.sobol_total || 0)) + " %" : "—",
      hasSobol ? fmt(100 * (e.sobol_first || 0)) + " %" : "—",
      fmt(e.src), fmt(e.pearson), `${fmt(e.sampled_min)} … ${fmt(e.sampled_max)}`]);
    const optRows = r.optimum ? r.parameters.map((n, i) => [esc(n), `<b>${fmt(r.optimum.x[i])}</b>`,
      `${fmt(r.bounds[i][0])} … ${fmt(r.bounds[i][1])}`, r.next_run ? fmt(r.next_run.x[i]) : "—"]) : [];
    body.innerHTML = `<div class="big">
        ${tile("Best run so far", fmt(ob.value) + u, esc(ob.label || ""))}
        ${r.optimum ? tile("Predicted optimum", fmt(r.optimum.predicted) + u, `± ${fmt(r.optimum.std)} (1σ of the surrogate)`) : ""}
        ${cv ? tile("Surrogate accuracy", "R² " + fmt(cv.r2), `${cv.folds === r.n_points ? "leave-one-out" : cv.folds + "-fold"} · RMSE ${fmt(cv.rmse)}${u}`) : ""}
        ${tile("Runs used", String(r.n_points), `${r.parameters.length} parameter${r.parameters.length === 1 ? "" : "s"} · ${goalTxt}`)}</div>
      <div class="rgrid">
        <div class="box" style="grid-column:1/-1"><div class="bt">Key design factors <span class="hint">${hasSobol ? "share of the response variance each input explains" : "standardised regression coefficients"}</span></div>
          ${canvas("optSens", 30 * Math.max(eff.length, 1) + 26)}
          ${tbl(["Parameter", "Sobol total", "Sobol first order", "Std. regression coeff.", "Pearson r", "Range covered"], rows)}
          <p class="hint">The total index counts everything an input takes part in, the first-order index only what it does on its own; the difference is interaction with other inputs. The standardised coefficient keeps the sign, so it also says in which direction the response moves.${r.linear_r2 === null || r.linear_r2 === undefined ? "" : ` A purely linear model explains R² = ${fmt(r.linear_r2)} of this response.`}</p></div>
        ${cv ? `<div class="box"><div class="bt">Surrogate check <span class="hint">${esc(r.surrogate.kind)}</span></div>${canvas("optParity", 280)}
          <p class="hint">Every point was predicted by a model fitted without it, so this is the error to expect from the optimum below, not the fit quality on its own training data.</p></div>` : ""}
        <div class="box"><div class="bt">Optimum and the next run</div>
          ${tbl(["Parameter", "Optimum", "Range searched", "Suggested next run"], optRows)}
          ${r.next_run ? '<p class="hint">The suggestion maximises expected improvement: a good predicted value weighed against the uncertainty of the surrogate, so it is where one more real run adds most.</p>' : ""}
          <div class="row gap"><button class="btn sm primary" data-optqueue="optimum" ${r.optimum ? "" : "disabled"}>${I("play")}Queue a run at the optimum</button>
            <button class="btn sm" data-optqueue="next" ${r.next_run ? "" : "disabled"}>${I("plus")}Queue the suggested run</button></div></div>
      </div>`;
    if (eff.length) CH.bars($("#optSens"), { labels: eff.map((e) => e.name),
      values: eff.map((e) => 100 * (hasSobol ? (e.sobol_total || 0) : Math.abs(e.src || 0))),
      unit: " %", max: 100, colors: eff.map((_, i) => PHASE_COLORS[i % PHASE_COLORS.length]) });
    if (cv) CH.scatter($("#optParity"), { points: cv.actual.map((a, i) => ({ x: a, y: cv.predicted[i] })),
      xlabel: `Computed ${m.label}${u}`, ylabel: `Predicted${u}`, unity: true, label: "Cross-validated prediction" });
  }

  /* ================================================================ viewer */
  function syncViewerControls() {
    const S = V.state;
    $("#vField").value = S.field;
    $("#vLog").checked = S.log;
    $("#vRange").value = S.range;
    $("#vFaces").checked = S.faces;
    $("#vParticles").checked = S.particles;
    $$("#vSurfStyle button").forEach((b) => b.classList.toggle("on", b.dataset.style === S.surfStyle));
    const hasVox = S.meta && (S.meta.voxel_style || S.meta.voxel_volume);
    const vsb = $("#vSurfStyle"); if (vsb) vsb.style.display = hasVox ? "" : "none";
    const vsh = $("#vSurfStyleHint"); if (vsh) vsh.style.display = hasVox ? "" : "none";
    // a large RVE whose particles are drawn in a corner region says so
    const rn = $("#vRegionNote");
    if (rn) { const t = V.regionNote ? V.regionNote() : ""; rn.textContent = t; rn.hidden = !t; }
    $("#vVolume").checked = S.volume;
    $("#vNetwork").checked = S.networkOn;
    $("#vShowX").checked = S.show.x; $("#vShowY").checked = S.show.y; $("#vShowZ").checked = S.show.z;
    $$("#vMode button").forEach((b) => b.classList.toggle("on", b.dataset.m === S.mode));
    $$("#vAxis2d button").forEach((b) => b.classList.toggle("on", b.dataset.a === S.axis2d));
    const op = $("#vOpacity");
    if (op) { op.value = S.opacity; $("#vOpacityV").textContent = S.opacity.toFixed(2); }
    // the phase list belongs to the loaded run, so the swatches are rebuilt
    // whenever the controls are synchronised rather than once at start-up
    renderColourSwatches();
    [["vAmbient", "ambient"], ["vDiffuse", "diffuse"], ["vSpecular", "specular"]].forEach(([id, k]) => {
      const el = $("#" + id), out = $("#" + id + "V");
      if (el) { el.value = S.light[k]; if (out) out.textContent = S.light[k].toFixed(2); }
    });
    const sh = $("#vShade"); if (sh) sh.checked = !!S.light.shade;
    const ed = $("#vEdges"); if (ed) ed.checked = !!S.light.edges;
    // arrows exist only for a quantity that was solved as a vector
    const vm = V.vectorFor ? V.vectorFor(S.field) : null;
    const vv = $("#vVectors");
    if (vv) { vv.checked = S.vectors && !!vm; vv.disabled = !vm; }
    const vrow = $("#vVectorsRow");
    if (vrow) vrow.title = vm ? `Arrows of ${vm.label}` : "The displayed quantity has no vector field";
    const ps = $("#vPath");
    if (ps) ps.value = S.pathKey || "";
    const an = $("#vAnimate");
    if (an) an.checked = S.pathAnim;
    const pinfo = $("#vPathInfo");
    if (pinfo) {
      const pm = V.pathMeta ? V.pathMeta() : null, sm = pm ? pm.summary || {} : {};
      pinfo.textContent = !pm ? "Field lines are written for solved vector fields; a percolation run adds the route of the largest sphere that passes."
        : pm.kind === "particle" ? `A sphere of ${fmt(pm.diameter_um)} µm follows the route of its centre; the tube is the route itself.`
        : `${pm.n_lines} field lines${sm.tortuosity ? `, mean length / straight distance ${fmt(sm.tortuosity)}` : ""}.`;
    }
    const [lo, hi] = V.currentRange();
    if (S.range !== "manual") { $("#vMin").value = S.field ? +lo.toPrecision(4) : ""; $("#vMax").value = S.field ? +hi.toPrecision(4) : ""; }
  }
  async function openViewer(prefer) {
    const id = state.resultJob;
    const info = $("#vInfo");
    if (!id || !state.result || !state.result.view) {
      V.showEmpty(`${I("cube")}<div><b>No structure is displayed.</b><br>A preset is selected on the Home tab; <i>Generate structure</i> or <i>Run analyses</i> then fills this view.</div>`);
      info.textContent = "";
      return;
    }
    try {
      const meta = await V.open(id);
      const S = V.state;
      // the lists are rebuilt when the run's view has gained fields too, not
      // only for another run: a run continued with more analyses keeps its id
      const sig = V.metaSig(meta);
      if (state.viewerJob !== id || state.viewerSig !== sig) {
        state.viewerJob = id;
        state.viewerSig = sig;
        const groups = {};
        meta.fields.forEach((f) => { (groups[f.group] = groups[f.group] || []).push(f); });
        $("#vField").innerHTML = `<option value="">Structure (phases)</option>` + Object.entries(groups).map(([g, fs]) => `<optgroup label="${esc(g)}">${fs.map((f) => `<option value="${f.key}">${esc(f.label)}</option>`).join("")}</optgroup>`).join("");
        $("#vPhaseToggles").innerHTML = meta.surfaces.map((sf) => { const l = meta.labels[sf.label]; return `<label class="phtog"><input type="checkbox" data-hide="${sf.label}" checked><i style="background:${l.color}"></i>${esc(l.name)}</label>`; }).join("") || '<div class="hint">No surfaces.</div>';
        $("#vNetworkRow").style.display = meta.network ? "" : "none";
        $("#vVolumeRow").style.display = V.can.volume() ? "" : "none";
        const paths = meta.paths || [], vecs = meta.vectors || [];
        const psel = $("#vPath"), psec = $("#vPathsSec");
        if (psel) psel.innerHTML = `<option value="">None</option>` + paths.map((p) => `<option value="${p.key}">${esc(p.label)}</option>`).join("");
        if (psec) psec.style.display = paths.length || vecs.length ? "" : "none";
        V.syncSliders();
        info.textContent = `Grid ${meta.full_shape.join("×")} → viewer ${meta.shape.join("×")} (${fmt(meta.spacing_um)} µm) · ${meta.surfaces.reduce((s, x) => s + x.n_tris, 0).toLocaleString()} surface triangles`;
        syncViewerControls();
        setTimeout(async () => { await V.refresh(true); V.camera("iso"); }, 30);
      }
      if (prefer !== undefined) {
        // "lines:<key>" and "path_…" ask for a route through the structure,
        // anything else for a scalar field
        const want = prefer.startsWith("lines:") ? prefer.slice(6) : (prefer.startsWith("path_") ? prefer : null);
        const pp = want ? (meta.paths || []).find((x) => x.key === want || x.key.startsWith(want)) : null;
        if (pp) {
          S.pathKey = pp.key;
          S.pathAnim = true;
          S.networkOn = false;
          // a translucent structure with the route inside it, the way a
          // percolation path is normally presented
          S.field = "";
          S.faces = false;
          S.particles = true;
          S.opacity = 0.22;
        } else {
          const f = prefer === "" ? null : meta.fields.find((x) => x.key.startsWith(prefer));
          S.field = f ? f.key : (prefer === "" ? "" : S.field);
          if (f) S.log = !!f.log;
          S.particles = true;
          if (!(meta.surfaces || []).length) S.faces = !!S.field;
          S.range = "auto";
        }
      }
      syncViewerControls();
      await V.refresh(true);
    } catch (err) {
      V.showEmpty(`${I("warn")}<div>The viewer could not be opened: ${esc(err.message)}</div>`);
    }
  }
  let rafPending = false;
  function sliceMoved() {
    V.syncSliders();
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(async () => { rafPending = false; await V.refresh(false); });
  }
  async function screenshot() {
    const url = await V.screenshot();
    if (!url) { toast("No 3D view is available."); return; }
    download(`${state.result ? state.result.name : "view"}_screen.png`, await (await fetch(url)).blob());
  }
  async function recordGif() {
    if (!V.hasData()) { toast("No structure is displayed."); return; }
    if (V.state.mode === "2d") { toast("The animated GIF is made of the 3D view."); return; }
    const btn = $("#vGif");
    btn.disabled = true;
    try {
      const S = V.state, what = S.pathKey || S.field || "structure";
      const name = `${(state.result ? state.result.name : "view").replace(/[^\w.-]+/g, "_")}_${String(what).replace(/[^\w.-]+/g, "_")}.gif`;
      const size = await V.recordGif({ filename: name, onProgress: (i, n) => { btn.innerHTML = `<span style="font:600 9px system-ui">${Math.round(100 * i / n)}%</span>`; } });
      if (size) toast(`Saved ${name} (${(size / 1e6).toFixed(1)} MB).`);
    } catch (e) { toast("The GIF could not be made: " + e.message, 6000); }
    finally { btn.disabled = false; btn.innerHTML = `<span style="font:600 10px system-ui;letter-spacing:.3px">GIF</span>`; }
  }
  async function hiresRender() {
    if (!V.hasData()) { toast("No structure is displayed."); return; }
    const S = V.state, req = V.renderRequest(), kind = S.mode === "2d" ? "slice" : "scene";
    const btn = $("#vRender");
    btn.disabled = true; btn.innerHTML = "…";
    try {
      const r = await fetch(`/api/jobs/${state.resultJob}/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, req }) });
      if (!r.ok) { const j = await r.json().catch(() => ({})); throw new Error(j.error || r.statusText); }
      download(`${state.result.name}_${req.field || "structure"}_${kind}.png`, await r.blob());
      toast("The PyVista figure has been saved.");
    } catch (err) { toast("Rendering failed: " + err.message.slice(0, 400), 7000); }
    finally { btn.disabled = false; btn.innerHTML = I("image"); }
  }
  /* One swatch per phase. The colour belongs to the phase, not to the view, so
     it is taken from the labels of the loaded run and applies at once to the 3D
     surfaces, the 2D sections and every legend. */
  function renderColourSwatches() {
    const box = $("#vColours");
    if (!box) return;
    const S = V.state, labels = (S.meta && S.meta.labels) || [];
    if (!labels.length) { box.innerHTML = '<div class="hint">Open a run to choose its phase colours.</div>'; return; }
    // not .slrow: that is a 46px / 1fr / 62px grid built for sliders, and it
    // crushes a phase name like "Epoxy (EMC / underfill matrix)" into its label column
    box.innerHTML = labels.map((l, i) => (l.kind === "interphase" ? ""
      : `<label class="colrow" title="${esc(l.name)}">
           <input type="color" data-phcol="${i}" value="${S.colours[i] || l.color}">
           <span>${esc(l.name)}</span></label>`)).join("");
    $$("[data-phcol]", box).forEach((el) => {
      el.oninput = () => { V.state.colours[+el.dataset.phcol] = el.value; V.refresh(true); };
    });
  }

  function bindViewer() {
    const S = V.state;
    $("#vMode").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      S.mode = b.dataset.m; syncViewerControls(); V.syncSliders();
      setTimeout(() => V.refresh(true), 20);
    });
    $("#vAxis2d").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; S.axis2d = b.dataset.a; syncViewerControls(); V.syncSliders(); V.refresh(false); });
    $("#v2dSlice").addEventListener("input", (e) => { S.slices[S.axis2d] = +e.target.value; sliceMoved(); });
    $("#vField").addEventListener("change", (e) => {
      S.field = e.target.value;
      const f = S.meta && S.meta.fields.find((x) => x.key === S.field);
      // a field reads best on the RVE faces; surfaces return for the structure
      // the field colours the surfaces; whether the RVE faces are drawn is the
      // viewer's own setting and is left as the user put it
      S.log = f ? !!f.log : false; S.range = "auto"; S.particles = true;
      if (S.field) S.networkOn = false;
      syncViewerControls(); V.refresh(true);
    });
    $("#vCmap").addEventListener("change", (e) => { S.cmap = e.target.value; V.refresh(true); });
    $("#vRange").addEventListener("change", (e) => { S.range = e.target.value; syncViewerControls(); V.refresh(true); });
    ["vMin", "vMax"].forEach((id) => $("#" + id).addEventListener("change", () => {
      S.vmin = $("#vMin").value === "" ? null : +$("#vMin").value; S.vmax = $("#vMax").value === "" ? null : +$("#vMax").value;
      S.range = "manual"; $("#vRange").value = "manual"; V.refresh(true);
    }));
    $("#vLog").addEventListener("change", (e) => { S.log = e.target.checked; syncViewerControls(); V.refresh(true); });
    ["x", "y", "z"].forEach((a) => {
      $("#vS" + a).addEventListener("input", (e) => { S.slices[a] = +e.target.value; sliceMoved(); });
      $("#vShow" + a.toUpperCase()).addEventListener("change", (e) => { S.show[a] = e.target.checked; V.refresh(true); });
    });
    // a control missing from an older cached page must not stop the rest of the UI
    const tog = (id, key) => { const el = $("#" + id); if (el) el.addEventListener("change", (e) => { S[key] = e.target.checked; V.refresh(true); }); };
    tog("vFaces", "faces"); tog("vClip", "clip"); tog("vParticles", "particles"); tog("vColorParticles", "colorParticles");
    const styleSeg = $("#vSurfStyle");
    if (styleSeg) styleSeg.addEventListener("click", (e) => {
      const b = e.target.closest("[data-style]");
      if (!b) return;
      S.surfStyle = b.dataset.style; S.particles = true;
      syncViewerControls(); V.refresh(true);
    });
    tog("vOutline", "outline"); tog("vOutlines2d", "outlines2d"); tog("vParallel", "parallel"); tog("vVolume", "volume"); tog("vNetwork", "networkOn"); tog("vAxes", "axes");
    tog("vAnimate", "pathAnim");
    // lighting is re-applied to the actors already on screen, so a slider moves
    // the structure that is being looked at rather than only the next one
    const lightSlider = (id, key) => {
      const el = $("#" + id), out = $("#" + id + "V");
      if (!el) return;
      el.addEventListener("input", () => {
        S.light[key] = +el.value;
        if (out) out.textContent = (+el.value).toFixed(2);
        V.refresh(true);
      });
    };
    lightSlider("vAmbient", "ambient"); lightSlider("vDiffuse", "diffuse"); lightSlider("vSpecular", "specular");
    const lightTog = (id, key) => { const el = $("#" + id); if (el) el.addEventListener("change", (e) => { S.light[key] = e.target.checked; V.refresh(true); }); };
    lightTog("vShade", "shade"); lightTog("vEdges", "edges");
    const creset = $("#vColourReset");
    if (creset) creset.addEventListener("click", () => { V.state.colours = {}; renderColourSwatches(); V.refresh(true); });
    // arrows and routes live inside the sample: opaque RVE faces would hide
    // every one of them, so switching either on clears the view first
    const vecBox = $("#vVectors");
    if (vecBox) vecBox.addEventListener("change", (e) => {
      S.vectors = e.target.checked;
      // the arrows stand in the matrix and inside the particles: opaque
      // particles hid most of them, so the particles turn see-through
      if (S.vectors) { S.faces = false; S.volume = false; if (S.opacity > 0.35) { S._opacityBefore = S.opacity; S.opacity = 0.3; } }
      else { if (S.field) S.faces = true; if (S._opacityBefore) { S.opacity = S._opacityBefore; S._opacityBefore = null; } }
      syncViewerControls(); V.refresh(true);
    });
    // the label follows the slider, the scene is rebuilt when it is released
    [["vVecDensity", "vecDensity"], ["vVecScale", "vecScale"]].forEach(([id, key]) => {
      const el = $("#" + id);
      if (!el) return;
      el.addEventListener("input", (e) => { S[key] = +e.target.value; $("#" + id + "V").textContent = S[key].toFixed(1); });
      el.addEventListener("change", () => V.refresh(true));
    });
    const vecUni = $("#vVecUniform");
    if (vecUni) vecUni.addEventListener("change", (e) => { S.vecUniform = e.target.checked; V.refresh(true); });
    const pathSel = $("#vPath");
    if (pathSel) pathSel.addEventListener("change", (e) => {
      S.pathKey = e.target.value;
      if (S.pathKey) {
        S.faces = false; S.networkOn = false; S.particles = true;
        S.opacity = Math.min(S.opacity, 0.25);
      } else if (S.field) {
        S.faces = true; S.particles = false; S.opacity = 1;
      }
      syncViewerControls(); V.refresh(true);
    });
    $("#vOpacity").addEventListener("input", (e) => { S.opacity = +e.target.value; $("#vOpacityV").textContent = S.opacity.toFixed(2); V.refresh(true); });
    $("#vPhaseToggles").addEventListener("change", (e) => { const l = e.target.dataset.hide; if (l !== undefined) { S.hidden[l] = !e.target.checked; V.refresh(true); } });
    $$("[data-cam]").forEach((b) => b.addEventListener("click", () => V.camera(b.dataset.cam)));
    $("#vShot").addEventListener("click", screenshot);
    $("#vGif").addEventListener("click", recordGif);
    const lbox = $("#liveBox");
    if (lbox) lbox.addEventListener("click", (e) => {
      const f = e.target.closest("[data-livepick]");
      if (f) { state.livePick = f.dataset.livepick; renderLive(); return; }
      const img = e.target.closest(".livemain img");
      if (img) window.open(img.src, "_blank");
    });
    $("#vRender").addEventListener("click", hiresRender);
    window.addEventListener("resize", () => { if (state.center === "vis") V.refresh(false); });
  }

  /* ================================================================ report and database */
  function renderReport() {
    const id = state.resultJob;
    if (!id) { $("#reportBar").innerHTML = '<span class="hint">No completed run is selected.</span>'; $("#reportFrame").removeAttribute("src"); return; }
    $("#reportBar").innerHTML = `${I("report")}<b>${esc(state.result ? state.result.name : id)}</b><div class="grow"></div>
      <a class="btn sm" href="/api/jobs/${id}/report" target="_blank">Open in a new tab</a><a class="btn sm" href="/api/jobs/${id}/download/report">${I("download")}Save HTML</a><a class="btn sm primary" href="/api/jobs/${id}/download/bundle">${I("zip")}Complete ZIP</a>`;
    const src = `/api/jobs/${id}/report`;
    if ($("#reportFrame").getAttribute("src") !== src) $("#reportFrame").setAttribute("src", src);
  }
  function openDatabase() {
    const g = Object.fromEntries(state.lib.groups.map((x) => [x.id, x.label]));
    const rows = allMaterials();
    $("#mTitle").innerHTML = `Material database <span class="hint">${rows.length} materials · ${esc(state.lib.note)}</span>`;
    $("#mBody").innerHTML = `<table class="grid" id="dbTable"><tr><th>Material</th><th>Group</th>${PROPS.map(([, l, un]) => `<th class="n">${l}<br><span class="hint">${un}</span></th>`).join("")}<th></th></tr>` +
      rows.map((r) => `<tr data-name="${esc((r.name + " " + (g[groupOf(r)] || "user")).toLowerCase())}" title="${esc(r.note || "")}"><td>${esc(r.name)}${isOverride(r.id) ? " (edited)" : ""}</td><td>${esc(g[groupOf(r)] || "User materials")}</td>` +
        PROPS.map(([k]) => `<td class="n">${fmt(r.props[k])}</td>`).join("") +
        `<td style="white-space:nowrap"><button class="btn sm" data-libmatrix="${r.id}">Use as matrix</button> <button class="btn sm" data-libadd="${r.id}">Add as phase</button></td></tr>`).join("") + "</table>";
    $("#mFilter").value = "";
    $("#modal").hidden = false;
    $("#mFilter").focus();
  }

  /* ================================================================ binding */
  function bind() {
    $("#brand").innerHTML = `${I("logo")}<div><b>MPSim</b><small>Material Property Simulation · RVE homogenisation of semiconductor materials</small></div>`;
    $("#btnRun").innerHTML = `${I("play")}Run`;
    $("#btnStop").innerHTML = `${I("stop")}Stop`;
    $("#ctabs [data-c=vis]").innerHTML = `${I("view3d")}3D / 2D View`;
    $("#ctabs [data-c=res]").innerHTML = `${I("table")}Result Viewer`;
    $("#ctabs [data-c=doe]").innerHTML = `${I("preset")}DOE`;
    const optTab = $("#ctabs [data-c=opt]");
    if (optTab) optTab.innerHTML = `${I("optimize")}Optimization`;
    $("#ctabs [data-c=rep]").innerHTML = `${I("report")}Report`;
    $("#ctabs [data-c=cal]").innerHTML = `${I("calibrate")}Calibration`;
    $("#doeCsv").innerHTML = `${I("download")}CSV`;
    $("#doeDelete").innerHTML = `${I("delete")}Delete study`;
    $("#rptabs [data-r=display]").innerHTML = `${I("display")}Display`;
    $("#rptabs [data-r=runs]").innerHTML = `${I("runs")}Runs`;
    $("#hScene").innerHTML = `${I("view3d")}Scene`;
    $("#hPhases").innerHTML = `${I("grains")}Surfaces`;
    $("#hSections").innerHTML = `${I("matrix")}Section planes`;
    // guarded like #hVectors: a page held in an old cache will not have these
    const hCol = $("#hColours");
    if (hCol) hCol.innerHTML = `${I("image")}Phase colours`;
    const hLit = $("#hLight");
    if (hLit) hLit.innerHTML = `${I("cube")}Lighting and depth`;
    const hVec = $("#hVectors");
    if (hVec) hVec.innerHTML = `${I("tortuosity")}Vectors and paths`;
    $("#camFit").innerHTML = I("fit");
    $("#vShot").innerHTML = I("camera");
    $("#vGif").innerHTML = `<span style="font:600 10px system-ui;letter-spacing:.3px">GIF</span>`;
    $("#vRender").innerHTML = I("image");
    $("#dockToggle").innerHTML = I("collapse");

    $("#rtabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setTab(b.dataset.tab); });
    $("#rbar").addEventListener("click", (e) => {
      if (e.target.matches("input[type=file]")) return;
      const b = e.target.closest("[data-rb]"); if (b) ribbonAction(b.dataset.rb, b);
    });
    $("#rbar").addEventListener("change", async (e) => {
      if (!e.target.matches("[data-rbfile]")) return;
      const f = e.target.files[0]; if (!f) return;
      try { state.form = completeForm(JSON.parse(await f.text())); state.sel = 0; renderAll(); schedulePlan(0); saveForm(); toast("The input has been opened."); }
      catch (err) { toast("The JSON file could not be read: " + err.message); }
      e.target.value = "";
    });
    $("#ctabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-c]"); if (b) setCenter(b.dataset.c); });
    $("#rptabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setRight(b.dataset.r); });
    $("#dtabs").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.id === "dockToggle") toggleDock(); else setDock(b.dataset.d, true);
    });
    $("#btnRun").addEventListener("click", () => submit(false));
    $("#btnStop").addEventListener("click", stopActive);
    $("#caseName").addEventListener("input", onField);
    $("#doeSel").addEventListener("change", (e) => openDoe(e.target.value));
    const optSel = $("#optSel");
    if (optSel) optSel.addEventListener("change", (e) => { state.opt.data = null; openDoe(e.target.value); });
    const optResp = $("#optResp");
    if (optResp) optResp.addEventListener("change", (e) => { state.opt.response = e.target.value; state.opt.data = null; renderOpt(); });
    const optGoal = $("#optGoal");
    if (optGoal) optGoal.addEventListener("change", (e) => { state.opt.goal = e.target.value; state.opt.data = null; renderOpt(); });
    const optTgt = $("#optTarget");
    if (optTgt) optTgt.addEventListener("change", (e) => { state.opt.target = e.target.value === "" ? null : +e.target.value; });
    const optRunBtn = $("#optRun");
    if (optRunBtn) optRunBtn.addEventListener("click", runOptimize);
    $("#doeCsv").addEventListener("click", () => { if (state.doeId) window.location.href = `/api/doe/${state.doeId}/csv`; });
    $("#doeDelete").addEventListener("click", async () => {
      if (!state.doeId || !confirm("Delete this study and all of its runs?")) return;
      try {
        await api("DELETE", `/api/doe/${state.doeId}?runs=1`);
        state.doeId = null; state.doeData = null;
        await loadDoeList(); renderDoe(); pollRuns();
        toast("The study has been deleted.");
      } catch (err) { toast(err.message); }
    });

    const lb = $("#lbody");
    lb.addEventListener("input", (e) => {
      const t = e.target;
      if (!t.dataset) return;
      if (t.dataset.lk) onLibField(e);
      else if (t.dataset.libfilter !== undefined) {
        state.libFilter = t.value; const pos = t.selectionStart; renderLeft();
        const el = $("#lbody [data-libfilter]"); if (el) { el.focus(); el.setSelectionRange(pos, pos); }
      }
      else if (t.dataset.k) onField(e);
      else if (t.dataset.pi !== undefined || t.dataset.doe) onDoeField(e);
      else if (t.dataset.cm !== undefined || t.dataset.cu !== undefined || t.dataset.cal || t.dataset.calopt) onCalField(e);
    });
    lb.addEventListener("change", (e) => {
      const t = e.target;
      if (t.dataset.lk) onLibField(e);
      else if (t.dataset.libimport !== undefined) importLibrary(t);
      else if (t.dataset.k) onField(e);
      else if (t.dataset.mat) onMaterial(e);
      else if (t.dataset.an) { state.form.analyses[t.dataset.an] = t.checked; saveForm(); schedulePlan(); renderLeft(); renderRibbon(); }
      else if (t.dataset.dir !== undefined) { state.form.options.directions = $$("[data-dir]", lb).filter((c) => c.checked).map((c) => c.dataset.dir).join("") || "z"; saveForm(); schedulePlan(); }
      else if (t.dataset.wfpick !== undefined) useWorkFolder(t.value);
      else if (t.dataset.wspick !== undefined) useWorkspace(t.value);
      else if (t.dataset.casepick !== undefined) useCase(t.value);
      // on change rather than on input: a half-typed core count must not be sent
      else if (t.dataset.set) onSetting(e);
      else if (t.dataset.opt) onOptOption(e);
      else if (t.dataset.pi !== undefined || t.dataset.doe) onDoeField(e);
      else if (t.dataset.cm !== undefined || t.dataset.cu !== undefined || t.dataset.cal || t.dataset.calopt) onCalField(e);
    });
    lb.addEventListener("click", (e) => {
      const t = e.target.closest("[data-libsel],[data-editmat],[data-uselib],[data-preset],[data-go],[data-sel],[data-shape],[data-add],[data-act],[data-q],[data-film],[data-anall],[data-annone],[data-pdel],[data-doeopen],[data-wfmake],[data-wsmake],[data-casemake],[data-cmdel],[data-cudel],[data-csadd],[data-csdel],[data-cutheory]");
      if (!t) return;
      if (onCalClick(t)) return;
      if (t.dataset.libsel) { state.libSel = t.dataset.libsel; renderLeft(); renderRibbon(); return; }
      // an explicit button, not a click on the matrix row, opens the library
      if (t.dataset.editmat) { state.libSel = t.dataset.editmat; setTab("material"); return; }
      if (t.dataset.uselib) {
        const en = getPath(state.form, t.dataset.uselib), m = mat(en.material_id);
        if (m) { en.props = clone(m.props); saveForm(); schedulePlan(0); renderLeft(); toast("The library values are used."); }
        return;
      }
      if (t.dataset.film !== undefined) {
        const on = t.dataset.film === "1";
        state.form.rve.film = on;
        if (on && !state.form.rve.T_um) {
          // a first guess the user overwrites: three of the largest particles
          const pl = state.plan && state.plan.plan;
          const dmax = pl ? Math.max(0, ...pl.features.filter((x) => !x.network).map((x) => x.eqd_max || 0)) : 0;
          state.form.rve.T_um = dmax ? +(3 * dmax).toPrecision(3) : pl ? +(pl.L_um / 4).toPrecision(3) : 10;
        }
        saveForm(); renderLeft(); renderRibbon(); schedulePlan(0);
        return;
      }
      if (t.dataset.wsmake !== undefined) {
        const box = $("[data-wsnew]", lb);
        if (box && box.value.trim()) useWorkspace(box.value.trim());
        else toast("Type the full path of the workspace first.", 5000);
        return;
      }
      if (t.dataset.wfmake !== undefined) {
        const box = $("[data-wfnew]", lb);
        useWorkFolder((box && box.value.trim()) || todayName());
        return;
      }
      if (t.dataset.casemake !== undefined) {
        const box = $("[data-casenew]", lb);
        const n = (state.case && state.case.cases ? state.case.cases.length : 0) + 1;
        useCase((box && box.value.trim()) || ("case-" + String(n).padStart(2, "0")));
        return;
      }
      if (t.dataset.anall || t.dataset.annone) {
        const keys = (t.dataset.anall || t.dataset.annone).split(",");
        const on = !!t.dataset.anall;
        keys.forEach((k) => { if (!on || !disabledReason(k)) state.form.analyses[k] = on; });
        saveForm(); schedulePlan(); renderLeft(); renderRibbon();
        return;
      }
      if (t.dataset.pdel !== undefined) { state.doe.parameters.splice(+t.dataset.pdel, 1); renderLeft(); return; }
      if (t.dataset.doeopen) { openDoe(t.dataset.doeopen); setCenter("doe"); return; }
      if (t.dataset.preset) {
        const p = state.presets.find((x) => x.id === t.dataset.preset);
        state.form = completeForm(p.form); state.sel = 0; renderAll(); schedulePlan(0); saveForm();
        toast(`Preset loaded: ${p.name}`);
      } else if (t.dataset.go) { if (t.dataset.go === "run") submit(false); else setTab(t.dataset.go); }
      else if (t.dataset.sel !== undefined) { state.sel = +t.dataset.sel; renderLeft(); }
      else if (t.dataset.shape) {
        const ph = state.form.phases[state.sel];
        ph.shape = t.dataset.shape;
        if (ph.shape === "network") ph.overlap = false;
        if (ph.shape === "spherocylinder" && +ph.size.length < +ph.size.d) ph.size.length = 6 * ph.size.d;
        renderLeft(); schedulePlan(0); saveForm();
      } else if (t.dataset.add) ribbonAction("add:" + t.dataset.add);
      else if (t.dataset.q) ribbonAction("q:" + t.dataset.q);
      else if (t.dataset.act) ribbonAction(t.dataset.act);
    });

    document.body.addEventListener("click", async (e) => {
      const b = e.target.closest("[data-open],[data-stop],[data-del],[data-goviewer],[data-editspec],[data-center],[data-sec],[data-network],[data-libmatrix],[data-libadd],[data-doerun],[data-optopen],[data-optqueue],[data-doetoggle],[data-doeopen2],[data-doegroup],[data-doeretry]");
      if (!b) return;
      // the cases of a study are shown only when asked for; the study row itself
      // opens the study rather than any one of its runs
      if (b.dataset.doetoggle !== undefined && b.dataset.doetoggle) {
        e.stopPropagation();
        state.openDoe[b.dataset.doetoggle] = !state.openDoe[b.dataset.doetoggle];
        $("#runsList").innerHTML = runsHTML();
        if (state.tab === "results") renderLeft();
        return;
      }
      if (b.dataset.doeopen2) { e.stopPropagation(); await openDoe(b.dataset.doeopen2); setCenter("doe"); return; }
      if (b.dataset.doegroup) {
        state.openDoe[b.dataset.doegroup] = !state.openDoe[b.dataset.doegroup];
        $("#runsList").innerHTML = runsHTML();
        if (state.tab === "results") renderLeft();
        return;
      }
      if (b.dataset.doerun) { await loadResult(b.dataset.doerun); setCenter("res"); return; }
      if (b.dataset.doeretry !== undefined && state.doeId) {
        try {
          const r = await api("POST", `/api/doe/${state.doeId}/retry`, {});
          toast(r.requeued ? `${r.requeued} case(s) queued again` : "No case needed re-running");
        } catch (err) { toast("The cases could not be queued: " + err.message, 8000); }
        await openDoe(state.doeId); pollRuns();
        return;
      }
      if (b.dataset.optopen) { state.opt.data = null; await openDoe(b.dataset.optopen); renderLeft(); return; }
      if (b.dataset.optqueue) { await queueOptPoint(b.dataset.optqueue); return; }
      if (b.dataset.stop) { e.stopPropagation(); await api("POST", `/api/jobs/${b.dataset.stop}/stop`); pollRuns(); }
      else if (b.dataset.del) {
        e.stopPropagation();
        if (confirm("Delete this result permanently?")) {
          try { await api("DELETE", `/api/jobs/${b.dataset.del}`); if (state.resultJob === b.dataset.del) { state.result = null; state.resultJob = null; state.viewerJob = null; } pollRuns(); if (state.center === "res") renderResults(); }
          catch (err) { toast(err.message); }
        }
      }
      else if (b.dataset.open) openJob(b.dataset.open);
      else if (b.dataset.goviewer !== undefined) { setCenter("vis"); openViewer(b.dataset.goviewer); }
      else if (b.dataset.network !== undefined) { setCenter("vis"); await openViewer(""); V.state.networkOn = true; V.state.field = ""; V.state.faces = false; syncViewerControls(); V.refresh(true); }
      else if (b.dataset.center) setCenter(b.dataset.center);
      else if (b.dataset.sec) {
        $$("#rvNav button").forEach((x) => x.classList.toggle("on", x === b));
        const s = $("#sec-" + b.dataset.sec); if (s) s.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      else if (b.dataset.editspec !== undefined) {
        try { const f = await api("GET", `/api/jobs/${state.resultJob}/download/spec`); delete f._limits; delete f.preview; state.form = completeForm(f); state.sel = 0; renderAll(); schedulePlan(0); saveForm(); setTab("structure"); toast("The input of this run has been loaded."); }
        catch (err) { toast(err.message); }
      }
      else if (b.dataset.libmatrix) { state.form.matrix = matEntry(b.dataset.libmatrix); state.sel = -1; $("#modal").hidden = true; setTab("structure"); schedulePlan(0); saveForm(); toast("The matrix material has been changed."); }
      else if (b.dataset.libadd) { const m = mat(b.dataset.libadd); state.form.phases.push(phaseEntry(m.id, "sphere")); state.sel = state.form.phases.length - 1; $("#modal").hidden = true; setTab("structure"); schedulePlan(0); saveForm(); toast(`Phase added: ${m.name}`); }
    });
    $("#rvBody").addEventListener("scroll", () => {
      const top = $("#rvBody").getBoundingClientRect().top;
      let cur = null;
      $$(".rsec").forEach((s) => { if (s.getBoundingClientRect().top - top < 80) cur = s.id.slice(4); });
      if (cur) $$("#rvNav button").forEach((x) => x.classList.toggle("on", x.dataset.sec === cur));
    });
    $("#mClose").addEventListener("click", () => ($("#modal").hidden = true));
    $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").hidden = true; });
    $("#mFilter").addEventListener("input", (e) => { const q = e.target.value.toLowerCase(); $$("#dbTable tr[data-name]").forEach((tr) => (tr.style.display = tr.dataset.name.includes(q) ? "" : "none")); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#modal").hidden = true; });
    bindViewer();
  }
  function renderAll() { $("#caseName").value = state.form.name || ""; renderRibbon(); renderLeft(); }

  async function init() {
    bind();
    try {
      const [lib, presets, health, settings] = await Promise.all([
        api("GET", "/api/materials"), api("GET", "/api/presets"),
        api("GET", "/api/health").catch(() => null),
        api("GET", "/api/settings").catch(() => null)]);
      state.lib = lib; lib.materials.forEach((m) => (state.byId[m.id] = m));
      state.presets = presets.presets; state.health = health; state.settings = settings;
      try { state.wf = await api("GET", "/api/workfolder"); } catch (e) { state.wf = null; }
      try { state.ws = await api("GET", "/api/workspace"); } catch (e) { state.ws = null; }
      try { state.case = await api("GET", "/api/case"); } catch (e) { state.case = null; }
    } catch (err) { toast("The server could not be reached: " + err.message, 10000); return; }
    loadCustom();
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem("mpsim.form") || "null"); } catch (e) { }
    // the openEMS full-wave run became the default; a case kept from before
    // carries the old default (off) and is switched on once
    try {
      if (localStorage.getItem("mpsim.fullwave_on") !== "1") {
        if (saved && saved.options && saved.options.emi) saved.options.emi.fullwave = true;
        localStorage.setItem("mpsim.fullwave_on", "1");
      }
    } catch (e) { }
    try { state.form = saved ? completeForm(saved) :completeForm(state.presets.find((p) => p.id === "porous_filter").form); }
    catch (e) { state.form = completeForm(state.presets[0].form); }
    // the dock starts collapsed so the viewport gets the height; a run opens it
    try { toggleDock(localStorage.getItem("mpsim.dock") !== "0"); } catch (e) { toggleDock(true); }
    renderAll();
    renderPartial();
    schedulePlan(0);
    if (state.health) {
      const v = state.health.versions;
      // same source and same wording as the poll below, so the first paint does
      // not disagree with the Home tab for the first four seconds
      $("#sbServer").textContent = state.settings
        ? `${state.settings.concurrency} at a time × ${state.settings.threads} of ${state.settings.cores} cores`
        : `${state.health.cores} cores · ${state.health.concurrency} concurrent run(s)`;
    }
    await pollRuns();
    await loadDoeCatalogue();
    await loadDoeList();
    // the run list was drawn before the studies were known, so a study row
    // showed a generic name until the next poll four seconds later
    $("#runsList").innerHTML = runsHTML();
    setInterval(pollRuns, 4000);
    setInterval(pollActive, 1000);
    setInterval(pollDoe, 5000);
    const last = localStorage.getItem("mpsim.active");
    const lj = last && state.jobs.find((j) => j.id === last);
    if (lj && ["running", "queued"].includes(lj.state)) attach(last, !(lj.analyses || []).length);
    const done = state.jobs.find((j) => j.state === "done");
    if (done) await loadResult(done.id);
    setCenter("vis");
  }
  init();
})();
