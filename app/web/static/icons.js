/* Line icons for the ribbon, panels and result sections (24 × 24, two-tone:
   stroke uses currentColor, fills use --ic so each command group has its own
   colour). No icon font or CDN: the page must work on an isolated network. */
(function () {
  "use strict";
  const S = 'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"';
  const F = 'fill="var(--ic, #cfe2ff)" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"';
  const P = {
    logo: `<path d="M12 2.5 20.5 7.3v9.4L12 21.5 3.5 16.7V7.3z" fill="#0b62c4"/><path d="M12 12 20.5 7.3M12 12v9.5M12 12 3.5 7.3" stroke="#fff" stroke-width="1.4"/><circle cx="12" cy="7.2" r="1.6" fill="#fff"/><circle cx="7.6" cy="13.6" r="1.3" fill="#9cc3f0"/><circle cx="16.3" cy="14.4" r="1.9" fill="#9cc3f0"/>`,
    play: `<path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/>`,
    calibrate: `<circle ${F} cx="12" cy="12" r="7.5"/><circle ${S} cx="12" cy="12" r="3.2"/><path ${S} d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/><circle cx="12" cy="12" r="1.1" fill="currentColor"/>`,
    stop: `<rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor"/>`,
    cube: `<path ${F} d="M12 3 20 7.5v9L12 21l-8-4.5v-9z"/><path ${S} d="M4 7.5 12 12l8-4.5M12 12v9"/><circle cx="9" cy="14.5" r="1.4" fill="currentColor"/><circle cx="15.5" cy="13.5" r="1.1" fill="currentColor"/><circle cx="12" cy="7.8" r="1.2" fill="currentColor"/>`,
    new: `<path ${F} d="M6 3h8l4 4v14H6z"/><path ${S} d="M14 3v4h4M12 11v6M9 14h6"/>`,
    open: `<path ${F} d="M3 7h6l2 2h10v10H3z"/><path ${S} d="M3 11h18"/>`,
    save: `<path ${F} d="M5 4h11l3 3v13H5z"/><path ${S} d="M8 4v5h7V4M8 20v-6h8v6"/>`,
    preset: `<rect ${F} x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect ${S} x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect ${S} x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect ${F} x="13.5" y="13.5" width="7" height="7" rx="1.5"/>`,
    database: `<ellipse ${F} cx="12" cy="6" rx="7" ry="2.6"/><path ${S} d="M5 6v12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6M5 12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6"/>`,
    matrix: `<rect ${F} x="4" y="4" width="16" height="16" rx="2"/><path ${S} d="M4 9.5h16M4 14.5h16M9.5 4v16M14.5 4v16" opacity=".55"/>`,
    custom: `<path ${F} d="M12 3.5l2.6 5.3 5.8.8-4.2 4.1 1 5.8L12 16.8l-5.2 2.7 1-5.8L3.6 9.6l5.8-.8z"/>`,
    reset: `<path ${S} d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path ${S} d="M4.5 4v4.5H9"/>`,
    sphere: `<circle ${F} cx="12" cy="12" r="8"/><ellipse ${S} cx="12" cy="12" rx="8" ry="2.8" opacity=".6"/><circle cx="9.3" cy="9" r="1.6" fill="#fff" opacity=".8"/>`,
    spheroid: `<ellipse ${F} cx="12" cy="12" rx="9.5" ry="5"/><ellipse ${S} cx="12" cy="12" rx="2.4" ry="5" opacity=".6"/>`,
    cylinder: `<path ${F} d="M4 8.5h16v7H4z"/><ellipse ${F} cx="20" cy="12" rx="1.8" ry="3.5"/><ellipse ${S} cx="4" cy="12" rx="1.8" ry="3.5"/>`,
    spherocylinder: `<rect ${F} x="2.5" y="8" width="19" height="8" rx="4"/><path ${S} d="M6.5 8v8M17.5 8v8" opacity=".5"/>`,
    shape_cube: `<path ${F} d="M5 8.5 12 5l7 3.5v8L12 20l-7-3.5z"/><path ${S} d="M5 8.5 12 12l7-3.5M12 12v8"/>`,
    cuboid: `<path ${F} d="M3 10 9 6.5h12v7.5L15 17.5H3z"/><path ${S} d="M3 10h12l6-3.5M15 10v7.5"/>`,
    superellipsoid: `<rect ${F} x="4" y="6" width="16" height="12" rx="5"/><path ${S} d="M4 12h16" opacity=".5"/>`,
    polyhedron: `<path ${F} d="M12 3 20 8.5 17.5 18h-11L4 8.5z"/><path ${S} d="M12 3l-2.5 7h5zM4 8.5l5.5 1.5M20 8.5l-5.5 1.5M9.5 10 6.5 18M14.5 10l3 8M9.5 10h5"/>`,
    helix: `<path ${S} stroke-width="2.2" d="M4 18c2 0 3-4 5-4s1 4 3 4 3-4 5-4 1 4 3 4M4 10c2 0 3-4 5-4s1 4 3 4 3-4 5-4 1 4 3 4" opacity=".35"/><path fill="none" stroke="currentColor" stroke-width="4.6" stroke-linecap="round" d="M3 16c2.5-6 5.5-6 6.5 0s4 6 6.5 0 4-6 5 0"/><path fill="none" stroke="var(--ic, #cfe2ff)" stroke-width="2" stroke-linecap="round" d="M3 16c2.5-6 5.5-6 6.5 0s4 6 6.5 0 4-6 5 0"/>`,
    pores: `<rect ${S} x="3.5" y="3.5" width="17" height="17" rx="2"/><circle ${F} cx="9" cy="9" r="2.6"/><circle ${F} cx="15.5" cy="14.5" r="3"/><circle ${F} cx="8" cy="16" r="1.5"/><circle ${F} cx="16" cy="7.5" r="1.3"/>`,
    network: `<path ${F} d="M4 15c1-5 5-2 6-6s6-5 9-2-1 5 1 8-4 6-8 4-9 1-8-4z"/><circle cx="9" cy="14" r="1.3" fill="#fff"/><circle cx="15" cy="10" r="1.6" fill="#fff"/>`,
    duplicate: `<rect ${F} x="8" y="8" width="12" height="12" rx="2"/><path ${S} d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>`,
    delete: `<path ${S} d="M4 7h16M9 7V4h6v3"/><path ${F} d="M6 7l1 13h10l1-13"/><path ${S} d="M10 11v6M14 11v6"/>`,
    gauge: `<path ${F} d="M3.5 16a8.5 8.5 0 0 1 17 0z"/><path ${S} d="M12 16l{X}"/><circle cx="12" cy="16" r="1.4" fill="currentColor"/>`,
    auto: `<rect ${S} x="3.5" y="3.5" width="12" height="12"/><path ${S} d="M3.5 7.5h12M3.5 11.5h12M7.5 3.5v12M11.5 3.5v12" opacity=".5"/><path ${F} d="M18 12l1 2.5 2.5 1-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z"/>`,
    checks: `<rect ${F} x="4" y="3.5" width="16" height="17" rx="2"/><path ${S} d="M7.5 8.5l1.5 1.5 3-3M7.5 14.5l1.5 1.5 3-3M14 9h3M14 15h3"/>`,
    thermal: `<path ${F} d="M10 4a2 2 0 0 1 4 0v9.3a4 4 0 1 1-4 0z"/><path ${S} d="M12 9v7"/><circle cx="12" cy="16.5" r="1.8" fill="currentColor"/>`,
    electrical: `<path ${F} d="M13.5 2.5 5.5 13.5h6l-1.5 8 8-11h-6z"/>`,
    dielectric: `<path ${S} d="M12 3v6M12 15v6"/><rect ${F} x="4" y="9" width="16" height="2" rx="1"/><rect ${F} x="4" y="13" width="16" height="2" rx="1"/><path ${S} d="M7 6.5h2M8 5.5v2M7 18h2" opacity=".7"/>`,
    magnetic: `<path ${F} d="M5 4h4v8a3 3 0 0 0 6 0V4h4v8a7 7 0 0 1-14 0z"/><path ${S} d="M5 8h4M15 8h4"/>`,
    emi: `<path ${F} d="M12 3 19.5 6v5.5c0 4.5-3.2 8-7.5 9.5-4.3-1.5-7.5-5-7.5-9.5V6z"/><path ${S} d="M8.5 11.5l2.4 2.4 4.6-4.6"/>`,
    cte: `<rect ${F} x="8" y="8" width="8" height="8" rx="1"/><path ${S} d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M10.5 4 12 2.5 13.5 4M10.5 20l1.5 1.5 1.5-1.5M4 10.5 2.5 12 4 13.5M20 10.5l1.5 1.5-1.5 1.5"/>`,
    permeability: `<path ${F} d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/><path ${S} d="M9 14.5c.5 1.6 1.6 2.5 3 2.8"/>`,
    tortuosity: `<path ${S} d="M3 18c3 0 3-12 6-12s3 12 6 12 3-12 6-12"/><circle ${F} cx="3.5" cy="18" r="1.8"/><circle ${F} cx="20.5" cy="6" r="1.8"/>`,
    optimize: `<circle ${F} cx="12" cy="12" r="8.5"/><circle ${S} cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><path ${S} d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/>`,
    surrogate: `<path ${S} d="M3.5 18.5c3.5 0 5-11 8.5-11s5 8 8.5 8"/><circle ${F} cx="7.5" cy="13.5" r="1.9"/><circle ${F} cx="12" cy="7.5" r="1.9"/><circle ${F} cx="17.5" cy="14.5" r="1.9"/>`,
    acoustics: `<path ${F} d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path ${S} d="M15.5 9a4.5 4.5 0 0 1 0 6M18 6.5a8 8 0 0 1 0 11"/>`,
    radiation: `<circle ${F} cx="12" cy="12" r="4"/><path ${S} d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>`,
    morphology: `<path ${S} d="M3.5 20.5h17"/><rect ${F} x="5" y="12" width="3" height="8.5"/><rect ${F} x="10.5" y="6" width="3" height="14.5"/><rect ${F} x="16" y="10" width="3" height="10.5"/>`,
    porosimetry: `<circle ${F} cx="12" cy="13" r="7.5"/><path ${S} d="M12 13l4-3.5M9.5 3h5M12 3v2.5"/><circle cx="12" cy="13" r="1.3" fill="currentColor"/>`,
    pore_network: `<path ${S} d="M6 7l6 4 6-5M12 11l-5 7M12 11l6 6"/><circle ${F} cx="6" cy="7" r="2.4"/><circle ${F} cx="18" cy="6" r="2"/><circle ${F} cx="12" cy="11" r="2.8"/><circle ${F} cx="7" cy="18" r="1.8"/><circle ${F} cx="18" cy="17" r="2.2"/>`,
    grains: `<circle ${F} cx="8" cy="8.5" r="4"/><ellipse ${F} cx="16.5" cy="9" rx="3.5" ry="2.2" transform="rotate(-30 16.5 9)"/><circle ${F} cx="10" cy="17" r="3"/><rect ${F} x="14.5" y="14" width="6" height="3" rx="1.5" transform="rotate(20 17.5 15.5)"/>`,
    solver: `<circle ${S} cx="12" cy="12" r="3"/><path ${F} d="M12 2.8l1.6 2.3 2.7-.7.6 2.7 2.6 1-1 2.6 1.8 2.1-2.1 1.8 1 2.6-2.6 1-.6 2.7-2.7-.7L12 21.2l-1.6-2.3-2.7.7-.6-2.7-2.6-1 1-2.6-1.8-2.1 2.1-1.8-1-2.6 2.6-1 .6-2.7 2.7.7z" fill-opacity=".35"/><circle ${S} cx="12" cy="12" r="3"/>`,
    view3d: `<path ${F} d="M12 4 19 8v8l-7 4-7-4V8z"/><path ${S} d="M5 8l7 4 7-4M12 12v8"/>`,
    table: `<rect ${F} x="3.5" y="4.5" width="17" height="15" rx="2"/><path ${S} d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10"/>`,
    report: `<path ${F} d="M6 3h8l4 4v14H6z"/><path ${S} d="M14 3v4h4M9 11h6M9 14h6M9 17h4"/>`,
    json: `<path ${S} d="M9 4c-2 0-3 1-3 3v2c0 1.3-.7 2-2 3 1.3 1 2 1.7 2 3v2c0 2 1 3 3 3M15 4c2 0 3 1 3 3v2c0 1.3.7 2 2 3-1.3 1-2 1.7-2 3v2c0 2-1 3-3 3"/>`,
    layers: `<path ${F} d="M12 4 21 8.5 12 13 3 8.5z"/><path ${S} d="M3 12.5l9 4.5 9-4.5M3 16.5l9 4.5 9-4.5"/>`,
    image: `<rect ${F} x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="10" r="1.8" fill="currentColor"/><path ${S} d="M4 18l5-5 3.5 3.5 2.5-2.5 5 4"/>`,
    zip: `<path ${F} d="M6 3h12v18H6z"/><path ${S} d="M12 3v2M12 7v2M12 11v2M10.5 14h3v4h-3z"/>`,
    camera: `<path ${F} d="M4 8h3l1.8-2.5h6.4L17 8h3v11H4z"/><circle ${S} cx="12" cy="13.3" r="3.3"/>`,
    runs: `<path ${S} d="M8 6h12M8 12h12M8 18h12"/><circle ${F} cx="4.5" cy="6" r="1.6"/><circle ${F} cx="4.5" cy="12" r="1.6"/><circle ${F} cx="4.5" cy="18" r="1.6"/>`,
    display: `<path ${S} d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle ${F} cx="16" cy="6" r="2"/><circle ${F} cx="10" cy="12" r="2"/><circle ${F} cx="18" cy="18" r="2"/>`,
    info: `<circle ${F} cx="12" cy="12" r="8.5"/><path ${S} d="M12 11v5.5M12 7.6v.1"/>`,
    warn: `<path ${F} d="M12 3.5 21 19.5H3z"/><path ${S} d="M12 9.5v4.5M12 16.8v.1"/>`,
    home: `<path ${F} d="M4 11 12 4l8 7v9h-5.5v-5.5h-5V20H4z"/>`,
    chevron: `<path ${S} d="M9 6l6 6-6 6"/>`,
    plus: `<path ${S} d="M12 5v14M5 12h14"/>`,
    download: `<path ${S} d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19.5h14"/>`,
    fit: `<path ${S} d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>`,
    collapse: `<path ${S} d="M6 9l6 6 6-6"/>`,
  };
  const GAUGE = { fast: "3.2-4.2", standard: "0-5", accurate: "-3.2-4.2" };
  window.ICON = function (name, cls) {
    let body = P[name];
    if (!body && GAUGE[name]) body = P.gauge.replace("{X}", GAUGE[name]);
    if (!body) body = P.info;
    return `<svg viewBox="0 0 24 24" class="ico ${cls || ""}" aria-hidden="true">${body}</svg>`;
  };
})();
