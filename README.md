<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_dark.png">
    <img src="docs/images/logo_light.png" alt="MPSim - Material Property Simulation" width="54%">
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.2.5-0b62c4" alt="version 5.2.5">
  <img src="https://img.shields.io/badge/python-3.11-3776ab" alt="Python 3.11">
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011%20x64-555555" alt="Windows 10 | 11 x64">
  <img src="https://img.shields.io/badge/license-MIT-2da44e" alt="MIT licence">
  <a href="https://doi.org/10.5281/zenodo.23120194"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23120194.svg" alt="DOI 10.5281/zenodo.23120194"></a>
  <img src="https://img.shields.io/badge/self--test-64%20checks-2da44e" alt="64 self-test checks">
  <a href="#documentation"><img src="https://img.shields.io/badge/user%20guide-EN%20%7C%20KO-8250df" alt="user guide in English and Korean"></a>
</p>

<p align="center">
  <b>MPSim (Material Property Simulation)</b> computes the effective properties of semiconductor materials
  from their microstructure, in a browser workspace.
</p>

<p align="center">
  <a href="https://hyoseokbyun-opt.github.io/MPSim/">Website</a> ·
  <a href="#software-description">Description</a> ·
  <a href="#analyses">Analyses</a> ·
  <a href="#application-example-a-mould-underfill-from-flow-to-cure">Example</a> ·
  <a href="#verification">Verification</a> ·
  <a href="#installation">Installation</a> ·
  <a href="https://github.com/HyoseokByun-opt/MPSim/releases/latest/download/MPSim-User-Guide-EN.pdf">User guide</a> ·
  <a href="#citing">Citing</a>
</p>

<p align="center">
  <img src="docs/images/banner.png" alt="Microstructures and fields computed with MPSim" width="100%">
</p>

## Software description

MPSim (Material Property Simulation) has been developed to compute effective material properties and to perform
material response simulations on digital microstructures of composite and porous materials. MPSim generates
artificial microstructures on a voxel representative volume element (RVE), with fillers of arbitrary shape, size
distribution and orientation, pores, coatings and thin films, and solves the governing equations on the RVE with
validated open-source solvers. The results comprise the effective property tensors, the local fields (temperature,
heat flux, electric field, stress, water activity, flow velocity) and a set of checks on the representativeness of the
RVE.

The software targets the materials of semiconductor packaging, devices and process equipment: moulding compounds,
underfills, thermal interface materials, conductive adhesives, low-k and high-k dielectrics, EMI absorbers, porous
ceramics and filters. The workflow is carried out in a single browser workspace: material definition, microstructure
generation, voxel grid, RVE checks, solution, and the evaluation of fields, charts and reports. MPSim runs on a single
PC or on a server that is accessed from other computers on the network; data do not leave the machine.

The main capabilities are listed below.

- **Microstructure generation.** Spheres, spheroids (needles and platelets), cylinders, capsules, cubes, cuboids,
  superellipsoids, polyhedra, helices and connected networks with full 3D rotation; log-normal size distributions,
  coated and hollow particles, several filler phases in one RVE, pores as a material phase and gas matrices. Dense
  packings are obtained by rigid-clump relaxation and Monte-Carlo compression, and every blend is checked so that no
  voxel belongs to two particles. Sphere packings can be sheared in the resin by particle dynamics before they are
  drawn, which yields the arrangement of a compound after flow.
- **RVE sizing.** The RVE edge is 7 times (Standard) or 10 times (Accurate) the largest particle, a criterion determined
  with the statistical method of Kanit et al. (2003). The volume-fraction error, octant homogeneity, directional scatter
  and convergence are checked on every run; several realisations provide a 95 % confidence interval.
- **Thin films.** Periodic in x and y, with the real film thickness in z and every particle inside the film.
- **Conduction.** Thermal, electrical, dielectric and magnetic conduction with a periodic finite-volume solver
  (multi-core, FFT-preconditioned) and, alternatively, the finite elements of NASA PuMA. Interfacial thermal resistance
  between filler and matrix and at filler-filler contacts is applied on the voxel faces. Anisotropic fillers (for
  example h-BN platelets) carry their conductivity tensor rotated with each particle.
- **Mechanics.** Stiffness, engineering constants and the thermal-expansion tensor from FANS, the periodic voxel finite
  elements of NASA PuMA solved with an FFT preconditioner whose iteration count does not grow with the grid (seven load
  cases on 96³ voxels in 49 s, 11 times faster than MINRES on the same discretisation), optionally above the glass
  transition.
- **Rheology.** Effective viscosity of underfills, moulding compounds and pastes from a creeping-flow solution on the RVE
  and, for dense fillers of any shape, from particle dynamics in which every filler translates, rotates and collides in
  the sheared resin (lubrication, friction, van der Waals attraction; cubes, platelets and rods as rigid clumps of
  spheres; NVIDIA GPU or CPU). The particle dynamics also provides the particle-size effect and the shear thinning
  caused by attraction. Krieger-Dougherty with the maximum packing fraction of the actual size distribution, flow
  curves, capillary underfill filling time and filler settling complete the analysis.
- **Moisture.** Effective moisture diffusivity (solved on the water activity), saturated uptake, the uptake curve of a
  part and hygroscopic swelling.
- **Electromagnetic shielding.** Frequency-resolved complex homogenisation coupled with a transmission-line model, and a
  full-wave simulation of the RVE slab in openEMS (FDTD, voxel by voxel) that is run by default to verify the
  homogenisation.
- **Porous media.** Permeability (Stokes flow), filtration efficiency by particle tracking, diffusion and tortuosity,
  acoustic absorption (Johnson-Champoux-Allard), radiative extinction, porosimetry, pore networks, percolation paths
  and grain statistics.
- **Calibration.** Unknown material values (interfacial and contact resistance, the conductivity of a filler grade, the
  bound resin layer or the adhesion of a filler surface) are determined from measured effective properties, including
  viscosity: the acoustic and diffuse mismatch models as a starting point, an identifiability check, a Bayesian calibration
  on an RVE or particle-dynamics surrogate with credible intervals, and the next measurement that adds the most
  information.
- **Studies.** Design of experiments (full factorial, one factor at a time, Latin hypercube; the share of a filler in a
  fixed filler content as a factor; one fixed grid for all cases or a grid per case), Gaussian-process surrogates,
  Sobol sensitivity indices, single-objective optima and NSGA-II Pareto fronts. The result table lists, next to each
  input, the filler content and filler shares realised in the generated structure.
- **Linked inputs.** A DOE factor or a calibration unknown can set further inputs: the same value (`x`), a ratio
  (`0.8*x`) or a function of several factors (`sqrt(x1*x2)`). Two grades of one filler can thus share one interfacial
  resistance, which reduces the number of factors and lets the measurements determine a single value.
- **Parallel execution.** Independent solutions (each property and direction, each Stokes and diffusion direction,
  each EMI frequency, the structure analyses) run concurrently on the cores assigned to a run within the free memory,
  and several runs can share one PC. A porous RVE with permeability, diffusion and structure analyses took 1231 s
  sequentially and 693 s in parallel on 8 cores.
- **Visualisation.** An interactive 3D view (vtk.js) with particles as smooth surfaces or as voxels, fields on the RVE
  faces and on sections, arrows, animated field lines and pore networks; publication figures rendered on the server
  (PyVista); animations saved as GIF; live fields during the solution; a single-file HTML report for every run.

### The workspace

<p align="center">
  <img src="docs/images/screenshot.png" alt="The workspace: ribbon, command panel, 3D view and inspector" width="100%">
</p>

The ribbon follows the order of a study (Home, Material, Structure, Domain, Structure Analysis, Simulation, Results).
The command panel on the left holds the inputs of the active tab, the centre shows the 3D/2D view, the Result Viewer
or the report, and the inspector on the right controls the display. Chapter 3 of the user guide describes the
workspace in detail.

## Analyses

| Group | Analysis | Solver | Main results |
|---|---|---|---|
| Conduction | Thermal conductivity | Periodic FV (default), NASA PuMA FE | k tensor, directional k(n), diffusivity, area-specific resistance, bounds and mean-field models |
| | Electrical conductivity | same | σ tensor, resistivity, sheet resistance, percolation |
| Electromagnetics | Permittivity and loss | same | Dk tensor, Df, local field enhancement per region |
| | Magnetic permeability | same | μr tensor |
| | EMI shielding | Complex homogenisation → scikit-rf; openEMS FDTD check | SE(f) with reflection, absorption and multiple-reflection terms, validity limits |
| Mechanics | Elasticity and thermal expansion | FANS voxel FE (default), NASA PuMA FE | C\*, S\*, E, G, ν, Hill moduli, anisotropy index, α tensor, thermal stress per region; optionally α2 and E above Tg |
| Rheology | Viscosity and flowability | FANS creeping flow; particle dynamics (lubrication DEM); Krieger–Dougherty with Farr–Groot φm | Relative viscosity, flow curve, φm, underfill filling time, settling, local shear-rate field |
| Reliability | Moisture uptake and swelling | Periodic FV on the water activity; FANS eigenstrain | D_eff tensor, saturated uptake, uptake curve, swelling strain and CME |
| Porous media | Permeability | NASA PuMA Stokes FE | K tensor, Kozeny–Carman constant, hydraulic tortuosity |
| | Filtration efficiency | Particle tracking in the Stokes field | Efficiency per particle size, most penetrating size, quality factor |
| | Diffusion and tortuosity | NASA PuMA | τ, D_eff/D₀, formation factor, MacMullin number, Knudsen and Bosanquet diffusivities |
| | Acoustic absorption | JCA model from the computed fields | Absorption, surface impedance, octave bands, NRC |
| Radiation | Radiative extinction | NASA PuMA ray casting | Extinction coefficient, Rosseland k_rad(T) |
| Structure | Porosity and size distributions | PoreSpy, NASA PuMA | Open/closed porosity, local thickness, chords, two-point correlation |
| | Porosimetry | PoreSpy | Intrusion/extrusion curves, bubble points |
| | Pore network | PoreSpy SNOW2 | Pores, throats, coordination numbers |
| | Percolation paths | Connected components, geodesic distance | Spanning fractions, geodesic tortuosity, largest passing sphere |
| | Grain analysis | Particle list, PoreSpy, PuMA structure tensor | Size distribution, sphericity, orientation tensor, spacing, clusters |
| Inverse problems | Calibration | RVE surrogate (Gaussian process), Bayesian posterior | Unknown material values with credible intervals, identifiability, next measurement |

Each result states the method that produced it. Conduction and thermal-expansion results are compared with rigorous
bounds (Wiener, Hashin–Shtrikman, Schapery); a value outside a rigorous bound is reported as a failed check.

## Application example: a mould underfill from flow to cure

This example follows one compound through the analyses that are relevant for a moulded underfill (MUF): epoxy filled
with fused silica of 15 µm and 3 µm at 55 vol% (38.5 vol% and 16.5 vol%). The viscosity of the uncured compound is
computed by particle dynamics. The same particle dynamics then shears a random packing of the compound in the resin,
and the arrangement it leaves is taken as the cured microstructure. Thermal conductivity, thermal expansion, stiffness,
permittivity, dielectric loss and moisture uptake are solved on this one structure (60 µm RVE, 240³ voxels). All
results below were obtained with the presets *MUF · epoxy + silica 15 + 3 µm 55 vol%* and *MUF · from flow to the cured
part* on a 14-core laptop without a GPU.

<p align="center">
  <img src="docs/images/muf_datasheet.png" alt="Summary of the properties of the cured compound" width="72%">
</p>

### Viscosity of the uncured compound

The particle dynamics resolves the lubrication forces between neighbouring particles, frictional contacts and van der
Waals attraction; every particle translates and rotates freely in a periodic box sheared by Lees-Edwards boundaries.
The relative viscosity is computed from the stress carried by the particles. In the animation, 2,222 particles are
coloured by their velocity relative to the imposed shear flow; red particles move fastest against their neighbours.

<p align="center">
  <img src="docs/images/viscosity_shear.gif" alt="Particles of the mould underfill sheared in the resin (particle dynamics)" width="46%">
</p>
<p align="center">
  <img src="docs/images/viscosity_showcase.png" alt="Relative viscosity over the sheared strain and the flow curve of the compound" width="100%">
</p>

The relative viscosity overshoots while the shear microstructure forms and reaches a steady value from a strain of
about 6. The value is averaged over the second half of the run (a strain of 10 by default): μr = 24 ± 3, where the
uncertainty is the expected scatter between runs from independent packings. With the shear-thinning (Carreau) resin
the compound reaches 33 Pa·s at 10 s⁻¹. Derived results:

| Quantity | Value |
|---|---|
| Maximum packing fraction φm (Farr–Groot, actual size distribution) | 0.77 |
| Capillary filling of a 50 µm × 10 mm gap (Washburn, parallel plates, contact angle 30°) | 108 min (resin alone: 6.3 min) |
| Settling of the 15 µm silica in 30 min before gelation (hindered) | 2.9 µm, 41 times slower than an isolated particle |

### Microstructure after shear flow

*Domain → Particle arrangement → After shear flow* shears the random packing with the particle dynamics above, with
the flow along x and the velocity gradient along z, to a whole-number strain (10), at which the sheared periodic images
coincide with the ordinary ones. The spheres carry half of the packing gap during the shear, so that the final positions
need corrections of a few nanometres only (6 nm on average here).

<p align="center">
  <img src="docs/images/muf_structure.png" alt="The cured microstructure and the orientation of close particle pairs" width="100%">
</p>

The number of close pairs (surfaces within 5 % of the sum of the radii) increases from 2.5 to 4.5 per sphere. The
pairs orient along the compression axis of the shear (135° from the flow direction in the x-z plane) and are depleted
along the extension axis (45°); the mean of n_x·n_z changes from -0.004 to -0.024. This anisotropic pair distribution
is characteristic of sheared non-Brownian suspensions (J. F. Morris, *Rheologica Acta* **48** (8), 909–923 (2009),
[doi:10.1007/s00397-009-0352-1](https://doi.org/10.1007/s00397-009-0352-1)). The filler contents of the drawn structure
are 38.50 and 16.50 vol%, equal to the input. The shear of 2,568 spheres to a strain of 10 took 36 minutes on 14 CPU
cores; on an NVIDIA GPU it takes minutes.

### Thermal conductivity

Steady heat conduction is solved on the cured RVE with periodic finite volumes for a temperature difference along x,
y and z. The animation shows the magnitude of the heat flux for heat driven along z, from the die side to the
substrate side of the gap; a cut plane travels through the RVE.

<p align="center">
  <img src="docs/images/heat_flux.gif" alt="Heat flux inside the cured compound, a cut plane travelling through the RVE" width="46%">
</p>
<p align="center">
  <img src="docs/images/muf_heat.png" alt="Thermal conductivity per direction, random packing and after shear flow" width="100%">
</p>

| Direction | Random packing (W/m·K) | After shear flow (W/m·K) | Change |
|---|---|---|---|
| x (flow) | 0.573 | 0.573 | 0.0 % |
| y (vorticity) | 0.570 | 0.579 | +1.6 % |
| z (velocity gradient, through the gap) | 0.569 | 0.573 | +0.7 % |

The conductivity through the gap is 0.573 W/m·K, 2.9 times that of the resin. It lies 5 % above the Maxwell-Garnett
estimate, which coincides with the lower Hashin–Shtrikman bound (0.544), and well below the Bruggeman estimate (0.685):
the particles are separated by resin films, and the composite behaves as a dispersion of particles. The heat flux is
highest in the silica and in the thin resin gaps between near-contacting particles. The changes caused by the shear
(up to 1.6 %) are smaller than the statistical scatter of a single realisation (±4.3 % at 50 vol% for a 7-diameter
RVE, and more for this 4-diameter RVE); for
this low-contrast filler the arrangement after flow does not change the conductivity measurably. Refining the grid
from 240³ to 360³ changes k_z by +0.12 %. A 1 mm layer has an area-specific thermal resistance of 1745 mm²K/W, and the
thermal diffusivity is 0.386 mm²/s.

### Thermal expansion and stiffness

The effective stiffness tensor and the thermal-expansion tensor are obtained with FANS from six unit strains and a
uniform temperature change. The render shows the von Mises stress for a temperature rise of 100 K with the RVE held at
its size.

<p align="center">
  <img src="docs/images/muf_mechanics.png" alt="Thermal stress, thermal expansion and Young's modulus per direction" width="100%">
</p>

| Direction | α (ppm/K) | E (GPa) |
|---|---|---|
| x (flow) | 18.7 | 13.4 |
| y (vorticity) | 17.8 | 13.7 |
| z (velocity gradient) | 18.5 | 13.8 |

The volumetric mean CTE is 18.4 ppm/K, below the Kerner (Mori-Tanaka) estimate for separated spheres (22.6 ppm/K) and
above the Turner estimate (4.7 ppm/K); near-contacting particles restrain the resin more than separated spheres do.
The Hill average of the Young's modulus is 13.3 GPa (Poisson's ratio 0.26), within the Hashin–Shtrikman bounds
(9.4–30.2 GPa); the shear moduli G_yz, G_xz and G_xy are 5.13, 5.01 and 5.39 GPa. The direction dependence is small but consistent with the pair orientation: fewer close pairs lie along
the flow direction x, where E is lowest and α highest. Its magnitude (up to 5 %) is comparable with the scatter of one
realisation. The thermal stress concentrates in the fine filler and at near contacts: the mean von Mises stress is
4 MPa in the resin (99th percentile 13 MPa), 8 MPa in the 15 µm silica (33 MPa) and 15 MPa in the 3 µm silica
(68 MPa). These local stresses indicate where filler-matrix debonding is likely to start during temperature cycling.

Properties above the glass transition are not reported in this example: in the rubbery state the particles that touch
on the voxel grid form a rigid skeleton that the real compound, with resin films between the particles, does not have
(user guide, section 14.5).

### Permittivity and dielectric loss

Permittivity and dielectric loss are obtained from a quasi-static complex conduction solution along x, y and z. The
render shows the local electric field relative to the applied field for a field along z.

<p align="center">
  <img src="docs/images/muf_dielectric.png" alt="Local electric field, dielectric constant and dissipation factor" width="100%">
</p>

The dielectric constant is 3.845 in all three directions. The permittivities of silica (3.8) and epoxy (3.9) differ by
2.6 %, so that the Hashin–Shtrikman bounds coincide (3.8448) and the local field departs from the applied field by at
most 2.9 %. The dissipation factor is 0.0068, equal to the volume-weighted mean of the phases (0.45 × 0.015 + 0.55 ×
0.0001): fused silica leaves the dielectric constant unchanged and lowers the loss in proportion to the resin it
replaces.

### Moisture uptake and swelling

Moisture diffusion is solved on the water activity c/c_sat, which is continuous across the phase boundaries; the silica
does not absorb water. The uptake curve of a part follows from the effective diffusivity, and the swelling strain from
an eigenstrain solution with FANS.

<p align="center">
  <img src="docs/images/muf_moisture.png" alt="Water activity, uptake curve of a 1 mm part and diffusivity per direction" width="100%">
</p>

| Quantity | Cured compound | Epoxy |
|---|---|---|
| Saturated uptake | 0.77 wt% | 2.5 wt% |
| Diffusivity D_x / D_y / D_z (10⁻¹² m²/s) | 2.00 / 2.04 / 1.99 | 3.0 |
| 1 mm part, both faces exposed: 50 % / 95 % of saturation | 6.9 h / 39 h | 4.6 h / 26 h |
| Swelling strain at saturation (per direction) | 0.23 % | 0.75 % |

The saturated uptake equals the resin uptake scaled by the mass fraction of the resin. The effective diffusivity is
0.67 times that of the resin, 15 % below the Maxwell estimate for impermeable spheres (0.78), because at 55 vol% the
diffusion paths wind between closely spaced particles. The silica restrains the hygroscopic swelling of the resin to
0.23 %, which corresponds to a coefficient of moisture expansion of 0.29 per unit mass fraction of water.

## Verification

`5_SELF_TEST.bat` solves **64 problems with known solutions** (42 in the quick set) and compares the results with the
reference values. A selection:

| Check | Reference | Result |
|---|---|---|
| Series and parallel laminates (bulk and thin film) | 1.81818 / 5.5 | exact |
| Interfacial thermal resistance | Hasselman–Johnson | within 0.26 % |
| Simple cubic sphere array, extrapolated from 256³ | converged reference | 64³: FV +0.7 %, FE +4.0 % |
| Two-phase thermal expansion | Levin's exact relation | relative error 5e-7 |
| Complex laminates (EMI homogenisation) | series/parallel closed form | error 1e-13 |
| Dilute spheres, complex admittivity | Maxwell-Garnett | within 0.06 % |
| Homogeneous slab in openEMS | exact slab transmission | within 0.001 dB |
| Cylindrical pore | Poiseuille; Λ = Λ′ = r | passed |
| Sobol indices of the Ishigami function | closed form | passed |
| FANS elasticity against PuMA (corrected) | same discretisation | 3.5e-9 |
| Dilute rigid sphere in a viscous liquid | Einstein [η] = 2.5 | within 15 % (voxelised sphere) |
| Particle dynamics, frictionless spheres at 50 vol% | Krieger–Dougherty with random close packing, 11.4 | 12.6 (200 spheres, strain 2) |
| Particle dynamics of cubes (rigid multi-sphere particles) | dissipated power = shear stress × rate | within 0.1 % |
| Anisotropic conduction: rotated tensor; tilted anisotropic laminate | exact; laminate formula | 2e-15; 2.6 % at 16 cells per layer (first order) |
| Maximum packing of bidisperse spheres | Farr–Groot simulations, 27 cases | within 0.011 |
| Impermeable sphere, moisture diffusion | Maxwell 2/(2+φ) | within 2 % |
| Calibration of R_int and filler k from two particle sizes | known values 2e-7 m²K/W, 2.0 W/m·K | 2.0e-7, 2.01 |
| Calibration of a bound resin layer from a viscosity (particle dynamics) | known value 10 nm | 10.6 nm (95 % 8.8–12.5) |
| Blends: two sphere sizes at 55 vol%, cubes with flakes at 45 vol% | no voxel inside two particles | 0 shared voxels, none trimmed |
| Complete pipelines: filled RVE, thin film, porous RVE, viscosity with live results | all files, all analyses | passed |
| DOE: share of a filler at a fixed total content; fixed grid | 5–50 % of 55 vol%, 160³ on every case | passed |
| Linked inputs: one R_int for two grades of a filler (calibration) | known value 3e-8 m²K/W | 3.0e-8 linked (4.5e-8 when only one grade is calibrated) |
| Stop of a run during parallel solutions | all processes of the run end | within seconds |
| Arrangement after shear flow: silica 4 + 1.6 µm, 45 vol%, strain 2 | contents kept, periodic, pairs oriented by the flow | close pairs 3.7 → 5.5 per sphere, n_x·n_z +0.014 → -0.024 |

Two presets also verify the solver against an exact solution. Copper vias that pass straight through a laminate conduct
heat as parallel paths, k_z = f·k_Cu + (1 − f)·k_laminate for a copper area fraction f. The presets *PCB thermal vias*
and *Glass core substrate · Cu TGVs* give 9.2851 and 36.0295 W/m·K, which agree with this value to the fourth decimal;
in version 5.2.3 this check revealed that thin films ignored the anisotropy of their laminate.

**Comparison with published measurements.** Four presets reproduce materials whose properties have been measured and
published. Each preset was run as loaded; the preset descriptions discuss the remaining differences.

| Preset | Reference | Reported | Computed |
|---|---|---|---|
| Heat spreader · Cu + diamond 50 vol% (400 µm) | Wang et al., *Materials* 2025 [1] | 640 W/m·K | 644 W/m·K, with 12.5 MW/m²K at the interface |
| Power-module baseplate · AlSiC 63 vol% | Guo et al., *J. USTB* 2014 [2] | 178 W/m·K, 8.21 ppm/K | 189 W/m·K, 8.84 ppm/K |
| mmWave substrate · PTFE + silica 60 wt% | Murali et al., *Compos. Part A* 2009 [3] | Dk 2.9 | Dk 3.01 |
| Power inductor · epoxy + carbonyl iron 54 vol% | Sugawa et al., *IEEE Trans. Magn.* 2013 [4] | μr 7.5 (printed) | μr 5.1 (separate spheres) |

1. Q. Wang, Z. Ye, L. Liu, J. Bai, Y. Zhao, Q. Hu, H. Liu, L. Hu, X. Guo, Y. Xiao, W. Cao, Z. Yang, “High thermal conductivity diamond–copper composites prepared via hot pressing with tungsten-coated interfacial layer optimization”, *Materials* **18** (16), 3882 (2025). [doi:10.3390/ma18163882](https://doi.org/10.3390/ma18163882) — 400 µm diamond, 50 vol%: 640 W/m·K.
2. M. Guo, J. Liu, C. Jia, S. Guo, Y. Li, H. Zhou, “Microstructure and properties of SiCp/Al electronic packaging materials fabricated by pseudo-semi-solid thixoforming”, *Journal of University of Science and Technology Beijing* **36** (4), 489–495 (2014). [doi:10.13374/j.issn1001-053x.2014.04.011](https://doi.org/10.13374/j.issn1001-053x.2014.04.011) — as tabulated in L. Lai et al., *Micromachines* **14** (8), 1491 (2023), Table 2, [doi:10.3390/mi14081491](https://doi.org/10.3390/mi14081491).
3. K. P. Murali, S. Rajesh, O. Prakash, A. R. Kulkarni, R. Ratheesh, “Preparation and properties of silica filled PTFE flexible laminates for microwave circuit applications”, *Composites Part A* **40** (8), 1179–1185 (2009). [doi:10.1016/j.compositesa.2009.05.007](https://doi.org/10.1016/j.compositesa.2009.05.007) — 60 wt% micron silica, X band.
4. Y. Sugawa, K. Ishidate, M. Sonehara, T. Sato, “Carbonyl-iron/epoxy composite magnetic core for planar power inductor used in package-level power grid”, *IEEE Transactions on Magnetics* **49** (7), 4172–4175 (2013). [doi:10.1109/TMAG.2013.2250925](https://doi.org/10.1109/TMAG.2013.2250925) — screen-printed 54 vol% core.

**RVE size.** The 7× and 10× criteria were determined from repeated runs on independent random structures, evaluated
with the statistical method of Kanit et al. (2003). The random scatter of one realisation (95 % interval) was ±0.12 % /
±0.14 % for the conductivity of 30 vol% fillers, ±4.3 % / ±1.9 % at 50 vol%, and ±0.92 % / ±0.38 % for the Young's
modulus at 30 vol% (7× / 10×). The size bias was within the scatter from 5× onwards.

<p align="center">
  <img src="docs/images/rve_size_study.png" alt="RVE size study" width="85%">
</p>

Every run also reports its own checks (RVE size, resolution, particle count, convergence, bounds) as passed, warning or
failed, together with the reason. Chapter 31 of the user guide explains these checks.

## Documentation

| User guide | PDF |
|---|---|
| English, 180 pages | [MPSim-User-Guide-EN.pdf](https://github.com/HyoseokByun-opt/MPSim/releases/latest/download/MPSim-User-Guide-EN.pdf) |
| Korean, 168 pages | [MPSim-User-Guide-KO.pdf](https://github.com/HyoseokByun-opt/MPSim/releases/latest/download/MPSim-User-Guide-KO.pdf) |

Each simulation chapter describes how the property is measured in the laboratory, how MPSim computes it, how the
numbers and renderings are read, and the underlying assumptions; chapter 32 treats two semiconductor-material problems
from start to finish. The guides are attached to every [release](https://github.com/HyoseokByun-opt/MPSim/releases)
(the links above always open the latest version) and are included, together with an HTML version, in the offline
package.

## System requirements

| | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 10 or 11, 64-bit | Windows 11 |
| CPU | 4 cores | 16 or more cores (solvers and independent solutions run in parallel) |
| Memory | 16 GB (grids up to about 240³ for conduction and elasticity) | 64 GB or more; 512 GB runs 500³ elasticity |
| Disk | 2 GB for the installation, plus the runs | SSD |
| GPU | not required | NVIDIA GPU with CUDA for the particle dynamics |
| Browser | Chrome or Edge (WebGL 2) | same |

The RVE grid limit is 500³ by default and can be changed per server. The planner estimates the peak memory of each
solver before the run and warns with the estimate instead of refusing. Estimates per voxel: conduction about 230 bytes
(500³ ≈ 29 GB); elasticity about 300 bytes with FANS, the default (240³ measured at 3 GB, 500³ ≈ 38 GB) and 2.5 kB
with PuMA FE; Stokes flow about 2.3 kB.

## Installation

The program is distributed in two forms:

- **The source (this repository)**, installed with conda on a PC with internet access.
- **The offline package**, the same program with the official Python runtime, all 78 wheels and openEMS (about
  500 MB), for PCs without internet access. Binaries are not kept in the repository; the package is available on the
  [**Releases**](https://github.com/HyoseokByun-opt/MPSim/releases) page or can be built with `tools\build_offline.py`
  (see below).

### With conda (from the source)

On a PC with internet access and Miniconda or Anaconda:

```bat
git clone https://github.com/HyoseokByun-opt/MPSim.git C:\MPSim
cd C:\MPSim
1_INSTALL.bat
```

`1_INSTALL.bat` creates a private environment in `env\` from `environment.yml` (conda-forge), downloads openEMS into
`tools\openEMS` for the EMI full-wave simulation and runs the quick self-test; the equivalent manual command is
`conda env create -p .\env -f environment.yml`. The program is started with `2_RUN_LOCAL.bat`. A short installation
path such as `C:\MPSim` is recommended (see [Common errors](#common-errors)).

NASA PuMA is distributed on conda-forge as the package `puma`. The PyPI project `pumapy` is an unrelated package and
must not be installed.

### On an offline PC

The offline package contains the official Python 3.11.9 runtime from python.org, 78 wheels (320 MB) including NASA
PuMA, and openEMS. Neither internet access nor conda is required.

1. Copy the folder to the offline PC, preferably to a short path such as `C:\MPSim`.
2. Run **`1_INSTALL_OFFLINE.bat`**. It unpacks the runtime into `python\`, installs the wheels with pinned versions,
   unpacks openEMS into `tools\openEMS`, checks every import and runs the quick self-test. Nothing outside the folder
   is changed (no registry entries, no PATH changes).
3. Start the program with `2_RUN_LOCAL.bat`.

The program is removed by deleting the folder.

### Building the offline package

On a PC with internet access and the tested conda environment (`environment.yml`):

```bat
conda run -p .\env python tools\build_offline.py            :: runtime, openEMS, PuMA wheel, 78 wheels
conda run -p .\env python tools\build_offline.py --no-openems
```

PuMA is packed from the conda-forge build into a standard wheel. Its compiled modules import only `python311.dll`,
`VCRUNTIME140.dll` and the Windows C runtime, so that the same files work in any CPython 3.11 on 64-bit Windows. The
versions are pinned in `requirements-offline.txt` and `constraints-offline.txt`: PuMA 3.2.2 requires numpy < 2 and
scipy 1.11, and vtk 9.5.2 is pinned together with pyvista 0.47.1.

## Running

| File | Purpose |
|---|---|
| `2_RUN_LOCAL.bat` | This PC only; opens `http://127.0.0.1:8000` in the browser |
| `3_RUN_SERVER.bat` | Serves the local network at `http://SERVER-IP:8000`; `--token auto` adds a shared access token |
| `4_NETWORK_CHECK.bat` | Reports why other PCs cannot connect (changes nothing) |
| `5_SELF_TEST.bat` | Verifies the engine against known solutions (`--quick` for the short set) |
| `6_RUN_COMMANDLINE.bat` | Runs a preset without the browser, e.g. `--preset tim_aln --quality fast` |

Workflow in the browser: **Home** (preset) → **Material** → **Structure** (*Generate structure*) → **Domain**
(*Generate voxel grid*) → **Structure Analysis** and **Simulation** (select analyses) → **Run** →
**3D / 2D View**, **Result Viewer**, **Report**. **DOE**, **Optimization** and **Calibration** extend a single case to a
study.

Server options (`app\run_web.py`):

```text
--host 0.0.0.0            address to listen on (127.0.0.1 = this PC only)
--port 8000
--token auto              shared access token ("auto" generates one)
--workspace PATH          where runs are kept (default: workspace\)
--concurrency 1           jobs solved at the same time
--max-voxels 500          grid limit per edge for conduction; 0 = off
--max-voxels-elastic 500  for elasticity and thermal expansion
--max-voxels-flow 500     for Stokes flow
--check                   report why other machines may not connect, then exit
```

After an update, the browser page should be reloaded with **Ctrl+F5** so that the new scripts are loaded.

## Project layout

```text
1_INSTALL.bat, 1_INSTALL_OFFLINE.bat … 6_RUN_COMMANDLINE.bat, _common.bat
environment.yml, requirements-offline.txt, constraints-offline.txt
app/
  mpsim/                 engine, independent of the browser
    spec.py              input validation and label table
    rve.py               RVE criteria, automatic sizing, memory estimate
    shapes.py, geometry.py, generate.py      shapes, size distributions, RVE generation (numba)
    solvers/             conduction.py (periodic FV), fans.py and elastic.py (FANS mechanics),
                         viscosity.py (creeping flow), suspension.py (particle dynamics),
                         puma_backend.py (NASA PuMA), complex_cond.py and emi.py (EMI), fullwave.py (openEMS)
    morphology.py, poro.py, percolation.py, grains.py, filtration.py, acoustics.py, streamlines.py
    analytical.py        rigorous bounds and mean-field models
    doe.py, optimise.py, calibrate.py, links.py        studies, surrogate models, calibration, linked inputs
    visual.py, report.py viewer data, publication figures, HTML report
    pipeline.py          one run end to end
    data/                materials.json (68 materials), presets.json (30 presets)
  web/                   Flask server, job queue, workspace UI (vtk.js)
  run_web.py, run_cli.py, selftest.py
docs/images/             the README images (the user guides are attached to the releases)
tools/                   build_offline.py, check_imports.py, make_download.py
runtime/, wheels/        Python runtime, openEMS and wheels (offline package only, not in the repository)
workspace/               results, kept across restarts (created at the first start)
```

## Common errors

**`WinError 206` or "filename too long" during installation.** A few packages contain deep paths that exceed the
260-character limit of Windows. The folder should be moved to a short path such as `C:\MPSim`, or long paths enabled
(`LongPathsEnabled` = 1 under `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`; administrator rights are required).

**Other PCs cannot open the server.** In most cases the Windows firewall blocks the port or the network profile is
*Public*. `4_NETWORK_CHECK.bat` on the server lists the addresses and what blocks them. The connection is best tested
from another PC with `curl -m 5 http://SERVER-IP:8000/healthz` rather than with ping.

**`cannot import name 'vtkCapsuleSource'`** or a PuMA import error in a self-made environment. vtk ≥ 9.7 or numpy ≥ 2
was installed next to PuMA 3.2.2. Installing with `constraints-offline.txt` or `environment.yml` pins compatible
versions.

**A run stops with a memory warning.** The planner's estimate exceeds the free memory. The grid can be reduced (larger
voxel or smaller RVE), fewer analyses solved at once, or `--concurrency` lowered.

**The page looks outdated after an update.** The browser keeps the old scripts; reload with Ctrl+F5.

**The full-wave EMI check reports "openEMS not found".** openEMS is searched in `tools\openEMS`, the `MPSIM_OPENEMS`
environment variable, `C:\openEMS` and the PATH. The offline installer unpacks it into `tools\openEMS`.

Further cases are listed in appendix D of the user guide.

## Citing

Published work that uses MPSim should cite the software (GitHub shows the citation under **Cite this repository**,
from [CITATION.cff](CITATION.cff)):

> H. Byun, *MPSim: Material Property Simulation*, version 5.2.5, Zenodo (2026). https://doi.org/10.5281/zenodo.23120194

The DOI 10.5281/zenodo.23120194 always resolves to the latest version; each release also has its own DOI on Zenodo.

The solvers that produced the results should be cited as well:

- J. C. Ferguson, F. Panerai, A. Borner, N. N. Mansour, "PuMA: the Porous Microstructure Analysis software",
  *SoftwareX* 7, 81–87 (2018).
- J. C. Ferguson, F. Semeraro, J. M. Thornton, F. Panerai, A. Borner, N. N. Mansour, "Update 3.0 to PuMA: the Porous
  Microstructure Analysis software", *SoftwareX* 15, 100775 (2021).
- J. T. Gostick et al., "PoreSpy: A Python toolkit for quantitative analysis of porous media images", *Journal of
  Open Source Software* 4(37), 1296 (2019).
- T. Liebig, A. Rennings, S. Held, D. Erni, "openEMS – a free and open source equivalent-circuit (EC) FDTD
  simulation platform supporting cylindrical coordinates suitable for the analysis of traveling wave MRI
  applications", *International Journal of Numerical Modelling* 26(6), 680–696 (2013).
- A. Arsenovic et al., "scikit-rf: An open source Python package for microwave network creation, analysis, and
  calibration", *IEEE Microwave Magazine* 23(1), 98–105 (2022).
- R. Mari, R. Seto, J. F. Morris, M. M. Denn, "Shear thickening, frictionless and frictional rheologies in
  non-Brownian suspensions", *Journal of Rheology* 58(6), 1693–1724 (2014).
- F. Boyer, É. Guazzelli, O. Pouliquen, "Unifying suspension and granular rheology", *Physical Review Letters* 107,
  188301 (2011).
- Y. Hu, T.-M. Li, L. Anderson, J. Ragan-Kelley, F. Durand, "Taichi: a language for high-performance computation on
  spatially sparse data structures", *ACM Transactions on Graphics* 38(6), 201 (2019).
- T. Kanit, S. Forest, I. Galliet, V. Mounoury, D. Jeulin, "Determination of the size of the representative volume
  element for random composites: statistical and numerical approach", *International Journal of Solids and
  Structures* 40(13–14), 3647–3679 (2003) (the RVE size criteria).

## Contributing and bug reports

Bug reports and suggestions are welcome as GitHub issues. A report should include the run folder, or at least its
`spec.normalized.json`, `result.json` and the console log (**Results → Complete ZIP** collects all of them), the version
shown in the title bar and the output of `5_SELF_TEST.bat --quick`. Pull requests should keep the self-test passing and
add a check for any new solver or analysis.

## Authors

**Hyoseok Byun** ([GitHub](https://github.com/HyoseokByun-opt) · [LinkedIn](https://www.linkedin.com/in/hyoseok-byun-72992120a)): concept, design, direction and
validation.

**Declaration of AI assistance.** The code, the verification tests and the user guides were written with the assistance
of Claude (Anthropic; Claude Code with Claude Opus 5.5), an AI assistant, working under the author's direction. The
author reviewed and edited the output and takes full responsibility for the content. Following the practice of
scientific publishing, the AI assistant is acknowledged here rather than listed as an author; commits written with its
help carry a `Co-Authored-By: Claude` trailer.

MPSim builds on the open-source projects listed under [License](#license); their authors deserve the credit for the
solvers.

## License

The code and documentation of MPSim are released under the MIT License (see [LICENSE](LICENSE)). Redistributions must
keep the copyright notice and the licence text.

The program builds on the open-source projects below. They are installed from conda-forge or PyPI; only vtk.js and
gifenc are included in the repository (`app/web/static/vendor`, with their licences). The offline package
redistributes them unchanged, each under its own licence:

| Component | Licence | Use |
|---|---|---|
| NASA PuMA 3.2.2 | NASA Open Source Agreement 1.3 (licence included in the wheel) | FE conduction, elasticity, Stokes flow, tortuosity, ray casting |
| openEMS v0.37.0-rc3 | GPL-3.0 (CSXCAD LGPL-3.0), run as a separate program; source at [thliebig/openEMS-Project](https://github.com/thliebig/openEMS-Project) | EMI full-wave check |
| PoreSpy, OpenPNM | MIT | Morphology, porosimetry, pore network |
| Taichi | Apache-2.0 | Particle dynamics of the viscosity (CPU or CUDA GPU) |
| scikit-rf | BSD-3-Clause | EMI transmission line |
| VTK / PyVista | BSD-3-Clause / MIT | Surfaces, figures |
| vtk.js | BSD-3-Clause | 3D view in the browser |
| gifenc | MIT | Animated GIFs of the views, in the browser |
| NumPy, SciPy, numba, scikit-image, scikit-learn, pandas, Flask, psutil, tifffile, pyamg | BSD or MIT | Numerics, server |
| Matplotlib | Matplotlib licence (PSF-based) | Figures |
| Python 3.11.9 | PSF License | Runtime |
