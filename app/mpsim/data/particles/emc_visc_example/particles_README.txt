Particles of the structure (90 particles), as MPSim built them.

Box: 48 x 48 x 48 um (x, y, z), from the origin; periodic in x, y and z (a particle may cross a face and continue on the opposite one). Lengths in micrometres.
Voxel: 0.5 um, the grid the run drew them on (Structure - List (CSV) reads this file and draws them again).
The voxel structure (structure.tif) samples these particles at the voxel centres ((i + 0.5) h, (j + 0.5) h, (k + 0.5) h):
a voxel belongs to a particle when its centre lies inside it.

particles.csv - one row per particle
  x_um, y_um, z_um   the centre
  d_um               the size d of the shape (sphere: diameter; spheroid: 2a along its local x; cylinder and
                     capsule: diameter; cube and cuboid: edge along local x; superellipsoid: 2a along local x;
                     polyhedron: circumscribed diameter)
  eqd_um, volume_um3 the volume-equivalent diameter and the volume
  lx_um, ly_um, lz_um  the extent along the particle's own axes x, y, z (spheroid and superellipsoid: the axes
                     2a, 2b, 2c; cuboid: the edges; cylinder: diameter, diameter, length; capsule: the same,
                     with the hemispherical ends in the length; polyhedron: its circumscribed diameter)
  length_um          cylinder and capsule: the length along local z (capsule: total, ends included)
  aspect_c_a, aspect_b_a  spheroid: c/a (local z) and b/a (local y)
  n_exponent         superellipsoid: |x/a|^n + |y/b|^n + |z/c|^n <= 1
  variant            polyhedron: its shape in polyhedra.csv
  shell_um           a coating's thickness (the sizes above include it)
  r11 ... r33        the rotation: row 1 (r11 r12 r13) is the particle's local x axis in the box, row 2 its y
                     axis, row 3 its z axis. A point (u, v, w) of the particle's own frame lies at
                     centre + u * row1 + v * row2 + w * row3.

polyhedra.csv - the vertices of every polyhedron shape in its own frame at circumradius 1: multiply by d_um / 2
and place them with the rotation and the centre above; the particle is their convex hull.

psa.csv - the size distributions per phase and dimension in bins: the realised (the particles above) and the
input (what the entered distribution draws, 20,000 particles sampled as the generator samples them), as number %
and volume % per bin and cumulative.
psa_summary.csv - their statistics: number mean, standard deviation, CV, D10/D50/D90 by number, the volume-weighted
mean (D[4,3] for spheres) and D10/D50/D90 by volume.
