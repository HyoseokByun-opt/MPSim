"""Body-fitted tetrahedral finite elements for the conduction-type properties (5.9.0).

The voxel solvers see a particle as the voxels whose centres it holds: its surface is a staircase, and a contact
between two particles is as wide as the voxels that happen to touch. Here the mesh follows the particles' exact
shapes instead:

* the lattice of the voxel corners is cut into tetrahedra (six per cube, Kuhn's subdivision), periodic like the RVE;
* every lattice node belongs to the particle that holds that point (the same exact point test the generator
  uses), and every tetrahedron edge whose ends lie in different particles (or in a particle and the matrix) is
  cut where it crosses the particle's surface, found by bisection on the exact shape;
* every lattice tetrahedron is divided by particle: a particle with one corner keeps a tetrahedron, with two or
  three a prism, split into three tetrahedra with the diagonals through the vertex of lowest rank (Dompierre et
  al. 1999), so that neighbours split a shared face alike and the mesh is conforming; what is left is matrix,
  fanned from its lowest matrix corner over the cut surfaces;
* an edge from one particle straight into another crosses a film of matrix thinner than the lattice (or none,
  where they overlap). The film is not meshed: the two particles meet at its middle, each with a vertex of its
  own, joined through the film's vertex by interface elements of conductance k_matrix / width per area - the
  lubrication limit of a thin gap (Batchelor & O'Brien 1977). Overlapping particles share the vertex (a direct
  contact);
* linear (P1) elements on that mesh, the periodic cell problem (or plates on a film's faces), solved by conjugate
  gradients preconditioned with smoothed-aggregation algebraic multigrid (PyAMG).

This is the conformal decomposition finite element method (Noble, Newren, Lechman 2010) on a regular background
mesh. On the same mesh FEniCSx gives the same numbers (self-test), since both solve the same P1 Galerkin system.

Reductions: interfaces are perfect except for the films above (an interface or contact resistance entered by the
user lives on voxel faces and stays with the finite-volume solver); a cut closer than DELTA of an edge to its end
is moved to DELTA (bounded element shapes); a film thinner than GAP_MIN of the lattice conducts as GAP_MIN.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit
from scipy import sparse

from .. import generate as GEN
from .. import shapes as SH

DELTA = 0.05                 # a cut stays this share of the edge away from its ends
GAP_MIN = 0.02               # the thinnest film carried, in lattice spacings (a closer contact conducts as this)
CHUNK = 1_500_000            # tetrahedra assembled at a time
_PERMS = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], np.int64)
# edge type from the offset between two corners, index dx*4 + dy*2 + dz
_ETYPE = np.array([-1, 2, 1, 5, 0, 4, 3, 6], np.int64)
_EOFF = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], np.int64)
# a prism (v0 v1 v2 | v3 v4 v5, edges v_i - v_i+3) turned so that a given vertex comes first
_PR = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4],
                [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]], np.int64)
_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)


# =========================================================================
# the mesh
# =========================================================================
@njit(cache=True)
def _inside(m, x, y, z, codes, centers, R, P, planes, nfaces, Lg, gper):
    dx = x - centers[m, 0]
    dy = y - centers[m, 1]
    dz = z - centers[m, 2]
    if gper[0]:
        dx -= Lg[0] * math.floor(dx / Lg[0] + 0.5)
    if gper[1]:
        dy -= Lg[1] * math.floor(dy / Lg[1] + 0.5)
    if gper[2]:
        dz -= Lg[2] * math.floor(dz / Lg[2] + 0.5)
    lx = R[m, 0, 0] * dx + R[m, 0, 1] * dy + R[m, 0, 2] * dz
    ly = R[m, 1, 0] * dx + R[m, 1, 1] * dy + R[m, 1, 2] * dz
    lz = R[m, 2, 0] * dx + R[m, 2, 1] * dy + R[m, 2, 2] * dz
    return SH.inside_local(codes[m], lx, ly, lz, P[m], planes, nfaces, 0.0)


@njit(cache=True)
def _exit(m, ax, ay, az, bx, by, bz, codes, centers, R, P, planes, nfaces, Lg, gper):
    """Where the segment a -> b leaves particle m (a inside): the share of the way, by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(18):
        t = 0.5 * (lo + hi)
        if _inside(m, ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az), codes, centers, R, P, planes,
                   nfaces, Lg, gper):
            lo = t
        else:
            hi = t
    return 0.5 * (lo + hi)


@njit(cache=True)
def _crossings(idx, n, per, h, owner, codes, centers, R, P, planes, nfaces, Lg, gper, delta, pts, tcut, gedge, film):
    """Where every cut edge crosses the particle surfaces. An edge from one particle to another leaves the first
    at t1 and enters the second at t2: its vertex sits half way; with t2 > t1 a film of matrix lies between
    (film[e] = 1, gedge[e] its width along the edge), else the two overlap (film[e] = 0, a direct contact)."""
    n0, n1, n2 = n[0], n[1], n[2]
    for e in range(idx.shape[0]):
        base = idx[e] // 7
        et = idx[e] % 7
        k = base % n2
        j = (base // n2) % n1
        i = base // (n1 * n2)
        ii, jj, kk = i + _EOFF[et, 0], j + _EOFF[et, 1], k + _EOFF[et, 2]
        ax, ay, az = i * h, j * h, k * h
        bx, by, bz = ii * h, jj * h, kk * h
        if per[0] and ii >= n0:
            ii -= n0
        if per[1] and jj >= n1:
            jj -= n1
        if per[2] and kk >= n2:
            kk -= n2
        oa = owner[base]
        ob = owner[(ii * n1 + jj) * n2 + kk]
        film[e] = 0
        gedge[e] = 0.0
        if oa > 0 and ob == 0:
            t = _exit(oa - 1, ax, ay, az, bx, by, bz, codes, centers, R, P, planes, nfaces, Lg, gper)
        elif oa == 0 and ob > 0:
            t = 1.0 - _exit(ob - 1, bx, by, bz, ax, ay, az, codes, centers, R, P, planes, nfaces, Lg, gper)
        else:
            t1 = _exit(oa - 1, ax, ay, az, bx, by, bz, codes, centers, R, P, planes, nfaces, Lg, gper)
            t2 = 1.0 - _exit(ob - 1, bx, by, bz, ax, ay, az, codes, centers, R, P, planes, nfaces, Lg, gper)
            t = 0.5 * (t1 + t2)
            if t2 > t1:
                film[e] = 1
                gedge[e] = (t2 - t1) * h * math.sqrt(float(_EOFF[et, 0] + _EOFF[et, 1] + _EOFF[et, 2]))
        if t < delta:
            t = delta
        elif t > 1.0 - delta:
            t = 1.0 - delta
        tcut[e] = t
        pts[e, 0] = ax + t * (bx - ax)
        pts[e, 1] = ay + t * (by - ay)
        pts[e, 2] = az + t * (bz - az)


@njit(cache=True)
def _emit(fill, tets, mat, cell, nt, a, b, c, d, m, ci):
    if fill:
        tets[nt, 0] = a
        tets[nt, 1] = b
        tets[nt, 2] = c
        tets[nt, 3] = d
        mat[nt] = m
        cell[nt] = ci
    return nt + 1


@njit(cache=True)
def _prism(fill, tets, mat, cell, nt, v, r, m, ci):
    """A prism (v0 v1 v2 | v3 v4 v5) as three tetrahedra, the diagonals through the vertex of lowest rank r."""
    s = 0
    for q in range(1, 6):
        if r[q] < r[s]:
            s = q
    w0, w1, w2 = v[_PR[s, 0]], v[_PR[s, 1]], v[_PR[s, 2]]
    w3, w4, w5 = v[_PR[s, 3]], v[_PR[s, 4]], v[_PR[s, 5]]
    if min(r[_PR[s, 1]], r[_PR[s, 5]]) < min(r[_PR[s, 2]], r[_PR[s, 4]]):
        nt = _emit(fill, tets, mat, cell, nt, w0, w1, w2, w5, m, ci)
        nt = _emit(fill, tets, mat, cell, nt, w0, w1, w5, w4, m, ci)
    else:
        nt = _emit(fill, tets, mat, cell, nt, w0, w1, w2, w4, m, ci)
        nt = _emit(fill, tets, mat, cell, nt, w0, w4, w2, w5, m, ci)
    return _emit(fill, tets, mat, cell, nt, w0, w4, w5, w3, m, ci)


@njit(cache=True)
def _tri_film(xa, xb, xc, ea, eb, ec, sa, sb, sc, da, db, dc, gedge, film, gmin, wside):
    """A triangle of a particle's cut surface: to each of its vertices that lie on a film, a third of its area
    over half the film's width normal to it (e: the cut edge, s: the side of the edge the particle is on, d:
    the edge's direction)."""
    ux, uy, uz = xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]
    vx, vy, vz = xc[0] - xa[0], xc[1] - xa[1], xc[2] - xa[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nrm <= 0.0:
        return
    area = 0.5 * nrm
    for q in range(3):
        e = ea if q == 0 else (eb if q == 1 else ec)
        if e < 0 or film[e] == 0:
            continue
        sd = sa if q == 0 else (sb if q == 1 else sc)
        d = da if q == 0 else (db if q == 1 else dc)
        dl = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        cosn = abs(nx * d[0] + ny * d[1] + nz * d[2]) / (nrm * dl)
        g = gedge[e] * cosn
        if g < gmin:
            g = gmin
        wside[e, sd] += 2.0 * (area / 3.0) / g


@njit(cache=True)
def _decompose(n, per, lab, owner, ecut, nn, h, tcut, gedge, film, gmin, fill, tets, mat, cell, wside):
    """The tetrahedra of the cut lattice (fill False: only counted). Returns (tetrahedra, lattice tetrahedra
    that hold two or more particles)."""
    n0, n1, n2 = n[0], n[1], n[2]
    c0 = n0 if per[0] else n0 - 1
    c1 = n1 if per[1] else n1 - 1
    c2 = n2 if per[2] else n2 - 1
    nt, nfilm = 0, 0
    v = np.empty(4, np.int64)
    o = np.empty(4, np.int64)
    lb = np.empty(4, np.int64)
    off = np.zeros((4, 3), np.int64)
    X = np.zeros((4, 3))                     # corners, unwrapped
    sid = np.empty((4, 4), np.int64)         # the cut vertex of edge (m, q) on m's side
    mid = np.empty((4, 4), np.int64)         # its matrix vertex
    rk = np.empty((4, 4), np.int64)          # its rank
    ed = np.empty((4, 4), np.int64)          # its cut edge
    cx = np.zeros((4, 4, 3))                 # its position
    dr = np.zeros((4, 4, 3))                 # the edge's direction
    pr = np.empty(6, np.int64)
    rr = np.empty(6, np.int64)
    xs = np.empty(4, np.int64)
    ys = np.empty(4, np.int64)
    pid = np.empty((8, 4), np.int64)
    prk = np.empty((8, 4), np.int64)
    pln = np.empty(8, np.int64)
    for i in range(c0):
        for j in range(c1):
            for k in range(c2):
                ci = (i * c1 + j) * c2 + k
                for p in range(6):
                    for a in range(3):
                        off[0, a] = 0
                    for m in range(1, 4):
                        for a in range(3):
                            off[m, a] = off[m - 1, a]
                        off[m, _PERMS[p, m - 1]] += 1
                    for m in range(4):
                        ii, jj, kk = i + off[m, 0], j + off[m, 1], k + off[m, 2]
                        X[m, 0], X[m, 1], X[m, 2] = ii * h, jj * h, kk * h
                        if ii >= n0:
                            ii -= n0
                        if jj >= n1:
                            jj -= n1
                        if kk >= n2:
                            kk -= n2
                        v[m] = (ii * n1 + jj) * n2 + kk
                        o[m] = owner[v[m]]
                        lb[m] = lab[v[m]]
                    if o[0] == o[1] and o[0] == o[2] and o[0] == o[3]:
                        nt = _emit(fill, tets, mat, cell, nt, v[0], v[1], v[2], v[3], lb[0], ci)
                        continue
                    for m in range(4):
                        for q in range(m + 1, 4):
                            if o[m] != o[q]:
                                et = _ETYPE[(off[q, 0] - off[m, 0]) * 4 + (off[q, 1] - off[m, 1]) * 2
                                            + (off[q, 2] - off[m, 2])]
                                e = ecut[v[m] * 7 + et]
                                ed[m, q] = e
                                rk[m, q] = nn + e
                                if film[e] == 1:
                                    sid[m, q] = nn + 3 * e
                                    mid[m, q] = nn + 3 * e + 1
                                    sid[q, m] = nn + 3 * e + 2
                                else:
                                    sid[m, q] = nn + 3 * e
                                    mid[m, q] = nn + 3 * e
                                    sid[q, m] = nn + 3 * e
                                t = tcut[e]
                                for a in range(3):
                                    cx[m, q, a] = X[m, a] + t * (X[q, a] - X[m, a])
                                    dr[m, q, a] = X[q, a] - X[m, a]
                            else:
                                ed[m, q] = -1
                                rk[m, q] = -1
                                sid[m, q] = -1
                                mid[m, q] = -1
                                sid[q, m] = -1
                            ed[q, m] = ed[m, q]
                            rk[q, m] = rk[m, q]
                            mid[q, m] = mid[m, q]
                            for a in range(3):
                                cx[q, m, a] = cx[m, q, a]
                                dr[q, m, a] = dr[m, q, a]
                    npoly = 0
                    ngroup = 0
                    nmat = 0
                    for m in range(4):
                        if o[m] == 0:
                            nmat += 1
                            continue
                        first = True
                        for q in range(m):
                            if o[q] == o[m]:
                                first = False
                        if not first:
                            continue
                        ngroup += 1
                        kx, ky = 0, 0
                        for q in range(4):
                            if o[q] == o[m]:
                                xs[kx] = q
                                kx += 1
                            else:
                                ys[ky] = q
                                ky += 1
                        if kx == 1:
                            x = xs[0]
                            y0, y1, y2 = ys[0], ys[1], ys[2]
                            nt = _emit(fill, tets, mat, cell, nt, v[x], sid[x, y0], sid[x, y1], sid[x, y2], lb[x], ci)
                            pid[npoly, 0], prk[npoly, 0] = mid[x, y0], rk[x, y0]
                            pid[npoly, 1], prk[npoly, 1] = mid[x, y1], rk[x, y1]
                            pid[npoly, 2], prk[npoly, 2] = mid[x, y2], rk[x, y2]
                            pln[npoly] = 3
                            if fill:
                                _tri_film(cx[x, y0], cx[x, y1], cx[x, y2], ed[x, y0], ed[x, y1], ed[x, y2],
                                          0 if x < y0 else 1, 0 if x < y1 else 1, 0 if x < y2 else 1,
                                          dr[x, y0], dr[x, y1], dr[x, y2], gedge, film, gmin, wside)
                        elif kx == 2:
                            x1, x2, y1, y2 = xs[0], xs[1], ys[0], ys[1]
                            pr[0], pr[1], pr[2] = v[x1], sid[x1, y1], sid[x1, y2]
                            pr[3], pr[4], pr[5] = v[x2], sid[x2, y1], sid[x2, y2]
                            rr[0], rr[1], rr[2] = v[x1], rk[x1, y1], rk[x1, y2]
                            rr[3], rr[4], rr[5] = v[x2], rk[x2, y1], rk[x2, y2]
                            nt = _prism(fill, tets, mat, cell, nt, pr, rr, lb[x1], ci)
                            # the cut surface, around: (x1,y1) (x1,y2) (x2,y2) (x2,y1)
                            pid[npoly, 0], prk[npoly, 0] = mid[x1, y1], rk[x1, y1]
                            pid[npoly, 1], prk[npoly, 1] = mid[x1, y2], rk[x1, y2]
                            pid[npoly, 2], prk[npoly, 2] = mid[x2, y2], rk[x2, y2]
                            pid[npoly, 3], prk[npoly, 3] = mid[x2, y1], rk[x2, y1]
                            pln[npoly] = 4
                            if fill:
                                s11 = 0 if x1 < y1 else 1
                                s12 = 0 if x1 < y2 else 1
                                s22 = 0 if x2 < y2 else 1
                                s21 = 0 if x2 < y1 else 1
                                if min(rk[x1, y1], rk[x2, y2]) < min(rk[x1, y2], rk[x2, y1]):
                                    _tri_film(cx[x1, y1], cx[x1, y2], cx[x2, y2], ed[x1, y1], ed[x1, y2], ed[x2, y2],
                                              s11, s12, s22, dr[x1, y1], dr[x1, y2], dr[x2, y2], gedge, film, gmin,
                                              wside)
                                    _tri_film(cx[x1, y1], cx[x2, y2], cx[x2, y1], ed[x1, y1], ed[x2, y2], ed[x2, y1],
                                              s11, s22, s21, dr[x1, y1], dr[x2, y2], dr[x2, y1], gedge, film, gmin,
                                              wside)
                                else:
                                    _tri_film(cx[x1, y2], cx[x2, y2], cx[x2, y1], ed[x1, y2], ed[x2, y2], ed[x2, y1],
                                              s12, s22, s21, dr[x1, y2], dr[x2, y2], dr[x2, y1], gedge, film, gmin,
                                              wside)
                                    _tri_film(cx[x1, y2], cx[x2, y1], cx[x1, y1], ed[x1, y2], ed[x2, y1], ed[x1, y1],
                                              s12, s21, s11, dr[x1, y2], dr[x2, y1], dr[x1, y1], gedge, film, gmin,
                                              wside)
                        else:
                            x1, x2, x3, y = xs[0], xs[1], xs[2], ys[0]
                            pr[0], pr[1], pr[2] = sid[x1, y], sid[x2, y], sid[x3, y]
                            pr[3], pr[4], pr[5] = v[x1], v[x2], v[x3]
                            rr[0], rr[1], rr[2] = rk[x1, y], rk[x2, y], rk[x3, y]
                            rr[3], rr[4], rr[5] = v[x1], v[x2], v[x3]
                            nt = _prism(fill, tets, mat, cell, nt, pr, rr, lb[x1], ci)
                            pid[npoly, 0], prk[npoly, 0] = mid[x1, y], rk[x1, y]
                            pid[npoly, 1], prk[npoly, 1] = mid[x2, y], rk[x2, y]
                            pid[npoly, 2], prk[npoly, 2] = mid[x3, y], rk[x3, y]
                            pln[npoly] = 3
                            if fill:
                                _tri_film(cx[x1, y], cx[x2, y], cx[x3, y], ed[x1, y], ed[x2, y], ed[x3, y],
                                          0 if x1 < y else 1, 0 if x2 < y else 1, 0 if x3 < y else 1,
                                          dr[x1, y], dr[x2, y], dr[x3, y], gedge, film, gmin, wside)
                        npoly += 1
                    if ngroup >= 2:
                        nfilm += 1
                    if nmat == 0 and ngroup < 3:
                        continue
                    # the matrix that is left: its hub
                    hub = -1
                    if nmat > 0:
                        for m in range(4):
                            if o[m] == 0 and (hub < 0 or v[m] < hub):
                                hub = v[m]
                    else:
                        hr = -1
                        for m in range(4):
                            for q in range(m + 1, 4):
                                if rk[m, q] >= 0 and (hr < 0 or rk[m, q] < hr):
                                    hr = rk[m, q]
                                    hub = mid[m, q]
                    # what the matrix keeps of the four faces
                    for f in range(4):
                        a, b, c = _FACES[f, 0], _FACES[f, 1], _FACES[f, 2]
                        fm = (1 if o[a] == 0 else 0) + (1 if o[b] == 0 else 0) + (1 if o[c] == 0 else 0)
                        if fm == 3:
                            pid[npoly, 0], pid[npoly, 1], pid[npoly, 2] = v[a], v[b], v[c]
                            prk[npoly, 0], prk[npoly, 1], prk[npoly, 2] = v[a], v[b], v[c]
                            pln[npoly] = 3
                            npoly += 1
                        elif fm == 2:
                            if o[a] != 0:
                                x, m1, m2 = a, b, c
                            elif o[b] != 0:
                                x, m1, m2 = b, a, c
                            else:
                                x, m1, m2 = c, a, b
                            pid[npoly, 0], prk[npoly, 0] = v[m1], v[m1]
                            pid[npoly, 1], prk[npoly, 1] = v[m2], v[m2]
                            pid[npoly, 2], prk[npoly, 2] = mid[x, m2], rk[x, m2]
                            pid[npoly, 3], prk[npoly, 3] = mid[x, m1], rk[x, m1]
                            pln[npoly] = 4
                            npoly += 1
                        elif fm == 1:
                            if o[a] == 0:
                                m1, x1, x2 = a, b, c
                            elif o[b] == 0:
                                m1, x1, x2 = b, a, c
                            else:
                                m1, x1, x2 = c, a, b
                            pid[npoly, 0], prk[npoly, 0] = v[m1], v[m1]
                            pid[npoly, 1], prk[npoly, 1] = mid[x1, m1], rk[x1, m1]
                            if o[x1] == o[x2]:
                                pid[npoly, 2], prk[npoly, 2] = mid[x2, m1], rk[x2, m1]
                                pln[npoly] = 3
                            else:
                                pid[npoly, 2], prk[npoly, 2] = mid[x1, x2], rk[x1, x2]
                                pid[npoly, 3], prk[npoly, 3] = mid[x2, m1], rk[x2, m1]
                                pln[npoly] = 4
                            npoly += 1
                        elif o[a] != o[b] and o[a] != o[c] and o[b] != o[c]:
                            pid[npoly, 0], prk[npoly, 0] = mid[a, b], rk[a, b]
                            pid[npoly, 1], prk[npoly, 1] = mid[b, c], rk[b, c]
                            pid[npoly, 2], prk[npoly, 2] = mid[a, c], rk[a, c]
                            pln[npoly] = 3
                            npoly += 1
                    for q in range(npoly):
                        skip = False
                        for a in range(pln[q]):
                            if pid[q, a] == hub:
                                skip = True
                        if skip:
                            continue
                        if pln[q] == 3:
                            nt = _emit(fill, tets, mat, cell, nt, hub, pid[q, 0], pid[q, 1], pid[q, 2], 0, ci)
                        elif min(prk[q, 0], prk[q, 2]) < min(prk[q, 1], prk[q, 3]):
                            nt = _emit(fill, tets, mat, cell, nt, hub, pid[q, 0], pid[q, 1], pid[q, 2], 0, ci)
                            nt = _emit(fill, tets, mat, cell, nt, hub, pid[q, 0], pid[q, 2], pid[q, 3], 0, ci)
                        else:
                            nt = _emit(fill, tets, mat, cell, nt, hub, pid[q, 1], pid[q, 2], pid[q, 3], 0, ci)
                            nt = _emit(fill, tets, mat, cell, nt, hub, pid[q, 1], pid[q, 3], pid[q, 0], 0, ci)
    return nt, nfilm


def supported(spec, ginfo, table):
    """None when the tetrahedral mesh can be made of this structure, else the reason it cannot."""
    parts = (ginfo or {}).get("particles") or {}
    if ginfo is None or ginfo.get("image"):
        return "a structure from an image has no particle shapes to mesh"
    if ginfo.get("shape") or ginfo.get("region_map") is not None or ginfo.get("pressed"):
        return "a compressed structure lives on its voxels"
    if any(t["kind"] == "shell" for t in table):
        return "coated and hollow particles are not meshed yet"
    if sum(t["kind"] == "matrix" for t in table) > 1:
        return "sections with a further matrix are not meshed yet"
    if any(ph["shape"] in ("network", "image") and float(ph.get("vf") or 0) > 0 for ph in spec["phases"]):
        return "a network phase has no particle shapes to mesh"
    if not len(parts.get("codes", [])):
        return "there are no particles to mesh"
    return None


def build_mesh(N, h, ginfo, gen_phases, mat_of_phase, nz=None, periodic=None, delta=DELTA, log=None):
    """The body-fitted mesh of a generated structure on the lattice of its voxel corners.

    N, h, nz: the voxel grid (nz: a film's voxels in z). mat_of_phase[i]: the material label of generator phase
    i (the matrix is label 0). periodic: per axis, whether the mesh wraps (default: x and y, and z unless a film).
    Returns {"points" µm, "tets", "mat", "cell" (the voxel a tetrahedron lies in), "springs" and "spring_w"
    (the films), "n" lattice nodes per axis, "cells", "per", "L" µm, "h", counts}."""
    t0 = time.time()
    film = bool(ginfo.get("film"))
    nzv = int(nz) if nz is not None else (int(round(ginfo["T_um"] / h)) if film else N)
    gper = np.array([True, True, not film])
    per = np.array(gper if periodic is None else periodic, np.bool_)
    # the particle at every lattice node: the particles drawn at the nodes (half a voxel from the voxel centres)
    owner, phase_of = GEN.rasterize_owner(N, h, ginfo, gen_phases, nz=(nzv + 1) if film else None, offset_um=-0.5 * h)
    for ax in range(3):
        if not per[ax] and gper[ax]:
            pad = [(0, 0)] * 3
            pad[ax] = (0, 1)
            owner = np.pad(owner, pad, mode="wrap")
    n = np.array(owner.shape, np.int64)
    lut = np.zeros(len(phase_of), np.int64)
    for q, ph in enumerate(phase_of):
        if ph >= 0:
            lut[q] = int(mat_of_phase[int(ph)])
    lut[0] = 0
    lab3 = lut[owner]
    nn = int(owner.size)
    # the cut edges: ends in different particles (or a particle and the matrix)
    idx = []
    for et in range(7):
        nb, ok = owner, np.ones(owner.shape, np.bool_)
        for ax in range(3):
            if _EOFF[et, ax]:
                nb = np.roll(nb, -1, axis=ax)
                if not per[ax]:
                    sl = [slice(None)] * 3
                    sl[ax] = -1
                    ok[tuple(sl)] = False
        cut = np.flatnonzero(((owner != nb) & ok).ravel())
        idx.append(cut * 7 + et)
    idx = np.sort(np.concatenate(idx)).astype(np.int64)
    ne = int(len(idx))
    ecut = np.full(nn * 7, -1, np.int32)
    ecut[idx] = np.arange(ne, dtype=np.int32)
    parts = ginfo["particles"]
    planes, nfaces = ginfo["planes"]
    Lg = np.array([N * h, N * h, nzv * h], float)
    cpt = np.zeros((max(ne, 1), 3))
    tcut = np.zeros(max(ne, 1))
    gedge = np.zeros(max(ne, 1))
    film_e = np.zeros(max(ne, 1), np.int8)
    lab = np.ascontiguousarray(lab3.ravel())
    own = np.ascontiguousarray(owner.ravel().astype(np.int64))
    if ne:
        _crossings(idx, n, per, float(h), own, parts["codes"].astype(np.int64),
                   np.ascontiguousarray(parts["centers"], float), np.ascontiguousarray(parts["R"], float),
                   np.ascontiguousarray(parts["P"], float), planes, nfaces, Lg, gper, float(delta), cpt, tcut, gedge,
                   film_e)
    gmin = float(GAP_MIN * h)
    wside = np.zeros((max(ne, 1), 2))
    nt, nfilm = _decompose(n, per, lab, own, ecut, nn, float(h), tcut, gedge, film_e, gmin, False,
                           np.zeros((1, 4), np.int64), np.zeros(1, np.uint8), np.zeros(1, np.int32), wside)
    tets = np.empty((nt, 4), np.int64)
    mat = np.empty(nt, np.uint8)
    cell = np.empty(nt, np.int32)
    _decompose(n, per, lab, own, ecut, nn, float(h), tcut, gedge, film_e, gmin, True, tets, mat, cell, wside)
    # the vertices in use, renumbered: the lattice nodes first, then up to three per cut edge
    used = np.zeros(nn + 3 * ne, np.bool_)
    used[:nn] = True
    used[tets.ravel()] = True
    new = np.cumsum(used) - 1
    tets = new[tets]
    g = np.meshgrid(np.arange(n[0]) * h, np.arange(n[1]) * h, np.arange(n[2]) * h, indexing="ij")
    cut_ids = np.flatnonzero(used[nn:])
    points = np.concatenate([np.stack([a.ravel() for a in g], axis=1), cpt[cut_ids // 3]])
    # the films: a spring between each particle's vertex and the film's vertex, per unit matrix conductivity
    fe = np.flatnonzero(film_e[:ne] == 1)
    if len(fe):
        fe = fe[used[nn + 3 * fe] & used[nn + 3 * fe + 2]]
    springs = np.zeros((2 * len(fe), 2), np.int64)
    sw = np.zeros(2 * len(fe))
    if len(fe):
        # a film vertex no matrix element uses is still the node the two springs meet at
        lone = ~used[nn + 3 * fe + 1]
        extra = np.arange(int(lone.sum())) + len(points)
        midv = np.where(lone, 0, new[nn + 3 * fe + 1])
        midv[lone] = extra
        points = np.concatenate([points, cpt[fe[lone]]])
        springs[0::2, 0], springs[0::2, 1] = new[nn + 3 * fe], midv
        springs[1::2, 0], springs[1::2, 1] = new[nn + 3 * fe + 2], midv
        sw[0::2], sw[1::2] = wside[fe, 0], wside[fe, 1]
    cells = np.array([n[a] if per[a] else n[a] - 1 for a in range(3)], np.int64)
    L = cells * float(h)
    clamped = int(np.count_nonzero((tcut[:ne] <= delta * (1 + 1e-9)) | (tcut[:ne] >= 1 - delta * (1 + 1e-9)))) if ne else 0
    mesh = {"points": points, "tets": tets, "mat": mat, "cell": cell, "n": n, "cells": cells, "per": per, "L": L,
            "h": float(h), "n_nodes": nn, "n_cut": ne, "n_film_edges": int(len(fe)), "n_film_tets": int(nfilm),
            "springs": springs, "spring_w": sw, "n_clamped": clamped, "seconds": time.time() - t0,
            "film_gap_um": (float(np.median(gedge[fe])) if len(fe) else 0.0)}
    if log:
        log(f"    tetrahedral mesh: {nt:,} tetrahedra on {len(points):,} vertices ({ne:,} edges cut at the particles' "
            f"surfaces; {len(fe):,} of them cross a film of matrix between two particles thinner than the lattice - "
            f"median {mesh['film_gap_um']:.3g} µm - carried as interface elements) ({mesh['seconds']:.1f} s)")
    return mesh


# =========================================================================
# P1 elements
# =========================================================================
@njit(cache=True)
def _grads(tets, t, points, per, L, e, g):
    """Gradients of the four hat functions of tetrahedron t into g; returns its volume."""
    v0 = tets[t, 0]
    for m in range(3):
        vm = tets[t, m + 1]
        for a in range(3):
            d = points[vm, a] - points[v0, a]
            if per[a]:
                d -= L[a] * math.floor(d / L[a] + 0.5)
            e[m, a] = d
    c0 = e[1, 1] * e[2, 2] - e[1, 2] * e[2, 1]
    c1 = e[1, 2] * e[2, 0] - e[1, 0] * e[2, 2]
    c2 = e[1, 0] * e[2, 1] - e[1, 1] * e[2, 0]
    det = e[0, 0] * c0 + e[0, 1] * c1 + e[0, 2] * c2
    vol = abs(det) / 6.0
    if vol <= 0.0:
        return 0.0
    inv = 1.0 / det
    g[1, 0], g[1, 1], g[1, 2] = c0 * inv, c1 * inv, c2 * inv
    g[2, 0] = (e[2, 1] * e[0, 2] - e[2, 2] * e[0, 1]) * inv
    g[2, 1] = (e[2, 2] * e[0, 0] - e[2, 0] * e[0, 2]) * inv
    g[2, 2] = (e[2, 0] * e[0, 1] - e[2, 1] * e[0, 0]) * inv
    g[3, 0] = (e[0, 1] * e[1, 2] - e[0, 2] * e[1, 1]) * inv
    g[3, 1] = (e[0, 2] * e[1, 0] - e[0, 0] * e[1, 2]) * inv
    g[3, 2] = (e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]) * inv
    for a in range(3):
        g[0, a] = -(g[1, a] + g[2, a] + g[3, a])
    return vol


@njit(cache=True)
def _elem(tets, points, kt, per, L, F, rows, cols, vals):
    """Element matrices of k grad(u).grad(v) as COO entries, and F[i, d] = -sum k V dphi_i/dx_d."""
    e = np.empty((3, 3))
    g = np.empty((4, 3))
    vol_sum = 0.0
    for t in range(tets.shape[0]):
        vol = _grads(tets, t, points, per, L, e, g)
        base = 16 * t
        if vol <= 0.0:
            for q in range(16):
                rows[base + q] = tets[t, 0]
                cols[base + q] = tets[t, 0]
                vals[base + q] = 0.0
            continue
        kv = kt[t] * vol
        vol_sum += vol
        for m in range(4):
            vm = tets[t, m]
            for a in range(3):
                F[vm, a] -= kv * g[m, a]
            for q in range(4):
                rows[base + 4 * m + q] = vm
                cols[base + 4 * m + q] = tets[t, q]
                vals[base + 4 * m + q] = kv * (g[m, 0] * g[q, 0] + g[m, 1] * g[q, 1] + g[m, 2] * g[q, 2])
    return vol_sum


@njit(cache=True)
def _post(tets, points, kt, mat, u, per, L, G, cell, qcell, E, V):
    """Volume integrals of k grad(T) with T = G.x + u: the total, per voxel, and the energy and volume per material."""
    e = np.empty((3, 3))
    g = np.empty((4, 3))
    qs = np.zeros(3)
    for t in range(tets.shape[0]):
        vol = _grads(tets, t, points, per, L, e, g)
        if vol <= 0.0:
            continue
        gx, gy, gz = G[0], G[1], G[2]
        for m in range(4):
            um = u[tets[t, m]]
            gx += um * g[m, 0]
            gy += um * g[m, 1]
            gz += um * g[m, 2]
        kv = kt[t] * vol
        qs[0] += kv * gx
        qs[1] += kv * gy
        qs[2] += kv * gz
        c = cell[t]
        qcell[c, 0] += kv * gx
        qcell[c, 1] += kv * gy
        qcell[c, 2] += kv * gz
        E[mat[t]] += kv * (gx * gx + gy * gy + gz * gz)
        V[mat[t]] += vol
    return qs


def assemble(mesh, vals):
    """The stiffness matrix (CSR), the load of a unit gradient along each axis, and the mesh volume."""
    tets, points = mesh["tets"], mesh["points"]
    nv = len(points)
    kt = np.asarray(vals, float)[mesh["mat"]]
    F = np.zeros((nv, 3))
    K = None
    vol = 0.0
    for a in range(0, len(tets), CHUNK):
        b = min(len(tets), a + CHUNK)
        rows = np.empty(16 * (b - a), np.int32 if nv < 2 ** 31 else np.int64)
        cols = np.empty_like(rows)
        v = np.empty(16 * (b - a))
        vol += _elem(tets[a:b], points, kt[a:b], mesh["per"], mesh["L"], F, rows, cols, v)
        Kc = sparse.coo_matrix((v, (rows, cols)), shape=(nv, nv)).tocsr()
        del rows, cols, v
        K = Kc if K is None else K + Kc
    sp, sw = mesh.get("springs"), mesh.get("spring_w")
    if sp is not None and len(sp):
        c = float(np.asarray(vals, float)[0]) * sw             # the matrix is label 0
        a, b = sp[:, 0], sp[:, 1]
        K = K + sparse.coo_matrix((np.concatenate([c, c, -c, -c]),
                                   (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), shape=(nv, nv)).tocsr()
    return K, F, vol


def _pcg(A, b, tol, maxiter, callback=None, should_stop=None, x0=None):
    """Conjugate gradients with smoothed-aggregation AMG (PyAMG); Jacobi where PyAMG is missing."""
    from scipy.sparse.linalg import LinearOperator, cg
    how = "AMG"
    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=400, keep=False)
        M = ml.aspreconditioner(cycle="V")
    except Exception:                                                # noqa: BLE001
        d = A.diagonal()
        d[d == 0] = 1.0
        M = LinearOperator(A.shape, matvec=lambda x: x / d)
        how = "Jacobi"
    bn = float(np.linalg.norm(b)) or 1.0
    st = {"it": 0}

    def cb(xk):
        st["it"] += 1
        if should_stop is not None and should_stop():
            raise InterruptedError
        if callback is not None and st["it"] % 5 == 0:
            callback(st["it"], float(np.linalg.norm(b - A @ xk)) / bn)
    try:
        x, info = cg(A, b, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
    except TypeError:                                                # SciPy before 1.12
        x, info = cg(A, b, x0=x0, tol=tol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
    res = float(np.linalg.norm(b - A @ x)) / bn
    return x, st["it"], res, info == 0, how


def conductivity(mesh, vals, direction, tol=1e-8, maxiter=3000, callback=None, should_stop=None, fields=False,
                 system=None, flux_h=None):
    """The effective conductivity column for a unit gradient along `direction` on the body-fitted mesh.

    A periodic axis carries the periodic cell problem (T = x_d + u, u periodic); an axis that does not wrap (a
    film's thickness) has plates at its two faces. system: (K, F, volume) of assemble(), reused over directions.
    Returns the keys of the finite-volume solver: column, k_eff, energy_frac (per material label), iterations,
    residual, converged, seconds, and with fields the potential ("u", absolute, on the voxels) and the flux "q"."""
    t0 = time.time()
    d = int(direction)
    K, F, vol = system if system is not None else assemble(mesh, vals)
    points, per, L = mesh["points"], mesh["per"], mesh["L"]
    nv = len(points)
    G = np.zeros(3)
    if per[d]:
        G[d] = 1.0
        u, it, res, ok, how = _pcg(K, F[:, d].copy(), tol, maxiter, callback, should_stop)
        u -= u.mean()
    else:
        x = points[:, d]
        eps = 1e-9 * L[d]
        fixed = (x < eps) | (x > L[d] - eps)
        Tb = np.where(fixed, x, 0.0)
        free = np.flatnonzero(~fixed)
        Kff = K[free][:, free].tocsr()
        rhs = -(K[free] @ Tb)
        uf, it, res, ok, how = _pcg(Kff, rhs, tol, maxiter, callback, should_stop, x0=x[free].copy())
        u = Tb.copy()
        u[free] = uf
    kt = np.asarray(vals, float)[mesh["mat"]]
    nlab = int(len(vals))
    ncell = int(np.prod(mesh["cells"]))
    qcell = np.zeros((ncell if fields else 1, 3))
    cell = mesh["cell"] if fields else np.zeros(len(mesh["tets"]), np.int32)
    E = np.zeros(nlab)
    V = np.zeros(nlab)
    qs = _post(mesh["tets"], points, kt, mesh["mat"], u, per, L, G, cell, qcell, E, V)
    e_film = 0.0
    sp = mesh.get("springs")
    if sp is not None and len(sp):
        e_film = float(np.sum(float(np.asarray(vals, float)[0]) * mesh["spring_w"] * (u[sp[:, 0]] - u[sp[:, 1]]) ** 2))
        E[0] += e_film
    col = qs / vol
    out = {"column": col, "k_eff": float(col[d]), "energy_frac": (E / max(E.sum(), 1e-300)).tolist(),
           "energy_frac_interface": float(e_film / max(E.sum(), 1e-300)),
           "hill_mandel_gap": float(abs(E.sum() / vol - col[d]) / max(abs(col[d]), 1e-300)),
           "iterations": int(it), "residual": float(res), "converged": bool(ok), "target": float(tol),
           "seconds": time.time() - t0, "method": "tet-fe" + ("" if per[d] else "-plates"), "preconditioner": how,
           "vf_mesh": (V / max(V.sum(), 1e-300)).tolist(), "n_tets": int(len(mesh["tets"])), "n_dof": int(nv)}
    if fields:
        n = mesh["n"]
        cells = mesh["cells"]
        nn = mesh["n_nodes"]
        # the deviation from the uniform gradient at the lattice nodes, averaged to the voxel centres
        w = ((u[:nn] - (0.0 if per[d] else points[:nn, d])) / L[d]).reshape(tuple(n))
        acc = np.zeros(tuple(cells))
        for o in range(8):
            a = w
            for ax in range(3):
                if (o >> ax) & 1:
                    a = np.roll(a, -1, axis=ax)
            acc += a[:cells[0], :cells[1], :cells[2]]
        shape = [1, 1, 1]
        shape[d] = int(cells[d])
        ramp = ((np.arange(cells[d]) + 0.5) / cells[d]).reshape(shape)
        out["u"] = ramp + acc / 8.0
        out["absolute_potential"] = True
        fh = float(flux_h) if flux_h else mesh["h"] * 1e-6
        out["q"] = (-qcell / mesh["h"] ** 3 / (cells[d] * fh)).reshape(tuple(cells) + (3,))
    return out


def write_fenics(mesh, path):
    """The mesh as an .npz for the FEniCSx check (solvers/fenicsx_tet.py): points, tetrahedra, materials."""
    np.savez_compressed(path, points=mesh["points"], tets=mesh["tets"], mat=mesh["mat"], L=mesh["L"],
                        springs=mesh["springs"], spring_w=mesh["spring_w"])
