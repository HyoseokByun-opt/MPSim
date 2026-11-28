"""A graded body-fitted tetrahedral mesh (5.10.0): fine where the particles are fine, coarse where they are not.

The mesh of tetfem.build_mesh has one lattice node per voxel corner everywhere, so a blend of sizes pays the
resolution of its finest particle through the whole RVE. Here the background mesh is an octree:

* the RVE is tiled with cubes of the coarsest size; a cube is divided in eight as long as the surface of a particle
  runs through it and the cube is larger than that particle's cell size (its smallest dimension over `res`), or two
  particles meet in it (one level finer still: the contact), so that the cells are as small as the particle they
  describe and no smaller;
* neighbours across a face or an edge differ by one level at most (2:1 balance);
* every cube becomes tetrahedra from its centre to its faces - a face against smaller cubes as its four quarters,
  a face with a neighbour's corner on an edge fanned from its centre, any other split by the diagonal from its
  low corner to its high one - which makes the mesh conforming without hanging nodes; a cube whose faces and
  edges carry no further node is Kuhn's six tetrahedra without a centre, so a uniform region is the lattice
  mesh of tetfem;
* that mesh is then cut at the particles' exact surfaces exactly as the lattice is (tetfem._cut_tet: conformal
  decomposition, films of matrix between two particles as interface elements).

The finest cell is the voxel halved `refine` times, so the mesh can describe particles finer than the voxel grid
resolves; the coarsest is the voxel doubled as often as the grid divides.

Reductions: a particle is found by the points a cube is sampled at (3 x 3 x 3) once the cube is smaller than the
particle's smallest dimension - a tip that reaches into a cube between those points is cut by the neighbours only.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict

from .. import generate as GEN
from .. import shapes as SH
from . import tetfem as TF

B = 1 << 14                  # leaf coordinates per axis (finest cells)
B2 = 1 << 16                 # node coordinates per axis (half finest cells)
_D18 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                 [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                 [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]], np.int64)


@njit(cache=True)
def _lkey(i, j, k, l):
    return ((l * B + i) * B + j) * B + k


@njit(cache=True)
def _nkey(i, j, k):
    return (i * B2 + j) * B2 + k


@njit(cache=True)
def _grow4(a, n):
    if n < a.shape[0]:
        return a
    b = np.empty((2 * a.shape[0], a.shape[1]), a.dtype)
    b[:n] = a[:n]
    return b


@njit(cache=True)
def _extents(codes, R, P, planes, nfaces, ext):
    for m in range(codes.shape[0]):
        hx, hy, hz = SH.local_half_extents(codes[m], P[m], planes, nfaces)
        for a in range(3):
            ext[m, a] = abs(R[m, 0, a]) * hx + abs(R[m, 1, a]) * hy + abs(R[m, 2, a]) * hz


@njit(cache=True)
def _root_lists(nroot, S, centers, ext, gper, fill, ptr, idx):
    """The particles whose box reaches each root cube (cubes of S µm): counted into ptr, or filled into idx."""
    n0, n1, n2 = nroot[0], nroot[1], nroot[2]
    for m in range(centers.shape[0]):
        lo = np.empty(3, np.int64)
        hi = np.empty(3, np.int64)
        for a in range(3):
            lo[a] = int(math.floor((centers[m, a] - ext[m, a]) / S)) - 1
            hi[a] = int(math.floor((centers[m, a] + ext[m, a]) / S)) + 1
            if hi[a] - lo[a] + 1 > nroot[a]:
                lo[a], hi[a] = 0, nroot[a] - 1
        for i in range(lo[0], hi[0] + 1):
            ii = i % n0 if gper[0] else i
            if ii < 0 or ii >= n0:
                continue
            for j in range(lo[1], hi[1] + 1):
                jj = j % n1 if gper[1] else j
                if jj < 0 or jj >= n1:
                    continue
                for k in range(lo[2], hi[2] + 1):
                    kk = k % n2 if gper[2] else k
                    if kk < 0 or kk >= n2:
                        continue
                    c = (ii * n1 + jj) * n2 + kk
                    if fill:
                        idx[ptr[c]] = m
                    ptr[c] += 1


@njit(cache=True)
def _near(m, x, y, z, rc, centers, R, P, codes, planes, nfaces, Lg, gper):
    """Whether a ball of radius rc at (x, y, z) reaches the particle's box (in its own frame)."""
    dx = x - centers[m, 0]
    dy = y - centers[m, 1]
    dz = z - centers[m, 2]
    if gper[0]:
        dx -= Lg[0] * math.floor(dx / Lg[0] + 0.5)
    if gper[1]:
        dy -= Lg[1] * math.floor(dy / Lg[1] + 0.5)
    if gper[2]:
        dz -= Lg[2] * math.floor(dz / Lg[2] + 0.5)
    hx, hy, hz = SH.local_half_extents(codes[m], P[m], planes, nfaces)
    if abs(R[m, 0, 0] * dx + R[m, 0, 1] * dy + R[m, 0, 2] * dz) > hx + rc:
        return False
    if abs(R[m, 1, 0] * dx + R[m, 1, 1] * dy + R[m, 1, 2] * dz) > hy + rc:
        return False
    return abs(R[m, 2, 0] * dx + R[m, 2, 1] * dy + R[m, 2, 2] * dz) <= hz + rc


@njit(cache=True)
def _octree(nroot, lmax, s0, codes, centers, R, P, planes, nfaces, Lg, gper, lev, wmin, cext, ptr, idx, nk, grow):
    """The leaves (i, j, k, level) the particles ask for (not yet balanced)."""
    S = 1 << lmax
    maxc = 1
    for c in range(ptr.shape[0] - 1):
        if ptr[c + 1] - ptr[c] > maxc:
            maxc = ptr[c + 1] - ptr[c]
    buf = np.empty(8 * (lmax + 2) * maxc + 64, np.int64)
    stack = np.empty((8 * (lmax + 2) + 8, 6), np.int64)
    leaves = np.empty((4096, 4), np.int32)
    nl = 0
    for ri in range(nroot[0]):
        for rj in range(nroot[1]):
            for rk in range(nroot[2]):
                c = (ri * nroot[1] + rj) * nroot[2] + rk
                n = ptr[c + 1] - ptr[c]
                for q in range(n):
                    buf[q] = idx[ptr[c] + q]
                stack[0, 0], stack[0, 1], stack[0, 2], stack[0, 3] = ri * S, rj * S, rk * S, lmax
                stack[0, 4], stack[0, 5] = 0, n
                sp = 1
                while sp > 0:
                    sp -= 1
                    i, j, k, l = stack[sp, 0], stack[sp, 1], stack[sp, 2], stack[sp, 3]
                    st, ln = stack[sp, 4], stack[sp, 5]
                    top = st + ln
                    sz = (1 << l) * s0
                    x, y, z = (i + 0.5 * (1 << l)) * s0, (j + 0.5 * (1 << l)) * s0, (k + 0.5 * (1 << l)) * s0
                    rc = 0.8660254037844386 * sz
                    need = False
                    ncut = 0
                    cutlev = 99
                    if l > 0:
                        for q in range(ln):
                            m = buf[st + q]
                            lp = lev[m]
                            lc = lp - cext if lp > cext else 0
                            if not _near(m, x, y, z, rc + grow, centers, R, P, codes, planes, nfaces, Lg, gper):
                                continue
                            nin = 0
                            for a in range(-1, 2):
                                for b in range(-1, 2):
                                    for cc in range(-1, 2):
                                        if TF._inside(m, x + 0.5 * a * sz, y + 0.5 * b * sz, z + 0.5 * cc * sz, codes,
                                                      centers, R, P, planes, nfaces, Lg, gper, nk):
                                            nin += 1
                            if nin == 27:
                                continue
                            if nin == 0:
                                if sz > 1.0001 * wmin[m] and l > lp:
                                    need = True
                                    break
                                continue
                            ncut += 1
                            if lc < cutlev:
                                cutlev = lc
                            if l > lp or (ncut >= 2 and l > cutlev):
                                need = True
                                break
                    if not need:
                        leaves = _grow4(leaves, nl)
                        leaves[nl, 0], leaves[nl, 1], leaves[nl, 2], leaves[nl, 3] = i, j, k, l
                        nl += 1
                        continue
                    h2 = 1 << (l - 1)
                    rc2 = 0.5 * rc
                    for o in range(8):
                        ci, cj, ck = i + (o & 1) * h2, j + ((o >> 1) & 1) * h2, k + ((o >> 2) & 1) * h2
                        cx, cy, cz = (ci + 0.5 * h2) * s0, (cj + 0.5 * h2) * s0, (ck + 0.5 * h2) * s0
                        n = 0
                        for q in range(ln):
                            m = buf[st + q]
                            if _near(m, cx, cy, cz, rc2 + grow, centers, R, P, codes, planes, nfaces, Lg, gper):
                                buf[top + n] = m
                                n += 1
                        stack[sp, 0], stack[sp, 1], stack[sp, 2], stack[sp, 3] = ci, cj, ck, l - 1
                        stack[sp, 4], stack[sp, 5] = top, n
                        top += n
                        sp += 1
    return leaves[:nl]


@njit(cache=True)
def _balance(leaves, nf, per, lmax):
    """2:1 balance across faces and edges; returns the leaves (i, j, k, level)."""
    d = Dict.empty(types.int64, types.int64)
    cap = max(2 * leaves.shape[0], 1024)
    lv = np.empty((cap, 4), np.int32)
    lv[:leaves.shape[0]] = leaves
    alive = np.ones(cap, np.bool_)
    n = leaves.shape[0]
    for q in range(n):
        d[_lkey(lv[q, 0], lv[q, 1], lv[q, 2], lv[q, 3])] = q
    q = 0
    c = np.empty(3, np.int64)
    while q < n:
        if not alive[q]:
            q += 1
            continue
        l = lv[q, 3]
        S = 1 << l
        for t in range(18):
            ok = True
            for a in range(3):
                c[a] = lv[q, a] + _D18[t, a] * S
                if c[a] < 0 or c[a] >= nf[a]:
                    if per[a]:
                        c[a] %= nf[a]
                    else:
                        ok = False
            if not ok:
                continue
            for m in range(lmax, l + 1, -1):
                M = 1 << m
                key = _lkey((c[0] // M) * M, (c[1] // M) * M, (c[2] // M) * M, m)
                if key in d:
                    p = d[key]
                    del d[key]
                    alive[p] = False
                    h2 = M >> 1
                    if n + 8 > lv.shape[0]:
                        lv2 = np.empty((2 * lv.shape[0], 4), np.int32)
                        lv2[:n] = lv[:n]
                        lv = lv2
                        al2 = np.ones(2 * alive.shape[0], np.bool_)
                        al2[:n] = alive[:n]
                        alive = al2
                    for o in range(8):
                        lv[n, 0] = lv[p, 0] + (o & 1) * h2
                        lv[n, 1] = lv[p, 1] + ((o >> 1) & 1) * h2
                        lv[n, 2] = lv[p, 2] + ((o >> 2) & 1) * h2
                        lv[n, 3] = m - 1
                        d[_lkey(lv[n, 0], lv[n, 1], lv[n, 2], m - 1)] = n
                        n += 1
        q += 1
    return lv[:n][alive[:n]]


@njit(cache=True)
def _node(nd, xyz, nn, i, j, k, n2, per):
    """The id of the node at half-cell coordinates (i, j, k), made when new. Returns (id, nodes, array)."""
    if per[0]:
        i %= n2[0]
    if per[1]:
        j %= n2[1]
    if per[2]:
        k %= n2[2]
    key = _nkey(i, j, k)
    if key in nd:
        return nd[key], nn, xyz
    nd[key] = nn
    xyz = _grow4(xyz, nn)
    xyz[nn, 0], xyz[nn, 1], xyz[nn, 2] = i, j, k
    return nn, nn + 1, xyz


@njit(cache=True)
def _has(nd, i, j, k, n2, per):
    if per[0]:
        i %= n2[0]
    if per[1]:
        j %= n2[1]
    if per[2]:
        k %= n2[2]
    return _nkey(i, j, k) in nd


@njit(cache=True)
def _tets(leaves, nf, per):
    """Nodes (half-cell coordinates) and tetrahedra of the balanced octree."""
    n2 = 2 * nf
    d = Dict.empty(types.int64, types.int64)
    for q in range(leaves.shape[0]):
        d[_lkey(leaves[q, 0], leaves[q, 1], leaves[q, 2], leaves[q, 3])] = q
    nd = Dict.empty(types.int64, types.int64)
    xyz = np.empty((max(4 * leaves.shape[0], 1024), 3), np.int32)
    nn = 0
    for q in range(leaves.shape[0]):
        S2 = 2 << leaves[q, 3]
        for o in range(8):
            _, nn, xyz = _node(nd, xyz, nn, 2 * leaves[q, 0] + (o & 1) * S2, 2 * leaves[q, 1] + ((o >> 1) & 1) * S2,
                               2 * leaves[q, 2] + ((o >> 2) & 1) * S2, n2, per)
    tets = np.empty((max(16 * leaves.shape[0], 1024), 4), np.int64)
    nt = 0
    p = np.empty(3, np.int64)
    c = np.empty(3, np.int64)
    w = np.empty(3, np.int64)
    ring = np.empty(8, np.int64)
    cu = np.array([0, 1, 1, 0], np.int64)
    cv = np.array([0, 0, 1, 1], np.int64)
    for q in range(leaves.shape[0]):
        l = leaves[q, 3]
        S = 1 << l
        S2 = 2 * S
        for a in range(3):
            p[a] = 2 * leaves[q, a]
        # a cube whose faces and edges carry no further node: Kuhn's six tetrahedra, no centre node
        plain = True
        if l > 0:
            for ax in range(3):
                u, v = (ax + 1) % 3, (ax + 2) % 3
                for e in range(4):
                    c[ax] = p[ax] + S
                    c[u] = p[u] + (e & 1) * S2
                    c[v] = p[v] + ((e >> 1) & 1) * S2
                    if _has(nd, c[0], c[1], c[2], n2, per):
                        plain = False
        if plain:
            if nt + 8 > tets.shape[0]:
                t2 = np.empty((2 * tets.shape[0], 4), np.int64)
                t2[:nt] = tets[:nt]
                tets = t2
            for pm in range(6):
                for a in range(3):
                    c[a] = p[a]
                tets[nt, 0] = nd[_nkey(c[0] % n2[0] if per[0] else c[0], c[1] % n2[1] if per[1] else c[1],
                                       c[2] % n2[2] if per[2] else c[2])]
                for m in range(3):
                    c[TF._PERMS[pm, m]] += S2
                    tets[nt, m + 1] = nd[_nkey(c[0] % n2[0] if per[0] else c[0], c[1] % n2[1] if per[1] else c[1],
                                               c[2] % n2[2] if per[2] else c[2])]
                nt += 1
            continue
        ctr, nn, xyz = _node(nd, xyz, nn, p[0] + S, p[1] + S, p[2] + S, n2, per)
        for ax in range(3):
            u, v = (ax + 1) % 3, (ax + 2) % 3
            for side in range(2):
                # the neighbour across this face: the same size, larger, or smaller (then the face is four squares)
                finer = False
                for a in range(3):
                    c[a] = leaves[q, a]
                c[ax] += S if side == 1 else -S
                inside = True
                if c[ax] < 0 or c[ax] >= nf[ax]:
                    if per[ax]:
                        c[ax] %= nf[ax]
                    else:
                        inside = False
                if inside and _lkey(c[0], c[1], c[2], l) not in d:
                    M = 2 * S
                    if l == 0 or _lkey((c[0] // M) * M, (c[1] // M) * M, (c[2] // M) * M, l + 1) in d:
                        finer = False
                    else:
                        finer = True
                nsq = 2 if finer else 1
                z = S if finer else S2
                for su in range(nsq):
                    for sv in range(nsq):
                        w[ax] = p[ax] + (S2 if side == 1 else 0)
                        w[u] = p[u] + su * z
                        w[v] = p[v] + sv * z
                        # the ring: corners, and a neighbour's corner at the middle of an edge
                        nr = 0
                        hang = False
                        for e in range(4):
                            c[ax] = w[ax]
                            c[u] = w[u] + cu[e] * z
                            c[v] = w[v] + cv[e] * z
                            ring[nr], nn, xyz = _node(nd, xyz, nn, c[0], c[1], c[2], n2, per)
                            nr += 1
                            e2 = (e + 1) % 4
                            c[u] = w[u] + (cu[e] + cu[e2]) * z // 2
                            c[v] = w[v] + (cv[e] + cv[e2]) * z // 2
                            if z >= 2 and _has(nd, c[0], c[1], c[2], n2, per):
                                ring[nr] = nd[_nkey(c[0] % n2[0] if per[0] else c[0], c[1] % n2[1] if per[1] else c[1],
                                                    c[2] % n2[2] if per[2] else c[2])]
                                nr += 1
                                hang = True
                        if nt + 8 > tets.shape[0]:
                            t2 = np.empty((2 * tets.shape[0], 4), np.int64)
                            t2[:nt] = tets[:nt]
                            tets = t2
                        if hang:
                            c[ax] = w[ax]
                            c[u] = w[u] + z // 2
                            c[v] = w[v] + z // 2
                            fc, nn, xyz = _node(nd, xyz, nn, c[0], c[1], c[2], n2, per)
                            for e in range(nr):
                                tets[nt, 0], tets[nt, 1], tets[nt, 2], tets[nt, 3] = ctr, fc, ring[e], ring[(e + 1) % nr]
                                nt += 1
                        else:
                            a0, a1, a2, a3 = ring[0], ring[1], ring[2], ring[3]
                            tets[nt, 0], tets[nt, 1], tets[nt, 2], tets[nt, 3] = ctr, a0, a1, a2
                            nt += 1
                            tets[nt, 0], tets[nt, 1], tets[nt, 2], tets[nt, 3] = ctr, a0, a2, a3
                            nt += 1
    return xyz[:nn], tets[:nt]


@njit(cache=True)
def _owners(points, nbin, Lg, gper, codes, centers, R, P, planes, nfaces, ext, owner, nk):
    """owner[v]: the lowest particle (1-based) that holds node v, 0 in the matrix. Nodes binned on a uniform grid."""
    nv = points.shape[0]
    bs = np.empty(3)
    for a in range(3):
        bs[a] = Lg[a] / nbin[a]
    cnt = np.zeros(nbin[0] * nbin[1] * nbin[2] + 1, np.int64)
    bi = np.empty(nv, np.int64)
    for v in range(nv):
        i = min(max(int(points[v, 0] / bs[0]), 0), nbin[0] - 1)
        j = min(max(int(points[v, 1] / bs[1]), 0), nbin[1] - 1)
        k = min(max(int(points[v, 2] / bs[2]), 0), nbin[2] - 1)
        bi[v] = (i * nbin[1] + j) * nbin[2] + k
        cnt[bi[v] + 1] += 1
    for c in range(1, cnt.shape[0]):
        cnt[c] += cnt[c - 1]
    order = np.empty(nv, np.int64)
    fillp = cnt[:-1].copy()
    for v in range(nv):
        order[fillp[bi[v]]] = v
        fillp[bi[v]] += 1
    lo = np.empty(3, np.int64)
    hi = np.empty(3, np.int64)
    for m in range(codes.shape[0]):
        for a in range(3):
            lo[a] = int(math.floor((centers[m, a] - ext[m, a]) / bs[a]))
            hi[a] = int(math.floor((centers[m, a] + ext[m, a]) / bs[a]))
            if hi[a] - lo[a] + 1 > nbin[a]:
                lo[a], hi[a] = 0, nbin[a] - 1
        for i in range(lo[0], hi[0] + 1):
            ii = i % nbin[0] if gper[0] else i
            if ii < 0 or ii >= nbin[0]:
                continue
            for j in range(lo[1], hi[1] + 1):
                jj = j % nbin[1] if gper[1] else j
                if jj < 0 or jj >= nbin[1]:
                    continue
                for k in range(lo[2], hi[2] + 1):
                    kk = k % nbin[2] if gper[2] else k
                    if kk < 0 or kk >= nbin[2]:
                        continue
                    c = (ii * nbin[1] + jj) * nbin[2] + kk
                    for s in range(cnt[c], cnt[c + 1]):
                        v = order[s]
                        if owner[v] != 0 and owner[v] <= m + 1:
                            continue
                        if TF._inside(m, points[v, 0], points[v, 1], points[v, 2], codes, centers, R, P, planes, nfaces,
                                      Lg, gper, nk):
                            owner[v] = m + 1


@njit(cache=True)
def _cross(ea, eb, points, per, L, owner, codes, centers, R, P, planes, nfaces, Lg, gper, delta, pts, tcut, gedge,
           film, nk):
    """tetfem._crossings for edges given by their end nodes (ea -> eb)."""
    for e in range(ea.shape[0]):
        ax, ay, az = points[ea[e], 0], points[ea[e], 1], points[ea[e], 2]
        dx, dy, dz = points[eb[e], 0] - ax, points[eb[e], 1] - ay, points[eb[e], 2] - az
        if per[0]:
            dx -= L[0] * math.floor(dx / L[0] + 0.5)
        if per[1]:
            dy -= L[1] * math.floor(dy / L[1] + 0.5)
        if per[2]:
            dz -= L[2] * math.floor(dz / L[2] + 0.5)
        bx, by, bz = ax + dx, ay + dy, az + dz
        oa, ob = owner[ea[e]], owner[eb[e]]
        film[e] = 0
        gedge[e] = 0.0
        if oa > 0 and ob == 0:
            t = TF._exit(oa - 1, ax, ay, az, bx, by, bz, codes, centers, R, P, planes, nfaces, Lg, gper, nk)
        elif oa == 0 and ob > 0:
            t = 1.0 - TF._exit(ob - 1, bx, by, bz, ax, ay, az, codes, centers, R, P, planes, nfaces, Lg, gper, nk)
        else:
            t1 = TF._exit(oa - 1, ax, ay, az, bx, by, bz, codes, centers, R, P, planes, nfaces, Lg, gper, nk)
            t2 = 1.0 - TF._exit(ob - 1, bx, by, bz, ax, ay, az, codes, centers, R, P, planes, nfaces, Lg, gper, nk)
            t = 0.5 * (t1 + t2)
            # signed: negative where the two overlap (read by the separated treatment of overlaps)
            gedge[e] = (t2 - t1) * math.sqrt(dx * dx + dy * dy + dz * dz)
            if t2 > t1:
                film[e] = 1
        if t < delta:
            t = delta
        elif t > 1.0 - delta:
            t = 1.0 - delta
        tcut[e] = t
        pts[e, 0] = ax + t * dx
        pts[e, 1] = ay + t * dy
        pts[e, 2] = az + t * dz


@njit(cache=True)
def _decompose(bt, te, points, per, L, owner, lab, nn, tcut, gedge, film, gmin, fill, tets, mat, cell, wside, wn, aside):
    """The cut of the background tetrahedra bt (corners in rising order; te: their six cut edges or -1, in the
    order (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)). Returns (tetrahedra, tetrahedra holding two or more particles)."""
    nt, nfilm = 0, 0
    v = np.empty(4, np.int64)
    o = np.empty(4, np.int64)
    lb = np.empty(4, np.int64)
    X = np.zeros((4, 3))
    sid = np.empty((4, 4), np.int64)
    mid = np.empty((4, 4), np.int64)
    rk = np.empty((4, 4), np.int64)
    ed = np.empty((4, 4), np.int64)
    cx = np.zeros((4, 4, 3))
    dr = np.zeros((4, 4, 3))
    pr = np.empty(6, np.int64)
    rr = np.empty(6, np.int64)
    xs = np.empty(4, np.int64)
    ys = np.empty(4, np.int64)
    pid = np.empty((8, 4), np.int64)
    prk = np.empty((8, 4), np.int64)
    pln = np.empty(8, np.int64)
    for t in range(bt.shape[0]):
        for m in range(4):
            v[m] = bt[t, m]
            o[m] = owner[v[m]]
            lb[m] = lab[v[m]]
        for a in range(3):
            X[0, a] = points[v[0], a]
        for m in range(1, 4):
            for a in range(3):
                dd = points[v[m], a] - X[0, a]
                if per[a]:
                    dd -= L[a] * math.floor(dd / L[a] + 0.5)
                X[m, a] = X[0, a] + dd
        q = 0
        for m in range(4):
            for m2 in range(m + 1, 4):
                ed[m, m2] = te[t, q]
                ed[m2, m] = te[t, q]
                q += 1
        nt, ng = TF._cut_tet(fill, tets, mat, cell, nt, t, v, X, o, lb, ed, nn, tcut, gedge, film, gmin, wside, wn,
                             sid, mid, rk, cx, dr, pr, rr, xs, ys, pid, prk, pln, aside)
        if ng >= 2:
            nfilm += 1
    return nt, nfilm


def plan(N, h, ginfo, nz=None, res=4.0, refine=None, max_refine=3, coarsest=None):
    """The levels of the graded mesh: (finest cell µm, halvings of the voxel, coarsest level, level per particle,
    smallest dimension per particle)."""
    parts = ginfo["particles"]
    meta = ginfo.get("poly_meta")
    film = bool(ginfo.get("film"))
    nzv = int(nz) if nz is not None else (int(round(ginfo["T_um"] / h)) if film else N)
    w = np.array([SH.min_dim(int(c), P, meta) for c, P in zip(parts["codes"], parts["P"])], float)
    if refine is None:
        refine = int(min(max_refine, max(0, math.ceil(math.log2(max(h * res / max(float(w.min()), 1e-12), 1.0)) - 1e-9))))
    s0 = h / (1 << refine)
    g = math.gcd(int(N), int(nzv)) << refine
    lmax = 0
    while g % 2 == 0 and lmax < 10:
        g //= 2
        lmax += 1
    while lmax > 0 and ((min(int(N), int(nzv)) << refine) >> lmax) < 4:
        lmax -= 1                                   # four coarsest cells along the shortest side at least
    if coarsest is not None:
        lmax = min(lmax, int(coarsest))
    lev = np.clip(np.floor(np.log2(np.maximum(w / res / s0, 1e-12)) + 1e-9), 0, lmax).astype(np.int64)
    return s0, int(refine), int(lmax), lev, w


def build_mesh(N, h, ginfo, gen_phases, mat_of_phase, nz=None, periodic=None, delta=TF.DELTA, log=None, res=4.0,
               refine=None, contact=1, coarsest=None, neck=0.0, neck_law=None, overlap="merged", split_phases=None,
               t0_um=None):
    """The graded body-fitted mesh of a generated structure. Arguments as tetfem.build_mesh, and: res, cells
    across the smallest dimension of a particle at its surface; refine, the halvings of the voxel for the finest
    cell (default: as many as the finest particle needs, three at most); contact, the further levels where two
    particles meet; coarsest, a cap on the largest cell (levels above the finest).
    Returns the mesh of tetfem.build_mesh with "graded": True and the cells per level."""
    t0 = time.time()
    film = bool(ginfo.get("film"))
    nzv = int(nz) if nz is not None else (int(round(ginfo["T_um"] / h)) if film else N)
    gper = np.array([True, True, not film])
    per = np.array(gper if periodic is None else periodic, np.bool_)
    parts = ginfo["particles"]
    planes, nfaces = ginfo["planes"]
    s0, refine, lmax, lev, w = plan(N, h, ginfo, nz=nz, res=res, refine=refine, coarsest=coarsest)
    nf = np.array([N << refine, N << refine, nzv << refine], np.int64)
    if int(nf.max()) >= B:
        raise ValueError("the graded mesh holds at most 16,384 finest cells along an axis")
    Lg = np.array([N * h, N * h, nzv * h], float)
    codes = parts["codes"].astype(np.int64)
    centers = np.ascontiguousarray(parts["centers"], float)
    R = np.ascontiguousarray(parts["R"], float)
    P = np.ascontiguousarray(parts["P"], float)
    ext = np.zeros((len(codes), 3))
    _extents(codes, R, P, planes, nfaces, ext)
    nk, grow = TF.neck_pairs(ginfo, float(neck), Lg, gper, neck_law)
    ext += grow                                       # a neck reaches half its width beyond the particle
    S = (1 << lmax) * s0
    nroot = (nf >> lmax).astype(np.int64)
    ptr = np.zeros(int(nroot.prod()) + 1, np.int64)
    _root_lists(nroot, S, centers, ext, gper, False, ptr, np.zeros(1, np.int64))
    cnt = ptr[:-1].copy()
    ptr[1:] = np.cumsum(cnt)
    ptr[0] = 0
    idx = np.zeros(max(int(ptr[-1]), 1), np.int64)
    fillp = ptr.copy()
    _root_lists(nroot, S, centers, ext, gper, True, fillp, idx)
    leaves = _octree(nroot, lmax, s0, codes, centers, R, P, planes, nfaces, Lg, gper, lev, w, int(contact), ptr, idx, nk, grow)
    leaves = _balance(leaves, nf, per, lmax)
    xyz, bt = _tets(leaves, nf, per)
    points0 = xyz.astype(float) * (0.5 * s0)
    nn = int(len(points0))
    # the particle at every node
    owner = np.zeros(nn, np.int64)
    nb = max(1, int(round((nn / 12.0) ** (1.0 / 3.0))))
    nbin = np.maximum(1, np.round(nb * Lg / Lg.max()).astype(np.int64))
    _owners(points0, nbin, Lg, gper, codes, centers, R, P, planes, nfaces, ext, owner, nk)
    lab = np.zeros(nn, np.int64)
    has = owner > 0
    lab[has] = np.asarray(mat_of_phase, np.int64)[parts["phase"].astype(np.int64)[owner[has] - 1]]
    # the cut edges
    bt = np.sort(bt, axis=1)
    ot = owner[bt]
    cutm = (ot != ot[:, :1]).any(axis=1)
    ct = bt[cutm]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    keys = np.stack([ct[:, a] * nn + ct[:, b] for a, b in pairs], axis=1)
    iscut = np.stack([owner[ct[:, a]] != owner[ct[:, b]] for a, b in pairs], axis=1)
    uk, inv = np.unique(keys[iscut], return_inverse=True)
    ne = int(len(uk))
    te = np.full(keys.shape, -1, np.int64)
    te[iscut] = inv
    del keys, inv
    ea, eb = uk // nn, uk % nn
    cpt = np.zeros((max(ne, 1), 3))
    tcut = np.zeros(max(ne, 1))
    gedge = np.zeros(max(ne, 1))
    film_e = np.zeros(max(ne, 1), np.int8)
    if ne:
        _cross(ea, eb, points0, per, Lg, owner, codes, centers, R, P, planes, nfaces, Lg, gper, float(delta), cpt, tcut,
               gedge, film_e, nk)
    # the owners at the ends of every cut edge and its direction (for the separated overlaps and the interfaces)
    oa_e, ob_e = owner[ea], owner[eb]
    de_e = points0[eb] - points0[ea]
    for a in range(3):
        if per[a]:
            de_e[:, a] -= Lg[a] * np.floor(de_e[:, a] / Lg[a] + 0.5)
    lut = np.zeros(len(parts["phase"]) + 1, np.int64)
    lut[1:] = np.asarray(mat_of_phase, np.int64)[parts["phase"].astype(np.int64)]
    n_sep = n_if = 0
    if overlap == "separate" and ne:
        intended = np.array([bool(gen_phases[int(p)].get("overlap")) for p in parts["phase"]], bool)
        n_sep = TF._separate_overlaps(oa_e, ob_e, de_e, gedge, film_e, centers, Lg, gper,
                                      shapes=(codes, R, P, planes, nfaces), xa=np.ascontiguousarray(points0[ea], float),
                                      ev=np.ascontiguousarray(de_e, float), intended=intended, bonded=nk)
    if split_phases and ne:
        split = np.isin(np.asarray(parts["phase"], np.int64), np.asarray(sorted(split_phases), np.int64))
        n_if = TF._split_interfaces(oa_e, ob_e, film_e, split)
    gmin = float(max(t0_um, TF.GAP_FLOOR * s0)) if t0_um is not None else float(TF.GAP_MIN * s0)
    wside = np.zeros((max(ne, 1), 2))
    wn = np.zeros((max(ne, 2), 2, 3))
    aside = np.zeros((max(ne, 1), 2))
    ntc, nfilm = _decompose(ct, te, points0, per, Lg, owner, lab, nn, tcut, gedge, film_e, gmin, False,
                            np.zeros((1, 4), np.int64), np.zeros(1, np.uint8), np.zeros(1, np.int32), wside, wn, aside)
    tets_c = np.empty((ntc, 4), np.int64)
    mat_c = np.empty(ntc, np.uint8)
    _decompose(ct, te, points0, per, Lg, owner, lab, nn, tcut, gedge, film_e, gmin, True, tets_c, mat_c,
               np.empty(ntc, np.int32), wside, wn, aside)
    plain = bt[~cutm]
    tets = np.concatenate([plain, tets_c])
    mat = np.concatenate([lab[plain[:, 0]].astype(np.uint8), mat_c])
    del plain, tets_c, mat_c, ct, te
    used = np.zeros(nn + 3 * ne, np.bool_)
    used[:nn] = True
    used[tets.ravel()] = True
    new = np.cumsum(used) - 1
    tets = new[tets]
    cut_ids = np.flatnonzero(used[nn:])
    points = np.concatenate([points0, cpt[cut_ids // 3]])
    points, fe, springs, sw, sa, sn, slab, iface, ia, ilab = TF._springs(nn, ne, used, new, film_e, wside, wn, aside,
                                                                        cpt, points, lut, oa_e, ob_e)
    # the voxel each tetrahedron's centre lies in
    cells = np.array([N, N, nzv], np.int64)
    p0 = points[tets[:, 0]]
    cen = p0.copy()
    for m in range(1, 4):
        dd = points[tets[:, m]] - p0
        for a in range(3):
            if per[a]:
                dd[:, a] -= Lg[a] * np.floor(dd[:, a] / Lg[a] + 0.5)
        cen += 0.25 * dd
    ci = np.floor(cen / h).astype(np.int64)
    for a in range(3):
        ci[:, a] = ci[:, a] % cells[a] if per[a] else np.clip(ci[:, a], 0, cells[a] - 1)
    cell = ((ci[:, 0] * cells[1] + ci[:, 1]) * cells[2] + ci[:, 2]).astype(np.int32)
    levels = np.bincount(leaves[:, 3], minlength=lmax + 1)
    clamped = int(np.count_nonzero((tcut[:ne] <= delta * (1 + 1e-9)) | (tcut[:ne] >= 1 - delta * (1 + 1e-9)))) if ne else 0
    mesh = {"points": points, "tets": tets, "mat": mat, "cell": cell, "n": cells + (~per).astype(np.int64),
            "cells": cells, "per": per, "L": Lg, "h": float(h), "n_nodes": nn, "n_cut": ne,
            "n_film_edges": int(len(fe)), "n_film_tets": int(nfilm), "springs": springs, "spring_w": sw,
            "spring_n": sn, "n_clamped": clamped, "seconds": time.time() - t0, "graded": True, "finest_um": float(s0),
            "spring_a": sa, "spring_lab": slab, "iface": iface, "iface_a": ia, "iface_lab": ilab, "t0_um": gmin,
            "overlap": overlap, "n_separated": int(n_sep), "n_iface_edges": int(n_if),
            "neck": float(neck), "n_bonded": int(len(nk[1]) // 2) if grow > 0.0 else 0,
            "coarsest_um": float(S), "levels": levels.tolist(), "refine": int(refine), "n_leaves": int(len(leaves)),
            "film_gap_um": (float(np.median(gedge[fe])) if len(fe) else 0.0)}
    if log:
        log(f"    graded tetrahedral mesh: {len(tets):,} tetrahedra on {len(points):,} vertices; cells from "
            f"{s0:.3g} µm (at the finest particles) to {S:.3g} µm, {len(leaves):,} of them "
            f"(a uniform mesh of the finest cell: {int(nf.prod()):,}); {len(fe):,} film edges"
            + (f", bonded contacts (neck {neck:g}) between {mesh['n_bonded']:,} pairs" if neck > 0 else "")
            + (f"; {n_sep:,} overlapping pairs met as touching ones" if n_sep else "")
            + (f"; {len(iface):,} interface elements carry the particle-matrix resistance" if len(iface) else "")
            + (f"; thinnest film {1000 * gmin:.3g} nm" if t0_um is not None else "")
            + f" ({mesh['seconds']:.1f} s)")
    return mesh
