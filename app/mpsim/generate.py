"""RVE microstructure generation on a periodic voxel grid (v4).

Placement routes, chosen per job:

* **growth with overlap relaxation + Monte-Carlo** - every particle phase is a
  non-overlapping sphere. The spheres start small and grow while overlaps are
  pushed apart (the force-biased / collective-rearrangement family), run by the
  compiled clump engine below; if they stall short of the target, Monte-Carlo
  compression (Metropolis sweeps, then inflation to the first contact) takes
  over. Equal spheres reach random close packing (~64 %), a size distribution
  more. A Metropolis pass then removes the "shrink-wrapped" artefact of every
  pair left at exactly contact.
* **voxel RSA** - any shape and any mixture, low to moderate loading. Each
  candidate is rasterised against the occupancy grid, so the overlap test is
  exact at the resolution the solver will see.
* **rigid clump relaxation** - when RSA jams before the target (it stalls near
  38 % for spheres and much earlier for elongated shapes). Every particle is
  represented for packing by a clump of spheres (the multi-sphere method of
  discrete-element codes); the clumps start small and grow while overlaps are
  pushed apart and the bodies are allowed to translate and rotate
  (orientation constraints of the phase are kept). The exact shapes are
  rasterised at the end and any residual voxel overlap is measured.
* **Gaussian random field** - the bicontinuous "network" phase, of any
  material (open porosity, a sintered skeleton, a foam's solid).

Particles keep their shape. Two particles whose surfaces come within a voxel
of each other become voxel neighbours, and a voxel solver then sees a contact
the continuous geometry does not have. The voxel finite-element solver even
conducts through a shared edge or corner (measured: +27 % k at 30 vol %
alumina in epoxy, against a face-based FV solve). The contact modes:

* ``apart`` (default) - particles are *moved*, never trimmed, so that no voxel
  of one particle touches a voxel of another in any of the 26 directions:
  RSA rejects such positions, and the packings keep a surface gap of 2 voxels
  (>= sqrt(3) voxels rules out corner neighbours). Where the target is too
  dense for that gap, the gap closes as far as the packing needs and the
  remaining contacts are counted as real ones (they can carry R_c).
* ``keep`` - no separation: particles may touch wherever the packing puts
  them (a contact network, e.g. with a contact resistance).
* ``separate`` - v3: touching voxels are cleared to matrix, which trims the
  particles.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange

from . import geometry as G
from . import shapes as SH

_NB26 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                  if (i, j, k) != (0, 0, 0)], dtype=np.int64)
_FWD26 = np.array([o for o in _NB26.tolist() if tuple(o) > (0, 0, 0)], dtype=np.int64)
_FWD6 = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.int64)


# =========================================================================
# rasterisation
# =========================================================================
@njit(cache=True)
def _place(owner, phase_of, overlap_of, net_of, h, code, cx, cy, cz, R, P, planes, nfaces,
           gap, inset, phase, pid, sep1, check, conflict, buf, zwall=False):
    """Rasterise one particle. Returns voxels written, or -1 on collision.

    check=True  : write nothing unless the whole particle (grown by `gap`) is free.
    check=False : write the free voxels; voxels already held by another
                  particle are counted in conflict[0] and left to their owner.
    `inset` shrinks the shape (a shell's inner boundary).
    zwall=True (a film): z does not wrap; a particle that reaches beyond the
    bottom or top face collides with it (check=True) or loses those voxels.
    """
    n0, n1, n2 = owner.shape
    hx, hy, hz = SH.local_half_extents(code, P, planes, nfaces)
    hx += gap
    hy += gap
    hz += gap
    ex = abs(R[0, 0]) * hx + abs(R[1, 0]) * hy + abs(R[2, 0]) * hz
    ey = abs(R[0, 1]) * hx + abs(R[1, 1]) * hy + abs(R[2, 1]) * hz
    ez = abs(R[0, 2]) * hx + abs(R[1, 2]) * hy + abs(R[2, 2]) * hz
    i0 = int(math.floor((cx - ex) / h - 0.5))
    i1 = int(math.ceil((cx + ex) / h - 0.5))
    j0 = int(math.floor((cy - ey) / h - 0.5))
    j1 = int(math.ceil((cy + ey) / h - 0.5))
    k0 = int(math.floor((cz - ez) / h - 0.5))
    k1 = int(math.ceil((cz + ez) / h - 0.5))
    n_in = 0
    same_ok = overlap_of[phase] == 1
    for i in range(i0, i1 + 1):
        dx = (i + 0.5) * h - cx
        ii = i % n0
        for j in range(j0, j1 + 1):
            dy = (j + 0.5) * h - cy
            jj = j % n1
            for k in range(k0, k1 + 1):
                dz = (k + 0.5) * h - cz
                lx = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
                ly = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
                lz = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz
                if not SH.inside_local(code, lx, ly, lz, P, planes, nfaces, -gap if gap > 0.0 else inset):
                    continue
                if zwall and (k < 0 or k >= n2):
                    if check:
                        return -1
                    continue
                kk = k % n2
                if check:
                    o = owner[ii, jj, kk]
                    if o != 0 and o != pid:
                        if not (same_ok and phase_of[o] == phase):
                            return -1
                if gap > 0.0:
                    if not SH.inside_local(code, lx, ly, lz, P, planes, nfaces, inset):
                        continue
                if check and sep1:
                    for nb in range(26):
                        ni = (ii + _NB26[nb, 0]) % n0
                        nj = (jj + _NB26[nb, 1]) % n1
                        nk = (kk + _NB26[nb, 2]) % n2
                        o = owner[ni, nj, nk]
                        if o != 0 and o != pid:
                            po = phase_of[o]
                            if net_of[po] == 0 and not (same_ok and po == phase):
                                return -1
                buf[n_in] = (ii * n1 + jj) * n2 + kk
                n_in += 1
    written = 0
    for m in range(n_in):
        idx = buf[m]
        kk = idx % n2
        t = idx // n2
        jj = t % n1
        ii = t // n1
        o = owner[ii, jj, kk]
        if o == 0:
            owner[ii, jj, kk] = pid
            written += 1
        elif o != pid and not check:
            if not (same_ok and phase_of[o] == phase):
                conflict[0] += 1
    return written


@njit(cache=True)
def _rsa(owner, phase_of, overlap_of, net_of, sep1_of, h, L, codes, Rs, Ps, planes, nfaces,
         gap_arr, phase_arr, target, count, n_placed, jammed, centers, placed_id,
         pid, max_attempts, jam_limit, seed, buf, zwall=False):
    np.random.seed(seed)
    fails = np.zeros(target.shape[0], np.int64)
    conflict = np.zeros(1, np.int64)
    for m in range(codes.shape[0]):
        ph = phase_arr[m]
        if jammed[ph] or count[ph] >= target[ph]:
            continue
        ok = False
        for _t in range(max_attempts):
            cx = np.random.random() * L
            cy = np.random.random() * L
            cz = np.random.random() * (owner.shape[2] * h)
            phase_of[pid] = ph
            w = _place(owner, phase_of, overlap_of, net_of, h, codes[m], cx, cy, cz, Rs[m], Ps[m],
                       planes, nfaces, gap_arr[m], 0.0, ph, pid, sep1_of[ph] == 1, True, conflict, buf,
                       zwall)
            if w > 0:
                count[ph] += w
                n_placed[ph] += 1
                centers[m, 0] = cx
                centers[m, 1] = cy
                centers[m, 2] = cz
                placed_id[m] = pid
                pid += 1
                ok = True
                break
        if ok:
            fails[ph] = 0
        else:
            phase_of[pid] = -1
            fails[ph] += 1
            if fails[ph] >= jam_limit:
                jammed[ph] = True
    return pid


@njit(cache=True)
def raster_all(owner, phase_of, overlap_of, net_of, h, codes, centers, Rs, Ps, planes, nfaces,
               insets, phase_arr, pid0, buf, zwall=False):
    """Rasterise a particle list without collision checks. Returns (next id, conflicts)."""
    pid = pid0
    conflict = np.zeros(1, np.int64)
    for m in range(codes.shape[0]):
        phase_of[pid] = phase_arr[m]
        _place(owner, phase_of, overlap_of, net_of, h, codes[m], centers[m, 0], centers[m, 1],
               centers[m, 2], Rs[m], Ps[m], planes, nfaces, 0.0, insets[m], phase_arr[m], pid,
               False, False, conflict, buf, zwall)
        pid += 1
    return pid, conflict[0]


@njit(cache=True)
def _raster_track(owner, phase_of, overlap_of, h, codes, centers, Rs, Ps, planes, nfaces,
                  phase_arr, pid0, csum, ccnt, fsum, tsum, buf, zwall=False):
    """Rasterise all particles; for every voxel two particles both claim,
    add its position to both particles' conflict sums (for the repair step),
    and a unit push along the line between the two centres - with the torque
    it exerts at that voxel - to their force and torque sums. A voxel
    beyond a film face pushes back into the film."""
    n0, n1, n2 = owner.shape
    L0 = n0 * h
    Lz0 = n2 * h
    for m in range(codes.shape[0]):
        pid = pid0 + m
        phase_of[pid] = phase_arr[m]
    total = 0
    for m in range(codes.shape[0]):
        pid = pid0 + m
        code = codes[m]
        cx, cy, cz = centers[m, 0], centers[m, 1], centers[m, 2]
        R = Rs[m]
        P = Ps[m]
        hx, hy, hz = SH.local_half_extents(code, P, planes, nfaces)
        ex = abs(R[0, 0]) * hx + abs(R[1, 0]) * hy + abs(R[2, 0]) * hz
        ey = abs(R[0, 1]) * hx + abs(R[1, 1]) * hy + abs(R[2, 1]) * hz
        ez = abs(R[0, 2]) * hx + abs(R[1, 2]) * hy + abs(R[2, 2]) * hz
        same_ok = overlap_of[phase_arr[m]] == 1
        for i in range(int(math.floor((cx - ex) / h - 0.5)), int(math.ceil((cx + ex) / h - 0.5)) + 1):
            dx = (i + 0.5) * h - cx
            ii = i % n0
            for j in range(int(math.floor((cy - ey) / h - 0.5)), int(math.ceil((cy + ey) / h - 0.5)) + 1):
                dy = (j + 0.5) * h - cy
                jj = j % n1
                for k in range(int(math.floor((cz - ez) / h - 0.5)), int(math.ceil((cz + ez) / h - 0.5)) + 1):
                    dz = (k + 0.5) * h - cz
                    lx = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
                    ly = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
                    lz = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz
                    if not SH.inside_local(code, lx, ly, lz, P, planes, nfaces, 0.0):
                        continue
                    if zwall and (k < 0 or k >= n2):
                        # beyond the film face: a conflict with the wall, which
                        # pushes the particle back inside
                        total += 1
                        csum[m, 0] += dx
                        csum[m, 1] += dy
                        csum[m, 2] += dz
                        ccnt[m] += 1
                        wz = 1.0 if k < 0 else -1.0
                        fsum[m, 2] += wz
                        tsum[m, 0] += dy * wz
                        tsum[m, 1] -= dx * wz
                        continue
                    kk = k % n2
                    o = owner[ii, jj, kk]
                    if o == 0:
                        owner[ii, jj, kk] = pid
                    elif o != pid:
                        if same_ok and phase_of[o] == phase_arr[m]:
                            continue
                        total += 1
                        # conflict: the voxel (as an offset from each centre)
                        om = o - pid0
                        csum[m, 0] += dx
                        csum[m, 1] += dy
                        csum[m, 2] += dz
                        ccnt[m] += 1
                        ox = (i + 0.5) * h - centers[om, 0]
                        oy = (j + 0.5) * h - centers[om, 1]
                        oz = (k + 0.5) * h - centers[om, 2]
                        ox -= L0 * math.floor(ox / L0 + 0.5)
                        oy -= L0 * math.floor(oy / L0 + 0.5)
                        if not zwall:
                            oz -= Lz0 * math.floor(oz / Lz0 + 0.5)
                        csum[om, 0] += ox
                        csum[om, 1] += oy
                        csum[om, 2] += oz
                        ccnt[om] += 1
                        # unit push from the other centre towards this one
                        ux = cx - centers[om, 0]
                        uy = cy - centers[om, 1]
                        uz = cz - centers[om, 2]
                        ux -= L0 * math.floor(ux / L0 + 0.5)
                        uy -= L0 * math.floor(uy / L0 + 0.5)
                        if not zwall:
                            uz -= Lz0 * math.floor(uz / Lz0 + 0.5)
                        un = math.sqrt(ux * ux + uy * uy + uz * uz)
                        if un < 1e-12:
                            ux, uy, uz, un = 1.0, 0.0, 0.0, 1.0
                        ux /= un
                        uy /= un
                        uz /= un
                        fsum[m, 0] += ux
                        fsum[m, 1] += uy
                        fsum[m, 2] += uz
                        tsum[m, 0] += dy * uz - dz * uy
                        tsum[m, 1] += dz * ux - dx * uz
                        tsum[m, 2] += dx * uy - dy * ux
                        fsum[om, 0] -= ux
                        fsum[om, 1] -= uy
                        fsum[om, 2] -= uz
                        tsum[om, 0] -= oy * uz - oz * uy
                        tsum[om, 1] -= oz * ux - ox * uz
                        tsum[om, 2] -= ox * uy - oy * ux
    return total


@njit(cache=True)
def _touch_track(owner, phase_of, overlap_of, net_of, pid0, n, h, centers, csum, ccnt, fwd, zper=True):
    """For every pair of neighbouring voxels (26 directions) held by two
    different particles, add the contact point to both particles' sums."""
    n0, n1, n2 = owner.shape
    L0 = n0 * h
    total = 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                o = owner[i, j, k]
                if o < pid0 or o >= pid0 + n:
                    continue
                po = phase_of[o]
                for nb in range(fwd.shape[0]):
                    ni = (i + fwd[nb, 0]) % n0
                    nj = (j + fwd[nb, 1]) % n1
                    kz = k + fwd[nb, 2]
                    if not zper and (kz < 0 or kz >= n2):
                        continue
                    nk = kz % n2
                    o2 = owner[ni, nj, nk]
                    if o2 == o or o2 < pid0 or o2 >= pid0 + n:
                        continue
                    p2 = phase_of[o2]
                    if net_of[p2] == 1 or (overlap_of[po] == 1 and p2 == po):
                        continue
                    total += 1
                    x = (i + 0.5 + 0.5 * fwd[nb, 0]) * h
                    y = (j + 0.5 + 0.5 * fwd[nb, 1]) * h
                    z = (k + 0.5 + 0.5 * fwd[nb, 2]) * h
                    for m in (o - pid0, o2 - pid0):
                        dx = x - centers[m, 0]
                        dy = y - centers[m, 1]
                        dz = z - centers[m, 2]
                        dx -= L0 * math.floor(dx / L0 + 0.5)
                        dy -= L0 * math.floor(dy / L0 + 0.5)
                        dz -= L0 * math.floor(dz / L0 + 0.5)
                        csum[m, 0] += dx
                        csum[m, 1] += dy
                        csum[m, 2] += dz
                        ccnt[m] += 1
    return total


def _rotate_rows(Rs, idx, axis, ang):
    """Turn the particles idx by `ang` (radians, per particle) about `axis`
    (world, per particle): every row of Rs[idx] (a local axis in the world) is
    rotated, then the frame is made orthonormal again."""
    u = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    K = np.zeros((len(idx), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -u[:, 2], u[:, 1], -u[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = u[:, 2], -u[:, 1], u[:, 0]
    sa, oc = np.sin(ang)[:, None, None], (1.0 - np.cos(ang))[:, None, None]
    Q = np.eye(3)[None] + sa * K + oc * np.einsum("nij,njk->nik", K, K)
    R = np.einsum("nij,nkj->nik", Rs[idx], Q)
    z = R[:, 2] / np.linalg.norm(R[:, 2], axis=1, keepdims=True)
    x = R[:, 0] - np.sum(R[:, 0] * z, axis=1, keepdims=True) * z
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    Rs[idx] = np.stack([x, np.cross(z, x), z], axis=1)


def repair_overlaps(owner, phase_of, overlap_of, h, codes, C, Rs, Ps, planes, nfaces, ph_arr, pid0,
                    log=None, max_rounds=160, net_of=None, apart=False, apart_rounds=300, zwall=False,
                    rotmode=None, shrink=True):
    """Push and turn particles off the voxels they share until no voxel is
    claimed twice; what pushing cannot clear is cleared by shrinking.

    The clump approximation misses the corners, edges and rims of boxes,
    flakes, cylinders and polyhedra (a flake 8 x 8 x 1.5 um is covered to
    35 % by its 40 packing spheres, a cube to 82 %), so after the clump
    packing the exact shapes overlap there. Each round rasterises the exact
    shapes; every voxel two particles share pushes them apart along the line
    between their centres, and turns them by the torque that push exerts at
    the voxel (rotmode per particle: -1 none, 0 free, 1 about its own axis,
    2 also about z - the orientation the phase asks for is kept). Corners
    caught in each other come free by turning, which pushing alone could not
    do in a dense packing (cubes and flakes at 45 vol% kept 271 shared voxels).

    The pushing goes on past max_rounds while it still makes progress (a new
    fewest count of shared voxels within the last 40 rounds), up to four times
    max_rounds - all counted in rounds, not seconds.
    If voxels are still shared after that, the particles involved shrink
    in steps of 2 % - shape and orientation kept - until none is: particles
    never overlap and are never trimmed (v5.0 and earlier left the shared
    voxel to the first particle, cutting a corner off the other).

    apart=True goes on once the overlaps are gone: particles with a voxel next
    to another particle's voxel (26 directions) are pushed apart in smaller
    steps while the count keeps falling (it stops after 10 rounds without a new
    best: in a dense packing the pushes only trade contacts for overlaps), at
    most `apart_rounds` rounds (counted, not timed, so that the structure does
    not depend on the speed of the PC), and the overlap-free state with the
    fewest touching pairs is kept.
    Returns (owner filled, remaining shared voxels, rounds, touching pairs or
    -1, shrink {"n": particles shrunk, "min": smallest size factor, "mean":
    mean factor of those shrunk})."""
    n = len(codes)
    L = owner.shape[0] * h
    Lz = owner.shape[2] * h
    ext = max(2.0 * SH.circumradius(int(c), P) for c, P in zip(codes, Ps))
    buf = np.empty(int(math.ceil(ext / h) + 4) ** 3, np.int64)
    rg = np.array([SH.circumradius(int(c), P) for c, P in zip(codes, Ps)])
    rot = np.full(n, -1, np.int64) if rotmode is None else np.asarray(rotmode, np.int64)
    no_shrink = {"n": 0, "min": 1.0, "mean": 1.0}
    conf = 0
    best = None                      # (touching pairs, C, Rs) of the best overlap-free state
    fewest = None                    # (shared voxels, C, Rs) while none is overlap-free
    since_clean = stale = 0
    touch = -1
    rnd = 0
    last_gain = 0

    def track():
        owner[:] = 0
        cs = np.zeros((n, 3))
        cc = np.zeros(n, np.int64)
        fs = np.zeros((n, 3))
        ts = np.zeros((n, 3))
        c_ = int(_raster_track(owner, phase_of, overlap_of, float(h), codes, C, Rs, Ps, planes, nfaces,
                               ph_arr, pid0, cs, cc, fs, ts, buf, zwall))
        return c_, cs, cc, fs, ts

    for rnd in range(4 * max_rounds + (apart_rounds if apart else 0)):
        conf, csum, ccnt, fsum, tsum = track()
        step_max = 0.5
        if conf == 0:
            if not apart:
                return owner, 0, rnd, -1, no_shrink
            touch = int(_touch_track(owner, phase_of, overlap_of, net_of, pid0, n, float(h), C, csum, ccnt,
                                     _FWD26, not zwall))
            if best is None or touch < best[0]:
                best = (touch, C.copy(), Rs.copy())
                stale = 0
            else:
                stale += 1
            if touch == 0 or since_clean >= apart_rounds or stale >= 10:
                break
            step_max = 0.25
        else:
            if best is None and (fewest is None or conf < fewest[0]):
                fewest = (conf, C.copy(), Rs.copy())
                last_gain = rnd
            if best is not None:
                stale += 1
                if stale >= 10 or since_clean >= apart_rounds:
                    break
            elif rnd >= max_rounds and (rnd - last_gain >= 40 or rnd >= 4 * max_rounds - 1):
                break
        if best is not None:
            since_clean += 1
        hit = ccnt > 0
        idx = np.flatnonzero(hit)
        away = -csum[hit] / ccnt[hit][:, None]
        if conf > 0:
            # shared voxels push along the lines between the centres
            fh = fsum[hit]
            strong = np.linalg.norm(fh, axis=1) > 0.2 * ccnt[hit]
            away[strong] = fh[strong]
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        stepl = np.minimum(step_max, 0.15 * np.sqrt(ccnt[hit])) * h
        moved = C[hit] + stepl[:, None] * away / norm
        moved[:, :2] = np.mod(moved[:, :2], L)
        if not zwall:
            moved[:, 2] = np.mod(moved[:, 2], Lz)
        C[hit] = moved
        if conf > 0:
            T = tsum[hit].copy()
            rm = rot[idx]
            zax = Rs[idx, 2, :]
            keep_ax = (rm == 1) | (rm == 2)
            if keep_ax.any():
                ka = np.flatnonzero(keep_ax)
                sp = np.sum(T[ka] * zax[ka], axis=1, keepdims=True)
                tz = T[ka, 2].copy()
                T[ka] = sp * zax[ka]
                m2 = rm[ka] == 2
                T[ka[m2], 2] += tz[m2]
            tn = np.linalg.norm(T, axis=1)
            lever = tn / (ccnt[hit] * np.maximum(rg[idx], 1e-30))
            ang = np.minimum(0.06, stepl / np.maximum(rg[idx], 1e-30) * np.minimum(1.0, 2.0 * lever))
            turn = (rm >= 0) & (tn > 1e-12) & (ang > 1e-5)
            if turn.any():
                _rotate_rows(Rs, idx[turn], T[turn], ang[turn])
        if log is not None and rnd % 10 == 9:
            log(f"    overlap repair round {rnd+1}: {conf:,} shared voxels"
                + (f", {touch:,} touching voxel pairs" if touch >= 0 else ""))
    if best is not None:
        if best[0] != touch or conf != 0:
            C[:] = best[1]
            Rs[:] = best[2]
            conf = track()[0]
        return owner, conf, rnd + 1, best[0], no_shrink
    if fewest is not None and fewest[0] < conf:
        C[:] = fewest[1]
        Rs[:] = fewest[2]
        conf = track()[0]
    if not (conf and shrink):
        return owner, conf, rnd + 1, touch, no_shrink
    # the last shared voxels: shrink the particles involved, 2 % a step
    fac = np.ones(n)
    for _ in range(60):
        conf, csum, ccnt, fsum, tsum = track()
        if conf == 0:
            break
        for m in np.flatnonzero(ccnt > 0):
            Ps[m, _linear_slots(int(codes[m]))] *= 0.98
            fac[m] *= 0.98
    shr = fac < 1.0
    return owner, conf, rnd + 1, touch, {"n": int(shr.sum()), "min": float(fac.min()),
                                         "mean": float(fac[shr].mean()) if shr.any() else 1.0}


@njit(cache=True)
def _cleanup(owner, phase_of, overlap_of, net_of, fwd):
    """v3 behaviour, kept as an option: clear one voxel at every contact between
    two different particles (3 offsets = faces only, 13 = faces, edges, corners)."""
    n0, n1, n2 = owner.shape
    removed = 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                o = owner[i, j, k]
                if o == 0:
                    continue
                po = phase_of[o]
                if net_of[po] == 1:
                    continue
                for ax in range(fwd.shape[0]):
                    ni = (i + fwd[ax, 0]) % n0
                    nj = (j + fwd[ax, 1]) % n1
                    nk = (k + fwd[ax, 2]) % n2
                    o2 = owner[ni, nj, nk]
                    if o2 == 0 or o2 == o:
                        continue
                    p2 = phase_of[o2]
                    if net_of[p2] == 1:
                        continue
                    if overlap_of[po] == 1 and p2 == po:
                        continue
                    removed += 1
                    if o2 > o:
                        owner[ni, nj, nk] = 0
                    else:
                        owner[i, j, k] = 0
                        break
    return removed


@njit(parallel=True, cache=True)
def _contact_rows(owner, phase_of, net_of, out, rows, zper=True):
    n0, n1, n2 = owner.shape
    for i0 in prange(n0):
        i = np.int64(i0)
        cnt = 0
        for j in range(n1):
            for k in range(n2):
                o = owner[i, j, k]
                bits = 0
                if o != 0 and net_of[phase_of[o]] == 0:
                    for ax in range(3):
                        ni, nj, nk = i, j, k
                        if ax == 0:
                            ni = i + 1 if i + 1 < n0 else 0
                        elif ax == 1:
                            nj = j + 1 if j + 1 < n1 else 0
                        else:
                            if not zper and k + 1 >= n2:
                                continue
                            nk = k + 1 if k + 1 < n2 else 0
                        o2 = owner[ni, nj, nk]
                        if o2 != 0 and o2 != o and net_of[phase_of[o2]] == 0:
                            bits |= 1 << ax
                            cnt += 1
                out[i, j, k] = bits
        rows[i] = cnt


def contact_faces(owner, phase_of, net_of, out, zper=True):
    """Bit per +axis face (1 x, 2 y, 4 z) that separates two different particles.

    These faces are where two fillers touch; the thermal solve can give them a
    contact resistance. Returns the number of such faces."""
    rows = np.zeros(owner.shape[0], np.int64)
    _contact_rows(owner, phase_of, net_of, out, rows, zper)
    return int(rows.sum())


@njit(parallel=True, cache=True)
def _touch_rows(owner, phase_of, net_of, fwd, rows, zper=True):
    n0, n1, n2 = owner.shape
    for i0 in prange(n0):
        i = np.int64(i0)
        cnt = 0
        for j in range(n1):
            for k in range(n2):
                o = owner[i, j, k]
                if o == 0 or net_of[phase_of[o]] == 1:
                    continue
                for nb in range(fwd.shape[0]):
                    kz = k + fwd[nb, 2]
                    if not zper and (kz < 0 or kz >= n2):
                        continue
                    o2 = owner[(i + fwd[nb, 0]) % n0, (j + fwd[nb, 1]) % n1, kz % n2]
                    if o2 != 0 and o2 != o and net_of[phase_of[o2]] == 0:
                        cnt += 1
        rows[i] = cnt


def touching_pairs(owner, phase_of, net_of, zper=True):
    """Voxel pairs of two different particles that are neighbours in any of
    the 26 directions (faces, edges and corners)."""
    rows = np.zeros(owner.shape[0], np.int64)
    _touch_rows(owner, phase_of, net_of, _FWD26, rows, zper)
    return int(rows.sum())


SEP_APART = 2.0          # surface gap, in voxels, that contacts="apart" asks of the packings
# Packing budgets are counted in steps, never in seconds, so that the same
# inputs and seed give the same structure for any box size and on any PC.
# 250 nm spheres at 55 vol% reached full size in 31,000 relaxation steps in a
# 5 µm box; the limit leaves room for denser and larger packings, which
# otherwise end on their stall criterion.
MAX_RELAX_STEPS = 150_000
MAX_COMPRESS_ITER = 1_500
# a tighter budget for jamming_fraction, which only needs where the packing
# stops (set and reset around its call; None outside it)
_BUDGET = None
# random close packing of equal spheres (0.64) over the value this relaxation
# jams at with them (0.6117, 500 spheres, the stall rule of jamming_fraction):
# its scale. With it a lognormal size distribution of CV 0.5 gives 0.680.
JAM_CAL = 0.64 / 0.6117
# Equal spheres stall after about 30,000 steps (8 s). Rotating clumps (cubes)
# keep gaining 0.1 % every few rounds far longer; the step limit bounds the
# time, and a value that reached it is reported as a lower estimate.
JAM_MAX_STEPS = 60_000
# Excluded-volume fraction a packing reaches quickly (spheres: random close
# packing is ~0.64, reached in seconds up to ~0.60; boxes and polyhedra jam
# earlier). The separation gap is cut to fit under it, so the packing never
# spends its time budget on a gap that cannot fit.
_FAST_PACK = {"sphere": 0.60, "other": 0.50}


def apart_gap(vol, gap_user, h, box_vol, kind):
    """Surface gap for contacts="apart": SEP_APART voxels, reduced (not below
    the user's gap) so that the particles grown by half of it fill no more
    than the fast-packing fraction. vol: particle volumes; gap_user: per
    particle; returns an array."""
    vol = np.asarray(vol, float)
    d_eq = (6.0 * vol / math.pi) ** (1.0 / 3.0)
    tot = float(np.sum(vol))
    lim = _FAST_PACK["sphere" if kind == "sphere" else "other"]

    def frac(g):
        return float(np.sum(math.pi / 6.0 * (d_eq + g) ** 3))

    target_vol = lim * box_vol
    g_hi = SEP_APART * h
    if frac(g_hi) <= target_vol:
        g = g_hi
    elif frac(0.0) >= target_vol or tot <= 0:
        g = 0.0
    else:
        lo, hi = 0.0, g_hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if frac(mid) <= target_vol else (lo, mid)
        g = lo
    return np.maximum(np.asarray(gap_user, float), g)



# =========================================================================
# sphere packing (force-biased + Monte-Carlo)
# =========================================================================
@njit(cache=True, fastmath=True)
def _cell_of(x, y, z, cs, nc):
    cx = int(x / cs) % nc
    cy = int(y / cs) % nc
    cz = int(z / cs) % nc
    return (cx * nc + cy) * nc + cz


@njit(cache=True, fastmath=True)
def _cell3(x, y, z, cs, nc, csz, ncz):
    cx = int(x / cs) % nc
    cy = int(y / cs) % nc
    cz = int(z / csz) % ncz
    return (cx * nc + cy) * ncz + cz


@njit(cache=True, fastmath=True)
def _mc_sweeps(pos, rad, L, n_sweeps, disp0, seed, Lz=0.0, Wt=0.0):
    """Athermal hard-sphere Metropolis in a periodic box L x L x Lz (Lz = L
    when 0). Wt > 0: a film, whose spheres must stay within 0 <= z <= Wt."""
    np.random.seed(seed)
    if Lz <= 0.0:
        Lz = L
    n = pos.shape[0]
    cs0 = 2.0 * rad.max()
    nc = max(int(L / cs0), 3)
    cs = L / nc
    ncz = max(int(Lz / cs0), 3)
    csz = Lz / ncz
    head = -np.ones(nc * nc * ncz, np.int64)
    nxt = -np.ones(n, np.int64)
    for i in range(n):
        c = _cell3(pos[i, 0], pos[i, 1], pos[i, 2], cs, nc, csz, ncz)
        nxt[i] = head[c]
        head[c] = i
    disp = disp0
    acc_total = 0
    half = 0.5 * L
    halfz = 0.5 * Lz
    reach = int(math.ceil(cs0 / cs))
    reachz = int(math.ceil(cs0 / csz))
    for s in range(n_sweeps):
        acc = 0
        for _ in range(n):
            i = np.random.randint(0, n)
            x = (pos[i, 0] + (np.random.random() - 0.5) * 2.0 * disp) % L
            y = (pos[i, 1] + (np.random.random() - 0.5) * 2.0 * disp) % L
            z = pos[i, 2] + (np.random.random() - 0.5) * 2.0 * disp
            if Wt > 0.0:
                if z - rad[i] < 0.0 or z + rad[i] > Wt:
                    continue
            else:
                z = z % Lz
            cx = int(x / cs) % nc
            cy = int(y / cs) % nc
            cz = int(z / csz) % ncz
            hit = False
            for ox in range(-reach, reach + 1):
                if hit:
                    break
                ax = (cx + ox) % nc
                for oy in range(-reach, reach + 1):
                    if hit:
                        break
                    ay = (cy + oy) % nc
                    for oz in range(-reachz, reachz + 1):
                        az = (cz + oz) % ncz
                        j = head[(ax * nc + ay) * ncz + az]
                        while j != -1:
                            if j != i:
                                dx = pos[j, 0] - x
                                dy = pos[j, 1] - y
                                dz = pos[j, 2] - z
                                if dx > half:
                                    dx -= L
                                elif dx < -half:
                                    dx += L
                                if dy > half:
                                    dy -= L
                                elif dy < -half:
                                    dy += L
                                if dz > halfz:
                                    dz -= Lz
                                elif dz < -halfz:
                                    dz += Lz
                                sr = rad[i] + rad[j]
                                if dx * dx + dy * dy + dz * dz < sr * sr:
                                    hit = True
                                    break
                            j = nxt[j]
                        if hit:
                            break
            if hit:
                continue
            c_old = _cell3(pos[i, 0], pos[i, 1], pos[i, 2], cs, nc, csz, ncz)
            c_new = _cell3(x, y, z, cs, nc, csz, ncz)
            pos[i, 0] = x
            pos[i, 1] = y
            pos[i, 2] = z
            if c_new != c_old:
                j = head[c_old]
                if j == i:
                    head[c_old] = nxt[i]
                else:
                    while j != -1:
                        if nxt[j] == i:
                            nxt[j] = nxt[i]
                            break
                        j = nxt[j]
                nxt[i] = head[c_new]
                head[c_new] = i
            acc += 1
        acc_total += acc
        if s < n_sweeps // 2 and s % 20 == 19:
            rate = acc / n
            if rate > 0.40:
                disp *= 1.08
            elif rate < 0.30:
                disp *= 0.93
            disp = min(disp, 0.25 * L)
            disp = max(disp, 1e-6 * L)
    return acc_total / max(1, n_sweeps * n)


def _wrap(pos, L):
    pos = np.mod(pos, L)
    pos[pos >= L] = 0.0
    return pos


def _sphere_pairs(pos, rad, box):
    """(i, j, centre distance) of every pair closer than the sum of radii
    plus the largest radius (periodic in all three directions of `box`)."""
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    cand = tree.query_pairs(r=2.0 * float(rad.max()), output_type="ndarray")
    if cand.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    dv = pos[cand[:, 1]] - pos[cand[:, 0]]
    dv -= box * np.round(dv / box)
    return cand[:, 0], cand[:, 1], np.linalg.norm(dv, axis=1)


def separate_spheres(pos, rad, L, Lz=None, Wt=0.0, max_iter=200, tol=1e-6):
    """Push the last overlaps of a sphere packing out.

    The relaxation stops at a relative overlap of 1e-3 of the smaller
    sphere, and a dense multimodal packing (8, 3 and 1.2 µm spheres at
    60 vol%) kept 23 pairs overlapping by up to 0.015 voxel. Each pass pushes
    every pair overlapping by more than `tol` (relative to the sum of radii)
    apart along its centre line to contact, inversely to the radii; a film
    keeps every sphere between its faces. In a jammed packing a push can
    press a sphere into a third one, so what is left is reported - the
    caller makes it exactly zero by the contact ratio (contact_ratio).
    Returns (positions, pairs left, largest relative overlap left)."""
    Lz = float(L if Lz is None else Lz)
    box = np.array([L, L, Lz])
    pos = pos.copy()
    left, worst = 0, 0.0
    for _ in range(max_iter):
        i, j, d = _sphere_pairs(pos, rad, box)
        need = rad[i] + rad[j]
        m = d < need * (1.0 - tol)
        left = int(m.sum())
        worst = float(np.max(1.0 - d[m] / need[m])) if left else 0.0
        if left == 0:
            break
        i, j, d, need = i[m], j[m], d[m], need[m]
        dv = pos[j] - pos[i]
        dv -= box * np.round(dv / box)
        bad = d < 1e-12 * need
        if bad.any():
            dv[bad] = np.random.default_rng(len(dv)).standard_normal((int(bad.sum()), 3))
            d[bad] = np.linalg.norm(dv[bad], axis=1)
        u = dv / d[:, None]
        gap = (need - d) * (1.0 + 1e-3)
        wi = rad[j] / need                       # the smaller sphere moves more
        disp = np.zeros_like(pos)
        np.add.at(disp, i, -(gap * wi)[:, None] * u)
        np.add.at(disp, j, (gap * (1.0 - wi))[:, None] * u)
        pos += disp
        pos[:, :2] = np.mod(pos[:, :2], L)
        if Wt > 0.0:
            pos[:, 2] = np.clip(pos[:, 2], rad, Wt - rad)
        else:
            pos[:, 2] = np.mod(pos[:, 2], Lz)
    return pos, left, worst


def contact_ratio(pos, rad, L, Lz=None):
    """Smallest centre distance over the sum of radii (inf with no pair in
    reach): the factor all spheres can grow by before two of them meet."""
    Lz = float(L if Lz is None else Lz)
    i, j, d = _sphere_pairs(pos, rad, np.array([L, L, Lz]))
    if d.size == 0:
        return math.inf
    return float(np.min(d / (rad[i] + rad[j])))


def force_biased_spheres(L, d_excl, rng, growth=1.03, max_outer=20000,
                         max_inner=80, log=None, should_stop=None, budget_s=240.0):
    """Periodic polydisperse hard-sphere packing by growth + overlap relaxation.

    Returns (positions, scale); scale < 1 means the target could not be reached
    and the spheres are `scale` times the requested size. The growth factor
    shrinks as the packing approaches jamming, which is what lets it reach
    random close packing instead of oscillating short of it.
    """
    n = len(d_excl)
    pos = rng.random((n, 3)) * L
    scale = 0.70
    stalled = 0
    t_log = t_start = time.time()
    best, best_pos, no_improve = 0.0, pos.copy(), 0
    g = growth
    for outer in range(max_outer):
        if should_stop is not None and outer % 20 == 0 and should_stop():
            raise InterruptedError
        r_cur = 0.5 * scale * d_excl
        rmax = r_cur.max()
        # near jamming the overlaps need many more relaxation sweeps to clear;
        # shrinking after a fixed small number is what stalled v3 short of RCP
        inner = 6 * max_inner
        for _ in range(inner):
            tree = cKDTree(pos, boxsize=L)
            cand = tree.query_pairs(r=2.0 * rmax, output_type="ndarray")
            if cand.size == 0:
                break
            i, j = cand[:, 0], cand[:, 1]
            delta = pos[j] - pos[i]
            delta -= L * np.round(delta / L)
            dist = np.linalg.norm(delta, axis=1)
            need = r_cur[i] + r_cur[j]
            m = dist < need
            if not m.any():
                break
            i, j, delta, dist, need = i[m], j[m], delta[m], dist[m], need[m]
            bad = dist < 1e-12
            if bad.any():
                delta[bad] = rng.standard_normal((int(bad.sum()), 3))
                dist[bad] = np.linalg.norm(delta[bad], axis=1)
            push = (0.5 * (need - dist) / dist)[:, None] * delta
            disp = np.zeros_like(pos)
            np.add.at(disp, i, -push)
            np.add.at(disp, j, push)
            pos = _wrap(pos + disp, L)
        tree = cKDTree(pos, boxsize=L)
        cand = tree.query_pairs(r=2.0 * rmax, output_type="ndarray")
        clean = True
        if cand.size:
            delta = pos[cand[:, 1]] - pos[cand[:, 0]]
            delta -= L * np.round(delta / L)
            dist = np.linalg.norm(delta, axis=1)
            clean = bool(np.all(dist >= r_cur[cand[:, 0]] + r_cur[cand[:, 1]] - 1e-9 * rmax))
        if clean:
            stalled = 0
            if scale > best + 1e-6:
                best, best_pos, no_improve = scale, pos.copy(), 0
            else:
                no_improve += 1
            if scale >= 1.0 - 1e-9:
                return pos, 1.0
            scale = min(scale * g, 1.0)
        else:
            # the last step was too large: go back to the best clean size and
            # try half the growth (bisection on the jamming size)
            # (a few more relaxation rounds first: early on, overlaps from the
            # random start simply need more sweeps)
            stalled += 1
            no_improve += 1
            if stalled >= 3 and best > 0:
                stalled = 0
                g = 1.0 + 0.5 * (g - 1.0)
                scale = min(1.0, best * g)
                if g - 1.0 < 2e-4:
                    break
        if no_improve > 800 or time.time() - t_start > budget_s:
            if log is not None:
                log(f"  !! the target packing is unreachable for these spheres; stopped at size scale {best:.3f}")
            break
        if log is not None and time.time() - t_log > 5.0:
            log(f"    packing: size scale {scale:.3f} (best {best:.3f})")
            t_log = time.time()
    return best_pos, float(best if best > 0 else scale)


def mc_compress(pos, d_excl, L, scale, rng, max_iter=MAX_COMPRESS_ITER, log=None, should_stop=None, Lz=None,
                Wt=0.0):
    """Monte-Carlo compression from an overlap-free packing.

    Alternate hard-sphere Metropolis sweeps (which spread the free volume
    evenly) with a uniform inflation to the first contact (the smallest
    centre distance over contact distance). Every state is overlap-free, and
    the loop is the classical route to random close packing, which the
    force-biased growth approaches but stalls short of.
    Returns (positions, scale). Lz, Wt: an L x L x Lz box with film walls.
    Ends at full size, after 40 rounds without progress or after max_iter
    rounds (of 20 sweeps each) - not on a clock, see clump_pack."""
    Lz = float(L if Lz is None else Lz)
    if _BUDGET is not None:
        max_iter = min(max_iter, _BUDGET[1])
    t_log = time.time()
    stall = 0
    it = 0
    while scale < 1.0 and it < max_iter:
        it += 1
        if should_stop is not None and should_stop():
            raise InterruptedError
        r = 0.5 * scale * d_excl
        _mc_sweeps(pos, r, float(L), 20, 0.02 * float(r.min()), int(rng.integers(1 << 30)), Lz, float(Wt))
        box = np.array([L, L, Lz])
        tree = cKDTree(np.mod(pos, box), boxsize=box)
        # every pair that would touch before full size limits the inflation:
        # search to the full-size contact distance. Searching only to the
        # current one found no pair whenever the sweeps had left a little room
        # everywhere, and the size then jumped to full in one step with the
        # spheres overlapping (equal spheres "reached" 82 vol%; v4 and earlier)
        cand = tree.query_pairs(r=float(d_excl.max()) * (1.0 + 1e-9), output_type="ndarray")
        if cand.size == 0:
            ratio = 1.0 / scale
        else:
            dv = pos[cand[:, 1]] - pos[cand[:, 0]]
            dv -= box * np.round(dv / box)
            ratio = float(np.min(np.linalg.norm(dv, axis=1) / (r[cand[:, 0]] + r[cand[:, 1]])))
        if Wt > 0.0:
            # a sphere may grow until it touches a face as well
            ratio = min(ratio, float(np.min(np.minimum(pos[:, 2], Wt - pos[:, 2]) / r)))
        new = min(1.0, scale * ratio * (1.0 - 1e-9))
        stall = stall + 1 if new - scale < 1e-5 * scale else 0
        scale = max(scale, new)
        if stall > 40:
            break
        if log is not None and time.time() - t_log > 5.0:
            log(f"    Monte-Carlo compression: size scale {scale:.4f}")
            t_log = time.time()
    return pos, float(scale)


# =========================================================================
# rigid clump relaxation (non-spherical dense packing)
# =========================================================================
@njit(cache=True, fastmath=True, parallel=True)
def _relax_step(C, Rm, s, tpl, toff, tcnt, tq, tr, gap, ph, same_ok, rotmode, L, g,
                wpos, wrad, wown, head, nxt, F, T, ncon, max_move, soff, wmax, Lz=0.0, Wt=0.0):
    """One steepest-descent step on the clump overlap energy.

    The box is periodic, L x L x Lz (Lz = L when 0). Wt > 0 is a film: walls
    at z = 0 and z = Wt push every sub-sphere back inside, and the centres do
    not wrap in z (Lz then exceeds Wt by more than a particle, so no
    particle meets another through the z period).

    Runs on all cores: each clump sums the pushes on its own sub-spheres
    (every overlapping pair is evaluated from both sides), so no two threads
    write the same clump and the sums do not depend on the thread count.
    On one core the cost grew with the particle count while the packing's
    time budget did not, and a 10 µm box of 250 nm spheres stopped short of
    the size a 5 µm box reached.
    soff (n + 1) and wmax (n) are work arrays.
    Returns the largest overlap relative to the smaller sub-sphere radius."""
    if Lz <= 0.0:
        Lz = L
    n = C.shape[0]
    soff[0] = 0
    for i in range(n):
        soff[i + 1] = soff[i] + tcnt[tpl[i]]
    m = soff[n]
    for i in prange(n):
        t = tpl[i]
        a = soff[i]
        for k in range(toff[t], toff[t] + tcnt[t]):
            qx = tq[k, 0] * g * s[i]
            qy = tq[k, 1] * g * s[i]
            qz = tq[k, 2] * g * s[i]
            wpos[a, 0] = C[i, 0] + Rm[i, 0, 0] * qx + Rm[i, 1, 0] * qy + Rm[i, 2, 0] * qz
            wpos[a, 1] = C[i, 1] + Rm[i, 0, 1] * qx + Rm[i, 1, 1] * qy + Rm[i, 2, 1] * qz
            wpos[a, 2] = C[i, 2] + Rm[i, 0, 2] * qx + Rm[i, 1, 2] * qy + Rm[i, 2, 2] * qz
            wrad[a] = tr[k] * g * s[i] + 0.5 * gap[i]
            wown[a] = i
            a += 1
    rmax = 0.0
    rsum = 0.0
    for a in range(m):
        rsum += wrad[a]
        if wrad[a] > rmax:
            rmax = wrad[a]
    # Cells of the mean sub-sphere diameter (at least a quarter of the largest
    # radius); each sub-sphere searches as far as its own radius plus the
    # largest one. With cells of the largest diameter, a 5:1 bimodal mix put
    # some 60 fines in every cell and each fine tested 1,600 candidates a
    # step; for equal spheres both are the same.
    rc = max(rsum / m, 0.25 * rmax)
    nc = max(int(L / (2.0 * rc)), 3)
    ncz = max(int(Lz / (2.0 * rc)), 3)
    while nc * nc * ncz > head.shape[0] and (nc > 3 or ncz > 3):
        nc = max(3, (nc * 7) // 8)
        ncz = max(3, (ncz * 7) // 8)
    cs = L / nc
    csz = Lz / ncz
    for c in range(nc * nc * ncz):
        head[c] = -1
    for a in range(m):
        x = wpos[a, 0] % L
        y = wpos[a, 1] % L
        z = wpos[a, 2] % Lz
        c = (int(x / cs) % nc * nc + int(y / cs) % nc) * ncz + int(z / csz) % ncz
        nxt[a] = head[c]
        head[c] = a
    for i in prange(n):
        f0 = 0.0
        f1 = 0.0
        f2 = 0.0
        t0 = 0.0
        t1 = 0.0
        t2 = 0.0
        cnt = 0
        worst_i = 0.0
        pi = ph[i]
        for a in range(soff[i], soff[i + 1]):
            x = wpos[a, 0] % L
            y = wpos[a, 1] % L
            z = wpos[a, 2] % Lz
            cx = int(x / cs) % nc
            cy = int(y / cs) % nc
            cz = int(z / csz) % ncz
            ra0 = wpos[a, 0] - C[i, 0]
            ra1 = wpos[a, 1] - C[i, 1]
            ra2 = wpos[a, 2] - C[i, 2]
            ra0 -= L * math.floor(ra0 / L + 0.5)
            ra1 -= L * math.floor(ra1 / L + 0.5)
            ra2 -= Lz * math.floor(ra2 / Lz + 0.5)
            if Wt > 0.0:
                # the film faces: overlap with a wall pushes the sub-sphere back
                lo = wrad[a] - wpos[a, 2]           # > 0: through the bottom face
                hi = wpos[a, 2] + wrad[a] - Wt      # > 0: through the top face
                for side in range(2):
                    dl = lo if side == 0 else -hi
                    if (side == 0 and lo > 0.0) or (side == 1 and hi > 0.0):
                        rel = abs(dl) / wrad[a]
                        if rel > worst_i:
                            worst_i = rel
                        f2 += dl
                        t0 += (wpos[a, 1] - C[i, 1]) * dl
                        t1 -= (wpos[a, 0] - C[i, 0]) * dl
                        cnt += 1
            # the cells within reach, each once even where the reach spans
            # the whole box
            reach = wrad[a] + rmax
            kx = int(math.ceil(reach / cs - 1e-9))
            kz = int(math.ceil(reach / csz - 1e-9))
            if 2 * kx + 1 >= nc:
                x0 = 0
                nx_ = nc
            else:
                x0 = cx - kx
                nx_ = 2 * kx + 1
            if 2 * kz + 1 >= ncz:
                z0 = 0
                nz_ = ncz
            else:
                z0 = cz - kz
                nz_ = 2 * kz + 1
            y0 = cy - kx if 2 * kx + 1 < nc else 0
            for jx in range(nx_):
                for jy in range(nx_):
                    for jz in range(nz_):
                        b = head[(((x0 + jx) % nc) * nc + (y0 + jy) % nc) * ncz + (z0 + jz) % ncz]
                        while b != -1:
                            ib = wown[b]
                            if ib != i and not (same_ok[pi] == 1 and pi == ph[ib]):
                                dx = wpos[b, 0] - wpos[a, 0]
                                dy = wpos[b, 1] - wpos[a, 1]
                                dz = wpos[b, 2] - wpos[a, 2]
                                dx -= L * math.floor(dx / L + 0.5)
                                dy -= L * math.floor(dy / L + 0.5)
                                dz -= Lz * math.floor(dz / Lz + 0.5)
                                d2 = dx * dx + dy * dy + dz * dz
                                need = wrad[a] + wrad[b]
                                if d2 < need * need:
                                    d = math.sqrt(d2)
                                    if d < 1e-12:
                                        # coincident centres: opposite pushes for the two
                                        dx = 1e-6 if a < b else -1e-6
                                        dy = 0.0
                                        dz = 0.0
                                        d = 1e-6
                                    dl = need - d
                                    rel = dl / min(wrad[a], wrad[b])
                                    if rel > worst_i:
                                        worst_i = rel
                                    # a is pushed away from b
                                    fx = -0.5 * dl * dx / d
                                    fy = -0.5 * dl * dy / d
                                    fz = -0.5 * dl * dz / d
                                    f0 += fx
                                    f1 += fy
                                    f2 += fz
                                    t0 += ra1 * fz - ra2 * fy
                                    t1 += ra2 * fx - ra0 * fz
                                    t2 += ra0 * fy - ra1 * fx
                                    cnt += 1
                            b = nxt[b]
        F[i, 0] = f0
        F[i, 1] = f1
        F[i, 2] = f2
        T[i, 0] = t0
        T[i, 1] = t1
        T[i, 2] = t2
        ncon[i] = cnt
        wmax[i] = worst_i
    worst = 0.0
    for i in range(n):
        if wmax[i] > worst:
            worst = wmax[i]
    # rigid-body update
    for i in prange(n):
        if ncon[i] == 0:
            continue
        w = 1.0 / math.sqrt(ncon[i])
        mx = F[i, 0] * w
        my = F[i, 1] * w
        mz = F[i, 2] * w
        mv = math.sqrt(mx * mx + my * my + mz * mz)
        if mv > max_move[i]:
            f = max_move[i] / mv
            mx *= f
            my *= f
            mz *= f
        C[i, 0] = (C[i, 0] + mx) % L
        C[i, 1] = (C[i, 1] + my) % L
        C[i, 2] = C[i, 2] + mz if Wt > 0.0 else (C[i, 2] + mz) % Lz
        if rotmode[i] < 0:
            continue
        rg = max_move[i] * 4.0              # ~ radius of gyration (max_move = 0.25 rg)
        ax = T[i, 0] * w / (rg * rg)
        ay = T[i, 1] * w / (rg * rg)
        az = T[i, 2] * w / (rg * rg)
        if rotmode[i] == 1 or rotmode[i] == 2:
            # keep the particle axis: spin about it only (mode 2 also about world z)
            zx = Rm[i, 2, 0]
            zy = Rm[i, 2, 1]
            zz = Rm[i, 2, 2]
            sp = ax * zx + ay * zy + az * zz
            nx_ = sp * zx
            ny_ = sp * zy
            nz_ = sp * zz
            if rotmode[i] == 2:
                nz_ += az
            ax = nx_
            ay = ny_
            az = nz_
        ang = math.sqrt(ax * ax + ay * ay + az * az)
        if ang < 1e-12:
            continue
        if ang > 0.08:
            f = 0.08 / ang
            ax *= f
            ay *= f
            az *= f
            ang = 0.08
        ux = ax / ang
        uy = ay / ang
        uz = az / ang
        ca = math.cos(ang)
        sa = math.sin(ang)
        oc = 1.0 - ca
        q00 = ca + ux * ux * oc
        q01 = ux * uy * oc - uz * sa
        q02 = ux * uz * oc + uy * sa
        q10 = uy * ux * oc + uz * sa
        q11 = ca + uy * uy * oc
        q12 = uy * uz * oc - ux * sa
        q20 = uz * ux * oc - uy * sa
        q21 = uz * uy * oc + ux * sa
        q22 = ca + uz * uz * oc
        for r in range(3):
            a0 = Rm[i, r, 0]
            a1 = Rm[i, r, 1]
            a2 = Rm[i, r, 2]
            Rm[i, r, 0] = q00 * a0 + q01 * a1 + q02 * a2
            Rm[i, r, 1] = q10 * a0 + q11 * a1 + q12 * a2
            Rm[i, r, 2] = q20 * a0 + q21 * a1 + q22 * a2
        # re-orthonormalise (Gram-Schmidt)
        nz0 = math.sqrt(Rm[i, 2, 0] ** 2 + Rm[i, 2, 1] ** 2 + Rm[i, 2, 2] ** 2)
        for c in range(3):
            Rm[i, 2, c] /= nz0
        dp = Rm[i, 0, 0] * Rm[i, 2, 0] + Rm[i, 0, 1] * Rm[i, 2, 1] + Rm[i, 0, 2] * Rm[i, 2, 2]
        for c in range(3):
            Rm[i, 0, c] -= dp * Rm[i, 2, c]
        nx0 = math.sqrt(Rm[i, 0, 0] ** 2 + Rm[i, 0, 1] ** 2 + Rm[i, 0, 2] ** 2)
        for c in range(3):
            Rm[i, 0, c] /= nx0
        Rm[i, 1, 0] = Rm[i, 2, 1] * Rm[i, 0, 2] - Rm[i, 2, 2] * Rm[i, 0, 1]
        Rm[i, 1, 1] = Rm[i, 2, 2] * Rm[i, 0, 0] - Rm[i, 2, 0] * Rm[i, 0, 2]
        Rm[i, 1, 2] = Rm[i, 2, 0] * Rm[i, 0, 1] - Rm[i, 2, 1] * Rm[i, 0, 0]
    return worst


def clump_pack(L, C, Rm, s, tpl, templates, gap, ph, same_ok, rotmode, circ,
               log=None, should_stop=None, max_steps=MAX_RELAX_STEPS, tol=0.01, Lz=None, Wt=0.0, g0=0.55,
               stall_window=0, stall_gain=0.0):
    """Grow and relax rigid clumps until they reach full size without overlap.

    templates : list of (centres (k,3), radii (k,)) at nominal size
    circ      : circumradius per particle at nominal size (sets the step size)
    g0        : starting size scale (a packing that is already clean at some
                scale starts there)
    stall_window, stall_gain: also stop once the best size grew by less than
                stall_gain (relative) over the last stall_window rounds - for
                finding where a packing jams, not for building a structure
    Returns (C, Rm, scale) of the best overlap-free state; scale < 1 means the
    target size (and so the target fraction) was not reached.

    The packing ends when it reaches full size, when it stalls, or after
    max_steps relaxation steps - never on a clock. With a time budget the
    result depended on the particle count and on the PC: a 10 µm box of
    250 nm spheres at 55 vol% ran out of time short of the size a 5 µm box
    reached with the same spheres, and the same seed could give another
    structure on a slower or busier PC.
    """
    if _BUDGET is not None:
        max_steps = min(max_steps, _BUDGET[0])
    toff = np.zeros(len(templates), np.int64)
    tcnt = np.zeros(len(templates), np.int64)
    tq_l, tr_l = [], []
    off = 0
    for t, (q, r) in enumerate(templates):
        toff[t], tcnt[t] = off, len(r)
        tq_l.append(np.asarray(q, float).reshape(-1, 3))
        tr_l.append(np.asarray(r, float))
        off += len(r)
    tq = np.concatenate(tq_l)
    tr = np.concatenate(tr_l)
    msub = int(sum(tcnt[t] for t in tpl))
    wpos = np.zeros((msub, 3))
    wrad = np.zeros(msub)
    wown = np.zeros(msub, np.int64)
    rmin = float(np.min([tr[toff[t]:toff[t] + tcnt[t]].max() for t in range(len(templates))]) * s.min())
    Lz = float(L if Lz is None else Lz)
    ncell = max(int(L / (2.0 * rmin * 0.3)), 3) ** 2 * max(int(Lz / (2.0 * rmin * 0.3)), 3)
    head = np.empty(min(ncell, 60_000_000), np.int64)
    nxt = np.empty(msub, np.int64)
    n = len(s)
    soff = np.empty(n + 1, np.int64)
    wmax = np.empty(n)
    F = np.zeros((n, 3))
    T = np.zeros((n, 3))
    ncon = np.zeros(n, np.int64)
    g = float(min(max(g0, 0.05), 1.0))
    best_g, best = 0.0, (C.copy(), Rm.copy())
    growth = 1.02
    t0 = t_log = time.time()
    stuck = 0
    it_total = 0
    hist = []
    # The clumps are kept in the order of the cells they sit in, so that the
    # neighbours a step reads lie close together in memory: in random order
    # nearly every neighbour was a cache miss. perm maps back to the order
    # the caller gave; the sums of every clump do not depend on it.
    perm = np.arange(n)
    cs_sort = 2.0 * float(np.max(circ * s)) + 1e-12
    ncs = max(int(L / cs_sort), 1)
    ncz_s = max(int(Lz / cs_sort), 1)

    def in_caller_order(a):
        out = np.empty_like(a)
        out[perm] = a
        return out
    while True:
        if should_stop is not None and should_stop():
            raise InterruptedError
        cxyz = np.floor(np.mod(C, [L, L, Lz]) / [L / ncs, L / ncs, Lz / ncz_s]).astype(np.int64)
        o = np.argsort((cxyz[:, 0] * ncs + cxyz[:, 1]) * ncz_s + cxyz[:, 2], kind="stable")
        C, Rm = np.ascontiguousarray(C[o]), np.ascontiguousarray(Rm[o])
        s, tpl, gap, ph, rotmode, circ = s[o], tpl[o], gap[o], ph[o], rotmode[o], circ[o]
        perm = perm[o]
        max_move = 0.25 * g * s * circ * 0.5
        clean = False
        for _ in range(300):
            w = _relax_step(C, Rm, s, tpl, toff, tcnt, tq, tr, gap, ph, same_ok, rotmode, float(L), g,
                            wpos, wrad, wown, head, nxt, F, T, ncon, max_move, soff, wmax, Lz, float(Wt))
            it_total += 1
            if w < tol:
                clean = True
                break
        if clean:
            # a clean state counts as progress only if it beats the best size;
            # near jamming the size otherwise oscillates just below the best
            # (5 spheres in a 2-diameter box sat at 0.946-0.947 until the time
            # budget ran out), which is a stall as much as a failed step
            if g > best_g * (1.0 + 1e-4):
                stuck = 0
            else:
                stuck += 1
            if g > best_g:
                best_g, best = g, (in_caller_order(C), in_caller_order(Rm))
            if g >= 1.0:
                break
            g = min(1.0, g * growth)
        else:
            stuck += 1
            g = max(best_g, g / (1.0 + 0.5 * (growth - 1.0)))
            growth = max(1.001, 1.0 + 0.7 * (growth - 1.0))
        if stuck > 60:
            break
        hist.append(best_g)
        if stall_window and len(hist) > stall_window and                 hist[-1] - hist[-1 - stall_window] < stall_gain * max(hist[-1], 1e-12):
            break
        if it_total >= max_steps:
            if log is not None:
                log(f"    clump relaxation: step limit ({max_steps:,}) reached at size scale {best_g:.4f}")
            break
        if log is not None and time.time() - t_log > 5.0:
            log(f"    clump relaxation: size scale {g:.3f} (best {best_g:.3f}, {it_total} steps)")
            t_log = time.time()
    return best[0], best[1], float(best_g)


# =========================================================================
# network phase
# =========================================================================
def random_field(shape, h, pore_size, rng):
    """Band-limited periodic Gaussian field; thresholding it gives a
    bicontinuous structure whose features are ~pore_size wide."""
    from scipy import fft as sfft
    nx, ny, nz = shape
    lam = 2.0 * pore_size
    k0 = 2.0 * math.pi / lam
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=h)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=h)
    kz = 2.0 * math.pi * np.fft.rfftfreq(nz, d=h)
    K = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    filt = np.exp(-0.5 * ((K - k0) / (0.3 * k0)) ** 2)
    filt[0, 0, 0] = 0.0
    from .solvers.conduction import _threads
    w = _threads()
    F = sfft.rfftn(rng.standard_normal(shape), workers=w)
    F *= filt
    return sfft.irfftn(F, s=shape, workers=w)


# =========================================================================
# orchestration
# =========================================================================
def _phase_arrays(phases, contacts):
    nph = len(phases)
    overlap_of = np.zeros(nph + 1, np.uint8)
    net_of = np.zeros(nph + 1, np.uint8)
    sep1_of = np.zeros(nph + 1, np.uint8)
    for i, ph in enumerate(phases):
        net_of[i] = 1 if ph["shape"] == G.NETWORK else 0
        overlap_of[i] = 1 if ph.get("overlap") else 0
        sep1_of[i] = 1 if (contacts in ("apart", "separate") and not ph.get("overlap")) else 0
    return overlap_of, net_of, sep1_of


def _rotmode(mode, spread):
    """-1 no rotation needed (sphere), 0 free, 1 keep the axis, 2 keep it in the xy plane."""
    if mode == "iso":
        return 0
    return 2 if mode == "xy" else 1


def phase_templates(phases, rng):
    """Nominal (code, P) per phase, plus the shared polyhedron plane table."""
    table = SH.PolyTable()
    nominal = []
    for ph in phases:
        if ph["shape"] == G.NETWORK:
            nominal.append(None)
            continue
        size = ph["size_um"]
        ids = None
        if ph["shape"] == "polyhedron":
            kind = size.get("poly", "icosahedron")
            ids = table.add(kind, n_vertices=int(size.get("n_vertices", 14)), rng=rng,
                            n_variants=int(size.get("variants", 8)))
        code, P = SH.params(ph["shape"], size, ids)
        nominal.append({"code": code, "P": P, "poly_ids": ids})
    planes, nfaces = table.arrays()
    return nominal, planes, nfaces, table.meta


def _draw(ph, nom, n, rng, poly_meta):
    """n particles of one phase: code, scaled P, rotation, scale factors."""
    s = G.sample_scales(n, ph.get("dist"), rng)
    ori = ph.get("orientation") or {}
    Rs = SH.sample_rotations(n, ori.get("mode", "iso"), ori.get("spread_deg", 0.0), rng)
    Ps = np.tile(nom["P"], (n, 1))
    lin = _linear_slots(nom["code"])
    Ps[:, lin] *= s[:, None]
    if nom["code"] == 6 and nom["poly_ids"] and len(nom["poly_ids"]) > 1:
        Ps[:, 1] = rng.choice(nom["poly_ids"], n)
    return Ps, Rs, s


def _linear_slots(code):
    """Indices of P that are lengths (scale with the particle size)."""
    return {0: [0], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [0, 1, 2], 5: [0, 1, 2],
            6: [0], 7: [0, 1, 2]}[code]


def rasterize(N, h, info, phases, nz=None):
    """The particles of a generated structure drawn on another grid: the same
    centres, sizes and orientations, in the order they were placed (the first
    particle keeps a shared voxel, as in generation). Returns labels with
    0 = matrix and i + 1 = phase i, or None when a network phase is present
    (a random field is made on its own grid and has no particle list)."""
    if any(ph["shape"] == G.NETWORK and ph.get("vf", 0) > 0 for ph in phases):
        return None
    owner, phase_of = rasterize_owner(N, h, info, phases, nz)
    lut = np.zeros(len(phase_of), np.uint8)
    valid = phase_of >= 0
    lut[valid] = (phase_of[valid] + 1).astype(np.uint8)
    return lut[owner]


def rasterize_owner(N, h, info, phases, nz=None, offset_um=0.0):
    """The particle each voxel belongs to (0 where none is), and the phase of
    every particle number, on a grid of N x N x nz voxels of size h. Network
    phases have no particle list and stay 0 here. The viewer draws a separate
    surface per particle from this, so that touching particles of one phase
    are not merged into one skin.
    offset_um: the particles are drawn this much towards the origin, so that
    a coarse grid samples the points another grid's pick does."""
    parts = info["particles"]
    planes, nfaces = info["planes"]
    n = len(parts["codes"])
    film = bool(info.get("film"))
    if nz is None:
        nz = N if not film else int(round(info["T_um"] / h))
    owner = np.zeros((N, N, nz), np.int32)
    phase_of = np.full(n + 2, -1, np.int32)
    if n:
        overlap_of, net_of, _ = _phase_arrays(phases, "keep")
        ext = max(2.0 * SH.circumradius(int(c), P) for c, P in zip(parts["codes"], parts["P"]))
        buf = np.empty(int(math.ceil(ext / h) + 6) ** 3, np.int64)
        raster_all(owner, phase_of, overlap_of, net_of, float(h), parts["codes"].astype(np.int64),
                   np.ascontiguousarray(parts["centers"] - float(offset_um)), np.ascontiguousarray(parts["R"]),
                   np.ascontiguousarray(parts["P"]), planes, nfaces, np.zeros(n),
                   parts["phase"].astype(np.int64), 1, buf, film)
    return owner, phase_of


def generate(N, h, phases, seed=0, log=print, should_stop=None, contacts="apart",
             want_contact_faces=False, nz=None, skin=0):
    """Build one realisation.

    `phases` items: shape, size_um, dist, vf, orientation {mode, spread_deg},
    overlap (bool), gap_frac. Returns (labels uint8 with 0 = matrix and i+1 =
    phase i, info dict).

    nz: a film of nz voxels in z (x and y stay periodic, N voxels each). Every
    particle then lies wholly inside 0 <= z <= nz h - none is cut by the
    faces or wraps through them. The packings run in a box whose z period
    exceeds the film by more than the largest particle, with walls at the two
    faces; RSA rejects a position that crosses a face.

    skin: voxels of matrix kept at each face of a film. A particle that
    touches a face touches the electrode plate there, through a contact whose
    area is set by the voxel size: on one 40 vol% film, 0.6 % of the face
    voxels in particles raised the through-thickness k by 25 %, and the
    contact vanished (with the excess) on a 1.5x finer grid. The film is
    built inside the skin and padded with matrix, for every packing route.
    """
    if nz is not None and skin and int(nz) - 2 * int(skin) >= 2:
        ks = int(skin)
        inner = int(nz) - 2 * ks
        grow = float(nz) / inner                       # the fractions refer to the whole film
        lab, info = generate(N, h, [dict(ph, vf=float(ph["vf"]) * grow) for ph in phases], seed=seed,
                             log=log, should_stop=should_stop, contacts=contacts,
                             want_contact_faces=want_contact_faces, nz=inner, skin=0)
        pad = ((0, 0), (0, 0), (ks, ks))
        lab = np.pad(lab, pad)
        if info.get("contact_mask") is not None:
            cm = info["contact_mask"]
            info["contact_mask"] = np.pad(cm, pad + ((0, 0),) * (cm.ndim - 3))
        parts = info.get("particles")
        if parts is not None and len(parts["centers"]):
            parts["centers"] = parts["centers"] + np.array([0.0, 0.0, ks * h])
        for pi in info["phases"]:
            for k in ("vf_target", "vf_voxel"):
                if k in pi:
                    pi[k] = float(pi[k]) / grow
        info["nz"], info["T_um"], info["skin_um"] = int(nz), int(nz) * h, ks * h
        log(f"  matrix skin of {ks} voxel(s) ({ks * h:.3g} µm) at each face of the film")
        return lab, info
    t0 = time.time()
    L = N * h
    film = nz is not None
    nz = int(nz) if film else N
    T = nz * h                           # film thickness (= L for a cube)
    V = L * L * T
    rng = np.random.default_rng(seed)
    shape = (N, N, nz)
    nvox = N * N * nz
    box = np.array([L, L, T])
    owner = np.zeros(shape, np.int32)
    nph = len(phases)
    overlap_of, net_of, sep1_of = _phase_arrays(phases, contacts)
    target = np.array([int(round(ph["vf"] * nvox)) for ph in phases] + [0], np.int64)
    count = np.zeros(nph + 1, np.int64)
    n_placed = np.zeros(nph + 1, np.int64)
    jammed = np.zeros(nph + 1, np.bool_)
    info = {"route": None, "phases": [dict() for _ in phases], "contacts_mode": contacts}
    nominal, planes, nfaces, poly_meta = phase_templates(phases, rng)
    info["poly_meta"] = poly_meta

    part_idx = [i for i, ph in enumerate(phases) if ph["shape"] != G.NETWORK and target[i] > 0]
    mean_vol = {}
    for i in part_idx:
        s = G.sample_scales(4000, phases[i].get("dist"), np.random.default_rng(7))
        nom = nominal[i]
        lin = _linear_slots(nom["code"])
        vols = []
        for sc in s[:400]:
            P = nom["P"].copy()
            P[lin] *= sc
            vols.append(SH.volume(nom["code"], P, poly_meta))
        mean_vol[i] = float(np.mean(vols))
    cap = 2 + sum(int(3.0 * phases[i]["vf"] * V / mean_vol[i]) + 64 for i in part_idx)
    phase_of = np.full(cap + nph + 16, -1, np.int32)
    pid = 1
    conflicts = 0

    parts = {"codes": [], "P": [], "R": [], "centers": [], "phase": []}

    def keep(codes, Ps, Rs, cen, pha):
        parts["codes"].append(np.asarray(codes, np.int64))
        parts["P"].append(np.asarray(Ps, float))
        parts["R"].append(np.asarray(Rs, float))
        parts["centers"].append(np.asarray(cen, float))
        parts["phase"].append(np.asarray(pha, np.int64))

    def buffer_for(Ps, codes):
        ext = max(2.0 * SH.circumradius(int(c), P) for c, P in zip(codes, Ps)) if len(codes) else h
        return np.empty(int(math.ceil(ext / h) + 4) ** 3, np.int64)

    sphere_route = (bool(part_idx) and all(phases[i]["shape"] == "sphere" and not phases[i].get("overlap")
                                           for i in part_idx))
    if sphere_route:
        info["route"] = "growth with overlap relaxation + Monte-Carlo (spheres)"
        diam, ph_of, d_ex = [], [], []
        for i in part_idx:
            ph = phases[i]
            d0 = float(ph["size_um"]["d"])
            n = max(1, int(round(ph["vf"] * V / mean_vol[i])))
            s = G.sample_scales(n, ph.get("dist"), rng)
            s *= (ph["vf"] * V / np.sum(math.pi * (d0 * s) ** 3 / 6.0)) ** (1.0 / 3.0)
            d = d0 * s
            gap = float(ph.get("gap_frac", 0.0) or 0.0) * d
            if contacts == "separate":
                gap = np.maximum(gap, 0.5 * h)
            diam.append(d)
            ph_of.append(np.full(n, i))
            d_ex.append(d + gap)
        diam = np.concatenate(diam)
        ph_arr = np.concatenate(ph_of).astype(np.int64)
        d_base = np.concatenate(d_ex)
        n = len(diam)
        if film:
            # A film may hold only two or three particles across its
            # thickness. The film, not the size distribution, then sets the
            # largest size: a particle the film cannot hold (with its own gap)
            # is made as thick as the film allows, and the others of its phase
            # grow slightly so that the phase keeps its volume fraction.
            room = T - (d_base - diam)
            over = diam > room
            if over.all():
                raise ValueError(f"particles of {float(diam.min()):.4g} µm do not fit in a film of {T:.4g} µm")
            if over.any():
                d_new = diam.copy()
                for i in np.unique(ph_arr[over]):
                    m = ph_arr == i
                    v_want = float(np.sum(diam[m] ** 3))
                    d_i = np.minimum(diam[m], room[m])
                    for _ in range(50):
                        free = d_i < room[m] * (1.0 - 1e-12)
                        rest = v_want - float(np.sum(d_i[~free] ** 3))
                        if not free.any() or rest <= 0.0:
                            break
                        d_i[free] = np.minimum(d_i[free] * (rest / float(np.sum(d_i[free] ** 3))) ** (1.0 / 3.0),
                                               room[m][free])
                        if abs(float(np.sum(d_i ** 3)) / v_want - 1.0) < 1e-9:
                            break
                    d_new[m] = d_i
                log(f"  {int(over.sum())} of {n} particles are larger than the {T:.4g} µm film can hold; they are drawn "
                    f"{float(room[over].min()):.4g} µm across, the others of their phase slightly larger to keep the fraction")
                d_base = d_base - diam + d_new
                diam = d_new
        if contacts == "apart":
            g_sep = apart_gap(math.pi / 6.0 * diam ** 3, d_base - diam, h, V, "sphere")
            d_excl = diam + g_sep
            g_med = float(np.median(g_sep)) / h
            if g_med < SEP_APART - 1e-6:
                log(f"  a {SEP_APART:g}-voxel gap between all particles does not fit at this loading; "
                    f"particles kept {g_med:.2f} voxels apart where the packing allows")
        else:
            d_excl = d_base.copy()
        # Growth with overlap relaxation, run by the compiled clump engine with
        # one sphere per clump. Measured against the vectorised force-biased
        # loop of v3 on 250 spheres: 60 % in 3.5 s instead of 21 s, and 0.997 of
        # the size at 64 % (random close packing) instead of 0.971.
        log(f"  packing {n:,} spherical particles (growth with overlap relaxation)")
        # a film packs in a z period longer than the film by more than a
        # particle, with walls at the faces: nothing meets through the period
        Lz_pack = T + 1.5 * float(d_excl.max()) if film else T
        Wt = T if film else 0.0
        if film:
            # the separation gap is wanted, not required: a particle as thick
            # as the film keeps what gap the film leaves it
            d_excl = np.minimum(d_excl, T)
            Lz_pack = T + 1.5 * float(d_excl.max())
        C0 = rng.random((n, 3)) * box
        if film:
            C0[:, 2] = 0.5 * d_excl + rng.random(n) * (T - d_excl)
        pos, _, scale = clump_pack(float(L), C0, np.tile(np.eye(3), (n, 1, 1)), d_excl.copy(),
                                   np.zeros(n, np.int64), [(np.zeros((1, 3)), np.array([0.5]))], np.zeros(n),
                                   ph_arr, overlap_of.astype(np.int64), np.full(n, -1, np.int64), np.full(n, 0.5),
                                   log=log, should_stop=should_stop, tol=1e-3, Lz=Lz_pack, Wt=Wt)
        pos = np.ascontiguousarray(pos)
        if scale < 1.0 and contacts == "apart" and np.any(d_excl > d_base):
            # The separation gap does not fit at this loading. Keep the gap the
            # packing did reach (a uniform fraction of it) and let the rest
            # close: particles touch only where the target needs it.
            d_prev = scale * d_excl                    # clean at these sizes
            d_excl = np.maximum(d_base, d_prev)
            left = float(np.median((d_excl - diam) / h))
            log(f"  the {SEP_APART:g}-voxel separation does not fit at this loading (size scale {scale:.4f}); "
                f"median gap kept {max(left, 0.0):.2f} voxels, contacts allowed where needed")
            if np.all(d_excl > d_base):
                scale = 1.0
            else:
                pos, _, scale = clump_pack(float(L), pos, np.tile(np.eye(3), (n, 1, 1)), d_excl.copy(),
                                           np.zeros(n, np.int64), [(np.zeros((1, 3)), np.array([0.5]))],
                                           np.zeros(n), ph_arr, overlap_of.astype(np.int64),
                                           np.full(n, -1, np.int64), np.full(n, 0.5),
                                           log=log, should_stop=should_stop, tol=1e-3,
                                           Lz=Lz_pack, Wt=Wt, g0=0.999 * float(np.min(d_prev / d_excl)))
                pos = np.ascontiguousarray(pos)
        if scale < 1.0:
            s0 = scale
            # the relaxation leaves a tolerance's worth of overlap; start the
            # compression from a strictly overlap-free size
            scale *= 0.999
            pos, scale = mc_compress(pos, d_excl, L, scale, rng, log=log, should_stop=should_stop,
                                     Lz=Lz_pack, Wt=Wt)
            log(f"  growth stopped at size scale {s0:.4f}; Monte-Carlo compression reached {scale:.4f}")
        if scale < 1.0:
            log(f"  !! the target packing fraction was not reached; packing stopped at a size scale of {scale:.3f}")
            diam *= scale
            d_excl *= scale
        pos, left, worst = separate_spheres(pos, 0.5 * d_excl, float(L), float(Lz_pack), float(Wt))
        if worst > 1e-3:
            log(f"  !! {left} sphere pairs still overlap after the separation (up to {100 * worst:.2f} % of the "
                f"contact distance); the spheres are drawn smaller by that much so that none overlaps")
        sweeps = int(min(3000, max(200, 3.0e6 / n)))
        rate = _mc_sweeps(pos, 0.5 * d_excl, float(L), sweeps, 0.05 * float(d_excl.min()),
                          int(rng.integers(1 << 30)), float(Lz_pack), float(Wt))
        log(f"  Monte-Carlo equilibration: {sweeps} sweeps (acceptance {100*rate:.0f} %)")
        info["pack_scale"] = scale
        codes = np.zeros(n, np.int64)
        Rs = np.tile(np.eye(3), (n, 1, 1))
        buf = np.empty((int(math.ceil(diam.max() / h)) + 6) ** 3, np.int64)
        want = float(sum(target[i] for i in part_idx))
        # the radius correction below may grow the spheres only until two of
        # them would touch: never into each other
        f_cap = contact_ratio(pos, 0.5 * diam, float(L), float(Lz_pack)) * (1.0 - 1e-9)
        f = min(1.0, f_cap)
        # A sphere a few voxels across rasterises to less than its analytic
        # volume; scale the radius until the voxel fraction (what the solvers
        # see) matches the target.
        for it in range(8):
            owner[owner >= pid] = 0
            Ps = np.zeros((n, 6))
            Ps[:, 0] = 0.5 * diam * f
            _, conf = raster_all(owner, phase_of, overlap_of, net_of, float(h), codes, pos, Rs, Ps,
                                 planes, nfaces, np.zeros(n), ph_arr, pid, buf, film)
            got = float(np.count_nonzero(owner >= pid))
            if want <= 0 or abs(got / want - 1.0) < 0.002:
                break
            f *= (want / max(got, 1.0)) ** (1.0 / 3.0)
            # when the packing fell short, growing the radii to make up the
            # fraction would push the spheres into each other: only the
            # discretisation correction is allowed then
            f = min(max(f, 0.75), 1.35 if scale >= 1.0 else 1.0, max(f_cap, 0.75))
        conflicts = int(conf)
        info["radius_factor"] = f
        if want > 0 and got < 0.998 * want and f >= f_cap * (1.0 - 1e-6) and f_cap < 1.349:
            log(f"  !! the voxel fraction is {100 * (1 - got / want):.2f} % short of the target: growing the radii "
                f"to make it up would push touching spheres into each other (a finer voxel size closes it)")
        if f >= 1.349 or f <= 0.751:
            log(f"  !! the voxel volume fraction could not be matched (radius factor {f:.3f}); "
                f"the voxel size is too coarse for these particles")
        for m in range(n):
            n_placed[ph_arr[m]] += 1
        keep(codes, Ps, Rs, pos, ph_arr)
        pid += n
    elif part_idx:
        # ---- 1. voxel RSA ---------------------------------------------------
        info["route"] = "voxel random sequential addition (arbitrary shapes)"
        rounds = 0
        remaining = {i: True for i in part_idx}
        rsa_parts = {"codes": [], "P": [], "R": [], "centers": [], "phase": []}
        while remaining and rounds < 6:
            if should_stop is not None and should_stop():
                raise InterruptedError
            rounds += 1
            cl, Pl, Rl, gl, phl = [], [], [], [], []
            for i in list(remaining):
                ph = phases[i]
                deficit = max(target[i] - count[i], 0) * h ** 3
                n = int(math.ceil(1.3 * deficit / mean_vol[i])) + 4
                Ps, Rs, s = _draw(ph, nominal[i], n, rng, poly_meta)
                gap = float(ph.get("gap_frac", 0.0) or 0.0) * np.array(
                    [2.0 * SH.circumradius(nominal[i]["code"], P) for P in Ps])
                cl.append(np.full(n, nominal[i]["code"]))
                Pl.append(Ps)
                Rl.append(Rs)
                gl.append(gap)
                phl.append(np.full(n, i))
            codes = np.concatenate(cl).astype(np.int64)
            Ps = np.concatenate(Pl)
            Rs = np.ascontiguousarray(np.concatenate(Rl))
            gap = np.concatenate(gl)
            ph_arr = np.concatenate(phl).astype(np.int64)
            vol = np.array([SH.volume(int(c), P, poly_meta) for c, P in zip(codes, Ps)])
            order = np.argsort(-vol, kind="stable")
            codes, Ps, Rs, gap, ph_arr = codes[order], Ps[order], Rs[order], gap[order], ph_arr[order]
            buf = buffer_for(Ps, codes)
            need = pid + len(codes) + 8
            if need > len(phase_of):
                grown = np.full(int(need * 1.5), -1, np.int32)
                grown[:len(phase_of)] = phase_of
                phase_of = grown
            centers = np.zeros((len(codes), 3))
            placed_id = np.zeros(len(codes), np.int64)
            pid = _rsa(owner, phase_of, overlap_of, net_of, sep1_of, float(h), float(L), codes, Rs, Ps,
                       planes, nfaces, gap, ph_arr, target, count, n_placed, jammed, centers, placed_id,
                       pid, 400, 25, int(rng.integers(1 << 30)), buf, film)
            ok = placed_id > 0
            for key, arr in (("codes", codes), ("P", Ps), ("R", Rs), ("centers", centers), ("phase", ph_arr)):
                rsa_parts[key].append(arr[ok])
            for i in list(remaining):
                if jammed[i] or count[i] >= target[i]:
                    del remaining[i]
            log(f"  RSA round {rounds}: " + ", ".join(
                f"{phases[i].get('name', i+1)} {100*count[i]/nvox:.2f}/{100*target[i]/nvox:.2f} %"
                for i in part_idx))
        short = [i for i in part_idx if not phases[i].get("overlap") and count[i] < 0.995 * target[i]]
        if not short:
            for key in parts:
                parts[key].extend(rsa_parts[key])
        else:
            # ---- 2. rigid clump relaxation ------------------------------------
            log("  RSA jammed below the target for " + ", ".join(phases[i].get("name", str(i + 1)) for i in short)
                + " -> dense packing by rigid clump relaxation")
            info["route"] = "rigid clump relaxation (multi-sphere, translation + rotation)"
            owner[:] = 0
            count[:] = 0
            n_placed[:] = 0
            jammed[:] = False
            pid = 1
            cl, Pl, Rl, sl, tpl_l, phl, gl, rot_l, circ_l = [], [], [], [], [], [], [], [], []
            templates, tkey = [], {}
            for i in part_idx:
                ph = phases[i]
                nom = nominal[i]
                n = max(1, int(round(ph["vf"] * V / mean_vol[i])))
                Ps, Rs, s = _draw(ph, nom, n, rng, poly_meta)
                # hit the target volume exactly with the drawn sizes
                v = np.array([SH.volume(nom["code"], P, poly_meta) for P in Ps])
                corr = (ph["vf"] * V / v.sum()) ** (1.0 / 3.0)
                lin = _linear_slots(nom["code"])
                Ps[:, lin] *= corr
                s = s * corr
                tids = []
                for P in Ps:
                    t_id = int(P[1]) if nom["code"] == 6 else -1
                    key = (i, t_id)
                    if key not in tkey:
                        Pn = nom["P"].copy()
                        if nom["code"] == 6:
                            Pn[1] = t_id
                        q, r = SH.clump_template(nom["code"], Pn, planes, nfaces, max_spheres=40, coverage=0.98)
                        tkey[key] = len(templates)
                        templates.append((q, r))
                    tids.append(tkey[key])
                gap_i = float(ph.get("gap_frac", 0.0) or 0.0) * 2.0 * SH.circumradius(nom["code"], nom["P"]) * s
                ori = ph.get("orientation") or {}
                rm = -1 if nom["code"] == 0 else _rotmode(ori.get("mode", "iso"), ori.get("spread_deg", 0))
                cl.append(np.full(n, nom["code"]))
                Pl.append(Ps)
                Rl.append(Rs)
                sl.append(s)
                tpl_l.append(np.array(tids))
                phl.append(np.full(n, i))
                gl.append(gap_i)
                rot_l.append(np.full(n, rm))
                circ_l.append(np.full(n, SH.circumradius(nom["code"], nom["P"])))
            codes = np.concatenate(cl).astype(np.int64)
            Ps = np.concatenate(Pl)
            Rs = np.ascontiguousarray(np.concatenate(Rl))
            s = np.concatenate(sl)
            tpl = np.concatenate(tpl_l).astype(np.int64)
            ph_arr = np.concatenate(phl).astype(np.int64)
            gap = np.concatenate(gl)
            rotm = np.concatenate(rot_l).astype(np.int64)
            circ = np.concatenate(circ_l)
            C = rng.random((len(codes), 3)) * box
            # the film: pack in a z period one particle longer than the film,
            # with walls at the faces; start every centre inside the film
            ext_max = float(max(2.0 * c_ * s_ for c_, s_ in zip(circ, s))) if len(s) else 0.0
            Lz_pack = T + 1.5 * ext_max if film else T
            Wt = T if film else 0.0
            if film:
                half_ext = 0.5 * circ * s
                if float(2.0 * half_ext.max()) > T:
                    log(f"  !! the largest particle ({float(2 * half_ext.max()):.4g} µm across) is thicker than the "
                        f"film ({T:.4g} µm); it can only fit if oriented flat")
                C[:, 2] = np.minimum(half_ext, 0.5 * T) + rng.random(len(codes)) * np.maximum(T - 2.0 * half_ext, 0.0)
            log(f"  clump packing of {len(codes):,} particles ({len(templates)} clump templates, "
                f"{sum(len(templates[t][1]) for t in tpl):,} spheres)")
            gap_user = gap.copy()
            if contacts == "apart":
                vols = np.array([SH.volume(int(c), P, poly_meta) for c, P in zip(codes, Ps)])
                gap = apart_gap(vols, gap_user, h, V, "other")
                g_med = float(np.median(gap)) / h
                if g_med < SEP_APART - 1e-6:
                    log(f"  a {SEP_APART:g}-voxel gap between all particles does not fit at this loading; "
                        f"particles kept {g_med:.2f} voxels apart where the packing allows")
            C, Rs, scale = clump_pack(float(L), C, Rs, s, tpl, templates, gap, ph_arr,
                                      overlap_of.astype(np.int64), rotm, circ, log=log, should_stop=should_stop,
                                      Lz=Lz_pack, Wt=Wt)
            if scale < 1.0 and np.any(gap > gap_user):
                # the separation gap does not fit: continue from this state
                # without it, so particles touch only where the target needs it
                log(f"  the {SEP_APART:g}-voxel separation does not fit at this loading (size scale "
                    f"{scale:.4f}); packing on with contacts allowed where needed")
                C, Rs, scale = clump_pack(float(L), C, Rs, s, tpl, templates, gap_user, ph_arr,
                                          overlap_of.astype(np.int64), rotm, circ, log=log,
                                          should_stop=should_stop, Lz=Lz_pack, Wt=Wt)
            info["pack_scale"] = scale
            if scale < 1.0:
                log(f"  !! the target could not be packed without overlap; particles are {scale:.3f} x the "
                    f"requested size (volume fraction x {scale**3:.3f})")
            for c in set(codes.tolist()):
                sel = codes == c
                Ps[np.ix_(sel, _linear_slots(c))] *= scale
            buf = buffer_for(Ps, codes)
            need = pid + len(codes) + 8
            if need > len(phase_of):
                grown = np.full(int(need * 1.5), -1, np.int32)
                grown[:len(phase_of)] = phase_of
                phase_of = grown
            owner, conflicts, rounds, touch, shr = repair_overlaps(
                owner, phase_of, overlap_of, float(h), codes, C, Rs, Ps, planes, nfaces, ph_arr, pid, log=log,
                net_of=net_of, apart=contacts == "apart", zwall=film, rotmode=rotm)
            log(f"  exact-shape overlap repair (push and turn): {rounds} round(s), {conflicts:,} shared voxels left"
                + (f", {touch:,} touching voxel pairs" if touch >= 0 else ""))
            info["shrunk"] = shr
            if shr["n"]:
                log(f"  !! {shr['n']} of {len(codes)} particles were caught in each other at their corners and "
                    f"were shrunk (shape kept) to clear the overlap: to {100 * shr['min']:.0f} % of their size at "
                    f"most, {100 * shr['mean']:.1f} % on average")
            pid += len(codes)
            for i in part_idx:
                n_placed[i] = int(np.count_nonzero(ph_arr == i))
            keep(codes, Ps, Rs, C, ph_arr)
            if conflicts:
                log(f"  !! {conflicts:,} voxels claimed by two particles (corners the clumps do not cover) "
                    f"were kept by the first particle")
    else:
        info["route"] = "random-field network" if any(net_of[:nph]) else "matrix only"
    info["voxel_conflicts"] = int(conflicts)

    # ---- v3-style separation (option) ------------------------------------
    removed = 0
    if contacts == "separate" and part_idx and any(not phases[i].get("overlap") for i in part_idx):
        removed = int(_cleanup(owner, phase_of, overlap_of, net_of, _FWD26))
        if removed:
            log(f"  {removed:,} particle contacts separated by matrix voxels (contacts = separate)")
    info["contacts_removed"] = removed

    # ---- contacts (geometry kept) -----------------------------------------
    cf = None
    if part_idx:
        cf = np.zeros(shape, np.uint8)
        n_contact = int(contact_faces(owner, phase_of, net_of, cf, not film))
        info["contact_faces"] = n_contact
        n_touch = touching_pairs(owner, phase_of, net_of, not film)
        info["touching_pairs"] = n_touch
        if n_touch and contacts != "separate":
            log(f"  particles touch at {n_touch:,} voxel neighbour pairs, {n_contact:,} of them across a face "
                f"(geometry kept; a contact resistance can be set on the faces)")
        elif contacts == "apart":
            log("  no two particles touch (no voxel of one is a neighbour of a voxel of another)")
        if not want_contact_faces:
            cf = None
    info["contact_mask"] = cf

    # ---- network phases in the remaining space -----------------------------
    for i, ph in enumerate(phases):
        if ph["shape"] != G.NETWORK or target[i] == 0:
            continue
        if should_stop is not None and should_stop():
            raise InterruptedError
        width = float(ph["size_um"]["d"])
        field = random_field(shape, h, width, rng)
        free = owner == 0
        vals = field[free]
        if vals.size == 0:
            continue
        kth = min(int(target[i]), vals.size) - 1
        thr = np.partition(vals, kth)[kth]
        mask = free & (field <= thr)
        del field, vals, free
        phase_of[pid] = i
        owner[mask] = pid
        count[i] = int(mask.sum())
        n_placed[i] = 1
        pid += 1
        log(f"  bicontinuous network '{ph.get('name', i+1)}': "
            f"{100*count[i]/nvox:.2f} vol % (characteristic width {width:g} µm)")

    # ---- labels and statistics ----------------------------------------------
    lut = np.zeros(len(phase_of), np.uint8)
    valid = phase_of >= 0
    lut[valid] = (phase_of[valid] + 1).astype(np.uint8)
    lut[0] = 0
    labels = lut[owner]
    del owner

    counts = np.bincount(labels.ravel(), minlength=nph + 1)
    cat = lambda lst, shp: (np.concatenate(lst) if lst else np.zeros(shp))
    p_codes = cat(parts["codes"], (0,)).astype(np.int64)
    p_P = cat(parts["P"], (0, 6))
    p_R = cat(parts["R"], (0, 3, 3))
    p_phase = cat(parts["phase"], (0,)).astype(np.int64)
    p_centers = cat(parts["centers"], (0, 3))

    for i, ph in enumerate(phases):
        pi = info["phases"][i]
        pi["name"] = ph.get("name", f"phase {i+1}")
        pi["vf_target"] = float(ph["vf"])
        pi["vf_voxel"] = float(counts[i + 1] / nvox)
        pi["n_particles"] = int(n_placed[i])
        pi["jammed"] = bool(jammed[i])
        sel = p_phase == i
        if ph["shape"] != G.NETWORK and sel.any():
            pi["surface_true_um2"] = float(sum(SH.surface(int(c), P, poly_meta)
                                               for c, P in zip(p_codes[sel], p_P[sel])))
        else:
            pi["surface_true_um2"] = None
        m = labels == (i + 1)
        faces = 0
        for ax in range(3):
            faces += int(np.count_nonzero(m != np.roll(m, -1, axis=ax)))
        pi["surface_voxel_um2"] = faces * h * h
        hN, hZ = N // 2, max(nz // 2, 1)
        occ = [float(m[x0:x0 + hN, y0:y0 + hN, z0:z0 + hZ].mean())
               for x0 in (0, hN) for y0 in (0, hN) for z0 in (0, hZ)]
        mean_o = float(np.mean(occ))
        pi["octant_cov"] = float(np.std(occ) / mean_o) if mean_o > 0 else 0.0
        del m

    info["particles"] = {"codes": p_codes, "P": p_P, "R": p_R, "phase": p_phase, "centers": p_centers,
                         "axes": p_R[:, 2, :] if len(p_R) else np.zeros((0, 3))}
    info["planes"] = (planes, nfaces)
    info["seconds"] = time.time() - t0
    info["N"], info["h_um"], info["L_um"] = N, h, L
    info["nz"], info["film"], info["T_um"] = nz, film, T
    return labels, info


def jamming_fraction(phases, n_target=500, seed=11, log=None, should_stop=None, max_particles=4000,
                     target=0.80):
    """Maximum (random jammed) packing fraction of these particles - their
    shapes, size distributions and relative amounts - packed as the generator
    packs them (rigid clumps, growth with overlap relaxation, rotation for
    non-spheres) from a loading no packing reaches, without any voxel grid:
    the size scale where the growth stops gives phi_m = target x scale^3. The
    growth stops when it gains less than 0.05 % over 30 rounds.

    The relaxation jams a little below the random close packing of the
    literature (equal spheres 0.612 against 0.64; a lognormal size
    distribution of CV 0.5, 0.650 against about 0.68), so the value is
    scaled by JAM_CAL = 0.64 / 0.6117 and both are returned. At least 40 of the
    largest particles are packed (a bimodal or wide distribution has many
    small ones per large one), at most max_particles. Returns (phi_m, info)."""
    parts = [dict(ph) for ph in phases if ph["shape"] != G.NETWORK and float(ph.get("vf", 0)) > 0]
    if not parts:
        raise ValueError("no particle phase to pack")
    t0 = time.time()
    tot = sum(float(p["vf"]) for p in parts)
    rng = np.random.default_rng(seed)
    nominal, planes, nfaces, poly_meta = phase_templates(parts, rng)
    mean_vol = []
    for i, p in enumerate(parts):
        p["vf"] = float(p["vf"]) * target / tot
        sc = G.sample_scales(2000, p.get("dist"), np.random.default_rng(7))
        nom = nominal[i]
        lin = _linear_slots(nom["code"])
        vols = []
        for x in sc[:300]:
            P = nom["P"].copy()
            P[lin] *= x
            vols.append(SH.volume(nom["code"], P, poly_meta))
        mean_vol.append(float(np.mean(vols)))
    dens = sum(p["vf"] / mv for p, mv in zip(parts, mean_vol))
    # rotating clumps cost some 40 sub-spheres each and creep toward jamming
    # for many more steps (500 cubes: 60,000 steps took 18 min): fewer of them
    # and half the steps, the value then a lower estimate
    clumps = any(nominal[i]["code"] != 0 for i in range(len(parts)))
    steps = JAM_MAX_STEPS // 2 if clumps else JAM_MAX_STEPS
    if clumps:
        n_target = min(n_target, 250)
    k_big = int(np.argmax(mean_vol))
    n_tot = int(min(max_particles, max(n_target, 40.0 * dens / (parts[k_big]["vf"] / mean_vol[k_big]))))
    V = n_tot / dens
    L = V ** (1.0 / 3.0)
    Cl, Rl, sl, tpl_l, phl, rot_l, circ_l = [], [], [], [], [], [], []
    templates, tkey = [], {}
    for i, p in enumerate(parts):
        nom = nominal[i]
        n = max(1, int(round(p["vf"] * V / mean_vol[i])))
        Ps, Rs, s_ = _draw(p, nom, n, rng, poly_meta)
        v = np.array([SH.volume(nom["code"], P, poly_meta) for P in Ps])
        corr = (p["vf"] * V / v.sum()) ** (1.0 / 3.0)
        s_ = s_ * corr
        tids = []
        for P in Ps:
            t_id = int(P[1]) if nom["code"] == 6 else -1
            key = (i, t_id)
            if key not in tkey:
                if nom["code"] == 0:
                    q, r = np.zeros((1, 3)), np.array([0.5 * float(nom["P"][0]) * 2.0 / max(float(nom["P"][0]) * 2.0, 1e-30)])
                else:
                    Pn = nom["P"].copy()
                    if nom["code"] == 6:
                        Pn[1] = t_id
                    q, r = SH.clump_template(nom["code"], Pn, planes, nfaces, max_spheres=40, coverage=0.98)
                tkey[key] = len(templates)
                templates.append((q, r))
            tids.append(tkey[key])
        ori = p.get("orientation") or {}
        rm = -1 if nom["code"] == 0 else _rotmode(ori.get("mode", "iso"), ori.get("spread_deg", 0))
        circ = SH.circumradius(nom["code"], nom["P"])
        if nom["code"] == 0:
            # a sphere clump is one sphere of radius 0.5 at scale = diameter
            s_ = s_ * 2.0 * circ
            circ = 0.5
        Rl.append(Rs)
        sl.append(s_)
        tpl_l.append(np.array(tids))
        phl.append(np.full(n, i))
        rot_l.append(np.full(n, rm))
        circ_l.append(np.full(n, circ))
    Rs = np.ascontiguousarray(np.concatenate(Rl))
    s_all = np.concatenate(sl)
    tpl = np.concatenate(tpl_l).astype(np.int64)
    ph_arr = np.concatenate(phl).astype(np.int64)
    rotm = np.concatenate(rot_l).astype(np.int64)
    circ = np.concatenate(circ_l)
    C = rng.random((len(s_all), 3)) * L
    said = []
    _, _, scale = clump_pack(float(L), C, Rs, s_all, tpl, templates, np.zeros(len(s_all)), ph_arr,
                             np.zeros(len(parts) + 1, np.int64), rotm, circ, log=said.append, should_stop=should_stop,
                             tol=1e-3, stall_window=30, stall_gain=5e-4, max_steps=steps)
    limited = any("step limit" in m for m in said)
    phi_raw = target * scale ** 3
    phi_m = min(0.95, phi_raw * JAM_CAL)
    out = {"phi_m": phi_m, "phi_m_raw": phi_raw, "calibration": JAM_CAL, "target": target, "scale": float(scale),
           "n_particles": int(len(s_all)), "L_um": L, "seconds": time.time() - t0,
           "route": "rigid clumps, growth with overlap relaxation", "step_limited": limited}
    if log:
        log(f"    maximum packing fraction: {phi_m:.4f} (jammed {phi_raw:.4f} x {JAM_CAL:.4f}; "
            f"{len(s_all)} particles, {out['seconds']:.0f} s" + ("; step limit reached: a lower estimate" if limited else "") + ")")
    return phi_m, out
