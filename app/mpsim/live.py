"""Live frames of the solves running now, for the browser.

Every solve that can show its state while it iterates - the finite-volume
conduction, the FFT (FANS) elasticity and viscosity, the particle dynamics -
writes its latest frame into the run folder (live/), at most once a second:
a section of its current field, or the positions of the particles. A slot
is one solve (one property and direction, one load, one shear rate); its
header (<slot>.json) carries a sequence number, so the browser takes the
binary (<slot>.f32 ...) only when it changed. Solves side by side, in this
process or in the workers of the parallel pool, write their own slots - the
browser picks which one to watch.
"""
import json
import math
import os
import re
import time

import numpy as np

MAX_SIDE = 256                 # a section is sampled down to this many pixels a side
EVERY_S = 1.0                  # the shortest time between two frames of a slot


def _atomic(path, data, binary):
    tmp = path + ".part"
    with open(tmp, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as fh:
        fh.write(data)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:          # the browser is reading it (Windows): the next frame then
            time.sleep(0.02)
    return False


def section(arr, axis=2):
    """The middle section of a 3D field normal to axis, sampled down to at
    most MAX_SIDE a side: rows the second remaining axis, columns the first."""
    a = np.asarray(arr)
    k = a.shape[axis] // 2
    s = np.take(a, k, axis=axis)
    st = max(1, math.ceil(max(s.shape) / MAX_SIDE))
    return np.ascontiguousarray(s[::st, ::st].T, dtype=np.float32)


class Slot:
    """One solve's live frames."""

    def __init__(self, live_dir, slot, name, kind="field", every=EVERY_S):
        self.dir = live_dir
        self.slot = re.sub(r"[^A-Za-z0-9_.-]+", "_", slot).strip("_") or "slot"
        self.name, self.kind, self.every = name, kind, every
        self.t, self.seq = 0.0, 0
        self.static, self.last = {}, {}
        os.makedirs(live_dir, exist_ok=True)

    def due(self):
        return time.time() - self.t >= self.every

    def _header(self, **h):
        self.last = {k: v for k, v in h.items() if k != "state"}
        self.seq += 1
        head = {"slot": self.slot, "name": self.name, "kind": self.kind, "seq": self.seq, "t": time.time(),
                **self.static, **h}
        _atomic(os.path.join(self.dir, self.slot + ".json"), json.dumps(head), False)

    def field(self, img, it=None, res=None, label="", plane="", force=False):
        """A 2D section (rows y, columns x) of the solve's current field."""
        if not force and not self.due():
            return
        self.t = time.time()
        img = np.asarray(img, np.float32)
        fin = img[np.isfinite(img)]
        lo, hi = (float(np.percentile(fin, 1)), float(np.percentile(fin, 99))) if fin.size else (0.0, 1.0)
        if _atomic(os.path.join(self.dir, self.slot + ".f32"), img.tobytes(), True):
            self._header(shape=list(img.shape), vmin=lo, vmax=hi if hi > lo else lo + 1e-12,
                         it=None if it is None else int(it), res=None if res is None else float(res),
                         label=label, plane=plane)

    def particles_static(self, radii_um, phase, box_um, names, colors, shown_of=None):
        """The spheres, written once: radii (um), phase index, box edge (um)."""
        _atomic(os.path.join(self.dir, self.slot + "_radii.f32"), np.asarray(radii_um, np.float32).tobytes(), True)
        _atomic(os.path.join(self.dir, self.slot + "_phase.u8"), np.asarray(phase, np.uint8).tobytes(), True)
        self.static = {"n": int(len(radii_um)), "box_um": float(box_um), "names": list(names),
                       "colors": list(colors), "n_total": int(shown_of or len(radii_um))}

    def particles(self, pos_um, speed, strain, eta, force=False):
        """The current positions (um) and speeds of the spheres."""
        if not force and not self.due():
            return
        self.t = time.time()
        ok = _atomic(os.path.join(self.dir, self.slot + "_pos.f32"), np.asarray(pos_um, np.float32).tobytes(), True)
        ok &= _atomic(os.path.join(self.dir, self.slot + "_speed.f32"), np.asarray(speed, np.float32).tobytes(), True)
        if ok:
            sp = np.asarray(speed, float)
            self._header(strain=float(strain), eta=float(eta) if eta is not None and math.isfinite(eta) else None,
                         speed_p99=float(np.percentile(sp, 99)) if sp.size else 1.0)

    def done(self, **h):
        """The solve has finished: its last frame stays, marked as such."""
        if self.seq:
            self._header(**{**self.last, **h, "state": "done"})


def index(live_dir):
    """The headers of every slot of a run, newest first."""
    out = []
    try:
        names = os.listdir(live_dir)
    except OSError:
        return out
    for f in names:
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(live_dir, f), encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda h: -float(h.get("t") or 0))
    return out


def clear(live_dir):
    """Removes the frames of an earlier run in the same folder."""
    try:
        for f in os.listdir(live_dir):
            try:
                os.remove(os.path.join(live_dir, f))
            except OSError:
                pass
    except OSError:
        pass


# ------------------------------------------------------------ solver snapshots
LABEL = {"thermal": "Temperature", "electrical": "Electric potential", "dielectric": "Electric potential",
         "magnetic": "Magnetic potential", "moisture": "Water activity"}


def _down(s2):
    st = max(1, math.ceil(max(s2.shape) / MAX_SIDE))
    return np.ascontiguousarray(s2[::st, ::st].T, dtype=np.float32)


def cond_snapshot(slot, d, film, label):
    """callback.snapshot for a conduction-type solve along axis d: the
    potential on the middle section that contains d (the applied gradient
    added back, so the isotherms bend round the fillers); a film shows its
    xz section (the plates at the bottom and top)."""
    def snap(u, it, res, final=False):
        if not final and not slot.due():
            return
        if film:
            img, plane = _down(np.take(u, u.shape[1] // 2, axis=1)), "y mid-section (x right, z up)"
        else:
            axis = 1 if d == 2 else 2
            s2 = np.take(u, u.shape[axis] // 2, axis=axis).astype(np.float64)
            ramp = np.arange(u.shape[d], dtype=float)
            s2 = s2 + (ramp[:, None] if [a for a in range(3) if a != axis].index(d) == 0 else ramp[None, :])
            img = _down(s2)
            plane = "z mid-section (x right, y up)" if axis == 2 else "y mid-section (x right, z up)"
        slot.field(img, it, res, label=f"{label} ({'between the plates' if film else 'gradient along ' + 'xyz'[d]})",
                   plane=plane, force=final)
        if final:
            slot.done()
    return snap


class Multi:
    """snapshot(name, x, it, res, final) for a solver of several loads (the
    FANS elasticity and viscosity): a slot per load, showing the magnitude of
    the vector field x (3, nx, ny, nz) on the z mid-section."""

    def __init__(self, live_dir, prefix, title, label):
        self.dir, self.prefix, self.title, self.label = live_dir, prefix, title, label
        self.slots = {}

    def __call__(self, name, x, it, res, final=False):
        slot = self.slots.get(name)
        if slot is None:
            slot = self.slots[name] = Slot(self.dir, f"{self.prefix}_{name}", f"{self.title}, {name}")
        if not final and not slot.due():
            return
        xs = np.asarray(x)[:, :, :, x.shape[3] // 2]
        img = _down(np.sqrt((xs.astype(np.float64) ** 2).sum(axis=0)))
        slot.field(img, it, res, label=self.label, plane="z mid-section (x right, y up)", force=final)
        if final:
            slot.done()
