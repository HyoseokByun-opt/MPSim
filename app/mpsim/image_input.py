"""A structure from a TIFF image instead of generated particles.

A 3D image (a multi-page TIFF stack, an ImageJ hyperstack or a single 3D page)
of grey values, colours (RGB) or palette indices is read once, stored with its
histogram, and turned into the voxel labels the solvers take: every grey value,
grey-value range or colour of the image is a class, and every class is the
matrix or a phase with a material. A single 2D image may be extruded along z
(a structure uniform through z: a section of aligned fibres, for example).

Axes: the pages of a stack are z, the rows y and the columns x, as ImageJ and
MPSim's own structure.tif have them, so that an exported structure reads back
unchanged. The image is taken as one periodic cell: the solvers wrap it around
its faces as they wrap a generated RVE (an image cut from a larger sample is not
periodic; the faces then meet a mismatched neighbour).
"""
from __future__ import annotations

import json
import os
import time
import uuid

import numpy as np

MAX_CLASSES = 64            # distinct values or colours taken one by one as classes
MAX_THRESHOLDS = 8
IMAGE_FILE = "image.npy"
META_FILE = "meta.json"


def _rgb_key(a):
    """Colours (..., 3) as one integer each: 0xRRGGBB."""
    a = np.asarray(a)
    return (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2].astype(np.uint32)


def _hex(key):
    return f"#{int(key) & 0xFFFFFF:06x}"


def read_tif(path, extrude=0):
    """The image as an array in x, y, z order and how its values are read: (keys, mode, info). mode 'gray'
    (the values themselves), 'rgb' (keys 0xRRGGBB) or 'palette' (indices into info['palette'])."""
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        axes = s.axes.upper()
        a = s.asarray()
        page = tf.pages[0]
        photometric = getattr(page.photometric, "name", str(page.photometric)).upper()
        cmap = page.colormap if photometric == "PALETTE" else None
        ij = tf.imagej_metadata or {}
        res = page.tags.get("XResolution")
        spacing = ij.get("spacing")
        unit = ij.get("unit")
    info = {"axes": axes, "dtype": str(a.dtype), "photometric": photometric}
    # a plain sequence of pages (I, Q), a time series, or a stack ImageJ calls channels, without a z axis: its
    # pages are the slices (a C of 3 or 4 without samples is the colour of an RGB image instead)
    if "Z" not in axes:
        for ax in "IQTC":
            if ax in axes and a.shape[axes.index(ax)] > 1 and not (ax == "C" and "S" not in axes
                                                                   and a.shape[axes.index(ax)] in (3, 4)):
                axes = axes.replace(ax, "Z", 1)
                break
    # samples: S (or C of 3/4 for an RGB hyperstack) are the colour channels
    mode = "gray"
    if "S" in axes and a.shape[axes.index("S")] in (3, 4):
        a = np.moveaxis(a, axes.index("S"), -1)[..., :3]
        axes = axes.replace("S", "") + "S"
        mode = "rgb"
    elif "C" in axes and a.shape[axes.index("C")] in (3, 4) and photometric in ("RGB", "MINISBLACK"):
        c = axes.index("C")
        if a.shape[c] == 3 or photometric == "RGB":
            a = np.moveaxis(a, c, -1)[..., :3]
            axes = axes.replace("C", "") + "S"
            mode = "rgb"
    # time, channels of other counts and the like: the first of each
    for ax in "TCQIE":
        if ax in axes and not (mode == "rgb" and ax == "S"):
            k = axes.index(ax)
            a = np.take(a, 0, axis=k)
            axes = axes[:k] + axes[k + 1:]
    spatial = axes.replace("S", "")
    if mode == "rgb":
        keys = _rgb_key(a)
    else:
        if a.dtype.kind == "f":
            if not np.all(np.isfinite(a)) or not np.allclose(a, np.round(a)):
                raise ValueError("the image holds non-integer grey values; save it as an 8- or 16-bit image "
                                 "(or segment it into labels first)")
            a = np.round(a).astype(np.int64)
        if a.dtype.kind not in "ui" and a.dtype != np.bool_:
            raise ValueError(f"grey values of type {a.dtype} are not read; save the image as 8- or 16-bit")
        if a.dtype.kind == "i" and a.size and int(a.min()) < 0:
            raise ValueError("the image holds negative grey values")
        if a.size and int(a.max()) >= 1 << 32:
            raise ValueError("grey values above 2³² - 1 are not read; save the image as 8-, 16- or 32-bit")
        keys = a if a.dtype.kind == "u" else a.astype(np.uint32)
        if cmap is not None:
            mode = "palette"
            info["palette"] = [_hex(_rgb_key((np.asarray(cmap)[:, k] / (257.0 if np.asarray(cmap).max() > 255 else 1.0))
                                              .round().astype(np.uint32))) for k in range(np.asarray(cmap).shape[1])]
    # to x, y, z (pages z, rows y, columns x)
    if spatial == "ZYX":
        keys = keys.transpose(2, 1, 0)
    elif spatial == "YX":
        if not extrude:
            info["is_2d"] = True
            keys = keys.T[:, :, None]
        else:
            keys = np.repeat(keys.T[:, :, None], int(extrude), axis=2)
    elif len(spatial) == 3:
        keys = keys.transpose(2, 1, 0)
    else:
        raise ValueError(f"an image of axes {s.axes} is not a 3D stack")
    # grey values and palette indices in the smallest type that holds them (an 8-bit stack stays one byte a voxel)
    if mode != "rgb" and keys.size:
        top = int(keys.max())
        keys = keys.astype(np.uint8 if top < 1 << 8 else np.uint16 if top < 1 << 16 else np.uint32, copy=False)
    info["shape"] = [int(v) for v in keys.shape]
    # the voxel size the file gives, if any (ImageJ spacing in z, resolution in x)
    try:
        if res is not None:
            num, den = res.value
            if num:
                info["voxel_from_file"] = float(den) / float(num)
        if spacing:
            info["voxel_from_file_z"] = float(spacing)
        if unit:
            info["unit_from_file"] = str(unit).replace("\\u00B5", "u").replace("µ", "u").replace("micron", "um")
    except Exception:                                                      # noqa: BLE001
        pass
    return np.ascontiguousarray(keys), mode, info


def histogram(keys):
    """The distinct keys of the image and how many voxels hold each: (values, counts), sorted by value."""
    flat = keys.ravel()
    if flat.dtype.kind == "u" and flat.size and int(flat.max()) < 1 << 17:
        c = np.bincount(flat)
        v = np.flatnonzero(c)
        return v.astype(np.int64), c[v].astype(np.int64)
    v, c = np.unique(flat, return_counts=True)
    return v.astype(np.int64), c.astype(np.int64)


def auto_thresholds(values, counts, n_classes=2):
    """Multi-level Otsu thresholds of a grey-value histogram (n_classes - 1 values; a voxel of value >= t lies
    above the threshold t)."""
    v, c = np.asarray(values, float), np.asarray(counts, float)
    n_classes = int(max(2, min(n_classes, MAX_THRESHOLDS + 1)))
    if len(v) <= n_classes:
        return [float(x) for x in v[1:]]
    try:
        from skimage.filters import threshold_multiotsu
        # a histogram of at most 256 bins over the values' range
        nb = int(min(256, len(v)))
        h, e = np.histogram(v, bins=nb, weights=c)
        t = threshold_multiotsu(hist=(h, 0.5 * (e[1:] + e[:-1])), classes=n_classes)
        return [float(x) for x in np.atleast_1d(t)]
    except Exception:                                                      # noqa: BLE001
        # equal shares of the voxels
        cum = np.cumsum(c) / c.sum()
        return [float(v[min(np.searchsorted(cum, k / n_classes), len(v) - 1)]) for k in range(1, n_classes)]


def classes_of(meta, thresholds=None):
    """The classes of an image: one per distinct value or colour (at most MAX_CLASSES), or, with thresholds,
    one per grey-value range. Each {key, label, color, count, fraction} with key the value (or 0xRRGGBB, or
    the range index)."""
    v = np.asarray(meta["values"], np.int64)
    c = np.asarray(meta["counts"], np.int64)
    tot = float(c.sum())
    out = []
    if thresholds:
        t = sorted(float(x) for x in thresholds)
        idx = np.searchsorted(t, v, side="right")              # range k: t[k-1] <= v < t[k]
        edges = [float(v.min())] + t + [float(v.max())]
        for k in range(len(t) + 1):
            n = int(c[idx == k].sum())
            g = int(round(255 * k / max(len(t), 1)))
            lo_, hi_ = edges[k], edges[k + 1]
            if not n:
                lab = (f"below {t[0]:g}" if k == 0 else f"from {t[-1]:g}" if k == len(t)
                       else f"{t[k - 1]:g} – {t[k]:g}") + " (no voxels)"
            else:
                lab = f"{lo_:g} – {hi_:g}" if k < len(t) else f"{lo_:g} – {hi_:g} (max)"
            out.append({"key": k, "label": lab,
                        "color": f"#{g:02x}{g:02x}{g:02x}", "count": n, "fraction": n / tot})
        return out
    if len(v) > MAX_CLASSES:
        return None
    pal = meta.get("palette")
    vmax = float(v.max()) if len(v) else 1.0
    for val, n in zip(v, c):
        if meta["mode"] == "rgb":
            col, lab = _hex(val), _hex(val)
        elif pal and int(val) < len(pal):
            col, lab = pal[int(val)], f"index {int(val)}"
        else:
            g = int(round(255 * float(val) / vmax)) if vmax > 0 else 0
            col, lab = f"#{g:02x}{g:02x}{g:02x}", f"value {int(val)}"
        out.append({"key": int(val), "label": lab, "color": col, "count": int(n), "fraction": float(n) / tot})
    return out


def store(root, path, name, extrude=0):
    """Read a TIFF and keep it under root/<id>: the keys (x, y, z) and its description. Returns the meta."""
    keys, mode, info = read_tif(path, extrude)
    iid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = os.path.join(root, iid)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, IMAGE_FILE), keys)
    v, c = histogram(keys)
    meta = dict(info, id=iid, name=name, mode=mode, dir=os.path.abspath(d), voxels=int(keys.size),
                values=v.tolist() if len(v) <= 70000 else None, counts=c.tolist() if len(v) <= 70000 else None,
                n_values=int(len(v)), value_min=int(v.min()), value_max=int(v.max()))
    if meta["values"] is None:
        # more than 70,000 distinct values: a histogram of 4096 bins
        h, e = np.histogram(keys, bins=4096)
        keep = h > 0
        meta["values"] = np.round(0.5 * (e[1:] + e[:-1])[keep]).astype(np.int64).tolist()
        meta["counts"] = h[keep].astype(np.int64).tolist()
        meta["binned"] = True
    meta["classes"] = classes_of(meta)
    if meta["classes"] is None:
        meta["auto_thresholds"] = auto_thresholds(meta["values"], meta["counts"], 2)
    if mode == "rgb" and meta["classes"] is None:
        raise ValueError(f"the image has {len(v):,} colours; a colour image is read as labels, one colour per "
                         f"material (at most {MAX_CLASSES}) - reduce it to its label colours first")
    with open(os.path.join(d, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return meta


EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images")


def example_dir(name):
    """The folder of an example image shipped with MPSim (data/images), read once into the user's temporary
    folder (the program folder may be read-only)."""
    import tempfile
    name = os.path.basename(str(name))
    src = os.path.join(EXAMPLES, name)
    if not os.path.isfile(src):
        raise ValueError(f"no example image '{name}'")
    root = os.path.join(tempfile.gettempdir(), "mpsim_example_images")
    d = os.path.join(root, os.path.splitext(name)[0] + f"-{int(os.path.getmtime(src))}")
    if not os.path.isfile(os.path.join(d, META_FILE)):
        meta = store(root, src, name)
        try:
            os.replace(meta["dir"], d)
        except OSError:
            return meta["dir"]
        meta["dir"] = os.path.abspath(d)
        with open(os.path.join(d, META_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    return os.path.abspath(d)


def load_meta(d):
    with open(os.path.join(d, META_FILE), encoding="utf-8") as fh:
        meta = json.load(fh)
    if "values" not in meta or "mode" not in meta:
        raise ValueError(f"{d} is not an MPSim image")
    return meta


def load_keys(d):
    return np.load(os.path.join(d, IMAGE_FILE), mmap_mode="r")


def class_index(keys, img):
    """The class of every voxel: the index into img['classes'] (thresholds: the range; values or colours: the
    class whose key it is). Returns uint8 indices, 255 where no class holds the value. Slab by slab along x, so
    that a large stack needs little more than its labels."""
    cls = img["classes"]
    out = np.empty(keys.shape, np.uint8)
    step = max(1, (1 << 24) // max(1, int(np.prod(keys.shape[1:]))))
    if img.get("thresholds"):
        t = np.asarray(sorted(img["thresholds"]), float)
        for x0 in range(0, keys.shape[0], step):
            out[x0:x0 + step] = np.searchsorted(t, np.asarray(keys[x0:x0 + step]), side="right")
        return out
    ks = np.asarray([c["key"] for c in cls], np.int64)
    order = np.argsort(ks)
    for x0 in range(0, keys.shape[0], step):
        flat = np.asarray(keys[x0:x0 + step]).astype(np.int64)
        pos = np.clip(np.searchsorted(ks[order], flat), 0, len(ks) - 1)
        out[x0:x0 + step] = np.where(ks[order][pos] == flat, order[pos], 255)
    return out


def labels(img, log=None):
    """The labels of the generator's convention from an image spec (spec['image']): 0 where a class is the
    matrix, i + 1 where it is phase i. Returns (labels uint8, info) with the generator's info keys."""
    t0 = time.time()
    keys = load_keys(img["dir"])
    if img.get("extrude") and keys.shape[2] == 1:
        keys = np.repeat(np.asarray(keys), int(img["extrude"]), axis=2)
    ci = class_index(keys, img)
    lut = np.zeros(256, np.uint8)
    for k, c in enumerate(img["classes"]):
        lut[k] = int(c["phase"]) + 1 if c["role"] == "phase" else 0
    if np.any(ci == 255):
        raise ValueError("the image holds values no class covers; open it again in the Structure tab")
    lab = lut[ci]
    nx, ny, nz = lab.shape
    h = float(img["h_um"])
    counts = np.bincount(lab.ravel(), minlength=int(lut.max()) + 1)
    from . import shapes as SH
    info = {"route": f"image {img.get('name') or ''} ({nx} × {ny} × {nz} voxels of {h:g} µm)".replace("  ", " "),
            "planes": SH.EMPTY_PLANES,
            "seconds": time.time() - t0, "particles": {"codes": np.zeros(0, np.int64), "P": np.zeros((0, 8)),
                                                       "R": np.zeros((0, 3, 3)), "phase": np.zeros(0, np.int64),
                                                       "centers": np.zeros((0, 3)), "axes": np.zeros((0, 3))},
            "poly_meta": [], "contacts_mode": "keep", "contacts_removed": 0, "contact_faces": 0, "touching_pairs": 0,
            "voxel_conflicts": 0, "N": nx, "h_um": h, "L_um": nx * h, "nz": nz, "film": False, "T_um": nz * h,
            "shape": [nx, ny, nz], "image": True,
            "phases": [{"name": p["name"], "vf_target": float(p["vf"]), "vf_voxel": float(counts[i + 1] / lab.size)
                        if i + 1 < len(counts) else 0.0, "n_particles": 0, "jammed": False, "surface_true_um2": None,
                        "surface_voxel_um2": 0.0, "octant_cov": 0.0} for i, p in enumerate(img["phase_list"])]}
    if log:
        log(f"   structure from the image {img.get('name') or ''}: {nx} × {ny} × {nz} voxels of {h:g} µm "
            f"({time.time() - t0:.1f} s)")
    return lab, info
