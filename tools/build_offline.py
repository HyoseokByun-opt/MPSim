"""Builds the offline bundle of MPSim (Material Property Simulation), v5.

Run once on a PC with internet access and the conda environment the program
was tested in; copy the whole v5 folder to the offline PC afterwards and run
1_INSTALL_OFFLINE.bat there. The offline PC needs neither internet nor conda.

    conda run -p .\\env python tools\\build_offline.py

What it writes next to this folder:

    runtime\\python-3.11.9-embed-amd64.zip  the official Python runtime from
                                           python.org (no installer, no
                                           registry entries on the offline PC)
    wheels\\pumapy-3.2.2-cp311-cp311-win_amd64.whl
                                           NASA PuMA, packed from the
                                           conda-forge build in this
                                           environment (see below)
    wheels\\*.whl                           every other package, exactly the
                                           versions in requirements-offline.txt
                                           and their dependencies

Why PuMA can travel as a wheel: conda-forge's "puma" package puts a plain
Python package (pumapy) into site-packages. Its four compiled modules import
only python311.dll, VCRUNTIME140.dll (shipped with every python.org Python)
and the Windows C runtime - no library from the conda environment - so the
same files work in any CPython 3.11 on 64-bit Windows. PuMA is released by
NASA under the NASA Open Source Agreement 1.3, which allows redistribution
with its licence; the licence file travels inside the wheel.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEELS = os.path.join(ROOT, "wheels")
RUNTIME = os.path.join(ROOT, "runtime")
PY_VERSION = "3.11.9"                     # the last 3.11 release with Windows binaries
PY_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
TAG = "cp311-cp311-win_amd64"
# openEMS (GPL, https://openems.de): the full-wave check of the EMI analysis.
# The official Windows build is copied unchanged; its source is on GitHub
# (thliebig/openEMS-Project) as the GPL requires to be offered.
OEMS_VERSION = "v0.37.0-rc3"
OEMS_URL = (f"https://github.com/thliebig/openEMS-Project/releases/download/{OEMS_VERSION}/"
            f"openEMS_x64_{OEMS_VERSION}_msvc.zip")
PIP_TARGET = ["--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.11",
              "--implementation", "cp", "--abi", "cp311"]


def log(msg):
    print(msg, flush=True)


def fetch_runtime():
    os.makedirs(RUNTIME, exist_ok=True)
    dst = os.path.join(RUNTIME, os.path.basename(PY_URL))
    if not os.path.exists(dst):
        log(f"downloading {PY_URL}")
        urllib.request.urlretrieve(PY_URL, dst + ".part")
        os.replace(dst + ".part", dst)
    digest = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    log(f"runtime  {os.path.basename(dst)}  {os.path.getsize(dst)/1e6:.1f} MB  sha256 {digest}")


def fetch_openems():
    os.makedirs(RUNTIME, exist_ok=True)
    dst = os.path.join(RUNTIME, os.path.basename(OEMS_URL))
    if not os.path.exists(dst):
        log(f"downloading {OEMS_URL}")
        urllib.request.urlretrieve(OEMS_URL, dst + ".part")
        os.replace(dst + ".part", dst)
    log(f"openEMS  {os.path.basename(dst)}  {os.path.getsize(dst)/1e6:.1f} MB")


def _record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build_puma_wheel(prefix):
    """Packs pumapy from a conda environment into a standard wheel."""
    sp = os.path.join(prefix, "Lib", "site-packages")
    pkg = os.path.join(sp, "pumapy")
    eggs = glob.glob(os.path.join(sp, "pumapy-*.egg-info")) + glob.glob(os.path.join(sp, "pumapy-*.dist-info"))
    if not os.path.isdir(pkg) or not eggs:
        raise SystemExit(f"NASA PuMA (pumapy) was not found in {sp}")
    info = eggs[0]
    version = re.match(r"pumapy-([0-9][0-9.]*)", os.path.basename(info)).group(1)
    pyds = glob.glob(os.path.join(pkg, "**", "*.pyd"), recursive=True)
    bad = [p for p in pyds if "cp311-win_amd64" not in os.path.basename(p)]
    if bad:
        raise SystemExit(f"compiled modules for another Python: {bad}")
    meta_src = os.path.join(info, "PKG-INFO") if info.endswith("egg-info") else os.path.join(info, "METADATA")
    metadata = open(meta_src, "rb").read()
    lic = None
    for cand in (os.path.join(info, "LICENSE.txt"), os.path.join(prefix, "info", "licenses", "LICENSE.txt"),
                 *glob.glob(os.path.join(prefix, "conda-meta", "..", "info", "licenses", "*"))):
        if os.path.isfile(cand):
            lic = open(cand, "rb").read()
            break
    if lic is None:
        # the conda package keeps its licence under pkgs/<build>/info/licenses
        conda_root = os.path.dirname(os.path.dirname(prefix)) if os.path.basename(os.path.dirname(prefix)) == "envs" else prefix
        for cand in glob.glob(os.path.join(conda_root, "pkgs", "puma-*", "info", "licenses", "*")):
            if os.path.isfile(cand):
                lic = open(cand, "rb").read()
                break
    dist = f"pumapy-{version}.dist-info"
    os.makedirs(WHEELS, exist_ok=True)
    out = os.path.join(WHEELS, f"pumapy-{version}-{TAG}.whl")
    records = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def put(arc, data):
            z.writestr(arc, data)
            records.append(f"{arc},{_record_hash(data)},{len(data)}")
        for root, dirs, files in os.walk(pkg):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(files):
                if f.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(root, f)
                put(os.path.relpath(full, sp).replace(os.sep, "/"), open(full, "rb").read())
        put(f"{dist}/METADATA", metadata)
        put(f"{dist}/WHEEL", (f"Wheel-Version: 1.0\nGenerator: mpsim tools/build_offline.py\n"
                              f"Root-Is-Purelib: false\nTag: {TAG}\n").encode())
        put(f"{dist}/top_level.txt", b"pumapy\n")
        if lic is not None:
            put(f"{dist}/LICENSE.txt", lic)
        records.append(f"{dist}/RECORD,,")
        z.writestr(f"{dist}/RECORD", "\n".join(records) + "\n")
    log(f"wheel    {os.path.basename(out)}  {os.path.getsize(out)/1e6:.1f} MB  "
        f"({len(pyds)} compiled modules, licence {'included' if lic else 'NOT FOUND'})")


def download(req_file, label):
    """pip download for CPython 3.11 on 64-bit Windows, whatever runs this."""
    reqs = [ln.strip() for ln in open(os.path.join(ROOT, req_file), encoding="utf-8")
            if ln.strip() and not ln.lstrip().startswith("#")]
    # PuMA comes from the wheel packed above; the PyPI "pumapy" is another project
    reqs = [r for r in reqs if not r.lower().startswith("pumapy")]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(reqs + ["pip", "setuptools"]) + "\n")
        tmp = fh.name
    try:
        cmd = [sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "-d", WHEELS,
               *PIP_TARGET, "-c", os.path.join(ROOT, "constraints-offline.txt"), "-r", tmp]
        log(f"pip download ({label}) ...")
        subprocess.run(cmd, check=True)
    finally:
        os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puma-prefix", default=sys.prefix,
                    help="conda environment that has NASA PuMA (default: the one running this script)")
    ap.add_argument("--no-openems", action="store_true", help="leave out openEMS (the EMI full-wave check, 65 MB)")
    a = ap.parse_args()
    fetch_runtime()
    if not a.no_openems:
        fetch_openems()
    build_puma_wheel(a.puma_prefix)
    download("requirements-offline.txt", "required")
    whl = glob.glob(os.path.join(WHEELS, "*.whl"))
    size = sum(os.path.getsize(p) for p in whl) / 1e6
    log(f"\n{len(whl)} wheels, {size:.0f} MB in {WHEELS}")
    log("Copy the whole folder to the offline PC and run 1_INSTALL_OFFLINE.bat there.")


if __name__ == "__main__":
    main()
