"""Assemble the files attached to a GitHub release, in Release\\:

    python tools\\make_release.py

- MPSim-<version>-offline-win64.zip - the offline package (tools\\make_download.py,
  zipped with one top folder MPSim-<version>; files that are already
  compressed are stored as they are)
- MPSim-User-Guide-EN.pdf, MPSim-User-Guide-KO.pdf - the user guides under fixed
  names, so that .../releases/latest/download/<name> always opens the newest
- SHA256SUMS.txt - checksums of the three files

The guides are attached to releases instead of being kept in the repository:
every edit of a guide would otherwise add some 70 MB to the history.

Publishing (with the GitHub CLI logged in):
    gh release create v<version> Release\\MPSim-<version>-offline-win64.zip
        Release\\MPSim-User-Guide-EN.pdf Release\\MPSim-User-Guide-KO.pdf
        Release\\SHA256SUMS.txt --title "MPSim <version>" --notes-file <notes>
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Release")
STORED = (".whl", ".zip", ".7z", ".gz", ".png", ".jpg", ".pdf")


def version():
    with open(os.path.join(ROOT, "app", "mpsim", "__init__.py"), encoding="utf-8") as fh:
        return re.search(r'__version__\s*=\s*"([^"]+)"', fh.read()).group(1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    v = version()
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_download                                       # noqa: E402
    if make_download.main():
        return 1
    os.makedirs(OUT, exist_ok=True)
    names = []
    zname = f"MPSim-{v}-offline-win64.zip"
    zpath = os.path.join(OUT, zname)
    src = os.path.join(ROOT, "Download")
    with zipfile.ZipFile(zpath + ".part", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, dirs, files in os.walk(src):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(base, f)
                arc = "/".join(("MPSim-" + v, os.path.relpath(p, src).replace(os.sep, "/")))
                z.write(p, arc, compress_type=zipfile.ZIP_STORED if f.lower().endswith(STORED)
                        else zipfile.ZIP_DEFLATED)
    os.replace(zpath + ".part", zpath)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
    if bad:
        sys.exit(f"{zname}: damaged entry {bad}")
    names.append(zname)
    for lang in ("EN", "KO"):
        name = f"MPSim-User-Guide-{lang}.pdf"
        shutil.copy2(os.path.join(ROOT, "docs", f"USER_GUIDE_{lang}.pdf"), os.path.join(OUT, name))
        names.append(name)
    with open(os.path.join(OUT, "SHA256SUMS.txt"), "w", encoding="ascii", newline="\n") as fh:
        for n in names:
            fh.write(f"{sha256(os.path.join(OUT, n))}  {n}\n")
    for n in names + ["SHA256SUMS.txt"]:
        print(f"  {n:38s} {os.path.getsize(os.path.join(OUT, n)) / 2**20:8.1f} MB")
    print(f"Release {v}: files in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
