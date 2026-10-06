"""Assemble the offline distribution in Download\\ - the folder to copy to a PC
without internet. Everything the program and its installer need, nothing a
run produced:

    python tools\\make_download.py

Copied: the batch files and read-me files, app\\ (without __pycache__ and
numba caches), tools\\ (the bundle builder and the import check), runtime\\
(the python.org runtime and openEMS), wheels\\, and docs\\ with the English and
Korean PDF guides, their HTML versions and only the figures they show. Left out:
workspace\\ (results), python\\ or env\\ (a local installation) and the guide's
source chapters.
"""
from __future__ import annotations

import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Download")
TOP = [f for f in os.listdir(ROOT)
       if f.lower().endswith((".bat", ".md", ".txt", ".yml")) or f in ("LICENSE", "CITATION.cff")]
# app/logs: the solver logs a run started from app/ writes (run_cli.py)
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "logs"}
SKIP_EXT = (".pyc", ".pyo", ".nbi", ".nbc")


def copytree(src, dst):
    n = 0
    for base, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(base, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            if f.endswith(SKIP_EXT):
                continue
            shutil.copy2(os.path.join(base, f), os.path.join(dst, rel, f))
            n += 1
    return n


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for f in TOP:
        shutil.copy2(os.path.join(ROOT, f), os.path.join(OUT, f))
    n = copytree(os.path.join(ROOT, "app"), os.path.join(OUT, "app"))
    os.makedirs(os.path.join(OUT, "tools"))
    for f in ("build_offline.py", "check_imports.py", "make_download.py"):
        shutil.copy2(os.path.join(ROOT, "tools", f), os.path.join(OUT, "tools", f))
    n += copytree(os.path.join(ROOT, "runtime"), os.path.join(OUT, "runtime"))
    n += copytree(os.path.join(ROOT, "wheels"), os.path.join(OUT, "wheels"))
    docs, odocs = os.path.join(ROOT, "docs"), os.path.join(OUT, "docs")
    os.makedirs(os.path.join(odocs, "guide", "figures"))
    n += copytree(os.path.join(docs, "images"), os.path.join(odocs, "images"))
    figs = set()
    for lang in ("en", "ko"):
        shutil.copy2(os.path.join(docs, f"USER_GUIDE_{lang.upper()}.pdf"), odocs)
        name = f"user_guide_{lang}.html"
        shutil.copy2(os.path.join(docs, "guide", name), os.path.join(odocs, "guide"))
        with open(os.path.join(docs, "guide", name), encoding="utf-8") as fh:
            figs |= set(re.findall(r'src="figures/([^"]+)"', fh.read()))
    figs = sorted(figs)
    missing = []
    for f in figs:
        src = os.path.join(docs, "guide", "figures", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(odocs, "guide", "figures", f))
        else:
            missing.append(f)
    total = sum(os.path.getsize(os.path.join(b, f)) for b, _, fs in os.walk(OUT) for f in fs)
    print(f"Download: {n + len(figs) + len(TOP) + 4} files, {total / 2**20:.0f} MB, {len(figs)} guide figures")
    if missing:
        print("missing figures:", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
