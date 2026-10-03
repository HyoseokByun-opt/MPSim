"""Run a simulation without the browser.

    python run_cli.py --list                         # bundled presets
    python run_cli.py --preset tim_aln --out runs/tim
    python run_cli.py --form my_case.json --out runs/case1

It drives the same engine as the web server, so the two cannot give different
answers. A form is the JSON the browser sends; materials may be given by
`material_id` alone and are then filled from the library.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset")
    ap.add_argument("--form")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quality", choices=["fast", "standard", "accurate"])
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "mpsim", "data", "presets.json"), encoding="utf-8") as fh:
        presets = {p["id"]: p for p in json.load(fh)["presets"]}
    if a.list or not (a.preset or a.form):
        for pid, p in presets.items():
            print(f"  {pid:16s} {p['name']}")
        return 0
    if a.preset:
        form = presets[a.preset]["form"]
    else:
        with open(a.form, encoding="utf-8") as fh:
            form = json.load(fh)
    if a.quality:
        form.setdefault("rve", {})["quality"] = a.quality
    out = a.out or os.path.join(here, "..", "workspace", "cli", time.strftime("%Y%m%d-%H%M%S"))

    from mpsim import pipeline

    def event(kind, **p):
        if kind == "stage":
            print(f"[{100*p.get('progress', 0):5.1f}%] {p['name']}  {p.get('detail', '')}", flush=True)
    summary = pipeline.run(form, out, log=lambda *x: print(*x, flush=True), event=event)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("Output:", os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
