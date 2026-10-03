"""Assemble the user guide from its chapter files: one HTML document per language.

    python build.py            -> user_guide_ko.html and user_guide_en.html, beside figures/
    python build.py en         -> only the English one

Chapters are plain HTML fragments in parts_ko/ and parts_en/, taken in filename
order, so a new chapter is added by dropping a file in with the right number
(the same number in both languages). The style lives here so every chapter
stays pure content; figures are shared by both languages and referenced
relative to this folder as figures/<name>.png.

Figures are placed as in a journal paper: one that would not fill the width
without growing too tall (taller than about 3/4 of its width) floats left at
36-48 % of the text width, by its proportions, and the text that follows
runs beside it; wide figures keep the full width. The PDF is printed by
make_pdf.js.
"""
from __future__ import annotations

import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The document has to be written BESIDE figures/, never one level up. Each
# chapter names its screenshots as figures/<name>.png and the browser resolves
# those against the html file itself when it prints. Writing it to ../ gave the
# worst possible combination: the build checked the paths against HERE and
# reported no missing figures, while the printed PDF contained none of them.
LANGS = {"ko": ("parts_ko", "user_guide_ko.html", "MPSim · 사용 안내서"),
         "en": ("parts_en", "user_guide_en.html", "MPSim · User Guide")}

STYLE = """
  @page { size: A4; margin: 18mm 16mm 17mm 16mm; }
  * { box-sizing: border-box; }
  body { font-family: "Segoe UI", "Malgun Gothic", system-ui, sans-serif;
         font-size: 10pt; line-height: 1.65; color: #1f2328; margin: 0; }
  h1 { font-size: 23pt; margin: 0 0 4pt; color: #0b62c4; letter-spacing: -0.5pt; }
  h2 { font-size: 15pt; margin: 0 0 11pt; padding: 7pt 0 6pt;
       border-top: 2.5pt solid #0b62c4; border-bottom: 0.7pt solid #d0d7de;
       color: #0b62c4; break-after: avoid; }
  h3 { font-size: 12pt; margin: 15pt 0 5pt; color: #0d1117; break-after: avoid; }
  h4 { font-size: 10.5pt; margin: 11pt 0 3pt; color: #1f2328; break-after: avoid; }
  p  { margin: 0 0 7pt; }
  ul, ol { margin: 0 0 8pt; padding-left: 17pt; }
  li { margin-bottom: 3pt; }
  table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
          font-size: 8.8pt; }
  tr { break-inside: avoid; }
  tr:first-child { break-after: avoid; }   /* no header row left alone at a page foot */
  th { background: #eef3f9; text-align: left; font-weight: 600; color: #0d1117; }
  th, td { border: 0.5pt solid #d0d7de; padding: 3.5pt 5pt; vertical-align: top; }
  td.q { width: 30%; font-weight: 600; color: #0d1117; }
  code, .mono { font-family: Consolas, monospace; font-size: 9pt; }
  .eq { font-family: Consolas, monospace; font-size: 10pt; text-align: center;
        background: #f6f8fa; border: 0.5pt solid #e1e6eb; border-radius: 3pt;
        padding: 6pt 8pt; margin: 7pt 0; }
  .code { font-family: Consolas, monospace; font-size: 8.6pt; text-align: left;
          background: #f6f8fa; border: 0.5pt solid #e1e6eb; border-radius: 3pt;
          padding: 6pt 8pt; margin: 7pt 0; line-height: 1.55;
          white-space: normal; break-inside: avoid; }
  .ui { font-family: Consolas, monospace; font-size: 9pt; background: #eef3f9;
        border: 0.5pt solid #d0d7de; border-radius: 2pt; padding: 0 3pt; }
  .lead { font-size: 10.5pt; color: #424a53; margin-bottom: 11pt; }
  .note { border-left: 3pt solid #0b62c4; background: #f5f9ff; padding: 7pt 10pt;
          margin: 8pt 0; font-size: 9.3pt; break-inside: avoid; }
  .warn { border-left: 3pt solid #e36209; background: #fff7f0; padding: 7pt 10pt;
          margin: 8pt 0; font-size: 9.3pt; break-inside: avoid; }
  .src { font-size: 9pt; color: #57606a; margin: 2pt 0 8pt; }
  .src b { color: #0b62c4; }
  figure { margin: 9pt 0 11pt; break-inside: avoid; }
  figure img { width: 100%; border: 0.5pt solid #d0d7de; border-radius: 3pt; display: block; }
  figure img.half { width: 62%; }
  figure .g3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5pt; }
  figure .g2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 5pt; }
  figure .g3 div, figure .g2 div { break-inside: avoid; }
  figure .g3 span, figure .g2 span { display: block; font-size: 8pt; color: #57606a; text-align: center; margin-top: 2pt; }
  figcaption { font-size: 8.6pt; color: #57606a; padding-top: 3.5pt; }
  /* paper-style placement: a tall figure floats left and the text runs beside
     it; headings and full-width blocks start below it */
  figure.side { float: left; margin: 3pt 13pt 8pt 0; }
  figure.side figcaption { font-size: 8.3pt; line-height: 1.5; }
  h2, h3, figure:not(.side) { clear: both; }
  .note, .warn, .eq, .code, ul, ol, .tbl { overflow: hidden; }
  .tbl table { margin: 6pt 0 10pt; }
  .sec { break-before: page; }
  .cover { padding-top: 52mm; break-after: page; }
  .cover .full { font-size: 16pt; color: #0b62c4; margin-top: 2pt; letter-spacing: -0.2pt; }
  .cover .sub { font-size: 13.5pt; color: #424a53; margin-top: 7pt; }
  .cover .meta { margin-top: 26mm; font-size: 9.5pt; color: #6e7781; line-height: 1.95; }
  .cover .ai { margin-top: 18mm; font-size: 8.3pt; color: #6e7781; line-height: 1.6; max-width: 150mm;
               border-top: 0.5pt solid #d0d7de; padding-top: 5pt; }
  .toc { columns: 2; column-gap: 13mm; font-size: 9.3pt; }
  .toc div { break-inside: avoid; margin-bottom: 2.5pt; }
  .toc .g { font-weight: 700; color: #0b62c4; margin-top: 8pt; }
"""


def png_size(path):
    """(width, height) from a PNG header, or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except OSError:
        return None
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


FIGURE = re.compile(r"<figure>(.*?)</figure>", re.S)


def place_figures(text):
    """Float a tall single-image figure left (see the module notes)."""
    def one(m):
        inner = m.group(1)
        imgs = re.findall(r"<img [^>]*>", inner)
        if len(imgs) != 1 or 'class="g' in inner:
            return m.group(0)
        src = re.search(r'src="([^"]+)"', imgs[0]).group(1)
        size = png_size(os.path.join(HERE, src))
        if not size:
            return m.group(0)
        ratio = size[1] / size[0]
        if ratio < 0.72:
            return m.group(0)
        width = 36 if ratio >= 1.6 else 42 if ratio >= 1.15 else 48
        img = re.sub(r'\s(class="half"|style="max-width:[^"]*")', "", imgs[0])
        return f'<figure class="side" style="width:{width}%">{inner.replace(imgs[0], img)}</figure>'
    text = FIGURE.sub(one, text)
    # a floated figure right under a heading moves below the first paragraph:
    # the heading and its text stay on the page and only the figure floats on
    # when it does not fit (a heading is kept with what follows it)
    return re.sub(r'(<h([34])[^>]*>[^\n]*?</h\2>\s*)(<figure class="side"[^>]*>.*?</figure>)(\s*)(<p>.*?</p>)',
                  r'\1\5\4\3', text, flags=re.S)


def build(lang):
    parts_dir, out_name, title = LANGS[lang]
    parts = os.path.join(HERE, parts_dir)
    out = os.path.join(HERE, out_name)
    if not os.path.isdir(parts):
        print(f"no {parts_dir}/ folder next to build.py", file=sys.stderr)
        return 1
    names = sorted(n for n in os.listdir(parts) if n.endswith(".html"))
    if not names:
        print(f"{parts_dir}/ holds no .html chapter", file=sys.stderr)
        return 1
    body = []
    missing = []
    n_side = 0
    for n in names:
        with open(os.path.join(parts, n), encoding="utf-8") as fh:
            text = fh.read()
        # every figure a chapter names must actually exist, otherwise the PDF
        # would be printed with silent gaps where the screenshots should be
        for ref in re.findall(r'src="(figures/[^"]+)"', text):
            if not os.path.exists(os.path.join(HERE, ref)):
                missing.append(f"{n}: {ref}")
        placed = place_figures(text)
        # a table sits in a box of its own, which narrows beside a floated
        # figure instead of dropping below it and leaving the side empty
        placed = re.sub(r"<table", '<div class="tbl"><table', placed)
        placed = placed.replace("</table>", "</table></div>")
        n_side += placed.count('<figure class="side"')
        body.append(f"<!-- ==== {n} ==== -->\n{placed}")
    html = (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{lang}: {len(names)} chapters -> {out}  ({len(html)/1024:.0f} kB, {n_side} figures beside the text)")
    if missing:
        print(f"WARNING: {len(missing)} figure(s) referenced but not present:")
        for m in missing[:20]:
            print("   " + m)
        return 2
    return 0


def main():
    langs = sys.argv[1:] or [k for k in LANGS if os.path.isdir(os.path.join(HERE, LANGS[k][0]))]
    return max(build(lang) for lang in langs)


if __name__ == "__main__":
    sys.exit(main())
