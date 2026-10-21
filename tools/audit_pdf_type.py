#!/usr/bin/env python3
"""Read the type sizes straight out of a figure's PDF.

An independent check on what the generator reports. The generator measures the
matplotlib artists before saving; this measures what actually landed in the
file, which catches anything the artist walk does not reach -- a size set by a
library, a legend the walk misses, text drawn by a helper.

Every glyph in a PDF is set with a `Tf` operator giving a font size in text
space, which the current transform then scales. The page transform is read
from the content stream so nested scales (a page scaled to column width, for
instance) are accounted for rather than ignored.

    python tools/audit_pdf_type.py main_text_figures/figure4.pdf
    python tools/audit_pdf_type.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR_PT = 5.0


def content_streams(path):
    raw = Path(path).read_bytes()
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        blob = m.group(1)
        try:
            out.append(zlib.decompress(blob).decode("latin-1"))
        except Exception:
            try:
                out.append(blob.decode("latin-1"))
            except Exception:
                pass
    return out


def type_sizes(path):
    """[(effective_pt, raw_pt, scale)] for every Tf in the file."""
    sizes = []
    for s in content_streams(path):
        # Track the uniform scale in effect: `a b c d e f cm` concatenates a
        # matrix, and a page scaled by k shows up as a=d=k. Only the uniform
        # part matters for type size.
        scale = 1.0
        stack = []
        for tok in re.finditer(
                r"(?P<save>\bq\b)|(?P<rest>\bQ\b)"
                r"|(?P<cm>[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+"
                r"[-\d.]+\s+[-\d.]+\s+cm\b)"
                r"|(?P<tf>/[A-Za-z0-9_+.-]+\s+(?P<sz>[-\d.]+)\s+Tf\b)", s):
            if tok.group("save"):
                stack.append(scale)
            elif tok.group("rest"):
                scale = stack.pop() if stack else 1.0
            elif tok.group("cm"):
                a, _b, _c, d = (float(v) for v in tok.group("cm").split()[:4])
                if a > 0 and d > 0:
                    scale *= (abs(a) + abs(d)) / 2.0
            elif tok.group("tf"):
                raw = float(tok.group("sz"))
                if raw > 0:
                    # Grab the text this size is about to set, so a size below
                    # the floor can be named rather than just counted.
                    tail = s[tok.end():tok.end() + 400]
                    words = re.findall(r"\((?:\\.|[^()\\])*\)", tail)[:6]
                    txt = "".join(w[1:-1] for w in words)
                    txt = re.sub(r"\\([()\\])", r"\1", txt)[:48]
                    sizes.append((raw * scale, raw, scale, txt))
    return sizes


def report(path):
    sizes = type_sizes(path)
    if not sizes:
        return dict(path=str(path), error="no text found")
    eff = sorted(v for v, _r, _s, _t in sizes)
    below = [v for v in eff if v < FLOOR_PT - 0.005]
    hist = Counter(round(v, 2) for v in eff)
    small = sorted((v, t) for v, _r, _s, t in sizes if v < FLOOR_PT - 0.005)
    return dict(path=str(path), n=len(eff), smallest=eff[0], largest=eff[-1],
                below=len(below), sizes=sorted(hist.items())[:8],
                examples=small[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    paths = ([str(p) for p in sorted((ROOT / "main_text_figures").glob("figure?.pdf"))]
             if a.all or not a.pdf else a.pdf)

    bad = 0
    print(f"  {'file':34}{'strings':>9}{'smallest':>10}{'largest':>9}"
          f"{'under 5pt':>11}")
    rows = []
    for p in paths:
        r = report(p)
        rows.append(r)
        if r.get("error"):
            print(f"  {Path(p).name:34}{r['error']}")
            bad += 1
            continue
        flag = "" if not r["below"] else "   <-- BELOW FLOOR"
        bad += 1 if r["below"] else 0
        print(f"  {Path(p).name:34}{r['n']:>9}{r['smallest']:>10.2f}"
              f"{r['largest']:>9.2f}{r['below']:>11}{flag}")
    for r in rows:
        if not r.get("error"):
            got = ", ".join(f"{v:g}pt x{n}" for v, n in r["sizes"])
            print(f"    {Path(r['path']).name}: {got}")
            for v, t in r.get("examples", []):
                print(f"        {v:.2f} pt  {t!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
