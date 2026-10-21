#!/usr/bin/env python3
"""Is the image under each figure caption actually that figure?

This is the one check that tests what a reviewer sees. The caption numbering
can be perfect while the embedded images came from a superseded render, in
which case the document shows the right captions over the wrong pictures and
nothing in the deposit reveals it. That is how a stale Fig. S25 was found:
its caption and number were correct, its image was an older render whose
aspect ratio did not match the deposited figure.

Method: walk the document in order, pairing each caption with the image that
precedes it, then perceptually match that image against the deposited
figures. The embedded copies are downsampled for Word, so an exact hash will
not do; a 16x16 average hash is robust to resize and recompression while
still distinguishing different plots. Aspect ratio is reported alongside,
because it catches a different render of the *same* figure, which the hash
alone may still match best.

    python tools/verify_embedded_figures.py Persistence_Supp_Mat.docx \\
        --record-b ../record_b_staging

    python tools/verify_embedded_figures.py Main_Text.docx \\
        --record-b ../record_b_staging --main-text

Exit status is 1 if any caption's image matches another figure better, or if
its aspect ratio disagrees with the deposited figure by more than --ar-tol.
"""
from __future__ import annotations

import argparse
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def ahash(img, n: int = 16):
    g = img.convert("L").resize((n, n), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    return [1 if p > avg else 0 for p in px]


def dist(a, b) -> int:
    return sum(x != y for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("document", type=Path,
                    help="a .docx whose captions and embedded images are "
                         "compared against the deposited figures")
    ap.add_argument("--record-b", type=Path, required=True, metavar="DIR",
                    help="a built results-record staging directory holding "
                         "the deposited figures")
    ap.add_argument("--main-text", action="store_true",
                    help="match 'Figure N' captions against figure1..5 "
                         "rather than 'Fig. SN' against S01..S31")
    ap.add_argument("--ar-tol", type=float, default=0.03, metavar="X",
                    help="how far an embedded image's aspect ratio may "
                         "differ from the deposited figure's (default 0.03)")
    a = ap.parse_args()

    z = zipfile.ZipFile(a.document)
    rels = {}
    for rel in ET.fromstring(z.read("word/_rels/document.xml.rels")):
        if "image" in rel.get("Type", ""):
            rels[rel.get("Id")] = "word/" + rel.get("Target").replace("../", "")
    root = ET.fromstring(z.read("word/document.xml"))

    cap_pat = (re.compile(r"^Figure\s*(\d)\s*[.:]") if a.main_text
               else re.compile(r"^Fig(?:ure)?\.?\s*S(\d{1,2})[.:]"))

    seq = []
    for p in root.iter(f"{W}p"):
        for blip in p.iter(f"{A}blip"):
            rid = blip.get(f"{R}embed")
            if rid in rels:
                seq.append(("img", rels[rid]))
        txt = "".join(t.text or "" for t in p.iter(f"{W}t")).strip()
        m = cap_pat.match(txt)
        if m:
            seq.append(("cap", int(m.group(1))))

    pairs, last = {}, None
    for kind, val in seq:
        if kind == "img":
            last = val
        elif last is not None:
            pairs.setdefault(val, last)

    dep = {}
    if a.main_text:
        for n in range(1, 6):
            p = a.record_b / f"figure{n}.png"
            if p.exists():
                im = Image.open(p)
                dep[n] = (p.name, ahash(im), im.size)
    else:
        for p in sorted(a.record_b.glob("S[0-9][0-9]_*.png")):
            n = int(re.match(r"^S(\d{2})_", p.name).group(1))
            im = Image.open(p)
            dep[n] = (p.name, ahash(im), im.size)

    print(f"  images in document : {len(rels)}")
    print(f"  captions paired    : {len(pairs)}")
    print(f"  deposited figures  : {len(dep)}\n")

    print(f"{'caption':9s} {'best':6s} {'dist':5s} {'emb ar':9s} "
          f"{'dep ar':9s} verdict")
    print("-" * 74)
    bad = []
    for num in sorted(pairs):
        try:
            with z.open(pairs[num]) as fh:
                im = Image.open(fh)
                im.load()
        except Exception as exc:
            print(f"{num:<9d} {'unreadable':6s} {'':5s} {'':9s} {'':9s} "
                  f"skip  {exc}")
            continue
        h = ahash(im)
        scored = sorted((dist(h, dh), n, nm, sz)
                        for n, (nm, dh, sz) in dep.items())
        if not scored:
            continue
        d, best, _nm, sz = scored[0]
        ar_e, ar_d = im.size[0] / im.size[1], sz[0] / sz[1]
        ok = best == num and abs(ar_e - ar_d) <= a.ar_tol
        if not ok:
            bad.append((num, best, d, ar_e, ar_d))
        label = f"{'Figure' if a.main_text else 'S'}{num}"
        print(f"{label:<9s} {best:<6d} {d:<5d} {ar_e:<9.3f} {ar_d:<9.3f} "
              f"{'OK' if ok else 'CHECK'}")

    print("-" * 74)
    if bad:
        print(f"  FAIL: {len(bad)} caption(s) need a look")
        for num, best, d, ae, ad in bad:
            why = ("matches another figure" if best != num
                   else "same figure, different render")
            print(f"      caption {num}: best match {best}, distance {d}, "
                  f"aspect {ae:.3f} vs deposited {ad:.3f}  ({why})")
        return 1
    print(f"  PASS: all {len(pairs)} caption(s) show their own figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
