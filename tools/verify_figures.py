#!/usr/bin/env python3
"""Verify manuscript figures by image matching against the .docx ground truth.

Instead of guessing which pipeline output corresponds to each additional figure, this
extracts the real figures embedded in a manuscript .docx, maps each to its "Figure A<n>"
caption (in document order), and finds the best-matching pipeline figure by blur-tolerant
image correlation. This tolerates the resolution / compression / mild blur introduced by
the Word (and any prior PDF) conversion.

Matching: each image is converted to greyscale, letter-boxed to a square (so aspect ratio
is preserved and becomes discriminative), Gaussian-blurred, z-normalised, and compared by
normalised correlation in [-1, 1]. Candidates are the pipeline's **medium-font** PNGs
(the manuscript uses the medium variant, not the _large one).

Outputs a report (per figure: best match + score + runners-up) so mappings can be accepted
only when they actually match. EMF (vector) figures cannot be rasterised here and are
reported as unverifiable.

    python tools/verify_figures.py --docx /path/Additional_figures_0605.docx --prefix A
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ANS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def extract_docx_figures(docx: Path, prefix: str, out_dir: Path) -> dict[int, Path]:
    """Return {fig_number: extracted_image_path} using document order + captions."""
    from lxml import etree
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx) as z:
        rels = etree.fromstring(z.read("word/_rels/document.xml.rels"))
        rel_map = {r.get("Id"): r.get("Target") for r in rels}
        body = etree.fromstring(z.read("word/document.xml"))[0]
        cap_re = re.compile(rf"(?:Figure|Fig\.?)\s*{prefix}\s*(\d+)")
        events = []  # ("cap", n) | ("img", target)
        for p in body.iter(WNS + "p"):
            txt = "".join(t.text or "" for t in p.iter(WNS + "t")).strip()
            m = re.match(cap_re, txt)
            if m:
                events.append(("cap", int(m.group(1))))
            for b in p.iter(ANS + "blip"):
                rid = b.get(RNS + "embed")
                if rid in rel_map:
                    events.append(("img", rel_map[rid]))
        # images appear just before their caption; walk and assign by nearest caption
        imgs = [e for e in events if e[0] == "img"]
        caps = [e for e in events if e[0] == "cap"]
        # robust when 1:1 and in order: the i-th image is figure caps[i]
        result = {}
        if len(imgs) == len(caps):
            for (_, tgt), (_, n) in zip(imgs, caps):
                result[n] = tgt
        else:  # fall back to sequential number = position
            for i, (_, tgt) in enumerate(imgs, 1):
                result[i] = tgt
        paths = {}
        for n, tgt in result.items():
            data = z.read("word/" + tgt)
            ext = os.path.splitext(tgt)[1].lower()
            dest = out_dir / f"{prefix}{n}{ext}"
            dest.write_bytes(data)
            paths[n] = dest
    return paths


def feat(path: Path, size: int = 256, sigma: float = 3.0):
    im = Image.open(path).convert("L")
    aspect = im.width / max(im.height, 1)
    im = ImageOps.pad(im, (size, size), color=255)  # letterbox -> aspect preserved
    a = np.asarray(im, dtype=float)
    a = gaussian_filter(a, sigma)
    a -= a.mean()
    s = a.std()
    if s > 1e-6:
        a /= s
    return a, aspect


def collect_candidates(results_dir: Path, extra_dirs: list[Path], refs: list[str]) -> list[Path]:
    # The additional figures use the Carter (UHGG) and combined references only — never
    # chimp/lev — so restrict the pool to those (plus non-ref dirs like occupancy_accumulation).
    cands: list[Path] = []
    for p in results_dir.rglob("*.png"):
        if "_large" in p.name:            # honour the medium-font requirement
            continue
        parts = p.relative_to(results_dir).parts
        top = parts[0] if parts else ""
        if top in ("UHGG", "lev", "combined", "chimp") and top not in refs:
            continue
        cands.append(p)
    for d in extra_dirs:
        if d.is_dir():
            cands += [p for p in d.rglob("*.png") if "_large" not in p.name]
        elif d.suffix == ".png" and d.exists():
            cands.append(d)
    return cands


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", type=Path, required=True)
    ap.add_argument("--prefix", default="A", help="figure prefix (A for additional figures)")
    ap.add_argument("--results-dir", type=Path, default=root / "results")
    ap.add_argument("--extra", type=Path, nargs="*",
                    default=[root / "main_text_figures", root / "accumulation_summary.png",
                             root / "accumulation_summary_v3.png"])
    ap.add_argument("--out", type=Path, default=root / "tools" / "figure_match_report.csv")
    ap.add_argument("--accept", type=float, default=0.45, help="min correlation to call a match")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--refs", nargs="*", default=["UHGG", "combined"],
                    help="reference dirs to search (additional figures use Carter/UHGG + combined)")
    args = ap.parse_args()

    ref_dir = Path("/tmp/_docx_figs")
    refs = extract_docx_figures(args.docx, args.prefix, ref_dir)
    print(f"extracted {len(refs)} {args.prefix}-figures from {args.docx.name}")

    cands = collect_candidates(args.results_dir, list(args.extra), args.refs)
    print(f"candidate pool (refs={args.refs}, medium-font, non-_large): {len(cands)} PNGs")
    print("precomputing candidate features …")
    cfeat = {}
    for c in cands:
        try:
            cfeat[c] = feat(c)
        except Exception:
            pass

    rel = lambda p: str(p.relative_to(root)) if p and root in p.parents else (str(p) if p else "")
    scored_by_fig: dict[int, list[tuple[float, Path]]] = {}
    emf = []
    for n in sorted(refs):
        rp = refs[n]
        if rp.suffix.lower() != ".png":
            emf.append(n)
            continue
        ra, rasp = feat(rp)
        s = []
        for c, (ca, casp) in cfeat.items():
            if abs(np.log((rasp + 1e-6) / (casp + 1e-6))) > 0.5:   # aspect within ~+/-50%
                continue
            s.append((float((ra * ca).mean()), c))
        s.sort(reverse=True)
        scored_by_fig[n] = s[:max(args.topk, 8)]

    # greedy unique assignment: each candidate used at most once, best global score first
    triples = sorted(((sc, n, c) for n, lst in scored_by_fig.items() for sc, c in lst),
                     key=lambda t: -t[0])
    assigned: dict[int, tuple[float, Path]] = {}
    used: set[Path] = set()
    for sc, n, c in triples:
        if n in assigned or c in used:
            continue
        assigned[n] = (sc, c)
        used.add(c)

    rows = []
    for n in sorted(refs):
        if n in emf:
            rows.append({"figure": f"{args.prefix}{n}", "status": "EMF_unverifiable",
                         "best": "", "score": "", "notes": "vector EMF; render to PNG to verify"})
            continue
        sc, c = assigned.get(n, (float("nan"), None))
        status = "MATCH" if (c is not None and sc >= args.accept) else "NO_MATCH"
        alts = " | ".join(f"{s:.3f}:{rel(cc)}" for s, cc in scored_by_fig.get(n, [])[:args.topk]
                          if cc != c)
        rows.append({"figure": f"{args.prefix}{n}", "status": status,
                     "best": rel(c), "score": f"{sc:.3f}", "notes": alts})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["figure", "status", "best", "score", "notes"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'fig':5} {'status':16} {'score':>6}  best match")
    for r in rows:
        print(f"{r['figure']:5} {r['status']:16} {r['score']:>6}  {r['best']}")
    n_match = sum(1 for r in rows if r["status"] == "MATCH")
    print(f"\n{n_match}/{len(rows)} matched (>= {args.accept}); report -> {args.out}")


if __name__ == "__main__":
    main()
