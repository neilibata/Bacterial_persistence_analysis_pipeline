#!/usr/bin/env python3
"""Resize pipeline PNGs to the geometry the manuscript documents already use.

The pipeline writes `*_notitle.png` at 300 dpi and page width, so Word's
natural insert size is ~15.8 in — about 2.5x a text column. The figures already
embedded in Persistence_Additional_figures_*.docx are 220 dpi at ~1378 px, i.e.
a natural size of 6.3 in. Dropping raw pipeline output in beside them therefore
renders the type about 40% of the size of the surrounding figures, even though
the font design is identical (glyph height per unit width matches to ~3%).

This resamples to that same 6.3 in / 220 dpi geometry so new figures drop in at
the size the document expects.

    python tools/stage_figures_for_docx.py --out staged_figures FILE [FILE ...]
    python tools/stage_figures_for_docx.py --out staged_figures --width-in 6.3 \
        results/UHGG/depth_audit/depth_by_lifestyle_notitle.png
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from PIL import Image

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--manifest", type=Path,
                    help="stage every slot listed in a manifest "
                         "(slot|reference|stem)")
    ap.add_argument("--out", type=Path, default=Path("staged_figures"))
    ap.add_argument("--results", type=Path, default=Path("results"),
                    help="results/ tree the manifest stems resolve under "
                         "(default: ./results; a package's own for a redraw)")
    ap.add_argument("--width-in", type=float, default=6.3,
                    help="target width in inches (default 6.3, the text column)")
    ap.add_argument("--dpi", type=int, default=220,
                    help="dpi to stamp (default 220, matching the existing figures)")
    ap.add_argument("--clean", action="store_true",
                    help="remove PNGs in --out that this run did not write, so "
                         "renumbered or withdrawn slots cannot linger")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any manifest slot has no output")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target_px = int(round(args.width_in * args.dpi))

    files: list[tuple[Path, str]] = [(f, f.name) for f in args.files]
    skipped = 0
    if args.manifest:
        for line in args.manifest.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            slot, ref, stem = parts[0], parts[1], parts[2]
            src = None
            # The no-suptitle twin is the one that goes in the document; the
            # caption is carried by the document itself.
            for cand in (f"{stem}_notitle", stem):
                p = args.results / ref / (cand + ".png")
                if p.exists():
                    src = p
                    break
            if src is None:
                print(f"  [skip] {slot} — no output under {args.results}/{ref}/{stem}")
                skipped += 1
                continue
            # Zero-pad S-numbers so the directory sorts in slot order, matching
            # supplementary_figures/.
            m = re.fullmatch(r"S(\d+)([a-z]?)", slot)
            label = f"S{int(m.group(1)):02d}{m.group(2)}" if m else slot
            files.append((src, f"{label}_{Path(stem).name}.png"))

    n_missing = sum(1 for f, _ in files if not f.exists()) + skipped
    written: set[str] = set()
    for f, name in files:
        if not f.exists():
            print(f"  [skip] {f} — not found"); continue
        im = Image.open(f)
        w, h = im.size
        if w == target_px:
            out_im = im
        else:
            out_im = im.resize((target_px, max(1, int(round(h * target_px / w)))),
                               Image.LANCZOS)
        dest = args.out / name
        out_im.save(dest, dpi=(args.dpi, args.dpi))
        written.add(name)
        ow, oh = out_im.size
        print(f"  {f.name}: {w}x{h} -> {ow}x{oh} @ {args.dpi} dpi "
              f"({ow/args.dpi:.1f} x {oh/args.dpi:.1f} in)  -> {dest}")
    if args.clean:
        for old in sorted(args.out.glob("*.png")):
            if old.name not in written:
                old.unlink()
                print(f"  [clean] removed {old.name} — not produced by this run")
    if n_missing:
        print(f"  {n_missing} slot(s) had no output")
        if args.strict:
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
