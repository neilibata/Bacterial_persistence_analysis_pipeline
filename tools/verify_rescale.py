#!/usr/bin/env python3
"""Check that reducing a figure to column width changed nothing but the size.

The claim made by persistence.figure_rescale is that the reduced figure is
the original photographically reduced: same layout, same relative type, same
everything, just smaller. That is checkable rather than assertable -- downscale
the original to the new figure's pixel dimensions and the two should agree.

Reports, per figure, the mean and 95th-percentile absolute pixel difference on
a 0-255 scale, and writes a side-by-side contact sheet with the difference map
so the check can also be made by eye.

    python tools/verify_rescale.py figure1
    python tools/verify_rescale.py --all

Writes docs/rescale_check/<name>.png and prints the numbers.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
BEFORE = ROOT / "archive" / "pre_column_width" / "main_text_figures"
AFTER = ROOT / "main_text_figures"
OUT = ROOT / "docs" / "rescale_check"

#: The verdict comes from comparing the two PDFs through the same rasteriser
#: at the same pixel size. That isolates the only question worth asking -- did
#: the drawing move -- from the fact that matplotlib's Agg and poppler's cairo
#: antialias differently, which on a dense figure costs ~10 grey levels per
#: pixel and says nothing about layout.
MEAN_OK = 0.5
P95_OK = 1.0

#: Figures whose layout was changed on purpose, with the reason. These are
#: expected to differ from the archived baseline; listing them keeps the tool
#: from reporting an intended edit as a regression, and keeps the reason with
#: the check rather than in someone's memory.
INTENTIONAL = {
    "figure1": "lifestyle colourbars moved inside the frame, into the empty "
               "Pacific west of the Americas",
    "figure3": "column headings added; width taken from 6.55 in to the full "
               "7.24 in column",
    "figure4": "legend raised from 9.5 to 11 pt (it printed at 4.6, under the "
               "5 pt floor); ticks, labels and titles up one point with it",
    "figure5": "panel A legend entries broken over two lines and panel E "
               "columns abbreviated, paying for 14 -> 16 pt so the smallest "
               "type clears the 5 pt floor",
}


def load_rgb(p):
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def raster_pdf(pdf, width_px, height_px=None):
    """Render a PDF to an exact pixel size with poppler."""
    pdf = Path(pdf)
    if not pdf.exists() or not shutil.which("pdftoppm"):
        return None
    with tempfile.TemporaryDirectory() as td:
        stem = str(Path(td) / "ref")
        r = subprocess.run(["pdftoppm", "-png", "-r", "300",
                            "-scale-to-x", str(width_px),
                            "-scale-to-y", str(height_px or -1),
                            str(pdf), stem], capture_output=True)
        if r.returncode != 0:
            return None
        pages = sorted(Path(td).glob("ref*.png"))
        if not pages:
            return None
        im = Image.open(pages[0])
        im.load()
        return im.convert("RGB")


def baseline_raster(name, width_px, height_px=None):
    """The untouched figure, rendered straight to the target pixel width.

    The archived PDF is vector, so rasterising it at the new width *is* the
    original photographically reduced -- a better reference than downscaling
    the old PNG, which bakes in its own resampling and its own padding.
    """
    return raster_pdf(BEFORE / f"{name}.pdf", width_px, height_px)


def _unused(name, width_px, height_px=None):
    pdf = BEFORE / f"{name}.pdf"
    if not pdf.exists() or not shutil.which("pdftoppm"):
        return None
    with tempfile.TemporaryDirectory() as td:
        stem = str(Path(td) / "ref")
        # Force both dimensions: a one-pixel difference in height would
        # otherwise be absorbed by a resample, and on a text-dense figure that
        # sub-pixel shift dominates the difference metric and hides whether
        # anything actually moved.
        r = subprocess.run(["pdftoppm", "-png", "-r", "300",
                            "-scale-to-x", str(width_px),
                            "-scale-to-y", str(height_px or -1),
                            str(pdf), stem],
                           capture_output=True)
        if r.returncode != 0:
            return None
        pages = sorted(Path(td).glob("ref*.png"))
        if not pages:
            return None
        im = Image.open(pages[0])
        im.load()
        return im.convert("RGB")


def compare(name):
    b_pdf = BEFORE / f"{name}.pdf"
    a_pdf, a_png = AFTER / f"{name}.pdf", AFTER / f"{name}.png"
    if not a_png.exists():
        return dict(name=name, error=f"no current figure at {a_png}")

    after = load_rgb(a_png)
    w, h = after.size

    # The verdict: the same drawing through the same renderer at the same size.
    verdict_src = "old PDF vs new PDF, one renderer"
    ref_pdf = (baseline_raster(name, w, h) if b_pdf.exists() else None)
    cur_pdf = (raster_pdf(a_pdf, w, h) if a_pdf.exists() else None)
    if ref_pdf is not None and cur_pdf is not None:
        d = np.asarray(ImageChops.difference(ref_pdf, cur_pdf),
                       dtype=np.float32).mean(axis=2)
        mean, p95 = float(d.mean()), float(np.percentile(d, 95))
    else:
        # No vector pair to compare: fall back to the raster, and say so, with
        # thresholds that admit the renderer difference.
        verdict_src = "archived PNG vs new PNG (no vector pair)"
        b_png = BEFORE / f"{name}.png"
        if not b_png.exists():
            return dict(name=name, error="nothing to compare against")
        ref = load_rgb(b_png).resize((w, h), Image.LANCZOS)
        d = np.asarray(ImageChops.difference(ref, after),
                       dtype=np.float32).mean(axis=2)
        mean, p95 = float(d.mean()), float(np.percentile(d, 95))
        ref_pdf = ref

    ar_b = (ref_pdf.size[0] / ref_pdf.size[1]) if ref_pdf else w / h
    res = dict(name=name, before=f"{ref_pdf.size[0]}x{ref_pdf.size[1]}",
               after=f"{w}x{h}", aspect_before=ar_b, aspect_after=w / h,
               aspect_drift=abs(ar_b - w / h) / ar_b,
               mean=mean, p95=p95, ref=verdict_src)
    thr_mean = MEAN_OK if cur_pdf is not None else 6.0
    thr_p95 = P95_OK if cur_pdf is not None else 24.0
    res["ok"] = (mean <= thr_mean and p95 <= thr_p95
                 and res["aspect_drift"] <= 0.01)

    # Contact sheet from the rasters, which is what a person wants to look at.
    OUT.mkdir(parents=True, exist_ok=True)
    shown = cur_pdf if cur_pdf is not None else after
    dv = np.asarray(ImageChops.difference(ref_pdf, shown),
                    dtype=np.float32).mean(axis=2)
    amp = np.clip(dv * 6, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(255 - amp).convert("RGB")
    pad, label = 8, 26
    sheet = Image.new("RGB", (w * 3 + pad * 4, h + label + pad * 2), "white")
    for i, (img, cap) in enumerate(((ref_pdf, "before, at the new size"),
                                    (shown, "after"),
                                    (diff_img, "difference x6"))):
        x = pad + i * (w + pad)
        sheet.paste(img, (x, label + pad))
        ImageDraw.Draw(sheet).text((x, pad), cap, fill="black")
    out = OUT / f"{name}.png"
    sheet.thumbnail((2400, 2400), Image.LANCZOS)
    sheet.save(out, dpi=(150, 150))
    res["sheet"] = str(out.relative_to(ROOT))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    names = (sorted(p.stem for p in BEFORE.glob("figure*.png"))
             if a.all or not a.name else a.name)
    rows = [compare(n) for n in names]

    print(f"  {'figure':10}{'before':>13}{'after':>13}{'mean':>8}{'p95':>8}"
          f"{'aspect':>9}   verdict")
    bad = 0
    for r in rows:
        if r.get("error"):
            print(f"  {r['name']:10}{r['error']}")
            bad += 1
            continue
        if r["ok"]:
            verdict = "same picture"
        elif r["name"] in INTENTIONAL:
            verdict = "changed on purpose"
        else:
            verdict = "DIFFERS"
            bad += 1
        print(f"  {r['name']:10}{r['before']:>13}{r['after']:>13}"
              f"{r['mean']:>8.2f}{r['p95']:>8.1f}"
              f"{r['aspect_drift'] * 100:>8.2f}%   {verdict}")
    for n, why in INTENTIONAL.items():
        if any(r["name"] == n for r in rows):
            print(f"\n  {n} was changed deliberately: {why}")
    for r in rows:
        if r.get("sheet"):
            print(f"    {r['name']}: {r['sheet']}   (reference: {r['ref']})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
