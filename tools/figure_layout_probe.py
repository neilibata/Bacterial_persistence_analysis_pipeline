#!/usr/bin/env python3
"""Probe rendered figures for layout and text problems, without re-plotting.

Everything here is measured from the saved raster: connected components of dark
ink are split into text-like glyphs and everything else, glyphs are grouped into
runs (words/lines), and each run is reported with its bounding box and the point
size it would carry at the placement width. Panels are inferred from whitespace
projections, so gutters and outer margins can be measured too.

The point of the tool is diagnosis only. It changes nothing and re-runs no
plotting code, so a full sweep of the additional figures takes a few seconds.

    python tools/figure_layout_probe.py staged_figures/*.png -o docs/layout_probe.json

Glyph ink height is 0.714 x nominal point size (same calibration as
tools/audit_figure_text.py, measured against text rendered at known sizes).
"""
from __future__ import annotations

import argparse, base64, io, json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

GLYPH_INK_RATIO = 0.714
PLACEMENT_WIDTH_IN = 6.3     # a typical Word page text column
DARK = 110                   # grey level below which a pixel counts as ink
ANY_INK = 245                # grey level below which a pixel is not background

# A run is text if its glyphs sit on a common baseline and are close together.
RUN_GAP_EM = 1.10            # horizontal gap, in units of glyph height
RUN_BASELINE_EM = 0.55       # vertical centre offset tolerated within a run

BAND_INK_FRAC = 0.004        # a row/column this empty counts as whitespace
BAND_MIN_FRAC = 0.030        # ...and is only a "gap" if it spans this much
OCCLUSION_INK = 0.020        # non-text ink under a run that counts as overlap


def _components(a: np.ndarray):
    lab, n = ndimage.label(a < DARK)
    if n == 0:
        return [], lab
    return ndimage.find_objects(lab), lab


def _split_glyphs(objs, H, W):
    """Partition components into text-like glyphs and everything else."""
    glyphs, other = [], []
    for sl in objs:
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        box = (x0, y0, x1, y1)
        if h < 3 or w < 2:
            continue                                   # speckle
        if h > 0.10 * H or w > 12 * h or (w / max(h, 1) > 6 and h < 6):
            other.append(box)                          # axes, curves, rules, ticks
        else:
            glyphs.append(box)
    return glyphs, other


def _runs(glyphs, px_per_pt):
    """Group glyphs into words/lines sharing a baseline."""
    if not glyphs:
        return []
    hs = np.array([b[3] - b[1] for b in glyphs], dtype=float)
    med = float(np.median(hs))
    # Fragments (the dot of an i, commas, subscripts) are real ink but not a
    # legibility constraint; they would drag every run's size downward.
    keep = [b for b, h in zip(glyphs, hs) if h >= 0.45 * med]
    keep.sort(key=lambda b: (round((b[1] + b[3]) / 2 / max(med, 1)), b[0]))

    runs, cur = [], []
    for b in keep:
        if not cur:
            cur = [b]; continue
        p = cur[-1]
        ph = p[3] - p[1]
        same_line = abs((b[1] + b[3]) / 2 - (p[1] + p[3]) / 2) <= RUN_BASELINE_EM * max(ph, 1)
        close = (b[0] - p[2]) <= RUN_GAP_EM * max(ph, 1)
        if same_line and close and b[0] >= p[0]:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    if cur:
        runs.append(cur)

    out = []
    for r in runs:
        x0 = min(b[0] for b in r); y0 = min(b[1] for b in r)
        x1 = max(b[2] for b in r); y1 = max(b[3] for b in r)
        ink_h = float(np.median([b[3] - b[1] for b in r]))
        out.append(dict(box=[x0, y0, x1, y1], n=len(r),
                        pt=round(ink_h / GLYPH_INK_RATIO / px_per_pt, 2)))
    return out


def _bands(ink_rows, total, axis_len):
    """Contiguous stretches of near-empty rows/columns."""
    empty = ink_rows < BAND_INK_FRAC
    bands, i = [], 0
    while i < axis_len:
        if empty[i]:
            j = i
            while j < axis_len and empty[j]:
                j += 1
            if (j - i) >= BAND_MIN_FRAC * axis_len:
                bands.append([int(i), int(j)])
            i = j
        else:
            i += 1
    return bands


def _occluded(run, other_lab, non_text_mask):
    x0, y0, x1, y1 = run["box"]
    sub = non_text_mask[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    return float(sub.mean())


def _overlaps(runs):
    """Pairs of text runs whose boxes genuinely overlap."""
    hits = []
    for i in range(len(runs)):
        ax0, ay0, ax1, ay1 = runs[i]["box"]
        aa = max((ax1 - ax0) * (ay1 - ay0), 1)
        for j in range(i + 1, len(runs)):
            bx0, by0, bx1, by1 = runs[j]["box"]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox <= 0 or oy <= 0:
                continue
            bb = max((bx1 - bx0) * (by1 - by0), 1)
            frac = (ox * oy) / min(aa, bb)
            if frac > 0.18:                       # a touch is not a collision
                hits.append([i, j, round(frac, 3)])
    return hits


def _thumb(im: Image.Image, width: int):
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    best = None
    for fmt, kw in (("PNG", dict(optimize=True)),
                    ("JPEG", dict(quality=82, optimize=True))):
        buf = io.BytesIO()
        im.convert("RGB" if fmt == "JPEG" else "P",
                   **({"palette": Image.ADAPTIVE, "colors": 256} if fmt == "PNG" else {})
                   ).save(buf, fmt, **kw)
        if best is None or buf.tell() < best[1]:
            best = (fmt.lower(), buf.tell(), base64.b64encode(buf.getvalue()).decode())
    return f"data:image/{best[0]};base64,{best[2]}", best[1]


def authored_size(src: Path):
    """Figure size in inches, from the source PNG header (no pixels decoded)."""
    try:
        with Image.open(src) as im:
            w, h = im.size
            dpi = im.info.get("dpi")
        if not dpi or not dpi[0]:
            return None
        return dict(src=str(src), w_in=round(w / dpi[0], 2),
                    h_in=round(h / dpi[1], 2), dpi=round(float(dpi[0])))
    except Exception:
        return None


def probe(path: Path, width_in: float, thumb_px: int, src: Path | None = None):
    im = Image.open(path)
    g = np.array(im.convert("L"))
    H, W = g.shape
    px_per_pt = (W / width_in) / 72.0

    objs, _ = _components(g)
    glyphs, other = _split_glyphs(objs, H, W)
    runs = _runs(glyphs, px_per_pt)

    non_text = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in other:
        non_text[y0:y1, x0:x1] = True

    for r in runs:
        r["occ"] = round(_occluded(r, None, non_text), 4)

    ink = (g < ANY_INK)
    row_f = ink.mean(axis=1)
    col_f = ink.mean(axis=0)

    # The space budget. Text real estate is the union of glyph boxes; graphics
    # ink is everything else that is not background. Their ratio is the number
    # that matters: a figure where text outweighs data has given the page to
    # the labels, and shrinking the canvas afterwards only makes the labels
    # illegible without giving the panels their space back.
    tmask = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in glyphs:
        tmask[y0:y1, x0:x1] = True
    data = ink & ~tmask
    # How far the graphics reach across the page; the rest is gutter and margin.
    reach_r = float((data.mean(axis=1) > 0.004).mean())
    reach_c = float((data.mean(axis=0) > 0.004).mean())

    pts = np.array([r["pt"] for r in runs]) if runs else np.array([])
    summary = dict(
        n_runs=len(runs),
        pt10=round(float(np.percentile(pts, 10)), 2) if pts.size else None,
        pt50=round(float(np.percentile(pts, 50)), 2) if pts.size else None,
        pt90=round(float(np.percentile(pts, 90)), 2) if pts.size else None,
        span=round(float(np.percentile(pts, 90) / max(np.percentile(pts, 10), 1e-6)), 2)
              if pts.size else None,
        ink=round(float(ink.mean()), 4),
        text_frac=round(float(tmask.mean()), 4),
        data_frac=round(float(data.mean()), 4),
        gt=round(float(data.sum() / max(tmask.sum(), 1)), 2),
        reach=round(reach_r * reach_c, 3),
        reach_r=round(reach_r, 3), reach_c=round(reach_c, 3),
    )

    auth = authored_size(src) if src else None
    if auth and pts.size:
        # Rendered pt = nominal pt x (placement width / authored width), so the
        # nominal sizes actually set in the plotting code invert back out, and
        # the authored width needed to hit a target follows from the same ratio.
        k = auth["w_in"] / width_in
        summary["authored_w_in"] = auth["w_in"]
        summary["authored_h_in"] = auth["h_in"]
        summary["shrink"] = round(k, 2)
        summary["nominal_pt50"] = round(summary["pt50"] * k, 1)
        summary["nominal_pt10"] = round(summary["pt10"] * k, 1)
        summary["nominal_pt90"] = round(summary["pt90"] * k, 1)

    thumb, nbytes = _thumb(im, thumb_px)
    return dict(
        name=path.stem, file=str(path), w=W, h=H, authored=auth,
        width_in=width_in, aspect=round(W / H, 3),
        runs=runs,
        overlaps=_overlaps(runs),
        row_bands=_bands(row_f, H, H),
        col_bands=_bands(col_f, W, W),
        summary=summary, thumb=thumb, thumb_bytes=nbytes,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, default=[])
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--width-in", type=float, default=PLACEMENT_WIDTH_IN,
                    help="placement width the figure will occupy (default 6.3)")
    ap.add_argument("--thumb-px", type=int, default=1000)
    ap.add_argument("--manifest", type=Path,
                    help="slot manifest, to resolve each staged file back to the "
                         "source PNG and recover the authored figure size")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--from-manifest", action="store_true",
                    help="probe the full-resolution source PNGs named by the "
                         "manifest instead of any staged copies. Downsampled "
                         "copies lose thin data strokes and drop sub-3px glyphs "
                         "below the noise floor, which biases both the ink "
                         "budget and the median point size; measure the original.")
    a = ap.parse_args()

    # slot -> source PNG, so authored figure size can be read from the original
    srcmap = {}
    if a.manifest and a.manifest.exists():
        for line in a.manifest.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            slot, ref, stem = parts[0], parts[1], parts[2]
            for cand in (f"{stem}_notitle", stem):
                pth = a.results / ref / f"{cand}.png"
                if pth.exists():
                    srcmap[slot] = pth
                    break

    def _key(t):
        return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", t)]

    if a.from_manifest:
        todo = [(slot, src, src) for slot, src in sorted(srcmap.items(), key=lambda kv: _key(kv[0]))]
    else:
        todo = [(f.stem.split("_")[0], f, srcmap.get(f.stem.split("_")[0]))
                for f in sorted(a.files, key=lambda p: _key(p.stem))]

    figs, total = [], 0
    for slot, f, src in todo:
        if not f.exists():
            print(f"  [skip] {f}", file=sys.stderr); continue
        d = probe(f, a.width_in, a.thumb_px, src)
        if a.from_manifest:
            d["name"] = f"{slot}_{f.stem}"
        total += d.pop("thumb_bytes")
        figs.append(d)
        s = d["summary"]
        print(f"  {d['name'][:44]:<46}{s['n_runs']:>5} runs  "
              f"pt50 {s['pt50']}  span {s['span']}x  "
              f"g:t {s['gt']:<6}reach {s['reach']*100:>3.0f}%  "
              f"authored {s.get('authored_w_in','?')}in")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(width_in=a.width_in, figures=figs)))
    print(f"\n  {len(figs)} figures -> {a.out}  "
          f"({a.out.stat().st_size/1e6:.1f} MB, thumbs {total/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
