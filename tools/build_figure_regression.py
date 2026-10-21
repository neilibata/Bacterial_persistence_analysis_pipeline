#!/usr/bin/env python3
"""Build a before/after viewer for the Additional (supplementary) figures.

"Before" is every figure embedded in a manuscript docx (the layout the
reader last saw); "after" is a run's staged S01..S32 PNGs. Each pair is
measured the same way, on the image alone, so a layout regression shows up
as a number as well as a picture:

  gap_v / gap_h   blank horizontal / vertical bands *inside* the content box,
                  as a fraction of its height / width. More white space
                  between panels raises these.
  max_v / max_h   the single widest such band.
  margin          blank border outside the content box.
  aspect          content-box width / height.

A slot is flagged when a gap fraction, widest band or the outer margin grows
by more than --flag-pp percentage points, or the aspect ratio moves by >10%.

The page is one self-contained HTML file (JPEG previews as data URIs), so
it can be opened anywhere or shared as it is.

    python tools/build_figure_regression.py \\
        --before-docx path/to/additional_figures.docx \\
        --after-dir   <run>/staged_figures \\
        --out         docs/figure_regression.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
CAPTION = re.compile(r"^\s*(?:Figure|Fig\.)\s*S(\d+)\b")
MEASURE_W = 1600          # measurement resolution (fractions are scale-free)
PREVIEW_W = 1500
INK = 235                 # grey level below which a pixel counts as ink


# ── docx → {slot: image bytes} ────────────────────────────────────────────────

def docx_figures(path: Path) -> dict[int, tuple[str, bytes]]:
    """Map 'Figure S<n>' captions to the image placed with them.

    An image belongs to the caption in its own paragraph, or else to the next
    caption after it (figures sit above their captions).
    """
    z = zipfile.ZipFile(path)
    rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels.findall("rel:Relationship", NS)}
    body = ET.fromstring(z.read("word/document.xml")).find("w:body", NS)
    out: dict[int, tuple[str, bytes]] = {}
    pending: list[str] = []
    for p in body.iter(f"{{{NS['w']}}}p"):
        embeds = [b.get(f"{{{NS['r']}}}embed") for b in p.iter(f"{{{NS['a']}}}blip")]
        pending += [e for e in embeds if e]
        text = "".join(t.text or "" for t in p.iter(f"{{{NS['w']}}}t"))
        m = CAPTION.match(text)
        if m and pending:
            n = int(m.group(1))
            rid = pending[-1]            # the image nearest the caption
            name = "word/" + target[rid].lstrip("/").removeprefix("word/")
            out.setdefault(n, (Path(name).name, z.read(name)))
            pending = []
    return out


# ── measurement ───────────────────────────────────────────────────────────────

def _vector_to_png(data: bytes) -> bytes | None:
    """Rasterise an EMF/WMF picture from Word.

    Word for Mac stores pasted PDFs as EMF with the original PDF inside an
    EMR_COMMENT; PIL cannot draw EMF, so pull that PDF (or an embedded PNG)
    out and rasterise it with macOS `sips`. Returns None if neither exists.
    """
    i = data.find(b"\x89PNG\r\n\x1a\n")
    if i >= 0:
        j = data.find(b"IEND", i)
        return data[i:j + 8]
    i, j = data.find(b"%PDF"), data.rfind(b"%%EOF")
    if i < 0 or j < i:
        return None
    import shutil, subprocess, tempfile
    if not shutil.which("sips"):
        return None
    with tempfile.TemporaryDirectory() as td:
        pdf, png = Path(td) / "f.pdf", Path(td) / "f.png"
        pdf.write_bytes(data[i:j + 5])
        subprocess.run(["sips", "-s", "format", "png", "--resampleWidth", "3000",
                        str(pdf), "--out", str(png)], capture_output=True)
        return png.read_bytes() if png.exists() else None


def _load(img_bytes_or_path) -> Image.Image:
    if isinstance(img_bytes_or_path, bytes) and img_bytes_or_path[:4] in (
            b"\x01\x00\x00\x00", b"\xd7\xcd\xc6\x9a"):          # EMF / placeable WMF
        png = _vector_to_png(img_bytes_or_path)
        if png is None:
            raise ValueError("vector picture with no embedded raster or PDF")
        img_bytes_or_path = png
    src = io.BytesIO(img_bytes_or_path) if isinstance(img_bytes_or_path, bytes) else img_bytes_or_path
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def _bands(ink_1d: np.ndarray, min_run: int):
    """Blank runs strictly inside the first..last ink index."""
    idx = np.flatnonzero(ink_1d)
    if idx.size == 0:
        return 0, 0, 0, 0
    lo, hi = idx[0], idx[-1]
    inner = ~ink_1d[lo:hi + 1]
    runs, cur = [], 0
    for v in inner:
        if v:
            cur += 1
        else:
            if cur >= min_run:
                runs.append(cur)
            cur = 0
    span = hi - lo + 1
    return sum(runs) / span, (max(runs) / span if runs else 0.0), lo, hi


def measure(im: Image.Image) -> dict:
    w, h = im.size
    small = im.resize((MEASURE_W, max(1, round(MEASURE_W * h / w))), Image.LANCZOS)
    g = np.asarray(small.convert("L"))
    ink = g < INK
    H, W = ink.shape
    gv, mv, top, bot = _bands(ink.any(axis=1), max(2, round(0.006 * H)))
    gh, mh, left, right = _bands(ink.any(axis=0), max(2, round(0.006 * W)))
    ch, cw = bot - top + 1, right - left + 1
    margin = 1 - (ch * cw) / (H * W) if ch > 0 and cw > 0 else 0
    return dict(px=f"{w}×{h}", gap_v=gv, max_v=mv, gap_h=gh, max_h=mh,
                margin=margin, aspect=cw / ch if ch else 0)


def preview(im: Image.Image) -> str:
    w, h = im.size
    if w > PREVIEW_W:
        im = im.resize((PREVIEW_W, round(PREVIEW_W * h / w)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── page ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before-docx", type=Path, required=True)
    ap.add_argument("--after-dir", type=Path, required=True,
                    help="Directory of S01_*.png … S32_*.png (staged_figures/).")
    ap.add_argument("--override", action="append", default=[], metavar="N=PNG",
                    help="Use this PNG as 'after' for slot N (e.g. a redrawn figure).")
    ap.add_argument("--before-label", default=None)
    ap.add_argument("--after-label", default=None)
    ap.add_argument("--flag-pp", type=float, default=2.0)
    ap.add_argument("--template", type=Path,
                    default=Path(__file__).with_name("figure_regression_template.html"))
    ap.add_argument("--out", type=Path, default=Path("docs/figure_regression.html"))
    args = ap.parse_args()

    before = docx_figures(args.before_docx.expanduser())
    overrides = {int(k): Path(v).expanduser() for k, v in (o.split("=", 1) for o in args.override)}
    after = {}
    for p in sorted(args.after_dir.expanduser().glob("S[0-9][0-9]_*.png")):
        after[int(p.name[1:3])] = p
    after.update(overrides)

    slots = []
    for n in sorted(set(before) | set(after)):
        row = dict(n=n)
        if n in before:
            name, data = before[n]
            try:
                im = _load(data)
                row["before"] = dict(src=preview(im), file=name, **measure(im))
            except Exception as exc:
                print(f"  S{n:02d}  before image unreadable ({name}): {exc}")
        if n in after:
            im = _load(after[n])
            row["after"] = dict(src=preview(im), file=after[n].name, **measure(im))
        if "before" in row and "after" in row:
            b, a = row["before"], row["after"]
            d = {k: round(100 * (a[k] - b[k]), 1) for k in ("gap_v", "max_v", "gap_h", "max_h", "margin")}
            d["aspect_pct"] = round(100 * (a["aspect"] / b["aspect"] - 1), 1) if b["aspect"] else 0
            reasons = [k for k in ("gap_v", "max_v", "gap_h", "max_h", "margin") if d[k] > args.flag_pp]
            if abs(d["aspect_pct"]) > 10:
                reasons.append("aspect")
            row["delta"], row["flags"] = d, reasons
        slots.append(row)
        print(f"  S{n:02d}  " + (f"flags={row.get('flags')}" if "delta" in row else "unpaired"))

    data = dict(
        before_label=args.before_label or args.before_docx.name,
        after_label=args.after_label or str(args.after_dir),
        flag_pp=args.flag_pp, slots=slots)
    html = args.template.read_text().replace(
        "/*DATA*/", json.dumps(data).replace("</", "<\\/"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), "
          f"{sum(bool(s.get('flags')) for s in slots)} flagged of {len(slots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
