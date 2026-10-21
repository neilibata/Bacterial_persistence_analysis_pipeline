"""
make_supp_figure_fixed_bases.py
--------------------------------
Assembles 24 subplots (4 rows × 6 columns) for a given reference genome.

Rows    : Pooled accumulation | Pooled params vs N |
          Continent accumulation | Continent params vs N
Columns : 10 kb | 50 kb | 100 kb | 200 kb | 500 kb | 1 Mb

Outputs : supp_figure_fixed_bases_{ref}.pdf
          supp_figure_fixed_bases_{ref}.png

Usage
-----
    python make_supp_figure_fixed_bases.py --ref UHGG
    python make_supp_figure_fixed_bases.py --ref lev
    python make_supp_figure_fixed_bases.py --ref UHGG --indir results/UHGG --outdir figures/

Requirements
------------
    pip install matplotlib pillow

Axis-label visibility
---------------------
Every panel retains its tick marks and tick-number labels.
The axis *title* label is stripped selectively:

  Left column   : y-axis title kept;  x-axis title stripped from image,
                  canonical "Number of samples (N)" drawn by matplotlib.
  Bottom row    : x-tick labels kept; x-axis title stripped from image
                  and replaced by canonical text.  y-axis title stripped.
  Bottom-left   : both y-axis title and canonical x-axis title shown.
  All others    : both axis titles stripped; tick marks/numbers remain.

Spine detection is fully automatic (no hard-coded pixel offsets).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

# ── layout constants ──────────────────────────────────────────────────────────

SIZES      = ["10kb", "30kb", "100kb", "300kb", "500kb", "1Mb"]
COL_LABELS = ["10 kb", "50 kb", "100 kb", "200 kb", "500 kb", "1 Mb"]

ROW_LABELS = [
    "Pooled\naccumulation",
    "Pooled\nparams vs N",
    "Continent\naccumulation",
    "Continent\nparams vs N",
]

# Row indices (0-based) whose images should have the top 15 % cropped.
CROP_TOP_ROWS = {1, 3}
CROP_TOP_FRAC = 0.15

FILE_TEMPLATES = [
    "accumulation_50pct_{ref}_flat_{size}_pooled_notitle.png",
    "accum_flat_{size}_pooled_params_vsN_notitle.png",
    "accumulation_50pct_{ref}_flat_{size}_continent_notitle.png",
    "accum_flat_{size}_continent_params_vsN_notitle.png",
]

NROWS = len(FILE_TEMPLATES)
NCOLS = len(SIZES)

FIG_DPI          = 150
LABEL_FONTSIZE   = 36
X_AXIS_LABEL     = "Number of samples (N)"
LEFT_MARGIN      = 2.20   # inches — space for row labels + full left margin of col 0
TOP_MARGIN       = 1.10
RIGHT_MARGIN     = 0.25
H_GAP_INNER      = 0.05   # inches gap between col j axes-right and col j+1 tick-labels-left
V_GAP            = 0.10
BOTTOM_LABEL_H   = 0.70   # inches reserved for the x-axis title below the bottom row
BOTTOM_MARGIN    = 0.20


# ── spine / margin detection ──────────────────────────────────────────────────

def _max_run(arr: np.ndarray) -> int:
    """Length of the longest contiguous True run in a 1-D boolean array."""
    arr = np.asarray(arr, bool)
    if not arr.any():
        return 0
    pad = np.concatenate([[False], arr, [False]])
    d = np.diff(pad.astype(np.int8))
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def _clusters(profile: np.ndarray, thresh: float = 0.005) -> list[tuple[int, int]]:
    """Return (start, end) pairs for contiguous above-threshold runs."""
    ok, res, in_c = profile > thresh, [], False
    for i, v in enumerate(ok):
        if v and not in_c:
            in_c, start = True, i
        elif not v and in_c:
            in_c = False
            res.append((start, i - 1))
    if in_c:
        res.append((start, len(profile) - 1))
    return res


def _gap_boundary(clusters: list[tuple[int, int]]) -> int:
    """
    Return the index just after the last tick cluster (the start of the gap
    before the axis *title* cluster, which sits at the end of the sequence).
    We identify this as the start of the **largest gap** between any two
    consecutive clusters.
    """
    if not clusters:
        return 0
    if len(clusters) == 1:
        return clusters[0][1] + 1
    max_gap, boundary = -1, clusters[-1][1] + 1
    for a, b in zip(clusters[:-1], clusters[1:]):
        g = b[0] - a[1] - 1
        if g > max_gap:
            max_gap, boundary = g, a[1] + 1
    return boundary


def detect_panel_margins(img: np.ndarray, dark_thresh: int = 80,
                          min_frac: float = 0.18) -> dict:
    """
    Detect spine positions and the boundary between axis titles and tick labels.

    Returns a dict with keys:
      left_spine   : x column of the left spine
      right_spine  : x column of the right spine
      bottom_spine : y row of the bottom spine
      y_label_end  : first column that contains tick labels (strip 0..y_label_end-1
                     for non-left-column panels; keeps ticks & tick numbers)
      tick_bottom  : rows below bottom_spine to keep (tick marks + tick numbers,
                     up to but not including the axis title region)
      axes_w       : right_spine - left_spine
    """
    gray = img.mean(axis=2)
    dark = gray < dark_thresh
    h, w = dark.shape

    mh = max(1, int(h * min_frac))
    mw = max(1, int(w * min_frac))

    left_spine   = next((c for c in range(w)        if _max_run(dark[:, c]) >= mh), 0)
    right_spine  = next((c for c in range(w-1,-1,-1) if _max_run(dark[:, c]) >= mh), w-1)
    bottom_spine = next((r for r in range(h-1,-1,-1) if _max_run(dark[r, :]) >= mw), h-1)

    # Y-axis: cluster 0 = rotated axis title, clusters 1+ = tick labels + tick marks.
    col_prof = dark[:, :left_spine].mean(axis=0) if left_spine > 0 else np.zeros(0)
    lc = _clusters(col_prof)
    if len(lc) >= 2:
        y_label_end = lc[1][0]          # start of first tick-label cluster
    elif lc:
        y_label_end = lc[0][1] + 1
    else:
        y_label_end = 0
    y_label_end = min(y_label_end, left_spine)

    # X-axis: rows below bottom spine.  Find the gap that separates tick labels
    # from the axis title text.
    bm = dark[bottom_spine+1:, :].mean(axis=1) if bottom_spine + 1 < h else np.zeros(0)
    bc = _clusters(bm)
    tick_bottom = _gap_boundary(bc)
    tick_bottom = min(tick_bottom, h - bottom_spine - 1)

    return {
        "left_spine":   left_spine,
        "right_spine":  right_spine,
        "bottom_spine": bottom_spine,
        "y_label_end":  y_label_end,
        "tick_bottom":  tick_bottom,
        "axes_w":       right_spine - left_spine,
    }


def get_row_margins(imgs: list[list[np.ndarray]]) -> list[dict]:
    """Compute per-row median margins (robust to single outlier images)."""
    row_margins = []
    for row_idx, row in enumerate(imgs):
        detections = [detect_panel_margins(img) for img in row]
        rm = {
            key: int(np.median([d[key] for d in detections]))
            for key in ("left_spine", "right_spine", "bottom_spine",
                        "y_label_end", "tick_bottom", "axes_w")
        }
        print(
            f"Row {row_idx}: left_spine={rm['left_spine']}  "
            f"y_label_end={rm['y_label_end']}  "
            f"tick_label_w={rm['left_spine'] - rm['y_label_end']}  "
            f"bottom_spine={rm['bottom_spine']}  "
            f"tick_bottom={rm['tick_bottom']}  "
            f"axes_w={rm['axes_w']}"
        )
        row_margins.append(rm)
    return row_margins


# ── image loading ─────────────────────────────────────────────────────────────

def load_images(indir: Path, ref: str) -> list[list[np.ndarray]]:
    imgs = []
    for row_idx, tmpl in enumerate(FILE_TEMPLATES):
        row = []
        for size in SIZES:
            fname = tmpl.format(ref=ref, size=size)
            fpath = indir / fname
            if not fpath.exists():
                sys.exit(f"ERROR: file not found: {fpath}")
            img = np.asarray(Image.open(fpath).convert("RGB"))
            if row_idx in CROP_TOP_ROWS:
                crop_px = int(img.shape[0] * CROP_TOP_FRAC)
                img     = img[crop_px:, :, :]
            row.append(img)
        imgs.append(row)
    return imgs


# ── figure assembly ───────────────────────────────────────────────────────────

def build_figure(imgs: list[list[np.ndarray]]) -> plt.Figure:
    row_margins = get_row_margins(imgs)

    # Global layout measures
    target_axes_w_px    = max(rm["axes_w"] for rm in row_margins)
    max_tick_label_w_px = max(rm["left_spine"] - rm["y_label_end"] for rm in row_margins)

    def px(n: float) -> float:
        return n / FIG_DPI

    axes_w_in           = px(target_axes_w_px)
    max_tick_label_w_in = px(max_tick_label_w_px)

    # Column spacing: axes area  +  tick-label band  +  small inner gap
    col_spacing = axes_w_in + max_tick_label_w_in + H_GAP_INNER

    # col_x_in[j] = figure-left distance to the LEFT SPINE of column j
    col_x_in = [LEFT_MARGIN + j * col_spacing for j in range(NCOLS)]

    # Row panel heights (axes area + tick region below, no axis title)
    row_panel_h_in: list[float] = []
    for i, rm in enumerate(row_margins):
        row_panel_h_in.append(px(rm["bottom_spine"] + rm["tick_bottom"]))

    # Figure dimensions
    fig_w = col_x_in[-1] + axes_w_in + RIGHT_MARGIN
    fig_h = (TOP_MARGIN
             + sum(row_panel_h_in)
             + (NROWS - 1) * V_GAP
             + BOTTOM_LABEL_H
             + BOTTOM_MARGIN)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=FIG_DPI)

    # row_tops_in[i] = inches from figure TOP to top of panel i
    row_tops_in: list[float] = []
    y_cursor = TOP_MARGIN
    for h_in in row_panel_h_in:
        row_tops_in.append(y_cursor)
        y_cursor += h_in + V_GAP

    # ── column labels (top of figure) ────────────────────────────────────────
    for j, label in enumerate(COL_LABELS):
        x_mid = (col_x_in[j] + axes_w_in / 2) / fig_w
        y_pos = 1.0 - (TOP_MARGIN * 0.38) / fig_h
        fig.text(x_mid, y_pos, label,
                 ha="center", va="center",
                 fontsize=LABEL_FONTSIZE, fontweight="bold")

    # ── row labels (left of figure) ───────────────────────────────────────────
    for i, label in enumerate(ROW_LABELS):
        y_mid = 1.0 - (row_tops_in[i] + row_panel_h_in[i] / 2) / fig_h
        x_pos = (LEFT_MARGIN * 0.38) / fig_w
        fig.text(x_pos, y_mid, label,
                 ha="center", va="center",
                 fontsize=LABEL_FONTSIZE, fontweight="bold",
                 rotation=90)

    # ── shared x-axis title (below bottom row) ───────────────────────────────
    bottom_edge_in = row_tops_in[-1] + row_panel_h_in[-1]
    x_center = (col_x_in[0] + col_x_in[-1] + axes_w_in) / 2 / fig_w
    y_xlabel  = 1.0 - (bottom_edge_in + BOTTOM_LABEL_H * 0.48) / fig_h
    fig.text(x_center, y_xlabel, X_AXIS_LABEL,
             ha="center", va="center",
             fontsize=LABEL_FONTSIZE, fontweight="bold")

    # ── panels ────────────────────────────────────────────────────────────────
    for i in range(NROWS):
        rm = row_margins[i]

        for j in range(NCOLS):
            img = imgs[i][j]
            full_w = img.shape[1]

            # ---- crop rules ----
            # X: left column keeps y-axis title (x0=0); others strip it, keep tick labels.
            x0 = 0 if j == 0 else rm["y_label_end"]
            # Right: include right spine pixel, discard any right margin whitespace.
            x1 = rm["right_spine"] + 1

            # Y: all rows strip the axis title from the image bottom.
            #    We keep tick marks + tick numbers (tick_bottom rows below spine).
            y0 = 0
            y1 = rm["bottom_spine"] + rm["tick_bottom"]

            cropped    = img[y0:y1, x0:x1]
            panel_h_in = px(cropped.shape[0])
            panel_w_in = px(cropped.shape[1])

            # The left spine of this panel is at col_x_in[j].
            # spine_offset is how many pixels from x0 the spine sits.
            spine_offset = rm["left_spine"] - x0
            panel_left_in = col_x_in[j] - px(spine_offset)
            panel_top_in  = row_tops_in[i]

            # Convert to matplotlib figure coords (origin = bottom-left)
            left_fig   = panel_left_in / fig_w
            bottom_fig = 1.0 - (panel_top_in + panel_h_in) / fig_h
            width_fig  = panel_w_in / fig_w
            height_fig = panel_h_in / fig_h

            ax = fig.add_axes([left_fig, bottom_fig, width_fig, height_fig])
            ax.imshow(cropped, interpolation="none", aspect="auto")
            ax.axis("off")

    return fig


# ── saving ────────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    pdf_path = outdir / f"{stem}.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PDF : {pdf_path.resolve()}")

    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PNG : {png_path.resolve()}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble fixed-bases supplementary figure"
    )
    parser.add_argument("--ref",    required=True,
                        help="Reference name: UHGG or lev")
    parser.add_argument("--indir",  default=None,
                        help="Directory with PNG files (default: results/{ref})")
    parser.add_argument("--outdir", default=".",
                        help="Output directory")
    args = parser.parse_args()

    ref    = args.ref
    indir  = Path(args.indir) if args.indir else Path("results") / ref
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reference : {ref}")
    print(f"Input dir : {indir.resolve()}")

    imgs = load_images(indir, ref)
    fig  = build_figure(imgs)
    save_figure(fig, outdir, f"supp_figure_fixed_bases_{ref}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
