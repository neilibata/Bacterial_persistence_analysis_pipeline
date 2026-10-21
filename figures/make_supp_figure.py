"""
make_supp_figure.py
-------------------
Assembles 24 subplots from results/UHGG into a 4-row × 6-column supplementary figure.

Rows    : Pooled accumulation | Pooled params vs N |
          Continent accumulation | Continent params vs N
Columns : 10 kb | 50 kb | 100 kb | 200 kb | 500 kb | 1 Mb

Outputs : supp_figure.pdf  (lossless — PNGs embedded at native DPI)
          supp_figure.png  (composite at native panel resolution)

Usage
-----
    python make_supp_figure.py                        # looks in results/UHGG/
    python make_supp_figure.py --indir path/to/pngs  # custom directory
    python make_supp_figure.py --outdir figures/      # custom output dir

Requirements
------------
    pip install matplotlib pillow
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

# ── layout metadata ──────────────────────────────────────────────────────────

SIZES = ["10kb", "30kb", "100kb", "300kb", "500kb", "1Mb"]
COL_LABELS = ["10 kb", "50 kb", "100 kb", "200 kb", "500 kb", "1 Mb"]

ROW_LABELS = [
    "Pooled accumulation",
    "Pooled params vs N",
    "Continent accumulation",
    "Continent params vs N",
]

# Ordered by row (plot type), each entry is the filename template for that row
FILE_TEMPLATES = [
    "accumulation_50pct_UHGG_flat_{size}_pooled_notitle.png",
    "accum_flat_{size}_pooled_params_vsN_notitle.png",
    "accumulation_50pct_UHGG_flat_{size}_continent_notitle.png",
    "accum_flat_{size}_continent_params_vsN_notitle.png",
]

NROWS = len(FILE_TEMPLATES)   # 4 plot types
NCOLS = len(SIZES)             # 6 bin sizes


def load_images(indir: Path) -> list[list[np.ndarray]]:
    """Return imgs[row][col] as numpy arrays (RGB).
    row = plot type (0-3), col = bin size (0-5)."""
    imgs = []
    for tmpl in FILE_TEMPLATES:
        row = []
        for size in SIZES:
            fname = tmpl.format(size=size)
            fpath = indir / fname
            if not fpath.exists():
                sys.exit(f"ERROR: file not found: {fpath}")
            img = Image.open(fpath).convert("RGB")
            row.append(np.asarray(img))
        imgs.append(row)
    return imgs


def build_figure(imgs: list[list[np.ndarray]]) -> plt.Figure:
    """
    Build and return the assembled figure.

    Layout
    ------
    Extra column on the left  → row labels
    Extra row on the top      → column labels
    Then 6 × 4 image panels   → actual subplots

    Sizing strategy: each panel is rendered at its native pixel size so that
    matplotlib does NOT resample the image content. The figure DPI is set so
    that 1 figure-inch == 1 pixel at the source DPI.
    """

    # Assume all panels share the same pixel dimensions (use first panel)
    ph, pw = imgs[0][0].shape[:2]          # panel height / width in px

    # We'll work at 150 dpi for the figure canvas.  Scale panel to that.
    # Because we are using imshow with interpolation='none', the image pixels
    # map 1-to-1 onto the canvas pixels only when the axes size matches the
    # image size exactly — which we ensure below.
    FIG_DPI = 150

    LABEL_FONTSIZE = 36   # 4× the original 9 pt

    # Spacing / label sizes in inches
    LEFT_MARGIN   = 2.20   # wider to fit large rotated row labels
    TOP_MARGIN    = 1.10   # taller to fit large column labels
    RIGHT_MARGIN  = 0.15
    BOTTOM_MARGIN = 0.15
    H_GAP         = 0.08   # gap between columns (inches)
    V_GAP         = 0.08   # gap between rows    (inches)

    panel_w_in = pw / FIG_DPI
    panel_h_in = ph / FIG_DPI

    fig_w = LEFT_MARGIN + NCOLS * panel_w_in + (NCOLS - 1) * H_GAP + RIGHT_MARGIN
    fig_h = TOP_MARGIN  + NROWS * panel_h_in + (NROWS - 1) * V_GAP + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=FIG_DPI)

    # ── draw column labels ────────────────────────────────────────────────────
    for j, label in enumerate(COL_LABELS):
        # x-centre of column j in figure coordinates
        x_left = (LEFT_MARGIN + j * (panel_w_in + H_GAP)) / fig_w
        x_mid  = x_left + (panel_w_in / 2) / fig_w
        y_top  = 1.0 - (TOP_MARGIN * 0.35) / fig_h   # near top

        fig.text(
            x_mid, y_top, label,
            ha="center", va="center",
            fontsize=LABEL_FONTSIZE, fontweight="bold",
            transform=fig.transFigure,
        )

    # ── draw row labels ───────────────────────────────────────────────────────
    for i, label in enumerate(ROW_LABELS):
        y_top = 1.0 - (TOP_MARGIN + i * (panel_h_in + V_GAP)) / fig_h
        y_mid = y_top - (panel_h_in / 2) / fig_h
        x_mid = (LEFT_MARGIN * 0.45) / fig_w

        fig.text(
            x_mid, y_mid, label,
            ha="center", va="center",
            fontsize=LABEL_FONTSIZE, fontweight="bold",
            rotation=90,
            transform=fig.transFigure,
        )

    # ── place image panels ────────────────────────────────────────────────────
    for i in range(NROWS):
        for j in range(NCOLS):
            # Panel position in figure coordinates [left, bottom, width, height]
            left   = (LEFT_MARGIN + j * (panel_w_in + H_GAP)) / fig_w
            bottom = 1.0 - (TOP_MARGIN + (i + 1) * panel_h_in + i * V_GAP) / fig_h
            width  = panel_w_in / fig_w
            height = panel_h_in / fig_h

            ax = fig.add_axes([left, bottom, width, height])
            ax.imshow(imgs[i][j], interpolation="none", aspect="auto")
            ax.axis("off")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Assemble supplementary figure")
    parser.add_argument("--indir",  default="results/UHGG",   help="Directory containing the 24 PNG files")
    parser.add_argument("--outdir", default=".",               help="Output directory for PDF and PNG")
    args = parser.parse_args()

    indir  = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from: {indir.resolve()}")
    imgs = load_images(indir)

    print("Building figure …")
    fig = build_figure(imgs)

    # ── save PDF ──────────────────────────────────────────────────────────────
    pdf_path = outdir / "supp_figure.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PDF : {pdf_path.resolve()}")

    # ── save PNG ──────────────────────────────────────────────────────────────
    png_path = outdir / "supp_figure.png"
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PNG : {png_path.resolve()}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
