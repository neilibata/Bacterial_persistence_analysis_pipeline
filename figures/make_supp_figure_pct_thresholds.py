"""
make_supp_figure_pct_thresholds.py
------------------------------------
Assembles 24 subplots (4 rows × 6 columns) varying coverage threshold.

Rows    : Pooled accumulation | Pooled params vs N |
          Continent accumulation | Continent params vs N
Columns : 1% | 3% | 10% | 30% | 50% | 90%

Outputs : supp_figure_pct_thresholds_{ref}.pdf
          supp_figure_pct_thresholds_{ref}.png

Usage
-----
    python make_supp_figure_pct_thresholds.py --ref UHGG
    python make_supp_figure_pct_thresholds.py --ref lev
    python make_supp_figure_pct_thresholds.py --ref UHGG --indir results/UHGG --outdir figures/

Requirements
------------
    pip install matplotlib pillow

File naming conventions expected in --indir
-------------------------------------------
Row 0  accumulation_{pct}pct_{ref}_pooled_notitle.png
Row 1  accum_pooled_params_vsN_cov{cov}_notitle.png
Row 2  accumulation_{pct}pct_{ref}_continent_notitle.png
Row 3  accum_continent_params_vsN_cov{cov}_notitle.png

where {pct} ∈ {1, 3, 10, 30, 50, 90}
and   {cov} ∈ {1p0, 3p0, 10p0, 30p0, 50p0, 90p0}
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

# ── layout ────────────────────────────────────────────────────────────────────

# Each entry: (pct_label, pct_str used in accumulation filenames, cov_str used in params filenames)
THRESHOLDS = [
    ("1%",  "1",  "1p0"),
    ("3%",  "3",  "3p0"),
    ("10%", "10", "10p0"),
    ("30%", "30", "30p0"),
    ("50%", "50", "50p0"),
    ("90%", "90", "90p0"),
]

COL_LABELS = [t[0] for t in THRESHOLDS]   # "1%", "3%", …

ROW_LABELS = [
    "Pooled accumulation",
    "Pooled params vs N",
    "Continent accumulation",
    "Continent params vs N",
]

# Templates per row.  Available keys: {ref}, {pct}, {cov}
# Rows whose filenames don't contain a reference name still accept {ref} safely
# because we always supply it (it simply won't appear in the string).
FILE_TEMPLATES = [
    "accumulation_{pct}pct_{ref}_pooled_notitle.png",
    "accum_pooled_params_vsN_cov{cov}_notitle.png",
    "accumulation_{pct}pct_{ref}_continent_notitle.png",
    "accum_continent_params_vsN_cov{cov}_notitle.png",
]

NROWS = len(FILE_TEMPLATES)
NCOLS = len(THRESHOLDS)

FIG_DPI        = 150
LABEL_FONTSIZE = 36
LEFT_MARGIN    = 2.20
TOP_MARGIN     = 1.10
RIGHT_MARGIN   = 0.15
BOTTOM_MARGIN  = 0.15
H_GAP          = 0.08
V_GAP          = 0.08


# ── helpers ───────────────────────────────────────────────────────────────────

def load_images(indir: Path, ref: str) -> list[list[np.ndarray]]:
    """imgs[row][col]  row=plot-type (0-3), col=threshold (0-5)"""
    imgs = []
    for row_idx, tmpl in enumerate(FILE_TEMPLATES):
        row = []
        for _, pct, cov in THRESHOLDS:
            fname = tmpl.format(ref=ref, pct=pct, cov=cov)
            fpath = indir / fname
            if not fpath.exists():
                sys.exit(f"ERROR: file not found: {fpath}")
            row.append(np.asarray(Image.open(fpath).convert("RGB")))
        imgs.append(row)
    return imgs


def build_figure(imgs: list[list[np.ndarray]]) -> plt.Figure:
    ph, pw = imgs[0][0].shape[:2]

    panel_w_in = pw / FIG_DPI
    panel_h_in = ph / FIG_DPI

    fig_w = LEFT_MARGIN + NCOLS * panel_w_in + (NCOLS - 1) * H_GAP + RIGHT_MARGIN
    fig_h = TOP_MARGIN  + NROWS * panel_h_in + (NROWS - 1) * V_GAP + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=FIG_DPI)

    # column labels
    for j, label in enumerate(COL_LABELS):
        x_left = (LEFT_MARGIN + j * (panel_w_in + H_GAP)) / fig_w
        x_mid  = x_left + (panel_w_in / 2) / fig_w
        y_top  = 1.0 - (TOP_MARGIN * 0.35) / fig_h
        fig.text(x_mid, y_top, label,
                 ha="center", va="center",
                 fontsize=LABEL_FONTSIZE, fontweight="bold")

    # row labels
    for i, label in enumerate(ROW_LABELS):
        y_top = 1.0 - (TOP_MARGIN + i * (panel_h_in + V_GAP)) / fig_h
        y_mid = y_top - (panel_h_in / 2) / fig_h
        x_mid = (LEFT_MARGIN * 0.45) / fig_w
        fig.text(x_mid, y_mid, label,
                 ha="center", va="center",
                 fontsize=LABEL_FONTSIZE, fontweight="bold",
                 rotation=90)

    # panels
    for i in range(NROWS):
        for j in range(NCOLS):
            left   = (LEFT_MARGIN + j * (panel_w_in + H_GAP)) / fig_w
            bottom = 1.0 - (TOP_MARGIN + (i + 1) * panel_h_in + i * V_GAP) / fig_h
            ax = fig.add_axes([left, bottom, panel_w_in / fig_w, panel_h_in / fig_h])
            ax.imshow(imgs[i][j], interpolation="none", aspect="auto")
            ax.axis("off")

    return fig


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    pdf_path = outdir / f"{stem}.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PDF : {pdf_path.resolve()}")

    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PNG : {png_path.resolve()}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Assemble pct-threshold supplementary figure")
    parser.add_argument("--ref",    required=True, help="Reference name: UHGG or lev")
    parser.add_argument("--indir",  default=None,  help="Directory with PNG files (default: results/{ref})")
    parser.add_argument("--outdir", default=".",   help="Output directory")
    args = parser.parse_args()

    ref    = args.ref
    indir  = Path(args.indir) if args.indir else Path("results") / ref
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reference : {ref}")
    print(f"Input dir : {indir.resolve()}")

    imgs = load_images(indir, ref)
    fig  = build_figure(imgs)
    save_figure(fig, outdir, f"supp_figure_pct_thresholds_{ref}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
