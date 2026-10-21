"""Graphical abstract outline from the 10% Figure3 accumulation fits."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .accumulation import (
    _figure3_fit_grid,
    _figure3_model_y,
    _make_figure3_series,
    _study_col,
)
from .loading import load_reference
from .plot_utils import savefig_dual


def plot_graphical_abstract(data, model="richards", out_dir=None):
    """Save a clean 10% log-space fit-only graphical abstract outline."""
    ref = data.ref_name
    out_dir = out_dir or os.path.join(config.outdir(ref), "Figure3")
    os.makedirs(out_dir, exist_ok=True)

    meta = data.meta.copy()
    style = {
        "Industrial": (config.OI["sky"], "-"),
        "Non-Industrial": (config.OI["orange"], "-"),
    }
    masks = {
        "Industrial": meta["Lifestyle"] == "Industrial",
        "Non-Industrial": meta["Lifestyle"].isin(["H-G", "Transitional"]),
    }

    _, sample_series, _ = _make_figure3_series(
        data.breadth, meta, masks, style, _study_col(meta),
        config.ACCUM_MAIN_TEXT_BREADTH, np.random.default_rng(123))

    fig, ax = plt.subplots(figsize=(9, 6))
    for sd in sample_series:
        x_fit = _figure3_fit_grid(sd["N_vec"])
        y_fit = _figure3_model_y(model, x_fit, sd)
        if y_fit is None:
            continue
        ax.plot(x_fit, y_fit, color=sd["color"], lw=4.0, label=sd["name"])

    ax.set_xscale("log")
    ax.set_xlabel("Number of samples (log scale)")
    ax.set_ylabel("SGBs detected")
    ax.grid(False)
    ax.legend(fontsize=16, loc="upper left", frameon=True)
    fig.tight_layout()

    fpath = os.path.join(out_dir, f"graphical_abstract_10pct_{model}.png")
    savefig_dual(fig, fpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {fpath}")
    return fpath


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create graphical abstract outline from 10% Figure3 fits.")
    parser.add_argument("--ref", default="combined",
                        help="Reference database (default: combined)")
    parser.add_argument("--model", default="richards",
                        choices=["richards", "hill"],
                        help="Fit curve family to draw (default: richards)")
    parser.add_argument("--out-dir", default=None,
                        help="Optional output directory")
    args = parser.parse_args(argv)

    data = load_reference(args.ref)
    plot_graphical_abstract(data, model=args.model, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
