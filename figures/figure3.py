#!/usr/bin/env python3
"""Regenerate main-text Figure 3.

This is a small manuscript-facing wrapper around make_combined_figure3.py. The
historical generator keeps its full build products and cache in an internal
directory; this script copies the manuscript-scaled Richards PNG/PDF to:

    main_text_figures/figure3.png
    main_text_figures/figure3.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p") + "pct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=Path("main_text_figures"), type=Path)
    ap.add_argument("--threshold", default=10.0, type=float)
    ap.add_argument("--model", default="richards", choices=["richards", "hill", "rsac"])
    ap.add_argument(
        "--min-occurrences",
        default=1,
        type=int,
        help="Require each SGB to be detected in at least this many sampled samples/studies.",
    )
    ap.add_argument(
        "--match-existing-rng",
        action="store_true",
        help="Match the RNG state used by the historical multi-threshold accumulation run.",
    )
    args = ap.parse_args()

    # This script lives in figures/; sibling scripts are alongside it (script_dir),
    # while data/results/cwd are resolved from the repo root (parent of figures/).
    script_dir = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    build_dir = out_dir / "_figure3_build"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_dir / "make_combined_figure3.py"),
        "--root",
        str(root),
        "--out-dir",
        str(build_dir),
        "--threshold",
        str(args.threshold),
        "--min-occurrences",
        str(args.min_occurrences),
        "--models",
        args.model,
        "--data-slot",
        "F3",
    ]
    if args.match_existing_rng:
        cmd.append("--match-existing-rng")

    subprocess.run(cmd, cwd=root, check=True)

    tag = _threshold_tag(args.threshold)
    suffix = f"{tag}_min{args.min_occurrences}" if args.min_occurrences > 1 else tag
    source = build_dir / f"figure3_combined_regenerated_{suffix}_{args.model}.png"
    source_pdf = source.with_suffix(".pdf")
    for expected in (source, source_pdf):
        if not expected.exists():
            raise FileNotFoundError(f"Expected Figure 3 build output was not created: {expected}")

    dest = out_dir / "figure3.png"
    shutil.copy2(source, dest)
    print(f"Saved: {dest}")
    dest_pdf = out_dir / "figure3.pdf"
    shutil.copy2(source_pdf, dest_pdf)
    print(f"Saved: {dest_pdf}")

    # The data object is written beside the build output; move it next to the
    # figure it belongs to, so every main-text figure's pickle sits in one
    # place for tools/replot_main.py and for the submitted data file.
    from persistence.figure_rescale import adopt_report_row
    adopt_report_row(build_dir, source.stem, out_dir, "figure3")

    src_obj = build_dir / "figure_data" / "Figure_F3_data.pkl"
    if src_obj.exists():
        dest_obj = out_dir / "figure_data" / "Figure_F3_data.pkl"
        dest_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_obj, dest_obj)
        print(f"Saved: {dest_obj}")


if __name__ == "__main__":
    main()
