#!/usr/bin/env python3
"""Regenerate main-text Figure 4.

This is a manuscript-facing wrapper around section2_bayesian_figure.py using
the hard 10% combined-reference view matched to the submitted manuscript.

Default outputs:
    main_text_figures/figure4.png
    main_text_figures/figure4.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=Path("main_text_figures"), type=Path)
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--ra-module-path", default=None)
    args = ap.parse_args()

    # This script lives in figures/; sibling scripts are alongside it (script_dir),
    # while data/results/cwd are resolved from the repo root (parent of figures/).
    script_dir = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    build_dir = out_dir / "_figure4_build"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir or str(root / "results" / args.ref)
    results_dir = args.results_dir or str(root / "results" / args.ref)
    ra_module_path = args.ra_module_path or str(root / "persistence" / "rank_abundance_analysis.py")

    cmd = [
        sys.executable,
        str(script_dir / "section2_bayesian_figure.py"),
        "--ref",
        args.ref,
        "--hard-10pct",
        "--out-dir",
        str(build_dir),
        "--data-dir",
        data_dir,
        "--results-dir",
        results_dir,
        "--ra-module-path",
        ra_module_path,
    ]
    subprocess.run(cmd, cwd=root, check=True)

    stem = f"section2_hard10pct_{args.ref}"
    for ext in (".png", ".pdf"):
        source = build_dir / f"{stem}{ext}"
        if not source.exists():
            raise FileNotFoundError(f"Expected Figure 4 build output was not created: {source}")
        dest = out_dir / f"figure4{ext}"
        shutil.copy2(source, dest)
        print(f"Saved: {dest}")

    # File the reduction measurement, and the data object, under the name the
    # figure is published as rather than the build name.
    sys.path.insert(0, str(root))
    from persistence.figure_rescale import adopt_report_row
    adopt_report_row(build_dir, stem, out_dir, "figure4")

    src_obj = build_dir / "figure_data" / "Figure_F4_data.pkl"
    if src_obj.exists():
        dest_obj = out_dir / "figure_data" / "Figure_F4_data.pkl"
        dest_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_obj, dest_obj)
        print(f"Saved: {dest_obj}")


if __name__ == "__main__":
    main()
