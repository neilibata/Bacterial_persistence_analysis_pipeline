#!/usr/bin/env python3
"""Regenerate main-text Figure 1.

Figure 1 is the Equal Earth global sample distribution map:
lifestyle is encoded by hue, country sample count by log-shading, and total
sequenced bases by circle area.

Default output:
    main_text_figures/figure1.png
    main_text_figures/figure1.pdf
"""

from __future__ import annotations

import argparse
import os

from make_metadata_equal_earth import plot_world_map_B_eqearth
from make_metadata_summary import (load_and_preprocess,
                                   COHORT_REF as _COHORT_REF)


def _default_cohort_ref() -> str:
    return _COHORT_REF


def _default_meta() -> str:
    """Metadata file from config, so a repointed cohort reaches this script.

    This was hard-coded to metadata_Mar_16_2026.csv, which silently kept the
    pre-2026-08-20 cohort while the analysis ran on the newer metadata.
    """
    import persistence.config as _c
    return _c.META_FILE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--meta",
        default=_default_meta(),
        help="Input metadata CSV (default: config.META_FILE)",
    )
    ap.add_argument(
        "--secondary",
        default=None,
        help="Optional secondary-run BC screening CSV",
    )
    ap.add_argument(
        "--out-dir",
        default="main_text_figures",
        help="Output directory (default: main_text_figures/)",
    )
    ap.add_argument(
        "--cohort-ref",
        default=_default_cohort_ref(),
        help=("Reference whose analytic cohort defines the map (default: the "
              "main-text reference). Reads results/<ref>/cohort_samples.csv so "
              "the reported Ns match the analyses."),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta = load_and_preprocess(args.meta, args.secondary, args.cohort_ref)
    plot_world_map_B_eqearth(meta, os.path.join(args.out_dir, "figure1"),
                             show_title=False)


if __name__ == "__main__":
    main()
