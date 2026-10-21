#!/usr/bin/env python3
"""Standalone driver for Supplementary S5 outputs (Tables S4/S5, Figures S5A/S5B).

The pipeline only wires `supp_s5.run_supp_s5` behind the (optional) Bayesian
section, so a standard run never emits Figure S5B (the VANISH/BloSSUM list-overlap
analysis = Additional Figure A21). This driver loads a reference and the four list
files already written by section 2, then regenerates the S5 outputs — in particular
`fig_s5b_overlap_analysis_{medium,large}{,_notitle}.png`.

    python figures/make_supp_s5_figure.py --ref combined
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import os as _os
import sys as _sys
# Moved out of the repository root on 2026-10-20. Run as
# `python figures/make_supp_s5_figure.py`, which puts figures/ on sys.path rather than
# the root, so the package has to be found explicitly -- the same
# bootstrap figure2.py, figure5.py and the rest of this directory use.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from persistence import config
# Reinstatement is part of cohort construction: load_reference alone returns
# the pre-reinstatement cohort, which is not what the paper's numbers use.
from persistence.loading import load_reference_analytic as load_reference
from persistence.supp_s5 import run_supp_s5


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing list file: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="combined")
    args = ap.parse_args()

    data = load_reference(args.ref)
    s2_main = Path(config.section_dir(args.ref, "s2", "main"))
    vanish_plus = _read_ids(s2_main / "vanish_plus_ids.txt")
    blossum_plus = _read_ids(s2_main / "blossum_plus_ids.txt")
    vanish_h = _read_ids(s2_main / "vanishH_ids.txt")
    blossum_h = _read_ids(s2_main / "blossumH_ids.txt")
    print(f"  [supp_s5 driver] ref={args.ref}  "
          f"V+={len(vanish_plus)} B+={len(blossum_plus)} "
          f"VH+={len(vanish_h)} BH+={len(blossum_h)}")

    run_supp_s5(data, vanish_plus, blossum_plus, vanish_h, blossum_h)
    print("  [supp_s5 driver] done")


if __name__ == "__main__":
    main()
