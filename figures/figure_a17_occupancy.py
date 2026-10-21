#!/usr/bin/env python3
"""Figure A17 — occupancy / detect-in-at-least-k-samples accumulation.

Thin wrapper that runs `persistence.occupancy_accumulation_sim` so the analysis is
reachable from the pipeline's figure runner:

    python -m persistence --only figures --figure occupancy_a17
    python figures/figure_a17_occupancy.py --refs combined          # standalone

Writes to `results/occupancy_accumulation/` by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importable
from persistence import occupancy_accumulation_sim as occ  # noqa: E402


def _meta_default(root: Path) -> str:
    """Metadata file, in config's preference order (not a local hard-coding)."""
    from persistence import config
    for cand in config.META_FILE_CANDIDATES:
        if (root / cand).exists():
            return cand
    return config.META_FILE_CANDIDATES[0]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", nargs="+", default=["combined"],
                    help="reference catalogue(s); the manuscript A17 uses 'combined'")
    ap.add_argument("--metadata", default=_meta_default(root))
    ap.add_argument("--out-dir", default=str(root / "results" / "occupancy_accumulation"))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-fit-boot", type=int, default=200)
    args = ap.parse_args()

    # occ.main() reads argv; hand it the resolved options.
    sys.argv = ["occupancy_accumulation_sim",
                "--metadata", args.metadata,
                "--refs", *args.refs,
                "--output-dir", args.out_dir,
                "--n-perm", str(args.n_perm),
                "--n-fit-boot", str(args.n_fit_boot)]
    occ.main()


if __name__ == "__main__":
    main()
