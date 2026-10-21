#!/usr/bin/env python3
"""Chimp-catalogue negative control for the accumulation argument.

The manuscript's claim is that non-detection is not absence: pool enough human
samples and the accumulation curve converges on the *whole* reference catalogue.
That argument is only meaningful if the same machinery can also say "absent"
when a taxon really is absent.

This applies the **identical Figure 3 fit** (`accumulation._fit_figure3_curve`,
4-parameter Richards in log-N space, weighted by the bootstrap spread and
bounded above by the catalogue size) to a chimpanzee-derived SGB catalogue
mapped against the same human metagenomes. If the pipeline detected taxa
indiscriminately, the chimp catalogue would saturate like a human one.

    python tools/chimp_negative_control.py
    python tools/chimp_negative_control.py --thresholds 10pct 30pct 50pct

Writes results/chimp/negative_control_asymptotes.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence.accumulation import _fit_figure3_curve  # noqa: E402

# reference -> catalogue size (SGBs), matching all_SGB_lengths_<ref>.txt
CATALOGUE = {"UHGG": 3971, "lev": 3594, "combined": 5081, "chimp": 1682}
SERIES = ["Industrial", "Non-Industrial"]


def fit_one(ref: str, threshold: str, series: str, unit: str = "samples"):
    path = ROOT / "results" / ref / "Figure3" / f"figure3_accumulation_{threshold}.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[(d["unit"] == unit) & (d["series"] == series)].sort_values("N")
    if len(d) < 5:
        return None

    n_mic = CATALOGUE[ref]
    scale, hill_p, hill_r2, rich_p, rich_r2 = _fit_figure3_curve(
        d["N"].to_numpy(float), d["sgb"].to_numpy(float),
        d["q16"].to_numpy(float), d["q84"].to_numpy(float), n_mic)
    if rich_p is None:
        return None

    return {
        "reference": ref, "catalogue_sgbs": n_mic, "threshold": threshold,
        "series": series, "unit": unit,
        "n_points": int(len(d)), "N_max": int(d["N"].max()),
        "observed_max_sgbs": float(d["sgb"].max()),
        "observed_pct": 100.0 * float(d["sgb"].max()) / n_mic,
        "richards_asymptote": float(rich_p[1]),
        "richards_pct_of_catalogue": 100.0 * float(rich_p[1]) / n_mic,
        "richards_R2": float(rich_r2),
        "hill_asymptote": float(hill_p[0]) if hill_p is not None else np.nan,
        "hill_pct_of_catalogue": (100.0 * float(hill_p[0]) / n_mic
                                  if hill_p is not None else np.nan),
        "hill_R2": float(hill_r2) if hill_r2 is not None else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", nargs="+", default=["10pct", "30pct", "50pct"])
    ap.add_argument("--refs", nargs="+", default=["UHGG", "lev", "combined", "chimp"])
    args = ap.parse_args()

    rows = [r for ref in args.refs for thr in args.thresholds for s in SERIES
            if (r := fit_one(ref, thr, s)) is not None]
    if not rows:
        sys.exit("no accumulation curves found — run the pipeline first")
    out = pd.DataFrame(rows)

    dest = ROOT / "results" / "chimp" / "negative_control_asymptotes.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    for thr in args.thresholds:
        sub = out[out.threshold == thr]
        if not len(sub):
            continue
        print(f"\n=== {thr} detection threshold — Richards asymptote (Figure 3 fit) ===")
        print(f"{'reference':<10} {'catalogue':>9} {'series':<16} {'N_max':>6} "
              f"{'observed':>9} {'asymptote':>10} {'% of cat':>9} {'R2':>7}")
        for _, r in sub.iterrows():
            print(f"{r.reference:<10} {r.catalogue_sgbs:>9,} {r.series:<16} {r.N_max:>6,} "
                  f"{r.observed_max_sgbs:>9,.0f} {r.richards_asymptote:>10,.1f} "
                  f"{r.richards_pct_of_catalogue:>8.1f}% {r.richards_R2:>7.4f}")
    print(f"\nWrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
