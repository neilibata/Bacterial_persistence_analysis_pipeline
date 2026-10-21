#!/usr/bin/env python3
"""Per-sample breadth profile from the coverage matrix, to test what a sample IS.

ENA says three CarterM samples are Bifidobacterium isolates and ninety-four
IjazUZ samples are "bacillota bacterium". That is the archive's label; this
checks the data. A shotgun gut metagenome spreads reads across hundreds of SGBs;
a single-organism run concentrates them on one, so the two are separable without
trusting either label.

For every sample in the matrix, computes from the breadth (%-genome-covered)
columns:

  richness_10pct   SGBs at >= 10% breadth        metagenome: hundreds
  richness_50pct   SGBs at >= 50% breadth
  max_breadth      highest single-SGB breadth    isolate: near 100
  top1_share       top SGB's breadth / total     isolate: dominant
  top5_share       top 5 SGBs' share

Writes one row per sample. Interpretation is left to the caller.

    python tools/profile_samples_from_matrix.py --matrix combined_coverage_matrix_20260820.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "combined_coverage_matrix_20260820.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "sample_breadth_profile.csv")
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args()

    header = pd.read_csv(args.matrix, nrows=0)
    idcol = header.columns[0]
    mean_cols = [c for c in header.columns if c.endswith("_mean")]
    print(f"{args.matrix.name}: {len(header.columns):,} columns, {len(mean_cols):,} breadth columns")

    rows = []
    seen = 0
    for chunk in pd.read_csv(args.matrix, usecols=[idcol] + mean_cols,
                             chunksize=args.chunk, low_memory=False):
        ids = chunk[idcol].astype(str).values
        v = chunk[mean_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        tot = v.sum(axis=1)
        part = np.sort(v, axis=1)[:, ::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            top1 = np.where(tot > 0, part[:, 0] / tot, 0.0)
            top5 = np.where(tot > 0, part[:, :5].sum(axis=1) / tot, 0.0)
        for i, sid in enumerate(ids):
            rows.append(dict(Sample_ID=sid,
                             richness_10pct=int((v[i] >= 10).sum()),
                             richness_50pct=int((v[i] >= 50).sum()),
                             max_breadth=float(part[i, 0]),
                             top1_share=float(top1[i]),
                             top5_share=float(top5[i]),
                             total_breadth=float(tot[i])))
        seen += len(chunk)
        print(f"  {seen:,} samples", end="\r")
    print()
    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"  wrote {args.out.relative_to(ROOT)} ({len(d):,} samples)")
    print("\n  cohort distribution of richness at 10% breadth:")
    print(d.richness_10pct.describe(percentiles=[.01, .05, .25, .5, .75]).to_string())


if __name__ == "__main__":
    main()
