#!/usr/bin/env python3
"""Flag samples whose breadth and depth are physically inconsistent.

Random shotgun sequencing links the two: at genome-wide depth d the expected
breadth is 1 - e^-d, so at the low depths typical here breadth ~= depth and the
depth over the covered bases tends to 1x. Reads that pile onto one narrow window
break that -- the covered fraction stays tiny while the depth on it climbs.

That is what 16S amplicon data looks like when it is mapped against whole
genomes, and it is how the MehtaRS 2018 libraries registered as WGS were caught:
379 bp covered per reference at up to 45,262x on that window, against 1-2x for
genuine shotgun.

Per sample, over every SGB with non-zero coverage:

  dpcb = mean_depth / (mean_coverage/100)     depth per covered base

  dpcb_p99 / dpcb_max     the pileup signature; ~1-2 for shotgun
  median_breadth_pct      how much of each reference is seen at all
  n_sgb_any               how many references are touched
  frac_sgb_dpcb_gt10      share of touched references showing >10x pileup

Reports; excludes nothing.

    python tools/scan_breadth_depth_consistency.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _mean_cols(path):
    h = pd.read_csv(path, nrows=0)
    return h.columns[0], [c for c in h.columns if c.endswith("_mean")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", type=Path,
                    default=ROOT / "combined_coverage_matrix_20260820.csv")
    ap.add_argument("--depth", type=Path,
                    default=ROOT / "combined_depth_matrix_20260820.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "breadth_depth_consistency.csv")
    ap.add_argument("--chunk", type=int, default=300)
    args = ap.parse_args()

    cid, ccols = _mean_cols(args.coverage)
    did, dcols = _mean_cols(args.depth)
    if ccols != dcols:
        raise SystemExit("ERROR: coverage and depth matrices have different species columns")
    print(f"{len(ccols):,} species columns; streaming both matrices in step", flush=True)

    ci = pd.read_csv(args.coverage, usecols=[cid] + ccols, chunksize=args.chunk, low_memory=False)
    di = pd.read_csv(args.depth, usecols=[did] + dcols, chunksize=args.chunk, low_memory=False)

    rows, seen = [], 0
    for cch, dch in zip(ci, di):
        cids = cch[cid].astype(str).str.strip().values
        dids = dch[did].astype(str).str.strip().values
        if not np.array_equal(cids, dids):
            # Row order is not guaranteed to agree; align explicitly rather than
            # silently pairing a sample's breadth with another's depth.
            dch = dch.set_index(dch[did].astype(str).str.strip()).reindex(cids).reset_index(drop=True)
        C = cch[ccols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        D = dch[dcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        for i, sid in enumerate(cids):
            c, d = C[i], D[i]
            hit = c > 0
            n = int(hit.sum())
            if n == 0:
                rows.append(dict(Sample_ID=sid, n_sgb_any=0))
                continue
            frac = c[hit] / 100.0
            dpcb = d[hit] / np.maximum(frac, 1e-12)
            rows.append(dict(
                Sample_ID=sid, n_sgb_any=n,
                median_breadth_pct=float(np.median(c[hit])),
                mean_depth_total=float(d.sum()),
                dpcb_median=float(np.median(dpcb)),
                dpcb_p99=float(np.percentile(dpcb, 99)),
                dpcb_max=float(dpcb.max()),
                frac_sgb_dpcb_gt10=float((dpcb > 10).mean()),
            ))
        seen += len(cids)
        print(f"  {seen:,}", end="\r", flush=True)
    print()

    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"wrote {args.out.relative_to(ROOT)} ({len(d):,} samples)\n")

    ok = d[d.n_sgb_any > 0]
    print("cohort distribution of depth-per-covered-base:")
    for c in ["dpcb_median", "dpcb_p99", "dpcb_max", "median_breadth_pct", "frac_sgb_dpcb_gt10"]:
        q = ok[c].describe(percentiles=[.5, .9, .99])
        print(f"  {c:22s} median {q['50%']:10.3f}  p90 {q['90%']:10.3f}  "
              f"p99 {q['99%']:12.3f}  max {q['max']:12.1f}")


if __name__ == "__main__":
    main()
