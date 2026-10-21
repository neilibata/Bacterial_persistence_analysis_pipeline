#!/usr/bin/env python3
"""Show that the deposited workflow reproduces the original preprocessing.

The cohort was processed in manually orchestrated stages over an extended
period; re-executing all ~18,000 samples through the deposited Snakemake
workflow in one pass costs about a month of cluster time. That expense buys
very little: what needs demonstrating is that the reimplementation produces the
same numbers as the original execution, and equivalence is measurable on a
subset.

    # 1. choose a stratified subset already present in the published matrix
    python tools/concordance_check.py select --ref UHGG --n 30

    # 2. reprocess resources/concordance_UHGG.tsv through the workflow, then
    python tools/concordance_check.py compare --ref UHGG \\
        --new /path/to/rerun/results/matrices/depth_matrix_YYYYMMDD.csv

Selection is seeded, so the subset is reproducible from the deposited code
rather than something a reader has to take on trust.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402

OUT = ROOT / "results" / "concordance"
CLASSIFIED = ("H-G", "Transitional", "Industrial")
RUN = re.compile(r"[SEDR]RR\d+")
SEED = 20260811


def matrix_index(path: Path) -> set[str]:
    ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[0], chunksize=200_000, dtype=str):
        ids.update(chunk.iloc[:, 0].astype(str).str.strip())
    ids.discard("")
    return ids


def cmd_select(args) -> int:
    m1 = pd.read_csv(ROOT / "metadata_M1.csv", low_memory=False, dtype=str)
    m1 = m1.drop_duplicates("Chosen_match_ID").copy()
    m1["_key"] = m1["Chosen_match_ID"].astype(str).str.strip()
    m1["_runs"] = m1["Paired SRR List"].apply(
        lambda v: [] if pd.isna(v) else list(dict.fromkeys(RUN.findall(str(v)))))
    lab = m1["Lifestyle"].astype(str).str.strip()
    cls = m1[lab.isin(CLASSIFIED)]

    bf = ROOT / config.REFERENCES[args.ref]["breadth_file"]
    if not bf.exists():
        sys.exit(f"{bf} not found")
    print(f"  scanning {bf.name} ...", flush=True)
    present = matrix_index(bf)
    pool = cls[cls["_key"].isin(present) & (cls["_runs"].str.len() > 0)].copy()
    print(f"  {len(pool):,} classified samples present in the {args.ref} matrix")

    # Stratify by lifestyle and by sequencing-depth tercile: a subset drawn only
    # from deep industrial samples would not exercise the regime where
    # detection is marginal, which is exactly where any discrepancy matters.
    tb = pd.to_numeric(pool["Total bases"], errors="coerce")
    pool["_depth_bin"] = pd.qcut(tb.rank(method="first"), 3,
                                 labels=["low", "mid", "high"])
    rng = np.random.default_rng(SEED)
    picks = []
    strata = list(pool.groupby(["Lifestyle", "_depth_bin"], observed=True))
    per = max(1, args.n // max(1, len(strata)))
    for (life, dbin), grp in strata:
        take = min(per, len(grp))
        idx = rng.choice(len(grp), size=take, replace=False)
        sel = grp.iloc[sorted(idx)]
        picks.append(sel)
        print(f"    {str(life):<13} {str(dbin):<5} {take:>3} of {len(grp):>6}")
    sel = pd.concat(picks).drop_duplicates("_key").sort_values("_key")

    OUT.mkdir(parents=True, exist_ok=True)
    res = ROOT / "resources"
    res.mkdir(exist_ok=True)
    tsv = res / f"concordance_{args.ref}.tsv"
    with open(tsv, "w") as fh:
        fh.write("sample_id\truns\n")
        for _, r in sel.iterrows():
            fh.write(f"{r['_key']}\t{';'.join(r['_runs'])}\n")
    n_runs = sum(len(r) for r in sel["_runs"])
    gbp = pd.to_numeric(sel["Total bases"], errors="coerce").sum() / 1e9
    print(f"\n  selected {len(sel)} samples, {n_runs} runs, {gbp:,.0f} Gbp")
    print(f"  wrote {tsv}")
    print(f"\n  reprocess with:\n    snakemake --profile profiles/slurm \\\n"
          f"      --config samples_tsv=resources/{tsv.name} "
          f"matrix_version=concordance")
    return 0


def cmd_compare(args) -> int:
    ref = args.ref
    old_p = ROOT / config.REFERENCES[ref]["depth_file"]
    new_p = Path(args.new)
    for p in (old_p, new_p):
        if not p.exists():
            sys.exit(f"{p} not found")

    new = pd.read_csv(new_p, index_col=0)
    new.index = new.index.astype(str).str.strip()
    want = set(new.index)
    print(f"  rerun matrix: {len(new):,} samples x {new.shape[1]:,} columns")

    keep = []
    for chunk in pd.read_csv(old_p, index_col=0, chunksize=2000):
        chunk.index = chunk.index.astype(str).str.strip()
        hit = chunk[chunk.index.isin(want)]
        if len(hit):
            keep.append(hit)
    if not keep:
        sys.exit("no overlapping samples between the two matrices")
    old = pd.concat(keep)
    print(f"  overlap     : {len(old):,} samples")

    cols = [c for c in new.columns if c in old.columns]
    print(f"  shared cols : {len(cols):,}")
    old, new = old.loc[old.index, cols], new.loc[old.index, cols]

    rows = []
    for s in old.index:
        a = pd.to_numeric(old.loc[s], errors="coerce").fillna(0.0).to_numpy()
        b = pd.to_numeric(new.loc[s], errors="coerce").fillna(0.0).to_numpy()
        both = (a > 0) | (b > 0)
        r = np.corrcoef(a[both], b[both])[0, 1] if both.sum() > 2 else np.nan
        denom = np.maximum(np.abs(a), np.abs(b))
        rel = np.abs(a - b) / np.where(denom > 0, denom, 1.0)
        rows.append(dict(
            sample=s, n_nonzero_old=int((a > 0).sum()),
            n_nonzero_new=int((b > 0).sum()),
            pearson_r=r, median_rel_diff=float(np.median(rel[both])) if both.sum() else 0.0,
            max_rel_diff=float(rel[both].max()) if both.sum() else 0.0,
            total_bp_old=float(a.sum()), total_bp_new=float(b.sum())))
    df = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"concordance_{ref}.csv"
    df.to_csv(out, index=False)

    print(f"\n  per-sample Pearson r : median {df['pearson_r'].median():.6f}  "
          f"min {df['pearson_r'].min():.6f}")
    print(f"  median rel. diff     : {df['median_rel_diff'].median():.2e}")
    print(f"  max rel. diff        : {df['max_rel_diff'].max():.2e}")
    print(f"  detected-SGB delta   : "
          f"{(df['n_nonzero_new'] - df['n_nonzero_old']).abs().max()} (max abs)")
    print(f"\n  wrote {out}")
    worst = df.nsmallest(3, "pearson_r")[["sample", "pearson_r", "max_rel_diff"]]
    if len(worst):
        print("\n  least concordant:")
        print(worst.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="choose a stratified reprocessing subset")
    s.add_argument("--ref", default="UHGG", choices=list(config.REFERENCES))
    s.add_argument("--n", type=int, default=30)
    s.set_defaults(fn=cmd_select)
    c = sub.add_parser("compare", help="compare a rerun matrix to the published one")
    c.add_argument("--ref", default="UHGG", choices=list(config.REFERENCES))
    c.add_argument("--new", required=True, help="depth matrix from the rerun")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
