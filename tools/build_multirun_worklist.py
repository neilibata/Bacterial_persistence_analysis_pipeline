#!/usr/bin/env python3
"""Emit a sample->runs worklist for the preprocessing pipeline.

A biological sample is often sequenced across several SRA runs. The analysis
keys on `Chosen_match_ID`, one row per sample, so every run belonging to a
sample must be concatenated *before* alignment -- aligning each run separately
produces one matrix row per run, which the analysis then cannot use. Breadth is
a percentage of genome covered, so per-run breadths cannot be pooled after the
fact; the merge has to happen on the reads.

`Paired SRR List` is authoritative: it is a superset of `SRA_Accession` for all
18,000 classified samples (2,653 samples list runs that `SRA_Accession` omits),
and by construction contains only paired-end runs.

    python tools/build_multirun_worklist.py \\
        --want results/cohort_reconciliation/lev_liss_multirun_ids.txt \\
        --out  results/cohort_reconciliation/lev_multirun_samples.tsv

Writes a two-column TSV (`sample_id<TAB>run1;run2;...`) that the Snakemake
pipeline reads directly, plus a flat run list for download accounting.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = re.compile(r"[SEDR]RR\d+")
CLASSIFIED = ("H-G", "Transitional", "Industrial")


def runs_of(value) -> list[str]:
    """Run accessions in listing order, de-duplicated, order preserved."""
    if pd.isna(value):
        return []
    return list(dict.fromkeys(RUN.findall(str(value))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1", type=Path, default=ROOT / "metadata_M1.csv")
    ap.add_argument("--want", type=Path,
                    help="file of Chosen_match_IDs, one per line "
                         "(default: every multi-run classified sample)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multi-only", action="store_true",
                    help="emit only samples with more than one run")
    args = ap.parse_args()

    m1 = pd.read_csv(args.m1, low_memory=False, dtype=str)
    m1 = m1.drop_duplicates("Chosen_match_ID").copy()
    m1["_key"] = m1["Chosen_match_ID"].astype(str).str.strip()
    m1["_runs"] = m1["Paired SRR List"].apply(runs_of)
    m1["_sra"] = m1["SRA_Accession"].apply(runs_of)

    lab = m1["Lifestyle"].astype(str).str.strip()
    cls = m1[lab.isin(CLASSIFIED)]

    if args.want:
        want = [l.strip() for l in open(args.want) if l.strip()]
        sel = cls[cls["_key"].isin(want)]
        missing = sorted(set(want) - set(sel["_key"]))
        if missing:
            print(f"  WARNING: {len(missing)} requested IDs are not classified "
                  f"samples in M1 (e.g. {missing[:3]})", file=sys.stderr)
    else:
        sel = cls

    if args.multi_only:
        sel = sel[sel["_runs"].str.len() > 1]

    empty = sel[sel["_runs"].str.len() == 0]
    if len(empty):
        sys.exit(f"ERROR: {len(empty)} samples have no paired runs "
                 f"(e.g. {list(empty['_key'][:3])}) -- refusing to emit a "
                 f"worklist that would silently skip them")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("sample_id\truns\n")
        for _, r in sel.sort_values("_key").iterrows():
            fh.write(f"{r['_key']}\t{';'.join(r['_runs'])}\n")

    flat = sorted({x for rs in sel["_runs"] for x in rs})
    runs_p = args.out.with_suffix(".runs.txt")
    runs_p.write_text("\n".join(flat) + "\n")

    n_multi = int((sel["_runs"].str.len() > 1).sum())
    tb = pd.to_numeric(sel.get("Total bases"), errors="coerce").sum()
    print(f"  samples          : {len(sel):,}  ({n_multi:,} multi-run)")
    print(f"  runs to download : {len(flat):,}")
    print(f"  total bases      : {tb/1e9:,.0f} Gbp")
    print(f"  runs per sample  : min {sel['_runs'].str.len().min()}  "
          f"median {int(sel['_runs'].str.len().median())}  "
          f"max {sel['_runs'].str.len().max()}")
    extra = int((sel["_runs"].str.len() > sel["_sra"].str.len()).sum())
    if extra:
        print(f"  NOTE: {extra:,} samples have paired runs absent from "
              f"SRA_Accession -- using Paired SRR List is what recovers them")
    print(f"\n  wrote {args.out}")
    print(f"  wrote {runs_p}")


if __name__ == "__main__":
    main()
