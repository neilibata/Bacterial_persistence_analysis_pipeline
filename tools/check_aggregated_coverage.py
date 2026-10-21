#!/usr/bin/env python3
"""Which samples are already aggregated, matching on sample identity not filename.

A per-sample CSV is named after whichever run accession the preprocessing
pipeline happened to use. The analysis keys on `Chosen_match_ID`, which may be a
*different* accession for the same biological sample, or not an accession at all
(e.g. "FAT_006-22-0-0"). Comparing filenames to Chosen_match_IDs therefore
understates coverage: a sample can be fully processed and still look missing.

This resolves every accession a sample is known by -- Chosen_match_ID,
SRA_Accession, Paired SRR List, Secondary_run_IDs, BioSample -- and reports a
sample as covered if a CSV exists under *any* of them.

    python tools/check_aggregated_coverage.py \\
        --metadata metadata_M1.csv \\
        --aggregated-dir /path/to/aggregated_stats_lev_high_merged \\
        --want results/cohort_reconciliation/lev_missing_sample_ids.txt

Writes <want>.covered.txt and <want>.todo.txt next to the want-list.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

ACC_COLS = ["Chosen_match_ID", "SRA_Accession", "Paired SRR List",
            "Secondary_run_IDs", "BioSample", "Sample ID"]
SUFFIXES = ("_depth_and_coverage_stats.csv", "_aggregated_stats.csv")
TOKEN = re.compile(r"[A-Za-z0-9_.\-]+")


def sample_aliases(m1: pd.DataFrame) -> dict[str, set[str]]:
    """Chosen_match_ID -> every identifier that sample might be filed under."""
    out: dict[str, set[str]] = {}
    for _, row in m1.iterrows():
        key = str(row["Chosen_match_ID"]).strip()
        if not key or key.lower() == "nan":
            continue
        al = out.setdefault(key, {key})
        for c in ACC_COLS:
            v = row.get(c)
            if pd.isna(v):
                continue
            for tok in TOKEN.findall(str(v)):
                tok = tok.strip()
                if tok and tok.lower() != "nan":
                    al.add(tok)
    return out


def present(agg_dir: Path) -> set[str]:
    """Basenames present in the aggregated directory, suffix stripped."""
    got = set()
    for f in os.listdir(agg_dir):
        for suf in SUFFIXES:
            if f.endswith(suf):
                got.add(f[: -len(suf)])
                break
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--aggregated-dir", required=True, type=Path)
    ap.add_argument("--want", required=True, type=Path,
                    help="file of Chosen_match_IDs to check (one per line)")
    args = ap.parse_args()

    m1 = pd.read_csv(args.metadata, low_memory=False, dtype=str)
    aliases = sample_aliases(m1)
    have = present(args.aggregated_dir)
    want = [l.strip() for l in open(args.want) if l.strip()]

    print(f"aggregated dir : {len(have):,} CSVs")
    print(f"metadata       : {len(aliases):,} samples, "
          f"{sum(len(v) for v in aliases.values()):,} identifiers")
    print(f"checking       : {len(want):,} samples\n")

    covered, todo, by_alias = [], [], 0
    for s in want:
        al = aliases.get(s, {s})
        hit = al & have
        if hit:
            covered.append((s, sorted(hit)[0]))
            if sorted(hit)[0] != s:
                by_alias += 1
        else:
            todo.append(s)

    print(f"  COVERED  {len(covered):,}   (of which {by_alias:,} matched under a "
          f"different accession than Chosen_match_ID)")
    print(f"  TODO     {len(todo):,}\n")

    cov_p = args.want.with_suffix(".covered.txt")
    todo_p = args.want.with_suffix(".todo.txt")
    with open(cov_p, "w") as fh:
        for s, f in covered:
            fh.write(f"{s}\t{f}\n")
    open(todo_p, "w").write("\n".join(todo) + ("\n" if todo else ""))
    print(f"  wrote {cov_p}  (Chosen_match_ID <TAB> filename to read it from)")
    print(f"  wrote {todo_p}")

    if by_alias:
        print("\n  examples matched under an alias:")
        for s, f in [c for c in covered if c[1] != c[0]][:5]:
            print(f"    {s:<22} found as {f}")

    # A stricter check the filename test cannot do: are there CSVs in the
    # directory that belong to no sample in the metadata at all?
    known = set().union(*aliases.values()) if aliases else set()
    orphan = have - known
    if orphan:
        print(f"\n  note: {len(orphan):,} CSVs match no metadata sample "
              f"(e.g. {sorted(orphan)[:3]})")


if __name__ == "__main__":
    main()
