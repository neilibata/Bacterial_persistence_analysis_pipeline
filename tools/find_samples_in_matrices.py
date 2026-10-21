#!/usr/bin/env python3
"""Search every matrix build for samples missing from the canonical one.

Large projects accumulate matrix variants -- _merged, _combined, _dedup, _new,
_new_extra -- built at different times from different batches. A sample absent
from the one the analysis uses may still be present, fully processed, in
another. Recovering a row costs nothing; reprocessing the sample costs hours.

Only the first column of each matrix is read, so this is cheap even on
multi-gigabyte files.

    python tools/find_samples_in_matrices.py \\
        --metadata metadata_M1.csv \\
        --want lev_missing_sample_ids.todo.txt \\
        --matrices /path/to/*lev*.csv

Reports, per matrix, how many of the wanted samples it contains, and writes a
recovery plan naming the best source for each.
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
TOKEN = re.compile(r"[A-Za-z0-9_.\-]+")


def aliases(m1: pd.DataFrame) -> dict[str, set[str]]:
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
                if tok and tok.lower() != "nan":
                    al.add(tok)
    return out


def first_column(path: Path) -> set[str]:
    """Row identifiers, reading only up to the first comma of each line."""
    ids = set()
    with open(path, "r", errors="replace") as fh:
        fh.readline()                      # header
        for line in fh:
            i = line.find(",")
            ids.add((line[:i] if i > 0 else line).strip())
    ids.discard("")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--want", required=True, type=Path)
    ap.add_argument("--matrices", required=True, nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("recovery_plan.tsv"))
    args = ap.parse_args()

    m1 = pd.read_csv(args.metadata, low_memory=False, dtype=str)
    alias = aliases(m1)
    want = [l.strip() for l in open(args.want) if l.strip()]
    want_alias = {s: alias.get(s, {s}) for s in want}
    print(f"looking for {len(want):,} samples across {len(args.matrices)} matrix build(s)\n")

    found_in: dict[str, list[tuple[str, str]]] = {s: [] for s in want}
    print(f"{'matrix':<62} {'rows':>9} {'wanted found':>13}")
    for mp in args.matrices:
        if not mp.exists():
            print(f"{mp.name:<62} {'MISSING':>9}")
            continue
        try:
            ids = first_column(mp)
        except Exception as exc:
            print(f"{mp.name:<62} error: {exc}")
            continue
        n = 0
        for s in want:
            hit = want_alias[s] & ids
            if hit:
                found_in[s].append((mp.name, sorted(hit)[0]))
                n += 1
        print(f"{mp.name:<62} {len(ids):>9,} {n:>13,}")

    recoverable = {s: v for s, v in found_in.items() if v}
    print(f"\n  RECOVERABLE from an existing matrix : {len(recoverable):,}")
    print(f"  still need reprocessing             : {len(want) - len(recoverable):,}")

    with open(args.out, "w") as fh:
        fh.write("Chosen_match_ID\tsource_matrix\trow_id\n")
        for s, v in recoverable.items():
            name, rid = v[0]
            fh.write(f"{s}\t{name}\t{rid}\n")
    still = [s for s in want if not found_in[s]]
    todo_p = args.out.with_suffix(".still_todo.txt")
    open(todo_p, "w").write("\n".join(still) + ("\n" if still else ""))
    print(f"\n  wrote {args.out}")
    print(f"  wrote {todo_p}")

    if recoverable:
        print("\n  examples:")
        for s, v in list(recoverable.items())[:5]:
            print(f"    {s:<22} in {v[0][0]}  as {v[0][1]}")


if __name__ == "__main__":
    main()
