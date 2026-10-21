#!/usr/bin/env python3
"""Merge the appended samples into their studies and back-fill what siblings agree on.

The 260 samples added on 2026-08-20 sit as a contiguous block at the end of the
metadata, so a study like BrambleMS 2021 occupies row 3,110 and then rows
18,130-18,303. A reader scanning the file sees six samples where there are 180.

Two operations, both conservative:

1. Stable sort by (Author, Year, Country) so each study is contiguous. Stability
   matters: loading.py resolves duplicate Chosen_match_IDs with
   drop_duplicates(keep="first"), so a reshuffle could silently change which row
   survives. All 21 duplicate pairs are within CarterM 2023 and both members are
   pre-existing, so a stable sort keeps the same winner -- the tool asserts this
   rather than assuming it.

2. Back-fill only fields the rest of the study agrees on. A value is inherited
   only where every non-blank sibling row carries the same one, which admits
   study-constant fields (Age group where a study is single-cohort, Obesity
   flags, Healthy) and correctly refuses per-person fields: Gender, Age and BMI
   have no unanimous value in any affected study and are left blank for external
   lookup.

    python tools/reorganise_new_samples.py                # dry run
    python tools/reorganise_new_samples.py --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "Chosen_match_ID"

# Considered for inheritance. Per-person fields are deliberately absent.
INHERIT = ["Age group", "Obesity", "Overweight", "Healthy",
           "Diseased", "Diseased (metabolic) or obese", "median_read_length",
           "non_westernized", "Index"]


def _blank(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["nan", "", "none"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "metadata_Aug_26_2026.csv")
    ap.add_argument("--baseline", type=Path, default=ROOT / "archive" / "Metadata_M1_081426.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "metadata_Aug_26_2026_organised.csv")
    ap.add_argument("--changelog", type=Path, default=ROOT / "docs" / "new_sample_backfill.csv")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.src, low_memory=False)
    base = pd.read_csv(args.baseline, low_memory=False)
    added = set(m[ID].astype(str).str.strip()) - set(base[ID].astype(str).str.strip())
    is_new = m[ID].astype(str).str.strip().isin(added)
    print(f"source {args.src.name}: {len(m):,} rows, {int(is_new.sum())} appended samples")

    # ── which duplicate row currently wins ────────────────────────────────
    ids = m[ID].astype(str).str.strip()
    winners_before = m[~ids.duplicated(keep="first")].index

    # ── 1. stable sort ────────────────────────────────────────────────────
    m["_orig"] = range(len(m))
    key = ["Author", "Year", "Country"]
    key = [k for k in key if k in m.columns]
    ordered = m.sort_values(key + ["_orig"], kind="mergesort").reset_index(drop=True)

    ids2 = ordered[ID].astype(str).str.strip()
    winners_after = set(ordered.loc[~ids2.duplicated(keep="first"), "_orig"])
    same = winners_after == set(winners_before)
    print(f"  duplicate resolution preserved by the sort: {same}")
    if not same:
        raise SystemExit("ABORT: the sort would change which duplicate row survives.")

    moved = int((ordered["_orig"] != range(len(ordered))).sum())
    print(f"  rows whose position changes: {moved:,}")

    # ── 2. back-fill from unanimous siblings ─────────────────────────────
    o_new = ordered[ID].astype(str).str.strip().isin(added)
    rows = []
    for c in INHERIT:
        if c not in ordered.columns:
            continue
        need = o_new & _blank(ordered[c])
        if not need.any():
            continue
        filled = 0
        for (a, y), g in ordered[need].groupby(["Author", "Year"]):
            sib = ordered[(ordered.Author == a) & (ordered.Year == y) & ~o_new][c]
            sv = sib[~_blank(sib)].astype(str).str.strip()
            if len(sv) and sv.nunique() == 1:
                val = sv.iloc[0]
                for i in g.index:
                    rows.append(dict(Chosen_match_ID=ordered.at[i, ID], Author=a, Year=y,
                                     column=c, value=val,
                                     basis=f"unanimous across {len(sv)} sibling rows"))
                    if args.apply:
                        ordered.at[i, c] = val
                filled += len(g)
        print(f"  {c:32s} back-filled {filled:4d} of {int(need.sum()):4d} missing")

    if not args.apply:
        print("\n  dry run — nothing written. Re-run with --apply.")
        return

    ordered = ordered.drop(columns=["_orig"])
    ordered.to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.changelog, index=False)
    print(f"\n  wrote {args.out.name} ({len(ordered):,} rows)")
    print(f"  wrote {args.changelog.relative_to(ROOT)} ({len(rows):,} cells back-filled)")


if __name__ == "__main__":
    main()
