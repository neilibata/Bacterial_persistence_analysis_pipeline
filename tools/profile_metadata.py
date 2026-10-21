#!/usr/bin/env python3
"""Column-by-column profile of the metadata the pipeline reads.

Completeness is the first thing a reader of Table S3 or a Zenodo depositor
needs and the last thing a CSV tells you, so this writes it down: per column,
how many values are present, how many are missing, how many are distinct, and
what the values actually look like.

"Missing" here is not just NaN. These files came from 100+ studies via several
ingestion passes, and absence is spelled many ways -- empty string, "NA",
"na", "nan", "none", "null", "unknown", "-", "?". A column that is 100%
non-null and 60% the string "unknown" is not complete, and counting only NaN
would say it was.

Also reports whether each column is referenced anywhere in the pipeline code,
so a reader can tell a live input from a carried-along passenger.

    python tools/profile_metadata.py                     # config's META_FILE
    python tools/profile_metadata.py --file other.csv
    python tools/profile_metadata.py --out docs/metadata_profile.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Spellings of absence seen across the contributing studies.
NULL_TOKENS = {"", "na", "nan", "none", "null", "unknown", "-", "?", "n/a",
               "not applicable", "not provided", "missing", "undetermined"}


def blank_mask(s: pd.Series):
    """True where the value carries no information."""
    if s.dtype.kind in "ifc":
        return s.isna()
    t = s.astype(str).str.strip().str.lower()
    return s.isna() | t.isin(NULL_TOKENS)


def code_references(names, search_dirs=("persistence", "figures", "tools")):
    """{column: n_files referencing it} by literal string search."""
    hits = {n: 0 for n in names}
    for d in search_dirs:
        for f in (ROOT / d).rglob("*.py"):
            try:
                src = f.read_text()
            except Exception:
                continue
            for n in names:
                if f'"{n}"' in src or f"'{n}'" in src:
                    hits[n] += 1
    return hits


def describe(s: pd.Series, blank, n):
    """A short, honest summary of what is in the column."""
    present = s[~blank]
    if present.empty:
        return "(entirely empty)"
    num = pd.to_numeric(present, errors="coerce")
    if num.notna().sum() >= 0.95 * len(present) and num.notna().any():
        return (f"numeric  min {num.min():g}  med {num.median():g}  "
                f"max {num.max():g}")
    vc = present.astype(str).str.strip().value_counts()
    top = "; ".join(f"{k}={v:,}" for k, v in vc.head(3).items())
    return (f"{len(vc):,} distinct  |  {top}"
            + (" ; ..." if len(vc) > 3 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--out", default="docs/metadata_profile.csv")
    a = ap.parse_args()

    if a.file:
        path = a.file
    else:
        from persistence import config
        path = config.META_FILE
    df = pd.read_csv(path, low_memory=False)
    n = len(df)
    print(f"  {path}\n  {n:,} rows x {len(df.columns)} columns\n")

    refs = code_references(list(df.columns))
    rows = []
    for i, c in enumerate(df.columns, 1):
        s = df[c]
        bl = blank_mask(s)
        n_present = int((~bl).sum())
        rows.append(dict(
            n=i, column=c, dtype=str(s.dtype),
            non_null=int(s.notna().sum()),
            nan=int(s.isna().sum()),
            present=n_present,
            blank=int(bl.sum()),
            pct_complete=round(100.0 * n_present / n, 2),
            n_unique=int(s[~bl].nunique()),
            used_in_code=refs.get(c, 0),
            summary=describe(s, bl, n),
        ))
    prof = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.to_csv(out, index=False)

    hdr = (f"  {'#':>3} {'column':<34}{'compl%':>8}{'NaN':>8}"
           f"{'blank':>8}{'uniq':>8}{'code':>6}  summary")
    print(hdr)
    print("  " + "-" * (len(hdr) + 30))
    for r in rows:
        print(f"  {r['n']:>3} {r['column'][:34]:<34}{r['pct_complete']:>8.2f}"
              f"{r['nan']:>8,}{r['blank']:>8,}{r['n_unique']:>8,}"
              f"{r['used_in_code']:>6}  {r['summary'][:58]}")

    print(f"\n  wrote {out}")
    full = [r for r in rows if r["pct_complete"] == 100.0]
    empty = [r for r in rows if r["present"] == 0]
    print(f"\n  {len(full)} columns 100% complete; {len(empty)} entirely empty")
    if empty:
        print("  entirely empty: " + ", ".join(r["column"] for r in empty))
    print(f"  {sum(1 for r in rows if r['used_in_code'] == 0)} columns are "
          "not referenced by any code (carried through to the deposit only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
