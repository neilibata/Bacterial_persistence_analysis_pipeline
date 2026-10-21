#!/usr/bin/env python3
"""Data S2: the curated per-sample metadata, as deposited, in a workbook.

Data S2 is Metadata Table M1. Its authority is `final_run/metadata_M1.csv`,
which is what the pipeline reads and what the input record deposits, so the
submitted workbook has to be that table and nothing else: same rows, same
columns, same order.

Why this exists rather than `build_m1_workbooks.py`. That tool renders M1 for
*inspection*, shading every cell that changed against the pre-correction
cohort, and it aligns the two tables by position. The cohort has since lost
four samples, so the comparison no longer broadcasts and the tool cannot run:

    ValueError: operands could not be broadcast together
                with shapes (18299,) (18303,)

The workbook it last produced is therefore baseline-era -- 18,303 rows by 110
columns against the current 18,299 by 118. It lacks `Subject_ID`,
`mdb_case_eligible`, `subject_primary` and nine other columns the health
analysis depends on, and carries `Obesity (old)`, `disease_BD` and two review
annotations that no longer exist. Shading is a reviewing aid; agreeing with
the deposited CSV is a requirement. This does the second and drops the first.

The conversion is verified rather than assumed: every cell is read back from
the written workbook and compared with the CSV, and the tool fails if any
value did not survive the round trip.

    python tools/build_data_s2.py
    python tools/build_data_s2.py --out Data_S2_metadata_M1.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M1_CSV = ROOT / "final_run" / "metadata_M1.csv"
SHEET = "Metadata M1"


def _as_text(df: pd.DataFrame) -> pd.DataFrame:
    """A comparable string view, insensitive to float formatting.

    The workbook stores numbers as numbers so a reader can filter and compute
    on them, which means a value's text form can change across the round trip
    (1.0 for 1, say). Comparing through a common normalisation checks that the
    value survived without demanding its spelling did.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        num = pd.to_numeric(s, errors="coerce")
        # A column is numeric only where every non-blank cell parses.
        blank = s.isna() | (s.astype(str).str.strip() == "")
        if (num.notna() | blank).all():
            out[c] = num.map(lambda v: "" if pd.isna(v) else f"{float(v):.10g}")
        else:
            out[c] = s.fillna("").astype(str).str.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=M1_CSV,
                    help="the deposited metadata table (default: "
                         "final_run/metadata_M1.csv)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "Data_S2_metadata_M1.xlsx",
                    help="workbook to write")
    a = ap.parse_args()

    if not a.csv.exists():
        print(f"  no such table: {a.csv}")
        return 2

    src = pd.read_csv(a.csv, low_memory=False)
    print(f"  {a.csv.name}: {len(src):,} rows x {len(src.columns)} columns")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.out, engine="openpyxl") as xl:
        src.to_excel(xl, sheet_name=SHEET, index=False)
        ws = xl.sheets[SHEET]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    print(f"  wrote {a.out}  ({a.out.stat().st_size / 1e6:.1f} MB)")

    back = pd.read_excel(a.out, sheet_name=SHEET)
    if list(back.columns) != list(src.columns):
        print("  FAIL: column names or order did not survive the round trip")
        return 1
    if back.shape != src.shape:
        print(f"  FAIL: shape changed, {src.shape} -> {back.shape}")
        return 1
    ls, lb = _as_text(src), _as_text(back)
    diff = (ls != lb)
    n = int(diff.to_numpy().sum())
    if n:
        print(f"  FAIL: {n:,} cell(s) did not survive the round trip")
        for c in [c for c in diff.columns if diff[c].any()][:8]:
            i = int(diff[c].to_numpy().nonzero()[0][0])
            print(f"      {c}: row {i + 2}  "
                  f"{ls[c].iat[i]!r} -> {lb[c].iat[i]!r}")
        return 1
    print(f"  verified: {back.shape[0]:,} x {back.shape[1]}, "
          f"every cell matches {a.csv.name}")
    print(f"  sheet {SHEET!r}, header frozen, auto-filter on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
