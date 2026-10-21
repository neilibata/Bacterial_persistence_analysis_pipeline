#!/usr/bin/env python3
"""Recompute Table S1 age and health statistics for the rows that change.

`table_s1a.py` derives these from the analytic cohort already, so this is a
check on that derivation rather than a replacement for it: it loads the same
cohort through the same entry point, aggregates independently, and reports any
disagreement. Agreement is the expected result and the point of running it --
the values are going into a manuscript table and were last computed before the
Leviatan backfill and the duplicate-ID fix.

Age is reported as median [Q1-Q3] over samples with a numeric age, and the
count of those is reported alongside, since several studies record age for only
part of their cohort and a median over 3 of 71 samples should not be quoted
without saying so.

    python tools/recompute_s1_rows.py
    python tools/recompute_s1_rows.py --all      # every study, not just changed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402
from persistence.loading import load_reference_analytic  # noqa: E402

# Studies whose Table S1 rows change (tools/patch_table_s1.py)
CHANGED = [
    ("CarterM", "2023", "Tanzania"),
    ("LokmerA", "2019", "Cameroon"),
    ("Obregon-TitoAJ", "2015", "Peru"),
    ("RubelMA", "2020", "Cameroon"),
    ("YeZ", "2020", "China"),
]

YES = {"yes", "1", "true", "unhealthy", "y"}


def is_unhealthy(s) -> bool:
    return str(s).strip().lower() in YES


def fmt_stat(vals: pd.Series, dp: int = 1) -> str:
    """median [Q1-Q3]. Age is quoted to 1 dp, Gbp/sample to 2, matching table_s1a."""
    v = vals.dropna()
    if len(v) == 0:
        return ""
    q1, med, q3 = v.quantile(.25), v.median(), v.quantile(.75)
    return f"{med:.{dp}f} [{q1:.{dp}f}-{q3:.{dp}f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="UHGG")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    print(f"  loading the {args.ref} analytic cohort ...", flush=True)
    data = load_reference_analytic(args.ref)
    meta = data.meta.copy()
    print(f"  cohort: {len(meta):,} samples", flush=True)

    meta["_age"] = pd.to_numeric(meta.get("Age"), errors="coerce")
    meta["_unh"] = meta.get("Unhealthy", pd.Series("", index=meta.index)).apply(is_unhealthy)
    meta["_gbp"] = (data.total_bp.reindex(meta.index) / 1e9)

    def country_full(df):
        for c in ("Country (full)", "Country_full", "Country"):
            if c in df.columns:
                return df[c].astype(str).str.strip()
        return pd.Series("", index=df.index)

    meta["_cty"] = country_full(meta)
    meta["_auth"] = meta.get("Author", pd.Series("", index=meta.index)).astype(str).str.strip()
    meta["_yr"] = meta.get("Year", pd.Series("", index=meta.index)).astype(str).str.strip()
    meta["_life"] = meta.get("Lifestyle", pd.Series("", index=meta.index)).astype(str).str.strip()

    g = meta.groupby(["_auth", "_yr", "_cty", "_life"], dropna=False)
    rows = []
    for (a, y, c, life), grp in g:
        rows.append({
            "Author": a, "Year": y, "Country": c, "Lifestyle": life,
            "N": len(grp),
            "Age median [IQR]": fmt_stat(grp["_age"], 1),
            "age_n": int(grp["_age"].notna().sum()),
            "Gbp/sample median [IQR]": fmt_stat(grp["_gbp"], 2),
            "unhealthy_pct": round(100.0 * grp["_unh"].mean(), 1),
            "unhealthy_n": int(grp["_unh"].sum()),
        })
    mine = pd.DataFrame(rows)

    pipe = pd.read_csv(ROOT / "results" / args.ref / "table_s1a_study_summary_formatted.csv",
                       dtype=str).fillna("")

    def norm(s):
        return str(s).replace("–", "-").replace("−", "-").strip()

    want = mine if args.all else mine[
        mine.apply(lambda r: (r.Author, r.Year, r.Country) in
                   [(a, y, c) for a, y, c in CHANGED], axis=1)]

    print(f"\n  {'study':<44}{'N':>5}{'age median [IQR]':>22}{'age n':>7}"
          f"{'unhealthy%':>12}")
    print("  " + "-" * 90)
    disagree = 0
    for _, r in want.sort_values(["Author", "Year", "Country", "Lifestyle"]).iterrows():
        label = f"{r.Author} {r.Year} {r.Country} [{r.Lifestyle}]"
        print(f"  {label:<44}{r.N:>5}{r['Age median [IQR]']:>22}{r.age_n:>7}"
              f"{r.unhealthy_pct:>11.1f}%")
        m = pipe[(pipe.Author == r.Author) & (pipe.Year == str(r.Year)) &
                 (pipe.Country_full == r.Country) & (pipe.Lifestyle == r.Lifestyle)]
        if len(m) != 1:
            print(f"      ! pipeline has {len(m)} matching rows")
            disagree += 1
            continue
        p = m.iloc[0]
        for col, mineval in (("N", str(r.N)),
                             ("Age median [IQR]", r["Age median [IQR]"]),
                             ("Gbp/sample median [IQR]", r["Gbp/sample median [IQR]"]),
                             ("unhealthy_pct", f"{r.unhealthy_pct:g}")):
            pv = norm(p[col])
            mv = norm(mineval)
            if pv != mv:
                # unhealthy_pct may differ only in trailing-zero formatting
                try:
                    if abs(float(pv) - float(mv)) < 0.05:
                        continue
                except ValueError:
                    pass
                print(f"      ! {col}: pipeline={pv!r}  recomputed={mv!r}")
                disagree += 1

    print(f"\n  disagreements with the pipeline table: {disagree}")
    if disagree == 0:
        print("  the pipeline's Table S1 values are confirmed for these rows.")
    out = ROOT / "results" / args.ref / "table_s1a_recomputed.csv"
    mine.to_csv(out, index=False)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
