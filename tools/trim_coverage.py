#!/usr/bin/env python3
"""How much of the cohort the surviving Trim Galore reports actually cover.

The original preprocessing was carried out in manually orchestrated stages over
an extended period and its per-run logs survive only in part. A trimming
statistic quoted for "the cohort" is therefore only meaningful alongside the
fraction of the cohort it rests on, and that fraction has to be measured rather
than assumed.

Reports are named per run; the analysis is keyed on `Chosen_match_ID`. A sample
is counted as fully covered only when every one of its paired runs has a
report, since a partially covered multi-run sample would understate its own
input volume.

    python tools/trim_coverage.py
    python tools/trim_coverage.py --trim resources/original_trim_summary.csv
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1", type=Path, default=ROOT / "metadata_M1.csv")
    ap.add_argument("--trim", type=Path,
                    default=ROOT / "resources" / "original_trim_summary.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "trim_coverage.csv")
    args = ap.parse_args()

    trim = pd.read_csv(args.trim)
    trim["sample"] = trim["sample"].astype(str).str.strip()
    have = dict(zip(trim["sample"], trim.index))
    tin = dict(zip(trim["sample"], pd.to_numeric(trim["trim_bp_in"], errors="coerce")))
    tout = dict(zip(trim["sample"], pd.to_numeric(trim["trim_bp_out"], errors="coerce")))
    print(f"  trim reports        : {len(trim):,} runs")

    m1 = pd.read_csv(args.m1, low_memory=False, dtype=str)
    m1 = m1.drop_duplicates("Chosen_match_ID").copy()
    m1["_key"] = m1["Chosen_match_ID"].astype(str).str.strip()
    m1["_runs"] = m1["Paired SRR List"].apply(
        lambda v: [] if pd.isna(v) else list(dict.fromkeys(RUN.findall(str(v)))))
    lab = m1["Lifestyle"].astype(str).str.strip()
    cls = m1[lab.isin(CLASSIFIED)].copy()
    print(f"  classified samples  : {len(cls):,}")

    cls["n_runs"] = cls["_runs"].str.len()
    cls["n_covered"] = cls["_runs"].apply(lambda rs: sum(1 for r in rs if r in have))
    cls["status"] = "none"
    cls.loc[cls["n_covered"] > 0, "status"] = "partial"
    cls.loc[(cls["n_covered"] == cls["n_runs"]) & (cls["n_runs"] > 0), "status"] = "full"
    cls["bp_in"] = cls["_runs"].apply(
        lambda rs: sum(tin.get(r, 0) or 0 for r in rs))
    cls["bp_out"] = cls["_runs"].apply(
        lambda rs: sum(tout.get(r, 0) or 0 for r in rs))

    n = len(cls)
    print("\n  COVERAGE BY SAMPLE")
    for st in ("full", "partial", "none"):
        k = int((cls["status"] == st).sum())
        print(f"    {st:<8} {k:>7,}  ({k/n*100:5.1f}%)")

    all_runs = {r for rs in cls["_runs"] for r in rs}
    print(f"\n  runs in cohort      : {len(all_runs):,}")
    print(f"  runs with a report  : {len(all_runs & set(have)):,}  "
          f"({len(all_runs & set(have))/len(all_runs)*100:.1f}%)")
    orphan = set(have) - all_runs
    if orphan:
        print(f"  reports not in cohort: {len(orphan):,}  "
              f"(runs processed but excluded during curation)")

    print("\n  COVERAGE BY LIFESTYLE (fully covered samples)")
    for life, grp in cls.groupby("Lifestyle"):
        k = int((grp["status"] == "full").sum())
        print(f"    {life:<14} {k:>6,} / {len(grp):>6,}  ({k/len(grp)*100:5.1f}%)")

    full = cls[cls["status"] == "full"]
    if len(full):
        bi, bo = full["bp_in"].sum(), full["bp_out"].sum()
        print(f"\n  TRIMMING (fully covered samples, n={len(full):,})")
        print(f"    bases before trim : {bi/1e12:.2f} Tbp")
        print(f"    bases after trim  : {bo/1e12:.2f} Tbp")
        print(f"    retained          : {bo/bi*100:.2f}%")
        # Sanity: does the archive's own base count agree with the reports?
        tb = pd.to_numeric(full["Total bases"], errors="coerce")
        ok = tb.notna() & (full["bp_in"] > 0)
        if ok.sum():
            ratio = (full.loc[ok, "bp_in"] / tb[ok]).median()
            print(f"    report bp_in / metadata 'Total bases', median ratio: "
                  f"{ratio:.3f}  (n={int(ok.sum()):,})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cls[["_key", "Lifestyle", "StudyFolder", "n_runs", "n_covered",
         "status", "bp_in", "bp_out"]].rename(
        columns={"_key": "Chosen_match_ID"}).to_csv(args.out, index=False)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
