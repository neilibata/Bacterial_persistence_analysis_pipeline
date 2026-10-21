#!/usr/bin/env python3
"""Make obesity count as unhealthy, consistently, in the cohort metadata.

The metadata contradicts itself: a sample can be flagged obese and still be
labelled Healthy. It is not a rounding error at the edge of a derived
criterion -- almost all of these carry an explicit obesity column, not just a
BMI over 30 -- and it is concentrated in a handful of studies that recorded
obese participants as healthy controls.

That matters because `Unhealthy` is the control/case split for every Section 3
comparison. Leaving it as it stands means the same trait counts as disease in
one study and as health in another, and the obese samples labelled Healthy sit
in the control group of every other condition's test.

This rewrites `Unhealthy` to "Unhealthy" wherever the obesity flag is set, and
writes a new dated metadata file rather than editing the one on disk, so the
previous cohort stays reproducible. Every changed row is listed in
docs/obesity_health_corrections.csv.

    python tools/apply_obesity_health_rule.py --dry-run
    python tools/apply_obesity_health_rule.py

Afterwards the new file has to become the one config picks up -- it is written
with a name that sorts ahead of the current default in
config.META_FILE_CANDIDATES, which the script prints a reminder about.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "docs" / "obesity_health_corrections.csv"


def obesity_flag(meta):
    """The pipeline's own obesity flag: the obesity columns, or BMI >= 30.

    Reproduced from loading.py rather than imported, because loading builds it
    while reading a full reference and this needs only the metadata.
    """
    from persistence import config
    from persistence.common import is_yes

    cols = [c for c in getattr(config, "OBESITY_COLUMNS", []) if c in meta.columns]
    if cols:
        flag = meta[cols].apply(lambda r: any(is_yes(v) for v in r), axis=1)
    else:
        flag = pd.Series(False, index=meta.index)
    by_col = flag.copy()
    if "BMI" in meta.columns:
        bmi = pd.to_numeric(meta["BMI"], errors="coerce")
        flag = flag | (bmi >= 30).fillna(False)
    return flag, by_col, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output metadata csv (default: dated name)")
    a = ap.parse_args()

    from persistence import config
    src = Path(config.META_FILE)
    print(f"  reading {src}")
    meta = pd.read_csv(src, low_memory=False)

    flag, by_col, cols = obesity_flag(meta)
    print(f"  obesity columns used: {cols}")
    print(f"  obesity flag positive: {int(flag.sum()):,} of {len(meta):,}")

    target = flag & (meta["Unhealthy"] == "Healthy")
    n = int(target.sum())
    print(f"  obese AND labelled Healthy: {n:,}")
    if n == 0:
        print("  nothing to change; the metadata is already consistent")
        return

    only_bmi = int((target & ~by_col).sum())
    print(f"    of these, {n - only_bmi:,} carry an explicit obesity column "
          f"and {only_bmi:,} are flagged by BMI >= 30 alone")
    for col in ("Study", "study", "Study_ID"):
        if col in meta.columns:
            counts = meta.loc[target, col].value_counts()
            print(f"    by {col}: "
                  + ", ".join(f"{k}={v}" for k, v in counts.head(6).items()))
            break

    # What the split looks like before and after, which is the number that
    # actually moves in every Section 3 comparison.
    ind = meta["Lifestyle"] == "Industrial"
    before_h = int((ind & (meta["Unhealthy"] == "Healthy")).sum())
    before_u = int((ind & (meta["Unhealthy"] == "Unhealthy")).sum())
    moved = int((ind & target).sum())
    print(f"  Industrial Healthy {before_h:,} -> {before_h - moved:,}; "
          f"Unhealthy {before_u:,} -> {before_u + moved:,}")

    log = meta.loc[target, [c for c in
                            ("Chosen_match_ID", "Study", "Lifestyle", "BMI",
                             "Unhealthy") + tuple(cols)
                            if c in meta.columns]].copy()
    log["Unhealthy_new"] = "Unhealthy"
    log["reason"] = ["explicit obesity column" if b else "BMI >= 30"
                     for b in by_col[target]]

    if a.dry_run:
        print("  dry run; nothing written")
        return

    meta.loc[target, "Unhealthy"] = "Unhealthy"

    stamp = datetime.datetime.now().strftime("%b_%d_%Y").replace(" ", "")
    out = Path(a.out) if a.out else ROOT / f"metadata_{stamp}_obesity_unhealthy.csv"
    meta.to_csv(out, index=False)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False)
    print(f"  wrote {out.name}  ({len(meta):,} rows)")
    print(f"  wrote {LOG.relative_to(ROOT)}  ({len(log):,} changed rows)")
    print("\n  Next: put this file at the head of "
          "config.META_FILE_CANDIDATES, then re-run Section 3.")


if __name__ == "__main__":
    main()
