#!/usr/bin/env python3
"""Add the cheap registry statistics to an existing stats report, without re-running.

Some manuscript numbers are derivable from files the pipeline has already
written — catalogue sizes from the length files, matrix coverage from a header
line, accumulation fit quality from the persisted fit CSVs. Re-running whole
sections to register them would cost hours for arithmetic that costs
milliseconds.

This reads those artifacts and merges the resulting tags into
``results/<ref>/stats_report.csv`` using the same rule as the pipeline: tags
computed here replace their previous values, everything else is left alone.

    python tools/refresh_registry_stats.py                 # all references
    python tools/refresh_registry_stats.py --refs UHGG combined
    python tools/refresh_registry_stats.py --dry-run

It is not a substitute for a run — it only registers quantities that are already
determined by existing outputs.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402

COLUMNS = ["tag", "section", "placement", "description", "value"]


def _row(tag, section, placement, description, value):
    return {"tag": tag, "section": section, "placement": placement,
            "description": description, "value": value}


def registry_rows(ref: str) -> list[dict]:
    rows: list[dict] = []
    cfg = config.REFERENCES.get(ref)
    if cfg is None:
        return rows

    # ── catalogue size, straight from the length file ────────────────────────
    lp = ROOT / cfg["lengths_file"]
    if lp.exists():
        d = pd.read_csv(lp, sep=r"\s+", header=None, names=["sgb", "bp"])
        rows.append(_row("n_sgbs_catalogue", "s1", "main",
                         f"SGBs in the {ref} reference catalogue", int(len(d))))
        rows.append(_row("catalogue_total_bp", "s1", "main",
                         f"Total catalogue length ({ref})",
                         f"{d['bp'].sum()/1e9:.2f} Gb"))

    # ── how many of them are actually columns of the breadth matrix ──────────
    bp_ = ROOT / cfg["breadth_file"]
    if bp_.exists():
        with open(bp_) as fh:
            header = fh.readline().rstrip("\n").split(",")
        sgbs = {c[:-5] for c in header[1:] if c.endswith("_mean")}
        rows.append(_row("n_sgbs_analysed", "s1", "main",
                         f"SGBs present as breadth-matrix columns ({ref})", len(sgbs)))

    # ── accumulation fit quality, from the fits the pipeline already wrote ───
    n_cat = int(len(pd.read_csv(lp, sep=r"\s+", header=None))) if lp.exists() else 0
    r2_seen = []
    for path in sorted(glob.glob(str(ROOT / "results" / ref / "accum_pooled_fits_vsN_cov*.csv"))):
        thr = Path(path).name.split("cov")[-1].replace(".csv", "")
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if not {"Series", "N_max", "a_abs_central", "R2_central"} <= set(d.columns):
            continue
        for series, g in d.groupby("Series"):
            r = g.sort_values("N_max").iloc[-1]
            slug = str(series).replace(" ", "_").replace("(", "").replace(")", "")
            if pd.notna(r["R2_central"]):
                r2_seen.append(float(r["R2_central"]))
                rows.append(_row(f"accum_R2_{slug}_{thr}", "s1", "supplement",
                                 f"Accumulation fit R2: {series} [cov {thr}]",
                                 f"{float(r['R2_central']):.4f}"))
            if pd.notna(r["a_abs_central"]) and n_cat:
                a = float(r["a_abs_central"])
                rows.append(_row(f"accum_asymptote_{slug}_{thr}", "s1", "supplement",
                                 f"Accumulation asymptote: {series} [cov {thr}]",
                                 f"{a:.1f} SGBs ({100*a/n_cat:.1f}% of the {n_cat:,}-SGB catalogue)"))
    if r2_seen:
        rows.append(_row("accum_R2_range", "s1", "main",
                         "Range of accumulation fit R2 across series and thresholds",
                         f"{min(r2_seen):.3f}-{max(r2_seen):.3f}"))
    return rows


def merge(ref: str, rows: list[dict], dry_run: bool = False) -> tuple[int, int]:
    path = ROOT / "results" / ref / "stats_report.csv"
    if not path.exists():
        print(f"  {ref}: no stats report yet — run the pipeline first")
        return (0, 0)
    prev = pd.read_csv(path)
    new = pd.DataFrame(rows)
    fresh = set(new["tag"])
    kept = prev[~prev["tag"].astype(str).isin(fresh)]
    updated = len(prev) - len(kept)
    out = pd.concat([kept, new], ignore_index=True, sort=False)
    if not dry_run:
        out.to_csv(path, index=False)
    return (len(rows) - updated, updated)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", default=list(config.REFERENCES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for ref in args.refs:
        rows = registry_rows(ref)
        if not rows:
            print(f"  {ref}: nothing derivable (missing inputs)")
            continue
        added, updated = merge(ref, rows, args.dry_run)
        verb = "would add" if args.dry_run else "added"
        print(f"  {ref:<9} {verb} {added} tag(s), refreshed {updated}"
              f"   ({len(rows)} registry statistics)")
    if args.dry_run:
        print("\n  (dry run — nothing written)")


if __name__ == "__main__":
    main()
