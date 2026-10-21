#!/usr/bin/env python3
"""Emit the two provenance maps that tie pipeline outputs to the paper.

Without these, every number in the manuscript is only traceable by reading the
code that produced it. Two files are written to docs/:

  results_mapping.csv  one row per tagged statistic: which manuscript section
                       it belongs to, what it currently evaluates to for each
                       reference, and the artefact it was computed from.
  plots_mapping.csv    one row per paper figure/table named in
                       tools/paper_manifest.csv: where its file actually is,
                       when it was last built, and whether that predates the
                       current pipeline run.

Both are regenerated, never hand-edited:

    python tools/build_mappings.py
    python tools/build_mappings.py --stale-before 2026-08-20
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFS = ["combined", "UHGG", "lev", "chimp"]
# Where a manifest source_glob may live. Order matters: a reference-specific
# hit should win over the same basename sitting in the repo root.
def _search_bases(ref: str) -> list[str]:
    bases = [f"results/{ref}"] if ref in REFS else []
    return bases + ["main_text_figures", "results", "."] + [f"results/{r}" for r in REFS]


def build_results_mapping(refs: list[str]) -> pd.DataFrame:
    """One row per (tag, reference) with its value and originating section."""
    frames = []
    for ref in refs:
        p = ROOT / "results" / ref / "stats_report.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False).drop_duplicates("tag")
        keep = [c for c in ("tag", "section", "placement", "description", "value") if c in d.columns]
        d = d[keep].copy()
        d["reference"] = ref
        d["source_artifact"] = f"results/{ref}/stats_report.csv"
        d["fill_in"] = f"results/{ref}/manuscript_fill_ins.txt"
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # A tag whose value differs between references is one a reader can get
    # wrong by consulting the wrong results/ directory -- flag it explicitly.
    spread = out.groupby("tag")["value"].nunique(dropna=False)
    out["varies_by_reference"] = out["tag"].map(spread.gt(1))
    return out.sort_values(["section", "placement", "tag", "reference"])


def build_plots_mapping(stale_before: dt.date) -> pd.DataFrame:
    manifest = ROOT / "tools" / "paper_manifest.csv"
    rows = []
    for line in manifest.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) < 7 or f[0] == "item_id":
            continue
        rows.append(f)

    out = []
    for item, cat, ref, src, paper, desc, conf in rows:
        hits: list[str] = []
        for base in _search_bases(ref):
            hits += glob.glob(str(ROOT / base / src))
        hits = sorted({h for h in hits if os.path.exists(h)})
        if hits:
            built = dt.date.fromtimestamp(max(os.path.getmtime(h) for h in hits))
            status = "stale" if built < stale_before else "current"
            resolved = ", ".join(os.path.relpath(h, ROOT) for h in hits[:3])
        else:
            built, status, resolved = None, "MISSING", ""
        out.append(dict(item_id=item, category=cat, paper_name=paper, reference=ref,
                        source_glob=src, resolved_path=resolved,
                        last_built=built, status=status,
                        confidence=conf, description=desc))
    return pd.DataFrame(out).sort_values(["category", "item_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", default=REFS)
    ap.add_argument("--stale-before", default="2026-08-20",
                    help="Builds older than this date are reported as stale (YYYY-MM-DD)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = ap.parse_args()
    cutoff = dt.date.fromisoformat(args.stale_before)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = build_results_mapping(args.refs)
    rp = args.out_dir / "results_mapping.csv"
    res.to_csv(rp, index=False)
    print(f"results_mapping : {len(res):5,d} rows  ({res['tag'].nunique():,} tags x {res['reference'].nunique()} refs)")
    if len(res):
        v = res[res.varies_by_reference].tag.nunique()
        print(f"                  {v:,} tags differ between references")
        print(res.groupby('placement').size().to_string())

    plots = build_plots_mapping(cutoff)
    pp = args.out_dir / "plots_mapping.csv"
    plots.to_csv(pp, index=False)
    print(f"\nplots_mapping   : {len(plots)} entries")
    print(plots.groupby('status').size().to_string())
    bad = plots[plots.status != "current"]
    if len(bad):
        print("\n  needs attention:")
        for _, r in bad.iterrows():
            print(f"    {r.status:8s} {r.item_id:14s} {r.source_glob[:58]}")
    print(f"\nWrote {rp.relative_to(ROOT)} and {pp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
