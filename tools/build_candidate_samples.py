#!/usr/bin/env python3
"""Candidate samples that exist in ENA, are not in M1, and were never processed.

"Not in M1" and "not processed" are different questions and only the second one
costs compute. A sample absent from M1 but present in the coverage matrices was
fetched and aligned already and was dropped later, on a curation call; a sample
absent from both has never been touched. Only the latter is a candidate, so
presence in the matrices is checked first and reported separately.

Every criterion is then evaluated and recorded per sample rather than applied as
a filter, because the interesting output is not a shorter list but a table that
says which criterion each sample fails. A sample failing only on depth is a
judgement call; one failing on three is not.

Two traps are checked explicitly, both found the hard way:

  Cultured isolates. 71 samples in the Hadza project and 63 in the Fiji project
  are single-organism cultures -- Bifidobacterium infantis, Segatella copri --
  derived from faecal samples. They are paired, WGS, gut-derived and deep, so
  every structured field says include. Only the free-text description betrays
  them ("This sample represents a ... genome isolated and sequenced from ...").
  Including one injects a single genome as though it were a community.

  Archive labels. `library_layout` is wrong for 100 of 101 runs checked against
  SRA's own read statistics, so a SINGLE label is recorded as unconfirmed rather
  than as a disqualification.

    python tools/build_candidate_samples.py --cache <ena_dir> --out candidates.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from persistence import config as cfg           # noqa: E402
from geo_resolve import resolve                    # noqa: E402
from rebuild_cohort_from_ena import body_site_verdict, MIN_BASES  # noqa: E402

ISOLATE = re.compile(
    r"isolated and sequenced from|represents a .{0,40}genome|"
    r"cultured bacterial|cultured archaeal|MIGS Cultured|single[- ]colony|"
    r"strain .{0,20}genome", re.I)

LIFESTYLES = {"H-G", "Transitional", "Industrial"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--rebuild", type=Path, required=True,
                    help="cohort_rebuild.csv, for the lifestyle assignment")
    ap.add_argument("--out", type=Path, default=Path("candidate_samples.csv"))
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    runs = pd.concat([pd.read_csv(f, sep="\t", dtype=str)
                      for f in glob.glob(str(args.cache / "*.tsv"))],
                     ignore_index=True)
    runs["_bases"] = pd.to_numeric(runs["base_count"], errors="coerce").fillna(0)
    runs["_sample"] = (runs["sample_accession"]
                       .fillna(runs["secondary_sample_accession"])
                       .fillna(runs["run_accession"]))

    # ---- identifiers already in M1, and what was already processed ----------
    in_m1 = set()
    for col in ("Chosen_match_ID", "SRA_Accession", "SRS ID", "BioSample"):
        if col in meta.columns:
            for v in meta[col].dropna():
                in_m1.update(x.strip() for x in str(v).split(";") if x.strip())
    secondary = set()
    if "Secondary_run_IDs" in meta.columns:
        for v in meta["Secondary_run_IDs"].dropna():
            secondary.update(x.strip() for x in
                             str(v).replace(",", ";").split(";") if x.strip())

    processed = set()
    for ref in ("UHGG", "lev", "combined", "chimp"):
        f = cfg.REFERENCES.get(ref, {}).get("breadth_file")
        if f and os.path.exists(f):
            processed |= set(pd.read_csv(f, usecols=[0], dtype=str)
                             .iloc[:, 0].astype(str))
    print(f"  M1 identifiers      : {len(in_m1):,}")
    print(f"  already processed   : {len(processed):,} matrix rows over 4 references")

    life = {}
    if args.rebuild.exists():
        rb = pd.read_csv(args.rebuild, dtype=str)
        for s, l in zip(rb["_sample"], rb.get("lifestyle")):
            if isinstance(l, str) and l in LIFESTYLES:
                life[s] = l

    # ---- collapse to samples and score every criterion ----------------------
    rows = []
    for sample, g in runs.groupby("_sample"):
        keys = {sample}
        keys |= {x for x in g["run_accession"].dropna()}
        keys |= {x for x in g["secondary_sample_accession"].dropna()}
        if keys & in_m1:
            continue                                   # already curated in
        text = " ".join(str(v) for v in
                        list(g["sample_title"].dropna()) +
                        list(g.get("description", pd.Series(dtype=str)).dropna()))
        paired = (g["library_layout"].fillna("").str.upper() == "PAIRED")
        wgs = (g["library_strategy"].fillna("").str.upper() == "WGS")
        qual = paired & wgs
        rows.append({
            "sample": sample,
            "study": g["study_accession"].dropna().iloc[0] if g["study_accession"].notna().any() else "",
            "runs": ";".join(sorted(g["run_accession"].dropna())),
            "n_runs": len(g),
            "already_processed": bool(keys & processed),
            "is_secondary_run_of_m1": bool(keys & secondary),
            "layout": "PAIRED" if paired.any() else "SINGLE (unconfirmed)",
            "strategy": ";".join(sorted(set(g["library_strategy"].dropna()))),
            "site": body_site_verdict(*g["host_body_site"].tolist(),
                                      *g["isolation_source"].tolist(),
                                      *g["sample_title"].tolist()),
            "looks_like_isolate": bool(ISOLATE.search(text)),
            "bases_qualifying": float(g.loc[qual, "_bases"].sum()),
            "bases_all_runs": float(g["_bases"].sum()),
            "country": resolve(g["country"].dropna().iloc[0]) if g["country"].notna().any() else None,
            "lifestyle": life.get(sample),
            "title": (g["sample_title"].dropna().iloc[0][:90]
                      if g["sample_title"].notna().any() else ""),
        })
    cand = pd.DataFrame(rows)
    print(f"\n  samples in these studies but absent from M1: {len(cand):,}")

    # ---- the cascade, reported not applied ---------------------------------
    cand["fail_processed"] = cand.already_processed
    cand["fail_isolate"] = cand.looks_like_isolate
    cand["fail_site"] = cand.site.eq("other")
    cand["fail_wgs"] = ~cand.strategy.str.contains("WGS", na=False)
    cand["fail_depth"] = cand.bases_qualifying < MIN_BASES
    cand["fail_lifestyle"] = cand.lifestyle.isna()
    fails = ["fail_processed", "fail_isolate", "fail_site", "fail_wgs",
             "fail_depth", "fail_lifestyle"]
    cand["n_fail"] = cand[fails].sum(axis=1)

    print(f"\n  {'criterion':<26}{'fails':>8}")
    for f in fails:
        print(f"  {f.replace('fail_',''):<26}{int(cand[f].sum()):>8,}")
    clean = cand[cand.n_fail == 0]
    print(f"\n  CANDIDATES passing every criterion : {len(clean):,}")
    print(f"  failing on depth alone             : "
          f"{int(((cand.n_fail == 1) & cand.fail_depth).sum()):,}")

    if len(clean):
        print(f"\n  {'study':<14}{'n':>6}  {'lifestyle':<14}{'median Gbp':>11}  country")
        for (st, ls), g in clean.groupby(["study", "lifestyle"]):
            print(f"  {st:<14}{len(g):>6}  {ls:<14}"
                  f"{g.bases_qualifying.median()/1e9:>11.2f}  "
                  f"{g.country.dropna().iloc[0] if g.country.notna().any() else '?'}")
    cand.to_csv(args.out, index=False)
    print(f"\n  wrote {args.out} ({len(cand):,} rows, all criteria flagged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
