#!/usr/bin/env python3
"""Find processed samples whose run coverage looks incomplete.

Distinct from the "extras" question. These samples are already in M1 and already
in the coverage matrices; the concern is that only some of their sequencing runs
were fetched and merged, so their depth -- and therefore their detected richness
-- is understated relative to what the archive actually holds.

Two independent signals, because either alone is weak:

  Run-count gap. The archive lists more runs for the sample's BioSample than M1
  records in `SRA_Accession` + `Secondary_run_IDs`. On its own this over-fires:
  a BioSample legitimately carries 16S companion runs, and Table M1 deliberately
  screens technical replicates rather than merging them.

  Depth shortfall. M1's `Total bases` is materially below the sum of ENA's
  `base_count` over the sample's qualifying runs. This is the substantive test,
  and the diagnostic pattern is a clean sub-multiple: a sample recording almost
  exactly 1/2 or 1/3 of the archive total processed that fraction of its runs.
  SRR061226 was confirmed this way -- recorded depth x1.96 short, one of two runs.

Reported together with the implied ratio so a k/N pattern is visible, and never
filtered on one signal alone. A sample flagged only on run count is usually a 16S
companion; one flagged on both, at a near-integer ratio, is a real shortfall.

Direction matters for the manuscript: a shortfall depresses detection, so it can
only weaken a claim that non-industrial samples carry more taxa -- it cannot
manufacture one. That makes this a completeness issue rather than a correctness
one, and is why it is worth fixing but was not urgent.

    python tools/find_incomplete_samples.py --cache <ena_dir> --out incomplete.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A recorded depth this far below the archive total counts as a shortfall.
# 0.85 rather than 1.0 because base_count is pre-QC and M1's Total bases is
# post-trimming, so a few percent loss is expected and not evidence of anything.
DEPTH_RATIO_FLAG = 0.85
NEAR_INTEGER_TOL = 0.12          # how close to k/N counts as a clean sub-multiple


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("incomplete_samples.csv"))
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    runs = pd.concat([pd.read_csv(f, sep="\t", dtype=str)
                      for f in glob.glob(str(args.cache / "*.tsv"))],
                     ignore_index=True)
    runs["_bases"] = pd.to_numeric(runs["base_count"], errors="coerce").fillna(0)
    runs["_wgs"] = runs["library_strategy"].fillna("").str.upper() == "WGS"

    # Every run the archive attaches to each sample-level accession, as two flat
    # dicts rather than a dict of DataFrames: the latter builds ~60,000 frames
    # and stalls before emitting anything.
    wgs_runs = runs[runs["_wgs"]]
    long = pd.concat([
        wgs_runs[[c, "run_accession", "_bases"]].rename(columns={c: "_key"})
        for c in ("sample_accession", "secondary_sample_accession")
    ], ignore_index=True).dropna(subset=["_key"])
    long = long.drop_duplicates(["_key", "run_accession"])
    grp = long.groupby("_key")
    runs_by_sample = grp["run_accession"].apply(set).to_dict()
    bases_by_sample = grp["_bases"].sum().to_dict()

    rows = []
    for _, r in meta.iterrows():
        recorded = set()
        for col in ("SRA_Accession", "Secondary_run_IDs"):
            for x in str(r.get(col) or "").replace(",", ";").split(";"):
                if x.strip():
                    recorded.add(x.strip())
        # find the archive's view via any sample-level identifier we hold
        arch_runs, arch_bases = None, 0.0
        for col in ("BioSample", "SRS ID", "Chosen_match_ID", "SRA_Accession"):
            key = str(r.get(col) or "").strip()
            if key in runs_by_sample:
                arch_runs = runs_by_sample[key]
                arch_bases = float(bases_by_sample.get(key, 0.0))
                break
        if not arch_runs:
            continue
        m1_bases = pd.to_numeric(pd.Series([r.get("Total bases")]),
                                 errors="coerce").iloc[0]
        if not arch_bases or pd.isna(m1_bases) or m1_bases <= 0:
            continue
        ratio = m1_bases / arch_bases
        missing = arch_runs - recorded
        rows.append({
            "Chosen_match_ID": r.get("Chosen_match_ID"),
            "Author": r.get("Author"), "StudyFolder": r.get("StudyFolder"),
            "Lifestyle": r.get("Lifestyle"),
            "n_runs_archive": len(arch_runs),
            "n_runs_recorded": len(arch_runs & recorded),
            "runs_not_recorded": ";".join(sorted(missing)),
            "m1_bases": m1_bases, "archive_bases": arch_bases,
            "depth_ratio": ratio,
        })
    df = pd.DataFrame(rows)
    print(f"  M1 samples resolvable against the archive: {len(df):,}")

    df["flag_runcount"] = df.n_runs_archive > df.n_runs_recorded
    df["flag_depth"] = df.depth_ratio < DEPTH_RATIO_FLAG
    # is the shortfall a clean k/N?
    inv = 1.0 / df.depth_ratio.replace(0, pd.NA)
    df["implied_multiple"] = inv
    df["near_integer"] = (inv - inv.round()).abs() < NEAR_INTEGER_TOL
    df["both"] = df.flag_runcount & df.flag_depth

    print(f"\n  {'signal':<34}{'samples':>9}")
    print(f"  {'archive lists unrecorded runs':<34}{int(df.flag_runcount.sum()):>9,}")
    print(f"  {'recorded depth < 85% of archive':<34}{int(df.flag_depth.sum()):>9,}")
    print(f"  {'BOTH':<34}{int(df.both.sum()):>9,}")
    print(f"  {'BOTH, at a near-integer ratio':<34}"
          f"{int((df.both & df.near_integer).sum()):>9,}")

    strong = df[df.both & df.near_integer].sort_values("depth_ratio")
    if len(strong):
        print(f"\n  strongest candidates (depth shortfall at a clean sub-multiple):")
        print(f"  {'sample':<14}{'study':<26}{'runs':>7}{'ratio':>8}{'x short':>9}")
        for _, r in strong.head(20).iterrows():
            print(f"  {str(r.Chosen_match_ID):<14}{str(r.StudyFolder)[:25]:<26}"
                  f"{r.n_runs_recorded}/{r.n_runs_archive:<4}{r.depth_ratio:>8.2f}"
                  f"{r.implied_multiple:>9.2f}")
        print(f"\n  by lifestyle: {strong.Lifestyle.value_counts().to_dict()}")
        print(f"  by study    : {strong.StudyFolder.value_counts().head(6).to_dict()}")

    df.to_csv(args.out, index=False)
    print(f"\n  wrote {args.out} ({len(df):,} rows, all signals retained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
