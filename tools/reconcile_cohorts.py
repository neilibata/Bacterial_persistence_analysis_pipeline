#!/usr/bin/env python3
"""Reconcile the curated metadata (M1) against each reference's breadth matrix.

Produces the exact ID lists needed to close two gaps found before submission:

1. **Coverage gaps** — M1 samples with no row in a reference's matrix. Leviatan
   is missing 398 of them, skewed toward agro-pastoralists and concentrated in a
   few studies, so the Carter-vs-Leviatan reference comparison is not run on
   identical cohorts.

2. **Orphan matrix rows** — matrix rows with no M1 entry. Harmless (they are
   filtered out on load) but they explain why matrix row counts exceed the cohort.

3. **ID collisions in M1** — ``Chosen_match_ID`` values used by more than one
   row. These are *not* duplicate records: all 21 differ across ~65 fields, so
   they are distinct samples sharing a key. ``loading.py`` resolves them with
   ``drop_duplicates(keep="first")``, which silently keeps one arbitrarily —
   they need resolving upstream, not filtering away.

    python tools/reconcile_cohorts.py
    python tools/reconcile_cohorts.py --refs UHGG lev

Writes per-reference ID lists and a summary under
``results/cohort_reconciliation/``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402

OUT = ROOT / "results" / "cohort_reconciliation"
CLASSIFIED = ("H-G", "Transitional", "Industrial")
CONTEXT = ["Author", "Year", "Country", "Lifestyle", "StudyFolder",
           "SRA_Accession", "BioSample", "Total bases"]


def matrix_ids(ref: str) -> set[str]:
    """Sample IDs in a reference's breadth matrix (index column only)."""
    path = ROOT / config.REFERENCES[ref]["breadth_file"]
    if not path.exists():
        return set()
    ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[0], chunksize=200_000, dtype=str):
        ids.update(chunk.iloc[:, 0].astype(str).str.strip())
    ids.discard("")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1", type=Path, default=ROOT / "metadata_M1.csv")
    ap.add_argument("--refs", nargs="+", default=["UHGG", "lev", "combined"])
    args = ap.parse_args()

    if not args.m1.exists():
        sys.exit(f"{args.m1} not found")
    OUT.mkdir(parents=True, exist_ok=True)

    m1 = pd.read_csv(args.m1, low_memory=False, dtype=str)
    key = m1["Chosen_match_ID"].astype(str).str.strip()
    m1 = m1.assign(_key=key)
    lab = m1["Lifestyle"].astype(str).str.strip()
    classified = m1[lab.isin(CLASSIFIED)]
    cls_ids = set(classified["_key"])
    cols = ["_key"] + [c for c in CONTEXT if c in m1.columns]

    # ── 3. ID collisions ─────────────────────────────────────────────────────
    dup_mask = key.duplicated(keep=False)
    collisions = m1.loc[dup_mask, cols].sort_values("_key")
    collisions.to_csv(OUT / "m1_id_collisions.csv", index=False)
    n_coll = collisions["_key"].nunique()

    print(f"M1: {len(m1):,} rows, {key.nunique():,} unique IDs, "
          f"{len(cls_ids):,} classified\n")
    if n_coll:
        print(f"  ID COLLISIONS: {n_coll} IDs used by {len(collisions)} rows "
              f"-> m1_id_collisions.csv")
        print(f"    these are distinct samples sharing a key; loading.py keeps the "
              f"first arbitrarily\n")

    # ── 1 & 2. per-reference coverage, both directions ───────────────────────
    summary = []
    per_ref_ids = {}
    for ref in args.refs:
        print(f"  scanning {ref} matrix ...", flush=True)
        mids = matrix_ids(ref)
        per_ref_ids[ref] = mids
        if not mids:
            print(f"    {ref}: matrix not found, skipped")
            continue

        missing = cls_ids - mids                       # in M1, not in matrix
        orphan = mids - set(key)                       # in matrix, not in M1

        miss_df = classified[classified["_key"].isin(missing)][cols]
        miss_df.to_csv(OUT / f"{ref}_in_m1_not_in_matrix.csv", index=False)
        pd.DataFrame({"sample_id": sorted(orphan)}).to_csv(
            OUT / f"{ref}_in_matrix_not_in_m1.csv", index=False)

        by_life = dict(miss_df["Lifestyle"].value_counts()) if len(miss_df) else {}
        by_study = dict(miss_df["Author"].value_counts().head(8)) if "Author" in miss_df else {}
        # Reprocessing worklist: resolve each missing sample to its run
        # accessions so the preprocessing pipeline can be pointed straight at
        # them without re-deriving the mapping.
        if len(miss_df):
            def _runs(row):
                out = []
                for c in ("SRA_Accession", "Paired SRR List", "Secondary_run_IDs"):
                    v = row.get(c)
                    if pd.notna(v):
                        out += re.findall(r"[SEDR]RR\d+", str(v))
                return sorted(dict.fromkeys(out))
            full = m1[m1["_key"].isin(missing)].drop_duplicates("_key")
            runs = full.apply(_runs, axis=1)
            work = full[[c for c in cols if c in full.columns]].copy()
            work["n_runs"] = runs.str.len().values
            work["run_accessions"] = runs.str.join(";").values
            work.to_csv(OUT / f"{ref}_missing_accessions.csv", index=False)
            flat = sorted({r for rs in runs for r in rs})
            (OUT / f"{ref}_missing_runs.txt").write_text("\n".join(flat) + "\n")
            print(f"      worklist: {len(flat):,} run accessions "
                  f"-> {ref}_missing_runs.txt")

        summary.append({"reference": ref, "matrix_ids": len(mids),
                        "m1_classified_covered": len(cls_ids & mids),
                        "m1_classified_missing": len(missing),
                        "orphan_matrix_rows": len(orphan)})
        print(f"    {ref}: covers {len(cls_ids & mids):,}/{len(cls_ids):,}   "
              f"missing {len(missing):,}   orphan rows {len(orphan):,}")
        if by_life:
            print(f"      missing by lifestyle: {by_life}")
            print(f"      missing by study    : {by_study}")

    common = cls_ids.intersection(*[per_ref_ids[r] for r in args.refs if per_ref_ids.get(r)]) \
        if any(per_ref_ids.values()) else set()
    pd.DataFrame({"sample_id": sorted(common)}).to_csv(
        OUT / "common_core_sample_ids.csv", index=False)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary.csv", index=False)
    print(f"\n  common core across {', '.join(args.refs)}: {len(common):,} samples "
          f"-> common_core_sample_ids.csv")
    print(f"  wrote {len(list(OUT.glob('*.csv')))} files to "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
