#!/usr/bin/env python3
"""Build Metadata Table M1: the one curated per-sample metadata file.

M1 is what a reviewer is handed and what the pipeline reads. Those have not
been the same object: metadata_M1.csv was built from the March cohort and never
received the August health curation, while the pipeline read a chain of patched
files. This collapses the chain into one file.

What it does, in order:

  1. Starts from the curated cohort (the August corrections, three isolate
     genomes removed) -- not from a patched copy of it.
  2. Applies the obesity rule: a sample flagged obese is unhealthy. Obesity is
     a disease state in its own right, and leaving it as health put obese
     participants in the control group of every other condition tested.
     Overweight alone is *not* unhealthy; an overweight participant with a
     disease is unhealthy through the disease.
  3. Carries read length and read count across from the previous M1. Read
     length is a per-sequencing-run property and near-constant within a study,
     so a sample the old M1 never saw inherits its study's value and has
     ``number_reads`` computed from it. No network access.
  4. Drops the unnamed columns a CSV round-trip left behind.

What it deliberately does NOT do is re-derive the health label from scratch.
A declarative rule would move 652 further samples, and those are curation
decisions rather than bugs -- see the report it prints. Nothing is changed
without being asked for.

    python tools/build_curated_m1.py --dry-run
    python tools/build_curated_m1.py --out final_run/metadata_M1.csv
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: the curated cohort this is built from
BASE = ROOT / "metadata_Aug_26_2026_final.csv"
#: the previous M1, used only as the source of read length / read count
PREV_M1 = ROOT / "metadata_M1.csv"

KEY = "Chosen_match_ID"
STUDY = "StudyFolder"
NOISY = {
    "disease_metastases", "disease_metastases_nodes", "disease_metastases_lung",
    "disease_metastases_liver", "disease_metastases_bone", "disease_metastases_SQ",
    "disease_metastases_adrenal", "disease_melanoma_surgery",
    "disease_carcinoma_surgery_history", "disease_treatment_colitis",
}
MB = "disease [MBIOTEMP_disease]"


def obesity_flag(m):
    """The pipeline's own obesity flag: an obesity column, or BMI >= 30."""
    from persistence import config
    from persistence.common import is_yes
    cols = [c for c in getattr(config, "OBESITY_COLUMNS", []) if c in m.columns]
    flag = (pd.concat([m[c].apply(is_yes) for c in cols], axis=1).any(axis=1)
            if cols else pd.Series(False, index=m.index))
    by_col = flag.copy()
    if "BMI" in m.columns:
        flag = flag | (pd.to_numeric(m["BMI"], errors="coerce") >= 30).fillna(False)
    return flag, by_col, cols


def unreviewed_conflicts(m):
    """Samples a declarative rule would move, which nobody has ruled on.

    Reported, never applied. The existing curation required a disease flag and
    the free-text field to agree before calling a sample unhealthy; a rule that
    accepts either signal on its own disagrees with that in two directions, and
    which is right is a judgement about the source studies, not about code.
    """
    from persistence.common import is_yes
    dz = [c for c in m.columns if c.startswith("disease_") and c not in NOISY]
    anydz = pd.concat([m[c].apply(is_yes) for c in dz], axis=1).any(axis=1)
    mb = m[MB].astype(str).str.strip().str.lower() if MB in m.columns else None
    named = (~mb.isin(["healthy", "nan", ""])) if mb is not None else pd.Series(
        False, index=m.index)
    healthy = m["Unhealthy"].astype(str).str.strip() == "Healthy"
    ob, _bc, _c = obesity_flag(m)
    return {
        "a disease flag is set, but the study's own free-text field says "
        "healthy or is blank": int((healthy & anydz & ~ob).sum()),
        "the free-text field names a condition, but no disease flag is set":
            int((healthy & named & ~anydz & ~ob).sum()),
    }


def carry_read_length(new, prev):
    """Read length and read count, from the previous M1, per study."""
    out = new.copy()
    p = prev.set_index(KEY)
    p = p[~p.index.duplicated()]
    idx = out[KEY].astype(str)

    for col in ("median_read_length", "number_reads"):
        if col in p.columns:
            carried = idx.map(p[col])
            out[col] = carried.where(carried.notna(), out.get(col))

    src = pd.Series("original", index=out.index)
    src[out["median_read_length"].isna()] = "missing"
    # Per study: read length is near-constant within one, which is the whole
    # basis of the original imputation, so a sample the old M1 never saw takes
    # its study's value rather than a network lookup.
    if STUDY in out.columns and STUDY in prev.columns:
        by_study = (prev.groupby(STUDY)["median_read_length"]
                    .median().dropna())
        need = out["median_read_length"].isna()
        filled = out.loc[need, STUDY].map(by_study)
        out.loc[need, "median_read_length"] = filled
        src[need & out["median_read_length"].notna()] = "study_median"

    # number_reads follows from total bases and read length.
    if "Total bases" in out.columns:
        need_n = out["number_reads"].isna() & out["median_read_length"].notna()
        tb = pd.to_numeric(out.loc[need_n, "Total bases"], errors="coerce")
        rl = pd.to_numeric(out.loc[need_n, "median_read_length"], errors="coerce")
        out.loc[need_n, "number_reads"] = (tb / rl).round()

    out["median_read_length_source"] = src
    nsrc = pd.Series("original", index=out.index)
    nsrc[out["number_reads"].isna()] = "missing"
    if "Total bases" in out.columns:
        nsrc[(src == "study_median")] = "computed"
    out["number_reads_source"] = nsrc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--prev-m1", default=str(PREV_M1))
    ap.add_argument("--out", default=str(ROOT / "final_run" / "metadata_M1.csv"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = pd.read_csv(a.base, low_memory=False)
    print(f"  base cohort      : {Path(a.base).name}  {len(base):,} rows, "
          f"{len(base.columns)} cols")

    junk = [c for c in base.columns if c.startswith("Unnamed")]
    if junk:
        base = base.drop(columns=junk)
        print(f"  dropped {len(junk)} unnamed column(s)")

    ob, by_col, obcols = obesity_flag(base)
    before = base["Unhealthy"].astype(str).str.strip()
    target = ob & (before == "Healthy")
    print(f"  obesity columns  : {obcols}")
    print(f"  obese samples    : {int(ob.sum()):,}")
    print(f"  obese AND labelled Healthy -> Unhealthy: {int(target.sum()):,} "
          f"({int((target & by_col).sum()):,} by an obesity column, "
          f"{int((target & ~by_col).sum()):,} by BMI >= 30 alone)")
    base.loc[target, "Unhealthy"] = "Unhealthy"

    prev = pd.read_csv(a.prev_m1, low_memory=False)
    out = carry_read_length(base, prev)
    for col in ("median_read_length", "number_reads"):
        print(f"  {col:20} filled {int(out[col].notna().sum()):,}/{len(out):,}")
    print("  read-length source: "
          + ", ".join(f"{k}={v:,}" for k, v in
                      out["median_read_length_source"].value_counts().items()))

    u = out["Unhealthy"].astype(str).str.strip()
    print(f"\n  Unhealthy {int((u == 'Unhealthy').sum()):,}   "
          f"Healthy {int((u == 'Healthy').sum()):,}   "
          f"unlabelled {int((~u.isin(['Healthy', 'Unhealthy'])).sum()):,}")

    print("\n  NOT applied -- curation decisions still open:")
    for why, n in unreviewed_conflicts(
            pd.read_csv(a.base, low_memory=False)).items():
        print(f"    {n:>5,}  {why}")

    if a.dry_run:
        print("\n  dry run; nothing written")
        return
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(f"\n  wrote {p.relative_to(ROOT)}  "
          f"({len(out):,} rows, {len(out.columns)} cols)")
    stamp = p.with_name("metadata_M1_provenance.txt")
    stamp.write_text(
        "Metadata Table M1\n"
        f"built {datetime.datetime.now():%Y-%m-%d %H:%M} by "
        "tools/build_curated_m1.py\n\n"
        f"base cohort      : {Path(a.base).name}\n"
        f"read length from : {Path(a.prev_m1).name} (per sample, else per study)\n"
        f"obesity rule     : a sample flagged obese is Unhealthy "
        f"({int(target.sum())} relabelled). Overweight alone is not.\n"
        f"rows             : {len(out):,}\n"
        f"Unhealthy        : {int((u == 'Unhealthy').sum()):,}\n"
        f"Healthy          : {int((u == 'Healthy').sum()):,}\n")
    print(f"  wrote {stamp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
