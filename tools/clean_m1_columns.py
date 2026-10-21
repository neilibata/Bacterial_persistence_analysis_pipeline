#!/usr/bin/env python3
"""Tidy M1's competing and inconsistent columns, so a third party reading the
file alone reaches the same cohort the pipeline does.

Every change here was chosen from a full audit (see docs/the session record *.md):

DROPPED
  Diseased, Diseased (all)      Health-status roll-ups that contradict the
                                authoritative `Unhealthy` on 2,037 and 1,816
                                samples. Unread by the analysis.
  Obesity (old)                 Superseded by `Obesity`; disagreed on part of
                                the 1,328-sample obesity disagreement.
  Summary totals, Which column? 0.1% populated -- scratch columns.

REBUILT
  Diseased (gut and metabolic)  Kept because the category is useful, but the
                                shipped column was wrong both ways: 858 samples
                                with a gut/metabolic flag were marked No, and 39
                                marked Yes carried only melanoma/metastases
                                flags. Now DERIVED from the flag classification
                                below, so it cannot drift from the flags again.

NORMALISED (values only; no row or column lost)
  Gender                        Female/female/Male/male -> Female/Male/unknown.
  Age group                     Elderly/Senior/senior -> Senior, and the rest
                                title-cased, so a groupby yields real strata.
  BMI                           Values outside config.BMI_PLAUSIBLE blanked
                                (a 0.0 and three others); the raw value is kept
                                in `BMI (raw)` so nothing is destroyed.

KEPT DELIBERATELY
  Obesity (metadata)            Only 26% populated, but it is the only record of
                                a study calling a participant obese where no BMI
                                exists. Not redundant with `Obesity`.
  Antibiotics                   1.1% populated and unusable as a covariate, but
                                retained on request rather than silently dropped.

    python tools/clean_m1_columns.py            # report only
    python tools/clean_m1_columns.py --apply
"""
from __future__ import annotations
import argparse, shutil, sys, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from persistence import config  # noqa: E402
sys.path.insert(0, str(ROOT / "tools"))
import _disease_categories as dcat  # noqa: E402

DROP = ["Diseased", "Diseased (all)", "Obesity (old)",
        "Summary totals", "Which column?"]

# Classification behind the rebuilt `Diseased (gut and metabolic)`. Stated
# explicitly so it can be argued with rather than inferred from behaviour.
GUT = {"IBD", "Crohn's", "Ulcerative_colitis", "CRC", "adenoma", "coeliac",
       "CDI", "acute_diarrhoea", "treatment_colitis", "irritable_bowel", "STEC",
       "infectiousgastroenteritis", "salmonellosis", "few_polyps",
       "carcinoma_surgery_history", "STH", "schistosoma", "Worm-Infected",
       "TKI_dependent_diarrhoea"}
METABOLIC = {"T1D", "T2D", "IGT", "metabolic_syndrome", "fatty_liver",
             "cirrhosis", "hypertension", "ACVD", "ascites", "hepatitis",
             "gestational_diabetes", "wilson"}
# Explicitly NOT gut or metabolic, listed so the exclusion is reviewable:
# BD (bone disease), MS, PD, RA, ankylosing_spondylitis, schizofrenia, ME_CFS,
# myalgic_encephalomyelitis, melanoma*, metastases*, konzo, CMV, MDRB.

GENDER = {"female": "Female", "male": "Male", "f": "Female", "m": "Male",
          "unknown": "unknown"}
AGEGROUP = {"adult": "Adult", "child": "Child", "schoolage": "Schoolage",
            "teenager": "Teenager", "senior": "Senior", "elderly": "Senior",
            "newborn": "Newborn", "unknown": "Unknown"}


def is_yes(s):
    return s.astype(str).str.strip().str.lower().isin(
        {"1", "1.0", "y", "yes", "true", "t"})


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    m = pd.read_csv(config.META_FILE, low_memory=False)
    out = m.copy()
    print(f"M1: {len(m):,} rows x {m.shape[1]} cols\n")

    print("── dropping ──")
    for c in DROP:
        if c in out.columns:
            print(f"  {c:<32} ({100*m[c].notna().mean():.1f}% populated)")
            out = out.drop(columns=[c])
        else:
            print(f"  {c:<32} already absent")

    print("\n── deriving the three Diseased* roll-ups ──")
    out = dcat.apply_rollups(out, config.OBESITY_COLUMNS)
    print(f"  classification: {len(dcat.GUT)} gut + {len(dcat.METABOLIC)} metabolic, "
          f"rest -> other (see tools/_disease_categories.py)")

    print("\n── normalising values ──")
    if "Gender" in out.columns:
        g = out["Gender"].astype(str).str.strip().str.lower()
        before = out["Gender"].dropna().astype(str).str.strip().nunique()
        out["Gender"] = g.map(GENDER).where(g.isin(GENDER), out["Gender"])
        print(f"  Gender     {before} distinct -> "
              f"{out['Gender'].dropna().astype(str).str.strip().nunique()}  "
              f"{sorted(out['Gender'].dropna().astype(str).unique())}")
    if "Age group" in out.columns:
        g = out["Age group"].astype(str).str.strip().str.lower()
        before = out["Age group"].dropna().astype(str).str.strip().nunique()
        out["Age group"] = g.map(AGEGROUP).where(g.isin(AGEGROUP), out["Age group"])
        print(f"  Age group  {before} distinct -> "
              f"{out['Age group'].dropna().astype(str).str.strip().nunique()}  "
              f"{sorted(out['Age group'].dropna().astype(str).unique())}")
    if "BMI" in out.columns:
        raw = pd.to_numeric(out["BMI"], errors="coerce")
        lo, hi = config.BMI_PLAUSIBLE
        bad = raw.notna() & ((raw < lo) | (raw > hi))
        vals = sorted(round(float(v), 1) for v in raw[bad].dropna())
        # Never clobber an existing BMI (raw): on a re-run `BMI` is already
        # blanked, so copying it would destroy the originals this exists to keep.
        if "BMI (raw)" not in out.columns:
            out["BMI (raw)"] = out["BMI"]
        else:
            print(f"  BMI (raw)  already present -- preserved, not overwritten")
        out.loc[bad, "BMI"] = pd.NA
        print(f"  BMI        blanked {int(bad.sum())} implausible value(s) "
              f"outside [{lo}, {hi}]: {vals}")
        print(f"             originals preserved in 'BMI (raw)'")

    print(f"\n── result ──")
    print(f"  cols {m.shape[1]} -> {out.shape[1]}")
    print(f"  rows {len(m):,} -> {len(out):,}")
    if a.apply:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_column_clean")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"  backed up -> {bak}")
        out.to_csv(src, index=False); print(f"  applied to {src}")
    else:
        print("  (report only; rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
