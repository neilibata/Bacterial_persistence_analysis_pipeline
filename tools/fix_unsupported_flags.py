#!/usr/bin/env python3
"""Remove disease flags no source supports, and Unhealthy labels with no reason.

Five study/flag pairs carried a condition alien to the cohort and supported by
none of the three sources that can set a flag:

  ZhangX_2015_CHN              schizofrenia       27   (an RA cohort)
  Macdonaldo-GomezMX_2016_USA  metastases         39   (a probiotic engraftment study)
  RettedalAE_2021_NZL          STH                30   (an osteoporosis cohort)
  LeChatelierE_2013_DNK        treatment_colitis  15   (MetaHIT gene-richness)
  JacobsonDK_2021_BFA          ACVD               15   (a lifestyle/SCFA meta-analysis)

Evidence that no source sets them:
  * da Silva's table has no such COLUMN -- its conditions are Diabetes,
    Colorectal Cancer, Liver Cirrhosis, Obesity, Ankylosing spondylitis, Bone
    disease, Crohn's, Healthy, MS, Myalgic encephalomyelitis, Parkinson's, RA,
    Type 1 diabetes, Ulcerative colitis.
  * None of the five studies appears in the MicrobiomeDB download.
  * ENA/NCBI record no disease attribute for any flagged sample. RettedalAE is
    the clearest: the archive carries full bone densitometry (FEMORAL_NECK_BMD,
    SPINE_T-Score, Whole Body_% FAT) and no parasitology field whatsoever.
  * All five predate the source-based rebuild with identical counts, so they are
    inherited from an undocumented curation rather than derived.

Also corrected, per explicit instruction:
  * Overweight is NOT a reason to be Unhealthy and never has been. Samples whose
    only ground was `Overweight` are relabelled Healthy.
  * Samples labelled Unhealthy with no disease flag, no obesity and no BMI >= 30
    are relabelled Healthy.

Not touched, because the right answer is unclear rather than absent:
  * disease_BD conflates two conditions -- bone disease in RettedalAE_2021_NZL
    (54, matching da Silva's "Bone disease" count exactly) and something else in
    YeZ_2018_CHN (20), a Vogt-Koyanagi-Harada / Behcet's cohort. Splitting it
    needs a naming decision, so it is reported, not guessed at.

    python tools/fix_unsupported_flags.py          # report only
    python tools/fix_unsupported_flags.py --apply
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

UNSUPPORTED = [
    ("ZhangX_2015_CHN", "schizofrenia"),
    ("Macdonaldo-GomezMX_2016_USA", "metastases"),
    ("RettedalAE_2021_NZL", "STH"),
    ("LeChatelierE_2013_DNK", "treatment_colitis"),
    ("JacobsonDK_2021_BFA", "ACVD"),
]


def is_yes(s):
    return s.astype(str).str.strip().str.lower().isin(
        {"1", "1.0", "y", "yes", "true", "t"})


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    m = pd.read_csv(config.META_FILE, low_memory=False)
    out = m.copy()
    print(f"M1: {len(m):,} rows\n")

    print("── clearing unsupported flags ──")
    cleared = 0
    for sf, flag in UNSUPPORTED:
        col = f"disease_{flag}"
        if col not in out.columns:
            print(f"  {sf:<30}{flag:<20} column absent"); continue
        sel = out["StudyFolder"].astype(str).eq(sf) & is_yes(out[col])
        n = int(sel.sum()); cleared += n
        out.loc[sel, col] = "No"
        print(f"  {sf:<30}{flag:<20} cleared {n}")
        rest = int(is_yes(out[col]).sum())
        print(f"       {col} remaining elsewhere: {rest}")
    print(f"  total flag instances cleared: {cleared}")

    # recompute what Unhealthy can be justified by
    dcols = [c for c in out.columns if c.startswith("disease_")]
    anyflag = pd.concat([is_yes(out[c]) for c in dcols], axis=1).any(axis=1)
    ob = pd.concat([is_yes(out[c]) for c in config.OBESITY_COLUMNS
                    if c in out.columns], axis=1).any(axis=1)
    bmi = pd.to_numeric(out.get("BMI"), errors="coerce")
    obese = ob | (bmi >= 30)
    u = out["Unhealthy"].astype(str).str.strip()
    ow = is_yes(out["Overweight"]) if "Overweight" in out.columns else pd.Series(
        False, index=out.index)

    print("\n── Unhealthy labels with no remaining reason ──")
    no_reason = u.eq("Unhealthy") & ~anyflag & ~obese
    ow_only = no_reason & ow
    nothing = no_reason & ~ow
    print(f"  Unhealthy, only ground was Overweight : {int(ow_only.sum())}"
          f"   -> Healthy (Overweight is not a disease)")
    print(f"  Unhealthy, no ground at all           : {int(nothing.sum())}"
          f"   -> Healthy")
    for tag, sel in (("overweight-only", ow_only), ("no ground", nothing)):
        if sel.any():
            print(f"    [{tag}] by study: "
                  f"{out.loc[sel,'StudyFolder'].value_counts().head(6).to_dict()}")
    newly = int((no_reason).sum())
    out.loc[no_reason, "Unhealthy"] = "Healthy"

    # flags cleared can also strip a sample's only reason
    print("\n── after clearing flags, did any sample lose its only reason? ──")
    u2 = out["Unhealthy"].astype(str).str.strip()
    still = u2.eq("Unhealthy") & ~anyflag & ~obese
    print(f"  Unhealthy with no reason remaining: {int(still.sum())}  (must be 0)")

    # Clearing flags invalidates the derived roll-ups, so refresh all three
    # from the shared classification rather than leaving them stale.
    print("\n── refreshing the derived roll-ups ──")
    out = dcat.apply_rollups(out, config.OBESITY_COLUMNS)

    print("\n── result ──")
    v = out["Unhealthy"].astype(str).str.strip().value_counts()
    print(f"  Unhealthy: {v.get('Healthy',0):,} Healthy / "
          f"{v.get('Unhealthy',0):,} Unhealthy   (relabelled {newly})")
    tot = sum(int(is_yes(out[c]).sum()) for c in dcols)
    tot0 = sum(int(is_yes(m[c]).sum()) for c in dcols if c in m.columns)
    print(f"  positive disease flags: {tot0:,} -> {tot:,}")
    print(f"  rows {len(m):,} -> {len(out):,}   cols {m.shape[1]} -> {out.shape[1]}")

    if a.apply:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_flag_fix")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"  backed up -> {bak}")
        out.to_csv(src, index=False); print(f"  applied to {src}")
    else:
        print("  (report only; rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
