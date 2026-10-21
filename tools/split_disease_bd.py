#!/usr/bin/env python3
"""Split `disease_BD`, which conflates two unrelated diseases.

The flag carries 74 positives spanning two cohorts that have nothing to do with
each other:

  RettedalAE_2021_NZL  54  PMID 33778322 -- "The Gut Microbiome Is Altered in
                           Postmenopausal Women With Osteoporosis and Osteopenia"
                           (JBMR Plus 2021). This count matches da Silva's
                           "Bone disease" column exactly, which is its source.
  YeZ_2018_CHN         20  PMID 30077182 / PRJNA431482 -- "A metagenomic study of
                           the gut microbiome in Behcet's disease" (Microbiome
                           2018). All 20 sit in that subset; the one sample from
                           the same folder's VKH study (PMID 31928124) has none.

The codebase held three different readings of the flag at once:
build_verified_m1.py mapped da Silva's "Bone disease" onto it, the metadata
explorer labelled it "Behcet's disease", and the data was both. Any analysis
grouping on it was pooling osteoporosis with an autoimmune vasculitis.

Splits into `disease_bone_disease` and `disease_Behcets`, assigned by study, and
drops `disease_BD`. Neither is in DISEASE_NCD_MAP, so no plotted contrast moves;
both land in the "other" roll-up category exactly as BD did.

    python tools/split_disease_bd.py            # report only
    python tools/split_disease_bd.py --apply
"""
from __future__ import annotations
import argparse, shutil, sys, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools"))
from persistence import config  # noqa: E402
import _disease_categories as dcat  # noqa: E402

ASSIGN = {  # StudyFolder -> new flag column
    "RettedalAE_2021_NZL": "disease_bone_disease",
    "YeZ_2018_CHN": "disease_Behcets",
}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    m = pd.read_csv(config.META_FILE, low_memory=False)
    if "disease_BD" not in m.columns:
        print("disease_BD already absent -- nothing to do"); return 0
    out = m.copy()
    bd = dcat.is_yes(out["disease_BD"])
    print(f"disease_BD positives: {int(bd.sum())}")
    print(f"  by study: {out.loc[bd,'StudyFolder'].value_counts().to_dict()}")

    for col in set(ASSIGN.values()):
        out[col] = "No"
    assigned = 0
    for sf, col in ASSIGN.items():
        sel = bd & out["StudyFolder"].astype(str).eq(sf)
        out.loc[sel, col] = "Yes"
        assigned += int(sel.sum())
        print(f"  {sf:<24} -> {col:<24}{int(sel.sum()):>4}")
    orphan = bd & ~out["StudyFolder"].astype(str).isin(ASSIGN)
    if orphan.any():
        print(f"  UNASSIGNED (study not in the map): {int(orphan.sum())}"
              f"  {out.loc[orphan,'StudyFolder'].value_counts().to_dict()}")
        print("  -> add them to ASSIGN before applying; they would be lost")
        if a.apply:
            raise SystemExit("refusing to apply with unassigned positives")
    out = out.drop(columns=["disease_BD"])
    print(f"\n  assigned {assigned} of {int(bd.sum())}; dropped disease_BD")

    # both new flags are "other" category, as BD was -- refresh to be sure
    print("\n── refreshing the derived roll-ups ──")
    out = dcat.apply_rollups(out, config.OBESITY_COLUMNS)
    print(f"\n  cols {m.shape[1]} -> {out.shape[1]}  rows {len(out):,}")

    if a.apply:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_bd_split")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"  backed up -> {bak}")
        out.to_csv(src, index=False); print(f"  applied to {src}")
    else:
        print("  (report only; rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
