#!/usr/bin/env python3
"""Reconcile the disease flags with the studies' own free-text disease field.

Two corrections, both audited per study before being written.

ADD -- the free-text field names a condition that no `disease_*` column
records. Each value occurs in exactly one study, which is what a per-study
ingestion gap looks like rather than scattered noise, and every value maps to a
column that already exists. The field is semicolon-separated, so a row can
name more than one condition.

DROP -- a `disease_*` flag that contradicts everything else known about its
study. In each case below the flag is the *only* disease flag anywhere in that
study, the free-text field is blank or says healthy, and the condition does not
fit what the study is about. The same flag is kept wherever it is genuine:
`disease_melanoma` stays on the LeeKA_2022 melanoma cohorts (164 samples, all
already unhealthy) and `disease_CDI` stays on IaniroG_2022_ITA, an
FMT-for-C.-difficile trial.

Dropping a flag does not by itself relabel a sample: `Unhealthy` is curated,
not derived. What it changes is the disease roll-ups, the per-condition
analyses, and what a declarative rule would conclude. Any sample left unhealthy
with no remaining support is reported.

    python tools/curate_disease_flags.py --dry-run
    python tools/curate_disease_flags.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_IO = ROOT / "final_run" / "metadata_M1.csv"
MB = "disease [MBIOTEMP_disease]"

#: free-text token -> the column that records it
ADD_MAP = {
    "ibd": "disease_IBD",
    "sth": "disease_STH",
    "igt": "disease_IGT",
    "crc": "disease_CRC",
    "t2d": "disease_T2D",
    "fatty_liver": "disease_fatty_liver",
    "hypertension": "disease_hypertension",
}

#: (column, study) pairs to clear, with the reason recorded in the changelog
DROP = [
    ("disease_melanoma", "LeChatelierE_2013_DNK",
     "obesity/gene-richness cohort; melanoma is the only disease flag in the "
     "study and the free-text field is blank"),
    ("disease_melanoma", "Macdonaldo-GomezMX_2016_USA",
     "melanoma is the only disease flag in the study, every flagged sample is "
     "labelled healthy, and the free-text field is blank"),
    ("disease_melanoma", "PehrssonEC_2016",
     "antibiotic-resistance/habitat study; free-text field says healthy for "
     "all 178 samples"),
    ("disease_melanoma_surgery", "PehrssonEC_2016",
     "same misassignment as disease_melanoma in this study"),
    ("disease_CDI", "HMP_2012_USA",
     "screened-healthy reference cohort; CDI is the only disease flag and "
     "every flagged sample is labelled healthy"),
]

NOISY = {
    "disease_metastases", "disease_metastases_nodes", "disease_metastases_lung",
    "disease_metastases_liver", "disease_metastases_bone", "disease_metastases_SQ",
    "disease_metastases_adrenal", "disease_melanoma_surgery",
    "disease_carcinoma_surgery_history", "disease_treatment_colitis",
}


def tokens(val):
    s = str(val).strip().lower()
    if s in ("", "nan", "healthy"):
        return []
    return [t.strip() for t in s.split(";") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default=str(DEFAULT_IO))
    ap.add_argument("--changelog",
                    default=str(ROOT / "final_run" / "disease_flag_curation.csv"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from persistence.common import is_yes
    from persistence import config

    m = pd.read_csv(a.io, low_memory=False)
    print(f"  {Path(a.io).name}: {len(m):,} rows")
    u = m["Unhealthy"].astype(str).str.strip()
    rows = []

    # ── ADD ───────────────────────────────────────────────────────────────
    dz = [c for c in m.columns if c.startswith("disease_") and c not in NOISY]
    anydz = pd.concat([m[c].apply(is_yes) for c in dz], axis=1).any(axis=1)
    obc = [c for c in config.OBESITY_COLUMNS if c in m.columns]
    ob = pd.concat([m[c].apply(is_yes) for c in obc], axis=1).any(axis=1)
    if "BMI" in m.columns:
        ob = ob | (pd.to_numeric(m["BMI"], errors="coerce") >= 30).fillna(False)

    target = (u == "Healthy") & ~anydz & ~ob & m[MB].map(lambda v: bool(tokens(v)))
    print(f"\n  ADD: {int(target.sum())} samples name a condition in the "
          "free-text field with no flag set")
    added = {}
    unknown = {}
    for i in m.index[target]:
        for tok in tokens(m.at[i, MB]):
            col = ADD_MAP.get(tok)
            if col is None or col not in m.columns:
                unknown[tok] = unknown.get(tok, 0) + 1
                continue
            if not is_yes(m.at[i, col]):
                m.at[i, col] = "Yes"
                added[col] = added.get(col, 0) + 1
                rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                                 StudyFolder=m.at[i, "StudyFolder"],
                                 column=col, old="", new="Yes",
                                 action="add",
                                 reason=f"free-text disease field says {tok!r}"))
        if u.at[i] != "Unhealthy":
            rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                             StudyFolder=m.at[i, "StudyFolder"],
                             column="Unhealthy", old=m.at[i, "Unhealthy"],
                             new="Unhealthy", action="add",
                             reason="a condition is now recorded for this sample"))
            m.at[i, "Unhealthy"] = "Unhealthy"
    for col, n in sorted(added.items(), key=lambda kv: -kv[1]):
        print(f"      {col:32} +{n}")
    if unknown:
        print(f"      *** unmapped free-text tokens, nothing set: {unknown}")

    # ── DROP ──────────────────────────────────────────────────────────────
    print("\n  DROP: flags that contradict their study")
    for col, study, why in DROP:
        if col not in m.columns:
            print(f"      {col} absent; skipped")
            continue
        sel = (m["StudyFolder"] == study) & m[col].apply(is_yes)
        n = int(sel.sum())
        if not n:
            print(f"      {col} / {study}: nothing set; skipped")
            continue
        for i in m.index[sel]:
            rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                             StudyFolder=study, column=col,
                             old=m.at[i, col], new="", action="drop",
                             reason=why))
        m.loc[sel, col] = ""
        print(f"      {col:32} -{n:<4} {study}")

    # ── does anything end up unhealthy with no support left? ─────────────
    dz2 = [c for c in m.columns if c.startswith("disease_") and c not in NOISY]
    any2 = pd.concat([m[c].apply(is_yes) for c in dz2], axis=1).any(axis=1)
    u2 = m["Unhealthy"].astype(str).str.strip()
    named2 = m[MB].map(lambda v: bool(tokens(v)))
    unsupported = (u2 == "Unhealthy") & ~any2 & ~ob & ~named2
    print(f"\n  samples left Unhealthy with no disease flag, obesity or "
          f"free-text condition: {int(unsupported.sum()):,}")
    if int(unsupported.sum()):
        print("      by study: "
              + str(dict(list(m.loc[unsupported, "StudyFolder"]
                              .value_counts().items())[:5])))

    print(f"\n  Unhealthy {int((u2 == 'Unhealthy').sum()):,}   "
          f"Healthy {int((u2 == 'Healthy').sum()):,}   "
          f"(was {int((u == 'Unhealthy').sum()):,} / "
          f"{int((u == 'Healthy').sum()):,})")

    if a.dry_run:
        print("\n  dry run; nothing written")
        return
    m.to_csv(a.io, index=False)
    pd.DataFrame(rows).to_csv(a.changelog, index=False)
    print(f"\n  wrote {Path(a.io).relative_to(ROOT)}")
    print(f"  wrote {Path(a.changelog).relative_to(ROOT)}  "
          f"({len(rows):,} cell changes)")


if __name__ == "__main__":
    main()
