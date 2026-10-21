#!/usr/bin/env python3
"""Write a corrected metadata file: Unhealthy mislabels and disease categories.

Never edits the input. Writes a new dated metadata CSV plus a per-cell changelog,
and leaves config.META_FILE_CANDIDATES alone -- adopting the new file is a
separate, deliberate step, because it moves samples between the arms of the
Section 3 contrast.

Two corrections, from different evidence:

1. Unhealthy mislabels (affects analysis)
   A sample carrying a substantive disease_* flag whose `disease
   [MBIOTEMP_disease]` field independently names a disease, but whose `Unhealthy`
   column reads Healthy. Two sources against one; Unhealthy is the outlier.
   Samples where MBIOTEMP instead agrees with Unhealthy are left alone -- there
   the flag is the outlier, which is what a study-level annotation applied to a
   study's own controls looks like.

2. Disease category assignment (does not affect analysis)
   disease_acute_diarrhoea -> gut and metabolic, and -> other
   disease_IBD             -> gut and metabolic
   disease_ACVD            -> other
   applied only where the flag is affirmatively yes, never on No or blank.
   `Diseased (all)` is then recomputed as the union of the two categories.

    python tools/apply_health_label_corrections.py                 # dry run
    python tools/apply_health_label_corrections.py --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "metadata_Aug_20_2026.csv"

G = "Diseased (gut and metabolic)"
O = "Diseased (other)"
A = "Diseased (all)"
MB = "disease [MBIOTEMP_disease]"

# Flags that describe a procedure or a metastatic site rather than a primary
# condition; they should not by themselves make a sample Unhealthy.
NOISY = {
    "disease_metastases", "disease_metastases_nodes", "disease_metastases_lung",
    "disease_metastases_liver", "disease_metastases_bone", "disease_metastases_SQ",
    "disease_metastases_adrenal", "disease_melanoma_surgery",
    "disease_carcinoma_surgery_history", "disease_treatment_colitis",
}

CATEGORY_RULES = [
    ("disease_acute_diarrhoea", G),
    ("disease_IBD", G),
    ("disease_acute_diarrhoea", O),
    ("disease_ACVD", O),
]


def _yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "yes", "true", "y"])


def _pos(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["yes", "1", "1.0"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=ROOT / "metadata_Aug_26_2026.csv")
    ap.add_argument("--changelog", type=Path, default=ROOT / "docs" / "health_label_corrections.csv")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.src, low_memory=False)
    orig = m.copy()
    sub_d = [c for c in m.columns if c.startswith("disease_") and c not in NOISY]
    flag = pd.concat([_yes(m[c]) for c in sub_d], axis=1).any(axis=1)
    unh = m["Unhealthy"].astype(str).str.strip().str.lower()
    mb = m[MB].astype(str).str.strip().str.lower()

    # ── 1. Unhealthy mislabels ───────────────────────────────────────────
    mislabel = flag & ~mb.isin(["healthy", "nan", ""]) & unh.eq("healthy")
    print(f"1. Unhealthy mislabels (flag + MBIOTEMP disease, Unhealthy=Healthy): {int(mislabel.sum())}")
    for (a, y), n in m[mislabel].groupby(["Author", "Year"]).size().sort_values(ascending=False).items():
        print(f"     {a} {y}: {n}")

    # ── 2. Category assignment ───────────────────────────────────────────
    cat_changes = {}
    for col, target in CATEGORY_RULES:
        add = _yes(m[col]) & ~_pos(m[target])
        cat_changes[(col, target)] = int(add.sum())
        print(f"2. {col:26s} -> {target:30s} {int(add.sum()):4d} to set")

    if not args.apply:
        print("\n  dry run — nothing written. Re-run with --apply.")
        return

    rows = []
    for i in m.index[mislabel]:
        rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"], column="Unhealthy",
                         old=m.at[i, "Unhealthy"], new="Unhealthy",
                         reason=f"MBIOTEMP={m.at[i, MB]!r} and a disease flag both indicate disease",
                         affects_analysis="yes"))
    m.loc[mislabel, "Unhealthy"] = "Unhealthy"

    for col, target in CATEGORY_RULES:
        add = _yes(m[col]) & ~_pos(m[target])
        for i in m.index[add]:
            rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"], column=target,
                             old=m.at[i, target], new="Yes",
                             reason=f"{col} is yes", affects_analysis="no"))
        m.loc[add, target] = "Yes"

    # roll-up recomputed, never hand-set
    newA = _pos(m[G]) | _pos(m[O])
    bump = newA & ~_pos(m[A])
    for i in m.index[bump]:
        rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"], column=A,
                         old=m.at[i, A], new="Yes",
                         reason="union of the two category fields", affects_analysis="no"))
    m.loc[bump, A] = "Yes"

    m.to_csv(args.out, index=False)
    cl = pd.DataFrame(rows)
    cl.to_csv(args.changelog, index=False)

    print(f"\n  wrote {args.out.name}  ({len(m):,} rows, unchanged row count: {len(m) == len(orig)})")
    print(f"  wrote {args.changelog.relative_to(ROOT)}  ({len(cl):,} cell changes)")
    print("\n  before -> after:")
    for c in ["Unhealthy", G, O, A]:
        b = int(_pos(orig[c]).sum()) if c != "Unhealthy" else int(orig[c].astype(str).str.strip().str.lower().eq("unhealthy").sum())
        a_ = int(_pos(m[c]).sum()) if c != "Unhealthy" else int(m[c].astype(str).str.strip().str.lower().eq("unhealthy").sum())
        print(f"    {c:32s} {b:6,d} -> {a_:6,d}  ({a_-b:+d})")
    print("\n  config.META_FILE_CANDIDATES is unchanged: the pipeline still reads the old file.")


if __name__ == "__main__":
    main()
