"""One classification of M1's disease_* flags, and the roll-ups derived from it.

Single source of truth, imported by clean_m1_columns.py and
fix_unsupported_flags.py. Before this existed the three `Diseased*` roll-ups were
hand-curated and drifted badly from the flags they claimed to summarise:
`Diseased (gut and metabolic)` marked 858 flagged samples No and 39 unflagged
Yes; `Diseased (other)` was 802 rows out, with 741 Yes carrying no supporting
flag at all. Deriving them means a flag correction propagates automatically --
clearing 15 ACVD flags used to leave the roll-up still saying Yes.

The classification is a judgement and is meant to be argued with; edit it here
and both tools follow.
"""
from __future__ import annotations
import pandas as pd

GUT = {
    "IBD", "Crohn's", "Ulcerative_colitis", "CRC", "adenoma", "coeliac", "CDI",
    "acute_diarrhoea", "treatment_colitis", "irritable_bowel", "STEC",
    "infectiousgastroenteritis", "salmonellosis", "few_polyps",
    "carcinoma_surgery_history", "STH", "schistosoma", "Worm-Infected",
    "TKI_dependent_diarrhoea",
}
METABOLIC = {
    "T1D", "T2D", "IGT", "metabolic_syndrome", "fatty_liver", "cirrhosis",
    "hypertension", "ACVD", "ascites", "hepatitis", "gestational_diabetes",
    "wilson",
}
# Everything else is "other" by construction, so a new flag lands there rather
# than silently vanishing from all three roll-ups.


def is_yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(
        {"1", "1.0", "y", "yes", "true", "t"})


def flag_names(df: pd.DataFrame) -> set[str]:
    return {c[len("disease_"):] for c in df.columns if c.startswith("disease_")}


def _any(df: pd.DataFrame, names) -> pd.Series:
    names = sorted(n for n in names if f"disease_{n}" in df.columns)
    if not names:
        return pd.Series(False, index=df.index)
    return pd.concat([is_yes(df[f"disease_{n}"]) for n in names], axis=1).any(axis=1)


def derive_rollups(df: pd.DataFrame, obesity_columns=()) -> dict[str, pd.Series]:
    """{column name: boolean Series} for the three derived roll-ups."""
    have = flag_names(df)
    other = have - GUT - METABOLIC
    ob = _any(df, ())  # placeholder, replaced below
    obs = [is_yes(df[c]) for c in obesity_columns if c in df.columns]
    bmi = pd.to_numeric(df.get("BMI"), errors="coerce")
    ob = (pd.concat(obs, axis=1).any(axis=1) if obs
          else pd.Series(False, index=df.index)) | (bmi >= 30)
    return {
        "Diseased (gut and metabolic)": _any(df, GUT | METABOLIC),
        "Diseased (other)": _any(df, other),
        "Diseased (metabolic) or obese": _any(df, METABOLIC) | ob,
    }


def apply_rollups(df: pd.DataFrame, obesity_columns=(), verbose=True) -> pd.DataFrame:
    for col, val in derive_rollups(df, obesity_columns).items():
        if verbose and col in df.columns:
            was = df[col].astype(str).str.strip().str.lower().eq("yes")
            print(f"  {col:<32} {int(was.sum()):>6,} -> {int(val.sum()):>6,}"
                  f"   (changed {int((was != val).sum()):,})")
        df[col] = val.map({True: "Yes", False: "No"})
    return df
