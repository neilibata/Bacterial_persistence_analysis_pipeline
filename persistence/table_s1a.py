"""
Table S1A builder plugin.

Resolves two open comments:
  1. Rachel/Neil thread: Age and Gbp/sample should be reported with
     uncertainty (mean ± SD), not just means.
  2. Condition column: each study row should list the specific health
     conditions compared, not just "Unhealthy %".

Outputs
-------
results/<ref>/table_s1a_study_summary.csv
    One row per (Author, Country, Year, Lifestyle) with columns:
        Author, Year, Country, Country_full,
        N (samples), n_hosts,
        age_median/_q25/_q75, age_n         ← per HOST
        bmi_median/_q25/_q75, bmi_n         ← per HOST
        gbp_median/_q25/_q75, gbp_n         ← per SAMPLE (sequencing depth)
        Lifestyle,
        unhealthy_pct                       ← % of HOSTS
        conditions, n_conditions,
        Reference

Hosts, not samples. Several studies sampled the same people repeatedly
(Lloyd-Price 2019: 1,125 samples from 104 hosts), so a per-sample age, BMI or
Unhealthy % weights each person by how often they were sampled. Each host is
represented by one sample -- the one the health analyses use
(`subject_primary`), or, for a host whose primary sample is not in the cohort,
its deepest sample -- and age, BMI and health status are read from it. A host
with no archived identifier carries its own Subject_ID and counts once.
Gbp/sample stays per sample: it describes the sequencing, not the person.

results/<ref>/table_s1a_study_summary_formatted.csv
    Same but with Age and Gbp/sample already formatted as "mean ± SD"
    strings, ready to paste into the supplement table.

Integration
-----------
Call standalone::

    python -m persistence --ref UHGG --only table_s1a

Or it runs automatically as part of a full pipeline run.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

from . import config


# ── Disease column → readable condition label mapping ────────────────────────
# Keys match the disease_* columns in metadata.
# Values are the labels that appear in the "conditions" column of Table S1A.

DISEASE_COL_LABELS: dict[str, str] = {
    "disease_T2D":               "Type 2 diabetes",
    "disease_IGT":               "Impaired glucose tolerance",
    "disease_T1D":               "Type 1 diabetes",
    "disease_fatty_liver":       "MASLD",
    "disease_metabolic_syndrome":"Metabolic syndrome",
    "disease_ACVD":              "Atherosclerotic CVD",
    "disease_hypertension":      "Hypertension",
    "disease_IBD":               "Inflammatory bowel disease",
    "disease_Crohn's":           "Crohn's disease",
    "disease_Ulcerative_colitis":"Ulcerative colitis",
    "disease_CRC":               "Colorectal cancer",
    "disease_adenoma":           "Colorectal adenoma",
    "disease_coeliac":           "Coeliac disease",
    "disease_PD":                "Parkinson's disease",
    "disease_MS":                "Multiple sclerosis",
    "disease_RA":                "Rheumatoid arthritis",
    # These three were keyed to column names the metadata does not use, so the
    # labels could never fire: cirrhosis (132 samples), schizophrenia (107) and
    # ME/CFS (60) were silently absent from the Conditions column. The metadata
    # spells the last two differently -- keep both keys so either survives a
    # future rename of the source column.
    "disease_CFS":               "Chronic fatigue syndrome",
    "disease_cirrhosis":         "Liver cirrhosis",
    "disease_liver_cirrhosis":   "Liver cirrhosis",
    "disease_schizofrenia":      "Schizophrenia",
    "disease_schizophrenia":     "Schizophrenia",
    # Conditions present in the metadata that had no label at all. Without these
    # a study whose only flag is one of them falls through to the generic
    # "Unspecified illness".
    "disease_myalgic_encephalomyelitis": "Myalgic encephalomyelitis",
    "disease_STH":               "Soil-transmitted helminth infection",
    "disease_Worm-Infected":     "Helminth infection",
    "disease_melanoma":          "Melanoma",
    "disease_hepatitis":         "Hepatitis",
    "disease_CDI":               "C. difficile infection",
    "disease_ankylosing_spondylitis": "Ankylosing spondylitis",
    "disease_acute_diarrhoea":   "Acute diarrhoea",
    # Deliberately NOT mapped: disease_STEC and disease_ME_CFS. Both are set for
    # samples the authoritative `disease [MBIOTEMP_disease]` field calls healthy
    # -- 82% and 67% of their flagged rows respectively -- so labelling from them
    # would put conditions on control samples. See docs/disease_flag_audit.csv.
}

# Columns that flag obesity / overweight (not disease per se)
OBESITY_LABELS: dict[str, str] = {
    "Obesity":              "Obesity",
    "Obesity (metadata)":   "Obesity",
    "Obesity_flag":         "Obesity",
    "Overweight":           "Overweight",
}


def _is_yes(val) -> bool:
    """Flexible truthy check for yes/no/1/0/True columns."""
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in {"yes", "1", "true", "unhealthy", "y"}


def _conditions_for_study(grp: pd.DataFrame) -> str:
    """
    Return a comma-separated string of conditions present in this study group.

    A condition is "present" if at least one sample in the group has a
    truthy value for that condition's column.
    """
    conditions = []

    # Specific disease columns
    for col, label in DISEASE_COL_LABELS.items():
        if col in grp.columns:
            if grp[col].apply(_is_yes).any():
                if label not in conditions:
                    conditions.append(label)

    # Obesity — check multiple possible columns, deduplicate
    obesity_present = False
    overweight_present = False
    for col, label in OBESITY_LABELS.items():
        if col in grp.columns:
            if grp[col].apply(_is_yes).any():
                if label == "Obesity":
                    obesity_present = True
                elif label == "Overweight":
                    overweight_present = True

    # Also check Obesity_flag_bool
    if "Obesity_flag_bool" in grp.columns:
        if grp["Obesity_flag_bool"].any():
            obesity_present = True

    if obesity_present:
        conditions.append("Obesity")
    if overweight_present:
        conditions.append("Overweight")

    # BMI-based check as last resort
    if not obesity_present and "BMI" in grp.columns:
        bmi = pd.to_numeric(grp["BMI"], errors="coerce")
        if (bmi >= 30).any():
            conditions.append("Obesity (BMI ≥ 30)")

    # If Unhealthy column exists but no specific condition was found,
    # note it as "Unspecified illness"
    if not conditions and "Unhealthy" in grp.columns:
        if grp["Unhealthy"].apply(_is_yes).any():
            conditions.append("Unspecified illness")

    return ", ".join(conditions) if conditions else "Healthy controls only"


def _author_key(meta: pd.DataFrame) -> pd.Series:
    """
    Return the study grouping key column (Author preferred, then StudyFolder).
    Returns a Series aligned with meta.index.
    """
    for col in ["Author", "StudyFolder", "study_name [MBIOTEMP_study_name]"]:
        if col in meta.columns:
            return meta[col].fillna("Unknown")
    return pd.Series("Unknown", index=meta.index)


def _country_full(meta: pd.DataFrame) -> pd.Series:
    """Return Country (full) if available, else Country.

    Earlier metadata generations spelled some countries several ways — the USA
    appeared as "United States of America", "United States" and "USA" across
    different studies. Table S1 took whichever spelling the group's first sample
    carried, so one country could occupy several apparent rows and inflate a
    naive count of distinct countries. The curated table no longer does this:
    every country code in it maps to exactly one spelling, so canonicalising to
    the most common spelling per code is now a guard rather than a repair. Keep
    it, because the failure is silent and surfaces only as a country count that
    disagrees with the cohort's.
    """
    code = meta.get("Country", pd.Series("", index=meta.index)).astype(str).str.strip()
    if "Country (full)" not in meta.columns:
        return code

    full = meta["Country (full)"].fillna(code).astype(str).str.strip()
    named = full != ""
    canon = (pd.DataFrame({"code": code[named], "full": full[named]})
             .groupby("code")["full"]
             .agg(lambda s: s.value_counts().idxmax()))
    return code.map(canon).fillna(full)


def _representatives(meta: pd.DataFrame, depth: pd.Series) -> pd.Series:
    """Map each sample to its host's representative sample.

    The representative is the host's `subject_primary` sample -- the one the
    health analyses keep -- when it is in the cohort; otherwise the host's
    deepest sample, then the first. Samples without a Subject_ID represent
    themselves.
    """
    subj = meta.get("Subject_ID", pd.Series(np.nan, index=meta.index))
    subj = subj.where(subj.notna(), pd.Series(meta.index, index=meta.index)).astype(str)
    prim = (meta.get("subject_primary", pd.Series("", index=meta.index))
                .astype(str).str.strip().str.lower().eq("yes"))
    rank = pd.DataFrame({"subj": subj, "prim": prim,
                         "depth": pd.to_numeric(depth.reindex(meta.index),
                                                errors="coerce").fillna(-1.0),
                         "pos": np.arange(len(meta))}, index=meta.index)
    rep = (rank.sort_values(["subj", "prim", "depth", "pos"],
                            ascending=[True, False, False, True])
               .reset_index().drop_duplicates("subj")
               .set_index("subj")[rank.index.name or "index"])
    return subj.map(rep)


def _median_iqr(vals: pd.Series, decimals: int):
    """(median, median-Q25, Q75-median, Q25, Q75, n) of the non-missing values."""
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if not len(v):
        return (np.nan,) * 5 + (0,)
    med, q25, q75 = (float(np.median(v)), float(np.percentile(v, 25)),
                     float(np.percentile(v, 75)))
    return (round(med, decimals), round(med - q25, decimals),
            round(q75 - med, decimals), round(q25, decimals),
            round(q75, decimals), int(len(v)))


def run_table_s1a(data, stats, outdir=None) -> pd.DataFrame:
    """
    Build Table S1A from the post-reinstatement metadata and total_bp.

    Parameters
    ----------
    data   : ReferenceData  (post-reinstatement)
    stats  : StatsReport

    Returns
    -------
    pd.DataFrame — the raw (numeric) summary table
    """
    ref = data.ref_name
    outdir = outdir or os.path.join("results", ref)
    os.makedirs(outdir, exist_ok=True)

    meta = data.meta
    total_bp = data.total_bp   # Series, samples → mapped bp

    print(f"\n{'='*60}")
    print(f"  Table S1A builder [{ref}]")
    print(f"{'='*60}")
    print(f"  Samples in cohort: {len(meta):,}")

    # Gbp per sample from total_bp
    gbp = (total_bp / 1e9).reindex(meta.index)

    # Age as numeric
    if "Age" in meta.columns:
        age = pd.to_numeric(meta["Age"], errors="coerce")
    else:
        age = pd.Series(np.nan, index=meta.index)
        print("  [WARN] No 'Age' column found — age columns will be empty")

    # BMI as numeric, blanked outside the plausible window (config.BMI_PLAUSIBLE)
    if "BMI" in meta.columns:
        bmi = pd.to_numeric(meta["BMI"], errors="coerce")
        lo_ok, hi_ok = getattr(config, "BMI_PLAUSIBLE", (10.0, 60.0))
        bmi = bmi.where(bmi.between(lo_ok, hi_ok))
    else:
        bmi = pd.Series(np.nan, index=meta.index)

    # One representative sample per host; age, BMI and health are read from it
    rep_of = _representatives(meta, total_bp)
    subj = meta.get("Subject_ID", pd.Series(np.nan, index=meta.index))
    subj = subj.where(subj.notna(), pd.Series(meta.index, index=meta.index)).astype(str)
    uh_all = meta.get("Unhealthy", pd.Series("Healthy", index=meta.index)).apply(_is_yes)
    n_disc = int((uh_all.groupby(subj).nunique() > 1).sum())
    if n_disc:
        print(f"  {n_disc} hosts have both Healthy and Unhealthy samples; "
              f"each is counted by its representative sample")

    # Author and Country grouping keys
    author_col = _author_key(meta)
    country_col = meta.get("Country", pd.Series("", index=meta.index))
    country_full_col = _country_full(meta)

    # Year
    year_col = None
    for c in ["Year", "year", "Publication_year"]:
        if c in meta.columns:
            year_col = meta[c]
            break

    # Lifestyle (majority vote within group)
    lifestyle_col = meta.get("Lifestyle", pd.Series("", index=meta.index))

    # Unhealthy flag
    unhealthy_col = meta.get("Unhealthy", pd.Series("Healthy", index=meta.index))

    # Reference / citation
    ref_col = None
    for c in ["Reference", "Citation", "reference", "citation"]:
        if c in meta.columns:
            ref_col = meta[c]
            break

    # ── Group by Author x Country x Year x Lifestyle ─────────────────────────
    # Lifestyle is included so that studies spanning multiple lifestyles (e.g.
    # LokmerA 2019 CMR with both Transitional and H-G samples) produce one
    # row per lifestyle rather than collapsing to the majority class.
    # Year ensures same-author same-country different-year studies are kept
    # separate; studies with no year get "" as a sentinel.
    year_series = pd.Series("", index=meta.index)
    if year_col is not None:
        year_series = year_col.fillna("").astype(str)

    group_keys = pd.DataFrame({
        "author":    author_col,
        "country":   country_col,
        "year":      year_series,
        "lifestyle": lifestyle_col.fillna("Unknown"),
    }, index=meta.index)

    rows = []
    for (author, country, year_key, lifestyle), idx in group_keys.groupby(
            ["author", "country", "year", "lifestyle"], sort=True).groups.items():

        grp     = meta.loc[idx]
        grp_gbp = gbp.loc[idx]

        n = len(grp)

        # ── Hosts: one representative sample each ────────────────────────────
        # A host's representative is chosen cohort-wide, so it can sit in
        # another row of the same study (a different lifestyle split); fall
        # back to the host's own samples in this row then.
        reps = rep_of.loc[idx]
        in_row = reps.isin(idx)
        reps = reps.where(in_row, pd.Series(idx, index=idx))
        hosts = (pd.DataFrame({"subj": subj.loc[idx], "rep": reps})
                   .drop_duplicates("subj")["rep"])
        n_hosts = int(len(hosts))

        # ── Age and BMI: per host, median + asymmetric IQR bars ──────────────
        age_med, age_lo, age_hi, age_q25, age_q75, age_n = _median_iqr(
            age.loc[hosts.values], 1)
        bmi_med, bmi_lo, bmi_hi, bmi_q25, bmi_q75, bmi_n = _median_iqr(
            bmi.loc[hosts.values], 1)

        # ── Gbp/sample stats: median + asymmetric IQR bars ───────────────────
        gbp_vals = grp_gbp.dropna()
        gbp_n    = int(len(gbp_vals))
        if gbp_n >= 1:
            gbp_med = float(np.median(gbp_vals))
            gbp_q25 = float(np.percentile(gbp_vals, 25))
            gbp_q75 = float(np.percentile(gbp_vals, 75))
            gbp_lo  = round(gbp_med - gbp_q25, 2)
            gbp_hi  = round(gbp_q75 - gbp_med, 2)
            gbp_med = round(gbp_med, 2)
        else:
            gbp_med = gbp_q25 = gbp_q75 = gbp_lo = gbp_hi = np.nan

        # Unhealthy %: of hosts, by their representative sample
        uh = unhealthy_col.loc[hosts.values].apply(_is_yes)
        unhealthy_pct = round(float(uh.mean() * 100), 1) if n_hosts > 0 else 0.0

        # Conditions
        conditions = _conditions_for_study(grp)

        # Year comes directly from the groupby key (year_key)
        year = year_key if year_key != "" else ""

        # Country (full)
        cfull = country_full_col.loc[idx].iloc[0] if len(idx) else country

        # Reference
        ref_str = ""
        if ref_col is not None:
            rv = ref_col.loc[idx].dropna()
            if len(rv):
                ref_str = str(rv.iloc[0])

        rows.append({
            "Author":        author,
            "Year":          year,
            "Country":       country,
            "Country_full":  cfull,
            "N":             n,
            "n_hosts":       n_hosts,
            "age_median":    age_med  if np.isfinite(age_med)  else "",
            "age_lo":        age_lo   if np.isfinite(age_lo)   else "",
            "age_hi":        age_hi   if np.isfinite(age_hi)   else "",
            "age_q25":       age_q25  if np.isfinite(age_q25)  else "",
            "age_q75":       age_q75  if np.isfinite(age_q75)  else "",
            "age_n":         age_n,
            "bmi_median":    bmi_med  if np.isfinite(bmi_med)  else "",
            "bmi_lo":        bmi_lo   if np.isfinite(bmi_lo)   else "",
            "bmi_hi":        bmi_hi   if np.isfinite(bmi_hi)   else "",
            "bmi_q25":       bmi_q25  if np.isfinite(bmi_q25)  else "",
            "bmi_q75":       bmi_q75  if np.isfinite(bmi_q75)  else "",
            "bmi_n":         bmi_n,
            "gbp_median":    gbp_med  if np.isfinite(gbp_med)  else "",
            "gbp_lo":        gbp_lo   if np.isfinite(gbp_lo)   else "",
            "gbp_hi":        gbp_hi   if np.isfinite(gbp_hi)   else "",
            "gbp_q25":       round(gbp_q25, 2) if np.isfinite(gbp_q25) else "",
            "gbp_q75":       round(gbp_q75, 2) if np.isfinite(gbp_q75) else "",
            "gbp_n":         gbp_n,
            "Lifestyle":     lifestyle,
            "unhealthy_pct": unhealthy_pct,
            "conditions":    conditions,
            "n_conditions":  0 if conditions == "Healthy controls only"
                               else len(conditions.split(", ")),
            "Reference":     ref_str,
        })

    summary_df = pd.DataFrame(rows)

    # ── Save raw (numeric) CSV ────────────────────────────────────────────────
    raw_path = os.path.join(outdir, "table_s1a_study_summary.csv")
    summary_df.to_csv(raw_path, index=False)
    print(f"  Saved: {raw_path}  ({len(summary_df)} study × country rows)")

    # ── Save formatted CSV with Word equation linearized strings ─────────────
    #
    # Word's equation editor (Alt+=) accepts a linearized math format.
    # Asymmetric error bars in Word equation notation:
    #
    #   value_(−lower)^(+upper)
    #
    # Example: a study with median age 45.1, Q25=38.0, Q75=52.3 produces:
    #   age_lo = 45.1 − 38.0 = 7.1   (lower bar, below median)
    #   age_hi = 52.3 − 45.1 = 7.2   (upper bar, above median)
    #   → "45.1_(−7.1)^(+7.2)"
    #
    # To use in Word:
    #   1. Press Alt+= to open an equation box
    #   2. Paste the string — Word auto-renders the subscript/superscript
    #   3. For the table cell: copy the rendered equation into the cell
    #
    # Studies with n=1 (or no age data) get a plain value with no bars.

    fmt_df = summary_df.copy()

    def _word_asymmetric(med, lo, hi, n_obs, decimals=1):
        """
        Format as Word equation linearized string:  med_(−lo)^(+hi)
        Falls back to plain value string when bars are unavailable.
        """
        if med == "" or (isinstance(med, float) and not np.isfinite(med)):
            return ""
        med_s = f"{float(med):.{decimals}f}"
        if (lo == "" or hi == "" or n_obs < 2
                or (isinstance(lo, float) and not np.isfinite(lo))
                or (isinstance(hi, float) and not np.isfinite(hi))):
            return med_s
        lo_s = f"{float(lo):.{decimals}f}"
        hi_s = f"{float(hi):.{decimals}f}"
        # Word linearized: value_(subscript text)^(superscript text)
        return f"{med_s}_(\u2212{lo_s})^(+{hi_s})"

    fmt_df["Age (Word eq.)"] = [
        _word_asymmetric(r["age_median"], r["age_lo"], r["age_hi"],
                         r["age_n"], decimals=1)
        for _, r in summary_df.iterrows()
    ]
    fmt_df["BMI (Word eq.)"] = [
        _word_asymmetric(r["bmi_median"], r["bmi_lo"], r["bmi_hi"],
                         r["bmi_n"], decimals=1)
        for _, r in summary_df.iterrows()
    ]
    fmt_df["Gbp/sample (Word eq.)"] = [
        _word_asymmetric(r["gbp_median"], r["gbp_lo"], r["gbp_hi"],
                         r["gbp_n"], decimals=2)
        for _, r in summary_df.iterrows()
    ]

    # Also provide a plain "median [Q25–Q75]" column for reference/checking
    def _plain_iqr(med, q25, q75, decimals=1):
        if med == "" or (isinstance(med, float) and not np.isfinite(med)):
            return ""
        m = f"{float(med):.{decimals}f}"
        if q25 == "" or q75 == "":
            return m
        return f"{m} [{float(q25):.{decimals}f}\u2013{float(q75):.{decimals}f}]"

    fmt_df["Age median [IQR]"]      = [
        _plain_iqr(r["age_median"], r["age_q25"], r["age_q75"], 1)
        for _, r in summary_df.iterrows()
    ]
    fmt_df["BMI median [IQR]"] = [
        _plain_iqr(r["bmi_median"], r["bmi_q25"], r["bmi_q75"], 1)
        for _, r in summary_df.iterrows()
    ]
    fmt_df["Gbp/sample median [IQR]"] = [
        _plain_iqr(r["gbp_median"], r["gbp_q25"], r["gbp_q75"], 2)
        for _, r in summary_df.iterrows()
    ]

    fmt_df = fmt_df.rename(columns={"n_hosts": "Hosts", "age_n": "Age N (hosts)",
                                    "bmi_n": "BMI N (hosts)",
                                    "unhealthy_pct": "Unhealthy % (hosts)"})
    present_cols = [
        "Author", "Year", "Country", "Country_full", "N", "Hosts",
        "Age (Word eq.)", "Age median [IQR]", "Age N (hosts)",
        "BMI (Word eq.)", "BMI median [IQR]", "BMI N (hosts)",
        "Gbp/sample (Word eq.)", "Gbp/sample median [IQR]",
        "Lifestyle", "Unhealthy % (hosts)", "conditions", "Reference",
    ]
    present_cols = [c for c in present_cols if c in fmt_df.columns]
    fmt_path = os.path.join(outdir, "table_s1a_study_summary_formatted.csv")
    fmt_df[present_cols].to_csv(fmt_path, index=False)
    print(f"  Saved: {fmt_path}")
    print(f"  Word equation format: value_(−lower)^(+upper)")
    print(f"  Example: {fmt_df['Age (Word eq.)'].dropna().iloc[0] if len(fmt_df) else 'n/a'}")

    # ── Stats report entries ──────────────────────────────────────────────────
    n_studies  = len(summary_df)
    n_with_age = int((summary_df["age_n"] > 0).sum())
    n_conds    = int((summary_df["n_conditions"] > 0).sum())

    stats.add("table_s1a_n_studies", "qc", "supplement",
              "Table S1A: total study × country rows",
              f"{n_studies}")
    stats.add("table_s1a_n_with_age", "qc", "supplement",
              "Table S1A: rows with at least one age value",
              f"{n_with_age}/{n_studies}")
    stats.add("table_s1a_n_hosts", "qc", "supplement",
              "Table S1A: hosts summed over rows (a host split across two "
              "lifestyle rows of one study counts in each)",
              int(summary_df["n_hosts"].sum()))
    stats.add("table_s1a_hosts_discordant_health", "qc", "supplement",
              "Table S1A: hosts with both Healthy and Unhealthy samples "
              "(counted by their representative sample)", n_disc)
    stats.add("table_s1a_n_with_conditions", "qc", "supplement",
              "Table S1A: rows with at least one named health condition",
              f"{n_conds}/{n_studies}")

    print(f"\n  Summary:")
    print(f"    {n_studies} study × country rows")
    print(f"    {n_with_age} with age data")
    print(f"    {n_conds} with named health conditions")

    return summary_df
