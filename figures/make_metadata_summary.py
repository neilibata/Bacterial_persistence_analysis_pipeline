#!/usr/bin/env python3
"""
make_metadata_summary.py
========================
Standalone script: summarise metadata_Mar_16_2026.csv using the same
preprocessing logic as the pipeline (depth filter, study exclusion,
non-Industrial NA Unhealthy → Healthy recode, obesity flag consolidation,
secondary-run BC screening & reinstatement).

Does NOT require the matrix files — it only needs the metadata CSV and,
optionally, the secondary-run CSV to mark reinstated samples.

Outputs
-------
  samples_summary_by_study.csv            — per-study aggregated table
  metadata_annotated.csv                  — full sample-level annotated table
  fig_lifestyle_continent.png/pdf         — stacked bar: Lifestyle × Continent
  fig_health_by_lifestyle.png/pdf         — stacked bar: Health × Lifestyle
  fig_age_distribution.png/pdf            — violin: Age by Lifestyle
  fig_depth_by_lifestyle.png/pdf          — violin: sequencing depth by Lifestyle
  fig_samples_by_country_year.png/pdf     — bubble map (requires geopandas) or
                                            table plot fallback

Usage
-----
  python make_metadata_summary.py \\
      --meta  metadata_Mar_16_2026.csv \\
      --secondary  Chosen_population_samples.csv   # optional
      --out-dir    ./metadata_summary/
"""

# The repo root on the path BEFORE the first `persistence` import. This
# script is run by path -- figure_runner launches it as a subprocess and the
# README shows the same -- so sys.path[0] is `figures/` and the package is not
# importable from it. A reviewer running it from a clean clone got
# ModuleNotFoundError on the next line. See figure2.py for the full note.
import os as _os, sys as _sys
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import persistence.config as _cfg
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  Okabe-Ito palette  (matches pipeline colour scheme exactly)
# ─────────────────────────────────────────────────────────────────────────────
OI = {
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermillion":"#D55E00",
    "pink":      "#CC79A7",
    "black":     "#000000",
}

LIFESTYLE_COLORS = {
    "H-G":          OI["vermillion"],
    "Transitional": OI["yellow"],
    "Industrial":   OI["sky"],
}
LIFESTYLE_DISPLAY = {
    "H-G": "Hunter-Gatherer",
    "Transitional": "Agro-Pastoralist",
    "Agro-Pastoralist": "Agro-Pastoralist",
    "Agro-Pastorialist": "Agro-Pastoralist",
    "Industrial": "Industrial",
}

HEALTH_COLORS = {
    "Healthy":   OI["green"],
    "Unhealthy": OI["blue"],
}

CONTINENT_COLORS = {
    "Africa":        OI["vermillion"],
    "Asia":          OI["orange"],
    "Europe":        OI["sky"],
    "North America": OI["blue"],
    "South America": OI["green"],
    "Oceania":       OI["pink"],
    "Other":         OI["black"],
}


# Delegates to config, which holds the single list of display-time renames
# (PLOT_TEXT_REPLACEMENTS). This was a hard-coded local copy that knew only
# about Agro-Pastorialist/Transitional, so it silently missed every rename
# added since: Fatty Liver to MASLD, Obese to Obesity, and the lifestyle
# sentence casing. The fallback is for running this script standalone, with
# the package absent.
try:
    from persistence.config import display_text as _display_text
except Exception as _exc:                          # package genuinely absent
    # Not a second rename table: a local copy drifts from
    # config.PLOT_TEXT_REPLACEMENTS and disagrees with the rest of the figure
    # set, which is worse than an obviously raw key.
    print(f"  [warn] persistence.config unavailable ({_exc}); figure text "
          f"will carry internal keys, not published labels", file=sys.stderr)

    def _display_text(text):
        return str(text)


def _apply_display_text(fig):
    for artist in fig.findobj(match=matplotlib.text.Text):
        old = artist.get_text()
        new = _display_text(old)
        if new != old:
            artist.set_text(new)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  Geography helpers  (identical to pipeline's common.py + config.py)
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_ALIASES = {
    "SWE": "Sweden",          "Sweden": "Sweden",
    "DNK": "Denmark",         "DEN": "Denmark",       "Denmark": "Denmark",
    "DEU": "Germany",         "GER": "Germany",        "Germany": "Germany",
    "FRA": "France",           "France": "France",
    "NLD": "Netherlands",     "Netherlands": "Netherlands",
    "GBR": "United Kingdom",  "UK": "United Kingdom",  "United Kingdom": "United Kingdom",
    "ESP": "Spain",            "Spain": "Spain",
    "AUT": "Austria",         "AUS": "Austria",        "Austria": "Austria",
    "ITA": "Italy",            "Italy": "Italy",
    "KAZ": "Kazakhstan",       "Kazakhstan": "Kazakhstan",
    "FIN": "Finland",          "Finland": "Finland",
    "BEL": "Belgium",          "Belgium": "Belgium",
    "ISL": "Iceland",          "Iceland": "Iceland",
    "RUS": "Russia",           "Russia": "Russia",
    "USA": "United States of America",
    "United States": "United States of America",
    "United States of America": "United States of America",
    "CAN": "Canada",           "Canada": "Canada",
    "BRA": "Brazil",           "Brazil": "Brazil",
    "ARG": "Argentina",        "Argentina": "Argentina",
    "PER": "Peru",             "Peru": "Peru",
    "COL": "Colombia",         "Colombia": "Colombia",
    "SLV": "El Salvador",      "El Salvador": "El Salvador",
    "HND": "Honduras",         "HON": "Honduras",      "Honduras": "Honduras",
    "ZAF": "South Africa",     "SA": "South Africa",   "South Africa": "South Africa",
    "ETH": "Ethiopia",         "Ethiopia": "Ethiopia",
    "GHA": "Ghana",            "Ghana": "Ghana",
    "TZA": "Tanzania",         "Tanzania": "Tanzania",
    "GNB": "Guinea-Bissau",    "Guinea-Bissau": "Guinea-Bissau",
    "MDG": "Madagascar",       "MAD": "Madagascar",    "Madagascar": "Madagascar",
    "CMR": "Cameroon",         "Cameroon": "Cameroon",
    "LBR": "Liberia",          "LIB": "Liberia",       "Liberia": "Liberia",
    "BFA": "Burkina Faso",     "BKF": "Burkina Faso",  "Burkina Faso": "Burkina Faso",
    "COD": "Democratic Republic of the Congo",
    "DRC": "Democratic Republic of the Congo",
    "DRC Congo": "Democratic Republic of the Congo",
    "CHN": "China",  "CHI": "China",  "China": "China",
    "JPN": "Japan",            "Japan": "Japan",
    "NPL": "Nepal",            "NEP": "Nepal",         "Nepal": "Nepal",
    "THA": "Thailand",         "Thailand": "Thailand",
    "MNG": "Mongolia",         "Mongolia": "Mongolia",
    "IDN": "Indonesia",        "IND": "Indonesia",     "Indonesia": "Indonesia",
    "ISR": "Israel",           "Israel": "Israel",
    "KOR": "South Korea",      "South Korea": "South Korea",
    "BGD": "Bangladesh",       "Bangladesh": "Bangladesh",
    "PHL": "Philippines",      "PHN": "Philippines",   "Philippines": "Philippines",
    "FJI": "Fiji",             "Fiji": "Fiji",
    "NZL": "New Zealand",      "New Zealand": "New Zealand",
}

COUNTRY_TO_CONTINENT = {
    "Sweden": "Europe",        "Denmark": "Europe",       "Germany": "Europe",
    "France": "Europe",        "Netherlands": "Europe",   "United Kingdom": "Europe",
    "Spain": "Europe",         "Austria": "Europe",       "Italy": "Europe",
    "Kazakhstan": "Europe",    "Finland": "Europe",       "Belgium": "Europe",
    "Iceland": "Europe",       "Russia": "Europe",
    "United States of America": "North America",
    "Canada": "North America", "El Salvador": "North America", "Honduras": "North America",
    "Brazil": "South America", "Peru": "South America",
    "Argentina": "South America", "Colombia": "South America",
    "South Africa": "Africa",  "Ethiopia": "Africa",      "Ghana": "Africa",
    "Tanzania": "Africa",      "Guinea-Bissau": "Africa", "Madagascar": "Africa",
    "Cameroon": "Africa",      "Liberia": "Africa",       "Burkina Faso": "Africa",
    "Democratic Republic of the Congo": "Africa",
    "Israel": "Asia",   "China": "Asia",    "Japan": "Asia",   "Nepal": "Asia",
    "Thailand": "Asia", "Mongolia": "Asia", "Indonesia": "Asia","South Korea": "Asia",
    "Bangladesh": "Asia",      "Philippines": "Asia",
    "Fiji": "Oceania",  "New Zealand": "Oceania",
}

# keep in step with config.OBESITY_COLUMNS
OBESITY_COLUMNS = ["Obesity", "Obesity (metadata)"]
# A floor on *sequenced* bases ("Total bases" in M1). This is NOT the pipeline's
# >=30 Mb cut, which is applied to mapped bases derived from the depth matrix and
# is ~100x smaller for the same sample; it only screens out obviously empty rows
# when no cohort manifest is available. The real cohort comes from COHORT_REF's
# manifest -- see _apply_pipeline_cohort.
MIN_TOTAL_BP    = 30_000_000
try:                            # the main-text reference: config.MAIN_COHORT_REF
    from persistence.config import MAIN_COHORT_REF as COHORT_REF
except Exception:               # standalone, package absent
    COHORT_REF  = "combined"


def normalise_country(s):
    if pd.isna(s):
        return None
    return COUNTRY_ALIASES.get(str(s).strip(), str(s).strip())


def get_continent(country):
    if pd.isna(country) or country is None:
        return "Other"
    s = str(country).strip()
    result = COUNTRY_TO_CONTINENT.get(s)
    if result:
        return result
    norm = normalise_country(s)
    return COUNTRY_TO_CONTINENT.get(norm, "Other")


def is_yes(val):
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in ("1", "1.0", "yes", "y", "true", "positive", "obese", "overweight"):
        return True
    try:
        return float(s) > 0
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline preprocessing (metadata only — no matrix files needed)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pipeline_cohort(meta: pd.DataFrame, cohort_ref: str) -> pd.DataFrame:
    """Restrict `meta` to the samples the analyses actually ran on.

    The pipeline's two exclusions -- >=30 Mb of *mapped* bases, and a floor on
    SGBs detected at the breadth threshold -- are computed from the matrices, not
    from Table M1, so they cannot be reproduced here. `run_reference` writes the
    surviving index to results/<ref>/cohort_samples.csv after reinstatement;
    intersecting against that is the only way for a metadata-only script to
    report the same N as the rest of the manuscript.

    Falls back to a sequenced-bases floor, loudly, when no manifest exists -- an
    approximation that over-counts by a couple of hundred samples.
    """
    from persistence.cohort_manifest import manifest_path, read_cohort_manifest

    cohort = read_cohort_manifest(cohort_ref)
    if cohort is None:
        print(f"  WARNING no cohort manifest at {manifest_path(cohort_ref)} — "
              f"falling back to a 'Total bases' floor, which OVER-COUNTS the "
              f"analytic cohort. Run the pipeline for --ref {cohort_ref} to "
              f"write it, then regenerate this figure.")
        if "Total bases" in meta.columns:
            low = meta["Total bases"] < MIN_TOTAL_BP
            if low.any():
                meta = meta[~low]
                print(f"  Dropped {low.sum():,} samples with "
                      f"< {MIN_TOTAL_BP/1e6:.0f} Mb sequenced (approximate)")
        return meta

    ids = set(cohort["Chosen_match_ID"].astype(str))
    keep = meta.index.astype(str).isin(ids)
    n_before = len(meta)
    missing = len(ids - set(meta.index.astype(str)))
    meta = meta[keep]
    print(f"  Analytic cohort ({cohort_ref}): {len(meta):,} of {n_before:,} "
          f"(dropped {n_before - len(meta):,} not in the analysis)")
    if missing:
        # The manifest is built from this same metadata file, so a sample in the
        # cohort but not in M1 means the two have drifted out of step.
        print(f"  WARNING {missing:,} cohort samples are absent from the "
              f"metadata — manifest and {os.path.basename(_cfg.META_FILE)} "
              f"disagree; regenerate the manifest.")
    return meta


def load_and_preprocess(meta_path: str, secondary_csv: str | None,
                        cohort_ref: str = COHORT_REF) -> pd.DataFrame:
    """
    Load metadata and apply the same preprocessing as the pipeline:
      1. Numeric coercions (Age, Total bases, BMI)
      2. Drop missing Lifestyle
      3. Restrict to the pipeline's analytic cohort (cohort_samples.csv)
      4. Non-Industrial NA Unhealthy → Healthy recode
      5. Obesity_flag from multiple columns + BMI >= 30
      6. Country normalisation + Continent assignment
      7. MacroGroup, Group_fine classification
      8. Mark reinstated secondaries (if secondary CSV provided)
    """
    print(f"Loading metadata: {meta_path}")
    meta = pd.read_csv(meta_path, low_memory=False, na_values=["", "NA", "NaN"])
    # Table M1 stores the manuscript's display labels (Agro-pastoralist,
    # Hunter-gatherer); analysis code compares against the internal keys.
    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = _cfg.canonicalize_lifestyle_series(meta["Lifestyle"])
    print(f"  Raw rows: {len(meta):,}")

    # Use the sample ID column as the index
    for id_col in ["Chosen_match_ID", "SampleID", "sample_id", "Run"]:
        if id_col in meta.columns:
            meta = meta.set_index(id_col)
            print(f"  Index column: {id_col}")
            break

    # Numeric coercions
    for col in ["Age", "Total bases", "BMI"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    # Drop missing Lifestyle, and any label that does not resolve to Industrial
    # or Non-Industrial ("Unclear"). Mirrors loading.py, which drops these at
    # load under config.DROP_UNCLASSIFIED_LIFESTYLE — without this, Figure 1 and
    # the metadata summaries would plot samples the analyses never saw.
    n_before = len(meta)
    valid_lifestyles = {"H-G", "Transitional", "Industrial"}
    keep = meta["Lifestyle"].notna() & meta["Lifestyle"].isin(valid_lifestyles)
    dropped = meta.loc[meta["Lifestyle"].notna() & ~keep, "Lifestyle"]
    if len(dropped):
        print(f"  Dropping {len(dropped):,} samples with an unclassifiable "
              f"Lifestyle: {dropped.value_counts().to_dict()}")
    meta = meta[keep]
    print(f"  After Lifestyle filter: {len(meta):,} (dropped {n_before - len(meta):,})")

    # Restrict to the cohort the analyses actually ran on.
    meta = _apply_pipeline_cohort(meta, cohort_ref)

    # Non-Industrial NA Unhealthy → Healthy
    if "Unhealthy" in meta.columns:
        nonind_na = (meta["Lifestyle"].isin(["H-G", "Transitional"]) &
                     meta["Unhealthy"].isna())
        if nonind_na.any():
            meta.loc[nonind_na, "Unhealthy"] = "Healthy"
            print(f"  Re-coded {nonind_na.sum():,} non-Industrial NA → Healthy")

    # Obesity flag
    ob_cols = [c for c in OBESITY_COLUMNS if c in meta.columns]
    if ob_cols:
        meta["Obesity_flag"] = meta[ob_cols].apply(
            lambda row: any(is_yes(v) for v in row), axis=1)
    else:
        meta["Obesity_flag"] = False
    if "BMI" in meta.columns:
        bmi_obese = pd.to_numeric(meta["BMI"], errors="coerce") >= 30
        meta["Obesity_flag"] = meta["Obesity_flag"] | bmi_obese.fillna(False)
    print(f"  Obesity_flag: {meta['Obesity_flag'].sum():,} positive")

    # Geography
    meta["Country_norm"] = meta["Country"].apply(normalise_country)
    if "Country (full)" in meta.columns:
        no_cont = meta["Country_norm"].isna()
        if no_cont.any():
            fallback = meta.loc[no_cont, "Country (full)"].apply(normalise_country)
            meta.loc[no_cont, "Country_norm"] = fallback
    meta["Continent"] = meta["Country_norm"].apply(get_continent)

    # Group classification
    def _macro(row):
        life = row.get("Lifestyle")
        if life == "Industrial":    return "Industrial"
        if life in ("H-G", "Transitional"): return "Non-Industrial"
        return None

    def _group_fine(row):
        uh   = str(row.get("Unhealthy", "")).strip()
        life = row.get("Lifestyle")
        if uh == "Unhealthy":
            return "Industrial Unhealthy" if life == "Industrial" else "Non-Industrial Unhealthy"
        elif uh == "Healthy":
            return "Industrial" if life == "Industrial" else "Non-Industrial"
        return None

    meta["MacroGroup"]  = meta.apply(_macro, axis=1)
    meta["Group_fine"]  = meta.apply(_group_fine, axis=1)

    # Reinstatement flag (secondary CSV only marks which runs were reinstated)
    meta["Reinstated"] = False
    if secondary_csv and os.path.exists(secondary_csv):
        try:
            sec = pd.read_csv(secondary_csv, low_memory=False)
            # The pipeline marks reinstated IDs in reinstate_secondary.py;
            # we look for a 'reinstated' column in the secondary CSV, or
            # simply mark all secondary run IDs that appear in our index.
            if "reinstated" in sec.columns and "secondary_id" in sec.columns:
                reinstated_ids = set(
                    sec.loc[sec["reinstated"] == True, "secondary_id"])
            else:
                # Fallback: check Secondary_run_IDs column in main metadata
                reinstated_ids = set()
            meta.loc[meta.index.isin(reinstated_ids), "Reinstated"] = True
            print(f"  Reinstated secondary samples: {meta['Reinstated'].sum():,}")
        except Exception as e:
            print(f"  [WARN] Could not parse secondary CSV: {e}")

    print(f"  Final dataset: {len(meta):,} samples")
    ls_counts = meta["Lifestyle"].value_counts().to_dict()
    print(f"  Lifestyle: {ls_counts}")
    if "Unhealthy" in meta.columns:
        uh_counts = meta["Unhealthy"].value_counts(dropna=False).to_dict()
        print(f"  Health:    {uh_counts}")

    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_csv(meta: pd.DataFrame, out_path: str):
    """Per-study summary table mirroring the original script but using
    pipeline-preprocessed metadata."""
    group_cols = []
    for c in ["Author", "Study", "StudyFolder", "Year", "Country", "Country (full)"]:
        if c in meta.columns:
            group_cols.append(c)
    if not group_cols:
        group_cols = ["Country_norm"]

    rows = []
    for keys, g in meta.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_samples"] = len(g)

        for lf in ["H-G", "Industrial", "Transitional"]:
            row[f"n_{lf}"] = int((g["Lifestyle"] == lf).sum())
        row["n_Reinstated"] = int(g.get("Reinstated", pd.Series(False)).sum())

        if "Total bases" in g.columns:
            row["median_total_bases"] = g["Total bases"].median()
        if "Age" in g.columns:
            row["median_age"]  = g["Age"].median()
            row["mean_age"]    = g["Age"].mean()
        if "Gender" in g.columns:
            gen = g["Gender"].astype(str).str.strip().str.lower()
            row["pct_female"] = (gen == "female").mean()
        if "Unhealthy" in g.columns:
            uh = g["Unhealthy"].astype(str).str.strip()
            row["pct_Unhealthy"] = (uh == "Unhealthy").mean()
        row["n_Obese"]      = int(g["Obesity_flag"].sum()) if "Obesity_flag" in g else 0
        row["Continent"]    = g["Continent"].mode().iloc[0] if not g["Continent"].empty else ""

        # Top disease flags
        disease_cols = [c for c in g.columns if c.startswith("disease_")]
        for dc in disease_cols[:10]:
            row[f"n_{dc}"] = int(
                (g[dc].astype(str).str.strip().str.lower() == "yes").sum())

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(group_cols, kind="mergesort")
    out_df.to_csv(out_path, index=False)
    print(f"  Summary CSV: {out_path}  ({len(out_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

FS = 11  # base font size


def _savefig(fig, path_stem: str, dpi: int = 180):
    """Save as PNG and PDF."""
    _apply_display_text(fig)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        fpath = path_stem + ext
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {fpath}")
    plt.close(fig)


def plot_lifestyle_by_continent(meta: pd.DataFrame, out_stem: str):
    """Stacked horizontal bar: one bar per continent, split by Lifestyle."""
    lifestyles = ["H-G", "Transitional", "Industrial"]
    conts = [c for c in CONTINENT_COLORS if c in meta["Continent"].values]
    conts = sorted(conts, key=lambda c: (meta["Continent"] == c).sum(), reverse=True)

    counts = {}
    for cont in conts:
        sub = meta[meta["Continent"] == cont]
        counts[cont] = {lf: int((sub["Lifestyle"] == lf).sum()) for lf in lifestyles}

    fig, ax = plt.subplots(figsize=(11, max(4, len(conts) * 0.65 + 1.5)))
    y = np.arange(len(conts))
    left = np.zeros(len(conts))

    for lf in lifestyles:
        vals = np.array([counts[c][lf] for c in conts], dtype=float)
        bars = ax.barh(y, vals, left=left, color=LIFESTYLE_COLORS[lf],
                       label=LIFESTYLE_DISPLAY.get(lf, lf),
                       edgecolor="white", lw=0.4, height=0.7)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, i, f"{int(v):,}", ha="center", va="center",
                        fontsize=FS - 3, color="white", fontweight="bold")
        left += vals

    # Total label at the right
    for i, cont in enumerate(conts):
        total = sum(counts[cont].values())
        ax.text(left[i] + 30, i, f"n={total:,}", va="center", fontsize=FS - 2)

    ax.set_yticks(y)
    ax.set_yticklabels(conts, fontsize=FS)
    ax.set_xlabel("Number of samples", fontsize=FS)
    ax.set_title("Samples by continent and lifestyle\n(pipeline-preprocessed dataset)",
                 fontsize=FS + 2, fontweight="bold")
    ax.legend(fontsize=FS - 1, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.2)
    _savefig(fig, out_stem)


def plot_health_by_lifestyle(meta: pd.DataFrame, out_stem: str):
    """Grouped bar: Healthy / Unhealthy counts per Lifestyle."""
    if "Unhealthy" not in meta.columns:
        print("  [SKIP] health_by_lifestyle: no 'Unhealthy' column")
        return

    lifestyles = [lf for lf in ["H-G", "Transitional", "Industrial"]
                  if lf in meta["Lifestyle"].values]
    health_cats = ["Healthy", "Unhealthy"]
    x = np.arange(len(lifestyles))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (hc, col) in enumerate(zip(health_cats, [HEALTH_COLORS["Healthy"],
                                                      HEALTH_COLORS["Unhealthy"]])):
        vals = []
        for lf in lifestyles:
            sub = meta[meta["Lifestyle"] == lf]
            uh  = sub["Unhealthy"].astype(str).str.strip()
            vals.append(int((uh == hc).sum()))
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.88, color=col, label=hc,
                      edgecolor="white", lw=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 20, f"{val:,}",
                        ha="center", va="bottom", fontsize=FS - 2, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(lifestyles, fontsize=FS)
    ax.set_ylabel("Number of samples", fontsize=FS)
    ax.set_title("Health status by lifestyle", fontsize=FS + 2, fontweight="bold")
    ax.legend(fontsize=FS - 1, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    _savefig(fig, out_stem)


def plot_age_distribution(meta: pd.DataFrame, out_stem: str):
    """Violin + strip plot of Age by Lifestyle."""
    if "Age" not in meta.columns:
        print("  [SKIP] age_distribution: no 'Age' column")
        return

    lifestyles = [lf for lf in ["H-G", "Transitional", "Industrial"]
                  if lf in meta["Lifestyle"].values]

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(42)

    for i, lf in enumerate(lifestyles):
        vals = meta.loc[meta["Lifestyle"] == lf, "Age"].dropna().values
        if len(vals) < 4:
            continue
        col = LIFESTYLE_COLORS[lf]
        parts = ax.violinplot([vals], positions=[i], showmedians=True,
                              widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.55)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color(col)
        # Strip
        jitter = rng.uniform(-0.2, 0.2, min(len(vals), 500))
        sample = vals[rng.choice(len(vals), min(len(vals), 500), replace=False)]
        ax.scatter(i + jitter, sample, color=col, s=6, alpha=0.3, zorder=3)
        ax.text(i, np.median(vals) + 1, f"n={len(vals):,}\nmed={np.median(vals):.0f}",
                ha="center", va="bottom", fontsize=FS - 2, color=col, fontweight="bold")

    ax.set_xticks(range(len(lifestyles)))
    ax.set_xticklabels(lifestyles, fontsize=FS)
    ax.set_ylabel("Age (years)", fontsize=FS)
    ax.set_title("Age distribution by lifestyle", fontsize=FS + 2, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    _savefig(fig, out_stem)


def plot_depth_by_lifestyle(meta: pd.DataFrame, out_stem: str):
    """Violin of sequencing depth (Total bases in Gb) by Lifestyle."""
    depth_col = None
    for c in ["Total bases", "Total_depth_bp", "Total_mapped_bp"]:
        if c in meta.columns:
            depth_col = c
            break
    if depth_col is None:
        print("  [SKIP] depth_by_lifestyle: no depth column found")
        return

    lifestyles = [lf for lf in ["H-G", "Transitional", "Industrial"]
                  if lf in meta["Lifestyle"].values]

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(42)

    for i, lf in enumerate(lifestyles):
        vals_raw = meta.loc[meta["Lifestyle"] == lf, depth_col].dropna().values
        vals_gb  = vals_raw / 1e9
        if len(vals_gb) < 4:
            continue
        log_vals = np.log10(np.maximum(vals_gb, 1e-6))
        col = LIFESTYLE_COLORS[lf]
        parts = ax.violinplot([log_vals], positions=[i], showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.55)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color(col)
        jitter = rng.uniform(-0.2, 0.2, min(len(log_vals), 500))
        sample = log_vals[rng.choice(len(log_vals), min(len(log_vals), 500), replace=False)]
        ax.scatter(i + jitter, sample, color=col, s=6, alpha=0.3, zorder=3)
        med_gb = 10**np.median(log_vals)
        ax.text(i, np.median(log_vals) + 0.05,
                f"n={len(vals_gb):,}\nmed={med_gb:.1f} Gb",
                ha="center", va="bottom", fontsize=FS - 2, color=col, fontweight="bold")

    # Log-scale y-axis labels in Gb
    ax.set_xticks(range(len(lifestyles)))
    ax.set_xticklabels(lifestyles, fontsize=FS)
    ax.set_ylabel("Sequencing depth (log₁₀ Gb)", fontsize=FS)
    # Relabel ticks in Gb
    nice_gb = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    tick_pos = [np.log10(v) for v in nice_gb]
    ymin, ymax = ax.get_ylim()
    in_range = [(p, str(v)) for p, v in zip(tick_pos, nice_gb)
                if ymin <= p <= ymax]
    if in_range:
        ax.set_yticks([p for p, _ in in_range])
        ax.set_yticklabels([l for _, l in in_range], fontsize=FS - 1)
        ax.set_ylabel("Sequencing depth (Gb)", fontsize=FS)
    ax.set_title("Sequencing depth by lifestyle", fontsize=FS + 2, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    _savefig(fig, out_stem)


def plot_samples_by_country(meta: pd.DataFrame, out_stem: str):
    """Bubble chart: one bubble per country, size = n_samples, colour = continent."""
    country_stats = (
        meta.groupby(["Country_norm", "Continent"])
        .agg(n_samples=("Lifestyle", "count"),
             pct_hg=("Lifestyle", lambda x: (x == "H-G").mean()),
             pct_trans=("Lifestyle", lambda x: (x == "Transitional").mean()),
             pct_ind=("Lifestyle", lambda x: (x == "Industrial").mean()))
        .reset_index()
        .sort_values("n_samples", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    y_pos = np.arange(len(country_stats))
    rng = np.random.default_rng(42)

    for i, row in enumerate(country_stats.itertuples()):
        col = CONTINENT_COLORS.get(row.Continent, OI["black"])
        # Stacked bar segments: H-G / Transitional / Industrial
        left = 0
        for lf, frac, lcol in [
            ("H-G",          row.pct_hg,    LIFESTYLE_COLORS["H-G"]),
            ("Transitional", row.pct_trans, LIFESTYLE_COLORS["Transitional"]),
            ("Industrial",   row.pct_ind,   LIFESTYLE_COLORS["Industrial"]),
        ]:
            w = row.n_samples * frac
            if w > 0:
                ax.barh(i, w, left=left, color=lcol,
                        edgecolor="white", lw=0.3, height=0.7)
                left += w
        # Country label
        ax.text(-20, i, row.Country_norm if row.Country_norm else "Unknown",
                ha="right", va="center", fontsize=max(FS - 4, 6))
        # n label
        ax.text(row.n_samples + 10, i, f"{row.n_samples:,}",
                ha="left", va="center", fontsize=max(FS - 4, 6))

    ax.set_yticks([])
    ax.set_xlabel("Number of samples", fontsize=FS)
    ax.set_title("Samples per country (coloured by Lifestyle composition)",
                 fontsize=FS + 2, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)

    legend_handles = [mpatches.Patch(facecolor=LIFESTYLE_COLORS[lf],
                                     label=LIFESTYLE_DISPLAY.get(lf, lf))
                      for lf in ["H-G", "Transitional", "Industrial"]]
    ax.legend(handles=legend_handles, fontsize=FS - 1,
              loc="lower right", framealpha=0.9)

    fig.set_size_inches(14, max(6, len(country_stats) * 0.28 + 1.5))
    _savefig(fig, out_stem)



def _find_ne_path():
    """Locate the Natural Earth 1:110m countries shapefile.

    The repo bundles it (figures/basemap/, see its README) and that copy is
    read first. The fallbacks below are what the pipeline used to rely on:
    pyogrio's *test fixture*, which exists or not depending on how pyogrio was
    installed -- a fresh pip environment could lack it, and Figure 1 then
    failed with nothing drawn. geodatasets' "naturalearth.land" has no
    country names, so it is not a usable fallback and is no longer tried.
    """
    import pathlib, sys

    # 0. The copy bundled with the repo
    bundled = (pathlib.Path(__file__).resolve().parent / "basemap"
               / "naturalearth_lowres" / "naturalearth_lowres.shp")
    if bundled.exists():
        return str(bundled)

    # 1. geopandas >= 1.0 removed datasets; try pyogrio bundled fixture
    for pkg in ["pyogrio", "geopandas"]:
        try:
            mod = __import__(pkg)
            base = pathlib.Path(mod.__file__).parent
            for candidate in base.rglob("naturalearth_lowres.shp"):
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass

    # 2. geopandas < 1.0 had a built-in datasets module
    try:
        import geopandas
        return geopandas.datasets.get_path("naturalearth_lowres")
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate the Natural Earth countries shapefile. It is bundled at "
        "figures/basemap/naturalearth_lowres/ -- check that directory came with "
        "the checkout or package."
    )

NE_PATH = None   # resolved lazily in _load_world()

# Shapefile 'name' column → pipeline canonical name
WORLD_ALIASES = {
    "Dem. Rep. Congo": "Democratic Republic of the Congo",
    "eSwatini":        "Swaziland",
    "S. Sudan":        "South Sudan",
    "W. Sahara":       "Western Sahara",
    "Cote d'Ivoire":  "Ivory Coast",
    "N. Korea":        "North Korea",
}


def _blend_white(hex_color, t):
    """Lighten hex_color toward white by factor t (0=original, 1=white)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r + (1-r)*t, g + (1-g)*t, b + (1-b)*t)


def _load_world():
    import geopandas as gpd
    path = _find_ne_path()
    world = gpd.read_file(path)
    world["name_norm"] = world["name"].replace(WORLD_ALIASES)
    return world


def _country_aggregates(meta):
    """Compute per-country aggregates from preprocessed meta DataFrame."""
    import math
    rows = []
    for country, g in meta.groupby("Country_norm", dropna=False):
        if not country or str(country) == "nan":
            continue
        n = len(g)
        depth_col = next((c for c in ["Total bases", "Total_depth_bp"] if c in g.columns), None)
        avg_depth = float(g[depth_col].mean()) if depth_col and not g[depth_col].isna().all() else float("nan")
        n_hg   = int((g["Lifestyle"] == "H-G").sum())
        n_tran = int((g["Lifestyle"] == "Transitional").sum())
        n_ind  = int((g["Lifestyle"] == "Industrial").sum())
        n_uh   = int((g.get("Unhealthy", pd.Series("", index=g.index)).astype(str).str.strip() == "Unhealthy").sum()) if "Unhealthy" in g.columns else 0
        n_h    = int((g.get("Unhealthy", pd.Series("", index=g.index)).astype(str).str.strip() == "Healthy").sum())  if "Unhealthy" in g.columns else 0

        # Dominant lifestyle (for map A hue)
        counts = {"H-G": n_hg, "Transitional": n_tran, "Industrial": n_ind}
        dom = max(counts, key=counts.get) if any(v > 0 for v in counts.values()) else None

        rows.append(dict(
            country_norm=str(country),
            n_samples=n,
            avg_depth=avg_depth,
            n_hg=n_hg, n_tran=n_tran, n_ind=n_ind,
            n_unhealthy=n_uh, n_healthy=n_h,
            dominant_lifestyle=dom,
            frac_unhealthy=n_uh / max(n_uh + n_h, 1),
        ))
    return pd.DataFrame(rows)


# ─── Map helpers (ported from map_metagenomes.py) ────────────────────────────

import warnings as _warnings
import math as _math
import numpy as _np
from matplotlib.colors import LogNorm as _LogNorm, LinearSegmentedColormap as _LSCmap
from matplotlib.cm import ScalarMappable as _ScalarMappable
from matplotlib.lines import Line2D as _L2D
from matplotlib import colors as _mcolors
from matplotlib.ticker import LogLocator as _LogLocator

_MAP_LIFESTYLE_COLORS = {
    "Industrial":   "#56B4E9",   # Sky blue
    "Transitional": "#F0E442",   # Yellow
    "H-G":          "#D55E00",   # Vermillion
}
_MAP_LIFESTYLE_HATCH = {
    "Industrial":   "///",
    "Transitional": "\\\\",
    "H-G":          "||",
}
_CB_LEFT_X = 0.12
_CB_WIDTH  = 0.014
_CB_HEIGHT = 0.18
_CB_GAP    = 0.014
_CB_BOTTOM = 0.52

def _fmt_sci(x):
    if not _np.isfinite(x) or x <= 0: return ""
    exp = int(_np.floor(_np.log10(x)))
    base = x / (10**exp)
    br = int(round(base)) if abs(base - round(base)) < 1e-3 else round(base, 1)
    sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"{br}×10{sup}" if not _np.isclose(br, 1.0, rtol=1e-3) else f"10{sup}"

def _fmt_pow10(x):
    if x <= 0 or not _np.isfinite(x): return ""
    exp = int(_np.floor(_np.log10(x)))
    base = x / (10**exp)
    sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    if _np.isclose(base, 1.0, rtol=1e-3): return f"10{sup}"
    br = int(round(base)) if abs(base - round(base)) < 1e-3 else round(base, 1)
    return f"{br}×10{sup}"

def _scale_bubbles(x, xmax, base=60.0):
    if not _np.isfinite(x) or xmax <= 0 or x <= 0: return 0.0
    return (_math.sqrt(x / float(xmax)) * base) ** 2

def _white_to_color_cmap(hex_color, name):
    rgb = _mcolors.to_rgb(hex_color)
    return _LSCmap.from_list(name, [(1,1,1), rgb], N=256)

def _make_lifestyle_facecolors(merged_df, values, dom_labels):
    """Shared LogNorm across all lifestyles (per_lifestyle=False)."""
    vs = _np.array(values, dtype=float)
    vs_pos = vs[_np.isfinite(vs) & (vs > 0)]
    vmin = float(vs_pos.min()) if vs_pos.size else 1.0
    vmax = float(vs_pos.max()) if vs_pos.size else 10.0
    if vmax <= vmin: vmax = vmin * 10.0
    shared_norm = _LogNorm(vmin=vmin, vmax=vmax)
    norms = {ls: shared_norm for ls in _MAP_LIFESTYLE_COLORS}
    vmins = {ls: vmin for ls in _MAP_LIFESTYLE_COLORS}
    vmaxes = {ls: vmax for ls in _MAP_LIFESTYLE_COLORS}

    fallback = _mcolors.to_rgba("#f2f2f2", 1.0)
    facecolors = []
    for i, r in merged_df.iterrows():
        ls = dom_labels[i]
        v  = values[i]
        if ls not in _MAP_LIFESTYLE_COLORS or not _np.isfinite(v) or v <= 0:
            facecolors.append(fallback); continue
        cmap = _white_to_color_cmap(_MAP_LIFESTYLE_COLORS[ls], f"w2{ls}")
        facecolors.append(_mcolors.to_rgba(cmap(shared_norm(v))))
    return facecolors, norms, vmins, vmaxes

def _add_lifestyle_colorbars(fig, norms, title_text, fontsize=11, fixed_ticks=None):
    order = ["Industrial", "Transitional", "H-G"]
    shared_norm = norms[order[0]]
    vmin, vmax = shared_norm.vmin, shared_norm.vmax
    for k, ls in enumerate(order):
        left   = _CB_LEFT_X
        bottom = _CB_BOTTOM - k*(_CB_HEIGHT + _CB_GAP)
        cax = fig.add_axes([left, bottom, _CB_WIDTH, _CB_HEIGHT])
        cmap = _white_to_color_cmap(_MAP_LIFESTYLE_COLORS[ls], f"w2{ls}_cb")
        sm = _ScalarMappable(norm=shared_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.tick_right()
        if fixed_ticks:
            ticks = [t for t in fixed_ticks if t > 0 and vmin <= t <= vmax]
            if len(ticks) < 2: ticks = [vmin, vmax]
        else:
            loc = _LogLocator(base=10, numticks=4)
            try:
                ticks = [t for t in loc.tick_values(vmin, vmax) if vmin <= t <= vmax]
                if len(ticks) < 2: ticks = [vmin, vmax]
            except Exception: ticks = [vmin, vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([_fmt_pow10(t) for t in ticks])
        cbar.ax.tick_params(labelsize=fontsize, pad=2)
        cbar.set_label(f"{ls}\n{title_text}", fontsize=fontsize+1, labelpad=10)

def _make_circle_legend(ax, values, title, loc="lower right",
                        fmt="{:,.0f}", fixed_values=None, base=80.0):
    vals = _np.array(values, dtype=float)
    vals = vals[_np.isfinite(vals) & (vals > 0)]
    if vals.size == 0: return
    vmax = vals.max()
    refs = [v for v in (fixed_values or []) if v > 0] or            _np.unique(_np.quantile(vals, _np.linspace(0.4,1,4))).tolist()
    handles, labels = [], []
    for r in refs:
        handles.append(plt.scatter([], [], s=_scale_bubbles(r, max(vmax, max(refs)), base=base),
                                   edgecolor="#333", facecolor="none", linewidths=1.0))
        labels.append(fmt(r) if callable(fmt) else fmt.format(r))
    leg = ax.legend(handles, labels, title=title, scatterpoints=1, frameon=True, loc=loc,
                    fontsize=11, title_fontsize=12, borderpad=0.6, labelspacing=0.5)
    leg._legend_box.align = "left"

def _draw_mixed_hatches(ax, merged, norms, value_cols_by_ls):
    """Draw primary fill + secondary hatches for multi-lifestyle countries."""
    import geopandas as gpd
    for idx, row in merged.iterrows():
        counts = {
            "Industrial":   row.get("l_ind",  0) or 0,
            "Transitional": row.get("l_tran", 0) or 0,
            "H-G":          row.get("l_hg",   0) or 0,
        }
        present = [ls for ls, c in counts.items() if c > 0]
        if len(present) < 2: continue
        order  = ["Industrial","Transitional","H-G"]
        primary = max(present, key=lambda ls: (counts[ls], -order.index(ls)))
        total   = sum(counts[ls] for ls in present)
        geom = gpd.GeoSeries([row.geometry], crs=merged.crs)
        # Primary fill
        vcol = value_cols_by_ls.get(primary)
        if vcol and vcol in row and _np.isfinite(row[vcol]) and row[vcol] > 0:
            cmap = _white_to_color_cmap(_MAP_LIFESTYLE_COLORS[primary], f"w2{primary}_p")
            base_col = cmap(norms[primary](row[vcol]))
        else:
            base_col = _MAP_LIFESTYLE_COLORS.get(primary, "#f2f2f2")
        geom.plot(ax=ax, color=base_col, edgecolor="#666666", linewidth=0.3)
        # Secondary hatches
        for ls in present:
            if ls == primary or counts[ls] <= 0: continue
            frac = counts[ls] / total
            vcol2 = value_cols_by_ls.get(ls)
            if vcol2 and vcol2 in row and _np.isfinite(row[vcol2]) and row[vcol2] > 0:
                cmap2 = _white_to_color_cmap(_MAP_LIFESTYLE_COLORS[ls], f"w2{ls}_s")
                sc = cmap2(norms[ls](row[vcol2]))
            else:
                sc = _MAP_LIFESTYLE_COLORS.get(ls, "#444444")
            geom.plot(ax=ax, facecolor=(0,0,0,0), edgecolor=sc,
                      hatch=_MAP_LIFESTYLE_HATCH.get(ls,"///"), linewidth=0.25+2.0*frac)


def _build_country_grp(meta):
    """Aggregate sample-level meta to per-country grp_country DataFrame."""
    depth_col = next((c for c in ["Total bases","Total_depth_bp"] if c in meta.columns), None)
    rows = []
    for country, g in meta.groupby("Country_norm", dropna=False):
        if not country or str(country) == "nan": continue
        n_total = len(g)
        avg_bases = float(g[depth_col].mean()) if depth_col else _np.nan
        total_bases = float(g[depth_col].sum()) if depth_col else _np.nan
        n_hg   = int((g["Lifestyle"] == "H-G").sum())
        n_ind  = int((g["Lifestyle"] == "Industrial").sum())
        n_tran = int((g["Lifestyle"] == "Transitional").sum())
        n_present = (n_hg > 0) + (n_ind > 0) + (n_tran > 0)
        counts = {"H-G": n_hg, "Industrial": n_ind, "Transitional": n_tran}
        dom = None
        if any(v > 0 for v in counts.values()):
            mv = max(counts.values())
            for opt in ["Industrial","Transitional","H-G"]:
                if counts[opt] == mv: dom = opt; break
        rows.append(dict(
            country_norm=str(country),
            total_samples=n_total,
            avg_bases_per_sample=avg_bases,
            total_bases=total_bases,
            # Backward-compatible aliases for older helper code. These values
            # are sequencing bases, not read counts.
            avg_reads_per_sample=avg_bases,
            total_reads=total_bases,
            l_hg=n_hg, l_ind=n_ind, l_tran=n_tran,
            n_lifestyles_present=n_present,
            dominant_lifestyle=dom,
        ))
    return pd.DataFrame(rows)


def plot_world_map_A(meta, out_stem):
    """Map A: hue=Lifestyle, log-shade=avg bases/sample, circles=#samples."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  [SKIP] world map A: geopandas not installed"); return

    world = _load_world()
    grp   = _build_country_grp(meta)

    unmatched = sorted(set(grp["country_norm"]) - set(world["name_norm"]))
    if unmatched: print(f"  [MAP-A] Unmatched ({len(unmatched)}): {unmatched}")

    merged = world.merge(grp, how="left", left_on="name_norm", right_on="country_norm")
    merged["centroid"] = merged.geometry.representative_point()

    dom       = merged["dominant_lifestyle"].fillna("NA")
    avg_bases = merged["avg_bases_per_sample"].fillna(_np.nan).to_numpy()
    nsamp     = merged["total_samples"].fillna(0).to_numpy(dtype=float)
    mixed_mask = (merged["n_lifestyles_present"].fillna(0) >= 2).to_numpy()

    faceA, normsA, vminsA, vmaxesA = _make_lifestyle_facecolors(merged, avg_bases, dom)
    sizesA = _np.array([_scale_bubbles(x, nsamp.max() if nsamp.max()>0 else 1, 65.0) for x in nsamp])

    figA, axA = plt.subplots(figsize=(14, 7), dpi=150)
    merged.plot(ax=axA, color="#f5f5f5", edgecolor="#666666", linewidth=0.3)
    # Non-mixed
    nm = ~mixed_mask
    merged[nm].plot(ax=axA, color=[c for c, k in zip(faceA, nm) if k],
                    edgecolor="#666666", linewidth=0.3)
    # Mixed
    _draw_mixed_hatches(axA, merged, normsA, {
        "Industrial":"avg_bases_per_sample","Transitional":"avg_bases_per_sample","H-G":"avg_bases_per_sample"})
    # Bubbles
    sel = merged.loc[nsamp > 0]
    axA.scatter(sel["centroid"].x, sel["centroid"].y, s=sizesA[nsamp>0],
                facecolor="none", edgecolor="#333", linewidth=0.9, alpha=0.85)
    # Colorbars
    _add_lifestyle_colorbars(figA, normsA, "bases\nper sample", 12, [3e9, 1e10, 3e10])
    # Lifestyle legend
    axA.legend(handles=[
        _L2D([0],[0], marker="s", color="w",
             label=LIFESTYLE_DISPLAY.get(ls, ls),
             markerfacecolor=_MAP_LIFESTYLE_COLORS[ls], markersize=12)
        for ls in ["Industrial","Transitional","H-G"]
    ], loc="lower left", frameon=True, fontsize=11)
    # Bubble legend
    _make_circle_legend(axA, nsamp, "# samples", "lower right", "{:,.0f}",
                        [10,30,100,300,1000], base=65.0)
    try: axA.get_legend().set_bbox_to_anchor((0.75, 0.07))
    except Exception: pass

    axA.set_title("Map A — Hue: Lifestyle • Log-shading: Avg bases per sample • Circles: # samples", pad=12)
    axA.axis("off")
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        figA.savefig(out_stem + ext, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_stem + ext}")
    plt.close(figA)


def plot_world_map_B(meta, out_stem):
    """Map B: hue=Lifestyle, log-shade=#samples, circles=total bases."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  [SKIP] world map B: geopandas not installed"); return

    world = _load_world()
    grp   = _build_country_grp(meta)

    unmatched = sorted(set(grp["country_norm"]) - set(world["name_norm"]))
    if unmatched: print(f"  [MAP-B] Unmatched ({len(unmatched)}): {unmatched}")

    merged = world.merge(grp, how="left", left_on="name_norm", right_on="country_norm")
    merged["centroid"] = merged.geometry.representative_point()

    dom       = merged["dominant_lifestyle"].fillna("NA")
    nsamp     = merged["total_samples"].fillna(0).to_numpy(dtype=float)
    totbases  = merged["total_bases"].fillna(0).to_numpy(dtype=float)
    mixed_mask = (merged["n_lifestyles_present"].fillna(0) >= 2).to_numpy()

    faceB, normsB, vminsB, vmaxesB = _make_lifestyle_facecolors(merged, nsamp, dom)
    sizesB = _np.array([_scale_bubbles(x, totbases.max() if totbases.max()>0 else 1, 75.0) for x in totbases])

    figB, axB = plt.subplots(figsize=(14, 7), dpi=150)
    merged.plot(ax=axB, color="#f5f5f5", edgecolor="#666666", linewidth=0.3)
    nm = ~mixed_mask
    merged[nm].plot(ax=axB, color=[c for c, k in zip(faceB, nm) if k],
                    edgecolor="#666666", linewidth=0.3)
    _draw_mixed_hatches(axB, merged, normsB,
                        {"Industrial":"l_ind","Transitional":"l_tran","H-G":"l_hg"})
    sel = merged.loc[totbases > 0]
    axB.scatter(sel["centroid"].x, sel["centroid"].y, s=sizesB[totbases>0],
                facecolor="none", edgecolor="#222", linewidth=0.9, alpha=0.88)
    _add_lifestyle_colorbars(figB, normsB, "# samples", 12)
    axB.legend(handles=[
        _L2D([0],[0], marker="s", color="w",
             label=LIFESTYLE_DISPLAY.get(ls, ls),
             markerfacecolor=_MAP_LIFESTYLE_COLORS[ls], markersize=12)
        for ls in ["Industrial","Transitional","H-G"]
    ], loc="lower left", frameon=True, fontsize=11)
    _make_circle_legend(axB, totbases, "Total bases", "lower right", _fmt_sci,
                        [1e11,3e11,1e12,3e12,1e13], base=75.0)
    try: axB.get_legend().set_bbox_to_anchor((0.75, 0.07))
    except Exception: pass

    axB.set_title("Map B — Hue: Lifestyle • Log-shading: # samples • Circles: Total bases (country sum)", pad=12)
    axB.axis("off")
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        figB.savefig(out_stem + ext, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_stem + ext}")
    plt.close(figB)


def _default_meta() -> str:
    """Metadata file from config, so a repointed cohort reaches this script.

    This was hard-coded to metadata_Mar_16_2026.csv, which silently kept the
    pre-2026-08-20 cohort while the analysis ran on the newer metadata.
    """
    import persistence.config as _c
    return _c.META_FILE


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta",
        default=_default_meta(),
        help="Input metadata CSV (default: config.META_FILE)")
    ap.add_argument("--secondary",
        default=None,
        help="Optional secondary-run BC screening CSV (secondary_bc_screening.csv)")
    ap.add_argument("--out-dir", dest="out_dir",
        default="metadata_summary",
        help="Output directory (default: metadata_summary/)")
    ap.add_argument("--cohort-ref", dest="cohort_ref", default=COHORT_REF,
                    help=("Reference whose analytic cohort to report (default: "
                          f"{COHORT_REF}). Reads results/<ref>/cohort_samples.csv."))
    ap.add_argument("--dpi", type=int, default=180,
        help="Figure DPI (default: 180)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Preprocessing ──────────────────────────────────────────────────────
    meta = load_and_preprocess(args.meta, args.secondary, args.cohort_ref)

    # ── Save annotated full table ──────────────────────────────────────────
    ann_path = os.path.join(args.out_dir, "metadata_annotated.csv")
    meta.to_csv(ann_path)
    print(f"  Annotated metadata: {ann_path}  ({len(meta):,} rows)")

    # ── Summary CSV ────────────────────────────────────────────────────────
    make_summary_csv(meta, os.path.join(args.out_dir, "samples_summary_by_study.csv"))

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")

    plot_lifestyle_by_continent(
        meta, os.path.join(args.out_dir, "fig_lifestyle_continent"))

    plot_health_by_lifestyle(
        meta, os.path.join(args.out_dir, "fig_health_by_lifestyle"))

    plot_age_distribution(
        meta, os.path.join(args.out_dir, "fig_age_distribution"))

    plot_depth_by_lifestyle(
        meta, os.path.join(args.out_dir, "fig_depth_by_lifestyle"))

    plot_samples_by_country(
        meta, os.path.join(args.out_dir, "fig_samples_by_country"))

    plot_world_map_A(
        meta, os.path.join(args.out_dir, "metagenome_world_map_A_full"))

    plot_world_map_B(
        meta, os.path.join(args.out_dir, "metagenome_world_map_B_full"))

    print("\nDone.")


if __name__ == "__main__":
    main()
