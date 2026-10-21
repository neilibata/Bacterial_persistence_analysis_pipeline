"""
Metadata visualization: country summaries and interactive world map.

Generates:
  - country_summary.csv   — per-country sample counts, demographics, disease fractions
  - world_map.html        — interactive Plotly choropleth with variable dropdown
  - metadata_sanity.txt   — raw distribution diagnostics for key columns
"""

import os
import numpy as np
import pandas as pd

from . import config
from .common import normalise_country, is_yes


# ════════════════════════════════════════════════════════════════════
#  METADATA DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════

def print_metadata_diagnostics(meta, out_dir):
    """Print and save raw distribution diagnostics for key metadata columns."""
    # The roll-up and second-opinion health columns were removed from the
    # metadata: only `Unhealthy` is read by the analysis, and the rest were a
    # standing invitation to compare against the wrong one.
    diag_cols = [
        "Gender", "Unhealthy", "Obesity", "Overweight",
    ]

    lines = ["=== RAW METADATA DISTRIBUTIONS ===\n"]
    for c in diag_cols:
        if c not in meta.columns:
            lines.append(f"[WARN] Column missing: {c}\n")
            continue
        lines.append(f"\n---- {c} ----")
        lines.append(str(meta[c].value_counts(dropna=False)))

    report = "\n".join(lines)
    print(report[:2000])  # truncate for console

    fpath = os.path.join(out_dir, "metadata_sanity.txt")
    with open(fpath, "w") as f:
        f.write(report)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  COUNTRY SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════

def build_country_summary(meta):
    """
    Build a per-country summary DataFrame with:
      - n_samples, total_bases, mean_bases_per_samp
      - mean_age, mean_BMI, frac_female
      - frac_healthy, frac_unhealthy
      - frac_gut_metab_disease, frac_other_disease, frac_metab_or_obese
    """
    df = meta.copy()
    df["country_norm"] = df["Country"].apply(normalise_country)

    # Total bases (best available depth proxy)
    depth_col = None
    for c in ["Total_depth_bp", "Total_mapped_bp", "Total bases"]:
        if c in df.columns:
            depth_col = c
            break
    if depth_col is None:
        df["_total_bases"] = 0
    else:
        df["_total_bases"] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0)

    rows = []
    for country, grp in df.groupby("country_norm", dropna=False):
        if pd.isna(country) or country is None:
            continue

        n = len(grp)
        total_bases = grp["_total_bases"].sum()
        mean_bases  = grp["_total_bases"].mean()

        mean_age = grp["Age"].mean() if "Age" in grp.columns else np.nan
        mean_bmi = grp["BMI"].mean() if "BMI" in grp.columns else np.nan

        frac_female = np.nan
        if "Gender" in grp.columns:
            gen = grp["Gender"].astype(str).str.strip().str.lower()
            n_female = gen.isin(["female", "f"]).sum()
            frac_female = n_female / n if n > 0 else np.nan

        # Health fractions
        def _frac_yes(col):
            if col not in grp.columns:
                return np.nan
            return grp[col].apply(is_yes).sum() / n if n > 0 else np.nan

        frac_healthy = np.nan
        if "Unhealthy" in grp.columns:
            frac_healthy = (grp["Unhealthy"] == "Healthy").sum() / n
        frac_unhealthy = np.nan
        if "Unhealthy" in grp.columns:
            frac_unhealthy = (grp["Unhealthy"] == "Unhealthy").sum() / n

        rows.append({
            "country_norm": country,
            "n_samples": n,
            "total_bases": total_bases,
            "mean_bases_per_samp": mean_bases,
            "mean_age": mean_age,
            "mean_BMI": mean_bmi,
            "frac_female": frac_female,
            "frac_healthy": frac_healthy,
            "frac_unhealthy": frac_unhealthy,
            # These three are the only place in the analysis code that reads
            # a `Diseased*` column, and they are descriptive only: they
            # colour the per-country diagnostic map and CSV, neither of which
            # is a shipped figure or table.
            #
            # The statistical contrast is `Unhealthy` everywhere -- Healthy
            # vs Unhealthy, via health_cmh.build_group_masks. Nothing in
            # Sections 1-3, no figures/ script, and no table builder consults
            # a `Diseased*` column. Do not use one as a case/control split:
            # the roll-ups predate the curated labels and disagree with
            # `Unhealthy` on 1,848 samples (1,668 of them Unhealthy with no
            # disease flag at all, mostly the BMI >= 30 rule).
            "frac_gut_metab_disease": _frac_yes("Diseased (gut and metabolic)"),
            "frac_other_disease": _frac_yes("Diseased (other)"),
            "frac_metab_or_obese": _frac_yes("Diseased (metabolic) or obese"),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
#  INTERACTIVE WORLD MAP (Plotly)
# ════════════════════════════════════════════════════════════════════

def build_world_map(summary_df, out_html):
    """
    Create an interactive Plotly choropleth map with dropdown to switch
    between variables (sample counts, depth, demographics, disease fracs).

    Requires: plotly (optional dependency, will skip gracefully).
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("    [WARN] plotly not installed — skipping world map")
        return

    df = summary_df.copy()
    df["log10_n_samples"]   = np.log10(df["n_samples"].clip(lower=1))
    df["log10_total_bases"] = np.log10(df["total_bases"].clip(lower=1))

    # Variable definitions for dropdown
    var_defs = {
        "log10_n_samples": {
            "label": "Number of samples (log₁₀)",
            "tickvals": [0, 1, 2, 3, 4],
            "ticktext": ["1", "10", "100", "1000", "10000"],
        },
        "log10_total_bases": {
            "label": "Total bases (log₁₀)",
            "tickvals": [9, 10, 11, 12, 13],
            "ticktext": ["1e9", "1e10", "1e11", "1e12", "1e13"],
        },
        "mean_age": {"label": "Mean age"},
        "mean_BMI": {"label": "Mean BMI"},
        "frac_female": {"label": "Fraction female"},
        "frac_healthy": {"label": "Fraction healthy"},
        "frac_unhealthy": {"label": "Fraction unhealthy"},
        "frac_gut_metab_disease": {"label": "Fraction gut/metabolic disease"},
        "frac_other_disease": {"label": "Fraction other disease"},
        "frac_metab_or_obese": {"label": "Fraction metabolic/obese"},
    }

    # Build all choropleth traces
    fig = go.Figure()
    first = True
    for var_col, vdef in var_defs.items():
        if var_col not in df.columns:
            continue
        trace = go.Choropleth(
            locations=df["country_norm"],
            locationmode="country names",
            z=df[var_col],
            text=df["country_norm"],
            colorscale="Viridis",
            colorbar=dict(title=vdef["label"]),
            visible=first,
            name=vdef["label"],
        )
        fig.add_trace(trace)
        first = False

    # Dropdown buttons
    buttons = []
    var_keys = [k for k in var_defs if k in df.columns]
    for i, var_col in enumerate(var_keys):
        vis = [False] * len(var_keys)
        vis[i] = True
        buttons.append(dict(
            label=var_defs[var_col]["label"],
            method="update",
            args=[{"visible": vis},
                  {"title": var_defs[var_col]["label"]}],
        ))

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down",
                          x=0.17, xanchor="left", y=1.15, yanchor="top")],
        title_text="Microbiome Dataset — Country-level Summary",
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"    Saved: {out_html}")


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_visualization(data):
    """
    Generate metadata visualizations for one reference database.

    Produces:
      - metadata_sanity.txt
      - country_summary_{ref}.csv
      - world_map_{ref}.html (if plotly available)
    """
    ref = data.ref_name
    out = config.outdir(ref)
    meta = data.meta

    print(f"\n  [{ref}] Metadata visualization")

    # Diagnostics
    print_metadata_diagnostics(meta, out)

    # Country summary
    summary = build_country_summary(meta)
    csv_path = os.path.join(out, f"country_summary_{ref}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}  ({len(summary)} countries)")

    # Interactive map
    html_path = os.path.join(out, f"world_map_{ref}.html")
    build_world_map(summary, html_path)

    return summary
