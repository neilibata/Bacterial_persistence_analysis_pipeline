#!/usr/bin/env python3
"""
rarefaction_plugin.py
─────────────────────
Replaces the Hill-MM fitted accumulation curve with the nonparametric
plug-in rarefaction estimator:

    E[C(N)] = sum_g  1 - (1 - p_g)^N

where  p_g = fraction of samples in the group in which SGB g is detected
             (breadth >= threshold).

Analytical variance (independence approximation):

    Var[C(N)] ≈ sum_g  (1 - p_g)^N  *  (1 - (1 - p_g)^N)

Extrapolation beyond N_max uses the Chao et al. (2014) formula:

    E[C(N_max + m)] = S_obs + f0_hat * (1 - (1 - f1/(N_max*f0_hat + f1))^m)

where  f0_hat = f1^2 / (2*f2)  (Chao1 unseen-species estimate),
and f1, f2 are the expected frequencies-of-frequencies derived from p_g.

Cohort
──────
This script builds its own cohort. It does NOT reproduce the pipeline's, and
earlier revisions of this docstring wrongly claimed that it did. What it
applies (`load_samples`):
  - Breadth matrix indexed by Chosen_match_ID
  - Depth matrix used to compute total_bp = sum(mean_depth * genome_length)
  - Samples filtered to those present in the breadth matrix index
  - Lifestyle.notna()
  - Minimum depth filter: total_bp >= MIN_TOTAL_BP (local copy of 30 Mb)
  - Study exclusion (local copy of config.EXCLUDE_STUDIES)
  - Presence/absence: breadth_df >= threshold  (breadth_pct mode)

What `loading.py:load_reference` applies and this script does NOT:
  - config.DROP_UNCLASSIFIED_LIFESTYLE -- the 27 "Unclear" rows are kept here
  - config.EXCLUDE_SAMPLES -- the 104 amplicon libraries of
    docs/amplicon_flagged_recommended.csv are kept here
  - config.ONE_SAMPLE_PER_BIOSAMPLE -- the 17 extra libraries of an
    already-kept BioSample are kept here
  - config.MIN_SGBS_DETECTED (>= 5 SGBs at MIN_SGBS_BREADTH_PCT) -- which
    config.py:157-166 records as removing a further ~55 samples that the
    amplicon list does not catch

So this cohort is roughly 200 samples larger than the analytic cohort
(results/<ref>/cohort_samples.csv). Counted against final_run/metadata_M1.csv
on 2026-10-07: 27 + 104 + 17 = 148 metadata-derivable additions, none of which
appear in any reference's cohort_samples.csv, plus the ~55 removed by the
min-SGBs floor alone.

Of those, the 27 "Unclear" rows are harmless to the drawn curves: every series
is selected by `Group_fine`, and classify_group returns None for them, so they
are counted in the cohort line this function prints but enter no curve. The
other ~176 do enter the curves, and they are the worst possible samples for
this particular estimator: the amplicon libraries clear 30 Mb while detecting
almost nothing (config.py:157-166), so they depress every per-SGB detection
fraction p_g, and therefore every curve, every analytical variance band and
every Chao f1/f2 extrapolation built from p_g.

The numbers this script draws are therefore NOT comparable with any pipeline
output, and nothing published depends on it: its only consumer is the
"rarefaction" task in persistence/figure_runner.py, whose four PNGs
(rarefaction_plugin_{UHGG,lev,combined,chimp}_10p0pct.png) appear in neither
tools/paper_manifest.csv, tools/additional_figures_manifest.txt,
tools/figure_sets.py nor verification_manifest.json, and it writes no CSV.
Treat it as a standalone diagnostic. Before any output of it is quoted, make
`load_samples` read the analytic cohort instead -- the cheapest correct fix is
to intersect its index with results/<ref>/cohort_samples.csv, which is the
post-exclusion, post-reinstatement cohort written by
persistence/cohort_manifest.py and is not derivable from metadata alone.

Usage
─────
Run from the directory that contains the breadth/depth/metadata files:

    python rarefaction_plugin.py --ref UHGG --threshold 10

Or with explicit paths:

    python rarefaction_plugin.py \\
        --breadth meancoverage_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv \\
        --depth   meandepth_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv \\
        --lengths all_SGB_lengths_UHGG.txt \\
        --meta    metadata_Mar_16_2026.csv \\
        --csv     results/UHGG/accum_pooled_fits_vsN_cov10p0.csv \\
        --threshold 10 --ref UHGG --out rarefaction_UHGG_10pct.png

    # With Chao extrapolation out to N_extrap samples
    python rarefaction_plugin.py --ref UHGG --threshold 10 --extrap 30000
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
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    def _display_text(text: str) -> str:
        return str(text)


def _apply_display_text(fig):
    for artist in fig.findobj(match=matplotlib.text.Text):
        old = artist.get_text()
        new = _display_text(old)
        if new != old:
            artist.set_text(new)
    return fig

# ── Okabe-Ito palette (matches pipeline) ─────────────────────────────────────
OI = {
    "sky":        "#56B4E9",
    "orange":     "#E69F00",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
}

SERIES_CONFIG = [
    ("Industrial (all)",
     {"Industrial", "Industrial Unhealthy"},
     OI["sky"],        "-"),
    ("Non-Industrial (all)",
     {"Non-Industrial", "Non-Industrial Unhealthy"},
     OI["orange"],     "-"),
    ("Industrial Unhealthy",
     {"Industrial Unhealthy"},
     OI["blue"],       "--"),
    ("Non-Industrial Unhealthy",
     {"Non-Industrial Unhealthy"},
     OI["vermillion"], "--"),
]

# A literal COPY of config.EXCLUDE_STUDIES, not a read of it, so it can drift.
# Verified identical to config.py:314-322 on 2026-10-07 (7 prefixes, same set);
# STUDY_EXCLUSION_COLS below likewise matches config.STUDY_EXCLUSION_COLS.
# `_cfg` is already imported above, so `_cfg.EXCLUDE_STUDIES` /
# `_cfg.STUDY_EXCLUSION_COLS` / `_cfg.MIN_TOTAL_BP` would remove the drift.
EXCLUDE_STUDIES = frozenset({
    "Bengtsson-PalmeJ", "DhakanDB", "FerrettiP",
    "Heintz-BuschartA", "KaurK", "UKN", "eKmeta",
})
STUDY_EXCLUSION_COLS = [
    "StudyFolder",
    "study_name [MBIOTEMP_study_name]",
    "Author",
]

# Another literal copy; equal to config.MIN_TOTAL_BP on 2026-10-07. Note that
# this is the ONLY one of the pipeline's two depth-related floors reproduced
# here -- config.MIN_SGBS_DETECTED is not applied anywhere in this file.
MIN_TOTAL_BP = 3e7   # 30 Mb


# ── Group classification (mirrors common.py exactly) ──────────────────────────

def classify_group(row):
    uh   = row.get("Unhealthy", None)
    life = row.get("Lifestyle",  None)
    if uh == "Unhealthy":
        if life in ("H-G", "Transitional", "Transitional"):
            return "Non-Industrial Unhealthy"
        if life == "Industrial":
            return "Industrial Unhealthy"
    elif uh == "Healthy":
        if life in ("H-G", "Transitional", "Transitional"):
            return "Non-Industrial"
        if life == "Industrial":
            return "Industrial"
    return None


# ── File auto-discovery ───────────────────────────────────────────────────────

# Derived from config.REFERENCES rather than hard-coded. The private copy went
# stale at the 2026-08-20 repointing and had no "combined" entry at all, so the
# pipeline's `--ref combined` call died on an argparse invalid-choice error.
REFS = {
    name: {
        "breadth":  spec["breadth_file"],
        "depth":    spec["depth_file"],
        "lengths":  spec["lengths_file"],
    }
    for name, spec in _cfg.REFERENCES.items()
}
META_CANDIDATES = list(_cfg.META_FILE_CANDIDATES)

def _find_file(name, search_dirs):
    for d in search_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def _find_meta(search_dirs):
    for d in search_dirs:
        for c in META_CANDIDATES:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p
    return None

def _find_sweep_csv(threshold, search_dirs):
    thr_label = f"{threshold:.1f}".replace(".", "p")
    for d in search_dirs:
        p = os.path.join(d, f"accum_pooled_fits_vsN_cov{thr_label}.csv")
        if os.path.exists(p):
            return p
    return None


# ── Matrix loading (mirrors loading.py: _load_matrix) ────────────────────────

def _load_matrix(path, tag):
    print(f"    Loading {tag}: {os.path.basename(path)}")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str).str.strip()
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if mean_cols:
        df = df[mean_cols].copy()
        df.columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]
    df = df.fillna(0.0)
    print(f"      {df.shape[0]:,} samples x {df.shape[1]:,} SGBs")
    return df


# ── Full sample loading — a REDUCED re-implementation of loading.py ──────────

def load_samples(breadth_path, depth_path, lengths_path, meta_path):
    """
    Load and align breadth matrix, depth matrix, genome lengths, and metadata.

    Applies a SUBSET of the pipeline's sample-level filters -- see the "Cohort"
    section of the module docstring for the full accounting and for what is
    missing. In short, this applies:
      - Chosen_match_ID join against the breadth-matrix index
      - Lifestyle.notna()
      - Minimum total_bp (30 Mb)
      - Study exclusion
    and does NOT apply DROP_UNCLASSIFIED_LIFESTYLE, EXCLUDE_SAMPLES,
    ONE_SAMPLE_PER_BIOSAMPLE or MIN_SGBS_DETECTED. The resulting cohort is
    roughly 200 samples larger than results/<ref>/cohort_samples.csv.

    Returns
    -------
    breadth_df : DataFrame (filtered samples x SGBs), values = % breadth
    meta       : DataFrame (filtered), indexed by Chosen_match_ID,
                 with 'Group_fine' column added
    """
    breadth_df = _load_matrix(breadth_path, "breadth")
    depth_df   = _load_matrix(depth_path,   "depth")

    common_sgbs = sorted(set(breadth_df.columns) & set(depth_df.columns))
    breadth_df  = breadth_df[common_sgbs]
    depth_df    = depth_df[common_sgbs]
    print(f"    Common SGBs (breadth + depth): {len(common_sgbs):,}")

    print(f"    Loading genome lengths: {os.path.basename(lengths_path)}")
    len_df = pd.read_csv(lengths_path, sep=r"\s+", header=None,
                         names=["microbe", "length"])
    len_df["microbe"] = len_df["microbe"].str.strip()
    length_dict = dict(zip(len_df["microbe"], len_df["length"]))
    sgbs_with_len = [s for s in common_sgbs if s in length_dict]
    n_drop = len(common_sgbs) - len(sgbs_with_len)
    if n_drop:
        print(f"    Dropped {n_drop} SGBs with no genome length")
    breadth_df = breadth_df[sgbs_with_len]
    depth_df   = depth_df[sgbs_with_len]
    lengths    = pd.Series({s: length_dict[s] for s in sgbs_with_len})

    # total_bp = sum(mean_depth * genome_length) — mirrors loading.py exactly
    total_bp = depth_df.mul(lengths, axis=1).sum(axis=1)

    # Metadata — join on Chosen_match_ID (must match breadth matrix row index)
    print(f"    Loading metadata: {os.path.basename(meta_path)}")
    meta = pd.read_csv(meta_path, low_memory=False)
    # Table M1 stores the manuscript's display labels (Agro-pastoralist,
    # Hunter-gatherer); analysis code compares against the internal keys.
    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = _cfg.canonicalize_lifestyle_series(meta["Lifestyle"])
    if "Chosen_match_ID" not in meta.columns:
        sys.exit(
            "ERROR: 'Chosen_match_ID' column not found in metadata.\n"
            "Make sure you are using the correct metadata file (e.g. "
            "metadata_Mar_16_2026.csv)."
        )
    meta["Chosen_match_ID"] = meta["Chosen_match_ID"].astype(str).str.strip()
    meta = meta[meta["Chosen_match_ID"].isin(breadth_df.index)].copy()
    # keep="first" is row order. loading.py:309-320 instead keeps the row with
    # the largest "Total bases", because the matrices carry the larger
    # sequencing run and keep="first" retained the smaller in 20 of 21 cases.
    # Inert against the current metadata -- final_run/metadata_M1.csv has no
    # duplicated Chosen_match_ID (checked 2026-10-07) -- but it is a latent
    # divergence if a metadata file with ingestion duplicates is passed.
    meta = meta.drop_duplicates(subset="Chosen_match_ID", keep="first")
    meta = meta.set_index("Chosen_match_ID")
    print(f"    Samples after Chosen_match_ID join: {len(meta):,}")

    # Align matrices to meta index
    breadth_df = breadth_df.loc[meta.index]
    depth_df   = depth_df.loc[meta.index]
    total_bp   = total_bp.loc[meta.index]
    meta["Total_depth_bp"] = total_bp

    # No-op, left from the relabelling that renamed the old lifestyle key to
    # "Transitional": both arguments are now the same string. The real
    # normalisation is _cfg.canonicalize_lifestyle_series above.
    meta["Lifestyle"] = meta["Lifestyle"].replace("Transitional", "Transitional")

    # Group classification
    meta["Group_fine"] = meta.apply(classify_group, axis=1)

    # Drop samples with no Lifestyle. NOTE: notna() only. loading.py:411-419
    # additionally drops every Lifestyle that does not resolve to
    # Industrial/Non-Industrial (config.DROP_UNCLASSIFIED_LIFESTYLE), i.e. the
    # 27 "Unclear" rows, which this keeps. classify_group below gives them a
    # Group_fine of None, so they fall out of every drawn series but are still
    # counted in the cohort printed at the end of this function.
    valid = meta["Lifestyle"].notna()
    meta       = meta[valid]
    breadth_df = breadth_df.loc[meta.index]
    total_bp   = total_bp.loc[meta.index]

    # Depth filter
    keep = total_bp >= MIN_TOTAL_BP
    n_drop_depth = int((~keep).sum())
    if n_drop_depth:
        print(f"    Dropped {n_drop_depth:,} samples below "
              f"{MIN_TOTAL_BP/1e6:.0f} Mb depth threshold")
    meta       = meta[keep]
    breadth_df = breadth_df.loc[meta.index]
    total_bp   = total_bp.loc[meta.index]

    # Study exclusion (mirrors _apply_study_exclusion in loading.py)
    excl_mask = pd.Series(False, index=meta.index)
    for col in STUDY_EXCLUSION_COLS:
        if col not in meta.columns:
            continue
        s = meta[col].astype(str).str.strip()
        if col == "Author":
            excl_mask |= s.isin(EXCLUDE_STUDIES)
        else:
            for prefix in EXCLUDE_STUDIES:
                excl_mask |= s.str.startswith(prefix)
    n_excl = int(excl_mask.sum())
    if n_excl:
        print(f"    Study exclusion: {n_excl:,} samples removed")
    meta       = meta[~excl_mask]
    breadth_df = breadth_df.loc[meta.index]

    print(f"    Final: {len(meta):,} samples")
    for grp, cnt in meta["Group_fine"].value_counts(dropna=False).items():
        print(f"      {grp}: {cnt:,}")

    return breadth_df, meta


# ── Core: plug-in rarefaction estimator ──────────────────────────────────────

def plugin_curve(p_g, N_values):
    """
    E[C(N)] = sum_g 1 - (1-p_g)^N   (computed in log-space for numerical stability)
    Var[C(N)] approx sum_g (1-p_g)^N * (1-(1-p_g)^N)   (independence approx)
    """
    p_g      = np.asarray(p_g, dtype=np.float64)
    N_values = np.asarray(N_values, dtype=np.float64)
    log1mp       = np.log1p(-np.clip(p_g, 0.0, 1.0 - 1e-15))
    log_survival = np.outer(N_values, log1mp)
    survival     = np.exp(log_survival)
    E_C   = np.sum(1.0 - survival, axis=1)
    var_C = np.sum(survival * (1.0 - survival), axis=1)
    return E_C, np.sqrt(np.maximum(var_C, 0.0))


# ── Chao et al. (2014) extrapolation ─────────────────────────────────────────

def chao_extrapolation(p_g, N_max, m_values):
    """
    Chao1-based extrapolation using expected f1, f2 from the binomial model.
    """
    p_g      = np.asarray(p_g, dtype=np.float64)
    N        = int(N_max)
    m_values = np.asarray(m_values, dtype=np.float64)
    f1_hat   = float(np.sum(N * p_g * (1.0 - p_g) ** (N - 1)))
    f2_hat   = float(np.sum((N * (N - 1) / 2.0) * p_g**2 * (1.0 - p_g) ** (N - 2)))
    S_obs    = float(np.sum(1.0 - (1.0 - p_g) ** N))
    f0_hat   = (f1_hat**2 / (2.0 * f2_hat)) if f2_hat > 1e-9 else f1_hat
    if f0_hat < 1e-9 or f1_hat < 1e-9:
        return S_obs + np.zeros_like(m_values)
    lam = f1_hat / (N * f0_hat + f1_hat)
    return S_obs + f0_hat * (1.0 - (1.0 - lam) ** m_values)


# ── Per-series computation ────────────────────────────────────────────────────

def compute_series(breadth_df, meta, groups, threshold, N_values, N_max_extrap=None):
    mask       = meta["Group_fine"].isin(groups)
    common_ids = breadth_df.index.intersection(meta.index[mask])
    if len(common_ids) < 5:
        return None

    block = breadth_df.loc[common_ids]
    N_max = len(common_ids)
    print(f"    n={N_max:,} samples, {block.shape[1]:,} SGBs")

    # Presence/absence — exactly as pipeline breadth_pct mode
    p_g = (block >= threshold).mean(axis=0).values
    p_g = p_g[p_g > 0.0]

    N_interp  = N_values[N_values <= N_max]
    E_C, SD_C = plugin_curve(p_g, N_interp)

    result = dict(N_max=N_max, p_g=p_g,
                  N_interp=N_interp, E_C=E_C, SD_C=SD_C)

    if N_max_extrap is not None and N_max_extrap > N_max:
        n_m      = max(int((N_max_extrap - N_max) / N_max * 60), 30)
        m_values = np.linspace(0, N_max_extrap - N_max, n_m)
        result["N_extrap"] = N_max + m_values
        result["E_extrap"] = chao_extrapolation(p_g, N_max, m_values)

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plot(series_results, sweep_df, threshold, ref, out_path, N_max_extrap=None):
    fig, ax = plt.subplots(figsize=(13, 8))

    for name, groups, color, ls in SERIES_CONFIG:
        res = series_results.get(name)
        if res is None:
            continue

        # Hill-MM overlay from sweep CSV
        if sweep_df is not None and not sweep_df.empty:
            sub = sweep_df[sweep_df["Series"] == name].sort_values("N_max")
            if not sub.empty:
                last = sub.iloc[-1]

                def _hill(N, a, b, c, n_mic):
                    a_frac = np.clip(a / n_mic, 1e-9, 1 - 1e-9)
                    a_raw  = np.log(a_frac / (1 - a_frac))
                    sigma  = 1.0 / (1.0 + np.exp(-np.clip(a_raw, -20, 20)))
                    return sigma * np.power(N, c) / (np.power(b, c) + np.power(N, c)) * n_mic

                N_fit  = np.logspace(np.log10(max(sub["N_max"].min(), 3)),
                                     np.log10(sub["N_max"].max()), 400)
                y_hill = _hill(N_fit, last["a_abs_central"], last["b_central"],
                               last["c_central"], last["n_microbes"])
                ax.plot(N_fit, y_hill, color=color, ls=ls, lw=1.8, alpha=0.45,
                        label=f"{name} Hill-MM")

                N_sw  = sub["N_max"].values.astype(float)
                n_mic = sub["n_microbes"].values.astype(float)
                y_q16 = _hill(N_sw, sub["a_q16_boot"].values,
                               sub["b_q50_boot"].values, sub["c_q50_boot"].values, n_mic)
                y_q84 = _hill(N_sw, sub["a_q84_boot"].values,
                               sub["b_q50_boot"].values, sub["c_q50_boot"].values, n_mic)
                ax.fill_between(N_sw, y_q16, y_q84, color=color, alpha=0.10, linewidth=0)

        # Plug-in curve
        ax.plot(res["N_interp"], res["E_C"], color=color, ls=ls, lw=2.5,
                label=f"{name} plug-in")
        ax.fill_between(res["N_interp"],
                        res["E_C"] - res["SD_C"], res["E_C"] + res["SD_C"],
                        color=color, alpha=0.20, linewidth=0)

        # Chao extrapolation
        if "N_extrap" in res:
            ax.plot(res["N_extrap"], res["E_extrap"],
                    color=color, ls=":", lw=2.0, alpha=0.80,
                    label=f"{name} Chao extrap.")

    ax.set_xscale("log")
    ax.set_xlabel("Number of samples N (log scale)", fontsize=14)
    ax.set_ylabel(f"SGBs detected (>={threshold:.0f}% breadth)", fontsize=14)
    ax.set_title(
        f"Plug-in rarefaction vs Hill-MM  --  {ref}  |  {threshold:.0f}% breadth\n"
        "Bold = plug-in estimator  |  Faded = Hill-MM (last sweep fit)  "
        "|  Shading = +/-1 SD analytical (independence)  |  Dotted = Chao extrap.",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85, ncol=2)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)
    plt.tight_layout()
    _apply_display_text(fig)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref",       default="UHGG", choices=list(REFS.keys()),
                   help="Reference database key (default: UHGG)")
    p.add_argument("--breadth",   default=None,
                   help="Breadth coverage matrix CSV (auto-detected if omitted)")
    p.add_argument("--depth",     default=None,
                   help="Depth matrix CSV (auto-detected if omitted)")
    p.add_argument("--lengths",   default=None,
                   help="Genome lengths file (auto-detected if omitted)")
    p.add_argument("--meta",      default=None,
                   help="Metadata CSV (auto-detected if omitted)")
    p.add_argument("--csv",       default=None,
                   help="accum_pooled_fits_vsN_cov*.csv for Hill-MM overlay")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="Breadth detection threshold in %% (default: 10.0)")
    p.add_argument("--extrap",    type=int,   default=None,
                   help="Extrapolate Chao curve out to this N (optional)")
    p.add_argument("--n-points",  type=int,   default=300,
                   help="Number of N grid points for plug-in curve (default: 300)")
    p.add_argument("--out",       default=None,
                   help="Output PNG path (auto-named if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    search_dirs  = (".", "..", "../..")
    ref_files    = REFS[args.ref]

    breadth_path = args.breadth or _find_file(ref_files["breadth"], search_dirs)
    depth_path   = args.depth   or _find_file(ref_files["depth"],   search_dirs)
    lengths_path = args.lengths or _find_file(ref_files["lengths"],  search_dirs)
    meta_path    = args.meta    or _find_meta(search_dirs)

    for label, path in [("breadth", breadth_path), ("depth", depth_path),
                         ("lengths", lengths_path), ("meta", meta_path)]:
        if path is None:
            sys.exit(f"ERROR: {label} file not found — pass --{label} <path>.")

    csv_path = args.csv or _find_sweep_csv(args.threshold, search_dirs)
    if csv_path is None:
        print("  No sweep CSV found — Hill-MM overlay will be skipped.")

    thr_label = f"{args.threshold:.1f}".replace(".", "p")
    out_path  = args.out or f"rarefaction_plugin_{args.ref}_{thr_label}pct.png"

    print(f"\nLoading data for {args.ref} ...")
    breadth_df, meta = load_samples(breadth_path, depth_path, lengths_path, meta_path)

    sweep_df = None
    if csv_path and os.path.exists(csv_path):
        print(f"  Loading sweep CSV: {csv_path}")
        sweep_df = pd.read_csv(csv_path)

    max_n    = max(meta["Group_fine"].isin(groups).sum()
                   for _, groups, _, _ in SERIES_CONFIG)
    N_values = np.unique(
        np.round(np.logspace(0, np.log10(max(max_n, 10)), args.n_points)).astype(int)
    )

    print(f"\nComputing plug-in rarefaction (threshold={args.threshold}%) ...")
    series_results = {}
    for name, groups, color, ls in SERIES_CONFIG:
        print(f"  Series: {name}")
        res = compute_series(breadth_df, meta, groups, args.threshold,
                             N_values, N_max_extrap=args.extrap)
        if res is not None:
            series_results[name] = res
        else:
            print(f"    [SKIP] fewer than 5 samples")

    if not series_results:
        sys.exit("ERROR: no series produced results.")

    make_plot(series_results, sweep_df, args.threshold,
              args.ref, out_path, N_max_extrap=args.extrap)


if __name__ == "__main__":
    main()
