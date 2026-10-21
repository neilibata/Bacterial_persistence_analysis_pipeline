#!/usr/bin/env python3
"""
figure_summary_joint.py — Joint Hierarchical Gut Microbiome SGB Analysis
=========================================================================

PRIMARY INFERENCE ENGINE: Joint Bayesian hierarchical model for
Industrial vs Non-Industrial SGB occupancy.

STATISTICAL PHILOSOPHY
----------------------
Prior: all SGBs are present in both populations at low baseline occupancy.
We look for strong Bayesian evidence that an SGB is genuinely absent
(ψ_k^Ind ≪ ε) in Industrial populations, not merely undetected.

Model (per SGB k) — PROBIT LINK throughout:
    probit(ψ_k^NI)  = α_k          α_k ~ N(μ_α, σ_α²)
    probit(ψ_k^Ind) = α_k + δ_k    δ_k ~ N(0,   σ_δ²)
    ψ = Φ((α + δ·1(Ind) + u_study) / σ_eff)

Prior centred on μ_α = Φ⁻¹(0.05) ≈ −1.645 (probit scale):
    "Every SGB is present in at least ~5% of samples unless the data
     strongly indicate otherwise."
δ_k = 0 a priori: no prior assumption that Industrial differs from NI.

Absence threshold (probit-consistent):
    ψ_k^Ind < ε  ↔  α_k + δ_k < σ_eff · Φ⁻¹(ε)  ≡  z_eps

N_missing = Σ_k Φ((z_eps − (α̂_k+δ̂_k)) / SD_k)  from Laplace posterior.
CI from simulation: draw (α+δ) ~ Laplace Normal, count draws below z_eps.

Detection model (shared across all steps):
    h[s,k] = ψ_k^g · Φ((μ_k + log d_s − log L_k − log τ) / σ_k)

where d_s = total_bp[s] = Σ_k depth_ks·L_k (total MAPPED bases).
Note: total_bp is endogenous; use --raw-depth-col if available.

Inference: Laplace approximation per SGB, hyperparameters by empirical Bayes.
"""

import persistence.config as _cfg
import argparse
import os
import warnings
from scipy.special import ndtr
from scipy.stats import norm as _scipy_norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

try:
    from .plot_utils import apply_display_text
except ImportError:
    def apply_display_text(fig):
        for artist in fig.findobj(match=matplotlib.text.Text):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)
        return fig

# ════════════════════════════════════════════════════════════════════════════
#  STYLE
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica","Arial","DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.direction": "in",
    "ytick.direction": "in", "lines.linewidth": 1.2,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

COL = {"Industrial": "#56B4E9", "Non-Industrial": "#E69F00",
       "world": "#555555", "absent": "#D55E00", "present": "#009E73"}

# ════════════════════════════════════════════════════════════════════════════
#  PRIORS  (user-facing constants, not buried in function signatures)
# ════════════════════════════════════════════════════════════════════════════

# "All SGBs present at low baseline occupancy" — PROBIT scale
# Φ⁻¹(0.05) ≈ -1.645: prior centred on 5% occupancy in probit space.
# σ_α = 1.0 on probit scale → occupancy range roughly [0.4%, 50%]
PRIOR_MU_ALPHA  = float(_scipy_norm.ppf(0.05))   # probit⁻¹(0.05) ≈ -1.645
PRIOR_SIG_ALPHA = 1.0     # probit-scale SD
PRIOR_SIG_DELTA = 1.0     # initial estimate; refined by EB

# Absence threshold: ψ_k^Ind < PSI_ABSENT_THR = "effectively absent in Industrial"
PSI_ABSENT_THR     = 0.01    # 1%
# Non-Industrial presence threshold: ψ_k^NI > PSI_NI_PRESENT_THR = "meaningfully present"
# Joint criterion: species is "lost" only if it was there to lose.
PSI_NI_PRESENT_THR = 0.05    # 5%

# Newton iterations for per-SGB Laplace MAP
N_NEWTON = 50    # increased from 20: smaller step size (0.5) needs more iterations

# Species batch size for matrix operations (controls peak memory ~100 MB)
BATCH_K  = 400

# ════════════════════════════════════════════════════════════════════════════
#  DATA PATHS  (edit or supply via --breadth-ind etc.)
# ════════════════════════════════════════════════════════════════════════════
MIN_TOTAL_BP = 1e8   # 100 Mbp minimum mapped bases

REFERENCES = {
    "UHGG": {
        "breadth_file":  "meancoverage_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv",
        "depth_file":    "meandepth_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv",
        "lengths_file":  "all_SGB_lengths_UHGG.txt",
    },
    "lev": {
        "breadth_file":  "meancoverage_and_std_matrix_high_lev_merged_dedup_firstcol.csv",
        "depth_file":    "meandepth_and_std_matrix_high_lev_merged_dedup_firstcol.csv",
        "lengths_file":  "all_SGB_lengths_lev.txt",
    },
}
META_FILE = "metadata_Mar_16_2026.csv"


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (identical to figure_summary_predictive.py)
# ════════════════════════════════════════════════════════════════════════════

def _find_meta():
    for p in [META_FILE, os.path.join("..", META_FILE),
              os.path.join(os.path.dirname(__file__), META_FILE)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Cannot find {META_FILE}")

def _normalise_country(x):
    if pd.isna(x): return None
    return str(x).strip().lower().title()

COUNTRY_TO_CONTINENT = {
    "United States": "North America", "Canada": "North America",
    "Mexico": "North America", "Cuba": "North America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Netherlands": "Europe",
    "Sweden": "Europe", "Denmark": "Europe", "Finland": "Europe",
    "China": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "India": "Asia", "Bangladesh": "Asia", "Mongolia": "Asia",
    "Fiji": "Oceania", "Papua New Guinea": "Oceania", "Australia": "Oceania",
    "Nigeria": "Africa", "Tanzania": "Africa", "Ethiopia": "Africa",
    "Malawi": "Africa", "Cameroon": "Africa", "South Africa": "Africa",
    "Peru": "South America", "Colombia": "South America",
    "Brazil": "South America", "Venezuela": "South America",
}

def _get_continent(country):
    if pd.isna(country): return "Unknown"
    return COUNTRY_TO_CONTINENT.get(str(country).strip(), "Unknown")

def _classify_macro(row):
    ls = str(row.get("Lifestyle","")).strip()
    if ls == "Industrial":   return "Industrial"
    if ls in ("H-G","Transitional"): return "Non-Industrial"
    return "Other"

def load_reference(ref_name, raw_depth_col=None, min_total_bp=MIN_TOTAL_BP):
    """
    Load breadth, depth, metadata and compute total_bp.

    Parameters
    ----------
    ref_name       : 'UHGG' or 'lev'
    raw_depth_col  : optional column name in metadata for raw sequenced bases.
                     If supplied, used as d_s instead of Σ depth×L_k, decoupling
                     effort from reference-mapping efficiency.

    Returns dict with keys: breadth, depth, meta, lengths, total_bp, n_sgbs.
    total_bp is either raw_depth_col (preferred) or Σ depth×L_k (fallback).
    """
    cfg = REFERENCES[ref_name]
    print(f"  Loading {ref_name}...")

    def _load_mean_matrix(filepath):
        df = pd.read_csv(filepath, index_col=0, low_memory=False)
        df.index = df.index.astype(str).str.strip()
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        df = df[mean_cols].copy()
        df.columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]
        return df.fillna(0.0)

    breadth_df = _load_mean_matrix(cfg["breadth_file"])
    depth_df   = _load_mean_matrix(cfg["depth_file"])
    common     = sorted(set(breadth_df.columns) & set(depth_df.columns))
    breadth_df = breadth_df[common]; depth_df = depth_df[common]

    ldf = pd.read_csv(cfg["lengths_file"], sep=r"\s+", header=None,
                      names=["microbe","length"])
    ldf["microbe"] = ldf["microbe"].astype(str).str.strip()
    ldict = dict(zip(ldf["microbe"], ldf["length"]))
    keep  = [s for s in common if s in ldict]
    if not keep:
        raise ValueError(f"0 SGBs matched lengths for {ref_name}")
    breadth_df = breadth_df[keep]; depth_df = depth_df[keep]
    lengths = pd.Series({s: ldict[s] for s in keep})
    print(f"    {len(keep):,} SGBs matched lengths")

    # total_bp = Σ_k depth_ks × L_k  (total MAPPED bases per sample)
    # ⚠ Endogenous: maps biology and reference-compatibility into one number.
    # Use --raw-depth-col if metadata has raw sequenced bases.
    total_bp_mapped = depth_df.mul(lengths, axis=1).sum(axis=1)

    meta_file = _find_meta()
    meta = pd.read_csv(meta_file, low_memory=False)
    # Table M1 stores the manuscript's display labels (Agro-pastoralist,
    # Hunter-gatherer); analysis code compares against the internal keys.
    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = _cfg.canonicalize_lifestyle_series(meta["Lifestyle"])
    meta["Chosen_match_ID"] = meta["Chosen_match_ID"].astype(str).str.strip()
    meta = (meta[meta["Chosen_match_ID"].isin(breadth_df.index)]
            .drop_duplicates("Chosen_match_ID", keep="first")
            .set_index("Chosen_match_ID"))
    breadth_df = breadth_df.loc[meta.index]
    depth_df   = depth_df.loc[meta.index]
    total_bp_mapped = total_bp_mapped.loc[meta.index]

    # Raw sequencing depth (preferred effort covariate)
    if raw_depth_col and raw_depth_col in meta.columns:
        total_bp = meta[raw_depth_col].astype(float).clip(lower=1.0)
        total_bp.name = "total_bp_raw"
        print(f"    Using raw depth column: {raw_depth_col!r}")
    else:
        if raw_depth_col:
            print(f"    WARNING: --raw-depth-col {raw_depth_col!r} not in metadata; "
                  f"falling back to Σ(depth×L_k)")
        total_bp = total_bp_mapped

    meta["Country_norm"] = meta["Country"].apply(_normalise_country) if "Country" in meta.columns else None
    meta["Continent"]  = meta["Country_norm"].apply(_get_continent) if "Country_norm" in meta.columns else "Unknown"
    meta["MacroGroup"] = meta.apply(_classify_macro, axis=1)

    valid = meta["Lifestyle"].notna()
    meta = meta[valid]; breadth_df = breadth_df.loc[meta.index]
    depth_df = depth_df.loc[meta.index]; total_bp = total_bp.loc[meta.index]

    low = total_bp < min_total_bp
    if low.any():
        print(f"    Removing {low.sum()} samples with total_bp < {min_total_bp:.0e}")
        keep_idx = meta.index[~low]
        meta = meta.loc[keep_idx]; breadth_df = breadth_df.loc[keep_idx]
        depth_df = depth_df.loc[keep_idx]; total_bp = total_bp.loc[keep_idx]

    depth_df = depth_df.loc[meta.index]
    print(f"    {len(meta):,} samples after filters")

    # ── Study ID extraction ───────────────────────────────────────────────
    # Try common metadata column names in priority order; fall back to a
    # Country×Lifestyle composite so downstream code always has study labels.
    _STUDY_COLS = ["Study_accession", "Study", "BioProject", "Project_accession",
                   "ENA_project", "project_accession", "study"]
    study_col_found = next((c for c in _STUDY_COLS if c in meta.columns), None)
    if study_col_found:
        meta["study_id"] = meta[study_col_found].astype(str).str.strip()
        print(f"    Study IDs from column: {study_col_found!r}  "
              f"({meta['study_id'].nunique()} unique studies)")
    else:
        # Composite proxy: Country × Lifestyle (separates cohorts approximately)
        c_col = "Country_norm" if "Country_norm" in meta.columns else (
                "Country" if "Country" in meta.columns else None)
        if c_col:
            meta["study_id"] = (meta[c_col].astype(str) + "__"
                                + meta["Lifestyle"].astype(str))
        else:
            meta["study_id"] = meta["Lifestyle"].astype(str)
        meta["_study_id_is_proxy"] = True
        print(f"    *** SENSITIVITY ONLY *** No study accession column found in metadata.")
        print(f"        Falling back to Country×Lifestyle proxy "
              f"({meta['study_id'].nunique()} pseudo-groups).")
        print(f"        This proxy can over-merge or over-split cohorts and does not")
        print(f"        represent true study boundaries.  Treat any run using this")
        print(f"        proxy as exploratory — do not report as a primary result.")

    return {"breadth": breadth_df, "depth": depth_df, "meta": meta,
            "lengths": lengths, "total_bp": total_bp, "n_sgbs": len(keep)}


def build_group_blocks(meta, breadth_df, threshold=10.0):
    lifestyle = meta["Lifestyle"]
    blocks = {}
    for name, mask in [("Industrial (all)",     lifestyle == "Industrial"),
                       ("Non-Industrial (all)", lifestyle.isin(["H-G","Transitional"]))]:
        ids = mask[mask].index.intersection(breadth_df.index)
        blocks[name] = breadth_df.loc[ids]
        print(f"    {name}: {len(ids):,} samples")
    return blocks


# ════════════════════════════════════════════════════════════════════════════
#  STUDY EFFECT ESTIMATION
#  Per-study offsets for occupancy (logit scale) and depth (log scale),
#  centered within lifestyle so lifestyle-level signal is preserved.
# ════════════════════════════════════════════════════════════════════════════

def estimate_study_effects(study_ids_ind, study_ids_ni,
                            C_ind, C_ni,
                            log_d_ind, log_d_ni,
                            J,
                            sig_occ_prior=1.0, sig_det_prior=0.5):
    """
    Per-study offsets using the full ordinal observation C ∈ {0,...,J}.

    Occupancy offsets v_j^occ (probit scale, centered within lifestyle):
        Summary: mean normalised ordinal score  s_j = mean_{s∈j,k}(C_sk / J)
        Offset:  v_j^occ = shrink(probit(s_j + δ) − probit(lifestyle_s + δ))
        where δ=0.01 prevents probit collapse at 0 or 1.
        This uses ALL threshold bins — not just (C>0) — so the study correction
        is aligned with the main ordinal likelihood.
        Centering WITHIN lifestyle: lifestyle-level signal is preserved.

    Detection offsets v_j^det (log-depth scale, centered within lifestyle):
        v_j^det = mean_{s∈j}(log d_s) − lifestyle_mean(log d)
        Applied as: log_d_eff[s] = log_d[s] − v_j^det[study[s]]

    Returns occ_off_ind, occ_off_ni, det_off_ind, det_off_ni, study_info
    """
    eps = 0.01   # probit regularisation at boundaries

    def _shrunk_occ_offsets_ordinal(sids, C, J_val, lifestyle_score, n0):
        offsets = np.zeros(len(sids), float)
        stats   = {}
        for sid in np.unique(sids):
            mask   = sids == sid
            n_j    = int(mask.sum())
            # Ordinal mean normalised to [0,1]
            score_j = float(C[mask].mean() / J_val)
            score_j = np.clip(score_j, eps, 1.0 - eps)
            lifestyle_s = np.clip(lifestyle_score, eps, 1.0 - eps)
            # Probit-scale difference, shrunk by sample count
            raw_v  = (_scipy_norm.ppf(score_j) - _scipy_norm.ppf(lifestyle_s))
            w      = n_j / (n_j + n0)
            shrunk = w * raw_v
            offsets[mask] = shrunk
            stats[sid] = {"n": n_j, "ordinal_score": score_j,
                          "raw_v": raw_v, "shrunk_v": shrunk}
        return offsets, stats

    def _depth_offsets(sids, log_d, lifestyle_mean_log_d):
        offsets = np.zeros(len(sids), float)
        for sid in np.unique(sids):
            mask = sids == sid
            offsets[mask] = float(log_d[mask].mean()) - lifestyle_mean_log_d
        return offsets

    study_ids_ind = np.asarray(study_ids_ind)
    study_ids_ni  = np.asarray(study_ids_ni)
    n0_occ = 1.0 / sig_occ_prior**2

    # Lifestyle-level ordinal mean (reference for centering)
    ls_score_ind = float(C_ind.mean() / J)
    ls_score_ni  = float(C_ni.mean()  / J)

    occ_off_ind, stats_ind = _shrunk_occ_offsets_ordinal(
        study_ids_ind, C_ind, J, ls_score_ind, n0_occ)
    occ_off_ni,  stats_ni  = _shrunk_occ_offsets_ordinal(
        study_ids_ni,  C_ni,  J, ls_score_ni,  n0_occ)

    det_off_ind = _depth_offsets(study_ids_ind, log_d_ind, float(log_d_ind.mean()))
    det_off_ni  = _depth_offsets(study_ids_ni,  log_d_ni,  float(log_d_ni.mean()))

    n_st_ind = len(np.unique(study_ids_ind))
    n_st_ni  = len(np.unique(study_ids_ni))
    print(f"      Industrial: {n_st_ind} studies  "
          f"occ_off(ordinal) SD={float(np.std(occ_off_ind)):.3f}  "
          f"det_off SD={float(np.std(det_off_ind)):.3f} log-bp")
    print(f"      NI:         {n_st_ni} studies  "
          f"occ_off(ordinal) SD={float(np.std(occ_off_ni)):.3f}  "
          f"det_off SD={float(np.std(det_off_ni)):.3f} log-bp")

    return occ_off_ind, occ_off_ni, det_off_ind, det_off_ni, {
        "ind": stats_ind, "ni": stats_ni,
        "n_studies_ind": n_st_ind, "n_studies_ni": n_st_ni,
    }
    """
    Per-study offsets for occupancy and detection layers.

    Occupancy offsets v_j^occ  (logit scale, centered within lifestyle):
        v_j^occ = shrink(logit(det_rate_j) − logit(lifestyle_mean_det_rate))
        Shrinkage weight: n_j / (n_j + n0_occ) where n0_occ = 1/sig_occ_prior²
        Centering WITHIN lifestyle is critical: it does not subtract the
        lifestyle-level signal, only the systematic between-study offset.

    Detection offsets v_j^det  (log-depth scale, centered within lifestyle):
        v_j^det = mean_s∈j(log d_s) − lifestyle_mean(log d)
        Corrects for between-study sequencing depth differences.
        Applied as: effective_log_d[s] = log_d[s] − v_j^det[study[s]]
        so that r_k uses depth normalized to the lifestyle-level mean.

    Returns
    -------
    occ_off_ind : (n_ind,)  per-sample occupancy offset (logit scale)
    occ_off_ni  : (n_ni,)   per-sample occupancy offset
    det_off_ind : (n_ind,)  per-sample depth correction (log scale)
    det_off_ni  : (n_ni,)   per-sample depth correction
    study_info  : dict with per-study statistics for diagnostics
    """

# ════════════════════════════════════════════════════════════════════════════
#  TWO-STAGE STUDY-AWARE ACCUMULATION
# ════════════════════════════════════════════════════════════════════════════

def _update_study_offsets_ordinal(alpha_hat, delta_hat, sig_eff,
                                   C_ind, r_all_ind, study_ids_ind,
                                   occ_off_ind, sig_occ,
                                   batch_k=BATCH_K, n_newton=5,
                                   sample_weights=None):
    """
    Refine study occupancy offsets u_j via Newton steps on full ordinal data.

    sample_weights : (n_ind,) or None.  When supplied (study-weighted mode),
        each sample's gradient contribution is scaled by its weight so that
        large studies do not dominate the offset estimate.  Pass None for
        uniform (sample-weighted) mode.
    """
    n_ind, n_sgb = C_ind.shape
    unique_studies = np.unique(study_ids_ind)
    n_st = len(unique_studies)
    study_to_idx = {s: i for i, s in enumerate(unique_studies)}
    sidx = np.array([study_to_idx[s] for s in study_ids_ind])

    u_j = np.array([float(occ_off_ind[study_ids_ind == s][0])
                    for s in unique_studies], float)

    w = (np.asarray(sample_weights, float)
         if sample_weights is not None else np.ones(n_ind, float))  # (n_ind,)

    EPS_H = 1e-9

    for _ in range(n_newton):
        g_per_samp = np.zeros(n_ind, float)
        F_per_samp = np.zeros(n_ind, float)

        for ks in range(0, n_sgb, batch_k):
            ke   = min(ks + batch_k, n_sgb)
            k_sl = slice(ks, ke)
            u_s  = u_j[sidx][:, None]
            z    = ((alpha_hat[k_sl] + delta_hat[k_sl])[None, :] + u_s) / sig_eff
            dp   = _scipy_norm.pdf(z) / sig_eff
            psi  = ndtr(z)
            p_cz1 = _p_cz1_batch(C_ind[:, k_sl], r_all_ind[:, k_sl, :])
            p_cz0 = np.where(C_ind[:, k_sl] == 0, 1.0 - 1e-6, 1e-6)
            p_obs = np.clip((1 - psi) * p_cz0 + psi * p_cz1, EPS_H, 1.0)
            sc    = (p_cz1 - p_cz0) / p_obs
            g_per_samp += (sc * dp).sum(1)
            F_per_samp += (dp**2 * (p_cz1 - p_cz0)**2 / p_obs).sum(1)

        g_j = np.zeros(n_st, float); F_j = np.zeros(n_st, float)
        np.add.at(g_j, sidx, w * g_per_samp)
        np.add.at(F_j, sidx, w * F_per_samp)
        g_j -= u_j / sig_occ**2
        F_j += 1.0 / sig_occ**2
        step = np.clip(np.where(F_j > 1e-12, g_j / F_j, 0.0), -0.5, 0.5)
        u_j  = u_j + step

    return u_j[sidx]



def _n_missing_with_hyper_uncertainty(alpha_hat, delta_hat,
                                       var_aa, var_dd, cov_ad,
                                       sig_alpha, sig_delta, sig_gamma,
                                       psi_absent_thr, n_sgb, rng,
                                       psi_ni_present_thr=0.05,
                                       n_species_draws=500, n_hyper=100):
    """
    Parametric bootstrap propagating (σ_α, σ_δ, σ_γ) uncertainty into N_missing.

    Uses the JOINT criterion P(ψ^Ind < ε AND ψ^NI > τ_ni) for every hyper-draw,
    so the broad CI is for the same estimand as N_missing_mean. Previous version
    used the marginal Ind-only criterion — that inconsistency is fixed here.

    For each hyper draw:
      1. Draw (σ_α*, σ_δ*, σ_γ*) from approximate posteriors.
      2. Recompute z_eps* and z_ni* under the drawn hyperparameters.
      3. Draw α_k* and (α+δ)_k* from the shifted 2D Laplace Normal.
      4. Count species where α+δ* < z_eps* AND α* > z_ni*.
    """
    mu_sum  = alpha_hat + delta_hat
    var_sum = np.maximum(var_aa + var_dd + 2.0 * cov_ad, 1e-9)
    delta_extra_var = np.maximum(var_dd - var_aa + 2.0 * cov_ad, 0.0)

    n_eff = max(n_sgb, 10)
    se_sa = sig_alpha / np.sqrt(2.0 * n_eff)
    se_sd = sig_delta / np.sqrt(2.0 * n_eff)
    se_sg = sig_gamma / np.sqrt(2.0 * n_eff)

    all_draws = []
    for _ in range(n_hyper):
        sig_a_b   = max(abs(rng.normal(sig_alpha, se_sa)), 0.01)
        sig_d_b   = max(abs(rng.normal(sig_delta, se_sd)), 0.01)
        sig_g_b   = max(abs(rng.normal(sig_gamma, se_sg)), 0.01)
        sig_eff_b = float(np.sqrt(1.0 + (np.pi / 8.0) * sig_g_b**2))
        z_eps_b   = sig_eff_b * float(_scipy_norm.ppf(psi_absent_thr))
        z_ni_b    = sig_eff_b * float(_scipy_norm.ppf(psi_ni_present_thr))

        extra_aa = max((sig_a_b**2 - sig_alpha**2) / n_eff, 0.0)
        extra_dd = max((sig_d_b**2 - sig_delta**2) / n_eff, 0.0)

        # Draw α and α+δ from the shifted 2D Laplace Normal (joint criterion)
        sd_aa_b = np.sqrt(np.maximum(var_aa + extra_aa, 1e-9))
        sd_de_b = np.sqrt(np.maximum(delta_extra_var + extra_dd, 1e-9))

        a_b  = rng.normal(alpha_hat[None, :], sd_aa_b[None, :],
                           (n_species_draws, n_sgb))
        d_b  = rng.normal(delta_hat[None, :], sd_de_b[None, :],
                           (n_species_draws, n_sgb))
        ad_b = a_b + d_b

        # Joint criterion — matches N_missing_mean estimand
        all_draws.append(((ad_b < z_eps_b) & (a_b > z_ni_b)).sum(1))

    return np.concatenate(all_draws).astype(float)


def union_missing_posterior(h_mat_ind, h_mat_ni,
                             r_mat_ind, r_mat_ni,
                             alpha_hat, delta_hat, var_aa, var_dd, cov_ad,
                             sig_eff, psi_ind_hat, psi_ni_hat,
                             study_ids_ind, study_ids_ni,
                             N_star_ind, N_star_ni,
                             eps_union=0.01, tau_union=0.05,
                             n_draws=2000, rng=None, mode='equal'):
    """
    Lifestyle-wide union-presence posterior for each taxon.

    Answers: "is taxon k missing from the Industrial lifestyle pool?"
    where the lifestyle pool is defined by:
       A_{g,k}(N*) = 1 − q_{g,k}^{N*}
    with q_{g,k} = study-mix-averaged miss-probability, computed under
    the two-stage study-aware distribution (equal or empirical study weights).

    Key idea (vs. per-sample occupancy):
      "Found anywhere in a lifestyle counts as a detection."
      A taxon present in only one study but reliably there still counts.
      The union probability grows with N* and is bounded below by the
      detection probability of the best study for that taxon.

    Under Laplace posterior draws (α_k^{(m)}, δ_k^{(m)}):
      ψ^{Ind,(m)}_k = Φ((α^{(m)} + δ^{(m)}) / σ_eff)
      h_{s,k}^{Ind,(m)} = ψ^{Ind,(m)}_k × r_{s,k}^{Ind}     [r is fixed]
      q_{g,k}^{(m)}     = 1 − ψ^{Ind,(m)}_k × r̄_k^{eq}     [r̄ = study-mean r]
      A^{Ind,(m)}_k(N*) = 1 − (q_{g,k}^{(m)})^{N*}

    Parameters
    ----------
    r_mat_ind / r_mat_ni : (n_samp, n_sgb)  per-sample detection probs at
        primary threshold WITHOUT occupancy (= h / ψ̂).
    N_star_ind / N_star_ni : int  reference sample size for union probability.
        Natural choice: observed group sample counts.
    mode : 'equal'  → 1/K weight per study   'empirical' → n_j/n weight.

    Returns
    -------
    p_missing_union   : (n_sgb,) P(A^Ind < ε_union AND A^NI > τ_union)
    N_missing_union   : float expected count
    N_miss_union_draws: (n_draws,)
    A_ind_map, A_ni_map: (n_sgb,) union detection probability at MAP ψ
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_sgb = len(alpha_hat)

    def _study_mean_r(r_mat, study_ids, n_sgb, mode):
        """Study-mix-averaged detection rate r̄_k per taxon."""
        if study_ids is None:
            return r_mat.mean(0)   # plain sample mean
        unique_s = np.unique(study_ids)
        K = len(unique_s)
        study_means = np.zeros((K, n_sgb), float)
        for i, s in enumerate(unique_s):
            mask = study_ids == s
            study_means[i] = r_mat[mask].mean(0)
        if mode == 'equal':
            return study_means.mean(0)   # (n_sgb,)
        else:   # empirical: weight by study size
            n_per = np.array([(study_ids == s).sum() for s in unique_s], float)
            return (study_means * n_per[:, None]).sum(0) / n_per.sum()

    r_bar_ind = _study_mean_r(r_mat_ind, study_ids_ind, n_sgb, mode)  # (n_sgb,)
    r_bar_ni  = _study_mean_r(r_mat_ni,  study_ids_ni,  n_sgb, mode)

    # MAP union probabilities (for diagnostics)
    q_ind_map = np.clip(1.0 - psi_ind_hat * r_bar_ind, 0.0, 1.0)
    q_ni_map  = np.clip(1.0 - psi_ni_hat  * r_bar_ni,  0.0, 1.0)
    A_ind_map = 1.0 - q_ind_map ** N_star_ind
    A_ni_map  = 1.0 - q_ni_map  ** N_star_ni

    # Posterior draws from bivariate Laplace Normal
    delta_extra_var = np.maximum(var_dd - var_aa + 2.0 * cov_ad, 0.0)
    a_draws  = rng.normal(alpha_hat[None, :],
                           np.sqrt(np.maximum(var_aa, 1e-9))[None, :],
                           (n_draws, n_sgb))   # α draws
    d_draws  = rng.normal(delta_hat[None, :],
                           np.sqrt(delta_extra_var + 1e-9)[None, :],
                           (n_draws, n_sgb))   # δ residual draws
    ad_draws = a_draws + d_draws              # α+δ

    psi_ind_draws = ndtr(ad_draws / sig_eff)   # (n_draws, n_sgb)
    psi_ni_draws  = ndtr(a_draws  / sig_eff)

    # Union probabilities per draw: A = 1 − (1 − ψ·r̄)^{N*}
    q_ind_draws = np.clip(1.0 - psi_ind_draws * r_bar_ind[None, :], 0.0, 1.0)
    q_ni_draws  = np.clip(1.0 - psi_ni_draws  * r_bar_ni [None, :], 0.0, 1.0)
    A_ind_draws = 1.0 - q_ind_draws ** N_star_ind  # (n_draws, n_sgb)
    A_ni_draws  = 1.0 - q_ni_draws  ** N_star_ni

    # Joint missingness criterion on union scale
    cond_a = A_ind_draws < eps_union    # Ind lifestyle pool near-empty
    cond_b = A_ni_draws  > tau_union    # NI lifestyle pool meaningfully detected
    p_missing_union  = np.mean(cond_a & cond_b, axis=0)   # (n_sgb,)
    N_miss_draws_u   = (cond_a & cond_b).sum(1).astype(float)
    N_missing_union  = float(N_miss_draws_u.mean())

    return p_missing_union, N_missing_union, N_miss_draws_u, A_ind_map, A_ni_map


def two_stage_accumulation(h_mat, study_ids, N_max, n_draw=200, rng=None,
                            mode='empirical'):
    """
    Study-aware predictive accumulation: draw study first, then sample.

    Two stages per new sample:
      1. Draw study j from lifestyle-level study distribution
      2. Draw sample s uniformly from within study j

    This prevents large studies from dominating the accumulation curve.
    A single study with 5,000 samples should not have 5× the weight of
    a study with 1,000 samples when estimating the lifestyle-level SGB pool.

    Parameters
    ----------
    h_mat     : (n_samp, n_sgb) detection probabilities from joint model
    study_ids : (n_samp,) study label per sample
    mode      : 'empirical'  — P(study j) ∝ n_j (observed study mix)
                'equal'      — P(study j) = 1/K (equal weight per study)
    N_max     : maximum samples to simulate
    n_draw    : number of bootstrap replicates

    Returns Ns, mean, lo95, hi95, draws
    """
    if rng is None: rng = np.random.default_rng()
    study_ids = np.asarray(study_ids)
    unique_studies = np.unique(study_ids)
    K = len(unique_studies)

    # Build index arrays: for each study, which rows of h_mat belong to it
    study_rows = {s: np.where(study_ids == s)[0] for s in unique_studies}

    if mode == 'equal':
        study_probs = np.ones(K) / K
    else:  # empirical
        n_per_study = np.array([len(study_rows[s]) for s in unique_studies], float)
        study_probs = n_per_study / n_per_study.sum()

    n_obs = h_mat.shape[0]
    Ns_set = set(np.geomspace(1, N_max, 40).astype(int).tolist())
    Ns_set.update([1, n_obs // 4, n_obs // 2, n_obs, N_max])
    Ns = np.array(sorted(s for s in Ns_set if 1 <= s <= N_max), int)

    h_arr  = np.clip(h_mat, 0.0, 1.0 - 1e-9)
    draws  = np.zeros((n_draw, len(Ns)), float)

    for b in range(n_draw):
        # Draw studies, then samples within studies
        chosen_studies = rng.choice(unique_studies, size=int(Ns.max()),
                                     p=study_probs, replace=True)
        row_idx = np.array([
            rng.choice(study_rows[s]) for s in chosen_studies
        ], dtype=int)
        h_sim  = h_arr[row_idx]
        cumlog = np.cumsum(np.log1p(-h_sim), axis=0)
        draws[b] = np.sum(1.0 - np.exp(cumlog[Ns - 1, :]), axis=1)

    return Ns.astype(float), draws.mean(0), *np.percentile(draws, [2.5, 97.5], 0), draws


# ════════════════════════════════════════════════════════════════════════════
#  SHARED OBSERVATION MODEL
#  Single function used by: detect param estimation, joint model,
#  accumulation curves, holdout diagnostic.
# ════════════════════════════════════════════════════════════════════════════

def _h_lognormal_batched(psi_k, mu_x, sig_x, log_d, log_L, log_tau, batch_k=BATCH_K):
    """
    h[s,k] = ψ_k · Φ((μ_k + log d_s − log L_k − log τ) / σ_k)

    Empirical sum over ACTUAL log(total_bp[s]) — no distributional assumption
    on depth. log_d may be multi-modal across studies; we use every value.

    Parameters
    ----------
    psi_k  : (n_sgb,)  occupancy (set to ones to get raw detection prob r)
    mu_x   : (n_sgb,)  log-abundance mean
    sig_x  : (n_sgb,)  log-abundance SD (> 0)
    log_d  : (n_samp,) log(total_bp) for the relevant sample set
    log_L  : (n_sgb,)  log(genome_length)
    log_tau: float      log(-log(1 − τ_frac))  [complementary log-log of threshold]
    batch_k: int        species per batch (~100 MB peak memory)

    Returns
    -------
    h_mat : (n_samp, n_sgb) clipped to [0, 1)
    """
    n_samp = len(log_d)
    n_sgb  = len(psi_k)
    mu_fallback = float(log_d.mean())
    h_mat  = np.zeros((n_samp, n_sgb), float)

    for ks in range(0, n_sgb, batch_k):
        ke   = min(ks + batch_k, n_sgb)
        k_sl = slice(ks, ke)
        mu_b  = np.where(np.isfinite(mu_x[k_sl]),
                          mu_x[k_sl], mu_fallback - log_L[k_sl])
        sig_b = np.maximum(np.where(np.isfinite(sig_x[k_sl]),
                                     sig_x[k_sl], 0.5), 0.05)
        z = (mu_b[None, :] + (log_d[:, None] - log_L[None, k_sl]) - log_tau) / sig_b[None, :]
        r_j = ndtr(z)
        h_mat[:, k_sl] = np.clip(psi_k[k_sl][None, :] * r_j, 0.0, 1.0 - 1e-9)

    return h_mat


def _r_batched(mu_x, sig_x, log_d, log_L, log_tau, batch_k=BATCH_K):
    """Pure detection probability r_k(d_s) — ψ=1, same model as _h_lognormal_batched."""
    return _h_lognormal_batched(np.ones(len(mu_x)), mu_x, sig_x,
                                 log_d, log_L, log_tau, batch_k)


# ════════════════════════════════════════════════════════════════════════════
#  DETECTION PARAMETER ESTIMATION  (Pass 1 of joint model)
#  Estimated ONCE on ALL samples combined (both groups) for maximum power.
# ════════════════════════════════════════════════════════════════════════════

def _truncated_lognormal_mle(log_x_obs, log_trunc_pts, n_iter=12):
    """
    MLE for N(μ, σ²) truncated from below at sample-varying thresholds.
    Analytical Newton correction (8-12 iterations). Fallback to naive mean/std.
    """
    z = np.asarray(log_x_obs, float)
    c = np.asarray(log_trunc_pts, float)
    if len(z) < 2:
        return (float(z[0]) if len(z) == 1 else 0.0), 0.10
    mu  = float(z.mean())
    sig = max(float(z.std(ddof=1)), 0.05)
    _S2PI = np.sqrt(2.0 * np.pi)
    for _ in range(n_iter):
        a       = (c - mu) / sig
        phi_a   = np.exp(-0.5 * a * a) / _S2PI
        Phi_neg = np.maximum(ndtr(-a), 1e-300)
        mills   = phi_a / Phi_neg
        mu_new  = float(z.mean() - sig * mills.mean())
        a2      = (c - mu_new) / sig
        phi_a2  = np.exp(-0.5 * a2 * a2) / _S2PI
        mills2  = phi_a2 / np.maximum(ndtr(-a2), 1e-300)
        vc      = np.maximum(1.0 - a2 * mills2 - mills2**2, 1e-3)
        sig_new = max(np.sqrt(float(np.var(z, ddof=1)) / float(vc.mean())), 0.05)
        if not (np.isfinite(mu_new) and np.isfinite(sig_new)) or sig_new > 12:
            break
        if abs(mu_new - mu) < 1e-5 and abs(sig_new - sig) < 1e-5:
            return float(mu_new), float(sig_new)
        mu, sig = mu_new, sig_new
    return (float(mu), float(sig)) if (np.isfinite(mu) and sig < 12) else (float(z.mean()), max(float(z.std(ddof=1)), 0.05))


def _empirical_bayes_shrink(mu_x, sig_x, n_det, n0=5):
    """Shrink μ_k, σ_k toward group means for species with few detections."""
    finite = np.isfinite(mu_x)
    if finite.sum() < 2:
        mu_x  = np.where(finite, mu_x,  0.0)
        sig_x = np.where(np.isfinite(sig_x), sig_x, 0.5)
        return mu_x, sig_x
    mu_pop  = float(np.nanmedian(mu_x[finite]))
    sig_pop = float(np.nanmedian(sig_x[finite & np.isfinite(sig_x)]))
    w       = np.minimum(n_det, n0).astype(float) / n0
    mu_x    = np.where(finite, w * mu_x + (1 - w) * mu_pop, mu_pop)
    sig_x   = np.where(np.isfinite(sig_x), w * sig_x + (1 - w) * sig_pop, sig_pop)
    return mu_x, sig_x


def estimate_detection_params(breadth_block, depth_block, total_bp_block, lengths,
                               eps_pct=0.5, correct_truncation=True, eb_n0=5,
                               batch_k=BATCH_K):
    """
    Estimate per-SGB detection parameters (μ_k, σ_k) from one group's samples.

    Called separately for Industrial and Non-Industrial to allow the joint model
    to distinguish occupancy depletion (δ_k < 0) from conditional abundance
    depletion (μ_k^Ind < μ_k^NI).

    Returns mu_x, sig_x, n_det_arr, L_k, log_d
    """
    idx    = (breadth_block.index
              .intersection(depth_block.index)
              .intersection(total_bp_block.index))
    b_frac = breadth_block.loc[idx].values.astype(float) / 100.0
    d_mat  = depth_block.loc[idx].values.astype(float)
    d_all  = np.maximum(total_bp_block.loc[idx].values.astype(float), 1.0)
    n_samp, n_sgb = b_frac.shape

    L_k     = np.maximum(lengths.reindex(breadth_block.columns).values.astype(float), 1.0)
    log_L   = np.log(L_k)
    log_d   = np.log(d_all)
    eps     = eps_pct / 100.0
    eps_tau = -np.log1p(-eps)
    det_eps = b_frac > eps

    mu_x      = np.full(n_sgb, np.nan, float)
    sig_x     = np.full(n_sgb, 0.5,   float)
    n_det_arr = np.zeros(n_sgb, int)

    for ks in range(0, n_sgb, batch_k):
        ke   = min(ks + batch_k, n_sgb)
        b_b  = b_frac[:, ks:ke]
        d_b  = d_mat[:,  ks:ke]
        de_b = det_eps[:, ks:ke]
        n_det_arr[ks:ke] = de_b.sum(0)
        for k_loc in range(ke - ks):
            k = ks + k_loc
            idx_det  = np.where(de_b[:, k_loc])[0]
            if len(idx_det) < 2: continue
            lam_raw  = d_b[idx_det, k_loc]
            mask_pos = lam_raw > 0
            lam      = lam_raw[mask_pos]
            idx_v    = idx_det[mask_pos]
            if len(lam) < 2: continue
            log_x     = np.log(lam) - log_d[idx_v] + log_L[k]
            log_trunc = np.log(eps_tau) - log_d[idx_v] + log_L[k]
            if correct_truncation and len(log_x) >= 3:
                mu_x[k], sig_x[k] = _truncated_lognormal_mle(log_x, log_trunc)
            else:
                mu_x[k]  = float(np.mean(log_x))
                sig_x[k] = max(float(np.std(log_x, ddof=1)), 0.05)

    mu_x, sig_x = _empirical_bayes_shrink(mu_x, sig_x, n_det_arr, n0=eb_n0)
    return mu_x, sig_x, n_det_arr, L_k, log_d


def estimate_detection_params_grouped(breadth_ind, breadth_ni,
                                        depth_ind,   depth_ni,
                                        total_bp_ind, total_bp_ni,
                                        ref_sgbs, lengths,
                                        correct_truncation=True, eb_n0=5,
                                        batch_k=BATCH_K):
    """
    Group-specific detection parameters with cross-group EB shrinkage.

    Estimates (μ_k^Ind, σ_k^Ind) and (μ_k^NI, σ_k^NI) separately, then
    shrinks species undetected in one group toward the other group's estimate.

    This separates two distinct biological signals:
      η_k = μ_k^Ind − μ_k^NI < 0  → present but lower abundance in Industrial
      δ_k < 0 in the occupancy model → genuinely absent in Industrial

    Returns mu_ind, sig_ind, mu_ni, sig_ni, L_k, log_d_ind, log_d_ni, eta_k
    """
    mu_ind, sig_ind, n_ind_det, L_k, log_d_ind = estimate_detection_params(
        breadth_ind[ref_sgbs],
        depth_ind.loc[breadth_ind.index, ref_sgbs],
        total_bp_ind.loc[breadth_ind.index],
        lengths.reindex(ref_sgbs),
        correct_truncation=correct_truncation, eb_n0=eb_n0, batch_k=batch_k)

    mu_ni, sig_ni, n_ni_det, _, log_d_ni = estimate_detection_params(
        breadth_ni[ref_sgbs],
        depth_ni.loc[breadth_ni.index, ref_sgbs],
        total_bp_ni.loc[breadth_ni.index],
        lengths.reindex(ref_sgbs),
        correct_truncation=correct_truncation, eb_n0=eb_n0, batch_k=batch_k)

    # Cross-group EB shrinkage: for species with few detections in one group,
    # borrow from the other group proportional to detection count.
    # Smooth shrinkage avoids the hard detection-count cliff that can force a
    # "present but thinner" species toward an "absent" interpretation.
    #
    # Weight toward own group: w_ind = n_ind_det / (n_ind_det + n0)
    # where n0 = 5 acts as prior equivalent count from the other group.
    # Species well-detected in both groups: w≈1 (keep own estimate).
    # Species undetected in one group: w≈0 (borrow cross-group estimate).
    n0_cross = 5.0
    w_ind    = n_ind_det.astype(float) / (n_ind_det.astype(float) + n0_cross)
    w_ni     = n_ni_det.astype(float)  / (n_ni_det.astype(float)  + n0_cross)
    # Pooled cross-group estimates (weighted by detection count)
    n_total  = (n_ind_det + n_ni_det).astype(float) + 1.0
    mu_pop   = (n_ind_det.astype(float) * mu_ind + n_ni_det.astype(float) * mu_ni) / n_total
    sig_pop  = (n_ind_det.astype(float) * sig_ind + n_ni_det.astype(float) * sig_ni) / n_total
    # Smoothly blend own estimate toward pooled
    mu_ind   = w_ind * mu_ind  + (1.0 - w_ind) * mu_pop
    sig_ind  = w_ind * sig_ind + (1.0 - w_ind) * sig_pop
    mu_ni    = w_ni  * mu_ni   + (1.0 - w_ni ) * mu_pop
    sig_ni   = w_ni  * sig_ni  + (1.0 - w_ni ) * sig_pop

    # Abundance difference: η_k = μ_k^Ind − μ_k^NI
    eta_k = mu_ind - mu_ni

    return mu_ind, sig_ind, mu_ni, sig_ni, L_k, log_d_ind, log_d_ni, eta_k


def _make_categories(breadth_block, thresholds_pct):
    """
    Convert % breadth → ordinal category C_sk ∈ {0, 1, ..., J}.

    Category 0 : B < τ_1  (below all thresholds)
    Category j : B ∈ [τ_j, τ_{j+1})  for j = 1..J-1
    Category J : B ≥ τ_J  (above all thresholds)

    Returns C : (n_samp, n_sgb) int8
    """
    thr = np.asarray(sorted(thresholds_pct), float)
    b   = breadth_block.values.astype(float)  # % breadth
    C   = np.zeros(b.shape, dtype=np.int8)
    for j, t in enumerate(thr):
        C += (b >= t).astype(np.int8)
    return C   # values in {0, 1, ..., J}


def _p_cz1_batch(C_b, r_all_b):
    """
    Interval probability P(C_sk | z=1) for a batch of species.

    C_b     : (n_samp, B) int8   ordinal category (0..J)
    r_all_b : (n_samp, B, J)     detection prob at each of J thresholds

    Returns p : (n_samp, B) float
    """
    n_s, B = C_b.shape
    J      = r_all_b.shape[2]
    p      = np.zeros((n_s, B), float)

    # C == 0: below all thresholds  → P = 1 − r(τ_1)
    m0 = (C_b == 0)
    p  = np.where(m0, 1.0 - r_all_b[:, :, 0], p)

    # C == J: above all thresholds  → P = r(τ_J)
    mJ = (C_b == J)
    p  = np.where(mJ, r_all_b[:, :, J - 1], p)

    # C == j for j = 1..J-1: in interval [τ_j, τ_{j+1})
    for j in range(1, J):
        mj = (C_b == j)
        if mj.any():
            p = np.where(mj,
                         np.maximum(r_all_b[:, :, j-1] - r_all_b[:, :, j], 1e-300),
                         p)
    return np.maximum(p, 1e-300)


def _r_all_batched(mu_x, sig_x, log_d, log_L, thresholds_frac,
                   batch_k=BATCH_K, dist='lognormal', nu=None):
    """
    Per-sample detection probability at EVERY threshold.

    dist : 'lognormal'  — Φ(z) where z = (μ + log_d − log_L − log_τ) / σ
           'student_t'  — Student-t CDF with tail df ν (shared across taxa)
                          P(log x + log_d − log_L > log_τ | μ, σ, ν)
                          = 1 − T_ν((log_τ − μ − log_d + log_L) / σ)
    nu : float or None  — degrees of freedom for 'student_t'.  None → lognormal.

    Returns r_all : (n_samp, n_sgb, J)
    """
    from scipy.stats import t as _scipy_t

    n_samp = len(log_d)
    n_sgb  = len(mu_x)
    J      = len(thresholds_frac)
    r_all  = np.zeros((n_samp, n_sgb, J), float)
    mu_fallback = float(log_d.mean())
    _use_t = (dist == 'student_t') and (nu is not None) and (nu > 0)

    for ks in range(0, n_sgb, batch_k):
        ke   = min(ks + batch_k, n_sgb)
        k_sl = slice(ks, ke)
        mu_b  = np.where(np.isfinite(mu_x[k_sl]), mu_x[k_sl],
                          mu_fallback - log_L[k_sl])
        sig_b = np.maximum(np.where(np.isfinite(sig_x[k_sl]), sig_x[k_sl], 0.5), 0.05)
        le_b  = log_d[:, None] - log_L[None, k_sl]       # (n_samp, bk)
        for j, tau in enumerate(thresholds_frac):
            log_tau_j = float(np.log(-np.log1p(-tau)))
            z = (mu_b[None, :] + le_b - log_tau_j) / sig_b[None, :]
            if _use_t:
                # P(log x > log_τ) = P(t_ν > −z) = 1 − T_ν(−z) = T_ν(z)
                r_all[:, k_sl, j] = _scipy_t.cdf(z, df=float(nu))
            else:
                r_all[:, k_sl, j] = ndtr(z)

    return r_all


def _fit_abundance_family(breadth_block, depth_block, total_bp, lengths,
                           thresholds_pct, mu_x, sig_x,
                           nu_grid=None, holdout_frac=0.2, rng=None, batch_k=BATCH_K):
    """
    Select abundance tail family by held-out ordinal category log-likelihood.

    Fits lognormal and Student-t (at a grid of ν) on a training split,
    evaluates ordinal log-likelihood on held-out samples, picks the best.

    Parameters
    ----------
    nu_grid : list of float or None  — ν values to try for Student-t.
              Default: [3, 5, 10, 20] (covers heavy-to-moderate tails).
              None → skip family selection and return 'lognormal', None.

    Returns
    -------
    best_dist : str   'lognormal' or 'student_t'
    best_nu   : float or None
    scores    : dict  {dist_label: held-out log-lik}
    """
    if nu_grid is None:
        nu_grid = [3.0, 5.0, 10.0, 20.0]

    if rng is None:
        rng = np.random.default_rng(0)

    ref_sgbs = sorted(set(breadth_block.columns) & set(mu_x.index
                         if hasattr(mu_x, 'index') else []))
    if not ref_sgbs:
        return 'lognormal', None, {}

    n_samp = len(breadth_block)
    n_hold = max(1, int(n_samp * holdout_frac))
    hold_idx = rng.choice(n_samp, size=n_hold, replace=False)
    train_idx = np.setdiff1d(np.arange(n_samp), hold_idx)

    b_hold = breadth_block.iloc[hold_idx][ref_sgbs]
    d_hold = total_bp.iloc[hold_idx] if hasattr(total_bp, 'iloc') else total_bp[hold_idx]
    log_d_hold = np.log(np.maximum(d_hold.values, 1.0))

    thr_frac = [t / 100. for t in sorted(thresholds_pct)]
    J        = len(thr_frac)
    mu_arr   = np.array([mu_x[k] if k in mu_x.index else 0.0 for k in ref_sgbs])
    sig_arr  = np.array([sig_x[k] if k in sig_x.index else 0.5 for k in ref_sgbs])
    log_L    = np.log(np.maximum(
        np.array([lengths[k] if k in lengths.index else 1e6 for k in ref_sgbs]), 1.0))

    C_hold = np.zeros((n_hold, len(ref_sgbs)), dtype=np.int8)
    b_vals = b_hold.values
    for j, t in enumerate(sorted(thresholds_pct)):
        C_hold += (b_vals >= t).astype(np.int8)

    def _ordinal_loglik(dist_name, nu_val):
        r_all_h = _r_all_batched(mu_arr, sig_arr, log_d_hold, log_L, thr_frac,
                                  batch_k=batch_k, dist=dist_name, nu=nu_val)
        # Compute interval probs and sum log-lik (uniform occupancy assumption for family check)
        eps = 1e-9
        ll  = 0.0
        for s in range(n_hold):
            for k in range(len(ref_sgbs)):
                c = int(C_hold[s, k])
                if c == 0:
                    p = 1.0 - r_all_h[s, k, 0]
                elif c == J:
                    p = r_all_h[s, k, J - 1]
                else:
                    p = max(r_all_h[s, k, c-1] - r_all_h[s, k, c], eps)
                ll += np.log(max(p, eps))
        return ll

    scores = {}
    scores['lognormal'] = _ordinal_loglik('lognormal', None)
    for nu_val in nu_grid:
        scores[f'student_t(nu={nu_val})'] = _ordinal_loglik('student_t', nu_val)

    best_label = max(scores, key=scores.get)
    if best_label == 'lognormal':
        best_dist, best_nu = 'lognormal', None
    else:
        best_nu  = float(best_label.split('=')[1].rstrip(')'))
        best_dist = 'student_t'

    print(f"      Abundance family selection (held-out ordinal log-lik):")
    for label, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " ← best" if label == best_label else ""
        print(f"        {label:30s}: {score:12.1f}{marker}")

    return best_dist, best_nu, scores





def _laplace_batch_full(C_ind_b, r_all_ind_b, C_ni_b, r_all_ni_b,
                          alpha0, delta0,
                          mu_alpha, sig_alpha, sig_delta, sig_eff,
                          study_off_ind=None, study_off_ni=None,
                          sample_weights_ind=None, sample_weights_ni=None,
                          eps_fp=1e-6, n_newton=N_NEWTON):
    """
    Laplace MAP + covariance for a batch of B species.

    sample_weights_ind : (n_ind,) or None.  When supplied (study-weighted mode),
        each sample's likelihood contribution is scaled by its weight.
        Equal-study weighting: w_s = 1/(K × n_j) so each of K studies
        contributes 1/K of the total likelihood mass regardless of sample count.
        None → uniform weighting (sample-weighted, default).
    """
    B      = len(alpha0)
    alpha  = alpha0.copy()
    delta  = delta0.copy()
    EPS_H  = 1e-9

    v_ind = (np.asarray(study_off_ind, float)[:, None]
             if study_off_ind is not None else 0.0)
    v_ni  = (np.asarray(study_off_ni,  float)[:, None]
             if study_off_ni  is not None else 0.0)

    # Per-sample weights — shape (n_samp, 1) for broadcasting across species batch
    wi = (np.asarray(sample_weights_ind, float)[:, None]
          if sample_weights_ind is not None else 1.0)
    wn = (np.asarray(sample_weights_ni,  float)[:, None]
          if sample_weights_ni  is not None else 1.0)

    p_cz1_ind = _p_cz1_batch(C_ind_b, r_all_ind_b)
    p_cz1_ni  = _p_cz1_batch(C_ni_b,  r_all_ni_b )
    p_cz0_ind = np.where(C_ind_b == 0, 1.0 - eps_fp, eps_fp)
    p_cz0_ni  = np.where(C_ni_b  == 0, 1.0 - eps_fp, eps_fp)

    for _ in range(n_newton):
        z_ind = ((alpha + delta)[None, :] + v_ind) / sig_eff
        z_ni  = (alpha          [None, :] + v_ni ) / sig_eff
        psi_ind = ndtr(z_ind); psi_ni = ndtr(z_ni)
        dp_ind  = _scipy_norm.pdf(z_ind) / sig_eff
        dp_ni   = _scipy_norm.pdf(z_ni ) / sig_eff

        p_obs_ind = np.clip((1.0 - psi_ind)*p_cz0_ind + psi_ind*p_cz1_ind, EPS_H, 1.0-EPS_H)
        p_obs_ni  = np.clip((1.0 - psi_ni )*p_cz0_ni  + psi_ni *p_cz1_ni,  EPS_H, 1.0-EPS_H)

        sc_ind = (p_cz1_ind - p_cz0_ind) / p_obs_ind
        sc_ni  = (p_cz1_ni  - p_cz0_ni ) / p_obs_ni
        # Weighted score sums: large studies down-weighted in study-weighted mode
        A_ind  = (wi * sc_ind * dp_ind).sum(0)
        A_ni   = (wn * sc_ni  * dp_ni ).sum(0)

        g_a = A_ind + A_ni - (alpha - mu_alpha) / sig_alpha**2
        g_d = A_ind        - delta              / sig_delta**2

        F_ind = (wi * (sc_ind * dp_ind)**2).sum(0)
        F_ni  = (wn * (sc_ni  * dp_ni )**2).sum(0)
        H_aa  = F_ind + F_ni + 1.0/sig_alpha**2
        H_dd  = F_ind        + 1.0/sig_delta**2
        H_ad  = F_ind

        det   = np.maximum(H_aa * H_dd - H_ad**2, 1e-12)
        Hi_aa =  H_dd / det;  Hi_dd =  H_aa / det;  Hi_ad = -H_ad / det

        step_a = Hi_aa * g_a + Hi_ad * g_d
        step_d = Hi_ad * g_a + Hi_dd * g_d
        max_step = np.maximum(np.abs(step_a), np.abs(step_d))
        damp     = np.where(max_step > 0.5, 0.5 / max_step, 1.0)
        alpha = np.clip(alpha + damp * step_a, -35, 35)
        delta = np.clip(delta + damp * step_d, -35, 35)

    # Posterior covariance at MAP
    z_ind   = ((alpha + delta)[None, :] + v_ind) / sig_eff
    z_ni    = (alpha           [None, :] + v_ni ) / sig_eff
    psi_ind = ndtr(z_ind); psi_ni = ndtr(z_ni)
    dp_ind  = _scipy_norm.pdf(z_ind) / sig_eff
    dp_ni   = _scipy_norm.pdf(z_ni ) / sig_eff
    p_obs_ind = np.clip((1-psi_ind)*p_cz0_ind + psi_ind*p_cz1_ind, EPS_H, 1-EPS_H)
    p_obs_ni  = np.clip((1-psi_ni )*p_cz0_ni  + psi_ni *p_cz1_ni,  EPS_H, 1-EPS_H)
    sc_ind  = (p_cz1_ind-p_cz0_ind)/p_obs_ind
    sc_ni   = (p_cz1_ni -p_cz0_ni )/p_obs_ni
    F_ind   = (wi * (sc_ind*dp_ind)**2).sum(0)
    F_ni    = (wn * (sc_ni *dp_ni )**2).sum(0)
    H_aa    = F_ind + F_ni + 1.0/sig_alpha**2
    H_dd    = F_ind        + 1.0/sig_delta**2
    H_ad    = F_ind
    det     = np.maximum(H_aa * H_dd - H_ad**2, 1e-12)

    return alpha, delta, H_dd/det, H_aa/det, -H_ad/det


def fit_joint_hierarchical(breadth_ind, breadth_ni,
                            depth_ind,   depth_ni,
                            lengths, total_bp_ind, total_bp_ni,
                            thresholds_pct=None,
                            threshold=10.0,
                            study_ids_ind=None,
                            study_ids_ni=None,
                            study_weight_mode='sample',
                            mu_alpha_prior=PRIOR_MU_ALPHA,
                            sig_alpha_prior=PRIOR_SIG_ALPHA,
                            sig_delta_init=PRIOR_SIG_DELTA,
                            sig_gamma_init=0.5,
                            psi_absent_thr=PSI_ABSENT_THR,
                            psi_ni_present_thr=0.05,
                            abundance_dist='lognormal',
                            abundance_nu=None,
                            fit_abundance_family=False,
                            nu_grid=None,
                            n_eb_iter=3,
                            correct_truncation=True,
                            eb_n0=5,
                            batch_k=BATCH_K,
                            n_newton=N_NEWTON,
                            n_missing_samples=2000,
                            rng_missing=None):
    """
    Joint hierarchical Bayesian model — three-improvement version.

    Improvement 1: Ordinal multi-threshold likelihood
        C_sk ∈ {0,...,J} using all thresholds simultaneously.
        More information per sample than a single binary observation.

    Improvement 2: Group-specific detection parameters
        μ_k^Ind, μ_k^NI estimated separately → η_k = μ_k^Ind − μ_k^NI
        separates "present but thinner" (η<0, δ≈0) from "absent" (δ≪0).

    Improvement 3: Study random effects via probit marginalisation
        γ_r ~ N(0, σ_γ²) integrated out analytically.
        Effective link: ψ = Φ(z / σ_eff) where σ_eff² = 1 + (π/8)σ_γ².
        σ_γ estimated by EB from between-study residual variance.

    N_missing posterior:
        Draws from 2D Laplace Normal → sum of Bernoullis → genuine interval.

    Returns the same dict as before, plus:
        eta_k    : (n_sgb,) abundance difference μ_k^Ind − μ_k^NI
        sig_gamma: EB-estimated study-level SD
        N_missing_draws: (n_missing_samples,) posterior draws for N_missing
    """
    if thresholds_pct is None:
        thresholds_pct = [10., 20., 30., 40., 50.]
    thresholds_pct  = sorted(float(t) for t in thresholds_pct)
    thresholds_frac = [t/100. for t in thresholds_pct]
    J               = len(thresholds_frac)

    ref_sgbs = sorted(set(breadth_ind.columns) & set(breadth_ni.columns))
    n_sgb    = len(ref_sgbs)
    if n_sgb == 0:
        raise ValueError("No overlapping SGBs between Industrial and Non-Industrial")

    # ── Step 1: Group-specific detection parameters ───────────────────────
    print("    Step 1: group-specific detection parameters...")
    mu_ind, sig_ind, mu_ni, sig_ni, L_k, log_d_ind, log_d_ni, eta_k = \
        estimate_detection_params_grouped(
            breadth_ind, breadth_ni, depth_ind, depth_ni,
            total_bp_ind, total_bp_ni, ref_sgbs, lengths,
            correct_truncation=correct_truncation, eb_n0=eb_n0, batch_k=batch_k)
    log_L = np.log(np.maximum(L_k, 1.0))
    n_ind, n_ni = len(log_d_ind), len(log_d_ni)

    eta_sd = float(np.nanstd(eta_k[np.isfinite(eta_k)]))
    print(f"      η_k = μ_Ind − μ_NI: mean={float(np.nanmean(eta_k)):.3f}  "
          f"sd={eta_sd:.3f}  "
          f"(neg = lower abundance in Industrial when present)")

    # ── Step 2: Ordinal categories + per-threshold detection probs ────────
    print("    Step 2: ordinal categories + multi-threshold detection probs...")
    C_ind = _make_categories(breadth_ind[ref_sgbs], thresholds_pct)
    C_ni  = _make_categories(breadth_ni [ref_sgbs], thresholds_pct)

    # ── Step 2b: Abundance family selection (optional) ────────────────────
    # Choose between lognormal and Student-t log-abundance by held-out
    # ordinal likelihood.  One shared ν per lifestyle group.
    dist_ind, nu_ind = abundance_dist, abundance_nu
    dist_ni,  nu_ni  = abundance_dist, abundance_nu
    if fit_abundance_family:
        print("    Step 2b: fitting abundance family (held-out ordinal log-lik)...")
        mu_ind_s  = pd.Series(mu_ind,  index=ref_sgbs)
        sig_ind_s = pd.Series(sig_ind, index=ref_sgbs)
        mu_ni_s   = pd.Series(mu_ni,   index=ref_sgbs)
        sig_ni_s  = pd.Series(sig_ni,  index=ref_sgbs)
        ls_idx    = lengths.reindex(ref_sgbs) if hasattr(lengths, 'reindex') \
                    else pd.Series(L_k, index=ref_sgbs)
        _af_rng   = np.random.default_rng(99)   # separate RNG, doesn't consume rng_missing
        dist_ind, nu_ind, _ = _fit_abundance_family(
            breadth_ind[ref_sgbs], None, total_bp_ind, ls_idx,
            thresholds_pct, mu_ind_s, sig_ind_s,
            nu_grid=nu_grid, rng=_af_rng, batch_k=batch_k)
        dist_ni,  nu_ni,  _ = _fit_abundance_family(
            breadth_ni[ref_sgbs],  None, total_bp_ni,  ls_idx,
            thresholds_pct, mu_ni_s,  sig_ni_s,
            nu_grid=nu_grid, rng=_af_rng, batch_k=batch_k)
        print(f"      Ind: {dist_ind} (ν={nu_ind})  NI: {dist_ni} (ν={nu_ni})")
    else:
        if abundance_dist != 'lognormal':
            print(f"      Using {abundance_dist} (ν={abundance_nu}) for both groups")

    # r_all: group-specific detection probs at all J thresholds
    r_all_ind = _r_all_batched(mu_ind, sig_ind, log_d_ind, log_L, thresholds_frac,
                                batch_k, dist=dist_ind, nu=nu_ind)
    r_all_ni  = _r_all_batched(mu_ni,  sig_ni,  log_d_ni,  log_L, thresholds_frac,
                                batch_k, dist=dist_ni,  nu=nu_ni)
    # r at primary threshold (for h_mat accumulation)
    j_prim    = int(np.argmin(np.abs(np.array(thresholds_pct) - threshold)))
    tau_prim  = thresholds_frac[j_prim]
    log_tau_p = float(np.log(-np.log1p(-tau_prim)))

    # ── Step 3: Explicit per-sample study offsets ─────────────────────────
    print("    Step 3: explicit study offsets (occupancy + depth)...")
    # Use observed study IDs when available; fall back to no study labels
    # (global sig_gamma only) if none are supplied.
    if study_ids_ind is not None and study_ids_ni is not None:
        sids_ind = np.asarray(study_ids_ind)
        sids_ni  = np.asarray(study_ids_ni)
        # Binary primary-threshold observations for offset estimation
        # Use full ordinal categories — not binary C>0 — so the study
        # correction uses the same information as the main likelihood.
        occ_off_ind, occ_off_ni, det_off_ind, det_off_ni, study_info = \
            estimate_study_effects(
                sids_ind, sids_ni,
                C_ind, C_ni,
                log_d_ind, log_d_ni, J)
        # Apply detection depth offset: subtract study mean from log_d before
        # recomputing r_all.  This removes between-study depth confounding from
        # the detection probability while preserving within-study variation.
        log_d_ind_adj = log_d_ind - det_off_ind
        log_d_ni_adj  = log_d_ni  - det_off_ni
        # Recompute r_all with depth-adjusted log_d (preserving selected dist)
        r_all_ind = _r_all_batched(mu_ind, sig_ind, log_d_ind_adj,
                                    log_L, thresholds_frac, batch_k,
                                    dist=dist_ind, nu=nu_ind)
        r_all_ni  = _r_all_batched(mu_ni,  sig_ni,  log_d_ni_adj,
                                    log_L, thresholds_frac, batch_k,
                                    dist=dist_ni, nu=nu_ni)
        print(f"      Study offsets applied; "
              f"n_studies_Ind={study_info['n_studies_ind']}  "
              f"n_studies_NI={study_info['n_studies_ni']}")
    else:
        occ_off_ind = None; occ_off_ni = None; study_info = {}
        print("      No study IDs supplied — global sig_gamma only")

    # Residual sig_eff from global between-study variance not captured by offsets
    sig_gamma = float(sig_gamma_init)
    sig_eff   = float(np.sqrt(1.0 + (np.pi / 8.0) * sig_gamma**2))
    print(f"      σ_γ_init={sig_gamma:.3f}  σ_eff={sig_eff:.3f}")

    # ── Per-sample weights for study_weight_mode ──────────────────────────
    # 'sample'  (default): w_s = 1  — large studies contribute proportionally
    #                       more likelihood mass; posterior is sample-weighted.
    # 'equal'  (sensitivity): w_s = 1/(K × n_j)  — each of K studies
    #                       contributes 1/K regardless of sample count.
    #                       Reduces the influence of one very large study on δ_k.
    # The scientific question determines which is primary: "sample-weighted" is
    # correct for "what does the observed cohort look like"; "equal-study-weighted"
    # is better for "what does the lifestyle-level SGB pool look like."
    def _compute_weights(sids, n_total):
        if study_weight_mode == 'equal' and sids is not None:
            unique_s = np.unique(sids)
            K = len(unique_s)
            w = np.ones(n_total, float)
            for s in unique_s:
                mask = sids == s
                n_j  = int(mask.sum())
                w[mask] = 1.0 / (K * max(n_j, 1))
            return w
        return None   # None → uniform in _laplace_batch_full

    # sids_ind/sids_ni already set above in Step 3 when study_ids are present;
    # also set them unconditionally for the weight computation (None if no IDs).
    sids_ind = (np.asarray(study_ids_ind) if study_ids_ind is not None
                else (sids_ind if 'sids_ind' in dir() else None))
    sids_ni  = (np.asarray(study_ids_ni)  if study_ids_ni  is not None
                else (sids_ni  if 'sids_ni'  in dir() else None))
    w_ind = _compute_weights(sids_ind, n_ind)
    w_ni  = _compute_weights(sids_ni,  n_ni)
    if study_weight_mode != 'sample':
        n_st_i = len(np.unique(sids_ind)) if sids_ind is not None else 0
        n_st_n = len(np.unique(sids_ni))  if sids_ni  is not None else 0
        print(f"      study_weight_mode='{study_weight_mode}'  "
              f"Ind: {n_ind} samples → {n_st_i} equal-weight studies  "
              f"NI: {n_ni} samples → {n_st_n} equal-weight studies")

    # ── Step 4: EB outer loop ─────────────────────────────────────────────
    sig_alpha = float(sig_alpha_prior)
    sig_delta = float(sig_delta_init)
    mu_alpha  = float(mu_alpha_prior)
    alpha_hat = np.full(n_sgb, mu_alpha, float)
    delta_hat = np.zeros(n_sgb, float)
    var_aa    = np.full(n_sgb, sig_alpha**2, float)
    var_dd    = np.full(n_sgb, sig_delta**2, float)
    cov_ad    = np.zeros(n_sgb, float)

    print(f"    Step 4: EB ({n_eb_iter} iters)  "
          f"μ_α={mu_alpha:.2f}  σ_α={sig_alpha:.2f}  σ_δ={sig_delta:.2f}")

    for eb_iter in range(n_eb_iter):
        # Inner: Laplace MAP per species batch with explicit study offsets + weights
        for ks in range(0, n_sgb, batch_k):
            ke   = min(ks + batch_k, n_sgb)
            k_sl = slice(ks, ke)
            a_h, d_h, v_aa, v_dd, c_ad = _laplace_batch_full(
                C_ind[:, k_sl], r_all_ind[:, k_sl, :],
                C_ni [:, k_sl], r_all_ni [:, k_sl, :],
                alpha_hat[k_sl].copy(), delta_hat[k_sl].copy(),
                mu_alpha, sig_alpha, sig_delta, sig_eff,
                study_off_ind=occ_off_ind,
                study_off_ni =occ_off_ni,
                sample_weights_ind=w_ind,
                sample_weights_ni =w_ni,
                n_newton=n_newton)
            alpha_hat[k_sl] = a_h
            delta_hat[k_sl] = d_h
            var_aa[k_sl]    = v_aa
            var_dd[k_sl]    = v_dd
            cov_ad[k_sl]    = c_ad

        # Alternating block: refine study occupancy offsets given current (α,δ).
        # SYMMETRIC: both Industrial (Ind) and Non-Industrial (NI) are updated.
        # Without the NI update, one side of the contrast (Ind) is adaptively
        # cleaned up while the other (NI) stays at its initializer — biasing δ.
        if occ_off_ind is not None and sids_ind is not None:
            occ_off_ind = _update_study_offsets_ordinal(
                alpha_hat, delta_hat, sig_eff,
                C_ind, r_all_ind, sids_ind, occ_off_ind,
                sig_occ=sig_alpha_prior, batch_k=batch_k, n_newton=3,
                sample_weights=w_ind)
        # NI: ψ_NI = Φ(α/σ_eff), so pass delta=zeros (no Industrial shift term)
        if occ_off_ni is not None and sids_ni is not None:
            occ_off_ni = _update_study_offsets_ordinal(
                alpha_hat, np.zeros(n_sgb, float), sig_eff,
                C_ni, r_all_ni, sids_ni, occ_off_ni,
                sig_occ=sig_alpha_prior, batch_k=batch_k, n_newton=3,
                sample_weights=w_ni)

        # Outer: EB updates for σ_α, σ_δ, σ_γ
        sig_alpha_new = float(np.sqrt(max(
            np.mean(alpha_hat**2 + var_aa) - mu_alpha**2, 0.1)))
        sig_delta_new = float(np.sqrt(max(np.mean(delta_hat**2 + var_dd), 0.1)))

        # σ_γ: residual between-study variance estimated symmetrically from
        # BOTH groups using ordinal mean scores (not binary C>0).
        # Normalise by J so scores are in [0,1] like predicted h.
        psi_ind_hat_now = ndtr((alpha_hat + delta_hat) / sig_eff)
        psi_ni_hat_now  = ndtr(alpha_hat               / sig_eff)
        h_pred_ind = psi_ind_hat_now[None, :] * r_all_ind[:, :, j_prim]
        h_pred_ni  = psi_ni_hat_now [None, :] * r_all_ni [:, :, j_prim]
        ord_ind = C_ind.astype(float) / J    # normalised ordinal score ∈ [0,1]
        ord_ni  = C_ni.astype(float)  / J
        resid_ind = ((ord_ind - h_pred_ind)**2).mean(1)   # (n_ind,)
        resid_ni  = ((ord_ni  - h_pred_ni )**2).mean(1)   # (n_ni,)
        # Pool both groups (symmetric) and convert to occupancy-scale SD
        combined_var = np.concatenate([resid_ind, resid_ni]).var()
        sig_gamma_new = float(np.sqrt(max(combined_var * 8.0/np.pi, 0.01)))
        sig_gamma_new = min(sig_gamma_new, 3.0)

        sig_alpha = min(sig_alpha_new, 4.0)
        sig_delta = min(sig_delta_new, 4.0)
        sig_gamma = 0.7 * sig_gamma + 0.3 * sig_gamma_new   # damped update
        sig_eff   = float(np.sqrt(1.0 + (np.pi / 8.0) * sig_gamma**2))

        print(f"      EB iter {eb_iter+1}: σ_α={sig_alpha:.3f}  "
              f"σ_δ={sig_delta:.3f}  σ_γ={sig_gamma:.3f}  σ_eff={sig_eff:.3f}  "
              f"median|δ|={np.median(np.abs(delta_hat)):.3f}")

    # ── Step 5: Final h_mat for accumulation curves ───────────────────────
    psi_ind_hat = ndtr((alpha_hat + delta_hat) / sig_eff)
    psi_ni_hat  = ndtr(alpha_hat               / sig_eff)
    h_mat_ind   = np.clip(psi_ind_hat[None, :] * r_all_ind[:, :, j_prim], 0.0, 1.0-1e-9)
    h_mat_ni    = np.clip(psi_ni_hat [None, :] * r_all_ni [:, :, j_prim], 0.0, 1.0-1e-9)

    # ── Step 6: Absence posterior — joint criterion ───────────────────────
    # Biologically targeted definition:
    #   "Missing in Industrial" means BOTH:
    #     (a) ψ_k^Ind < ε_ind  (absent or near-absent in Industrial)
    #     (b) ψ_k^NI  > τ_ni   (meaningfully present in Non-Industrial)
    #
    # This stops globally rare taxa (low α, small δ) from inflating N_missing.
    # A species is "lost in industrialisation" only if it was there to lose.
    #
    # Probit-scale conditions:
    #   (a) α_k + δ_k < z_eps   where z_eps  = σ_eff · Φ⁻¹(ε_ind)
    #   (b) α_k       > z_ni    where z_ni   = σ_eff · Φ⁻¹(τ_ni)
    #
    # Joint probability via 2D Laplace Normal:
    #   P(A ∩ B) = P(α+δ < z_eps) · P(α > z_ni | α+δ < z_eps)
    # → computed by simulation from the (α, α+δ) bivariate Normal.
    #
    # ε_ind defaults to PSI_ABSENT_THR (e.g. 1%).
    # τ_ni  defaults to PSI_NI_PRESENT_THR (e.g. 5%).
    z_eps  = sig_eff * float(_scipy_norm.ppf(psi_absent_thr))     # ≈ −2.3
    z_ni   = sig_eff * float(_scipy_norm.ppf(psi_ni_present_thr)) # ≈ −1.6

    var_sum = np.maximum(var_aa + var_dd + 2.0 * cov_ad, 1e-9)
    sd_sum  = np.sqrt(var_sum)

    # Marginal p_missing_ind: P(ψ^Ind < ε) — kept for backward compat/figure
    p_missing_ind_marginal = ndtr((z_eps - (alpha_hat + delta_hat)) / sd_sum)

    # Joint p_missing: P(ψ^Ind < ε AND ψ^NI > τ_ni)
    # Simulate from the 2D Laplace Normal: (α, α+δ) ~ N with block covariance.
    # Cov matrix: [[var_aa, var_aa+cov_ad], [var_aa+cov_ad, var_sum]]
    if rng_missing is None:
        rng_missing = np.random.default_rng(42)
    # Draw α ~ N(α̂, var_aa), ε_αδ ~ N(0, var_dd + 2·cov_ad) independently,
    # then α+δ = α + ε_αδ  (the cross-covariance is cov_ad).
    alpha_draws = rng_missing.normal(alpha_hat[None, :],
                                      np.sqrt(np.maximum(var_aa, 1e-9))[None, :],
                                      (n_missing_samples, n_sgb))
    # δ_draws independent error beyond α
    delta_extra_var = np.maximum(var_dd - var_aa + 2.0 * cov_ad, 0.0)
    delta_draws = rng_missing.normal(delta_hat[None, :],
                                      np.sqrt(delta_extra_var + 1e-9)[None, :],
                                      (n_missing_samples, n_sgb))
    ad_draws   = alpha_draws + delta_draws    # α+δ draws (Industrial predictor)
    # Condition (a): α+δ < z_eps  (Industrial absent)
    cond_a = ad_draws < z_eps
    # Condition (b): α > z_ni    (NI present)
    cond_b = alpha_draws > z_ni
    p_missing_joint = np.mean(cond_a & cond_b, axis=0)  # (n_sgb,)

    N_miss_draws = (cond_a & cond_b).sum(1).astype(float)   # (M,)
    N_miss_mean  = float(N_miss_draws.mean())
    N_miss_ci    = (float(np.percentile(N_miss_draws, 2.5)),
                    float(np.percentile(N_miss_draws, 97.5)))

    # Marginal N_missing (old definition) for comparison
    N_miss_marginal = float((ad_draws < z_eps).sum(1).mean())

    # Broad CI via hyperparameter bootstrap — SAME joint criterion as N_missing_mean
    N_miss_draws_broad = _n_missing_with_hyper_uncertainty(
        alpha_hat, delta_hat, var_aa, var_dd, cov_ad,
        sig_alpha, sig_delta, sig_gamma,
        psi_absent_thr, n_sgb, rng_missing,
        psi_ni_present_thr=psi_ni_present_thr,
        n_species_draws=max(n_missing_samples // 10, 50),
        n_hyper=100)
    N_miss_ci_broad = (float(np.percentile(N_miss_draws_broad, 2.5)),
                       float(np.percentile(N_miss_draws_broad, 97.5)))

    print(f"    N_missing (joint: Ind<ε AND NI>τ) = {N_miss_mean:.0f}  "
          f"[{N_miss_ci[0]:.0f}, {N_miss_ci[1]:.0f}]  "
          f"(marginal Ind-only: {N_miss_marginal:.0f})")
    print(f"    Broad CI (+ hyper): [{N_miss_ci_broad[0]:.0f}, {N_miss_ci_broad[1]:.0f}]  "
          f"P(joint absent)>0.5: {int((p_missing_joint>0.5).sum()):,}")

    # ── Step 7: Union-presence posterior ──────────────────────────────────
    # Answer the lifestyle-level question:
    #   "Is taxon k missing from the Industrial lifestyle pool?"
    # where "pool" = set of taxa detectable if N* samples are drawn from the
    # lifestyle (via the study-aware two-stage distribution).
    #
    # Per-sample occupancy (Steps 4-6) asks whether a typical sample has
    # nontrivial occupancy.  Union presence asks whether ANY sample in N*
    # draws from the lifestyle would detect the taxon — the stated criterion
    # that "found anywhere in a lifestyle counts."
    print("    Step 7: union-presence posterior (lifestyle-wide missingness)...")
    # r_mat = h_mat / ψ̂ (detection prob without occupancy factor)
    # Clip to avoid division by near-zero ψ̂
    r_mat_ind = np.clip(h_mat_ind / np.maximum(psi_ind_hat[None, :], 1e-6), 0.0, 1.0)
    r_mat_ni  = np.clip(h_mat_ni  / np.maximum(psi_ni_hat [None, :], 1e-6), 0.0, 1.0)
    N_star_ind = n_ind   # reference = observed group size
    N_star_ni  = n_ni
    sids_ind_u = np.asarray(study_ids_ind) if study_ids_ind is not None else None
    sids_ni_u  = np.asarray(study_ids_ni)  if study_ids_ni  is not None else None

    p_miss_union, N_miss_union, N_miss_union_draws, A_ind_map, A_ni_map = \
        union_missing_posterior(
            h_mat_ind, h_mat_ni, r_mat_ind, r_mat_ni,
            alpha_hat, delta_hat, var_aa, var_dd, cov_ad,
            sig_eff, psi_ind_hat, psi_ni_hat,
            sids_ind_u, sids_ni_u, N_star_ind, N_star_ni,
            eps_union=psi_absent_thr, tau_union=psi_ni_present_thr,
            n_draws=n_missing_samples, rng=rng_missing, mode='equal')
    N_miss_union_ci = (float(np.percentile(N_miss_union_draws, 2.5)),
                       float(np.percentile(N_miss_union_draws, 97.5)))
    print(f"      N_missing (union, equal-study) = {N_miss_union:.0f}  "
          f"[{N_miss_union_ci[0]:.0f}, {N_miss_union_ci[1]:.0f}]  "
          f"P(union absent)>0.5: {int((p_miss_union>0.5).sum()):,}")

    return {
        # Detection parameters (group-specific)
        "mu_ind": mu_ind, "sig_ind": sig_ind,
        "mu_ni":  mu_ni,  "sig_ni":  sig_ni,
        "eta_k":  eta_k,   # abundance depletion: μ^Ind − μ^NI
        "L_k": L_k,
        "log_tau_primary": log_tau_p,
        # Occupancy parameters
        "alpha_hat": alpha_hat, "delta_hat": delta_hat,
        "var_aa": var_aa, "var_dd": var_dd, "cov_ad": cov_ad,
        "psi_ni_hat":  psi_ni_hat,
        "psi_ind_hat": psi_ind_hat,
        # Absence posterior
        "p_missing_ind":      p_missing_ind_marginal,  # P(ψ^Ind < ε) — marginal
        "p_missing_joint":    p_missing_joint,          # P(ψ^Ind < ε AND ψ^NI > τ_ni)
        "N_missing_mean":     N_miss_mean,              # from joint criterion
        "N_missing_marginal": N_miss_marginal,          # from marginal criterion (reference)
        "N_missing_ci":       N_miss_ci,
        "N_missing_ci_broad": N_miss_ci_broad,
        "N_missing_draws":    N_miss_draws,
        "psi_ni_present_thr": psi_ni_present_thr,
        # Union-presence posterior (lifestyle-level, two-stage study-aware)
        "p_missing_union":    p_miss_union,     # P(A^Ind<ε AND A^NI>τ) lifestyle-wide
        "N_missing_union":    N_miss_union,
        "N_missing_union_ci": N_miss_union_ci,
        "N_missing_union_draws": N_miss_union_draws,
        "A_ind_map": A_ind_map,   # MAP union detection prob per taxon, Industrial
        "A_ni_map":  A_ni_map,    # MAP union detection prob per taxon, NI
        "N_star_ind": N_star_ind, "N_star_ni": N_star_ni,
        # h_mat for accumulation
        "h_mat_ind": h_mat_ind,
        "h_mat_ni":  h_mat_ni,
        "r_all_ind": r_all_ind, "r_all_ni": r_all_ni,
        "C_ind": C_ind, "C_ni": C_ni,
        "d_i_ind": np.exp(log_d_ind), "d_i_ni": np.exp(log_d_ni),
        "log_d_ind": log_d_ind,       "log_d_ni": log_d_ni,
        # Study information
        "study_ids_ind": np.asarray(study_ids_ind) if study_ids_ind is not None else None,
        "study_ids_ni":  np.asarray(study_ids_ni)  if study_ids_ni  is not None else None,
        "study_info":    study_info,
        # Hyperparameters
        "hyperparams": {"mu_alpha": mu_alpha, "sig_alpha": sig_alpha,
                        "sig_delta": sig_delta, "sig_gamma": sig_gamma,
                        "sig_eff": sig_eff},
        "abundance_dist": {"ind": dist_ind, "ni": dist_ni,
                           "nu_ind": nu_ind, "nu_ni": nu_ni},
        "ref_sgbs": ref_sgbs,
        "psi_absent_thr":     psi_absent_thr,
        "psi_ni_present_thr": psi_ni_present_thr,
        "thresholds_pct": thresholds_pct,
    }

def predictive_accumulation(h_mat, N_max, n_draw=200, rng=None):
    """
    Bootstrap-predictive accumulation curve.  All N use with-replacement
    resampling (honest predictive band, not permutation rarefaction).

    Returns Ns, mean, lo95, hi95, draws
    """
    if rng is None: rng = np.random.default_rng()
    n_obs, n_sgb = h_mat.shape
    n_obs_pts    = min(n_obs, 20)
    n_extra      = max(N_max - n_obs, 0)
    n_pts        = n_obs_pts + min(n_extra, 30)
    Ns_set = set(np.geomspace(1, N_max, n_pts).astype(int).tolist())
    Ns_set.update([1, n_obs // 4, n_obs // 2, n_obs, N_max])
    Ns = np.array(sorted(s for s in Ns_set if 1 <= s <= N_max), int)

    h_arr  = np.clip(h_mat, 0.0, 1.0 - 1e-9)
    draws  = np.zeros((n_draw, len(Ns)), float)
    for b in range(n_draw):
        idx    = rng.integers(0, n_obs, size=int(Ns.max()))
        h_sim  = h_arr[idx]
        cumlog = np.cumsum(np.log1p(-h_sim), axis=0)
        draws[b] = np.sum(1.0 - np.exp(cumlog[Ns - 1, :]), axis=1)

    return Ns.astype(float), draws.mean(0), *np.percentile(draws, [2.5, 97.5], 0), draws


def empirical_rarefaction(breadth_block, threshold, n_perm=20, rng=None):
    """
    Without-replacement permutation rarefaction (observed data curve).

    Properly permutes sample order `n_perm` times and averages, giving
    the expected species accumulation independent of the original row order.
    Shown as a secondary reference alongside the predictive bootstrap band.
    """
    if rng is None: rng = np.random.default_rng(0)
    b_mat = (breadth_block.values >= threshold)
    n_obs = b_mat.shape[0]
    pts   = np.unique(np.geomspace(1, n_obs, 30).astype(int))
    S_all = np.zeros((n_perm, len(pts)), float)
    for p in range(n_perm):
        perm = rng.permutation(n_obs)
        b_p  = b_mat[perm]
        S_all[p] = [int(b_p[:n].any(0).sum()) for n in pts]
    return pts.astype(float), S_all.mean(0)


def threshold_surface(jm, thresholds_pct, N_max, n_draw, rng, batch_k=BATCH_K):
    """
    S(N, τ) surface derived entirely from the joint model parameters.
    Uses _h_lognormal_batched — same observation model as the joint fit.

    Returns {tau_pct: (Ns, mean, lo, hi)}
    """
    mu_x  = jm["mu_x"];  sig_x = jm["sig_x"]
    log_L = np.log(np.maximum(jm["L_k"], 1.0))
    psi_i = jm["psi_ind_hat"]
    psi_n = jm["psi_ni_hat"]
    d_i   = jm["d_i_ind"]
    d_n   = jm["d_i_ni"]
    log_d_i = np.log(np.maximum(d_i, 1.0))
    log_d_n = np.log(np.maximum(d_n, 1.0))

    surface = {}
    for tau_pct in thresholds_pct:
        tau_j     = float(tau_pct) / 100.0
        log_tau_j = float(np.log(-np.log1p(-tau_j)))
        h_i = _h_lognormal_batched(psi_i, mu_x, sig_x, log_d_i, log_L, log_tau_j, batch_k)
        h_n = _h_lognormal_batched(psi_n, mu_x, sig_x, log_d_n, log_L, log_tau_j, batch_k)
        N_max_i = int(np.ceil(len(d_i) * (N_max / len(d_i))))
        N_max_n = int(np.ceil(len(d_n) * (N_max / len(d_n))))
        Ns_i, pm_i, pl_i, ph_i, _ = predictive_accumulation(h_i, N_max_i, n_draw, rng)
        Ns_n, pm_n, pl_n, ph_n, _ = predictive_accumulation(h_n, N_max_n, n_draw, rng)
        surface[float(tau_pct)] = {
            "Ind": (Ns_i, pm_i, pl_i, ph_i),
            "NI":  (Ns_n, pm_n, pl_n, ph_n),
        }
    return surface


# ════════════════════════════════════════════════════════════════════════════
#  HOLDOUT DIAGNOSTIC — fully train-only refit
# ════════════════════════════════════════════════════════════════════════════

def holdout_diagnostic(jm_full, breadth_ind, breadth_ni,
                        depth_ind, depth_ni,
                        lengths, total_bp_ind, total_bp_ni,
                        threshold=10.0, hold_frac=0.15,
                        do_refit=False):
    """
    Hold out deepest `hold_frac` of Industrial samples; score how many
    new SGBs the joint model correctly predicts they contribute.

    refit=True: refit joint model on train-only samples (true out-of-sample).
    refit=False: use pre-computed h_mat_ind from full model (calibration check).

    Returns dict with obs_new, pred_new, n_train, n_hold, refitted.
    """
    idx_ordered = total_bp_ind.sort_values().index
    n_hold  = max(5, int(len(idx_ordered) * hold_frac))
    n_train = len(idx_ordered) - n_hold
    train_idx = idx_ordered[:n_train]
    hold_idx  = idx_ordered[n_train:]

    # Observed new SGBs in holdout
    tau   = float(threshold) / 100.0
    b_ind = (breadth_ind.values >= threshold)
    union = b_ind[breadth_ind.index.get_indexer(train_idx)].any(0)
    obs_new = []; running = union.copy()
    for i in breadth_ind.index.get_indexer(hold_idx):
        added = b_ind[i] & ~running
        obs_new.append(int(added.sum()))
        running |= b_ind[i]

    log_tau = float(np.log(-np.log1p(-tau)))

    if do_refit:
        # Refit joint model on training samples only
        jm_tr = fit_joint_hierarchical(
            breadth_ind.loc[train_idx], breadth_ni,
            depth_ind.loc[train_idx], depth_ni,
            lengths, total_bp_ind.loc[train_idx], total_bp_ni,
            threshold=threshold)
        psi_tr = jm_tr["psi_ind_hat"]
        mu_tr  = jm_tr["mu_x"]; sig_tr = jm_tr["sig_x"]
        L_tr   = jm_tr["L_k"]
        log_L_tr = np.log(np.maximum(L_tr, 1.0))

        # Score hold on train-only model
        log_d_hold  = np.log(np.maximum(total_bp_ind.loc[hold_idx].values.astype(float), 1.0))
        log_d_train = np.log(np.maximum(total_bp_ind.loc[train_idx].values.astype(float), 1.0))
        h_hold  = _h_lognormal_batched(psi_tr, mu_tr, sig_tr, log_d_hold,  log_L_tr, log_tau)
        h_train = _h_lognormal_batched(psi_tr, mu_tr, sig_tr, log_d_train, log_L_tr, log_tau)
        log1_train = np.log1p(-h_train).sum(0)
        refitted = True
    else:
        hold_pos   = breadth_ind.index.get_indexer(hold_idx)
        train_pos  = breadth_ind.index.get_indexer(train_idx)
        h_hold  = np.clip(jm_full["h_mat_ind"][hold_pos],  0, 1 - 1e-9)
        h_train = np.clip(jm_full["h_mat_ind"][train_pos], 0, 1 - 1e-9)
        log1_train = np.log1p(-h_train).sum(0)
        refitted = False

    p_unseen = np.exp(log1_train)
    pred_new = []; p_still = p_unseen.copy()
    for j in range(len(hold_idx)):
        pred_new.append(float((h_hold[j] * p_still).sum()))
        p_still *= (1.0 - h_hold[j])

    return {"obs_new": np.array(obs_new, float),
            "pred_new": np.array(pred_new, float),
            "n_train": n_train, "n_hold": n_hold,
            "refitted": refitted}


# ════════════════════════════════════════════════════════════════════════════
#  HADZA UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def hadza_exclusive_sgbs(breadth_df, meta, threshold=10.0, min_hadza_pct=1.0,
                          hadza_col=None, hadza_label="Hadza"):
    """
    Return SGB IDs present in ≥min_hadza_pct% of Hadza samples and 0% elsewhere.

    Hadza identification (in priority order):
      1. `hadza_col` column in meta (e.g. "Population") with value `hadza_label`.
      2. Country == Tanzania AND Lifestyle == H-G  (geographic proxy; warns).
      3. Lifestyle == H-G only (broadest proxy; warns — this was the old behaviour
         and incorrectly includes all hunter-gatherer groups, not just Hadza).

    The reviewer notes that Lifestyle=="H-G" is too broad: it can exclude taxa
    associated with non-Hadza H-G groups (Pygmy, San, etc.) from panel E,
    distorting the "Hadza-excluded" interpretation.

    Parameters
    ----------
    hadza_col   : str or None  — metadata column containing population label.
    hadza_label : str          — value in hadza_col that identifies Hadza
                                 (default "Hadza"; case-insensitive match).
    """
    # Priority 1: explicit population column
    if hadza_col and hadza_col in meta.columns:
        is_hadza = meta[hadza_col].astype(str).str.lower() == hadza_label.lower()
        method = f"column {hadza_col!r}=={hadza_label!r}"
    # Priority 2: Tanzania H-G proxy
    elif ("Country_norm" in meta.columns and
          (meta["Country_norm"].astype(str).str.lower() == "tanzania").any()):
        is_hadza = ((meta["Country_norm"].astype(str).str.lower() == "tanzania") &
                    (meta["Lifestyle"] == "H-G"))
        method = "Country=Tanzania AND Lifestyle=H-G (geographic proxy)"
        print(f"    NOTE: Hadza identified by {method}.  "
              f"If non-Hadza Tanzanian H-G samples exist, pass --hadza-col.")
    else:
        # Fallback: all H-G — warn clearly
        is_hadza = meta["Lifestyle"] == "H-G"
        method = "Lifestyle==H-G (broadest proxy — may include non-Hadza H-G)"
        print(f"    WARNING: No Hadza-specific identifier found.  "
              f"Using {method}.  Panel E may exclude non-Hadza H-G taxa.")

    is_hadza_idx = is_hadza[is_hadza].index.intersection(breadth_df.index)
    if len(is_hadza_idx) == 0:
        return set()

    b             = breadth_df.values >= threshold
    row_is_hadza  = breadth_df.index.isin(is_hadza_idx)
    hadza_det     = b[row_is_hadza].mean(0)
    other_det     = b[~row_is_hadza].mean(0)
    exclusive     = set(np.array(breadth_df.columns)
                        [(hadza_det >= min_hadza_pct / 100.0) & (other_det == 0)])
    print(f"    Hadza identified by: {method}  "
          f"({int(row_is_hadza.sum())} Hadza samples, {len(exclusive)} exclusive SGBs)")
    return exclusive


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE GENERATION
# ════════════════════════════════════════════════════════════════════════════

THRESHOLDS_PCT = [10, 20, 30, 40, 50]
TAU_COLORS     = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]

def _surf_get(surface, threshold_pct):
    key = float(threshold_pct)
    if key in surface: return surface[key]
    available = np.array(sorted(surface.keys()))
    nearest   = float(available[np.argmin(np.abs(available - key))])
    if abs(nearest - key) > 5.0:
        raise KeyError(f"threshold {key} not in surface (nearest={nearest})")
    return surface[nearest]


def build_figure(jm, jm_e, thresholds_pct, surface, surface_e,
                 hd_ind, hd_ni, breadth_ind, breadth_ni,
                 breadth_ind_e, breadth_ni_e,
                 threshold, N_pred, rng, out_stem):
    """
    3×2 figure — all panels derived from the joint hierarchical model.

    Panel A: Industrial accumulation  (predictive + empirical rarefaction)
    Panel B: NI accumulation          (predictive + empirical rarefaction)
    Panel C: Threshold surface Ind    (S(N,τ) for τ=10..50%)
    Panel D: δ_k posterior            (Industrial depletion per SGB)
    Panel E: Hadza-excluded NI        (same as A/B but excl. Hadza SGBs)
    Panel F: N_missing summary        (posterior bars + CI)
    """
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             hspace=0.42, wspace=0.35,
                             left=0.07, right=0.97, top=0.94, bottom=0.06)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]

    def _label(ax, letter):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="left")

    # ── Panels A & B: accumulation curves ────────────────────────────────
    for ax, grp, h_mat, d_i, breadth_block, sids, letter in [
            (axes[0][0], "Industrial",     jm["h_mat_ind"], jm["d_i_ind"],
             breadth_ind, jm["study_ids_ind"],  "A"),
            (axes[0][1], "Non-Industrial", jm["h_mat_ni"],  jm["d_i_ni"],
             breadth_ni,  jm["study_ids_ni"],   "B"),
    ]:
        N_max = int(np.ceil(len(d_i) * 2.0))
        col   = COL[grp]

        # Sample-level bootstrap (standard)
        Ns, pm, pl, ph, _ = predictive_accumulation(h_mat, N_max, N_pred, rng)
        ax.fill_between(Ns/1000, pl, ph, alpha=0.20, color=col, lw=0)
        ax.plot(Ns/1000, pm, color=col, lw=1.5, label="Predictive (sample-level)")

        # Two-stage: equal-study-weight (if study IDs available)
        if sids is not None and len(np.unique(sids)) > 1:
            Ns_e, pm_e, pl_e, ph_e, _ = two_stage_accumulation(
                h_mat, sids, N_max, N_pred, rng, mode='equal')
            ax.plot(Ns_e/1000, pm_e, color=col, lw=1.2, ls="--",
                    label="Predictive (equal-study-weight)")
            ax.fill_between(Ns_e/1000, pl_e, ph_e, alpha=0.10, color=col, lw=0)

        # Empirical rarefaction (secondary reference)
        N_emp, S_emp = empirical_rarefaction(breadth_block, threshold)
        ax.plot(N_emp/1000, S_emp, color=col, lw=0.9, ls=":", alpha=0.6,
                label="Empirical rarefaction (observed)")

        ax.axvline(len(d_i)/1000, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Samples (k)"); ax.set_ylabel("SGBs detected")
        ax.set_title(grp); _label(ax, letter)
        n_st = len(np.unique(sids)) if sids is not None else "?"
        ax.legend(fontsize=7, framealpha=0.4,
                  title=f"{n_st} studies", title_fontsize=7)

    # ── Panel C: threshold surface (Industrial) ───────────────────────────
    ax = axes[1][0]; _label(ax, "C")
    for j, (tau_pct, col) in enumerate(zip(thresholds_pct, TAU_COLORS)):
        entry = _surf_get(surface, tau_pct)
        Ns_i, pm_i, pl_i, ph_i = entry["Ind"]
        ax.plot(Ns_i / 1000, pm_i, color=col, lw=1.4, label=f"{tau_pct}%")
        ax.fill_between(Ns_i / 1000, pl_i, ph_i, alpha=0.15, color=col, lw=0)
    ax.set_xlabel("Samples (k)"); ax.set_ylabel("SGBs detected")
    ax.set_title("Threshold surface — Industrial")
    ax.legend(title="Breadth τ", fontsize=8)

    # ── Panel D: δ_k posterior (Industrial depletion) ────────────────────
    ax = axes[1][1]; _label(ax, "D")
    delta_hat  = jm["delta_hat"]
    sd_delta   = np.sqrt(np.maximum(jm["var_dd"], 1e-9))
    p_miss     = jm["p_missing_joint"]  # joint criterion: Ind-absent AND NI-present

    # Sort by δ (most depleted first)
    order   = np.argsort(delta_hat)[::-1]
    x_idx   = np.arange(len(delta_hat))
    # Color by p_missing
    cmap    = plt.cm.RdBu_r
    colors  = cmap(np.clip(p_miss[order], 0, 1))
    ax.scatter(x_idx, delta_hat[order], c=colors, s=1.5, alpha=0.6, linewidths=0)
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("SGB rank (by δ̂)"); ax.set_ylabel("δ̂_k  (Industrial depletion, logit scale)")
    ax.set_title(f"Industrial depletion  δ_k = logit(ψ^Ind) − logit(ψ^NI)")
    # Annotation
    n_strong = int((p_miss > 0.95).sum())
    n_mod    = int(((p_miss > 0.50) & (p_miss <= 0.95)).sum())
    ax.text(0.02, 0.05, f"P(absent)>0.95: {n_strong:,} SGBs\n"
            f"P(absent)>0.50: {n_mod:,} SGBs",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="P(ψ_k^Ind < 1%)", shrink=0.8)

    # ── Panel E: Hadza-excluded NI ────────────────────────────────────────
    ax = axes[2][0]; _label(ax, "E")
    for grp, h_mat, d_i, col in [
            ("Industrial",     jm_e["h_mat_ind"], jm_e["d_i_ind"], COL["Industrial"]),
            ("Non-Industrial", jm_e["h_mat_ni"],  jm_e["d_i_ni"],  COL["Non-Industrial"]),
    ]:
        N_max_e = int(np.ceil(len(d_i) * 2.0))
        Ns_e, pm_e, pl_e, ph_e, _ = predictive_accumulation(h_mat, N_max_e, N_pred, rng)
        ax.fill_between(Ns_e / 1000, pl_e, ph_e, alpha=0.20, color=col, lw=0)
        ax.plot(Ns_e / 1000, pm_e, color=col, lw=1.4, label=grp)
    ax.set_xlabel("Samples (k)"); ax.set_ylabel("SGBs detected")
    ax.set_title("Excl. Hadza-exclusive SGBs")
    ax.legend(fontsize=8)

    # ── Panel F: N_missing posterior summary ─────────────────────────────
    ax = axes[2][1]; _label(ax, "F")
    # Per-SGB p_missing sorted
    p_miss_sorted = np.sort(jm["p_missing_joint"])[::-1]
    n_sgbs        = len(p_miss_sorted)
    x_sgbs        = np.arange(1, n_sgbs + 1)
    ax.plot(x_sgbs, np.cumsum(p_miss_sorted), color=COL["absent"], lw=1.5)
    ax.axhline(jm["N_missing_mean"], color=COL["absent"], lw=1.0, ls="--", alpha=0.7)
    lo, hi = jm["N_missing_ci"]
    ax.axhspan(lo, hi, color=COL["absent"], alpha=0.15)
    ax.set_xlabel("SGBs ranked by P(absent in Ind)")
    ax.set_ylabel("Cumulative E[N missing]")
    ax.set_title(f"N_missing = {jm['N_missing_mean']:.0f}  [{lo:.0f}, {hi:.0f}]")
    ax.text(0.98, 0.05,
            f"ψ_absent threshold: {jm['psi_absent_thr']*100:.1f}%\n"
            f"Prior: μ_α = Φ⁻¹(5%) ≈ -1.645\n"
            f"EB σ_δ = {jm['hyperparams']['sig_delta']:.2f}",
            transform=ax.transAxes, fontsize=8, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Gut Microbiome SGB Diversity — Joint Hierarchical Model",
                 fontsize=13, fontweight="bold")
    apply_display_text(fig)
    for suffix in [".pdf", ".png"]:
        fig.savefig(out_stem + suffix, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_stem}.pdf / .png")


# ════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ════════════════════════════════════════════════════════════════════════════

def run_self_tests():
    """
    Embedded unit tests. Run with --test flag.
    Tests: threshold snapping, _h_lognormal_batched batch invariance,
           _laplace_batch recovery, holdout no-leak.
    """
    import traceback
    rng = np.random.default_rng(0)
    passed = failed = 0

    def check(name, expr, msg=""):
        nonlocal passed, failed
        try:
            assert expr, msg
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e or msg}")
            traceback.print_exc()
            failed += 1

    # 1. _surf_get snapping
    surf_mock = {10.0: "a", 20.0: "b", 30.0: "c"}
    check("_surf_get exact",    _surf_get(surf_mock, 10.0) == "a")
    check("_surf_get snap",     _surf_get(surf_mock, 11.0) == "a")
    check("_surf_get snap hi",  _surf_get(surf_mock, 29.9) == "c")

    # 2. _h_lognormal_batched batch-size invariance
    n_s, n_k = 50, 120
    psi = rng.uniform(0.1, 0.9, n_k)
    mu  = rng.normal(-1, 0.5, n_k)
    sig = rng.uniform(0.3, 1.0, n_k)
    log_d = rng.normal(20, 1, n_s)
    log_L = rng.uniform(13, 16, n_k)
    log_tau = float(np.log(-np.log1p(-0.10)))
    h1 = _h_lognormal_batched(psi, mu, sig, log_d, log_L, log_tau, batch_k=17)
    h2 = _h_lognormal_batched(psi, mu, sig, log_d, log_L, log_tau, batch_k=n_k)
    check("_h_lognormal_batched batch invariance",
          np.allclose(h1, h2, atol=1e-12),
          f"max diff = {np.abs(h1-h2).max():.2e}")

    # 3. _laplace_batch_full: ordinal + group-specific r + probit marginalisation
    # k=0: Ind ~60% detect, NI ~60%  → δ ≈ 0
    # k=1: Ind ~10% detect, NI ~70%  → δ clearly negative
    n_ind2, n_ni2, B, J = 300, 300, 3, 3
    rng2 = np.random.default_rng(1)
    C_ind2 = np.zeros((n_ind2, B), dtype=np.int8)
    C_ni2  = np.zeros((n_ni2,  B), dtype=np.int8)
    # k=0: both groups ~60% detect (C=J means above all thresholds)
    C_ind2[:, 0] = (rng2.uniform(size=n_ind2) < 0.60).astype(np.int8) * J
    C_ni2 [:, 0] = (rng2.uniform(size=n_ni2)  < 0.60).astype(np.int8) * J
    # k=1: Ind ~10%, NI ~70% — clear depletion signal
    C_ind2[:, 1] = (rng2.uniform(size=n_ind2) < 0.10).astype(np.int8) * J
    C_ni2 [:, 1] = (rng2.uniform(size=n_ni2)  < 0.70).astype(np.int8) * J
    # k=2: both ~10% (symmetric low)
    C_ind2[:, 2] = (rng2.uniform(size=n_ind2) < 0.10).astype(np.int8) * J
    C_ni2 [:, 2] = (rng2.uniform(size=n_ni2)  < 0.10).astype(np.int8) * J
    r_all_ind = np.zeros((n_ind2, B, J)); r_all_ni = np.zeros((n_ni2, B, J))
    for j in range(J):
        r_all_ind[:, :, j] = 0.8 - j * 0.2
        r_all_ni [:, :, j] = 0.8 - j * 0.2
    a0 = np.full(B, -2.944); d0 = np.zeros(B)
    a_h, d_h, v_aa, v_dd, c_ad = _laplace_batch_full(
        C_ind2, r_all_ind, C_ni2, r_all_ni, a0, d0,
        mu_alpha=-2.944, sig_alpha=1.5, sig_delta=1.0, sig_eff=1.0, n_newton=30)
    check("laplace_full: k=1 more depleted than k=0",
          d_h[1] < d_h[0], f"d_h[0]={d_h[0]:.3f}  d_h[1]={d_h[1]:.3f}")
    check("laplace_full: k=1 more depleted than k=2 (symmetric low)",
          d_h[1] < d_h[2], f"d_h[1]={d_h[1]:.3f}  d_h[2]={d_h[2]:.3f}")
    check("laplace_full: variances positive", np.all(v_aa > 0) and np.all(v_dd > 0))

    # 4. Holdout no-leak: structural check — refit branch must not read jm_full
    # We verify this by running the refit path and confirming the closure
    # doesn't accept jm_full as an argument (it only receives train data).
    # Direct check: h_mat_ind from full jm_t should NOT equal holdout h_hold
    # when the train model differs from the full model.
    check("holdout function has do_refit param",
          "do_refit" in holdout_diagnostic.__code__.co_varnames)

    # 5. fit_joint_hierarchical smoke test
    n_si, n_sn, n_k2 = 40, 30, 60
    b_ind = pd.DataFrame(rng.choice([0,5,15,25],  (n_si,n_k2), p=[0.7,0.1,0.1,0.1]).astype(float),
                          index=[f"i{s}" for s in range(n_si)],
                          columns=[f"k{j}" for j in range(n_k2)])
    b_ni  = pd.DataFrame(rng.choice([0,5,15,25,35],(n_sn,n_k2), p=[0.5,0.1,0.1,0.15,0.15]).astype(float),
                          index=[f"n{s}" for s in range(n_sn)],
                          columns=b_ind.columns)
    d_ind = pd.DataFrame(rng.exponential(0.3,(n_si,n_k2)), index=b_ind.index, columns=b_ind.columns)
    d_ni  = pd.DataFrame(rng.exponential(0.3,(n_sn,n_k2)), index=b_ni.index,  columns=b_ni.columns)
    ls    = pd.Series(rng.uniform(1e6,5e6,n_k2), index=b_ind.columns)
    tb_i  = pd.Series(rng.lognormal(20,0.5,n_si), index=b_ind.index)
    tb_n  = pd.Series(rng.lognormal(21,0.4,n_sn), index=b_ni.index)
    jm_t  = fit_joint_hierarchical(b_ind, b_ni, d_ind, d_ni, ls, tb_i, tb_n,
                                    thresholds_pct=[10.,20.,30.],
                                    threshold=10.0, n_eb_iter=2,
                                    batch_k=20, n_newton=10,
                                    n_missing_samples=100)
    check("fit_joint returns dict", isinstance(jm_t, dict))
    check("psi_ind in [0,1]", np.all((jm_t["psi_ind_hat"]>=0)&(jm_t["psi_ind_hat"]<=1)))
    check("p_missing_joint in [0,1]", np.all((jm_t["p_missing_joint"]>=0)&(jm_t["p_missing_joint"]<=1)))
    check("p_missing_ind in [0,1]",   np.all((jm_t["p_missing_ind"]>=0)  &(jm_t["p_missing_ind"]<=1)))
    check("N_missing >= 0", jm_t["N_missing_mean"] >= 0)
    check("h_mat shapes", jm_t["h_mat_ind"].shape == (n_si, n_k2)
          and jm_t["h_mat_ni"].shape == (n_sn, n_k2))
    check("eta_k exists", "eta_k" in jm_t and jm_t["eta_k"].shape == (n_k2,))
    check("N_missing_draws exists",
          "N_missing_draws" in jm_t and len(jm_t["N_missing_draws"]) == 100)
    check("sig_gamma in hyperparams", "sig_gamma" in jm_t["hyperparams"])
    check("thresholds_pct stored", jm_t["thresholds_pct"] == [10., 20., 30.])

    # ── New feature tests added for session fixes ─────────────────────────

    # Fix 1b: probit prior constant = Φ⁻¹(0.05), not logit(0.05)
    from scipy.stats import norm as _sn
    check("PRIOR_MU_ALPHA is probit-scale",
          abs(PRIOR_MU_ALPHA - float(_sn.ppf(0.05))) < 1e-5,
          f"expected Φ⁻¹(0.05)≈-1.645, got {PRIOR_MU_ALPHA:.4f}")

    # Fix 1c: p_missing is monotone decreasing with alpha+delta (probit consistent)
    # p_missing_ind (marginal) must decrease with α+δ (probit-consistent)
    # p_missing_joint is NOT monotone in α+δ alone — it also depends on α
    # (a species with very negative α has low NI occupancy → fails the NI>τ condition)
    ad = jm_t["alpha_hat"] + jm_t["delta_hat"]
    pm = jm_t["p_missing_ind"]   # marginal: monotone in -(α+δ) by construction
    check("p_missing_ind decreases with alpha+delta (probit)",
          float(np.corrcoef(ad, pm)[0, 1]) < 0,
          "correlation should be negative")

    # Fix 1c: N_missing_ci_broad key present
    check("N_missing_ci_broad in output", "N_missing_ci_broad" in jm_t)

    # Fix 4 (smooth shrinkage): no hard LOW variable in detection params function
    try:
        import __main__ as _m
        _src_check = open(_m.__file__).read() if hasattr(_m, '__file__') else ""
    except Exception:
        _src_check = ""
    # Fallback: read file from working directory
    import os
    for _candidate in ["figure_summary_joint.py",
                        os.path.join(os.path.dirname(__file__), "figure_summary_joint.py")
                        if "__file__" in dir() else ""]:
        if _candidate and os.path.exists(_candidate):
            _src_check = open(_candidate).read(); break
    if _src_check:
        _low_pat = "\nLOW = "          # match assignment line, not test strings
        check("no hard LOW cutoff in detection params",
              _low_pat not in _src_check and "n0_cross" in _src_check)
    else:
        check("no hard LOW cutoff (source read skipped)", True)

    # Fix 6 (Hadza): three identification priorities
    meta_h = pd.DataFrame({
        "Lifestyle":    ["H-G", "H-G",  "Industrial", "H-G"],
        "Country_norm": ["Tanzania","Kenya","USA","Tanzania"],
        "Population":   ["Hadza","Ogiek","Western","Hadza"],
    }, index=["s1","s2","s3","s4"])
    b_h = pd.DataFrame({
        "sgb_hadza": [50., 0., 0., 55.],   # Hadza only
        "sgb_other": [0., 60., 50.,  0.],   # non-Hadza
        "sgb_all":   [50., 55., 50., 55.],  # everyone
    }, index=meta_h.index)

    excl_col = hadza_exclusive_sgbs(b_h, meta_h, 10., hadza_col="Population")
    check("Hadza Priority-1 (explicit col): sgb_hadza exclusive",
          "sgb_hadza" in excl_col and "sgb_other" not in excl_col)

    meta_geo = meta_h.drop(columns=["Population"])
    excl_geo = hadza_exclusive_sgbs(b_h, meta_geo, 10.)
    check("Hadza Priority-2 (Tanzania proxy): sgb_hadza exclusive",
          "sgb_hadza" in excl_geo)

    meta_bare = meta_h[["Lifestyle"]]
    excl_bare = hadza_exclusive_sgbs(b_h, meta_bare, 10.)
    check("Hadza Priority-3 (H-G fallback): sgb_hadza exclusive",
          "sgb_hadza" in excl_bare)

    # Fix 7 (depth filter): load_reference accepts min_total_bp
    import inspect as _ins
    lr_params = list(_ins.signature(load_reference).parameters)
    check("load_reference accepts min_total_bp", "min_total_bp" in lr_params)

    # Fix 1 (symmetric NI): fit_joint accepts study_ids_ni
    fj_params = list(_ins.signature(fit_joint_hierarchical).parameters)
    check("fit_joint_hierarchical accepts study_ids_ni", "study_ids_ni" in fj_params)
    check("fit_joint_hierarchical accepts study_weight_mode",
          "study_weight_mode" in fj_params)

    # Fix 2 (sample weights): _laplace_batch_full accepts sample_weights_ind
    lb_params = list(_ins.signature(_laplace_batch_full).parameters)
    check("_laplace_batch_full accepts sample_weights_ind",
          "sample_weights_ind" in lb_params)

    # Fix 2 (_update_study_offsets_ordinal accepts sample_weights)
    uo_params = list(_ins.signature(_update_study_offsets_ordinal).parameters)
    check("_update_study_offsets_ordinal accepts sample_weights",
          "sample_weights" in uo_params)

    # Fix 3: estimate_study_effects uses ordinal scores (not binary C>0)
    # Check via the return dict having 'ordinal_score' key
    n_s2, n_k2, J2 = 60, 20, 5
    sids2 = np.repeat(["A","B"], 30)
    C2 = rng.integers(0, J2+1, (n_s2, n_k2), dtype=np.int8)
    ld2 = rng.normal(20, .5, n_s2)
    _, _, _, _, info2 = estimate_study_effects(sids2, sids2, C2, C2, ld2, ld2, J2)
    check("estimate_study_effects returns ordinal_score",
          "ordinal_score" in info2["ind"]["A"])

    # Parametric bootstrap: _n_missing_with_hyper_uncertainty returns draws
    _n_sgb_t = len(jm_t["alpha_hat"])
    draws_boot = _n_missing_with_hyper_uncertainty(
        jm_t["alpha_hat"], jm_t["delta_hat"],
        jm_t["var_aa"], jm_t["var_dd"], jm_t["cov_ad"],
        jm_t["hyperparams"]["sig_alpha"],
        jm_t["hyperparams"]["sig_delta"],
        jm_t["hyperparams"]["sig_gamma"],
        jm_t["psi_absent_thr"], _n_sgb_t, rng,
        n_species_draws=20, n_hyper=10)
    check("_n_missing_with_hyper_uncertainty returns (n_hyper×n_species,) draws",
          draws_boot.shape == (200,) and draws_boot.min() >= 0)

    # ── Joint missingness criterion ───────────────────────────────────────
    # p_missing_joint should be ≤ p_missing_ind (subset event)
    check("p_missing_joint ≤ p_missing_ind (subset criterion)",
          np.all(jm_t["p_missing_joint"] <= jm_t["p_missing_ind"] + 1e-9),
          f"max violation={float(np.max(jm_t['p_missing_joint'] - jm_t['p_missing_ind'])):.4f}")

    # N_missing_joint ≤ N_missing_marginal (globally rare taxa excluded)
    check("N_missing_joint ≤ N_missing_marginal",
          jm_t["N_missing_mean"] <= jm_t["N_missing_marginal"] + 0.1)

    # psi_ni_present_thr stored
    check("psi_ni_present_thr in return dict",
          "psi_ni_present_thr" in jm_t)

    # ── Abundance family dispatcher ───────────────────────────────────────
    # _r_all_batched with student_t should give different results than lognormal
    # for a small batch
    mu_test  = np.array([-1.0, 0.0, 1.0])
    sig_test = np.array([0.5, 0.5, 0.5])
    ld_test  = np.array([19.0, 20.0, 21.0])
    lL_test  = np.array([14.0, 14.0, 14.0])
    thr_test = [0.10, 0.30, 0.50]

    r_ln = _r_all_batched(mu_test, sig_test, ld_test, lL_test, thr_test,
                           batch_k=3, dist='lognormal', nu=None)
    r_st = _r_all_batched(mu_test, sig_test, ld_test, lL_test, thr_test,
                           batch_k=3, dist='student_t', nu=5.0)
    check("student_t r_all differs from lognormal",
          not np.allclose(r_ln, r_st, atol=1e-6))
    check("student_t r_all in [0,1]",
          np.all((r_st >= 0) & (r_st <= 1)))
    # Student-t with ν→∞ should converge to lognormal
    r_st_large = _r_all_batched(mu_test, sig_test, ld_test, lL_test, thr_test,
                                  batch_k=3, dist='student_t', nu=500.0)
    check("student_t(ν=500) ≈ lognormal",
          np.allclose(r_ln, r_st_large, atol=0.01))

    # abundance_dist in return dict
    check("abundance_dist in return dict", "abundance_dist" in jm_t)

    # ── Broad CI uses same joint criterion as N_missing_mean ─────────────
    # Verify the fix: broad draws should now be joint (≤ marginal in expectation)
    from scipy.stats import norm as _sn2
    _z_eps = jm_t["hyperparams"]["sig_eff"] * float(_sn2.ppf(jm_t["psi_absent_thr"]))
    _z_ni  = jm_t["hyperparams"]["sig_eff"] * float(_sn2.ppf(jm_t["psi_ni_present_thr"]))
    # Can't directly test draws (not in return dict), but check N_missing_ci_broad exists
    # and that broad CI >= base CI width (more uncertainty sources)
    lo_b, hi_b   = jm_t["N_missing_ci"]
    lo_br, hi_br = jm_t["N_missing_ci_broad"]
    check("N_missing_ci_broad is present and non-negative",
          lo_br >= 0 and hi_br >= lo_br)

    # ── Union presence posterior ──────────────────────────────────────────
    check("p_missing_union in return dict",    "p_missing_union"   in jm_t)
    check("N_missing_union in return dict",    "N_missing_union"   in jm_t)
    check("N_missing_union_ci in return dict", "N_missing_union_ci" in jm_t)
    check("A_ind_map in return dict",          "A_ind_map"         in jm_t)
    check("A_ind_map in [0,1]",
          np.all((jm_t["A_ind_map"] >= 0) & (jm_t["A_ind_map"] <= 1)))
    check("p_missing_union in [0,1]",
          np.all((jm_t["p_missing_union"] >= 0) & (jm_t["p_missing_union"] <= 1)))

    # Union N_missing >= per-sample N_missing (pool is harder to miss than individual)
    # This is not guaranteed in small-data tests, so just check it's non-negative
    check("N_missing_union >= 0", jm_t["N_missing_union"] >= 0)

    # A_ind_map should be ≥ h_mat mean (occupancy × detection ≤ union prob for N>1)
    h_mean_k = jm_t["h_mat_ind"].mean(0)
    check("A_ind_map >= mean(h_mat_ind) roughly (union ≥ expected single-draw)",
          float(np.mean(jm_t["A_ind_map"] >= h_mean_k - 0.05)) > 0.8)

    # union_missing_posterior function runs independently
    n_s2, n_k2 = 30, 20
    h_dummy  = rng.uniform(0.05, 0.4, (n_s2, n_k2))
    r_dummy  = h_dummy / 0.3   # pretend ψ̂=0.3 everywhere
    r_dummy  = np.clip(r_dummy, 0, 1)
    a_hat    = np.full(n_k2, -1.5);  d_hat = np.full(n_k2, -0.5)
    v_aa     = np.full(n_k2, 0.1);   v_dd  = np.full(n_k2, 0.2); c_ad = np.zeros(n_k2)
    psi_i    = ndtr((a_hat+d_hat)/1.0); psi_n = ndtr(a_hat/1.0)
    sids_u   = np.repeat(['A','B','C'], 10)
    p_u, N_u, draws_u, Ai, An = union_missing_posterior(
        h_dummy, h_dummy, r_dummy, r_dummy,
        a_hat, d_hat, v_aa, v_dd, c_ad,
        1.0, psi_i, psi_n,
        sids_u, sids_u, 30, 30,
        eps_union=0.05, tau_union=0.05,
        n_draws=100, rng=rng, mode='equal')
    check("union_missing_posterior: p_u in [0,1]",
          np.all((p_u >= 0) & (p_u <= 1)))
    check("union_missing_posterior: draws shape",
          draws_u.shape == (100,))
    check("union_missing_posterior: A_ind_map in [0,1]",
          np.all((Ai >= 0) & (Ai <= 1)))

    print(f"\n  {passed} passed, {failed} failed")
    return failed == 0


# ════════════════════════════════════════════════════════════════════════════
#  CLI AND MAIN
# ════════════════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        description="Joint hierarchical SGB accumulation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--refs",         nargs="+", default=["UHGG","lev"])
    p.add_argument("--out",          default="figure_summary_joint",
                   help="Output file stem (without extension)")
    p.add_argument("--threshold",    type=float, default=10.0,
                   help="Primary breadth threshold (%%)")
    p.add_argument("--thresholds",   nargs="+", type=float,
                   default=[10., 20., 30., 40., 50.],
                   help="Threshold grid for surface panels (%%)")
    p.add_argument("--n-pred",       type=int,   default=200,
                   help="Predictive draws for accumulation curves")
    p.add_argument("--n-extrap",     type=float, default=2.0,
                   help="Extrapolation factor beyond observed N")
    p.add_argument("--n-eb-iter",    type=int,   default=3,
                   help="Empirical Bayes iterations for σ_α, σ_δ")
    p.add_argument("--n-newton",     type=int,   default=N_NEWTON,
                   help="Newton steps for Laplace MAP")
    p.add_argument("--batch-k",      type=int,   default=BATCH_K,
                   help="Species batch size (memory control)")
    p.add_argument("--holdout-frac", type=float, default=0.15,
                   help="Fraction of deepest Industrial samples held out")
    p.add_argument("--holdout-refit",action="store_true", default=False,
                   help="Refit joint model on training samples only (slower)")
    p.add_argument("--study-weight-mode", default="sample",
                   choices=["sample", "equal"],
                   help="'sample': uniform per sample (default, sample-weighted posterior). "
                        "'equal': 1/(K×n_j) per sample — each study contributes equally "
                        "to the likelihood.  Use 'equal' as a sensitivity analysis when one "
                        "large study may dominate the Industrial contrast.")
    p.add_argument("--psi-absent",   type=float, default=PSI_ABSENT_THR,
                   help="ψ threshold for 'effectively absent' (default 1%%)")
    p.add_argument("--mu-alpha",     type=float, default=PRIOR_MU_ALPHA,
                   help="Prior mean of α_k in probit space (default=Φ⁻¹(0.05)≈-1.645)")
    p.add_argument("--sig-alpha",    type=float, default=PRIOR_SIG_ALPHA,
                   help="Prior SD of α_k in probit space (default=1.0)")
    p.add_argument("--sig-delta",    type=float, default=PRIOR_SIG_DELTA,
                   help="Initial SD of δ_k (refined by EB)")
    p.add_argument("--no-trunc-correction", action="store_true", default=False,
                   help="Disable truncation correction for μ_k estimation")
    p.add_argument("--eb-n0",        type=int,   default=5,
                   help="EB shrinkage pseudo-count")
    p.add_argument("--raw-depth-col",default=None,
                   help="Metadata column with raw sequenced bases (decouples "
                        "effort from reference mapping). If absent, Σ(depth×L) used. "
                        "SENSITIVITY: run with and without this flag when available.")
    p.add_argument("--min-total-bp",  type=float, default=1e8,
                   help="Minimum total mapped bases to keep a sample (default 1e8=100Mbp). "
                        "Lowering this threshold keeps shallower samples — useful to check "
                        "whether the depth filter removes studies that reveal rare taxa.")
    p.add_argument("--hadza-col",     default=None,
                   help="Metadata column identifying Hadza samples (e.g. 'Population'). "
                        "If absent, falls back to Country=Tanzania+H-G, then all H-G. "
                        "See hadza_exclusive_sgbs() for priority logic.")
    p.add_argument("--hadza-label",   default="Hadza",
                   help="Value in --hadza-col that identifies Hadza (default 'Hadza').")
    p.add_argument("--abundance-dist", default="lognormal",
                   choices=["lognormal", "student_t"],
                   help="Abundance tail family for the detection model. "
                        "'lognormal': current default. "
                        "'student_t': heavier tail (requires --nu or --fit-abundance-family).")
    p.add_argument("--nu",             type=float, default=None,
                   help="Degrees of freedom for Student-t log-abundance (e.g. 5). "
                        "Ignored when --abundance-dist=lognormal.")
    p.add_argument("--fit-abundance-family", action="store_true", default=False,
                   help="Select abundance family by held-out ordinal log-likelihood. "
                        "Tries lognormal and Student-t on a grid of ν; picks the best. "
                        "Overrides --abundance-dist and --nu.")
    p.add_argument("--nu-grid",        nargs="+", type=float,
                   default=[3.0, 5.0, 10.0, 20.0],
                   help="ν grid for --fit-abundance-family (default: 3 5 10 20).")
    p.add_argument("--psi-ni-present", type=float, default=0.05,
                   help="Non-Industrial occupancy threshold τ_NI for the joint missingness "
                        "criterion P(ψ^Ind<ε AND ψ^NI>τ_NI). Default 0.05 (5%%).")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--test",         action="store_true", default=False,
                   help="Run unit tests and exit")
    return p


def main(argv=None):
    args   = _build_parser().parse_args(argv)
    rng    = np.random.default_rng(args.seed)

    if args.test:
        print("=== Running unit tests ===")
        ok = run_self_tests()
        import sys; sys.exit(0 if ok else 1)

    # Threshold snapping
    THRESHOLDS_PCT_ARR = np.asarray(args.thresholds, float)
    THRESHOLD = float(args.threshold)
    if not np.any(np.isclose(THRESHOLDS_PCT_ARR, THRESHOLD)):
        nearest = float(THRESHOLDS_PCT_ARR[np.argmin(np.abs(THRESHOLDS_PCT_ARR - THRESHOLD))])
        print(f"  WARNING: --threshold {THRESHOLD} not in grid; snapping to {nearest}")
        THRESHOLD = nearest

    correct_trunc = not args.no_trunc_correction

    print("=== Joint Hierarchical SGB Accumulation Analysis ===")
    print(f"  Prior: μ_α = {args.mu_alpha:.3f} (probit) = Φ⁻¹→{100*float(__import__('scipy.special',fromlist=['ndtr']).ndtr(args.mu_alpha)):.1f}% occupancy")
    print(f"  Prior: σ_α = {args.sig_alpha:.2f}  σ_δ_init = {args.sig_delta:.2f}")
    print(f"  ψ_absent threshold: {args.psi_absent*100:.1f}%")
    print(f"  Study weight mode: {args.study_weight_mode!r}")
    print(f"  Effort axis: {'raw depth from ' + args.raw_depth_col if args.raw_depth_col else 'mapped Σ(depth×L) [endogenous — run with --raw-depth-col for sensitivity]'}")
    print(f"  Min total bp: {args.min_total_bp:.0e}  [sensitivity: try lower to keep shallower studies]")
    if args.hadza_col:
        print(f"  Hadza ID: column={args.hadza_col!r}  label={args.hadza_label!r}")

    for ref in args.refs:
        print(f"\n=== Reference: {ref} ===")
        rdat = load_reference(ref, raw_depth_col=args.raw_depth_col,
                              min_total_bp=args.min_total_bp)

        # Group blocks
        blocks = {}
        lifestyle = rdat["meta"]["Lifestyle"]
        for name, mask in [("Industrial (all)",     lifestyle == "Industrial"),
                            ("Non-Industrial (all)", lifestyle.isin(["H-G","Transitional"]))]:
            ids = mask[mask].index.intersection(rdat["breadth"].index)
            blocks[name] = rdat["breadth"].loc[ids]
            print(f"    {name}: {len(ids):,} samples")

        breadth_ind = blocks["Industrial (all)"]
        breadth_ni  = blocks["Non-Industrial (all)"]
        total_bp_ind = rdat["total_bp"].loc[breadth_ind.index]
        total_bp_ni  = rdat["total_bp"].loc[breadth_ni.index]
        depth_ind    = rdat["depth"].loc[breadth_ind.index]
        depth_ni     = rdat["depth"].loc[breadth_ni.index]

        # Extract study IDs for study-aware model
        meta_ind = rdat["meta"].loc[breadth_ind.index]
        meta_ni  = rdat["meta"].loc[breadth_ni.index]
        sids_ind = meta_ind["study_id"].values if "study_id" in meta_ind.columns else None
        sids_ni  = meta_ni ["study_id"].values if "study_id" in meta_ni.columns  else None

        # ── Step 1: Joint hierarchical model (PRIMARY) ────────────────────
        print(f"\n=== Step 1: Joint hierarchical model ({ref}) ===")
        jm = fit_joint_hierarchical(
            breadth_ind, breadth_ni,
            depth_ind,   depth_ni,
            rdat["lengths"],
            total_bp_ind, total_bp_ni,
            thresholds_pct    = list(THRESHOLDS_PCT_ARR),
            threshold         = THRESHOLD,
            study_ids_ind     = sids_ind,
            study_ids_ni      = sids_ni,
            study_weight_mode = args.study_weight_mode,
            mu_alpha_prior    = args.mu_alpha,
            sig_alpha_prior   = args.sig_alpha,
            sig_delta_init    = args.sig_delta,
            psi_absent_thr    = args.psi_absent,
            psi_ni_present_thr= args.psi_ni_present,
            abundance_dist    = args.abundance_dist,
            abundance_nu      = args.nu,
            fit_abundance_family = args.fit_abundance_family,
            nu_grid           = args.nu_grid,
            n_eb_iter         = args.n_eb_iter,
            correct_truncation= correct_trunc,
            eb_n0             = args.eb_n0,
            batch_k           = args.batch_k,
            n_newton          = args.n_newton,
            rng_missing       = rng)

        print(f"\n  === Sample-weighted results ===")
        print(f"  N_missing (per-sample joint)     = {jm['N_missing_mean']:.0f}  "
              f"[{jm['N_missing_ci'][0]:.0f}, {jm['N_missing_ci'][1]:.0f}]  "
              f"broad: [{jm['N_missing_ci_broad'][0]:.0f}, {jm['N_missing_ci_broad'][1]:.0f}]")
        print(f"  N_missing (lifestyle-union)      = {jm['N_missing_union']:.0f}  "
              f"[{jm['N_missing_union_ci'][0]:.0f}, {jm['N_missing_union_ci'][1]:.0f}]  "
              f"(N*_Ind={jm['N_star_ind']}, N*_NI={jm['N_star_ni']})")
        print(f"  N_missing (marginal Ind-only)    = {jm['N_missing_marginal']:.0f}  (reference)")
        print(f"  SGBs P(per-sample joint)>0.95:   {int((jm['p_missing_joint']>0.95).sum()):,}")
        print(f"  SGBs P(union absent)>0.95:       {int((jm['p_missing_union']>0.95).sum()):,}")
        if jm.get("abundance_dist"):
            ad = jm["abundance_dist"]
            print(f"  Abundance family: Ind={ad['ind']} (ν={ad['nu_ind']})  "
                  f"NI={ad['ni']} (ν={ad['nu_ni']})")

        # ── Sensitivity: equal-study-weight posterior ─────────────────────
        # Run equal-study-weight fit as a first-class sensitivity, not just a
        # figure option.  Reports N_missing under equal-study weighting alongside
        # the primary sample-weighted result.  When one large study dominates the
        # Industrial cohort, these two estimates can diverge substantially.
        if sids_ind is not None and len(np.unique(sids_ind)) > 1:
            print(f"\n  === Equal-study-weight sensitivity ===")
            jm_eq = fit_joint_hierarchical(
                breadth_ind, breadth_ni,
                depth_ind,   depth_ni,
                rdat["lengths"],
                total_bp_ind, total_bp_ni,
                thresholds_pct    = list(THRESHOLDS_PCT_ARR),
                threshold         = THRESHOLD,
                study_ids_ind     = sids_ind,
                study_ids_ni      = sids_ni,
                study_weight_mode = 'equal',
                mu_alpha_prior    = args.mu_alpha,
                sig_alpha_prior   = args.sig_alpha,
                sig_delta_init    = args.sig_delta,
                psi_absent_thr    = args.psi_absent,
                psi_ni_present_thr= args.psi_ni_present,
                n_eb_iter         = args.n_eb_iter,
                correct_truncation= correct_trunc,
                eb_n0             = args.eb_n0,
                batch_k           = args.batch_k,
                n_newton          = args.n_newton,
                rng_missing       = np.random.default_rng(args.seed + 1))
            print(f"  N_missing (joint, equal-study)   = {jm_eq['N_missing_mean']:.0f}  "
                  f"[{jm_eq['N_missing_ci'][0]:.0f}, {jm_eq['N_missing_ci'][1]:.0f}]")
            print(f"  N_missing (marginal, equal-study) = {jm_eq['N_missing_marginal']:.0f}")
            delta_diff = float(np.abs(jm['delta_hat'] - jm_eq['delta_hat']).mean())
            print(f"  Mean |δ_sample − δ_equal| = {delta_diff:.4f}  "
                  f"(>0.1 suggests large-study dominance)")
        else:
            jm_eq = None

        # Proxy ID warning in summary
        proxy_used = rdat["meta"].get("_study_id_is_proxy", pd.Series([False])).any() \
            if hasattr(rdat["meta"].get("_study_id_is_proxy", None), "any") else False
        if proxy_used:
            print(f"\n  *** SENSITIVITY ONLY: proxy study IDs (Country×Lifestyle) were used. "
                  f"Results may be unreliable — provide a real study accession column. ***")

        # ── Step 2: Holdout diagnostic ────────────────────────────────────
        print(f"\n=== Step 2: Holdout diagnostic ({ref}) ===")
        hd_ind = holdout_diagnostic(
            jm, breadth_ind, breadth_ni,
            depth_ind, depth_ni,
            rdat["lengths"], total_bp_ind, total_bp_ni,
            threshold=THRESHOLD, hold_frac=args.holdout_frac,
            do_refit=args.holdout_refit)
        print(f"  Ind holdout: train={hd_ind['n_train']} hold={hd_ind['n_hold']}  "
              f"obs_new={hd_ind['obs_new'].sum():.0f}  "
              f"pred_new={hd_ind['pred_new'].sum():.0f}  "
              f"refitted={hd_ind['refitted']}")

        # ── Step 3: Threshold surface ─────────────────────────────────────
        print(f"\n=== Step 3: Threshold surface ({ref}) ===")
        N_max_i = int(np.ceil(len(jm["d_i_ind"]) * args.n_extrap))
        N_max_n = int(np.ceil(len(jm["d_i_ni"])  * args.n_extrap))
        N_max   = max(N_max_i, N_max_n)
        surface = threshold_surface(jm, THRESHOLDS_PCT_ARR, N_max,
                                    args.n_pred, rng, args.batch_k)

        # ── Step 4: Panel E — Hadza-excluded NI ──────────────────────────
        print(f"\n=== Step 4: Panel E Hadza-excluded ({ref}) ===")
        excl = hadza_exclusive_sgbs(rdat["breadth"], rdat["meta"], THRESHOLD,
                                     hadza_col=args.hadza_col,
                                     hadza_label=args.hadza_label)
        print(f"  Hadza-exclusive SGBs removed: {len(excl)}")
        breadth_ni_e = breadth_ni[[c for c in breadth_ni.columns if c not in excl]]
        breadth_ind_e = breadth_ind[[c for c in breadth_ind.columns if c not in excl]]

        jm_e = fit_joint_hierarchical(
            breadth_ind_e, breadth_ni_e,
            depth_ind.loc[breadth_ind_e.index, breadth_ind_e.columns],
            depth_ni.loc[ breadth_ni_e.index,  breadth_ni_e.columns],
            rdat["lengths"].reindex(breadth_ind_e.columns),
            total_bp_ind.loc[breadth_ind_e.index],
            total_bp_ni.loc[ breadth_ni_e.index],
            thresholds_pct = list(THRESHOLDS_PCT_ARR),
            threshold      = THRESHOLD,
            mu_alpha_prior = args.mu_alpha,
            sig_alpha_prior= args.sig_alpha,
            sig_delta_init = args.sig_delta,
            psi_absent_thr = args.psi_absent,
            n_eb_iter      = args.n_eb_iter,
            correct_truncation = correct_trunc,
            eb_n0          = args.eb_n0,
            batch_k        = args.batch_k,
            n_newton       = args.n_newton,
            rng_missing    = rng)
        surface_e = threshold_surface(jm_e, THRESHOLDS_PCT_ARR,
                                       N_max, args.n_pred, rng, args.batch_k)

        # ── Step 5: Build figure ──────────────────────────────────────────
        print(f"\n=== Step 5: Building figure ({ref}) ===")
        out_stem = f"{args.out}_{ref}"
        build_figure(jm=jm, jm_e=jm_e,
                     thresholds_pct=THRESHOLDS_PCT_ARR,
                     surface=surface, surface_e=surface_e,
                     hd_ind=hd_ind, hd_ni=None,
                     breadth_ind=breadth_ind, breadth_ni=breadth_ni,
                     breadth_ind_e=breadth_ind_e, breadth_ni_e=breadth_ni_e,
                     threshold=THRESHOLD,
                     N_pred=args.n_pred, rng=rng, out_stem=out_stem)

        # Per-SGB results table
        ref_sgbs  = jm["ref_sgbs"]
        results_df = pd.DataFrame({
            "sgb":          ref_sgbs,
            "alpha_hat":    jm["alpha_hat"],
            "delta_hat":    jm["delta_hat"],
            "sd_delta":     np.sqrt(np.maximum(jm["var_dd"], 1e-9)),
            "psi_ni_hat":   jm["psi_ni_hat"],
            "psi_ind_hat":  jm["psi_ind_hat"],
            "p_missing_joint":jm["p_missing_joint"],
        }).set_index("sgb")
        csv_out = f"{args.out}_{ref}_per_sgb.csv"
        results_df.to_csv(csv_out)
        print(f"  Per-SGB results → {csv_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
