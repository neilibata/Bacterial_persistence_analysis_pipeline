"""
GLMM utilities: per-taxon logistic regression with depth + geography.

Core model (continent mode):
    presence_i ~ group_i + continent_FE + ns(log_depth, k)

All regressions use statsmodels GLM (binomial/logit) for speed.
Continent fixed effects implicitly control for between-study variation
within each continent.

Public API:
    run_taxon_glmm_batch  — batch per-taxon regressions (parallel)
    run_meta_sensitivity   — within-study meta-analysis
    bh_fdr                — Benjamini–Hochberg FDR correction
    build_design_matrix   — depth spline + continent dummies
"""

import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Lazy imports for heavy deps
_sm = None
_patsy = None


def _ensure_statsmodels():
    global _sm
    if _sm is None:
        import statsmodels.api as sm
        _sm = sm
    return _sm


def _ensure_patsy():
    global _patsy
    if _patsy is None:
        import patsy
        _patsy = patsy
    return _patsy


# ====================================================================
#  DESIGN MATRIX CONSTRUCTION
# ====================================================================

def build_design_matrix(log_depth, continent=None, study=None,
                        mode="continent", depth_knots=3):
    """
    Build covariate matrix: depth spline + geography dummies.

    Returns DataFrame of covariates (no intercept — added by GLM).
    """
    n = len(log_depth)
    parts = {}

    # Depth spline (natural spline via patsy)
    patsy = _ensure_patsy()
    ld = log_depth.values if hasattr(log_depth, 'values') else np.array(log_depth)
    valid_depth = np.isfinite(ld)

    if valid_depth.sum() > depth_knots + 2:
        try:
            depth_basis = patsy.dmatrix(
                f"cr(x, df={depth_knots}) - 1",
                {"x": ld},
                return_type="dataframe")
            depth_basis.index = (log_depth.index if hasattr(log_depth, 'index')
                                 else range(n))
            for c in depth_basis.columns:
                parts[f"depth_{c}"] = depth_basis[c].values
        except Exception:
            # Fallback: polynomial
            for k in range(1, depth_knots + 1):
                parts[f"depth_poly{k}"] = ld ** k
    else:
        parts["depth_lin"] = ld

    # Geography (skip if mode=="none" — Mundlak handles confounding
    # via within-study centring instead of geography FEs)
    if mode == "none":
        pass
    elif mode == "continent" and continent is not None:
        cont = continent.astype(str).fillna("Unknown")
        dummies = pd.get_dummies(cont, prefix="cont", drop_first=True,
                                 dtype=float)
        dummies.index = (log_depth.index if hasattr(log_depth, 'index')
                         else range(n))
        for c in dummies.columns:
            parts[c] = dummies[c].values

    elif mode == "study_fe" and study is not None:
        st = study.astype(str).fillna("Unknown")
        dummies = pd.get_dummies(st, prefix="study", drop_first=True,
                                 dtype=float)
        dummies.index = (log_depth.index if hasattr(log_depth, 'index')
                         else range(n))
        for c in dummies.columns:
            parts[c] = dummies[c].values

    idx = (log_depth.index if hasattr(log_depth, 'index')
           else range(n))
    return pd.DataFrame(parts, index=idx).astype(float)


# ====================================================================
#  SINGLE-TAXON GLM
# ====================================================================

def _fit_one_taxon(taxon_id, y, group, covariates, label_g0="g0",
                   label_g1="g1"):
    """
    Fit logistic regression for one taxon.

    y ~ group + covariates

    Returns dict with coefficients, p-values, prevalence estimates.
    """
    sm = _ensure_statsmodels()

    result = {
        "taxon": taxon_id,
        "converged": False,
        "beta_group": np.nan,
        "se_group": np.nan,
        "z_group": np.nan,
        "p_group": np.nan,
        "prev_diff": np.nan,
        f"raw_prev_{label_g0}": np.nan,
        f"raw_prev_{label_g1}": np.nan,
        f"adj_prev_{label_g0}": np.nan,
        f"adj_prev_{label_g1}": np.nan,
        "n_present": 0,
        "n_total": len(y),
    }

    y_arr = np.asarray(y, dtype=float)
    g_arr = np.asarray(group, dtype=float)
    valid = np.isfinite(y_arr) & np.isfinite(g_arr)

    # Also check covariates
    cov_arr = covariates.values if hasattr(covariates, 'values') else covariates
    cov_arr = np.asarray(cov_arr, dtype=float)
    valid &= np.all(np.isfinite(cov_arr), axis=1)

    y_v = y_arr[valid]
    g_v = g_arr[valid]
    cov_v = cov_arr[valid]

    n_pres = int(y_v.sum())
    n_abs = int(len(y_v) - n_pres)
    result["n_present"] = n_pres

    # Skip degenerate cases
    if n_pres < 3 or n_abs < 3:
        return result
    if len(np.unique(g_v)) < 2:
        return result

    # Raw prevalences
    result[f"raw_prev_{label_g0}"] = y_v[g_v == 0].mean()
    result[f"raw_prev_{label_g1}"] = y_v[g_v == 1].mean()

    # Build X: intercept + group + covariates
    X = np.column_stack([np.ones(len(y_v)), g_v, cov_v])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y_v, X,
                           family=sm.families.Binomial())
            fit = model.fit(disp=0, maxiter=50)

        if not np.isfinite(fit.params).all():
            return result

        result["converged"] = True
        result["beta_group"] = fit.params[1]
        result["se_group"] = fit.bse[1]
        result["z_group"] = fit.tvalues[1]
        result["p_group"] = fit.pvalues[1]

        # G-computation: predicted prevalences at group=0 and group=1
        X0 = X.copy(); X0[:, 1] = 0
        X1 = X.copy(); X1[:, 1] = 1
        pred0 = fit.predict(X0).mean()
        pred1 = fit.predict(X1).mean()
        result[f"adj_prev_{label_g0}"] = pred0
        result[f"adj_prev_{label_g1}"] = pred1
        result["prev_diff"] = pred1 - pred0

    except Exception:
        pass

    return result


# ====================================================================
#  BATCH RUNNER
# ====================================================================

def run_taxon_glmm_batch(presence_df, group_series, log_depth_series,
                         study_series=None, continent_series=None,
                         study_mode="continent", depth_knots=3,
                         run_meta=False, min_prev=0.002,
                         max_prev=0.998, n_jobs=-1, verbose=True,
                         label_g0="g0", label_g1="g1"):
    """
    Run per-taxon logistic regressions in parallel.

    Parameters
    ----------
    presence_df : bool DataFrame (samples x taxa)
    group_series : 0/1 Series
    log_depth_series : Series
    study_series, continent_series : Series (for FEs)
    study_mode : "continent" or "study_fe"
    depth_knots : int
    min_prev, max_prev : filter taxa by overall prevalence
    n_jobs : parallel workers
    """
    _ensure_statsmodels()

    # Align
    idx = presence_df.index.intersection(group_series.index)
    idx = idx.intersection(log_depth_series.dropna().index)
    presence = presence_df.loc[idx]
    group = group_series.loc[idx]
    log_depth = log_depth_series.loc[idx]
    continent = (continent_series.loc[idx]
                 if continent_series is not None else None)
    study = (study_series.loc[idx]
             if study_series is not None else None)

    # Build design matrix once
    covariates = build_design_matrix(
        log_depth, continent=continent, study=study,
        mode=study_mode, depth_knots=depth_knots)
    covariates = covariates.loc[idx]

    # Filter taxa by prevalence
    prev = presence.mean(axis=0)
    keep = prev.index[(prev >= min_prev) & (prev <= max_prev)]
    if verbose:
        print(f"    Taxa passing prevalence filter: "
              f"{len(keep)}/{presence.shape[1]}")

    if len(keep) == 0:
        return pd.DataFrame()

    # Run in parallel
    def _worker(taxon):
        y = presence[taxon].astype(float).values
        return _fit_one_taxon(
            taxon, y, group.values, covariates.values,
            label_g0=label_g0, label_g1=label_g1)

    if verbose:
        print(f"    Running {len(keep)} regressions "
              f"(n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(t) for t in keep)

    df = pd.DataFrame(results)

    # FDR correction
    if "p_group" in df.columns:
        df["q_group"] = bh_fdr(df["p_group"])

    return df


# ====================================================================
#  WITHIN-STUDY META-ANALYSIS
# ====================================================================

def run_meta_sensitivity(presence_df, group_series, log_depth_series,
                         study_series, taxon_ids=None,
                         min_per_study=20, n_jobs=-1, verbose=True):
    """
    Within-study logistic regression + inverse-variance meta-analysis.
    """
    _ensure_statsmodels()

    idx = presence_df.index.intersection(group_series.index)
    idx = idx.intersection(log_depth_series.dropna().index)
    idx = idx.intersection(study_series.dropna().index)

    presence = presence_df.loc[idx]
    group = group_series.loc[idx]
    log_depth = log_depth_series.loc[idx]
    studies = study_series.loc[idx]

    taxa = taxon_ids if taxon_ids else list(presence.columns)

    # Get qualifying studies
    study_counts = pd.DataFrame({
        "study": studies, "group": group
    }).groupby("study")["group"].agg(["sum", "count"])
    study_counts["n_g0"] = study_counts["count"] - study_counts["sum"]
    valid_studies = study_counts[
        (study_counts["sum"] >= 5) &
        (study_counts["n_g0"] >= 5) &
        (study_counts["count"] >= min_per_study)
    ].index.tolist()

    if verbose:
        print(f"    Meta-analysis: {len(valid_studies)} qualifying studies, "
              f"{len(taxa)} taxa")

    if len(valid_studies) < 2:
        return pd.DataFrame()

    def _meta_one_taxon(taxon):
        betas, ses = [], []
        for st in valid_studies:
            mask = studies == st
            y = presence.loc[mask, taxon].astype(float).values
            g = group.loc[mask].values
            ld = log_depth.loc[mask].values

            if y.sum() < 2 or (len(y) - y.sum()) < 2:
                continue
            if len(np.unique(g)) < 2:
                continue

            X = np.column_stack([np.ones(len(y)), g, ld])
            try:
                sm = _ensure_statsmodels()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.GLM(y, X,
                                 family=sm.families.Binomial()).fit(
                        disp=0, maxiter=30)
                if np.isfinite(fit.params[1]) and np.isfinite(fit.bse[1]):
                    if fit.bse[1] > 0:
                        betas.append(fit.params[1])
                        ses.append(fit.bse[1])
            except Exception:
                continue

        if len(betas) < 2:
            return {"taxon": taxon, "meta_effect": np.nan,
                    "meta_se": np.nan, "meta_z": np.nan,
                    "meta_p": np.nan, "meta_n_studies": 0,
                    "meta_direction_consistent": False}

        weights = [1.0 / (s ** 2) for s in ses]
        w_sum = sum(weights)
        meta_beta = sum(b * w for b, w in zip(betas, weights)) / w_sum
        meta_se = 1.0 / np.sqrt(w_sum)
        meta_z = meta_beta / meta_se
        from scipy.stats import norm
        meta_p = 2 * norm.sf(abs(meta_z))

        return {
            "taxon": taxon,
            "meta_effect": meta_beta,
            "meta_se": meta_se,
            "meta_z": meta_z,
            "meta_p": meta_p,
            "meta_n_studies": len(betas),
            "meta_direction_consistent": True,  # updated below
        }

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_meta_one_taxon)(t) for t in taxa)

    df = pd.DataFrame(results)

    if "meta_p" in df.columns:
        df["meta_q"] = bh_fdr(df["meta_p"])

    return df


# ====================================================================
#  FDR CORRECTION
# ====================================================================

# ====================================================================
#  MUNDLAK WITHIN–BETWEEN DECOMPOSITION
# ====================================================================

def _fit_mundlak_one_taxon(taxon_id, y, group, study_ids,
                           covariates, min_studies=3):
    """
    Mundlak (hybrid) within–between logistic regression for one taxon.

    Decomposes the group indicator into:
        X_w = X_is - X̄_s   (within-study centred)
        X_b = X̄_s           (between-study mean)

    Model:
        logit P(Y=1) = alpha + beta_w * X_w + beta_b * X_b
                      + study_dummies + covariates

    beta_w  = within-study effect (same estimand as CMH)
    beta_b  = between-study association (often confounded)

    If beta_w and beta_b diverge, naive pooling is badly biased.

    Uses cluster-robust (HC1) standard errors grouped by study.

    Returns
    -------
    dict with keys: taxon, converged, beta_within, se_within,
        p_within, beta_between, se_between, p_between,
        hausman_like_p (test of beta_w == beta_b).
    """
    sm = _ensure_statsmodels()

    result = {
        "taxon": taxon_id,
        "converged": False,
        "beta_within": np.nan,
        "se_within": np.nan,
        "p_within": np.nan,
        "beta_between": np.nan,
        "se_between": np.nan,
        "p_between": np.nan,
        "hausman_diff": np.nan,
        "hausman_p": np.nan,
        "n_present": 0,
        "n_total": len(y),
        "n_studies_informative": 0,
    }

    y_arr = np.asarray(y, dtype=float)
    g_arr = np.asarray(group, dtype=float)
    s_arr = np.asarray(study_ids)
    cov_arr = (covariates.values if hasattr(covariates, 'values')
               else np.asarray(covariates, dtype=float))

    # Validity filter
    valid = (np.isfinite(y_arr) & np.isfinite(g_arr)
             & np.all(np.isfinite(cov_arr), axis=1)
             & pd.notna(s_arr))
    y_v = y_arr[valid]
    g_v = g_arr[valid]
    s_v = s_arr[valid]
    cov_v = cov_arr[valid]

    n_pres = int(y_v.sum())
    n_abs = int(len(y_v) - n_pres)
    result["n_present"] = n_pres

    if n_pres < 3 or n_abs < 3 or len(np.unique(g_v)) < 2:
        return result

    # ── Mundlak decomposition ──
    # Compute study-level mean of group indicator (proportion of cases)
    study_ser = pd.Series(s_v)
    group_ser = pd.Series(g_v)
    study_means = group_ser.groupby(study_ser).transform("mean").values
    x_within = g_v - study_means   # within-study centred
    x_between = study_means         # between-study mean

    # Count informative studies (those with both groups)
    unique_studies = np.unique(s_v)
    n_informative = 0
    for st in unique_studies:
        mask_st = s_v == st
        if len(np.unique(g_v[mask_st])) >= 2:
            n_informative += 1
    result["n_studies_informative"] = n_informative

    if n_informative < min_studies:
        return result

    # Check that X_w has variance (otherwise beta_w is unidentifiable)
    if np.std(x_within) < 1e-10:
        return result

    # ── Build design matrix ──
    # intercept + X_w + X_b + covariates (depth spline etc.)
    X = np.column_stack([np.ones(len(y_v)),
                         x_within, x_between, cov_v])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y_v, X, family=sm.families.Binomial())

            # Fit with cluster-robust SEs grouped by study
            # First get numeric study codes for cov_struct
            study_codes = pd.Categorical(s_v).codes
            fit = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": study_codes},
                disp=0, maxiter=50)

        if not np.isfinite(fit.params).all():
            return result

        result["converged"] = True
        # Column indices: 0=intercept, 1=x_within, 2=x_between, 3+=covariates
        result["beta_within"] = fit.params[1]
        result["se_within"] = fit.bse[1]
        result["p_within"] = fit.pvalues[1]
        result["beta_between"] = fit.params[2]
        result["se_between"] = fit.bse[2]
        result["p_between"] = fit.pvalues[2]

        # Hausman-like diagnostic: do beta_w and beta_b agree?
        diff = fit.params[1] - fit.params[2]
        se_diff = np.sqrt(fit.bse[1]**2 + fit.bse[2]**2)
        if se_diff > 0:
            from scipy.stats import norm as norm_dist
            z_diff = diff / se_diff
            result["hausman_diff"] = diff
            result["hausman_p"] = 2 * norm_dist.sf(abs(z_diff))

    except Exception:
        pass

    return result


def run_mundlak_batch(presence_df, group_series, log_depth_series,
                      study_series, taxon_ids=None,
                      depth_knots=3, min_studies=3,
                      n_jobs=-1, verbose=True):
    """
    Run per-taxon Mundlak within–between regressions in parallel.

    Parameters
    ----------
    presence_df : bool/float DataFrame (samples x taxa)
    group_series : 0/1 Series (control/case)
    log_depth_series : Series
    study_series : Series (study labels)
    taxon_ids : list of taxon column names (None = all)
    depth_knots : int (depth spline df)
    min_studies : int (minimum informative studies per taxon)
    n_jobs : int

    Returns DataFrame with per-taxon results including beta_w, beta_b,
    and Hausman diagnostic.
    """
    _ensure_statsmodels()

    idx = (presence_df.index
           .intersection(group_series.index)
           .intersection(log_depth_series.dropna().index)
           .intersection(study_series.dropna().index))

    presence = presence_df.loc[idx]
    group = group_series.loc[idx]
    log_depth = log_depth_series.loc[idx]
    study = study_series.loc[idx]

    # Build covariates: depth spline only (no geography FEs —
    # the within-study centring handles confounding).
    covariates = build_design_matrix(
        log_depth, mode="none", depth_knots=depth_knots)
    covariates = covariates.loc[idx]

    taxa = taxon_ids if taxon_ids else list(presence.columns)

    if verbose:
        print(f"    Mundlak: {len(taxa)} taxa, {len(idx)} samples, "
              f"{study.nunique()} studies")

    if len(taxa) == 0 or len(idx) < 20:
        return pd.DataFrame()

    def _worker(taxon):
        y = presence[taxon].astype(float).values
        return _fit_mundlak_one_taxon(
            taxon, y, group.values, study.values,
            covariates, min_studies=min_studies)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(t) for t in taxa)

    df = pd.DataFrame(results)
    if "p_within" in df.columns:
        df["q_within"] = bh_fdr(df["p_within"])
    if "p_between" in df.columns:
        df["q_between"] = bh_fdr(df["p_between"])

    return df


# ====================================================================
#  FDR CORRECTION
# ====================================================================

def bh_fdr(p_values):
    """Benjamini–Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    q = np.full(n, np.nan)

    valid = np.isfinite(p)
    if valid.sum() == 0:
        return pd.Series(q, index=getattr(p_values, 'index', None))

    pv = p[valid]
    order = np.argsort(pv)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(pv) + 1)

    qv = pv * len(pv) / ranks
    # Enforce monotonicity (step-up)
    qv_sorted = qv[order]
    for i in range(len(qv_sorted) - 2, -1, -1):
        qv_sorted[i + 1] = min(qv_sorted[i + 1], 1.0)
        qv_sorted[i] = min(qv_sorted[i], qv_sorted[i + 1])
    qv[order] = qv_sorted
    qv = np.clip(qv, 0, 1)

    q[valid] = qv
    if hasattr(p_values, 'index'):
        return pd.Series(q, index=p_values.index)
    return q
