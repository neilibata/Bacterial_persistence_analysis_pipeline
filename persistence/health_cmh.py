"""
Health comparison statistical helpers (v4).

Provides:
  - Cochran-Mantel-Haenszel (CMH) test aggregating 2x2 tables across studies
  - Per-taxon Fisher's exact tests (pooled, no stratification)
  - Within-study depth-matched CMH wrappers
  - Wilson confidence intervals
  - Group mask builder for all comparison types

v4 changes vs v3:
  - Added proper CMH implementation (replaces misleadingly-named Fisher-only module)
  - Within-study stratification for per-condition analyses
  - Nearest-neighbour depth matching within each study stratum
  - Symmetric obesity comparison (healthy non-obese vs healthy obese)
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm
from scipy.stats import chi2 as chi2_dist
from scipy.spatial import cKDTree

from . import config


# ====================================================================
#  CONFIDENCE INTERVALS
# ====================================================================

def wilson_ci(count, nobs, alpha=0.05):
    """Wilson score confidence interval for a proportion."""
    if nobs == 0:
        return 0.0, 0.0
    z = norm.ppf(1 - alpha / 2)
    p_hat = count / nobs
    denom = 1 + z**2 / nobs
    centre = (p_hat + z**2 / (2 * nobs)) / denom
    margin = (z * np.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * nobs)) / nobs) / denom)
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ====================================================================
#  COCHRAN-MANTEL-HAENSZEL TEST
# ====================================================================

def cochran_mantel_haenszel(tables):
    """
    CMH test over a list of 2x2 np.arrays.

    Table layout:
        row 0 = group 1 (e.g. healthy/control)
        row 1 = group 2 (e.g. unhealthy/case)
        col 0 = present, col 1 = absent

    Returns
    -------
    (pval, log_common_OR, direction)
        direction: +1 = group1 more prevalent (depleted in g2)
                   -1 = group2 more prevalent (enriched in g2)
                    0 = no data
    """
    valid = []
    for t in tables:
        t = np.asarray(t, dtype=float)
        if t.shape != (2, 2):
            continue
        n = t.sum()
        if n == 0 or any(t.sum(axis=1) == 0) or any(t.sum(axis=0) == 0):
            continue
        valid.append(t)

    if len(valid) == 0:
        return np.nan, 0.0, 0

    numerator = 0.0
    denominator = 0.0
    R_num = 0.0
    R_den = 0.0

    for t in valid:
        a, b, c, d = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
        n = a + b + c + d
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        E_a = r1 * c1 / n
        V_a = (r1 * r2 * c1 * c2 / (n**2 * (n - 1))) if n > 1 else 0.0
        numerator += (a - E_a)
        denominator += V_a
        R_num += a * d / n
        R_den += b * c / n

    if denominator <= 0:
        return np.nan, 0.0, 0

    chi2_stat = (abs(numerator) - 0.5)**2 / denominator
    pval = chi2_dist.sf(chi2_stat, df=1)
    # Complete separation -- a taxon present in every g1 sample and no g2
    # sample, or the reverse -- gives R_den == 0 or R_num == 0. That is a
    # valid table with a tiny p-value and an odds ratio of 0 or infinity, not
    # a missing one. log_or is held finite so the published column stays
    # usable, but scoring these as `direction = 0` meant every caller counted
    # them as neither depleted nor enriched, and they are the most extreme
    # taxa there are: the published fractions were biased low by dropping
    # exactly the strongest signal. The MH numerator sum(a - E_a) is signed
    # and well defined under separation, so it carries the direction instead.
    # An odds ratio of exactly 1 is a different case and still scores 0.
    separated = (R_den <= 0) or (R_num <= 0)
    log_or = 0.0 if separated else float(np.log(R_num / R_den))
    if separated:
        direction = int(np.sign(numerator))
    else:
        direction = 1 if log_or > 0 else (-1 if log_or < 0 else 0)
    return pval, log_or, direction


# ====================================================================
#  PER-TAXON FISHER (POOLED, NO STRATIFICATION)
# ====================================================================

def fisher_enrichment_test(bdf, taxon_ids, ids_g1, ids_g2, thresh,
                           list_name="", condition="", method="Fisher"):
    """
    Per-taxon Fisher's exact test comparing prevalence in g1 vs g2.

    Returns dict with depleted_count, enriched_count, fractions, CIs.
    """
    n1, n2 = len(ids_g1), len(ids_g2)
    mics = [m for m in taxon_ids if m in bdf.columns]
    total = len(mics)

    if total == 0 or n1 == 0 or n2 == 0:
        return None

    dep = enr = 0
    for m in mics:
        a = int((bdf.loc[ids_g1, m] >= thresh).sum())
        c = int((bdf.loc[ids_g2, m] >= thresh).sum())
        _, p = fisher_exact([[a, n1 - a], [c, n2 - c]])
        if p < 0.05:
            if a / n1 > c / n2:
                dep += 1   # more prevalent in g1 (healthy)
            elif c / n2 > a / n1:
                enr += 1   # more prevalent in g2 (unhealthy)

    return _pack_result(dep, enr, total, n1, n2,
                        condition=condition, method=method,
                        list_name=list_name)


# ====================================================================
#  WITHIN-STUDY CMH (UNMATCHED)
# ====================================================================

def cmh_enrichment_test(bdf, taxon_ids, meta, mask_g1, mask_g2, thresh,
                        list_name="", condition="",
                        method="Within-study CMH",
                        min_stratum=None):
    """
    Per-taxon CMH test comparing prevalence across study strata.

    For each taxon, builds a 2x2 table per study and aggregates
    with the CMH statistic.
    """
    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    strata = _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum)
    if not strata:
        return None

    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None

    n1 = sum(len(g1) for _, g1, _ in strata)
    n2 = sum(len(g2) for _, _, g2 in strata)
    dep = enr = 0

    for mic in mics:
        tables = []
        for _, g1_idx, g2_idx in strata:
            a = int((bdf.loc[g1_idx, mic] >= thresh).sum())
            c = int((bdf.loc[g2_idx, mic] >= thresh).sum())
            tables.append(np.array([[a, len(g1_idx) - a],
                                    [c, len(g2_idx) - c]]))
        p, _, d = cochran_mantel_haenszel(tables)
        if not np.isnan(p) and p < 0.05:
            if d > 0:
                dep += 1
            elif d < 0:
                enr += 1

    return _pack_result(dep, enr, len(mics), n1, n2,
                        condition=condition, method=method,
                        list_name=list_name, n_studies=len(strata))


# ====================================================================
#  WITHIN-STUDY DEPTH-MATCHED CMH
# ====================================================================

def depth_matched_cmh_enrichment_test(
        bdf, taxon_ids, meta, mask_g1, mask_g2, thresh,
        list_name="", condition="",
        method="Depth-matched CMH",
        min_stratum=None):
    """
    Per-taxon CMH test with 1:1 nearest-neighbour depth matching
    within each study stratum.
    """
    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    strata = _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum)
    if not strata:
        return None

    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None

    matched_strata = []
    total_g1 = total_g2 = 0
    depth_diffs = []

    for study, g1_idx, g2_idx in strata:
        g1_depths = meta.loc[g1_idx, "_log_depth"].dropna()
        g2_depths = meta.loc[g2_idx, "_log_depth"].dropna()

        if len(g1_depths) < min_stratum or len(g2_depths) < min_stratum:
            continue

        mg1, mg2 = _nearest_neighbour_match(g1_depths, g2_depths)

        if len(mg1) < min_stratum:
            continue

        matched_strata.append((study, mg1, mg2))
        total_g1 += len(mg1)
        total_g2 += len(mg2)

        n_pairs = min(len(mg1), len(mg2))
        if n_pairs > 0:
            d1 = meta.loc[mg1, "_log_depth"].values
            d2 = meta.loc[mg2, "_log_depth"].values
            depth_diffs.extend(
                np.abs(d1[:n_pairs] - d2[:n_pairs]).tolist())

    if not matched_strata:
        return None

    dep = enr = 0
    for mic in mics:
        tables = []
        for _, mg1, mg2 in matched_strata:
            a = int((bdf.loc[mg1, mic] >= thresh).sum())
            c = int((bdf.loc[mg2, mic] >= thresh).sum())
            tables.append(np.array([[a, len(mg1) - a],
                                    [c, len(mg2) - c]]))
        p, _, d = cochran_mantel_haenszel(tables)
        if not np.isnan(p) and p < 0.05:
            if d > 0:
                dep += 1
            elif d < 0:
                enr += 1

    median_dd = np.median(depth_diffs) if depth_diffs else np.nan

    return _pack_result(
        dep, enr, len(mics), total_g1, total_g2,
        condition=condition, method=method, list_name=list_name,
        n_studies=len(matched_strata),
        median_log_depth_diff=round(median_dd, 4))


# ====================================================================
#  WITHIN-STUDY GEOM-MEAN-TARGET WEIGHTED CMH
# ====================================================================

def geom_mean_weighted_cmh_enrichment_test(
        bdf, taxon_ids, meta, mask_g1, mask_g2, thresh,
        list_name="", condition="",
        method="Geom-mean-weighted CMH",
        min_stratum=None):
    """
    Per-taxon CMH test using geometric-mean-target importance weights
    to correct for sequencing-depth confounding, without discarding samples.

    Within each study stratum, KDEs are fit on log10(depth) for g1 and g2
    separately.  Each sample i receives weight:

        w_i = sqrt( f_other(depth_i) / f_own(depth_i) )

    mean-normalised to 1 within its group x stratum cell and clipped at
    WEIGHT_CLIP (imported from depth_bias_ipw).  The weighted 2x2 table
    for each stratum then becomes:

        [ sum(w_i : g1, present)   sum(w_i : g1, absent)  ]
        [ sum(w_j : g2, present)   sum(w_j : g2, absent)  ]

    These effective counts are passed to the standard CMH aggregator.
    Unlike depth-matched CMH, all samples are retained — ESS loss comes
    only from up/down-weighting, not deletion.

    Diagnostics reported:
      - mean_ess_g1 / mean_ess_g2  : mean effective sample size across strata
      - mean_ess_frac               : ESS / n, averaged across both groups
    """
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights, WEIGHT_CLIP  # noqa: F401

    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    strata = _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum)
    if not strata:
        return None

    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None

    # ── Pre-compute per-stratum weights (expensive but done once) ──
    # KEY INVARIANT: every stratum returned by _get_study_strata is kept —
    # samples with missing _log_depth are assigned weight=1 (i.e. pass through
    # unweighted) rather than being excluded.  This guarantees n1/n2 here
    # exactly match the Within-study CMH counts.
    weighted_strata = []   # list of (g1_idx, g2_idx, w1, w2)
    ess_g1_list, ess_g2_list = [], []

    for study, g1_idx, g2_idx in strata:
        # Start with the full stratum index lists
        g1_all = list(g1_idx)
        g2_all = list(g2_idx)

        # Separate out samples with valid vs missing depth
        ld1_series = meta.loc[g1_all, "_log_depth"]
        ld2_series = meta.loc[g2_all, "_log_depth"]

        g1_has_depth = [i for i in g1_all if pd.notna(ld1_series.loc[i])]
        g2_has_depth = [i for i in g2_all if pd.notna(ld2_series.loc[i])]

        # Build weight arrays initialised to 1.0 (uniform = no reweighting)
        w1 = np.ones(len(g1_all))
        w2 = np.ones(len(g2_all))

        # Only attempt KDE-based reweighting when both groups have enough
        # distinct depth values; otherwise leave weights uniform
        if len(g1_has_depth) >= min_stratum and len(g2_has_depth) >= min_stratum:
            ld1_arr = ld1_series.loc[g1_has_depth].values.astype(float)
            ld2_arr = ld2_series.loc[g2_has_depth].values.astype(float)

            if len(np.unique(ld1_arr)) >= 3 and len(np.unique(ld2_arr)) >= 3:
                kde1 = _BinnedKDE(ld1_arr)
                kde2 = _BinnedKDE(ld2_arr)

                # Compute weights for depth-valid samples
                w1_kde = _geomean_weights(ld1_arr, kde1, kde2)
                w2_kde = _geomean_weights(ld2_arr, kde2, kde1)

                # Map back: depth-valid samples get KDE weight; others stay 1.0
                g1_depth_pos = {idx: pos for pos, idx in enumerate(g1_has_depth)}
                g2_depth_pos = {idx: pos for pos, idx in enumerate(g2_has_depth)}
                for pos, idx in enumerate(g1_all):
                    if idx in g1_depth_pos:
                        w1[pos] = w1_kde[g1_depth_pos[idx]]
                for pos, idx in enumerate(g2_all):
                    if idx in g2_depth_pos:
                        w2[pos] = w2_kde[g2_depth_pos[idx]]

        # ESS diagnostics
        def _ess(w):
            s = w.sum()
            return float(s ** 2 / (w ** 2).sum()) if s > 0 else 0.0

        ess_g1_list.append(_ess(w1))
        ess_g2_list.append(_ess(w2))

        weighted_strata.append((g1_all, g2_all, w1, w2))

    if not weighted_strata:
        return None

    n1_total = sum(len(g1) for g1, _, _, _ in weighted_strata)
    n2_total = sum(len(g2) for _, g2, _, _ in weighted_strata)

    dep = enr = 0

    for mic in mics:
        tables = []
        for g1_idx, g2_idx, w1, w2 in weighted_strata:
            present1 = (bdf.loc[g1_idx, mic].values >= thresh).astype(float)
            present2 = (bdf.loc[g2_idx, mic].values >= thresh).astype(float)

            # Weighted cell counts
            a = float(np.dot(w1, present1))           # g1 present
            b = float(np.dot(w1, 1.0 - present1))     # g1 absent
            c = float(np.dot(w2, present2))            # g2 present
            d = float(np.dot(w2, 1.0 - present2))     # g2 absent

            if a + b < 1e-9 or c + d < 1e-9:
                continue
            tables.append(np.array([[a, b], [c, d]]))

        p, _, direction = cochran_mantel_haenszel(tables)
        if not np.isnan(p) and p < 0.05:
            if direction > 0:
                dep += 1
            elif direction < 0:
                enr += 1

    mean_ess_g1  = float(np.mean(ess_g1_list))  if ess_g1_list  else np.nan
    mean_ess_g2  = float(np.mean(ess_g2_list))  if ess_g2_list  else np.nan
    mean_ess_frac = float(np.mean(
        [e / max(len(g), 1) for (g, _, _, _), e in
         zip(weighted_strata, ess_g1_list)] +
        [e / max(len(g), 1) for (_, g, _, _), e in
         zip(weighted_strata, ess_g2_list)]
    )) if ess_g1_list else np.nan

    result = _pack_result(
        dep, enr, len(mics), n1_total, n2_total,
        condition=condition, method=method, list_name=list_name,
        n_studies=len(weighted_strata))

    result["mean_ess_g1"]   = round(mean_ess_g1,   1) if np.isfinite(mean_ess_g1)   else np.nan
    result["mean_ess_g2"]   = round(mean_ess_g2,   1) if np.isfinite(mean_ess_g2)   else np.nan
    result["mean_ess_frac"] = round(mean_ess_frac, 3) if np.isfinite(mean_ess_frac) else np.nan

    return result


# ====================================================================
#  GMW PER-TAXON  (directional, no significance threshold)
# ====================================================================

def build_weighted_strata(strata, meta, min_stratum=None):
    """Attach geom-mean-target depth weights to each study stratum.

    `strata` is the output of `_get_study_strata`: (study, g1_idx, g2_idx) per
    study. Returns [(g1_ids, g2_ids, w1, w2), ...] in the same order, with
    weights of 1.0 wherever a stratum is too small or too degenerate in log
    depth to fit the two KDEs.

    Extracted so that every per-taxon estimator weights the samples
    identically -- `geom_mean_weighted_cmh_per_taxon` (pooled prevalences) and
    `stratified_shifts_per_taxon` (within-study Mantel-Haenszel) must not be
    allowed to drift apart. `geom_mean_weighted_cmh_enrichment_test` keeps its
    own inline copy on purpose: it is the estimator behind the published
    Figure 5A-D and is left byte-for-byte alone.
    """
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights

    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    weighted_strata = []
    for study, g1_idx, g2_idx in strata:
        g1_all = list(g1_idx)
        g2_all = list(g2_idx)
        w1 = np.ones(len(g1_all))
        w2 = np.ones(len(g2_all))

        ld1_series = meta.loc[g1_all, "_log_depth"]
        ld2_series = meta.loc[g2_all, "_log_depth"]
        g1_has_depth = [i for i in g1_all if pd.notna(ld1_series.loc[i])]
        g2_has_depth = [i for i in g2_all if pd.notna(ld2_series.loc[i])]

        if len(g1_has_depth) >= min_stratum and len(g2_has_depth) >= min_stratum:
            ld1_arr = ld1_series.loc[g1_has_depth].values.astype(float)
            ld2_arr = ld2_series.loc[g2_has_depth].values.astype(float)
            if len(np.unique(ld1_arr)) >= 3 and len(np.unique(ld2_arr)) >= 3:
                kde1 = _BinnedKDE(ld1_arr)
                kde2 = _BinnedKDE(ld2_arr)
                w1_kde = _geomean_weights(ld1_arr, kde1, kde2)
                w2_kde = _geomean_weights(ld2_arr, kde2, kde1)
                g1_depth_pos = {idx: pos for pos, idx in enumerate(g1_has_depth)}
                g2_depth_pos = {idx: pos for pos, idx in enumerate(g2_has_depth)}
                for pos, idx in enumerate(g1_all):
                    if idx in g1_depth_pos:
                        w1[pos] = w1_kde[g1_depth_pos[idx]]
                for pos, idx in enumerate(g2_all):
                    if idx in g2_depth_pos:
                        w2[pos] = w2_kde[g2_depth_pos[idx]]

        weighted_strata.append((g1_all, g2_all, w1, w2))

    return weighted_strata


def geom_mean_weighted_cmh_per_taxon(bdf, taxon_ids, meta,
                                      mask_g1, mask_g2, thresh,
                                      min_stratum=None):
    """
    Compute per-taxon weighted prevalence differences using geom-mean-target
    IPW weights, with NO significance threshold applied.

    Unlike geom_mean_weighted_cmh_enrichment_test (which only counts taxa
    that pass p<0.05), this returns a row for every taxon in taxon_ids,
    including those with weak or non-significant shifts.  This is the input
    for the directional-prevalence plot: even when n2 is small and power is
    low, the distribution of signed prevalence differences across the taxon
    list carries information about the net direction of shift.

    Returns
    -------
    DataFrame with columns:
        taxon          — SGB identifier
        prev_g1        — weighted prevalence in g1 (control), pooled across strata
        prev_g2        — weighted prevalence in g2 (case/disease), pooled across strata
        prev_diff      — prev_g1 - prev_g2  (positive = depleted in g2)
        log_or         — CMH log common OR
        pval           — CMH p-value (not thresholded)
        direction      — +1 depleted in g2, -1 enriched in g2, 0 no data
    Returns None if no valid strata.
    """
    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    strata = _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum)
    if not strata:
        return None

    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None

    weighted_strata = build_weighted_strata(strata, meta, min_stratum)

    rows = []
    for mic in mics:
        tables = []
        sum_a = sum_ab = sum_c = sum_cd = 0.0

        for g1_idx, g2_idx, w1, w2 in weighted_strata:
            present1 = (bdf.loc[g1_idx, mic].values >= thresh).astype(float)
            present2 = (bdf.loc[g2_idx, mic].values >= thresh).astype(float)
            a = float(np.dot(w1, present1))
            b = float(np.dot(w1, 1.0 - present1))
            c = float(np.dot(w2, present2))
            d = float(np.dot(w2, 1.0 - present2))
            if a + b < 1e-9 or c + d < 1e-9:
                continue
            tables.append(np.array([[a, b], [c, d]]))
            sum_a += a;  sum_ab += (a + b)
            sum_c += c;  sum_cd += (c + d)

        if not tables:
            continue

        pval, log_or, direction = cochran_mantel_haenszel(tables)
        prev_g1 = sum_a / sum_ab if sum_ab > 0 else np.nan
        prev_g2 = sum_c / sum_cd if sum_cd > 0 else np.nan
        prev_diff = (prev_g1 - prev_g2) if (not np.isnan(prev_g1) and not np.isnan(prev_g2)) else np.nan

        rows.append({
            "taxon":     mic,
            "prev_g1":   round(prev_g1,   4) if np.isfinite(prev_g1)   else np.nan,
            "prev_g2":   round(prev_g2,   4) if np.isfinite(prev_g2)   else np.nan,
            "prev_diff": round(prev_diff,  4) if np.isfinite(prev_diff) else np.nan,
            "log_or":    round(log_or,     4) if np.isfinite(log_or)    else np.nan,
            "pval":      pval if not np.isnan(pval) else np.nan,
            "direction": int(direction),
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


# ====================================================================
#  STRATIFIED PER-TAXON SHIFT  (Mantel-Haenszel risk difference)
# ====================================================================

def stratified_shifts_per_taxon(bdf, taxon_ids, meta, mask_g1, mask_g2, thresh,
                                min_stratum=None):
    """Within-study case-control prevalence difference for every taxon.

    `geom_mean_weighted_cmh_per_taxon` returns a *pooled* prevalence
    difference: sum_s a_s / sum_s n1_s  -  sum_s c_s / sum_s n2_s. Only its
    p-value and log OR are stratified. Because the case:control ratio ranges
    from 0.02 to 8.2 across studies, the studies enter the two arms in
    different proportions, so the pooled difference carries between-study
    composition as if it were a case-control effect. It exaggerates the
    depletion of common taxa (Unhealthy, 30-70% prevalence: -11 to -14 pp
    pooled against -5 to -6 pp stratified) and reverses the sign for Obesity.

    This estimator keeps every study's own contrast:

        rd_mh    = sum_s W_s (p1_s - p2_s) / sum_s W_s,  W_s = n1_s n2_s / (n1_s + n2_s)
        rd_mh_se = Sato (1989) variance estimator for that statistic
        p_ctrl_mh = sum_s W_s p1_s / sum_s W_s   (control prevalence, same weighting)

    where p1_s and p2_s are the GMW depth-weighted prevalences in study s, from
    exactly the weights `build_weighted_strata` gives the pooled estimator.
    `p_ctrl_mh` is the x-axis the background curve is fitted against: matching
    on the pooled control prevalence while testing a stratified shift would
    mismatch the two axes.

    Sign convention follows `prev_diff`: positive = more prevalent in g1
    (control) = DEPLETED in disease. Figures that plot "case - control" negate
    it.

    Returns
    -------
    (taxa, strata) : (DataFrame, DataFrame) or (None, None)
        taxa    — one row per taxon: taxon, rd_mh, rd_mh_se, t_mh, p_mh,
                  p_ctrl_mh, p_case_mh, pooled_diff, p_ctrl_pooled, n_studies
        strata  — one row per study: study, n_ctrl_w, n_case_w, W_mh
    """
    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    strata = _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum)
    if not strata:
        return None, None

    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None, None

    weighted_strata = build_weighted_strata(strata, meta, min_stratum)

    n_s = len(weighted_strata)
    A  = np.zeros((n_s, len(mics)))      # weighted count present, controls
    C  = np.zeros((n_s, len(mics)))      # weighted count present, cases
    N1 = np.zeros(n_s)                   # weighted arm sizes
    N2 = np.zeros(n_s)
    names = []

    for k, ((g1_idx, g2_idx, w1, w2), (study, _, _)) in enumerate(
            zip(weighted_strata, strata)):
        p1 = (bdf.loc[g1_idx, mics].values >= thresh)
        p2 = (bdf.loc[g2_idx, mics].values >= thresh)
        A[k]  = w1 @ p1
        C[k]  = w2 @ p2
        N1[k] = w1.sum()
        N2[k] = w2.sum()
        names.append(study)

    keep = (N1 > 0) & (N2 > 0)
    if not keep.any():
        return None, None
    A, C, N1, N2 = A[keep], C[keep], N1[keep], N2[keep]
    names = [s for s, k in zip(names, keep) if k]

    p1s = A / N1[:, None]
    p2s = C / N2[:, None]
    W   = N1 * N2 / (N1 + N2)
    sW  = W.sum()

    rd    = (W[:, None] * (p1s - p2s)).sum(0) / sW
    pctrl = (W[:, None] * p1s).sum(0) / sW
    pcase = (W[:, None] * p2s).sum(0) / sW

    # Sato (1989): Var(rd) = (rd * sum_s P_s + sum_s Q_s) / (sum_s W_s)^2
    #
    # With arm 1 = controls (A present of N1) and arm 2 = cases, Q_s below is
    # Sato's Q term exactly, which fixes the mapping; P_s then has to be
    # (n1^2*C - n2^2*A + n1*n2*(n2-n1)/2)/N^2. It was written with every sign
    # reversed until 2026-10-07, which flipped the whole rd-dependent
    # correction. The effect on each taxon's SE runs to tens of percent where
    # the arms are unbalanced; it reaches the curve only through the sigma
    # floor in health_background.background_z, so mean z moved by at most
    # 0.016 in the plotted cells, but rd_mh_se / t_mh / p_mh are published
    # per taxon and were wrong by up to 18%.
    n1, n2 = N1[:, None], N2[:, None]
    NN = n1 + n2
    P_s = (n1 ** 2 * C - n2 ** 2 * A + n1 * n2 * (n2 - n1) / 2) / NN ** 2
    Q_s = (A * (n2 - C) + C * (n1 - A)) / (2 * NN)
    var = (rd * P_s.sum(0) + Q_s.sum(0)) / sW ** 2
    se  = np.sqrt(np.maximum(var, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, rd / se, np.nan)
    p = 2 * norm.sf(np.abs(t))

    taxa = pd.DataFrame({
        "taxon":         mics,
        "rd_mh":         rd,
        "rd_mh_se":      se,
        "t_mh":          t,
        "p_mh":          p,
        "p_ctrl_mh":     pctrl,
        "p_case_mh":     pcase,
        "pooled_diff":   A.sum(0) / N1.sum() - C.sum(0) / N2.sum(),
        "p_ctrl_pooled": A.sum(0) / N1.sum(),
        "n_studies":     len(names),
    })
    strata_df = pd.DataFrame({"study": names, "n_ctrl_w": N1,
                              "n_case_w": N2, "W_mh": W})
    return taxa, strata_df


# ====================================================================
#  WITHIN-STUDY MUNDLAK GLMM (WITHIN–BETWEEN DECOMPOSITION)
# ====================================================================

def mundlak_enrichment_test(bdf, taxon_ids, meta, mask_g1, mask_g2, thresh,
                            list_name="", condition="",
                            method="Mundlak GLMM",
                            min_stratum=None, depth_knots=3):
    """
    Per-taxon Mundlak within–between logistic regression.

    Decomposes the group indicator into within-study (X_w) and
    between-study (X_b) components.  β_w is the CMH-analogous
    within-study effect; β_b captures (potentially confounded)
    between-study variation.

    Uses cluster-robust SEs grouped by study.
    Significance determined by FDR < 0.05 on β_w (the within effect).

    Returns dict compatible with CMH result format, plus Mundlak-
    specific diagnostics (n_hausman_sig, median_hausman_diff).
    """
    from .glmm_utils import (
        _fit_mundlak_one_taxon, build_design_matrix, bh_fdr,
        _ensure_statsmodels,
    )
    import warnings

    _ensure_statsmodels()

    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM

    # ── Subset to g1 ∪ g2, require study + depth ──
    idx = meta.index[mask_g1 | mask_g2]
    idx = [i for i in idx if i in bdf.index]

    valid = (meta.loc[idx, "_log_depth"].notna()
             & meta.loc[idx, "_study"].notna())
    idx = [i for i in idx if valid.get(i, False)]

    if len(idx) < 20:
        return None

    group = mask_g2.loc[idx].astype(int)   # 0=control, 1=case
    log_depth = meta.loc[idx, "_log_depth"]
    study = meta.loc[idx, "_study"]

    n_g1 = int((group == 0).sum())
    n_g2 = int((group == 1).sum())
    if n_g1 < 10 or n_g2 < 10:
        return None

    # Check minimum informative studies
    informative = 0
    for s in study.unique():
        sm = study == s
        g_sub = group[sm]
        if g_sub.sum() >= min_stratum and (len(g_sub) - g_sub.sum()) >= min_stratum:
            informative += 1
    if informative < 3:
        return None

    # ── Build depth covariates (no geography — Mundlak handles it) ──
    covariates = build_design_matrix(
        log_depth, mode="none", depth_knots=depth_knots)
    covariates = covariates.loc[idx]

    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None

    # ── Per-taxon Mundlak regressions ──
    results = []
    for mic in mics:
        y = (bdf.loc[idx, mic] >= thresh).astype(float).values
        res = _fit_mundlak_one_taxon(
            mic, y, group.values, study.values,
            covariates, min_studies=3)
        results.append(res)

    df = pd.DataFrame(results)
    if df.empty or "p_within" not in df.columns:
        return None

    # FDR correction on the within-study p-values
    df["q_within"] = bh_fdr(df["p_within"])

    # Count significant at FDR < 0.05 on β_w (within-study effect)
    sig = df[df["q_within"] < 0.05]
    dep = int((sig["beta_within"] < 0).sum())   # depleted in case
    enr = int((sig["beta_within"] > 0).sum())   # enriched in case
    total = len(mics)
    n_converged = int(df["converged"].sum())

    # ── Hausman diagnostic: how many taxa show β_w ≠ β_b? ──
    hausman_sig = df["hausman_p"].dropna()
    n_hausman_sig = int((hausman_sig < 0.05).sum()) if len(hausman_sig) > 0 else 0
    median_hausman = float(df["hausman_diff"].median()) if df["hausman_diff"].notna().any() else np.nan

    result = _pack_result(
        dep, enr, total, n_g1, n_g2,
        condition=condition, method=method, list_name=list_name,
        n_studies=informative)

    # Add Mundlak-specific diagnostics
    result["n_converged"] = n_converged
    result["n_hausman_sig"] = n_hausman_sig
    result["median_hausman_diff"] = round(median_hausman, 4) if np.isfinite(median_hausman) else np.nan

    return result


# ====================================================================
#  GROUP-MASK BUILDER
# ====================================================================


def subject_primary_mask(meta):
    """True for the one sample per subject chosen to represent that subject.

    Written by tools/build_verified_m1.py as `subject_primary`. Gated by
    config.DEDUP_SUBJECTS. Absent column or a disabled flag means every sample
    passes, so behaviour is unchanged where the column has not been built.
    """
    if not getattr(config, "DEDUP_SUBJECTS", False):
        return pd.Series(True, index=meta.index)
    col = meta.get("subject_primary")
    if col is None:
        return pd.Series(True, index=meta.index)
    return col.astype(str).str.strip().str.lower().ne("no")


def case_eligible_mask(meta):
    """True where a sample may take part in a per-condition case/control test.

    MicrobiomeDB's `study_condition` distinguishes states that its `disease`
    field collapses: LiJ_2017's 56 pre-hypertensive participants are recorded as
    `hypertension` by `disease`, and the post-FMT timepoints in LiSS_2016 and
    IaniroG_2022, and YachidaS_2019's post-surgery cases, are recorded as their
    underlying condition. Those samples are neither clean cases nor clean
    controls, so they are dropped from BOTH arms of every per-condition contrast
    rather than being silently counted as either.

    tools/build_verified_m1.py writes the `mdb_case_eligible` column this reads
    ("Yes", "No:pre-disease", "No:intervention", or "" where MicrobiomeDB has no
    study_condition for the sample). Absent column or blank value means eligible,
    so behaviour is unchanged for samples the source says nothing about.

    The `unhealthy` and `obese` contrasts are deliberately NOT filtered: they key
    on the curated `Unhealthy` label and the obesity columns rather than on the
    exploded disease flags, so the ambiguity this guards against does not arise
    there.
    """
    col = meta.get("mdb_case_eligible")
    if col is None:
        return pd.Series(True, index=meta.index)
    v = col.astype(str).str.strip()
    return ~v.str.startswith("No")


def build_group_masks(meta, comparison_type, disease_cols=None):
    """
    Returns (mask_g1, mask_g2) boolean Series.

    comparison_type:
      'unhealthy' : Industrial Healthy vs Industrial Unhealthy
      'obese'     : Industrial Healthy non-obese vs Industrial Healthy obese
      'overweight': Industrial non-overweight vs Industrial overweight
      'disease'   : Industrial not-diseased vs Industrial diseased

    Demographic strata (config.DEMOGRAPHIC_MAP):
      'age_band:<band>'      : Industrial Healthy reference band vs that band
      'sex'                  : Industrial Healthy male vs female
      'bmi_obese_vs_rest'    : Industrial BMI 18.5-30 vs BMI >=30
      'bmi_excess_vs_normal' : Industrial BMI 18.5-25 vs BMI >=25
    """
    ind = meta["Lifestyle"] == "Industrial"
    healthy = meta["Unhealthy"] == "Healthy"

    # One sample per subject for every case/control contrast. Applied to `ind`
    # so each branch below inherits it; accumulation does not route through this
    # function and is therefore untouched.
    _primary = subject_primary_mask(meta)
    _n_sec = int((ind & ~_primary).sum())
    if _n_sec:
        print(f"    [subject dedup] {_n_sec:,} secondary samples excluded "
              f"({int((ind & _primary).sum()):,} Industrial primaries remain)")
    ind = ind & _primary

    if comparison_type == "unhealthy":
        return ind & healthy, ind & (meta["Unhealthy"] == "Unhealthy")

    elif comparison_type == "obese":
        # Industrial non-obese vs Industrial obese (all health statuses).
        # NOTE: we do NOT restrict to Healthy here because the Unhealthy
        # flag is partly driven by obesity itself, which would make
        # Healthy ∩ Obese near-empty by construction.
        ob = meta.get("Obesity_flag", pd.Series(False, index=meta.index))
        mask_g1 = ind & (~ob)
        mask_g2 = ind & ob
        print(f"    [obese masks] Industrial={ind.sum():,}, "
              f"Obesity_flag={ob.sum():,}, "
              f"g1(Ind non-obese)={mask_g1.sum():,}, "
              f"g2(Ind obese)={mask_g2.sum():,}")
        return mask_g1, mask_g2

    elif comparison_type == "obese_healthy_only":
        # Sensitivity: healthy non-obese vs healthy obese (within Industrial)
        ob = meta.get("Obesity_flag", pd.Series(False, index=meta.index))
        mask_g1 = ind & healthy & (~ob)
        mask_g2 = ind & healthy & ob
        print(f"    [obese healthy-only] "
              f"g1(Ind H non-obese)={mask_g1.sum():,}, "
              f"g2(Ind H obese)={mask_g2.sum():,}")
        return mask_g1, mask_g2


    elif comparison_type == "overweight":
        ow = meta.get("Overweight_flag", pd.Series(False, index=meta.index))
        return ind & ~ow, ind & ow

    elif comparison_type == "disease":
        valid_cols = [c for c in (disease_cols or []) if c in meta.columns]
        if not valid_cols:
            return (pd.Series(False, index=meta.index),
                    pd.Series(False, index=meta.index))
        diseased = meta[valid_cols].apply(
            lambda row: any(_is_positive(v) for v in row), axis=1)
        elig = case_eligible_mask(meta)
        n_drop = int((ind & ~elig).sum())
        if n_drop:
            print(f"    [case-eligibility] dropping {n_drop:,} pre-disease / "
                  f"intervention-arm samples from both arms")
        return ind & ~diseased & elig, ind & diseased & elig

    elif comparison_type.startswith("age_band:"):
        # Within Industrial Healthy, so a band difference is not simply the
        # disease burden that accumulates with age.
        band = comparison_type.split(":", 1)[1]
        ref = age_band_series(meta).eq(config.AGE_REFERENCE_BAND)
        sel = age_band_series(meta).eq(band)
        mask_g1, mask_g2 = ind & healthy & ref, ind & healthy & sel
        print(f"    [age {band} vs {config.AGE_REFERENCE_BAND}] "
              f"g1(ref)={mask_g1.sum():,}, g2({band})={mask_g2.sum():,}")
        return mask_g1, mask_g2

    elif comparison_type == "sex":
        sx = sex_series(meta)
        mask_g1, mask_g2 = ind & healthy & sx.eq("Male"), \
            ind & healthy & sx.eq("Female")
        print(f"    [sex] g1(male)={mask_g1.sum():,}, "
              f"g2(female)={mask_g2.sum():,}")
        return mask_g1, mask_g2

    elif comparison_type in ("bmi_obese_vs_rest", "bmi_excess_vs_normal"):
        # Not restricted to Healthy: obesity is itself what sets the Unhealthy
        # label, so Industrial & Healthy & Obese is empty by construction.
        # Underweight is excluded from both arms rather than pooled into the
        # reference -- it is a distinct phenotype, not a milder normal.
        bb = bmi_band_series(meta)
        if comparison_type == "bmi_obese_vs_rest":
            g1, g2 = bb.isin(["Normal", "Overweight"]), bb.eq("Obese")
        else:
            g1, g2 = bb.eq("Normal"), bb.isin(["Overweight", "Obese"])
        mask_g1, mask_g2 = ind & g1, ind & g2
        print(f"    [{comparison_type}] g1={mask_g1.sum():,}, "
              f"g2={mask_g2.sum():,} "
              f"(underweight excluded: {(ind & bb.eq('Underweight')).sum():,})")
        return mask_g1, mask_g2

    raise ValueError(f"Unknown comparison: {comparison_type}")


# ====================================================================
#  DEMOGRAPHIC STRATA
# ====================================================================
# The metadata carries these three fields as free text from 100+ studies, so
# each needs normalising before it can be banded. None of this belongs in the
# mask builder itself -- Table S3 reports the band sizes and has to derive
# them the same way.

def age_band_series(meta):
    """Age band label per sample, NA where age is missing.

    Bands come from the numeric `Age` column. The `Age group` column is not
    used: it is a coarse, case-inconsistent mix (Adult/adult, Elderly/senior/
    Senior, Child/child/schoolage) that cannot be cut at 18/30/45/60.
    """
    if "Age" not in meta.columns:
        return pd.Series(pd.NA, index=meta.index, dtype="object")
    age = pd.to_numeric(meta["Age"], errors="coerce")
    out = pd.Series(pd.NA, index=meta.index, dtype="object")
    for label, lo, hi in config.AGE_BANDS:
        out = out.mask(age.ge(lo) & age.lt(hi), label)
    return out


def sex_series(meta):
    """'Female'/'Male' per sample, NA otherwise.

    Case-folded because the column holds both `Female` and `female`; anything
    that is not one of the two (`unknown`, blank) becomes NA rather than being
    forced into an arm.
    """
    col = next((c for c in ("Gender", "Sex") if c in meta.columns), None)
    if col is None:
        return pd.Series(pd.NA, index=meta.index, dtype="object")
    return (meta[col].astype(str).str.strip().str.lower()
            .map({"female": "Female", "male": "Male"}))


def bmi_band_series(meta):
    """WHO BMI category per sample, NA where BMI is missing or implausible.

    The plausibility window drops a handful of sentinel and out-of-range
    values (a 0.0 and three above 60) that would otherwise land in the
    Underweight and Obese arms.
    """
    if "BMI" not in meta.columns:
        return pd.Series(pd.NA, index=meta.index, dtype="object")
    lo_ok, hi_ok = config.BMI_PLAUSIBLE
    bmi = pd.to_numeric(meta["BMI"], errors="coerce")
    bmi = bmi.where(bmi.between(lo_ok, hi_ok))
    out = pd.Series(pd.NA, index=meta.index, dtype="object")
    for label, lo, hi in config.BMI_BANDS:
        out = out.mask(bmi.ge(lo) & bmi.lt(hi), label)
    return out


# ====================================================================
#  INTERNAL HELPERS
# ====================================================================

def _is_positive(val):
    """Check if a value indicates disease-positive."""
    if pd.isna(val):
        return False
    if isinstance(val, (bool, np.bool_)):
        return val
    if isinstance(val, (int, float)):
        return val > 0
    s = str(val).strip().lower()
    return s in ("yes", "true", "1", "positive", "y")


def _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum):
    """Return list of (study_name, g1_idx, g2_idx) qualifying strata."""
    md = meta.copy()
    md["_g1"] = mask_g1
    md["_g2"] = mask_g2
    md = md[md.index.isin(bdf.index)]
    strata = []
    for s in md["_study"].dropna().unique():
        sm = md["_study"] == s
        g1s = md.index[sm & md["_g1"]]
        g2s = md.index[sm & md["_g2"]]
        if len(g1s) >= min_stratum and len(g2s) >= min_stratum:
            strata.append((s, g1s, g2s))
    return strata


def _nearest_neighbour_match(g1_depths, g2_depths, replace=False):
    """1:1 nearest-neighbour matching on log-depth (greedy, no-replacement)."""
    if len(g1_depths) == 0 or len(g2_depths) == 0:
        return [], []

    if len(g1_depths) <= len(g2_depths):
        query, ref = g1_depths, g2_depths
        query_is_g1 = True
    else:
        query, ref = g2_depths, g1_depths
        query_is_g1 = False

    ref_vals = ref.values.reshape(-1, 1)
    tree = cKDTree(ref_vals)
    query_vals = query.values.reshape(-1, 1)
    _, indices = tree.query(query_vals, k=1)

    if not replace:
        used = set()
        matched_query, matched_ref = [], []
        dists = np.abs(query_vals.ravel() - ref_vals.ravel()[indices])
        order = np.argsort(dists)
        for qi in order:
            ri = indices[qi]
            if ri not in used:
                used.add(ri)
                matched_query.append(query.index[qi])
                matched_ref.append(ref.index[ri])
    else:
        matched_query = query.index.tolist()
        matched_ref = [ref.index[i] for i in indices]

    if query_is_g1:
        return matched_query, matched_ref
    else:
        return matched_ref, matched_query


def _pack_result(dep, enr, total, n1, n2, **extra):
    """Package enrichment counts into a standardized result dict."""
    def _ci(count, tot):
        f = count / tot if tot > 0 else 0.0
        if tot > 0:
            lo, hi = wilson_ci(count, tot)
        else:
            lo, hi = 0.0, 0.0
        return f, max(0, f - lo), max(0, hi - f)

    fd, ld, hd = _ci(dep, total)
    fe, le, he = _ci(enr, total)

    d = {
        "n_group1": n1, "n_group2": n2, "total_taxa": total,
        "depleted_count": dep, "depleted_frac": fd,
        "depleted_ci_lo": ld, "depleted_ci_hi": hd,
        "enriched_count": enr, "enriched_frac": fe,
        "enriched_ci_lo": le, "enriched_ci_hi": he,
    }
    d.update(extra)
    return d
