"""
Differential prevalence (Fisher's exact) and abundance (Mann-Whitney U on CLR)
between Industrial and Non-Industrial populations.

Produces volcano plots with 7-colour scheme:
  Grey          - not in any list
  Blue shades   - VANISH only / VANISH+ only / VANISH  &  VANISH+
  Red shades    - BloSSUM only / BloSSUM+ only / BloSSUM  &  BloSSUM+
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_utils import apply_oi_style, savefig_dual
apply_oi_style()

from scipy.special import gammaln, logsumexp
from scipy.stats import (mannwhitneyu, fisher_exact, norm, chi2_contingency,
                         chi2 as chi2_dist)

from . import config
from .common import (clr_transform, assign_volcano_category,
                     assign_volcano_category_7)


FLOAT_TINY_LOG10 = np.log10(np.nextafter(0.0, 1.0))


def _pvalue_from_log10(log10_pval):
    """Return float p-value when representable; otherwise preserve underflow as 0."""
    if not np.isfinite(log10_pval) or log10_pval < FLOAT_TINY_LOG10:
        return 0.0
    return float(10.0 ** log10_pval)


def _neg_log10_bonferroni(log10_pval, n_tests):
    """Compute -log10(min(1, p * n_tests)) from log10(p)."""
    return np.maximum(-log10_pval - np.log10(n_tests), 0.0)


def _fisher_two_sided_log10_pvalue(d1, n1, d2, n2):
    """Two-sided Fisher exact log10 p-value for [[d1, n1-d1], [d2, n2-d2]]."""
    M = n1 + n2
    K = d1 + d2
    N = n1
    lo = max(0, N - (M - K))
    hi = min(N, K)

    def logpmf(x):
        return (
            gammaln(K + 1) - gammaln(x + 1) - gammaln(K - x + 1)
            + gammaln(M - K + 1) - gammaln(N - x + 1)
            - gammaln(M - K - (N - x) + 1)
            - (gammaln(M + 1) - gammaln(N + 1) - gammaln(M - N + 1))
        )

    obs = logpmf(d1)
    vals = np.array([logpmf(x) for x in range(lo, hi + 1)], dtype=float)
    logp = float(logsumexp(vals[vals <= obs + 1e-12]))
    return logp / np.log(10.0)


def _chi2_1df_log10_pvalue(chi2):
    """Stable log10 survival p-value for chi-square with one degree of freedom."""
    return float((np.log(2.0) + norm.logsf(np.sqrt(max(float(chi2), 0.0))))
                 / np.log(10.0))


# ════════════════════════════════════════════════════════════════════
#  GENERIC TEST FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def test_prevalence(breadth_df, meta, g1_mask_fn, g2_mask_fn,
                    g1_label, g2_label, threshold=None):
    """Per-SGB Fisher's exact test on breadth >= threshold."""
    if threshold is None:
        threshold = config.BREADTH_THRESHOLD

    g1 = meta.index[g1_mask_fn(meta)]
    g2 = meta.index[g2_mask_fn(meta)]
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return pd.DataFrame()

    print(f"    Prevalence: {g1_label} (n={n1:,}) vs {g2_label} (n={n2:,})")

    results = []
    for mic in breadth_df.columns:
        d1 = int((breadth_df.loc[g1, mic] >= threshold).sum())
        d2 = int((breadth_df.loc[g2, mic] >= threshold).sum())
        if d1 == 0 and d2 == 0:
            continue
        odds, _ = fisher_exact([[d1, n1-d1], [d2, n2-d2]])
        log10_pval = _fisher_two_sided_log10_pvalue(d1, n1, d2, n2)
        results.append({
            "microbe": mic,
            "prev_g1": d1/n1, "prev_g2": d2/n2,
            "prevalence_diff": d1/n1 - d2/n2,
            "odds_ratio": odds, "pval": _pvalue_from_log10(log10_pval),
            "log10_pval": log10_pval,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    n_tests = len(df)
    df["neg_log10_p_bonf"] = _neg_log10_bonferroni(df["log10_pval"], n_tests)
    return df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)


def test_abundance(abundance_df, meta, g1_mask_fn, g2_mask_fn,
                   g1_label, g2_label):
    """Per-SGB Mann-Whitney U on CLR-transformed abundance."""
    g1 = meta.index[g1_mask_fn(meta)]
    g2 = meta.index[g2_mask_fn(meta)]
    if len(g1) < 5 or len(g2) < 5:
        return pd.DataFrame()

    print(f"    Abundance:  {g1_label} (n={len(g1):,}) vs {g2_label} (n={len(g2):,})")

    all_idx = list(g1) + list(g2)
    clr = clr_transform(abundance_df.loc[all_idx])

    results = []
    for mic in clr.columns:
        x, y = clr.loc[g1, mic].dropna(), clr.loc[g2, mic].dropna()
        if len(x) < 5 or len(y) < 5:
            continue
        try:
            U, _ = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        except TypeError:
            U, _ = mannwhitneyu(x, y, alternative="two-sided")

        n1u, n2u = len(x), len(y)
        std_U = np.sqrt(n1u * n2u * (n1u + n2u + 1) / 12)
        if std_U == 0:
            log10p = 0.0
        else:
            z = (U - n1u*n2u/2) / std_U
            log10p = np.log10(2) + norm.logsf(abs(z)) / np.log(10)
        results.append({
            "microbe": mic, "clr_diff": x.mean() - y.mean(),
            "U_stat": U, "pval": _pvalue_from_log10(log10p), "log10_pval": log10p,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    n_tests = len(df)
    df["neg_log10_p_bonf"] = _neg_log10_bonferroni(df["log10_pval"], n_tests)
    return df.sort_values("clr_diff", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════
#  VOLCANO PLOTS - 7-COLOUR SCHEME
# ════════════════════════════════════════════════════════════════════

def plot_volcano_7(df, effect_col, vanish_set, blossum_set,
                   vanish_plus_set, blossum_plus_set,
                   title, filepath, xlabel=None,
                   sig_col="neg_log10_p_bonf"):
    """
    Volcano plot with 7-colour scheme (publication fonts).
    """
    if df.empty:
        return

    # Publication font context (consistent with figures/section1.py)
    import matplotlib as _mpl
    _saved = {k: _mpl.rcParams.get(k) for k in [
        "font.size", "axes.linewidth", "pdf.fonttype", "ps.fonttype",
        "font.family", "font.sans-serif", "xtick.major.width", "ytick.major.width",
        "xtick.major.size", "ytick.major.size"]}
    _mpl.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
    })

    df = df.copy()
    df["category_7"] = df["microbe"].apply(
        lambda m: assign_volcano_category_7(
            m, vanish_set, blossum_set, vanish_plus_set, blossum_plus_set))

    fig, ax = plt.subplots(figsize=(11, 8))

    for cat in config.VOLCANO_DRAW_ORDER_7:
        sub = df[df["category_7"] == cat]
        if sub.empty:
            continue
        color = config.VOLCANO_PALETTE_7.get(cat, "#AAA")
        is_bg = (cat == "No list")
        ax.scatter(sub[effect_col], sub[sig_col],
                   c=color, s=8 if is_bg else 18,
                   alpha=0.30 if is_bg else 0.75,
                   edgecolors="none",
                   zorder=2 if is_bg else 3,
                   rasterized=is_bg)

    ax.axhline(-np.log10(0.05), ls="--", color="grey", alpha=0.5, lw=0.6)
    ax.axvline(0, ls="-", color="grey", alpha=0.3, lw=0.5)
    ax.set_xlabel(xlabel or effect_col, fontsize=10)
    ax.set_ylabel("\u2212log\u2081\u2080(p) [Bonferroni]", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)

    from matplotlib.patches import Patch
    legend_elements = []
    for cat in config.VOLCANO_DRAW_ORDER_7:
        sub = df[df["category_7"] == cat]
        if not sub.empty:
            legend_elements.append(
                Patch(facecolor=config.VOLCANO_PALETTE_7[cat],
                      edgecolor="grey", linewidth=0.3,
                      label=f"{cat} (n={len(sub)})"))
    ax.legend(handles=legend_elements, fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="grey",
              handlelength=1.0, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)

    plt.tight_layout()
    savefig_dual(fig, filepath, dpi=300)
    plt.close()
    print(f"    Saved: {filepath}")

    _mpl.rcParams.update(_saved)


def plot_volcano_legacy(df, effect_col, vanish_set, blossum_set,
                        title, filepath, xlabel=None,
                        sig_col="neg_log10_p_bonf"):
    """Legacy volcano plot with original VANISH/BloSSUM colouring only."""
    if df.empty:
        return

    df = df.copy()
    df["category"] = df["microbe"].apply(
        lambda m: assign_volcano_category(m, vanish_set, blossum_set))

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat in ["Other", "BloSSUM", "VANISH", "Both"]:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub[effect_col], sub[sig_col],
                   c=config.VOLCANO_PALETTE.get(cat, "#AAA"),
                   s=15, alpha=0.6, label=f"{cat} ({len(sub)})",
                   edgecolors="none", zorder=2 if cat == "Other" else 3)

    ax.axhline(-np.log10(0.05), ls="--", color="grey", alpha=0.5)
    ax.axvline(0, ls="-", color="grey", alpha=0.3)
    ax.set_xlabel(xlabel or effect_col)
    ax.set_ylabel("-log10(p) [Bonferroni]")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    savefig_dual(fig, filepath, dpi=200)
    plt.close()
    print(f"    Saved: {filepath}")


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_differential(data, matched_idx):
    """
    Industrial vs Non-Industrial differential analysis on depth-matched samples.

    Produces:
        - Prevalence + abundance volcano plots (7-colour + legacy)
        - Result CSVs with 7-category column
        - Top-100 NI-enriched (VANISH+) and Ind-enriched (BloSSUM+) lists

    Returns
    -------
    prev_df, abund_df, vanish_plus_ids, blossum_plus_ids
    """
    ref = data.ref_name
    out = config.outdir(ref)

    breadth_m = data.breadth.loc[matched_idx]
    abund_m   = data.abundance.loc[matched_idx]
    meta_m    = data.meta.loc[matched_idx]

    print(f"\n  [{ref}] Differential: Industrial vs Non-Industrial")

    ni  = lambda df: df["Lifestyle"].isin(["H-G", "Transitional"])
    ind = lambda df: df["Lifestyle"] == "Industrial"

    vanish_plus_ids  = []
    blossum_plus_ids = []

    # ── Prevalence ──
    prev = test_prevalence(breadth_m, meta_m, ind, ni, "Industrial", "Non-Industrial")
    if not prev.empty:
        # Top-100 lists (tail = most NI-enriched, head = most Ind-enriched)
        if len(prev) >= 100:
            vanish_plus_ids  = prev.tail(100)["microbe"].tolist()
            blossum_plus_ids = prev.head(100)["microbe"].tolist()
            _save_top100(vanish_plus_ids,
                         f"{out}/top100_NI_enriched_prevalence.txt")
            _save_top100(blossum_plus_ids,
                         f"{out}/top100_Ind_enriched_prevalence.txt")

        # Add 7-category column to CSV
        vp_set = set(vanish_plus_ids)
        bp_set = set(blossum_plus_ids)
        prev["category_7"] = prev["microbe"].apply(
            lambda m: assign_volcano_category_7(
                m, data.vanish_set, data.blossum_set, vp_set, bp_set))
        prev.to_csv(f"{out}/prevalence_ind_vs_nonind.csv", index=False)

        # 7-colour volcano
        plot_volcano_7(
            prev, "prevalence_diff",
            data.vanish_set, data.blossum_set,
            vp_set, bp_set,
            f"{ref}: Prevalence (Industrial - Non-Industrial)",
            f"{out}/volcano_prevalence_7col.png",
            xlabel="Prevalence difference")

        # Legacy volcano (for comparison)
        plot_volcano_legacy(
            prev, "prevalence_diff",
            data.vanish_set, data.blossum_set,
            f"{ref}: Prevalence (Industrial - Non-Industrial)",
            f"{out}/volcano_prevalence.png",
            xlabel="Prevalence difference")

    # ── Abundance ──
    abund = test_abundance(abund_m, meta_m, ind, ni, "Industrial", "Non-Industrial")
    if not abund.empty:
        vp_set = set(vanish_plus_ids)
        bp_set = set(blossum_plus_ids)
        abund["category_7"] = abund["microbe"].apply(
            lambda m: assign_volcano_category_7(
                m, data.vanish_set, data.blossum_set, vp_set, bp_set))
        abund.to_csv(f"{out}/abundance_ind_vs_nonind.csv", index=False)

        # 7-colour volcano
        plot_volcano_7(
            abund, "clr_diff",
            data.vanish_set, data.blossum_set,
            vp_set, bp_set,
            f"{ref}: CLR Abundance (Industrial - Non-Industrial)",
            f"{out}/volcano_abundance_7col.png",
            xlabel="CLR difference")

        # Legacy volcano
        plot_volcano_legacy(
            abund, "clr_diff",
            data.vanish_set, data.blossum_set,
            f"{ref}: CLR Abundance (Industrial - Non-Industrial)",
            f"{out}/volcano_abundance.png",
            xlabel="CLR difference")

    # Overlap summary
    if vanish_plus_ids:
        top_ni  = set(vanish_plus_ids)
        top_ind = set(blossum_plus_ids)
        print(f"    VANISH  &  NI-enriched (VANISH+):  {len(data.vanish_set & top_ni)}")
        print(f"    BloSSUM  &  Ind-enriched (BloSSUM+): {len(data.blossum_set & top_ind)}")
        print(f"    VANISH+ saved:  {len(vanish_plus_ids)} taxa")
        print(f"    BloSSUM+ saved: {len(blossum_plus_ids)} taxa")

    return prev, abund, vanish_plus_ids, blossum_plus_ids


def _save_top100(microbe_series, path):
    with open(path, "w") as f:
        for m in microbe_series:
            f.write(f"{m}\n")


# ════════════════════════════════════════════════════════════════════
#  GEOM-MEAN-WEIGHTED DIFFERENTIAL  (GMW - IPW on depth)
# ════════════════════════════════════════════════════════════════════

def test_prevalence_weighted(breadth_df, g1_idx, g2_idx, w1, w2,
                              g1_label, g2_label, threshold=None):
    """
    Per-SGB weighted prevalence test using geom-mean-target IPW weights.

    Instead of counting binary presence, each sample i contributes w_i
    to the cell total, so the effective 2x2 table becomes:

        [ Σw (g1,present)   Σw (g1,absent) ]
        [ Σw (g2,present)   Σw (g2,absent) ]

    chi2_contingency (with Yates correction off) accepts non-integer
    counts and returns a valid chi-squared p-value.  Prevalence
    difference is computed as weighted fractions:
        D = Σw·present_g1 / Σw_g1  -  Σw·present_g2 / Σw_g2

    Returns
    -------
    DataFrame sorted descending by weighted prevalence_diff
    (g1 - g2, so positive = enriched in g1 / Industrial).
    """
    if threshold is None:
        threshold = config.BREADTH_THRESHOLD

    g1_idx = list(g1_idx)
    g2_idx = list(g2_idx)
    W1 = float(w1.sum())
    W2 = float(w2.sum())
    if W1 == 0 or W2 == 0:
        return pd.DataFrame()

    # Rao-Scott first-order design effect. a, b, c, d below are WEIGHTED
    # sums, but chi2_contingency treats them as counts of independent
    # observations, so chi2 is inflated by roughly n/ESS and p is too small.
    # The ESS was already being printed on this line and simply not applied.
    # deff is the weighted mean of the two arms' n/ESS; chi2 is divided by it,
    # which is the standard first-order correction for a survey-weighted
    # contingency table. Set DIFFERENTIAL_DESIGN_EFFECT = False to reproduce
    # the uncorrected p-values.
    ess1 = W1 ** 2 / float((w1 ** 2).sum())
    ess2 = W2 ** 2 / float((w2 ** 2).sum())
    print(f"    Weighted prevalence: {g1_label} (ESS~{ess1:.0f}/{len(g1_idx)}) "
          f"vs {g2_label} (ESS~{ess2:.0f}/{len(g2_idx)})")
    _apply_deff = bool(getattr(config, "DIFFERENTIAL_DESIGN_EFFECT", True))
    deff = 1.0
    if _apply_deff and ess1 > 0 and ess2 > 0:
        deff = max(1.0, (len(g1_idx) + len(g2_idx)) / (ess1 + ess2))
        print(f"      design effect {deff:.3f} applied to chi2 "
              f"(Rao-Scott first order); without it every p here is too small")

    b1 = breadth_df.loc[g1_idx]
    b2 = breadth_df.loc[g2_idx]

    results = []
    for mic in breadth_df.columns:
        p1 = (b1[mic].values >= threshold).astype(float)
        p2 = (b2[mic].values >= threshold).astype(float)

        a = float(np.dot(w1, p1))          # g1 present
        b = float(np.dot(w1, 1.0 - p1))   # g1 absent
        c = float(np.dot(w2, p2))          # g2 present
        d = float(np.dot(w2, 1.0 - p2))   # g2 absent

        if a + b < 1e-9 or c + d < 1e-9:
            continue
        if a + c == 0 or b + d == 0:
            continue

        try:
            chi2, pval, _, _ = chi2_contingency(
                [[a, b], [c, d]], correction=False)
        except ValueError:
            continue
        if deff > 1.0:
            chi2 = chi2 / deff
            pval = float(chi2_dist.sf(chi2, df=1))

        prev_g1 = a / (a + b)
        prev_g2 = c / (c + d)
        log10_pval = _chi2_1df_log10_pvalue(chi2)
        results.append({
            "microbe":          mic,
            "prev_g1":          prev_g1,
            "prev_g2":          prev_g2,
            "prevalence_diff":  prev_g1 - prev_g2,
            "chi2":             chi2,
            "pval":             _pvalue_from_log10(log10_pval),
            "log10_pval":       log10_pval,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    n_tests = len(df)
    df["neg_log10_p_bonf"] = _neg_log10_bonferroni(df["log10_pval"], n_tests)
    return df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)


def run_differential_gmw(data, mask_g1, mask_g2,
                          g1_label, g2_label,
                          out_suffix, top_n=100):
    """
    Geom-mean-weighted (IPW) differential prevalence between two groups
    defined by boolean meta masks, without discarding any samples.

    KDEs are fit on log10(_depth) for each group over the FULL cohort
    (not just matched samples), and importance weights reweight each
    group toward the geometric-mean target distribution.

    Parameters
    ----------
    data       : ReferenceData
    mask_g1    : boolean Series indexed on data.meta - group 1 (e.g. Industrial)
    mask_g2    : boolean Series indexed on data.meta - group 2 (e.g. NI)
    g1_label   : str   label for g1
    g2_label   : str   label for g2
    out_suffix : str   appended to output filenames, e.g. "ind_all_vs_ni"
    top_n      : int   number of taxa returned in each direction (default 100)

    Returns
    -------
    (prev_df, g1_enriched_ids, g2_enriched_ids)
        g1_enriched_ids - top_n taxa most enriched in g1 (BloSSUM-like)
        g2_enriched_ids - top_n taxa most enriched in g2 (VANISH-like)
    """
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights

    meta     = data.meta
    breadth  = data.breadth
    ref      = data.ref_name
    out      = config.outdir(ref)

    g1_idx = meta.index[mask_g1 & meta.index.isin(breadth.index)]
    g2_idx = meta.index[mask_g2 & meta.index.isin(breadth.index)]

    if len(g1_idx) < 20 or len(g2_idx) < 20:
        print(f"    [GMW] Too few samples: {g1_label}={len(g1_idx)}, "
              f"{g2_label}={len(g2_idx)} - skipping")
        return pd.DataFrame(), [], []

    # Build weight arrays - all samples included, weight=1 for those without depth
    g1_all = list(g1_idx)
    g2_all = list(g2_idx)
    w1 = np.ones(len(g1_all))
    w2 = np.ones(len(g2_all))

    ld1_series = meta["_log_depth"].reindex(g1_all)
    ld2_series = meta["_log_depth"].reindex(g2_all)

    g1_has_depth = [i for i in g1_all if pd.notna(ld1_series.loc[i])]
    g2_has_depth = [i for i in g2_all if pd.notna(ld2_series.loc[i])]

    if len(g1_has_depth) >= 20 and len(g2_has_depth) >= 20:
        ld1_arr = ld1_series.loc[g1_has_depth].values.astype(float)
        ld2_arr = ld2_series.loc[g2_has_depth].values.astype(float)

        if len(np.unique(ld1_arr)) >= 3 and len(np.unique(ld2_arr)) >= 3:
            kde1 = _BinnedKDE(ld1_arr)
            kde2 = _BinnedKDE(ld2_arr)
            w1_kde = _geomean_weights(ld1_arr, kde1, kde2)
            w2_kde = _geomean_weights(ld2_arr, kde2, kde1)

            # Map KDE weights back to full index; depth-less samples keep w=1
            g1_depth_pos = {idx: pos for pos, idx in enumerate(g1_has_depth)}
            g2_depth_pos = {idx: pos for pos, idx in enumerate(g2_has_depth)}
            for pos, idx in enumerate(g1_all):
                if idx in g1_depth_pos:
                    w1[pos] = w1_kde[g1_depth_pos[idx]]
            for pos, idx in enumerate(g2_all):
                if idx in g2_depth_pos:
                    w2[pos] = w2_kde[g2_depth_pos[idx]]

    ess1 = float(w1.sum()**2 / (w1**2).sum())
    ess2 = float(w2.sum()**2 / (w2**2).sum())
    print(f"    [GMW {out_suffix}] n1={len(g1_all)} (ESS={ess1:.0f}), "
          f"n2={len(g2_all)} (ESS={ess2:.0f})")

    # Weighted prevalence test
    prev_df = test_prevalence_weighted(
        breadth, g1_all, g2_all, w1, w2, g1_label, g2_label)

    if prev_df.empty:
        return prev_df, [], []

    # Save CSV
    csv_path = os.path.join(out, f"prevalence_gmw_{out_suffix}.csv")
    prev_df.to_csv(csv_path, index=False)

    # Top-n lists: tail = most g2-enriched (VANISH-like), head = most g1-enriched (BloSSUM-like)
    n = min(top_n, len(prev_df))
    g2_enriched_ids = prev_df.tail(n)["microbe"].tolist()   # NI-enriched -> VANISH-like
    g1_enriched_ids = prev_df.head(n)["microbe"].tolist()   # Ind-enriched -> BloSSUM-like

    _save_top100(g2_enriched_ids,
                 os.path.join(out, f"top{n}_g2_enriched_gmw_{out_suffix}.txt"))
    _save_top100(g1_enriched_ids,
                 os.path.join(out, f"top{n}_g1_enriched_gmw_{out_suffix}.txt"))

    print(f"    [GMW {out_suffix}] g2-enriched (VANISH-like): {len(g2_enriched_ids)}, "
          f"g1-enriched (BloSSUM-like): {len(g1_enriched_ids)}")

    return prev_df, g1_enriched_ids, g2_enriched_ids
