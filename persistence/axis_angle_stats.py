"""
axis_angle_stats.py — Statistical tests for the extinction-axis angle analysis.

Four tests supporting three claims in the manuscript:

  Claim 1 — Intermediacy (along axis):
    Jonckheere-Terpstra (JT) ordered trend test on p_frac.
    H1: median_HG ≤ median_Trans ≤ median_Ind  (ordered alternative).
    More powerful than KW for this ordered hypothesis.

  Claim 2 — Off-axis displacement:
    Wilcoxon signed-rank on Transitional perp_signed vs 0.
    H1: median > 0 (Trans displaced toward its own side, not on the H-G→Ind line).
    Sanity-checks H-G and Industrial as controls (both expected ≈ 0).

  Claim 3 — Group distinctness in (p, perp) space:
    PERMANOVA on Euclidean distance in 2-D (p, perp_signed) space.
    Planned pairwise contrasts with Bonferroni correction.

  Consistency — Persistent non-collinearity across depth bins:
    One-sample t-test on per-bin theta_deg vs 0.
    H1: mean θ > 0°.

Primary analysis uses per-bin median values (n = n_qualifying_bins per lifestyle),
which are independent across bins (each bin is a separate PCoA embedding).
Per-sample sensitivity analyses are reported separately.

Outputs
-------
results/<ref>/section1_diversity/main/axis_angle_v2_stats_<spec>.csv
results/<ref>/section1_diversity/main/axis_angle_v2_stats_<spec>_fillins.txt

Standalone
----------
    python -m persistence --ref UHGG --only s1_axis_stats
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp
from scipy.spatial.distance import pdist, squareform


LIFESTYLE_ORDER = ["H-G", "Transitional", "Industrial"]
N_PERM_JT      = 10_000
N_PERM_PERM    = 999
MAX_SAMPLE_N   = 600    # per group for per-sample sensitivity check


# ── Jonckheere-Terpstra ordered trend statistic ───────────────────────────────

def _jt_stat(g1: np.ndarray, g2: np.ndarray, g3: np.ndarray) -> float:
    """
    JT = U(g1,g2) + U(g1,g3) + U(g2,g3)
    where U(A,B) = # pairs (a,b) with a < b  (concordant with ordering A<B).
    Vectorised: O(n1·n2 + n1·n3 + n2·n3).
    """
    U12 = float(np.sum(g1[:, None] < g2[None, :]))
    U13 = float(np.sum(g1[:, None] < g3[None, :]))
    U23 = float(np.sum(g2[:, None] < g3[None, :]))
    return U12 + U13 + U23


def _jt_permutation(g1: np.ndarray, g2: np.ndarray, g3: np.ndarray,
                    n_perm: int, rng: np.random.Generator
                    ) -> tuple[float, float, float]:
    """
    Permutation JT test with Cliff's delta effect size.
    Returns (JT_obs, p_value, cliffs_delta_HG_Ind).
    """
    JT_obs = _jt_stat(g1, g2, g3)
    n1, n2, n3 = len(g1), len(g2), len(g3)
    combined = np.concatenate([g1, g2, g3])

    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        null[i] = _jt_stat(perm[:n1], perm[n1:n1+n2], perm[n1+n2:])

    p_val = (np.sum(null >= JT_obs) + 1) / (n_perm + 1)

    # Cliff's delta between extreme groups (H-G and Industrial)
    # delta = (U(g1<g3) - U(g3<g1)) / (n1*n3)
    U_hg_ind = float(np.sum(g1[:, None] < g3[None, :]))
    U_ind_hg = float(np.sum(g3[:, None] < g1[None, :]))
    cliffs_d  = (U_hg_ind - U_ind_hg) / (n1 * n3)

    return JT_obs, p_val, cliffs_d


# ── PERMANOVA ─────────────────────────────────────────────────────────────────

def _permanova(coords: np.ndarray, labels: np.ndarray,
               n_perm: int, rng: np.random.Generator
               ) -> tuple[float, float, float]:
    """
    Permutational MANOVA on Euclidean distance.
    Returns (F_obs, p_value, R2).
    """
    D    = squareform(pdist(coords, "euclidean"))
    n    = len(labels)
    grps = np.unique(labels)
    a    = len(grps)

    def _F(lbs):
        ss_w = sum(
            D[np.ix_(np.where(lbs == g)[0],
                      np.where(lbs == g)[0])].sum()
            / (2 * max(np.sum(lbs == g), 1))
            for g in grps
            if np.sum(lbs == g) >= 2
        )
        ss_t = D.sum() / (2 * n)
        ss_b = ss_t - ss_w
        if ss_w < 1e-15:
            return np.inf
        return (ss_b / (a - 1)) / (ss_w / (n - a))

    F_obs = _F(labels)
    R2    = 1.0 - 1.0 / (1.0 + F_obs * (a - 1) / max(n - a, 1))

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _F(rng.permutation(labels))

    p_val = (np.sum(null >= F_obs) + 1) / (n_perm + 1)
    return F_obs, p_val, R2


def _pairwise_permanova(coords: np.ndarray, labels: np.ndarray,
                        pairs: list, n_perm: int, rng: np.random.Generator,
                        alpha: float = 0.05) -> list[dict]:
    """
    Planned pairwise contrasts with Bonferroni correction.
    pairs: list of (label_a, label_b) tuples.
    """
    results = []
    alpha_adj = alpha / len(pairs)
    for a, b in pairs:
        mask  = np.isin(labels, [a, b])
        c2    = coords[mask]
        lbs2  = labels[mask]
        F, p, r2 = _permanova(c2, lbs2, n_perm, rng)
        results.append(dict(
            contrast=f"{a} vs {b}",
            F=round(F, 3), p=round(p, 4), R2=round(r2, 4),
            bonferroni_alpha=round(alpha_adj, 4),
            significant=(p < alpha_adj),
        ))
    return results


# ── p-value formatter ─────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.3f}"


# ── Core per-spec analysis ────────────────────────────────────────────────────

def _run_stats_for_spec(samples_df: pd.DataFrame,
                        bin_df: pd.DataFrame,
                        spec_tag: str,
                        outdir: str,
                        ref: str,
                        stats,
                        rng_seed: int = 42) -> dict:
    """
    Run all four tests and return a dict of results.
    Primary analysis: per-bin medians (independent across bins).
    Sensitivity: per-sample (subsampled, correlated within bins).
    """
    rng = np.random.default_rng(rng_seed)
    print(f"\n  [{spec_tag}] Statistical tests")

    # ── Step 1: per-bin medians per lifestyle ─────────────────────────────────
    # Aggregate samples to per-bin-per-lifestyle medians.
    # Only include bins that have all three lifestyles represented.
    bin_lf = (
        samples_df
        .groupby(["bin_idx", "Lifestyle"])[["p", "perp_signed"]]
        .median()
        .reset_index()
    )
    # Bins with all three lifestyles
    bin_counts = bin_lf.groupby("bin_idx")["Lifestyle"].nunique()
    complete_bins = bin_counts[bin_counts == 3].index
    bin_lf = bin_lf[bin_lf["bin_idx"].isin(complete_bins)]

    n_bins = len(complete_bins)
    print(f"    {n_bins} complete bins (all three lifestyles present)")

    def _get_bin_vals(lf: str, col: str) -> np.ndarray:
        sub = bin_lf[bin_lf["Lifestyle"] == lf].sort_values("bin_idx")
        return sub[col].values

    p_hg    = _get_bin_vals("H-G",          "p")
    p_tr    = _get_bin_vals("Transitional",  "p")
    p_ind   = _get_bin_vals("Industrial",    "p")
    perp_hg  = _get_bin_vals("H-G",         "perp_signed")
    perp_tr  = _get_bin_vals("Transitional", "perp_signed")
    perp_ind = _get_bin_vals("Industrial",   "perp_signed")

    print(f"\n    Per-bin median p (fractional position):")
    print(f"      H-G:          {np.median(p_hg):.3f}  (IQR {np.percentile(p_hg,25):.3f}–{np.percentile(p_hg,75):.3f})")
    print(f"      Transitional: {np.median(p_tr):.3f}  (IQR {np.percentile(p_tr,25):.3f}–{np.percentile(p_tr,75):.3f})")
    print(f"      Industrial:   {np.median(p_ind):.3f}  (IQR {np.percentile(p_ind,25):.3f}–{np.percentile(p_ind,75):.3f})")
    print(f"\n    Per-bin median perp_signed (off-axis displacement):")
    print(f"      H-G:          {np.median(perp_hg):.3f}")
    print(f"      Transitional: {np.median(perp_tr):.3f}")
    print(f"      Industrial:   {np.median(perp_ind):.3f}")

    # ── Test 1: Jonckheere-Terpstra (intermediacy along axis) ─────────────────
    print(f"\n    Test 1 — Jonckheere-Terpstra (ordered trend: H-G ≤ Trans ≤ Ind)")
    JT_obs, jt_p, cliffs_d = _jt_permutation(
        p_hg, p_tr, p_ind, N_PERM_JT, rng)
    JT_max = len(p_hg) * len(p_tr) + len(p_hg) * len(p_ind) + len(p_tr) * len(p_ind)
    print(f"      JT = {JT_obs:.0f} (max possible = {JT_max:.0f}),  "
          f"{_fmt_p(jt_p)},  n_perm = {N_PERM_JT:,}")
    print(f"      Cliff's δ (H-G vs Industrial) = {cliffs_d:.3f}")

    # ── Test 2: Wilcoxon signed-rank (off-axis displacement ≠ 0) ─────────────
    print(f"\n    Test 2 — Wilcoxon signed-rank vs 0 (off-axis displacement)")
    wx_results = {}
    for lf, perp_vals in [("H-G", perp_hg),
                            ("Transitional", perp_tr),
                            ("Industrial", perp_ind)]:
        side = "greater" if lf == "Transitional" else "two-sided"
        try:
            W, wx_p = wilcoxon(perp_vals, alternative=side)
        except Exception:
            W, wx_p = np.nan, np.nan
        wx_results[lf] = (W, wx_p)
        med = np.median(perp_vals)
        q25 = np.percentile(perp_vals, 25)
        q75 = np.percentile(perp_vals, 75)
        print(f"      {lf:<14}: W = {W:.0f},  {_fmt_p(wx_p)}  "
              f"({'one-sided >' if side=='greater' else 'two-sided'}),  "
              f"median = {med:.3f}  [IQR {q25:.3f}–{q75:.3f}]")

    # ── Test 3: PERMANOVA in (p, perp) space ──────────────────────────────────
    print(f"\n    Test 3 — PERMANOVA on (p, perp) [n_perm = {N_PERM_PERM}]")
    # Stack all three groups
    coords_all = np.vstack([
        np.column_stack([p_hg,  perp_hg]),
        np.column_stack([p_tr,  perp_tr]),
        np.column_stack([p_ind, perp_ind]),
    ])
    labels_all = np.array(
        ["H-G"]          * len(p_hg)  +
        ["Transitional"] * len(p_tr)  +
        ["Industrial"]   * len(p_ind)
    )
    F_obs, perm_p, R2 = _permanova(coords_all, labels_all, N_PERM_PERM, rng)
    print(f"      Overall: F = {F_obs:.3f},  {_fmt_p(perm_p)},  R² = {R2:.4f}")

    pairs = [("H-G", "Transitional"),
             ("Transitional", "Industrial"),
             ("H-G", "Industrial")]
    pw = _pairwise_permanova(coords_all, labels_all, pairs, N_PERM_PERM, rng)
    for r in pw:
        sig_str = "✓" if r["significant"] else "ns"
        print(f"      {r['contrast']:<30}: F = {r['F']:.3f},  "
              f"{_fmt_p(r['p'])},  R² = {r['R2']:.4f}  "
              f"[Bonferroni α = {r['bonferroni_alpha']:.3f}  {sig_str}]")

    # ── Test 4: t-test on per-bin theta (persistent non-collinearity) ─────────
    print(f"\n    Test 4 — One-sample t-test on per-bin θ vs 0°")
    theta_vals = bin_df["theta_deg"].dropna().values
    mean_t  = float(np.mean(theta_vals))
    sd_t    = float(np.std(theta_vals, ddof=1))
    t_stat, t_p = ttest_1samp(theta_vals, 0.0, alternative="greater")
    # Also 95% CI on mean theta
    from scipy.stats import t as t_dist
    se_t = sd_t / np.sqrt(len(theta_vals))
    ci_lo = mean_t - t_dist.ppf(0.975, df=len(theta_vals)-1) * se_t
    ci_hi = mean_t + t_dist.ppf(0.975, df=len(theta_vals)-1) * se_t
    print(f"      n_bins = {len(theta_vals)},  mean θ = {mean_t:.1f}° ± {sd_t:.1f}°  "
          f"(95% CI {ci_lo:.1f}°–{ci_hi:.1f}°)")
    print(f"      t({len(theta_vals)-1}) = {t_stat:.2f},  {_fmt_p(t_p)}  (one-sided > 0°)")

    # ── Sensitivity: per-sample analysis (subsampled) ─────────────────────────
    print(f"\n    Sensitivity — per-sample analysis "
          f"(subsampled to ≤{MAX_SAMPLE_N} per group)")
    sens = {}
    for lf in LIFESTYLE_ORDER:
        sub = samples_df[samples_df["Lifestyle"] == lf]
        ok  = np.isfinite(sub["p"]) & np.isfinite(sub["perp_signed"])
        sub = sub[ok]
        n_g = min(len(sub), MAX_SAMPLE_N)
        if len(sub) > MAX_SAMPLE_N:
            sub = sub.sample(n=MAX_SAMPLE_N, random_state=42)
        sens[lf] = sub[["p", "perp_signed"]].values
        print(f"      {lf}: {len(sub):,} samples used "
              f"(of {ok.sum():,} available)")

    s_p_hg  = sens["H-G"][:, 0]
    s_p_tr  = sens["Transitional"][:, 0]
    s_p_ind = sens["Industrial"][:, 0]
    _, jt_p_sens, cliffs_d_sens = _jt_permutation(
        s_p_hg, s_p_tr, s_p_ind,
        min(N_PERM_JT, 2000), rng)
    print(f"      JT (per-sample): {_fmt_p(jt_p_sens)},  "
          f"Cliff's δ = {cliffs_d_sens:.3f}")

    s_perp_tr = sens["Transitional"][:, 1]
    try:
        _, wx_p_sens = wilcoxon(s_perp_tr, alternative="greater")
    except Exception:
        wx_p_sens = np.nan
    print(f"      Wilcoxon Trans perp (per-sample): {_fmt_p(wx_p_sens)}")

    # ── Assemble result dict ──────────────────────────────────────────────────
    W_tr, wx_p_tr = wx_results["Transitional"]
    W_hg, wx_p_hg = wx_results["H-G"]
    W_ind, wx_p_ind = wx_results["Industrial"]
    result = dict(
        spec_tag=spec_tag,
        n_bins=n_bins,
        # T1 — JT
        JT_obs=round(JT_obs, 1), JT_max=JT_max,
        jt_p=round(jt_p, 5), cliffs_d=round(cliffs_d, 4),
        # T2 — Wilcoxon
        W_hg=round(W_hg, 1), wx_p_hg=round(wx_p_hg, 5),
        W_trans=round(W_tr, 1), wx_p_trans=round(wx_p_tr, 5),
        W_ind=round(W_ind, 1), wx_p_ind=round(wx_p_ind, 5),
        perp_trans_median=round(float(np.median(perp_tr)), 4),
        perp_trans_q25=round(float(np.percentile(perp_tr, 25)), 4),
        perp_trans_q75=round(float(np.percentile(perp_tr, 75)), 4),
        # T3 — PERMANOVA
        permanova_F=round(F_obs, 3), permanova_p=round(perm_p, 4),
        permanova_R2=round(R2, 4),
        **{f"pw_{r['contrast'].replace(' vs ','_vs_')}_F": r["F"] for r in pw},
        **{f"pw_{r['contrast'].replace(' vs ','_vs_')}_p": r["p"] for r in pw},
        **{f"pw_{r['contrast'].replace(' vs ','_vs_')}_R2": r["R2"] for r in pw},
        **{f"pw_{r['contrast'].replace(' vs ','_vs_')}_sig": r["significant"] for r in pw},
        # T4 — t-test on theta
        theta_mean=round(mean_t, 2), theta_sd=round(sd_t, 2),
        theta_ci_lo=round(ci_lo, 2), theta_ci_hi=round(ci_hi, 2),
        t_stat=round(t_stat, 3), t_p=round(t_p, 5),
        t_df=len(theta_vals)-1,
        # Medians (for reporting)
        p_hg_median=round(float(np.median(p_hg)), 3),
        p_trans_median=round(float(np.median(p_tr)), 3),
        p_ind_median=round(float(np.median(p_ind)), 3),
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(outdir, f"axis_angle_v2_stats_{spec_tag}.csv")
    pd.DataFrame([result]).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Manuscript fill-ins ───────────────────────────────────────────────────
    pw_hg_tr  = next(r for r in pw if "H-G" in r["contrast"] and "Trans" in r["contrast"])
    pw_tr_ind = next(r for r in pw if "Trans" in r["contrast"] and "Ind" in r["contrast"])
    pw_hg_ind = next(r for r in pw if "H-G" in r["contrast"] and "Ind" in r["contrast"])

    fillin_lines = [
        f"=== Manuscript fill-ins: axis-angle stats [{ref} / {spec_tag}] ===",
        "",
        "── Overview sentence ──────────────────────────────────────────────────",
        f"Industrial and non-industrial microbiota were compositionally distinct",
        f"(PERMANOVA: F = {F_obs:.2f}, {_fmt_p(perm_p)}, R² = {R2:.3f}), but",
        f"transitional populations were not simply intermediate along a single",
        f"linear gradient: when projected into a coordinate-invariant H-G→Industrial",
        f"axis space, transitional samples lay partway along the main contrast",
        f"(median fractional position = {np.median(p_tr):.2f} [IQR",
        f"{np.percentile(p_tr,25):.2f}–{np.percentile(p_tr,75):.2f}]; ordered",
        f"Jonckheere-Terpstra: JT = {JT_obs:.0f}, {_fmt_p(jt_p)},",
        f"Cliff's δ(H-G, Industrial) = {cliffs_d:.2f}) while remaining",
        f"systematically displaced off-axis (Wilcoxon signed-rank vs 0:",
        f"W = {W_tr:.0f}, {_fmt_p(wx_p_tr)}, median perpendicular displacement",
        f"= {np.median(perp_tr):.2f} [IQR {np.percentile(perp_tr,25):.2f}–{np.percentile(perp_tr,75):.2f}]),",
        f"consistent with a distinct agriculturalization component of microbiome",
        f"change preceding the later shift toward industrial composition",
        f"(mean angle between trajectories = {mean_t:.1f}° ± {sd_t:.1f}°,",
        f"95% CI {ci_lo:.1f}°–{ci_hi:.1f}°; t({len(theta_vals)-1}) = {t_stat:.2f},",
        f"{_fmt_p(t_p)}).",
        "",
        "── Individual test values ─────────────────────────────────────────────",
        "",
        f"Test 1 — JT (intermediacy along axis; per-bin medians, n = {n_bins} bins):",
        f"  JT = {JT_obs:.0f}  (max = {JT_max:.0f})",
        f"  {_fmt_p(jt_p)}  (permutation, n_perm = {N_PERM_JT:,})",
        f"  Cliff's δ (H-G vs Industrial) = {cliffs_d:.3f}",
        f"  Per-bin median p:  H-G = {np.median(p_hg):.3f},  "
        f"Trans = {np.median(p_tr):.3f},  Ind = {np.median(p_ind):.3f}",
        "",
        f"Test 2 — Wilcoxon signed-rank vs 0 (off-axis displacement; per-bin medians):",
        f"  Transitional:  W = {W_tr:.0f},  {_fmt_p(wx_p_tr)}  (one-sided >0)",
        f"    median perp = {np.median(perp_tr):.3f}  [IQR {np.percentile(perp_tr,25):.3f}–{np.percentile(perp_tr,75):.3f}]",
        f"  H-G (control): W = {W_hg:.0f},  {_fmt_p(wx_p_hg)}  (two-sided)",
        f"    median perp = {np.median(perp_hg):.3f}",
        f"  Industrial (control): W = {W_ind:.0f},  {_fmt_p(wx_p_ind)}  (two-sided)",
        f"    median perp = {np.median(perp_ind):.3f}",
        "",
        f"Test 3 — PERMANOVA (per-bin medians, n_perm = {N_PERM_PERM}):",
        f"  Overall:  F = {F_obs:.3f},  {_fmt_p(perm_p)},  R² = {R2:.4f}",
        f"  H-G vs Transitional:   F = {pw_hg_tr['F']:.3f},  "
        f"{_fmt_p(pw_hg_tr['p'])},  R² = {pw_hg_tr['R2']:.4f}  "
        f"[{'✓ sig' if pw_hg_tr['significant'] else 'ns'} at Bonferroni α = {pw_hg_tr['bonferroni_alpha']:.3f}]",
        f"  Transitional vs Industrial: F = {pw_tr_ind['F']:.3f},  "
        f"{_fmt_p(pw_tr_ind['p'])},  R² = {pw_tr_ind['R2']:.4f}  "
        f"[{'✓ sig' if pw_tr_ind['significant'] else 'ns'} at Bonferroni α = {pw_tr_ind['bonferroni_alpha']:.3f}]",
        f"  H-G vs Industrial:     F = {pw_hg_ind['F']:.3f},  "
        f"{_fmt_p(pw_hg_ind['p'])},  R² = {pw_hg_ind['R2']:.4f}  "
        f"[{'✓ sig' if pw_hg_ind['significant'] else 'ns'} at Bonferroni α = {pw_hg_ind['bonferroni_alpha']:.3f}]",
        "",
        f"Test 4 — one-sample t-test on per-bin θ (non-collinearity; n = {len(theta_vals)}):",
        f"  Mean θ = {mean_t:.1f}° ± {sd_t:.1f}° SD  (95% CI {ci_lo:.1f}°–{ci_hi:.1f}°)",
        f"  t({len(theta_vals)-1}) = {t_stat:.3f},  {_fmt_p(t_p)}  (one-sided > 0°)",
        "",
        "── Sensitivity (per-sample, subsampled) ───────────────────────────────",
        f"  JT per-sample:  {_fmt_p(jt_p_sens)},  Cliff's δ = {cliffs_d_sens:.3f}",
        f"  Wilcoxon Trans perp per-sample:  {_fmt_p(wx_p_sens)}",
    ]

    txt_path = os.path.join(outdir, f"axis_angle_v2_stats_{spec_tag}_fillins.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(fillin_lines) + "\n")
    print(f"  Saved: {txt_path}")
    print()
    print("\n".join(fillin_lines[:35]))  # print overview to console

    # ── Stats report ─────────────────────────────────────────────────────────
    if stats is not None:
        stats.add(f"axis_stats_JT_{spec_tag}", "s1", "main",
                  f"JT ordered trend: H-G≤Trans≤Ind on p_frac [{spec_tag}]",
                  f"JT={JT_obs:.0f}/{JT_max}, {_fmt_p(jt_p)}, "
                  f"Cliff's delta={cliffs_d:.3f}, n_bins={n_bins}",
                  p_value=jt_p, n1=n_bins)
        stats.add(f"axis_stats_wilcox_trans_{spec_tag}", "s1", "main",
                  f"Wilcoxon Trans perp vs 0 [{spec_tag}]",
                  f"W={W_tr:.0f}, {_fmt_p(wx_p_tr)}, "
                  f"median perp={np.median(perp_tr):.3f}, n_bins={n_bins}",
                  p_value=wx_p_tr, n1=n_bins)
        stats.add(f"axis_stats_permanova_{spec_tag}", "s1", "main",
                  f"PERMANOVA on (p, perp) [{spec_tag}]",
                  f"F={F_obs:.3f}, {_fmt_p(perm_p)}, R2={R2:.4f}, n={len(labels_all)}",
                  p_value=perm_p, effect_size=F_obs, n1=len(labels_all))
        stats.add(f"axis_stats_theta_t_{spec_tag}", "s1", "main",
                  f"t-test on per-bin theta vs 0 [{spec_tag}]",
                  f"mean={mean_t:.1f}°±{sd_t:.1f}°, CI [{ci_lo:.1f}°,{ci_hi:.1f}°], "
                  f"t({len(theta_vals)-1})={t_stat:.3f}, {_fmt_p(t_p)}",
                  p_value=t_p, n1=len(theta_vals))

    return result


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_axis_angle_stats(data, stats, specs=None):
    """Called from __main__.py under --only s1_axis_stats."""
    from . import config

    ref    = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")

    if specs is None:
        from .detection import DetectionSpec
        specs = [
            DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    print(f"\n{'='*60}")
    print(f"  Extinction-axis statistical tests [{ref}]")
    print(f"{'='*60}")

    all_results = []
    for spec in specs:
        spec_tag   = spec.short_label
        smp_path   = os.path.join(outdir, f"axis_angle_v2_samples_{spec_tag}.csv")
        bin_path   = os.path.join(outdir, f"axis_angle_v2_{spec_tag}.csv")

        if not os.path.exists(smp_path):
            print(f"\n  [{spec_tag}] Samples CSV not found: {smp_path}")
            print(f"  Run --only s1_axis_angle first, then re-run --only s1_axis_stats")
            continue
        if not os.path.exists(bin_path):
            print(f"\n  [{spec_tag}] Bin CSV not found: {bin_path}")
            continue

        samples_df = pd.read_csv(smp_path)
        bin_df     = pd.read_csv(bin_path)
        print(f"\n  [{spec_tag}] Loaded {len(samples_df):,} sample rows, "
              f"{len(bin_df):,} bin rows")

        result = _run_stats_for_spec(
            samples_df, bin_df, spec_tag, outdir, ref, stats)
        all_results.append(result)

    if len(all_results) > 1:
        # Combined CSV
        combined_path = os.path.join(outdir, "axis_angle_v2_stats_combined.csv")
        pd.DataFrame(all_results).to_csv(combined_path, index=False)
        print(f"\n  Saved combined: {combined_path}")
