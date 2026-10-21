"""
Section 1 analyses using anchored 1-to-K depth-matched sample sets.

This module adds depth-controlled diversity comparisons anchored on the
smallest group:

  A. **H-G-anchored** (3-way):
       - Alpha diversity: H-G vs Transitional vs Industrial
       - Beta diversity: PCoA + PERMANOVA
       - Depth-matching QC plots

  B. **NI-anchored** (2-way):
       - Alpha diversity: Non-Industrial vs Industrial
       - Beta diversity: PCoA + PERMANOVA
       - Depth-matching QC plots

All analyses save traces to:
    results/{ref}/section1_diversity/traces/anchored/

Figures go to:
    results/{ref}/section1_diversity/main/
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import apply_oi_style, savefig_dual, current_fs
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, kruskal

from . import config
from .detection import DetectionSpec, build_presence


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

SPECS = [
    DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
    DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
]

N_PCOA_AXES = config.DIVERSITY_PCOA_AXES
N_PERMANOVA = config.DIVERSITY_N_PERMANOVA

COLORS = {
    "H-G":            config.LIFESTYLE_COLORS["H-G"],
    "Transitional":   config.LIFESTYLE_COLORS["Transitional"],
    "Industrial":     config.LIFESTYLE_COLORS["Industrial"],
    "Non-Industrial": config.OI["orange"],
}


# ╔══════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _shannon(presence_row, abundance_row):
    """Shannon H' from abundance of detected SGBs, renormalised to them.

    WARNING -- there are two Shannon definitions in Section 1 and they are not
    interchangeable. This one is detection-filtered: it restricts to the SGBs
    that pass the DetectionSpec, renormalises their abundances to sum to 1, and
    is therefore spec-dependent and consistent with the `richness` written
    beside it by _compute_richness_shannon. It is what lands in the per-sample
    table `<trace_dir>/alpha/<tag>.csv`.

    `_plot_anchored_alpha` draws a *different* quantity under the same label:
    -(a * log a) summed over ALL catalogue SGBs, with no detection filter and
    no renormalisation (data.abundance already sums to 1 across the catalogue).
    section1_diversity.compute_alpha_matched_pair uses that all-SGB form too.

    Measured on the 2026-09-29 combined run (not re-derived -- taken from the
    run's own outputs: the all-SGB values recovered from the cached figures in
    section1_diversity/main/.figcache/alpha_anchored_*.pkl, against the
    detection-filtered values in traces/anchored/alpha/*.csv for the same
    samples): the all-SGB form is higher by +0.47 to +0.62 nats, i.e. +10.5% to
    +16.8% of the detection-filtered value, per group. Per sample over the
    5,807 samples common to the anchored traces and
    alpha_diversity_matched_Industrial_vs_Non-Industrial_10pct.csv: mean
    +0.583 nats (+16.4%), median +0.544 (+14.4%), range +0.12 to +3.29. The gap
    is driven by the ~2,800 SGBs per sample that carry non-zero mapped depth
    but fail detection; `richness` agrees exactly on all 5,807, so the detection
    call is identical and only the Shannon formula differs.

    Consequence to know before quoting either: the anchored alpha PNG cannot be
    reproduced from the CSV the pipeline writes beside it, and the median-ratio
    annotation on its Shannon panel moves -- e.g. Industrial/H-G reads 0.863 as
    drawn and 0.821 under this definition (hg_anchored, 10pct). No Shannon
    number is registered by this module (only richness, Kruskal and PERMANOVA)
    and none reaches the manuscript text, so nothing published turns on the
    choice today. Do not unify the two without re-running Section 1: see
    the 2026-10-07 audit.
    """
    det = presence_row.astype(bool)
    a = abundance_row[det].values.astype(float)
    a = a[a > 0]
    if len(a) == 0:
        return 0.0
    p = a / a.sum()
    return float(-np.sum(p * np.log(p)))


def _compute_richness_shannon(data, ids, spec):
    """Return DataFrame with richness + Shannon for sample IDs."""
    presence = build_presence(
        data.breadth.loc[ids], data.depth.loc[ids],
        data.lengths, spec=spec)
    richness = presence.sum(axis=1)

    abund = data.abundance.loc[ids]
    shannon = pd.Series(
        [_shannon(presence.loc[s], abund.loc[s]) for s in ids],
        index=ids, name="Shannon")

    df = pd.DataFrame({
        "sample_id": ids,
        "richness": richness.loc[ids].values,
        "shannon": shannon.values,
        "total_bp": data.total_bp.loc[ids].values,
        "log10_bp": np.log10(data.total_bp.loc[ids].values.astype(float)),
        "Lifestyle": data.meta.loc[ids, "Lifestyle"].values,
    })
    if "Unhealthy" in data.meta.columns:
        df["Health"] = data.meta.loc[ids, "Unhealthy"].values
    df.index = ids
    return df


def _pcoa(D_sq, n_axes=3):
    """Classical PCoA from a square distance matrix."""
    D2 = D_sq ** 2
    n = D2.shape[0]
    row_m = D2.mean(axis=1)
    col_m = D2.mean(axis=0)
    B = -0.5 * (D2 - row_m[:, None] - col_m[None, :] + D2.mean())

    try:
        from scipy.linalg import eigh
        evals, evecs = eigh(B, subset_by_index=[n - n_axes, n - 1])
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
    except (ImportError, TypeError):
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx[:n_axes]]
        evecs = evecs[:, idx[:n_axes]]

    total = max(np.trace(B), 1e-10)
    var_pct = np.array([max(ev, 0) / total * 100 for ev in evals])
    coords = evecs * np.sqrt(np.maximum(evals, 0))
    return coords, var_pct


def _permanova(D_sq, labels, n_perm=999):
    """PERMANOVA F-stat and permutation p-value."""
    rng = np.random.default_rng(42)
    n = D_sq.shape[0]
    groups = np.unique(labels)
    a = len(groups)

    def _ss(D, labs):
        ss_w = 0
        for g in groups:
            idx = np.where(labs == g)[0]
            if len(idx) < 2:
                continue
            ss_w += np.sum(D[np.ix_(idx, idx)] ** 2) / (2 * len(idx))
        ss_t = np.sum(D ** 2) / (2 * n)
        return ss_t - ss_w, ss_w

    ss_a, ss_w = _ss(D_sq, labels)
    if ss_w == 0:
        return np.inf, 1 / (n_perm + 1)
    F_obs = (ss_a / max(a - 1, 1)) / (ss_w / max(n - a, 1))

    n_ge = 1
    for _ in range(n_perm):
        plab = rng.permutation(labels)
        sa, sw = _ss(D_sq, plab)
        if sw > 0:
            Fp = (sa / max(a - 1, 1)) / (sw / max(n - a, 1))
        else:
            Fp = 0
        if Fp >= F_obs:
            n_ge += 1
    return F_obs, n_ge / (n_perm + 1)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TRACE EXPORT: MATCHING METADATA                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def _save_anchored_matching_metadata(data, am, trace_dir):
    """Save anchored matching pair/group IDs and depths."""
    match_dir = _ensure(os.path.join(trace_dir, "matching"))
    total_bp = data.total_bp

    # ── H-G-anchored ──
    if am.hg_anchor_ids:
        rows = []
        for i, hg_id in enumerate(am.hg_anchor_ids):
            hg_bp = float(total_bp[hg_id])
            trans_ids = am.hg_anchor_to_trans.get(hg_id, [])
            ind_ids = am.hg_anchor_to_ind.get(hg_id, [])
            for j, tid in enumerate(trans_ids):
                t_bp = float(total_bp[tid])
                rows.append({
                    "anchor_idx": i, "hg_id": hg_id, "hg_bp": hg_bp,
                    "ref_group": "Transitional", "ref_idx": j,
                    "ref_id": tid, "ref_bp": t_bp,
                    "frac_diff": abs(hg_bp - t_bp) / max(hg_bp, t_bp),
                })
            for j, iid in enumerate(ind_ids):
                i_bp = float(total_bp[iid])
                rows.append({
                    "anchor_idx": i, "hg_id": hg_id, "hg_bp": hg_bp,
                    "ref_group": "Industrial", "ref_idx": j,
                    "ref_id": iid, "ref_bp": i_bp,
                    "frac_diff": abs(hg_bp - i_bp) / max(hg_bp, i_bp),
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(match_dir, "hg_anchored_pairs.csv"),
                  index=False)
        print(f"      H-G anchored: {len(am.hg_anchor_ids)} anchors, "
              f"{len(rows)} total pairs")

    # ── NI-anchored ──
    if am.ni_anchor_ids:
        rows = []
        for i, ni_id in enumerate(am.ni_anchor_ids):
            ni_bp = float(total_bp[ni_id])
            ind_ids = am.ni_anchor_to_ind.get(ni_id, [])
            for j, iid in enumerate(ind_ids):
                i_bp = float(total_bp[iid])
                rows.append({
                    "anchor_idx": i, "ni_id": ni_id, "ni_bp": ni_bp,
                    "ni_lifestyle": data.meta.loc[ni_id, "Lifestyle"],
                    "ref_idx": j, "ind_id": iid, "ind_bp": i_bp,
                    "frac_diff": abs(ni_bp - i_bp) / max(ni_bp, i_bp),
                })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(match_dir, "ni_anchored_pairs.csv"),
                  index=False)
        print(f"      NI anchored: {len(am.ni_anchor_ids)} anchors, "
              f"{len(rows)} total pairs")

    # Summary
    summary_rows = []
    if am.hg_stats:
        summary_rows.append({"analysis": "hg_anchored", **{
            k: v for k, v in am.hg_stats.items()
            if not isinstance(v, dict)
        }})
    if am.ni_stats:
        summary_rows.append({"analysis": "ni_anchored", **{
            k: v for k, v in am.ni_stats.items()
            if not isinstance(v, dict)
        }})
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(match_dir, "anchored_matching_summary.csv"),
            index=False)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TRACE EXPORT: ALPHA + BETA DIVERSITY                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def _run_anchored_alpha_beta(
    data, group_ids_dict, comparison_name, spec,
    trace_dir, stats, permanova_rows,
):
    """
    Compute alpha + beta diversity for a multi-group anchored comparison.

    Parameters
    ----------
    group_ids_dict : dict  group_label -> list of sample IDs
    """
    spec_tag = spec.short_label
    tag = f"{comparison_name}_{spec_tag}"

    alpha_dir = _ensure(os.path.join(trace_dir, "alpha"))
    beta_dir = _ensure(os.path.join(trace_dir, "beta"))

    all_ids = []
    all_labels = []
    for label, ids in group_ids_dict.items():
        all_ids.extend(ids)
        all_labels.extend([label] * len(ids))

    n_total = len(all_ids)
    group_sizes = {label: len(ids) for label, ids in group_ids_dict.items()}
    print(f"      [{tag}] " + ", ".join(
        f"{l}={n:,}" for l, n in group_sizes.items()))

    # ── Alpha ──
    frames = []
    for label, ids in group_ids_dict.items():
        df = _compute_richness_shannon(data, ids, spec)
        df["group"] = label
        frames.append(df)
    alpha_df = pd.concat(frames, ignore_index=True)
    alpha_df["comparison"] = comparison_name
    alpha_df["detection"] = spec.description
    alpha_df.to_csv(os.path.join(alpha_dir, f"{tag}.csv"), index=False)

    # Pairwise Mann-Whitney tests between groups
    group_labels = list(group_ids_dict.keys())
    for i in range(len(group_labels)):
        for j in range(i + 1, len(group_labels)):
            l1, l2 = group_labels[i], group_labels[j]
            r1 = alpha_df[alpha_df["group"] == l1]["richness"].values
            r2 = alpha_df[alpha_df["group"] == l2]["richness"].values
            med1, med2 = np.median(r1), np.median(r2)
            ratio = med1 / max(med2, 1e-10)
            _, p = mannwhitneyu(r1, r2, alternative="two-sided")

            stats.add(
                f"anchored_{comparison_name}_{l1}_vs_{l2}_{spec_tag}",
                "s1", "main",
                f"Anchored richness: {l1} vs {l2} ({spec.description})",
                f"median {l1}={med1:.0f}, {l2}={med2:.0f}, ratio={ratio:.2f}x",
                test="Mann-Whitney U", p_value=p,
                n1=len(r1), n2=len(r2))
            print(f"        {l1} vs {l2}: ratio={ratio:.2f}x, p={p:.2e}")

    # Kruskal-Wallis if >2 groups
    if len(group_labels) > 2:
        groups_rich = [alpha_df[alpha_df["group"] == l]["richness"].values
                       for l in group_labels]
        H, p_kw = kruskal(*groups_rich)
        stats.add(f"anchored_{comparison_name}_kruskal_{spec_tag}",
                  "s1", "main",
                  f"Anchored Kruskal-Wallis ({comparison_name}, {spec.description})",
                  f"H={H:.2f}", test="Kruskal-Wallis", p_value=p_kw,
                  n1=n_total)
        print(f"        Kruskal-Wallis: H={H:.2f}, p={p_kw:.2e}")

    # ── Beta (Bray-Curtis PCoA + PERMANOVA) ──
    abund = data.abundance.loc[all_ids]
    vals = np.nan_to_num(abund.values, nan=0.0)

    t0 = time.time()
    D = pdist(vals, metric="braycurtis")
    D = np.nan_to_num(D, nan=0.0)
    D_sq = squareform(D)
    print(f"        Bray-Curtis: {time.time() - t0:.1f}s")

    coords, var_pct = _pcoa(D_sq, N_PCOA_AXES)

    pcoa_df = pd.DataFrame(
        coords, index=all_ids,
        columns=[f"PCo{i+1}" for i in range(N_PCOA_AXES)])
    pcoa_df["sample_id"] = all_ids
    pcoa_df["group"] = all_labels
    pcoa_df["Lifestyle"] = data.meta.loc[all_ids, "Lifestyle"].values
    pcoa_df["total_bp"] = data.total_bp.loc[all_ids].values
    pcoa_df["comparison"] = comparison_name
    pcoa_df["detection"] = spec.description
    for i in range(min(N_PCOA_AXES, len(var_pct))):
        pcoa_df[f"var_pct_PCo{i+1}"] = var_pct[i]
    pcoa_df.to_csv(os.path.join(beta_dir, f"{tag}_pcoa.csv"), index=False)

    print(f"        PCoA: {var_pct[0]:.1f}% / {var_pct[1]:.1f}%")

    # PERMANOVA (overall)
    lab_arr = np.array(all_labels)
    F_obs, p_perm = _permanova(D_sq, lab_arr, N_PERMANOVA)
    print(f"        PERMANOVA: F={F_obs:.2f}, p={p_perm:.4f}")

    stats.add(f"anchored_permanova_{tag}", "s1", "main",
              f"PERMANOVA: {comparison_name} ({spec.description})",
              f"F={F_obs:.2f}",
              test="PERMANOVA", p_value=p_perm, n1=n_total)

    permanova_rows.append(dict(
        comparison=comparison_name, detection=spec.description,
        label_a="anchored", label_b=f"{len(group_labels)}-group",
        n_a=n_total, n_b=len(group_labels),
        F=F_obs, p=p_perm,
        PCo1_var=var_pct[0], PCo2_var=var_pct[1]))

    # Pairwise PERMANOVAs
    for i in range(len(group_labels)):
        for j in range(i + 1, len(group_labels)):
            l1, l2 = group_labels[i], group_labels[j]
            ids1 = group_ids_dict[l1]
            ids2 = group_ids_dict[l2]
            sub_ids = list(ids1) + list(ids2)
            sub_idx = [all_ids.index(s) for s in sub_ids]
            D_sub = D_sq[np.ix_(sub_idx, sub_idx)]
            lab_sub = np.array([l1] * len(ids1) + [l2] * len(ids2))
            Fp, pp = _permanova(D_sub, lab_sub, N_PERMANOVA)
            print(f"        PERMANOVA {l1} vs {l2}: F={Fp:.2f}, p={pp:.4f}")

            permanova_rows.append(dict(
                comparison=f"{comparison_name}_{l1}_vs_{l2}",
                detection=spec.description,
                label_a=l1, label_b=l2,
                n_a=len(ids1), n_b=len(ids2),
                F=Fp, p=pp,
                PCo1_var=np.nan, PCo2_var=np.nan))

    return alpha_df, pcoa_df


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PLOTTING: DEPTH-MATCHING QC                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

def _plot_anchored_depth_qc(data, am, outdir):
    """
    QC plots showing depth distributions of the anchored matched sets.
    """
    meta = data.meta
    total_bp = data.total_bp

    # ── H-G-anchored ──
    if am.hg_anchor_ids:
        n_groups = 3
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5),
                                  sharey=True)

        groups = [
            (f"{config.LIFESTYLE_DISPLAY.get('H-G', 'H-G')} (anchor)",
             am.hg_anchor_ids, COLORS["H-G"]),
            ("Transitional (matched)", am.hg_trans_ids, COLORS["Transitional"]),
            ("Industrial (matched)", am.hg_ind_ids, COLORS["Industrial"]),
        ]

        all_log = np.log10(total_bp[
            am.hg_anchor_ids + am.hg_trans_ids + am.hg_ind_ids
        ].values.astype(float))
        all_log = all_log[np.isfinite(all_log)]
        bins = np.linspace(np.percentile(all_log, 1),
                          np.percentile(all_log, 99), 35)

        for ax, (label, ids, color) in zip(axes, groups):
            log_bp = np.log10(total_bp[ids].values.astype(float))
            log_bp = log_bp[np.isfinite(log_bp)]
            ax.hist(log_bp, bins=bins, alpha=0.7, color=color,
                    density=True, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("log₁₀(total mapped bp)")
            ax.set_title(f"{label}\n(n={len(ids):,})")
            ax.axvline(np.median(log_bp), color="black", ls="--",
                       alpha=0.7, label=f"median={np.median(log_bp):.2f}")
            ax.legend(fontsize=FS-2)
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel("Density")
        fig.suptitle(
            f"H-G-anchored depth matching QC\n"
            f"{len(am.hg_anchor_ids)} H-G x "
            f"K_Trans={am.hg_k_trans} x K_Ind={am.hg_k_ind}  "
            f"(frac_tol={am.hg_stats.get('frac_tol', '?'):.0%})",
            fontsize=FS+2, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(outdir, "depth_qc_hg_anchored.png")
        savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fpath}")

    # ── NI-anchored ──
    if am.ni_anchor_ids:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        groups = [
            ("Non-Industrial (anchor)", am.ni_anchor_ids, COLORS["Non-Industrial"]),
            ("Industrial (matched)", am.ni_ind_ids, COLORS["Industrial"]),
        ]

        all_log = np.log10(total_bp[
            am.ni_anchor_ids + am.ni_ind_ids
        ].values.astype(float))
        all_log = all_log[np.isfinite(all_log)]
        bins = np.linspace(np.percentile(all_log, 1),
                          np.percentile(all_log, 99), 35)

        for ax, (label, ids, color) in zip(axes, groups):
            log_bp = np.log10(total_bp[ids].values.astype(float))
            log_bp = log_bp[np.isfinite(log_bp)]
            ax.hist(log_bp, bins=bins, alpha=0.7, color=color,
                    density=True, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("log₁₀(total mapped bp)")
            ax.set_title(f"{label}\n(n={len(ids):,})")
            ax.axvline(np.median(log_bp), color="black", ls="--",
                       alpha=0.7, label=f"median={np.median(log_bp):.2f}")
            ax.legend(fontsize=FS-2)
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel("Density")
        fig.suptitle(
            f"NI-anchored depth matching QC\n"
            f"{len(am.ni_anchor_ids)} NI x K_Ind={am.ni_k_ind}  "
            f"(frac_tol={am.ni_stats.get('frac_tol', '?'):.0%})",
            fontsize=FS+2, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(outdir, "depth_qc_ni_anchored.png")
        savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fpath}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PLOTTING: ALPHA DIVERSITY BOXPLOTS                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def _plot_anchored_alpha(data, group_ids_dict, comparison_name, spec, outdir):
    """
    Box + strip plot of richness and Shannon for an anchored comparison.
    """
    spec_tag = spec.short_label
    group_labels = list(group_ids_dict.keys())
    n_groups = len(group_labels)

    # Compute richness/Shannon for all samples
    all_ids = []
    for ids in group_ids_dict.values():
        all_ids.extend(ids)

    presence = build_presence(
        data.breadth.loc[all_ids], data.depth.loc[all_ids],
        data.lengths, spec=spec)
    richness = presence.sum(axis=1)

    # NOT the same Shannon as the per-sample table this figure is published
    # beside. This is the all-SGB form: no detection filter, no renormalisation,
    # so `spec` constrains `richness` above but not this line -- the plotted
    # Shannon is identical at 10pct and 300kb, while the CSV's is not. It runs
    # +0.47 to +0.62 nats (+10.5% to +16.8%) above the detection-filtered
    # `_shannon()` that _compute_richness_shannon writes to alpha/<tag>.csv; see
    # _shannon's docstring for the full measurement and the decision it needs.
    # Left as-is deliberately. alpha_anchored_*.png is in none of the figure
    # manifests, so no numbered figure is at stake, but switching the formula
    # moves a drawn panel and its ratio annotations, and section1_diversity.py
    # would have to move with it; that is the author's call, not a drive-by fix.
    abund = data.abundance.loc[all_ids]
    abund_pos = abund.clip(lower=1e-10)
    shannon = -(abund_pos * np.log(abund_pos)).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(5 + 2 * n_groups, 6))
    rng = np.random.default_rng(42)

    for ax, metric, series in [
        (axes[0], "Richness", richness),
        (axes[1], "Shannon", shannon),
    ]:
        data_arrays = []
        labels = []
        colors = []
        for label, ids in group_ids_dict.items():
            vals = series.loc[ids].values
            data_arrays.append(vals)
            labels.append(f"{label}\n(n={len(ids):,})")
            colors.append(COLORS.get(label, "gray"))

        bplot = ax.boxplot(data_arrays, patch_artist=True, widths=0.5,
                           showfliers=False)
        for patch, c in zip(bplot["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        for i, (vals, c) in enumerate(zip(data_arrays, colors), 1):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       c=c, alpha=0.06, s=3, zorder=1, edgecolors="none")

        ax.set_xticklabels(labels, fontsize=FS-1)
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=FS+1)
        ax.grid(axis="y", alpha=0.3)

        # Annotate median ratios relative to first group
        medians = [np.median(v) for v in data_arrays]
        ref_med = medians[0]
        for i in range(1, len(medians)):
            ratio = medians[i] / max(ref_med, 1e-10)
            ax.text(i + 1, medians[i], f" {ratio:.2f}x",
                    fontsize=FS-2, va="bottom", ha="left", color=colors[i])

    fig.suptitle(
        f"{comparison_name} — anchored depth-matched\n"
        f"({spec.description})",
        fontsize=FS+2, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(
        outdir,
        f"alpha_anchored_{comparison_name}_{spec_tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PLOTTING: PCoA                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

def _plot_anchored_pcoa(pcoa_df, comparison_name, spec, outdir):
    """PCoA scatter (PC1 vs PC2) colored by group."""
    spec_tag = spec.short_label

    fig, ax = plt.subplots(figsize=(8, 7))

    groups = pcoa_df["group"].unique()
    for group in groups:
        mask = pcoa_df["group"] == group
        color = COLORS.get(group, "gray")
        n = mask.sum()
        ax.scatter(
            pcoa_df.loc[mask, "PCo1"],
            pcoa_df.loc[mask, "PCo2"],
            c=color, alpha=0.3, s=10, label=f"{group} (n={n:,})",
            edgecolors="none")

    v1 = pcoa_df["var_pct_PCo1"].iloc[0]
    v2 = pcoa_df["var_pct_PCo2"].iloc[0]
    ax.set_xlabel(f"PCo1 ({v1:.1f}%)")
    ax.set_ylabel(f"PCo2 ({v2:.1f}%)")
    ax.set_title(f"{comparison_name} — PCoA (anchored depth-matched)\n"
                 f"({spec.description})")
    ax.legend(fontsize=FS-1, markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fpath = os.path.join(
        outdir,
        f"pcoa_anchored_{comparison_name}_{spec_tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PLOTTING: PAIRED DEPTH SCATTER (anchor vs matched)             ║
# ╚══════════════════════════════════════════════════════════════════╝

def _plot_anchor_depth_scatter(data, am, outdir):
    """
    Scatter plot: each anchor's depth vs its matched samples' depths.
    Shows how tightly the 1-to-K matching controls for depth.
    """
    total_bp = data.total_bp

    # H-G anchored
    if am.hg_anchor_ids and am.hg_anchor_to_trans:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, gname, a2r, color in [
            (axes[0], "Transitional", am.hg_anchor_to_trans, COLORS["Transitional"]),
            (axes[1], "Industrial", am.hg_anchor_to_ind, COLORS["Industrial"]),
        ]:
            anchor_depths, ref_depths = [], []
            for aid, rids in a2r.items():
                a_bp = float(total_bp[aid])
                for rid in rids:
                    anchor_depths.append(np.log10(a_bp))
                    ref_depths.append(np.log10(float(total_bp[rid])))

            ax.scatter(anchor_depths, ref_depths, c=color, alpha=0.15,
                       s=8, edgecolors="none")
            lims = [min(min(anchor_depths), min(ref_depths)),
                    max(max(anchor_depths), max(ref_depths))]
            ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
            ax.set_xlabel("H-G anchor log₁₀(bp)")
            ax.set_ylabel(f"{gname} matched log₁₀(bp)")
            ax.set_title(f"H-G -> {gname} (K={len(a2r[am.hg_anchor_ids[0]])})")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)

        fig.suptitle("H-G-anchored depth matching: anchor vs matched",
                     fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(outdir, "depth_scatter_hg_anchored.png")
        savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fpath}")

    # NI anchored
    if am.ni_anchor_ids and am.ni_anchor_to_ind:
        fig, ax = plt.subplots(figsize=(6, 6))
        anchor_depths, ref_depths = [], []
        for aid, rids in am.ni_anchor_to_ind.items():
            a_bp = float(total_bp[aid])
            for rid in rids:
                anchor_depths.append(np.log10(a_bp))
                ref_depths.append(np.log10(float(total_bp[rid])))

        ax.scatter(anchor_depths, ref_depths, c=COLORS["Industrial"],
                   alpha=0.15, s=8, edgecolors="none")
        lims = [min(min(anchor_depths), min(ref_depths)),
                max(max(anchor_depths), max(ref_depths))]
        ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
        ax.set_xlabel("NI anchor log₁₀(bp)")
        ax.set_ylabel("Industrial matched log₁₀(bp)")
        k = len(am.ni_anchor_to_ind[am.ni_anchor_ids[0]])
        ax.set_title(f"NI -> Industrial (K={k})")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(outdir, "depth_scatter_ni_anchored.png")
        savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {fpath}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PUBLIC ENTRY POINT                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_anchored_analyses(data, am, stats):
    """
    Run full anchored depth-matched Section 1 analyses.

    Parameters
    ----------
    data : ReferenceData
    am : AnchoredMatchResult  (from depth_matching_anchored.build_anchored_matched_sets)
    stats : StatsReport
    """
    ref = data.ref_name
    trace_dir = os.path.join(
        "results", ref, "section1_diversity", "traces", "anchored")
    _ensure(trace_dir)

    s1_main = config.section_dir(ref, "s1", "main")
    s1_supp = config.section_dir(ref, "s1", "supplement")

    print(f"\n{'='*60}")
    print(f"  Section 1 — Anchored depth-matched analyses [{ref}]")
    print(f"  Output: {trace_dir}/")
    print(f"{'='*60}")

    t0 = time.time()

    # ── 1. Save matching metadata ──
    print(f"\n  [1/5] Saving anchored matching metadata …")
    _save_anchored_matching_metadata(data, am, trace_dir)

    # ── 2. H-G-anchored three-way analysis ──
    permanova_rows = []

    if am.hg_anchor_ids:
        print(f"\n  [2/5] H-G-anchored alpha + beta diversity …")

        for spec in SPECS:
            print(f"    [{spec.short_label}]:")

            group_ids = {
                "H-G": am.hg_anchor_ids,
                "Transitional": am.hg_trans_ids,
                "Industrial": am.hg_ind_ids,
            }

            alpha_df, pcoa_df = _run_anchored_alpha_beta(
                data, group_ids, "hg_anchored", spec,
                trace_dir, stats, permanova_rows)

            # Plots
            _plot_anchored_alpha(data, group_ids, "hg_anchored", spec, s1_main)
            _plot_anchored_pcoa(pcoa_df, "hg_anchored", spec, s1_main)
    else:
        print(f"\n  [2/5] H-G-anchored: no matched sets — skipping")

    # ── 3. NI-anchored two-way analysis ──
    if am.ni_anchor_ids:
        print(f"\n  [3/5] NI-anchored alpha + beta diversity …")

        for spec in SPECS:
            print(f"    [{spec.short_label}]:")

            group_ids = {
                "Non-Industrial": am.ni_anchor_ids,
                "Industrial": am.ni_ind_ids,
            }

            alpha_df, pcoa_df = _run_anchored_alpha_beta(
                data, group_ids, "ni_anchored", spec,
                trace_dir, stats, permanova_rows)

            # Plots
            _plot_anchored_alpha(data, group_ids, "ni_anchored", spec, s1_main)
            _plot_anchored_pcoa(pcoa_df, "ni_anchored", spec, s1_main)
    else:
        print(f"\n  [3/5] NI-anchored: no matched sets — skipping")

    # ── 4. Save combined PERMANOVA table ──
    beta_dir = _ensure(os.path.join(trace_dir, "beta"))
    if permanova_rows:
        pd.DataFrame(permanova_rows).to_csv(
            os.path.join(beta_dir, "permanova_anchored_all.csv"), index=False)

    # ── 5. Depth-matching QC plots ──
    print(f"\n  [4/5] Depth-matching QC plots …")
    _plot_anchored_depth_qc(data, am, s1_supp)
    _plot_anchor_depth_scatter(data, am, s1_supp)

    elapsed = time.time() - t0
    print(f"\n  [5/5] Anchored analyses complete [{ref}] ({elapsed / 60:.1f} min)")

    # Print inventory
    print(f"\n  Anchored trace file inventory:")
    total_size = 0
    for root, dirs, files in os.walk(trace_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            total_size += size
            rel = os.path.relpath(fpath, trace_dir)
            print(f"    {rel:<55s} {size / 1024:>8.1f} kB")
    print(f"  Total: {total_size / 1024 / 1024:.1f} MB")
