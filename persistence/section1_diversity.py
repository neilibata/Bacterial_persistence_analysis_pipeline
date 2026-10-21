"""
Section 1: Almost all human gut bacterial taxa are present in industrial
           populations.

Article outputs
---------------
  - Dataset overview table (88 studies, 46 countries, sample counts)
  - Sequencing depth stats per lifestyle (mean +/- SE)
  - Genome-length bias stat (reads-weighted mean Ind vs NI)
  - Alpha diversity: depth-matched 300kb + 10% breadth + redistribution correction
  - Beta diversity: PCoA + PERMANOVA (depth-matched Ind vs NI)
  - Accumulation curves (all + matched, multi-threshold)
  - Depth-matching QC plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import apply_oi_style, savefig_dual, current_fs
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import (mannwhitneyu, kruskal, spearmanr,
                          sem as scipy_sem, ks_2samp)

from . import config
from .common import classify_group, classify_macro_group
from .detection import DetectionSpec, build_presence
from .read_redistribution import redistribute_and_compute_shannon


# ════════════════════════════════════════════════════════════════════
#  1A. DATASET DESCRIPTION & SEQUENCING DEPTH
# ════════════════════════════════════════════════════════════════════

def compute_dataset_overview(data, stats):
    """Compute and log all dataset-description statistics."""
    meta = data.meta
    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")
    s1_supp = config.section_dir(ref, "s1", "supplement")

    n_total = len(meta)
    n_studies = meta["_study"].nunique() if "_study" in meta.columns else 0
    n_countries = meta["Country_norm"].nunique()

    stats.add("n_total_samples", "s1", "main", "Total metagenomes", n_total)
    stats.add("n_studies", "s1", "main", "Number of studies", n_studies)
    stats.add("n_countries", "s1", "main", "Number of countries", n_countries)

    # Catalogue size. The main text quotes these (3,971 / 3,594 / 5,081) but they
    # lived only in the length files, so no manuscript number could be bound back
    # to a computation. Recording them here closes that gap.
    stats.add("n_sgbs_catalogue", "s1", "main",
              f"SGBs in the {ref} reference catalogue", int(len(data.lengths)))
    stats.add("n_sgbs_analysed", "s1", "main",
              f"SGBs carried into the analysis for {ref} (in both matrices, with a length)",
              int(data.breadth.shape[1]))
    total_cat_bp = float(data.lengths.sum())
    stats.add("catalogue_total_bp", "s1", "main",
              f"Total catalogue length ({ref})", f"{total_cat_bp/1e9:.2f} Gb")

    depth_rows = []
    for lf in ["H-G", "Transitional", "Industrial"]:
        mask = meta["Lifestyle"] == lf
        n = int(mask.sum())

        # Mapped bases (Σ mean_depth x genome_length)
        mapped  = data.total_bp[mask].values.astype(float)
        mapped  = mapped[np.isfinite(mapped) & (mapped > 0)]
        total_mapped = data.total_bp[mask].sum()

        # Sequenced bases (raw FASTQ, from metadata "Total bases" column)
        seq = data.total_seq_bp.reindex(meta.index[mask]).values.astype(float)
        seq = seq[np.isfinite(seq) & (seq > 0)]
        total_seq = float(np.nansum(data.total_seq_bp.reindex(meta.index[mask])))

        stats.add(f"n_samples_{lf}", "s1", "main",
                  f"Number of {lf} samples", n)
        stats.add(f"total_bases_mapped_{lf}", "s1", "main",
                  f"Total mapped bases ({lf})",
                  f"{total_mapped/1e12:.1f} trillion")
        if total_seq > 0:
            stats.add(f"total_bases_sequenced_{lf}", "s1", "main",
                      f"Total sequenced bases ({lf})",
                      f"{total_seq/1e12:.1f} trillion")
            stats.add(f"mapping_rate_{lf}", "s1", "main",
                      f"Mapping rate ({lf})",
                      f"{100*total_mapped/total_seq:.1f}%")

        if len(mapped) == 0:
            continue
        m_map = np.mean(mapped);  se_map = scipy_sem(mapped)
        row = {
            "Lifestyle": lf, "n": n,
            "mean_mapped_bp": m_map, "se_mapped_bp": se_map,
            "median_mapped_bp": np.median(mapped),
        }
        stats.add(f"depth_mean_se_{lf}", "s1", "main",
                  f"Mapped depth mean +/- SE ({lf})",
                  f"{m_map/1e9:.2f} +/- {se_map/1e9:.2f} Gb")

        if len(seq) > 0:
            m_seq = np.mean(seq);  se_seq = scipy_sem(seq)
            row.update({"mean_seq_bp": m_seq, "se_seq_bp": se_seq,
                        "median_seq_bp": np.median(seq)})
            stats.add(f"seq_depth_mean_se_{lf}", "s1", "main",
                      f"Sequenced depth mean +/- SE ({lf})",
                      f"{m_seq/1e9:.2f} +/- {se_seq/1e9:.2f} Gb")

        depth_rows.append(row)

    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(os.path.join(s1_main, "sequencing_depth_by_lifestyle.csv"),
                    index=False)

    _build_study_table(meta, data, s1_main, s1_supp, stats)
    _plot_depth_violin(meta, data, s1_main)
    return depth_df


def _build_study_table(meta, data, s1_main, s1_supp, stats):
    """Build per-study and per-country summary tables."""
    rows = []
    for (country, lifestyle), grp in meta.groupby(
            ["Country_norm", "Lifestyle"], dropna=False):
        n = len(grp)
        total_bp = data.total_bp[grp.index].sum()
        rows.append({
            "Country": country, "Lifestyle": lifestyle,
            "n_samples": n, "total_bases_Tb": total_bp / 1e12,
            "mean_depth_Gb": (total_bp / n / 1e9) if n > 0 else 0,
        })
    overview = pd.DataFrame(rows).sort_values(
        ["Lifestyle", "Country"]).reset_index(drop=True)
    overview.to_csv(os.path.join(s1_main, "dataset_overview.csv"), index=False)

    if "_study" in meta.columns and meta["_study"].notna().any():
        study_rows = []
        for study, grp in meta.groupby("_study", dropna=False):
            total_bp = data.total_bp[grp.index].sum()
            study_rows.append({
                "Study": study, "n_samples": len(grp),
                "Lifestyle": grp["Lifestyle"].mode().iloc[0] if len(grp) > 0 else "",
                "Countries": ", ".join(sorted(grp["Country_norm"].dropna().unique())),
                "total_bases_Tb": total_bp / 1e12,
            })
        study_df = pd.DataFrame(study_rows).sort_values(
            "n_samples", ascending=False).reset_index(drop=True)
        study_df.to_csv(os.path.join(s1_supp, "per_study_metadata.csv"),
                        index=False)


def _plot_depth_violin(meta, data, outdir):
    """
    Paired violin plot: sequenced bases (light) vs mapped bases (dark),
    side-by-side within each lifestyle group.

    Sequenced = raw FASTQ bases from metadata "Total bases".
    Mapped    = Σ(mean_depth x genome_length), MAPQ >= 30 filter applied.
    """
    lifestyles = ["H-G", "Transitional", "Industrial"]

    # ── Collect distributions ─────────────────────────────────────────────────
    groups = {}   # lf -> {"mapped": array, "seq": array}
    for lf in lifestyles:
        mask = meta["Lifestyle"] == lf
        mapped = data.total_bp.reindex(meta.index[mask]).values.astype(float)
        mapped = np.log10(mapped[np.isfinite(mapped) & (mapped > 0)])

        seq_raw = data.total_seq_bp.reindex(meta.index[mask]).values.astype(float)
        seq = np.log10(seq_raw[np.isfinite(seq_raw) & (seq_raw > 0)])

        groups[lf] = {"mapped": mapped, "seq": seq,
                      "n": int(mask.sum()), "n_seq": len(seq)}

    # ── Layout: one column per lifestyle, two side-by-side violins each ───────
    fig, axes = plt.subplots(1, len(lifestyles),
                             figsize=(10, 5), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.10, right=0.97,
                        top=0.88, bottom=0.12)

    tick_gb   = [0.04, 0.1, 0.4, 1, 4, 10, 40, 100, 400]
    tick_log  = [np.log10(v) for v in tick_gb]

    for ax, lf in zip(axes, lifestyles):
        color = config.LIFESTYLE_COLORS.get(lf, "gray")
        g     = groups[lf]

        vdata = []
        positions = []
        face_colors = []

        if len(g["seq"]) > 1:
            vdata.append(g["seq"])
            positions.append(1)
            # Light version of lifestyle colour for sequenced
            face_colors.append(color)

        if len(g["mapped"]) > 1:
            vdata.append(g["mapped"])
            positions.append(2)
            face_colors.append(color)

        if vdata:
            parts = ax.violinplot(vdata, positions=positions,
                                  showmedians=True, showextrema=False,
                                  widths=0.7)
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(face_colors[j])
                pc.set_edgecolor("white")
                pc.set_linewidth(0.5)
                # Sequenced = translucent; Mapped = solid
                pc.set_alpha(0.35 if positions[j] == 1 else 0.80)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

        # Annotate medians
        for pos, arr, label in [
            (1, g["seq"],    "seq"),
            (2, g["mapped"], "mapped"),
        ]:
            if len(arr) > 1:
                med = np.median(arr)
                gb  = 10**med / 1e9
                ax.text(pos, med + 0.06, f"{gb:.1f} Gb",
                        ha="center", va="bottom", fontsize=FS-3.5,
                        color="black", fontweight="bold")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Sequenced", "Mapped"], fontsize=FS-2, rotation=20)
        ax.set_title(f"{lf}\n(n={g['n']:,})", fontsize=FS-1,
                     color=color, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        # Shared y-ticks on leftmost axis only
        valid_ticks = [t for t in tick_log
                       if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]

    # Apply y-ticks once after all violins drawn (sharey=True)
    all_vals = np.concatenate([
        np.concatenate([g["seq"], g["mapped"]])
        for g in groups.values()
        if len(g["seq"]) > 0 or len(g["mapped"]) > 0
    ])
    if len(all_vals):
        ylo = max(all_vals.min() - 0.3, np.log10(0.03e9))
        yhi = all_vals.max() + 0.3
        axes[0].set_ylim(ylo, yhi)
        valid_ticks = [t for t in tick_log if ylo <= t <= yhi]
        axes[0].set_yticks(valid_ticks)
        axes[0].set_yticklabels([f"{v} Gb" for v in tick_gb[:len(valid_ticks)]])

    axes[0].set_ylabel("Sequencing depth", fontsize=FS-1)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="gray", alpha=0.35, edgecolor="white",
              label="Sequenced bases (raw FASTQ)"),
        Patch(facecolor="gray", alpha=0.80, edgecolor="white",
              label="Mapped bases (MAPQ >= 30)"),
    ]
    fig.legend(handles=legend_els, loc="upper center",
               ncol=2, fontsize=FS-2, frameon=True,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Sequenced vs Mapped bases by lifestyle",
                 y=1.01, fontsize=FS+1)

    fpath = os.path.join(outdir, "sequencing_depth_by_lifestyle.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  1B. ALPHA DIVERSITY — DEPTH-MATCHED (300kb + 10% breadth)
# ════════════════════════════════════════════════════════════════════

def compute_alpha_matched_pair(
    data, ids_a, ids_b, label_a, label_b,
    stats, spec=None,
):
    """
    Alpha diversity comparison on a 1:1 depth-matched sample set.
    Uses the given DetectionSpec (flat_bp or breadth_pct).
    """
    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")

    if spec is None:
        spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    spec_tag = spec.short_label

    if not ids_a or not ids_b:
        print(f"    [alpha] No matched pairs for {label_a} vs {label_b}")
        return None

    all_ids = list(ids_a) + list(ids_b)
    presence = build_presence(
        data.breadth.loc[all_ids],
        data.depth.loc[all_ids],
        data.lengths, spec=spec)

    richness = presence.sum(axis=1)
    abund_m = data.abundance.loc[all_ids]
    abund_pos = abund_m.clip(lower=1e-10)
    shannon = -(abund_pos * np.log(abund_pos)).sum(axis=1)

    rich_a = richness.loc[ids_a].values
    rich_b = richness.loc[ids_b].values
    shan_a = shannon.loc[ids_a].values
    shan_b = shannon.loc[ids_b].values

    fname_tag = f"{label_a}_vs_{label_b}".replace(" ", "_")

    for metric, vals_a, vals_b in [
        ("Richness", rich_a, rich_b),
        ("Shannon", shan_a, shan_b),
    ]:
        med_a, med_b = np.median(vals_a), np.median(vals_b)
        ratio = med_a / med_b if med_b > 0 else np.inf
        U, p_mw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        stats.add(
            f"alpha_matched_{metric}_{fname_tag}_{spec_tag}",
            "s1", "main",
            f"{metric}: {label_a} vs {label_b} "
            f"(1:1 depth-matched, {spec.description})",
            f"median {label_a}={med_a:.0f}, {label_b}={med_b:.0f}, "
            f"ratio={ratio:.2f}x",
            test="Mann-Whitney U", p_value=p_mw,
            n1=len(vals_a), n2=len(vals_b), effect_size=U,
            extra={"median_a": med_a, "median_b": med_b, "ratio": ratio})

    # Save CSV
    alpha_df = pd.DataFrame({
        "sample": all_ids,
        "group": [label_a] * len(ids_a) + [label_b] * len(ids_b),
        "Richness": richness.loc[all_ids].values,
        "Shannon": shannon.loc[all_ids].values,
        "detection": spec.description,
    })
    alpha_df.to_csv(os.path.join(
        s1_main, f"alpha_diversity_matched_{fname_tag}_{spec_tag}.csv"),
        index=False)

    # Plot
    _plot_alpha_matched(
        rich_a, rich_b, shan_a, shan_b,
        label_a, label_b, spec, s1_main,
        suffix=f"_{fname_tag}_{spec_tag}")

    return alpha_df


def _plot_alpha_matched(rich_a, rich_b, shan_a, shan_b,
                        label_a, label_b, spec, outdir, suffix=""):
    """Two-panel boxplot: Richness + Shannon for a matched pair."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    color_a = config.LIFESTYLE_COLORS.get(label_a, config.OI["vermillion"])
    color_b = config.LIFESTYLE_COLORS.get(label_b, config.OI["blue"])
    rng = np.random.default_rng(42)

    for ax, metric, va, vb in zip(axes,
                                   ["Richness", "Shannon"],
                                   [rich_a, shan_a],
                                   [rich_b, shan_b]):
        labels = [f"{label_a}\n(n={len(va):,})",
                  f"{label_b}\n(n={len(vb):,})"]
        bplot = ax.boxplot([va, vb], patch_artist=True, widths=0.5,
                           showfliers=False)
        for patch, c in zip(bplot["boxes"], [color_a, color_b]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        for i, (vals, c) in enumerate(zip([va, vb], [color_a, color_b]), 1):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       c=c, alpha=0.08, s=4, zorder=1, edgecolors="none")

        U, p = mannwhitneyu(va, vb, alternative="two-sided")
        med_a, med_b = np.median(va), np.median(vb)
        ratio = med_a / med_b if med_b > 0 else np.inf
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}\nmedian ratio = {ratio:.2f}x, p = {p:.2e}",
                     fontsize=FS)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{label_a} vs {label_b} — 1:1 depth-matched\n"
                 f"({spec.description})", fontsize=FS+2, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir, f"alpha_diversity_matched{suffix}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


def run_alpha_matched(data, dm, stats):
    """
    Run alpha diversity for both depth-matched comparisons.
    Uses the flat 300kb threshold AND the 10% breadth coverage threshold.
    """
    flat_spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    breadth_spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)

    for spec in [flat_spec, breadth_spec]:
        print(f"    [{spec.short_label}] detection: {spec.description}")

        # Industrial vs Transitional
        compute_alpha_matched_pair(
            data,
            dm.ind_vs_trans_ind, dm.ind_vs_trans_trans,
            "Industrial", "Transitional",
            stats, spec=spec)

        # Industrial vs Non-Industrial
        compute_alpha_matched_pair(
            data,
            dm.ind_vs_ni_ind, dm.ind_vs_ni_ni,
            "Industrial", "Non-Industrial",
            stats, spec=spec)


# ════════════════════════════════════════════════════════════════════
#  1B′. GENOME-LENGTH CORRECTED ALPHA (redistribution)
# ════════════════════════════════════════════════════════════════════

def run_redistribution_corrected_alpha(data, dm, stats, L_ref=None):
    """
    Genome-length redistribution correction on depth-matched pairs.
    Produces 2x2 plots (observed vs corrected, richness vs Shannon).
    """
    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")
    flat_bp = config.ACCUM_MAIN_TEXT_FLAT_BP

    comparisons = [
        ("Industrial", "Transitional",
         dm.ind_vs_trans_ind, dm.ind_vs_trans_trans),
        ("Industrial", "Non-Industrial",
         dm.ind_vs_ni_ind, dm.ind_vs_ni_ni),
    ]

    for label_a, label_b, ids_a, ids_b in comparisons:
        if not ids_a or not ids_b:
            print(f"    [redist] No matched pairs for {label_a} vs {label_b}")
            continue

        all_ids = list(ids_a) + list(ids_b)
        result_df, L_ref_used = redistribute_and_compute_shannon(
            data.breadth, data.depth, data.lengths, all_ids,
            flat_threshold_bp=flat_bp, L_ref=L_ref)

        res_a = result_df.loc[result_df.index.isin(ids_a)]
        res_b = result_df.loc[result_df.index.isin(ids_b)]
        fname_tag = f"{label_a}_vs_{label_b}".replace(" ", "_")

        # Stats
        for metric, col_obs, col_corr in [
            ("Richness", "richness_observed", "richness_corrected"),
            ("Shannon", "shannon_observed", "shannon_corrected"),
        ]:
            obs_a, obs_b = res_a[col_obs].values, res_b[col_obs].values
            corr_a, corr_b = res_a[col_corr].values, res_b[col_corr].values

            ratio_obs = np.median(obs_a) / max(np.median(obs_b), 1e-10)
            ratio_corr = np.median(corr_a) / max(np.median(corr_b), 1e-10)
            _, p_corr = mannwhitneyu(corr_a, corr_b, alternative="two-sided")

            stats.add(
                f"alpha_redist_{metric}_{fname_tag}_300kb",
                "s1", "main",
                f"{metric} (genome-length corrected): {label_a} vs {label_b} "
                f"(L_ref={L_ref_used/1e6:.2f} Mbp)",
                f"observed ratio={ratio_obs:.2f}x, "
                f"corrected ratio={ratio_corr:.2f}x",
                test="Mann-Whitney U", p_value=p_corr,
                n1=len(corr_a), n2=len(corr_b),
                extra={"ratio_obs": ratio_obs, "ratio_corr": ratio_corr,
                       "L_ref_Mbp": L_ref_used / 1e6})

        # Diagnostics
        print(f"    [redist] {label_a} vs {label_b} "
              f"(L_ref={L_ref_used/1e6:.2f} Mbp)")
        print(f"      {label_a}: freed={res_a['freed_reads_pct'].mean():.1f}%, "
              f"DN={res_a['delta_N'].mean():+.1f}")
        print(f"      {label_b}: freed={res_b['freed_reads_pct'].mean():.1f}%, "
              f"DN={res_b['delta_N'].mean():+.1f}")

        # CSV
        out_df = result_df.copy()
        out_df["group"] = ""
        out_df.loc[out_df.index.isin(ids_a), "group"] = label_a
        out_df.loc[out_df.index.isin(ids_b), "group"] = label_b
        out_df.to_csv(os.path.join(
            s1_main, f"alpha_redist_matched_{fname_tag}_300kb.csv"))

        # Plot
        _plot_redistribution_alpha(
            res_a, res_b, label_a, label_b,
            flat_bp, L_ref_used, s1_main,
            suffix=f"_{fname_tag}_300kb")


def _plot_redistribution_alpha(res_a, res_b, label_a, label_b,
                                flat_bp, L_ref, outdir, suffix=""):
    """2x2: observed vs corrected x richness vs Shannon."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    color_a = config.LIFESTYLE_COLORS.get(label_a, config.OI["vermillion"])
    color_b = config.LIFESTYLE_COLORS.get(label_b, config.OI["blue"])
    rng = np.random.default_rng(42)

    panels = [
        (axes[0, 0], "richness_observed", "Observed Richness"),
        (axes[0, 1], "richness_corrected", "Corrected Richness"),
        (axes[1, 0], "shannon_observed", "Observed Shannon"),
        (axes[1, 1], "shannon_corrected", "Corrected Shannon"),
    ]

    for ax, col, title in panels:
        va = res_a[col].values
        vb = res_b[col].values
        labels = [f"{label_a}\n(n={len(va):,})",
                  f"{label_b}\n(n={len(vb):,})"]

        bplot = ax.boxplot([va, vb], patch_artist=True, widths=0.5,
                           showfliers=False)
        for patch, c in zip(bplot["boxes"], [color_a, color_b]):
            patch.set_facecolor(c); patch.set_alpha(0.6)

        for i, (vals, c) in enumerate(zip([va, vb], [color_a, color_b]), 1):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       c=c, alpha=0.08, s=4, zorder=1, edgecolors="none")

        U, p = mannwhitneyu(va, vb, alternative="two-sided")
        med_a, med_b = np.median(va), np.median(vb)
        ratio = med_a / med_b if med_b > 0 else np.inf
        ax.set_xticklabels(labels)
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(f"{title}\nmedian ratio = {ratio:.2f}x, p = {p:.2e}",
                     fontsize=FS)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{label_a} vs {label_b} — 1:1 depth-matched\n"
        f"Genome-length redistribution "
        f"(L_ref = {L_ref/1e6:.2f} Mbp, >={flat_bp//1000} kb)",
        fontsize=FS+2, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir,
                         f"alpha_diversity_redist_matched{suffix}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  1C. BETA DIVERSITY — DEPTH-MATCHED PCoA + PERMANOVA
# ════════════════════════════════════════════════════════════════════

def compute_beta_diversity_matched(data, dm, stats):
    """
    PCoA + PERMANOVA on Bray-Curtis distances using depth-matched
    Industrial vs Non-Industrial samples.
    """
    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")

    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind or not ids_ni:
        print("    [beta] No depth-matched pairs — skipping")
        return None

    all_ids = list(ids_ind) + list(ids_ni)
    abund_m = data.abundance.loc[all_ids]
    meta_m = data.meta.loc[all_ids]
    n_ind, n_ni = len(ids_ind), len(ids_ni)
    n = len(all_ids)
    print(f"    [beta] Depth-matched: Industrial={n_ind}, "
          f"Non-Industrial={n_ni}, total={n}")

    vals = np.nan_to_num(abund_m.values, nan=0.0, posinf=0.0, neginf=0.0)

    from scipy.spatial.distance import pdist, squareform

    print("    [beta] Computing Bray-Curtis distances...")
    D = pdist(vals, metric="braycurtis")
    D = np.nan_to_num(D, nan=0.0)
    D_sq = squareform(D)

    # PCoA — numerically stable double-centering
    D2 = D_sq ** 2
    row_means = D2.mean(axis=1)
    col_means = D2.mean(axis=0)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_means[:, np.newaxis] - col_means[np.newaxis, :] +
                grand_mean)

    n_axes = config.DIVERSITY_PCOA_AXES
    try:
        from scipy.linalg import eigh as scipy_eigh
        eigenvalues, eigenvectors = scipy_eigh(
            B, subset_by_index=[n - n_axes, n - 1])
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
    except (ImportError, TypeError):
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort[:n_axes]]
        eigenvectors = eigenvectors[:, idx_sort[:n_axes]]

    trace_B = np.trace(B)
    total_var = max(trace_B, 1e-10)
    var_explained = np.array([
        max(ev, 0) / total_var * 100 for ev in eigenvalues])

    top_ev = np.maximum(eigenvalues, 0)
    coords = eigenvectors * np.sqrt(top_ev)

    print(f"    [beta] PCoA axis 1: {var_explained[0]:.1f}%, "
          f"axis 2: {var_explained[1]:.1f}%")

    pcoa_df = pd.DataFrame(
        coords, index=abund_m.index,
        columns=[f"PCo{i+1}" for i in range(n_axes)])
    pcoa_df["Lifestyle"] = meta_m["Lifestyle"].values
    pcoa_df["Group"] = ["Industrial"] * n_ind + ["Non-Industrial"] * n_ni
    pcoa_df.to_csv(os.path.join(s1_main, "beta_diversity_pcoa_matched.csv"))

    # PERMANOVA — binary Industrial vs Non-Industrial
    print("    [beta] Running PERMANOVA...")
    binary_labels = np.array(
        ["Industrial"] * n_ind + ["Non-Industrial"] * n_ni)
    try:
        F_obs, p_perm = _permanova(
            D_sq, binary_labels, n_perm=config.DIVERSITY_N_PERMANOVA)
        print(f"    [beta] PERMANOVA: F={F_obs:.2f}, p={p_perm:.4f}")
        stats.add("beta_permanova_matched", "s1", "main",
                  "PERMANOVA (depth-matched Ind vs NI)",
                  f"F={F_obs:.2f}", test="PERMANOVA", p_value=p_perm,
                  n1=n_ind, n2=n_ni)
    except Exception as e:
        print(f"    [beta] PERMANOVA failed: {e}")

    for ax_i in range(min(n_axes, len(var_explained))):
        stats.add(f"pcoa_var_axis{ax_i+1}_matched", "s1", "main",
                  f"PCoA axis {ax_i+1} variance (depth-matched)",
                  f"{var_explained[ax_i]:.1f}%")

    _plot_pcoa_matched(pcoa_df, var_explained, s1_main)
    return pcoa_df


def _permanova(D_sq, labels, n_perm=999):
    """Simple PERMANOVA (Anderson 2001)."""
    rng = np.random.default_rng(42)
    n = D_sq.shape[0]
    unique_groups = np.unique(labels)
    a = len(unique_groups)

    def _ss(D, labs):
        ss_w = 0
        for g in unique_groups:
            idx = np.where(labs == g)[0]
            if len(idx) < 2:
                continue
            sub = D[np.ix_(idx, idx)]
            ss_w += np.sum(sub ** 2) / (2 * len(idx))
        ss_t = np.sum(D ** 2) / (2 * n)
        return ss_t - ss_w, ss_w

    ss_a_obs, ss_w_obs = _ss(D_sq, labels)
    if ss_w_obs == 0:
        return np.inf, 0.001
    F_obs = (ss_a_obs / (a - 1)) / (ss_w_obs / (n - a))

    n_ge = 1
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        ss_a_p, ss_w_p = _ss(D_sq, perm_labels)
        if ss_w_p > 0:
            F_p = (ss_a_p / (a - 1)) / (ss_w_p / (n - a))
        else:
            F_p = 0
        if F_p >= F_obs:
            n_ge += 1

    return F_obs, n_ge / (n_perm + 1)


def _plot_pcoa_matched(pcoa_df, var_explained, outdir):
    """PCoA scatter coloured by lifestyle."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for lf in ["H-G", "Transitional", "Industrial"]:
        sub = pcoa_df[pcoa_df["Lifestyle"] == lf]
        if sub.empty:
            continue
        color = config.LIFESTYLE_COLORS.get(lf, "gray")
        ax.scatter(sub["PCo1"], sub["PCo2"],
                   c=color, s=10, alpha=0.5,
                   label=f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} (n={len(sub):,})",
                   edgecolors="none")

    ax.set_xlabel(f"PCo1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PCo2 ({var_explained[1]:.1f}%)")
    ax.set_title("Beta Diversity: PCoA on Bray-Curtis\n"
                 "(1:1 depth-matched, Industrial vs Non-Industrial)")
    ax.legend(fontsize=FS-1, markerscale=3)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fpath = os.path.join(outdir, "beta_diversity_pcoa_matched.png")
    savefig_dual(fig, fpath, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  1D. GENOME-LENGTH BIAS STAT
# ════════════════════════════════════════════════════════════════════

def compute_genome_length_stat(data, dm, stats):
    """
    Article stat: reads-weighted mean genome length Industrial vs NI.
    Computed on the depth-matched sample set.
    """
    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")
    lengths = data.lengths

    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind or not ids_ni:
        return

    # Reads-weighted mean genome length per sample
    def _rw_mean_L(sample_ids):
        vals = []
        for sid in sample_ids:
            depth_row = data.depth.loc[sid].values.astype(float)
            sgbs = data.depth.columns
            mapped = depth_row * lengths.reindex(sgbs, fill_value=0).values
            det = mapped > 0
            if det.sum() > 0:
                L_arr = lengths.reindex(sgbs, fill_value=0).values
                vals.append(np.average(L_arr[det], weights=mapped[det]) / 1e6)
        return np.array(vals)

    rw_ind = _rw_mean_L(ids_ind)
    rw_ni = _rw_mean_L(ids_ni)

    if len(rw_ind) > 0 and len(rw_ni) > 0:
        U, p = mannwhitneyu(rw_ind, rw_ni, alternative="two-sided")
        stats.add("genome_length_rw_mean_ind_vs_ni", "s1", "main",
                  "Reads-weighted mean genome length: Industrial vs NI "
                  "(depth-matched)",
                  f"Ind={np.mean(rw_ind):.2f} Mbp, NI={np.mean(rw_ni):.2f} Mbp",
                  test="Mann-Whitney U", p_value=p,
                  n1=len(rw_ind), n2=len(rw_ni),
                  extra={"mean_ind_Mbp": np.mean(rw_ind),
                         "mean_ni_Mbp": np.mean(rw_ni)})
        print(f"    Genome length (reads-weighted): "
              f"Ind={np.mean(rw_ind):.2f} Mbp, NI={np.mean(rw_ni):.2f} Mbp, "
              f"p={p:.2e}")


# ════════════════════════════════════════════════════════════════════
#  1E. ACCUMULATION CURVES
# ════════════════════════════════════════════════════════════════════

def run_accumulation_with_stats(data, dm, matched_idx_all, stats,
                                all_specs=None, main_specs=None):
    """Run accumulation curves: all samples + depth-matched subset."""
    ref = data.ref_name

    if all_specs is None:
        from .detection import breadth_pct_specs
        all_specs = breadth_pct_specs(config.ACCUM_BREADTH_THRESHOLDS)
    if main_specs is None:
        main_specs = [DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)]

    from .accumulation import run_accumulation
    from .accumulation_matched import run_accumulation_matched

    breadth_specs = [s for s in all_specs if s.mode == "breadth_pct"]
    if breadth_specs:
        breadth_thrs = [s.threshold for s in breadth_specs]
        saved = config.ACCUM_BREADTH_THRESHOLDS
        config.ACCUM_BREADTH_THRESHOLDS = breadth_thrs

        print("    ── Accumulation: ALL samples (unmatched) ──")
        run_accumulation(data, matched_idx_all)

        print("    ── Accumulation: depth-matched subset ──")
        dm_idx = dm.all_matched_ind_vs_ni if dm else matched_idx_all
        run_accumulation_matched(data, dm_idx)

        config.ACCUM_BREADTH_THRESHOLDS = saved

    # Flat-bp accumulation (all samples)
    flat_specs = [s for s in all_specs if s.mode == "flat_bp"]
    if flat_specs:
        _run_flat_accumulation(data, matched_idx_all, flat_specs)

        # Flat-bp accumulation (depth-matched)
        if dm is not None:
            _run_flat_accumulation_matched(data, dm, flat_specs)

    for spec in main_specs:
        _log_recovery_stats(data, stats, spec=spec)


def _run_flat_accumulation(data, matched_idx, flat_specs):
    """
    Run accumulation curves using flat mapped-bases thresholds.

    Produces the same multi-series pooled + continent plots as the
    breadth-based accumulation, for each flat-bp threshold.
    """
    from .accumulation import (
        _run_sweep, fit_central_curve, plot_accumulation_curves,
        plot_param_vs_N,
    )
    from .common import classify_group, classify_macro_group

    ref = data.ref_name
    out = config.outdir(ref)
    meta = data.meta.copy()
    rng = np.random.default_rng(42)

    meta["Group"] = meta.apply(classify_group, axis=1)
    meta["MacroGroup"] = meta.apply(classify_macro_group, axis=1)
    group_map = meta["Group"]
    cont_col = "Continent" if "Continent" in meta.columns else None

    for spec in flat_specs:
        thr_label = spec.short_label
        thr_desc = spec.description
        print(f"\n    ═══ Flat threshold: {thr_desc} ═══")

        presence = build_presence(
            data.breadth, data.depth, data.lengths, spec=spec)
        # Convert boolean presence to pseudo-breadth (0 or 100%)
        pseudo_breadth = presence.astype(float) * 100.0

        # ── A. POOLED ──
        series_inputs = []
        style_map = {}
        for name, groups, color, ls in config.ACCUM_POOLED_SERIES:
            mask = group_map.isin(groups)
            block = pseudo_breadth.loc[mask]
            if block.shape[0] < config.ACCUM_MIN_SAMPLES:
                continue
            series_inputs.append((name, block, color, ls))
            style_map[name] = (color, ls)
            print(f"      [Pooled] {name}: {block.shape[0]:,} samples")

        if series_inputs:
            # Use thr=50 as the detection threshold on pseudo-breadth
            curve_data, sweep_df = _run_sweep(
                series_inputs, 50.0, rng, out, ref, "flat_pooled")

            if curve_data:
                plot_accumulation_curves(
                    curve_data, 50.0, ref, out,
                    suffix=f"_flat_{thr_label}_pooled",
                    ylabel=f"SGBs detected ({thr_desc})",
                    title=f"Accumulation: SGBs detected ({thr_desc})\n"
                          f"pooled per lifestyle ({ref} ref)")

            if not sweep_df.empty:
                sweep_df.to_csv(os.path.join(
                    out, f"accum_flat_{thr_label}_pooled_fits.csv"),
                    index=False)

                plot_param_vs_N(
                    sweep_df, 50.0,
                    f"Pooled ({ref}, {thr_desc})",
                    style_map,
                    os.path.join(out,
                        f"accum_flat_{thr_label}_pooled_params_vsN.png"))

        # ── B. CONTINENT BREAKDOWN ──
        if cont_col is None or meta[cont_col].isna().all():
            continue

        series_inputs_cont = []
        style_map_cont = {}

        for macro in ["Industrial", "Non-Industrial"]:
            macro_mask = meta["MacroGroup"] == macro
            sub_meta = meta.loc[macro_mask].copy()
            continents = sorted(
                sub_meta[cont_col].dropna().unique().tolist())

            for cont in continents:
                ids_cont = sub_meta.index[sub_meta[cont_col] == cont]
                block = pseudo_breadth.loc[
                    pseudo_breadth.index.intersection(ids_cont)]
                if block.shape[0] < config.ACCUM_MIN_SAMPLES:
                    continue

                label = f"{macro} - {cont}"
                color = config.CONTINENT_COLORS.get(cont, "gray")
                ls = "-" if macro == "Industrial" else "--"
                series_inputs_cont.append((label, block, color, ls))
                style_map_cont[label] = (color, ls)

        if series_inputs_cont:
            curve_data_cont, sweep_df_cont = _run_sweep(
                series_inputs_cont, 50.0, rng, out, ref, "flat_continent")

            if curve_data_cont:
                plot_accumulation_curves(
                    curve_data_cont, 50.0, ref, out,
                    suffix=f"_flat_{thr_label}_continent",
                    ylabel=f"SGBs detected ({thr_desc})",
                    title=f"Accumulation by continent ({thr_desc})\n"
                          f"Industrial (solid) vs NI (dashed) ({ref})")

            if not sweep_df_cont.empty:
                sweep_df_cont.to_csv(os.path.join(
                    out, f"accum_flat_{thr_label}_continent_fits.csv"),
                    index=False)

                plot_param_vs_N(
                    sweep_df_cont, 50.0,
                    f"By Continent ({ref}, {thr_desc})",
                    style_map_cont,
                    os.path.join(out,
                        f"accum_flat_{thr_label}_continent_params_vsN.png"))


def _run_flat_accumulation_matched(data, dm, flat_specs):
    """
    Run depth-matched accumulation curves using flat mapped-bases thresholds.

    Uses the same Ind ↔ NI depth-matched sets as the breadth accumulation.
    """
    from .accumulation_matched import (
        _run_matched_sweep, greedy_depth_match,
    )
    from .accumulation import plot_accumulation_curves, plot_param_vs_N
    from .common import classify_macro_group

    ref = data.ref_name
    out = config.outdir(ref)
    meta = data.meta.copy()
    rng = np.random.default_rng(123)

    # Use the same depth-matched IDs from dm
    m_ind = dm.ind_vs_ni_ind
    m_ni = dm.ind_vs_ni_ni
    if not m_ind or not m_ni:
        print("    [flat matched] No depth-matched pairs — skipping")
        return

    print(f"\n    ── Flat depth-matched accumulation ──")
    print(f"    Matched: Industrial={len(m_ind)}, "
          f"Non-Industrial={len(m_ni)}")

    for spec in flat_specs:
        thr_label = spec.short_label
        thr_desc = spec.description
        print(f"\n    ═══ Flat threshold: {thr_desc} (depth-matched) ═══")

        presence = build_presence(
            data.breadth, data.depth, data.lengths, spec=spec)
        pseudo_breadth = presence.astype(float) * 100.0

        series_inputs = [
            (f"Industrial (matched, n={len(m_ind)})",
             pseudo_breadth.loc[m_ind], config.OI["sky"], "-"),
            (f"Non-Industrial (matched, n={len(m_ni)})",
             pseudo_breadth.loc[m_ni], config.OI["orange"], "-"),
        ]

        curve_data, sweep_df = _run_matched_sweep(
            series_inputs, 50.0, rng, out, ref)

        if curve_data:
            plot_accumulation_curves(
                curve_data, 50.0, ref, out,
                suffix=f"_flat_{thr_label}_pooled_depthmatched",
                ylabel=f"SGBs detected ({thr_desc})",
                title=f"Depth-matched accumulation ({thr_desc})\n"
                      f"({ref} ref)")

        if not sweep_df.empty:
            sweep_df.to_csv(os.path.join(
                out, f"accum_flat_{thr_label}_matched_fits.csv"),
                index=False)


def _log_recovery_stats(data, stats, spec=None):
    """Count VANISH/BloSSUM recovered in opposite lifestyle."""
    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    spec_tag = spec.short_label

    meta = data.meta
    presence = build_presence(data.breadth, data.depth, data.lengths, spec=spec)

    # VANISH in Industrial
    ind_mask = meta["Lifestyle"] == "Industrial"
    vanish_in_ref = [v for v in data.vanish_set if v in presence.columns]
    ind_presence = presence.loc[ind_mask]
    detected_any = {v for v in vanish_in_ref
                    if v in ind_presence.columns and ind_presence[v].any()}
    stats.add(f"vanish_in_industrial_{spec_tag}", "s1", "main",
              f"VANISH taxa in >=1 Industrial sample ({spec.description})",
              f"{len(detected_any)}/{len(vanish_in_ref)}")

    # BloSSUM in Non-Industrial
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])
    blossum_in_ref = [b for b in data.blossum_set if b in presence.columns]
    ni_presence = presence.loc[ni_mask]
    detected_any_b = {b for b in blossum_in_ref
                      if b in ni_presence.columns and ni_presence[b].any()}
    stats.add(f"blossum_in_nonindustrial_{spec_tag}", "s1", "main",
              f"BloSSUM taxa in >=1 Non-Industrial sample ({spec.description})",
              f"{len(detected_any_b)}/{len(blossum_in_ref)}")


# ════════════════════════════════════════════════════════════════════
#  1F. DEPTH-MATCHING QC
# ════════════════════════════════════════════════════════════════════

def plot_depth_matching_qc(data, dm, outdir):
    """QC: depth distributions before and after matching."""
    meta = data.meta
    total_bp = data.total_bp

    comparisons = [
        ("Ind_vs_Trans", dm.ind_vs_trans_ind, dm.ind_vs_trans_trans,
         "Industrial", "Transitional"),
        ("Ind_vs_NI", dm.ind_vs_ni_ind, dm.ind_vs_ni_ni,
         "Industrial", "Non-Industrial"),
    ]

    for tag, ids_a, ids_b, lab_a, lab_b in comparisons:
        if not ids_a or not ids_b:
            continue

        mask_a = meta["Lifestyle"] == "Industrial"
        if lab_b == "Transitional":
            mask_b = meta["Lifestyle"] == "Transitional"
        else:
            mask_b = meta["Lifestyle"].isin(["H-G", "Transitional"])

        # Get log10 depths, excluding zeros
        bp_a_all = total_bp.loc[meta.index[mask_a]]
        bp_b_all = total_bp.loc[meta.index[mask_b]]
        log_a_pre = np.log10(bp_a_all[bp_a_all > 0])
        log_b_pre = np.log10(bp_b_all[bp_b_all > 0])
        n_a_zero = int((bp_a_all <= 0).sum())
        n_b_zero = int((bp_b_all <= 0).sum())

        bp_a_matched = total_bp.loc[ids_a]
        bp_b_matched = total_bp.loc[ids_b]
        log_a_post = np.log10(bp_a_matched[bp_a_matched > 0])
        log_b_post = np.log10(bp_b_matched[bp_b_matched > 0])

        # Compute bins from valid data
        all_valid = pd.concat([log_a_pre, log_b_pre]).dropna()
        if len(all_valid) == 0:
            continue
        bins = np.linspace(all_valid.quantile(0.01),
                           all_valid.quantile(0.99), 40)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        ax = axes[0]
        ax.hist(log_a_pre, bins=bins,
                alpha=0.5, density=True, color=config.OI["vermillion"],
                label=f"{lab_a} (n={len(log_a_pre):,})")
        ax.hist(log_b_pre, bins=bins,
                alpha=0.5, density=True, color=config.OI["blue"],
                label=f"{lab_b} (n={len(log_b_pre):,})")
        ax.set_title("Before matching")
        ax.set_xlabel("log$_{10}$(Total mapped bp)")
        ax.legend(fontsize=FS-2)
        if n_a_zero + n_b_zero > 0:
            ax.text(0.02, 0.95,
                    f"Excluded: {n_a_zero} {lab_a} + {n_b_zero} {lab_b}\n"
                    f"with 0 mapped bp",
                    transform=ax.transAxes, fontsize=FS-3, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              alpha=0.8))

        # After matching — also show per-pair frac diff distribution
        ax = axes[1]
        ax.hist(log_a_post, bins=bins,
                alpha=0.5, density=True, color=config.OI["vermillion"],
                label=f"{lab_a} matched (n={len(ids_a):,})")
        ax.hist(log_b_post, bins=bins,
                alpha=0.5, density=True, color=config.OI["blue"],
                label=f"{lab_b} matched (n={len(ids_b):,})")
        ax.set_title("After 1:1 matching")
        ax.set_xlabel("log$_{10}$(Total mapped bp)")
        ax.legend(fontsize=FS-2)

        # Annotate with matching quality
        st = dm.stats.get(tag, {})
        if "median_frac_diff" in st:
            ax.text(0.02, 0.95,
                    f"frac_tol={st['frac_tol']:.0%}\n"
                    f"median diff={st['median_frac_diff']:.4f}\n"
                    f"max diff={st['max_frac_diff']:.4f}",
                    transform=ax.transAxes, fontsize=FS-3, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              alpha=0.8))

        fig.suptitle(f"Depth-matching QC: {lab_a} vs {lab_b}", fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(outdir, f"depth_matching_qc_{tag}.png")
        savefig_dual(fig, fpath, dpi=300)
        plt.close(fig)
        print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_section1(data, dm, matched_idx_all, stats,
                 all_specs=None, main_specs=None, stop_after=None,
                 am=None):
    """
    Run all Section 1 analyses.

    Parameters
    ----------
    stop_after : str or None
        If "alpha", stop after 1B″ (robustness figure) — skips beta
        diversity, accumulation curves, and depth-matching QC.
    am : AnchoredMatchResult or None
        If provided, run anchored 1-to-K depth-matched analyses.
    """
    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 1: Diversity & Composition [{ref}]")
    print(f"{'='*60}")

    if all_specs is None:
        from .detection import breadth_pct_specs
        all_specs = breadth_pct_specs(config.ACCUM_BREADTH_THRESHOLDS)
    if main_specs is None:
        main_specs = [DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)]

    s1_main = config.section_dir(ref, "s1", "main")
    s1_supp = config.section_dir(ref, "s1", "supplement")

    # 1A. Dataset overview
    print("\n  1A. Dataset overview & sequencing depth")
    compute_dataset_overview(data, stats)

    # 1B. Alpha diversity — depth-matched, 300kb + 10% breadth thresholds
    print("\n  1B. Alpha diversity (1:1 depth-matched, >=300 kb + >=10% breadth)")
    if dm is not None:
        run_alpha_matched(data, dm, stats)
    else:
        print("    [WARN] No depth-matching — skipping alpha")

    # 1B′. Genome-length corrected alpha (redistribution)
    print("\n  1B′. Alpha diversity — genome-length corrected")
    if dm is not None:
        run_redistribution_corrected_alpha(data, dm, stats)

    # 1B″. Composite figure + triplet health richness
    if dm is not None:
        from .section1_figure import run_section1_figure
        run_section1_figure(data, dm, stats)

    if stop_after == "alpha":
        print(f"\n  Section 1 stopped after alpha (--only s1_alpha) [{ref}]")
        return

    # 1C. Beta diversity
    print("\n  1C. Beta diversity (PCoA + PERMANOVA, depth-matched)")
    if dm is not None:
        compute_beta_diversity_matched(data, dm, stats)

    # 1D. Genome-length bias stat
    print("\n  1D. Genome-length bias (reads-weighted mean)")
    if dm is not None:
        compute_genome_length_stat(data, dm, stats)

    # 1D′. Trace export (CSVs for standalone figure generation)
    if dm is not None:
        from .section1_traces import run_section1_traces
        run_section1_traces(data, dm, stats)

    # 1D″. Anchored 1-to-K depth-matched analyses (H-G-anchored + NI-anchored)
    if am is not None and (am.hg_anchor_ids or am.ni_anchor_ids):
        print("\n  1D″. Anchored 1-to-K depth-matched analyses")
        from .section1_anchored import run_anchored_analyses
        run_anchored_analyses(data, am, stats)

    if stop_after == "traces":
        print(f"\n  Section 1 stopped after traces (--only s1_traces) [{ref}]")
        return

    # 1E. Accumulation curves
    print("\n  1E. Accumulation curves (all + matched)")
    run_accumulation_with_stats(data, dm, matched_idx_all, stats,
                                all_specs=all_specs, main_specs=main_specs)

    # 1F. Depth-matching QC
    print("\n  1F. Depth-matching QC")
    if dm is not None:
        plot_depth_matching_qc(data, dm, s1_supp)

    # 1G. GMW depth-corrected richness (standalone figure)
    #     Runs on ALL samples — IPW handles confounding by design, no
    #     samples are discarded as in 1:1 matching.
    print("\n  1G. GMW depth-corrected richness")
    try:
        from .depth_bias_ipw import run_gmw_richness
        run_gmw_richness(
            data, stats,
            specs=main_specs if main_specs else None,
            n_boot=getattr(config, "OW_AIPWE_N_BOOT_MAIN", 500))
    except Exception as _e:
        print(f"    [WARN] GMW richness skipped: {_e}")

    # 1H. GMW threshold robustness sweep (supplement)
    print("\n  1H. GMW threshold robustness sweep")
    try:
        from .depth_bias_ipw import run_gmw_threshold_sweep
        run_gmw_threshold_sweep(
            data, stats,
            all_specs=all_specs if all_specs else None,
            n_boot=getattr(config, "OW_AIPWE_N_BOOT_SWEEP", 200))
    except Exception as _e:
        print(f"    [WARN] GMW sweep skipped: {_e}")

    # 1I. Pairwise depth-corrected richness  (Ind/Trans, Ind/H-G, Trans/H-G)
    #     Uses its own per-pair geometric-mean target — avoids the 3-way
    #     problem where the shared target lands in a depth range that none
    #     of the three groups actually occupies.
    print("\n  1I. Pairwise depth-corrected richness (all 3 lifestyle pairs)")
    try:
        from .depth_bias_ipw import run_geomean_richness_pairwise
        run_geomean_richness_pairwise(
            data, stats,
            specs=main_specs if main_specs else None,
            n_boot=getattr(config, "OW_AIPWE_N_BOOT_MAIN", 500))
    except Exception as _e:
        print(f"    [WARN] Pairwise richness skipped: {_e}")

    print(f"\n  Section 1 complete [{ref}]")
