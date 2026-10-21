"""
Depth-dependence audit plugin.

Resolves Rachel's Ruhlemann-style comment in the "Testing for extinction"
section, which asks for explicit correlations between sequencing depth and
recoverable diversity-related quantities — analogous to the assembly-size /
bins-recovered analysis in metagenome-assembled-genome studies.

Because this pipeline is reference-based rather than assembly-based, the
closest analogues are:

    library_size   → total mapped bp  (``total_bp``)
    assembly_size  → represented genome bp per sample
                     (``sum_j [ length_j × I(presence_{ij}) ]``)
    bins_recovered → per-sample richness (SGB count)

Three Spearman correlations, overall and by lifestyle:

    DDA-1  log10(total_bp)      vs  richness
    DDA-2  log10(total_bp)      vs  log10(represented_genome_bp)
    DDA-3  log10(repr_genome_bp) vs  richness

All computed under both main-text detection specs (breadth / flat-bp).

Outputs
-------
results/<ref>/depth_audit/depth_dependence_audit_<spec>.csv
    One row per (Lifestyle, correlation pair) with Spearman ρ, p, n.

results/<ref>/depth_audit/depth_dependence_audit_<spec>.png
    Three-panel scatter (library size vs richness, library size vs repr.
    genome, repr. genome vs richness) with LOWESS smoothers, coloured by
    lifestyle.

results/<ref>/stats_report.csv (via StatsReport)
    Stable tags:
        depth_vs_richness_spearman_all_<spec>
        depth_vs_reprbp_spearman_all_<spec>
        reprbp_vs_richness_spearman_all_<spec>
        … and per-lifestyle equivalents.

Integration
-----------
Add to run_reference() in __main__.py immediately after run_depth_audit()::

    from .depth_dependence_audit import run_depth_dependence_audit
    run_depth_dependence_audit(data, stats)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .plot_utils import apply_oi_style, savefig_dual, current_fs
apply_oi_style()

from . import config
from .detection import DetectionSpec, build_presence

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.grid": True,
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

LIFESTYLE_ORDER = ["H-G", "Transitional", "Industrial"]


# ── LOWESS smoother (pure-numpy, no statsmodels dep) ─────────────────────────

def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.4, n_pts: int = 200):
    """Weighted least-squares LOWESS smoother.  Returns (x_smooth, y_smooth)."""
    if len(x) < 4:
        return x, y
    x_eval = np.linspace(x.min(), x.max(), n_pts)
    y_smooth = np.empty(n_pts)
    h = np.sort(np.abs(x - np.median(x)))[int(frac * len(x))]
    h = max(h, 1e-6)
    for i, xi in enumerate(x_eval):
        dist = np.abs(x - xi)
        w = np.maximum(0.0, 1.0 - (dist / (1.5 * dist.max() + 1e-12)) ** 3) ** 3
        # Weighted LS
        W = np.diag(w)
        A = np.column_stack([np.ones(len(x)), x])
        try:
            coef = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
            y_smooth[i] = coef[0] + coef[1] * xi
        except np.linalg.LinAlgError:
            y_smooth[i] = np.nan
    return x_eval, y_smooth


# ── Represented genome bp ────────────────────────────────────────────────────

def _compute_repr_genome_bp(
    presence: pd.DataFrame,
    lengths: pd.Series,
) -> pd.Series:
    """
    For each sample, sum the genome lengths of all detected taxa.

    repr_genome_bp_i = sum_j [ length_j × presence_{ij} ]
    """
    shared_taxa = presence.columns.intersection(lengths.index)
    len_vec = lengths.reindex(shared_taxa).fillna(0).values
    pres_sub = presence[shared_taxa].values.astype(float)
    repr_bp = pres_sub @ len_vec
    return pd.Series(repr_bp, index=presence.index, name="repr_genome_bp")


# ── Spearman helper ──────────────────────────────────────────────────────────

def _spearman_row(
    x: np.ndarray,
    y: np.ndarray,
    label_x: str,
    label_y: str,
    lifestyle: str,
    spec_tag: str,
) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return {
            "Lifestyle": lifestyle,
            "x_var": label_x,
            "y_var": label_y,
            "spec": spec_tag,
            "n": n,
            "rho": np.nan,
            "p": np.nan,
        }
    rho, p = spearmanr(x[mask], y[mask])
    return {
        "Lifestyle": lifestyle,
        "x_var": label_x,
        "y_var": label_y,
        "spec": spec_tag,
        "n": n,
        "rho": round(float(rho), 4),
        "p": float(p),
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def run_depth_dependence_audit(data, stats, specs=None):
    """
    Run the depth-dependence audit for all main-text detection specs.

    Parameters
    ----------
    data   : ReferenceData
    stats  : StatsReport
    specs  : list[DetectionSpec] | None
    """
    ref = data.ref_name
    outdir = os.path.join("results", ref, "depth_audit")
    os.makedirs(outdir, exist_ok=True)

    if specs is None:
        specs = [
            DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    print(f"\n{'='*60}")
    print(f"  Depth-dependence audit [{ref}]")
    print(f"{'='*60}")

    for spec in specs:
        _run_one_spec(data, stats, spec, outdir)


def _run_one_spec(data, stats, spec: DetectionSpec, outdir: str) -> None:
    spec_tag = spec.short_label
    ref = data.ref_name
    meta = data.meta

    print(f"\n  [{spec_tag}] {spec.description}")

    # ── Build presence matrix over all valid samples ─────────────────────────
    valid_ids = meta.index.tolist()
    # Filter to samples present in breadth/depth
    valid_ids = [s for s in valid_ids
                 if s in data.breadth.index and s in data.depth.index]
    if not valid_ids:
        print("    [DDA] No valid samples — skipping")
        return

    presence = build_presence(
        data.breadth.loc[valid_ids],
        data.depth.loc[valid_ids],
        data.lengths,
        spec=spec,
    )
    richness = presence.sum(axis=1).rename("richness")
    repr_bp = _compute_repr_genome_bp(presence, data.lengths)
    total_bp = data.total_bp.reindex(valid_ids)

    # log-transform (add 1 to avoid log(0))
    log_total = np.log10(total_bp.clip(lower=1))
    log_repr  = np.log10(repr_bp.clip(lower=1))

    # Lifestyle labels
    lifestyle = meta.reindex(valid_ids)["Lifestyle"].fillna("Unknown")

    # ── Correlations ─────────────────────────────────────────────────────────
    corr_rows: list[dict] = []

    # Overall
    for label_x, xvals in [
        ("log10_total_bp", log_total.values),
        ("log10_repr_genome_bp", log_repr.values),
    ]:
        for label_y, yvals in [
            ("richness", richness.values),
            ("log10_repr_genome_bp", log_repr.values),
        ]:
            if label_x == label_y:
                continue
            # skip log10_repr vs richness twice
            if label_x == "log10_repr_genome_bp" and label_y == "log10_repr_genome_bp":
                continue
            row = _spearman_row(xvals, yvals, label_x, label_y,
                                "All", spec_tag)
            corr_rows.append(row)

            tag = f"{label_x}_vs_{label_y}_spearman_all_{spec_tag}"
            if np.isfinite(row["rho"]):
                stats.add(
                    tag, "s1", "main",
                    f"Spearman ρ: {label_x} vs {label_y} — all lifestyles "
                    f"({spec.description})",
                    f"ρ={row['rho']:+.3f}, p={row['p']:.3e}, n={row['n']}",
                    test="Spearman rank correlation",
                    p_value=row["p"],
                    effect_size=row["rho"],
                    n1=row["n"],
                )
                print(f"    All | {label_x} vs {label_y}: "
                      f"ρ={row['rho']:+.3f}, p={row['p']:.3e}, n={row['n']}")

    # Per lifestyle
    for lf in LIFESTYLE_ORDER:
        mask = lifestyle == lf
        ids_lf = [s for s, m in zip(valid_ids, mask) if m]
        if len(ids_lf) < 5:
            continue
        idx = [valid_ids.index(s) for s in ids_lf]
        lt = log_total.values[idx]
        lr = log_repr.values[idx]
        ri = richness.values[idx]

        for label_x, xv, label_y, yv in [
            ("log10_total_bp", lt, "richness", ri),
            ("log10_total_bp", lt, "log10_repr_genome_bp", lr),
            ("log10_repr_genome_bp", lr, "richness", ri),
        ]:
            row = _spearman_row(xv, yv, label_x, label_y, lf, spec_tag)
            corr_rows.append(row)
            tag = f"{label_x}_vs_{label_y}_spearman_{lf}_{spec_tag}"
            if np.isfinite(row["rho"]):
                stats.add(
                    tag, "s1", "supplement",
                    f"Spearman ρ: {label_x} vs {label_y} — {lf} only "
                    f"({spec.description})",
                    f"ρ={row['rho']:+.3f}, p={row['p']:.3e}, n={row['n']}",
                    test="Spearman rank correlation",
                    p_value=row["p"],
                    effect_size=row["rho"],
                    n1=row["n"],
                )

    # ── Save CSV ─────────────────────────────────────────────────────────────
    corr_df = pd.DataFrame(corr_rows)
    csv_path = os.path.join(outdir, f"depth_dependence_audit_{spec_tag}.csv")
    corr_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_depth_dependence(
        log_total, log_repr, richness, lifestyle,
        valid_ids, outdir, spec_tag, spec,
    )


def _plot_depth_dependence(
    log_total: pd.Series,
    log_repr: pd.Series,
    richness: pd.Series,
    lifestyle: pd.Series,
    valid_ids: list,
    outdir: str,
    spec_tag: str,
    spec: DetectionSpec,
) -> None:
    """Three-panel scatter with per-lifestyle colouring and LOWESS smoother."""

    panels = [
        ("log10_total_bp", "richness",
         log_total, richness,
         "log₁₀ total mapped bp",
         "Richness (SGBs detected)",
         "DDA-1: Library size vs richness"),
        ("log10_total_bp", "log10_repr_genome_bp",
         log_total, log_repr,
         "log₁₀ total mapped bp",
         "log₁₀ represented genome bp",
         "DDA-2: Library size vs represented genome"),
        ("log10_repr_genome_bp", "richness",
         log_repr, richness,
         "log₁₀ represented genome bp",
         "Richness (SGBs detected)",
         "DDA-3: Represented genome vs richness"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    FS = current_fs()

    for ax, (_, _, xvals, yvals, xlabel, ylabel, title) in zip(axes, panels):
        # Per-lifestyle scatter
        for lf in LIFESTYLE_ORDER:
            mask = lifestyle == lf
            xm = xvals.values[mask.values]
            ym = yvals.values[mask.values]
            fin = np.isfinite(xm) & np.isfinite(ym)
            color = config.LIFESTYLE_COLORS.get(lf, "gray")
            ax.scatter(
                xm[fin], ym[fin],
                c=color, alpha=0.25, s=8, edgecolors="none",
                label=f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} (n={fin.sum():,})",
            )

        # Overall LOWESS smoother
        xall = xvals.values
        yall = yvals.values
        fin_all = np.isfinite(xall) & np.isfinite(yall)
        if fin_all.sum() > 10:
            order = np.argsort(xall[fin_all])
            xs, ys = _lowess(xall[fin_all][order], yall[fin_all][order])
            ax.plot(xs, ys, color="black", lw=2, zorder=5, label="LOWESS")

        # Overall Spearman annotation
        if fin_all.sum() >= 5:
            rho, p = spearmanr(xall[fin_all], yall[fin_all])
            ax.text(
                0.05, 0.95,
                f"ρ = {rho:+.3f}\np = {p:.2e}",
                transform=ax.transAxes,
                fontsize=FS - 1,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_title(title, fontsize=FS + 1, fontweight="bold")
        ax.legend(fontsize=FS - 2, markerscale=2)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Depth-dependence audit — {spec.description}\n"
        "(Ruhlemann-style: library size → represented genome → richness)",
        fontsize=FS + 2,
        fontweight="bold",
    )
    plt.tight_layout()

    png_path = os.path.join(outdir, f"depth_dependence_audit_{spec_tag}.png")
    savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {png_path}")
