"""
3-way GMW-weighted PCoA: H-G / Transitional / Industrial.

Problem with the standard 2-group depth-matched PCoA
------------------------------------------------------
The existing ``compute_beta_diversity_matched`` collapses H-G and
Transitional into a single "Non-Industrial" pool and 1:1 depth-matches
against Industrial.  H-G samples are much deeper on average, so after
matching they are either excluded (too few pairs) or swamped.  The result
is a PCoA that cannot resolve H-G from Transitional in PC space.

Solution
--------
Use 3-way geometric-mean-target (GMW) importance weights so that all three
groups — H-G, Transitional, and Industrial — contribute equally as
``attractors`` in the distance matrix without discarding any samples.

The 3-way GMW target is the cubic-root of the product of the three depth
KDEs: target(x) = (f_HG(x) · f_Trans(x) · f_Ind(x))^(1/3).

Per-sample weights are:
    w_g(x) = target(x) / f_g(x)  ∝  [f_other_a · f_other_b / f_g^2]^(1/3)

Mean-normalised to 1, clipped at WEIGHT_CLIP.

These weights are applied to the Bray–Curtis distance matrix via
weighted double-centering (see Legendre & Gallagher 2001), producing a
weighted PCoA that treats the three groups symmetrically regardless of
their original sample sizes and depth imbalances.

For the figure, samples are also shown with transparency proportional to
their importance weight so low-weight samples (poor depth overlap) recede
visually.

Outputs
-------
results/<ref>/section1_diversity/main/pcoa_gmw3_<spec>.csv
    PCoA coordinates + Lifestyle + importance_weight for every sample.

results/<ref>/section1_diversity/main/pcoa_gmw3_<spec>.png
    Scatter (PCo1 × PCo2) coloured by H-G / Transitional / Industrial.
    Point alpha ∝ importance weight.
    Convex hulls per group.
    Inset ESS and PERMANOVA statistics.

results/<ref>/stats_report.csv entries:
    pcoa_gmw3_var_axis1_<spec>    — % variance on PCo1
    pcoa_gmw3_var_axis2_<spec>    — % variance on PCo2
    pcoa_gmw3_permanova_<spec>    — PERMANOVA F and p

Standalone usage
----------------
    python -m persistence --ref UHGG --only s1_pcoa3

    # or directly:
    from persistence.pcoa_gmw3 import run_pcoa_gmw3
    run_pcoa_gmw3(data, stats)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from . import config
from .detection import DetectionSpec, build_presence
from .depth_bias_ipw import _BinnedKDE, _ess
from .plot_utils import apply_oi_style, savefig_dual, current_fs

apply_oi_style()

LIFESTYLE_ORDER = ["H-G", "Transitional", "Industrial"]
WEIGHT_CLIP     = 10.0   # mirrors depth_bias_ipw.WEIGHT_CLIP
N_PERM_DEFAULT  = 999
MAX_SAMPLES     = 2000   # subsample cap for BC distance matrix (speed)


# ── 3-way GMW weights ─────────────────────────────────────────────────────────

def _gmw3_weights(log_depth: np.ndarray,
                  kde_own: _BinnedKDE,
                  kde_b: _BinnedKDE,
                  kde_c: _BinnedKDE,
                  clip: float = WEIGHT_CLIP) -> np.ndarray:
    """
    w_i = (f_b(x_i) · f_c(x_i) / f_own(x_i)^2)^(1/3)
    Mean-normalised to 1, clipped at `clip`.
    """
    f_own = np.maximum(kde_own(log_depth), 1e-10)
    f_b   = np.maximum(kde_b(log_depth),   1e-10)
    f_c   = np.maximum(kde_c(log_depth),   1e-10)
    w = (f_b * f_c / f_own ** 2) ** (1.0 / 3.0)
    w = w / w.mean()
    return np.clip(w, 0.0, clip)


# ── Weighted PCoA (weighted double-centering) ──────────────────────────────────

def _weighted_pcoa(D_sq: np.ndarray,
                   weights: np.ndarray,
                   n_axes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted PCoA via weighted double-centering (Legendre & Gallagher 2001).

    Parameters
    ----------
    D_sq    : (n, n) squared distance matrix
    weights : (n,) importance weights, need not sum to 1
    n_axes  : number of axes to return

    Returns
    -------
    coords        : (n, n_axes) PCoA coordinates
    var_explained : (n_axes,) % variance explained per axis
    """
    w = weights / weights.sum()            # normalise to sum=1
    W = np.diag(w)

    # Weighted double-centring
    D2   = D_sq ** 2
    Ic   = np.eye(len(w)) - np.ones((len(w), len(w))) * w      # row-centring
    B    = -0.5 * Ic @ D2 @ Ic.T

    # Weighted symmetric form: W^(1/2) B W^(1/2)
    W12  = np.diag(np.sqrt(w))
    Bw   = W12 @ B @ W12

    try:
        from scipy.linalg import eigh as scipy_eigh
        n = len(w)
        start = max(0, n - n_axes)
        eigenvalues, eigenvectors = scipy_eigh(
            Bw, subset_by_index=[start, n - 1])
        eigenvalues  = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(Bw)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx[:n_axes]]
        eigenvectors = eigenvectors[:, idx[:n_axes]]

    # Back-transform: V = W^(-1/2) * eigvec * sqrt(max(λ,0))
    W12_inv = np.diag(1.0 / np.sqrt(np.maximum(w, 1e-300)))
    top_ev   = np.maximum(eigenvalues[:n_axes], 0.0)
    coords   = W12_inv @ eigenvectors[:, :n_axes] * np.sqrt(top_ev)

    total_var = max(np.sum(np.maximum(eigenvalues, 0.0)), 1e-10)
    var_explained = top_ev / total_var * 100.0

    return coords, var_explained


# ── PERMANOVA (3-group) ───────────────────────────────────────────────────────

def _permanova3(D_sq: np.ndarray,
                labels: np.ndarray,
                n_perm: int = N_PERM_DEFAULT,
                rng_seed: int = 42) -> tuple[float, float]:
    """Label-permutation PERMANOVA, ≥ 2 groups."""
    rng = np.random.default_rng(rng_seed)
    n = D_sq.shape[0]
    unique_groups = np.unique(labels)
    a = len(unique_groups)
    if a < 2:
        return np.nan, np.nan

    def _F(labs):
        ss_w = 0.0
        for g in unique_groups:
            idx = np.where(labs == g)[0]
            if len(idx) < 2:
                continue
            sub = D_sq[np.ix_(idx, idx)]
            ss_w += sub.sum() / (2 * len(idx))
        ss_t = D_sq.sum() / (2 * n)
        ss_a = ss_t - ss_w
        if ss_w == 0:
            return np.inf
        return (ss_a / (a - 1)) / (ss_w / (n - a))

    F_obs = _F(labels)
    n_ge  = 1
    for _ in range(n_perm):
        if _F(rng.permutation(labels)) >= F_obs:
            n_ge += 1
    return F_obs, n_ge / (n_perm + 1)


# ── Figure ────────────────────────────────────────────────────────────────────

def _weighted_geometric_median(x, y, w, n_iter=30):
    """
    Weiszfeld algorithm for the weighted geometric median.
    More robust than weighted mean when a group has a heavy-tailed depth
    distribution (e.g. H-G, which has a few very deep samples that the
    GMW weights upweight and that can pull the mean toward Transitional).
    """
    w = w / w.sum()
    cx, cy = float(np.dot(w, x)), float(np.dot(w, y))
    for _ in range(n_iter):
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist = np.maximum(dist, 1e-8)
        ww   = w / dist
        ww  /= ww.sum()
        cx_new = float(np.dot(ww, x))
        cy_new = float(np.dot(ww, y))
        if abs(cx_new - cx) + abs(cy_new - cy) < 1e-7:
            break
        cx, cy = cx_new, cy_new
    return cx, cy


def _kde_contour(ax, x, y, weights, color,
                 levels=(0.90, 0.50),
                 bw_method="scott"):
    """
    Draw weighted KDE contours at the given probability-mass levels.
    levels=(0.90, 0.50) → outer contour encloses ~90% of mass,
                           inner contour encloses ~50% of mass.
    Uses scipy.stats.gaussian_kde with importance weights.
    """
    from scipy.stats import gaussian_kde
    pts = np.vstack([x, y])
    if pts.shape[1] < 5:
        return
    w = np.maximum(weights, 0.0)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return
    w = w / w_sum

    try:
        kde = gaussian_kde(pts, bw_method=bw_method, weights=w)
    except Exception:
        return

    # Evaluate on a grid
    margin = 0.05
    xlo, xhi = x.min() - margin, x.max() + margin
    ylo, yhi = y.min() - margin, y.max() + margin
    xi = np.linspace(xlo, xhi, 150)
    yi = np.linspace(ylo, yhi, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    # Convert density levels to probability-mass iso-contours
    # Sort all grid values descending; find the value at which cumsum
    # of (normalised) density reaches each probability level.
    z_flat   = Zi.ravel()
    order    = np.argsort(z_flat)[::-1]
    z_sorted = z_flat[order]
    z_cumsum = np.cumsum(z_sorted) / z_sorted.sum()

    contour_vals = []
    for level in sorted(levels):             # inner → outer
        idx = np.searchsorted(z_cumsum, level)
        contour_vals.append(z_sorted[min(idx, len(z_sorted)-1)])
    contour_vals = sorted(set(contour_vals))  # ascending for contourf

    if not contour_vals:
        return

    # Filled contour (outer level, low alpha) + line contours
    alphas_fill = [0.10, 0.18]   # outer fills are more transparent
    for i, (lo, hi) in enumerate(zip(
            [Zi.min()] + contour_vals[:-1], contour_vals)):
        alpha_f = alphas_fill[min(i, len(alphas_fill)-1)]
        ax.contourf(Xi, Yi, Zi, levels=[lo, hi],
                    colors=[color], alpha=alpha_f, zorder=1)

    # Contour lines
    line_styles = ["--", "-"]
    line_widths = [0.8, 1.4]
    line_alphas = [0.45, 0.70]
    for j, val in enumerate(contour_vals):
        ax.contour(Xi, Yi, Zi, levels=[val],
                   colors=[color],
                   linestyles=[line_styles[min(j, 1)]],
                   linewidths=[line_widths[min(j, 1)]],
                   alpha=line_alphas[min(j, 1)],
                   zorder=2)


def _plot_pcoa_gmw3(pcoa_df: pd.DataFrame,
                    var_explained: np.ndarray,
                    ess_by_lf: dict,
                    F_obs: float, p_perm: float,
                    spec: DetectionSpec,
                    outdir: str,
                    spec_tag: str) -> None:
    """
    Two-panel figure:
      Left  — PCo1 × PCo2
      Right — PCo1 × PCo3 (if available)

    Visualization:
      - Smooth weighted-KDE contours (50% and 90% probability-mass levels)
        replace convex hulls.  Contours show where the *density* of each
        lifestyle actually lives, rather than its worst-case bounding box,
        so overlapping sparse tails no longer obscure the core separation.
      - Centroids use the weighted geometric median (Weiszfeld algorithm)
        rather than the weighted mean.  The GMW scheme upweights shallower
        H-G samples to match the Industrial depth range; those samples
        happen to be microbiome-closer to Transitional, so the weighted mean
        centroid gets pulled toward Transitional.  The geometric median is
        equivariant and resists that pull.
    """
    FS = current_fs()
    n_axes   = pcoa_df.filter(like="PCo").shape[1]
    n_panels = 2 if n_axes >= 3 else 1

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(10 * n_panels, 9),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    panel_pairs = [(0, 1)] + ([(0, 2)] if n_axes >= 3 else [])

    for ax, (ax_x, ax_y) in zip(axes, panel_pairs):
        col_x = f"PCo{ax_x + 1}"
        col_y = f"PCo{ax_y + 1}"
        vx    = var_explained[ax_x] if ax_x < len(var_explained) else 0
        vy    = var_explained[ax_y] if ax_y < len(var_explained) else 0

        for lf in LIFESTYLE_ORDER:
            sub   = pcoa_df[pcoa_df["Lifestyle"] == lf]
            if sub.empty:
                continue
            color = config.LIFESTYLE_COLORS.get(lf, "gray")
            n_lf  = len(sub)
            ess   = ess_by_lf.get(lf, 0.0)

            xv = sub[col_x].values
            yv = sub[col_y].values

            # ── Importance weights for KDE and centroid ───────────────────
            if "importance_weight" in sub.columns:
                ww = sub["importance_weight"].values.astype(float)
            else:
                ww = np.ones(n_lf)
            ww = np.maximum(ww, 0.0)

            # ── Scatter: alpha ∝ normalised importance weight ─────────────
            w_norm = ww / max(ww.max(), 1e-9)
            alphas = np.clip(0.08 + 0.45 * w_norm, 0.04, 0.65)
            rgba   = np.array(list(matplotlib.colors.to_rgba(color)))
            rgba_arr = np.tile(rgba, (n_lf, 1))
            rgba_arr[:, 3] = alphas
            ax.scatter(xv, yv, c=rgba_arr, s=6,
                       edgecolors="none", zorder=3)

            # ── Weighted KDE contours (90% and 50% mass levels) ───────────
            _kde_contour(ax, xv, yv, ww, color,
                         levels=(0.90, 0.50))

            # ── Weighted geometric-median centroid ────────────────────────
            cx, cy = _weighted_geometric_median(xv, yv, ww)
            ax.scatter(cx, cy, c=color, s=160, marker="D",
                       edgecolors="black", linewidths=1.0,
                       zorder=6,
                       label=(f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} "
                              f"(n={n_lf:,}, ESS={ess:.0f})"))

        ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.35)
        ax.axvline(0, color="gray", lw=0.4, ls="--", alpha=0.35)
        ax.set_xlabel(f"{col_x} ({vx:.1f}%)", fontsize=FS)
        ax.set_ylabel(f"{col_y} ({vy:.1f}%)", fontsize=FS)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=FS - 2, markerscale=1.5, loc="best")

    p_str = (f"p = {p_perm:.4f}" if not np.isnan(p_perm) else "p = n/a")
    F_str = (f"F = {F_obs:.2f}"  if not np.isnan(F_obs)  else "F = n/a")
    fig.suptitle(
        f"3-way GMW-weighted PCoA — {spec.description}\n"
        f"Bray–Curtis; cubic-root depth target equalises H-G / Trans / Ind\n"
        f"PERMANOVA ({F_str}, {p_str}, n_perm={N_PERM_DEFAULT})  |  "
        f"diamond = weighted geometric-median centroid  |  "
        f"contours = 50% and 90% KDE mass  |  α ∝ importance weight",
        fontsize=FS + 1, fontweight="bold")

    png_path = os.path.join(outdir, f"pcoa_gmw3_{spec_tag}.png")
    savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {png_path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pcoa_gmw3(data, stats,
                  specs=None,
                  max_samples: int = MAX_SAMPLES,
                  n_perm: int = N_PERM_DEFAULT,
                  rng_seed: int = 42):
    """
    3-way GMW-weighted PCoA: H-G / Transitional / Industrial.

    Parameters
    ----------
    data        : ReferenceData  (post-reinstatement)
    stats       : StatsReport
    specs       : list[DetectionSpec] | None  (defaults to both main-text specs)
    max_samples : int  subsample cap per group for the distance matrix
    n_perm      : int  PERMANOVA permutations
    rng_seed    : int
    """
    ref    = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")
    os.makedirs(outdir, exist_ok=True)
    meta   = data.meta
    rng    = np.random.default_rng(rng_seed)

    if specs is None:
        specs = [
            DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    print(f"\n{'='*60}")
    print(f"  3-way GMW PCoA [{ref}]")
    print(f"{'='*60}")

    # ── Group membership ──────────────────────────────────────────────────────
    grp_ids: dict[str, list] = {}
    for lf in LIFESTYLE_ORDER:
        ids = [s for s in meta.index
               if s in data.breadth.index and s in data.depth.index
               and meta.loc[s, "Lifestyle"] == lf]
        grp_ids[lf] = ids
        print(f"  {lf}: {len(ids):,} samples")

    if any(len(v) < 10 for v in grp_ids.values()):
        print("  [WARN] One or more groups has < 10 samples — skipping")
        return

    # ── Log depth per group ───────────────────────────────────────────────────
    def _ld(ids):
        return np.log10(np.maximum(
            data.total_bp.reindex(ids).fillna(1.0).values, 1.0))

    log_depth = {lf: _ld(grp_ids[lf]) for lf in LIFESTYLE_ORDER}

    # ── KDEs ─────────────────────────────────────────────────────────────────
    kde = {lf: _BinnedKDE(log_depth[lf]) for lf in LIFESTYLE_ORDER}

    # ── 3-way GMW weights ─────────────────────────────────────────────────────
    lf_a, lf_b, lf_c = LIFESTYLE_ORDER
    weights: dict[str, np.ndarray] = {
        lf_a: _gmw3_weights(log_depth[lf_a], kde[lf_a], kde[lf_b], kde[lf_c]),
        lf_b: _gmw3_weights(log_depth[lf_b], kde[lf_b], kde[lf_a], kde[lf_c]),
        lf_c: _gmw3_weights(log_depth[lf_c], kde[lf_c], kde[lf_a], kde[lf_b]),
    }

    ess_by_lf = {lf: _ess(weights[lf]) for lf in LIFESTYLE_ORDER}
    for lf in LIFESTYLE_ORDER:
        print(f"  {lf}: ESS = {ess_by_lf[lf]:.0f} / {len(grp_ids[lf]):,}")

    for spec in specs:
        _run_one_spec(
            data, meta, grp_ids, log_depth, weights, ess_by_lf,
            spec, outdir, ref, stats, max_samples, n_perm, rng)


def _run_one_spec(data, meta, grp_ids, log_depth, weights, ess_by_lf,
                  spec, outdir, ref, stats,
                  max_samples, n_perm, rng):
    spec_tag = spec.short_label
    print(f"\n  [{spec_tag}] {spec.description}")

    # ── Subsample within each group, weighted ─────────────────────────────────
    # Draw from each group proportional to importance weight so the most
    # depth-compatible samples are preferentially selected.  H-G is small
    # enough that we always include all of it; larger groups are capped.
    sel_ids:    list[str]   = []
    sel_weights: list[float] = []
    sel_lf:     list[str]   = []

    for lf in LIFESTYLE_ORDER:
        ids = grp_ids[lf]
        w   = weights[lf]
        n   = len(ids)
        if n <= max_samples:
            sel_ids.extend(ids)
            sel_weights.extend(w.tolist())
            sel_lf.extend([lf] * n)
        else:
            # Weighted sampling without replacement, capped at max_samples
            w_prob = w / w.sum()
            idx    = rng.choice(n, size=max_samples, replace=False, p=w_prob)
            sel_ids.extend([ids[i] for i in idx])
            sel_weights.extend(w[idx].tolist())
            sel_lf.extend([lf] * max_samples)
            print(f"    {lf}: subsampled {max_samples:,} from {n:,} "
                  f"(weighted, ESS={ess_by_lf[lf]:.0f})")

    n_total = len(sel_ids)
    print(f"    Total samples in distance matrix: {n_total:,}")

    # ── Build abundance matrix and Bray–Curtis distances ──────────────────────
    presence = build_presence(
        data.breadth.loc[sel_ids],
        data.depth.loc[sel_ids],
        data.lengths,
        spec=spec)
    abund = data.abundance.loc[sel_ids]
    vals  = np.nan_to_num(abund.values, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Computing Bray–Curtis distances ({n_total}×{n_total})...")
    D = pdist(vals, metric="braycurtis")
    D = np.nan_to_num(D, nan=0.0)
    D_sq = squareform(D)

    # ── Weighted PCoA ─────────────────────────────────────────────────────────
    w_arr = np.array(sel_weights, dtype=float)
    n_axes = min(config.DIVERSITY_PCOA_AXES, n_total - 1)
    coords, var_explained = _weighted_pcoa(D_sq, w_arr, n_axes=n_axes)

    print(f"    PCo1: {var_explained[0]:.1f}%  "
          f"PCo2: {var_explained[1]:.1f}%"
          + (f"  PCo3: {var_explained[2]:.1f}%"
             if len(var_explained) >= 3 else ""))

    # ── PERMANOVA ─────────────────────────────────────────────────────────────
    print(f"    Running PERMANOVA (n_perm={n_perm})...")
    labels_arr = np.array(sel_lf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F_obs, p_perm = _permanova3(D_sq, labels_arr, n_perm=n_perm)
    print(f"    PERMANOVA: F={F_obs:.3f}, p={p_perm:.4f}")

    # ── Stats report ──────────────────────────────────────────────────────────
    for i, pct in enumerate(var_explained[:3]):
        stats.add(
            f"pcoa_gmw3_var_axis{i+1}_{spec_tag}", "s1", "main",
            f"3-way GMW PCoA axis {i+1} variance [{spec.description}]",
            f"{pct:.2f}%")
    stats.add(
        f"pcoa_gmw3_permanova_{spec_tag}", "s1", "main",
        f"3-way GMW PCoA PERMANOVA [{spec.description}]",
        f"F={F_obs:.3f}, p={p_perm:.4f}, n_perm={n_perm}",
        test="PERMANOVA", p_value=p_perm, effect_size=F_obs,
        n1=n_total)
    for lf in LIFESTYLE_ORDER:
        stats.add(
            f"pcoa_gmw3_ess_{lf}_{spec_tag}", "s1", "supplement",
            f"3-way GMW PCoA ESS for {lf} [{spec.description}]",
            f"ESS={ess_by_lf[lf]:.0f} / n={len(grp_ids[lf]):,}",
            n1=len(grp_ids[lf]))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    pcoa_df = pd.DataFrame(
        coords, index=sel_ids,
        columns=[f"PCo{i+1}" for i in range(n_axes)])
    pcoa_df["Lifestyle"]         = sel_lf
    pcoa_df["importance_weight"] = sel_weights
    pcoa_df["log10_total_bp"]    = [
        float(np.log10(max(data.total_bp.get(s, 1.0), 1.0)))
        for s in sel_ids]

    csv_path = os.path.join(outdir, f"pcoa_gmw3_{spec_tag}.csv")
    pcoa_df.to_csv(csv_path)
    print(f"    Saved: {csv_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    _plot_pcoa_gmw3(
        pcoa_df, var_explained, ess_by_lf,
        F_obs, p_perm, spec, outdir, spec_tag)
