"""
Section 1 figures: composite alpha diversity + robustness.

Produces two figures per reference:
    1. figure_alpha_diversity_composite.png  (main-text quality)
    2. figure_robustness_alpha.png           (supplement)

Robustness figure — three comparisons
--------------------------------------
1. **Uncorrected** — all samples, no depth correction (shows inflated gap).
2. **Depth-matched** — 1:1 nearest-neighbour pairs.
3. **IPW** — inverse-probability weighting to a common depth distribution.

Robustness layout (4 rows x 3 cols)
------------------------------------
    Row 0 DATA:   A. Depth imbalance  | B. Matched bins    | C. IPW dists
    Row 1 300kb:  D. Uncorr           | E. Matched         | F. IPW
    Row 2 10%:    G. Uncorr           | H. Matched         | I. IPW
    Row 3:        J. Forest plot (full width)

Composite layout (2 x 3)
-------------------------
    A. Ind vs NI (300kb)     B. Ind vs NI (10%)      C. Ind vs Trans (10%)
    D. Redistribution        E. Triplet (300kb)       F. Triplet (10%)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import (
    pin_ylabel,
    add_post_layout_hook,
    apply_oi_style,
    current_fs,
    page_figsize,
    PAGE_WIDTH_IN,
    panel_grid,
    savefig_dual,
    share_axis_labels,
    square_figsize,
)
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu, gaussian_kde

from . import config
from .detection import DetectionSpec, build_presence

plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "font.size": 10,
                     "font.family": "sans-serif",
                     # Arial first: macOS Helvetica is a .ttc collection and
                     # matplotlib cannot pick a face inside one, so every
                     # fontweight="bold" in this module rendered as regular.
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

IND_COLOR = config.OI["sky"]       # Industrial (all) = sky blue
NI_COLOR  = config.OI["orange"]    # Non-Industrial = orange


# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════

def _nonnegative_xerr(r, lo, hi):
    """Return Matplotlib-safe asymmetric xerr after normalizing CI bounds."""
    r = np.asarray(r, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ci_lo = np.minimum(lo, hi)
    ci_hi = np.maximum(lo, hi)
    valid = np.isfinite(r) & np.isfinite(ci_lo) & np.isfinite(ci_hi)
    xerr_lo = np.where(valid, np.maximum(r - ci_lo, 0.0), 0.0)
    xerr_hi = np.where(valid, np.maximum(ci_hi - r, 0.0), 0.0)
    return xerr_lo, xerr_hi, ci_lo, ci_hi


def _bootstrap_ratio(ri, rn, n_boot=2000, seed=42):
    """Bootstrap 95% CI for ratio of medians."""
    rng = np.random.default_rng(seed)
    n_i, n_n = len(ri), len(rn)
    br = np.array([
        np.median(ri[rng.integers(0, n_i, n_i)]) /
        max(np.median(rn[rng.integers(0, n_n, n_n)]), 1e-10)
        for _ in range(n_boot)])
    return np.percentile(br, [2.5, 97.5])


def _weighted_median(values, weights):
    """Weighted median via cumulative weight interpolation."""
    o = np.argsort(values)
    sv, sw = values[o], weights[o]
    cum = np.cumsum(sw)
    return sv[min(np.searchsorted(cum, 0.5 * cum[-1]), len(sv) - 1)]


def _adaptive_bins_matched(log_depths_paired, n_bins=18):
    """Equal-count bins on matched-pair depths."""
    edges = np.percentile(log_depths_paired, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 0.01
    edges[-1] += 0.01
    return edges


def _bin_richness(log_d, richness, edges):
    """Per-bin median, IQR, and n."""
    rows = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = ((log_d >= lo) & (log_d < hi)) if j < len(edges) - 2 \
            else ((log_d >= lo) & (log_d <= hi))
        v = richness[mask]
        if len(v) < 2:
            continue
        rows.append(dict(bin_mid=0.5 * (lo + hi), median=np.median(v),
                         q25=np.percentile(v, 25), q75=np.percentile(v, 75),
                         n=len(v)))
    return pd.DataFrame(rows)


def _fmt_p(p):
    """Format p-value with significance stars."""
    if p < 1e-100:
        s = "p < 10⁻¹⁰⁰"
    elif p < 0.001:
        s = f"p = {p:.1e}"
    else:
        s = f"p = {p:.3f}"
    if p < 0.001:
        s += " ***"
    elif p < 0.01:
        s += " **"
    elif p < 0.05:
        s += " *"
    return s


def _sig_bracket(ax, x1, x2, y, p, fs=9):
    """Draw significance bracket between two box positions."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="black")
    ax.text((x1 + x2) / 2, y + h, _fmt_p(p),
            ha="center", va="bottom", fontsize=fs)


# ══════════════════════════════════════════════════════════════════
#  ROBUSTNESS — COMPUTE METHODS
# ══════════════════════════════════════════════════════════════════

def _compute_uncorrected(data, spec):
    """Naive comparison — all samples, no depth correction."""
    meta = data.meta
    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])
    all_ids = meta.index[ind_mask | ni_mask].tolist()

    presence = build_presence(data.breadth.loc[all_ids],
                              data.depth.loc[all_ids],
                              data.lengths, spec=spec)
    richness = presence.sum(axis=1)

    ids_ind = meta.index[ind_mask].tolist()
    ids_ni = meta.index[ni_mask].tolist()
    ri = richness.loc[ids_ind].values
    rn = richness.loc[ids_ni].values
    di = np.log10(data.total_bp.loc[ids_ind].values.astype(float))
    dn = np.log10(data.total_bp.loc[ids_ni].values.astype(float))

    ok_i = np.isfinite(di) & np.isfinite(ri)
    ok_n = np.isfinite(dn) & np.isfinite(rn)
    ri, di = ri[ok_i], di[ok_i]
    rn, dn = rn[ok_n], dn[ok_n]

    ratio = np.median(ri) / max(np.median(rn), 1e-10)
    _, p = mannwhitneyu(ri, rn, alternative="two-sided")
    ci_lo, ci_hi = _bootstrap_ratio(ri, rn)

    return dict(rich_ind=ri, rich_ni=rn, depth_ind=di, depth_ni=dn,
                ratio=ratio, p=p, ci_lo=ci_lo, ci_hi=ci_hi,
                n_ind=len(ri), n_ni=len(rn), method="Uncorrected")


def _compute_matched(data, ids_ind, ids_ni, spec):
    """Depth-matched 1:1 nearest-neighbour comparison."""
    all_ids = list(ids_ind) + list(ids_ni)
    presence = build_presence(data.breadth.loc[all_ids],
                              data.depth.loc[all_ids],
                              data.lengths, spec=spec)
    richness = presence.sum(axis=1)
    ri = richness.loc[ids_ind].values
    rn = richness.loc[ids_ni].values
    di = np.log10(data.total_bp.loc[ids_ind].values.astype(float))
    dn = np.log10(data.total_bp.loc[ids_ni].values.astype(float))

    pair_d = 0.5 * (di + dn)
    edges = _adaptive_bins_matched(pair_d)
    bi = _bin_richness(di, ri, edges)
    bn = _bin_richness(dn, rn, edges)

    ratio = np.median(ri) / max(np.median(rn), 1e-10)
    _, p = mannwhitneyu(ri, rn, alternative="two-sided")
    ci_lo, ci_hi = _bootstrap_ratio(ri, rn)

    return dict(rich_ind=ri, rich_ni=rn, depth_ind=di, depth_ni=dn,
                bins_ind=bi, bins_ni=bn, edges=edges, pair_depths=pair_d,
                ratio=ratio, p=p, ci_lo=ci_lo, ci_hi=ci_hi,
                n=len(ri), method="Depth-matched")



def _depth_stratified_subsample(ids, log_depth, n_keep, rng, n_strata=10):
    """
    Draw n_keep indices into ids, stratified by log_depth decile.

    Preserves the depth distribution shape (retained for potential future use;
    the robustness figure now runs on the full dataset without subsampling).
    If n_keep is None or >= len(ids), returns all indices unchanged.
    """
    n = len(ids)
    if n_keep is None or n_keep >= n:
        return np.arange(n)
    edges = np.percentile(log_depth, np.linspace(0, 100, n_strata + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    chosen = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = np.where((log_depth >= lo) & (log_depth < hi))[0]
        if len(in_bin) == 0:
            continue
        k = max(1, round(n_keep * len(in_bin) / n))
        k = min(k, len(in_bin))
        chosen.append(rng.choice(in_bin, k, replace=False))
    idx = np.concatenate(chosen)
    if len(idx) > n_keep:
        idx = rng.choice(idx, n_keep, replace=False)
    return idx


def _compute_gmw_for_robustness(uncorrected_res):
    """
    Geometric-mean-target depth-corrected richness for the robustness figure.

    Takes the already-computed uncorrected result dict (which contains richness
    and log-depth for ALL samples) so we never call build_presence twice.
    Applies w = sqrt(f_other / f_own) weights via _gmw_estimate_2grp, runs
    a plug-in bootstrap (KDEs fixed), returns ratio + CI.

    No subsampling: the full dataset is used.  The plug-in bootstrap is fast
    (O(n_boot x n) dot products, no KDE re-estimation).
    """
    from .depth_bias_ipw import (_gmw_estimate_2grp, _gmw_bootstrap_2grp,
                                  _geomean_target_grid, _ess,
                                  _adaptive_n_boot)
    from . import config

    ri  = uncorrected_res["rich_ind"].astype(float)
    rn  = uncorrected_res["rich_ni"].astype(float)
    ld_i = uncorrected_res["depth_ind"]
    ld_n = uncorrected_res["depth_ni"]

    # Drop any non-finite entries
    ok_i = np.isfinite(ld_i) & np.isfinite(ri)
    ok_n = np.isfinite(ld_n) & np.isfinite(rn)
    ri, ld_i = ri[ok_i], ld_i[ok_i]
    rn, ld_n = rn[ok_n], ld_n[ok_n]

    n_boot = _adaptive_n_boot(
        len(ri), len(rn),
        base=getattr(config, "OW_AIPWE_N_BOOT_BASE",     1000),
        cap=getattr(config,  "OW_AIPWE_N_BOOT_ROBUSTNESS_CAP", 2000),
    )
    print(f"      GMW: {len(ri):,} Ind + {len(rn):,} NI  n_boot={n_boot}")

    rng = np.random.default_rng(42)
    mu1, mu0, kde_i, kde_n, w_ind, w_ni = _gmw_estimate_2grp(ld_i, ld_n, ri, rn)
    ratio  = mu1 / max(mu0, 1e-10)
    att    = mu1 - mu0

    boot   = _gmw_bootstrap_2grp(ld_i, ld_n, ri, rn, w_ind, w_ni, n_boot, rng)
    valid  = np.isfinite(boot[:, 0]) & np.isfinite(boot[:, 1]) & (boot[:, 1] > 0)
    boot_ratios = boot[valid, 0] / np.maximum(boot[valid, 1], 1e-10)
    ci_lo, ci_hi = (np.percentile(boot_ratios, [2.5, 97.5])
                    if valid.any() else [np.nan, np.nan])
    ci_att = (np.percentile(boot[valid, 2], [2.5, 97.5])
              if valid.any() else [np.nan, np.nan])

    _, p = mannwhitneyu(ri, rn, alternative="two-sided")

    depth_grid = np.linspace(min(ld_i.min(), ld_n.min()),
                             max(ld_i.max(), ld_n.max()), 300)
    geom_grid  = _geomean_target_grid(depth_grid, kde_i, kde_n)

    return dict(
        rich_ind=ri,  rich_ni=rn,
        log_d_ind=ld_i, log_d_ni=ld_n,
        ow_ind=w_ind,  ow_ni=w_ni,
        depth_grid=depth_grid, geom_grid=geom_grid,
        mu1=mu1, mu0=mu0, att=att,
        ratio=ratio, p=p, ci_lo=ci_lo, ci_hi=ci_hi, ci_att=ci_att,
        mu1_raw=float(ri.mean()), mu0_raw=float(rn.mean()),
        ess_ind=_ess(w_ind), ess_ni=_ess(w_ni),
        n_ind=len(ri), n_ni=len(rn),
        method="Geom-mean-target",
    )


def _plot_gmw_overlap(ax, ow_res):
    """
    Panel C of the robustness figure: depth KDE curves + geom-mean-target curve.

    IMPORTANT: does NOT use twinx().  A twinx would insert an extra axes object
    into fig.axes, offsetting the index-based y-limit sync that runs later in
    build_robustness_figure, which would then accidentally set the twinx ylim
    to the richness scale (~1000) and make the GMW curve invisible.

    The geom-mean-target curve sqrt(f_i*f_n) is normalised to 75% of plot
    height so it sits within the density y-axis without a second axis.
    """
    di, dn = ow_res["log_d_ind"], ow_res["log_d_ni"]
    lo = min(di.min(), dn.min()) - 0.2
    hi = max(di.max(), dn.max()) + 0.2
    dg = np.linspace(lo, hi, 300)

    # KDE density curves for display (scipy — one-time eval, fine here)
    kde_i_disp = gaussian_kde(di, bw_method="silverman")(dg)
    kde_n_disp = gaussian_kde(dn, bw_method="silverman")(dg)

    ax.plot(dg, kde_i_disp, color=IND_COLOR, lw=1.8,
            label=f"Industrial (n={ow_res['n_ind']:,})")
    ax.plot(dg, kde_n_disp, color=NI_COLOR, lw=1.8, ls="--",
            label=f"Non-Industrial (n={ow_res['n_ni']:,})")

    y_top = max(float(kde_i_disp.max()), float(kde_n_disp.max())) * 1.30
    ax.set_ylim(0, y_top)
    ax.set_yticks([])

    # Geometric-mean target: sqrt(f_i * f_n) — the reweighting target.
    # Scaled to 75% of plot height so it sits within the density y-axis.
    # (No twinx: that would insert an extra axes object into fig.axes and
    # break the index-based y-limit sync in build_robustness_figure.)
    geom_raw = np.maximum(
        np.asarray(ow_res.get("geom_grid",
                   np.sqrt(np.maximum(ow_res.get("ow_grid", np.array([])), 0.0))),
                   dtype=float), 0.0)
    dg_ow    = np.asarray(ow_res.get("depth_grid", dg), dtype=float)
    geom_max = float(np.nanmax(geom_raw)) if len(geom_raw) > 0 else 0.0

    if geom_max > 1e-12:
        scale    = y_top * 0.75 / geom_max
        geom_disp = geom_raw * scale
        # The peak depth used to be a boxed note with an arrow, pinned to a
        # corner. place_legends_clear picks the emptiest corner from the DATA
        # and cannot see a text artist, so it kept putting the legend straight
        # on top of that note -- the one collision left in this figure at every
        # type size. The number belongs in the key for the curve it describes.
        peak_idx = int(np.nanargmax(geom_raw))
        peak_x   = float(dg_ow[peak_idx])
        ax.plot(dg_ow, geom_disp, color="black", lw=2.5,
                label=r"$\sqrt{\hat{f}_i{\cdot}\hat{f}_n}$ target [scaled]"
                      "\n" + f"peak at log\u2081\u2080 = {peak_x:.2f}")
        ax.fill_between(dg_ow, geom_disp, alpha=0.10, color="black")
        ax.axvline(peak_x, color="black", lw=0.8, ls=":", alpha=0.55)

    ax.set_xlabel(r"$\log_{10}$(Total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title("Geometric-mean target", fontsize=FS-5)
    ax.legend(fontsize=FS-2, loc="upper right")


def _compute_ipw(data, spec):
    """Inverse-probability weighting to a common depth distribution."""
    meta = data.meta
    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])
    all_ids = meta.index[ind_mask | ni_mask].tolist()

    presence = build_presence(data.breadth.loc[all_ids],
                              data.depth.loc[all_ids],
                              data.lengths, spec=spec)
    richness = presence.sum(axis=1)

    ids_ind = meta.index[ind_mask].tolist()
    ids_ni = meta.index[ni_mask].tolist()
    ld_i = np.log10(data.total_bp.loc[ids_ind].values.astype(float))
    ld_n = np.log10(data.total_bp.loc[ids_ni].values.astype(float))

    vi, vn = np.isfinite(ld_i), np.isfinite(ld_n)
    ld_i, ld_n = ld_i[vi], ld_n[vn]
    ids_ind = [ids_ind[k] for k in range(len(ids_ind)) if vi[k]]
    ids_ni = [ids_ni[k] for k in range(len(ids_ni)) if vn[k]]
    ri = richness.loc[ids_ind].values
    rn = richness.loc[ids_ni].values

    kde_i = gaussian_kde(ld_i, bw_method="silverman")
    kde_n = gaussian_kde(ld_n, bw_method="silverman")
    ni, nn = len(ld_i), len(ld_n)
    fi, fn = ni / (ni + nn), nn / (ni + nn)
    f_target = lambda x: fi * kde_i(x) + fn * kde_n(x)

    wi = f_target(ld_i).ravel() / (kde_i(ld_i).ravel() + 1e-20)
    wn = f_target(ld_n).ravel() / (kde_n(ld_n).ravel() + 1e-20)
    for w in (wi, wn):
        lo, hi = np.percentile(w, [1, 99])
        w[:] = np.clip(w, lo, hi)
    wi /= wi.sum()
    wn /= wn.sum()

    wm_i = _weighted_median(ri, wi)
    wm_n = _weighted_median(rn, wn)
    ratio = wm_i / max(wm_n, 1e-10)

    rng = np.random.default_rng(42)
    br = np.empty(2000)
    for b in range(2000):
        ii = rng.choice(ni, ni, replace=True)
        jj = rng.choice(nn, nn, replace=True)
        bwi = wi[ii]; bwi /= bwi.sum()
        bwn = wn[jj]; bwn /= bwn.sum()
        br[b] = (_weighted_median(ri[ii], bwi) /
                 max(_weighted_median(rn[jj], bwn), 1e-10))
    ci_lo, ci_hi = np.percentile(br, [2.5, 97.5])
    _, p = mannwhitneyu(ri, rn, alternative="two-sided")

    return dict(rich_ind=ri, rich_ni=rn, log_d_ind=ld_i, log_d_ni=ld_n,
                w_ind=wi, w_ni=wn, kde_ind=kde_i, kde_ni=kde_n,
                f_target=f_target, ratio=ratio, p=p,
                ci_lo=ci_lo, ci_hi=ci_hi,
                wmed_ind=wm_i, wmed_ni=wm_n,
                n_ind=ni, n_ni=nn, method="IPW")


# ══════════════════════════════════════════════════════════════════
#  ROBUSTNESS — PLOT HELPERS
# ══════════════════════════════════════════════════════════════════

def _plot_uncorrected_data(ax, uncorr):
    """Panel A: depth imbalance between Industrial and NI."""
    di, dn = uncorr["depth_ind"], uncorr["depth_ni"]
    lo = min(di.min(), dn.min()) - 0.3
    hi = max(di.max(), dn.max()) + 0.3
    bins = np.linspace(lo, hi, 50)

    ax.hist(di, bins=bins, density=True, alpha=0.35, color=IND_COLOR,
            edgecolor="white", lw=0.3,
            label=f"Industrial (n={uncorr['n_ind']:,})")
    ax.hist(dn, bins=bins, density=True, alpha=0.35, color=NI_COLOR,
            edgecolor="white", lw=0.3,
            label=f"Non-Industrial (n={uncorr['n_ni']:,})")

    dg = np.linspace(lo, hi, 300)
    ax.plot(dg, gaussian_kde(di, bw_method="silverman")(dg),
            color=IND_COLOR, lw=2)
    ax.plot(dg, gaussian_kde(dn, bw_method="silverman")(dg),
            color=NI_COLOR, lw=2)

    med_i, med_n = np.median(di), np.median(dn)
    ax.axvline(med_i, color=IND_COLOR, ls="--", lw=1, alpha=0.7)
    ax.axvline(med_n, color=NI_COLOR, ls="--", lw=1, alpha=0.7)

    diff_gb = (10**med_n - 10**med_i) / 1e9
    # Anchored to the axes, not the data: at 85% of the y-range between the two
    # medians it sat on the peak of both curves. Upper right is empty here
    # because the distributions are left-skewed.
    ax.text(0.98, 0.98, f"\u0394 median ~ {diff_gb:.1f} Gb",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FS-1,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85,
                      edgecolor="0.7"))

    ax.set_xlabel("log₁₀(Total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title("Uncorrected", fontsize=FS-5)
    ax.legend(fontsize=FS-2, loc="upper left")


def _plot_matched_data(ax, matched, spec):
    """Panel B: adaptive-bin richness vs depth for matched pairs."""
    bi, bn = matched["bins_ind"], matched["bins_ni"]
    edges = matched["edges"]

    for bdf, color, label in [(bi, IND_COLOR, "Industrial"),
                               (bn, NI_COLOR, "Non-Industrial")]:
        if bdf.empty:
            continue
        ax.plot(bdf["bin_mid"], bdf["median"], "o-", color=color,
                ms=SYMBOL_MIN_PT, lw=1.5, label=label, zorder=3)
        ax.fill_between(bdf["bin_mid"], bdf["q25"], bdf["q75"],
                        color=color, alpha=0.12, zorder=2)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2 == 0:
            ax.axvspan(lo, hi, color="gray", alpha=0.04, zorder=0)

    ax.set_xlabel("log₁₀(Total mapped bp)")
    # Pinned: left to itself the label is placed outside the tick
    # labels and lands inside the panel to its left.
    pin_ylabel(ax, "Richness (SGBs)", x=-0.150)
    ax.set_title("Depth-matched", fontsize=FS-5)
    ax.legend(fontsize=FS-2, loc="upper left")


def _plot_ipw_data(ax, ipw):
    """Panel C: original vs reweighted depth distributions."""
    dg = np.linspace(
        min(ipw["log_d_ind"].min(), ipw["log_d_ni"].min()) - 0.2,
        max(ipw["log_d_ind"].max(), ipw["log_d_ni"].max()) + 0.2, 300)

    ax.plot(dg, ipw["kde_ind"](dg), "--", color=IND_COLOR, lw=1.2,
            alpha=0.6, label="Ind (original)")
    ax.plot(dg, ipw["kde_ni"](dg), "--", color=NI_COLOR, lw=1.2,
            alpha=0.6, label="NI (original)")

    bins_arr = np.linspace(dg[0], dg[-1], 40)
    for ld, w, c, lab in [
        (ipw["log_d_ind"], ipw["w_ind"], IND_COLOR, "Ind (reweighted)"),
        (ipw["log_d_ni"], ipw["w_ni"], NI_COLOR, "NI (reweighted)"),
    ]:
        ax.hist(ld, bins=bins_arr, weights=w * len(w), color=c,
                alpha=0.25, density=True, label=lab)

    ax.set_xlabel("log₁₀(Total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title(f"IPW: depth reweighting\n"
                 f"(Ind n={ipw['n_ind']:,}, NI n={ipw['n_ni']:,})",
                 fontsize=FS-1)
    ax.legend(fontsize=FS-2, loc="upper right")


def _two_group_box(ax, v_ind, v_ni, n_ind, n_ni, ratio, p,
                   ci_lo, ci_hi, title):
    """Compact two-group boxplot with ratio annotation."""
    rng = np.random.default_rng(42)
    bp = ax.boxplot([v_ind, v_ni], patch_artist=True, widths=0.5,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], [IND_COLOR, NI_COLOR]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    for i, (vals, c) in enumerate(
            zip([v_ind, v_ni], [IND_COLOR, NI_COLOR]), 1):
        ns = min(len(vals), 2000)
        ix = rng.choice(len(vals), ns, replace=False)
        jt = rng.uniform(-0.12, 0.12, ns)
        # s=2 was 1.4 pt across, under the 6 pt symbol minimum. At 6 pt the
        # mark covers 18x the area, so the opacity drops by about as much to
        # keep the cloud reading as density rather than as a solid block.
        ax.scatter(np.full(ns, i) + jt, vals[ix], c=c, alpha=0.012,
                   s=SYMBOL_MIN_S, zorder=1, edgecolors="none")

    # Where the boxes and whiskers stop. The jitter cloud behind them runs to
    # the largest single richness value, so letting matplotlib autoscale to the
    # scatter put the boxes in the bottom third of the panel and gave the top
    # two thirds to a few faint dots. The caller sets the limit from this.
    _top = max((w.get_ydata().max() for w in bp["whiskers"]), default=1.0)

    ax.set_xticklabels([f"Ind\n(n={n_ind:,})", f"NI\n(n={n_ni:,})"])
    # Labelled once for the row by the caller; ten copies of "Richness" cost a
    # column of width each and say nothing the first one did not.
    ax.set_ylabel("")
    ax.set_title(title, fontsize=FS-5)

    # The ratio, its interval and the p-value were a three-line box pinned to
    # the top-right of every panel, landing on the whiskers and on whatever the
    # neighbouring panel had at its left edge. The ratio is the one number a
    # reader needs while looking at the boxes; the interval and p-value are in
    # robustness_summary.csv for the caption.
    ax.text(0.97, 0.95, f"{ratio:.2f}x", transform=ax.transAxes,
            fontsize=FS-2, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75,
                      edgecolor="none"))
    ax.grid(axis="y", alpha=0.3)
    return float(_top)


def _plot_forest(ax, entries, ref):
    """Forest plot: ratio + 95% CI for each method x threshold."""
    labels, ratios, los, his = [], [], [], []
    for e in entries:
        labels.append(f"{e['method']} — {e['spec_label']}")
        ratios.append(e["ratio"])
        los.append(e.get("ci_lo", np.nan))
        his.append(e.get("ci_hi", np.nan))

    y = np.arange(len(labels))
    r, lo, hi = np.array(ratios), np.array(los), np.array(his)

    method_colors = {"Uncorrected": "#888888",
                     "Depth-matched": IND_COLOR,
                     "IPW": NI_COLOR}
    colors = [method_colors.get(e["method"], "black") for e in entries]

    xerr_lo, xerr_hi, lo, hi = _nonnegative_xerr(r, lo, hi)
    ax.errorbar(r, y, xerr=[xerr_lo, xerr_hi], fmt="none",
                ecolor="gray", capsize=4, lw=1.5, zorder=2)
    ax.scatter(r, y, c=colors, s=50, zorder=4,
               edgecolors="white", lw=0.5)
    ax.axvline(1.0, color="gray", ls="--", lw=1)

    # Shade corrected vs uncorrected ranges
    corrected = [e for e in entries if e["method"] != "Uncorrected"]
    if corrected:
        cr = [e["ratio"] for e in corrected]
        ax.axvspan(min(cr), max(cr), color="#C8E6C9", alpha=0.35, zorder=0,
                   label=f"Corrected range: {min(cr):.2f}–{max(cr):.2f}x")

    uncorr = [e for e in entries if e["method"] == "Uncorrected"]
    if uncorr:
        ur = [e["ratio"] for e in uncorr]
        ax.axvspan(min(ur), max(ur), color="#FFCDD2", alpha=0.35, zorder=0,
                   label=f"Uncorrected range: {min(ur):.2f}–{max(ur):.2f}x")

    ax.set_yticks(y)
    # Half a row of air top and bottom: on the default margins the
    # first and last interval sat on the spines.
    if len(y):
        ax.set_ylim(-0.75, len(y) - 1 + 0.75)
    # Short row labels; the full wording of each is in the caption.
    _short = {"\u2265300 kb mapped bases": "300 kb",
              ">=300 kb mapped bases": "300 kb",
              # the pre-2026-10-09 spelling, so a stored data object or an
              # older results tree still resolves rather than printing the
              # full sentence as a row label
              "\u2265300 kB mapped bases": "300 kb",
              ">=300 kB mapped bases": "300 kb",
              "\u226510% breadth coverage": "10% breadth",
              ">=10% breadth coverage": "10% breadth"}

    def _short_row(t):
        out = str(t)
        for long, short in _short.items():
            out = out.replace(long, short)
        # One line per row. Two-line labels doubled the height of the label
        # block and left the rows cramped against the panel edges; the panel
        # is shifted right by its own label width anyway, so a longer single
        # line costs width that was going spare.
        return out.replace(" \u2014 ", ", ").replace(" - ", ", ")

    for _lab in ax.set_yticklabels([_short_row(l) for l in labels],
                                   fontsize=FS-2):
        _lab._keep_text = True
    ax.set_xlabel("Richness ratio (Industrial / Non-Industrial)")
    ax.set_title("Robustness summary", fontsize=FS-4)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    # Outside, to the right: this panel is a forest of six rows spanning the
    # full width, so a key placed inside covers estimates whichever corner it
    # takes. Outside it can also be read at the figure's own type size.
    # Under the panel, not beside it: placed to the right it pushed the whole
    # figure past the three-column width, and everything on it was then
    # reduced -- which is what took the key entries below 5 pt.
    _lf = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=2, frameon=False, handlelength=1.4,
                    columnspacing=1.6)
    _lf._fixed_loc = True

    for i, (rv, lv, hv) in enumerate(zip(r, lo, hi)):
        if np.isfinite(lv):
            ax.text(hv + 0.005, i,
                    f"  {rv:.2f} [{lv:.2f}, {hv:.2f}]",
                    va="center", fontsize=FS-2, color="gray")


# ══════════════════════════════════════════════════════════════════
#  ROBUSTNESS FIGURE BUILDER
# ══════════════════════════════════════════════════════════════════

# Base type size for the robustness figure, in points on its own 28 in canvas.
# It is authored four and a half times page width because ten panels need the
# room, and point sizes do not scale with the canvas -- the module default of 10
# pt printed at 1.6 pt once the figure was placed at column width. Measured by
# sweeping the size against the collision map (tools/render_sizes.py): the
# figure stays clear of itself up to here and starts colliding above it.
ROBUSTNESS_FS: float = 32.0

#: Scatter area giving a 6 pt symbol, the Science minimum. `s` is an
#: area in points squared, so this is 6 squared.
SYMBOL_MIN_S: float = 36.0
#: The same floor as a marker diameter, for Line2D.
SYMBOL_MIN_PT: float = 6.0


class _RobustnessSpec:
    """Stands in for a DetectionSpec when redrawing from the table.

    The drawing reads only the short label and the description off a spec;
    carrying the real object would tie the data file to the detection module.
    """

    def __init__(self, short_label, description=""):
        self.short_label = short_label
        self.description = description or short_label


def _robustness_result_from_arrays(ri, rn, di, dn, method, spec_label,
                                   paired=False):
    """Rebuild one comparison from its richness and depth columns.

    Mirrors _compute_uncorrected and _compute_matched, which are themselves
    pure functions of these four arrays once presence has been called.
    """
    ok_i = np.isfinite(di) & np.isfinite(ri)
    ok_n = np.isfinite(dn) & np.isfinite(rn)
    ri, di = ri[ok_i], di[ok_i]
    rn, dn = rn[ok_n], dn[ok_n]

    ratio = np.median(ri) / max(np.median(rn), 1e-10)
    _, p = mannwhitneyu(ri, rn, alternative="two-sided")
    ci_lo, ci_hi = _bootstrap_ratio(ri, rn)
    out = dict(rich_ind=ri, rich_ni=rn, depth_ind=di, depth_ni=dn,
               ratio=ratio, p=p, ci_lo=ci_lo, ci_hi=ci_hi,
               method=method, spec_label=spec_label)
    if paired and len(ri) == len(rn):
        # The matched panel bins on the mean depth of each 1:1 pair, so the
        # two arrays have to stay in the order they were written.
        pair_d = 0.5 * (di + dn)
        edges = _adaptive_bins_matched(pair_d)
        out.update(bins_ind=_bin_richness(di, ri, edges),
                   bins_ni=_bin_richness(dn, rn, edges),
                   edges=edges, pair_depths=pair_d, n=len(ri))
    else:
        out.update(n_ind=len(ri), n_ni=len(rn))
    return out


def plot_robustness_alpha(tables, out_dir, ref="combined"):
    """Draw the robustness figure from its table, and from nothing else.

    The geometric-mean-target column is not stored separately: it is derived
    from the uncorrected comparison exactly as the pipeline derives it.
    """
    df = tables["samples"]
    if "detection_key" in df.columns:
        key_col = "detection_key"
    else:
        # Tables written before the key column carry only the prose
        # description, and the panels are looked up by short label, so
        # recover it. "300kb" and "10pct" are the two the figure draws.
        key_col = "_detection_key"
        df = df.copy()
        df[key_col] = np.where(
            df["detection"].astype(str).str.contains("300"), "300kb", "10pct")

    def _arrays(method, key):
        sub = df[(df["method"] == method) & (df[key_col] == key)]
        out = {}
        for grp, tag in (("Industrial", "i"), ("Non-Industrial", "n")):
            g = sub[sub["group"] == grp]
            out["r" + tag] = g["richness"].to_numpy(dtype=float)
            out["d" + tag] = (g["log10_total_mapped_bp"].to_numpy(dtype=float)
                              if "log10_total_mapped_bp" in g.columns
                              else np.full(len(g), np.nan))
        return out

    keys = list(dict.fromkeys(df[key_col]))
    descs = {k: str(df.loc[df[key_col] == k, "detection"].iloc[0]) for k in keys}
    specs = [_RobustnessSpec(k, descs[k]) for k in keys]
    flat_spec = next((sp for sp in specs if "300" in sp.short_label), specs[0])

    uncorrected, matched, ow_aipwe = {}, {}, {}
    for k in keys:
        a = _arrays("Uncorrected", k)
        uncorrected[k] = _robustness_result_from_arrays(
            a["ri"], a["rn"], a["di"], a["dn"], "Uncorrected", descs[k])
        b = _arrays("Depth-matched", k)
        matched[k] = _robustness_result_from_arrays(
            b["ri"], b["rn"], b["di"], b["dn"], "Depth-matched", descs[k],
            paired=True)
        ow_aipwe[k] = _compute_gmw_for_robustness(uncorrected[k])
        ow_aipwe[k]["spec_label"] = descs[k]

    fpath, _ = _draw_robustness_figure(uncorrected, matched, ow_aipwe, specs,
                                       flat_spec, ref, out_dir)
    return fpath


def _draw_robustness_figure(uncorrected, matched, ow_aipwe, specs,
                            flat_spec, ref, s1_main):
    """Draw the robustness figure from its three result sets."""
    global FS
    # The canvas is now authored at the printed width, so the shared
    # scale applies: a base tuned for a 20-28 in canvas made the type
    # three times too large once the figure stopped being reduced.
    apply_oi_style()
    FS = current_fs()

    # Authored at the printed width. A figure drawn four times too wide is
    # reduced to fit the page, and the reduction takes every line and
    # symbol with it -- a 1 pt rule landed at 0.25 pt, under the 0.5 pt
    # floor, and the type had to be inflated to stay legible.
    _PANEL = PAGE_WIDTH_IN / 3.0      # inches, square
    _COLS, _PANEL_ROWS = 3, 3
    _FOREST = 0.60                    # forest plot, as a fraction of a panel
    _W = _PANEL * _COLS
    _H = _PANEL * (_PANEL_ROWS + _FOREST) * 1.12      # 12% for titles/margins
    fig = plt.figure(figsize=(_W, _H))
    # wspace 0.16 left an 11 px gutter between columns while panel B's
    # rotated y label is 14 px wide, so the label had nowhere to go but into
    # the panel on its left. The gutter has to clear the label, not the other
    # way round.
    gs = gridspec.GridSpec(4, 3,
                           height_ratios=[1.0, 1.0, 1.0, _FOREST],
                           hspace=0.30, wspace=0.24)
    panels = "ABCDEFGHIJKLM"

    # Row 0: DATA panels
    ax = fig.add_subplot(gs[0, 0])
    _plot_uncorrected_data(ax, uncorrected["300kb"])
    _letter_a = ax.text(-0.10, 1.035, panels[0], transform=ax.transAxes,
                        fontsize=FS+6, fontweight="bold", va="bottom", ha="right")

    ax = fig.add_subplot(gs[0, 1])
    _plot_matched_data(ax, matched["300kb"], flat_spec)
    ax.text(-0.10, 1.035, panels[1], transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold", va="bottom", ha="right")

    ax = fig.add_subplot(gs[0, 2])
    _plot_gmw_overlap(ax, ow_aipwe["300kb"])
    ax.text(-0.10, 1.035, panels[2], transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold", va="bottom", ha="right")

    # The three data panels share an x scale, so each one's first and last tick
    # label sits right at the column gap and touches its neighbour's. Pruning
    # the end labels costs nothing a reader needs -- the interior ticks carry
    # the scale -- and it is one of the three things that capped the type size.
    from matplotlib.ticker import MaxNLocator
    for _ax in fig.axes[:3]:
        _ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    # Rows 1–2: RESULTS (300kb then 10%)
    _tops: dict = {}
    for row, sp in enumerate(specs, 1):
        sl = sp.short_label
        for col, (rd, mname) in enumerate([
            (uncorrected, "Uncorrected"),
            (matched, "Depth-matched"),
            (ow_aipwe, "Geom-mean-target"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            res = rd[sl]
            n_i = res.get("n_ind", res.get("n", 0))
            n_n = res.get("n_ni",  res.get("n", 0))
            vi  = res["rich_ind"]
            vn  = res["rich_ni"]
            if mname == "Geom-mean-target":
                rr = np.random.default_rng(42)
                ns = 5000
                # Resample proportional to geom-mean-target weights for display
                wi_norm = res["ow_ind"] / res["ow_ind"].sum()
                wn_norm = res["ow_ni"]  / res["ow_ni"].sum()
                vi = res["rich_ind"][rr.choice(n_i, ns, p=wi_norm, replace=True)]
                vn = res["rich_ni"][rr.choice(n_n, ns, p=wn_norm, replace=True)]
            _tops.setdefault(col, []).append(
                _two_group_box(ax, vi, vn, n_i, n_n,
                               res["ratio"], res["p"],
                               res["ci_lo"], res["ci_hi"],
                               ""))
            pi = (row - 1) * 3 + col + 3
            ax.text(-0.10, 1.035, panels[pi], transform=ax.transAxes,
                    fontsize=FS+6, fontweight="bold", va="bottom", ha="right")

            # The detection threshold names the row, once, on its right-hand
            # edge. Repeated in every panel title it ran three panels into
            # each other; the method is already the column heading.
            if col == 2:
                _short_row = {"\u2265300 kb mapped bases": "300 kb",
                              ">=300 kb mapped bases": "300 kb",
                              "\u2265300 kB mapped bases": "300 kb",
                              ">=300 kB mapped bases": "300 kb",
                              "\u226510% breadth coverage": "10% breadth",
                              ">=10% breadth coverage": "10% breadth"}
                _rtxt = _short_row.get(str(sp.description), str(sp.description))
                _rl = ax.text(1.02, 0.5, _rtxt, rotation=270,
                              transform=ax.transAxes, va="center", ha="left",
                              fontsize=FS-4, fontweight="bold", color="0.25")
                _rl._keep_text = True

    # Sync y-axes per column across rows 1 and 2, to the boxes rather than to
    # the jitter cloud. Autoscaling to the scatter took every column to the
    # largest richness in the cohort and left the boxes sitting in the bottom
    # third -- half the panel area spent on a faint haze of outlying dots.
    for col in range(3):
        ax1 = fig.axes[3 + col]
        ax2 = fig.axes[6 + col]
        ymax = max(_tops.get(col, [0.0])) * 1.18 or max(
            ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)

    # Row 3: Forest plot
    ax = fig.add_subplot(gs[3, :])
    _forest_ax = ax               # repositioned below, once its labels exist
    forest_entries = []
    for sp in specs:
        sl = sp.short_label
        for rd, mn in [(uncorrected, "Uncorrected"),
                       (matched, "Depth-matched"),
                       (ow_aipwe, "Geom-mean-target")]:
            e = dict(rd[sl])
            e["method"] = mn
            forest_entries.append(e)
    _plot_forest(ax, forest_entries, ref)
    _letter_j = ax.text(-0.02, 1.035, panels[9], transform=ax.transAxes,
                        fontsize=FS+6, fontweight="bold", va="bottom", ha="right")

    # No gridlines, and fewer ticks. Both were competing with the data rather
    # than supporting it: at this type size a gridline crossing a box is one
    # more line to read, and a tick every 100 SGBs puts six numbers down an
    # axis where three carry the scale. Fewer, larger numbers read faster than
    # more, smaller ones. apply_oi_style sets axes.grid globally, so this has
    # to be switched off on the axes rather than in the rcParams.
    for _ax in fig.axes:
        _ax.grid(False)
    for _ax in fig.axes[3:9]:                      # the six box panels
        _ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _forest_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.suptitle(
        f"Robustness of the industrial richness deficit [{ref}]\n"
        f"Uncorrected vs depth-matched vs geom-mean-target "
        r"($w = \sqrt{\hat{f}_{other}/\hat{f}_{own}}$) x two detection thresholds",
        fontsize=FS+2, fontweight="bold", y=0.99)

    fpath = os.path.join(s1_main, "figure_robustness_alpha.png")
    # One label for the whole grid. Ten panels each carried their own copy of
    # "Richness (SGBs)" and "log10(Total mapped bp)", and it was those, sitting
    # outside their axes, that reached the neighbouring panel's tick labels --
    # the 100% overlaps the collision map kept reporting.
    fig.supylabel("Richness (SGBs)", fontsize=FS-2)

    # Margins are the shared ones on purpose. Tightening them here looks like
    # it recovers whitespace and does the opposite: the canvas is far taller
    # than it is wide, bbox_inches="tight" crops whatever the axes do not fill,
    # and margins that let the axes fill more of a 38 in height only make the
    # saved image taller relative to its width. Saved aspect went 1.11 -> 1.32
    # that way, and a figure placed at a fixed width in Word shrinks by exactly
    # that ratio -- the type and the gaps between panels both came out smaller
    # on the page while every number in the code got bigger. Apparent size is
    # set by the saved aspect, not by the margin fractions.

    # The forest plot spans the full width, so its long category labels hang
    # out to the left of the panels above it and the figure reads as ragged.
    # Shift it right by exactly the width of its own labels, so the left edge
    # of its text lines up with the left spine of the plots above.
    def _align_forest(_fig=fig, _ax=_forest_ax, _ref=_letter_a, _j=_letter_j):
        """Line the forest plot's category labels up with the panels above.

        Its labels are long, so matching the axes' left edge to the grid leaves
        the text hanging out past everything else. Align the left edge of the
        text instead, which is what the eye reads as the figure's margin.

        Runs as a post-layout hook: subplots_adjust resets every
        gridspec-managed axes, and fit_tick_labels can shrink these very labels,
        so measuring before either had run aligned to text that then moved.
        """
        try:
            _fig.canvas.draw()
            _r = _fig.canvas.get_renderer()
            _lab_left = min(
                (t.get_window_extent(renderer=_r).x0
                 for t in _ax.get_yticklabels()
                 if t.get_visible() and t.get_text().strip()),
                default=None)
            if _lab_left is None:
                return
            _pos = _ax.get_position()
            _target = min(a.get_position().x0 for a in _fig.axes if a is not _ax)
            _shift = _pos.x0 - (_lab_left / _fig.bbox.width)
            _right = _pos.x0 + _pos.width
            _new_x0 = _target + _shift
            _ax.set_position([_new_x0, _pos.y0, _right - _new_x0, _pos.height])
            # J then belongs over the label block, not over the shifted axes,
            # so it lines up with A, D and G down the left edge.
            _x = _ref.get_window_extent(renderer=_r).x0 / _fig.bbox.width
            _j.set_transform(_fig.transFigure)
            _j.set_position((_x, _pos.y0 + _pos.height * 1.02))
        except Exception as _exc:
            print(f"    [forest] could not align labels: {_exc}")

    add_post_layout_hook(fig, _align_forest)
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath, forest_entries


def build_robustness_figure(data, dm, stats):
    """Build the 10-panel robustness figure for one reference."""
    global FS
    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")

    flat_spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    breadth_spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    specs = [flat_spec, breadth_spec]

    # ── Compute all methods x thresholds ──
    print(f"    Computing uncorrected comparisons …")
    uncorrected = {}
    for sp in specs:
        uncorrected[sp.short_label] = _compute_uncorrected(data, sp)
        uncorrected[sp.short_label]["spec_label"] = sp.description

    print(f"    Computing depth-matched comparisons …")
    matched = {}
    for sp in specs:
        matched[sp.short_label] = _compute_matched(
            data, dm.ind_vs_ni_ind, dm.ind_vs_ni_ni, sp)
        matched[sp.short_label]["spec_label"] = sp.description

    print(f"    Computing geom-mean-target depth correction …")
    ow_aipwe = {}
    for sp in specs:
        # Reuse richness already computed by _compute_uncorrected — no redundant build_presence
        ow_aipwe[sp.short_label] = _compute_gmw_for_robustness(uncorrected[sp.short_label])
        ow_aipwe[sp.short_label]["spec_label"] = sp.description

    # ── Log stats ──
    for tag, rd, mname in [("uncorrected", uncorrected, "Uncorrected"),
                            ("matched", matched, "Depth-matched"),
                            ("gmw", ow_aipwe, "Geom-mean-target")]:
        for sl, res in rd.items():
            ci_str = (f"[{res.get('ci_lo', np.nan):.3f}, "
                      f"{res.get('ci_hi', np.nan):.3f}]")
            stats.add(f"robustness_{tag}_{sl}", "s1", "main",
                      f"Richness ratio ({mname}, {res['spec_label']})",
                      f"ratio={res['ratio']:.3f}, 95% CI {ci_str}",
                      test=mname, p_value=res["p"],
                      extra={"ratio": res["ratio"],
                             "ci_lo": res.get("ci_lo"),
                             "ci_hi": res.get("ci_hi")})

    # ── Build figure ──
    # Canvas as the medium preset had it. Measured across five canvas scales,
    # the overlaps in this figure are fixed placements rather than crowding --
    # they persist from 6.3 in to 29 in -- so shrinking the canvas only cost
    # printed size. At 29 in the figure also stops overflowing its own bounds.
    # Square panels. The grid is three rows of three plus a full-width forest
    # plot, and the old height ratios gave each row about a fifth of a 22 in
    # canvas -- panels 9.7 in wide and 4.4 in tall. Squat panels waste the
    # vertical room that lets type grow, so size the canvas from the panel
    # instead: height follows from however many rows there are.
    # Set the type scale here, next to the figure, and not at the top of this
    # function. Fourteen modules call apply_oi_style() with no argument at
    # import, which resets the whole style to the page-width default -- and the
    # computation above lazily imports one of them (depth_bias_ipw, for the GMW
    # estimator). Applying the size at function entry therefore survived only
    # until that import fired: the figure came out with base-10 ticks and axis
    # labels while every explicit fontsize=FS-n, captured before the reset,
    # stayed large. It looked like a font bug and was an import-order bug, and
    # it hid in a sweep because the lazy import only fires on the first render
    # in a process. Style belongs adjacent to the figure it styles.

    # The figure's data object. One row per sample per method and detection
    # threshold: the richness it contributed and which group it is in. The
    # ratios, intervals and p-values shown in the panels and the forest plot
    # are all computed from these rows.
    fpath = os.path.join(s1_main, "figure_robustness_alpha.png")
    _blocks = []
    for _method, _rd in (("Uncorrected", uncorrected),
                         ("Depth-matched", matched),
                         ("Geom-mean-target", ow_aipwe)):
        for _sl, _res in _rd.items():
            for _grp, _key, _dkeys in (
                    ("Industrial", "rich_ind", ("depth_ind", "log_d_ind")),
                    ("Non-Industrial", "rich_ni", ("depth_ni", "log_d_ni"))):
                _v = _res.get(_key)
                if _v is None:
                    continue
                # Depth travels with richness: the geometric-mean-target
                # weights are a function of it, so without this column the
                # corrected ratio in the forest cannot be recomputed from the
                # table. The two methods name the same log10 quantity
                # differently, hence the pair of keys.
                _d = next((_res[k] for k in _dkeys if _res.get(k) is not None),
                          None)
                _row = {
                    "method": _method,
                    # The short key as well as the description: the panels are
                    # looked up by key, and matching on prose would break the
                    # moment a description is reworded.
                    "detection_key": _sl,
                    "detection": _res.get("spec_label", _sl),
                    "group": _grp,
                    "richness": np.asarray(_v, dtype=float),
                }
                if _d is not None and len(_d) == len(_v):
                    _row["log10_total_mapped_bp"] = np.asarray(_d, dtype=float)
                _blocks.append(pd.DataFrame(_row))
    # Keep the slot-resolving path: _draw_robustness_figure rebinds `fpath`.
    _emit_path = fpath

    fpath, forest_entries = _draw_robustness_figure(
        uncorrected, matched, ow_aipwe, specs, flat_spec, ref, s1_main)

    # Save summary CSV
    rows = [dict(reference=ref, method=e["method"],
                 detection=e["spec_label"], ratio=e["ratio"],
                 ci_lo=e.get("ci_lo", np.nan),
                 ci_hi=e.get("ci_hi", np.nan), p=e["p"])
            for e in forest_entries]
    csv_path = os.path.join(s1_main, "robustness_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    # The data object is emitted here, below the draw, rather than above it:
    # the forest panel's ratios, intervals and p-values are the draw's own
    # return value, and they are quoted directly in the caption ("Richness
    # ratios ... were 0.53 (uncorrected, 95% CI [0.51, 0.54]) ..."). Emitted
    # above, the object carried per-sample richness only, so the one panel that
    # is nothing but estimates and intervals had no tabulated data at all.
    if _blocks:
        from . import figure_source
        _t = {"samples": pd.concat(_blocks, ignore_index=True)}
        if rows:
            _t["forest"] = pd.DataFrame(rows)
        figure_source.emit_for_path(_emit_path, _t, reference=ref)


# ══════════════════════════════════════════════════════════════════
#  TRIPLET HEALTH RICHNESS
# ══════════════════════════════════════════════════════════════════

def compute_triplet_health_richness(data, dm, stats, spec):
    """
    Richness for triplet-matched NI / IndH / IndU with paired
    Wilcoxon signed-rank tests.
    """
    from scipy.stats import wilcoxon

    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")
    ids_ni = dm.triplet_ni
    ids_h = dm.triplet_ind_h
    ids_u = dm.triplet_ind_u

    if not ids_ni or not ids_h or not ids_u:
        return None

    n = len(ids_ni)
    spec_tag = spec.short_label

    all_ids = list(ids_ni) + list(ids_h) + list(ids_u)
    presence = build_presence(data.breadth.loc[all_ids],
                              data.depth.loc[all_ids],
                              data.lengths, spec=spec)
    richness = presence.sum(axis=1)
    rni = richness.loc[ids_ni].values
    rh = richness.loc[ids_h].values
    ru = richness.loc[ids_u].values

    _, p_hu = wilcoxon(rh, ru, alternative="two-sided")
    _, p_nih = wilcoxon(rni, rh, alternative="two-sided")
    _, p_niu = wilcoxon(rni, ru, alternative="two-sided")
    med_ni, med_h, med_u = np.median(rni), np.median(rh), np.median(ru)

    for tag, desc, p, m1, m2, l1, l2 in [
        ("IndH_vs_IndU", "Ind-Healthy vs Ind-Unhealthy",
         p_hu, med_h, med_u, "IndH", "IndU"),
        ("NI_vs_IndH", "NI vs Ind-Healthy",
         p_nih, med_ni, med_h, "NI", "IndH"),
        ("NI_vs_IndU", "NI vs Ind-Unhealthy",
         p_niu, med_ni, med_u, "NI", "IndU"),
    ]:
        ratio = m1 / max(m2, 1e-10)
        stats.add(
            f"health_richness_{tag}_{spec_tag}", "s1", "main",
            f"Richness: {desc} (triplet, {spec.description})",
            f"median {l1}={m1:.0f}, {l2}={m2:.0f}, ratio={ratio:.2f}x",
            test="Wilcoxon signed-rank (paired)", p_value=p, n1=n, n2=n)
        print(f"    {desc}: {m1:.0f} vs {m2:.0f} "
              f"(ratio={ratio:.2f}x), p={p:.2e}")

    pd.DataFrame({
        "triplet_idx": np.arange(n),
        "NI_id": ids_ni, "IndH_id": ids_h, "IndU_id": ids_u,
        "richness_NI": rni, "richness_IndH": rh, "richness_IndU": ru,
        "delta_H_minus_U": rh - ru,
        "detection": spec.description,
    }).to_csv(os.path.join(
        s1_main, f"health_triplet_richness_{spec_tag}.csv"), index=False)

    return dict(rni=rni, rh=rh, ru=ru,
                med_ni=med_ni, med_h=med_h, med_u=med_u,
                p_nih=p_nih, p_hu=p_hu, p_niu=p_niu,
                n=n, spec=spec)


# ══════════════════════════════════════════════════════════════════
#  COMPOSITE FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════

def _richness_box(ax, ri, rn, n_i, n_n, label_i, label_n,
                  color_i, color_n, rng):
    """Two-group richness boxplot."""
    bp = ax.boxplot([ri, rn], patch_artist=True, widths=0.5,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], [color_i, color_n]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, (vals, c) in enumerate(
            zip([ri, rn], [color_i, color_n]), 1):
        ns = min(len(vals), 2000)
        ix = rng.choice(len(vals), ns, replace=False)
        jt = rng.uniform(-0.12, 0.12, ns)
        ax.scatter(np.full(ns, i) + jt, vals[ix], c=c, alpha=0.05,
                   s=2, zorder=1, edgecolors="none")
    ax.set_xticklabels([f"{label_i}\n(n={n_i:,})",
                        f"{label_n}\n(n={n_n:,})"])
    ax.set_ylabel("Richness (SGBs detected)")
    ax.grid(axis="y", alpha=0.3)


def _triplet_box(ax, hr, rng):
    """Three-group health boxplot with significance brackets."""
    rni, rh, ru = hr["rni"], hr["rh"], hr["ru"]
    n = hr["n"]
    spec = hr["spec"]

    HC = {"NI": config.OI["orange"], "IndH": config.OI["green"], "IndU": config.OI["blue"],
      "Hadza": config.OI["pink"], "Carter": config.OI["pink"]}
    groups = [rni, rh, ru]
    labels = [f"Non-Ind\n(n={n:,})", f"Ind-Healthy\n(n={n:,})",
              f"Ind-Unhealthy\n(n={n:,})"]
    colors = [HC["NI"], HC["IndH"], HC["IndU"]]

    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, (vals, c) in enumerate(zip(groups, colors), 1):
        ns = min(len(vals), 2000)
        ix = rng.choice(len(vals), ns, replace=False)
        jt = rng.uniform(-0.15, 0.15, ns)
        ax.scatter(np.full(ns, i) + jt, vals[ix], c=c, alpha=0.06,
                   s=3, zorder=1, edgecolors="none")

    ax.set_xticklabels(labels, fontsize=FS-1)
    ax.set_ylabel("Richness (SGBs detected)")
    ax.grid(axis="y", alpha=0.3)

    ymax = max(np.percentile(g, 99) for g in groups)
    gap = ymax * 0.06
    _sig_bracket(ax, 2, 3, ymax + gap, hr["p_hu"])
    _sig_bracket(ax, 1, 2, ymax + gap * 3.5, hr["p_nih"])
    _sig_bracket(ax, 1, 3, ymax + gap * 6, hr["p_niu"])
    ax.set_ylim(0, ymax + gap * 9)

    ax.set_title(
        f"Triplet-matched ({spec.description})\n"
        f"median: {hr['med_ni']:.0f} / {hr['med_h']:.0f} / "
        f"{hr['med_u']:.0f}",
        fontsize=FS-1)


# ══════════════════════════════════════════════════════════════════
#  COMPOSITE FIGURE
# ══════════════════════════════════════════════════════════════════

def build_composite_figure(data, dm, stats):
    """
    Composite alpha diversity figure: lifestyle + health.

    Layout (2 x 3):
        A  Ind vs NI (300kb)  B  Ind vs NI (10%)   C  Ind vs Trans (10%)
        D  Redistribution     E  Triplet (300kb)    F  Triplet (10%)
    """
    global FS
    apply_oi_style()
    FS = current_fs()   # style is set per figure; read it now, not at import
    from .read_redistribution import redistribute_and_compute_shannon

    ref = data.ref_name
    s1_main = config.section_dir(ref, "s1", "main")

    flat_spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    breadth_spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)

    rng = np.random.default_rng(42)
    ind_c = IND_COLOR
    ni_c = NI_COLOR
    trans_c = config.LIFESTYLE_COLORS["Transitional"]

    # ── Richness for each panel ──
    def _rich(ids_a, ids_b, spec):
        all_ids = list(ids_a) + list(ids_b)
        pres = build_presence(data.breadth.loc[all_ids],
                              data.depth.loc[all_ids],
                              data.lengths, spec=spec)
        r = pres.sum(axis=1)
        ra = r.loc[ids_a].values
        rb = r.loc[ids_b].values
        _, p = mannwhitneyu(ra, rb, alternative="two-sided")
        return ra, rb, np.median(ra) / max(np.median(rb), 1e-10), p

    ri_200, rn_200, rat_ni_200, p_ni_200 = _rich(
        dm.ind_vs_ni_ind, dm.ind_vs_ni_ni, flat_spec)
    ri_10, rn_10, rat_ni_10, p_ni_10 = _rich(
        dm.ind_vs_ni_ind, dm.ind_vs_ni_ni, breadth_spec)
    rit_10, rtr_10, rat_tr_10, p_tr_10 = _rich(
        dm.ind_vs_trans_ind, dm.ind_vs_trans_trans, breadth_spec)

    # ── Redistribution ──
    all_c = list(dm.ind_vs_ni_ind) + list(dm.ind_vs_ni_ni)
    result_df, L_ref = redistribute_and_compute_shannon(
        data.breadth, data.depth, data.lengths, all_c,
        flat_threshold_bp=config.ACCUM_MAIN_TEXT_FLAT_BP)
    res_ind = result_df.loc[result_df.index.isin(dm.ind_vs_ni_ind)]
    res_ni = result_df.loc[result_df.index.isin(dm.ind_vs_ni_ni)]

    # ── Triplet health ──
    print(f"\n    Triplet health richness (300kb):")
    hr_200 = compute_triplet_health_richness(data, dm, stats, flat_spec)
    print(f"\n    Triplet health richness (10%):")
    hr_10 = compute_triplet_health_richness(data, dm, stats, breadth_spec)

    # ════════════════════════════════════════════════════════
    #  BUILD FIGURE
    # ════════════════════════════════════════════════════════
    fig = plt.figure(figsize=page_figsize(n_cols=3, n_rows=2))
    gs = gridspec.GridSpec(2, 3, hspace=0.28, wspace=0.22)

    # A: Ind vs NI, 300kb
    ax = fig.add_subplot(gs[0, 0])
    _richness_box(ax, ri_200, rn_200,
                  len(dm.ind_vs_ni_ind), len(dm.ind_vs_ni_ni),
                  "Industrial", "Non-Industrial", ind_c, ni_c, rng)
    ax.set_title(f"Ind vs NI (>=300 kb)\n"
                 f"ratio = {rat_ni_200:.2f}x, p = {p_ni_200:.1e}",
                 fontsize=FS-1)
    ax.text(-0.08, 1.05, "A", transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold")

    # B: Ind vs NI, 10%
    ax = fig.add_subplot(gs[0, 1])
    _richness_box(ax, ri_10, rn_10,
                  len(dm.ind_vs_ni_ind), len(dm.ind_vs_ni_ni),
                  "Industrial", "Non-Industrial", ind_c, ni_c, rng)
    ax.set_title(f"Ind vs NI (>=10% breadth)\n"
                 f"ratio = {rat_ni_10:.2f}x, p = {p_ni_10:.1e}",
                 fontsize=FS-1)
    ax.text(-0.08, 1.05, "B", transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold")

    # C: Ind vs Trans, 10%
    ax = fig.add_subplot(gs[0, 2])
    _richness_box(ax, rit_10, rtr_10,
                  len(dm.ind_vs_trans_ind), len(dm.ind_vs_trans_trans),
                  "Industrial", "Transitional", ind_c, trans_c, rng)
    ax.set_title(f"Ind vs Trans (>=10% breadth)\n"
                 f"ratio = {rat_tr_10:.2f}x, p = {p_tr_10:.1e}",
                 fontsize=FS-1)
    ax.text(-0.08, 1.05, "C", transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold")

    # Sync top row y-axes
    ymax_top = max(fig.axes[i].get_ylim()[1] for i in range(3))
    for i in range(3):
        fig.axes[i].set_ylim(0, ymax_top * 1.02)

    # D: Redistribution
    ax = fig.add_subplot(gs[1, 0])
    obs_i = res_ind["richness_observed"].values
    obs_n = res_ni["richness_observed"].values
    cor_i = res_ind["richness_corrected"].values
    cor_n = res_ni["richness_corrected"].values

    positions = [1, 2, 3.5, 4.5]
    bp = ax.boxplot([obs_i, obs_n, cor_i, cor_n], positions=positions,
                    patch_artist=True, widths=0.45, showfliers=False)
    for patch, c in zip(bp["boxes"], [ind_c, ni_c, ind_c, ni_c]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axvline(2.75, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(1.5, ax.get_ylim()[1] * 0.97, "Observed",
            ha="center", fontsize=FS-2, color="gray")
    ax.text(4.0, ax.get_ylim()[1] * 0.97, "Corrected",
            ha="center", fontsize=FS-2, color="gray")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Ind\n(obs)", "NI\n(obs)",
                        "Ind\n(corr)", "NI\n(corr)"], fontsize=FS-1)
    ratio_obs = np.median(obs_i) / max(np.median(obs_n), 1e-10)
    ratio_cor = np.median(cor_i) / max(np.median(cor_n), 1e-10)
    _, p_cor = mannwhitneyu(cor_i, cor_n, alternative="two-sided")
    ax.set_ylabel("Richness (SGBs detected)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Genome-length correction\n"
        f"obs {ratio_obs:.2f}x -> corr {ratio_cor:.2f}x, "
        f"p = {p_cor:.1e}", fontsize=FS-1)
    ax.text(-0.08, 1.05, "D", transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold")

    # E: Triplet 300kb
    ax = fig.add_subplot(gs[1, 1])
    if hr_200 is not None:
        _triplet_box(ax, hr_200, rng)
    else:
        ax.text(0.5, 0.5, "No triplet data",
                transform=ax.transAxes, ha="center")
    ax.text(-0.08, 1.05, "E", transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold")

    # F: Triplet 10%
    ax = fig.add_subplot(gs[1, 2])
    if hr_10 is not None:
        _triplet_box(ax, hr_10, rng)
    else:
        ax.text(0.5, 0.5, "No triplet data",
                transform=ax.transAxes, ha="center")
    ax.text(-0.08, 1.05, "F", transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold")

    # Sync E, F y-axes
    if hr_200 is not None and hr_10 is not None:
        ymax_bot = max(fig.axes[4].get_ylim()[1],
                       fig.axes[5].get_ylim()[1])
        fig.axes[4].set_ylim(0, ymax_bot)
        fig.axes[5].set_ylim(0, ymax_bot)

    fig.suptitle(
        f"Alpha diversity: lifestyle and health [{ref}]\n"
        f"All comparisons 1:1 or triplet depth-matched",
        fontsize=FS+2, fontweight="bold", y=0.99)

    fpath = os.path.join(s1_main,
                         "figure_alpha_diversity_composite.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ══════════════════════════════════════════════════════════════════
#  CROSS-REFERENCE FOREST PLOT
# ══════════════════════════════════════════════════════════════════

def build_cross_reference_forest(refs=None):
    """Combined forest plot from both UHGG and lev robustness CSVs."""
    if refs is None:
        refs = ["UHGG", "lev"]

    frames = []
    for ref in refs:
        p = os.path.join("results", ref, "section1_diversity", "main",
                         "robustness_summary.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"    [cross-ref] Missing {p}")
    if not frames:
        print("    [cross-ref] No data — skipping")
        return

    df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(
        figsize=page_figsize(height_in=max(2.6, len(df) * 0.28 + 1.2)))
    labels = [f"{r['reference']} | {r['method']} — {r['detection']}"
              for _, r in df.iterrows()]
    y = np.arange(len(labels))
    r = df["ratio"].values
    lo, hi = df["ci_lo"].values, df["ci_hi"].values
    colors = ["#1B3A6B" if rv == "UHGG" else "#B35A1F"
              for rv in df["reference"]]

    xerr_lo, xerr_hi, lo, hi = _nonnegative_xerr(r, lo, hi)
    ax.errorbar(r, y, xerr=[xerr_lo, xerr_hi], fmt="none",
                ecolor="gray", capsize=3, lw=1)
    ax.scatter(r, y, c=colors, s=40, zorder=4,
               edgecolors="white", lw=0.5)
    ax.axvline(1.0, color="gray", ls="--", lw=1)

    corr = df[df["method"] != "Uncorrected"]["ratio"]
    if len(corr):
        ax.axvspan(corr.min(), corr.max(), color="#C8E6C9", alpha=0.3,
                   zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS-2)
    ax.set_xlabel("Richness ratio (Industrial / Non-Industrial)")
    ax.set_title("Robustness across references, methods, and thresholds",
                 fontsize=FS+6, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1B3A6B", label="UHGG"),
                       Patch(color="#B35A1F", label="lev")],
              fontsize=FS-1, loc="lower right")
    plt.tight_layout()

    outdir = os.path.join("results", "combined")
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir,
                         "figure_robustness_cross_reference.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")



# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
#  ROBUSTNESS — NI SUB-POPULATION SPLITS
# ══════════════════════════════════════════════════════════════════
# NI SUB-POPULATION SPLITS


def _interpolate_richness_at_depth(data_obj, ids_list, spec, ref_log_depths):
    """
    For a group of samples, bin richness vs depth and interpolate
    predicted richness at reference depths.
    Returns dict {ref_label: predicted_median_richness}.
    """
    from scipy.interpolate import interp1d
    uids = list(dict.fromkeys(ids_list))
    if len(uids) < 20:
        return {}
    all_ids = uids
    presence = build_presence(data_obj.breadth.loc[all_ids],
                              data_obj.depth.loc[all_ids],
                              data_obj.lengths, spec=spec)
    richness = presence.sum(axis=1)
    richness = richness[~richness.index.duplicated(keep="first")]
    ld = np.log10(data_obj.total_bp.reindex(uids).values.astype(float))
    ri = richness.reindex(uids).values
    ok = np.isfinite(ld) & np.isfinite(ri)
    if ok.sum() < 20:
        return {}
    ld_ok, ri_ok = ld[ok], ri[ok]
    edges = np.percentile(ld_ok, np.linspace(0, 100, 19))
    edges[0] -= 0.01; edges[-1] += 0.01
    bdf = _bin_richness(ld_ok, ri_ok, edges)
    if len(bdf) < 3:
        return {}
    bx = bdf["bin_mid"].values
    by = bdf["median"].values
    f = interp1d(bx, by, kind="linear", bounds_error=False, fill_value=np.nan)
    # Extrapolation: log-linear fit on upper half of data (monotonic power law)
    pos = by > 0
    extrap_fn = None
    if pos.sum() >= 3:
        # Use upper half of bins for extrapolation slope
        n_upper = max(3, pos.sum() // 2)
        bx_up = bx[pos][-n_upper:]
        by_up = by[pos][-n_upper:]
        if len(bx_up) >= 2 and np.all(by_up > 0):
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(bx_up, np.log10(by_up), 1)  # linear in log-log
            extrap_fn = lambda x, c=coeffs: 10**P.polyval(x, c)
    result = {}
    for label, log_d in ref_log_depths.items():
        pred = float(f(log_d))
        is_extrap = False
        if (not np.isfinite(pred) or pred <= 0) and extrap_fn is not None:
            pred = float(extrap_fn(log_d))
            is_extrap = True
        if np.isfinite(pred) and pred > 0:
            result[label] = (pred, is_extrap)
    return result


def _audit_high_depth_studies(meta, total_bp, lifestyle, min_log_depth=9.5, top_n=10):
    """
    Identify studies contributing high-depth samples for a given lifestyle.
    Prints a diagnostic table.
    """
    mask = meta["Lifestyle"] == lifestyle
    ld = np.log10(total_bp.loc[meta.index[mask]].values.astype(float))
    ids = meta.index[mask]
    high = ld >= min_log_depth
    high_ids = ids[high]
    if len(high_ids) == 0:
        print(f"    No {lifestyle} samples with >= 10^{min_log_depth:.1f} bp")
        return
    study_col = None
    for c in ["StudyFolder", "Study", "Author"]:
        if c in meta.columns:
            study_col = c
            break
    if study_col is None:
        print(f"    No study column found")
        return
    high_meta = meta.loc[high_ids]
    counts = high_meta[study_col].value_counts().head(top_n)
    total_in_lf = mask.sum()
    print(f"    {lifestyle}: {len(high_ids)} / {total_in_lf} samples with "
          f"log10(bp) >= {min_log_depth} ({100*len(high_ids)/total_in_lf:.1f}%)")
    print(f"    Top studies ({study_col}):")
    for study, n in counts.items():
        # Get median richness for this study at high depth
        study_mask = high_meta[study_col] == study
        ld_study = np.log10(total_bp.loc[high_meta.index[study_mask]].values.astype(float))
        print(f"      {study}: n={n}, median_depth={np.median(10**ld_study)/1e9:.1f} Gb")


def _find_carter_mask(meta):
    """Carter et al. 2023 meta-analysis samples. See persistence.carter_cohort.

    Kept as a thin wrapper because callers expect (mask, diagnostic string). The
    string match this used to do found Carter's own sequencing (389 rows, 82%
    hunter-gatherer) rather than their 1,800-sample meta-analysis, so the figures
    built on it described the wrong cohort.
    """
    from .carter_cohort import find_carter_cohort, describe
    mask, diag = find_carter_cohort(meta, strict=False)
    return mask, describe(diag)

def _find_hadza_mask(meta):
    """
    Identify Hadza samples via flexible matching across multiple columns.
    Searches case-insensitively for 'hadza' in Population, Detailed location,
    Sample ID, StudyFolder, Author. Falls back to Tanzania H-G.
    """
    import pandas as pd
    mask = pd.Series(False, index=meta.index)
    search_cols = ["Population", "Detailed location", "Sample ID",
                   "StudyFolder", "Author"]
    matched_col = None
    for col in search_cols:
        if col not in meta.columns:
            continue
        col_match = meta[col].astype(str).str.contains(
            "hadza", case=False, na=False)
        if col_match.any():
            mask = mask | col_match
            if matched_col is None:
                matched_col = col
    if not mask.any():
        country_col = None
        for cc in ["Country", "Country (full)", "Country_norm"]:
            if cc in meta.columns:
                country_col = cc
                break
        if country_col is not None:
            tz_mask = meta[country_col].astype(str).str.contains(
                "TZA|Tanzania", case=False, na=False)
            hg_mask = meta["Lifestyle"] == "H-G"
            mask = tz_mask & hg_mask
            if mask.any():
                matched_col = f"{country_col}=Tanzania+H-G"
    n = mask.sum()
    diag = f"Hadza detection: n={n}"
    if matched_col:
        diag += f" (matched via {matched_col})"
    else:
        diag += " (NO MATCH)"
    if "Population" in meta.columns:
        hg = meta[meta["Lifestyle"] == "H-G"]
        if len(hg) > 0:
            pop_counts = hg["Population"].value_counts().head(10)
            diag += f"\n    H-G Population values: {pop_counts.to_dict()}"
    return mask, diag

def _plot_depth_distributions_3group(ax, depths, labels, colors, ns):
    lo = min(d.min() for d in depths if len(d) > 0) - 0.3
    hi = max(d.max() for d in depths if len(d) > 0) + 0.3
    bins = np.linspace(lo, hi, 50)
    dg = np.linspace(lo, hi, 300)
    for d, lbl, c, n in zip(depths, labels, colors, ns):
        if len(d) < 2: continue
        ax.hist(d, bins=bins, density=True, alpha=0.2, color=c,
                edgecolor="white", lw=0.3, label=f"{lbl} (n={n:,})")
        ax.plot(dg, gaussian_kde(d, bw_method="silverman")(dg), color=c, lw=2)
        ax.axvline(np.median(d), color=c, ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("log10(Total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title("Sequencing depth distributions", fontsize=FS-1)
    ax.legend(fontsize=FS-2, loc="upper left")


def _plot_binned_richness_3group(ax, data_obj, ids_groups, labels, colors, spec):
    all_ids = list(dict.fromkeys(i for grp in ids_groups for i in grp))
    presence = build_presence(data_obj.breadth.loc[all_ids],
                              data_obj.depth.loc[all_ids],
                              data_obj.lengths, spec=spec)
    richness = presence.sum(axis=1)
    richness = richness[~richness.index.duplicated(keep="first")]
    log_d_all = np.log10(data_obj.total_bp.loc[all_ids].values.astype(float))
    valid = np.isfinite(log_d_all)
    edges = np.percentile(log_d_all[valid], np.linspace(0, 100, 19))
    edges[0] -= 0.01
    edges[-1] += 0.01
    for ids, lbl, c in zip(ids_groups, labels, colors):
        if len(ids) < 10: continue
        uids = list(dict.fromkeys(ids))
        ld = np.log10(data_obj.total_bp.reindex(uids).values.astype(float))
        ri = richness.reindex(uids).values
        ok = np.isfinite(ld) & np.isfinite(ri)
        bdf = _bin_richness(ld[ok], ri[ok], edges)
        if bdf.empty: continue
        ax.plot(bdf["bin_mid"], bdf["median"], "o-", color=c, ms=4, lw=1.5, label=lbl, zorder=3)
        ax.fill_between(bdf["bin_mid"], bdf["q25"], bdf["q75"], color=c, alpha=0.12, zorder=2)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i % 2 == 0: ax.axvspan(lo, hi, color="gray", alpha=0.04, zorder=0)
    # Reference depth lines and interpolated/extrapolated richness
    ref_depths = {"1 Gb": 9.0, "3 Gb": np.log10(3e9), "10 Gb": 10.0}
    for label, ld_ref in ref_depths.items():
        ax.axvline(ld_ref, color="gray", ls=":", lw=0.7, alpha=0.5)
        ax.text(ld_ref, ax.get_ylim()[1] * 0.98, label,
                ha="center", va="top", fontsize=FS-2, color="gray", rotation=90)

    ann_lines = []
    for ids, lbl, c in zip(ids_groups, labels, colors):
        if len(ids) < 10:
            continue
        uids = list(dict.fromkeys(ids))
        ld = np.log10(data_obj.total_bp.reindex(uids).values.astype(float))
        ri = richness.reindex(uids).values
        ok = np.isfinite(ld) & np.isfinite(ri)
        bdf = _bin_richness(ld[ok], ri[ok], edges)
        if bdf.empty or len(bdf) < 3:
            continue

        bx = bdf["bin_mid"].values
        by = bdf["median"].values

        from scipy.interpolate import interp1d
        f_interp = interp1d(bx, by, kind="linear", bounds_error=False,
                            fill_value=np.nan)

        # Log-linear fit on upper half of data for monotonic extrapolation
        pos = by > 0
        n_upper = max(3, pos.sum() // 2)
        bx_up = bx[pos][-n_upper:]
        by_up = by[pos][-n_upper:]
        if len(bx_up) >= 2 and np.all(by_up > 0):
            from numpy.polynomial import polynomial as P
            coeffs_ext = P.polyfit(bx_up, np.log10(by_up), 1)
            extrap_fn = lambda x, c=coeffs_ext: 10**P.polyval(x, c)
        else:
            extrap_fn = lambda x: np.nan

        # Draw extrapolation dashes beyond data range
        data_max_ld = bx.max()
        if data_max_ld < 9.95:  # extrapolate to 10 Gb
            extrap_x = np.linspace(data_max_ld, 10.0, 30)
            extrap_y = np.array([extrap_fn(x) for x in extrap_x])
            valid_ext = np.isfinite(extrap_y) & (extrap_y > 0)
            if valid_ext.sum() > 2:
                ax.plot(extrap_x[valid_ext], extrap_y[valid_ext],
                        "--", color=c, lw=1.0, alpha=0.5, zorder=2)

        parts = []
        for ref_label, ref_ld in ref_depths.items():
            # Try interpolation first, fall back to extrapolation
            pred = f_interp(ref_ld)
            is_extrap = False
            if not np.isfinite(pred) or pred <= 0:
                pred = extrap_fn(ref_ld)
                is_extrap = True
            if np.isfinite(pred) and pred > 0:
                marker = "D" if is_extrap else "o"
                alpha_m = 0.6 if is_extrap else 1.0
                ax.plot(ref_ld, pred, marker, color=c, ms=5, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        alpha=alpha_m)
                suffix = "*" if is_extrap else ""
                parts.append(f"{ref_label}={pred:.0f}{suffix}")
        if parts:
            ann_lines.append(f"{lbl}: " + ", ".join(parts))
    if ann_lines:
        ann_text = chr(10).join(ann_lines)
        ax.text(0.02, 0.02, ann_text,
                transform=ax.transAxes, fontsize=FS-2, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, lw=0.5))
    ax.set_xlabel("log10(Total mapped bp)")
    ax.set_ylabel("Richness (SGBs)")
    ax.set_title("Binned median +/- IQR" + chr(10) + f"({spec.description})", fontsize=FS-1)
    ax.legend(fontsize=FS-2, loc="upper left")

def _three_group_box(ax, v_focal, v_other, v_ind,
                      n_focal, n_other, n_ind,
                      label_focal, label_other, label_ind,
                      color_focal, color_other, color_ind, title):
    rng = np.random.default_rng(42)
    groups = [v_focal, v_other, v_ind]
    colors = [color_focal, color_other, color_ind]
    labels_x = [f"{label_focal}\n(n={n_focal:,})",
                f"{label_other}\n(n={n_other:,})",
                f"{label_ind}\n(n={n_ind:,})"]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
        patch.set_edgecolor("none")
    for i, (vals, c) in enumerate(zip(groups, colors), 1):
        ns = min(len(vals), 2000)
        if ns == 0: continue
        ix = rng.choice(len(vals), ns, replace=False)
        jt = rng.uniform(-0.12, 0.12, ns)
        ax.scatter(np.full(ns, i) + jt, vals[ix], c=c, alpha=0.05,
                   s=2, zorder=1, edgecolors="none")
    ax.set_xticklabels(labels_x, fontsize=FS-2)
    ax.set_ylabel("Richness")
    ax.set_title(title, fontsize=FS-1)
    for i, vals in enumerate(groups):
        if len(vals) == 0: continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        x = i + 1
        ax.text(x, (med + q3) / 2.0, f"{med:.0f}",
                ha="center", va="center", fontsize=FS-2,
                color="white", fontweight="bold", zorder=5)
        ax.text(x, (q1 + med) / 2.0, f"[{q1:.0f}-{q3:.0f}]",
                ha="center", va="center", fontsize=FS-2,
                color="white", zorder=5)
    ax.grid(axis="y", alpha=0.3)

def _plot_split_forest(ax, entries, split_label, other_label,
                        color_focal, color_other, ref):
    labels_f = []
    for e in entries:
        sg = e["subgroup"]
        mt = e["method"]
        sl = e["spec_label"]
        labels_f.append(f"{sg} | {mt} -- {sl}")
    y = np.arange(len(labels_f))
    r = np.array([e["ratio"] for e in entries])
    lo = np.array([e["ci_lo"] for e in entries])
    hi = np.array([e["ci_hi"] for e in entries])
    colors_f = [color_focal if e["subgroup"] == split_label
                else color_other for e in entries]
    xerr_lo, xerr_hi, lo, hi = _nonnegative_xerr(r, lo, hi)
    ax.errorbar(r, y, xerr=[xerr_lo, xerr_hi], fmt="none",
                ecolor="gray", capsize=4, lw=1.5, zorder=2)
    ax.scatter(r, y, c=colors_f, s=50, zorder=4,
               edgecolors="white", lw=0.5)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_f, fontsize=FS-2)
    ax.set_xlabel("Richness ratio (subgroup / Industrial)")
    ax.set_title(f"Robustness: {split_label} vs {other_label} [{ref}]",
                 fontsize=FS, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=color_focal, label=split_label),
        Patch(color=color_other, label=other_label)],
        fontsize=FS-2, loc="lower right")
    for i, (rv, lv, hv) in enumerate(zip(r, lo, hi)):
        if np.isfinite(lv):
            ax.text(hv + 0.005, i,
                    f"  {rv:.2f} [{lv:.2f}, {hv:.2f}]",
                    va="center", fontsize=FS-2, color="gray")


def build_robustness_ni_split_figure(data, dm, stats,
                                      split_label, other_label,
                                      filename_suffix,
                                      focal_mask=None,
                                      split_col=None, split_value=None):
    ref = data.ref_name
    meta = data.meta
    s1_main = config.section_dir(ref, "s1", "main")

    flat_spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    breadth_spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    specs = [flat_spec, breadth_spec]

    # Filter samples below minimum depth
    min_bp = getattr(config, "MIN_TOTAL_BP", 1e8)
    valid_depth = data.total_bp >= min_bp
    n_dropped = (~valid_depth).sum()
    if n_dropped > 0:
        print(f"    Filtered {n_dropped} samples with < {min_bp:.0e} bp")

    if focal_mask is None:
        if split_col is not None and split_col in meta.columns:
            ni_mask_all = meta["Lifestyle"].isin(["H-G", "Transitional"])
            focal_mask = ni_mask_all & (meta[split_col] == split_value)
        else:
            print(f"    [WARN] Cannot build focal mask -- skipping")
            return

    n_focal = focal_mask.sum()
    if n_focal == 0:
        print(f"    [WARN] {split_label}: n=0 -- skipping {filename_suffix}")
        return

    COLOR_FOCAL = config.OI["orange"]
    COLOR_OTHER = config.OI["yellow"]

    forest_entries = []

    fig = plt.figure(figsize=page_figsize(n_cols=3, n_rows=5))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1.0, 0.85, 0.85, 0.85, 0.7],
                           hspace=0.30, wspace=0.22)
    panels = "ABCDEFGHIJKLM"
    panel_idx = 0

    # Row 0: DATA panels
    ax = fig.add_subplot(gs[0, 0])
    ind_mask_all = meta["Lifestyle"] == "Industrial"
    ni_mask_all = meta["Lifestyle"].isin(["H-G", "Transitional"])
    other_mask = ni_mask_all & (~focal_mask)

    ids_all_ind = meta.index[ind_mask_all & valid_depth].tolist()
    ids_all_focal = meta.index[focal_mask & valid_depth].tolist()
    ids_all_other = meta.index[other_mask & valid_depth].tolist()

    # Study audit for the "other" NI group (often Transitional)
    print(f"    --- Study audit: {other_label} at high depth ---")
    _audit_high_depth_studies(meta, data.total_bp, "Transitional")
    if split_label == "Hadza":
        print(f"    --- Study audit: Hadza at high depth ---")
        _audit_high_depth_studies(meta, data.total_bp, "H-G")

    # Compute depth-standardized richness predictions
    REF_DEPTHS = {"1 Gb": 9.0, "3 Gb": np.log10(3e9), "10 Gb": 10.0}
    depth_std_results = {}
    for grp_label, grp_ids in [(split_label, ids_all_focal),
                                (other_label, ids_all_other),
                                ("Industrial", ids_all_ind)]:
        for sp in specs:
            preds_raw = _interpolate_richness_at_depth(data, grp_ids, sp, REF_DEPTHS)
            # Unpack (value, is_extrap) tuples
            preds = {}
            preds_extrap = {}
            for k, (v, is_ext) in preds_raw.items():
                preds[k] = v
                preds_extrap[k] = is_ext
            depth_std_results[(grp_label, sp.short_label)] = preds
            depth_std_results[(grp_label, sp.short_label, "extrap")] = preds_extrap
            if preds:
                parts = [f"{k}={v:.0f}{'*' if preds_extrap.get(k) else ''}" for k, v in preds.items()]
                print(f"    Predicted richness ({grp_label}, {sp.description}): {', '.join(parts)}")

    d_ind = np.log10(data.total_bp.loc[ids_all_ind].values.astype(float))
    d_focal = np.log10(data.total_bp.loc[ids_all_focal].values.astype(float))
    d_other = np.log10(data.total_bp.loc[ids_all_other].values.astype(float))
    d_ind = d_ind[np.isfinite(d_ind)]
    d_focal = d_focal[np.isfinite(d_focal)]
    d_other = d_other[np.isfinite(d_other)]

    _plot_depth_distributions_3group(
        ax,
        [d_focal, d_other, d_ind],
        [split_label, other_label, "Industrial"],
        [COLOR_FOCAL, COLOR_OTHER, IND_COLOR],
        [len(ids_all_focal), len(ids_all_other), len(ids_all_ind)])
    ax.text(-0.10, 1.035, panels[panel_idx], transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
    panel_idx += 1

    ax = fig.add_subplot(gs[0, 1])
    _plot_binned_richness_3group(
        ax, data,
        [ids_all_focal, ids_all_other, ids_all_ind],
        [split_label, other_label, "Industrial"],
        [COLOR_FOCAL, COLOR_OTHER, IND_COLOR],
        flat_spec)
    ax.text(-0.10, 1.035, panels[panel_idx], transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
    panel_idx += 1

    ax = fig.add_subplot(gs[0, 2])
    _plot_binned_richness_3group(
        ax, data,
        [ids_all_focal, ids_all_other, ids_all_ind],
        [split_label, other_label, "Industrial"],
        [COLOR_FOCAL, COLOR_OTHER, IND_COLOR],
        breadth_spec)
    ax.text(-0.10, 1.035, panels[panel_idx], transform=ax.transAxes,
            fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
    panel_idx += 1

    # ── Row 3: Predicted richness bar charts (J, K, L) ─────────────
    # Compute method-specific predictions at reference depths
    REF_LABELS = ["1 Gb", "3 Gb", "10 Gb"]
    REF_LOG_D = [9.0, np.log10(3e9), 10.0]
    depth_bar_colors = ["#b3cde3", "#6baed6", "#08519c"]  # light->dark blue for 1/3/10 Gb

    def _predict_at_refs(ids_list, sp, weights=None):
        """Compute binned richness for a group and interpolate/extrapolate at ref depths."""
        uids = list(dict.fromkeys(ids_list))
        if len(uids) < 20:
            return [np.nan] * 3, [False] * 3
        pres = build_presence(data.breadth.loc[uids],
                               data.depth.loc[uids],
                               data.lengths, spec=sp)
        rich = pres.sum(axis=1)
        rich = rich[~rich.index.duplicated(keep="first")]
        ld = np.log10(data.total_bp.reindex(uids).values.astype(float))
        ri = rich.reindex(uids).values
        ok = np.isfinite(ld) & np.isfinite(ri)
        if ok.sum() < 20:
            return [np.nan] * 3, [False] * 3
        ld_ok, ri_ok = ld[ok], ri[ok]
        w_ok = None
        if weights is not None:
            w_ok = weights[ok]
            # Zero out negligible weights
            w_ok = np.where(w_ok > 1e-20, w_ok, 0)
            if w_ok.sum() < 1e-10:
                w_ok = None
        edges = np.percentile(ld_ok, np.linspace(0, 100, 19))
        edges[0] -= 0.01; edges[-1] += 0.01
        if w_ok is not None:
            bdf = _bin_richness_weighted(ld_ok, ri_ok, w_ok, edges)
        else:
            bdf = _bin_richness(ld_ok, ri_ok, edges)
        if len(bdf) < 3:
            return [np.nan] * 3, [False] * 3
        bx = bdf["bin_mid"].values
        by = bdf["median"].values
        from scipy.interpolate import interp1d
        f_interp = interp1d(bx, by, kind="linear", bounds_error=False, fill_value=np.nan)
        # Log-linear extrapolation on upper half
        pos = by > 0
        extrap_fn = None
        n_upper = max(3, pos.sum() // 2)
        bx_up = bx[pos][-n_upper:]
        by_up = by[pos][-n_upper:]
        if len(bx_up) >= 2 and np.all(by_up > 0):
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(bx_up, np.log10(by_up), 1)
            extrap_fn = lambda x, c=coeffs: 10**P.polyval(x, c)
        preds = []
        is_extrap = []
        for ref_ld in REF_LOG_D:
            v = float(f_interp(ref_ld))
            ext = False
            if (not np.isfinite(v) or v <= 0) and extrap_fn is not None:
                v = float(extrap_fn(ref_ld))
                ext = True
            preds.append(v if (np.isfinite(v) and v > 0) else np.nan)
            is_extrap.append(ext)
        return preds, is_extrap

    # Helper for weighted binned richness (for IPW)
    def _bin_richness_weighted(ld, ri, w, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (ld >= lo) & (ld < hi)
            if mask.sum() < 5:
                continue
            # Weighted median
            wm = _weighted_median(ri[mask], w[mask] / w[mask].sum())
            # Weighted quartiles (approximate via resampling)
            rows.append(dict(bin_mid=(lo + hi) / 2, median=wm, q25=np.nan, q75=np.nan))
        return pd.DataFrame(rows)

    # Compute predictions for each method x threshold x group
    method_preds = {}  # (method, sp_label, group) -> (preds_list, extrap_list)
    grp_labels = [split_label, other_label, "Industrial"]
    grp_colors = [COLOR_FOCAL, COLOR_OTHER, IND_COLOR]

    for sp in specs:
        sl = sp.short_label
        for method in ["Uncorrected", "Depth-matched", "IPW"]:
            if method == "Uncorrected":
                id_map = {split_label: ids_all_focal,
                          other_label: ids_all_other,
                          "Industrial": ids_all_ind}
            elif method == "Depth-matched":
                dm_ind_list = list(dm.ind_vs_ni_ind)
                dm_ni_list = list(dm.ind_vs_ni_ni)
                dm_focal = [s for s in dm_ni_list if focal_mask.get(s, False)]
                dm_other = [s for s in dm_ni_list if not focal_mask.get(s, False)]
                id_map = {split_label: dm_focal,
                          other_label: dm_other,
                          "Industrial": dm_ind_list}
            else:  # IPW
                id_map = {split_label: ids_all_focal,
                          other_label: ids_all_other,
                          "Industrial": ids_all_ind}

            for grp in grp_labels:
                ids = id_map[grp]
                if method == "IPW":
                    # Compute IPW weights for this group
                    uids = list(dict.fromkeys(ids))
                    ld_g = np.log10(data.total_bp.reindex(uids).values.astype(float))
                    ok_g = np.isfinite(ld_g)
                    # Target: pooled distribution of all 3 groups
                    all_ld = np.concatenate([
                        np.log10(data.total_bp.reindex(list(dict.fromkeys(id_map[g]))).values.astype(float))
                        for g in grp_labels])
                    all_ld = all_ld[np.isfinite(all_ld)]
                    if len(all_ld) > 50 and ok_g.sum() > 20:
                        kde_t = gaussian_kde(all_ld, bw_method="silverman")
                        kde_g = gaussian_kde(ld_g[ok_g], bw_method="silverman")
                        # Build full-length weight array aligned to uids
                        w_full = np.zeros(len(uids))
                        w_full[ok_g] = (kde_t(ld_g[ok_g]).ravel() /
                                        (kde_g(ld_g[ok_g]).ravel() + 1e-20))
                        lo_w, hi_w = np.percentile(w_full[ok_g], [1, 99])
                        w_full = np.clip(w_full, lo_w, hi_w)
                        w_full[~ok_g] = 0
                        preds, ext = _predict_at_refs(ids, sp, weights=w_full)
                    else:
                        preds, ext = _predict_at_refs(ids, sp)
                else:
                    preds, ext = _predict_at_refs(ids, sp)
                method_preds[(method, sl, grp)] = (preds, ext)


    # Rows 1-2: Boxplots per method x threshold
    # Pre-compute IPW weights for 3 groups
    def _compute_ipw_weights_3group(ids_f, ids_o, ids_i, tbp):
        """Compute IPW weights for 3 groups to a common target distribution."""
        groups = [ids_f, ids_o, ids_i]
        ld_groups = []
        for ids in groups:
            uids = list(dict.fromkeys(ids))
            ld = np.log10(tbp.reindex(uids).values.astype(float))
            ok = np.isfinite(ld)
            filt_ids = [s for s, v in zip(uids, ok) if v]
            ld_groups.append((filt_ids, ld[ok]))
        # Pooled target
        all_ld = np.concatenate([g[1] for g in ld_groups if len(g[1]) > 10])
        if len(all_ld) < 50:
            return None
        kde_target = gaussian_kde(all_ld, bw_method="silverman")
        weights_out = []
        for filt_ids, ld_g in ld_groups:
            if len(ld_g) < 10:
                weights_out.append((filt_ids, None))
                continue
            kde_g = gaussian_kde(ld_g, bw_method="silverman")
            w = kde_target(ld_g).ravel() / (kde_g(ld_g).ravel() + 1e-20)
            lo_w, hi_w = np.percentile(w, [1, 99])
            w = np.clip(w, lo_w, hi_w)
            w /= w.sum()
            weights_out.append((filt_ids, w))
        return weights_out

    for row, sp in enumerate(specs, 1):
        sl = sp.short_label
        for col, method in enumerate(["Uncorrected", "Depth-matched", "IPW"]):
            ax = fig.add_subplot(gs[row, col])
            if method == "Uncorrected":
                ids_ind = ids_all_ind
                ids_focal = ids_all_focal
                ids_other = ids_all_other
            elif method == "Depth-matched":
                ids_ind = list(dm.ind_vs_ni_ind)
                dm_ni = list(dm.ind_vs_ni_ni)
                ids_focal = [s for s in dm_ni if focal_mask.get(s, False)]
                ids_other = [s for s in dm_ni if not focal_mask.get(s, False)]
            else:  # IPW
                ids_ind = ids_all_ind
                ids_focal = ids_all_focal
                ids_other = ids_all_other

            all_ids = list(dict.fromkeys(ids_ind + ids_focal + ids_other))
            presence = build_presence(data.breadth.loc[all_ids],
                                      data.depth.loc[all_ids],
                                      data.lengths, spec=sp)
            richness = presence.sum(axis=1)
            richness = richness[~richness.index.duplicated(keep="first")]

            ri = richness.reindex(list(dict.fromkeys(ids_ind))).values.astype(float) if ids_ind else np.array([])
            rf = richness.reindex(list(dict.fromkeys(ids_focal))).values.astype(float) if ids_focal else np.array([])
            ro = richness.reindex(list(dict.fromkeys(ids_other))).values.astype(float) if ids_other else np.array([])

            if method == "IPW":
                # Compute weights and resample for display
                ipw_res = _compute_ipw_weights_3group(
                    ids_focal, ids_other, ids_ind, data.total_bp)
                if ipw_res is not None:
                    rr = np.random.default_rng(42)
                    ns_resample = 5000
                    grp_richness = []
                    grp_n_orig = []
                    grp_wmedians = []
                    for g_idx, (filt_ids, w_g) in enumerate(ipw_res):
                        r_g = richness.reindex(filt_ids).values.astype(float) if filt_ids else np.array([])
                        grp_n_orig.append(len(filt_ids))
                        if w_g is not None and len(r_g) > 0:
                            resampled = r_g[rr.choice(len(r_g), ns_resample, p=w_g, replace=True)]
                            grp_richness.append(resampled)
                            wmed = _weighted_median(r_g, w_g)
                            grp_wmedians.append(wmed)
                        else:
                            grp_richness.append(r_g)
                            grp_wmedians.append(np.median(r_g) if len(r_g) > 0 else 0)
                    rf, ro, ri = grp_richness[0], grp_richness[1], grp_richness[2]
                    wmed_f, wmed_o, wmed_i = grp_wmedians
                else:
                    wmed_f = np.median(rf) if len(rf) > 0 else 0
                    wmed_o = np.median(ro) if len(ro) > 0 else 0
                    wmed_i = np.median(ri) if len(ri) > 0 else 0

            # For IPW, pass original n and use weighted resample for display
            if method == "IPW" and ipw_res is not None:
                nf_display = grp_n_orig[0]
                no_display = grp_n_orig[1]
                ni_display = grp_n_orig[2]
            else:
                nf_display = len(rf)
                no_display = len(ro)
                ni_display = len(ri)
            _three_group_box(
                ax, rf, ro, ri,
                nf_display, no_display, ni_display,
                split_label, other_label, "Industrial",
                COLOR_FOCAL, COLOR_OTHER, IND_COLOR,
                f"{method}\n({sp.description})")
            # Annotate method-specific depth-standardized predictions
            pred_lines = []
            for grp_lbl in [split_label, other_label, "Industrial"]:
                p_vals, p_ext = method_preds.get((method, sl, grp_lbl), ([np.nan]*3, [False]*3))
                parts_ann = []
                for d_idx, ref_lbl in enumerate(REF_LABELS):
                    if np.isfinite(p_vals[d_idx]):
                        flag = "*" if p_ext[d_idx] else ""
                        parts_ann.append(f"{ref_lbl}={p_vals[d_idx]:.0f}{flag}")
                if parts_ann:
                    pred_lines.append(f"{grp_lbl}: " + ", ".join(parts_ann))
            if pred_lines:
                ax.text(0.97, 0.97, "\n".join(pred_lines),
                        transform=ax.transAxes, fontsize=FS-2,
                        ha="right", va="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85, lw=0.5))
            ax.text(-0.10, 1.035, panels[panel_idx],
                    transform=ax.transAxes, fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
            panel_idx += 1

            # Forest entries: use weighted medians for IPW
            if method == "IPW":
                med_i_forest = wmed_i
            else:
                med_i_forest = np.median(ri) if len(ri) else 0
            for lbl, rv, pval_pair, med_sub in [
                (split_label, rf, ri,
                 wmed_f if method == "IPW" else (np.median(rf) if len(rf) > 0 else 0)),
                (other_label, ro, ri,
                 wmed_o if method == "IPW" else (np.median(ro) if len(ro) > 0 else 0)),
            ]:
                if len(rv) > 0 and len(pval_pair) > 0:
                    ratio = med_sub / max(med_i_forest, 1e-10)
                    _, pv = mannwhitneyu(rv, pval_pair, alternative="two-sided")
                    ci = _bootstrap_ratio(rv, pval_pair) if len(rv) > 10 else (np.nan, np.nan)
                    forest_entries.append(dict(
                        method=method, spec_label=sp.description,
                        subgroup=lbl, ratio=ratio,
                        ci_lo=ci[0], ci_hi=ci[1], p=pv))

    for col in range(3):
        ax1 = fig.axes[3 + col]
        ax2 = fig.axes[6 + col]
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)

    # Plot the 3 prediction panels
    for col, method in enumerate(["Uncorrected", "Depth-matched", "IPW"]):
        ax = fig.add_subplot(gs[3, col])
        x_positions = np.arange(len(grp_labels))
        bar_width = 0.22
        offsets = [-bar_width, 0, bar_width]

        max_val = 0
        for d_idx, (ref_lbl, ref_color) in enumerate(zip(REF_LABELS, depth_bar_colors)):
            vals = []
            extrap_flags = []
            for grp in grp_labels:
                # Average across the two thresholds
                all_preds = []
                all_ext = []
                for sp in specs:
                    sl = sp.short_label
                    p, e = method_preds.get((method, sl, grp), ([np.nan]*3, [False]*3))
                    if np.isfinite(p[d_idx]):
                        all_preds.append(p[d_idx])
                        all_ext.append(e[d_idx])
                if all_preds:
                    vals.append(np.mean(all_preds))
                    extrap_flags.append(any(all_ext))
                else:
                    vals.append(0)
                    extrap_flags.append(False)

            bars = ax.bar(x_positions + offsets[d_idx], vals, bar_width,
                          color=ref_color, edgecolor="white", lw=0.5,
                          label=ref_lbl, zorder=3)
            for b_idx, (bar, v, ext_f) in enumerate(zip(bars, vals, extrap_flags)):
                if v > 0:
                    suffix = "*" if ext_f else ""
                    ax.text(bar.get_x() + bar.get_width()/2, v + 5,
                            f"{v:.0f}{suffix}", ha="center", va="bottom",
                            fontsize=FS-2, rotation=90)
                    if ext_f:
                        bar.set_hatch("//")
                        bar.set_edgecolor("gray")
                max_val = max(max_val, v)

        # Color-code x-axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(grp_labels, fontsize=FS-2)
        for tick_label, c in zip(ax.get_xticklabels(), grp_colors):
            tick_label.set_color(c)
            tick_label.set_fontweight("bold")
        ax.set_ylabel("Predicted richness (SGBs)")
        ax.set_title(f"{method}" + chr(10) + "(avg of both thresholds)", fontsize=FS-1)
        ax.legend(fontsize=FS-2, loc="upper right", title="Ref. depth", title_fontsize=FS-2)
        ax.set_ylim(0, max_val * 1.35)
        ax.grid(axis="y", alpha=0.2)
        panel_lbl = chr(ord("J") + col)
        ax.text(-0.08, 1.08, panel_lbl,
                transform=ax.transAxes, fontsize=FS+6, fontweight="bold")

    # Row 4: Forest plot (now panel M)
    ax = fig.add_subplot(gs[4, :])
    if forest_entries:
        _plot_split_forest(ax, forest_entries, split_label, other_label,
                            COLOR_FOCAL, COLOR_OTHER, ref)
    ax.text(-0.02, 1.06, "M",
            transform=ax.transAxes, fontsize=FS+6, fontweight="bold")

    fig.suptitle(
        f"Robustness: {split_label} vs {other_label} vs Industrial [{ref}]\n"
        f"Uncorrected + depth-matched + IPW x two detection thresholds",
        fontsize=FS+2, fontweight="bold", y=0.99)

    fpath = os.path.join(s1_main, f"figure_robustness_alpha_{filename_suffix}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")

    if forest_entries:
        csv_path = os.path.join(s1_main, f"robustness_summary_{filename_suffix}.csv")
        pd.DataFrame(forest_entries).to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════
class _SkipEmit(Exception):
    """Raised to skip publishing when the figure was drawn from the table."""


def plot_read_length_diagnostic(tables, out_dir, ref="combined"):
    """Draw the read-length diagnostic from its table, and nothing else."""
    build_read_length_diagnostic(tables=tables, ref=ref, s1_main=out_dir)
    import os
    return os.path.join(out_dir, "figure_read_length_diagnostic.png")


def build_read_length_diagnostic(data=None, dm=None, stats=None,
                                 tables=None, ref=None, s1_main=None):
    """
    Diagnostic figure: does read length systematically affect observed
    richness at a given sequencing depth?

    Computes mean_read_length = Total_bases / number_reads from metadata,
    splits into terciles (short/medium/long), and plots binned richness
    vs depth curves for each tercile.

    Layout (3 rows x 3 cols):
        Row 0 (300kb):  All samples | Industrial | Non-Industrial
        Row 1 (10%):    All samples | Industrial | Non-Industrial
        Row 2:          Read-length distribution | Residual scatter | Summary table
    """
    global FS
    # One drawing implementation, two sources. The pipeline passes `data` and
    # the panels are computed from the matrices; a redraw passes `tables` and
    # the same panels are read straight out of the published frames. Keeping
    # a single implementation is the point: a second copy of this plotting
    # code would drift from the first.
    from_table = tables is not None
    if from_table:
        ref = ref or "combined"
        s1_main = s1_main or "."
        _samp = tables["samples"].set_index("sample_id")
        _rbd = tables["richness_by_depth"]
        mean_rl = pd.to_numeric(_samp["mean_read_length_bp"], errors="coerce")
        rl_group = _samp["tercile"]
        # The published label is the display one; the comparisons below are
        # against the internal keys, so bridge it back.
        meta = pd.DataFrame(
            {"Lifestyle": config.canonicalize_lifestyle_series(
                _samp["lifestyle"])} if "lifestyle" in _samp.columns
            else {"Lifestyle": pd.Series("Industrial", index=_samp.index)},
            index=_samp.index)
        rl_source = "published table"
    else:
        ref = data.ref_name
        meta = data.meta
        s1_main = config.section_dir(ref, "s1", "main")
    _rl_rich_rows = []

    flat_spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    breadth_spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    # "300kb", not "300kB": these are mapped BASES, so the unit is the
    # kilobase. The label is the panel title AND the key the summary box
    # groups on, so both have to move together.
    specs = [(flat_spec, "300kb"), (breadth_spec, "10%")]

    # Compute mean read length
    has_reads = (not from_table) and "number_reads" in meta.columns
    has_bases = (not from_table) and "Total bases" in meta.columns
    has_median_rl = (not from_table) and "median_read_length" in meta.columns

    if from_table:
        pass
    elif has_reads and has_bases:
        n_reads = pd.to_numeric(meta["number_reads"], errors="coerce")
        t_bases = pd.to_numeric(meta["Total bases"], errors="coerce")
        mean_rl = t_bases / n_reads.replace(0, np.nan)
        rl_source = "Total bases / number_reads"
    elif has_median_rl:
        mean_rl = pd.to_numeric(meta["median_read_length"], errors="coerce")
        rl_source = "median_read_length"
    else:
        print("    [WARN] No read length info in metadata -- skipping")
        return

    valid = np.isfinite(mean_rl) & (mean_rl > 0) & (mean_rl < 500)
    n_valid = valid.sum()
    print(f"    Read length source: {rl_source}")
    print(f"    Valid samples: {n_valid:,} / {len(meta):,}")
    if n_valid < 100:
        print("    Too few samples with read length -- skipping")
        return

    # Assign terciles
    rl_valid = mean_rl[valid]
    t1, t2 = np.percentile(rl_valid, [33.3, 66.7])
    rl_group = pd.Series("Medium", index=meta.index)
    rl_group[mean_rl <= t1] = "Short"
    rl_group[mean_rl > t2] = "Long"
    rl_group[~valid] = np.nan

    tercile_colors = {"Short": "#1b9e77", "Medium": "#d95f02", "Long": "#7570b3"}
    tercile_order = ["Short", "Medium", "Long"]

    print(f"    Tercile boundaries: Short <= {t1:.0f} bp, Medium {t1:.0f}-{t2:.0f} bp, Long > {t2:.0f} bp")
    for t in tercile_order:
        n = (rl_group == t).sum()
        med = mean_rl[rl_group == t].median()
        print(f"      {t}: n={n:,}, median read length={med:.0f} bp")

    # Depth filter
    min_bp = getattr(config, "MIN_TOTAL_BP", 1e8)

    # Nine panels at page width are 1.7 in each, and each carries a title, a
    # legend and per-series labels. That produced 318 text collisions -- not a
    # placement problem, simply more text than the panels can hold. Same answer
    # as the other dense grids: widen the canvas and scale the base with it, so
    # the panels get room and the printed size holds. 24 in gives 6.5 in panels
    # at about 6.6 pt printed, which is where the robustness figure sits.
    # Authored at the printed width. A figure drawn four times too wide is
    # reduced to fit the page, and the reduction takes every line and
    # symbol with it -- a 1 pt rule landed at 0.25 pt, under the 0.5 pt
    # floor, and the type had to be inflated to stay legible.
    _W5 = PAGE_WIDTH_IN
    _MG5 = dict(top=0.90, bottom=0.05, left=0.055, right=0.985,
                wspace=0.26, hspace=0.26)
    # The canvas is now authored at the printed width, so the shared
    # scale applies: a base tuned for a 20-28 in canvas made the type
    # three times too large once the figure stopped being reduced.
    apply_oi_style()
    FS = current_fs()
    fig = plt.figure(figsize=square_figsize(n_cols=3, n_rows=3,
                                            width_in=_W5, margins=_MG5))
    # No longer wide on purpose: this figure is authored at the printed
    # width now, so any excess is overflow to be trimmed, not intent.
    fig._margin_overrides = _MG5
    # Drop the first tick on both axes: at the origin the x and y labels are a
    # few pixels apart and read as one smudged number.
    fig._plain_axes_kw = dict(x_ticks=4, y_ticks=4, prune="lower")
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 1.0, 1.0],
                           hspace=0.30, wspace=0.22)
    panels = "ABCDEFGHI"
    panel_idx = 0

    # Lifestyle subsets
    subsets = [
        ("All samples", meta.index.tolist()),
        ("Industrial", meta.index[meta["Lifestyle"] == "Industrial"].tolist()),
        ("Non-Industrial", meta.index[meta["Lifestyle"].isin(["H-G", "Transitional"])].tolist()),
    ]

    summary_rows = []
    # The residual panel draws its Spearman in the axes title and the caption
    # quotes it. Collected here so the data object can publish it: the
    # submitted source data has to carry the P values the figure shows.
    corr_rows = []

    for row, (sp, sp_label) in enumerate(specs):
        for col, (subset_label, subset_ids) in enumerate(subsets):
            ax = fig.add_subplot(gs[row, col])

            # Filter to valid read length + depth. From a table the panel is
            # already the set of samples that passed, with its richness and
            # depth alongside.
            if from_table:
                _sub = _rbd[(_rbd["spec"] == sp_label)
                            & (_rbd["subset"] == subset_label)]
                ok_ids = [str(x) for x in _sub["sample_id"]]
                richness = pd.Series(_sub["richness"].to_numpy(dtype=float),
                                     index=ok_ids)
                log_bp = pd.Series(
                    _sub["log10_total_mapped_bp"].to_numpy(dtype=float),
                    index=ok_ids)
            else:
                ok_ids = [s for s in subset_ids
                          if valid.get(s, False)
                          and data.total_bp.get(s, 0) >= min_bp]

            if len(ok_ids) < 50:
                ax.text(0.5, 0.5, "Insufficient data",
                        transform=ax.transAxes, ha="center")
                ax.text(-0.10, 1.035, panels[panel_idx],
                        transform=ax.transAxes, fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
                panel_idx += 1
                continue

            if not from_table:
                # Compute richness for all ok_ids
                presence = build_presence(data.breadth.loc[ok_ids],
                                          data.depth.loc[ok_ids],
                                          data.lengths, spec=sp)
                richness = presence.sum(axis=1)
                richness = richness[~richness.index.duplicated(keep="first")]
                log_bp = pd.Series(
                    np.log10(data.total_bp.reindex(ok_ids).values.astype(float)),
                    index=ok_ids)

            # The six binned curves are richness against depth within each
            # tercile, so both columns have to travel with the read lengths;
            # the summary table alone describes only the last panel.
            _rl_rich_rows.append(pd.DataFrame({
                "sample_id": [str(x) for x in ok_ids],
                "spec": sp_label,
                "subset": subset_label,
                "richness": richness.reindex(ok_ids).values.astype(float),
                "log10_total_mapped_bp": log_bp.reindex(ok_ids).values,
            }))

            # Common bin edges
            log_d_all = log_bp.reindex(ok_ids).values
            log_d_valid = log_d_all[np.isfinite(log_d_all)]
            edges = np.percentile(log_d_valid, np.linspace(0, 100, 19))
            edges[0] -= 0.01
            edges[-1] += 0.01

            # Plot each tercile
            for t in tercile_order:
                t_ids = [s for s in ok_ids if rl_group.get(s) == t]
                if len(t_ids) < 10:
                    continue
                uids = list(dict.fromkeys(t_ids))
                ld = log_bp.reindex(uids).values
                ri = richness.reindex(uids).values
                ok_mask = np.isfinite(ld) & np.isfinite(ri)
                bdf = _bin_richness(ld[ok_mask], ri[ok_mask], edges)
                if bdf.empty:
                    continue
                ax.plot(bdf["bin_mid"], bdf["median"], "o-",
                        color=tercile_colors[t], ms=SYMBOL_MIN_PT, lw=1.5,
                        label=f"{t} (n={len(t_ids):,})", zorder=3)
                ax.fill_between(bdf["bin_mid"], bdf["q25"], bdf["q75"],
                                color=tercile_colors[t], alpha=0.12, zorder=2)

                # Interpolate at 3 Gb for summary
                if len(bdf) >= 3:
                    from scipy.interpolate import interp1d
                    f_interp = interp1d(bdf["bin_mid"].values, bdf["median"].values,
                                        kind="linear", bounds_error=False, fill_value=np.nan)
                    pred_3gb = float(f_interp(np.log10(3e9)))
                    if np.isfinite(pred_3gb):
                        summary_rows.append(dict(
                            spec=sp_label, subset=subset_label,
                            tercile=t, n=len(t_ids),
                            median_rl=mean_rl[rl_group == t].median(),
                            predicted_richness_3Gb=pred_3gb))

            # Reference lines
            for label, ld_ref in [("1 Gb", 9.0), ("3 Gb", np.log10(3e9)), ("10 Gb", 10.0)]:
                ax.axvline(ld_ref, color="gray", ls=":", lw=0.7, alpha=0.5)

            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                if i % 2 == 0:
                    ax.axvspan(lo, hi, color="gray", alpha=0.04, zorder=0)

            ax.set_xlabel("log10(Total mapped bp)")
            ax.set_ylabel("Richness (SGBs)")
            ax.set_title(f"{subset_label} ({sp_label})", fontsize=FS-1)
            ax.legend(fontsize=FS-2, loc="upper left")
            ax.text(-0.10, 1.035, panels[panel_idx],
                    transform=ax.transAxes, fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
            panel_idx += 1

    # Row 2, Col 0: Read length distributions by lifestyle
    ax = fig.add_subplot(gs[2, 0])
    for lf, c in [("H-G", config.OI["vermillion"]), ("Transitional", config.OI["yellow"]), ("Industrial", IND_COLOR)]:
        lf_mask = (meta["Lifestyle"] == lf) & valid
        rl_vals = mean_rl[lf_mask].values
        if len(rl_vals) < 10:
            continue
        bins_rl = np.linspace(50, 300, 50)
        ax.hist(rl_vals, bins=bins_rl, density=True, alpha=0.3, color=c,
                label=(f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} "
                       f"(n={len(rl_vals):,}, med={np.median(rl_vals):.0f}bp)"))
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(rl_vals[np.isfinite(rl_vals)], bw_method="silverman")
        xg = np.linspace(50, 300, 200)
        ax.plot(xg, kde(xg), color=c, lw=2)
    ax.set_xlabel("Mean read length (bp)")
    ax.set_ylabel("Density")
    ax.set_title("Read length distributions by lifestyle", fontsize=FS-1)
    ax.legend(fontsize=FS-2, loc="upper right")
    ax.text(-0.10, 1.035, panels[panel_idx],
            transform=ax.transAxes, fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
    panel_idx += 1

    # Row 2, Col 1: Richness residual vs read length
    ax = fig.add_subplot(gs[2, 1])
    # Use 10% threshold, all samples
    # This panel is the 10% threshold over all samples, which is one of the
    # slices already published, so a redraw reads it rather than recomputing.
    if from_table:
        _all = _rbd[(_rbd["spec"] == "10%") & (_rbd["subset"] == "All samples")]
        all_ok = [str(x) for x in _all["sample_id"]]
        ld_all = _all["log10_total_mapped_bp"].to_numpy(dtype=float)
        ri_all = _all["richness"].to_numpy(dtype=float)
    else:
        all_ok = [s for s in meta.index
                  if valid.get(s, False) and data.total_bp.get(s, 0) >= min_bp]
    if len(all_ok) > 100:
        if not from_table:
            presence_all = build_presence(data.breadth.loc[all_ok],
                                           data.depth.loc[all_ok],
                                           data.lengths, spec=breadth_spec)
            rich_all = presence_all.sum(axis=1)
            rich_all = rich_all[~rich_all.index.duplicated(keep="first")]
            ld_all = np.log10(
                data.total_bp.reindex(all_ok).values.astype(float))
            ri_all = rich_all.reindex(all_ok).values
        rl_all = mean_rl.reindex(all_ok).values
        ok3 = np.isfinite(ld_all) & np.isfinite(ri_all) & np.isfinite(rl_all)
        if ok3.sum() > 100:
            # Fit simple linear regression of richness on log_depth
            from numpy.polynomial import polynomial as P
            coeffs = P.polyfit(ld_all[ok3], ri_all[ok3], 2)
            predicted = P.polyval(ld_all[ok3], coeffs)
            residuals = ri_all[ok3] - predicted

            rng = np.random.default_rng(42)
            n_show = min(ok3.sum(), 5000)
            idx = rng.choice(ok3.sum(), n_show, replace=False)

            for lf, c in [("Industrial", IND_COLOR), ("H-G", config.OI["vermillion"]),
                           ("Transitional", config.OI["yellow"])]:
                lf_mask_arr = meta.reindex(np.array(all_ok)[ok3]).iloc[idx]["Lifestyle"].values == lf
                ax.scatter(rl_all[ok3][idx][lf_mask_arr],
                           residuals[idx][lf_mask_arr],
                           # alpha was 0.02, which made the panel unreadable.
                           # The industrial and agro-pastoralist columns are
                           # dense enough to saturate at any opacity, but the
                           # hunter-gatherer cloud at ~280 bp is a few hundred
                           # points spread over 150 bp and disappeared
                           # altogether -- and that is the group the
                           # read-length confound is about. At 0.10 all three
                           # read while the dense columns still show their
                           # gradient instead of going solid.
                           c=c, alpha=0.10, s=SYMBOL_MIN_S, edgecolors="none",
                           label=config.LIFESTYLE_DISPLAY.get(lf, lf),
                           rasterized=True)

            # Correlation
            from scipy.stats import spearmanr
            rho, p_rho = spearmanr(rl_all[ok3], residuals)
            corr_rows.append({
                "panel": "read length vs richness residual",
                "detection": "10pct",
                "x": "mean read length (bp)",
                "y": "richness residual (obs - depth-predicted)",
                "n": int(np.sum(ok3)),
                "test": "Spearman rank correlation, two-sided",
                "rho": float(rho),
                "p": float(p_rho),
            })
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.set_xlabel("Mean read length (bp)")
            ax.set_ylabel("Richness residual (obs − depth-predicted)")
            # Printed the way the manuscript prints a correlation, so the
            # caption can quote the panel verbatim. It did not: the panel said
            # "rho=0.017, p=2.5e-02" while the caption carried -0.076 and
            # 4.6e-13 from an older run, and the main text agreed with the
            # panel. Whatever this run computes is what the caption must say.
            if np.isfinite(p_rho) and p_rho < 1e-3:
                _mant, _exp = f"{p_rho:.1e}".split("e")
                _p_txt = f"${_mant}\\times10^{{{int(_exp)}}}$"
            else:
                _p_txt = f"{p_rho:.3f}"
            ax.set_title("Read length vs richness residual (10%)" + chr(10) +
                         f"Spearman ρ = {rho:+.3f}, p = {_p_txt}",
                         fontsize=FS-1)
            # The cloud is drawn at reduced opacity so 6 pt symbols read as
            # density; the key has to opt out of that or its swatches are
            # invisible.
            _lg = ax.legend(fontsize=FS-2, markerscale=1)
            for _h in _lg.legend_handles:
                try:
                    _h.set_alpha(1.0)
                except Exception:
                    pass
    ax.text(-0.10, 1.035, panels[panel_idx],
            transform=ax.transAxes, fontsize=FS+6, fontweight="bold", va="bottom", ha="right")
    panel_idx += 1

    # Row 2, Col 2: Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        # Pivot for display
        # A column-aligned table instead of prose rows: the old form ran ~50
        # monospace characters per line, wider than its own panel, which forced
        # the saved canvas out under bbox_inches="tight".
        table_lines = ["Predicted richness at 3 Gb", ""]
        _abbr = {"All samples": "All", "Industrial": "Ind", "Non-Industrial": "NonInd"}
        for sp_label in ["300kb", "10%"]:
            table_lines.append(f"{sp_label:<7}" + "".join(f"{t[:1]:>6}" for t in tercile_order))
            for subset in ["All samples", "Industrial", "Non-Industrial"]:
                vals = []
                for t in tercile_order:
                    row = sdf[(sdf["spec"] == sp_label) & (sdf["subset"] == subset) & (sdf["tercile"] == t)]
                    vals.append(f"{row.iloc[0]['predicted_richness_3Gb']:.0f}" if len(row) else "-")
                table_lines.append(f"{_abbr[subset]:<7}" + "".join(f"{v:>6}" for v in vals))
            table_lines.append("")
        table_text = chr(10).join(table_lines)
        _tbl = ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
                       fontsize=FS-2, va="top", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))
        # Sized for its own content, not the figure: leave it out of the autofit
        # so scaling the rest of the panel text does not push it past its panel.
        _tbl._no_autoscale = True
    ax.set_title("Summary", fontsize=FS-1)
    ax.text(-0.10, 1.035, panels[min(panel_idx, len(panels) - 1)],
            transform=ax.transAxes, fontsize=FS+6, fontweight="bold", va="bottom", ha="right")

    fig.suptitle(
        f"Read length effect on observed richness [{ref}]" + chr(10) +
        f"Terciles: Short <= {t1:.0f} bp, Medium {t1:.0f}-{t2:.0f} bp, Long > {t2:.0f} bp",
        fontsize=FS+2, fontweight="bold", y=0.99)

    fpath = os.path.join(s1_main, "figure_read_length_diagnostic.png")
    # The figure's data object: the per-tercile summary the panels are drawn
    # from, plus the per-sample read length and richness the terciles are cut
    # from, so the cut points themselves are reproducible rather than assumed.
    from . import figure_source
    _t = {}
    if from_table:
        # A redraw reads the object; it does not rewrite it.
        _t = None
    if _t is not None and summary_rows:
        _t["tercile_summary"] = pd.DataFrame(summary_rows)
    if _t is not None and _rl_rich_rows:
        _t["richness_by_depth"] = pd.concat(_rl_rich_rows, ignore_index=True)
    # Per sample: the mean read length it was binned on and which tercile that
    # put it in, so the tercile boundaries printed in the title are derivable
    # rather than asserted.
    try:
        if _t is None:
            raise _SkipEmit
        _t["samples"] = pd.DataFrame({
            "sample_id": mean_rl.index.astype(str),
            "mean_read_length_bp": mean_rl.values,
            "tercile": rl_group.reindex(mean_rl.index).values,
            # One panel splits the read lengths by lifestyle, so the label
            # travels too. Published in display form; readers canonicalize
            # it back to the internal key.
            "lifestyle": [config.display_text(x) for x in
                          meta["Lifestyle"].reindex(mean_rl.index).astype(str)],
        })
    except _SkipEmit:
        pass
    except Exception as _exc:
        print(f"    [data] read-length per-sample table unavailable: {_exc}")
    if _t is not None and corr_rows:
        _t["stats"] = pd.DataFrame(corr_rows)
    if _t:
        figure_source.emit_for_path(fpath, _t)

    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")

    # Save summary CSV
    if summary_rows:
        csv_path = os.path.join(s1_main, "read_length_richness_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")


def run_section1_figure(data, dm, stats):
    global FS
    """Build both the composite and robustness figures."""
    ref = data.ref_name
    if dm is None or not dm.ind_vs_ni_ind:
        print("    [WARN] No depth-matched pairs — skipping figures")
        return

    print(f"\n  1B″a. Composite alpha diversity figure [{ref}]")
    build_composite_figure(data, dm, stats)

    print(f"\n  1B″b. Robustness figure "
          f"(uncorrected + 2 methods x 2 thresholds) [{ref}]")
    build_robustness_figure(data, dm, stats)

    # The three Non-Industrial split variants are orphans: nothing references
    # them, they are in neither the manuscript nor the Additional Figures, and
    # each carries ~800 text overlaps on a 28 in canvas that overflows to 21 in
    # when saved. Two regeneration runs have now been killed part-way through
    # drawing them. Keep them available, but off unless asked for.
    if not os.environ.get("S1_NI_SPLITS"):
        print(f"\n  1B″c-e. Non-Industrial split variants skipped "
              f"(set S1_NI_SPLITS=1 to build them)")
        print(f"\n  1B″f. Read length diagnostic [{ref}]")
        build_read_length_diagnostic(data, dm, stats)
        apply_oi_style()
        FS = current_fs()
        return

    print(f"\n  1B″c. Robustness: Hadza vs Other NI [{ref}]")
    hadza_mask, hadza_diag = _find_hadza_mask(data.meta)
    print(f"    {hadza_diag}")
    build_robustness_ni_split_figure(
        data, dm, stats,
        split_label="Hadza", other_label="Other NI",
        filename_suffix="hadza",
        focal_mask=hadza_mask)

    print(f"\n  1B″d. Robustness: H-G vs Transitional [{ref}]")
    hg_mask = (data.meta["Lifestyle"] == "H-G")
    print(f"    H-G mask: n={hg_mask.sum()}")
    build_robustness_ni_split_figure(
        data, dm, stats,
        split_label="H-G", other_label="Transitional",
        filename_suffix="hg",
        focal_mask=hg_mask)

    print("\n  1B″e. Robustness: Carter vs Other NI [" + ref + "]")
    carter_mask, carter_diag = _find_carter_mask(data.meta)
    print("    " + carter_diag)
    build_robustness_ni_split_figure(
        data, dm, stats,
        split_label="Carter", other_label="Other NI",
        filename_suffix="carter",
        focal_mask=carter_mask)

    print("\n  1B″f. Read length diagnostic [" + ref + "]")

    build_read_length_diagnostic(data, dm, stats)

    # The robustness figure pins a much larger base size for its own canvas.
    # Hand the module default back here, once every figure in this section is
    # written, rather than inside that builder: tick label sizes are read from
    # the rcParams at draw time, so restoring while the figure is still open
    # redraws its ticks at the wrong size. Without this, later sections would
    # be drawn at 32 pt on a page-width canvas.
    apply_oi_style()
    FS = current_fs()

