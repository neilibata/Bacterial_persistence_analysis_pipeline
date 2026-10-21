"""
depth_bias_ipw.py - Geometric-mean-target depth-bias correction.

Method
------
Target distribution:  f_target(x) = sqrt(f_A(x) * f_B(x))

This is the unique distribution equidistant (in KL-divergence) from both
groups - neither is favoured.  Importance weights reweight each group to
look as if it were drawn from this common target:

  Group A:  w_i = sqrt( f_B(x_i) / f_A(x_i) )   - mean-normalised, clipped
  Group B:  w_j = sqrt( f_A(x_j) / f_B(x_j) )

Direction guarantees (bidirectional):
  Deep NI where Industrial is rare:  f_A << f_B  ->  w_A >> 1  (Ind upweighted)
  Shallow Ind where NI is rare:      f_B << f_A  ->  w_B >> 1  (NI upweighted)

Weights are mean-normalised (mean(w)=1) and hard-clipped at WEIGHT_CLIP=10.
Bootstrap is plug-in: KDEs fixed from full data; only (w,y) pairs resampled.

Outputs (per reference, saved to results/{ref}/section1_diversity/main/)
------------------------------------------------------------------------
  gmw_richness_main.png      2-group Ind vs NI  (2 specs x 3 cols)
  gmw_richness_summary.csv   point estimates + 95 % CI
  richness_pairwise_corrected.png 3-pair (Ind/Trans, Ind/HG, Trans/HG)
  richness_pairwise_summary.csv

  results/{ref}/depth_bias_ipw/
    gmw_threshold_sweep.png  forest plot
    gmw_threshold_sweep.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import (
    style_for_figure,
    apply_oi_style,
    current_fs,
    page_figsize,
    pin_ylabel,
    square_panels,
    add_panel_letters,
    plain_axes,
    square_figsize,
    drawn_margins,
    panel_grid,
    savefig_dual,
    share_axis_labels,
)
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu, norm as _sp_norm

from . import config
from . import figure_source
from .detection import DetectionSpec, build_presence, breadth_pct_specs, flat_bp_specs

# ── Palette ───────────────────────────────────────────────────────────────────
_C_IND   = config.LIFESTYLE_COLORS.get("Industrial",   config.OI["green"])
_C_TRANS = config.LIFESTYLE_COLORS.get("Transitional",  config.OI["yellow"])
_C_HG   = config.LIFESTYLE_COLORS.get("H-G",            config.OI["vermillion"])
_C_NI   = config.LIFESTYLE_COLORS.get("Non-Industrial", config.OI["orange"])

_PALETTE3 = {"Industrial": _C_IND, "Transitional": _C_TRANS, "H-G": _C_HG,
             "Hadza": config.OI["pink"], "Carter": config.OI["pink"]}
_PALETTE2 = {"Industrial": _C_IND, "Non-Industrial": _C_NI}
_PALETTE2_ACCUM = {"Industrial (all)": _C_IND, "Non-Industrial (all)": _C_NI}

WEIGHT_CLIP = 10.0
# Tick positions in log10(bases) space to match _ld() = log10(total_bp)
# Labels are in Gb.  log10(1 Gb in bases) = log10(1e9) = 9.0
_TICK_GB    = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
_TICK_LOG   = [np.log10(v * 1e9) for v in _TICK_GB]   # log10(bases)


# ══════════════════════════════════════════════════════════════════════════════
#  Fast binned KDE  (O(G log G))
# ══════════════════════════════════════════════════════════════════════════════

class _BinnedKDE:
    """Histogram + FFT Gaussian convolution, Silverman bandwidth."""
    N_GRID = 512

    def __init__(self, data):
        data = np.asarray(data, dtype=float)
        n    = len(data)
        bw   = 1.06 * data.std(ddof=1) * n ** (-0.2)
        self._bw = max(bw, 1e-6)
        lo = data.min() - 4 * self._bw
        hi = data.max() + 4 * self._bw
        self._grid = np.linspace(lo, hi, self.N_GRID)
        dx = self._grid[1] - self._grid[0]
        counts, _ = np.histogram(data, bins=self.N_GRID, range=(lo, hi))
        half   = self.N_GRID
        k_x    = np.linspace(-half * dx, half * dx, 2 * half + 1)
        kernel = np.exp(-0.5 * (k_x / self._bw) ** 2)
        kernel /= kernel.sum()
        from numpy.fft import fft, ifft
        nfft   = 2 ** int(np.ceil(np.log2(len(counts) + len(kernel) - 1)))
        conv   = np.real(ifft(fft(counts, nfft) * fft(kernel, nfft)))
        start  = len(kernel) // 2
        density = np.maximum(conv[start: start + self.N_GRID], 0.0)
        density /= max(density.sum() * dx, 1e-300)
        self._density = density

    def __call__(self, x):
        return np.maximum(
            np.interp(np.asarray(x, dtype=float),
                      self._grid, self._density, left=0.0, right=0.0),
            0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Core weighting functions
# ══════════════════════════════════════════════════════════════════════════════

def _geomean_weights(x_own, kde_own, kde_other):
    """
    w = sqrt(f_other(x) / f_own(x)), mean-normalised to 1, clipped at WEIGHT_CLIP.
    """
    f_own   = np.maximum(kde_own(x_own),   1e-10)
    f_other = np.maximum(kde_other(x_own), 1e-10)
    w = np.sqrt(f_other / f_own)
    w = w / w.mean()
    return np.clip(w, 0.0, WEIGHT_CLIP)


def binned_geomean_weights(x_a, x_b, edges, smooth=0.5):
    """GMW weights from a coarse histogram instead of a KDE.

    Same estimator as _geomean_weights -- w = sqrt(f_other/f_own), normalised
    to mean 1, clipped at WEIGHT_CLIP -- with the two densities estimated by
    binning rather than by _BinnedKDE. That is the only difference.

    Why it exists. Applying GMW inside each study stratum is the right way to
    correct depth and stratify by study at the same time, but _BinnedKDE uses
    a Silverman bandwidth and needs a few dozen points per arm to estimate a
    density worth dividing by; study strata are routinely 5-19 per arm. A
    density is hard to estimate there, but the *ratio* of two densities over
    three or four bins is not, and the ratio is all the weights use.

    `smooth` is a pseudo-count added to every bin of both arms, so a bin one
    arm happens to miss gives a large but finite weight rather than dividing
    by zero and dropping the other arm's samples. Combined with WEIGHT_CLIP
    it bounds the influence any one sample can have.

    Returns (w_a, w_b), each mean-normalised to 1.
    """
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    edges = np.asarray(edges, dtype=float)
    nb = len(edges) - 1
    if nb < 2 or not len(x_a) or not len(x_b):
        return np.ones(len(x_a)), np.ones(len(x_b))

    ia = np.clip(np.digitize(x_a, edges[1:-1]), 0, nb - 1)
    ib = np.clip(np.digitize(x_b, edges[1:-1]), 0, nb - 1)
    ca = np.bincount(ia, minlength=nb).astype(float) + smooth
    cb = np.bincount(ib, minlength=nb).astype(float) + smooth
    pa, pb = ca / ca.sum(), cb / cb.sum()

    wa = np.sqrt(pb[ia] / pa[ia])
    wb = np.sqrt(pa[ib] / pb[ib])
    wa = np.clip(wa / wa.mean(), 0.0, WEIGHT_CLIP)
    wb = np.clip(wb / wb.mean(), 0.0, WEIGHT_CLIP)
    return wa, wb


def depth_bin_edges(all_log_depth, n_bins):
    """Quantile bin edges on the whole cohort's log depth.

    Defined once on every sample rather than per stratum, so a given depth
    falls in the same bin whichever stratum it is being weighted in, and the
    bins do not move with the arm sizes.
    """
    v = np.asarray(all_log_depth, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.array([])
    q = np.linspace(0.0, 1.0, int(n_bins) + 1)
    e = np.unique(np.quantile(v, q))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def _geomean_target_grid(depth_grid, kde_a, kde_b):
    """sqrt(f_A(x)*f_B(x)) on a dense grid - for visualisation."""
    fa = np.maximum(np.asarray(kde_a(depth_grid), dtype=float), 0.0)
    fb = np.maximum(np.asarray(kde_b(depth_grid), dtype=float), 0.0)
    return np.sqrt(fa * fb)


# kept for backward compatibility with section1_figure.py
def _overlap_grid(depth_grid, kde_i, kde_n):
    return _geomean_target_grid(depth_grid, kde_i, kde_n) ** 2


def _ess(weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        return 0.0
    return float(w.sum() ** 2 / (w ** 2).sum())


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical helpers  (ported directly from Carter script)
# ══════════════════════════════════════════════════════════════════════════════

def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _sig_bracket(ax, x1, x2, y_frac, p,
                  dy_bar=0.02, dy_text=0.01, fontsize=FS-2, color="black"):
    """Significance bracket at fractional y-height y_frac."""
    ylo, yhi = ax.get_ylim()
    yspan = yhi - ylo
    yb    = ylo + y_frac * yspan
    dy_b  = dy_bar  * yspan
    dy_t  = dy_text * yspan
    ax.plot([x1, x1, x2, x2], [yb, yb+dy_b, yb+dy_b, yb], lw=1.2, color=color)
    ax.text((x1+x2)/2, yb+dy_b+dy_t, _stars(p),
            ha="center", va="bottom", fontsize=fontsize, color=color)


def _violins(ax, sns, data, x, order, palette, inner="box", alpha=None,
             linewidth=0.8, **kw):
    """Richness violins, one per group in `order`, from seaborn or without it.

    seaborn is optional here, and without it the violins used to be skipped
    outright. Everything placed on these panels afterwards -- the mean labels,
    the brackets -- is in data coordinates, so on an axis left at its default
    0-1 range they landed a million pixels off the canvas and the tight save
    tried to allocate an image that size. Draw them with matplotlib instead.
    """
    if sns is not None:
        if alpha is not None:
            kw["alpha"] = alpha
        sns.violinplot(data=data, x=x, y="richness", order=order, hue=x,
                       hue_order=order, palette=palette, legend=False,
                       inner=inner, cut=0, linewidth=linewidth, ax=ax, **kw)
        return
    vals = [data.loc[data[x] == g, "richness"].to_numpy(dtype=float)
            for g in order]
    pos = [i for i, v in enumerate(vals) if len(v) > 1]
    if pos:
        # matplotlib evaluates the density over the data range only, which is
        # seaborn's cut=0.
        parts = ax.violinplot([vals[i] for i in pos], positions=pos,
                              widths=0.8, showextrema=False)
        for i, body in zip(pos, parts["bodies"]):
            body.set_facecolor(palette[order[i]])
            body.set_edgecolor("0.3")
            body.set_linewidth(linewidth)
            body.set_alpha(1.0 if alpha is None else alpha)
        if inner == "box":
            for i in pos:
                q1, med, q3 = np.percentile(vals[i], [25, 50, 75])
                ax.vlines(i, vals[i].min(), vals[i].max(), color="0.25", lw=1)
                ax.vlines(i, q1, q3, color="0.25", lw=4)
                ax.plot(i, med, "o", color="white", ms=2, zorder=3)
    ax.set_xlim(-0.5, len(order) - 0.5)


def _boot_pval(boot_diffs):
    n = len(boot_diffs)
    if n == 0: return np.nan
    p_pos = (boot_diffs > 0).mean()
    p_neg = (boot_diffs < 0).mean()
    return float(np.clip(2 * min(p_pos, p_neg), 1.0/n, 1.0))


def _raw_vs_corr_pval(raw_vals, corr_boot_means):
    raw_mean = float(np.mean(raw_vals))
    boot_se  = float(np.std(corr_boot_means))
    if boot_se == 0: return 1.0
    z = (np.mean(corr_boot_means) - raw_mean) / boot_se
    return float(2 * _sp_norm.sf(abs(z)))


def _weighted_sem(vals, weights, mu):
    """SE of the weighted mean sum(w y)/sum(w), with w treated as fixed.

    Var(mu) = sum(w^2 Var(y)) / (sum w)^2, so the weights enter the sum
    SQUARED. They were entering to the first power, which is the same number
    only when every weight is equal -- with GMW weights spanning up to the
    clip at 10 it understated the plotted error bars. This form agrees with
    the plug-in bootstrap in _gmw_bootstrap_2grp, which resamples the same
    (w, y) pairs with the weights held fixed; that agreement is checked in
    tools/test_audit_fixes.py.
    """
    w = np.asarray(weights, dtype=float)
    y = np.asarray(vals,    dtype=float)
    return float(np.sqrt((w**2 * (y - mu)**2).sum()) / max(w.sum(), 1e-300))


# ══════════════════════════════════════════════════════════════════════════════
#  Core 2-group estimator + plug-in bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def _gmw_estimate_2grp(ld_i, ld_n, ri, rn):
    """
    Fit KDEs, compute geom-mean-target weights, return weighted means.
    Returns: mu_ind, mu_ni, kde_i, kde_n, w_ind, w_ni
    """
    kde_i = _BinnedKDE(ld_i)
    kde_n = _BinnedKDE(ld_n)
    w_ind = _geomean_weights(ld_i, kde_i, kde_n)
    w_ni  = _geomean_weights(ld_n, kde_n, kde_i)
    mu_ind = float(np.dot(w_ind, ri) / max(w_ind.sum(), 1e-300))
    mu_ni  = float(np.dot(w_ni,  rn) / max(w_ni.sum(),  1e-300))
    return mu_ind, mu_ni, kde_i, kde_n, w_ind, w_ni


# backward-compat alias used by section1_figure.py
def _overlap_weighted_estimate(ld_i, ld_n, ri, rn):
    return _gmw_estimate_2grp(ld_i, ld_n, ri, rn)


def _gmw_bootstrap_2grp(ld_i, ld_n, ri, rn, w_ind, w_ni, n_boot, rng):
    """
    Plug-in bootstrap: KDEs fixed; only (w,y) pairs resampled within each group.
    Returns ndarray (n_boot, 3): [mu_ind, mu_ni, att].
    """
    n_i, n_n = len(ri), len(rn)
    boot = np.empty((n_boot, 3)); boot[:] = np.nan
    for b in range(n_boot):
        ii = rng.choice(n_i, n_i, replace=True)
        jj = rng.choice(n_n, n_n, replace=True)
        w_ib = w_ind[ii]; w_nb = w_ni[jj]
        d_i = w_ib.sum(); d_n = w_nb.sum()
        if d_i == 0 or d_n == 0: continue
        m1 = float((w_ib * ri[ii]).sum() / d_i)
        m0 = float((w_nb * rn[jj]).sum() / d_n)
        boot[b] = [m1, m0, m1-m0]
    return boot


# backward-compat alias used by section1_figure.py
def _bootstrap_overlap_weighted(ld_i, ld_n, ri, rn, kde_i, kde_n, n_boot, rng):
    w_ind = _geomean_weights(ld_i, kde_i, kde_n)
    w_ni  = _geomean_weights(ld_n, kde_n, kde_i)
    return _gmw_bootstrap_2grp(ld_i, ld_n, ri, rn, w_ind, w_ni, n_boot, rng)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-spec 2-group analysis
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_spec(log_depth, is_ind, richness, spec, n_boot, rng, stats):
    spec_tag = spec.short_label
    n_ind = int(is_ind.sum()); n_ni = int((~is_ind).sum())

    ld_i = log_depth[is_ind]; ld_n = log_depth[~is_ind]
    ri   = richness[is_ind];  rn   = richness[~is_ind]

    mu1, mu0, kde_i, kde_n, w_ind, w_ni = _gmw_estimate_2grp(ld_i, ld_n, ri, rn)
    att = mu1 - mu0

    ess_ind = _ess(w_ind); ess_ni = _ess(w_ni)
    print(f"      [{spec_tag}] Industrial:     n={n_ind}  ESS={ess_ind:.0f}  "
          f"({100*ess_ind/n_ind:.0f}%)")
    print(f"      [{spec_tag}] Non-Industrial: n={n_ni}  ESS={ess_ni:.0f}  "
          f"({100*ess_ni/n_ni:.0f}%)")

    print(f"      [{spec_tag}] Bootstrapping ({n_boot} iters)...")
    boot  = _gmw_bootstrap_2grp(ld_i, ld_n, ri, rn, w_ind, w_ni, n_boot, rng)
    valid = np.isfinite(boot[:, 2])
    ci_mu1 = np.percentile(boot[valid,0],[2.5,97.5]) if valid.any() else [np.nan]*2
    ci_mu0 = np.percentile(boot[valid,1],[2.5,97.5]) if valid.any() else [np.nan]*2
    ci_att = np.percentile(boot[valid,2],[2.5,97.5]) if valid.any() else [np.nan]*2

    mu1_raw = float(ri.mean()); mu0_raw = float(rn.mean())
    raw_pct  = 100 * mu1_raw / max(mu0_raw, 1e-10)
    corr_pct = 100 * mu1     / max(mu0,     1e-10)

    _, p_raw = mannwhitneyu(ri, rn, alternative="two-sided")
    p_corr   = _boot_pval(boot[valid,2]) if valid.any() else np.nan
    p_rvc_i  = _raw_vs_corr_pval(ri, boot[valid,0]) if valid.any() else np.nan
    p_rvc_n  = _raw_vs_corr_pval(rn, boot[valid,1]) if valid.any() else np.nan
    wsem_ind = _weighted_sem(ri, w_ind, mu1)
    wsem_ni  = _weighted_sem(rn, w_ni,  mu0)

    depth_grid = np.linspace(log_depth.min(), log_depth.max(), 300)
    geom_grid  = _geomean_target_grid(depth_grid, kde_i, kde_n)

    stats.add(f"gmw_att_{spec_tag}", "s1", "main",
        f"Geom-mean-target depth correction: Ind vs NI ({spec.description})",
        f"D={att:+.1f} SGBs [95%CI:{ci_att[0]:+.1f},{ci_att[1]:+.1f}] "
        f"raw={raw_pct:.1f}% corr={corr_pct:.1f}%",
        test="Geom-mean-target + plug-in bootstrap", n1=n_ind, n2=n_ni,
        extra={"att":att,"ci_lo":ci_att[0],"ci_hi":ci_att[1],
               "mu1_raw":mu1_raw,"mu0_raw":mu0_raw,"mu1_adj":mu1,"mu0_adj":mu0,
               "raw_pct":raw_pct,"corr_pct":corr_pct,
               "ess_ind":ess_ind,"ess_ni":ess_ni,
               "p_raw":p_raw,"p_corr":p_corr})
    stats.add(f"gmw_raw_mw_{spec_tag}", "s1", "main",
        f"Raw richness MW: Ind vs NI ({spec.description})",
        f"Ind={mu1_raw:.1f}+/-{ri.std()/np.sqrt(len(ri)):.1f}  "
        f"NI={mu0_raw:.1f}+/-{rn.std()/np.sqrt(len(rn)):.1f}",
        test="Mann-Whitney U", p_value=p_raw, n1=n_ind, n2=n_ni)

    return dict(
        spec=spec, mu1=mu1, mu0=mu0, att=att,
        ci_mu1=ci_mu1, ci_mu0=ci_mu0, ci_att=ci_att,
        mu1_raw=mu1_raw, mu0_raw=mu0_raw,
        raw_pct=raw_pct, corr_pct=corr_pct,
        wsem_ind=wsem_ind, wsem_ni=wsem_ni,
        ess_ind=ess_ind, ess_ni=ess_ni,
        p_raw=p_raw, p_corr=p_corr, p_rvc_i=p_rvc_i, p_rvc_n=p_rvc_n,
        richness=richness, is_ind=is_ind, log_depth=log_depth,
        ri=ri, rn=rn, ld_i=ld_i, ld_n=ld_n,
        w_ind=w_ind, w_ni=w_ni, kde_i=kde_i, kde_n=kde_n,
        depth_grid=depth_grid, geom_grid=geom_grid,
        ow_grid=geom_grid**2,   # legacy key for section1_figure.py
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1G: 2-group Industrial vs Non-Industrial  (matches gmw_richness.pdf)
# ══════════════════════════════════════════════════════════════════════════════

class _SpecLabel:
    """Stands in for a DetectionSpec when redrawing from a table.

    The table stores the spec's description, which is all the drawing code
    reads off it; carrying the real object would mean the data file could
    only be read by code that can import the detection module.
    """

    def __init__(self, description, short_label=""):
        self.description = description
        self.short_label = short_label or description


def _gmw_main_table(results_list):
    """Source rows behind the GMW richness figure: one per sample.

    The corrected means, the bootstrap intervals, the effective sample sizes
    and the significance brackets are all functions of these four columns,
    so the figure can be rebuilt from this table and nothing else.
    """
    blocks = []
    for r in results_list:
        for grp, rk, ldk, wk in (("Industrial", "ri", "ld_i", "w_ind"),
                                 ("Non-Industrial", "rn", "ld_n", "w_ni")):
            if rk in r and ldk in r:
                block = {
                    "detection": r.get("spec").description
                                 if r.get("spec") is not None else "",
                    "group": grp,
                    "richness": np.asarray(r[rk], dtype=float),
                    "log10_total_mapped_bp": np.asarray(r[ldk], dtype=float),
                }
                # The weight each sample carries in the corrected mean. It is
                # a deterministic function of the depths, so the figure can be
                # redrawn without it -- but a reader cannot then check the
                # correction itself, only re-run it, and the submitted data
                # should let the reweighting be inspected directly.
                w = r.get(wk)
                if w is not None and len(w) == len(r[rk]):
                    block["gmw_weight"] = np.asarray(w, dtype=float)
                blocks.append(pd.DataFrame(block))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _gmw_main_stats(results_list):
    """The tests this figure draws, as a table.

    `_gmw_main_table` carries the values the brackets are computed *from*, and
    `plot_gmw_main` recomputes them on the replot path, so the figure is
    reproducible without this. The journal asks for the P values themselves in
    the submitted source data, though, and a reader recomputing a bootstrap
    cannot confirm they got the same answer as the asterisk already drawn. So
    publish them rather than leave them to be re-derived.
    """
    rows = []
    for r in results_list or []:
        if not r or "ri" not in r:
            continue
        spec = r.get("spec")
        rows.append({
            "detection": spec.description if spec is not None else "",
            "comparison": "Industrial vs Non-Industrial",
            "n_industrial": int(len(r["ri"])),
            "n_non_industrial": int(len(r["rn"])),
            "ess_industrial": float(r["ess_ind"]),
            "ess_non_industrial": float(r["ess_ni"]),
            "mean_richness_raw_industrial": float(r["mu1_raw"]),
            "mean_richness_raw_non_industrial": float(r["mu0_raw"]),
            "mean_richness_gmw_industrial": float(r["mu1"]),
            "mean_richness_gmw_non_industrial": float(r["mu0"]),
            "gmw_difference": float(r["att"]),
            "gmw_difference_ci_lo": float(r["ci_att"][0]),
            "gmw_difference_ci_hi": float(r["ci_att"][1]),
            "test_raw": "Mann-Whitney U, two-sided",
            "p_raw": float(r["p_raw"]),
            "test_gmw": "plug-in bootstrap, two-sided",
            "p_gmw": float(r["p_corr"]),
            # A different test from either of the above: a normal-approximation
            # z-test on the shift from the raw to the reweighted mean, which is
            # what the two grey sub-brackets draw.
            "test_raw_vs_gmw": "bootstrap z-test on the raw-to-GMW shift, "
                               "two-sided",
            "p_raw_vs_gmw_industrial": float(r["p_rvc_i"]),
            "p_raw_vs_gmw_non_industrial": float(r["p_rvc_n"]),
        })
    return pd.DataFrame(rows)


def _gmw_spec_from_samples(sub, n_boot, rng):
    """Re-derive one row's worth of GMW results from the per-sample table.

    Mirrors _run_one_spec without the stats side effects. The bootstrap is
    redrawn here, so intervals move within Monte-Carlo error of the pipeline's
    own unless the caller passes the same seed.
    """
    ld = sub["log10_total_mapped_bp"].to_numpy(dtype=float)
    rich = sub["richness"].to_numpy(dtype=float)
    is_ind = (sub["group"].astype(str) == "Industrial").to_numpy()

    ld_i, ld_n = ld[is_ind], ld[~is_ind]
    ri, rn = rich[is_ind], rich[~is_ind]

    mu1, mu0, kde_i, kde_n, w_ind, w_ni = _gmw_estimate_2grp(ld_i, ld_n, ri, rn)
    boot = _gmw_bootstrap_2grp(ld_i, ld_n, ri, rn, w_ind, w_ni, n_boot, rng)
    valid = np.isfinite(boot[:, 2])
    pct = lambda col: (np.percentile(boot[valid, col], [2.5, 97.5])
                       if valid.any() else [np.nan, np.nan])

    _, p_raw = mannwhitneyu(ri, rn, alternative="two-sided")
    depth_grid = np.linspace(ld.min(), ld.max(), 300)

    return dict(
        spec=_SpecLabel(str(sub["detection"].iloc[0])),
        mu1=mu1, mu0=mu0, att=mu1 - mu0,
        ci_mu1=pct(0), ci_mu0=pct(1), ci_att=pct(2),
        mu1_raw=float(ri.mean()), mu0_raw=float(rn.mean()),
        wsem_ind=_weighted_sem(ri, w_ind, mu1),
        wsem_ni=_weighted_sem(rn, w_ni, mu0),
        ess_ind=_ess(w_ind), ess_ni=_ess(w_ni),
        p_raw=p_raw,
        p_corr=_boot_pval(boot[valid, 2]) if valid.any() else np.nan,
        p_rvc_i=_raw_vs_corr_pval(ri, boot[valid, 0]) if valid.any() else np.nan,
        p_rvc_n=_raw_vs_corr_pval(rn, boot[valid, 1]) if valid.any() else np.nan,
        richness=rich, is_ind=is_ind, log_depth=ld,
        ri=ri, rn=rn, ld_i=ld_i, ld_n=ld_n,
        w_ind=w_ind, w_ni=w_ni, kde_i=kde_i, kde_n=kde_n,
        depth_grid=depth_grid,
        geom_grid=_geomean_target_grid(depth_grid, kde_i, kde_n),
    )


#: Fields of a result dict that are bootstrap outputs rather than inputs, and
#: so cannot be reproduced on a redraw unless the seed and replicate count
#: match the run's. `_adopt_published_p` restores them from the data object.
_BOOT_P_FIELDS = ("p_raw", "p_corr", "p_rvc_i", "p_rvc_n",
                  "p_rvc_a", "p_rvc_b")


def _adopt_published_p(results, stats_df, key_of):
    """Take the drawn p-values from the data object, not from a fresh bootstrap.

    A redraw re-derives everything from the published samples, but the
    bootstrap p-values depend on the seed and the replicate count, and the
    redraw entry points default to seed 0 and OW_AIPWE_N_BOOT_BASE while the
    run uses seed 42/43 and an adaptive count up to OW_AIPWE_N_BOOT_CAP. Left
    alone, a replot would draw a different number of asterisks from the
    p-values published for the same panel in Data S1 -- the defect
    the 2026-10-07 audit recorded for S27. Where the data object carries the
    run's `stats` panel, its p-values win; otherwise the recomputation stands.
    """
    if stats_df is None or getattr(stats_df, "empty", True):
        return results
    try:
        by_key = {key_of(row): row for row in stats_df.to_dict("records")}
    except Exception:
        return results
    for r in results or []:
        if not r:
            continue
        row = by_key.get(_result_key(r))
        if row is None:
            continue
        for src, dst in (("p_raw", "p_raw"), ("p_gmw", "p_corr"),
                         ("p_raw_vs_gmw_industrial", "p_rvc_i"),
                         ("p_raw_vs_gmw_non_industrial", "p_rvc_n"),
                         ("p_raw_vs_gmw_a", "p_rvc_a"),
                         ("p_raw_vs_gmw_b", "p_rvc_b")):
            if src in row and dst in r:
                try:
                    r[dst] = float(row[src])
                except (TypeError, ValueError):
                    pass
    return results


def _result_key(r):
    """The key a result dict is matched on; set by the caller before lookup."""
    return r.get("_pub_key")


def plot_gmw_main(tables, out_path, rng=None, n_boot=None):
    """Draw the GMW richness figure from its table, and from nothing else."""
    if rng is None:
        rng = np.random.default_rng(0)
    if n_boot is None:
        n_boot = config.OW_AIPWE_N_BOOT_BASE
    df = tables["samples"]
    results_list = []
    for det, sub in df.groupby("detection", sort=False):
        r = _gmw_spec_from_samples(sub, n_boot, rng)
        if r:
            r["_pub_key"] = str(det)
        results_list.append(r)
    _adopt_published_p(results_list, tables.get("stats"),
                       lambda row: str(row.get("detection", "")))
    return _draw_main_figure(results_list, out_path, rng)


def _make_main_figure(results_list, out_path, rng):
    """Emit the figure's data object, then draw the figure."""
    table = _gmw_main_table(results_list)
    if not table.empty:
        tables = {"samples": table}
        st = _gmw_main_stats(results_list)
        if not st.empty:
            tables["stats"] = st
        figure_source.emit_for_path(out_path, tables)
    return _draw_main_figure(results_list, out_path, rng)


def _draw_main_figure(results_list, out_path, rng):
    """
    N_specs rows x 3 cols:
      Col 0  depth KDEs + twinx geometric-mean target curve
      Col 1  raw richness violins  (mean+/-SE, MW sig bracket)
      Col 2  depth-corrected violins (weighted resample) + 3 sig brackets
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    n_rows = len(results_list)
    # Square panels. Three columns of tall narrow panels gave each title a
    # column a third of the page wide to sit in, so two-line titles ran into
    # the panel titles beside and below them; at 1:1 they fit.
    # The gutters go into the sizing as well as the GridSpec: sized with the
    # shared defaults, the canvas came out taller than three square panels
    # need, and the surplus sat as blank bands between the rows.
    _MG = dict(top=0.86, hspace=0.42, wspace=0.28)
    fig = plt.figure(figsize=square_figsize(n_cols=3, n_rows=n_rows,
                                            margins=drawn_margins(_MG)))
    # 3 x n_rows panels: the type scale follows the panel count rather than a
    # fixed base, so a two-row figure is not held to a nine-panel size.
    FS = style_for_figure(n_panels=3 * n_rows)
    fig.suptitle("SGB richness: raw vs depth-corrected",
                 fontsize=FS+1, fontweight="bold")
    fig._margin_overrides = _MG
    # Equal columns: with a square plotting box, an unequal width ratio buys
    # a bigger square rather than a wider panel, and one oversized panel in a
    # grid reads as a mistake. The density column gave up 15% here.
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=_MG["hspace"],
                           wspace=_MG["wspace"], width_ratios=[1.0, 1.0, 1.0])
    order2 = ["Industrial", "Non-Industrial"]

    for row, r in enumerate(results_list):
        label    = r["spec"].description
        ld_i, ld_n = r["ld_i"], r["ld_n"]
        ri,    rn  = r["ri"],   r["rn"]

        # ── Col 0: depth KDEs + twinx geom-mean target ───────────────────
        ax_d  = fig.add_subplot(gs[row, 0])
        ax_ow = ax_d.twinx()
        ax_ow.tick_params(labelbottom=False, bottom=False)
        for grp, vals, ls in [("Industrial",    ld_i, "-"),
                               ("Non-Industrial", ld_n, "--")]:
            if len(vals) > 1:
                pd.Series(vals).plot.kde(ax=ax_d, color=_PALETTE2[grp],
                                         linestyle=ls, linewidth=1.5, label=grp)
        geom      = r["geom_grid"]
        dg        = r["depth_grid"]
        geom_disp = geom / max(geom.max(), 1e-300) * 0.25
        ax_ow.plot(dg, geom_disp, color="black", linewidth=2.5,
                   label=r"$\sqrt{\hat{f}_{Ind}{\cdot}\hat{f}_{NI}}$ (target)")
        ax_ow.fill_between(dg, geom_disp, alpha=0.10, color="black")
        ax_ow.set_ylim(0, 0.30)
        # Normalised curve: the numbers are meaningless and the ticks collide
        # with the panel to the right once margins are tight.
        ax_ow.set_yticks([])
        ax_ow.set_ylabel("")
        lo_x = min(ld_i.min(), ld_n.min()) - 0.1
        hi_x = max(ld_i.max(), ld_n.max()) + 0.1
        ticks_in = [t for t in _TICK_LOG if lo_x <= t <= hi_x]
        if len(ticks_in) < 3:
            ticks_in = [t for t in _TICK_LOG
                        if (lo_x - 0.6) <= t <= (hi_x + 0.6)]
        ax_d.set_xticks(ticks_in)
        ax_d.set_xticklabels([str(_TICK_GB[_TICK_LOG.index(t)]) for t in ticks_in],
                              rotation=30, fontsize=FS-2)
        ax_d.set_xlim(lo_x, hi_x)
        ax_d.set_xlabel("Sequencing depth (Gb)")
        ax_d.set_ylabel("Density", fontsize=FS-2)
        ax_d.set_yticks([])
        ax_d.set_title(label, fontsize=FS-1, fontweight="bold")
        lines1, labs1 = ax_d.get_legend_handles_labels()
        lines2, labs2 = ax_ow.get_legend_handles_labels()
        ax_d.legend(lines1 + lines2, labs1 + labs2, fontsize=FS-2,
                    framealpha=0.9)

        # ── Col 1: raw richness violins ───────────────────────────────────
        ax_v = fig.add_subplot(gs[row, 1])
        raw_df = pd.DataFrame({
            "Group":    np.where(r["is_ind"], "Industrial", "Non-Industrial"),
            "richness": r["richness"].astype(float),
        })
        _violins(ax_v, sns, raw_df, "Group", order2, _PALETTE2)
        for i, grp in enumerate(order2):
            v = raw_df.loc[raw_df["Group"]==grp, "richness"]
            ax_v.text(i, v.median(), f"  {v.mean():.0f}\u00b1{v.sem():.0f}",
                      va="center", fontsize=FS-2, fontweight="bold")
        ax_v.set_title("Raw richness", fontsize=FS-1)
        ax_v.set_xlabel("")
        pin_ylabel(ax_v, "SGBs per sample")
        n_i_lbl = len(raw_df[raw_df["Group"]=="Industrial"])
        n_n_lbl = len(raw_df[raw_df["Group"]=="Non-Industrial"])
        ax_v.set_xticks(range(2))
        ax_v.set_xticklabels([f"Industrial\n(n={n_i_lbl:,})",
                               f"Non-Industrial\n(n={n_n_lbl:,})"])
        _sig_bracket(ax_v, 0, 1, 0.88, r["p_raw"])

        # ── Col 2: depth-corrected violins (weighted resample) ────────────
        ax_c = fig.add_subplot(gs[row, 2])
        corr_subs = []
        for grp, vals, w_g in [("Industrial",    ri, r["w_ind"]),
                                ("Non-Industrial", rn, r["w_ni"])]:
            w_norm = w_g / w_g.sum()
            idx_rs = rng.choice(len(vals), size=len(vals), replace=True, p=w_norm)
            corr_subs.append(pd.DataFrame({"Group": grp, "richness": vals[idx_rs]}))
        corr_df = pd.concat(corr_subs, ignore_index=True)
        _violins(ax_c, sns, corr_df, "Group", order2, _PALETTE2, alpha=0.85)
        ci_los = [abs(r["mu1"]-r["ci_mu1"][0]), abs(r["mu0"]-r["ci_mu0"][0])]
        ci_his = [abs(r["ci_mu1"][1]-r["mu1"]), abs(r["ci_mu0"][1]-r["mu0"])]
        for i, (grp, mu, clo, chi, wsem) in enumerate(zip(
                order2, [r["mu1"],r["mu0"]], ci_los, ci_his,
                [r["wsem_ind"],r["wsem_ni"]])):
            ax_c.errorbar(i, mu, yerr=[[clo],[chi]], fmt="D", color="white",
                          ecolor="black", elinewidth=1.5, capsize=4,
                          markersize=5, zorder=5)
            ax_c.text(i, mu, f"  {mu:.0f}\u00b1{wsem:.0f}",
                      va="center", fontsize=FS-2, fontweight="bold")
        ax_c.set_title("Depth-corrected richness", fontsize=FS-1)
        # The y-axis is shared with the raw panel beside it; repeating the label
        # was costing a column of width for no information.
        ax_c.set_xlabel("")
        # Both violin columns carry the unit: the corrected panel is read on
        # its own as often as it is read against the raw one beside it, and an
        # unlabelled y-axis makes the reader look left to find out what it
        # counts.
        pin_ylabel(ax_c, "SGBs per sample")
        square_panels(ax_d, ax_v, ax_c)
        ax_c.set_xticks(range(2))
        ax_c.set_xticklabels([f"Industrial\n(ESS={r['ess_ind']:.0f})",
                               f"Non-Industrial\n(ESS={r['ess_ni']:.0f})"])
        _sig_bracket(ax_c, 0, 1, 0.88, r["p_corr"])
        _sig_bracket(ax_c, -0.15, 0.15, 0.76, r["p_rvc_i"],
                     dy_bar=0.015, dy_text=0.008, fontsize=FS-2, color="grey")
        _sig_bracket(ax_c, 0.85, 1.15, 0.76, r["p_rvc_n"],
                     dy_bar=0.015, dy_text=0.008, fontsize=FS-2, color="grey")

    savefig_dual(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1I: Pairwise 3-way  (matches richness_3way_corrected.pdf)
# ══════════════════════════════════════════════════════════════════════════════

_PAIRS       = [("Industrial","Transitional"),
                ("Industrial","H-G"),
                ("Transitional","H-G")]
# Panel headings, in the short forms the caption defines: I = Industrial,
# A-P = Agro-pastoralist, H-G = Hunter-gatherer. Spelled out these run wider
# than the panel at 18.4 cm; the internal key "Trans" must never reach a
# reader either way.
_PAIR_LABELS = ["I vs A-P", "I vs H-G", "A-P vs H-G"]


def _run_pair(ls_a, ls_b, meta_sub, log_depth_ser, richness_ser, n_boot, rng):
    """Geometric-mean-target comparison for one lifestyle pair."""
    mask_a = (meta_sub["Lifestyle"] == ls_a).values
    mask_b = (meta_sub["Lifestyle"] == ls_b).values

    return _pair_from_arrays(
        ls_a, ls_b,
        log_depth_ser[mask_a], log_depth_ser[mask_b],
        richness_ser[mask_a], richness_ser[mask_b], n_boot, rng)


def _pair_from_arrays(ls_a, ls_b, ld_a, ld_b, r_a, r_b, n_boot, rng):
    """The pair comparison itself, over depths and richnesses alone.

    Split out from _run_pair so the figure can be rebuilt from its published
    table, which carries these four arrays and no metadata frame.
    """
    ld_a = np.asarray(ld_a, dtype=float); ld_b = np.asarray(ld_b, dtype=float)
    r_a = np.asarray(r_a, dtype=float); r_b = np.asarray(r_b, dtype=float)

    # A reference need not carry every lifestyle — the chimp catalogue has no
    # H-G samples at all. A pair with an empty (or single-sample) side has no
    # depth distribution to re-weight towards, so skip it rather than dividing
    # by a zero-sample bandwidth.
    if len(ld_a) < 2 or len(ld_b) < 2:
        return None

    kde_a = _BinnedKDE(ld_a); kde_b = _BinnedKDE(ld_b)
    w_a   = _geomean_weights(ld_a, kde_a, kde_b)
    w_b   = _geomean_weights(ld_b, kde_b, kde_a)

    mu_a = float(np.dot(w_a, r_a) / max(w_a.sum(), 1e-300))
    mu_b = float(np.dot(w_b, r_b) / max(w_b.sum(), 1e-300))

    boot  = _gmw_bootstrap_2grp(ld_a, ld_b, r_a, r_b, w_a, w_b, n_boot, rng)
    valid = np.isfinite(boot[:, 2])
    ci_a   = np.percentile(boot[valid,0],[2.5,97.5]) if valid.any() else [np.nan]*2
    ci_b   = np.percentile(boot[valid,1],[2.5,97.5]) if valid.any() else [np.nan]*2
    ci_att = np.percentile(boot[valid,2],[2.5,97.5]) if valid.any() else [np.nan]*2

    _, p_raw  = mannwhitneyu(r_a, r_b, alternative="two-sided")
    p_corr    = _boot_pval(boot[valid,2]) if valid.any() else np.nan
    p_rvc_a   = _raw_vs_corr_pval(r_a, boot[valid,0]) if valid.any() else np.nan
    p_rvc_b   = _raw_vs_corr_pval(r_b, boot[valid,1]) if valid.any() else np.nan

    depth_grid = np.linspace(min(ld_a.min(),ld_b.min())-0.1,
                              max(ld_a.max(),ld_b.max())+0.1, 300)
    geom_grid  = _geomean_target_grid(depth_grid, kde_a, kde_b)

    sem_a = float(r_a.std(ddof=1)/np.sqrt(len(r_a))) if len(r_a)>1 else 0.0
    sem_b = float(r_b.std(ddof=1)/np.sqrt(len(r_b))) if len(r_b)>1 else 0.0

    return dict(
        ls_a=ls_a, ls_b=ls_b,
        ld_a=ld_a, ld_b=ld_b, r_a=r_a, r_b=r_b,
        w_a=w_a, w_b=w_b, kde_a=kde_a, kde_b=kde_b,
        mu_a=mu_a, mu_b=mu_b, att=mu_a-mu_b,
        ci_a=ci_a, ci_b=ci_b, ci_att=ci_att,
        raw_a=float(r_a.mean()), raw_b=float(r_b.mean()),
        raw_sem_a=sem_a, raw_sem_b=sem_b,
        ess_a=_ess(w_a), ess_b=_ess(w_b),
        n_a=len(r_a), n_b=len(r_b),
        p_raw=p_raw, p_corr=p_corr,
        p_rvc_a=p_rvc_a, p_rvc_b=p_rvc_b,
        depth_grid=depth_grid, geom_grid=geom_grid,
    )


def _gmw_pairwise_table(pair_results):
    """Source rows behind the pairwise figure: one per sample, per comparison.

    A sample's weight depends on which pair it is compared within, so it is
    carried per row rather than per sample: the same sample appears in two
    pairs with two different weights.
    """
    blocks = []
    for (mode, pi), r in (pair_results or {}).items():
        if not r:
            continue
        for side in ("a", "b"):
            block = {
                "detection_mode": mode,
                "pair_index": pi,
                "lifestyle": r[f"ls_{side}"],
                "richness": np.asarray(r[f"r_{side}"], dtype=float),
                "log10_total_mapped_bp": np.asarray(r[f"ld_{side}"], dtype=float),
            }
            w = r.get(f"w_{side}")
            if w is not None and len(w) == len(r[f"r_{side}"]):
                block["gmw_weight"] = np.asarray(w, dtype=float)
            blocks.append(pd.DataFrame(block))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def _gmw_pairwise_stats(pair_results):
    """The tests this figure draws, one row per lifestyle pair.

    Same reasoning as `_gmw_main_stats`: the brackets are recomputable from the
    published samples, but the submitted source data has to carry the P values
    it quotes.
    """
    rows = []
    for (mode, pi), r in (pair_results or {}).items():
        if not r:
            continue
        rows.append({
            "detection_mode": mode,
            "pair_index": pi,
            "lifestyle_a": r["ls_a"],
            "lifestyle_b": r["ls_b"],
            "n_a": int(r["n_a"]),
            "n_b": int(r["n_b"]),
            "ess_a": float(r["ess_a"]),
            "ess_b": float(r["ess_b"]),
            "mean_richness_raw_a": float(r["raw_a"]),
            "mean_richness_raw_b": float(r["raw_b"]),
            "mean_richness_gmw_a": float(r["mu_a"]),
            "mean_richness_gmw_b": float(r["mu_b"]),
            "gmw_difference": float(r["att"]),
            "gmw_difference_ci_lo": float(r["ci_att"][0]),
            "gmw_difference_ci_hi": float(r["ci_att"][1]),
            # p_raw is published for completeness; the pairwise figure draws
            # only the three that follow it.
            "test_raw": "Mann-Whitney U, two-sided (not drawn)",
            "p_raw": float(r["p_raw"]),
            "test_gmw": "plug-in bootstrap, two-sided",
            "p_gmw": float(r["p_corr"]),
            "test_raw_vs_gmw": "bootstrap z-test on the raw-to-GMW shift, "
                               "two-sided",
            "p_raw_vs_gmw_a": float(r["p_rvc_a"]),
            "p_raw_vs_gmw_b": float(r["p_rvc_b"]),
        })
    return pd.DataFrame(rows)


def plot_gmw_pairwise(tables, out_path, rng=None, n_boot=None):
    """Draw the pairwise figure from its table, and from nothing else."""
    if rng is None:
        rng = np.random.default_rng(0)
    if n_boot is None:
        n_boot = config.OW_AIPWE_N_BOOT_BASE
    df = tables["samples"]
    pair_results = {}
    for (mode, pi), sub in df.groupby(["detection_mode", "pair_index"],
                                      sort=False):
        # _PAIRS fixes which two lifestyles a pair index refers to, and in
        # which order, so the table needs to carry only the index.
        ls_a, ls_b = _PAIRS[int(pi)]
        a = sub[sub["lifestyle"].astype(str) == ls_a]
        b = sub[sub["lifestyle"].astype(str) == ls_b]
        r = _pair_from_arrays(
            ls_a, ls_b,
            a["log10_total_mapped_bp"], b["log10_total_mapped_bp"],
            a["richness"], b["richness"], n_boot, rng)
        if r:
            r["_pub_key"] = f"{mode}|{int(pi)}"
        pair_results[(mode, int(pi))] = r
    # See _adopt_published_p: the run's p-values win over a fresh bootstrap at
    # a different seed and replicate count.
    _adopt_published_p(list(pair_results.values()), tables.get("stats"),
                       lambda row: f"{row.get('detection_mode', '')}"
                                   f"|{int(row.get('pair_index', -1))}")
    return _draw_pairwise_figure(pair_results, out_path, rng)


def _make_pairwise_figure(pair_results, out_path, rng):
    """Emit the figure's data object, then draw the figure."""
    table = _gmw_pairwise_table(pair_results)
    if not table.empty:
        tables = {"samples": table}
        st = _gmw_pairwise_stats(pair_results)
        if not st.empty:
            tables["stats"] = st
        figure_source.emit_for_path(out_path, tables)
    return _draw_pairwise_figure(pair_results, out_path, rng)


def _draw_pairwise_figure(pair_results, out_path, rng):
    """
    3 cols (pairs) x 3 rows:
      Row 0  depth distributions + geometric-mean target
      Row 1  breadth-pct richness  (raw faded + corrected overlay)
      Row 2  flat-bp richness      (raw faded + corrected overlay)
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    # (detection mode, column). Column 0 is the depth panel.
    MODE_ROWS = [("breadth_pct", 1), ("flat_bp", 2)]

    # Sized for the 3 x 3 of square panels it draws, with the gutters it
    # draws them with; a 2 x 2 page size left the panels height-bound, and the
    # spare width went into the gaps between the columns.
    _MG = dict(top=0.87, hspace=0.42, wspace=0.28)
    fig = plt.figure(figsize=square_figsize(n_cols=3, n_rows=3,
                                            margins=drawn_margins(_MG)))
    FS = style_for_figure(n_panels=9)        # 3 pair columns x 3 rows
    # One line. The three-line version overlapped all three panel titles; the
    # target definition belongs in the caption, not on top of the figure.
    fig._margin_overrides = dict(top=_MG["top"])
    fig.suptitle("SGB richness: pairwise depth-corrected comparisons",
                 fontsize=FS+1, fontweight="bold")
    # One comparison per row, one measurement per column: the depth
    # densities first, then a column per detection threshold.
    # Equal columns, for the reason given on the main figure: square boxes
    # turn a width ratio into a size difference between panels.
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=_MG["hspace"],
                           wspace=_MG["wspace"], width_ratios=[1, 1, 1])

    for ci, (pi, (ls_a, ls_b)) in enumerate(zip(range(3), _PAIRS)):
        r0 = pair_results.get(("breadth_pct", pi))
        if r0 is None:
            continue

        # ── Row 0: depth KDEs + geom-mean target ─────────────────────────
        ax_d  = fig.add_subplot(gs[ci, 0])
        ax_d2 = ax_d.twinx()
        ax_d2.tick_params(labelbottom=False, bottom=False)
        # Square the density column with the violin columns beside it; the
        # twin has to match or it detaches from the panel it overlays.
        square_panels(ax_d, ax_d2)
        for ls, vals, ls_style in [(ls_a, r0["ld_a"], "-"),
                                   (ls_b, r0["ld_b"], "--")]:
            if len(vals) > 1:
                pd.Series(vals).plot.kde(ax=ax_d,
                                         color=_PALETTE3.get(ls, config.OI["pink"]),
                                         linestyle=ls_style, linewidth=1.5,
                                         label=config.LIFESTYLE_DISPLAY.get(ls, ls))
        dg   = r0["depth_grid"]
        geom = r0["geom_grid"]
        geom_disp = geom / max(geom.max(), 1e-300) * 0.25
        ax_d2.plot(dg, geom_disp, color="black", linewidth=2,
                   label=r"$\sqrt{\hat{f}_A{\cdot}\hat{f}_B}$ target")
        ax_d2.fill_between(dg, geom_disp, alpha=0.08, color="black")
        ax_d2.set_ylim(0, 0.30); ax_d2.set_yticks([])
        lo_x = min(r0["ld_a"].min(), r0["ld_b"].min()) - 0.1
        hi_x = max(r0["ld_a"].max(), r0["ld_b"].max()) + 0.1
        ticks_in = [t for t in _TICK_LOG if lo_x <= t <= hi_x]
        if len(ticks_in) < 3:
            ticks_in = [t for t in _TICK_LOG
                        if (lo_x - 0.6) <= t <= (hi_x + 0.6)]
        ax_d.set_xticks(ticks_in)
        ax_d.set_xticklabels([str(_TICK_GB[_TICK_LOG.index(t)]) for t in ticks_in],
                              rotation=30, fontsize=FS-2)
        ax_d.set_xlim(lo_x, hi_x)
        ax_d.set_xlabel("Sequencing depth (Gb)", fontsize=FS-2)
        # The row is a comparison now that the figure is transposed, so the
        # depth column carries its name.
        _yl = ax_d.set_ylabel(f"{_PAIR_LABELS[pi]}\nDensity", fontsize=FS-2)
        _yl._keep_text = True
        ax_d.set_yticks([])
        if ci == 0:
            ax_d.set_title("Sequencing depth", fontsize=FS-1)
        lines1, labs1 = ax_d.get_legend_handles_labels()
        lines2, labs2 = ax_d2.get_legend_handles_labels()
        ax_d.legend(lines1 + lines2, labs1 + labs2, fontsize=FS-2,
                    framealpha=0.9)

        # ── Rows 1-2: richness violins per mode ───────────────────────────
        for mode, vrow in MODE_ROWS:
            r = pair_results.get((mode, pi))
            if r is None:
                continue
            ax_v = fig.add_subplot(gs[ci, vrow])
            order_pair = [ls_a, ls_b]
            pal_pair   = {ls_a: _PALETTE3.get(ls_a, config.OI["pink"]),
                          ls_b: _PALETTE3.get(ls_b, config.OI["pink"])}

            raw_vdf = pd.DataFrame({
                "Lifestyle": np.concatenate([np.full(len(r["r_a"]),ls_a),
                                              np.full(len(r["r_b"]),ls_b)]),
                "richness":  np.concatenate([r["r_a"], r["r_b"]]),
            })
            _violins(ax_v, sns, raw_vdf, "Lifestyle", order_pair, pal_pair,
                     linewidth=0.6, alpha=0.35, density_norm="width")

            corr_subs = []
            for ls, vals, w_g in [(ls_a,r["r_a"],r["w_a"]),
                                  (ls_b,r["r_b"],r["w_b"])]:
                w_norm = w_g / w_g.sum()
                idx_rs = rng.choice(len(vals), size=len(vals),
                                    replace=True, p=w_norm)
                corr_subs.append(pd.DataFrame(
                    {"Lifestyle":ls, "richness":vals[idx_rs]}))
            corr_vdf = pd.concat(corr_subs, ignore_index=True)
            _violins(ax_v, sns, corr_vdf, "Lifestyle", order_pair, pal_pair,
                     inner=None, linewidth=1.2, alpha=0.65,
                     density_norm="width")

            for i, ls in enumerate(order_pair):
                mu       = r["mu_a"] if ls==ls_a else r["mu_b"]
                ci_bounds = r["ci_a"] if ls==ls_a else r["ci_b"]
                raw      = r["raw_a"] if ls==ls_a else r["raw_b"]
                rsem     = r["raw_sem_a"] if ls==ls_a else r["raw_sem_b"]
                ax_v.errorbar(i, mu, yerr=[[abs(mu-ci_bounds[0])],[abs(ci_bounds[1]-mu)]],
                              fmt="D", color="white", ecolor="black",
                              elinewidth=1.5, capsize=4, markersize=5, zorder=5)
                ax_v.text(i - 0.04, raw, f"{raw:.0f}\u00b1{rsem:.0f}  ",
                          ha="right",
                          va="center", fontsize=FS-2, color="gray", style="italic")
                ax_v.text(i, mu, f"  {mu:.0f}",
                          va="center", fontsize=FS-2, fontweight="bold")

            att      = r["mu_a"] - r["mu_b"]
            raw_diff = r["raw_a"] - r["raw_b"]
            mode_lbl = r["spec"].description if r.get("spec") is not None else (
                "\u226510% breadth" if mode == "breadth_pct"
                else f"\u2265{config.ACCUM_MAIN_TEXT_FLAT_BP//1000} kb mapped")
            # One line, and the confidence interval only on the corrected value:
            # the full "Raw x -> Corrected y [lo, hi]" string was wider than the
            # panel and ran into its neighbour.
            # The threshold names the panel; the effect size is a result and
            # lives in richness_pairwise_summary.csv for the caption.
            if ci == 0:
                ax_v.set_title(mode_lbl, fontsize=FS-2)
            # Shared down the column; labelling every panel cost width.
            ax_v.set_xlabel("")
            # Every violin column is labelled, not just the first: each is a
            # different detection threshold and gets read on its own.
            pin_ylabel(ax_v, "SGBs per sample", fontsize=FS-2)
            square_panels(ax_v)
            ax_v.set_xticks(range(2))
            ax_v.set_xticklabels([
                f"{ls}\nn={r['n_a'] if ls==ls_a else r['n_b']}"
                f"  ESS={r['ess_a'] if ls==ls_a else r['ess_b']:.0f}"
                for ls in order_pair], fontsize=FS-2)
            _sig_bracket(ax_v, 0, 1, 0.87, r["p_corr"])
            _sig_bracket(ax_v, -0.15, 0.15, 0.76, r["p_rvc_a"],
                         dy_bar=0.015, dy_text=0.008, fontsize=FS-2, color="grey")
            _sig_bracket(ax_v, 0.85, 1.15, 0.76, r["p_rvc_b"],
                         dy_bar=0.015, dy_text=0.008, fontsize=FS-2, color="grey")

    # Science: capital part labels, upper left of each panel.
    add_panel_letters(fig, [a for a in fig.axes if a.get_visible()])

    savefig_dual(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold sweep forest-plot
# ══════════════════════════════════════════════════════════════════════════════

def _sweep_corrected_handles(values, label_stem="GMW-corrected + 95%CI"):
    """Legend entries matching the colours the sweep actually drew.

    Marker colour encodes the direction of the effect -- _C_IND where Industrial
    is higher, _C_NI where Non-Industrial is -- so a single hard-coded swatch is
    wrong whenever the data runs the other way, which in practice it always
    does. Build the key from the signs present: one entry when the effect has a
    consistent direction, two when it does not, and say which is which.
    """
    from matplotlib.lines import Line2D
    import numpy as _np
    v = _np.asarray([x for x in values if _np.isfinite(x)], dtype=float)
    pos, neg = bool((v > 0).any()), bool((v < 0).any())
    out = []
    if pos and neg:
        out.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_IND,
                          markersize=7, label=f"{label_stem}, Ind higher"))
        out.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_NI,
                          markersize=7, label=f"{label_stem}, NI higher"))
    else:
        col = _C_IND if pos else _C_NI
        out.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                          markersize=7, label=label_stem))
    return out


def _make_sweep_figure(sweep_df, out_path):
    """
    Two-panel threshold robustness figure.

    Left  — Absolute richness difference (Δ SGBs): raw (grey diamond) and
             GMW-corrected (coloured CI bar), same as before.
    Right — Percentage difference (Ind as % of NI richness − 100%):
             raw% vs GMW-corrected%.  This directly compares what depth
             correction changes across thresholds.
    """
    if sweep_df.empty:
        return
    n = len(sweep_df)
    fig, (ax, ax2) = plt.subplots(
        # Height is set by the number of thresholds swept, one row each, so this
        # is deliberately not square -- squaring it would drop rows.
        1, 2, figsize=page_figsize(n_cols=2, height_in=max(2.6, n * 0.26 + 1.3)),
        # The panels share their y axis, so the right one carries no tick
        # labels and the gap between them was holding nothing. Closing it
        # gives both panels the width instead.
        gridspec_kw={"wspace": 0.08},
        sharey=True)
    y_pos = np.arange(n)

    # ── Left: absolute Δ ───────────────────────────────────────────────────
    ax.scatter(sweep_df["raw_diff"].values, y_pos, marker="D",
               color="#888888", s=30, alpha=0.7, zorder=2,
               label="Raw (unweighted)")
    for i in range(n):
        att = sweep_df["att"].iloc[i]
        if np.isnan(att): continue
        color = _C_IND if att > 0 else _C_NI
        ax.plot([sweep_df["ci_lo"].iloc[i], sweep_df["ci_hi"].iloc[i]],
                [y_pos[i], y_pos[i]], color=color, linewidth=2.0, zorder=3)
        ax.scatter([att], [y_pos[i]], color=color, s=55, zorder=4)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    # "\u2265500 kb mapped bases" x 12 is a wide block of repeated words sitting
    # outside the axes, which is what pushed the saved figure past page width.
    # The units alone carry it; the x-label says what the threshold is on.
    _short = (sweep_df["label"].astype(str)
              .str.replace(" mapped bases", "", regex=False)
              .str.replace(" breadth coverage", "%", regex=False)
              .str.replace("\u2265", "", regex=False)
              .str.replace("%%", "%", regex=False))
    ax.set_yticklabels(_short.values, fontsize=FS-1)
    ax.set_xlabel("\u0394 richness  (Ind \u2212 NI, # SGBs)", fontsize=FS)
    ax.set_title("Absolute richness difference",
                 fontsize=FS, fontweight="bold")
    from matplotlib.lines import Line2D as _L2D_abs
    ax.legend(handles=[
        _L2D_abs([0],[0], marker="D", color="w", markerfacecolor="#888888",
                 markersize=6, alpha=0.7, label="Raw (unweighted)"),
    ] + _sweep_corrected_handles(sweep_df["att"].values),
        fontsize=FS-1, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)

    # ── Right: % difference (Ind/NI − 1) × 100 ────────────────────────────
    has_pct = "raw_pct" in sweep_df.columns and "corr_pct" in sweep_df.columns
    has_pct = ("raw_pct" in sweep_df.columns and "corr_pct" in sweep_df.columns)
    if has_pct:
        ax2.scatter(sweep_df["raw_pct"].values, y_pos, marker="D",
                    color="#888888", s=30, alpha=0.7, zorder=2,
                    label="Raw %")
        # GMW-corrected % with CI bars
        has_pct_ci = ("corr_pct_lo" in sweep_df.columns and
                      "corr_pct_hi" in sweep_df.columns)
        for i in range(n):
            cp = sweep_df["corr_pct"].iloc[i]
            if np.isnan(cp):
                continue
            # Match the left panel: colour encodes which group is higher. This
            # was pinned to _C_IND, so the same corrected effect appeared blue
            # here and orange beside it, and neither matched its key.
            color = _C_IND if cp > 0 else _C_NI
            if has_pct_ci:
                ax2.plot([sweep_df["corr_pct_lo"].iloc[i],
                          sweep_df["corr_pct_hi"].iloc[i]],
                         [y_pos[i], y_pos[i]],
                         color=color, linewidth=2.0, zorder=3)
            ax2.scatter([cp], [y_pos[i]], color=color, s=55, zorder=4)
        ax2.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax2.set_xlabel("Ind richness relative to NI (% difference)", fontsize=FS)
        ax2.set_title("Percentage richness difference",
                      fontsize=FS, fontweight="bold")
        from matplotlib.lines import Line2D as _L2D
        ax2.legend(handles=[
            _L2D([0],[0], marker="D", color="w", markerfacecolor="#888888",
                 markersize=6, alpha=0.7, label="Raw %"),
        ] + _sweep_corrected_handles(sweep_df["corr_pct"].values,
                                     "GMW-corrected % + 95%CI"),
            fontsize=FS-1, framealpha=0.9)
        ax2.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        ax2.grid(axis="x", alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "% data not available",
                 transform=ax2.transAxes, ha="center", va="center")

    # Shared formatting
    n_breadth = (sweep_df["mode"]=="breadth_pct").sum()
    for _ax in (ax, ax2):
        _ax.invert_yaxis()
        if 0 < n_breadth < n:
            _ax.axhspan(-0.5, n_breadth-0.5, color="#FFF0F0", alpha=0.4, zorder=0)
            _ax.axhspan(n_breadth-0.5, n-0.5, color="#F0F4FF", alpha=0.4, zorder=0)

    fig._margin_overrides = dict(top=0.86)
    fig._aspect_exempt = True          # one row per threshold; see above
    fig.suptitle("GMW depth-corrected richness — threshold robustness",
                 fontsize=FS+2, fontweight="bold")
    # The sweep frame is exactly what the figure plots, one row per threshold.
    figure_source.emit_for_path(out_path, {"sweep": sweep_df.reset_index(drop=True)})

    savefig_dual(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry points
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Adaptive bootstrap count
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_n_boot(n_i, n_n, base=None, cap=None):
    """
    Scale bootstrap replicates with the smaller group size.

    Rationale
    ---------
    The plug-in bootstrap SE ∝ 1/sqrtn_boot x σ.  For the Carter study
    (~900 Industrial, ~800 NI) 500 reps gives CI half-widths of ~1-2 SGBs -
    acceptable for a pilot.  The full pipeline has ~14k Industrial and ~3.8k NI,
    so σ is similar but we can afford substantially more reps without the
    bottleneck that used to come from KDE re-estimation (which is now fixed).

    Rule: n_boot = clip(base x sqrt(min(n_i, n_n) / 500), base, cap)

    With base=1000, cap=5000:
      n_min =  500  (Carter-scale) ->  1,000  reps
      n_min = 1000                 ->  1,414
      n_min = 2000                 ->  2,000
      n_min = 3800  (full NI)      ->  2,757
      n_min = 5000                 ->  3,162
      n_min >= 7813                 ->  5,000  (cap)

    Parameters
    ----------
    n_i, n_n : int   raw group sizes (not ESS - ESS is unknown before fitting)
    base     : int   minimum reps (default config.OW_AIPWE_N_BOOT_BASE)
    cap      : int   maximum reps (default config.OW_AIPWE_N_BOOT_CAP)
    """
    if base is None:
        base = getattr(config, "OW_AIPWE_N_BOOT_BASE", 1000)
    if cap is None:
        cap  = getattr(config, "OW_AIPWE_N_BOOT_CAP",  5000)
    n_min  = min(n_i, n_n)
    scaled = int(base * np.sqrt(n_min / 500.0))
    return int(np.clip(scaled, base, cap))


def run_gmw_richness(data, stats, specs=None, n_boot=None, rng_seed=42):
    """
    1G. Geometric-mean-target depth-corrected richness: Industrial vs Non-Industrial.
    Produces gmw_richness_main.png + summary CSV.
    """
    ref      = data.ref_name
    out_main = config.section_dir(ref, "s1", "main")
    rng      = np.random.default_rng(rng_seed)
    meta  = data.meta
    valid = meta["MacroGroup"].isin(["Industrial", "Non-Industrial"])
    if not valid.any():
        print("      [WARN] MacroGroup not set - skipping"); return None

    idx       = meta.index[valid]
    is_ind    = (meta.loc[idx, "MacroGroup"] == "Industrial").values
    log_depth = np.log10(np.maximum(data.total_bp.loc[idx].values, 1.0))

    if is_ind.sum() < 10 or (~is_ind).sum() < 10:
        print("      [WARN] Too few samples - skipping"); return None

    # n_boot: use caller-supplied value, or scale adaptively with group sizes
    if n_boot is None:
        n_boot = _adaptive_n_boot(int(is_ind.sum()), int((~is_ind).sum()))
    print(f"\n  1G. Geom-mean-target depth-corrected richness [{ref}]"
          f"  n={is_ind.sum():,}/{(~is_ind).sum():,}  n_boot={n_boot}")

    if specs is None:
        specs = [DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
                 DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)]

    results_list = []; summary_rows = []
    for spec in specs:
        print(f"\n    Spec: {spec.description}")
        presence = build_presence(data.breadth.loc[idx], data.depth.loc[idx],
                                  data.lengths, spec=spec)
        richness = presence.sum(axis=1).values.astype(float)
        r = _run_one_spec(log_depth, is_ind, richness, spec, n_boot, rng, stats)
        results_list.append(r)
        summary_rows.append({
            "spec": spec.description, "mode": spec.mode, "threshold": spec.threshold,
            "n_ind": int(is_ind.sum()), "n_ni": int((~is_ind).sum()),
            "ess_ind": r["ess_ind"], "ess_ni": r["ess_ni"],
            "mu1_raw": r["mu1_raw"], "mu0_raw": r["mu0_raw"],
            "raw_diff": r["mu1_raw"]-r["mu0_raw"],
            "mu1_adj": r["mu1"], "mu0_adj": r["mu0"],
            "att": r["att"], "ci_lo": r["ci_att"][0], "ci_hi": r["ci_att"][1],
            "raw_pct": r["raw_pct"], "corr_pct": r["corr_pct"],
            "p_mw_raw": r["p_raw"], "p_corr": r["p_corr"],
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path   = os.path.join(out_main, "gmw_richness_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n    Saved: {csv_path}")

    fig_path = os.path.join(out_main, "gmw_richness_main.png")
    _make_main_figure(results_list, fig_path, rng)
    print(f"\n    Depth-correction complete [{ref}]")
    return summary_df


def run_geomean_richness_pairwise(data, stats, specs=None, n_boot=None, rng_seed=43,
                                  sample_mask=None, out_dir=None, stem=None):
    """
    1I. Pairwise depth-corrected richness for all 3 lifestyle pairs.
    Produces richness_pairwise_corrected.png + summary CSV.

    `sample_mask` restricts the comparison to a subset of the cohort (a boolean
    Series over `data.meta.index`); `out_dir` and `stem` keep those outputs
    separate from the whole-cohort run.
    """
    ref      = data.ref_name
    out_main = out_dir or config.section_dir(ref, "s1", "main")
    os.makedirs(out_main, exist_ok=True)
    stem     = stem or "richness_pairwise_corrected"
    rng      = np.random.default_rng(rng_seed)
    if specs is None:
        specs = [DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
                 DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP)]

    meta  = data.meta
    valid = meta["Lifestyle"].isin(["Industrial","Transitional","H-G"])
    if sample_mask is not None:
        valid = valid & sample_mask.reindex(meta.index).fillna(False)
    if not valid.any():
        print("      [WARN] No lifestyle-labelled samples - skipping"); return None

    idx      = meta.index[valid]
    meta_sub = meta.loc[idx]
    log_depth_arr = np.log10(np.maximum(data.total_bp.loc[idx].values, 1.0))

    print(f"\n  1I. Pairwise depth-corrected richness [{ref}]")
    print(f"      pairs: {_PAIR_LABELS}")

    pair_results = {}; summary_rows = []

    for spec in specs:
        print(f"\n    Spec: {spec.description}")
        presence = build_presence(data.breadth.loc[idx], data.depth.loc[idx],
                                  data.lengths, spec=spec)
        richness_arr = presence.sum(axis=1).values.astype(float)

        for pi, (ls_a, ls_b) in enumerate(_PAIRS):
            # Adaptive n_boot: scale with smaller group for this pair
            n_a = int((meta_sub["Lifestyle"] == ls_a).sum())
            n_b = int((meta_sub["Lifestyle"] == ls_b).sum())
            n_boot_pair = (n_boot if n_boot is not None
                           else _adaptive_n_boot(n_a, n_b))
            # Not every reference carries every lifestyle (the chimp catalogue
            # has no H-G samples), and a pair with an empty side has no depth
            # distribution to re-weight towards.
            if n_a < 2 or n_b < 2:
                print(f"      {ls_a}(n={n_a:,}) vs {ls_b}(n={n_b:,})"
                      f"  — skipped, needs >=2 samples per group")
                pair_results[(spec.mode, pi)] = None
                continue
            print(f"      {ls_a}(n={n_a:,}) vs {ls_b}(n={n_b:,})"
                  f"  n_boot={n_boot_pair}... ", end="", flush=True)
            r = _run_pair(ls_a, ls_b, meta_sub,
                          log_depth_arr, richness_arr, n_boot_pair, rng)
            pair_results[(spec.mode, pi)] = r
            att = r["mu_a"] - r["mu_b"]
            print(f"D={att:+.1f} [{r['ci_att'][0]:+.1f},{r['ci_att'][1]:+.1f}]  "
                  f"raw MW p={r['p_raw']:.2e}  corr p={r['p_corr']:.2e}")

            stats.add(
                f"gmw_pair_{spec.short_label}_{ls_a}_vs_{ls_b}",
                "s1", "main",
                f"Pairwise corrected: {ls_a} vs {ls_b} ({spec.description})",
                f"\u0394={att:+.1f} [95%CI:{r['ci_att'][0]:+.1f},{r['ci_att'][1]:+.1f}]  "
                f"raw {ls_a}={r['raw_a']:.1f} corr={r['mu_a']:.1f}  "
                f"raw {ls_b}={r['raw_b']:.1f} corr={r['mu_b']:.1f}",
                test="Geom-mean-target + plug-in bootstrap",
                n1=r["n_a"], n2=r["n_b"],
                extra={"att":att,"ci_lo":r["ci_att"][0],"ci_hi":r["ci_att"][1],
                       "p_corr":r["p_corr"],"p_raw":r["p_raw"]})
            summary_rows.append({
                "spec":spec.description,"mode":spec.mode,
                "ls_a":ls_a,"ls_b":ls_b,
                "n_a":r["n_a"],"n_b":r["n_b"],
                "ess_a":r["ess_a"],"ess_b":r["ess_b"],
                "raw_a":r["raw_a"],"raw_b":r["raw_b"],
                "raw_diff":r["raw_a"]-r["raw_b"],
                "mu_a":r["mu_a"],"mu_b":r["mu_b"],
                "att":att,"ci_lo":r["ci_att"][0],"ci_hi":r["ci_att"][1],
                "p_raw":r["p_raw"],"p_corr":r["p_corr"],
            })

    summary_df = pd.DataFrame(summary_rows)
    csv_path   = os.path.join(out_main, "richness_pairwise_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n    Saved: {csv_path}")

    fig_path = os.path.join(out_main, f"{stem}.png")
    _make_pairwise_figure(pair_results, fig_path, rng)
    return summary_df


def run_gmw_threshold_sweep(data, stats, all_specs=None,
                                  n_boot=None, rng_seed=42,
                                  sample_mask=None, out_dir=None,
                                  label=""):
    """1H. Robustness sweep across all detection thresholds.

    `sample_mask` restricts the sweep to a subset of the cohort (a boolean
    Series over `data.meta.index`), so the same estimator can be run on a study
    subset without duplicating the loop. `out_dir` and `label` let the caller
    keep those outputs separate from the whole-cohort run.
    """
    ref     = data.ref_name
    out_dir = out_dir or os.path.join("results", ref, "depth_bias_ipw")
    os.makedirs(out_dir, exist_ok=True)
    rng     = np.random.default_rng(rng_seed)
    if all_specs is None:
        all_specs = (breadth_pct_specs(config.ACCUM_BREADTH_THRESHOLDS) +
                     flat_bp_specs(config.ACCUM_FLAT_THRESHOLDS_BP))

    meta  = data.meta
    valid = meta["MacroGroup"].isin(["Industrial","Non-Industrial"])
    if sample_mask is not None:
        valid = valid & sample_mask.reindex(meta.index).fillna(False)
    if not valid.any():
        print("      [WARN] MacroGroup not set - skipping"); return None

    idx       = meta.index[valid]
    is_ind    = (meta.loc[idx,"MacroGroup"]=="Industrial").values
    log_depth = np.log10(np.maximum(data.total_bp.loc[idx].values, 1.0))

    if is_ind.sum() < 10 or (~is_ind).sum() < 10:
        print("      [WARN] Too few samples - skipping"); return None

    # Adaptive n_boot for the sweep (uses a lower cap to keep sweep fast)
    if n_boot is None:
        n_boot = _adaptive_n_boot(int(is_ind.sum()), int((~is_ind).sum()),
                                  base=getattr(config, "OW_AIPWE_N_BOOT_SWEEP", 200),
                                  cap=getattr(config, "OW_AIPWE_N_BOOT_SWEEP_CAP", 1000))

    print(f"\n    Threshold sweep [{ref}]{label}"
          f"  n={is_ind.sum():,}/{(~is_ind).sum():,}  n_boot={n_boot}")

    # Fit KDEs once (depth independent of detection spec)
    ld_i = log_depth[is_ind]; ld_n = log_depth[~is_ind]
    kde_i = _BinnedKDE(ld_i);  kde_n = _BinnedKDE(ld_n)
    w_ind = _geomean_weights(ld_i, kde_i, kde_n)
    w_ni  = _geomean_weights(ld_n, kde_n, kde_i)

    rows = []
    for spec in all_specs:
        print(f"      {spec.description}... ", end="", flush=True)
        presence = build_presence(data.breadth.loc[idx], data.depth.loc[idx],
                                  data.lengths, spec=spec)
        richness = presence.sum(axis=1).values.astype(float)
        ri = richness[is_ind]; rn = richness[~is_ind]
        mu1 = float(np.dot(w_ind,ri)/max(w_ind.sum(),1e-300))
        mu0 = float(np.dot(w_ni, rn)/max(w_ni.sum(), 1e-300))
        att = mu1 - mu0

        boot  = _gmw_bootstrap_2grp(ld_i,ld_n,ri,rn,w_ind,w_ni,n_boot,rng)
        valid_b = np.isfinite(boot[:,2])
        ci = (np.percentile(boot[valid_b,2],[2.5,97.5])
              if valid_b.any() else [np.nan,np.nan])
        raw_diff = float(ri.mean()-rn.mean())
        print(f"\u0394={att:+.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]  raw\u0394={raw_diff:+.1f}")

        stats.add(f"gmw_sweep_{spec.short_label}", "s1", "supplement",
            f"Threshold sweep: {spec.description}",
            f"\u0394={att:+.1f} [95%CI:{ci[0]:+.1f},{ci[1]:+.1f}]  raw\u0394={raw_diff:+.1f}",
            extra={"att":att,"ci_lo":ci[0],"ci_hi":ci[1],
                   "raw_diff":raw_diff,"mu1":mu1,"mu0":mu0})
        raw_pct       = (float(ri.mean()) / max(float(rn.mean()), 1e-6) - 1.0) * 100
        corr_pct      = (mu1 / max(mu0, 1e-6) - 1.0) * 100
        # % CIs: translate absolute att CIs onto percentage scale using mu0
        corr_pct_lo   = (ci[0] / max(mu0, 1e-6)) * 100  # ci_lo / mu0
        corr_pct_hi   = (ci[1] / max(mu0, 1e-6)) * 100  # ci_hi / mu0
        rows.append({"label":spec.description,"mode":spec.mode,
                     "threshold":spec.threshold,"att":att,
                     "ci_lo":ci[0],"ci_hi":ci[1],"raw_diff":raw_diff,
                     "raw_pct":raw_pct,"corr_pct":corr_pct,
                     "corr_pct_lo":corr_pct_lo,"corr_pct_hi":corr_pct_hi})

    sweep_df = pd.DataFrame(rows)
    stem = "gmw_threshold_sweep"
    sweep_df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    _make_sweep_figure(sweep_df, os.path.join(out_dir, f"{stem}.png"))
    return sweep_df
