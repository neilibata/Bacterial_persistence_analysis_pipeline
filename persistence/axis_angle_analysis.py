"""  # rebuilt 2026-04-16 00:32
Depth-binned extinction-axis projection and angle analysis (v2).

See docstring in original for full method description.
Changes in this version:
  - _run_bin now also returns per-sample (p, perp_signed) coordinates
  - _run_one_spec saves a per-sample CSV for the point-cloud figure
  - _plot: drops bottom panel; improves top-left with grand centroid,
    three conceptual arrows, right-angle marker, better labels;
    improves top-right with mean-angle line and stronger title
  - _plot_point_cloud: new second figure overlaying all bins' point
    clouds in the coordinate-invariant (p, perp) space with per-
    lifestyle KDE contours
"""

from __future__ import annotations
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import colorsys
from .plot_utils import (  # module level: the helpers below are used
    page_figsize,          # inside functions that do their own local
    panel_grid,            # plot_utils imports, which would otherwise
    add_post_layout_hook,  # shadow these names out of scope
    label_pad,
    plain_axes,
    share_axis_labels,
    square_figsize,
    PAGE_WIDTH_IN,
)

LIFESTYLE_ORDER     = ["H-G", "Transitional", "Industrial"]
GROUP_COLORS        = {"H-G": "#D55E00", "Transitional": "#F0E442",
                       "Industrial": "#56B4E9"}
DISPLAY_LABELS      = {"H-G": "Hunter-Gatherer",
                       "Transitional": "Transitional",
                       "Industrial": "Industrial",
                       "Ind": "Industrial",
                       "Trans": "Transitional"}


def _display(label: str) -> str:
    """The published label for a lifestyle key.

    Delegates to config, which is the single bridge between the internal keys
    the analysis compares against and the labels readers see. The local
    DISPLAY_LABELS table was a second copy of the same mapping and had fallen
    behind: it had no entry for "Transitional", so that key reached the
    figures verbatim.
    """
    from . import config as _cfg
    return _cfg.display_text(DISPLAY_LABELS.get(label, label))

# Continents ordered most-traditional → most-industrial; darker = more traditional
CONTINENT_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania", "Unknown"]
# Scaling factor applied to V (HSV brightness) per continent.
# Transitional yellow is near-max V so we scale S instead (see _continent_color).
CONTINENT_SCALE = {
    "Africa":   1.00,
    "Americas": 0.82,
    "Asia":     0.65,
    "Europe":   0.48,
    "Oceania":  0.35,
    "Unknown":  0.78,
}
# Marker shapes for per-study centroid panel (cycled)
STUDY_MARKERS = ['o', 's', '^', 'v', '<', '>', 'D', 'P', 'X',
                 'h', 'H', '8', 'p', '*']

def _continent_color(lifestyle: str, continent: str) -> tuple:
    """Return an RGB tuple that is a continent-specific shade of the lifestyle color."""
    base = GROUP_COLORS.get(lifestyle, "#888888")
    r = int(base[1:3], 16) / 255
    g = int(base[3:5], 16) / 255
    b = int(base[5:7], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    scale = CONTINENT_SCALE.get(continent, 0.78)
    if lifestyle == "Transitional":
        # Yellow is already maximum brightness; vary saturation instead
        r2, g2, b2 = colorsys.hsv_to_rgb(h, min(1.0, s * (0.3 + 0.7 * scale)), v)
    else:
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, min(1.0, max(0.15, v * scale)))
    return (r2, g2, b2)
N_BINS_DEFAULT      = 20
MIN_N_HG_DEFAULT    = 3
MIN_N_OTHER_DEFAULT = 10
N_BOOT_DEFAULT      = 300
N_PCOA_AXES         = 5


# == Weighted double-centring PCoA =============================================

def _weighted_pcoa(D_sq, weights, n_axes=N_PCOA_AXES):
    w    = weights / weights.sum()
    Ic   = np.eye(len(w)) - np.outer(np.ones(len(w)), w)
    B    = -0.5 * Ic @ (D_sq ** 2) @ Ic.T
    W12  = np.diag(np.sqrt(w))
    Bw   = W12 @ B @ W12
    try:
        from scipy.linalg import eigh
        n = len(w); start = max(0, n - n_axes)
        ev, evec = eigh(Bw, subset_by_index=[start, n - 1])
        ev = ev[::-1]; evec = evec[:, ::-1]
    except Exception:
        ev, evec = np.linalg.eigh(Bw)
        idx = np.argsort(ev)[::-1]
        ev = ev[idx[:n_axes]]; evec = evec[:, idx[:n_axes]]
    W12_inv = np.diag(1.0 / np.sqrt(np.maximum(w, 1e-300)))
    top_ev  = np.maximum(ev[:n_axes], 0.0)
    coords  = W12_inv @ evec[:, :n_axes] * np.sqrt(top_ev)
    total   = max(np.sum(np.maximum(ev, 0.0)), 1e-10)
    return coords, top_ev / total * 100.0


# == Coordinate-invariant geometry ============================================

def _centroid_geometry(c_hg, c_trans, c_ind):
    v     = c_ind - c_hg
    v_len = float(np.linalg.norm(v))
    if v_len < 1e-12:
        return dict(v_len=v_len, p=np.nan, perp=np.nan, theta=np.nan,
                    v_hat=None, e_perp=None)
    vhat  = v / v_len
    d     = c_trans - c_hg
    proj  = float(np.dot(d, vhat))
    p     = proj / v_len
    perp_vec = d - proj * vhat
    perp  = float(np.linalg.norm(perp_vec)) / v_len   # always >= 0
    # Signed perpendicular unit vector (positive = toward Trans)
    perp_raw_len = np.linalg.norm(perp_vec)
    e_perp = (perp_vec / perp_raw_len
              if perp_raw_len > 1e-12 else np.zeros_like(vhat))
    u     = c_ind - c_trans
    u_len = float(np.linalg.norm(u))
    theta = float(np.degrees(np.arccos(np.clip(
        np.dot(u / u_len, vhat), -1.0, 1.0)))) if u_len > 1e-12 else np.nan
    return dict(v_len=v_len, p=p, perp=perp, theta=theta,
                v_hat=vhat, e_perp=e_perp)


# == Bootstrap CI on theta =====================================================

def _bootstrap_theta(coords_by_group, n_boot, rng):
    thetas = np.empty(n_boot)
    for i in range(n_boot):
        cents = {}
        for lf, c in coords_by_group.items():
            n_g = len(c)
            if n_g == 0: continue
            cents[lf] = c[rng.integers(0, n_g, size=n_g)].mean(axis=0)
        if not all(lf in cents for lf in LIFESTYLE_ORDER):
            thetas[i] = np.nan; continue
        thetas[i] = _centroid_geometry(
            cents["H-G"], cents["Transitional"], cents["Industrial"])["theta"]
    valid = thetas[np.isfinite(thetas)]
    return (float(np.percentile(valid, 2.5)),
            float(np.percentile(valid, 97.5))) if len(valid) >= 5 else (np.nan, np.nan)


# == Per-bin Jaccard PCoA =====================================================

def _run_bin(ids_by_group, presence_all, n_boot, rng):
    """
    Returns centroid geometry + per-sample projected (p, perp_signed) coords.
    perp_signed uses e_perp defined by the Transitional centroid direction,
    giving a consistent sign convention across samples in this bin.
    """
    from scipy.spatial.distance import pdist, squareform
    all_ids = []; weights = []; lf_labels = []
    for lf in LIFESTYLE_ORDER:
        ids_lf = ids_by_group.get(lf, [])
        n_g    = len(ids_lf)
        if n_g == 0: continue
        all_ids.extend(ids_lf)
        weights.extend([1.0 / (3.0 * n_g)] * n_g)
        lf_labels.extend([lf] * n_g)
    n_total = len(all_ids)
    if n_total < 6: return None

    pres = presence_all.loc[all_ids].values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J = pdist(pres, metric="jaccard")
    J = np.nan_to_num(J, nan=1.0)
    D_sq = squareform(J)

    w_arr = np.array(weights, dtype=float)
    try:
        n_ax = min(N_PCOA_AXES, n_total - 1)
        coords, var_ex = _weighted_pcoa(D_sq, w_arr, n_axes=n_ax)
    except Exception:
        return None

    pos = 0; cents = {}; coords_by_group = {}
    for lf in LIFESTYLE_ORDER:
        ids_lf = ids_by_group.get(lf, [])
        n_g    = len(ids_lf)
        if n_g == 0: continue
        w_g = w_arr[pos:pos+n_g]; c_g = coords[pos:pos+n_g]
        wn  = w_g / max(w_g.sum(), 1e-300)
        cents[lf]           = c_g.T @ wn
        coords_by_group[lf] = c_g
        pos += n_g

    if not all(lf in cents for lf in LIFESTYLE_ORDER): return None
    geom = _centroid_geometry(cents["H-G"], cents["Transitional"],
                               cents["Industrial"])
    ci_lo, ci_hi = _bootstrap_theta(coords_by_group, n_boot, rng)
    ns = {lf: len(ids_by_group.get(lf, [])) for lf in LIFESTYLE_ORDER}

    # Per-sample projected coordinates in this bin's (p, perp_signed) space
    sample_projs = {}   # lf -> {"p": array, "perp": array}
    if geom["v_hat"] is not None:
        vhat   = geom["v_hat"]
        e_perp = geom["e_perp"]   # points toward Trans centroid (positive)
        v_len  = geom["v_len"]
        c_hg   = cents["H-G"]
        for lf in LIFESTYLE_ORDER:
            c_g = coords_by_group.get(lf)
            if c_g is None or len(c_g) == 0:
                continue
            deltas = c_g - c_hg           # (n_g, n_axes)
            p_arr  = deltas @ vhat / v_len
            perp_component = deltas - np.outer(deltas @ vhat, vhat)
            perp_signed = perp_component @ e_perp / v_len
            sample_projs[lf] = {"p": p_arr, "perp": perp_signed}

    return dict(geom=geom, ci_lo=ci_lo, ci_hi=ci_hi,
                n_hg=ns["H-G"], n_trans=ns["Transitional"],
                n_ind=ns["Industrial"],
                var_ex1=float(var_ex[0]) if len(var_ex)>0 else np.nan,
                var_ex2=float(var_ex[1]) if len(var_ex)>1 else np.nan,
                sample_projs=sample_projs)


# == Core per-spec function ===================================================

def _run_one_spec(data, spec, n_bins, min_n_hg, min_n_other,
                  n_boot, outdir, ref, stats=None, rng_seed=42):
    from .detection import build_presence
    spec_tag = spec.short_label
    rng      = np.random.default_rng(rng_seed)
    meta     = data.meta

    print(f"\n  [{spec_tag}] {spec.description}")

    valid_ids = [s for s in meta.index
                 if s in data.breadth.index and s in data.depth.index
                 and meta.loc[s, "Lifestyle"] in LIFESTYLE_ORDER]
    if not valid_ids:
        print("  [axis_angle] No valid samples"); return pd.DataFrame()

    log_depth  = np.log10(np.maximum(
        data.total_bp.reindex(valid_ids).fillna(1.0).values, 1.0))
    lifestyles = meta.reindex(valid_ids)["Lifestyle"].values

    print(f"    Total: {len(valid_ids):,}  " +
          "  ".join(f"{lf}: {int((lifestyles==lf).sum()):,}"
                    for lf in LIFESTYLE_ORDER))

    # Build per-sample continent + author lookups (used by point-cloud figure)
    valid_meta = meta.reindex(valid_ids)
    _cont_col  = next((c for c in ["Continent", "Country_norm"] if c in meta.columns), None)
    _auth_col  = next((c for c in ["Author", "_study", "StudyFolder"] if c in meta.columns), None)
    continent_lookup = (valid_meta[_cont_col].fillna("Unknown").to_dict()
                        if _cont_col else {})
    author_lookup    = (valid_meta[_auth_col].fillna("Unknown").to_dict()
                        if _auth_col else {})
    # Country is already ISO alpha-3 in the metadata, which is exactly the
    # three characters a per-study marker can carry without a leader line.
    _ctry_col  = next((c for c in ["Country", "Country_norm"]
                       if c in meta.columns), None)
    country_lookup   = (valid_meta[_ctry_col].fillna("").astype(str).to_dict()
                        if _ctry_col else {})

    print(f"    Building presence matrix...")
    presence_all = build_presence(
        data.breadth.loc[valid_ids], data.depth.loc[valid_ids],
        data.lengths, spec=spec)

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges     = np.unique(np.percentile(log_depth, quantiles))
    if len(edges) < 3:
        print("  [axis_angle] Too few unique depth values"); return pd.DataFrame()
    actual_bins  = len(edges) - 1
    bin_mids_log = 0.5 * (edges[:-1] + edges[1:])
    bin_mids_gbp = 10**bin_mids_log / 1e9
    print(f"    {actual_bins} quantile bins, "
          f"~{len(valid_ids)//actual_bins} samples each")

    rows = []
    all_sample_rows = []   # for point-cloud figure

    for b in range(actual_bins):
        lo, hi = edges[b], edges[b+1]
        in_bin = ((log_depth >= lo) & (log_depth < hi)
                  if b < actual_bins-1
                  else (log_depth >= lo) & (log_depth <= hi))
        ids_by_group = {lf: [valid_ids[i] for i in np.where(
            in_bin & (lifestyles == lf))[0]] for lf in LIFESTYLE_ORDER}
        if (len(ids_by_group["H-G"]) < min_n_hg
                or len(ids_by_group["Transitional"]) < min_n_other
                or len(ids_by_group["Industrial"]) < min_n_other):
            continue
        result = _run_bin(ids_by_group, presence_all, n_boot, rng)
        if result is None: continue
        g = result["geom"]
        row = dict(bin_idx=b,
                   bin_lo_log10bp=round(lo,4), bin_hi_log10bp=round(hi,4),
                   bin_mid_log10bp=round(bin_mids_log[b],4),
                   bin_mid_Gbp=round(bin_mids_gbp[b],4),
                   n_HG=result["n_hg"], n_Trans=result["n_trans"],
                   n_Ind=result["n_ind"],
                   v_len=round(g["v_len"],6),
                   p_frac=round(g["p"],4) if np.isfinite(g["p"]) else np.nan,
                   perp_norm=round(g["perp"],4) if np.isfinite(g["perp"]) else np.nan,
                   theta_deg=round(g["theta"],3) if np.isfinite(g["theta"]) else np.nan,
                   theta_ci_lo=round(result["ci_lo"],3) if np.isfinite(result["ci_lo"]) else np.nan,
                   theta_ci_hi=round(result["ci_hi"],3) if np.isfinite(result["ci_hi"]) else np.nan,
                   var_ex_pc1_pct=round(result["var_ex1"],2),
                   var_ex_pc2_pct=round(result["var_ex2"],2))
        rows.append(row)

        # Collect per-sample projected positions for point-cloud figure
        for lf, proj in result["sample_projs"].items():
            ids_lf = ids_by_group.get(lf, [])
            for i, (pv, perpv) in enumerate(zip(proj["p"], proj["perp"])):
                sid = ids_lf[i] if i < len(ids_lf) else ""
                all_sample_rows.append(dict(
                    bin_idx=b, bin_mid_Gbp=round(bin_mids_gbp[b], 4),
                    Lifestyle=lf, p=float(pv), perp_signed=float(perpv),
                    continent=continent_lookup.get(sid, "Unknown"),
                    country=country_lookup.get(sid, ""),
                    author=author_lookup.get(sid, "Unknown")))

        n_hg_v = result["n_hg"]; n_tr_v = result["n_trans"]
        n_in_v = result["n_ind"]
        p_v = g["p"]; perp_v = g["perp"]; theta_v = g["theta"]
        print(f"    Bin {b:2d} ({bin_mids_gbp[b]:.2f} Gbp): "
              f"n=({n_hg_v},{n_tr_v},{n_in_v})  "
              f"p={p_v:.3f}  perp={perp_v:.3f}  "
              f"theta={theta_v:.1f}deg")

    if not rows:
        print("  [axis_angle] No qualifying bins"); return pd.DataFrame()

    df          = pd.DataFrame(rows)
    samples_df  = pd.DataFrame(all_sample_rows)

    print(f"\n  {len(df)} qualifying bins")

    # Per-sample distribution of the Transitional position along the
    # HG->Ind axis. The per-bin summary above answers "where does the average
    # bin sit"; the manuscript quotes a median and IQR over samples, which is a
    # different statistic and was not previously emitted.
    if stats is not None and len(samples_df):
        tr = samples_df[samples_df["Lifestyle"] == "Transitional"]
        for col, tag, desc in [
            ("p", f"axis_sample_pfrac_{spec_tag}",
             "Per-sample Transitional fractional position (0=HG, 1=Ind)"),
            ("perp_signed", f"axis_sample_perp_{spec_tag}",
             "Per-sample signed perpendicular displacement of Transitional"),
        ]:
            v = tr[col].dropna()
            if len(v):
                stats.add(tag, "s1", "main", desc,
                          f"median={v.median():.3f}, IQR=[{v.quantile(.25):.3f}-"
                          f"{v.quantile(.75):.3f}], mean={v.mean():.3f}, n={len(v)}",
                          n1=int(len(v)))

    if stats is not None:
        for tag, col, desc in [
            (f"axis_angle_mean_theta_{spec_tag}", "theta_deg",
             "Mean Trans->Ind angle vs HG->Ind (degrees)"),
            (f"axis_angle_mean_pfrac_{spec_tag}", "p_frac",
             "Mean Trans fractional position (0=HG, 1=Ind)"),
            (f"axis_angle_mean_perp_{spec_tag}", "perp_norm",
             "Mean normalised perpendicular displacement of Trans"),
        ]:
            vals = df[col].dropna()
            if len(vals):
                stats.add(tag, "s1", "main", desc,
                          f"mean={vals.mean():.3f}, median={vals.median():.3f}, "
                          f"SD={vals.std():.3f}, n_bins={len(vals)}", n1=len(vals))

    csv_path = os.path.join(outdir, f"axis_angle_v2_{spec_tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    smp_csv = os.path.join(outdir, f"axis_angle_v2_samples_{spec_tag}.csv")
    samples_df.to_csv(smp_csv, index=False)
    print(f"  Saved: {smp_csv}")

    _plot(df, spec_tag, outdir, ref, n_bins, min_n_hg, min_n_other)
    if not samples_df.empty:
        _plot_point_cloud(samples_df, df, spec_tag, outdir, ref)
    return df


# == Figure 1: summary panel ==================================================

def _plot(df, spec_tag, outdir, ref, n_bins, min_n_hg, min_n_other):
    """
    Two-panel figure (bottom panel removed).
    Left : fan plot with grand centroid, three conceptual arrows,
           right-angle marker, lighter trajectory line.
    Right: angle vs depth with mean line, stronger title.
    """
    try:
        from .plot_utils import (
            apply_oi_style,
            current_fs,
            page_figsize,
            label_pad,
            plain_axes,
            square_figsize,
            panel_grid,
            savefig_dual,
            share_axis_labels,
        )
    except ImportError:
        def current_fs(): return 12
        def apply_oi_style(*a, **k): pass
        def savefig_dual(fig, path, **kw): fig.savefig(path, **kw)
    FS = current_fs()

    # Pin the size. Run after another section this figure came out at 8.5 pt
    # and on its own at 10, because nothing here asked for a size and the last
    # module to call the style setter won. At 10 the long strings overflow and
    # the saved canvas grows to 7.5 in from a 6.3 in page, which costs more
    # printed size than the extra point buys. Measured at 8.5 the canvas stays
    # on the page.
    # Side by side, on a canvas sized for it. Two panels across a 6.3 in page
    # are 2.6 in each and no type size fixes that. Widening the canvas and
    # raising the base together does: at 13.6 in each panel gets the same 5.7 in
    # it would have had stacked, and the type scales with it, so the figure
    # prints at about 10.5 pt once placed back in the column. Printed size is
    # (base - 3) x 6.3 / saved width -- the width and the base have to move
    # together or one cancels the other.
    # Bigger again. At 13.6 in the panels were 5.7 in and the key, the titles
    # and the annotations were all competing for the same panel. At 20 in they
    # are 8.2 in, which is where a seven-entry key fits inside the panel
    # without reaching the title. The base scales with the canvas so printed
    # size holds at about 9 pt: printed = (base - 3) x 6.3 / width.
    # Authored at the printed width: a canvas reduced to a third takes
    # every line and symbol with it, and a 1 pt rule landed at 0.36 pt,
    # under the 0.5 pt floor.
    _W = PAGE_WIDTH_IN
    _MG = dict(top=0.89, bottom=0.115, left=0.075, right=0.90, wspace=0.17)
    # The canvas is now authored at the printed width, so the shared
    # scale applies: a base tuned for a 20-28 in canvas made the type
    # three times too large once the figure stopped being reduced.
    apply_oi_style()
    FS = current_fs()
    fig, (ax_fan, ax_ang) = plt.subplots(
        1, 2, figsize=square_figsize(n_cols=2, width_in=_W, margins=_MG))
    # No longer wide on purpose: this figure is authored at the printed
    # width now, so any excess is overflow to be trimmed, not intent.

    depths  = df["bin_mid_log10bp"].values
    norm_d  = (depths - depths.min()) / max(np.ptp(depths), 1e-9)
    cmap    = cm.get_cmap("plasma")
    bcolors = [cmap(v) for v in norm_d]

    p    = df["p_frac"].values
    perp = df["perp_norm"].values
    valid = np.isfinite(p) & np.isfinite(perp)
    p_v  = p[valid];  perp_v = perp[valid]

    # Grand centroid of Transitional bins (median is robust to outlier bins)
    trans_cx = float(np.median(p_v))
    trans_cy = float(np.median(perp_v))

    # ── Top-left: Fan plot ────────────────────────────────────────────────────

    # Light horizontal baseline
    ax_fan.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.35, zorder=1)

    # Per-bin Transitional points (smaller, lighter connecting line)
    sort_idx = np.argsort(depths[valid])
    ax_fan.plot(p_v[sort_idx], perp_v[sort_idx],
                color="gray", lw=0.4, alpha=0.25, zorder=2)
    ax_fan.scatter(p_v, perp_v,
                   c=[bcolors[i] for i in np.where(valid)[0]],
                   s=40, edgecolors="black", linewidths=0.4,
                   alpha=0.75, zorder=3)

    # H-G and Industrial anchor points
    ax_fan.scatter([0], [0], c=GROUP_COLORS["H-G"], s=70, marker="^",
                   edgecolors="black", lw=1.0, zorder=6,
                   label=f"{_display('H-G')} anchor")
    ax_fan.scatter([1], [0], c=GROUP_COLORS["Industrial"], s=70, marker="D",
                   edgecolors="black", lw=1.0, zorder=6, label="Industrial anchor")

    # Transitional grand centroid — large outlined star
    ax_fan.scatter([trans_cx], [trans_cy],
                   c=GROUP_COLORS["Transitional"], s=110, marker="*",
                   edgecolors="black", lw=1.2, zorder=7,
                   # Its coordinates were in the key on a second line, which
                   # made this the longest entry and pushed the key into the
                   # title. They are readable off the axes it is plotted on.
                   label=f"{_display('Transitional')} grand centroid")

    # Right-angle marker from grand centroid down to axis
    box_size = 0.025
    ax_fan.plot([trans_cx, trans_cx, trans_cx - box_size],
                [box_size, box_size, box_size],
                color="gray", lw=0.9, alpha=0.6, zorder=5)

    # Arrow 1 — Hunter-Gatherer -> Industrial axis (VANISH/BloSSUM), just above y=0
    ofs = trans_cy * 0.06    # small vertical offset
    ax_fan.annotate(
        "", xy=(0.96, ofs), xycoords="data",
        xytext=(0.04, ofs), textcoords="data",
        arrowprops=dict(arrowstyle="-|>", color="#222222",
                        lw=2.0, mutation_scale=18),
        zorder=8)
    # Axes fractions, not data coordinates, and clipped. In data coordinates
    # these three labels ran wherever the centroid happened to be: the
    # Industrialisation one left the panel entirely and landed on the next
    # panel's tick labels (100% overlap in the collision map). Fixed fractions
    # also keep the three apart from each other, which data placement did not.
    # (named in the key -- see _AXIS_KEYS below)

    # Arrow 2 — Hunter-Gatherer -> Transitional grand centroid (Agriculturalization)
    ax_fan.annotate(
        "", xy=(trans_cx * 0.91, trans_cy * 0.91), xycoords="data",
        xytext=(0.02, 0.00), textcoords="data",
        arrowprops=dict(arrowstyle="-|>",
                        color=GROUP_COLORS["H-G"],
                        lw=1.8, mutation_scale=16,
                        linestyle="dashed"),
        zorder=8)


    # Arrow 3 — Transitional grand centroid -> Industrial (Industrialisation / VANISH+/BloSSUM+)
    ax_fan.annotate(
        "", xy=(0.96, 0.00), xycoords="data",
        xytext=(trans_cx + 0.03, trans_cy * 0.88), textcoords="data",
        arrowprops=dict(arrowstyle="-|>",
                        color=GROUP_COLORS["Industrial"],
                        lw=1.8, mutation_scale=16,
                        linestyle="dashed"),
        zorder=8)


    # The axis labels and titles used to carry their own definitions on a
    # second line -- what 0 and 1 mean, what the normalisation is, what a
    # positive Y indicates. That is caption material. Here it made the two
    # largest strings in the figure two lines each, which is what pushed the
    # saved canvas to 8.4 in from a 6.3 in page and took the printed type down
    # to 4.1 pt. The range of sizes was upside down as well: definitions at
    # full size, the key and the annotations that name the data at FS-2.
    # labelpad, because squaring the panels moved the labels in: at the default
    # pad the y label sat on the top tick and the x label on the ticks below it
    # (72% and 61% overlap). The pad is in points, so it holds at any size.
    ax_fan.set_xlabel("Position along Hunter-Gatherer to Industrial axis",
                      fontsize=FS - 1, labelpad=label_pad())
    ax_fan.set_ylabel("Off-axis displacement (normalised)",
                      fontsize=FS - 1, labelpad=label_pad())
    ax_fan.set_title("Transitional centroid per depth bin",
                     fontsize=FS, fontweight="bold")
    # The three arrows used to be captioned by floating italic text inside the
    # panel. Fixed data positions put them wherever the centroid happened to
    # be, fixed axes fractions put them on the key instead, and either way they
    # were three more boxes competing for the same empty corner. They name the
    # three arrows, so they belong in the key beside a swatch that matches each
    # arrow's colour and dash.
    from matplotlib.lines import Line2D as _Line
    _AXIS_KEYS = [
        _Line([0], [0], color="#222222", lw=2.0, ls="-",
              label="Hunter-Gatherer to Industrial (VANISH / BloSSUM)"),
        _Line([0], [0], color=GROUP_COLORS["H-G"], lw=1.8, ls="--",
              label="Agriculturalization axis"),
        _Line([0], [0], color=GROUP_COLORS["Industrial"], lw=1.8, ls="--",
              label="Industrialisation axis (VANISH+ / BloSSUM+)"),
    ]
    # One key, inside the panel, where a reader looks for it. Two columns: at
    # seven entries a single column is half the panel height whatever the size,
    # and an 8.2 in panel is wide enough to lay them out two across.
    # A band of headroom above the trajectory for the key to sit in. The panel
    # is what grows, not the type: with seven entries there is no corner of the
    # data-filled area big enough, and shrinking the key further would put it
    # below the size anything else in the figure is set at.
    _y0, _y1 = ax_fan.get_ylim()
    ax_fan.set_ylim(_y0, _y1 + 0.34 * (_y1 - _y0))

    _h, _l = ax_fan.get_legend_handles_labels()
    ax_fan.legend(_h + _AXIS_KEYS, _l + [a.get_label() for a in _AXIS_KEYS],
                  fontsize=FS - 9, loc="upper right", framealpha=0.92, ncol=2,
                  borderaxespad=0.7, labelspacing=0.35, handlelength=1.6,
                  columnspacing=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(vmin=10**depths.min()/1e9,
                             vmax=10**depths.max()/1e9))
    sm.set_array([])
    # One colorbar at the figure's right edge, spanning both panels. Vertical
    # beside the left panel it shared the gutter with the right panel's y label
    # and its ticks sat on that label; moved under the left panel it sat on that
    # panel's x label instead, and took height from one panel only, which broke
    # the square panels. Outside both, it conflicts with neither and costs each
    # panel the same width.
    cb = fig.colorbar(sm, ax=[ax_fan, ax_ang], location="right",
                      fraction=0.022, pad=0.015, aspect=34)
    cb.set_label("Depth bin midpoint (Gbp)", fontsize=FS - 3)
    cb.ax.tick_params(labelsize=FS - 4)

    # ── Top-right: Angle plot ─────────────────────────────────────────────────
    sub = df.dropna(subset=["theta_deg"])
    mean_t = sd_t = np.nan
    if not sub.empty:
        xg     = 10**sub["bin_mid_log10bp"].values / 1e9
        ci_lo  = sub["theta_ci_lo"].values
        ci_hi  = sub["theta_ci_hi"].values
        ci_ok  = np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if ci_ok.any():
            ax_ang.fill_between(xg[ci_ok], ci_lo[ci_ok], ci_hi[ci_ok],
                                color="#0072B2", alpha=0.18,
                                label="Bootstrap 95% CI")
        ax_ang.scatter(xg, sub["theta_deg"].values,
                       c="#0072B2", s=55, edgecolors="black",
                       lw=0.4, zorder=4,
                       label=f"\u03b8 per bin (n={len(sub)})")
        ax_ang.plot(xg, sub["theta_deg"].values,
                    color="#0072B2", lw=1.0, alpha=0.5)

        mean_t = float(sub["theta_deg"].mean())
        sd_t   = float(sub["theta_deg"].std(ddof=1))

        # Mean angle reference line
        ax_ang.axhline(mean_t, color="#0072B2", lw=1.5, ls="-.",
                       alpha=0.65,
                       label=f"Mean = {mean_t:.1f}\u00b0 "
                             f"\u00b1 {sd_t:.1f}\u00b0")

        ax_ang.axhline(0,  color="black", lw=1.2, ls="-",
                       label="\u03b8 = 0\u00b0 (parallel / collinear)")
        ax_ang.axhline(90, color="gray",  lw=0.8, ls="--", alpha=0.5,
                       label="\u03b8 = 90\u00b0 (orthogonal)")

        # The mean deviation used to be a boxed note pinned top-right, which is
        # a corner place_legends_clear also likes: the legend landed on it at
        # full overlap. It restates a line that is already in the key, so it
        # belongs there, beside the line it describes.

    ax_ang.set_xscale("log")
    # A log axis labels its minor ticks too, which doubles the number of
    # strings under the axis and is what the x label kept landing on.
    from matplotlib.ticker import NullFormatter as _NullFmt
    ax_ang.xaxis.set_minor_formatter(_NullFmt())
    ax_ang.set_xlabel("Depth bin midpoint (Gbp)", fontsize=FS - 1, labelpad=label_pad())
    ax_ang.set_ylabel("theta (degrees)", fontsize=FS - 1, labelpad=label_pad())
    ax_ang.set_title("Transitional to Industrial vs Hunter-Gatherer to Industrial",
                     fontsize=FS, fontweight="bold")
    ax_ang.legend(fontsize=FS - 1, loc="upper left")
    ax_ang.set_ylim(bottom=-5)

    fig.suptitle(
        f"Depth-binned extinction-axis analysis - {ref}  [{spec_tag}]",
        fontsize=FS + 1, fontweight="bold")
    fig._margin_overrides = _MG
    from matplotlib.ticker import MaxNLocator as _MaxN
    for _a in (ax_fan, ax_ang):
        _a.yaxis.set_major_locator(_MaxN(nbins=4))
    # The figure's data object: the per-bin frame it plots.
    from . import figure_source
    figure_source.emit_for_path(
        os.path.join(outdir, f"axis_angle_v2_{spec_tag}.png"),
        {"bins": df.reset_index(drop=True)})

    png_path = os.path.join(outdir, f"axis_angle_v2_{spec_tag}.png")
    try:
        savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    except Exception:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


# == Figure 2: two-panel point-cloud ==========================================

def _repel_labels(items, min_dx=0.05, min_dy=0.03, n_iter=200):
    """
    Iterative spring repulsion for label positions.
    items: list of dicts with keys 'lx','ly' (label pos, mutable).
    Operates in-place.  O(n² × n_iter) — fine for n < 150 studies.
    """
    for _ in range(n_iter):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                dx = items[i]["lx"] - items[j]["lx"]
                dy = items[i]["ly"] - items[j]["ly"]
                # Elliptical distance (axis-normalised)
                dn = np.sqrt((dx / min_dx) ** 2 + (dy / min_dy) ** 2)
                if 0 < dn < 1.0:
                    push  = (1.0 - dn) * 0.5
                    angle = np.arctan2(dy / min_dy, dx / min_dx)
                    items[i]["lx"] += push * min_dx * np.cos(angle)
                    items[i]["ly"] += push * min_dy * np.sin(angle)
                    items[j]["lx"] -= push * min_dx * np.cos(angle)
                    items[j]["ly"] -= push * min_dy * np.sin(angle)


def _kde_contours(ax, xv, yv, color, levels=(0.90, 0.50)):
    """Draw filled + line KDE contours at specified probability-mass levels."""
    from scipy.stats import gaussian_kde
    if len(xv) < 10:
        return
    try:
        kde = gaussian_kde(np.vstack([xv, yv]), bw_method="scott")
        xi = np.linspace(xv.min() - 0.1, xv.max() + 0.1, 180)
        yi = np.linspace(yv.min() - 0.06, yv.max() + 0.06, 180)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        z_flat = Zi.ravel()
        order  = np.argsort(z_flat)[::-1]
        z_cum  = np.cumsum(z_flat[order]) / z_flat.sum()
        iso = {}
        for lv in sorted(levels):
            iso[lv] = z_flat[order[np.searchsorted(z_cum, lv)]]
        # Filled bands (outer then inner)
        ax.contourf(Xi, Yi, Zi, levels=[iso[0.90], Zi.max()],
                    colors=[color], alpha=0.09, zorder=1)
        ax.contourf(Xi, Yi, Zi, levels=[iso[0.50], Zi.max()],
                    colors=[color], alpha=0.13, zorder=1)
        ax.contour(Xi, Yi, Zi, levels=[iso[0.90]],
                   colors=[color], linewidths=[0.9],
                   linestyles=["--"], alpha=0.55, zorder=3)
        ax.contour(Xi, Yi, Zi, levels=[iso[0.50]],
                   colors=[color], linewidths=[1.5],
                   linestyles=["-"], alpha=0.80, zorder=3)
    except Exception:
        pass


def _plot_pointcloud_main(samples_df, spec_tag, outdir, ref):
    """
    Figure 2A (main text): point-cloud of all projected samples.
    Solid lifestyle colours, KDE contours, grand centroids, conceptual arrows.
    Output: axis_angle_v2_pointcloud_<spec>.png
    """
    try:
        from .plot_utils import apply_oi_style, current_fs, savefig_dual
    except ImportError:
        def current_fs(): return 12
        def savefig_dual(fig, path, **kw): fig.savefig(path, **kw)
    FS = current_fs()

    # Square: this is a 2-D ordination, so equal units on both axes is the
    # honest shape, and a wide short panel distorts the geometry it exists to
    # show.
    apply_oi_style(base=11)   # one page-wide square panel carries this
    FS = current_fs()
    fig, ax = plt.subplots(figsize=square_figsize(margins=dict(top=0.91)))
    fig._margin_overrides = dict(top=0.91)
    centroid_of: dict = {}

    for lf in LIFESTYLE_ORDER:
        sub = samples_df[samples_df["Lifestyle"] == lf]
        if sub.empty:
            continue
        color = GROUP_COLORS.get(lf, "gray")
        xv = sub["p"].values
        yv = sub["perp_signed"].values
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]
        if len(xv) < 5:
            continue
        centroid_of[lf] = (float(np.median(xv)), float(np.median(yv)))

        # The cloud is the data; the centroid markers are annotation. At 9-14 pt
        # against 240-360 pt the annotation outweighed the thing it marks.
        dot_sz    = {"H-G": 20, "Transitional": 13, "Industrial": 13}.get(lf, 13)
        dot_alpha = {"H-G": 0.45, "Transitional": 0.18, "Industrial": 0.16}.get(lf, 0.18)
        ax.scatter(xv, yv, c=color, s=dot_sz, alpha=dot_alpha,
                   edgecolors="none", zorder=2)
        _kde_contours(ax, xv, yv, color)

        cx, cy = centroid_of[lf]
        mk = {"H-G": "^", "Transitional": "*", "Industrial": "D"}.get(lf, "o")
        sz = {"H-G": 70, "Transitional": 100, "Industrial": 70}.get(lf, 60)
        ax.scatter(cx, cy, c=color, s=sz, marker=mk,
                   edgecolors="black", lw=1.4, zorder=7,
                   label=f"{_display(lf)}  (n={len(xv):,})")

    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.35, zorder=1)

    c_hg  = centroid_of.get("H-G",  (0.02, 0.00))
    c_tr  = centroid_of.get("Transitional", (0.45, 0.30))
    c_ind = centroid_of.get("Industrial", (1.00, 0.00))

    def _arr(ax_, xy, xytext, color, lw=1.8, ls="solid"):
        ax_.annotate("", xy=xy, xycoords="data",
                     xytext=xytext, textcoords="data",
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=lw, mutation_scale=16,
                                     linestyle=ls),
                     zorder=8)

    # The two dashed arrows share an endpoint at the Transitional centroid and
    # ran almost on top of each other where they met. Offsetting the shared end
    # perpendicular to each arrow separates them without moving what they point
    # at. The solid arrow sits clear of both, on the axis line.
    ofs = c_tr[1] * 0.08
    _sep = max(abs(c_tr[1]), 0.05) * 0.09
    _arr(ax, (c_ind[0]*0.97, ofs), (c_hg[0]+0.02, ofs), "#222222", lw=2.0)
    _arr(ax, (c_tr[0]*0.92 - _sep, c_tr[1]*0.92), (c_hg[0]+0.01, 0.00),
         GROUP_COLORS["H-G"], lw=1.7, ls="dashed")
    _arr(ax, (c_ind[0]*0.97, 0.00), (c_tr[0]+0.03 + _sep, c_tr[1]*0.88),
         GROUP_COLORS["Industrial"], lw=1.7, ls="dashed")

    # The captions for those three arrows were floating italic text in data
    # coordinates, and the Industrialisation one was Industrial blue written
    # over a cloud of Industrial blue dots. In the key each gets a swatch that
    # matches its arrow, on the key's own background, where it reads.
    from matplotlib.lines import Line2D as _Line
    _axis_keys = [
        _Line([0], [0], color="#222222", lw=2.0, ls="-",
              label="Hunter-Gatherer to Industrial (VANISH / BloSSUM)"),
        _Line([0], [0], color=GROUP_COLORS["H-G"], lw=1.7, ls="--",
              label="Agriculturalization axis"),
        _Line([0], [0], color=GROUP_COLORS["Industrial"], lw=1.7, ls="--",
              label="Industrialisation axis (VANISH+ / BloSSUM+)"),
    ]
    lf_handles, _lbl = ax.get_legend_handles_labels()
    ax.legend(handles=lf_handles + _axis_keys,
              labels=_lbl + [k.get_label() for k in _axis_keys],
              fontsize=FS-3, loc="upper right", framealpha=0.92,
              borderaxespad=0.6, labelspacing=0.35, handlelength=1.8)
    # The second title line said what the contours and marker shapes mean --
    # a key in prose, running the full width of the figure. It is what pushed
    # the saved canvas to 7.4 in from a 6.3 in page, and at that width the whole
    # figure shrinks 17% on placement. Caption material; see caption_debt.csv.
    ax.set_xlabel("Position along Hunter-Gatherer to Industrial axis",
                  fontsize=FS - 1, labelpad=label_pad())
    ax.set_ylabel("Off-axis displacement", fontsize=FS - 1, labelpad=label_pad())
    ax.set_title(f"Sample positions on the extinction axis  [{spec_tag}]",
                 fontsize=FS, fontweight="bold")

    png_path = os.path.join(outdir, f"axis_angle_v2_pointcloud_{spec_tag}.png")
    try:
        savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    except Exception:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


def _plot_pointcloud_supp(samples_df, spec_tag, outdir, ref):
    """
    Figure 2B (supplement): per-study centroids with repelling black labels.
    Output: axis_angle_v2_study_centroids_<spec>.png
    """
    if "author" not in samples_df.columns:
        return

    try:
        from .plot_utils import apply_oi_style, current_fs, savefig_dual
    except ImportError:
        def current_fs(): return 12
        def savefig_dual(fig, path, **kw): fig.savefig(path, **kw)
    FS = current_fs()

    # Darken Industrial colour for white-background visibility
    import colorsys as _cs
    _ih = GROUP_COLORS["Industrial"]
    _r, _g, _b = tuple(int(_ih[i:i+2], 16)/255 for i in (1, 3, 5))
    _h, _s, _v = _cs.rgb_to_hsv(_r, _g, _b)
    IND_DARK = _cs.hsv_to_rgb(_h, min(1.0, _s*1.4), max(0.0, _v*0.65))

    # Per-(author, lifestyle) centroid
    study_data: dict = {}
    for (author, lf), grp in samples_df.groupby(["author", "Lifestyle"]):
        ok = np.isfinite(grp["p"]) & np.isfinite(grp["perp_signed"])
        g  = grp[ok]
        if len(g) < 5:
            continue
        # One country per study in almost every case; where a study spans
        # several, name the one most of its samples came from.
        _cc = ""
        if "country" in g.columns:
            _vc = g["country"].astype(str).str.strip()
            _vc = _vc[_vc.str.len() == 3]
            if len(_vc):
                _cc = _vc.value_counts().index[0]
        study_data[(author, lf)] = {
            "p":    float(g["p"].median()),
            "perp": float(g["perp_signed"].median()),
            "n":    len(g),
            "cc":   _cc,
        }

    from . import figure_source
    _rows = [dict(author=a, lifestyle=lf, p=v["p"], perp=v["perp"],
                  n=v["n"], country=v.get("cc", ""))
             for (a, lf), v in study_data.items()]
    _tables = {"study_centroids": pd.DataFrame(_rows)}
    figure_source.emit_for_path(
        os.path.join(outdir, f"axis_angle_v2_study_centroids_{spec_tag}.png"),
        _tables)
    return plot_study_centroids(_tables, spec_tag, outdir, ref)


def plot_study_centroids(tables, spec_tag, outdir, ref):
    """Draw the per-study centroid figure from its table, and nothing else.

    One row per study: its position along the axis, its off-axis
    displacement, its sample count and its country code. Marker size,
    which studies get a label, and the axis limits are all functions of
    those columns.
    """
    from .plot_utils import apply_oi_style, current_fs, savefig_dual
    import colorsys as _cs

    df = tables["study_centroids"]
    _ih = GROUP_COLORS["Industrial"]
    _r, _g, _b = tuple(int(_ih[i:i+2], 16) / 255 for i in (1, 3, 5))
    _h, _s, _v = _cs.rgb_to_hsv(_r, _g, _b)
    IND_DARK = _cs.hsv_to_rgb(_h, min(1.0, _s * 1.4), max(0.0, _v * 0.65))

    study_data = {
        (row["author"], row["lifestyle"]): {
            "p": float(row["p"]), "perp": float(row["perp"]),
            "n": int(row["n"]), "cc": str(row["country"] or ""),
        }
        for _, row in df.iterrows()
    }
    if not study_data:
        return None

    apply_oi_style(base=11)   # one page-wide square panel carries this
    FS = current_fs()
    fig, ax = plt.subplots(figsize=square_figsize(margins=dict(top=0.91)))

    # Plot all circles
    for (author, lf), sd in sorted(study_data.items(),
                                    key=lambda kv: kv[1]["n"]):
        color_plot = (IND_DARK if lf == "Industrial"
                      else GROUP_COLORS.get(lf, "gray"))
        # Area still encodes n, as the key says; the ceiling comes down so the
        # largest study does not swamp the panel it sits in.
        sz = max(14, min(210, sd["n"] ** 0.5 * 8))
        ax.scatter(sd["p"], sd["perp"],
                   c=[color_plot], s=sz, marker="o",
                   edgecolors="white", linewidths=0.4,
                   zorder=4, alpha=0.88)

    # Label selection.
    #
    # There are far more Industrial studies than there is room for labels, and
    # labelling a fixed twenty of them plus every outlier produced forty-two
    # label-on-label overlaps -- "HMP" sitting 89% on top of "QinJ". Spring
    # repulsion cannot fix that: past a certain density there is no arrangement
    # that fits, so something has to be dropped rather than nudged.
    #
    # So this is a priority list, not a filter, and how far down it we get is
    # decided by measurement at draw time (_keep_fitting_labels below):
    #   1. every non-Industrial study above the size floor -- these are the
    #      point of the figure and there are few enough to always fit;
    #   2. Industrial studies by size, largest first;
    #   3. Industrial studies by distance from the Industrial median, most
    #      distinctive first.
    # Anything that would land on a label already placed is dropped.
    MIN_N_OTHER  = 10

    ind_ps   = [v["p"]    for (a, lf), v in study_data.items() if lf == "Industrial"]
    ind_pp   = [v["perp"] for (a, lf), v in study_data.items() if lf == "Industrial"]
    ind_med_p    = float(np.median(ind_ps))   if ind_ps else 1.0
    ind_med_perp = float(np.median(ind_pp))   if ind_pp else 0.0

    def _dist(v):
        return float(np.hypot(v["p"] - ind_med_p, v["perp"] - ind_med_perp))

    ind_items = [(a, v) for (a, lf), v in study_data.items()
                 if lf == "Industrial"]
    by_size   = [a for a, _ in sorted(ind_items, key=lambda x: -x[1]["n"])]
    by_dist   = [a for a, _ in sorted(ind_items, key=lambda x: -_dist(x[1]))]
    # Largest first, then most distinctive, each name once.
    ind_order, _seen = [], set()
    for a in by_size + by_dist:
        if a not in _seen:
            _seen.add(a); ind_order.append(a)
    ind_rank = {a: i for i, a in enumerate(ind_order)}

    # Build label items with initial positions, in priority order
    label_items = []
    for (author, lf), sd in sorted(
            study_data.items(),
            key=lambda kv: (0, -kv[1]["n"]) if kv[0][1] != "Industrial"
            else (1, ind_rank.get(kv[0][0], 10**6))):
        if lf != "Industrial" and sd["n"] < MIN_N_OTHER:
            continue
        short = str(author).rstrip("0123456789")[:12]
        if sd.get("cc"):
            short = f"{short} ({sd['cc']})"
        if lf == "Industrial":
            dx = sd["p"]    - ind_med_p
            dy = sd["perp"] - ind_med_perp
            mag = max(np.sqrt(dx**2 + dy**2), 1e-6)
            lx = sd["p"]    + dx/mag * 0.09
            ly = sd["perp"] + dy/mag * 0.07
        elif lf == "H-G":
            lx = sd["p"] - 0.07
            ly = sd["perp"] + 0.02
        else:
            lx = sd["p"] + (0.05 if sd["p"] < 0.5 else -0.05)
            ly = sd["perp"] + 0.03
        label_items.append(dict(x=sd["p"], y=sd["perp"],
                                 lx=lx, ly=ly, label=short, lf=lf))

    _repel_labels(label_items, min_dx=0.055, min_dy=0.030, n_iter=300)

    placed = []
    for item in label_items:
        dist_moved = np.sqrt((item["lx"]-item["x"])**2
                              + (item["ly"]-item["y"])**2)
        placed.append(ax.annotate(
            item["label"],
            xy=(item["x"], item["y"]),
            xytext=(item["lx"], item["ly"]),
            fontsize=FS-2, ha="center", va="center",
            color="black",
            arrowprops=(dict(arrowstyle="-", color="#aaaaaa", lw=0.5,
                             connectionstyle="arc3,rad=0.05")
                        if dist_moved > 0.02 else None),
            zorder=6, annotation_clip=True))

    def _keep_fitting_labels(_fig=fig, _ax=ax, _placed=tuple(placed)):
        """Keep labels down the priority list until they stop fitting.

        Runs after the layout has settled, because whether two labels collide
        depends on the final panel size and type size, which are not known while
        the figure is being built. Walks the list in priority order and hides
        anything whose box touches a box already kept, or that has drifted
        outside the panel.
        """
        try:
            _fig.canvas.draw()
            _r = _fig.canvas.get_renderer()
            _panel = _ax.get_window_extent(renderer=_r)
            # Seed the obstacles with the key and the title. A label that lands
            # on the legend is as unreadable as one that lands on another
            # label, and the legend moves after the labels are placed, so it
            # has to be measured here rather than avoided when they are laid
            # out. "Lloyd-PriceJ (USA)" was sitting on top of "Industrial".
            kept, dropped = [], 0
            _leg = _ax.get_legend()
            if _leg is not None and _leg.get_visible():
                try:
                    kept.append(_leg.get_window_extent(renderer=_r))
                except Exception:
                    pass
            try:
                if _ax.title.get_text().strip():
                    kept.append(_ax.title.get_window_extent(renderer=_r))
            except Exception:
                pass
            for _a in _placed:
                _b = _a.get_window_extent(renderer=_r)
                _off = (_b.x0 < _panel.x0 or _b.x1 > _panel.x1 or
                        _b.y0 < _panel.y0 or _b.y1 > _panel.y1)
                _hit = any(not (_b.x1 <= k.x0 or _b.x0 >= k.x1 or
                                _b.y1 <= k.y0 or _b.y0 >= k.y1) for k in kept)
                if _off or _hit:
                    _a.set_visible(False)
                    if _a.arrow_patch is not None:
                        _a.arrow_patch.set_visible(False)
                    dropped += 1
                else:
                    kept.append(_b)
            print(f"    [labels] {len(kept)} study labels kept, "
                  f"{dropped} dropped for want of room")
        except Exception as exc:
            print(f"    [labels] could not fit study labels: {exc}")

    add_post_layout_hook(fig, _keep_fitting_labels)

    # Reference lines
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.35)
    ax.axvline(0, color=GROUP_COLORS["H-G"],       lw=1.0, ls="--", alpha=0.5)
    ax.axvline(1, color=GROUP_COLORS["Industrial"], lw=1.0, ls="--", alpha=0.5)

    all_x = [sd["p"]    for sd in study_data.values()]
    all_y = [sd["perp"] for sd in study_data.values()]
    ax.set_xlim(min(-0.22, min(all_x)-0.18), max(1.25, max(all_x)+0.18))
    ax.set_ylim(min(-0.30, min(all_y)-0.08), max(0.75, max(all_y)+0.08))

    from matplotlib.lines import Line2D as _L2D
    lf_patches = [
        mpatches.Patch(
            facecolor=(IND_DARK if lf == "Industrial"
                        else GROUP_COLORS.get(lf, "gray")),
            edgecolor="white", lw=0.5, label=_display(lf))
        for lf in LIFESTYLE_ORDER if any(k[1] == lf for k in study_data)
    ]
    size_handles = [
        _L2D([0],[0], marker="o", color="w", markerfacecolor="gray",
             markersize=ms, markeredgecolor="black", markeredgewidth=0.3,
             label=f"n = {ns}")
        for ms, ns in [(2.6, 25), (4.0, 60), (5.7, 120), (8.2, 250)]
    ]
    # Top right: the Industrial cluster sits lower right and the labelled
    # non-Industrial studies run up the left, so this is the free corner.
    _leg = ax.legend(handles=lf_patches + size_handles,
                     fontsize=FS-1, loc="upper right",
                     title="Lifestyle  /  size sqrt(n)",
                     title_fontsize=FS-1, framealpha=0.9)
    _leg._fixed_loc = True
    ax.set_xlabel("Position along Hunter-Gatherer to Industrial axis",
                  fontsize=FS)
    ax.set_ylabel("Off-axis displacement", fontsize=FS)
    # One line. The second line described which studies get a label and how the
    # labels are placed -- method, not finding, and caption material. It ran the
    # full width of the figure, which is what grew the saved canvas past the
    # page and took the printed type to 4.3 pt.
    # No title on the figure: this one is an axes title, so the figure-level
    # strip never saw it. It is recorded in tools/caption_debt.csv instead.
    fig._margin_overrides = dict(top=0.91)
    png_path = os.path.join(outdir,
                             f"axis_angle_v2_study_centroids_{spec_tag}.png")
    try:
        savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    except Exception:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")



    return png_path


def _plot_point_cloud(samples_df, centroid_df, spec_tag, outdir, ref):
    """Wrapper: produce both main-text and supplement point-cloud figures."""
    _plot_pointcloud_main(samples_df, spec_tag, outdir, ref)
    _plot_pointcloud_supp(samples_df, spec_tag, outdir, ref)


# == Pipeline entry point =====================================================

def run_axis_angle(data, stats,
                   n_bins=N_BINS_DEFAULT, min_n_hg=MIN_N_HG_DEFAULT,
                   min_n_other=MIN_N_OTHER_DEFAULT, n_boot=N_BOOT_DEFAULT,
                   specs=None):
    from . import config
    ref    = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")
    os.makedirs(outdir, exist_ok=True)
    if specs is None:
        from .detection import DetectionSpec
        specs = [DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
                 DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]
    print(f"\n{'='*60}\n  Extinction-axis angle analysis [{ref}]")
    print(f"  (per-bin Jaccard PCoA, quantile bins)\n{'='*60}")
    for spec in specs:
        _run_one_spec(data, spec, n_bins=n_bins, min_n_hg=min_n_hg,
                      min_n_other=min_n_other, n_boot=n_boot,
                      outdir=outdir, ref=ref, stats=stats)


# == Standalone CLI ===========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",         default="UHGG")
    parser.add_argument("--spec",        default="10pct")
    parser.add_argument("--n-bins",      type=int, default=N_BINS_DEFAULT)
    parser.add_argument("--min-n-hg",    type=int, default=MIN_N_HG_DEFAULT)
    parser.add_argument("--min-n-other", type=int, default=MIN_N_OTHER_DEFAULT)
    parser.add_argument("--n-boot",      type=int, default=N_BOOT_DEFAULT)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from persistence.loading import load_reference
    from persistence.stats_report import StatsReport
    from persistence.reinstate_secondary import reinstate_distinct_secondaries
    from persistence import config
    from persistence.detection import DetectionSpec

    stats = StatsReport(args.ref)
    data  = load_reference(args.ref)
    data  = reinstate_distinct_secondaries(data, stats)
    spec_map = {s.short_label: s for s in [
        DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
        DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)]}
    spec = spec_map.get(args.spec)
    if spec is None:
        print(f"Unknown spec '{args.spec}'. Available: {list(spec_map)}")
        sys.exit(1)
    outdir = config.section_dir(args.ref, "s1", "main")
    os.makedirs(outdir, exist_ok=True)
    _run_one_spec(data, spec, n_bins=args.n_bins, min_n_hg=args.min_n_hg,
                  min_n_other=args.min_n_other, n_boot=args.n_boot,
                  outdir=outdir, ref=args.ref, stats=stats)
    stats.save()
