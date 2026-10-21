"""
figure2.py  —  Standalone Figure 2 generator

Layout
------
  A  Depth KDE      (H-G / Transitional / Ind (all))
  B  Raw richness   (H-G / Transitional / Ind (all))
  C  GMW corrected  (H-G / Transitional / Ind (all))
  D  Jaccard boxplots  + pairwise significance
  E  Bray-Curtis boxplots + pairwise significance
  F  Axis-angle projection plot

Pipeline prerequisites
----------------------
    python -m persistence --ref combined --only health_richness s1_within s1_axis_angle

`--ref` below defaults to combined, which is what this figure is published
on, so this block named a run the figure does not read.

This writes:
    results/combined/section3_health/main/
        health_richness_gmw_combined_summary_lifestyle_<spec>_perSample.csv  (A/B/C)
    results/combined/section1_diversity/main/
        within_group_beta_raw_<spec>.npz
        axis_angle_v2_pointcloud_<spec>_notitle.png

Usage
-----
    python figure2.py
"""

from __future__ import annotations
import argparse, colorsys, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.image import imread
from scipy.stats import gaussian_kde, mannwhitneyu

# ── Palette (Okabe-Ito) ───────────────────────────────────────────────────────
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", pink="#CC79A7", black="#000000")

# Data keys (NPZ / CSV) use short forms; DISPLAY maps them to full labels.
DISPLAY = {
    "H-G":                 "Hunter-Gatherer",
    "Ind (all)":           "Industrial",
    "Industrial":          "Industrial",
    "Transitional":        "Agro-Pastoralist",
    "Agro-Pastoralist":    "Agro-Pastoralist",
    "Agro-Pastorialist":   "Agro-Pastoralist",
    "Non-Industrial":      "Non-Industrial",
    "Between (Ind vs NI)": "Between\n(Industrial vs\nNon-Industrial)",
}

GROUP_COLORS = {
    "H-G":          OI["vermillion"],
    "Transitional": OI["yellow"],
    "Ind (all)":    OI["sky"],
    "Industrial":   OI["sky"],
    "Non-Industrial": OI["orange"],
    "Between (Ind vs NI)": "#888888",
}

LIFESTYLE_ORDER = ["H-G", "Transitional", "Ind (all)"]
BETA_ORDER      = ["Between (Ind vs NI)", "Industrial", "Transitional", "H-G"]

FS      = 17

# Enforce uniform font throughout — overrides seaborn/matplotlib defaults
matplotlib.rcParams.update({
    "font.size":        FS,
    "axes.titlesize":   FS,
    "axes.labelsize":   FS,
    "xtick.labelsize":  FS,
    "ytick.labelsize":  FS,
    "legend.fontsize":  FS,
    "font.family":      "sans-serif",
})


# ══════════════════════════════════════════════════════════════════════════════
#  Micro-helpers
# ══════════════════════════════════════════════════════════════════════════════

def _star(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def _rank_biserial(u, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U."""
    return float(1 - 2*u / (n1*n2))

def _style(ax):
    ax.grid(False); ax.set_axisbelow(False)
    ax.tick_params(axis="both", labelsize=FS, length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)

# Delegates to config, which is the single list of display-time renames
# (PLOT_TEXT_REPLACEMENTS). This used to be a hard-coded local copy that knew
# only about Agro-Pastoralist/Transitional, so it silently missed every rename
# added since -- Fatty Liver to MASLD, Obese to Obesity, the lifestyle
# sentence casing. The fallback is for running this script standalone, with
# the package absent.
# The repo root goes on the path BEFORE the import, because these scripts are
# run by path -- figure_runner launches them as subprocesses and the README
# shows the same -- so sys.path[0] is `figures/` and `persistence` is not
# importable from it. Without this the except branch ran instead, silently, and
# that branch was a second rename table which had drifted from
# config.PLOT_TEXT_REPLACEMENTS -- it mapped Transitional to the title-case
# "Agro-Pastoralist" and knew nothing of Obese, Fatty Liver or the lifestyle
# sentence casing. figure_runner sets PYTHONPATH on the subprocesses it
# launches, so the pipeline's own runs bound config and were correct; the
# exposure was any other way of running these, which is what a reviewer
# cloning the repo does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from persistence.config import display_text as _display_text
except Exception as _exc:                          # package genuinely absent
    # Deliberately not a second rename table: a local copy drifts from
    # config.PLOT_TEXT_REPLACEMENTS and then disagrees with the rest of the
    # figure set, which is worse than an obviously raw key. Say so, loudly,
    # and pass the text through untouched.
    print(f"  [warn] persistence.config unavailable ({_exc}); figure text "
          f"will carry internal keys, not published labels", file=sys.stderr)

    def _display_text(text):
        return str(text)

def _apply_display_text(fig):
    for artist in fig.findobj(match=matplotlib.text.Text):
        old = artist.get_text()
        new = _display_text(old)
        if new != old:
            artist.set_text(new)
    return fig

def _bgcol(h, desat=0.35, bright=1.20):
    r,g,b = [int(h[i:i+2],16)/255 for i in (1,3,5)]
    hh,s,v = colorsys.rgb_to_hsv(r,g,b)
    nr,ng,nb = colorsys.hsv_to_rgb(hh, s*desat, min(v*bright,1.0))
    return "#{:02x}{:02x}{:02x}".format(int(nr*255),int(ng*255),int(nb*255))


# ══════════════════════════════════════════════════════════════════════════════
#  File discovery
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_spec_tag(s1_dir: str, s3_dir: str, spec_arg: str) -> str:
    defaults = {"breadth_pct": "10pct", "flat_bp": "300kb"}
    for tag in [spec_arg, defaults.get(spec_arg, spec_arg)]:
        probes = [
            os.path.join(s1_dir, f"within_group_beta_raw_{tag}.npz"),
            os.path.join(s1_dir, f"within_group_beta_boxplot_{tag}_notitle.png"),
            os.path.join(s1_dir, f"axis_angle_v2_pointcloud_{tag}_notitle.png"),
            glob.glob(os.path.join(
                s3_dir, f"health_richness_gmw_*_summary_lifestyle_{tag}_notitle.png")),
        ]
        if any((bool(p) if isinstance(p, list) else os.path.exists(p))
               for p in probes):
            return tag
    # Scan
    found = glob.glob(os.path.join(s1_dir, "within_group_beta_raw_*.npz"))
    if found:
        tag = os.path.basename(found[0]).replace(
            "within_group_beta_raw_","").replace(".npz","")
        print(f"  [INFO] Auto-detected spec_tag: {tag}")
        return tag
    print(f"  [WARN] Could not resolve spec_tag from '{spec_arg}'")
    return spec_arg


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_per_sample(s3_dir: str, ref: str, spec_tag: str,
                    explicit_path: str | None = None):
    """
    Per-sample richness/weight/depth for summary_lifestyle groups
    (H-G / Transitional / Ind (all)).
    Tries explicit_path first, then the standard pipeline output location.
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.append(os.path.join(
        s3_dir,
        f"health_richness_gmw_{ref}_summary_lifestyle_{spec_tag}_perSample.csv"))
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Normalise legacy "Transitional" label if present
            if "Group" in df.columns:
                df["Group"] = df["Group"].replace("Transitional", "Transitional")
            return df
    print(f"  [WARN] Per-sample CSV not found. Tried:")
    for p in candidates:
        print(f"           {p}")
    print("         Run:  python -m persistence --ref "
          f"{ref} --only health_richness")
    print("         Or pass an existing file with --ps-csv <path>")
    return None


def load_beta_raw(s1_dir: str, spec_tag: str, ref: str = "UHGG") -> dict | None:
    """
    Load raw distance arrays from the patched pipeline output.
    Returns {lifestyle_metric: np.ndarray}, e.g. {"H-G_jaccard": array, ...}
    """
    path = os.path.join(s1_dir, f"within_group_beta_raw_{spec_tag}.npz")
    if not os.path.exists(path):
        print(f"  [WARN] Raw beta arrays not found: {path}")
        print("         Run:  python -m persistence --ref "
              f"{ref} --only s1_within")
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_beta_summary(s1_dir: str, spec_tag: str) -> pd.DataFrame | None:
    path = os.path.join(s1_dir, f"within_group_beta_{spec_tag}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Beta summary CSV not found: {path}")
        return None
    return pd.read_csv(path)


def load_axis_angle_samples(s1_dir: str, spec_tag: str) -> pd.DataFrame | None:
    path = os.path.join(s1_dir, f"axis_angle_v2_samples_{spec_tag}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Axis-angle samples CSV not found: {path}")
        return None
    return pd.read_csv(path)


def load_pcoa(s1_dir: str):
    path = os.path.join(s1_dir, "beta_diversity_pcoa_matched.csv")
    if not os.path.exists(path):
        print(f"  [WARN] PCoA CSV not found: {path}")
        return None, None, None
    df = pd.read_csv(path, index_col=0)
    var1, var2 = None, None

    # Strategy 1: companion JSON written by patched pipeline
    json_path = path.replace(".csv", "_variance.json")
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path) as f:
                jd = json.load(f)
            var1 = float(jd["pcoa_var1"])
            var2 = float(jd["pcoa_var2"])
        except Exception as e:
            print(f"  [WARN] Could not parse {json_path}: {e}")

    # Strategy 2: stats_report.csv (lives at results/<ref>/stats_report.csv,
    # i.e. three levels up from section1_diversity/main/)
    if var1 is None:
        for stats_path in [
            os.path.join(s1_dir, "stats_report.csv"),
            os.path.join(s1_dir, "..", "stats_report.csv"),
            os.path.join(s1_dir, "..", "..", "stats_report.csv"),
            os.path.join(s1_dir, "..", "..", "..", "stats_report.csv"),
        ]:
            stats_path = os.path.normpath(stats_path)
            if not os.path.exists(stats_path):
                continue
            try:
                st = pd.read_csv(stats_path, low_memory=False)
                tag_col = "tag" if "tag" in st.columns else st.columns[0]
                val_col = "value" if "value" in st.columns else st.columns[4]
                for _, row in st.iterrows():
                    tag = str(row.get(tag_col, ""))
                    val = str(row.get(val_col, "")).replace("%", "").strip()
                    if "pcoa_var_axis1" in tag and var1 is None:
                        try: var1 = float(val)
                        except ValueError: pass
                    if "pcoa_var_axis2" in tag and var2 is None:
                        try: var2 = float(val)
                        except ValueError: pass
                if var1 is not None:
                    break
            except Exception:
                pass

    # Strategy 3: eigenvalue reconstruction — PCo_i = eigenvector_i * sqrt(λ_i)
    # so sum(PCo_i²) = λ_i, and var_explained_i = λ_i / Σλ_j.
    # This is exact (not an approximation) given the PCoA coordinate definition.
    if var1 is None:
        pco_cols = [c for c in df.columns if c.startswith("PCo")]
        if len(pco_cols) >= 2:
            ev    = df[pco_cols].pow(2).sum()   # λ per axis
            total = ev.sum()
            if total > 0:
                var1 = 100.0 * float(ev[pco_cols[0]]) / total
                var2 = 100.0 * float(ev[pco_cols[1]]) / total
                print(f"  [INFO] PCoA variance (eigenvalue reconstruction): "
                      f"PCo1={var1:.1f}%, PCo2={var2:.1f}%")
                print(f"         (If wrong, re-run pipeline --only s1 to generate "
                      f"beta_diversity_pcoa_matched_variance.json)")

    # Clean column names to plain PCo1/PCo2
    df.columns = [c.split("(")[0].strip() if c.startswith("PCo") else c
                  for c in df.columns]
    return df, var1, var2


def resolve_axis_angle_image(s1_dir: str, spec_tag: str,
                             explicit_path: str | None = None) -> str | None:
    """Find the title-less large axis-angle point-cloud panel for Figure 2."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    stem = os.path.join(s1_dir, f"axis_angle_v2_pointcloud_{spec_tag}")
    candidates.extend([
        # Current naming first. The preset suffixes are kept as a fallback so
        # this still finds a panel that has not been regenerated yet.
        f"{stem}_notitle.png",
        f"{stem}.png",
        f"{stem}_large_notitle.png",
        f"{stem}_medium_notitle.png",
        f"{stem}_large.png",
        f"{stem}_medium.png",
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    print("  [WARN] Axis-angle image not found. Tried:")
    for path in candidates:
        print(f"           {path}")
    return None


def resolve_gmw_image(s3_dir: str, ref: str, spec_tag: str,
                      explicit_path: str | None = None) -> str | None:
    """Find the title-less large 3-way GMW richness panel strip."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    stem = os.path.join(
        s3_dir, f"health_richness_gmw_{ref}_summary_lifestyle_{spec_tag}")
    candidates.extend([
        # Current naming first; preset suffixes kept as a fallback so this
        # still finds a panel that has not been regenerated yet.
        f"{stem}_notitle.png",
        f"{stem}.png",
        f"{stem}_large_notitle.png",
        f"{stem}_medium_notitle.png",
        f"{stem}_large.png",
        f"{stem}_medium.png",
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    print("  [WARN] 3-way GMW image not found. Tried:")
    for path in candidates:
        print(f"           {path}")
    return None


def resolve_beta_boxplot_image(s1_dir: str, spec_tag: str,
                               explicit_path: str | None = None) -> str | None:
    """Find the title-less large Jaccard/Bray-Curtis beta boxplot strip."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    stem = os.path.join(s1_dir, f"within_group_beta_boxplot_{spec_tag}")
    candidates.extend([
        # Current naming first; preset suffixes kept as a fallback so this
        # still finds a panel that has not been regenerated yet.
        f"{stem}_notitle.png",
        f"{stem}.png",
        f"{stem}_large_notitle.png",
        f"{stem}_medium_notitle.png",
        f"{stem}_large.png",
        f"{stem}_medium.png",
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    print("  [WARN] Beta boxplot image not found. Tried:")
    for path in candidates:
        print(f"           {path}")
    return None


def _show_image(ax, image_path: str | None, crop: tuple[float, float, float, float] | None = None,
                missing: str = "Missing image"):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if image_path is None:
        ax.text(0.5, 0.5, missing, transform=ax.transAxes,
                ha="center", va="center", fontsize=FS-2, color="#888",
                style="italic", multialignment="center")
        return
    img = imread(image_path)
    if crop is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = crop
        img = img[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
    ax.imshow(img)
    ax.set_aspect("auto")


def _label_image_axis(ax, xlabel: str | None = None, ylabel: str | None = None):
    """Add figure-level axis labels to imported raster panels."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS, labelpad=10)


# ══════════════════════════════════════════════════════════════════════════════
#  Panel A — depth KDE
# ══════════════════════════════════════════════════════════════════════════════

def draw_A(ax, ps_df, show_title: bool = True):
    _style(ax)
    if show_title:
        ax.set_title("Sequencing depth distributions by lifestyle",
                     fontsize=FS, fontweight="bold", pad=4)

    if ps_df is None or "log10_depth" not in ps_df.columns:
        ax.text(0.5, 0.5, "Run pipeline with\n--only health_richness",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS-2, color="#888", style="italic",
                multialignment="center")
        ax.set_ylabel("Normalised density", fontsize=FS)
        return

    NICE_GB  = [0.03, 0.1, 0.3, 1, 3, 10, 30]
    TICK_LOG = [np.log10(v*1e9) for v in NICE_GB]
    ls_map   = {"H-G": "-", "Transitional": "--", "Ind (all)": ":"}
    order    = [g for g in LIFESTYLE_ORDER if g in ps_df["Group"].values]

    all_ld   = ps_df["log10_depth"].dropna().values
    lo_x, hi_x = all_ld.min()-0.25, all_ld.max()+0.25
    dg = np.linspace(lo_x, hi_x, 400)

    geom = np.ones(len(dg))
    for grp in order:
        ld  = ps_df.loc[ps_df["Group"]==grp, "log10_depth"].dropna().values
        col = GROUP_COLORS.get(grp, OI["pink"])
        if len(ld) < 5: continue
        k  = gaussian_kde(ld, bw_method="silverman")
        yv = k(dg); yv /= max(yv.max(), 1e-300)
        ax.plot(dg, yv, color=col, ls=ls_map.get(grp,"-"), lw=2.2, label=DISPLAY.get(grp, grp))
        ax.fill_between(dg, yv, alpha=0.10, color=col)
        geom *= np.maximum(k(dg), 1e-300)

    geom = geom**(1/max(len(order), 1))
    gn   = geom / max(geom.max(), 1e-300)
    ax.plot(dg, gn, color="black", lw=2.2, label="GMW target")
    ax.fill_between(dg, gn, alpha=0.08, color="black")

    ticks = [(p, str(v)) for p, v in zip(TICK_LOG, NICE_GB) if lo_x<=p<=hi_x]
    if len(ticks) < 3:
        ticks = list(zip(TICK_LOG, [str(v) for v in NICE_GB]))[:5]
    ax.set_xticks([p for p,_ in ticks])
    ax.set_xticklabels([l for _,l in ticks], fontsize=FS)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0","0.5","1.0"], fontsize=FS)
    ax.set_xlabel("Sequencing depth (Gb)", fontsize=FS)
    ax.set_ylabel("Normalised density",    fontsize=FS)
    ax.legend(fontsize=FS, loc="upper left", framealpha=0.7,
              edgecolor="none", handlelength=2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Panels B + C — richness violins
# ══════════════════════════════════════════════════════════════════════════════

def draw_violin(ax, ps_df, corrected: bool, title: str, rng,
                show_title: bool = True):
    try: import seaborn as sns
    except ImportError: sns = None

    _style(ax)
    if show_title:
        ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)
    order   = [g for g in LIFESTYLE_ORDER if g in ps_df["Group"].values]
    palette = {g: GROUP_COLORS.get(g, OI["pink"]) for g in order}

    if corrected:
        rows = []
        for g in order:
            sub = ps_df[ps_df["Group"]==g]
            r   = sub["richness"].values.astype(float)
            w   = np.maximum(sub["weight"].values.astype(float), 0.0)
            w_s = w / max(w.sum(), 1e-300)
            idx = rng.choice(len(r), size=len(r), replace=True, p=w_s)
            for v in r[idx]: rows.append({"Group": g, "richness": v})
        vdf = pd.DataFrame(rows)
    else:
        vdf = ps_df[["Group","richness"]].copy()

    if sns is not None:
        sns.violinplot(data=vdf, x="Group", y="richness",
                       order=order, hue="Group", hue_order=order,
                       palette=palette, legend=False, inner="box",
                       cut=0, linewidth=0.8, ax=ax, density_norm="width")

    for i, g in enumerate(order):
        sub = ps_df[ps_df["Group"]==g]
        r = sub["richness"].values.astype(float)
        w_raw = np.maximum(sub["weight"].values.astype(float), 0.0)
        w_sum = max(w_raw.sum(), 1e-300)
        w = w_raw / w_sum
        mu = float(np.dot(w, r)) if corrected else float(r.mean())
        n = len(r)

        if corrected:
            # Weighted SEM for the GMW-corrected mean.  The per-sample
            # weights are mean-normalized, so using normalized weights here
            # would report the weighted SD instead of the SEM.
            wsem = float(np.sqrt((w_raw * (r - mu)**2).sum()) / w_sum) if n > 1 else 0.0
            ax.errorbar(i, mu, yerr=[[wsem], [wsem]],
                        fmt="D", color="white", ecolor="black",
                        elinewidth=2.0, capsize=5, markersize=8, zorder=6)
            ax.text(i+0.06, mu, f"{mu:.0f}±{wsem:.0f}",
                    va="center", fontsize=FS, fontweight="bold")
        else:
            sem = r.std(ddof=1)/np.sqrt(n) if n > 1 else 0.0
            ax.errorbar(i, mu, yerr=[[sem],[sem]],
                        fmt="D", color="white", ecolor="black",
                        elinewidth=2.0, capsize=5, markersize=8, zorder=6)
            ax.text(i+0.06, mu, f"{mu:.0f}±{sem:.0f}",
                    va="center", fontsize=FS, fontweight="bold")

    ax.set_xticks(range(len(order)))
    if corrected:
        xlabels = []
        for g in order:
            w_g = np.maximum(
                ps_df.loc[ps_df["Group"]==g, "weight"].values.astype(float), 0.0)
            ess = float(w_g.sum()**2 / max((w_g**2).sum(), 1e-300))
            label = DISPLAY.get(g, g).replace("Hunter-Gatherer", "Hunter-\nGatherer")
            xlabels.append(f"{label}\n(ESS={ess:.0f})")
        ax.set_xticklabels(xlabels, fontsize=FS)
        if show_title:
            ax.set_title("GMW corrected", fontsize=FS, fontweight="bold", pad=4)
    else:
        xlabels = []
        for g in order:
            label = DISPLAY.get(g, g).replace("Hunter-Gatherer", "Hunter-\nGatherer")
            xlabels.append(f"{label}\n(n={len(ps_df[ps_df['Group'] == g]):,})")
        ax.set_xticklabels(xlabels, fontsize=FS)
    ax.set_ylabel("# SGBs per sample", fontsize=FS)
    ax.set_xlabel(""); ax.set_ylim(bottom=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Panels D + E — density curves
# ══════════════════════════════════════════════════════════════════════════════

def draw_density(ax, beta_raw: dict, metric: str, title: str):
    """
    Overlaid KDE density of pairwise distances for each lifestyle + between.
    Vertical dashed lines mark the group medians.
    """
    _style(ax)
    ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)

    ls_map = {"H-G": "-", "Transitional": "--",
              "Industrial": "-", "Non-Industrial": "--",
              "Between (Ind vs NI)": ":"}

    # Single-line Between label for legend (no \n — fits legend box)
    DE_DISPLAY = {**DISPLAY,
                  "Between (Ind vs NI)": "Between (Industrial vs\nNon-Industrial)"}

    plot_order = [g for g in ["H-G","Transitional","Industrial",
                              "Between (Ind vs NI)"]
                  if f"{g}_{metric}" in beta_raw]

    for grp in plot_order:
        key = f"{grp}_{metric}"
        if key not in beta_raw: continue
        vals = beta_raw[key]
        col  = GROUP_COLORS.get(grp, OI["pink"])
        ls   = ls_map.get(grp, "-")
        n_samp  = max(2, int(round((1 + (1 + 8*len(vals))**0.5) / 2)))
        n_label = f"n={n_samp:,}"
        try:
            k  = gaussian_kde(vals, bw_method=0.05)
            xg = np.linspace(0, 1, 400)
            yg = k(xg)
            ax.plot(xg, yg, color=col, ls=ls, lw=1.8,
                    label=f"{DE_DISPLAY.get(grp,grp)} ({n_label})")
            ax.fill_between(xg, yg, alpha=0.06, color=col)
        except Exception:
            pass
        med = float(np.median(vals))
        ax.axvline(med, color=col, lw=1.0, ls=":", alpha=0.60)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Pairwise distance", fontsize=FS)
    ax.set_ylabel("Density", fontsize=FS)
    ax.legend(fontsize=FS, loc="upper left", framealpha=0.7,
              edgecolor="none", handlelength=2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Panel F — PCoA scatter
# ══════════════════════════════════════════════════════════════════════════════

def draw_F(ax, pcoa_df, var1, var2, max_pts=4000):
    _style(ax)
    title = "Beta Diversity: PCoA on Bray-Curtis\n(1:1 depth-matched, Industrial vs Non-Industrial)"
    ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)

    if pcoa_df is None:
        ax.text(0.5, 0.5, "Run pipeline with\n--only s1",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS-2, color="#888", style="italic",
                multialignment="center")
        return

    rng = np.random.default_rng(99)
    for lf in ["H-G", "Transitional", "Industrial"]:
        sub = pcoa_df[pcoa_df["Lifestyle"]==lf]
        if sub.empty:
            # Fall back to Group column
            g_map = {"Industrial":"Industrial","Non-Industrial":"Non-Industrial"}
            sub = pcoa_df[pcoa_df.get("Group","") == g_map.get(lf, lf)]
        if sub.empty: continue
        if len(sub) > max_pts:
            sub = sub.iloc[rng.choice(len(sub), max_pts, replace=False)]
        col  = GROUP_COLORS.get(lf, OI["pink"])
        n_lbl = f"n={len(pcoa_df[pcoa_df['Lifestyle']==lf]):,}"
        ax.scatter(sub["PCo1"], sub["PCo2"], s=3, alpha=0.35,
                   color=col, rasterized=True, label=f"{DISPLAY.get(lf,lf)} ({n_lbl})")

    x_lbl = f"PCo1 ({var1:.1f}%)" if var1 is not None else "PCo1"
    y_lbl = f"PCo2 ({var2:.1f}%)" if var2 is not None else "PCo2"
    ax.set_xlabel(x_lbl, fontsize=FS)
    ax.set_ylabel(y_lbl, fontsize=FS)
    ax.legend(fontsize=FS, loc="upper right", framealpha=0.7,
              edgecolor="none", markerscale=3)
    ax.axhline(0, color="black", lw=0.4, ls=":", alpha=0.4)
    ax.axvline(0, color="black", lw=0.4, ls=":", alpha=0.4)


def draw_axis_angle_image(ax, image_path: str | None):
    _show_image(ax, image_path, missing="Run pipeline --only s1_axis_angle")


def _axis_kde_contours(ax, xv, yv, color):
    if len(xv) < 20:
        return
    try:
        kde = gaussian_kde(np.vstack([xv, yv]), bw_method="scott")
        xi = np.linspace(xv.min() - 0.08, xv.max() + 0.08, 160)
        yi = np.linspace(yv.min() - 0.06, yv.max() + 0.06, 160)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        z = Zi.ravel()
        order = np.argsort(z)[::-1]
        z_cum = np.cumsum(z[order]) / max(z.sum(), 1e-300)
        iso90 = z[order[np.searchsorted(z_cum, 0.90)]]
        iso50 = z[order[np.searchsorted(z_cum, 0.50)]]
        ax.contourf(Xi, Yi, Zi, levels=[iso90, Zi.max()],
                    colors=[color], alpha=0.08, zorder=1)
        ax.contour(Xi, Yi, Zi, levels=[iso90], colors=[color],
                   linewidths=[0.8], linestyles=["--"],
                   alpha=0.45, zorder=3)
        ax.contour(Xi, Yi, Zi, levels=[iso50], colors=[color],
                   linewidths=[1.2], linestyles=["-"],
                   alpha=0.75, zorder=3)
    except Exception:
        pass


def draw_axis_angle_native(ax, samples_df: pd.DataFrame | None,
                           image_fallback: str | None = None):
    """Draw panel F natively so labels/fonts match the rest of Figure 2."""
    if samples_df is None or samples_df.empty:
        draw_axis_angle_image(ax, image_fallback)
        return

    _style(ax)
    ann_fs = 17
    axis_fs = 17
    legend_fs = 17
    centroid_of = {}
    for lf in ["H-G", "Transitional", "Industrial"]:
        sub = samples_df[samples_df["Lifestyle"] == lf]
        if sub.empty:
            continue
        xv = sub["p"].to_numpy(dtype=float)
        yv = sub["perp_signed"].to_numpy(dtype=float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]
        if len(xv) < 5:
            continue

        color = GROUP_COLORS.get(lf, OI["pink"])
        centroid_of[lf] = (float(np.median(xv)), float(np.median(yv)))
        dot_sz = {"H-G": 12, "Transitional": 8, "Industrial": 8}.get(lf, 8)
        dot_alpha = {"H-G": 0.45, "Transitional": 0.16,
                     "Industrial": 0.16}.get(lf, 0.18)
        ax.scatter(xv, yv, c=color, s=dot_sz, alpha=dot_alpha,
                   edgecolors="none", rasterized=True, zorder=2)
        _axis_kde_contours(ax, xv, yv, color)

        cx, cy = centroid_of[lf]
        marker = {"H-G": "^", "Transitional": "*", "Industrial": "D"}[lf]
        size = {"H-G": 160, "Transitional": 250, "Industrial": 160}[lf]
        ax.scatter(cx, cy, c=color, s=size, marker=marker,
                   edgecolors="black", lw=1.1, zorder=7,
                   label=f"{DISPLAY.get(lf, lf)} (n={len(xv):,})")

    c_hg = centroid_of.get("H-G", (0.02, 0.00))
    c_tr = centroid_of.get("Transitional", (0.45, 0.30))
    c_ind = centroid_of.get("Industrial", (1.00, 0.00))

    def _arr(xy, xytext, color, lw=1.5, ls="solid"):
        ax.annotate("", xy=xy, xycoords="data",
                    xytext=xytext, textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=14,
                                    linestyle=ls),
                    zorder=8)

    ofs = c_tr[1] * 0.08
    _arr((c_ind[0] * 0.97, ofs), (c_hg[0] + 0.02, ofs),
         "#222222", lw=1.8)
    ax.text(0.5 * (c_hg[0] + c_ind[0]), ofs + c_tr[1] * 0.07,
            "Hunter-Gatherer -> Industrial  (VANISH / BloSSUM)",
            ha="center", va="bottom", fontsize=ann_fs,
            color="#222222", fontweight="bold",
            fontstyle="italic", zorder=9)

    _arr((c_tr[0] * 0.92, c_tr[1] * 0.92), (c_hg[0] + 0.01, 0.00),
         GROUP_COLORS["H-G"], lw=1.5, ls="dashed")
    ax.text(c_tr[0] * 0.50, c_tr[1] * 0.52, "Agriculturalization axis",
            ha="center", va="bottom", fontsize=ann_fs,
            color=GROUP_COLORS["H-G"], fontweight="bold",
            fontstyle="italic", zorder=9)

    _arr((c_ind[0] * 0.97, 0.00), (c_tr[0] + 0.03, c_tr[1] * 0.88),
         GROUP_COLORS["Industrial"], lw=1.5, ls="dashed")
    ax.text(c_tr[0] + 0.08, c_tr[1] * 0.45,
            "Industrialisation axis\n(approx. VANISH+ / BloSSUM+)",
            ha="left", va="center", fontsize=ann_fs,
            color=GROUP_COLORS["Industrial"], fontweight="bold",
            fontstyle="italic", zorder=9)

    ax.set_xlabel("Fractional position along Hunter-Gatherer -> Industrial axis",
                  fontsize=axis_fs)
    ax.set_ylabel("Off-axis displacement (positive = toward Transitional side)",
                  fontsize=axis_fs)
    ax.tick_params(axis="both", labelsize=FS)
    ax.legend(fontsize=legend_fs, loc="upper right", framealpha=0.88,
              edgecolor="#cccccc", markerscale=0.95,
              borderpad=0.35, labelspacing=0.3, handletextpad=0.35)
    ax.grid(False)

    xs = samples_df["p"].to_numpy(dtype=float)
    ys = samples_df["perp_signed"].to_numpy(dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.any():
        ax.set_xlim(max(-0.28, np.percentile(xs[ok], 0.3) - 0.08),
                    min(1.35, np.percentile(xs[ok], 99.7) + 0.08))
        ax.set_ylim(max(-0.85, np.percentile(ys[ok], 0.3) - 0.08),
                    min(0.95, np.percentile(ys[ok], 99.7) + 0.08))


# ══════════════════════════════════════════════════════════════════════════════
#  Panels G + H — boxplots with significance brackets
#
# Significance style: brackets are drawn as HORIZONTAL lines (parallel to the
# brace) with short vertical ticks at each end, plus the star/r annotation
# placed to the RIGHT of the bracket — all in the area to the right of the
# boxplots.  This avoids the radiating-outward style that takes up too much
# horizontal space.
# ══════════════════════════════════════════════════════════════════════════════

def _mw_stats(a, b, rng=None, max_n=500):
    """
    Mann-Whitney U + rank-biserial r, subsampled to ~n_samples distances.

    Pairwise distance arrays have O(n²) entries but only n independent
    samples.  Testing all pairs gives absurdly small p-values even for tiny
    effects.  We subsample to min(max_n, sqrt(2*n_pairs)) distances per
    group so that effective n reflects the actual sample count.
    """
    if len(a) < 2 or len(b) < 2:
        return np.nan, "", np.nan
    if rng is None:
        rng = np.random.default_rng(42)

    def _n_eff(n_pairs):
        return max(20, min(max_n, int(0.5 + (0.25 + 2*n_pairs)**0.5)))

    na = _n_eff(len(a)); nb = _n_eff(len(b))
    sa = a[rng.choice(len(a), na, replace=False)]
    sb = b[rng.choice(len(b), nb, replace=False)]

    u, p = mannwhitneyu(sa, sb, alternative="two-sided")
    r    = _rank_biserial(u, na, nb)
    return p, _star(p), r


def draw_boxplot_with_sig(ax, beta_raw: dict, metric: str, title: str):
    """
    Horizontal boxplots for each lifestyle/group (y-axis) with pairwise
    significance brackets on the right side.

    Bracket style:  a short vertical line connects the two group y-positions,
    capped with a small horizontal tick, and the annotation (star + r) is
    placed to the right of the cap — all in a column whose x-position does
    not interfere with the data boxes.
    """
    _style(ax)
    ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)

    order = [g for g in BETA_ORDER if f"{g}_{metric}" in beta_raw]
    if not order:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", fontsize=FS-2, color="#888")
        return

    data_arrays = {g: beta_raw[f"{g}_{metric}"] for g in order}
    y_step = 1.28
    y_pos = {g: i * y_step for i, g in enumerate(order)}

    # Subsample for plotting to keep it fast
    rng = np.random.default_rng(42)
    MAX_PLOT = 2000
    for i, grp in enumerate(order):
        vals = data_arrays[grp]
        col  = GROUP_COLORS.get(grp, OI["pink"])
        n    = len(vals)
        if n > MAX_PLOT:
            vals_plot = rng.choice(vals, MAX_PLOT, replace=False)
        else:
            vals_plot = vals

        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        iqr  = q75 - q25
        lo_w = max(vals[vals >= q25 - 1.5*iqr].min(), vals.min())
        hi_w = min(vals[vals <= q75 + 1.5*iqr].max(), vals.max())

        y  = y_pos[grp]
        bh = 0.35   # box half-height

        # Box
        ax.fill_betweenx([y-bh, y+bh], q25, q75,
                         color=col, alpha=0.55, zorder=3)
        ax.plot([q25, q75], [y-bh, y-bh], color=col, lw=0.8, zorder=4)
        ax.plot([q25, q75], [y+bh, y+bh], color=col, lw=0.8, zorder=4)
        ax.plot([q25, q25], [y-bh, y+bh], color=col, lw=0.8, zorder=4)
        ax.plot([q75, q75], [y-bh, y+bh], color=col, lw=0.8, zorder=4)

        # Median line
        ax.plot([med, med], [y-bh, y+bh], color="black", lw=1.5, zorder=5)
        # Median dot
        ax.scatter([med], [y], color="white", s=25, zorder=6, edgecolors="black", lw=0.8)

        # Whiskers
        ax.plot([lo_w, q25], [y, y], color=col, lw=1.0, zorder=3)
        ax.plot([q75, hi_w], [y, y], color=col, lw=1.0, zorder=3)
        # Whisker caps
        for xc in [lo_w, hi_w]:
            ax.plot([xc, xc], [y-bh*0.4, y+bh*0.4], color=col, lw=0.8, zorder=3)

    def _n_samples(arr):
        """Back-calculate n_samples from n_pairs = n*(n-1)/2."""
        return max(2, int(round((1 + (1 + 8*len(arr))**0.5) / 2)))

    ax.set_yticks([y_pos[g] for g in order])
    ax.set_yticklabels(
        [f"{DISPLAY.get(g,g)}\n(n={_n_samples(data_arrays[g]):,})" for g in order],
        fontsize=FS)
    for txt in ax.get_yticklabels():
        txt.set_linespacing(1.05)
    ax.set_xlabel("Pairwise distance", fontsize=FS)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6 * y_step, (len(order) - 1) * y_step + 0.6 * y_step)

    # ── Pairwise significance brackets ───────────────────────────────
    # Sort pairs by span (innermost = shortest span first)
    mw_rng = np.random.default_rng(42)
    pairs_to_draw = []
    for i in range(len(order)):
        for j in range(i+1, len(order)):
            ga, gb = order[i], order[j]
            p, star, r = _mw_stats(data_arrays[ga], data_arrays[gb], rng=mw_rng)
            if star:
                pairs_to_draw.append((i, j, star, r, p))

    if not pairs_to_draw:
        return

    # Sort by span so innermost bracket is closest to data
    pairs_to_draw.sort(key=lambda x: x[1]-x[0])

    # Brackets to the RIGHT of x=1.0, stacked left-to-right by span.
    # Annotations rotated 90° placed to the LEFT of each bracket line
    # (between bracket and data), reading bottom-to-top.
    col_w   = 0.038           # stars only, no r= text
    x_start = 1.01
    ax.set_xlim(0, x_start + col_w * len(pairs_to_draw) + 0.012)
    ax.set_xticks(np.linspace(0, 1, 6))

    for idx, (i, j, star, r, p) in enumerate(pairs_to_draw):
        x_col  = x_start + idx * col_w
        y_i    = y_pos[order[i]]
        y_j    = y_pos[order[j]]
        tick_w = 0.010

        # Vertical connecting line
        ax.plot([x_col, x_col], [y_i, y_j],
                color="black", lw=1.0, clip_on=False, zorder=6)
        # End ticks
        for yy in (y_i, y_j):
            ax.plot([x_col-tick_w, x_col+tick_w], [yy, yy],
                    color="black", lw=1.0, clip_on=False, zorder=6)

        # Stars only — rotated 270°, to the right of the bracket line
        y_mid = (y_i + y_j) / 2.0
        sig   = star not in ("", "ns")
        label = r"$^{ns}$" if star == "ns" else star
        ax.text(x_col + tick_w * 0.09, y_mid, label,
                ha="center", va="center",
                rotation=270, rotation_mode="anchor",
                fontsize=FS,
                fontweight="bold" if sig else "normal",
                color="black" if sig else "#888888",
                clip_on=False)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_boxplot_summary_with_sig(ax, beta_summary: pd.DataFrame,
                                  metric: str, title: str):
    """Draw aligned distance boxes from within_group_beta_<spec>.csv."""
    _style(ax)
    ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)

    metric_rows = beta_summary[beta_summary["Metric"] == metric].copy()
    order = [g for g in BETA_ORDER if g in set(metric_rows["Lifestyle"])]
    if not order:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", fontsize=FS-2, color="#888")
        return

    rows = {r["Lifestyle"]: r for _, r in metric_rows.iterrows()}
    y_step = 1.28
    y_pos = {g: i * y_step for i, g in enumerate(order)}
    box_h = 0.52

    for grp in order:
        row = rows[grp]
        y = y_pos[grp]
        col = GROUP_COLORS.get(grp, OI["pink"])
        q25, med, q75 = float(row["q25"]), float(row["median"]), float(row["q75"])
        iqr = max(q75 - q25, 1e-9)
        lo_w = max(0.0, q25 - 1.5 * iqr)
        hi_w = min(1.0, q75 + 1.5 * iqr)

        ax.fill_betweenx([y - box_h/2, y + box_h/2], q25, q75,
                         color=col, alpha=0.55, zorder=3)
        ax.plot([q25, q75], [y - box_h/2, y - box_h/2], color=col, lw=1.0)
        ax.plot([q25, q75], [y + box_h/2, y + box_h/2], color=col, lw=1.0)
        ax.plot([q25, q25], [y - box_h/2, y + box_h/2], color=col, lw=1.0)
        ax.plot([q75, q75], [y - box_h/2, y + box_h/2], color=col, lw=1.0)
        ax.plot([med, med], [y - box_h/2, y + box_h/2],
                color="black", lw=1.4, zorder=5)
        ax.scatter([med], [y], color="white", s=28, zorder=6,
                   edgecolors="black", lw=0.8)
        ax.plot([lo_w, q25], [y, y], color=col, lw=1.0)
        ax.plot([q75, hi_w], [y, y], color=col, lw=1.0)
        for xc in (lo_w, hi_w):
            ax.plot([xc, xc], [y - box_h*0.22, y + box_h*0.22],
                    color=col, lw=1.0)

    ax.set_yticks([y_pos[g] for g in order])
    ax.set_yticklabels([
        f"{DISPLAY.get(g, g)}\n(n={int(rows[g]['n_samples']):,})"
        for g in order
    ], fontsize=FS)
    for txt in ax.get_yticklabels():
        txt.set_linespacing(1.05)
    ax.set_xlabel("Pairwise distance", fontsize=FS)
    ax.set_ylim(-0.6 * y_step, (len(order) - 1) * y_step + 0.6 * y_step)

    def _approx_p(ga, gb):
        a, b = rows[ga], rows[gb]
        na = max(float(a["n_samples"]), 2.0)
        nb = max(float(b["n_samples"]), 2.0)
        sda = max((float(a["q75"]) - float(a["q25"])) / 1.349, 1e-9)
        sdb = max((float(b["q75"]) - float(b["q25"])) / 1.349, 1e-9)
        se = np.sqrt(sda*sda/na + sdb*sdb/nb)
        z = abs(float(a["mean"]) - float(b["mean"])) / max(se, 1e-12)
        from scipy.stats import norm
        return 2.0 * norm.sf(z)

    pairs = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            p = _approx_p(order[i], order[j])
            pairs.append((i, j, _star(p)))
    pairs.sort(key=lambda x: x[1] - x[0])

    x_start = 1.01
    col_w = 0.045
    ax.set_xlim(0, x_start + col_w * len(pairs) + 0.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    for idx, (i, j, star) in enumerate(pairs):
        x = x_start + idx * col_w
        yi, yj = y_pos[order[i]], y_pos[order[j]]
        tick = 0.014
        ax.plot([x, x], [yi, yj], color="black", lw=1.0, clip_on=False)
        ax.plot([x - tick, x + tick], [yi, yi], color="black", lw=1.0, clip_on=False)
        ax.plot([x - tick, x + tick], [yj, yj], color="black", lw=1.0, clip_on=False)
        label = r"$^{ns}$" if star == "ns" else star
        ax.text(x + tick * 0.09, (yi + yj) / 2, label,
                rotation=270, rotation_mode="anchor",
                ha="center", va="center",
                fontsize=FS,
                fontweight="bold" if star != "ns" else "normal",
                color="black" if star != "ns" else "#777",
                clip_on=False)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  Main assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_figure2(s1_dir: str, s3_dir: str, ref: str, spec_tag: str,
                  out_path: str, ps_csv: str | None = None,
                  axis_angle_img: str | None = None,
                  gmw_img: str | None = None,
                  beta_boxplot_img: str | None = None,
                  tables=None):

    rng = np.random.default_rng(42)

    if tables is not None:
        # Rebuild from the published data object. Three panels of this figure
        # embed an already-rendered image rather than plotting from a table,
        # so those travel in the object as their own bytes; without them the
        # figure could not be reconstructed from the pickle alone.
        import tempfile as _tmp
        _ex = tables.get("_extras", {})
        ps_df = tables.get("per_sample")
        axis_samples = tables.get("axis_angle_samples")
        beta_summary = tables.get("beta_summary")
        beta_raw = _ex.get("beta_raw")
        _imgdir = _tmp.mkdtemp(prefix="fig2_imgs_")
        def _restore(key):
            blob = _ex.get(key)
            if not blob:
                return None
            q = os.path.join(_imgdir, f"{key}.png")
            with open(q, "wb") as fh:
                fh.write(blob)
            return q
        axis_angle_img = _restore("axis_angle_img")
        gmw_img = _restore("gmw_img")
        beta_boxplot_img = _restore("beta_boxplot_img")
    else:
        axis_angle_img = resolve_axis_angle_image(s1_dir, spec_tag, axis_angle_img)
        axis_samples = load_axis_angle_samples(s1_dir, spec_tag)
        gmw_img        = resolve_gmw_image(s3_dir, ref, spec_tag, gmw_img)
        beta_boxplot_img = resolve_beta_boxplot_image(
            s1_dir, spec_tag, beta_boxplot_img)
        ps_df = load_per_sample(s3_dir, ref, spec_tag, explicit_path=ps_csv)
        beta_raw = load_beta_raw(s1_dir, spec_tag, ref=ref)
        beta_summary = load_beta_summary(s1_dir, spec_tag)

        def _blob(q):
            try:
                with open(q, "rb") as fh:
                    return fh.read()
            except Exception:
                return None
        from persistence import figure_source as _fsrc
        _fsrc.emit(
            os.path.dirname(out_path) or ".", "F2",
            {k: v for k, v in (("per_sample", ps_df),
                               ("axis_angle_samples", axis_samples),
                               ("beta_summary", beta_summary))
             if v is not None},
            stem=os.path.splitext(os.path.basename(out_path))[0],
            reference=ref,
            extras={"beta_raw": beta_raw,
                    "axis_angle_img": _blob(axis_angle_img),
                    "gmw_img": _blob(gmw_img),
                    "beta_boxplot_img": _blob(beta_boxplot_img)},
            notes=("Main Figure 2. per_sample drives the richness panels, "
                   "beta_summary and the beta_raw arrays the within-group "
                   "distance densities, axis_angle_samples panel F. The three "
                   "image extras are panels rendered elsewhere and embedded "
                   f"here. spec_tag={spec_tag}, rng seed 42."))

    # ── Figure geometry ───────────────────────────────────────────────
    # Top: A/B/C 3-way GMW output.
    # Bottom: D/E stacked on the left; F axis-angle plot spanning both rows
    # on the right.

    FIG_W   = 18.5
    lm, rm  = 0.06, 0.975   # lower-panel margins (figure fraction)
    top_lm = 0.025          # let the GMW strip start at the figure plotting edge

    avail_w = (rm - lm) * FIG_W
    top_h = 5.7
    bottom_h = 7.5
    gap_in = 1.15
    FIG_H = top_h + bottom_h + gap_in + 0.35

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def _f(x): return x / FIG_H

    t_top = 0.98
    b_top = t_top - _f(top_h)
    t_bot = b_top - _f(gap_in)
    b_bot = t_bot - _f(bottom_h)

    top_gs = gridspec.GridSpec(1, 1, figure=fig,
        left=top_lm, right=rm, top=t_top, bottom=b_top)
    bot_gs = gridspec.GridSpec(2, 2, figure=fig,
        left=lm, right=rm, top=t_bot, bottom=b_bot,
        width_ratios=[1.16, 1.48], height_ratios=[1.0, 1.0],
        wspace=0.14, hspace=0.34)

    ax_D = fig.add_subplot(bot_gs[0, 0])
    ax_E = fig.add_subplot(bot_gs[1, 0])
    ax_F = fig.add_subplot(bot_gs[:, 1])

    # ── Draw panels ───────────────────────────────────────────────────
    if ps_df is not None:
        nested = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=top_gs[0, 0],
                                                  wspace=0.24)
        ax_A = fig.add_subplot(nested[0, 0])
        ax_B = fig.add_subplot(nested[0, 1])
        ax_C = fig.add_subplot(nested[0, 2])
        draw_A(ax_A, ps_df, show_title=True)
        draw_violin(ax_B, ps_df, corrected=False, title="Raw richness",
                    rng=rng, show_title=True)
        draw_violin(ax_C, ps_df, corrected=True, title="GMW corrected",
                    rng=rng, show_title=True)
    elif gmw_img is not None:
        nested = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=top_gs[0, 0],
                                                  wspace=0.12)
        ax_A = fig.add_subplot(nested[0, 0])
        ax_B = fig.add_subplot(nested[0, 1])
        ax_C = fig.add_subplot(nested[0, 2])

        # Crop the cached three-panel strip into real subplots so the
        # composite controls labels and edge alignment.
        for ax_, crop, xlabel, ylabel in [
            (ax_A, (0.000, 0.000, 0.342, 1.000),
             "Sequencing depth (Gb)", "Normalised density"),
            (ax_B, (0.330, 0.000, 0.674, 1.000),
             None, "# SGBs per sample"),
            (ax_C, (0.658, 0.000, 1.000, 1.000),
             None, "# SGBs per sample"),
        ]:
            _show_image(ax_, gmw_img, crop=crop,
                        missing="Run pipeline --only health_richness")
            _label_image_axis(ax_, xlabel=xlabel, ylabel=ylabel)
    else:
        nested = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=top_gs[0, 0],
                                                  wspace=0.28)
        ax_A = fig.add_subplot(nested[0, 0])
        ax_B = fig.add_subplot(nested[0, 1])
        ax_C = fig.add_subplot(nested[0, 2])
        draw_A(ax_A, ps_df)
        if ps_df is not None:
            draw_violin(ax_B, ps_df, corrected=False, title="Raw richness", rng=rng)
            draw_violin(ax_C, ps_df, corrected=True, title="GMW corrected", rng=rng)
        else:
            for ax_, t in [(ax_B, "Raw richness"), (ax_C, "GMW corrected")]:
                _style(ax_)
                ax_.text(0.5, 0.5, "Run pipeline --only health_richness",
                         transform=ax_.transAxes, ha="center", va="center",
                         fontsize=FS-2, color="#888", style="italic",
                         multialignment="center")
                ax_.set_title(t, fontsize=FS, fontweight="bold", pad=4)
        for ax_, lbl in [(ax_A, "A"), (ax_B, "B"), (ax_C, "C")]:
            ax_.text(-0.06, 1.065, lbl, transform=ax_.transAxes,
                     fontsize=FS+4, fontweight="bold", va="bottom")

    if beta_raw is not None:
        draw_boxplot_with_sig(ax_D, beta_raw, "jaccard",
                              "Jaccard (presence/absence)")
        draw_boxplot_with_sig(ax_E, beta_raw, "braycurtis",
                              "Bray-Curtis (abundance)")
    elif beta_summary is not None:
        draw_boxplot_summary_with_sig(ax_D, beta_summary, "jaccard",
                                      "Jaccard (presence/absence)")
        draw_boxplot_summary_with_sig(ax_E, beta_summary, "braycurtis",
                                      "Bray-Curtis (abundance)")
    elif beta_boxplot_img is not None:
        _show_image(ax_D, beta_boxplot_img, crop=(0.00, 0.00, 0.585, 1.00))
        _show_image(ax_E, beta_boxplot_img, crop=(0.415, 0.00, 1.00, 1.00))
        ax_D.set_title("Jaccard (presence/absence)", fontsize=FS,
                       fontweight="bold", pad=3)
        ax_E.set_title("Bray-Curtis (abundance)", fontsize=FS,
                       fontweight="bold", pad=3)
    else:
        for ax_, t in [(ax_D,"Jaccard (presence/absence)"),
                       (ax_E,"Bray-Curtis (abundance)")]:
            _style(ax_)
            ax_.text(0.5, 0.5, "Run pipeline --only s1_within",
                     transform=ax_.transAxes, ha="center", va="center",
                     fontsize=FS-2, color="#888", style="italic",
                     multialignment="center")
            ax_.set_title(t, fontsize=FS, fontweight="bold", pad=4)

    draw_axis_angle_native(ax_F, axis_samples, axis_angle_img)

    # ── Panel labels ──────────────────────────────────────────────────
    if ps_df is not None or gmw_img is not None:
        for ax_, lbl, xoff in [(ax_A, "A", -0.19),
                               (ax_B, "B", -0.07),
                               (ax_C, "C", -0.07)]:
            ax_.text(xoff, 1.02, lbl, transform=ax_.transAxes,
                     fontsize=FS+4, fontweight="bold", va="bottom")
    for ax_, lbl in [(ax_D,"D"),(ax_E,"E"),(ax_F,"F")]:
        ax_.text(-0.05, 1.02, lbl, transform=ax_.transAxes,
                 fontsize=FS+4, fontweight="bold", va="bottom")

    # ── Save both formats at the printed column width ─────────────────
    # The figure is authored much wider than it prints, so the page would
    # reduce it. save_column_width does that reduction in the file instead --
    # by raising the render resolution and scaling the PDF page, never by
    # touching the drawing, because this layout is hand-tuned.
    _apply_display_text(fig)
    from persistence.figure_rescale import save_column_width
    stem = os.path.splitext(out_path)[0]
    save_column_width(fig, stem, pad_inches=0.12)
    plt.close(fig)
    for _e in ("png", "pdf"):
        print(f"Saved: {stem}.{_e}")

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate Figure 2.  "
        "Run pipeline --only health_richness s1 s1_within first.")
    ap.add_argument("--ref",     default="combined",
                    help="reference to build from. The main text reports "
                         "combined; UHGG disagrees with the supplementary tables.")
    ap.add_argument("--spec",    default="breadth_pct",
                    help="Spec: breadth_pct | flat_bp | 10pct | 300kb")
    ap.add_argument("--s1",  default=None,
                    help="Section 1 output directory (default: results/<ref>/section1_diversity/main/)")
    ap.add_argument("--s3",  default=None,
                    help="Section 3 output directory (default: results/<ref>/section3_health/main/)")
    ap.add_argument("--ps-csv", default=None,
                    help="Optional: explicit path to per-sample richness CSV "
                         "(e.g. from a previous pipeline run). "
                         "Example: ../full_pipeline_v10/results/UHGG/"
                         "section1_diversity/main/"
                         "health_richness_gmw_UHGG_summary_lifestyle_10pct_perSample.csv")
    ap.add_argument("--axis-angle-img", default=None,
                    help="Optional: explicit axis-angle PNG. Defaults to "
                         "<s1>/axis_angle_v2_pointcloud_<spec>_notitle.png")
    ap.add_argument("--gmw-img", default=None,
                    help="Optional: explicit 3-way GMW PNG. Defaults to "
                         "<s3>/health_richness_gmw_<ref>_summary_lifestyle_<spec>_notitle.png")
    ap.add_argument("--beta-boxplot-img", default=None,
                    help="Optional: explicit beta boxplot PNG. Defaults to "
                         "<s1>/within_group_beta_boxplot_<spec>_notitle.png")
    ap.add_argument("--out",     default="main_text_figures/figure2.pdf")
    args = ap.parse_args()

    if args.s1 is None:
        args.s1 = os.path.join("results", args.ref, "section1_diversity", "main")
    if args.s3 is None:
        args.s3 = os.path.join("results", args.ref, "section3_health", "main")

    for d in [args.s1, args.s3]:
        if not os.path.isdir(d):
            print(f"Error: dir not found: {d}"); sys.exit(1)

    spec_tag = _resolve_spec_tag(args.s1, args.s3, args.spec)
    print(f"Using spec_tag: {spec_tag}")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    build_figure2(s1_dir=args.s1, s3_dir=args.s3, ref=args.ref,
                  spec_tag=spec_tag, out_path=args.out,
                  ps_csv=args.ps_csv, axis_angle_img=args.axis_angle_img,
                  gmw_img=args.gmw_img,
                  beta_boxplot_img=args.beta_boxplot_img)


if __name__ == "__main__":
    main()
