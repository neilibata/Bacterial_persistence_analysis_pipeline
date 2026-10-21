from .. import config
#!/usr/bin/env python3
"""
Publication figures for Section 2 — reads pipeline-generated CSVs.

Usage:
    python -m persistence.figures --section s2 --ref UHGG
    python -m persistence.figures.section2 --ref UHGG

Produces  figures/section2/
    main_figure2.{pdf,png}             Composite 2x2
    volcano_prevalence_7col.{pdf,png}  Standalone
    volcano_abundance_7col.{pdf,png}   Standalone
    section2_stats_summary.csv         Stats table for manuscript

Composite layout:
    A  Prevalence volcano (7-colour)     B  Abundance volcano (7-colour)
    C  VANISH-family pooling (Ind)       D  BloSSUM-family pooling (NI)
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
from ..plot_utils import apply_oi_style, savefig_dual
apply_oi_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Style (matches figures/section1.py) ──────────────────────────
FS_TITLE  = 10
FS_LABEL  = 9
FS_TICK   = 8
FS_LEGEND = 7

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "axes.linewidth": 0.6, "axes.grid": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3,    "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

VOLCANO_PALETTE_7 = config.VOLCANO_PALETTE_7
VOLCANO_DRAW_ORDER_7 = list(VOLCANO_PALETTE_7.keys())

POOL_STYLES = {
    "vanish":       {"c": "#A8C8E8", "lw": 1.5, "ls": "--", "lab": "VANISH"},
    "vanish_plus":  {"c": "#4A90D9", "lw": 1.8, "ls": "-",  "lab": "VANISH+"},
    "all_sgb_ind":  {"c": "#BBBBBB", "lw": 1.2, "ls": ":",  "lab": "All SGBs"},
    "blossum":      {"c": "#F0A0A0", "lw": 1.5, "ls": "--", "lab": "BloSSUM"},
    "blossum_plus": {"c": "#D94A4A", "lw": 1.8, "ls": "-",  "lab": "BloSSUM+"},
    "all_sgb_ni":   {"c": "#BBBBBB", "lw": 1.2, "ls": ":",  "lab": "All SGBs"},
}


def _ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════════════

class S2Data:
    """Lazy-load Section 2 CSVs for one reference."""

    def __init__(self, results_dir, ref="UHGG"):
        self.res = results_dir
        self.ref = ref
        self._s2 = os.path.join(results_dir, "section2_differential", "main")
        self._c  = {}

    def _ld(self, p):
        if p not in self._c:
            self._c[p] = pd.read_csv(p) if os.path.exists(p) else None
        return self._c[p]

    # Differentials (from 2A)
    def prev(self):  return self._ld(os.path.join(self.res, "prevalence_ind_vs_nonind.csv"))
    def abund(self): return self._ld(os.path.join(self.res, "abundance_ind_vs_nonind.csv"))

    # New pooling CSVs (from section2_pooling 2E)
    def vanish_pool(self, s="10pct"):       return self._ld(os.path.join(self._s2, f"vanish_pooling_industrial_{s}.csv"))
    def vanish_plus_pool(self, s="10pct"):  return self._ld(os.path.join(self._s2, f"vanish_plus_pooling_industrial_{s}.csv"))
    def all_ind(self, s="10pct"):           return self._ld(os.path.join(self._s2, f"all_sgb_pooling_industrial_{s}.csv"))
    def blossum_pool(self, s="10pct"):      return self._ld(os.path.join(self._s2, f"blossum_pooling_ni_{s}.csv"))
    def blossum_plus_pool(self, s="10pct"): return self._ld(os.path.join(self._s2, f"blossum_plus_pooling_ni_{s}.csv"))
    def all_ni(self, s="10pct"):            return self._ld(os.path.join(self._s2, f"all_sgb_pooling_ni_{s}.csv"))

    # Legacy pooling (from section2_differential 2B — fallback)
    def vanish_leg(self, s="10pct"):  return self._ld(os.path.join(self._s2, f"vanish_detection_pooling_industrial_{s}.csv"))
    def blossum_leg(self, s="10pct"): return self._ld(os.path.join(self._s2, f"blossum_detection_pooling_ni_{s}.csv"))

    # Extended analyses (2F, 2G)
    def macro(self, s="10pct"): return self._ld(os.path.join(self._s2, f"macro_pooling_detection_{s}.csv"))
    def extinct(self, s="10pct"):
        # Try spec-specific first, fall back to legacy
        p = os.path.join(self._s2, f"extinction_candidates_vanish_blossum_plus_{s}.csv")
        if os.path.exists(p):
            return self._ld(p)
        return self._ld(os.path.join(self._s2, "extinction_candidates_vanish_blossum_plus.csv"))


# ═══════════════════════════════════════════════════════════════════
#  PANELS
# ═══════════════════════════════════════════════════════════════════

def _panel_volcano(ax, df, ecol, xlabel, title, scol="neg_log10_p_bonf"):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return
    df = df.copy()
    if "category_7" not in df.columns:
        df["category_7"] = "None"
    for cat in VOLCANO_DRAW_ORDER_7:
        sub = df[df["category_7"] == cat]
        if sub.empty:
            continue
        bg = (cat == "None")
        ax.scatter(sub[ecol], sub[scol],
                   c=VOLCANO_PALETTE_7[cat],
                   s=6 if bg else 14,
                   alpha=0.25 if bg else 0.75,
                   edgecolors="none",
                   zorder=2 if bg else 3,
                   rasterized=bg)
    ax.axhline(-np.log10(0.05), ls="--", color="grey", alpha=0.5, lw=0.6)
    ax.axvline(0, ls="-", color="grey", alpha=0.3, lw=0.5)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel("\u2212log\u2081\u2080(p) [Bonferroni]", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    hdls = [Patch(fc=VOLCANO_PALETTE_7[c], ec="grey", lw=0.3,
                  label=f"{c} (n={len(df[df['category_7']==c])})")
            for c in VOLCANO_DRAW_ORDER_7
            if not df[df["category_7"] == c].empty]
    ax.legend(handles=hdls, fontsize=FS_LEGEND, loc="upper right",
              framealpha=0.9, edgecolor="grey",
              handlelength=1.0, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)


def _panel_pool(ax, pairs, title, max_n=None):
    """pairs = [(DataFrame, style_key), ...]"""
    good = [(d, k) for d, k in pairs if d is not None and not d.empty]
    if not good:
        ax.text(0.5, 0.5, "No pooling data", transform=ax.transAxes, ha="center")
        return
    if max_n is None:
        max_n = min(int(max(d["n_needed"].max() for d, _ in good) * 1.5), 15000)
    nr = np.sort(np.unique(np.concatenate([
        np.arange(1, min(100, max_n)),
        np.logspace(2, np.log10(max(max_n, 101)), 60).astype(int),
    ])))
    for d, k in good:
        st = POOL_STYLES.get(k, {})
        p  = d["prevalence"].dropna().values
        p  = p[p >= 0]
        if len(p) == 0:
            continue
        f = np.array([np.mean(1.0 - (1.0 - p)**n) for n in nr])
        ax.plot(nr, f,
                color=st.get("c", "#888"), lw=st.get("lw", 1.5),
                ls=st.get("ls", "-"), alpha=0.85,
                label=f"{st.get('lab', k)} (n={len(p)})", zorder=3)
        mn = d["n_needed"].median()
        if np.isfinite(mn) and 1 <= mn <= max_n:
            ix = min(np.searchsorted(nr, mn), len(f) - 1)
            ax.plot(mn, f[ix], "o", color=st.get("c", "#888"), ms=4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1, max_n)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, ls=":", color="grey", alpha=0.3, lw=0.5)
    ax.axhline(0.5, ls=":", color="grey", alpha=0.3, lw=0.5)
    ax.set_xlabel("Number of samples pooled", fontsize=FS_LABEL)
    ax.set_ylabel("Fraction of taxa detected\n(expected)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=0.9,
              edgecolor="grey", handlelength=1.5, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)


# ═══════════════════════════════════════════════════════════════════
#  FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_main_figure2(sd, out, spec="10pct"):
    """Main-text Figure 2: 2x2 composite."""
    print("  Building main figure 2 ...")
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)

    # A — Prevalence volcano
    ax = fig.add_subplot(gs[0, 0])
    _panel_volcano(ax, sd.prev(), "prevalence_diff",
                   "Prevalence difference\n(Industrial \u2212 Non-Industrial)",
                   f"{sd.ref}: Prevalence")
    ax.text(-0.10, 1.05, "A", transform=ax.transAxes, fontsize=16, fontweight="bold")

    # B — Abundance volcano
    ax = fig.add_subplot(gs[0, 1])
    _panel_volcano(ax, sd.abund(), "clr_diff",
                   "CLR difference\n(Industrial \u2212 Non-Industrial)",
                   f"{sd.ref}: CLR Abundance")
    ax.text(-0.10, 1.05, "B", transform=ax.transAxes, fontsize=16, fontweight="bold")

    # C — VANISH-family pooling in Industrial
    ax = fig.add_subplot(gs[1, 0])
    vp  = sd.vanish_pool(spec)
    vpp = sd.vanish_plus_pool(spec)
    ai  = sd.all_ind(spec)
    if vp is not None or vpp is not None:
        pairs = [(ai, "all_sgb_ind"), (vp, "vanish"), (vpp, "vanish_plus")]
    else:
        pairs = [(sd.vanish_leg(spec), "vanish")]
    _panel_pool(ax, pairs, "VANISH / VANISH+ detection\nin industrial samples")
    ax.text(-0.10, 1.05, "C", transform=ax.transAxes, fontsize=16, fontweight="bold")

    # D — BloSSUM-family pooling in Non-Industrial
    ax  = fig.add_subplot(gs[1, 1])
    bp  = sd.blossum_pool(spec)
    bpp = sd.blossum_plus_pool(spec)
    an  = sd.all_ni(spec)
    if bp is not None or bpp is not None:
        pairs = [(an, "all_sgb_ni"), (bp, "blossum"), (bpp, "blossum_plus")]
    else:
        pairs = [(sd.blossum_leg(spec), "blossum")]
    _panel_pool(ax, pairs, "BloSSUM / BloSSUM+ detection\nin non-industrial samples")
    ax.text(-0.10, 1.05, "D", transform=ax.transAxes, fontsize=16, fontweight="bold")

    fig.suptitle(
        "Differing sets of low-prevalence microbes in industrial\n"
        "and hunter-gatherer populations",
        fontsize=12, fontweight="bold", y=0.99)

    for fmt in ("pdf", "png"):
        savefig_dual(fig, os.path.join(out, f"main_figure2.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: main_figure2.pdf/png")


def build_standalone_volcanos(sd, out):
    """Standalone full-size volcano plots."""
    print("  Building standalone volcanos ...")
    for fn, ecol, xlab, title, stem in [
        (sd.prev, "prevalence_diff",
         "Prevalence difference (Industrial \u2212 Non-Industrial)",
         f"{sd.ref}: Prevalence (Industrial \u2212 Non-Industrial)",
         "volcano_prevalence_7col"),
        (sd.abund, "clr_diff",
         "CLR difference (Industrial \u2212 Non-Industrial)",
         f"{sd.ref}: CLR Abundance (Industrial \u2212 Non-Industrial)",
         "volcano_abundance_7col"),
    ]:
        df = fn()
        if df is None or df.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        _panel_volcano(ax, df, ecol, xlab, title)
        plt.tight_layout()
        for fmt in ("pdf", "png"):
            savefig_dual(fig, os.path.join(out, f"{stem}.{fmt}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {stem}")


def build_stats_table(sd, out, spec="10pct"):
    """Summary stats table for Section 2 manuscript text."""
    print("  Building stats summary ...")
    rows = []

    prev = sd.prev()
    if prev is not None and not prev.empty:
        nsig = (prev["neg_log10_p_bonf"] > -np.log10(0.05)).sum()
        rows += [
            dict(metric="n_tested", value=len(prev)),
            dict(metric="n_significant", value=nsig),
            dict(metric="n_NI_enriched",
                 value=int(((prev["prevalence_diff"] < 0) &
                            (prev["neg_log10_p_bonf"] > -np.log10(0.05))).sum())),
            dict(metric="n_Ind_enriched",
                 value=int(((prev["prevalence_diff"] > 0) &
                            (prev["neg_log10_p_bonf"] > -np.log10(0.05))).sum())),
        ]
        for c in VOLCANO_DRAW_ORDER_7:
            rows.append(dict(metric=f"n_{c}",
                             value=int((prev["category_7"] == c).sum())))

    for tag, pfn, afn, lab in [
        ("VANISH",   sd.vanish_pool,       sd.all_ind, "VANISH_Ind"),
        ("VANISH+",  sd.vanish_plus_pool,  sd.all_ind, "VANISH+_Ind"),
        ("BloSSUM",  sd.blossum_pool,      sd.all_ni,  "BloSSUM_NI"),
        ("BloSSUM+", sd.blossum_plus_pool, sd.all_ni,  "BloSSUM+_NI"),
    ]:
        p = pfn(spec)
        a = afn(spec)
        if p is not None and not p.empty:
            m  = p["n_needed"].median()
            ma = a["n_needed"].median() if a is not None else np.nan
            rows += [
                dict(metric=f"{lab}_median_N", value=f"{m:.0f}"),
                dict(metric=f"{lab}_fold",
                     value=f"{m/ma:.1f}x" if ma > 0 else ""),
                dict(metric=f"{lab}_n_taxa", value=len(p)),
            ]

    ec = sd.extinct(spec)
    if ec is not None and not ec.empty:
        rows.append(dict(metric="n_extinction_candidates", value=len(ec)))
        det_col = "n_detected" if "n_detected" in ec.columns else "n_detected_1pct"
        for _, r in ec.iterrows():
            rows.append(dict(
                metric=f"extinct_{r['taxon']}",
                value=(f"{r['direction']}|{r[det_col]}@spec"
                       f"|{r.get('taxonomy', '')}")))

    macro = sd.macro(spec)
    if macro is not None and not macro.empty:
        det_col = "n_detected" if "n_detected" in macro.columns else "n_detected_main"
        for lst in macro["list"].unique():
            sub  = macro[macro["list"] == lst]
            miss = sub.loc[sub[det_col] == 0, "taxon"].tolist()
            rows.append(dict(
                metric=f"pooled_missing_{lst}",
                value="; ".join(miss) if miss else "(all detected)"))

    pd.DataFrame(rows).to_csv(
        os.path.join(out, "section2_stats_summary.csv"), index=False)
    print(f"    Saved: section2_stats_summary.csv ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate Section 2 publication figures from pipeline CSVs")
    ap.add_argument("--ref", default="UHGG")
    ap.add_argument("--data-dir", default=None,
                    help="Path to results/{ref}/ directory")
    ap.add_argument("--out", default="figures/section2")
    ap.add_argument("--spec", default="10pct",
                    help="Detection spec tag (default: 10pct)")
    args = ap.parse_args()

    ddir = args.data_dir or os.path.join("results", args.ref)
    if not os.path.isdir(ddir):
        print(f"ERROR: {ddir} not found.  Run --only s2 first.")
        sys.exit(1)

    out = _ensure(args.out)
    sd  = S2Data(ddir, args.ref)
    print(f"\n  Data: {ddir}\n  Out:  {out}\n  {'='*50}\n")

    build_main_figure2(sd, out, args.spec)
    build_standalone_volcanos(sd, out)
    build_stats_table(sd, out, args.spec)
    print(f"\n  Done. Output: {out}/")


if __name__ == "__main__":
    main()
