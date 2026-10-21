#!/usr/bin/env python3
"""
summary_accumulation_figure.py
-------------------------------
2 × 2 summary figure: SGB accumulation vs samples (row 1) and studies (row 2),
for Carter (UHGG, col 1) and Leviatan (lev, col 2) references.

Usage
-----
    python summary_accumulation_figure.py \\
        --ref-carter UHGG --ref-lev lev \\
        --out accumulation_summary.png \\
        --module-path /path/to/rank_abundance_analysis.py \\
        --ref-carter UHGG --ref-lev lev \\
        --out accumulation_summary.png \\
        --n-iter 200 --n-iter-asym 100 --seed 0

The --module-path argument is strongly recommended: it prevents
accidentally loading the wrong version of rank_abundance_analysis.py
when multiple copies exist in the workspace.

What is plotted
---------------
Each panel shows POSTERIOR-WEIGHTED ACCUMULATION, not hard detections.
The y-axis E[S(k)] sums prior-corrected excess posteriors across SGBs
(prior-subtracted union accumulation).  This is softer than the
p_genuine_10pct >= 0.5 threshold used by fit_sgb_asymptote; the
asymptote S_inf is a liberal upper bound on the number of SGBs with
any positive above-prior evidence.  Label panels accordingly.
"""

import argparse, os, sys, importlib, importlib.util, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Global style ──────────────────────────────────────────────────────────────
_FS  = 13     # base — axis labels, tick labels, annotations, legend
_FST = 14     # panel title (column header, top row only)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         _FS,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
})

# Colours aligned to the Wong palette in rank_abundance_analysis.py
LS_COLORS  = {"Industrial": "#56B4E9", "Non-Industrial": "#E69F00"}
LS_MARKERS = {"Industrial": "o",       "Non-Industrial": "s"}
LS_ORDER   = ["Industrial", "Non-Industrial"]

# ── Pipeline import ───────────────────────────────────────────────────────────
def _bootstrap_pkg(pkg_dir, alias="mbo"):
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [pkg_dir]; pkg.__package__ = alias
    sys.modules[alias] = pkg; spec.loader.exec_module(pkg)
    return alias

def _resolve_pipeline():
    try:
        from .loading import load_reference as _lr
        from . import config as _cfg
        return _lr, _cfg
    except ImportError:
        pass
    for name in ("persistence", "microbiome_extinctions"):
        try:
            return (importlib.import_module(f"{name}.loading").load_reference,
                    importlib.import_module(f"{name}.config"))
        except ImportError:
            pass
    sd = os.path.dirname(os.path.abspath(__file__))
    for root in [sd, os.path.dirname(sd)]:
        for cand in ["persistence-patched", "persistence", "microbiome_extinctions"]:
            p = os.path.join(root, cand)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
                al = _bootstrap_pkg(p)
                return (importlib.import_module(f"{al}.loading").load_reference,
                        importlib.import_module(f"{al}.config"))
    raise ImportError("Pipeline package not found.")

load_reference, _cfg = _resolve_pipeline()

# ── Reinstatement guard ──────────────────────────────────────────────────────
# load_reference returns the PRE-reinstatement cohort. Every published number is
# computed after reinstate_distinct_secondaries promotes compositionally distinct
# secondary runs to independent samples, which changes N. Wrapping the resolved
# loader here keeps this script on the same cohort as the rest of the pipeline.
_lr_raw = load_reference


def load_reference(_ref, *a, **kw):            # noqa: F811 - deliberate shadow
    _d = _lr_raw(_ref, *a, **kw)
    try:
        import sys as _sys
        from pathlib import Path as _P
        _r = str(_P(__file__).resolve().parents[1])
        if _r not in _sys.path:
            _sys.path.insert(0, _r)
        from persistence.reinstate_secondary import reinstate_distinct_secondaries
        from persistence.stats_report import StatsReport
        return reinstate_distinct_secondaries(_d, StatsReport(str(_ref)))
    except Exception as _exc:
        print(f"    [WARN] reinstatement unavailable ({_exc}); this output will NOT "
              f"match the pipeline's cohort")
        return _d


# ── Import helpers from rank_abundance_analysis.py ───────────────────────────
def _import_ra(explicit_path=None):
    """
    Load helper functions from rank_abundance_analysis.py.

    Priority:
      1. explicit_path (from --module-path CLI argument — recommended)
      2. Same directory as this script
      3. persistence/ under this script directory
      4. Parent directory of this script

    The pipeline-import line is stubbed via regex so this survives any
    rewording of the _resolve_pipeline_imports() call in the main script.
    """
    import re as _re
    sd = os.path.dirname(os.path.abspath(__file__)) if not explicit_path else None
    candidates = (
        [explicit_path] if explicit_path
        else [os.path.join(sd, "rank_abundance_analysis.py"),
              os.path.join(sd, "persistence", "rank_abundance_analysis.py"),
              os.path.join(os.path.dirname(sd), "rank_abundance_analysis.py"),
              os.path.join(os.path.dirname(sd), "persistence", "rank_abundance_analysis.py")]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            spec   = importlib.util.spec_from_file_location("_ra", p)
            mod    = importlib.util.module_from_spec(spec)
            ra_src = open(p).read()
            # Stub the line that CALLS _resolve_pipeline_imports() via assignment.
            # Requiring '=' before the call ensures the function *definition* line
            # (def _resolve_pipeline_imports():) is never accidentally matched.
            ra_src = _re.sub(
                r'^[^\n]*=\s*_resolve_pipeline_imports\(\)[^\n]*$',
                'load_reference = None; _cfg = None  # stubbed for standalone import',
                ra_src, flags=_re.MULTILINE,
            )
            exec(compile(ra_src, p, "exec"), mod.__dict__)
            print(f"  RA module: {os.path.abspath(p)}")
            return mod
    raise FileNotFoundError(
        "rank_abundance_analysis.py not found.\n"
        "Specify it with --module-path /path/to/rank_abundance_analysis.py")


def _pre_parse_module_path():
    """Extract --module-path from sys.argv without full argparse."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--module-path", "--module_path") and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--module-path=") or a.startswith("--module_path="):
            return a.split("=", 1)[1]
    return None


_module_path = _pre_parse_module_path()
if _module_path is None:
    import warnings as _w
    _w.warn(
        "\n"
        "  --module-path not supplied: will auto-discover rank_abundance_analysis.py\n"
        "  from the script directory.  If multiple copies exist in the workspace,\n"
        "  this may silently load the wrong version and produce misleading figures.\n"
        "  Strongly recommended: pass --module-path /path/to/rank_abundance_analysis.py",
        stacklevel=1,
    )
print("  Importing helpers from rank_abundance_analysis.py ...")
_ra = _import_ra(explicit_path=_module_path)
_estimate_sigma_b             = _ra._estimate_sigma_b
_estimate_coverage_efficiency = _ra._estimate_coverage_efficiency
_compute_per_sample_posterior = _ra._compute_per_sample_posterior
_collapse_to_study_posteriors = _ra._collapse_to_study_posteriors
_posterior_accumulation       = _ra._posterior_accumulation
_logspace_checkpoints         = _ra._logspace_checkpoints
_latent_L_prior_mass          = _ra._latent_L_prior_mass

# ── Hill MM fit ───────────────────────────────────────────────────────────────
def _fit_hill_mm(ns, Ss, S_ref, S_sem=None):
    """S(n) = S_max * n^h / (K^h + n^h)  with inverse-variance weighting."""
    ns_f = ns.astype(float); Ss_f = Ss.astype(float)
    w = (1.0 / np.maximum(S_sem, 0.5)**2
         if S_sem is not None and len(S_sem) == len(Ss) else np.ones(len(Ss_f)))
    w /= w.sum()
    def obj(p):
        Sm, K, h = p; nh = ns_f**h; Kh = K**h
        return float(np.sum(w * (Sm*nh/(Kh+nh) - Ss_f)**2))
    bounds = [(1., float(S_ref)), (0.1, float(max(ns_f)*10)), (0.1, 5.)]
    best = None
    for K0 in [float(np.median(ns_f)), float(max(ns_f))]:
        for h0 in [0.5, 1., 2.]:
            r = minimize(obj, [min(Ss_f[-1]*2, S_ref), K0, h0],
                         bounds=bounds, method="L-BFGS-B")
            if best is None or r.fun < best.fun: best = r
    if best is None or (not best.success and best.fun > 1e8):
        return dict(S_max=np.nan, K=np.nan, h=np.nan,
                    S_pred=np.full_like(ns_f, np.nan), success=False)
    Sm, K, h = best.x; nh = ns_f**h; Kh = K**h
    return dict(S_max=float(Sm), K=float(K), h=float(h),
                S_pred=Sm*nh/(Kh+nh), success=True)

# ── Asymptote bootstrap CI ────────────────────────────────────────────────────
def _asymptote_ci(curves, ns, S_ref, ci=0.95, min_pts=10):
    """Fit Hill MM to each bootstrap curve; return (S_max_mean, lo, hi)."""
    S_mean = curves.mean(axis=0)
    S_sem  = curves.std(axis=0) / max(np.sqrt(len(curves)), 1.)
    mm     = _fit_hill_mm(ns, S_mean, S_ref, S_sem=S_sem)
    boot   = []
    for c in curves:
        if (c > 2).sum() < min_pts: continue
        r = _fit_hill_mm(ns, c, S_ref)
        if r["success"] and np.isfinite(r["S_max"]): boot.append(r["S_max"])
    if len(boot) < 5:
        return mm["S_max"], np.nan, np.nan
    tail = (1 - ci) / 2
    return (mm["S_max"],
            float(np.percentile(boot, 100*tail)),
            float(np.percentile(boot, 100*(1-tail))))

# ── Number formatters ─────────────────────────────────────────────────────────
def _kfmt(v, _=None):
    """Axis tick: 2500 → '2.5k', 1000 → '1k', 500 → '500'."""
    if v >= 1000:
        s = f"{v/1000:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return f"{v:.0f}"

def _ann(val, lo=None, hi=None):
    """
    Compact asymptote annotation with a single trailing unit.
    All values in thousands → share one 'k' at the end.
      3456, 3200, 3712  →  'S∞ 3.5 [3.2–3.7]k'
      890               →  'S∞ 890'
    """
    use_k = val >= 1000
    def _f(v): return f"{v/1000:.2g}" if use_k else f"{v:.0f}"
    unit  = "k" if use_k else ""
    base  = f"S\u221e {_f(val)}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        return f"{base} [{_f(lo)}\u2013{_f(hi)}]{unit}"
    return f"{base}{unit}"

# ── Accumulation computation ──────────────────────────────────────────────────
def compute_accumulation(bfrac_s, depth_s, meta_s, genome_lengths, level,
                         n_iter, rng, study_col=None,
                         sigma_bins=None, sigma_edges=None,
                         sigma_b_ref=0.015, c=1.0, n_pts=80):
    _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
    if study_col is None:
        study_col = next((col for col in _STUDY_COLS if col in meta_s.columns), None)
    print(f"      Posteriors ({len(bfrac_s)} × {len(bfrac_s.columns)}) ...", flush=True)
    det = _compute_per_sample_posterior(
        bfrac_s, depth_s, c=c, sigma_b_ref=sigma_b_ref,
        sigma_bins=sigma_bins, sigma_edges=sigma_edges,
        genome_lengths=genome_lengths)
    p_df = det["10pct"]
    if level == "study" and study_col is not None:
        n_u   = meta_s[study_col].nunique()
        p_acc = _collapse_to_study_posteriors(p_df, meta_s, study_col)
        N     = n_u
        xlabel = "Studies"
        print(f"      Study-level: {len(bfrac_s)} → {n_u} studies")
    else:
        p_acc  = p_df
        N      = len(bfrac_s)
        xlabel = "Samples"
    ns = _logspace_checkpoints(N, n_pts=n_pts)
    print(f"      Bootstrap ({n_iter} iter) ...", flush=True)
    curves, ns = _posterior_accumulation(p_acc, n_iter=n_iter, rng=rng, checkpoints=ns)
    return dict(ns_obs=ns, S_mean=curves.mean(0), S_sem=curves.std(0)/max(np.sqrt(n_iter),1.),
                curves=curves, N_total=N, x_label=xlabel,
                # n_det_any: SGBs with any positive prior-corrected excess.
                # Liberal upper bound — NOT equivalent to p_genuine>=0.5.
                n_det_any=int((p_df.values > 0).any(axis=0).sum()))

# ── Panel drawing ─────────────────────────────────────────────────────────────
def draw_panel(ax, results_by_ls, S_ref, ref_label,
               show_title,    # True only for top row (column header)
               show_xlabel,   # True only for bottom row
               show_ylabel,   # True only for left column
               row_label="",  # e.g. "Per-sample" / "Per-study" for left margin
               extrap_factor=2.0, n_pts_show=22, ci_level=0.95,
               n_iter_asym=50, rng_asym=None):
    """One accumulation panel."""
    if rng_asym is None:
        rng_asym = np.random.default_rng(0)

    max_N = max(r["N_total"] for r in results_by_ls.values())
    xlim  = max_N * extrap_factor
    asym  = {}

    for ls in LS_ORDER:
        if ls not in results_by_ls: continue
        r   = results_by_ls[ls]
        col = LS_COLORS[ls]; mk = LS_MARKERS[ls]
        ns  = r["ns_obs"]; Sm = r["S_mean"]; Ssem = r["S_sem"]

        idx = np.unique(np.round(np.linspace(0, len(ns)-1,
                                              min(n_pts_show, len(ns)))).astype(int))
        ax.plot(ns, Sm, color=col, lw=1.4, alpha=0.80, zorder=3)
        ax.scatter(ns[idx], Sm[idx], color=col, s=5**2, marker=mk,
                   zorder=5, linewidths=0)
        ax.errorbar(ns[idx], Sm[idx], yerr=Ssem[idx]*1.96,
                    fmt="none", ecolor=col, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, alpha=0.50, zorder=4)

        mm = _fit_hill_mm(ns, Sm, S_ref, S_sem=Ssem)
        if mm["success"]:
            ax.plot(ns, mm["S_pred"], color=col, lw=2.0, zorder=6)
            ns_ext   = np.linspace(1, xlim, 400)
            nh = ns_ext**mm["h"]; Kh = mm["K"]**mm["h"]
            ext_y    = mm["S_max"] * nh / (Kh + nh)
            ext_mask = ns_ext > r["N_total"]
            if ext_mask.any():
                ax.plot(ns_ext[ext_mask], ext_y[ext_mask],
                        color=col, lw=1.6, ls="--", alpha=0.55, zorder=5)

        cv  = r["curves"]
        sub = cv[rng_asym.choice(len(cv), min(n_iter_asym, len(cv)), replace=False)]
        asym[ls] = (*_asymptote_ci(sub, ns, S_ref, ci=ci_level), mm)

    # ── Asymptote bands + single-unit annotations ─────────────────────────
    ypos = {ls: asym[ls][0]
            for ls in asym
            if asym[ls][3]["success"] and np.isfinite(asym[ls][0])}
    if len(ypos) == 2:
        vals = list(ypos.values())
        if abs(vals[0] - vals[1]) < S_ref * 0.045:
            ypos[LS_ORDER[0]] = max(vals) + S_ref * 0.022
            ypos[LS_ORDER[1]] = min(vals) - S_ref * 0.022

    for ls in LS_ORDER:
        if ls not in asym: continue
        Sm_val, lo, hi, mm = asym[ls]
        if not mm["success"] or not np.isfinite(Sm_val): continue
        col = LS_COLORS[ls]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axhspan(lo, hi, color=col, alpha=0.14, zorder=1)
        ax.axhline(Sm_val, color=col, lw=0.9, ls=":", alpha=0.70, zorder=2)
        # "S∞ 3.5 [3.2–3.7]k" — unit appears once at the end
        ax.text(xlim * 0.015, ypos.get(ls, Sm_val) * 1.012,
                _ann(Sm_val, lo, hi),
                color=col, fontsize=_FS, va="bottom", fontweight="bold", zorder=10)

    # ── Extrapolation shading (no text) ──────────────────────────────────
    ax.axvspan(max_N, xlim, color="#f0f0f0", alpha=0.55, zorder=0)

    # ── Axes ─────────────────────────────────────────────────────────────
    x_label = next(iter(results_by_ls.values()))["x_label"]
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, S_ref * 1.10)
    ax.set_xlabel(x_label if show_xlabel else "", fontsize=_FS)
    # y-label: row_label prefix (e.g. "Per-sample") on left column only
    if show_ylabel:
        y_parts = [row_label, "Posterior-weighted SGBs\n(softer than p_genuine≥0.5 detector)"]
        ax.set_ylabel("\n".join(p for p in y_parts if p), fontsize=_FS - 1)
    else:
        ax.set_ylabel("", fontsize=_FS)
    ax.tick_params(labelsize=_FS - 1,
                   labelbottom=True)   # keep tick numbers on all rows
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_kfmt))
    if show_title:
        ax.set_title(ref_label, fontsize=_FST, fontweight="bold", pad=5)

    # ── Inset legend (bottom-right, ~2/3 panel width) ────────────────────
    ax_ins = ax.inset_axes([0.56, 0.04, 0.42, 0.30])
    ax_ins.legend(
        handles=[Line2D([0],[0], color=LS_COLORS[ls], lw=2.0,
                        marker=LS_MARKERS[ls], markersize=7, label=ls)
                 for ls in LS_ORDER],
        loc="center", fontsize=_FS, framealpha=0.94,
        edgecolor="#cccccc", handlelength=2.2)
    ax_ins.set_axis_off()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_prep(ref_key):
    print(f"    Loading '{ref_key}' ...")
    data    = load_reference(ref_key)
    meta    = data.meta.copy()
    breadth = data.breadth
    lengths = data.lengths
    raw_max = float(breadth.values[breadth.values > 0].max()) \
              if (breadth.values > 0).any() else 0.
    bfrac = (breadth / 100.) if raw_max > 1. else breadth
    samps = [s for s in meta.index if s in bfrac.index and s in data.depth.index]
    bfrac = bfrac.loc[samps]; depth = data.depth.loc[samps]; meta = meta.loc[samps]
    c = _estimate_coverage_efficiency(bfrac, depth)
    sb, se = _estimate_sigma_b(bfrac, depth, c=c)
    print(f"      c={c:.4f}  sigma [{sb.min():.4f},{sb.max():.4f}]")
    if hasattr(lengths, "items") and len(lengths) > 0:
        gl = {str(k): int(v) for k, v in lengths.items()}
    else:
        raise ValueError(
            f"load_and_prep('{ref_key}'): data.lengths is empty or missing. "
            "Ensure all_SGB_lengths_UHGG.txt / all_SGB_lengths_lev.txt are "
            "available and that load_reference populates data.lengths. "
            "Do NOT fall back to a 3 Mb default — genome lengths vary "
            "substantially between SGBs and affect every null curve and prior."
        )
    _SC = ("StudyID","Study","cohort","_study")
    sc  = next((col for col in _SC if col in meta.columns), None)
    if sc: print(f"      study col='{sc}' ({meta[sc].nunique()} studies)")
    return dict(bfrac=bfrac, depth=depth, meta=meta, genome_lengths=gl,
                c=c, sigma_bins=sb, sigma_edges=se, S_ref=bfrac.shape[1], study_col=sc)


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module-path", default=None,
                   help="Explicit path to rank_abundance_analysis.py. "
                        "Recommended: avoids loading the wrong version "
                        "when multiple copies exist in the workspace.")
    p.add_argument("--ref-carter",  default="UHGG")
    p.add_argument("--ref-lev",     default="lev")
    p.add_argument("--out",         default="accumulation_summary.png")
    p.add_argument("--n-iter",      type=int, default=200)
    p.add_argument("--n-iter-asym", type=int, default=100)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--extrap",      type=float, default=2.0)
    p.add_argument("--ci",          type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    rng  = np.random.default_rng(args.seed)

    refs = {"carter": (args.ref_carter, "Carter"),
            "lev":    (args.ref_lev,    "Leviatan")}

    print("\n══ Loading data ══")
    ref_data = {}
    for rid, (key, label) in refs.items():
        print(f"\n  {label} ({key})")
        try:
            ref_data[rid] = load_and_prep(key)
        except Exception as e:
            print(f"  ERROR: {e}"); traceback.print_exc(); sys.exit(1)

    print("\n══ Computing accumulations ══")
    results = {rid: {"sample": {}, "study": {}} for rid in refs}
    for rid, (key, label) in refs.items():
        d = ref_data[rid]
        for level in ["sample", "study"]:
            print(f"\n  {label}  [{level}]")
            for ls in LS_ORDER:
                mask  = d["meta"]["MacroGroup"] == ls
                samps = d["meta"].index[mask]
                if len(samps) < 5:
                    print(f"    skip {ls}: {len(samps)} samples"); continue
                print(f"    {ls}  n={len(samps)}")
                results[rid][level][ls] = compute_accumulation(
                    bfrac_s=d["bfrac"].loc[samps], depth_s=d["depth"].loc[samps],
                    meta_s=d["meta"].loc[samps], genome_lengths=d["genome_lengths"],
                    level=level, n_iter=args.n_iter, rng=rng,
                    study_col=d["study_col"], sigma_bins=d["sigma_bins"],
                    sigma_edges=d["sigma_edges"], c=d["c"])

    print("\n══ Drawing figure ══")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             constrained_layout=True, squeeze=False)
    # Column headers (top row) and axis labels carry all needed information

    for col_idx, (rid, (key, label)) in enumerate(refs.items()):
        S_ref = ref_data[rid]["S_ref"]
        for row_idx, level in enumerate(["sample", "study"]):
            ax      = axes[row_idx, col_idx]
            res_ls  = results[rid][level]
            if not res_ls:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_FS)
                continue
            _ROW_LABELS = ["Per-sample rarefaction", "Per-study rarefaction"]
            draw_panel(
                ax            = ax,
                results_by_ls = res_ls,
                S_ref         = S_ref,
                ref_label     = label,
                show_title    = (row_idx == 0),    # column header on top row only
                show_xlabel   = (row_idx == 1),    # x-label on bottom row only
                show_ylabel   = (col_idx == 0),    # y-label on left column only
                row_label     = _ROW_LABELS[row_idx] if col_idx == 0 else "",
                extrap_factor = args.extrap,
                ci_level      = args.ci,
                n_iter_asym   = args.n_iter_asym,
                rng_asym      = rng,
            )

    fig.text(
        0.5, -0.01,
        "NOTE: y-axis is posterior-weighted accumulation E[S(k)] = \u03a3\u2c7c 1\u2212\u220f(1\u2212p\u1d62\u2c7c), "
        "not SGB counts under the primary detector (p_genuine_10pct \u2265 0.5).\n"
        "S\u221e is a liberal upper bound on the number of SGBs with any above-prior evidence.",
        ha="center", va="top", fontsize=_FS - 2, color="#555555",
        style="italic",
    )
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
