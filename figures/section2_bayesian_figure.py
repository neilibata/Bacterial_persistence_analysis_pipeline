#!/usr/bin/env python3
"""
section2_bayesian_figure.py
---------------------------
Section 2 figure using Bayesian latent-L posteriors.

Figure layout (4 rows x 2 cols when study column present, 3 rows otherwise)
----------------------------------------------------------------------------
  A  VANISH-family  per-SAMPLE accumulation  (% of list detected vs N samples)
  B  BloSSUM-family per-SAMPLE accumulation
  C  VANISH-family  per-STUDY  accumulation  (% of list detected vs N studies)
  D  BloSSUM-family per-STUDY  accumulation
  E/C  VANISH / BloSSUM detection-burden dot plot
  F/D  BloSSUM detection-burden dot plot
  G/E  Prevalence volcano  (Industrial - Non-Industrial, Bayesian GMW)
  H/F  CLR Abundance volcano (Bayesian)

Accumulation panels use the SAME hard union detector as accumulation_summary_v2:
  S(k | perm) = sum_j I(p_union_j(k) >= 0.5)
  where p_union_j(k) = 1 - prod_{t=1}^{k}(1 - p_raw[perm[t],j])

Lists loaded from canonical pipeline outputs (not re-derived):
  data.vanish_set / data.blossum_set   — canonical VANISH / BloSSUM
  bayes_vanish_plus_ids.txt            — VANISH+  (Pass 2)
  bayes_blossum_plus_ids.txt           — BloSSUM+ (Pass 2)
  bayes_vanishH_ids.txt                — VanishH+ (Pass 3, Ind-healthy vs NI)
  bayes_blossumH_ids.txt               — BlosumH+ (Pass 3)

Cache: raw posterior matrices stored as compressed NPZ (version 2).
  Old soft_prev cache files (bayesian_soft_prev_*) are ignored.
  New cache: bayesian_plot_v2_{ref}_{null_key}.npz

p-value precision:
  Prefers neg_log10_p_bonf_exact column (computed via chi2.logsf, no cap).
  Falls back to chi2-column recomputation.
  Volcano loading is STRICT by default — fails loudly if Bayesian CSV absent.
  Use --allow-legacy-fallback to permit old non-Bayesian CSVs.
"""

import argparse
import os
from pathlib import Path
import re
import sys
import importlib
import importlib.util
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import norm as _norm_dist

warnings.filterwarnings("ignore", category=FutureWarning)

# The repo root goes on the path BEFORE the import: run by path, sys.path[0] is
# `figures/` and `persistence` is not importable from it, so the except
# branch ran instead and did so silently. See figure2.py for the full note.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from persistence.plot_utils import apply_display_text
except ImportError as _exc:                        # package genuinely absent
    # Not a second rename table: this one renamed Transitional to the
    # title-case "Agro-Pastoralist", which is neither the internal key nor the
    # published label, so a fallback render disagreed with every other figure.
    print(f"  [warn] persistence.plot_utils unavailable ({_exc}); figure "
          f"text will carry internal keys, not published labels",
          file=sys.stderr)

    def apply_display_text(fig):
        return fig

# ══════════════════════════════════════════════════════════════════════════════
#  Style constants
# ══════════════════════════════════════════════════════════════════════════════

# This figure is authored at 15 in and prints at 7.24, so everything here is
# reduced to 49% on the page. The legend at 9.5 pt printed at 4.6 pt, under
# the Science 5 pt floor; the rest had room to grow with it. Printed sizes at
# the bottom of each line.
FS_PANEL  = 22        # 10.7 pt
FS_TITLE  = 13        #  6.3 pt
FS_LABEL  = 15        #  7.3 pt
FS_TICK   = 14        #  6.8 pt
FS_LEGEND = 11        #  5.4 pt -- was 9.5, which printed below the floor

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.6, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

_W = dict(sky="#56B4E9", orange="#E69F00", yellow="#F0E442",
          blue="#0072B2", green="#009E73", vermillion="#D55E00",
          pink="#CC79A7", black="#000000", grey="#999999")

# VANISH-family: VANISH / VANISH+ / VanishH+
_VANISH_COLORS  = [_W["yellow"], _W["orange"], _W["vermillion"]]
# BloSSUM-family: BloSSUM / BloSSUM+ / BlosumH+
_BLOSSUM_COLORS = [_W["sky"],    _W["blue"],   _W["green"]]

# ASCII-safe volcano category labels (no Unicode that fails in Helvetica)
VOLCANO_PALETTE = {
    "None":                 "#CCCCCC",
    "VANISH only":          _W["yellow"],
    "BloSSUM only":         _W["sky"],
    "VANISH+ only":         _W["orange"],
    "BloSSUM+ only":        _W["green"],
    "VANISH and VANISH+":   _W["vermillion"],
    "BloSSUM and BloSSUM+": _W["blue"],
}

# Consistency-threshold registry: 1,3,10,30,50,90% and 10,30,100,300,500kB,1Mb
_BAYES_NULL_REGISTRY = {
    "1pct": (0.01, None), "3pct": (0.03, None), "10pct": (0.10, None),
    "30pct": (0.30, None), "50pct": (0.50, None), "90pct": (0.90, None),
    "10kb": (None, 10), "30kb": (None, 30), "100kb": (None, 100),
    "300kb": (None, 300), "500kb": (None, 500), "1Mb": (None, 1000),
}

_RICHNESS_KNOWN = {"Industrial": 500, "Non-Industrial": 800}

# Cache version — increment if format changes to invalidate old caches
# Bumped when the cohort definition changes, so a cache built on an older
# cohort cannot be silently reused. v4 = post-reinstatement, post-duplicate-ID fix.
_CACHE_VERSION = 4
_CACHE_NPZ = "bayesian_plot_v{ver}_{ref}_{null_key}.npz"
_CACHE_META = "bayesian_plot_v{ver}_{ref}_{null_key}_meta.json"

# ══════════════════════════════════════════════════════════════════════════════
#  Module bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_pkg(pkg_dir, alias="mbo"):
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [pkg_dir]; pkg.__package__ = alias
    sys.modules[alias] = pkg; spec.loader.exec_module(pkg); return alias


def _resolve_pipeline():
    try:
        from .loading import load_reference as _lr
        from . import config as _cfg; return _lr, _cfg
    except ImportError: pass
    for name in ("persistence", "microbiome_extinctions"):
        try:
            return (importlib.import_module(f"{name}.loading").load_reference,
                    importlib.import_module(f"{name}.config"))
        except ImportError: pass
    sd = os.path.dirname(os.path.abspath(__file__))
    for root in [sd, os.path.dirname(sd)]:
        for cand in ["persistence-patched", "persistence", "microbiome_extinctions"]:
            p = os.path.join(root, cand)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
                al = _bootstrap_pkg(p)
                return (importlib.import_module(f"{al}.loading").load_reference,
                        importlib.import_module(f"{al}.config"))
    raise ImportError("Pipeline package not found.")


def _import_ra(explicit_path=None):
    sd = os.path.dirname(os.path.abspath(__file__))
    for p in ([explicit_path] if explicit_path else [
        os.path.join(sd, "rank_abundance_analysis.py"),
        os.path.join(sd, "persistence", "rank_abundance_analysis.py"),
        os.path.join(os.path.dirname(sd), "rank_abundance_analysis.py"),
        os.path.join(os.path.dirname(sd), "persistence", "rank_abundance_analysis.py"),
    ]):
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("_ra", p)
            mod  = importlib.util.module_from_spec(spec)
            src  = re.sub(r'^[^\n]*=\s*_resolve_pipeline_imports\(\)[^\n]*$',
                          'load_reference = None; _cfg = None',
                          open(p).read(), flags=re.MULTILINE)
            exec(compile(src, p, "exec"), mod.__dict__)
            print(f"  [import] RA <- {os.path.abspath(p)}")
            return mod
    raise FileNotFoundError("rank_abundance_analysis.py not found. Pass --ra-module-path.")


def _patch_null_models(ra):
    added = []
    for k, v in _BAYES_NULL_REGISTRY.items():
        if k not in ra.NULL_MODELS:
            ra.NULL_MODELS[k] = v; added.append(k)
    if added:
        print(f"    [patch] NULL_MODELS: added {added}")


# ══════════════════════════════════════════════════════════════════════════════
#  Canonical list loading  (replaces build_lists / build_hplus)
# ══════════════════════════════════════════════════════════════════════════════

def _read_id_file(path):
    """Read newline-separated SGB IDs from a text file. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]


def load_canonical_lists(data, results_dir, prefer_bayes=True):
    """
    Load the canonical VANISH/BloSSUM list IDs from pipeline outputs.

    Sources
    -------
    V  / B  : data.vanish_set / data.blossum_set  (loaded with the reference)
    Vp / Bp : GMW Ind-all vs NI list files
    VH / BH : GMW Ind-healthy vs NI list files

    The pipeline writes these files with or without a "bayes_" prefix depending
    on which entry point was used:
      Bayesian:     bayes_vanish_plus_ids.txt  (section2_bayesian.py)
      Non-Bayesian: vanish_plus_ids.txt        (section2_differential.py)

    Both variants are tried. Bayesian mode prefers the Bayesian version; hard
    threshold mode prefers the standard non-Bayesian version. All sets are
    intersected with data.breadth.columns so only observed SGBs are retained.
    """
    cols = set(data.breadth.columns)
    s2   = os.path.join(results_dir, "section2_differential", "main")

    V = sorted(set(data.vanish_set  or []) & cols)
    B = sorted(set(data.blossum_set or []) & cols)

    def _load(bayes_name, plain_name, label):
        names = (bayes_name, plain_name) if prefer_bayes else (plain_name, bayes_name)
        for fname in names:
            path = os.path.join(s2, fname)
            if os.path.exists(path):
                ids = sorted(set(_read_id_file(path)) & cols)
                print(f"    {label}: {len(ids)} taxa  [{fname}]")
                return ids
        print(f"    [warn] {label}: 0 taxa loaded.")
        print(f"      Tried: {os.path.join(s2, bayes_name)}")
        print(f"      Tried: {os.path.join(s2, plain_name)}")
        print(f"      Run the Bayesian (or standard) Section 2 pipeline first.")
        return []

    Vp = _load("bayes_vanish_plus_ids.txt",  "vanish_plus_ids.txt",  "Vp")
    Bp = _load("bayes_blossum_plus_ids.txt", "blossum_plus_ids.txt", "Bp")
    VH = _load("bayes_vanishH_ids.txt",      "vanishH_ids.txt",      "VH")
    BH = _load("bayes_blossumH_ids.txt",     "blossumH_ids.txt",     "BH")

    print(f"    V: {len(V)} taxa  B: {len(B)} taxa")
    return dict(V=V, B=B, Vp=Vp, Bp=Bp, VH=VH, BH=BH)


def _collapse_to_study_posteriors(p_df, meta_s, study_col):
    """Collapse sample-level 0/1 detections or posteriors to study-level union."""
    studies = meta_s[study_col].unique()
    arr = np.zeros((len(studies), len(p_df.columns)), dtype=np.float32)
    for k, study in enumerate(studies):
        samp_ids = meta_s.index[meta_s[study_col] == study]
        rows = p_df.loc[p_df.index.intersection(samp_ids)].values
        if rows.shape[0] == 0:
            continue
        lnd = np.sum(np.log1p(-np.clip(rows.astype(np.float64), 0, 1 - 1e-6)), axis=0)
        arr[k] = np.clip(1.0 - np.exp(lnd), 0, 1)
    return pd.DataFrame(arr, index=pd.Index(studies, name=study_col), columns=p_df.columns)


# ══════════════════════════════════════════════════════════════════════════════
#  Posterior computation — identical to section2_bayesian.py
# ══════════════════════════════════════════════════════════════════════════════

def _compute_group(bfrac_s, depth_s, gl, c, sb, se, pi, ra, null_key):
    """
    Compute raw per-sample posteriors for one lifestyle group.
    Returns (soft_prev Series, hard_k Series, p_raw DataFrame[float32]).

    CRITICAL: must use subtract_prior=False (raw posteriors) to be consistent
    with section2_bayesian.build_bayesian_detection_bundle().
    If the local rank_abundance_analysis.py does not support subtract_prior,
    we raise a clear error rather than silently proceeding with wrong values.
    """
    import inspect
    sig = inspect.signature(ra._compute_per_sample_posterior)
    if "subtract_prior" not in sig.parameters:
        raise RuntimeError(
            "\n\n"
            "ERROR: Your local rank_abundance_analysis.py does not support\n"
            "subtract_prior=False. Without this, the figure's posteriors will\n"
            "be PRIOR-SUBTRACTED (excess scores), which are DIFFERENT from the\n"
            "pipeline's raw posteriors. This makes accumulation curves and\n"
            "N-needed estimates INCONSISTENT with the main analysis.\n\n"
            "Fix: copy the rank_abundance_analysis.py from the persistence-patched\n"
            "zip to your pipeline directory, OR use --no-cache and the pipeline's\n"
            "own soft_prev outputs via --use-pipeline-soft-prev.\n"
        )
    p_dict = ra._compute_per_sample_posterior(
        bfrac_s        = bfrac_s,
        depth_s        = depth_s.reindex(columns=bfrac_s.columns),
        c              = c,
        sigma_bins     = sb,
        sigma_edges    = se,
        genome_lengths = gl,
        pi_present     = pi,
        min_map_bp     = 10_000,
        subtract_prior = False,   # MUST be False — raw posteriors matching the pipeline
    )

    if null_key not in p_dict:
        raise KeyError(f"null_key='{null_key}' not in posteriors {list(p_dict)}")
    raw    = p_dict[null_key]
    soft   = raw.mean(axis=0)
    hard_k = (raw >= np.float32(0.5)).sum(axis=0).astype(int)
    return soft, hard_k, raw


# ══════════════════════════════════════════════════════════════════════════════
#  Raw-posterior cache  (NPZ, version-stamped)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_paths(out_dir, ref, null_key):
    stem = lambda t: os.path.join(
        out_dir, t.format(ver=_CACHE_VERSION, ref=ref, null_key=null_key))
    return stem(_CACHE_NPZ), stem(_CACHE_META)


def _save_cache(out_dir, ref, null_key, p_raw_I, p_raw_NI, p_raw_HG,
                meta_I, meta_NI, meta_HG, study_col, pi_ind, pi_ni,
                n_st_I, n_st_NI):
    """Save raw posterior matrices + metadata to compressed NPZ + JSON."""
    npz_path, meta_path = _cache_paths(out_dir, ref, null_key)
    np.savez_compressed(
        npz_path,
        p_raw_I     = p_raw_I.values.astype(np.float32),
        p_raw_NI    = p_raw_NI.values.astype(np.float32),
        p_raw_HG    = p_raw_HG.values.astype(np.float32),
        idx_I       = np.array(list(p_raw_I.index),  dtype=object),
        idx_NI      = np.array(list(p_raw_NI.index), dtype=object),
        idx_HG      = np.array(list(p_raw_HG.index), dtype=object),
        columns     = np.array(list(p_raw_I.columns), dtype=object),
    )
    # Study membership for study-level collapse
    stI  = meta_I[study_col].values.tolist()  if (study_col and study_col in meta_I.columns)  else []
    stNI = meta_NI[study_col].values.tolist() if (study_col and study_col in meta_NI.columns) else []
    stHG = meta_HG[study_col].values.tolist() if (study_col and study_col in meta_HG.columns) else []
    with open(meta_path, "w") as f:
        json.dump(dict(
            cache_version=_CACHE_VERSION,
            ref=ref, null_key=null_key,
            study_col=study_col, pi_ind=pi_ind, pi_ni=pi_ni,
            n_st_I=int(n_st_I), n_st_NI=int(n_st_NI),
            n_I=len(p_raw_I), n_NI=len(p_raw_NI), n_HG=len(p_raw_HG),
            study_labels_I=stI, study_labels_NI=stNI, study_labels_HG=stHG,
        ), f, indent=2)
    print(f"  [cache v{_CACHE_VERSION}] Saved: {npz_path}")


def _load_cache(out_dir, ref, null_key, all_cols):
    """
    Load cached raw posteriors.
    Returns (p_raw_I, p_raw_NI, p_raw_HG, meta_dict) or (None,)*4 if missing/stale.
    Searches out_dir first, then results/{ref}/ (written by section2_bayesian.py).
    """
    # Search order: figure out_dir, then pipeline results dir
    search_dirs = [out_dir, os.path.join("results", ref)]
    npz_path = meta_path = None
    for d in search_dirs:
        np_, mp_ = _cache_paths(d, ref, null_key)
        if os.path.exists(np_) and os.path.exists(mp_):
            npz_path, meta_path = np_, mp_
            break
    if npz_path is None:
        return None, None, None, None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, ValueError, OSError) as _ce:
        print(f"  [cache] Corrupt/unreadable meta file ({_ce}) — deleting and ignoring.")
        try:
            os.remove(meta_path)
        except OSError:
            pass
        return None, None, None, None
    if meta.get("cache_version", 0) != _CACHE_VERSION:
        print(f"  [cache] Version mismatch (found v{meta.get('cache_version','?')}, "
              f"need v{_CACHE_VERSION}) — recomputing.")
        return None, None, None, None

    z      = np.load(npz_path, allow_pickle=True)
    cols   = list(z["columns"])
    # Only keep columns that are still in the current dataset
    common = [c for c in cols if c in all_cols]
    ci     = np.array([cols.index(c) for c in common])

    def _df(mat_key, idx_key):
        return pd.DataFrame(z[mat_key][:, ci].astype(np.float32),
                            index=list(z[idx_key]),
                            columns=common)

    p_raw_I  = _df("p_raw_I",  "idx_I")
    p_raw_NI = _df("p_raw_NI", "idx_NI")
    p_raw_HG = _df("p_raw_HG", "idx_HG")
    print(f"  [cache v{_CACHE_VERSION}] Loaded: {npz_path}")
    return p_raw_I, p_raw_NI, p_raw_HG, meta


# ══════════════════════════════════════════════════════════════════════════════
#  Accumulation — hard union detector (identical to accumulation_summary_v2)
# ══════════════════════════════════════════════════════════════════════════════

def _logspace_cp(N, n_pts=60):
    if N <= n_pts:
        return np.arange(1, N + 1, dtype=int)
    ratio = N ** (1.0 / (n_pts - 1))
    cross = max(2, int(np.ceil(1.0 / (ratio - 1))))
    head  = np.arange(1, min(cross, N) + 1, dtype=int)
    tb    = n_pts - len(head)
    if tb > 0 and cross < N:
        raw  = np.logspace(np.log10(cross + 1), np.log10(N), tb * 4)
        tail = np.unique(np.round(raw).astype(int))
        tail = np.clip(tail, cross + 1, N)
        if len(tail) > tb:
            tail = tail[np.round(np.linspace(0, len(tail) - 1, tb)).astype(int)]
    else:
        tail = np.array([], dtype=int)
    return np.union1d(np.union1d(head, tail), [1, N]).astype(int)


def _summarize_curves(curves, ns, n_total, n_list):
    """Convert (n_iter, n_cp) fraction curves to quantile ribbons."""
    return dict(
        ns     = np.asarray(ns, dtype=np.int32),
        q025   = np.percentile(curves,  2.5, axis=0).astype(np.float32),
        q250   = np.percentile(curves, 25.0, axis=0).astype(np.float32),
        q500   = np.percentile(curves, 50.0, axis=0).astype(np.float32),
        q750   = np.percentile(curves, 75.0, axis=0).astype(np.float32),
        q975   = np.percentile(curves, 97.5, axis=0).astype(np.float32),
        N_total = int(n_total),
        n_list  = int(n_list),
        curves  = np.asarray(curves, dtype=np.float32),
    )


def _hard_fraction_accumulation(p_raw_df, taxon_ids, n_iter, rng, n_pts=60):
    """
    Bootstrap accumulation of FRACTION of a taxon list detected, using the same
    hard union rule as accumulation_summary_v2._hard_call_accumulation:

        S(k | perm) = (1/n_list) * sum_j I(p_union_j(k) >= 0.5)

    where p_union_j(k) = 1 - prod_{t=1}^{k}(1 - p_raw[perm[t], j])

    Parameters
    ----------
    p_raw_df  : DataFrame[float32], shape (n_samples_or_studies, n_SGBs)
    taxon_ids : list of SGB IDs in the list
    n_iter    : number of random permutations
    rng       : np.random.Generator

    Returns dict with ns, q025, q250, q500, q750, q975, N_total, n_list
    or None if there are no valid taxa.
    """
    cols = [t for t in taxon_ids if t in p_raw_df.columns]
    if not cols or p_raw_df.shape[0] < 2:
        return None

    p       = p_raw_df[cols].to_numpy(dtype=np.float32)       # (n_s, n_list)
    n_s, n_list = p.shape
    ns      = _logspace_cp(n_s, n_pts)
    log_thr = float(np.log(np.float32(1.0) - np.float32(0.5)))
    log1mp  = np.log1p(-np.clip(p, np.float32(0), np.float32(1.0 - 1e-6)))

    curves = np.zeros((n_iter, len(ns)), dtype=np.float32)
    for it in range(n_iter):
        perm    = rng.permutation(n_s)
        cum_log = np.cumsum(log1mp[perm, :], axis=0)           # (n_s, n_list)
        for qi, k in enumerate(ns):
            curves[it, qi] = float((cum_log[k - 1, :] <= log_thr).mean())

    return _summarize_curves(curves, ns, n_s, n_list)


def _accum_from_soft_prev(soft_prev, taxon_ids, n_total, n_iter, rng, n_pts=60):
    """
    Poisson-binomial accumulation fallback when p_raw matrix is unavailable.

    Uses mean posterior per taxon (soft_prev) to approximate the hard-union
    accumulation curve:
        P(taxon j detected after k samples) ≈ 1 - (1 - p_j)^k

    This is an approximation of the full sample-permutation bootstrap, and
    produces results consistent with the pipeline's _n_needed estimates.
    Used automatically when --use-pipeline-soft-prev is set and no NPZ cache
    exists.
    """
    valid = [t for t in taxon_ids if t in soft_prev.index]
    if not valid or n_total < 2:
        return None
    p = np.clip(
        np.array([float(soft_prev[t]) for t in valid], dtype=np.float64),
        1e-9, 1.0 - 1e-9)
    n_list = len(p)
    ns     = _logspace_cp(n_total, n_pts)
    curves = np.zeros((n_iter, len(ns)), dtype=np.float32)
    for it in range(n_iter):
        for qi, k in enumerate(ns):
            p_det = 1.0 - (1.0 - p) ** k
            curves[it, qi] = float((rng.random(n_list) < p_det).mean())
    return _summarize_curves(curves, ns, n_total, n_list)


# ══════════════════════════════════════════════════════════════════════════════
#  Hill-MM fitting (fraction scale, asymptote in [0,1])
# ══════════════════════════════════════════════════════════════════════════════

def _fit_hill(ns, fmean, cap=1.0):
    ns64 = np.asarray(ns, float); fs64 = np.asarray(fmean, float)
    def obj(p):
        A, b, c = p; nh = ns64 ** c; bh = b ** c
        return float(np.sum((A * nh / (bh + nh) - fs64) ** 2))
    bds  = [(0, cap), (0.1, float(ns64.max() * 10)), (0.05, 5.0)]
    best = None
    for K0 in [float(np.median(ns64)), float(ns64.max() * 0.2)]:
        for c0 in [0.5, 1.0, 1.5]:
            A0  = float(min(fs64[-1] * 1.5, cap))
            res = minimize(obj, x0=[A0, K0, c0], bounds=bds, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
    if best is None or (not best.success and best.fun > 1):
        return np.nan, np.nan, np.nan, np.full_like(ns64, np.nan), False
    A, b, c = best.x; nh = ns64 ** c; bh = b ** c
    return float(A), float(b), float(c), A * nh / (bh + nh), True


def _fit_hill_asymptote_bootstrap(ns, curves, cap=1.0):
    """Fit each bootstrap accumulation curve to estimate asymptote uncertainty."""
    if curves is None:
        return (np.nan, np.nan, np.nan)
    vals = []
    for row in np.asarray(curves, dtype=float):
        if np.sum(np.isfinite(row)) < 5:
            continue
        A, _, _, _, ok = _fit_hill(ns, row, cap=cap)
        if ok and np.isfinite(A):
            vals.append(A)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 50.0, 97.5]))


# ══════════════════════════════════════════════════════════════════════════════
#  Accumulation panel — accumulation_summary_v2 style
# ══════════════════════════════════════════════════════════════════════════════

def _draw_accum(ax, list_data, pop_label, ref_name, level_label, is_sample):
    """
    Accumulation panel with 95%/50% bootstrap ribbons, Hill-MM fit,
    asymptote annotation, and lower-right log-x inset.

    list_data: [(label, color, accum_dict_or_None), ...]
    """
    valid = [(lbl, col, d) for lbl, col, d in list_data if d is not None]
    if not valid:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    max_N = max(d["N_total"] for _, _, d in valid)
    xlim  = max_N * 1.05
    xlab  = ("Number of samples" if is_sample
              else "Number of studies (pooled within-study union)")

    ax.set_xlim(0, xlim); ax.set_ylim(-2, 105)
    ax.axhline(100, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(False)
    # Title removed — pop_label and level embedded in x-axis label
    ax.set_xlabel(f"{xlab} ({pop_label})", fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel("% of list taxa detected", fontsize=FS_LABEL, labelpad=3)
    ax.set_title("", pad=2)

    handles = []
    for lbl, col, acc in valid:
        ns    = acc["ns"]
        q025  = acc["q025"] * 100; q250 = acc["q250"] * 100
        q500  = acc["q500"] * 100; q750 = acc["q750"] * 100; q975 = acc["q975"] * 100
        n_list = acc["n_list"]

        ax.fill_between(ns, q025, q975, color=col, alpha=0.12, linewidth=0, zorder=2)
        ax.fill_between(ns, q250, q750, color=col, alpha=0.22, linewidth=0, zorder=3)
        ax.plot(ns, q500, color=col, lw=0.8, alpha=0.60, zorder=4)
        ax.scatter(ns, q500, color=col, s=9, marker="o", zorder=5, linewidths=0)

        A, b, c, pred, ok = _fit_hill(ns, q500 / 100.0)
        if ok:
            a_lo, a_med, a_hi = _fit_hill_asymptote_bootstrap(
                ns, acc.get("curves"), cap=1.0)
            ns_x = np.linspace(1, xlim, 600)
            p_x  = 100.0 * A * ns_x ** c / (b ** c + ns_x ** c)
            ax.plot(ns_x[ns_x <= max_N], p_x[ns_x <= max_N],
                    color=col, lw=2.0, zorder=6)
            ax.plot(ns_x[ns_x > max_N], p_x[ns_x > max_N],
                    color=col, lw=1.3, ls="--", alpha=0.45, zorder=6)
            a_pct = A * 100
            if np.isfinite(a_lo) and np.isfinite(a_hi):
                ax.axhspan(100.0 * a_lo, 100.0 * a_hi,
                           color=col, alpha=0.06, zorder=0)
            ax.axhline(a_pct, color=col, lw=0.8, ls=":", alpha=0.55, zorder=1)
            if np.isfinite(a_lo) and np.isfinite(a_hi):
                lbl_str = (f"{lbl} (n={n_list})  asymp={a_pct:.0f}% "
                           f"[{100.0 * a_lo:.0f}-{100.0 * a_hi:.0f}], "
                           f"b={b:.1f}, c={c:.2f}")
            else:
                lbl_str = (f"{lbl} (n={n_list})  "
                           f"asymp={a_pct:.0f}%, b={b:.1f}, c={c:.2f}")
        else:
            lbl_str = f"{lbl} (n={n_list})"

        handles.append(plt.Line2D([0], [0], color=col, lw=1.8, label=lbl_str))

    handles.append(plt.Line2D([0], [0], color="#aaaaaa", lw=0.8, ls=":",
                               label="100% of list"))

    # No legend on main axes — full legend lives inside the log inset.

    # ── Log-x inset: 75% wide x 75% tall, anchored lower-right ─────────────
    _IW, _IH = 0.75, 0.72
    ai = ax.inset_axes([1.0 - _IW - 0.01, 0.12, _IW, _IH])
    ai.set_xscale("log"); ai.set_xlim(1, xlim); ai.set_ylim(-2, 105)
    ai.spines["top"].set_visible(False); ai.spines["right"].set_visible(False)
    ai.tick_params(labelsize=FS_TICK - 1, which="both")
    ai.grid(False)
    ai.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ai.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ai.axhline(100, color="#aaaaaa", lw=0.5, ls=":", zorder=0)

    for lbl, col, acc in valid:
        ns     = acc["ns"]; N_tot = acc["N_total"]
        q500   = acc["q500"] * 100
        q025   = acc["q025"] * 100; q975 = acc["q975"] * 100
        q250   = acc["q250"] * 100; q750 = acc["q750"] * 100
        ai.fill_between(ns, q025, q975, color=col, alpha=0.12, linewidth=0, zorder=2)
        ai.fill_between(ns, q250, q750, color=col, alpha=0.22, linewidth=0, zorder=3)
        ai.scatter(ns, q500, color=col, s=5, marker="o", zorder=5, linewidths=0)
        A, b, c, _, ok = _fit_hill(ns, q500 / 100.0)
        if ok:
            a_lo, _, a_hi = _fit_hill_asymptote_bootstrap(
                ns, acc.get("curves"), cap=1.0)
            nl   = np.logspace(0, np.log10(xlim), 300)
            plg  = 100.0 * A * nl ** c / (b ** c + nl ** c)
            if np.isfinite(a_lo) and np.isfinite(a_hi):
                ai.axhspan(100.0 * a_lo, 100.0 * a_hi,
                           color=col, alpha=0.06, zorder=0)
            ai.plot(nl[nl <= N_tot], plg[nl <= N_tot], color=col, lw=1.4, zorder=6)
            ai.plot(nl[nl > N_tot],  plg[nl > N_tot],  color=col, lw=0.9,
                    ls="--", alpha=0.45, zorder=6)

    # Full fit-annotation legend in inset lower-right — this is the only legend.
    ai.legend(handles=handles, fontsize=FS_LEGEND,
              loc="lower right", framealpha=0.93, edgecolor="#cccccc",
              handlelength=1.8, borderpad=0.4, labelspacing=0.25, handletextpad=0.4)
    ai.set_xlabel("N (log scale)", fontsize=FS_TICK, labelpad=2, color="#555")
    ai.set_ylabel("% detected", fontsize=FS_TICK, labelpad=2, color="#555")


# ══════════════════════════════════════════════════════════════════════════════
#  Dot-plot panel
# ══════════════════════════════════════════════════════════════════════════════

def _n_needed(p, n_total):
    p   = float(np.clip(p, 1e-9, 1 - 1e-9))
    raw = np.log(0.5) / np.log(1 - p)
    return n_total if not np.isfinite(raw) or raw <= 0 else min(int(np.ceil(raw)), n_total)


def _nn_arr(soft, ids, n_tot):
    return np.array([_n_needed(float(soft[t]), n_tot)
                     for t in ids if t in soft.index], dtype=float)


def _draw_dotplot(ax, row_data, typical_nn, n_total, pop_label):
    rng  = np.random.default_rng(42)
    vnn  = typical_nn[np.isfinite(typical_nn)]
    blo  = np.percentile(vnn, 25) if len(vnn) else np.nan
    bhi  = np.percentile(vnn, 75) if len(vnn) else np.nan
    bmd  = np.median(vnn)         if len(vnn) else np.nan

    n_rows = len(row_data); yp = np.arange(n_rows, 0, -1, dtype=float)
    if np.isfinite(blo):
        ax.axvspan(blo, bhi, color="#dddddd", alpha=0.45, zorder=0,
                   label="Typical SGB range")
    if np.isfinite(bmd):
        ax.axvline(bmd, color="#aaaaaa", lw=1.0, ls="--",
                   label=f"Typical SGB median (N={bmd:.0f})")

    for y, (lbl, col, nn) in zip(yp, row_data):
        nc = np.array([x for x in nn if np.isfinite(x) and x >= 1])
        if not len(nc): continue
        ax.scatter(nc, y + rng.uniform(-0.28, 0.28, len(nc)),
                   color=col, s=20, alpha=0.65, linewidths=0.2,
                   edgecolors="white", zorder=3)
        med = np.median(nc)
        ax.plot([med, med], [y - 0.35, y + 0.35],
                color=col, lw=2.5, zorder=4, solid_capstyle="round")
        ax.text(med * 1.12, y + 0.08, f"N={med:.0f}", color="#111111",
                fontsize=FS_TICK + 2, va="bottom", fontweight="black", zorder=6)

    ax.set_yticks(yp)
    ax.set_yticklabels([f"{l}\n(n={len(n)})" for l, _, n in row_data], fontsize=FS_TICK)
    ax.set_ylim(0.3, n_rows + 0.5)
    ax.set_xscale("log"); ax.set_xlim(left=0.8)
    # Plain numerals, as on the accumulation insets. Left to itself a log axis
    # formats its ticks as mathtext powers, and matplotlib sets a superscript
    # at 70% of the surrounding face -- 6.8 pt of tick label became a 4.8 pt
    # exponent, the only type on this figure under the Science 5 pt floor.
    # These axes count samples, so "1,000" is also what a reader wants.
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(False)
    pop_full = "Industrial" if pop_label.strip() == "Industrial" else "Non-Industrial"
    ax.set_xlabel(
        f"N {pop_full} samples for 50% detection probability (n={n_total:,})",
        fontsize=FS_LABEL)
    ax.set_title("", pad=2)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.9,
              edgecolor="#cccccc", handlelength=1.5, borderpad=0.4)


# ══════════════════════════════════════════════════════════════════════════════
#  Volcano panel — high-precision p-values + overlap annotation
# ══════════════════════════════════════════════════════════════════════════════

def _exact_neg_log10p(df):
    """
    Return per-row -log10(p_Bonferroni) at unlimited precision.

    For a chi-squared statistic with df=1:
        P(chi2 > x) = 2 * norm.sf(sqrt(x))
        log(P) = log(2) + norm.logsf(sqrt(x))

    norm.logsf is stable for arbitrarily large values (e.g. chi2=50000 ->
    -log10(p) ~ 10860), unlike scipy.stats.chi2.logsf which overflows above
    chi2 ~ 1800.

    Priority:
      1. neg_log10_p_bonf_exact column (pre-computed in pipeline)
      2. chi2 column  -> recompute via norm.logsf(sqrt(chi2)) — unlimited range
      3. neg_log10_p_bonf column (legacy, capped at 300 by np.log10(1e-300))
    """
    if "neg_log10_p_bonf_exact" in df.columns:
        return df["neg_log10_p_bonf_exact"].values.astype(float)

    if "chi2" in df.columns:
        n       = len(df)
        log10e  = np.log10(np.e)
        chi2v   = np.asarray(df["chi2"].values, dtype=np.float64)
        # log(p_raw) = log(2) + norm.logsf(sqrt(chi2))   [chi2 df=1 identity]
        # norm.logsf is numerically stable for all finite inputs
        log_p_raw = np.log(2.0) + _norm_dist.logsf(np.sqrt(np.maximum(chi2v, 0.0)))
        log10_p_bon = log_p_raw * log10e + np.log10(n)
        return np.maximum(-log10_p_bon, 0.0)

    if "neg_log10_p_bonf" in df.columns:
        return df["neg_log10_p_bonf"].values.astype(float)

    raise KeyError("No usable p-value column in volcano DataFrame "
                   "(need chi2, neg_log10_p_bonf_exact, or neg_log10_p_bonf).")


def _fisher_two_sided_logp(d1, n1, d2, n2):
    """Two-sided Fisher exact log-p for [[d1, n1-d1], [d2, n2-d2]]."""
    M = n1 + n2
    K = d1 + d2
    N = n1
    lo = max(0, N - (M - K))
    hi = min(N, K)

    def _logpmf(x):
        return (
            gammaln(K + 1) - gammaln(x + 1) - gammaln(K - x + 1)
            + gammaln(M - K + 1) - gammaln(N - x + 1)
            - gammaln(M - K - (N - x) + 1)
            - (gammaln(M + 1) - gammaln(N + 1) - gammaln(M - N + 1))
        )

    obs = _logpmf(d1)
    vals = np.array([_logpmf(x) for x in range(lo, hi + 1)], dtype=float)
    return float(logsumexp(vals[vals <= obs + 1e-12]))


def _recompute_hard_prevalence_exact(data, bfrac, meta, cfg):
    """
    Recompute hard 10% depth-matched prevalence p-values with log-space Fisher
    exact arithmetic, avoiding the legacy 1e-300 cap in saved CSVs.
    """
    matched_idx = None
    try:
        pkg = data.__class__.__module__.rsplit(".", 1)[0]
        dm_mod = importlib.import_module(f"{pkg}.depth_matching")
        dm = dm_mod.build_depth_matched_sets(
            data,
            frac_tol=getattr(cfg, "DEPTH_MATCH_FRAC_TOL", 0.03),
            health_frac_tol=getattr(cfg, "HEALTH_TRIPLET_FRAC_TOL", 0.01),
        )
        matched_idx = list(dm.all_matched_ind_vs_ni)
        print(f"  Hard prevalence exact p-values: depth-matched n={len(matched_idx):,}")
    except Exception as exc:
        print(f"  [warn] Depth matching unavailable for exact hard p-values ({exc}); using all samples.")
        matched_idx = list(meta.index)

    matched_idx = [s for s in matched_idx if s in meta.index and s in bfrac.index]
    meta_m = meta.loc[matched_idx]
    b_m = bfrac.loc[matched_idx]
    g1 = meta_m.index[meta_m["Lifestyle"] == "Industrial"]
    g2 = meta_m.index[meta_m["Lifestyle"].isin(["H-G", "Transitional"])]
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return None

    rows = []
    for mic in b_m.columns:
        d1 = int((b_m.loc[g1, mic] >= 0.10).sum())
        d2 = int((b_m.loc[g2, mic] >= 0.10).sum())
        if d1 == 0 and d2 == 0:
            continue
        logp = _fisher_two_sided_logp(d1, n1, d2, n2)
        rows.append(dict(
            microbe=mic,
            prev_g1=d1 / n1,
            prev_g2=d2 / n2,
            prevalence_diff=d1 / n1 - d2 / n2,
            log10_pval=logp / np.log(10.0),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n_tests = len(df)
    df["neg_log10_p_bonf_exact"] = np.maximum(
        -(df["log10_pval"].values + np.log10(n_tests)), 0.0)
    df["neg_log10_p_bonf"] = df["neg_log10_p_bonf_exact"]
    return df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)


def _overlap_cat(sgb, vS, bS, vpS, bpS):
    inV  = sgb in vS;  inB  = sgb in bS
    inVp = sgb in vpS; inBp = sgb in bpS
    if inV  and inVp:   return "VANISH and VANISH+"
    if inB  and inBp:   return "BloSSUM and BloSSUM+"
    if inVp:            return "VANISH+ only"
    if inBp:            return "BloSSUM+ only"
    if inV:             return "VANISH only"
    if inB:             return "BloSSUM only"
    return "None"


def _draw_volcano(ax, df, ecol, xlabel, title, palette, vS, bS, vpS, bpS,
                  legend_loc="lower left", missing_msg=None):
    if df is None or df.empty:
        msg = missing_msg or (
            "Bayesian volcano CSV not found.\nRun section2_bayesian pipeline first.\n"
            "(bayes_prevalence_gmw_ind_all_vs_ni.csv\n"
            "or bayes_abundance_ind_vs_nonind.csv)"
        )
        ax.text(0.5, 0.5,
                msg,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_LABEL - 1, color="#888888",
                bbox=dict(boxstyle="round", fc="#f8f8f8", ec="#cccccc"))
        ax.set_title("", pad=2); return

    df = df.copy()
    if "microbe" in df.columns and df.index.name != "microbe":
        df = df.set_index("microbe", drop=False)

    # High-precision y-axis
    df["_y"] = _exact_neg_log10p(df)

    # Overlap categories
    df["_cat"] = [
        _overlap_cat(m, vS, bS, vpS, bpS)
        for m in (df.index if df.index.name == "microbe" else
                  df.get("microbe", df.index))
    ]

    for cat in palette:
        sub = df[df["_cat"] == cat]
        if sub.empty: continue
        bg = (cat == "None")
        ax.scatter(sub[ecol], sub["_y"],
                   c=palette[cat], s=6 if bg else 14,
                   alpha=0.25 if bg else 0.75,
                   edgecolors="none", zorder=2 if bg else 3, rasterized=bg)

    ax.axhline(-np.log10(0.05), ls="--", color="#888888", alpha=0.5, lw=0.6)
    ax.axvline(0, ls="-", color="#888888", alpha=0.3, lw=0.5)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel("-log10(p) [Bonferroni]", fontsize=FS_LABEL)
    ax.set_title("", pad=2)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(False)

    # Legend with counts
    hdls = []
    for cat in palette:
        sub = df[df["_cat"] == cat]
        if sub.empty: continue
        hdls.append(Patch(fc=palette[cat], ec="#888888", lw=0.3,
                          label=f"{cat} (n={len(sub)})"))
    ax.legend(handles=hdls, fontsize=FS_LEGEND, loc=legend_loc,
              framealpha=0.9, edgecolor="#888888",
              handlelength=1.0, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)




# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref",            default="UHGG")
    ap.add_argument("--out-dir",        default="figures/section2_bayesian")
    ap.add_argument("--ra-module-path", default=None)
    ap.add_argument("--data-dir",       default=None,
                    help="Base dir for Bayesian CSVs (default: results/{ref})")
    ap.add_argument("--results-dir",    default=None,
                    help="Pipeline results dir (default: results/{ref}) for list files")
    ap.add_argument("--null-key",       default="10pct")
    ap.add_argument("--n-iter",  type=int,   default=100,
                    help="Bootstrap permutations for accumulation ribbons")
    ap.add_argument("--n-pts",   type=int,   default=60,
                    help="Accumulation checkpoints per curve")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignore existing cache and recompute posteriors")
    ap.add_argument("--allow-legacy-fallback", action="store_true",
                    help="Permit non-Bayesian volcano CSVs as fallback when Bayesian absent")
    ap.add_argument("--use-pipeline-soft-prev", action="store_true",
                    help=("Load soft_prev from pipeline pooling CSVs instead of\n"
                          "recomputing. Guarantees figure N-medians match the\n"
                          "pipeline output exactly. Accumulation curves still\n"
                          "require the NPZ cache (recomputed if absent)."))
    ap.add_argument("--hard-10pct", action="store_true",
                    help=("Build an alternate non-Bayesian figure using only\n"
                          "hard breadth >=10%% detection in each sample."))
    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _emit_data_object(P):
    """Publish everything compose_figure draws from, as one pickle.

    The two volcano frames are tables and go in as panels; the accumulation
    ribbons, the per-taxon posterior prevalences and the list memberships are
    arrays and sets, so they travel as extras. Between them they are the whole
    input to the figure -- rebuild_from_data_object takes nothing else.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from persistence import figure_source as _fsrc
    except Exception as exc:                       # pragma: no cover
        print(f"  [data] could not import figure_source: {exc}")
        return
    try:
        import pandas as _pd
        panels = {}
        for _k in ("prev_df", "abund_df"):
            if isinstance(P.get(_k), _pd.DataFrame):
                panels[_k] = P[_k]
        extras = {k: P.get(k) for k in
                  ("aVs", "aVps", "aBs", "aBps", "aVst", "aVpst", "aBst",
                   "aBpst", "sI_soft", "sN_soft", "V", "B", "Vp", "Bp",
                   "n_I", "n_NI", "typ_I", "typ_NI", "ref", "null_key",
                   "hard_mode")}
        _fsrc.emit(
            str(Path(P["out_dir"])), "F4", panels,
            stem=(f"section2_hard10pct_{P['ref']}" if P["hard_mode"]
                  else f"section2_bayesian_{P['ref']}_{P['null_key']}"),
            reference=P["ref"], extras=extras,
            notes=("Main Figure 4. prev_df and abund_df are the two volcano "
                   "panels; the a* extras are the accumulation quantile "
                   "ribbons per list, sI_soft/sN_soft the per-taxon posterior "
                   "prevalences behind the dot plots, and V/B/Vp/Bp the list "
                   "memberships that colour every panel."))
    except Exception as exc:
        print(f"  [data] could not write the Figure 4 data object: {exc}")


def rebuild_from_data_object(obj, out_dir):
    """Redraw Figure 4 from its published data object."""
    P = dict(obj.get("extras", {}))
    P.update(obj.get("panels", {}))
    P["out_dir"] = str(out_dir)
    P.setdefault("prev_df", None)
    P.setdefault("abund_df", None)
    return compose_figure(P, emit_data=False)


def compose_figure(P, emit_data=True):
    """Draw and save the figure from its inputs alone.

    Split out of main() so the same composition serves both a fresh run
    and a redraw from the published data object: the figure a reader can
    rebuild is then the same code path that made the one in the paper,
    not a reimplementation of it.
    """
    if emit_data:
        _emit_data_object(P)
    vS = set(P['V']); bS = set(P['B']); vpS = set(P['Vp']); bpS = set(P['Bp'])

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\n  Building figure ...")
    has_study = (P['aVst'] is not None)
    n_rows    = 4 if has_study else 3
    fig = plt.figure(figsize=(15, 20 if has_study else 16))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig,
                            hspace=0.12, wspace=0.14,
                            top=0.97, bottom=0.035,
                            left=0.065, right=0.985)

    # Row 0: per-sample accumulation
    ax_A = fig.add_subplot(gs[0, 0])
    _draw_accum(ax_A,
                [("VANISH",   _VANISH_COLORS[0], P['aVs']),
                 ("VANISH+",  _VANISH_COLORS[1], P['aVps'])],
                "Industrial", P['ref'], "per-sample", True)
    ax_A.text(-0.10, 1.06, "A", transform=ax_A.transAxes,
              fontsize=FS_PANEL, fontweight="black")

    ax_B = fig.add_subplot(gs[0, 1])
    _draw_accum(ax_B,
                [("BloSSUM",  _BLOSSUM_COLORS[0], P['aBs']),
                 ("BloSSUM+", _BLOSSUM_COLORS[1], P['aBps'])],
                "Non-Industrial", P['ref'], "per-sample", True)
    ax_B.text(-0.10, 1.06, "B", transform=ax_B.transAxes,
              fontsize=FS_PANEL, fontweight="black")

    # Row 1: per-study accumulation (if available)
    if has_study:
        ax_C = fig.add_subplot(gs[1, 0])
        _draw_accum(ax_C,
                    [("VANISH",   _VANISH_COLORS[0], P['aVst']),
                     ("VANISH+",  _VANISH_COLORS[1], P['aVpst'])],
                    "Industrial", P['ref'], "per-study", False)
        ax_C.text(-0.10, 1.06, "C", transform=ax_C.transAxes,
                  fontsize=FS_PANEL, fontweight="black")

        ax_D = fig.add_subplot(gs[1, 1])
        _draw_accum(ax_D,
                    [("BloSSUM",  _BLOSSUM_COLORS[0], P['aBst']),
                     ("BloSSUM+", _BLOSSUM_COLORS[1], P['aBpst'])],
                    "Non-Industrial", P['ref'], "per-study", False)
        ax_D.text(-0.10, 1.06, "D", transform=ax_D.transAxes,
                  fontsize=FS_PANEL, fontweight="black")

        dr, vr = 2, 3
        dl, vl = ("E", "F"), ("G", "H")
    else:
        dr, vr = 1, 2
        dl, vl = ("C", "D"), ("E", "F")

    # Dot-plot row
    ax_dV = fig.add_subplot(gs[dr, 0])
    _draw_dotplot(ax_dV,
                  [("VANISH+",  _VANISH_COLORS[1], _nn_arr(P['sI_soft'], P['Vp'], P['n_I'])),
                   ("VANISH",   _VANISH_COLORS[0], _nn_arr(P['sI_soft'], P['V'],  P['n_I']))],
                  P['typ_I'], P['n_I'], "Industrial")
    ax_dV.text(-0.12, 1.06, dl[0], transform=ax_dV.transAxes,
               fontsize=FS_PANEL, fontweight="black")

    ax_dB = fig.add_subplot(gs[dr, 1])
    _draw_dotplot(ax_dB,
                  [("BloSSUM+", _BLOSSUM_COLORS[1], _nn_arr(P['sN_soft'], P['Bp'], P['n_NI'])),
                   ("BloSSUM",  _BLOSSUM_COLORS[0], _nn_arr(P['sN_soft'], P['B'],  P['n_NI']))],
                  P['typ_NI'], P['n_NI'], "Non-Industrial")
    ax_dB.text(-0.12, 1.06, dl[1], transform=ax_dB.transAxes,
               fontsize=FS_PANEL, fontweight="black")

    # Volcano row
    ax_pv = fig.add_subplot(gs[vr, 0])
    analysis_label = "Hard 10% coverage" if P['hard_mode'] else f"Bayesian, null={P['null_key']}"
    prevalence_missing = (
        "Hard-threshold prevalence CSV not found.\nRun standard Section 2 first.\n"
        "(prevalence_ind_vs_nonind.csv)"
    ) if P['hard_mode'] else None
    abundance_missing = (
        "Hard-threshold abundance CSV not found.\nRun standard Section 2 first.\n"
        "(abundance_ind_vs_nonind.csv)"
    ) if P['hard_mode'] else None
    _draw_volcano(ax_pv, P['prev_df'], "prevalence_diff",
                  "GMW-weighted prevalence difference (Industrial - Non-Industrial)",
                  f"{P['ref']}: Prevalence (Industrial - Non-Industrial)\n"
                  f"[{analysis_label}]",
                  VOLCANO_PALETTE, vS, bS, vpS, bpS,
                  legend_loc="upper center", missing_msg=prevalence_missing)
    ax_pv.text(-0.10, 1.06, vl[0], transform=ax_pv.transAxes,
               fontsize=FS_PANEL, fontweight="black")

    ax_ab = fig.add_subplot(gs[vr, 1])
    _draw_volcano(ax_ab, P['abund_df'], "clr_diff",
                  "GMW-weighted CLR abundance difference (Industrial - Non-Industrial)",
                  f"{P['ref']}: CLR Abundance (Industrial - Non-Industrial)\n"
                  f"[{analysis_label}]",
                  VOLCANO_PALETTE, vS, bS, vpS, bpS,
                  missing_msg=abundance_missing)
    ax_ab.text(-0.10, 1.06, vl[1], transform=ax_ab.transAxes,
               fontsize=FS_PANEL, fontweight="black")



    stem = (f"section2_hard10pct_{P['ref']}" if P['hard_mode'] else
            f"section2_bayesian_{P['ref']}_{P['null_key']}")
    apply_display_text(fig)
    # Both formats, at the printed column width. The drawing is untouched --
    # the raster is rendered at the resolution the print size needs and the
    # vector page is scaled -- so the hand-tuned layout is preserved exactly.
    from persistence.figure_rescale import save_column_width
    save_column_width(fig, os.path.join(P['out_dir'], stem))
    for fmt in ("pdf", "png"):
        print(f"  Saved: {os.path.join(P['out_dir'], f'{stem}.{fmt}')}")
    plt.close(fig)
    print(f"\n  Done. Output: {os.path.abspath(P['out_dir'])}/\n")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    hard_mode = bool(getattr(args, "hard_10pct", False))

    print(f"\n{'='*65}")
    if hard_mode:
        print(f"  Section 2 Hard 10% Figure | ref={args.ref}")
    else:
        print(f"  Section 2 Bayesian Figure | ref={args.ref} | null={args.null_key}")
    print(f"{'='*65}")

    ra = None if hard_mode else _import_ra(args.ra_module_path)
    load_reference, cfg = _resolve_pipeline()
    if not hard_mode:
        _patch_null_models(ra)

    # ── Always load data (needed for canonical list IDs and metadata) ──────────
    print(f"\n  Loading '{args.ref}' ...")
    data    = load_reference(args.ref)

    # Reinstatement is part of cohort construction, not an optional extra: the
    # pipeline runs it immediately after load_reference and before any statistic,
    # because it promotes compositionally distinct secondary runs to independent
    # samples and therefore changes N. Skipping it here computed this figure on a
    # different cohort from every other analysis in the paper (204 samples short
    # for the combined reference), which is why its "typical SGB" medians did not
    # match the pipeline's.
    try:
        # This script is run from figures/, so the package root is not implicitly
        # importable the way it is for `python -m persistence`.
        _root = str(Path(__file__).resolve().parents[1])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from persistence.reinstate_secondary import reinstate_distinct_secondaries
        from persistence.stats_report import StatsReport

        _n_before = len(data.meta)
        data = reinstate_distinct_secondaries(data, StatsReport(args.ref))
        print(f"    reinstatement: {_n_before:,} -> {len(data.meta):,} samples")
    except Exception as exc:                      # pragma: no cover - defensive
        print(f"    [WARN] could not apply reinstatement ({exc}); this figure will "
              f"be computed on the pre-reinstatement cohort and will NOT match the "
              f"pipeline's numbers")

    meta    = data.meta.copy()
    breadth = data.breadth
    gl      = {str(k): int(v) for k, v in data.lengths.items()}
    S_ref   = breadth.shape[1]
    all_cols = set(breadth.columns)

    raw_max = float(breadth.values[breadth.values > 0].max()) \
              if (breadth.values > 0).any() else 0.0
    bfrac   = (breadth / 100.0) if raw_max > 1.0 else breadth.copy()
    samps   = [s for s in meta.index
               if s in bfrac.index and s in data.depth.index]
    bfrac   = bfrac.loc[samps]
    depth   = data.depth.loc[samps]
    meta    = meta.loc[samps]

    # Lifestyle masks — identical to section2_bayesian.py
    # Uses Lifestyle column directly (same as build_bayesian_detection_bundle)
    is_ind = meta["Lifestyle"] == "Industrial"
    is_ni  = meta["Lifestyle"].isin(["H-G", "Transitional"])
    is_hg  = meta["Lifestyle"].isin(["H-G"])

    samps_I  = meta.index[is_ind]; n_I  = len(samps_I)
    samps_NI = meta.index[is_ni];  n_NI = len(samps_NI)
    samps_HG = meta.index[is_hg];  n_HG = len(samps_HG)
    print(f"    Ind={n_I:,}  NI={n_NI:,}  HG={n_HG:,}")

    meta_I  = meta.loc[samps_I]
    meta_NI = meta.loc[samps_NI]
    meta_HG = meta.loc[samps_HG]

    _SC = ("StudyID", "Study", "cohort", "_study")
    study_col = next((c for c in _SC if c in meta.columns), None)
    n_st_I = n_st_NI = 0
    if study_col:
        n_st_I  = meta_I[study_col].nunique()
        n_st_NI = meta_NI[study_col].nunique()
        print(f"    Studies: Ind={n_st_I}, NI={n_st_NI}")
    else:
        print("    [warn] No study column; per-study panels will be skipped.")

    # ── Load canonical list IDs from pipeline outputs ─────────────────────────
    results_dir = args.results_dir or os.path.join("results", args.ref)
    print(f"\n  Loading canonical list IDs from {results_dir} ...")
    lists = load_canonical_lists(data, results_dir, prefer_bayes=not hard_mode)
    V  = lists["V"];  Vp = lists["Vp"]; VH = lists["VH"]
    B  = lists["B"];  Bp = lists["Bp"]; BH = lists["BH"]

    # ── Load soft_prev from pipeline pooling CSVs if requested ──────────────────
    # When --use-pipeline-soft-prev is set, we read the per-SGB mean posteriors
    # (stored as 'prevalence' in the pooling CSVs) directly from the pipeline
    # output, guaranteeing N-median consistency.
    # Note: the accumulation curves still need the full p_raw matrix; those come
    # from the NPZ cache computed by this figure script.
    _pipeline_soft = {}
    sI_soft_pl = pd.Series(dtype=float)
    sN_soft_pl = pd.Series(dtype=float)
    if (not hard_mode) and getattr(args, "use_pipeline_soft_prev", False):
        print("\n  Loading Bayesian soft_prev from pipeline exports ...")
        # config.outdir(ref) = results/{ref} — use data_dir override if set
        _out = args.data_dir or os.path.join("results", args.ref)
        _nk  = args.null_key
        _ind_path = os.path.join(_out, f"bayes_soft_prev_industrial_{_nk}.csv")
        _ni_path  = os.path.join(_out, f"bayes_soft_prev_ni_{_nk}.csv")
        if os.path.exists(_ind_path):
            sI_soft_pl = pd.read_csv(_ind_path, index_col="taxon")["prevalence"]
            print(f"    Ind soft_prev: {len(sI_soft_pl)} taxa  [{_ind_path}]")
        else:
            print(f"    [warn] Not found: {_ind_path}")
        if os.path.exists(_ni_path):
            sN_soft_pl = pd.read_csv(_ni_path, index_col="taxon")["prevalence"]
            print(f"    NI soft_prev:  {len(sN_soft_pl)} taxa  [{_ni_path}]")
        else:
            print(f"    [warn] Not found: {_ni_path}")

    # ── Posterior computation / hard-threshold detection ───────────────────────
    # Decision logic:
    #   1. Try cache first (unless --no-cache).
    #   2. If cache miss AND --use-pipeline-soft-prev loaded both soft_prev CSVs:
    #      skip posterior recompute entirely — use pipeline soft_prev for the
    #      dotplot; accumulation panels will be skipped (p_raw stays None).
    #   3. If cache miss AND no soft_prev override: recompute posteriors
    #      (requires subtract_prior=False support in local RA).
    p_raw_I = p_raw_NI = p_raw_HG = None
    pi_ind = float(np.clip(_RICHNESS_KNOWN["Industrial"]     / S_ref, 0.01, 0.95))
    pi_ni  = float(np.clip(_RICHNESS_KNOWN["Non-Industrial"] / S_ref, 0.01, 0.95))

    if hard_mode:
        print("\n  Building hard detection matrices [breadth >= 10%] ...")
        p_raw_I  = (bfrac.loc[samps_I]  >= 0.10).astype(np.float32)
        p_raw_NI = (bfrac.loc[samps_NI] >= 0.10).astype(np.float32)
        p_raw_HG = (bfrac.loc[samps_HG] >= 0.10).astype(np.float32) if n_HG >= 5 else p_raw_NI
        sI_soft  = p_raw_I.mean(axis=0)
        sN_soft  = p_raw_NI.mean(axis=0)
        print(f"    Industrial detections: {int(p_raw_I.values.sum()):,}")
        print(f"    Non-Industrial detections: {int(p_raw_NI.values.sum()):,}")
    elif not args.no_cache:
        p_raw_I, p_raw_NI, p_raw_HG, cmeta = _load_cache(
            args.out_dir, args.ref, args.null_key, all_cols)

    # When --use-pipeline-soft-prev is set and files loaded, we don't need
    # to recompute posteriors for the dotplot at all.
    _have_pipeline_soft = (len(sI_soft_pl) > 0 and len(sN_soft_pl) > 0)
    _skip_recompute = (
        getattr(args, "use_pipeline_soft_prev", False) and _have_pipeline_soft
    )

    if (not hard_mode) and p_raw_I is None and not _skip_recompute:
        print(f"\n  Calibrating ...")
        c_est = ra._estimate_coverage_efficiency(bfrac, depth)
        sb, se = ra._estimate_sigma_b(bfrac, depth, c=c_est)
        print(f"    c={c_est:.4f}  pi_Ind={pi_ind:.4f}  pi_NI={pi_ni:.4f}")

        print(f"\n  Computing posteriors [null={args.null_key}] ...")
        print("    Industrial ...")
        sI_soft, kI, p_raw_I = _compute_group(
            bfrac.loc[samps_I], depth.loc[samps_I], gl, c_est, sb, se,
            pi_ind, ra, args.null_key)
        print("    Non-Industrial ...")
        sN_soft, kN, p_raw_NI = _compute_group(
            bfrac.loc[samps_NI], depth.loc[samps_NI], gl, c_est, sb, se,
            pi_ni, ra, args.null_key)
        if n_HG >= 5:
            print("    H-G only ...")
            sHG_soft, kHG, p_raw_HG = _compute_group(
                bfrac.loc[samps_HG], depth.loc[samps_HG], gl, c_est, sb, se,
                pi_ni, ra, args.null_key)
        else:
            p_raw_HG = p_raw_NI

        _save_cache(args.out_dir, args.ref, args.null_key,
                    p_raw_I, p_raw_NI, p_raw_HG,
                    meta_I, meta_NI, meta_HG,
                    study_col, pi_ind, pi_ni, n_st_I, n_st_NI)
    elif (not hard_mode) and p_raw_I is not None:
        # Cache hit: reconstruct soft from raw posteriors
        sI_soft  = p_raw_I.mean(axis=0)
        sN_soft  = p_raw_NI.mean(axis=0)
    elif not hard_mode:
        # _skip_recompute=True: pipeline soft_prev will supply sI_soft/sN_soft below.
        # p_raw remains None → accumulation panels will be skipped gracefully.
        sI_soft = sI_soft_pl.reindex(pd.Index(all_cols)).fillna(0.0)
        sN_soft = sN_soft_pl.reindex(pd.Index(all_cols)).fillna(0.0)
        print("  [pipeline mode] Skipping posterior recompute — using pipeline soft_prev.")
        print("  [pipeline mode] Accumulation panels require NPZ cache; run once with")
        print("                  a patched rank_abundance_analysis.py to build it.")

    # ── Study-level posteriors via ra._collapse_to_study_posteriors ───────────
    p_study_I = p_study_NI = None
    if p_raw_I is not None and study_col and n_st_I >= 5:
        print("\n  Collapsing to study-level posteriors ...")
        if hard_mode:
            p_study_I  = _collapse_to_study_posteriors(p_raw_I,  meta_I,  study_col)
            p_study_NI = _collapse_to_study_posteriors(p_raw_NI, meta_NI, study_col)
        else:
            p_study_I  = ra._collapse_to_study_posteriors(p_raw_I,  meta_I,  study_col)
            p_study_NI = ra._collapse_to_study_posteriors(p_raw_NI, meta_NI, study_col)
        print(f"    Ind: {p_raw_I.shape[0]} samples -> {p_study_I.shape[0]} studies")
        print(f"    NI:  {p_raw_NI.shape[0]} samples -> {p_study_NI.shape[0]} studies")

    # ── Bootstrap accumulation ────────────────────────────────────────────────
    # If p_raw is available (cache hit or fresh compute): hard-union detector.
    # If p_raw is None (pipeline-soft-prev mode): Poisson-binomial fallback.
    n_it, n_pt = args.n_iter, args.n_pts
    _using_soft_fallback = (p_raw_I is None)
    if hard_mode:
        print(f"\n  Bootstrap accumulation (hard 10% union, {n_it} iters, {n_pt} pts) ...")
    elif _using_soft_fallback:
        print(f"\n  Bootstrap accumulation (Poisson-binomial fallback, {n_it} iters) ...")
        print("  [note] Using pipeline soft_prev for accumulation curves.")
        print("         For exact hard-union curves, build NPZ cache with patched RA.")
    else:
        print(f"\n  Bootstrap accumulation (hard-union, {n_it} iters, {n_pt} pts) ...")

    def _acc(p_df_or_soft, ids, n_tot, tag, is_soft=False):
        if len(ids) == 0:
            return None
        print(f"    {tag} (n_list={len(ids)}) ...", flush=True)
        if is_soft:
            return _accum_from_soft_prev(p_df_or_soft, ids, n_tot, n_it, rng, n_pts=n_pt)
        if p_df_or_soft is None:
            return None
        return _hard_fraction_accumulation(p_df_or_soft, ids, n_it, rng, n_pts=n_pt)

    # Per-sample
    _sf = _using_soft_fallback
    aVs  = _acc(sI_soft if _sf else p_raw_I,  V,  n_I,  "VANISH/samp",   _sf)
    aVps = _acc(sI_soft if _sf else p_raw_I,  Vp, n_I,  "VANISH+/samp",  _sf)
    aVHs = _acc(sI_soft if _sf else p_raw_I,  VH, n_I,  "VH+/samp",      _sf)
    aBs  = _acc(sN_soft if _sf else p_raw_NI, B,  n_NI, "BloSSUM/samp",  _sf)
    aBps = _acc(sN_soft if _sf else p_raw_NI, Bp, n_NI, "BloSSUM+/samp", _sf)
    aBHs = _acc(sN_soft if _sf else p_raw_NI, BH, n_NI, "BH+/samp",      _sf)

    # Per-study (collapsed posterior matrices via ra._collapse_to_study_posteriors)
    aVst  = _acc(p_study_I,  V,  n_st_I,  "VANISH/stdy")
    aVpst = _acc(p_study_I,  Vp, n_st_I,  "VANISH+/stdy")
    aVHst = _acc(p_study_I,  VH, n_st_I,  "VH+/stdy")
    aBst  = _acc(p_study_NI, B,  n_st_NI, "BloSSUM/stdy")
    aBpst = _acc(p_study_NI, Bp, n_st_NI, "BloSSUM+/stdy")
    aBHst = _acc(p_study_NI, BH, n_st_NI, "BH+/stdy")

    # Override soft_prev with pipeline-derived values if requested
    if (not hard_mode) and getattr(args, "use_pipeline_soft_prev", False):
        # Override soft_prev with pipeline-exported authoritative values.
        # Gate on len(), NOT on _pipeline_soft dict (which stays empty now).
        used = False
        if len(sI_soft_pl) > 0:
            sI_soft = sI_soft_pl.reindex(sI_soft.index).fillna(sI_soft)
            print(f"  [soft_prev override] Ind: {sI_soft_pl.notna().sum()} taxa from pipeline")
            used = True
        if len(sN_soft_pl) > 0:
            sN_soft = sN_soft_pl.reindex(sN_soft.index).fillna(sN_soft)
            print(f"  [soft_prev override] NI: {sN_soft_pl.notna().sum()} taxa from pipeline")
            used = True
        if not used:
            print("  [warn] --use-pipeline-soft-prev set but no soft_prev CSVs found.\n"
                  f"        Expected: bayes_soft_prev_industrial_{args.null_key}.csv\n"
                  f"        In:       {args.data_dir or os.path.join('results', args.ref)}")

    # ── Dot-plot data ─────────────────────────────────────────────────────────
    # Typical SGB: ALL taxa (matching pipeline 2E "All SGBs" pooling output).
    # The pipeline figure-style N-medians exclude list taxa (321), but the
    # 2E table uses all SGBs — we match the 2E table here.
    typ_I   = _nn_arr(sI_soft, list(sI_soft.index), n_I)
    typ_NI  = _nn_arr(sN_soft, list(sN_soft.index), n_NI)

    # ── Volcano CSVs ─────────────────────────────────────────────────────────
    # Bayesian mode is strict by default. Hard-threshold mode reads the standard
    # non-Bayesian Section 2 CSVs produced by the hard 10% differential pass.
    ddir = args.data_dir or os.path.join("results", args.ref)
    prev_df = abund_df = None

    _prev_bayes = [
        os.path.join(ddir, "bayes_prevalence_gmw_ind_all_vs_ni.csv"),
        os.path.join(ddir, "section2_differential", "main",
                     "bayes_prevalence_gmw_ind_all_vs_ni.csv"),
    ]
    _abund_bayes = [
        os.path.join(ddir, "bayes_abundance_ind_vs_nonind.csv"),
        os.path.join(ddir, "section2_differential", "main",
                     "bayes_abundance_ind_vs_nonind.csv"),
    ]
    _prev_legacy  = [os.path.join(ddir, "section2_differential", "main",
                                   "prevalence_ind_vs_nonind.csv"),
                      os.path.join(ddir, "prevalence_ind_vs_nonind.csv")]
    _abund_legacy = [os.path.join(ddir, "section2_differential", "main",
                                   "abundance_ind_vs_nonind.csv"),
                      os.path.join(ddir, "abundance_ind_vs_nonind.csv")]

    _prev_order = _prev_legacy if hard_mode else _prev_bayes
    _abund_order = _abund_legacy if hard_mode else _abund_bayes

    if hard_mode:
        prev_df = _recompute_hard_prevalence_exact(data, bfrac, meta, cfg)
        if prev_df is not None:
            print(f"\n  Prevalence table [hard 10%, exact log-p]: {len(prev_df)} rows")
    else:
        for path in _prev_order:
            if os.path.exists(path):
                prev_df = pd.read_csv(path)
                print(f"\n  Prevalence CSV [bayes]: {path} ({len(prev_df)} rows)")
                break
    if prev_df is None:
        if (not hard_mode) and getattr(args, "allow_legacy_fallback", False):
            for path in _prev_legacy:
                if os.path.exists(path):
                    prev_df = pd.read_csv(path)
                    print(f"\n  Prevalence CSV [LEGACY — non-Bayesian]: {path} ({len(prev_df)} rows)")
                    break
        if prev_df is None:
            csv_kind = "hard 10% prevalence" if hard_mode else "Bayesian prevalence"
            print(f"\n  [ERROR] {csv_kind} CSV not found. Tried:")
            for p in _prev_order: print(f"    {p}")
            if hard_mode:
                print("  Run standard Section 2 first: python -m persistence --ref UHGG --only s2")
            else:
                print("  Run: python -m persistence --ref UHGG --only s2 --bayesian")
                print("  Or use --allow-legacy-fallback to permit non-Bayesian CSVs.")

    for path in _abund_order:
        if os.path.exists(path):
            abund_df = pd.read_csv(path)
            lab = "hard 10%" if hard_mode else "bayes"
            print(f"  Abundance CSV  [{lab}]: {path} ({len(abund_df)} rows)")
            break
    if abund_df is None:
        if (not hard_mode) and getattr(args, "allow_legacy_fallback", False):
            for path in _abund_legacy:
                if os.path.exists(path):
                    abund_df = pd.read_csv(path)
                    print(f"  Abundance CSV  [LEGACY — non-Bayesian]: {path} ({len(abund_df)} rows)")
                    break
        if abund_df is None:
            csv_kind = "hard 10% abundance" if hard_mode else "Bayesian abundance"
            print(f"  [ERROR] {csv_kind} CSV not found. Tried:")
            for p in _abund_order: print(f"    {p}")
            if hard_mode:
                print("  Run standard Section 2 first: python -m persistence --ref UHGG --only s2")
            else:
                print("  Run: python -m persistence --ref UHGG --only s2 --bayesian")
                print("  Or use --allow-legacy-fallback to permit non-Bayesian CSVs.")

    compose_figure(dict(
        ref=args.ref, null_key=args.null_key,
        out_dir=args.out_dir, hard_mode=hard_mode,
        aVs=aVs,
        aVps=aVps,
        aBs=aBs,
        aBps=aBps,
        aVst=aVst,
        aVpst=aVpst,
        aBst=aBst,
        aBpst=aBpst,
        sI_soft=sI_soft,
        sN_soft=sN_soft,
        V=V,
        B=B,
        Vp=Vp,
        Bp=Bp,
        n_I=n_I,
        n_NI=n_NI,
        typ_I=typ_I,
        typ_NI=typ_NI,
        prev_df=prev_df,
        abund_df=abund_df,
    ))


if __name__ == "__main__":
    main()
