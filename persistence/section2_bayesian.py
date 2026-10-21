"""
section2_bayesian.py — Bayesian-presence variant of Section 2.

Replaces the hard breadth threshold (breadth >= 10%) with per-sample
latent-L posteriors P(L >= tau | b_ij, d_ij, G_j) from the Beta-likelihood
mixture-prior model implemented in rank_abundance_analysis.py.

Detection estimands
-------------------
Two complementary estimands are maintained, matching the accumulation pipeline:

  sample_presence[null_key][i,j] = (p_raw[i,j] >= 0.5)
      Per-sample Boolean call.  Appropriate for Fisher / GMW tests
      (Pass 1, Pass 2, Pass 3, Hadza-specific).

  pooled union: p_union_j = 1 - prod_i(1 - p_raw[i,j])  >= 0.5
      Pooled detection over a group of samples, using the exact same
      cumulative log-survival rule as accumulation_summary_v2.
      _hard_call_accumulation.  Used for 2E pooling burden, 2F continent
      detection, and 2G extinction candidates.

Null threshold registry
-----------------------
BAYES_NULL_REGISTRY maps null_key strings to (f_frac_or_None, kb_or_None)
tuples in the same format as rank_abundance_analysis.NULL_MODELS.  All
thresholds needed by the default Section 2 pooling sweep are listed.

The registry is patched into _ra.NULL_MODELS before calling
_compute_per_sample_posterior, exactly the way accumulation_summary_v2.py
patches in "300kb".  One _compute_per_sample_posterior call then computes
ALL thresholds in a single pass — consistent, efficient, and identical to
the accumulation code.

BayesianDetectionBundle
-----------------------
A BayesianDetectionBundle holds:
  raw             : dict[null_key -> DataFrame[float32](samples x SGBs)]
                    Raw P(L >= tau) posteriors.  Use for pooled union detection.
  sample_presence : dict[null_key -> DataFrame[bool](samples x SGBs)]
                    Per-sample calls (raw >= 0.5).  Use for Fisher/GMW.
  c, sigma_bins, sigma_edges : calibrated noise model parameters.
  pi_by_lifestyle : dict[lifestyle -> float] priors used.

Consistency guarantee
---------------------
Any pooled detection statement produced by Section 2 now uses the same
estimand as accumulation_summary_v2._hard_call_accumulation():
  detected_pooled_j = (1 - prod_i(1 - p_raw[i,j])) >= 0.5
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from scipy.stats import fisher_exact, mannwhitneyu, norm, chi2_contingency

from . import config
from .common import (assign_volcano_category_7, clr_transform)
from .differential import (
    plot_volcano_7, plot_volcano_legacy,
    _save_top100,
)

# ── Richness-informed priors (matching accumulation_summary_v2.py) ─────────────
_RICHNESS_KNOWN = {"Industrial": 500, "Non-Industrial": 800}

# ── Full null registry — covers all Section 2 pooling specs ───────────────────
#
# Format: null_key -> (f_frac_or_None, kb_or_None)
# Mirrors the format of rank_abundance_analysis.NULL_MODELS exactly.
# All entries are patched into _ra.NULL_MODELS before calling
# _compute_per_sample_posterior so that one call computes ALL thresholds.
#
BAYES_NULL_REGISTRY: Dict[str, tuple] = {
    "1pct":   (0.01,  None),
    "3pct":   (0.03,  None),
    "10pct":  (0.10,  None),
    "30pct":  (0.30,  None),
    "50pct":  (0.50,  None),
    "90pct":  (0.90,  None),
    "10kb":   (None,  10),
    "30kb":   (None,  30),
    "100kb":  (None,  100),
    "300kb":  (None,  300),
    "500kb":  (None,  500),
    "1Mb":    (None,  1000),
}
# ── DetectionSpec -> null_key mapping ────────────────────────────────────────
# Two-level lookup:
#
#   Level 1: (kind_str, threshold_float) -> null_key
#     Tries all plausible attribute-name variants for "kind" and "threshold"
#     because DetectionSpec's internal attribute names are not fixed across
#     versions of the pipeline.
#
#   Level 2: short_label -> null_key
#     Falls back to spec.short_label so that even if attribute names change
#     again the mapping still works, provided short_label is stable.
#
# The kind strings and threshold values match DetectionSpec("breadth_pct", 10.0)
# and DetectionSpec("flat_bp", 300_000) as used throughout the pipeline.

_SPEC_TO_NULL_KEY: Dict[tuple, str] = {
    ("breadth_pct",   1.0): "1pct",
    ("breadth_pct",   3.0): "3pct",
    ("breadth_pct",  10.0): "10pct",
    ("breadth_pct",  30.0): "30pct",
    ("breadth_pct",  50.0): "50pct",
    ("breadth_pct",  90.0): "90pct",
    ("flat_bp",   10_000.0): "10kb",
    ("flat_bp",   30_000.0): "30kb",
    ("flat_bp",  100_000.0): "100kb",
    ("flat_bp",  300_000.0): "300kb",
    ("flat_bp",  500_000.0): "500kb",
    ("flat_bp", 1_000_000.0): "1Mb",
}
# Candidate attribute names for the "kind" field of DetectionSpec
_KIND_ATTRS  = ("kind", "spec_type", "type", "detection_type", "method", "mode", "name")
# Candidate attribute names for the "threshold/value" field
_VALUE_ATTRS = ("threshold", "value", "level", "cutoff", "amount", "pct", "bp", "limit")

# Fallback: map spec.short_label -> null_key for common short_label formats
_SHORT_LABEL_TO_NULL_KEY: Dict[str, str] = {
    # breadth-pct variants
    "b1": "1pct",     "b1pct": "1pct",    "1pct": "1pct",
    "b3": "3pct",     "b3pct": "3pct",    "3pct": "3pct",
    "b10": "10pct",   "b10pct": "10pct",  "10pct": "10pct",
    "b30": "30pct",   "b30pct": "30pct",  "30pct": "30pct",
    "b50": "50pct",   "b50pct": "50pct",  "50pct": "50pct",
    "b90": "90pct",   "b90pct": "90pct",  "90pct": "90pct",
    # flat-bp variants
    "10kb": "10kb",   "f10kb": "10kb",    "flat10kb": "10kb",
    "30kb": "30kb",   "f30kb": "30kb",    "flat30kb": "30kb",
    "100kb": "100kb", "f100kb": "100kb",  "flat100kb": "100kb",
    "300kb": "300kb", "f300kb": "300kb",  "flat300kb": "300kb",
    "500kb": "500kb", "f500kb": "500kb",  "flat500kb": "500kb",
    "1mb": "1Mb",     "f1mb": "1Mb",      "flat1mb": "1Mb",
}

def spec_to_null_key(spec) -> str:
    """
    Map a DetectionSpec to a Bayesian null_key string.

    Uses a two-level lookup that is robust to changes in DetectionSpec's
    internal attribute names:
      1. Tries all plausible (kind, value) attribute-name combinations.
      2. Falls back to spec.short_label matched against known short-label formats.

    Raises KeyError with the spec's repr and discovered attributes if both
    levels fail, to help diagnose DetectionSpec changes.
    """
    # Level 1: find kind + value via any plausible attribute name
    kind = next((getattr(spec, a) for a in _KIND_ATTRS if hasattr(spec, a)), None)
    val  = next((getattr(spec, a) for a in _VALUE_ATTRS if hasattr(spec, a)), None)

    if kind is not None and val is not None:
        key = (str(kind), float(val))
        if key in _SPEC_TO_NULL_KEY:
            return _SPEC_TO_NULL_KEY[key]

    # Level 2: match via short_label
    sl = getattr(spec, "short_label", None)
    if sl is not None and str(sl).lower() in _SHORT_LABEL_TO_NULL_KEY:
        return _SHORT_LABEL_TO_NULL_KEY[str(sl).lower()]

    # Diagnostic: show the object's actual attributes to help fix the mapping
    try:
        spec_attrs = {k: getattr(spec, k)
                      for k in dir(spec)
                      if not k.startswith("_") and not callable(getattr(spec, k, None))}
    except Exception:
        spec_attrs = {}
    raise KeyError(
        f"Cannot map DetectionSpec to a Bayesian null_key.\n"
        f"  repr(spec)  = {repr(spec)}\n"
        f"  short_label = {sl!r}\n"
        f"  kind (found via {_KIND_ATTRS}) = {kind!r}\n"
        f"  value (found via {_VALUE_ATTRS}) = {val!r}\n"
        f"  non-private attrs = {spec_attrs}\n"
        f"  Supported (kind, value) pairs: {list(_SPEC_TO_NULL_KEY)}\n"
        f"  Supported short_labels: {list(_SHORT_LABEL_TO_NULL_KEY)}\n"
        f"  To fix: add the correct (kind, value) pair or short_label to the "
        f"mappings in section2_bayesian.py."
    )


# ── BayesianDetectionBundle ───────────────────────────────────────────────────

@dataclass
class BayesianDetectionBundle:
    """
    Complete Bayesian detection state for one reference dataset.

    raw[null_key]            : DataFrame[float32], shape (n_samples, n_SGBs).
                               Raw P(L >= tau | b, d, G) posteriors
                               (subtract_prior=False).
                               Use for pooled union detection.

    sample_presence[null_key]: DataFrame[bool], shape (n_samples, n_SGBs).
                               Per-sample calls: raw[null_key] >= 0.5.
                               Use for Fisher / GMW differential tests.
    """
    raw:             Dict[str, pd.DataFrame]
    sample_presence: Dict[str, pd.DataFrame]
    c:               float
    sigma_bins:      np.ndarray
    sigma_edges:     np.ndarray
    pi_by_lifestyle: Dict[str, float]


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC IMPORT OF BAYESIAN HELPERS FROM rank_abundance_analysis.py
# ══════════════════════════════════════════════════════════════════════════════

def _load_ra_module(explicit_path: Optional[str] = None):
    """
    Load rank_abundance_analysis.py into a fresh module namespace.

    Uses the same exec-stub technique as accumulation_summary_v2.py: the
    _resolve_pipeline_imports() call at module level is patched out so the
    file can be exec'd without a second pipeline bootstrap.
    """
    if explicit_path is None:
        explicit_path = os.environ.get("RA_MODULE_PATH", None)

    sd = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        [explicit_path] if explicit_path
        else [
            os.path.join(sd,                  "rank_abundance_analysis.py"),
            os.path.join(os.path.dirname(sd), "rank_abundance_analysis.py"),
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            spec    = importlib.util.spec_from_file_location("_ra_bayes", path)
            mod     = importlib.util.module_from_spec(spec)
            ra_src  = open(path).read()
            ra_src  = re.sub(
                r'^[^\n]*=\s*_resolve_pipeline_imports\(\)[^\n]*$',
                'load_reference = None; _cfg = None  # stubbed',
                ra_src, flags=re.MULTILINE,
            )
            exec(compile(ra_src, path, "exec"), mod.__dict__)
            print(f"    [bayes] Loaded RA helpers from {os.path.abspath(path)}")
            return mod
    raise FileNotFoundError(
        "rank_abundance_analysis.py not found.\n"
        "Set RA_MODULE_PATH or pass ra_module_path= to build_bayesian_detection_bundle()."
    )


_RA_MODULE = None

def _get_ra(ra_module_path=None):
    global _RA_MODULE
    if _RA_MODULE is None:
        _RA_MODULE = _load_ra_module(ra_module_path)
    return _RA_MODULE


def _patch_null_models(ra) -> None:
    """
    Patch _ra.NULL_MODELS with every entry in BAYES_NULL_REGISTRY.

    Mirrors the technique in accumulation_summary_v2.py:
        for _k, _v in BAYES_NULL_REGISTRY.items():
            _ra.NULL_MODELS.setdefault(_k, _v)
    Applied to all keys in BAYES_NULL_REGISTRY so that a single call to
    _compute_per_sample_posterior computes every threshold we need.
    """
    added = []
    for key, val in BAYES_NULL_REGISTRY.items():
        if key not in ra.NULL_MODELS:
            ra.NULL_MODELS[key] = val
            added.append(key)
    if added:
        print(f"    [bayes] Patched _ra.NULL_MODELS: added {added}")


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE-MODEL CALIBRATION (once per reference dataset)
# ══════════════════════════════════════════════════════════════════════════════

def _calibrate(data, ra) -> tuple:
    """
    Estimate coverage efficiency c and depth-stratified breadth noise model.

    Uses all samples (both lifestyles) for a stable global estimate —
    the same strategy as load_and_prep() in accumulation_summary_v2.py.

    Returns (c, sigma_bins, sigma_edges, bfrac_all, depth_all).
    """
    bfrac = data.breadth.copy()
    raw_max = float(bfrac.values[bfrac.values > 0].max()) if (bfrac.values > 0).any() else 0.0
    if raw_max > 1.0:
        bfrac = bfrac / 100.0

    depth = data.depth.copy()
    samps = bfrac.index.intersection(depth.index).intersection(data.meta.index)
    bfrac = bfrac.loc[samps]
    depth = depth.loc[samps, bfrac.columns]

    c = ra._estimate_coverage_efficiency(bfrac, depth)
    sigma_bins, sigma_edges = ra._estimate_sigma_b(bfrac, depth, c=c)
    print(f"    [bayes] c={c:.4f}  sigma_b [{sigma_bins.min():.4f}, {sigma_bins.max():.4f}]")
    return c, sigma_bins, sigma_edges, bfrac, depth


# ══════════════════════════════════════════════════════════════════════════════
#  CORE: BUILD BAYESIAN DETECTION BUNDLE
# ══════════════════════════════════════════════════════════════════════════════

def build_bayesian_detection_bundle(
    data,
    null_keys:      List[str]       = None,
    ra_module_path: Optional[str]   = None,
    pi_industrial:  Optional[float] = None,
    pi_ni:          Optional[float] = None,
    min_map_bp:     int             = 10_000,
) -> BayesianDetectionBundle:
    """
    Build a BayesianDetectionBundle: raw posteriors + sample-level calls.

    This is the single shared backend for all Bayesian Section 2 analyses.
    It uses identical mechanics to accumulation_summary_v2.compute_posteriors():

      * Calibrates c and sigma_b once on ALL samples.
      * Patches _ra.NULL_MODELS with BAYES_NULL_REGISTRY so that ONE call to
        _compute_per_sample_posterior computes every requested threshold.
      * Uses subtract_prior=False (raw posteriors, correct for union detector).
      * Derives sample_presence by thresholding raw >= 0.5, same as
        _DETECT_THRESH = 0.5 in accumulation_summary_v2.

    Sample grouping uses data.meta["MacroGroup"] (values "Industrial" /
    "Non-Industrial"), matching accumulation_summary_v2._run_posteriors_parallel.
    Falls back to Lifestyle if MacroGroup is absent.

    Pi priors are clipped to [0.01, 0.95], identical to
    accumulation_summary_v2._richness_pi(pi_mode='fixed').

    Parameters
    ----------
    data           : ReferenceData
    null_keys      : list of null_key strings to compute.  Defaults to all
                     keys in BAYES_NULL_REGISTRY.
    ra_module_path : explicit path to rank_abundance_analysis.py.
    pi_industrial  : prior P(present) for Industrial.  Default clip(500/S_ref).
    pi_ni          : prior P(present) for Non-Industrial.  Default clip(800/S_ref).
    min_map_bp     : minimum mapped bases for a non-zero posterior.
    """
    ra = _get_ra(ra_module_path)

    # Patch _ra.NULL_MODELS BEFORE any posterior call — same as accumulation_summary_v2
    _patch_null_models(ra)

    if null_keys is None:
        null_keys = list(BAYES_NULL_REGISTRY.keys())

    missing = [k for k in null_keys if k not in ra.NULL_MODELS]
    if missing:
        raise KeyError(
            f"null_keys {missing} are not in ra.NULL_MODELS after patching. "
            f"Available: {list(ra.NULL_MODELS)}"
        )

    S_ref     = data.breadth.shape[1]
    pi_ind    = float(np.clip(
        pi_industrial if pi_industrial is not None else _RICHNESS_KNOWN["Industrial"]  / S_ref,
        0.01, 0.95))
    pi_ni_val = float(np.clip(
        pi_ni         if pi_ni         is not None else _RICHNESS_KNOWN["Non-Industrial"] / S_ref,
        0.01, 0.95))

    pi_by_lifestyle = {"Industrial": pi_ind, "Non-Industrial": pi_ni_val}

    print(f"\n    [bayes] Building detection bundle")
    print(f"           null_keys={null_keys}")
    print(f"           pi_ind={pi_ind:.4f}  pi_ni={pi_ni_val:.4f}")

    c, sigma_bins, sigma_edges, bfrac_all, depth_all = _calibrate(data, ra)

    genome_lengths = {str(k): int(v) for k, v in data.lengths.items()}

    meta = data.meta
    # Canonical group masks — Lifestyle is the single source of truth
    # across build_bayesian_detection_bundle, soft_prev export, and
    # section2_bayesian_figure.py recomputation.
    _GROUP_LIFESTYLE = {
        "Industrial":     ["Industrial"],
        "Non-Industrial": ["H-G", "Transitional"],
    }

    # parts[null_key] = list of per-lifestyle DataFrames
    parts: Dict[str, list] = {k: [] for k in null_keys}

    for ls_label, pi_val in [
        ("Industrial",     pi_ind),
        ("Non-Industrial", pi_ni_val),
    ]:
        mask = meta["Lifestyle"].isin(_GROUP_LIFESTYLE[ls_label])

        samps = meta.index[mask & meta.index.isin(bfrac_all.index)]
        if len(samps) == 0:
            print(f"    [bayes] {ls_label}: 0 samples — skipping")
            continue

        bfrac_s = bfrac_all.loc[samps]
        depth_s = depth_all.loc[samps, bfrac_all.columns]

        print(f"    [bayes] {ls_label}: n={len(samps)}, pi={pi_val:.4f} ...", flush=True)

        # _compute_per_sample_posterior iterates over ALL keys in ra.NULL_MODELS
        # (now patched with the full registry), computing every threshold in one pass.
        p_dict = ra._compute_per_sample_posterior(
            bfrac_s        = bfrac_s,
            depth_s        = depth_s,
            c              = c,
            sigma_bins     = sigma_bins,
            sigma_edges    = sigma_edges,
            genome_lengths = genome_lengths,
            pi_present     = pi_val,
            min_map_bp     = min_map_bp,
            subtract_prior = False,   # raw posteriors — consistent with accumulation
        )

        for k in null_keys:
            if k not in p_dict:
                raise KeyError(
                    f"null_key='{k}' not in posterior dict. "
                    f"Available: {list(p_dict.keys())}"
                )
            parts[k].append(p_dict[k])

    # Concatenate lifestyles; reindex to full breadth index (missing samples -> 0.0)
    full_idx = data.breadth.index
    all_cols  = data.breadth.columns

    raw: Dict[str, pd.DataFrame] = {}
    sample_presence: Dict[str, pd.DataFrame] = {}

    for k in null_keys:
        if not parts[k]:
            df = pd.DataFrame(
                np.zeros((len(full_idx), len(all_cols)), dtype=np.float32),
                index=full_idx, columns=all_cols,
            )
        else:
            df = pd.concat(parts[k], axis=0)
        df = (df
              .reindex(index=full_idx, fill_value=np.float32(0))
              .reindex(columns=all_cols, fill_value=np.float32(0))
              .fillna(np.float32(0))
              .astype(np.float32))
        raw[k] = df
        sample_presence[k] = df >= np.float32(0.5)

        n_det = int(sample_presence[k].any(axis=0).sum())
        print(f"    [bayes] [{k}] {n_det}/{S_ref} SGBs detected in >=1 sample "
              f"({100*n_det/max(S_ref,1):.1f}%)")

    return BayesianDetectionBundle(
        raw=raw,
        sample_presence=sample_presence,
        c=c,
        sigma_bins=sigma_bins,
        sigma_edges=sigma_edges,
        pi_by_lifestyle=pi_by_lifestyle,
    )


def build_bayesian_presence(
    data,
    null_key:       str            = "10pct",
    pi_industrial:  Optional[float] = None,
    pi_ni:          Optional[float] = None,
    ra_module_path: Optional[str]  = None,
    min_map_bp:     int            = 10_000,
) -> pd.DataFrame:
    """
    Convenience wrapper: build a bundle for one null_key, return sample_presence.

    Kept for backward compatibility with callers that only need one Boolean matrix.
    """
    bundle = build_bayesian_detection_bundle(
        data           = data,
        null_keys      = [null_key],
        ra_module_path = ra_module_path,
        pi_industrial  = pi_industrial,
        pi_ni          = pi_ni,
        min_map_bp     = min_map_bp,
    )
    return bundle.sample_presence[null_key]


# ══════════════════════════════════════════════════════════════════════════════
#  DIFFERENTIAL TESTS  (sample-level — appropriate for Fisher/GMW)
# ══════════════════════════════════════════════════════════════════════════════

def _test_prevalence_bayes(
    presence_df: pd.DataFrame,
    meta:        pd.DataFrame,
    g1_mask_fn,
    g2_mask_fn,
    g1_label:   str,
    g2_label:   str,
) -> pd.DataFrame:
    g1 = meta.index[g1_mask_fn(meta)]
    g2 = meta.index[g2_mask_fn(meta)]
    g1 = g1[g1.isin(presence_df.index)]
    g2 = g2[g2.isin(presence_df.index)]
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return pd.DataFrame()

    print(f"    [bayes] Fisher prevalence: {g1_label} (n={n1:,}) vs {g2_label} (n={n2:,})")

    pres1 = presence_df.loc[g1].values.astype(np.int8)
    pres2 = presence_df.loc[g2].values.astype(np.int8)

    results = []
    for ji, mic in enumerate(presence_df.columns):
        d1 = int(pres1[:, ji].sum())
        d2 = int(pres2[:, ji].sum())
        if d1 == 0 and d2 == 0:
            continue
        odds, pval = fisher_exact([[d1, n1 - d1], [d2, n2 - d2]])
        results.append({
            "microbe":         mic,
            "prev_g1":         d1 / n1,
            "prev_g2":         d2 / n2,
            "prevalence_diff": d1 / n1 - d2 / n2,
            "odds_ratio":      odds,
            "pval":            pval,
            "log10_pval":      np.log10(max(pval, 1e-300)),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    n_tests = len(df)
    df["neg_log10_p_bonf"] = np.maximum(-df["log10_pval"] - np.log10(n_tests), 0.0)
    return df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)


def _test_prevalence_weighted_bayes(
    presence_df: pd.DataFrame,
    g1_idx,
    g2_idx,
    w1:          np.ndarray,
    w2:          np.ndarray,
    g1_label:    str,
    g2_label:    str,
) -> pd.DataFrame:
    g1_idx = list(g1_idx)
    g2_idx = list(g2_idx)
    g1_in  = [i for i in g1_idx if i in presence_df.index]
    g2_in  = [i for i in g2_idx if i in presence_df.index]
    g1_pos_map = {idx: pos for pos, idx in enumerate(g1_idx)}
    g2_pos_map = {idx: pos for pos, idx in enumerate(g2_idx)}
    w1_f = np.array([w1[g1_pos_map[i]] for i in g1_in], dtype=np.float64)
    w2_f = np.array([w2[g2_pos_map[i]] for i in g2_in], dtype=np.float64)

    W1 = float(w1_f.sum()); W2 = float(w2_f.sum())
    if W1 == 0 or W2 == 0:
        return pd.DataFrame()

    ess1 = W1**2 / (w1_f**2).sum()
    ess2 = W2**2 / (w2_f**2).sum()
    print(f"    [bayes] Weighted prevalence: {g1_label} "
          f"(ESS~{ess1:.0f}/{len(g1_in)}) vs {g2_label} (ESS~{ess2:.0f}/{len(g2_in)})")

    pres1 = presence_df.loc[g1_in].values.astype(np.float64)
    pres2 = presence_df.loc[g2_in].values.astype(np.float64)

    results = []
    for ji, mic in enumerate(presence_df.columns):
        p1 = pres1[:, ji]; p2 = pres2[:, ji]
        a  = float(np.dot(w1_f, p1))
        b_ = float(np.dot(w1_f, 1.0 - p1))
        c_ = float(np.dot(w2_f, p2))
        d_ = float(np.dot(w2_f, 1.0 - p2))
        if a + b_ < 1e-9 or c_ + d_ < 1e-9:
            continue
        if a + c_ == 0 or b_ + d_ == 0:
            continue
        try:
            chi2, pval, _, _ = chi2_contingency([[a, b_], [c_, d_]], correction=False)
        except ValueError:
            continue
        prev_g1 = a / (a + b_); prev_g2 = c_ / (c_ + d_)
        results.append({
            "microbe":         mic,
            "prev_g1":         prev_g1,
            "prev_g2":         prev_g2,
            "prevalence_diff": prev_g1 - prev_g2,
            "chi2":            chi2,
            "pval":            pval,
            "log10_pval":      np.log10(max(pval, 1e-300)),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    n_tests = len(df)
    df["neg_log10_p_bonf"] = np.maximum(-df["log10_pval"] - np.log10(n_tests), 0.0)

    # High-precision Bonferroni -log10(p) using norm.logsf(sqrt(chi2)).
    # chi2(df=1) identity: P(chi2 > x) = 2*norm.sf(sqrt(x))
    # norm.logsf is numerically stable for all finite inputs — no float64 cap.
    from scipy.stats import norm as _norm_sp
    _log10e = np.log10(np.e)
    _log_p_raw = np.log(2.0) + _norm_sp.logsf(
        np.sqrt(np.maximum(df["chi2"].values.astype(np.float64), 0.0)))
    df["neg_log10_p_bonf_exact"] = np.maximum(
        (-_log_p_raw * _log10e) - np.log10(n_tests), 0.0)

    return df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)



def _weighted_mwu_log10p(x, y, w1, w2):
    """
    Weighted Mann-Whitney / Wilcoxon rank-sum test.

    Parameters
    ----------
    x, y  : 1D arrays of per-sample values for the two groups
    w1,w2 : non-negative sample weights on the same samples

    Returns
    -------
    auc_w      : weighted probability that x > y (plus 0.5 for ties)
    z_stat     : two-sided normal approximation under H0: auc_w = 0.5
    log10_pval : base-10 log p-value (full precision via norm.logsf)
    pval       : float p-value (0.0 if below float range)
    n1_eff     : Kish effective sample size for group 1
    n2_eff     : Kish effective sample size for group 2

    Uses Kish ESS in the null variance so it does not collapse when many
    posteriors are near 0 or 1.  Tie correction uses the standard unweighted
    tie factor as a practical approximation.
    """
    x  = np.asarray(x,  dtype=np.float64)
    y  = np.asarray(y,  dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)

    m1 = np.isfinite(x)  & np.isfinite(w1) & (w1 > 0)
    m2 = np.isfinite(y)  & np.isfinite(w2) & (w2 > 0)
    x, y, w1, w2 = x[m1], y[m2], w1[m1], w2[m2]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    W1 = float(w1.sum()); W2 = float(w2.sum())
    if W1 <= 0 or W2 <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Normalised within-group weights for the rank statistic
    a = w1 / W1
    b = w2 / W2

    # Kish effective sample sizes for the null variance
    n1_eff = float(W1 ** 2 / np.dot(w1, w1))
    n2_eff = float(W2 ** 2 / np.dot(w2, w2))
    if n1_eff <= 1 or n2_eff <= 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    vals  = np.concatenate([x, y])
    grp   = np.concatenate([np.zeros(len(x), dtype=np.int8),
                             np.ones (len(y), dtype=np.int8)])
    wt    = np.concatenate([a, b])
    order = np.argsort(vals, kind="mergesort")
    vals_o, grp_o, wt_o = vals[order], grp[order], wt[order]

    # Weighted AUC via tie blocks
    b_below  = 0.0
    auc_w    = 0.0
    tie_term = 0.0
    i = 0; n = len(vals_o)
    while i < n:
        j = i + 1
        while j < n and vals_o[j] == vals_o[i]:
            j += 1
        g_block = grp_o[i:j]; w_block = wt_o[i:j]
        x_block = float(w_block[g_block == 0].sum())
        y_block = float(w_block[g_block == 1].sum())
        auc_w  += x_block * (b_below + 0.5 * y_block)
        t = j - i
        if t > 1:
            tie_term += t ** 3 - t
        b_below += y_block
        i = j

    # Null variance with Kish ESS and tie correction
    N      = len(x) + len(y)
    tie_cf = 1.0
    if N > 1:
        denom = float(N ** 3 - N)
        if denom > 0:
            tie_cf = max(1.0 - tie_term / denom, 1e-12)
    var0   = float(max(tie_cf * (n1_eff + n2_eff + 1.0) / (12.0 * n1_eff * n2_eff), 1e-12))
    z_stat = (auc_w - 0.5) / np.sqrt(var0)

    log10_pval = float(np.log10(2.0) + norm.logsf(abs(z_stat)) / np.log(10.0))
    tiny_log10 = np.log10(np.finfo(np.float64).tiny)
    pval = 0.0 if (not np.isfinite(log10_pval) or log10_pval < tiny_log10)            else float(10.0 ** log10_pval)

    return auc_w, z_stat, log10_pval, pval, n1_eff, n2_eff



def _test_prevalence_weighted_posterior_bayes(
    posterior_df: pd.DataFrame,
    g1_idx,
    g2_idx,
    w1: np.ndarray,
    w2: np.ndarray,
    g1_label: str,
    g2_label: str,
) -> pd.DataFrame:
    """
    GMW-weighted rank test (weighted MWU) on posterior detection probabilities.

    Effect size:  weighted mean posterior difference (prev_g1 - prev_g2)
    Significance: _weighted_mwu_log10p on per-sample posterior probabilities
                  with the same GMW weights — uses Kish ESS in the null
                  variance, so p-values do not explode when posteriors cluster
                  near 0 or 1.
    """
    g1_idx = list(g1_idx)
    g2_idx = list(g2_idx)
    g1_in  = [i for i in g1_idx if i in posterior_df.index]
    g2_in  = [i for i in g2_idx if i in posterior_df.index]

    g1_pos = {idx: pos for pos, idx in enumerate(g1_idx)}
    g2_pos = {idx: pos for pos, idx in enumerate(g2_idx)}
    w1_f   = np.asarray([w1[g1_pos[i]] for i in g1_in], dtype=np.float64)
    w2_f   = np.asarray([w2[g2_pos[i]] for i in g2_in], dtype=np.float64)

    W1 = float(w1_f.sum()); W2 = float(w2_f.sum())
    if W1 <= 0 or W2 <= 0:
        return pd.DataFrame()

    wn1  = w1_f / W1
    wn2  = w2_f / W2
    ess1 = W1 ** 2 / float(np.dot(w1_f, w1_f))
    ess2 = W2 ** 2 / float(np.dot(w2_f, w2_f))
    print(f"    [bayes] Weighted MWU on posterior prevalence: {g1_label} "
          f"(ESS~{ess1:.0f}/{len(g1_in)}) vs {g2_label} (ESS~{ess2:.0f}/{len(g2_in)})")

    P1 = np.clip(posterior_df.loc[g1_in].to_numpy(dtype=np.float64), 0.0, 1.0)
    P2 = np.clip(posterior_df.loc[g2_in].to_numpy(dtype=np.float64), 0.0, 1.0)

    results = []
    for ji, mic in enumerate(posterior_df.columns):
        p1 = P1[:, ji]
        p2 = P2[:, ji]

        prev_g1 = float(np.dot(wn1, p1))
        prev_g2 = float(np.dot(wn2, p2))
        diff    = prev_g1 - prev_g2

        auc_w, z_stat, log10p, pval, _, _ = _weighted_mwu_log10p(p1, p2, w1_f, w2_f)
        if not np.isfinite(log10p):
            continue

        results.append({
            "microbe":         mic,
            "prev_g1":         prev_g1,
            "prev_g2":         prev_g2,
            "prevalence_diff": diff,
            "mw_auc_w":        auc_w,
            "z_stat":          z_stat,
            "pval":            pval,
            "log10_pval":      log10p,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    n_tests = len(df)
    df["neg_log10_p_bonf"]       = np.maximum(-df["log10_pval"] - np.log10(n_tests), 0.0)
    df["neg_log10_p_bonf_exact"] = df["neg_log10_p_bonf"]
    return df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)


def _test_prevalence_posterior_bayes(
    posterior_df: pd.DataFrame,
    meta:         pd.DataFrame,
    g1_mask_fn,
    g2_mask_fn,
    g1_label:    str,
    g2_label:    str,
) -> pd.DataFrame:
    """Unweighted wrapper: calls the weighted MWU with unit weights."""
    g1 = meta.index[g1_mask_fn(meta)]
    g2 = meta.index[g2_mask_fn(meta)]
    return _test_prevalence_weighted_posterior_bayes(
        posterior_df = posterior_df,
        g1_idx       = g1,
        g2_idx       = g2,
        w1           = np.ones(len(g1), dtype=np.float64),
        w2           = np.ones(len(g2), dtype=np.float64),
        g1_label     = g1_label,
        g2_label     = g2_label,
    )


def _run_abundance_gmw_bayes(data, mask_g1, mask_g2, g1_label, g2_label,
                              vanish_plus_ids, blossum_plus_ids):
    """
    GMW-weighted CLR abundance test on the full cohort.

    Mirrors _run_differential_gmw_bayes but for CLR abundance:
      - Uses all Ind / NI samples (not just depth-matched subset)
      - Builds GMW IPW weights from _log_depth (same as prevalence)
      - Effect size: GMW-weighted mean CLR difference (g1 - g2)
      - Significance: GMW-weighted MWU (_weighted_mwu_log10p) — same helper
        as prevalence, using GMW weights, consistent across both volcanoes
      - p-values computed via norm.logsf for unlimited precision

    Writes: bayes_abundance_ind_vs_nonind.csv
    """
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights

    meta = data.meta
    ref  = data.ref_name
    out  = config.outdir(ref)

    g1_idx = meta.index[mask_g1 & meta.index.isin(data.abundance.index)]
    g2_idx = meta.index[mask_g2 & meta.index.isin(data.abundance.index)]

    if len(g1_idx) < 20 or len(g2_idx) < 20:
        print(f"    [bayes abund GMW] Too few: {g1_label}={len(g1_idx)}, "
              f"{g2_label}={len(g2_idx)} — skipping")
        return pd.DataFrame()

    g1_all = list(g1_idx); g2_all = list(g2_idx)
    w1 = np.ones(len(g1_all)); w2 = np.ones(len(g2_all))

    ld1 = meta["_log_depth"].reindex(g1_all)
    ld2 = meta["_log_depth"].reindex(g2_all)
    g1_hd = [i for i in g1_all if pd.notna(ld1.loc[i])]
    g2_hd = [i for i in g2_all if pd.notna(ld2.loc[i])]

    if len(g1_hd) >= 20 and len(g2_hd) >= 20:
        a1 = ld1.loc[g1_hd].values.astype(float)
        a2 = ld2.loc[g2_hd].values.astype(float)
        if len(np.unique(a1)) >= 3 and len(np.unique(a2)) >= 3:
            kde1 = _BinnedKDE(a1); kde2 = _BinnedKDE(a2)
            w1k  = _geomean_weights(a1, kde1, kde2)
            w2k  = _geomean_weights(a2, kde2, kde1)
            g1p  = {idx: pos for pos, idx in enumerate(g1_hd)}
            g2p  = {idx: pos for pos, idx in enumerate(g2_hd)}
            for pos, idx in enumerate(g1_all):
                if idx in g1p: w1[pos] = w1k[g1p[idx]]
            for pos, idx in enumerate(g2_all):
                if idx in g2p: w2[pos] = w2k[g2p[idx]]

    W1 = w1.sum(); W2 = w2.sum()
    ess1 = W1**2 / (w1**2).sum()
    ess2 = W2**2 / (w2**2).sum()
    print(f"    [bayes abund GMW] {g1_label} (n={len(g1_all)}, ESS={ess1:.0f}) "
          f"vs {g2_label} (n={len(g2_all)}, ESS={ess2:.0f})")

    # CLR transform on union of all samples for a shared reference
    all_idx = g1_all + g2_all
    clr = clr_transform(data.abundance.loc[all_idx])

    w1_normed = w1 / W1
    w2_normed = w2 / W2

    results = []
    for mic in clr.columns:
        x = clr.loc[g1_all, mic].values.astype(float)
        y = clr.loc[g2_all, mic].values.astype(float)
        if np.isnan(x).all() or np.isnan(y).all():
            continue
        # GMW-weighted mean CLR difference as effect size
        clr_diff = float(np.nansum(w1_normed * x) - np.nansum(w2_normed * y))

        # GMW-weighted MWU — same test as prevalence, consistent across volcanoes
        auc_w, z_stat, log10p, pval, _, _ = _weighted_mwu_log10p(x, y, w1, w2)
        if not np.isfinite(log10p):
            continue

        results.append({
            "microbe":    mic,
            "clr_diff":   clr_diff,
            "mw_auc_w":   auc_w,
            "z_stat":     z_stat,
            "pval":       pval,
            "log10_pval": log10p,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    n_tests = len(df)
    df["neg_log10_p_bonf"]       = np.maximum(-df["log10_pval"] - np.log10(n_tests), 0.0)
    df["neg_log10_p_bonf_exact"] = df["neg_log10_p_bonf"]

    vp_set = set(vanish_plus_ids); bp_set = set(blossum_plus_ids)
    df["category_7"] = df["microbe"].apply(
        lambda m: assign_volcano_category_7(
            m, data.vanish_set, data.blossum_set, vp_set, bp_set))
    df = df.sort_values("clr_diff", ascending=False).reset_index(drop=True)
    df.to_csv(f"{out}/bayes_abundance_ind_vs_nonind.csv", index=False)

    plot_volcano_7(
        df, "clr_diff",
        data.vanish_set, data.blossum_set, vp_set, bp_set,
        f"{ref}: GMW CLR Abundance (Industrial - Non-Industrial) [Bayesian detection]",
        f"{out}/bayes_volcano_abundance_7col.png",
        xlabel="GMW-weighted CLR difference")

    print(f"    [bayes abund GMW] {n_tests} SGBs tested, written to bayes_abundance_ind_vs_nonind.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PASS 1 — DEPTH-MATCHED DIFFERENTIAL  (Bayesian)
# ══════════════════════════════════════════════════════════════════════════════

def _run_differential_bayes(data, matched_idx, presence_df, posterior_df=None):
    ref = data.ref_name
    out = config.outdir(ref)

    meta_m  = data.meta.loc[matched_idx]
    abund_m = data.abundance.loc[matched_idx]
    pres_m  = presence_df.reindex(matched_idx)
    post_m  = posterior_df.reindex(matched_idx) if posterior_df is not None else pres_m

    ni  = lambda df: df["Lifestyle"].isin(["H-G", "Transitional"])
    ind = lambda df: df["Lifestyle"] == "Industrial"

    # Use posterior z-test for Pass 1 (soft prevalence, not Fisher on hard calls)
    prev = _test_prevalence_posterior_bayes(post_m, meta_m, ind, ni, "Industrial", "Non-Industrial")

    vanish_plus_ids  = []
    blossum_plus_ids = []

    if not prev.empty:
        n = len(prev)
        if n >= 100:
            vanish_plus_ids  = prev.tail(100)["microbe"].tolist()
            blossum_plus_ids = prev.head(100)["microbe"].tolist()
            _save_top100(vanish_plus_ids,  f"{out}/bayes_top100_NI_enriched_prevalence.txt")
            _save_top100(blossum_plus_ids, f"{out}/bayes_top100_Ind_enriched_prevalence.txt")

        vp_set = set(vanish_plus_ids); bp_set = set(blossum_plus_ids)
        prev["category_7"] = prev["microbe"].apply(
            lambda m: assign_volcano_category_7(
                m, data.vanish_set, data.blossum_set, vp_set, bp_set))
        prev.to_csv(f"{out}/bayes_prevalence_ind_vs_nonind.csv", index=False)

        plot_volcano_7(
            prev, "prevalence_diff",
            data.vanish_set, data.blossum_set, vp_set, bp_set,
            f"{ref}: Bayesian Prevalence (Industrial - Non-Industrial)",
            f"{out}/bayes_volcano_prevalence_7col.png",
            xlabel="Bayesian delta_p  (Industrial - Non-Industrial)")

        plot_volcano_legacy(
            prev, "prevalence_diff",
            data.vanish_set, data.blossum_set,
            f"{ref}: Bayesian Prevalence (Industrial - Non-Industrial)",
            f"{out}/bayes_volcano_prevalence.png",
            xlabel="Bayesian delta_p")

    abund = pd.DataFrame()   # abundance now computed GMW full-cohort in orchestrator
    return prev, abund, vanish_plus_ids, blossum_plus_ids


# ══════════════════════════════════════════════════════════════════════════════
#  PASS 2 / 3 — GMW WEIGHTED DIFFERENTIAL  (Bayesian)
# ══════════════════════════════════════════════════════════════════════════════

def _run_differential_gmw_bayes(
    data, detect_df, mask_g1, mask_g2, g1_label, g2_label, out_suffix, top_n=100
):
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights

    meta = data.meta
    ref  = data.ref_name
    out  = config.outdir(ref)

    g1_idx = meta.index[mask_g1 & meta.index.isin(detect_df.index)]
    g2_idx = meta.index[mask_g2 & meta.index.isin(detect_df.index)]

    if len(g1_idx) < 20 or len(g2_idx) < 20:
        print(f"    [bayes GMW] Too few samples: {g1_label}={len(g1_idx)}, "
              f"{g2_label}={len(g2_idx)} — skipping")
        return pd.DataFrame(), [], []

    g1_all = list(g1_idx); g2_all = list(g2_idx)
    w1 = np.ones(len(g1_all)); w2 = np.ones(len(g2_all))

    ld1 = meta["_log_depth"].reindex(g1_all)
    ld2 = meta["_log_depth"].reindex(g2_all)
    g1_hd = [i for i in g1_all if pd.notna(ld1.loc[i])]
    g2_hd = [i for i in g2_all if pd.notna(ld2.loc[i])]

    if len(g1_hd) >= 20 and len(g2_hd) >= 20:
        a1 = ld1.loc[g1_hd].values.astype(float)
        a2 = ld2.loc[g2_hd].values.astype(float)
        if len(np.unique(a1)) >= 3 and len(np.unique(a2)) >= 3:
            kde1 = _BinnedKDE(a1); kde2 = _BinnedKDE(a2)
            w1k  = _geomean_weights(a1, kde1, kde2)
            w2k  = _geomean_weights(a2, kde2, kde1)
            g1p  = {idx: pos for pos, idx in enumerate(g1_hd)}
            g2p  = {idx: pos for pos, idx in enumerate(g2_hd)}
            for pos, idx in enumerate(g1_all):
                if idx in g1p: w1[pos] = w1k[g1p[idx]]
            for pos, idx in enumerate(g2_all):
                if idx in g2p: w2[pos] = w2k[g2p[idx]]

    prev_df = _test_prevalence_weighted_posterior_bayes(
        detect_df, g1_all, g2_all, w1, w2, g1_label, g2_label)

    if prev_df.empty:
        return prev_df, [], []

    prev_df.to_csv(os.path.join(out, f"bayes_prevalence_gmw_{out_suffix}.csv"), index=False)

    n = min(top_n, len(prev_df))
    g2_en = prev_df.tail(n)["microbe"].tolist()
    g1_en = prev_df.head(n)["microbe"].tolist()
    _save_top100(g2_en, os.path.join(out, f"bayes_top{n}_g2_enriched_gmw_{out_suffix}.txt"))
    _save_top100(g1_en, os.path.join(out, f"bayes_top{n}_g1_enriched_gmw_{out_suffix}.txt"))
    print(f"    [bayes GMW {out_suffix}] g2-enriched: {len(g2_en)}, g1-enriched: {len(g1_en)}")
    return prev_df, g1_en, g2_en


# ══════════════════════════════════════════════════════════════════════════════
#  2A ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_differential_with_stats_bayes(data, dm, stats, presence_df,
                                       posterior_df=None, main_specs=None):
    ref     = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")

    matched_idx = dm.all_matched_ind_vs_ni if dm else list(data.meta.index)
    n_ind = len(dm.ind_vs_ni_ind) if dm else 0
    n_ni  = len(dm.ind_vs_ni_ni)  if dm else 0

    print(f"\n    [bayes Pass 1] Depth-matched MWU on posterior prevalence: "
          f"Ind={n_ind}, NI={n_ni}, total={len(matched_idx)}")
    stats.add("bayes_differential_n_matched", "s2", "main",
              "Depth-matched samples for Bayesian weighted-MWU differential",
              f"Industrial={n_ind}, NI={n_ni}", n1=n_ind, n2=n_ni)

    prev_df, abund_df, _vp, _bp = _run_differential_bayes(
        data, matched_idx, presence_df, posterior_df=posterior_df)

    if prev_df is not None and not prev_df.empty:
        n_sig = (prev_df["neg_log10_p_bonf"] > -np.log10(0.05)).sum()
        stats.add("bayes_n_sig_differential_prevalence", "s2", "main",
                  "SGBs significant in Bayesian prevalence differential (Bonferroni p<0.05)",
                  n_sig)

    print("\n    [bayes Pass 2] GMW Ind(all) vs NI")
    meta  = data.meta
    ni_m  = meta["Lifestyle"].isin(["H-G", "Transitional"])
    ind_m = meta["Lifestyle"] == "Industrial"

    _gmw_df = posterior_df if posterior_df is not None else presence_df
    _, blossum_plus_ids, vanish_plus_ids = _run_differential_gmw_bayes(
        data, _gmw_df,  ind_m, ni_m,
        "Industrial (all)", "Non-Industrial", "ind_all_vs_ni")

    stats.add("bayes_n_vanish_plus", "s2", "main",
              "VANISH+ taxa (Bayesian GMW Ind-all vs NI)", len(vanish_plus_ids))
    stats.add("bayes_n_blossum_plus", "s2", "main",
              "BloSSUM+ taxa (Bayesian GMW Ind-all vs NI)", len(blossum_plus_ids))

    # ── GMW full-cohort abundance (consistent with prevalence) ───────────────
    print("\n    [bayes abund] GMW CLR abundance — full cohort Ind vs NI")
    abund_df = _run_abundance_gmw_bayes(
        data, ind_m, ni_m,
        "Industrial (all)", "Non-Industrial",
        vanish_plus_ids, blossum_plus_ids)
    if abund_df is not None and not abund_df.empty:
        n_sig_a = (abund_df["neg_log10_p_bonf"] > -np.log10(0.05)).sum()
        stats.add("bayes_n_sig_abundance_gmw", "s2", "main",
                  "SGBs significant in GMW abundance differential (Bonferroni p<0.05)",
                  n_sig_a)

    print("\n    [bayes Pass 3] GMW Ind(healthy) vs NI")
    ind_h_m = ind_m & (meta["Unhealthy"] == "Healthy")
    print(f"    Ind-Healthy n={int(ind_h_m.sum())}, NI n={int(ni_m.sum())}")

    _, blossum_h_ids, vanish_h_ids = _run_differential_gmw_bayes(
        data, _gmw_df, ind_h_m, ni_m,
        "Industrial (healthy)", "Non-Industrial", "ind_healthy_vs_ni")

    stats.add("bayes_n_vanish_h", "s2", "main",
              "VanishH+ taxa (Bayesian GMW Ind-healthy vs NI)", len(vanish_h_ids))
    stats.add("bayes_n_blossum_h", "s2", "main",
              "BlossumH+ taxa (Bayesian GMW Ind-healthy vs NI)", len(blossum_h_ids))

    def _sl(ids, path):
        with open(path, "w") as f: f.write("\n".join(ids))

    _sl(vanish_plus_ids,  os.path.join(s2_main, "bayes_vanish_plus_ids.txt"))
    _sl(blossum_plus_ids, os.path.join(s2_main, "bayes_blossum_plus_ids.txt"))
    _sl(vanish_h_ids,     os.path.join(s2_main, "bayes_vanishH_ids.txt"))
    _sl(blossum_h_ids,    os.path.join(s2_main, "bayes_blossumH_ids.txt"))

    print(f"    [bayes] Lists: V+={len(vanish_plus_ids)}  B+={len(blossum_plus_ids)}  "
          f"VH+={len(vanish_h_ids)}  BH+={len(blossum_h_ids)}")

    return (prev_df, abund_df,
            vanish_plus_ids, blossum_plus_ids,
            vanish_h_ids, blossum_h_ids)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_hadza_specific_bayes(data, presence_df, stats, taxonomy_df=None,
                                   posterior_df=None):
    from .section2_differential import _extract_lowest_rank, _extract_rank
    from .section2_pooling import _build_taxonomy_map

    ref     = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")
    meta    = data.meta

    hadza_mask    = (meta["Country_norm"].str.contains("Tanzania", na=False) &
                     meta["Lifestyle"].isin(["H-G", "Transitional"]))
    other_ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"]) & ~hadza_mask

    hadza_idx    = meta.index[hadza_mask    & meta.index.isin(presence_df.index)]
    other_ni_idx = meta.index[other_ni_mask & meta.index.isin(presence_df.index)]
    n_h, n_o     = len(hadza_idx), len(other_ni_idx)

    if n_h < 10 or n_o < 10:
        print(f"    [bayes hadza] Insufficient: Hadza={n_h}, other={n_o}")
        return

    print(f"    [bayes hadza] Hadza n={n_h:,}, other NI n={n_o:,}")
    tax_map = _build_taxonomy_map(taxonomy_df)

    # Use posterior z-test (soft) if posterior_df provided, else fall back to presence
    post_df = posterior_df if posterior_df is not None else presence_df
    df = _test_prevalence_weighted_posterior_bayes(
        posterior_df = post_df.loc[post_df.index.isin(list(hadza_idx) + list(other_ni_idx))],
        g1_idx = hadza_idx, g2_idx = other_ni_idx,
        w1 = np.ones(len(hadza_idx), dtype=np.float64),
        w2 = np.ones(len(other_ni_idx), dtype=np.float64),
        g1_label = "Hadza", g2_label = "other NI",
    )
    if df.empty:
        return
    # Rename columns to match downstream expectations
    df = df.rename(columns={"microbe": "sgb", "prev_g1": "hadza_prev",
                             "prev_g2": "other_ni_prev"})
    df["p_bonferroni"] = (df["pval"] * len(df)).clip(upper=1.0)
    df["gtdb_taxonomy"]    = df["sgb"].map(tax_map).fillna("")
    df["lowest_rank"]      = df["gtdb_taxonomy"].apply(_extract_lowest_rank)
    df = df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)

    df.to_csv(os.path.join(s2_main, f"bayes_hadza_vs_other_ni_{ref}.csv"), index=False)

    sig_mask = (df["p_bonferroni"] < 0.05) & (df["prevalence_diff"] > 0)
    n_sig    = sig_mask.sum()
    print(f"    [bayes hadza] {n_sig} SGBs enriched in Hadza (Bonferroni p<0.05)")
    stats.add("bayes_hadza_n_sig_enriched", "s2", "main",
              "SGBs Hadza-enriched by Bayesian posterior z-test (Bonferroni p<0.05)", n_sig)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL SECTION 2 ENTRY POINT  (Bayesian)
# ══════════════════════════════════════════════════════════════════════════════

def run_section2_bayesian(
    data,
    dm,
    stats,
    all_specs=None,
    main_specs=None,
    pooling_specs=None,
    taxonomy_df=None,
    null_key:        str            = "10pct",
    ra_module_path:  Optional[str]  = None,
    pi_industrial:   Optional[float] = None,
    pi_ni:           Optional[float] = None,
    run_supp_s5:     bool           = True,
):
    """
    Drop-in replacement for section2_differential.run_section2() using Bayesian
    detection throughout — including pooled analyses.

    Changes from previous implementation
    -------------------------------------
    1. Builds a BayesianDetectionBundle covering ALL null_keys needed by
       pooling_specs, in one posterior computation pass (not one per spec).

    2. 2B no longer calls compute_detection_with_pooling() (hard breadth thresholds).
       All pooling routes through run_section2_pooling() with the bundle.

    3. Pooled detection in 2E/2F/2G uses the raw-posterior union rule:
       detected_j = (1 - prod_i(1 - p_raw[i,j])) >= 0.5
       — the same estimand as accumulation_summary_v2._hard_call_accumulation().

    4. n_needed estimates use mean raw posterior per taxon via the analytic
       union formula, replacing the binary-prevalence approximation.
    """
    from .section2_pooling import run_section2_pooling
    from .detection import DetectionSpec

    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 2 (Bayesian) — Differential Prevalence [{ref}]")
    print(f"  primary null_key = {null_key}")
    print(f"{'='*60}")

    if main_specs is None:
        main_specs = [DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)]

    if pooling_specs is None:
        pooling_specs = [
            DetectionSpec("breadth_pct", 10.0),
            DetectionSpec("breadth_pct", 30.0),
            DetectionSpec("breadth_pct", 50.0),
            DetectionSpec("flat_bp", 100_000),   # 100 kb
            DetectionSpec("flat_bp", 300_000),   # 300 kb
            DetectionSpec("flat_bp", 500_000),   # 500 kb
        ]

    # Resolve null_keys for all pooling specs; skip those with no Bayesian equivalent
    valid_pooling = []
    spec_null_keys = []
    for spec in pooling_specs:
        try:
            nk = spec_to_null_key(spec)
            valid_pooling.append(spec)
            spec_null_keys.append(nk)
        except KeyError as e:
            print(f"  [bayes] WARNING: skipping pooling spec with no null_key: {e}")

    all_null_keys = sorted(set([null_key] + spec_null_keys))
    print(f"  [bayes] Bundle null_keys: {all_null_keys}")

    # ── 0: Build detection bundle once for all keys ───────────────────────────
    print("\n  [Bayes 0] Building Bayesian detection bundle ...")
    bundle = build_bayesian_detection_bundle(
        data           = data,
        null_keys      = all_null_keys,
        ra_module_path = ra_module_path,
        pi_industrial  = pi_industrial,
        pi_ni          = pi_ni,
    )

    # Keep both hard calls and raw posteriors
    presence_df  = bundle.sample_presence[null_key]
    posterior_df = bundle.raw[null_key]   # float32 [0,1] posterior probabilities

    # ── Save raw posterior NPZ cache (figure-compatible format) ──────────
    # The figure looks for bayesian_plot_v3_{ref}_{null_key}.npz in results/{ref}/
    # We write it here so the figure can load it without recomputing anything.
    try:
        import numpy as _np, json as _json
        _p_full  = bundle.raw[null_key]   # all samples x SGBs
        _meta_all = data.meta
        _sc = next((c for c in ("StudyID","Study","cohort","_study")
                    if c in _meta_all.columns), None)
        _ind_m = _meta_all["Lifestyle"] == "Industrial"
        _ni_m  = _meta_all["Lifestyle"].isin(["H-G", "Transitional"])
        _hg_m  = _meta_all["Lifestyle"].isin(["H-G"])
        _sI  = _meta_all.index[_ind_m & _meta_all.index.isin(_p_full.index)]
        _sNI = _meta_all.index[_ni_m  & _meta_all.index.isin(_p_full.index)]
        _sHG = _meta_all.index[_hg_m  & _meta_all.index.isin(_p_full.index)]
        _pI  = _p_full.loc[_sI];  _pNI = _p_full.loc[_sNI]
        _pHG = _p_full.loc[_sHG] if len(_sHG) >= 5 else _pNI
        _nst_I  = _meta_all.loc[_sI,  _sc].nunique() if _sc else 0
        _nst_NI = _meta_all.loc[_sNI, _sc].nunique() if _sc else 0
        _out_dir = config.outdir(ref)
        _npz  = os.path.join(_out_dir, f"bayesian_plot_v3_{ref}_{null_key}.npz")
        _jmeta = os.path.join(_out_dir, f"bayesian_plot_v3_{ref}_{null_key}_meta.json")
        _np.savez_compressed(
            _npz,
            p_raw_I  = _pI.values.astype(_np.float32),
            p_raw_NI = _pNI.values.astype(_np.float32),
            p_raw_HG = _pHG.values.astype(_np.float32),
            idx_I    = _np.array(list(_pI.index),  dtype=object),
            idx_NI   = _np.array(list(_pNI.index), dtype=object),
            idx_HG   = _np.array(list(_pHG.index), dtype=object),
            columns  = _np.array(list(_pI.columns), dtype=object),
        )
        _stI  = _meta_all.loc[_sI,  _sc].tolist() if _sc else []
        _stNI = _meta_all.loc[_sNI, _sc].tolist() if _sc else []
        with open(_jmeta, "w") as _f:
            _json.dump(dict(
                cache_version=3, ref=ref, null_key=null_key,
                study_col=_sc,
                pi_ind=float(bundle.pi_by_lifestyle.get("Industrial", 0.0)),
                pi_ni=float(bundle.pi_by_lifestyle.get("Non-Industrial", 0.0)),
                n_st_I=int(_nst_I), n_st_NI=int(_nst_NI),
                n_I=len(_pI), n_NI=len(_pNI), n_HG=len(_pHG),
                study_labels_I=_stI, study_labels_NI=_stNI, study_labels_HG=[],
            ), _f, indent=2)
        print(f"  [bayes] Figure cache saved: {_npz}")
    except Exception as _ce:
        print(f"  [bayes] Warning: could not save figure cache: {_ce}")

    # ── Export Bayesian soft-prev CSVs for figure consistency ────────────
    # These are the authoritative per-SGB mean posteriors used by
    # section2_bayesian_figure.py via --use-pipeline-soft-prev.
    # Using the same Lifestyle masks as build_bayesian_detection_bundle.
    _p_raw = bundle.raw[null_key]   # float32 DataFrame, all samples x SGBs
    _meta  = data.meta
    _ind_mask = _meta["Lifestyle"] == "Industrial"
    _ni_mask  = _meta["Lifestyle"].isin(["H-G", "Transitional"])
    _samps_ind = _meta.index[_ind_mask & _meta.index.isin(_p_raw.index)]
    _samps_ni  = _meta.index[_ni_mask  & _meta.index.isin(_p_raw.index)]
    _soft_ind = _p_raw.loc[_samps_ind].mean(axis=0)  # mean posterior per SGB, Ind
    _soft_ni  = _p_raw.loc[_samps_ni].mean(axis=0)   # mean posterior per SGB, NI

    def _write_soft_prev(path, soft_prev):
        soft_prev.rename("prevalence").rename_axis("taxon").to_csv(path)

    _sp_out = config.outdir(ref)
    _write_soft_prev(
        os.path.join(_sp_out, f"bayes_soft_prev_industrial_{null_key}.csv"), _soft_ind)
    _write_soft_prev(
        os.path.join(_sp_out, f"bayes_soft_prev_ni_{null_key}.csv"), _soft_ni)
    print(f"  [bayes] Soft-prev CSVs saved: Ind={len(_soft_ind)}, NI={len(_soft_ni)} SGBs")
    # Expose for downstream use in this function
    _soft_prev_ind = _soft_ind
    _soft_prev_ni  = _soft_ni

    det_per_sample = presence_df.sum(axis=1)
    det_per_sgb    = presence_df.sum(axis=0)
    stats.add("bayes_mean_sgbs_per_sample", "s2", "main",
              f"Mean SGBs detected per sample (Bayesian, {null_key})",
              f"{det_per_sample.mean():.1f} +/- {det_per_sample.std():.1f}")
    stats.add("bayes_n_sgbs_detected_any", "s2", "main",
              f"SGBs detected in >=1 sample (Bayesian, {null_key})",
              int((det_per_sgb > 0).sum()))

    # ── 2A: Differential (sample-level, unchanged) ───────────────────────────
    print("\n  [Bayes 2A] Differential analysis (all three passes)")
    (prev_df, abund_df,
     vanish_plus_ids, blossum_plus_ids,
     vanish_h_ids, blossum_h_ids) = run_differential_with_stats_bayes(
        data, dm, stats, presence_df,
        posterior_df=posterior_df, main_specs=main_specs)

    # 2B is now fully inside run_section2_pooling with the bundle
    # (previously was: compute_detection_with_pooling with hard breadth thresholds)

    # ── 2C: Hadza-specific ───────────────────────────────────────────────────
    print("\n  [Bayes 2C] Hadza-specific SGBs (Bayesian)")
    _compute_hadza_specific_bayes(data, presence_df, stats,
                                   taxonomy_df=taxonomy_df, posterior_df=posterior_df)

    # ── 2D–2G: Extended pooling with full Bayesian bundle ────────────────────
    print("\n  [Bayes 2D-2G] Extended pooling (Bayesian bundle, union detection)")
    run_section2_pooling(
        data             = data,
        vanish_plus_ids  = vanish_plus_ids,
        blossum_plus_ids = blossum_plus_ids,
        stats            = stats,
        main_specs       = valid_pooling,
        taxonomy_df      = taxonomy_df,
        detection_bundle = bundle,
        file_prefix      = "bayes_",
    )

    # ── Supplementary figures ─────────────────────────────────────────────────
    if run_supp_s5:
        print("\n  [Bayes 2E] Supplementary tables and figures (Bayesian lists)")
        try:
            from .supp_s5 import run_supp_s5 as _supp
            _supp(data, vanish_plus_ids, blossum_plus_ids,
                  vanish_h_ids, blossum_h_ids,
                  abund_df=abund_df, taxonomy_df=taxonomy_df)
        except Exception as exc:
            print(f"    [supp_s5 warn] skipped: {exc}")

    print(f"\n  Section 2 (Bayesian) complete [{ref}]")
    return vanish_plus_ids, blossum_plus_ids, vanish_h_ids, blossum_h_ids
