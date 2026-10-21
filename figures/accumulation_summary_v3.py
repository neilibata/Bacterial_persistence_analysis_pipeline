#!/usr/bin/env python3
"""
accumulation_summary_v3.py  (refactored — modelling fixes A–C + plot overhaul)
-------------------------------------------------------------------------------
2×2 summary figure: SGBs detected as a function of sampling effort.

Modelling fixes in this version
--------------------------------
Fix A — Below-floor sentinel separation (was: zero-conflation bug)
    `_gamma_collapse_to_study_posteriors` and `_scan_pi_gamma_consistent` now
    treat p_raw == 0.0 as "not evaluated / below floor" (neutral BF = 0) rather
    than as informative negative evidence.  Samples below the joint evidence
    floor no longer suppress study-level posteriors as if they were strong
    non-detections.  The sentinel test is `p_raw > 1e-7`; this is safe because
    `_latent_L_posterior_vec` always returns positive values for above-floor
    samples (it includes the prior as a lower bound).

Fix B — Scan/deployment algebraic consistency
    `_scan_pi_gamma_consistent` replaces `_scan_pi_gamma_soft`.  The held-out
    study target is now computed by the *same* gamma-damped log-odds collapse as
    `_gamma_collapse_to_study_posteriors` (the deployment path), rather than the
    per-sample union formula used previously.  The training prediction is the
    cross-study aggregate of per-training-study posteriors, also using the same
    gamma-collapse formula.  This closes the algebraic inconsistency: the gamma
    that is optimised in the scan is now the same gamma that governs the quantity
    being plotted.

    Residual limitation: the target p_study_h is still model-derived (it comes
    from the within-study gamma-collapse at the current pi value), not from
    external ground truth.  External validation would require independent
    sequencing or culture-based data.  Within that constraint, the new scan is
    a genuine study-level LOSO with a proper scoring rule and consistent algebra.

Fix C — Plotting improvements
    - Both axes hard-capped at 4000 SGBs.
    - Gridlines removed (already off via rcParams; confirmed).
    - Insets enlarged to ~2/3 of subplot area and shifted upward.
    - Asymptote annotations shifted down and vertically repelled.
    - Error bars on data points now show ± 1 SD (bootstrap std), not SEM.

Previously fixed (carried forward)
-----------------------------------
Fix 1 — Soft-label LOSO scoring (now subsumed by Fix B)
Fix 2 — Left-half MAD sigma estimator (`_estimate_sigma_b_lefthalf`)
Fix 3 — Evidence-space floor (`_evidence_floors`)
Fix 4 — fit_ok as A-fit diagnostic only (`_assert_not_detection_mask`)
Fix 5 — Single-point prior subtraction (no double-subtraction)

Known remaining limitations (documented, not yet fully resolved)
----------------------------------------------------------------
L1 — LOSO target is model-generated, not externally calibrated (see Fix B note).
L2 — Bootstrap CI reflects ordering uncertainty only; it does not propagate
     uncertainty in the detection model, pi_present, gamma, or sigma.
     Treat the asymptote band as a plotting envelope, not a full CI.
L3 — The accumulation curve is an *evidence-weighted accumulation score*
     (prior-subtracted excess posteriors fed into union-style rarefaction),
     not a literal posterior expected richness.  Manuscript descriptions should
     reflect this distinction.
L4 — The module import uses exec() over regex-modified source.  This is a
     reproducibility risk; see IMPORT WARNING below.

IMPORT WARNING
--------------
`_import_ra` reads rank_abundance_analysis.py, stubs one line via regex, and
exec()s the result.  If multiple copies of the file exist, auto-discovery may
load the wrong version silently.  Always pass --module-path explicitly.
The correct long-term fix is to install rank_abundance_analysis as a proper
package (pip install -e .) and import it with a standard import statement.

Layout
------
Row 0: accumulation vs NUMBER OF SAMPLES (per-sample rarefaction)
Row 1: accumulation vs NUMBER OF STUDIES (per-study rarefaction)
Col 0: Carter reference (UHGG)
Col 1: Leviatan reference (lev)

Usage
-----
    python accumulation_summary_v3.py \\
        --ref-carter UHGG --ref-lev lev \\
        --out accumulation_summary_v3.png \\
        --module-path /path/to/rank_abundance_analysis.py \\
        --n-iter 200 --seed 0 --n-workers 4
"""

import argparse, os, sys, importlib, importlib.util, warnings, traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy.optimize import minimize
from scipy.special import logit as _sp_logit, expit as _sp_expit

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Typography ────────────────────────────────────────────────────────────────
_FS  = 11
_FSL = 12
_FST = 13

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         _FS,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

PALETTE    = {"sky": "#56B4E9", "orange": "#E69F00"}
LS_COLORS  = {"Industrial": PALETTE["sky"], "Non-Industrial": PALETTE["orange"]}
LS_MARKERS = {"Industrial": "o",            "Non-Industrial": "s"}
LS_ORDER   = ["Industrial", "Non-Industrial"]

# ══════════════════════════════════════════════════════════════════════════════
#  Fix 4 — fit_ok is an A-fit diagnostic, never a detection/validity mask
# ══════════════════════════════════════════════════════════════════════════════
#
# NOTE: fit_ok records whether the asymptote-fit envelope converged.  It MUST
# NOT gate detection summaries, accumulation inputs, or validity checks.
# Detection is governed by p_genuine >= threshold (posterior-based) and
# has_data (any above-floor evidence exists regardless of envelope fit).
# Using fit_ok as a validity mask silently excludes floor-fit rows — those are
# real observations with posteriors, just without a converged A curve.
#
# The guard below is provided for callers.  Nothing in this file ever touches
# fit_ok except to emit it as a diagnostic column.

def _assert_not_detection_mask(label: str = "fit_ok") -> None:
    """
    Raise RuntimeError to prevent fit_ok from being used as a detection mask.

    Call immediately before any conditional branch or DataFrame filter that
    would exclude rows based on fit_ok.  Costs nothing at runtime in the
    normal (non-error) path since it always raises.

    Example usage:
        if using_fit_ok:
            _assert_not_detection_mask("fit_df['fit_ok']")
    """
    raise RuntimeError(
        f"`{label}` is an A-fit diagnostic only and must not be used as a "
        "detection or validity mask.  Use `has_data` (any above-floor evidence) "
        "for dataset validity and `p_genuine >= threshold` for detection."
    )


# ── Module import ─────────────────────────────────────────────────────────────
def _pre_parse_module_path():
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--module-path", "--module_path") and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(("--module-path=", "--module_path=")):
            return a.split("=", 1)[1]
    return None


def _import_ra(explicit_path=None):
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
            ra_src = _re.sub(
                r'^[^\n]*=\s*_resolve_pipeline_imports\(\)[^\n]*$',
                'load_reference = None; _cfg = None  # stubbed',
                ra_src, flags=_re.MULTILINE,
            )
            exec(compile(ra_src, p, "exec"), mod.__dict__)
            print(f"  RA module: {os.path.abspath(p)}")
            return mod
    raise FileNotFoundError(
        "rank_abundance_analysis.py not found. "
        "Pass --module-path /path/to/rank_abundance_analysis.py"
    )


_mp = _pre_parse_module_path()
if _mp is None:
    warnings.warn(
        "\n  --module-path not supplied: auto-discovering rank_abundance_analysis.py.\n"
        "  If multiple copies exist this may silently load the wrong version.",
        stacklevel=1,
    )
print("  Importing helpers from rank_abundance_analysis.py ...")
_ra = _import_ra(explicit_path=_mp)

# ── Low-level building blocks from RA module ─────────────────────────────────
# NOTE: _estimate_sigma_b and the _scan_pi_gamma / _joint_estimate_pi_gamma
# family are NOT imported here.  Both are overridden locally (Fixes 1 and 2).
_estimate_coverage_efficiency = _ra._estimate_coverage_efficiency
_posterior_accumulation       = _ra._posterior_accumulation
_logspace_checkpoints         = _ra._logspace_checkpoints
_latent_L_posterior_vec       = _ra._latent_L_posterior_vec
_latent_L_prior_mass          = _ra._latent_L_prior_mass
_hierarchical_aggregate       = _ra._hierarchical_aggregate


# ── Pipeline bootstrap ────────────────────────────────────────────────────────
def _bootstrap_pkg(pkg_dir, alias="mbo"):
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    pkg  = importlib.util.module_from_spec(spec)
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



# ══════════════════════════════════════════════════════════════════════════════
#  Fix 2 — Left-half MAD sigma estimator
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_sigma_b_lefthalf(
    bfrac_s:     pd.DataFrame,
    depth_s:     pd.DataFrame,
    c:           float,
    f_null:      float = 0.10,
    n_bins:      int   = 12,
    min_pts_bin: int   = 30,
) -> tuple:
    """
    Depth-stratified breadth noise model from left-half residuals only.

    Why left-half?
    --------------
    The RA module's `_estimate_sigma_b` fits a two-component EM mixture
    (null-Gaussian + non-null-Uniform) to the full residual distribution
    r = b - b_null.  The problem is that when genuine positives cluster near
    the null — low-breadth real signal — they are absorbed by the Uniform
    component only imperfectly, and the EM can incorrectly up-weight the null
    Gaussian to cover them.  This inflates sigma in exactly the low-breadth
    regime that matters most for distinguishing genuine signal from noise.

    Left-half residuals (r < 0, i.e. b < b_null) cannot belong to genuine
    positives: a real SGB can only add coverage, not remove it relative to
    the null curve.  They are therefore a clean, uncontaminated draw from
    the null noise distribution.  Under the symmetric noise assumption
    N(0, sigma^2) the left half has the same spread as the right, so:

        sigma_k = MAD(|r[r < 0]|) / 0.6745

    where 0.6745 = Phi^{-1}(0.75) is the Gaussian correction so that
    MAD(|N(0,1)|) = 0.6745.

    Parameters
    ----------
    bfrac_s, depth_s : (n_samples, n_sgbs) DataFrames — must be aligned
    c                : coverage efficiency scalar
    f_null           : null curve genome fraction (default 0.10)
    n_bins           : number of log10(depth) bins
    min_pts_bin      : minimum left-half points per bin for a local estimate;
                       bins below this fall back to the global estimate

    Returns
    -------
    sigma_by_bin : ndarray (n_bins,)   — per-bin sigma estimate
    bin_edges    : ndarray (n_bins+1,) — log10(depth) bin edges
    """
    b_flat = bfrac_s.values.ravel()
    d_flat = depth_s[bfrac_s.columns].values.ravel()

    valid  = (d_flat > 0) & np.isfinite(b_flat) & np.isfinite(d_flat)
    b_v    = b_flat[valid]
    d_v    = d_flat[valid]

    b_null = f_null * (1.0 - np.exp(-c * d_v / max(f_null, 1e-9)))
    resid  = b_v - b_null                          # negative ↔ below null

    # Global left-half fallback (no hand-tuned constant)
    left_all    = resid[resid < 0]
    global_sigma = (float(np.median(np.abs(left_all))) / 0.6745
                    if len(left_all) >= 10 else 0.015)
    global_sigma = max(global_sigma, 1e-4)

    log_d        = np.log10(np.maximum(d_v, 1e-9))
    edges        = np.linspace(log_d.min(), log_d.max(), n_bins + 1)
    sigma_by_bin = np.full(n_bins, global_sigma, dtype=float)

    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        in_bin = (log_d >= lo) & (log_d < hi)
        left_k = resid[in_bin & (resid < 0)]

        if len(left_k) < min_pts_bin:
            continue                               # keep global fallback

        sigma_by_bin[k] = max(float(np.median(np.abs(left_k))) / 0.6745, 1e-4)

    return sigma_by_bin, edges


# ══════════════════════════════════════════════════════════════════════════════
#  Fix 3 — Evidence-space floor (covered windows + depth floor)
# ══════════════════════════════════════════════════════════════════════════════

_WINDOW_BP = 1_000   # 1 kb window — atomic unit of coverage evidence


def _evidence_floors(
    G_arr:      np.ndarray,
    c:          float,
    min_map_bp: int = 10_000,
) -> tuple:
    """
    Per-SGB joint evidence floor expressed in covered 1 kb windows.

    Background
    ----------
    The original code tested `b >= min_map_bp / G` per SGB, which is a
    breadth-fraction threshold that scales with G.  For a 500 kb genome it
    is ~2% breadth; for a 5 Mb genome it is ~0.2%.  The threshold penalises
    short references far more than large ones for no probabilistic reason.

    This function expresses the floor as a fixed number of covered 1 kb
    windows:

        min_windows = min_map_bp // _WINDOW_BP   (10 windows by default)

    The breadth floor is then:

        b_floor[j] = min_windows / (G_j / _WINDOW_BP) = min_map_bp / G_j

    So the breadth formula is unchanged.  What IS new is the depth floor:

        E[covered windows | depth d] = (G_j/_WINDOW_BP) * (1 - exp(-c*d*_WINDOW_BP/G_j))

    Setting this equal to min_windows and solving for d gives:

        d_floor[j] = -log(1 - min_windows * _WINDOW_BP / G_j) * G_j / (c * _WINDOW_BP)

    The depth floor catches technical failures: a sample at high sequencing
    depth with near-zero breadth should not be treated as a valid non-detection
    because the model expects it to have covered min_windows windows.  The
    joint (b_floor AND d_floor) test is a bivariate evidence gate that the
    original single-threshold approach lacked entirely.

    Parameters
    ----------
    G_arr      : (n_g,) genome lengths in bp
    c          : coverage efficiency scalar
    min_map_bp : minimum evidence expressed as base-pairs

    Returns
    -------
    b_floor : (n_g,) breadth fraction floor
    d_floor : (n_g,) depth floor (Poisson-derived, NEW)
    """
    n_win    = np.maximum(G_arr / _WINDOW_BP, 1.0)         # windows per SGB
    min_win  = max(1, min_map_bp // _WINDOW_BP)             # scalar
    win_frac = min_win / n_win                              # (n_g,)

    b_floor   = np.clip(win_frac, 1e-4, 0.99)

    safe_frac = np.minimum(win_frac, 0.99)
    d_floor   = (
        -np.log(np.maximum(1.0 - safe_frac, 1e-9))
        * G_arr / np.maximum(c * _WINDOW_BP, 1e-9)
    )
    return b_floor, d_floor


# ══════════════════════════════════════════════════════════════════════════════
#  Fix 1 — Soft-label LOSO scoring
# ══════════════════════════════════════════════════════════════════════════════

def _gamma_collapse_vec(p_rows, lo_prior, gam):
    """
    Vectorised gamma-damped log-odds collapse for a block of samples × SGBs.

    Fix A: cells where p_raw == 0 (below floor / not evaluated) contribute
    BF = 0 rather than a large negative BF.  The sentinel is `p_raw > 1e-7`;
    this is safe because `_latent_L_posterior_vec` always returns a value
    >= alpha_j > 0 for above-floor samples.

    Parameters
    ----------
    p_rows   : (n_samples, n_sgbs) float64 — raw posteriors (0 = not evaluated)
    lo_prior : (n_sgbs,) float64 — logit(alpha_j)
    gam      : float — damping factor in [0, 1]

    Returns
    -------
    p_study : (n_sgbs,) float64 — raw study-level posterior
    """
    evaluated = p_rows > 1e-7                                          # (n, J)
    p_safe    = np.where(evaluated, np.clip(p_rows, 1e-9, 1-1e-9), 0.5)
    bf        = np.where(evaluated, _sp_logit(p_safe) - lo_prior, 0.0)
    bf_d      = np.where(bf >= 0.0, bf, gam * bf)
    lo_agg    = lo_prior + bf_d.sum(axis=0)                            # (J,)
    return np.clip(_sp_expit(lo_agg), 0.0, 1.0)


def _scan_pi_gamma_consistent(
    bfrac_s:        pd.DataFrame,
    depth_s:        pd.DataFrame,
    genome_lengths: dict,
    c:              float,
    sigma_bins:     np.ndarray,
    sigma_edges:    np.ndarray,
    study_labels,
    pi_grid:        tuple = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90),
    gamma_grid:     tuple = (0.05, 0.10, 0.20, 0.35, 0.50, 1.00),
    n_sgbs_sample:  int   = 300,
    min_map_bp:     int   = 10_000,
    seed:           int   = 0,
) -> dict:
    """
    Joint (pi_present, gamma) grid scan — algebraically consistent LOSO.

    Fix A: below-floor samples treated as uninformative (BF = 0), not as
    evidence of absence.  Previously p_raw == 0.0 was being clipped to 1e-9
    inside _gamma_collapse, yielding a strongly negative log-odds BF that
    suppressed the study-level posterior incorrectly.

    Fix B: scan now uses the *same* gamma-damped log-odds aggregation as
    `_gamma_collapse_to_study_posteriors` (the deployment path) for both:
      - the held-out study's target  (within-study collapse at current gamma)
      - the training prediction      (cross-study aggregate of per-study
                                      gamma-collapsed posteriors)

    Previously the scan used `_hierarchical_aggregate` for training and the
    union formula for the target — two different algebra systems.  Optimising
    gamma with one rule and deploying with another is an internal-consistency
    error: the selected gamma was not tuned for the quantity being plotted.

    Residual limitation (L1): the held-out target is still model-derived
    (gamma-collapse at the current pi value, not external ground truth).
    External calibration would require independent sequencing or culture data.
    Within that constraint this is a genuine study-level LOSO with a proper
    scoring rule and consistent algebra.

    Algorithm
    ---------
    For each (pi, gamma):
      1. Compute p_mat[n_samples, J] — raw posteriors (0 = below floor)
      2. For each study k: p_study_mat[k, J] = gamma_collapse_vec(study_k_rows)
      3. Compute study-level BFs: bf_study[k, J] = logit(p_study_mat[k]) - lo_prior
         and dampen: bf_d[k, J] = gamma * min(0, bf) + max(0, bf)
      4. LOSO: for held-out study h, training prediction =
              expit(lo_prior + sum_{k≠h} bf_d[k, :])
         target = p_study_mat[h, :]
         score  = mean_j CE(target_j, pred_j) over informative SGBs
    """
    def _fallback():
        sgbs_f   = list(bfrac_s.columns)
        floors_f = np.array([
            max(1e-4, min_map_bp / max(
                int(genome_lengths.get(str(s)) or genome_lengths.get(s)), 1))
            for s in sgbs_f
        ], dtype=np.float64)
        frac  = float((bfrac_s.values > floors_f[np.newaxis, :]).mean())
        pi0   = float(np.clip(frac, 0.05, 0.95))
        grid  = np.full((len(pi_grid), len(gamma_grid)), np.nan)
        return dict(best_pi=pi0, best_gamma=0.15,
                    score_grid=grid, pi_grid=pi_grid, gamma_grid=gamma_grid,
                    method="fallback_independent")

    if study_labels is None:
        return _fallback()
    # Drop NaN study labels — samples with no study assignment should not be
    # used as held-out units or included in training study counts.
    nan_mask     = pd.isnull(study_labels)
    if nan_mask.all():
        return _fallback()
    study_labels = np.asarray(study_labels, dtype=object)
    study_labels = np.where(nan_mask, None, study_labels)   # sentinel → None

    # Build valid-study list (exclude None sentinel)
    all_unique = np.unique(study_labels[~nan_mask])
    studies    = all_unique                                   # only real labels
    K = len(studies)
    if K < 3:
        return _fallback()

    rng  = np.random.default_rng(seed)
    sgbs = list(bfrac_s.columns)
    if len(sgbs) > n_sgbs_sample:
        sgbs = list(rng.choice(sgbs, n_sgbs_sample, replace=False))
    J = len(sgbs)

    b_mat = bfrac_s[sgbs].values.astype(np.float64)
    d_mat = depth_s[sgbs].values.astype(np.float64)

    G_arr = np.array([
        int(genome_lengths.get(str(s)) or genome_lengths.get(s)) for s in sgbs
    ], dtype=np.float64)
    tau_arr = 0.10 * G_arr

    b_floor_arr, d_floor_arr = _evidence_floors(G_arr, c, min_map_bp)

    log_d_flat = np.log10(np.maximum(d_mat.ravel(), 1e-9))
    k_flat     = np.searchsorted(sigma_edges[1:], log_d_flat)
    k_flat     = np.clip(k_flat, 0, len(sigma_bins) - 1).astype(int)
    sigma_mat  = sigma_bins[k_flat].reshape(d_mat.shape)

    # Precompute per-study sample row-index arrays (NaN rows get no mask entry)
    valid_rows  = np.where(~nan_mask)[0]           # row indices with a real study label
    study_masks = {s: valid_rows[study_labels[valid_rows] == s] for s in studies}

    score_grid = np.full((len(pi_grid), len(gamma_grid)), -np.inf)

    for i_pi, pi in enumerate(pi_grid):
        alpha_arr = np.array([
            _latent_L_prior_mass(tau_arr[j], G_arr[j], pi_present=pi)
            for j in range(J)
        ], dtype=np.float64)
        lo_prior = _sp_logit(np.clip(alpha_arr, 1e-9, 1.0 - 1e-9))  # (J,)

        # --- Step 1: raw per-sample posteriors (Fix A: 0 = below floor) -----
        p_mat = np.zeros((b_mat.shape[0], J), dtype=np.float64)
        for j in range(J):
            b_j = b_mat[:, j]; d_j = d_mat[:, j]
            s_j = np.maximum(sigma_mat[:, j], 1e-9)
            valid = (b_j >= float(b_floor_arr[j])) & (d_j >= float(d_floor_arr[j]))
            if not valid.any():
                continue
            p_mat[valid, j] = _latent_L_posterior_vec(
                b_arr=b_j[valid], d_arr=d_j[valid], G=G_arr[j],
                c=c, sigma_arr=s_j[valid], tau=tau_arr[j],
                n_quad=25, pi_present=pi,
            )

        for i_g, gam in enumerate(gamma_grid):
            # --- Step 2: per-study gamma-collapse (same formula as deployment) ---
            p_study_mat = np.tile(alpha_arr, (K, 1))  # default: prior if no samples
            for k, study in enumerate(studies):
                rows_k = p_mat[study_masks[study], :]          # (n_h, J)
                if rows_k.shape[0] == 0:
                    continue
                p_study_mat[k, :] = _gamma_collapse_vec(rows_k, lo_prior, gam)

            # --- Step 3: study-level BFs for cross-study aggregate ---------------
            lo_study  = _sp_logit(np.clip(p_study_mat, 1e-9, 1.0 - 1e-9))  # (K, J)
            bf_study  = lo_study - lo_prior[np.newaxis, :]                   # (K, J)
            bf_study_d = np.where(bf_study >= 0.0, bf_study, gam * bf_study) # (K, J)
            bf_total   = bf_study_d.sum(axis=0)                               # (J,)

            # SGBs with meaningful variation across studies
            informative = p_study_mat.std(axis=0) > 1e-3                     # (J,)

            # --- Step 4: LOSO cross-entropy scoring ---------------------------
            scores = []
            for k_h in range(K):
                # Training prediction (leave study k_h out)
                bf_tr  = bf_total - bf_study_d[k_h, :]        # (J,)
                lo_pred = lo_prior + bf_tr
                p_pred  = np.clip(_sp_expit(lo_pred), 1e-9, 1.0 - 1e-9)  # (J,)

                # Held-out study's target (same gamma-collapse as deployment)
                p_h = np.clip(p_study_mat[k_h, :], 1e-9, 1.0 - 1e-9)     # (J,)

                if not informative.any():
                    continue
                ce = (p_h * np.log(p_pred) + (1.0 - p_h) * np.log(1.0 - p_pred))
                scores.append(float(ce[informative].mean()))

            if scores:
                score_grid[i_pi, i_g] = float(np.mean(scores))

    if not np.isfinite(score_grid).any():
        fb = _fallback(); fb["score_grid"] = score_grid; return fb

    best_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    return dict(
        best_pi    = float(pi_grid[best_idx[0]]),
        best_gamma = float(gamma_grid[best_idx[1]]),
        score_grid = score_grid,
        pi_grid    = pi_grid,
        gamma_grid = gamma_grid,
        method     = "joint_scan_consistent",
    )


def _joint_estimate_pi_gamma(
    bfrac_s, depth_s, genome_lengths, c, sigma_bins, sigma_edges,
    study_labels,
    pi_grid        = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90),
    gamma_grid     = (0.05, 0.10, 0.20, 0.35, 0.50, 1.00),
    n_sgbs_sample  = 300,
    seed           = 0,
) -> tuple:
    """Wrapper around `_scan_pi_gamma_consistent`. Prints table, returns (pi, gamma, result)."""
    print("    Joint (pi_present, gamma) scan — consistent LOSO (Fix B) ...")
    result   = _scan_pi_gamma_consistent(
        bfrac_s, depth_s, genome_lengths, c,
        sigma_bins, sigma_edges, study_labels,
        pi_grid=pi_grid, gamma_grid=gamma_grid,
        n_sgbs_sample=n_sgbs_sample, seed=seed,
    )
    best_pi  = result["best_pi"]
    best_gam = result["best_gamma"]
    method   = result["method"]
    grid     = result["score_grid"]

    if method == "joint_scan_consistent":
        pi_vals  = list(result["pi_grid"])
        gam_vals = list(result["gamma_grid"])
        header   = "pi\\gamma  " + "  ".join(f"{g:6.2f}" for g in gam_vals)
        print(f"      {header}")
        for i, pi in enumerate(pi_vals):
            row = "  ".join(
                f"{'*' if (i == np.argmax(grid[:, j]) and j == np.argmax(grid[i, :])) else ' '}"
                f"{grid[i, j]:6.2f}"
                for j in range(len(gam_vals))
            )
            print(f"      pi={pi:.2f}  {row}{' <-- best' if pi == best_pi else ''}")
        print(f"    Best: pi={best_pi:.2f}  gamma={best_gam:.2f}  (consistent LOSO)")
    else:
        print(f"    Fallback: pi={best_pi:.3f}  gamma={best_gam:.3f}  (< 3 studies)")

    return best_pi, best_gam, result


# ══════════════════════════════════════════════════════════════════════════════
#  Fix 5 — Raw posterior computation (no prior subtraction)
# ══════════════════════════════════════════════════════════════════════════════

def _fast_compute_posterior_10pct(
    bfrac_s:        pd.DataFrame,
    depth_s:        pd.DataFrame,
    c:              float,
    sigma_bins:     np.ndarray,
    sigma_edges:    np.ndarray,
    genome_lengths: dict,
    pi_present:     float,
    min_map_bp:     int = 10_000,
    n_quad:         int = 25,
    n_workers:      int = 4,
) -> pd.DataFrame:
    """
    Per-sample RAW posteriors P(L >= tau_10pct | b, d, G).

    Fix 5: returns raw values — no prior subtraction.  Prior subtraction is
    deferred to `_apply_prior_subtraction` in `compute_accumulation`, called
    exactly once at whichever aggregation level (sample or study) feeds
    `_posterior_accumulation`.

    Fix 3: the evidence floor uses `_evidence_floors` (joint breadth + depth).

    Speed: only the 10pct model (4× fewer calls), parallel SGB loop.
    """
    sgbs  = bfrac_s.columns.tolist()
    samps = bfrac_s.index.tolist()
    n_s, n_g = len(samps), len(sgbs)

    b_mat = bfrac_s.values.astype(np.float32)
    d_mat = depth_s[sgbs].values.astype(np.float32)

    G_arr   = np.array([
        int(genome_lengths.get(str(g)) or genome_lengths.get(g)) for g in sgbs
    ], dtype=np.float64)
    tau_arr = 0.10 * G_arr

    b_floor, d_floor = _evidence_floors(G_arr, c, min_map_bp)

    if sigma_bins is not None and sigma_edges is not None:
        log_d_flat = np.log10(np.maximum(d_mat.ravel(), 1e-9))
        k_flat     = np.searchsorted(sigma_edges[1:], log_d_flat)
        k_flat     = np.clip(k_flat, 0, len(sigma_bins) - 1).astype(int)
        sigma_mat  = sigma_bins[k_flat].reshape(n_s, n_g).astype(np.float32)
    else:
        sigma_mat = np.full((n_s, n_g), 0.015, dtype=np.float32)
    sigma_mat = np.maximum(sigma_mat, 1e-9)

    p_mat = np.zeros((n_s, n_g), dtype=np.float32)   # RAW, no prior subtraction

    def _process_range(j_start: int, j_end: int) -> None:
        for j in range(j_start, j_end):
            tau_j = tau_arr[j]; G_j = G_arr[j]
            if tau_j >= G_j:
                continue
            b_j = b_mat[:, j].astype(np.float64)
            d_j = d_mat[:, j].astype(np.float64)
            s_j = sigma_mat[:, j].astype(np.float64)
            valid = (b_j >= float(b_floor[j])) & (d_j >= float(d_floor[j]))
            if not valid.any():
                continue
            p_vec = _latent_L_posterior_vec(
                b_arr=b_j[valid], d_arr=d_j[valid], G=G_j, c=c,
                sigma_arr=s_j[valid], tau=tau_j, n_quad=n_quad,
                pi_present=pi_present,
            ).astype(np.float32)
            p_col        = np.zeros(n_s, dtype=np.float32)
            p_col[valid] = p_vec
            p_mat[:, j]  = p_col

    ranges = np.array_split(np.arange(n_g), n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_process_range, int(r[0]), int(r[-1]) + 1)
            for r in ranges if len(r) > 0
        ]
        for f in futures:
            f.result()

    return pd.DataFrame(p_mat, index=samps, columns=sgbs, dtype=np.float32)


def _apply_prior_subtraction(
    p_df:           pd.DataFrame,
    genome_lengths: dict,
    pi_present:     float,
) -> pd.DataFrame:
    """
    Convert RAW posteriors → prior-subtracted excess posteriors (Fix 5).

        p_excess[i, j] = max(0, (p_raw[i, j] - alpha_j) / (1 - alpha_j))

    Called exactly once per `compute_accumulation` call.
    """
    sgbs  = p_df.columns.tolist()
    G_arr = np.array([
        int(genome_lengths.get(str(g)) or genome_lengths.get(g)) for g in sgbs
    ], dtype=np.float64)
    tau_arr   = 0.10 * G_arr
    alpha_arr = np.array([
        _latent_L_prior_mass(tau_arr[j], G_arr[j], pi_present=pi_present)
        for j in range(len(sgbs))
    ], dtype=np.float64)

    p_mat    = p_df.values.astype(np.float64)
    alpha_bc = alpha_arr[np.newaxis, :]
    denom    = np.maximum(1.0 - alpha_bc, 1e-9)
    p_excess = np.where(alpha_bc > 0,
                        np.clip((p_mat - alpha_bc) / denom, 0.0, 1.0),
                        p_mat)
    return pd.DataFrame(p_excess.astype(np.float32),
                        index=p_df.index, columns=p_df.columns)


# ══════════════════════════════════════════════════════════════════════════════
#  Fix 5 — Gamma-weighted study collapse: raw in, raw out
# ══════════════════════════════════════════════════════════════════════════════

def _gamma_collapse_to_study_posteriors(
    p_df:           pd.DataFrame,
    meta_s:         pd.DataFrame,
    study_col:      str,
    gamma:          float,
    genome_lengths: dict,
    pi_present:     float,
) -> pd.DataFrame:
    """
    Collapse RAW per-sample posteriors to RAW per-study posteriors.

    Fix 5: both input and output are raw posteriors — the prior is NOT
    subtracted here.  The prior subtraction step is deferred to
    `_apply_prior_subtraction` in `compute_accumulation`.

    Fix A: cells where p_raw == 0.0 (below floor / not evaluated) are treated
    as uninformative (BF = 0), not as evidence of absence.  This is handled
    by `_gamma_collapse_vec`, which checks `p > 1e-7` as the evaluated
    sentinel.  Previously, these cells were clipped to 1e-9 and converted to
    a large negative log-odds BF, effectively penalising studies with many
    shallow samples as if those samples were informative non-detections.

    Self-consistency: when all samples in a study are below floor (p_raw == 0),
    all BFs are 0, and p_study_raw = expit(lo_prior) = alpha_j — the prior
    exactly.  `_apply_prior_subtraction` then yields 0 excess, which is correct.

    The model (per study k, SGB j, samples i in study k, via _gamma_collapse_vec):
        evaluated[i,j]  = p_raw[i,j] > 1e-7
        BF[i,j]         = logit(p_raw[i,j]) - logit(alpha_j)  if evaluated
                        = 0                                     otherwise (Fix A)
        BF_damped[i,j]  = gamma * BF[i,j]  if BF[i,j] < 0 (non-detection)
                        = BF[i,j]           otherwise
        lo_study[k,j]   = logit(alpha_j) + sum_i BF_damped[i,j]
        p_study_raw[k,j] = expit(lo_study[k,j])
    """
    sgbs    = p_df.columns.tolist()
    studies = meta_s[study_col].unique()
    n_st    = len(studies)
    n_g     = len(sgbs)

    G_arr = np.array([
        int(genome_lengths.get(str(g)) or genome_lengths.get(g)) for g in sgbs
    ], dtype=np.float64)
    alpha_arr = np.array([
        _latent_L_prior_mass(0.10 * G_arr[j], G_arr[j], pi_present=pi_present)
        for j in range(n_g)
    ], dtype=np.float64)
    lo_prior = _sp_logit(np.clip(alpha_arr, 1e-9, 1.0 - 1e-9))  # (n_g,)
    gam      = float(np.clip(gamma, 0.0, 1.0))

    # Default for studies with no samples: return the prior exactly
    p_study  = np.tile(alpha_arr, (n_st, 1)).astype(np.float64)

    for k, study in enumerate(studies):
        mask     = meta_s[study_col] == study
        samp_ids = meta_s.index[mask]
        rows     = p_df.loc[p_df.index.intersection(samp_ids)].values.astype(np.float64)
        if rows.shape[0] == 0:
            continue
        # Fix A: uninformative cells (below floor, p==0) contribute BF=0
        p_study[k] = _gamma_collapse_vec(rows, lo_prior, gam)

    return pd.DataFrame(p_study.astype(np.float32),
                        index=pd.Index(studies, name=study_col),
                        columns=p_df.columns)


# ── Hill-MM fitting (unchanged) ───────────────────────────────────────────────
def _fit_hill(ns, Ss, S_ref, S_sem=None):
    ns_f = np.asarray(ns, float); Ss_f = np.asarray(Ss, float)
    if S_sem is not None and len(S_sem) == len(Ss_f):
        w = 1.0 / np.maximum(S_sem, 0.5) ** 2
    else:
        w = np.ones(len(Ss_f))
    w = w / w.sum()

    def obj(p):
        Sm, K, h = p
        nh = ns_f ** h; Kh = K ** h
        return float(np.sum(w * (Sm * nh / (Kh + nh) - Ss_f) ** 2))

    bounds = [(1.0, float(S_ref)), (0.1, float(max(ns_f) * 10)), (0.1, 5.0)]
    best = None
    for K0 in [float(np.median(ns_f)), float(max(ns_f))]:
        for h0 in [0.5, 1.0, 2.0]:
            S0  = float(min(Ss_f[-1] * 2, S_ref))
            res = minimize(obj, x0=[S0, K0, h0], bounds=bounds, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

    if best is None or (not best.success and best.fun > 1e8):
        return dict(S_max=np.nan, K=np.nan, h=np.nan,
                    S_pred=np.full_like(ns_f, np.nan), success=False, r2=np.nan)
    Sm, K, h = best.x
    nh = ns_f ** h; Kh = K ** h
    pred   = Sm * nh / (Kh + nh)
    ss_res = float(np.sum(w * (Ss_f - pred) ** 2))
    ss_tot = float(np.sum(w * (Ss_f - np.average(Ss_f, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(S_max=float(Sm), K=float(K), h=float(h),
                S_pred=pred, success=True, r2=float(r2))


def _asymptote_ci(curves, ns, S_ref, ci=0.95, n_boot_asym=400, rng=None):
    """
    Fit Hill-MM asymptote to the mean accumulation curve; CI by parametric
    bootstrap on the mean.

    Why the old approach was wrong
    --------------------------------
    The previous implementation fit Hill-MM to each *individual* bootstrap
    ordering curve and used percentiles of the resulting S_max estimates as
    the CI.  This is statistically incorrect for two reasons:

    1. Each ordering curve is a single random permutation of the N units.
       At early checkpoints (n = 1, 2, ...) the curve value is purely random —
       a single SGB or a handful.  These early points are enormously variable
       across permutations and, because they were weighted equally (S_sem=None),
       they drove the Hill fit to very different S_max values for different
       orderings.

    2. The resulting CI band measured "how much does S_max change when Hill-MM
       is fit to a noisy individual ordering curve?" — essentially Hill-fit
       instability, not biological uncertainty.  That is why the CI band
       extended down into the observed data: it was wider than the data itself.

    Correct approach
    ----------------
    S_max is estimated by fitting the mean curve E[S(k)], which is the
    statistically appropriate estimand (the mean across all orderings converges
    to the true expected rarefaction curve).  The CI reflects uncertainty in
    that mean, propagated through the Hill-MM model:

        Step 1. Fit Hill-MM to S_mean (weighted by 1/S_sem^2) → S_max.
        Step 2. Parametric bootstrap: for each of n_boot_asym iterations,
                draw S_perturbed = S_mean + N(0, S_sem^2) at every checkpoint
                and refit Hill-MM.  Collect S_max values.
        Step 3. CI = percentiles of the bootstrapped S_max values.

    The result is a narrow band around the asymptote dotted line, correctly
    capturing "how well-constrained is S_max by the observed mean curve."

    Residual limitation (L2): this CI does not propagate uncertainty from the
    detection model, pi_present, gamma, or study-level sampling variation.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_curves = len(curves)
    S_mean = curves.mean(axis=0)
    S_sd   = curves.std(axis=0)
    S_sem  = S_sd / max(np.sqrt(n_curves), 1.0)

    # Fit the mean curve (weighted by inverse variance)
    mm = _fit_hill(ns, S_mean, S_ref, S_sem=S_sem)
    if not mm["success"] or not np.isfinite(mm["S_max"]):
        return np.nan, np.nan, np.nan, mm

    # Parametric bootstrap on the mean curve
    # Noise at each checkpoint is N(0, S_sem^2) — reflects ordering uncertainty
    # in the *mean*, not per-ordering instability.
    boots = []
    for _ in range(n_boot_asym):
        noise = rng.standard_normal(len(S_mean)) * S_sem
        S_pert = np.clip(S_mean + noise, 0, S_ref)
        r = _fit_hill(ns, S_pert, S_ref, S_sem=S_sem)
        if r["success"] and np.isfinite(r["S_max"]):
            boots.append(r["S_max"])

    if len(boots) < 5:
        return mm["S_max"], np.nan, np.nan, mm

    tail = (1.0 - ci) / 2.0
    lo   = float(np.percentile(boots, 100 * tail))
    hi   = float(np.percentile(boots, 100 * (1.0 - tail)))
    return mm["S_max"], lo, hi, mm


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep(ref_key: str) -> dict:
    """
    Load reference data and jointly estimate (pi_present, gamma).

    Uses local overrides:
      - Fix 2: `_estimate_sigma_b_lefthalf`
      - Fix 1: local `_joint_estimate_pi_gamma` (soft-label LOSO)
    """
    print(f"    Loading '{ref_key}' ...")
    data    = load_reference(ref_key)
    meta    = data.meta.copy()
    breadth = data.breadth
    lengths = data.lengths

    if not (hasattr(lengths, "items") and len(lengths) > 0):
        raise ValueError(
            f"load_and_prep('{ref_key}'): data.lengths is empty or missing. "
            "Ensure all_SGB_lengths_*.txt is available."
        )
    gl = {str(k): int(v) for k, v in lengths.items()}

    raw_max = float(breadth.values[breadth.values > 0].max()) \
              if (breadth.values > 0).any() else 0.0
    bfrac   = (breadth / 100.0) if raw_max > 1.0 else breadth.copy()
    samps   = [s for s in meta.index if s in bfrac.index and s in data.depth.index]
    bfrac   = bfrac.loc[samps]
    depth   = data.depth.loc[samps]
    meta    = meta.loc[samps]

    c       = _estimate_coverage_efficiency(bfrac, depth)
    # Fix 2: left-half MAD — no null-curve anchor, uncontaminated by positives
    sb, se  = _estimate_sigma_b_lefthalf(bfrac, depth, c=c)
    print(f"      c={c:.4f}  σ_b left-half [{sb.min():.4f}, {sb.max():.4f}]")

    _SC = ("StudyID", "Study", "cohort", "_study")
    sc  = next((col for col in _SC if col in meta.columns), None)
    if sc:
        print(f"      study col='{sc}' ({meta[sc].nunique()} studies)")

    study_labels = meta[sc].values if sc else None

    # Fix 1: soft-label LOSO (local implementation, not RA module's version)
    pi_est, gamma_est, scan_info = _joint_estimate_pi_gamma(
        bfrac_s=bfrac, depth_s=depth, genome_lengths=gl,
        c=c, sigma_bins=sb, sigma_edges=se, study_labels=study_labels,
        pi_grid=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90),
        gamma_grid=(0.05, 0.10, 0.20, 0.35, 0.50, 1.00),
        n_sgbs_sample=300, seed=0,
    )
    print(f"      pi_present={pi_est:.3f}  gamma={gamma_est:.3f}  "
          f"(method={scan_info['method']})")

    return dict(
        bfrac=bfrac, depth=depth, meta=meta, genome_lengths=gl,
        c=c, sigma_bins=sb, sigma_edges=se,
        S_ref=bfrac.shape[1], study_col=sc,
        pi_present=pi_est, gamma=gamma_est, scan_info=scan_info,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Accumulation — single prior subtraction point (Fix 5)
# ══════════════════════════════════════════════════════════════════════════════

def compute_accumulation(
    bfrac_s, depth_s, meta_s, genome_lengths,
    level:      str,
    n_iter:     int,
    rng,
    study_col   = None,
    sigma_bins  = None,
    sigma_edges = None,
    c           = 1.0,
    n_pts       = 80,
    pi_present  = 0.5,
    gamma       = 0.15,
    n_workers   = 4,
) -> dict:
    """
    Bootstrap posterior-weighted rarefaction.

    Prior-subtraction flow (Fix 5)
    --------------------------------
    Step 1.  _fast_compute_posterior_10pct  →  RAW P(L >= tau | b, d)
    Step 2a. Sample-level: _apply_prior_subtraction(p_raw) → p_excess
    Step 2b. Study-level:  _gamma_collapse_to_study_posteriors(p_raw)
                           → p_study_raw (still raw)
                           → _apply_prior_subtraction(p_study_raw) → p_excess
    Step 3.  _posterior_accumulation(p_excess) → bootstrap curves

    Prior is subtracted exactly once, after the aggregation level is resolved.
    The double-subtraction in the previous version arose because prior
    subtraction happened inside _fast_compute_posterior_10pct AND implicitly
    again in _gamma_collapse_to_study_posteriors via the log-prior baseline.
    """
    _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
    if study_col is None:
        study_col = next((col for col in _STUDY_COLS if col in meta_s.columns), None)

    print(f"      Raw posteriors ({len(bfrac_s)} × {len(bfrac_s.columns)}) ...", flush=True)
    p_raw = _fast_compute_posterior_10pct(
        bfrac_s=bfrac_s, depth_s=depth_s, c=c,
        sigma_bins=sigma_bins, sigma_edges=sigma_edges,
        genome_lengths=genome_lengths, pi_present=pi_present,
        n_workers=n_workers,
    )

    if level == "study" and study_col is not None:
        n_u    = meta_s[study_col].nunique()
        xlabel = "Number of studies"
        print(f"      Study collapse: {len(bfrac_s)} samples → {n_u} studies "
              f"(γ={gamma:.2f})", flush=True)
        p_study_raw = _gamma_collapse_to_study_posteriors(
            p_df=p_raw, meta_s=meta_s, study_col=study_col,
            gamma=gamma, genome_lengths=genome_lengths, pi_present=pi_present,
        )
        # Single prior subtraction at study level
        p_acc = _apply_prior_subtraction(p_study_raw, genome_lengths, pi_present)
        N = n_u
    else:
        # Single prior subtraction at sample level
        p_acc  = _apply_prior_subtraction(p_raw, genome_lengths, pi_present)
        N      = len(bfrac_s)
        xlabel = "Number of samples"

    ns = _logspace_checkpoints(N, n_pts=n_pts)
    print(f"      Bootstrap ({n_iter} iter) ...", flush=True)
    curves, ns = _posterior_accumulation(p_acc, n_iter=n_iter, rng=rng, checkpoints=ns)

    # Uncertainty note (L2): S_sd captures bootstrap ordering uncertainty only.
    # It does not propagate uncertainty in the detection model, pi_present,
    # gamma, sigma, or study-level sampling variation.  Treat it as a
    # conditional plotting envelope, not a full credible interval for S_inf.
    return dict(ns=ns, curves=curves,
                S_mean=curves.mean(0),
                S_sd  =curves.std(0),                        # ← 1 SD (Fix C)
                S_sem =curves.std(0) / max(np.sqrt(n_iter), 1.0),  # kept for compat
                N_total=N, xlabel=xlabel)


# ── Panel drawing (unchanged from original) ───────────────────────────────────
def _yticks_integer(ax, y_max, step=1000):
    ticks = list(range(0, int(y_max) + step, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks], fontsize=_FS)


def _format_int(n):
    return f"{int(round(n)):,}"


def draw_panel(ax, results_by_ls, S_ref, ref_label,
               show_title, show_xlabel, show_ylabel, show_row_label,
               row_label="", extrap_factor=2.0, ci_level=0.95,
               n_iter_asym=50, rng_asym=None):
    """
    Draw one subplot panel (one reference × one rarefaction level).

    Fix C (plotting):
    - y-axis hard-capped at Y_CAP = 4000 SGBs.
    - Gridlines off (confirmed via rcParams).
    - Error bars on scatter points show ± 1 SD (bootstrap std), not SEM.
      Interpretation: captures ordering uncertainty conditional on the dataset.
      Does NOT propagate model, hyperparameter, or sampling uncertainty (L2).
    - Inset enlarged to ~2/3 of subplot dimensions and shifted upward.
    - Asymptote annotations shifted downward and vertically repelled to avoid
      crowding at the top of the axes.
    """
    if rng_asym is None:
        rng_asym = np.random.default_rng(0)

    Y_CAP = 4000   # Fix C: hard ceiling regardless of S_ref

    max_N = max(r["N_total"] for r in results_by_ls.values())
    xlim  = max_N * extrap_factor
    y_lim = Y_CAP   # Fix C

    fits = {}; r2s = {}
    for ls in LS_ORDER:
        if ls not in results_by_ls:
            continue
        r  = results_by_ls[ls]
        ns = r["ns"]; S_mean = r["S_mean"]
        S_sd = r.get("S_sd", r.get("S_sem", None))     # Fix C: prefer SD
        # Pass ALL curves so the mean is estimated from the full ensemble.
        # n_iter_asym controls parametric bootstrap iterations inside _asymptote_ci.
        Sm_val, lo, hi, mm = _asymptote_ci(
            r["curves"], ns, S_ref, ci=ci_level,
            n_boot_asym=n_iter_asym, rng=rng_asym,
        )
        fits[ls] = (Sm_val, lo, hi, mm, ns, S_mean, S_sd, r["N_total"])
        r2s[ls]  = mm.get("r2", np.nan)

    ax.set_xlim(0, xlim); ax.set_ylim(0, y_lim)
    ax.tick_params(labelsize=_FS)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _yticks_integer(ax, y_lim, step=1000)

    if show_xlabel:
        ax.set_xlabel(next(iter(results_by_ls.values()))["xlabel"],
                      fontsize=_FSL, labelpad=4)
    if show_ylabel:
        # Fix L3: "evidence-weighted accumulation score", not "posterior expectation"
        ylabel = "Per-sample rarefaction\nNumber of SGBs detected" \
                 if show_row_label and row_label and "sample" in row_label.lower() \
                 else ("Per-study rarefaction\nNumber of SGBs detected"
                       if show_row_label and row_label else
                       "Number of SGBs detected")
        ax.set_ylabel(ylabel, fontsize=_FSL - 1, labelpad=4)
    if show_title:
        ax.set_title(ref_label, fontsize=_FST, fontweight="bold", pad=6)

    ax.axvspan(max_N, xlim, color="#f4f4f4", alpha=0.7, zorder=0)
    asym_ypos = {}

    for ls in LS_ORDER:
        if ls not in fits:
            continue
        Sm_val, lo, hi, mm, ns, S_mean, S_sd, N_tot = fits[ls]
        col = LS_COLORS[ls]; mk = LS_MARKERS[ls]

        if np.isfinite(Sm_val):
            hi_band = min(hi, y_lim) if np.isfinite(hi) else y_lim
            lo_band = max(lo, 0)     if np.isfinite(lo) else 0
            if lo_band < hi_band:
                ax.axhspan(lo_band, hi_band, color=col, alpha=0.12, zorder=1)
            ax.axhline(min(Sm_val, y_lim), color=col, lw=1.0, ls=":", alpha=0.75, zorder=2)
            asym_ypos[ls] = Sm_val

        # Fix C: yerr = 1 SD (bootstrap std), not SEM
        if S_sd is not None:
            ax.errorbar(ns, S_mean, yerr=S_sd, fmt="none", ecolor=col,
                        elinewidth=0.6, capsize=1.5, capthick=0.6, alpha=0.35, zorder=3)
        ax.scatter(ns, S_mean, color=col, s=4**2, marker=mk,
                   zorder=4, linewidths=0, label=ls)

        if mm["success"]:
            ns_ext = np.linspace(1, xlim, 600)
            nh = ns_ext ** mm["h"]; Kh = mm["K"] ** mm["h"]
            pred = np.minimum(mm["S_max"] * nh / (Kh + nh), y_lim)
            ax.plot(ns_ext[ns_ext <= N_tot], pred[ns_ext <= N_tot],
                    color=col, lw=2.0, zorder=5)
            ax.plot(ns_ext[ns_ext > N_tot],  pred[ns_ext > N_tot],
                    color=col, lw=1.5, ls="--", alpha=0.55, zorder=5)

    # Fix C: annotations — lower, with vertical repulsion, capped below y_lim
    ann_positions = {}
    for ls in LS_ORDER:
        if ls in asym_ypos and np.isfinite(asym_ypos[ls]):
            ann_positions[ls] = min(asym_ypos[ls] * 0.92, y_lim * 0.86)

    if len(ann_positions) == 2:
        ls0, ls1 = LS_ORDER[0], LS_ORDER[1]
        if ls0 in ann_positions and ls1 in ann_positions:
            min_gap = y_lim * 0.07   # 7% of y_lim minimum separation
            gap = abs(ann_positions[ls0] - ann_positions[ls1])
            if gap < min_gap:
                mid = (ann_positions[ls0] + ann_positions[ls1]) / 2
                # Keep ordering consistent with asymptote ordering
                y0_higher = asym_ypos.get(ls0, 0) >= asym_ypos.get(ls1, 0)
                ann_positions[ls0] = mid + (min_gap / 2 if y0_higher else -min_gap / 2)
                ann_positions[ls1] = mid + (-min_gap / 2 if y0_higher else min_gap / 2)
            # Clamp both into safe range
            for ls in LS_ORDER:
                if ls in ann_positions:
                    ann_positions[ls] = float(np.clip(
                        ann_positions[ls], y_lim * 0.04, y_lim * 0.90))

    for ls in LS_ORDER:
        if ls not in fits or ls not in ann_positions:
            continue
        Sm_val, lo, hi, mm, *_ = fits[ls]
        if not np.isfinite(Sm_val):
            continue
        col   = LS_COLORS[ls]
        y_ann = ann_positions[ls]
        ann = (f"S∞ {_format_int(Sm_val)} [{_format_int(lo)}–{_format_int(hi)}]"
               if np.isfinite(lo) and np.isfinite(hi)
               else f"S∞ {_format_int(Sm_val)}")
        ax.text(xlim * 0.02, y_ann, ann,
                color=col, fontsize=_FS - 1, va="center",
                fontweight="bold", zorder=10, clip_on=True)

    r2_lines = [f"{ls[:3]}: R²={r2s[ls]:.3f}"
                for ls in LS_ORDER if ls in r2s and np.isfinite(r2s[ls])]
    if r2_lines:
        ax.text(0.97, 0.97, "\n".join(r2_lines), transform=ax.transAxes,
                fontsize=_FS - 1, va="top", ha="right", color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cccccc", alpha=0.85), zorder=11)

    # Fix C: larger inset (~2/3 of subplot each dimension) shifted upward
    ax_ins = ax.inset_axes([0.34, 0.10, 0.64, 0.58])
    ax_ins.set_xscale("log"); ax_ins.set_xlim(1, xlim); ax_ins.set_ylim(0, y_lim)
    ax_ins.spines["top"].set_visible(False); ax_ins.spines["right"].set_visible(False)
    ax_ins.tick_params(labelsize=_FS - 2, which="both")
    ax_ins.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ax_ins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax_ins.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    for ls in LS_ORDER:
        if ls not in fits:
            continue
        Sm_val, lo, hi, mm, ns, S_mean, S_sd, N_tot = fits[ls]
        col = LS_COLORS[ls]; mk = LS_MARKERS[ls]
        # Fix C: SD bars in inset too
        if S_sd is not None:
            ax_ins.errorbar(ns, S_mean, yerr=S_sd, fmt="none", ecolor=col,
                            elinewidth=0.5, capsize=1.0, capthick=0.5, alpha=0.30, zorder=3)
        ax_ins.scatter(ns, S_mean, color=col, s=3**2, marker=mk,
                       zorder=4, linewidths=0)
        if mm["success"]:
            ns_ext = np.logspace(0, np.log10(xlim), 400)
            nh = ns_ext ** mm["h"]; Kh = mm["K"] ** mm["h"]
            pred = np.minimum(mm["S_max"] * nh / (Kh + nh), y_lim)
            ax_ins.plot(ns_ext[ns_ext <= N_tot], pred[ns_ext <= N_tot],
                        color=col, lw=1.5, zorder=5)
            ax_ins.plot(ns_ext[ns_ext > N_tot],  pred[ns_ext > N_tot],
                        color=col, lw=1.2, ls="--", alpha=0.55, zorder=5)

    ax_ins.legend(
        handles=[Patch(facecolor=LS_COLORS[ls], label=ls, alpha=0.85)
                 for ls in LS_ORDER if ls in fits],
        loc="upper left", fontsize=_FS - 2, framealpha=0.92,
        edgecolor="#cccccc", handlelength=1.2, handleheight=0.8,
        borderpad=0.4, labelspacing=0.3)
    ax_ins.set_xlabel("log scale", fontsize=_FS - 2, labelpad=2, color="#555")
    ax_ins.set_ylabel("SGBs",      fontsize=_FS - 2, labelpad=2, color="#555")
    return ax_ins


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module-path",  default=None)
    p.add_argument("--ref-carter",   default="UHGG")
    p.add_argument("--ref-lev",      default="lev")
    p.add_argument("--out",          default="accumulation_summary_v3.png")
    p.add_argument("--n-iter",       type=int,   default=200)
    p.add_argument("--n-iter-asym",  type=int,   default=100)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--extrap",       type=float, default=2.0)
    p.add_argument("--ci",           type=float, default=0.95)
    p.add_argument("--n-workers",    type=int,   default=4)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    rng  = np.random.default_rng(args.seed)

    refs = {
        "carter": (args.ref_carter, "Carter"),
        "lev":    (args.ref_lev,    "Leviatan"),
    }

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
                    print(f"    skip {ls}: only {len(samps)} samples"); continue
                n_units = samps.nunique() if level == "sample" else \
                          d["meta"].loc[mask, d["study_col"]].nunique() \
                          if d["study_col"] else len(samps)
                print(f"    {ls}  n_samples={len(samps)}  n_{level}s={n_units}")
                results[rid][level][ls] = compute_accumulation(
                    bfrac_s=d["bfrac"].loc[samps],
                    depth_s=d["depth"].loc[samps],
                    meta_s =d["meta"].loc[samps],
                    genome_lengths=d["genome_lengths"],
                    level=level, n_iter=args.n_iter, rng=rng,
                    study_col=d["study_col"],
                    sigma_bins=d["sigma_bins"], sigma_edges=d["sigma_edges"],
                    c=d["c"],
                    pi_present=d["pi_present"],
                    gamma=d["gamma"],
                    n_workers=args.n_workers,
                )

    print("\n══ Drawing figure ══")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             constrained_layout=True, squeeze=False)
    row_labels = ["Per-sample rarefaction", "Per-study rarefaction"]

    for row_idx, level in enumerate(["sample", "study"]):
        for col_idx, (rid, (key, label)) in enumerate(refs.items()):
            ax     = axes[row_idx, col_idx]
            res_ls = results[rid][level]
            S_ref  = ref_data[rid]["S_ref"]
            if not res_ls:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_FSL); continue
            draw_panel(
                ax=ax, results_by_ls=res_ls, S_ref=S_ref, ref_label=label,
                show_title=(row_idx == 0), show_xlabel=True,
                show_ylabel=(col_idx == 0), show_row_label=(col_idx == 0),
                row_label=row_labels[row_idx], extrap_factor=args.extrap,
                ci_level=args.ci, n_iter_asym=args.n_iter_asym, rng_asym=rng,
            )

    fig.text(
        0.5, -0.015,
        "y-axis: evidence-weighted accumulation score E[S(k)] — prior-subtracted excess posteriors "
        "fed into union-style rarefaction (not literal posterior expected richness; see L3).  "
        "S∞: Hill-MM asymptote fit to the mean curve.  "
        "Shaded band: 95% CI from parametric bootstrap on the mean curve (S_mean ± S_sem noise → refit); "
        "reflects ordering uncertainty in the mean, NOT biological or model uncertainty (see L2).  "
        "Error bars on data points: ± 1 bootstrap SD.  "
        "σ_b: left-half MAD.  pi & gamma: consistent LOSO scan (Fixes A+B).  "
        "Prior subtracted once at aggregation level.",
        ha="center", va="top", fontsize=_FS - 2, color="#555", style="italic",
    )

    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
