#!/usr/bin/env python3
"""Additional Figures A16 & A17 — finite-catalogue S-infinity accumulation.

Regenerates the two combined-reference accumulation-diagnostic composites:
  - A16: SGB accumulation under increasingly stringent per-sample breadth
    thresholds (>=10% / >=30% / >=50%).            -> figure_a16_*
  - A17: SGB accumulation requiring detection in at least k samples
    (k = 1, 2, 3, 5).                              -> figure_a17_*

Each is a rows x 3-column grid: Accumulation curve | Support for S_inf
(profile-likelihood landscape with the chi^2 95% cutoff) | Bootstrap S_inf
(replicate asymptote histogram). Ported from the manuscript analysis
(`combined_summary_two_figures.py`, 2026-06-05); paths made repo-relative and
the numpy-2 bootstrap caches vendored under accumulation_sinf_inputs/ as
numpy-portable pickles. See accumulation_sinf_inputs/README.md for the upstream
precompute chain.

    python tools/make_accumulation_sinf_figures.py
"""
from __future__ import annotations

import os
import pickle
import sys
import hashlib
from pathlib import Path

# Restored from archive/pipeline_code_20260814.tar.gz, where it sat at the
# repository root; it now lives in tools/, so the root is one level up.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize, lsq_linear
from scipy.special import betaln, gammaln, logsumexp


INPUTS = ROOT / "accumulation_sinf_inputs"
_VENDORED_CACHES = INPUTS / "caches"
_REBUILT = ROOT / "results" / "Figure3_combined"


def _cache_in(name: str) -> Path:
    """A bootstrap cache, preferring one rebuilt on the current cohort.

    The vendored caches are the 5 June fits. Rebuilding them is the whole point
    of the exercise, and stage 1 writes the rebuilt ones beside the figures
    under the same names -- but this module read the vendored directory
    unconditionally, so a full rebuild still drew the old cohort and failed the
    consistency check exactly as before.
    """
    rebuilt = _REBUILT / name
    if rebuilt.exists():
        return rebuilt
    return _VENDORED_CACHES / name
# The vendored profiles are the 5 June fits, on the cohort of that date. After
# a rebuild the freshly scored ones live beside the figures, so prefer those
# when they exist -- otherwise a rebuild would draw new bootstrap data against
# old mixture fits, which is the mismatch this whole exercise is about.


def _profiles_dir():
    import os as _os
    env = _os.environ.get("SINF_PROFILES")
    if env:
        return Path(env)
    for k in (2, 3, 5):
        if not (_REBUILT / f"profile_sinf_mixture_min{k}" /
                f"profile_mix_k{k}_best_by_bic_asymptotes.csv").exists():
            return INPUTS / "profiles"
    print("  [sinf] using rebuilt mixture profiles from results/Figure3_combined")
    return _REBUILT


PROFILES = INPUTS / "profiles"
OUT = ROOT / "results" / "Figure3_combined"
CACHE = OUT / "sinf_cache"
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402
from persistence.accumulation import (  # noqa: E402
    _median_fit_scale,
    bootstrap_curve_variable_reps,
    fit_richards_weighted,
    richards_logN_abs,
)


class _CompatUnpickler(pickle.Unpickler):
    """Read caches whether written under numpy 1.x (numpy.core) or 2.x (numpy._core)."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as fh:
        return _CompatUnpickler(fh).load()


IND = "#56B4E9"
NON = "#E69F00"
IND_DARK = "#0072B2"
NON_DARK = "#D55E00"


# Line widths were tuned on a 17.5 in canvas that printed at 0.42 of its
# size; the canvas is now the printed width, so the same numbers drew every
# trace 2.4x as heavy. Scale them back to what printed, never below the 0.5 pt
# floor.
_LW_SCALE = 7.24 / 17.5


def _w(pt):
    return max(pt * _LW_SCALE, 0.5)


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "axes.linewidth": _w(1.45),
    }
)


def r2_rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan, float(np.sqrt(np.mean((y - yhat) ** 2)))


def sigma_from_band(sd):
    sig = (np.asarray(sd["q84_vec"], float) - np.asarray(sd["q16_vec"], float)) / 2.0
    repl = np.nanmedian(sig[sig > 0]) if np.any(sig > 0) else 1.0
    sig[~np.isfinite(sig) | (sig <= 0)] = repl
    return sig


def threshold_tag(threshold):
    return f"{threshold:g}".replace(".", "p") + "pct"


def _normalise_labels(fig):
    """Publication labels, on a figure that does not go through savefig_dual."""
    try:
        from persistence.plot_utils import apply_display_text
        apply_display_text(fig)
    except Exception as exc:
        print(f"  [labels] could not normalise display text: {exc}")


def _panels_of(payload):
    """The per-reference panels, whichever cache layout this is.

    The vendored June caches are a bare mapping of reference to panels. The
    ones make_combined_figure3.py writes now nest that under "panels" and carry
    schema, threshold, cohort and input provenance beside it -- a better
    format, and the reason a rebuilt cache read as an empty dictionary here.
    """
    return payload.get("panels", payload) if isinstance(payload, dict) else payload


def load_threshold_panel(threshold):
    tag = threshold_tag(threshold)
    path = _cache_in(f"figure3_combined_regenerated_{tag}_bootstrap_cache.pkl")
    return _panels_of(_load_pickle(path))["combined"][0]


def load_k_panel(k):
    if k == 1:
        return load_k1_panel_with_boot(10.0)
    path = _cache_in(
        f"figure3_combined_regenerated_10pct_min{k}_rsac_bootstrap_replicates_cache.pkl")
    return _panels_of(_load_pickle(path))["combined"][0]


def load_breadth_and_meta():
    # The matrix the rest of the pipeline reads, not the July one this script
    # was written against. Hard-coding the older file meant A16 and A17 were
    # built on a cohort 250 Non-Industrial samples short of every other figure,
    # which is exactly the inconsistency this regeneration exists to remove.
    breadth_path = ROOT / config.REFERENCES["combined"]["breadth_file"]
    if not breadth_path.exists():
        raise FileNotFoundError(
            f"Combined coverage matrix not found: {breadth_path}")
    print(f"  [sinf] coverage matrix: {breadth_path.name}", flush=True)
    meta_path = next((ROOT / p for p in config.META_FILE_CANDIDATES if (ROOT / p).exists()), None)
    if meta_path is None:
        raise FileNotFoundError("Could not resolve metadata file")
    breadth = pd.read_csv(
        breadth_path,
        index_col=0,
        low_memory=False,
        usecols=lambda c: c == "Sample_ID" or str(c).endswith("_mean"),
    )
    breadth.index = breadth.index.astype(str).str.strip()
    breadth.columns = [c.rsplit("_mean", 1)[0] for c in breadth.columns]
    breadth = breadth.fillna(0.0)
    meta = pd.read_csv(meta_path, low_memory=False)
    # M1 carries the published labels -- Hunter-gatherer, Agro-pastoralist --
    # while the analysis code compares against the internal keys H-G and
    # Transitional. config.canonicalize_lifestyle_series bridges them, and this
    # script predates it: without the call the Non-Industrial mask matched
    # nothing, the regenerated grid came back empty, and the consistency check
    # reported it as a cohort mismatch.
    if "Lifestyle" in meta.columns and hasattr(config, "canonicalize_lifestyle_series"):
        meta["Lifestyle"] = config.canonicalize_lifestyle_series(meta["Lifestyle"])
    meta["Chosen_match_ID"] = meta["Chosen_match_ID"].astype(str).str.strip()
    meta = meta[meta["Chosen_match_ID"].isin(breadth.index)].copy()
    meta = meta.set_index("Chosen_match_ID", drop=False)
    return breadth, meta


_BREADTH_META = None


def ensure_k1_boot_matrix(panel, threshold):
    """Attach real sample-level k=1 bootstrap matrices to an existing panel."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"k1_combined_{threshold_tag(threshold)}_bootstrap_replicates.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)["sample_panel"]

    global _BREADTH_META
    if _BREADTH_META is None:
        print("Reading combined coverage matrix once for k=1 bootstrap replicate caches...", flush=True)
        _BREADTH_META = load_breadth_and_meta()
    breadth, meta = _BREADTH_META
    rng = np.random.default_rng(123 + int(threshold * 10))
    masks = {
        "Industrial": meta["Lifestyle"] == "Industrial",
        "Non-Industrial": meta["Lifestyle"].isin(["H-G", "Transitional"]),
    }
    out_panel = []
    for sd in panel:
        sd = dict(sd)
        ids = masks[sd["name"]][masks[sd["name"]]].index.intersection(breadth.index)
        block = breadth.loc[ids]
        Ns = np.asarray(sd["N_vec"], int)
        boot_counts = {int(n): config.ACCUM_NUM_INNER_BOOT for n in Ns}
        N_vec, _mean, _std, n_mic, _med, _q16, _q84, boot_matrix = bootstrap_curve_variable_reps(
            block, Ns, boot_counts, rng, float(threshold), return_boot_matrix=True
        )
        if not np.array_equal(np.asarray(N_vec, int), Ns):
            raise RuntimeError(f"k=1 regenerated N grid mismatch for {sd['name']} threshold {threshold}")
        sd["boot_matrix"] = np.asarray(boot_matrix, dtype=np.float32)
        sd["n_mic"] = int(n_mic)
        out_panel.append(sd)
    with cache_path.open("wb") as fh:
        pickle.dump({"sample_panel": out_panel, "threshold": float(threshold)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out_panel


def load_k1_panel_with_boot(threshold):
    exact_cache = _cache_in("k1_combined_10pct_bootstrap_replicates.pkl")
    if float(threshold) == 10.0 and exact_cache.exists():
        return _load_pickle(exact_cache)["sample_panel"]
    return ensure_k1_boot_matrix(load_threshold_panel(threshold), threshold)


def mm(n, s, kscale):
    return s * n / (kscale + n)


def gen_mm(n, s, kscale, alpha):
    return s * (1.0 - np.power(1.0 + n / kscale, -alpha))


def hill(n, s, kscale, h):
    return s * np.power(n, h) / (np.power(kscale, h) + np.power(n, h))


def fit_first_detection(sd):
    n = np.asarray(sd["N_vec"], float)
    y = np.asarray(sd["y_center"], float)
    sig = sigma_from_band(sd)
    m = float(sd["n_mic"])
    ymax = max(float(np.nanmax(y)), 1.0)
    xmax = max(float(np.nanmax(n)), 1.0)
    candidates = {}
    specs = {
        "MM": (lambda x, s, kscale: mm(x, s, kscale), [ymax, 1e-6], [m, xmax * 100.0], [min(m, ymax * 1.1), np.median(n)]),
        "GenMM": (lambda x, s, kscale, a: gen_mm(x, s, kscale, a), [ymax, 1e-6, 1e-4], [m, xmax * 100.0, 10.0], [min(m, ymax * 1.1), np.median(n), 1.0]),
        "Hill": (lambda x, s, kscale, h: hill(x, s, kscale, h), [ymax, 1e-6, 0.05], [m, xmax * 100.0, 8.0], [min(m, ymax * 1.1), np.median(n), 1.0]),
    }
    for name, (func, lo, hi, p0) in specs.items():
        try:
            p, _ = curve_fit(func, n, y, p0=p0, bounds=(lo, hi), sigma=sig, maxfev=50000)
            yhat = func(n, *p)
            candidates[name] = {"model": name, "shape": yhat / p[0], "yhat": yhat, "S": float(p[0]), "r2": r2_rmse(y, yhat)[0]}
        except Exception:
            pass
    p, rr = fit_richards_weighted(n, y, sig, m)
    if p is not None:
        yhat = richards_logN_abs(np.log(n), *p)
        candidates["Richards"] = {"model": "Richards", "shape": yhat / p[1], "yhat": yhat, "S": float(p[1]), "r2": rr}
    return candidates[max(candidates, key=lambda name: candidates[name]["r2"])]


def beta_binom_logpmf(r, n, alpha, beta):
    return (
        gammaln(n + 1.0)
        - gammaln(r + 1.0)
        - gammaln(n - r + 1.0)
        + betaln(r + alpha, n - r + beta)
        - betaln(alpha, beta)
    )


def beta_tail(n, k, alpha, beta):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n, dtype=float)
    ok = n >= k
    if np.any(ok):
        cdf = np.zeros(np.sum(ok), dtype=float)
        for r in range(k):
            cdf += np.exp(beta_binom_logpmf(r, n[ok], alpha, beta))
        out[ok] = np.clip(1.0 - cdf, 0.0, 1.0)
    return out


def mix_params(row):
    h = int(row["H"])
    w = np.array([row.get(f"w{i}", np.nan) for i in range(1, h + 1)], dtype=float)
    a = np.array([row.get(f"alpha{i}", np.nan) for i in range(1, h + 1)], dtype=float)
    b = np.array([row.get(f"beta{i}", np.nan) for i in range(1, h + 1)], dtype=float)
    w /= np.nansum(w)
    return w, a, b


def equal_mix_shape(n, k, row):
    w, a, b = mix_params(row)
    out = np.zeros_like(np.asarray(n, float), dtype=float)
    for ww, aa, bb in zip(w, a, b):
        out += ww * beta_tail(n, k, aa, bb)
    return out


def unpack_mix_theta(theta, h):
    theta = np.asarray(theta, dtype=float)
    if h > 1:
        logits = np.r_[theta[: h - 1], 0.0]
        logits = logits - np.max(logits)
        w = np.exp(logits)
        w = w / np.sum(w)
        off = h - 1
    else:
        w = np.array([1.0])
        off = 0
    a = np.exp(theta[off : off + h])
    b = np.exp(theta[off + h : off + 2 * h])
    return w, a, b


def mix_shape_from_theta(n, k, theta, h):
    w, a, b = unpack_mix_theta(theta, h)
    out = np.zeros_like(np.asarray(n, float), dtype=float)
    for ww, aa, bb in zip(w, a, b):
        out += ww * beta_tail(n, k, aa, bb)
    return np.clip(out, 0.0, 1.0)


def init_mix_theta(row):
    w, a, b = mix_params(row)
    h = len(w)
    parts = []
    if h > 1:
        logits = np.log(np.clip(w[:-1], 1e-8, 1.0)) - np.log(np.clip(w[-1], 1e-8, 1.0))
        parts.extend(np.clip(logits, -6.0, 6.0))
    parts.extend(np.log(np.clip(a, 1e-8, 1e4)))
    parts.extend(np.log(np.clip(b, 1e-6, 1e8)))
    return np.asarray(parts, dtype=float)


def fit_beta_mix_curve(y, n, sigma, k, h, n_mic, init_row):
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    n_all = n.copy()
    sigma = np.asarray(sigma, dtype=float)
    m = np.isfinite(y) & np.isfinite(n) & np.isfinite(sigma) & (sigma > 0)
    if m.sum() < 5:
        return None
    y, n, sigma = y[m], n[m], sigma[m]
    s_min = max(float(np.nanmax(y)), 1.0)
    s_max = float(n_mic)
    theta0 = init_mix_theta(init_row)
    shape0 = mix_shape_from_theta(n, k, theta0, h)
    s0 = best_s(y, shape0, s_min, s_max, sigma)
    x0 = np.r_[s0, theta0]

    theta_bounds = []
    if h > 1:
        theta_bounds.extend([(-8.0, 8.0)] * (h - 1))
    theta_bounds.extend([(-8.0, 4.0)] * h)
    theta_bounds.extend([(-4.0, 18.0)] * h)
    bounds = [(s_min, s_max)] + theta_bounds

    def obj(x):
        s = float(x[0])
        shape = mix_shape_from_theta(n, k, x[1:], h)
        resid = (y - s * shape) / sigma
        return 0.5 * float(np.sum(resid * resid))

    starts = [x0]
    # A second start with a heavier rare tail helps avoid local minima at the cap.
    rare = theta0.copy()
    off = h - 1 if h > 1 else 0
    rare[off : off + h] = np.minimum(rare[off : off + h], np.log(0.25))
    rare[off + h : off + 2 * h] = np.maximum(rare[off + h : off + 2 * h], np.log(max(np.nanmax(n), 10.0)))
    starts.append(np.r_[min(s_max, max(s_min, s0)), rare])

    best = None
    for start in starts:
        try:
            res = minimize(obj, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 450, "ftol": 1e-7})
        except Exception:
            continue
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    shape_full = mix_shape_from_theta(n_all, k, best.x[1:], h)
    return {
        "S": float(best.x[0]),
        "theta": np.asarray(best.x[1:], dtype=float),
        "nll": float(best.fun),
        "shape_fit_n": shape_full,
    }


def fit_beta_mix_masses_curve(y, n, sigma, k, n_mic, init_row):
    """Fast bootstrap refit: fixed beta components, refit component masses.

    The coefficients are c_h = S_inf * w_h. Their sum is the replicate-specific
    asymptote, and the normalized coefficients define the replicate-specific
    mixture shape.
    """
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w0, a, b = mix_params(init_row)
    tails = np.vstack([beta_tail(n, k, aa, bb) for aa, bb in zip(a, b)]).T
    m = np.isfinite(y) & np.all(np.isfinite(tails), axis=1) & np.isfinite(sigma) & (sigma > 0)
    if m.sum() < 5:
        return None
    xw = tails[m] / sigma[m, None]
    yw = y[m] / sigma[m]
    try:
        res = lsq_linear(xw, yw, bounds=(0.0, float(n_mic)), max_iter=250, tol=1e-8)
    except Exception:
        return None
    if not res.success and res.x is None:
        return None
    coeff = np.clip(np.asarray(res.x, dtype=float), 0.0, float(n_mic))
    s = float(np.sum(coeff))
    if s > float(n_mic) and s > 0:
        coeff *= float(n_mic) / s
        s = float(n_mic)
    if s <= 0:
        coeff = np.asarray(w0, dtype=float) * max(float(np.nanmax(y)), 1.0)
        s = float(np.sum(coeff))
    weights = coeff / max(s, 1e-12)
    shape = tails @ weights
    yhat = s * shape
    nll = 0.5 * float(np.sum(np.square((y[m] - yhat[m]) / sigma[m])))
    return {"S": s, "weights": weights, "shape_fit_n": shape, "nll": nll}


def fit_repeated(sd, k):
    _pd_ = _profiles_dir()
    _f = (_pd_ / f"profile_sinf_mixture_min{k}" /
          f"profile_mix_k{k}_best_by_bic_asymptotes.csv")
    if not _f.exists():
        _f = _pd_ / f"profile_mix_k{k}_best_by_bic_asymptotes.csv"
    best_mix = pd.read_csv(_f)
    row = best_mix[(best_mix["ref"] == "combined") & (best_mix["unit"] == "samples") & (best_mix["series"] == sd["name"])].iloc[0]
    n = np.asarray(sd["N_vec"], float)
    y = np.asarray(sd["y_center"], float)
    sig = sigma_from_band(sd)
    shape = equal_mix_shape(n, k, row)
    s = best_s(y, shape, np.nanmax(y), float(sd["n_mic"]), sig)
    yhat = s * shape
    r2, rmse = r2_rmse(y, yhat)
    return {
        "model": f"H={int(row['H'])} beta-mix",
        "shape": shape,
        "S": s,
        "yhat": yhat,
        "r2": r2,
        "rmse": rmse,
        "k": int(k),
        "H": int(row["H"]),
        "mix_row": row.to_dict(),
    }


def best_s(y, shape, s_min, s_max, sigma=None):
    wt = np.ones_like(y) if sigma is None else 1.0 / np.square(sigma)
    denom = float(np.sum(wt * shape * shape))
    if denom <= 0:
        return float(s_min)
    return float(np.clip(np.sum(wt * y * shape) / denom, s_min, s_max))


def profile_delta(y, sigma, shape, s_grid):
    vals = np.array([0.5 * float(np.sum(np.square((y - s * shape) / sigma))) for s in s_grid])
    return 2.0 * (vals - vals.min())


def profile_nll_fixed_shape(y, sigma, shape, s_grid):
    return np.array([0.5 * float(np.sum(np.square((y - s * shape) / sigma))) for s in s_grid])


def fit_cache_key(sd, fit):
    boot = np.asarray(sd["boot_matrix"], float)
    n = np.asarray(sd["N_vec"], float)
    h = hashlib.sha1()
    h.update(b"bootstrap-refit-v3-bootstrap-mixture-likelihood")
    h.update(str(sd.get("name", "")).encode())
    h.update(str(fit.get("model", "")).encode())
    h.update(str(fit.get("k", "k1")).encode())
    h.update(np.asarray(boot.shape, dtype=np.int64).tobytes())
    h.update(np.nan_to_num(boot[:, -1] if boot.ndim == 2 and boot.shape[1] else boot.ravel(), nan=-1.0).astype("float64").tobytes())
    h.update(np.nan_to_num(n, nan=-1.0).astype("float64").tobytes())
    return h.hexdigest()[:16]


def bootstrap_replicate_refits(sd, fit):
    if "_bootstrap_refits" in fit:
        return fit["_bootstrap_refits"]

    cache_path = CACHE / f"bootstrap_refits_{fit_cache_key(sd, fit)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            fit["_bootstrap_refits"] = pickle.load(fh)
            return fit["_bootstrap_refits"]

    boot = np.asarray(sd["boot_matrix"], float)
    n_vec = np.asarray(sd["N_vec"], float)
    n_mic = float(sd["n_mic"])
    scale_vec = _median_fit_scale(boot)
    refits = []

    if str(fit.get("model", "")).lower() == "richards":
        for b, rep in enumerate(boot):
            m = np.isfinite(rep) & np.isfinite(n_vec)
            if m.sum() < 5 or not np.isfinite(rep[m][-1]):
                continue
            params, r2 = fit_richards_weighted(n_vec[m], rep[m], scale_vec[m], n_mic)
            if params is None or not np.isfinite(params[1]):
                continue
            yhat = richards_logN_abs(np.log(n_vec), *params)
            refits.append({"bootstrap": b, "S": float(params[1]), "shape": yhat / float(params[1]), "r2": float(r2)})
    else:
        k = int(fit["k"])
        init_row = pd.Series(fit["mix_row"])
        for b, rep in enumerate(boot):
            res = fit_beta_mix_masses_curve(rep, n_vec, scale_vec, k, n_mic, init_row)
            if res is None or not np.isfinite(res["S"]):
                continue
            refits.append({"bootstrap": b, "S": float(res["S"]), "shape": res["shape_fit_n"], "r2": np.nan})

    fit["_bootstrap_refits"] = refits
    with cache_path.open("wb") as fh:
        pickle.dump(refits, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return refits


def prepare_threshold_results():
    rows = []
    for thr in (10.0, 30.0, 50.0):
        panel = load_k1_panel_with_boot(thr)
        row = {}
        for sd in panel:
            row[sd["name"]] = {"sd": sd, "fit": fit_first_detection(sd)}
        rows.append((f">={thr:g}% breadth", row))
    return rows


def prepare_k_results():
    rows = []
    for k in (1, 2, 3, 5):
        panel = load_k_panel(k)
        row = {}
        for sd in panel:
            row[sd["name"]] = {"sd": sd, "fit": fit_first_detection(sd) if k == 1 else fit_repeated(sd, k)}
        label = f">={k} sample" if k == 1 else f">={k} samples"
        rows.append((label, row))
    return rows


def style_axis(ax, bottom=True):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(length=6 * _LW_SCALE, width=_w(1.35))
    if not bottom:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.grid(axis="y", color="0.9", lw=_w(0.9))


def _landscape_for(entry):
    """The profile-likelihood landscape, published or recomputed.

    A redraw from the data object carries it already; the pipeline does not,
    and refits it from the series and its fit as before.
    """
    cached = entry.get("landscape")
    if cached is not None:
        return cached
    return bootstrap_profile_landscape(entry["sd"], entry["fit"])


def _boot_vals_for(entry):
    """The bootstrap asymptote replicates, published or recomputed."""
    cached = entry.get("boot_vals")
    if cached is not None:
        return np.asarray(cached, dtype=float)
    return bootstrap_asymptote_values(entry["sd"], entry["fit"])


def add_row_accum(ax, row, bottom=True, show_legend=False):
    r2_text = []
    for name, color, dark in (("Industrial", IND, IND_DARK), ("Non-Industrial", NON, NON_DARK)):
        sd = row[name]["sd"]
        fit = row[name]["fit"]
        n = np.asarray(sd["N_vec"], float)
        y = np.asarray(sd["y_center"], float)
        ax.fill_between(n, sd["q16_vec"], sd["q84_vec"], color=color, alpha=0.12, lw=0)
        ax.plot(n, y, "o", ms=3.0 * _LW_SCALE, color=color, alpha=0.45)
        # The fit quality rides in the key rather than in a second box: both
        # want the one clear corner of an accumulation panel, and the legend
        # is where the reader is already looking to identify the series.
        ax.plot(n, fit["yhat"], "-", color=dark,
                lw=_w(2.5), label=f"{name}  $R^2$={fit['r2']:.4f}")
    ax.set_xlabel("Samples" if bottom else "")
    style_axis(ax, bottom=bottom)
    if True:
        ax.legend(loc="best", frameon=True, framealpha=0.92, borderpad=0.35, handlelength=1.6)


def add_row_profile(ax, row, bottom=True, show_legend=False):
    for name, dark in (("Industrial", IND_DARK), ("Non-Industrial", NON_DARK)):
        s_grid, q025, q50, q975 = _landscape_for(row[name])
        vals = _boot_vals_for(row[name])
        if vals.size:
            lo_ci, med_ci, hi_ci = np.nanpercentile(vals, [2.5, 50.0, 97.5])
            ax.axvspan(lo_ci, hi_ci, color=dark, alpha=0.08, lw=0)
            ax.axvline(med_ci, color=dark, ls="--", lw=_w(1.8), alpha=0.9)
        ax.plot(s_grid, q50, lw=_w(3.0), color=dark, label=f"{name} boot")
    m_catalog = float(next(iter(row.values()))["sd"]["n_mic"])
    ax.axvline(m_catalog, color="0.25", ls=":", lw=_w(1.5), label=r"$M$")
    ax.axhline(3.84, color="0.35", ls=":", lw=_w(1.8), label=r"$\chi^2_1$ 95% cutoff")
    ax.set_xlabel(r"$S_\infty$" if bottom else "")
    ax.set_ylim(0, 20.0)
    xs_in_view = []
    for name in ("Industrial", "Non-Industrial"):
        s_grid, _q025, _q50, _q975 = _landscape_for(row[name])
        xs_in_view.append(s_grid[np.isfinite(s_grid)])
    if xs_in_view:
        xs = np.concatenate(xs_in_view)
        lo, hi = float(xs.min()), float(xs.max())
        pad = max((hi - lo) * 0.08, 2.0)
        ax.set_xlim(lo - pad, hi + pad)
    style_axis(ax, bottom=bottom)


def bootstrap_profile_landscape(sd, fit):
    if "_bootstrap_profile_landscape" in fit:
        return fit["_bootstrap_profile_landscape"]

    boot = np.asarray(sd["boot_matrix"], float)
    n_vec = np.asarray(sd["N_vec"], float)
    n_mic = float(sd["n_mic"])
    lo = max(float(np.nanmax(boot)) if boot.size else 1.0, 1.0)
    hi = n_mic
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = max(float(np.nanmax(boot)), 1.0)
        hi = n_mic
    s_grid = np.linspace(lo, hi, 400)

    profile_mode = "bootstrap asymptote KDE likelihood"
    vals = bootstrap_asymptote_values(sd, fit)
    vals = vals[np.isfinite(vals)]

    if vals.size >= 5:
        sd_vals = float(np.nanstd(vals, ddof=1))
        bw = 1.06 * sd_vals * (vals.size ** (-1.0 / 5.0)) if sd_vals > 0 else 1.0
        bw = max(bw, (hi - lo) / 200.0, 1.0)
        z = (s_grid[:, None] - vals[None, :]) / bw
        log_density = logsumexp(-0.5 * z * z, axis=1) - np.log(vals.size * bw * np.sqrt(2.0 * np.pi))
        q50 = -2.0 * (log_density - np.nanmax(log_density))
        q50 = np.clip(q50, 0.0, 40.0)
        q025 = q975 = q50
        n_profiles = int(vals.size)
    else:
        y = np.asarray(sd["y_center"], float)
        sig = sigma_from_band(sd)
        q50 = profile_delta(y, sig, fit["shape"], s_grid)
        q025 = q975 = q50
        n_profiles = 0
    fit["_bootstrap_profile_landscape"] = (s_grid, q025, q50, q975)
    fit["_bootstrap_profile_audit"] = {
        "profile_mode": profile_mode,
        "n_bootstrap_curves": int(boot.shape[0]) if boot.ndim == 2 else 0,
        "n_profiles_used": int(n_profiles),
        "s_grid_min": float(np.nanmin(s_grid)),
        "s_grid_max": float(np.nanmax(s_grid)),
        "used_median_fallback": bool(n_profiles == 0),
    }
    return fit["_bootstrap_profile_landscape"]


def add_row_boot(ax, row, bottom=True, show_legend=False):
    vals_by_name = {
        name: _boot_vals_for(row[name])
        for name in ("Industrial", "Non-Industrial")
    }
    all_vals = np.concatenate([v[np.isfinite(v)] for v in vals_by_name.values() if v.size])
    if all_vals.size >= 2 and np.nanmax(all_vals) > np.nanmin(all_vals):
        bins = np.linspace(float(np.nanmin(all_vals)), float(np.nanmax(all_vals)), 24)
    else:
        center = float(np.nanmedian(all_vals)) if all_vals.size else 0.0
        bins = np.linspace(center - 0.5, center + 0.5, 24)
    for name, color, dark in (("Industrial", IND, IND_DARK), ("Non-Industrial", NON, NON_DARK)):
        sd = row[name]["sd"]
        vals = vals_by_name[name]
        ax.hist(vals, bins=bins, histtype="stepfilled", color=color, alpha=0.34, edgecolor="none", label=f"{name} med={np.median(vals):.0f}")
        ax.axvline(np.median(vals), color=dark, lw=_w(2.2))
    ax.axvline(float(next(iter(row.values()))["sd"]["n_mic"]), color="0.25", ls=":", lw=_w(1.8), label="M")
    ax.set_xlabel(r"Bootstrap $S_\infty$" if bottom else "")
    style_axis(ax, bottom=bottom)


def bootstrap_asymptote_values(sd, fit):
    """Use every bootstrap replicate curve.

    For k=1/Richards, match the original Figure 3 logic and refit Richards to
    each replicate curve. For beta-mixture k>1, the current fitted mixture shape
    is held fixed and S_inf is refit to each replicate because the mixture
    parameters come from catalogue-level/profile fits rather than curve-only
    refits.
    """
    if "_bootstrap_asymptotes" in fit:
        return fit["_bootstrap_asymptotes"]
    vals = [float(item["S"]) for item in bootstrap_replicate_refits(sd, fit) if np.isfinite(item.get("S", np.nan))]
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    fit["_bootstrap_asymptotes"] = vals
    return vals


def rows_from_tables(tables):
    """Rebuild the (label, row) list this figure draws from its tables.

    Each row carries the sampled curve and its fit, plus the profile landscape
    and the bootstrap replicates, so no panel needs refitting.
    """
    curves = tables["accumulation_curves"]
    metrics = tables["metrics"]
    support = tables.get("profile_support")
    reps = tables.get("bootstrap_replicates")

    rows = []
    for label in dict.fromkeys(curves["row_label"]):
        row = {}
        for name in ("Industrial", "Non-Industrial"):
            c = curves[(curves["row_label"] == label)
                       & (curves["series"] == name)]
            if c.empty:
                continue
            met = metrics[(metrics["row"] == label)
                          & (metrics["series"] == name)]
            rep = (reps[(reps["row_label"] == label)
                        & (reps["series"] == name)]
                   if reps is not None else None)
            sd = {
                "N_vec": c["n_samples"].to_numpy(dtype=float),
                "y_center": c["sgbs"].to_numpy(dtype=float),
                "q16_vec": c["sgbs_q16"].to_numpy(dtype=float),
                "q84_vec": c["sgbs_q84"].to_numpy(dtype=float),
                "n_mic": (float(rep["n_mic"].iloc[0])
                          if rep is not None and not rep.empty else np.nan),
            }
            fit = {
                "yhat": c["sgbs_fitted"].to_numpy(dtype=float),
                "r2": float(met["R2"].iloc[0]) if not met.empty else np.nan,
                "S": float(met["S_inf"].iloc[0]) if not met.empty else np.nan,
                "model": (str(met["model"].iloc[0]) if not met.empty else ""),
            }
            entry = {"sd": sd, "fit": fit}
            if support is not None:
                sup = support[(support["row_label"] == label)
                              & (support["series"] == name)]
                if not sup.empty:
                    entry["landscape"] = (
                        sup["S_inf"].to_numpy(dtype=float),
                        sup["support_q025"].to_numpy(dtype=float),
                        sup["support_q50"].to_numpy(dtype=float),
                        sup["support_q975"].to_numpy(dtype=float),
                    )
            if rep is not None and not rep.empty:
                entry["boot_vals"] = rep["S_inf_replicate"].to_numpy(dtype=float)
            row[name] = entry
        if row:
            rows.append((label, row))
    return rows


_KEY_KW = dict(frameon=True, framealpha=0.92, borderpad=0.35, handlelength=1.6)


def _key_prop(size):
    from persistence.plot_utils import FONT_STACK
    return {"family": FONT_STACK, "size": size}


def _key_size(axes):
    """The size the type pass settled the accumulation keys on."""
    for ax in axes[:, 0]:
        leg = ax.get_legend()
        if leg is not None and leg.get_texts():
            return float(leg.get_texts()[0].get_fontsize())
    return float(plt.rcParams["legend.fontsize"])


def _place_profile_key(col_axes, fontsize):
    """The support column's key, under the column, clear of every panel.

    No support panel has a corner its curves leave free in every figure, and
    a key that wandered from row to row between S16 and S17 would read as
    inconsistent; it describes the whole column, so it sits beneath it.
    """
    ax = col_axes[-1]
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              prop=_key_prop(fontsize), frameon=False, handlelength=1.6,
              columnspacing=1.2)


def _place_column_labels(fig, axes, labels, pad_in=0.04):
    """Set each column's y label just left of that column's tick labels.

    Runs after the type pass, since that resizes the tick labels it clears.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w_px = fig.get_size_inches()[0] * fig.dpi
    top = axes[0, 0].get_position().y1
    bottom = axes[-1, 0].get_position().y0
    for j, lab in enumerate(labels):
        x0 = min(ax.yaxis.get_tightbbox(r).x0 for ax in axes[:, j])
        lab.set_x(x0 / w_px - pad_in / fig.get_size_inches()[0])
        lab.set_y((top + bottom) / 2)


def _add_panel_letters(fig, axes):
    """A-L, upper left inside each panel, at the printed part-label size.

    Added after the type pass: that pass caps every string at the body-text
    ceiling, and part labels are meant to stand above it.
    """
    from persistence.plot_utils import PANEL_LETTER_PT, PAGE_WIDTH_IN, FONT_STACK
    fig.canvas.draw()
    w_in = float(fig.get_tightbbox(fig.canvas.get_renderer()).width)
    pt = PANEL_LETTER_PT / (PAGE_WIDTH_IN / max(w_in, 1e-6))
    for k, ax in enumerate(axes.ravel()):
        t = ax.text(0.018, 0.965, chr(ord("A") + k), transform=ax.transAxes,
                    va="top", ha="left", fontsize=pt, fontweight="bold",
                    fontfamily=FONT_STACK,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.78, "pad": 1.0})
        t._fixed_pt = True


def plot_sinf_figure(tables, title, path, metrics_path=None, boot_path=None):
    """Draw an S_inf accumulation figure from its table, and nothing else."""
    path = Path(path)
    metrics_path = Path(metrics_path or path.with_name(path.stem + "_metrics.csv"))
    boot_path = Path(boot_path or path.with_name(path.stem + "_bootstrap.csv"))
    make_figure(rows_from_tables(tables), title, path, metrics_path, boot_path)
    return str(path)


def make_figure(rows, title, path, metrics_path, boot_path):
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        # Printed width; the previous 17.5 in canvas was reduced to 0.42
        # and took the line widths under the 0.5 pt floor with it.
        figsize=(7.24, 7.24 * 3.55 * n_rows / 17.5),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.25, 1.05, 1.05]},
    )
    if len(rows) == 1:
        axes = np.asarray([axes])
    col_titles = ["Accumulation curve", r"Support for $S_\infty$", r"Bootstrap $S_\infty$"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, pad=6, fontweight="bold", fontsize=18)
    metrics = []
    boot_rows = []
    profile_rows = []
    for i, (label, row) in enumerate(rows):
        bottom = i == n_rows - 1
        # The threshold applies to the whole row, so it labels the row from
        # the right margin, reading downwards, rather than sitting inside the
        # accumulation panel where the Non-Industrial curve runs.
        axes[i, 2].text(
            1.035, 0.5, label,
            transform=axes[i, 2].transAxes,
            rotation=270, va="center", ha="left",
            fontsize=13, fontweight="bold", clip_on=False,
        )
        add_row_accum(axes[i, 0], row, bottom=bottom, show_legend=(i == 0))
        add_row_profile(axes[i, 1], row, bottom=bottom, show_legend=(i == 0))
        add_row_boot(axes[i, 2], row, bottom=bottom, show_legend=True)
        for name in ("Industrial", "Non-Industrial"):
            sd = row[name]["sd"]
            fit = row[name]["fit"]
            metrics.append(
                {
                    "row": label,
                    "series": name,
                    "model": fit["model"],
                    "R2": fit["r2"],
                    "RMSE": r2_rmse(sd["y_center"], fit["yhat"])[1],
                    "S_inf": fit["S"],
                }
            )
            # Through the helpers, so a redraw uses the published replicates
            # rather than refitting from a bootstrap matrix the table does
            # not carry.
            vals = _boot_vals_for(row[name])
            _s_grid, _q025, _q50, _q975 = _landscape_for(row[name])
            profile_audit = fit.get("_bootstrap_profile_audit", {})
            profile_rows.append(
                {
                    "row": label,
                    "series": name,
                    "model": fit["model"],
                    **profile_audit,
                }
            )
            boot_rows.append(
                {
                    "row": label,
                    "series": name,
                    "model": fit["model"],
                    "median": float(np.median(vals)),
                    "q025": float(np.quantile(vals, 0.025)),
                    "q975": float(np.quantile(vals, 0.975)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    for ax in axes.ravel():
        ax.set_ylabel("")
    axes[-1, 0].set_xlabel("Number of samples")
    axes[-1, 1].set_xlabel(r"$S_\infty$ (SGBs)")
    axes[-1, 2].set_xlabel(r"Bootstrap estimate of $S_\infty$ (SGBs)")

    fig.subplots_adjust(
        left=0.07,
        right=0.955,
        bottom=0.06,
        top=0.955,
        hspace=0.14,
        wspace=0.20,
    )
    # One label per column, centred on the column; x is set once the tick
    # labels have their final size (_place_column_labels).
    col_labels = [
        fig.text(0.0, 0.5, text, rotation=90, va="center", ha="right", fontsize=18)
        for text in ("Accumulated SGBs", r"Relative support, $\Delta$",
                     "Bootstrap replicates")
    ]
    _normalise_labels(fig)


    # The figure's data object. Three panels per row -- the accumulation curve,
    # the profile-likelihood support and the bootstrap asymptotes -- and all
    # three are views of the same two frames this function already writes as
    # CSV, so the object carries those rather than a third copy.
    try:
        sys.path.insert(0, str(ROOT))
        from persistence import figure_source
        # rows is a list of (label, row) pairs; each row is keyed by series
        # name and carries the sampled curve and its fit. Column 1 plots the
        # curve and the fitted line, column 2 the profile-likelihood landscape
        # and column 3 the bootstrap asymptotes, so all three are published.
        _curves, _landscape, _reps = [], [], []
        for _label, _row in rows:
            for _name in ("Industrial", "Non-Industrial"):
                _entry = _row.get(_name) if hasattr(_row, "get") else None
                if not _entry:
                    continue
                _sd, _fit = _entry["sd"], _entry["fit"]
                _n = np.asarray(_sd["N_vec"], dtype=float)
                _cur = {
                    "row_label": _label,
                    "series": _name,
                    "n_samples": _n,
                    "sgbs": np.asarray(_sd["y_center"], dtype=float),
                }
                _yhat = _fit.get("yhat")
                if _yhat is not None and len(_yhat) == len(_n):
                    _cur["sgbs_fitted"] = np.asarray(_yhat, dtype=float)
                for _q, _key in (("sgbs_q16", "q16_vec"), ("sgbs_q84", "q84_vec")):
                    _v = _sd.get(_key)
                    if _v is not None and len(_v) == len(_n):
                        _cur[_q] = np.asarray(_v, dtype=float)
                _curves.append(pd.DataFrame(_cur))

                # Column 3 is a histogram of the individual bootstrap
                # asymptotes, so the replicates themselves have to travel,
                # not just their quantiles; n_mic is the catalogue line
                # drawn across it.
                _vals = np.asarray(bootstrap_asymptote_values(_sd, _fit),
                                   dtype=float)
                _reps.append(pd.DataFrame({
                    "row_label": _label,
                    "series": _name,
                    "n_mic": float(_sd["n_mic"]),
                    "S_inf_replicate": _vals,
                }))

                _sg, _q025, _q50, _q975 = bootstrap_profile_landscape(_sd, _fit)
                _landscape.append(pd.DataFrame({
                    "row_label": _label,
                    "series": _name,
                    "S_inf": np.asarray(_sg, dtype=float),
                    "support_q025": np.asarray(_q025, dtype=float),
                    "support_q50": np.asarray(_q50, dtype=float),
                    "support_q975": np.asarray(_q975, dtype=float),
                }))

        _t = {"metrics": pd.DataFrame(metrics),
              "bootstrap_asymptotes": pd.DataFrame(boot_rows)}
        if _curves:
            _t["accumulation_curves"] = pd.concat(_curves, ignore_index=True)
        if _landscape:
            _t["profile_support"] = pd.concat(_landscape, ignore_index=True)
        if _reps:
            _t["bootstrap_replicates"] = pd.concat(_reps, ignore_index=True)
        figure_source.emit_for_path(str(path), _t)
    except Exception as _exc:
        print(f"  [data] S_inf figure table unavailable: {_exc}")

    # This script manages its own layout and so does not go through
    # savefig_dual; it still has to meet the same Science format.
    try:
        from persistence.plot_utils import prepare_for_publication
        prepare_for_publication(fig, Path(path).stem)
    except Exception as _exc:
        print(f'  [format] {_exc}')
    # Keys built at their final size: one made before the type pass keeps the
    # spacing of the size it was made at, and the pass only rebuilds keys it
    # places itself.
    key_pt = _key_size(axes)
    for ax in axes[:, 2]:
        # The replicates pile up against M at the right, and the left of the
        # panel holds only a thin tail, so the key goes there, mid-height.
        ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.5),
                  prop=_key_prop(key_pt), **_KEY_KW)
    _place_profile_key(axes[:, 1], key_pt)
    _place_column_labels(fig, axes, col_labels)
    _add_panel_letters(fig, axes)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    profile_path = metrics_path.with_name(metrics_path.stem.replace("_metrics", "_profile_audit") + ".csv")
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # Additional Figure A16 — stringent per-sample breadth thresholds (>=10/30/50%)
    fig_a16_rows = prepare_threshold_results()
    make_figure(
        fig_a16_rows,
        "Figure A16: Combined catalogue accumulation across detection thresholds",
        OUT / "figure_a16_accumulation_sinf_thresholds_medium.png",
        OUT / "figure_a16_threshold_metrics.csv",
        OUT / "figure_a16_threshold_bootstrap_asymptotes.csv",
    )
    # Additional Figure A17 — detection required in at least k samples (k=1,2,3,5)
    fig_a17_rows = prepare_k_results()
    make_figure(
        fig_a17_rows,
        "Figure A17: Combined catalogue accumulation across repeated-detection thresholds",
        OUT / "figure_a17_accumulation_sinf_mink_medium.png",
        OUT / "figure_a17_k_metrics.csv",
        OUT / "figure_a17_k_bootstrap_asymptotes.csv",
    )
    print(OUT)


if __name__ == "__main__":
    main()
