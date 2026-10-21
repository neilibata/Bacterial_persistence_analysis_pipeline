#!/usr/bin/env python3
"""
Occupancy / detection accumulation simulation for breadth-coverage matrices.

This script turns each breadth coverage matrix into a binary detection matrix
using a breadth threshold, then estimates how many reference taxa are detected
at least once as samples accumulate in random order.

The exact curve used here is the finite-population version of:

    E[C_m] = sum_k [1 - Pr(taxon k is missed in all m selected samples)]

For binary thresholded detections, if taxon k is detected in r_k of S samples,
then under sampling without replacement:

    Pr(miss k in m samples) = C(S - r_k, m) / C(S, m)

so:

    E[C_m] = sum_k [1 - C(S - r_k, m) / C(S, m)]

Permutation runs are also included to show random-order variability.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    from . import config
except ImportError:  # running outside the package
    try:
        import persistence.config as config
    except ImportError:
        import config

try:
    from .plot_utils import apply_display_text
except ImportError:
    from matplotlib.text import Text

    def apply_display_text(fig):
        for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)


# Read from config.REFERENCES rather than hard-coding: these names were fixed
# here, so repointing a reference (the 2026-08-20 rebuild) never reached this
# analysis and A17 silently kept the pre-rebuild matrices.
DEFAULT_REFS = {
    name: {
        "breadth": spec["breadth_file"],
        "depth": spec["depth_file"],
        "lengths": spec["lengths_file"],
    }
    for name, spec in config.REFERENCES.items()
}

EXCLUDE_STUDIES = frozenset({
    "Bengtsson-PalmeJ",
    "DhakanDB",
    "FerrettiP",
    "Heintz-BuschartA",
    "KaurK",
    "UKN",
    "eKmeta",
})

STUDY_EXCLUSION_COLS = [
    "StudyFolder",
    "study_name [MBIOTEMP_study_name]",
    "Author",
]

GROUP_COLORS = {
    "All": "#000000",
    "MacroGroup:Industrial": "#56B4E9",
    "MacroGroup:Non-Industrial": "#E69F00",
    "Group_fine:Industrial": "#009E73",
    "Group_fine:Industrial Unhealthy": "#0072B2",
    "Group_fine:Non-Industrial": "#E69F00",
    "Group_fine:Non-Industrial Unhealthy": "#D55E00",
    "Lifestyle:H-G": "#CC79A7",
    "Lifestyle:Industrial": "#56B4E9",
    "Lifestyle:Transitional": "#F0E442",
}


def clean_group_label(group: str) -> str:
    return group.split(":", 1)[1] if ":" in group else group


def safe_name(name: str) -> str:
    return (
        name.replace(":", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_mean_matrix(path: Path, tag: str) -> pd.DataFrame:
    print(f"Loading {tag}: {path}")
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str).str.strip()

    mean_cols = [c for c in df.columns if str(c).endswith("_mean")]
    if mean_cols:
        df = df[mean_cols].copy()
        df.columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]

    df = df.fillna(0.0)
    print(f"  {df.shape[0]:,} samples x {df.shape[1]:,} taxa")
    return df


def classify_macro_group(row: pd.Series) -> str | None:
    life = row.get("Lifestyle")
    if life == "Industrial":
        return "Industrial"
    if life in ("H-G", "Transitional"):
        return "Non-Industrial"
    return None


def classify_group(row: pd.Series) -> str | None:
    unhealthy = row.get("Unhealthy")
    life = row.get("Lifestyle")
    if unhealthy == "Unhealthy":
        if life in ("H-G", "Transitional"):
            return "Non-Industrial Unhealthy"
        if life == "Industrial":
            return "Industrial Unhealthy"
    if unhealthy == "Healthy":
        if life in ("H-G", "Transitional"):
            return "Non-Industrial"
        if life == "Industrial":
            return "Industrial"
    return None


def apply_study_exclusion(meta: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=meta.index)
    for col in STUDY_EXCLUSION_COLS:
        if col not in meta.columns:
            continue
        values = meta[col].astype(str).str.strip()
        if col == "Author":
            mask |= values.isin(EXCLUDE_STUDIES)
        else:
            for prefix in EXCLUDE_STUDIES:
                mask |= values.str.startswith(prefix)

    if mask.any():
        print(f"  Study exclusion removed {int(mask.sum()):,} samples")
    return meta.loc[~mask].copy()


def load_metadata(path: Path, sample_index: pd.Index) -> pd.DataFrame:
    print(f"Loading metadata: {path}")
    meta = pd.read_csv(path, low_memory=False)
    # M1 carries the manuscript's display labels; analysis compares keys.
    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = config.canonicalize_lifestyle_series(meta["Lifestyle"])
    if "Chosen_match_ID" not in meta.columns:
        raise ValueError(f"'Chosen_match_ID' was not found in {path}")

    meta["Chosen_match_ID"] = meta["Chosen_match_ID"].astype(str).str.strip()
    meta = meta[meta["Chosen_match_ID"].isin(sample_index)].copy()
    meta = meta.drop_duplicates(subset="Chosen_match_ID", keep="first")
    meta = meta.set_index("Chosen_match_ID")

    if "Lifestyle" in meta.columns:
        meta = meta[meta["Lifestyle"].notna()].copy()

    if {"Lifestyle", "Unhealthy"}.issubset(meta.columns):
        nonind_na = meta["Lifestyle"].isin(["H-G", "Transitional"]) & meta["Unhealthy"].isna()
        if nonind_na.any():
            print(f"  Re-coded {int(nonind_na.sum()):,} Non-Industrial NA health labels to Healthy")
            meta.loc[nonind_na, "Unhealthy"] = "Healthy"
        meta["MacroGroup"] = meta.apply(classify_macro_group, axis=1)
        meta["Group_fine"] = meta.apply(classify_group, axis=1)
    else:
        meta["MacroGroup"] = "All"
        meta["Group_fine"] = "All"

    meta = apply_study_exclusion(meta)
    print(f"  Linked/preprocessed samples: {len(meta):,}")
    return meta


def load_total_bp(depth_path: Path, lengths_path: Path, sample_index: pd.Index) -> pd.Series | None:
    if not depth_path.exists() or not lengths_path.exists():
        return None

    depth = load_mean_matrix(depth_path, "depth")
    length_df = pd.read_csv(lengths_path, sep=r"\s+", header=None, names=["taxon", "length"])
    length_df["taxon"] = length_df["taxon"].astype(str).str.strip()
    lengths = pd.Series(length_df["length"].values, index=length_df["taxon"])

    common = [c for c in depth.columns if c in lengths.index]
    if not common:
        return None

    total_bp = depth[common].mul(lengths.loc[common], axis=1).sum(axis=1)
    return total_bp.reindex(sample_index)


def make_n_grid(n_samples: int, n_points: int, n_min: int | None = None) -> np.ndarray:
    if n_samples <= 0:
        return np.array([], dtype=int)
    if n_min is None:
        n_min = 1
    n_min = max(1, min(int(n_min), n_samples))
    raw = np.geomspace(n_min, n_samples, num=min(n_points, n_samples))
    grid = np.unique(np.rint(raw).astype(int))
    grid = grid[(grid >= 1) & (grid <= n_samples)]
    if grid[-1] != n_samples:
        grid = np.append(grid, n_samples)
    return grid


def log_choose(n: int | np.ndarray, k: int) -> np.ndarray:
    n_arr = np.asarray(n, dtype=float)
    out = np.full(n_arr.shape, -np.inf, dtype=float)
    valid = (k >= 0) & (n_arr >= k)
    if np.any(valid):
        lgamma_vec = np.vectorize(math.lgamma)
        out[valid] = (
            lgamma_vec(n_arr[valid] + 1.0)
            - math.lgamma(k + 1.0)
            - lgamma_vec(n_arr[valid] - k + 1.0)
        )
    return out


def exact_without_replacement_curve(detected: np.ndarray, n_grid: Iterable[int]) -> pd.DataFrame:
    n_samples = detected.shape[0]
    richness = detected.sum(axis=0).astype(int)
    observed_taxa = int((richness > 0).sum())
    rows = []

    for m in n_grid:
        log_denom = float(log_choose(np.array([n_samples]), int(m))[0])
        eligible = richness <= n_samples - m
        miss = np.zeros_like(richness, dtype=float)
        if eligible.any():
            miss[eligible] = np.exp(log_choose(n_samples - richness[eligible], int(m)) - log_denom)
        expected = np.sum(1.0 - miss)
        rows.append({
            "n_samples": int(m),
            "expected_taxa": float(expected),
            "observed_detectable_taxa": observed_taxa,
            "n_reference_taxa": int(detected.shape[1]),
        })

    return pd.DataFrame(rows)


def permutation_curve(
    detected: np.ndarray,
    n_grid: Iterable[int],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_samples = detected.shape[0]
    n_grid = np.asarray(list(n_grid), dtype=int)
    curves = np.empty((n_perm, len(n_grid)), dtype=float)

    for b in range(n_perm):
        order = rng.permutation(n_samples)
        present_any = np.zeros(detected.shape[1], dtype=bool)
        next_pos = 0
        for j, n in enumerate(n_grid):
            if n > next_pos:
                present_any |= detected[order[next_pos:n], :].any(axis=0)
                next_pos = n
            curves[b, j] = present_any.sum()

    return pd.DataFrame({
        "n_samples": n_grid,
        "perm_mean": curves.mean(axis=0),
        "perm_sd": curves.std(axis=0, ddof=1) if n_perm > 1 else 0.0,
        "perm_p025": np.quantile(curves, 0.025, axis=0),
        "perm_p500": np.quantile(curves, 0.500, axis=0),
        "perm_p975": np.quantile(curves, 0.975, axis=0),
    })


def prevalence_laplace_curve(detected: np.ndarray, n_grid: Iterable[int]) -> pd.DataFrame:
    prevalence = detected.mean(axis=0)
    rows = []
    for m in n_grid:
        expected = np.sum(1.0 - np.power(1.0 - prevalence, int(m)))
        rows.append({"n_samples": int(m), "prevalence_laplace_taxa": float(expected)})
    return pd.DataFrame(rows)


def hill3(x: np.ndarray, n_inf: float, x50: float, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xb = np.power(np.maximum(x, 1e-12), beta)
    x50b = np.power(max(x50, 1e-12), beta)
    return n_inf * xb / (xb + x50b)


def hill4_asym(
    x: np.ndarray,
    n_inf: float,
    x50: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ratio = np.power(max(x50, 1e-12) / np.maximum(x, 1e-12), beta)
    asym = np.power(2.0, 1.0 / max(gamma, 1e-12)) - 1.0
    return n_inf * np.power(1.0 + asym * ratio, -gamma)


def _fit_init(x: np.ndarray, y: np.ndarray, n_reference_taxa: int) -> tuple[float, float, float]:
    y_max = float(np.nanmax(y))
    n_inf0 = min(max(y_max * 1.02, y_max + 1.0), float(n_reference_taxa))
    target = 0.5 * n_inf0
    x50 = float(x[np.argmin(np.abs(y - target))])
    return n_inf0, max(x50, 1.0), 0.8


def fit_accumulation_models(curve: pd.DataFrame) -> list[dict[str, float | str]]:
    x = curve["n_samples"].to_numpy(dtype=float)
    y = curve["expected_taxa"].to_numpy(dtype=float)
    n_reference_taxa = int(curve["n_reference_taxa"].iloc[0])
    y_max = float(np.nanmax(y))
    n_inf0, x500, beta0 = _fit_init(x, y, n_reference_taxa)
    upper_n = float(n_reference_taxa)
    rows: list[dict[str, float | str]] = []

    specs = [
        (
            "Hill3",
            hill3,
            [n_inf0, x500, beta0],
            ([y_max, 1.0, 0.05], [upper_n, max(float(x.max()) * 100.0, 10.0), 5.0]),
        ),
        (
            "Hill4_asym",
            hill4_asym,
            [n_inf0, x500, beta0, 1.0],
            ([y_max, 1.0, 0.05, 0.05], [upper_n, max(float(x.max()) * 100.0, 10.0), 5.0, 20.0]),
        ),
    ]

    for model, func, p0, bounds in specs:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=50000)
            yhat = func(x, *popt)
            ss_res = float(np.sum(np.square(y - yhat)))
            ss_tot = float(np.sum(np.square(y - y.mean())))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            row = {
                "model": model,
                "fit_ok": 1,
                "n_inf": float(popt[0]),
                "x50": float(popt[1]),
                "beta": float(popt[2]),
                "gamma": float(popt[3]) if model == "Hill4_asym" else np.nan,
                "r2": r2,
                "rss": ss_res,
            }
        except Exception:
            row = {
                "model": model,
                "fit_ok": 0,
                "n_inf": np.nan,
                "x50": np.nan,
                "beta": np.nan,
                "gamma": np.nan,
                "r2": np.nan,
                "rss": np.nan,
            }
        rows.append(row)
    return rows


def bootstrap_fit_asymptotes(
    detected: np.ndarray,
    n_grid: Iterable[int],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if n_boot <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n_samples = detected.shape[0]
    n_grid = np.asarray(list(n_grid), dtype=int)
    rows = []

    for b in range(n_boot):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        boot_curve = exact_without_replacement_curve(detected[sample_idx, :], n_grid)
        for fit in fit_accumulation_models(boot_curve):
            rows.append({
                "boot": b,
                "model": fit["model"],
                "fit_ok": fit["fit_ok"],
                "n_inf": fit["n_inf"],
                "x50": fit["x50"],
                "beta": fit["beta"],
                "gamma": fit["gamma"],
                "r2": fit["r2"],
            })

    return pd.DataFrame(rows)


def summarize_bootstrap_fits(boot: pd.DataFrame) -> pd.DataFrame:
    if boot.empty:
        return pd.DataFrame()

    rows = []
    for model, block in boot.groupby("model"):
        ok = block[block["fit_ok"] == 1].copy()
        row = {
            "model": model,
            "boot_n": int(len(block)),
            "boot_fit_ok": int(len(ok)),
        }
        for param in ["n_inf", "x50", "beta", "gamma", "r2"]:
            vals = ok[param].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{param}_boot_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{param}_boot_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{param}_boot_p025"] = float(vals.quantile(0.025)) if len(vals) else np.nan
            row[f"{param}_boot_p500"] = float(vals.quantile(0.500)) if len(vals) else np.nan
            row[f"{param}_boot_p975"] = float(vals.quantile(0.975)) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_single_group(
    curve: pd.DataFrame,
    fit_rows: list[dict[str, float | str]],
    path_base: Path,
    title: str,
) -> None:
    x = curve["n_samples"].to_numpy(dtype=float)
    y = curve["expected_taxa"].to_numpy(dtype=float)
    color = GROUP_COLORS.get(str(curve["group"].iloc[0]), "#000000")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.fill_between(
        x,
        curve["perm_p025"].to_numpy(dtype=float),
        curve["perm_p975"].to_numpy(dtype=float),
        color=color,
        alpha=0.16,
        linewidth=0,
        label="Random-order 95% interval",
    )
    ax.plot(x, y, color=color, linewidth=2.2, label="Exact expected curve")
    ax.scatter(x, y, color=color, s=16, alpha=0.8)

    xfine = np.geomspace(max(1.0, float(x.min())), float(x.max()), 250)
    for fit in fit_rows:
        if int(fit["fit_ok"]) != 1:
            continue
        if fit["model"] == "Hill3":
            yfit = hill3(xfine, float(fit["n_inf"]), float(fit["x50"]), float(fit["beta"]))
            ax.plot(xfine, yfit, color="#333333", linewidth=1.8, linestyle="-", label="Hill 3-param")
        elif fit["model"] == "Hill4_asym":
            yfit = hill4_asym(
                xfine,
                float(fit["n_inf"]),
                float(fit["x50"]),
                float(fit["beta"]),
                float(fit["gamma"]),
            )
            ax.plot(xfine, yfit, color="#333333", linewidth=1.8, linestyle="--", label="Asym Hill 4-param")

    ax.set_xscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Detected reference taxa")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    apply_display_text(fig)
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_comparison(
    curves: pd.DataFrame,
    fit_table: pd.DataFrame,
    groups: list[str],
    path_base: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for group in groups:
        block = curves[curves["group"] == group]
        if block.empty:
            continue
        color = GROUP_COLORS.get(group, "#000000")
        x = block["n_samples"].to_numpy(dtype=float)
        y = block["expected_taxa"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            block["perm_p025"].to_numpy(dtype=float),
            block["perm_p975"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(x, y, color=color, linewidth=2.0, label=clean_group_label(group))

        fit = fit_table[(fit_table["group"] == group) & (fit_table["model"] == "Hill4_asym")]
        if len(fit) and int(fit["fit_ok"].iloc[0]) == 1:
            xfine = np.geomspace(max(1.0, float(x.min())), float(x.max()), 250)
            ax.plot(
                xfine,
                hill4_asym(
                    xfine,
                    float(fit["n_inf"].iloc[0]),
                    float(fit["x50"].iloc[0]),
                    float(fit["beta"].iloc[0]),
                    float(fit["gamma"].iloc[0]),
                ),
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.75,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Detected reference taxa")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    apply_display_text(fig)
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def summarize_taxa(detected: np.ndarray, taxa: pd.Index) -> pd.DataFrame:
    n_samples = detected.shape[0]
    n_detected = detected.sum(axis=0).astype(int)
    prevalence = n_detected / float(n_samples)
    lambda_hat = -np.log(np.clip(1.0 - prevalence, 1e-15, 1.0))
    out = pd.DataFrame({
        "taxon": taxa.astype(str),
        "n_detected_samples": n_detected,
        "prevalence_at_threshold": prevalence,
        "lambda_hat_per_sample": lambda_hat,
    })
    return out.sort_values(["n_detected_samples", "taxon"], ascending=[False, True])


def group_masks(meta: pd.DataFrame) -> dict[str, pd.Index]:
    groups: dict[str, pd.Index] = {"All": meta.index}

    for col in ["MacroGroup", "Group_fine", "Lifestyle"]:
        if col not in meta.columns:
            continue
        for value in sorted(v for v in meta[col].dropna().unique() if str(v) != "None"):
            idx = meta.index[meta[col] == value]
            if len(idx):
                groups[f"{col}:{value}"] = idx

    return groups


def analyze_reference(
    ref_name: str,
    breadth_path: Path,
    metadata_path: Path,
    output_dir: Path,
    threshold: float,
    n_perm: int,
    n_points: int,
    seed: int,
    min_samples: int,
    min_total_bp: float | None,
    depth_path: Path | None,
    lengths_path: Path | None,
    n_fit_boot: int,
) -> None:
    breadth = load_mean_matrix(breadth_path, "breadth")
    meta = load_metadata(metadata_path, breadth.index)

    if min_total_bp is not None and depth_path is not None and lengths_path is not None:
        total_bp = load_total_bp(depth_path, lengths_path, breadth.index)
        if total_bp is not None:
            meta["Total_depth_bp"] = total_bp.reindex(meta.index)
            keep = meta["Total_depth_bp"] >= min_total_bp
            n_drop = int((~keep).sum())
            if n_drop:
                print(f"  Dropped {n_drop:,} samples below {min_total_bp / 1e6:.0f} Mb")
            meta = meta.loc[keep].copy()

    breadth = breadth.loc[meta.index]
    detected_df = breadth >= threshold

    ref_out = output_dir / ref_name
    ref_out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    taxon_rows = []
    fit_rows = []
    boot_rows = []
    for group, idx in group_masks(meta).items():
        if len(idx) < min_samples:
            continue

        group_detected = detected_df.loc[idx].to_numpy(dtype=bool, copy=True)
        n_grid = make_n_grid(group_detected.shape[0], n_points, n_min=min_samples)
        exact = exact_without_replacement_curve(group_detected, n_grid)
        perm = permutation_curve(group_detected, n_grid, n_perm, seed)
        laplace = prevalence_laplace_curve(group_detected, n_grid)

        curve = exact.merge(perm, on="n_samples").merge(laplace, on="n_samples")
        curve.insert(0, "group", group)
        curve.insert(0, "reference", ref_name)
        all_rows.append(curve)

        fits = fit_accumulation_models(curve)
        boot = bootstrap_fit_asymptotes(
            group_detected,
            n_grid,
            n_fit_boot,
            seed + 10_000 + len(fit_rows),
        )
        boot_summary = summarize_bootstrap_fits(boot)
        for fit in fits:
            fit_out = dict(fit)
            fit_out.update({
                "reference": ref_name,
                "group": group,
                "n_samples_total": int(group_detected.shape[0]),
                "observed_detectable_taxa": int((group_detected.sum(axis=0) > 0).sum()),
                "n_reference_taxa": int(group_detected.shape[1]),
            })
            if not boot_summary.empty:
                match = boot_summary[boot_summary["model"] == fit["model"]]
                if len(match):
                    fit_out.update(match.iloc[0].to_dict())
            fit_rows.append(fit_out)

        if not boot.empty:
            boot.insert(0, "group", group)
            boot.insert(0, "reference", ref_name)
            boot_rows.append(boot)

        taxa = summarize_taxa(group_detected, detected_df.columns)
        taxa.insert(0, "group", group)
        taxa.insert(0, "reference", ref_name)
        taxon_rows.append(taxa)

        plot_single_group(
            curve,
            fits,
            ref_out / f"{ref_name}_{safe_name(group)}_accumulation_fit",
            f"{ref_name}: {clean_group_label(group)}",
        )

        print(
            f"  {ref_name} {group}: {len(idx):,} samples, "
            f"{int((group_detected.sum(axis=0) > 0).sum()):,} taxa detected at >= {threshold:g}%"
        )

    curves = pd.concat(all_rows, ignore_index=True)
    taxa = pd.concat(taxon_rows, ignore_index=True)
    fit_table = pd.DataFrame(fit_rows)

    curves_path = ref_out / f"{ref_name}_occupancy_accumulation_curves.csv"
    taxa_path = ref_out / f"{ref_name}_taxon_detection_rates.csv"
    fits_path = ref_out / f"{ref_name}_accumulation_model_fits.csv"
    curves.to_csv(curves_path, index=False)
    taxa.to_csv(taxa_path, index=False)
    fit_table.to_csv(fits_path, index=False)

    if boot_rows:
        boot_path = ref_out / f"{ref_name}_accumulation_model_fit_bootstrap.csv"
        pd.concat(boot_rows, ignore_index=True).to_csv(boot_path, index=False)
        print(f"Saved {boot_path}")

    plot_comparison(
        curves,
        fit_table,
        ["MacroGroup:Industrial", "MacroGroup:Non-Industrial"],
        ref_out / f"{ref_name}_macrogroup_accumulation_fit",
        f"{ref_name}: Macro-group accumulation",
    )
    plot_comparison(
        curves,
        fit_table,
        [
            "Group_fine:Industrial",
            "Group_fine:Industrial Unhealthy",
            "Group_fine:Non-Industrial",
            "Group_fine:Non-Industrial Unhealthy",
        ],
        ref_out / f"{ref_name}_health_lifestyle_accumulation_fit",
        f"{ref_name}: Lifestyle x health accumulation",
    )

    print(f"Saved {curves_path}")
    print(f"Saved {taxa_path}")
    print(f"Saved {fits_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="metadata_Mar_16_2026.csv")
    parser.add_argument("--output-dir", default="results/occupancy_accumulation")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--n-fit-boot", type=int, default=200)
    parser.add_argument("--n-points", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-total-bp", type=float, default=3e7)
    parser.add_argument(
        "--refs",
        nargs="+",
        default=["UHGG", "lev"],
        choices=sorted(DEFAULT_REFS),
        help="Reference databases to analyze.",
    )
    parser.add_argument(
        "--skip-depth-filter",
        action="store_true",
        help="Do not use matching depth/length files to apply the minimum mapped-bp filter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ref_name in args.refs:
        cfg = DEFAULT_REFS[ref_name]
        breadth_path = Path(cfg["breadth"])
        if not breadth_path.exists():
            raise FileNotFoundError(f"Missing breadth file for {ref_name}: {breadth_path}")

        depth_path = Path(cfg["depth"])
        lengths_path = Path(cfg["lengths"])
        if args.skip_depth_filter:
            depth_path = None
            lengths_path = None

        analyze_reference(
            ref_name=ref_name,
            breadth_path=breadth_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            threshold=args.threshold,
            n_perm=args.n_perm,
            n_points=args.n_points,
            seed=args.seed,
            min_samples=args.min_samples,
            min_total_bp=None if args.skip_depth_filter else args.min_total_bp,
            depth_path=depth_path,
            lengths_path=lengths_path,
            n_fit_boot=args.n_fit_boot,
        )


if __name__ == "__main__":
    main()
