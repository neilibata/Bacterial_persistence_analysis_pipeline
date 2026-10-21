"""
Health-condition projection onto the Non-Industrial→Industrial extinction axis.

Question
--------
Do gut microbiome compositions of patients with specific conditions (T2D,
IBD, CRC, obesity, etc.) sit along the same Non-Industrial→Industrial
continuum, and if so, where?  And do they project *past* healthy Industrial
(i.e. further toward the Industrial extreme than even healthy Industrial
controls), or only partway?

Design
------
Step 1 — Axis definition (NI→Ind).
    Compute a Jaccard PCoA on two equally-weighted groups:
      Non-Industrial (H-G + Transitional)  →  contributes 1/2 total mass
      Healthy Industrial                    →  contributes 1/2 total mass
    Each NI sample gets weight 1/(2·n_NI); each Healthy Industrial sample
    gets weight 1/(2·n_IndH).  Both groups are subsampled to MAX_AXIS_N to
    keep the distance matrix tractable.
    The axis vector v̂ = (c_IndH − c_NI) / |c_IndH − c_NI|.

Step 2 — Per-condition projection.
    For each condition in DISEASE_NCD_MAP + combined Unhealthy:
      a. Identify disease cases:  Industrial samples positive for condition
      b. Identify depth-matched controls:  Healthy Industrial, nearest-
         neighbour matched on log10(total_bp), n_control = n_disease
      c. Draw n_ref = min(n_disease, MAX_REF) NI reference samples
      d. Run a fresh Jaccard PCoA on:
           NI ref (1/3 mass)  +  controls (1/3 mass)  +  cases (1/3 mass)
         This equal-group-weight scheme prevents sample-size artefacts:
         a condition with n=30 and one with n=2000 are treated identically.
      e. Extract weighted centroids c_NI, c_ctrl, c_case in PCoA space.
      f. Report coordinate-invariant quantities:
           p_ctrl = (c_ctrl − c_NI)·v̂_local / |c_IndH − c_NI|_local
           p_case = (c_case − c_NI)·v̂_local / |c_IndH − c_NI|_local
           shift  = p_case − p_ctrl    (case relative to its own controls)
         v̂_local is defined freshly within each per-condition PCoA using
         the NI ref and control centroids, so the axis is always commensurable
         with the two reference anchors in that embedding.
      g. Bootstrap CI (stratified within each of the 3 groups).

Step 3 — Figure.
    Horizontal forest plot:
      Y axis — conditions, sorted by p_case (position along axis)
      X axis — fractional position (0 = NI, 1 = Healthy Industrial)
      Each condition: two points joined by a line —
        open circle = controls (should cluster near 1.0)
        filled circle = cases
      Dashed reference lines at 0 (NI) and 1 (Healthy Industrial).
      95 % bootstrap CI bars on the case point.
      Colour by condition category (metabolic, gut, other).

Outputs
-------
results/<ref>/section3_health/main/health_axis_projection_<spec>.csv
results/<ref>/section3_health/main/health_axis_projection_<spec>.png

Standalone
----------
    python -m persistence --ref UHGG --only s3_axis
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .health_cmh import build_group_masks

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_AXIS_N   = 1500   # samples per group for axis-definition PCoA
MAX_REF      = 500    # NI reference samples per condition PCoA
MIN_N_CASE   = 15     # minimum cases to include a condition
N_BOOT       = 400
N_PCOA_AXES  = 6

# Condition categories for colour-coding
CONDITION_CATEGORY = {
    "Unhealthy (all)":   "combined",
    "Obese":             "metabolic",
    "Overweight":        "metabolic",
    "Glucose Intolerance":"metabolic",
    "Fatty Liver":       "metabolic",
    "Metabolic Syndrome":"metabolic",
    "ASCVD":             "metabolic",
    "Hypertension":      "metabolic",
    "IBD":               "gut",
    "CRC / Adenoma":     "gut",
    "Coeliac":           "gut",
}
CATEGORY_COLORS = {
    "combined":  "#333333",
    "metabolic": "#E69F00",
    "gut":       "#0072B2",
    "other":     "#CC79A7",
}


# ── Weighted PCoA (same as axis_angle_analysis) ───────────────────────────────

def _weighted_pcoa(D_sq: np.ndarray, weights: np.ndarray,
                   n_axes: int = N_PCOA_AXES):
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
    return coords


def _jaccard_sq(pres: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J = pdist(pres, metric="jaccard")
    return squareform(np.nan_to_num(J, nan=1.0))


def _weighted_centroid(coords: np.ndarray, w: np.ndarray) -> np.ndarray:
    wn = w / max(w.sum(), 1e-300)
    return coords.T @ wn


# ── Depth-matched controls ────────────────────────────────────────────────────

def _depth_match_controls(case_ids: list, healthy_ind_ids: list,
                           log_depth: dict, rng: np.random.Generator,
                           n_control: int) -> list:
    """
    Nearest-neighbour depth matching (with replacement).
    Returns n_control Healthy Industrial IDs matched to the case depth distribution.
    """
    if not healthy_ind_ids or not case_ids:
        return []
    case_depths = np.array([log_depth.get(s, 9.0) for s in case_ids])
    ctrl_depths = np.array([log_depth.get(s, 9.0) for s in healthy_ind_ids])
    ctrl_arr    = np.array(healthy_ind_ids)

    selected = []
    # For each case sample, find the closest-depth control (with replacement)
    for cd in case_depths[:n_control]:
        idx = int(np.argmin(np.abs(ctrl_depths - cd)))
        selected.append(ctrl_arr[idx])
    return selected


# ── Per-condition PCoA and projection ────────────────────────────────────────

def _project_condition(ni_ids: list, ctrl_ids: list, case_ids: list,
                        presence_all: pd.DataFrame,
                        n_boot: int, rng: np.random.Generator
                        ) -> dict | None:
    """
    Run a fresh equal-weight Jaccard PCoA on NI ref + controls + cases.
    Returns coordinate-invariant fractional positions and bootstrap CI.
    """
    n_ni   = len(ni_ids)
    n_ctrl = len(ctrl_ids)
    n_case = len(case_ids)
    n_total = n_ni + n_ctrl + n_case
    if n_total < 9 or n_ni < 2 or n_ctrl < 2 or n_case < 2:
        return None

    all_ids = ni_ids + ctrl_ids + case_ids
    weights = (
        [1.0 / (3.0 * n_ni)]   * n_ni   +
        [1.0 / (3.0 * n_ctrl)] * n_ctrl +
        [1.0 / (3.0 * n_case)] * n_case
    )
    w_arr = np.array(weights, dtype=float)

    pres = presence_all.loc[all_ids].values.astype(float)
    D_sq = _jaccard_sq(pres)

    try:
        n_ax   = min(N_PCOA_AXES, n_total - 1)
        coords = _weighted_pcoa(D_sq, w_arr, n_axes=n_ax)
    except Exception:
        return None

    # Weighted centroids per group
    c_ni   = _weighted_centroid(coords[:n_ni],             w_arr[:n_ni])
    c_ctrl = _weighted_centroid(coords[n_ni:n_ni+n_ctrl],  w_arr[n_ni:n_ni+n_ctrl])
    c_case = _weighted_centroid(coords[n_ni+n_ctrl:],      w_arr[n_ni+n_ctrl:])

    # Local axis vector (NI → controls in this embedding)
    v     = c_ctrl - c_ni
    v_len = float(np.linalg.norm(v))
    if v_len < 1e-12:
        return None
    v_hat = v / v_len

    def _proj(c):
        return float(np.dot(c - c_ni, v_hat)) / v_len   # 0=NI, 1=ctrl

    p_ctrl = _proj(c_ctrl)   # should be ≈ 1.0 by construction
    p_case = _proj(c_case)
    shift  = p_case - p_ctrl

    # Bootstrap CI on shift and p_case
    boot_shift = np.empty(n_boot)
    boot_pcase = np.empty(n_boot)
    slices = [slice(0, n_ni), slice(n_ni, n_ni+n_ctrl),
              slice(n_ni+n_ctrl, n_total)]
    for i in range(n_boot):
        boot_cents = []
        for sl in slices:
            c_grp = coords[sl]
            n_g   = len(c_grp)
            idx   = rng.integers(0, n_g, size=n_g)
            boot_cents.append(c_grp[idx].mean(axis=0))
        bv     = boot_cents[1] - boot_cents[0]
        bv_len = float(np.linalg.norm(bv))
        if bv_len < 1e-12:
            boot_shift[i] = boot_pcase[i] = np.nan
            continue
        bv_hat = bv / bv_len
        def _bp(c):
            return float(np.dot(c - boot_cents[0], bv_hat)) / bv_len
        boot_pcase[i]  = _bp(boot_cents[2])
        boot_shift[i]  = _bp(boot_cents[2]) - _bp(boot_cents[1])

    valid = np.isfinite(boot_shift)
    if valid.sum() < 5:
        ci_lo = ci_hi = shift_ci_lo = shift_ci_hi = np.nan
    else:
        ci_lo       = float(np.percentile(boot_pcase[valid], 2.5))
        ci_hi       = float(np.percentile(boot_pcase[valid], 97.5))
        shift_ci_lo = float(np.percentile(boot_shift[valid], 2.5))
        shift_ci_hi = float(np.percentile(boot_shift[valid], 97.5))

    return dict(
        p_ctrl=p_ctrl, p_case=p_case, shift=shift,
        p_case_ci_lo=ci_lo, p_case_ci_hi=ci_hi,
        shift_ci_lo=shift_ci_lo, shift_ci_hi=shift_ci_hi,
        n_ni=n_ni, n_ctrl=n_ctrl, n_case=n_case,
    )


# ── Main per-spec function ────────────────────────────────────────────────────

def _run_one_spec(data, spec, outdir, ref, stats, rng_seed=42):
    from .detection import build_presence
    spec_tag = spec.short_label
    rng      = np.random.default_rng(rng_seed)
    meta     = data.meta

    print(f"\n  [{spec_tag}] {spec.description}")

    # ── Sample pools ─────────────────────────────────────────────────────────
    ind     = meta["Lifestyle"] == "Industrial"
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])
    healthy = meta["Unhealthy"] == "Healthy"

    all_ni_ids    = [s for s in meta.index[ni_mask]
                     if s in data.breadth.index]
    all_indh_ids  = [s for s in meta.index[ind & healthy]
                     if s in data.breadth.index]

    log_depth = {s: float(np.log10(max(data.total_bp.get(s, 1.0), 1.0)))
                 for s in meta.index}

    print(f"    NI pool: {len(all_ni_ids):,}  |  "
          f"Healthy Ind pool: {len(all_indh_ids):,}")

    if len(all_ni_ids) < 20 or len(all_indh_ids) < 20:
        print("  [health_axis] Insufficient samples — skipping")
        return pd.DataFrame()

    # ── Build presence matrix (all samples at once, subset per condition) ─────
    all_needed = list(set(all_ni_ids + all_indh_ids +
                          [s for s in meta.index[ind]
                           if s in data.breadth.index]))
    print(f"    Building presence matrix ({len(all_needed):,} samples)...")
    presence_all = build_presence(
        data.breadth.loc[all_needed],
        data.depth.loc[all_needed],
        data.lengths, spec=spec)

    # ── Build condition list ──────────────────────────────────────────────────
    conditions = [("Unhealthy (all)", "unhealthy", None)]
    for cond_label, cond_spec in config.DISEASE_NCD_MAP.items():
        conditions.append((cond_label,
                           cond_spec["type"],
                           cond_spec.get("cols", None)))

    # ── Per-condition projections ─────────────────────────────────────────────
    rows = []
    for cond_label, ctype, dcols in conditions:
        mask_ctrl, mask_case = build_group_masks(meta, ctype, dcols)
        case_ids = [s for s in meta.index[mask_case]
                    if s in presence_all.index]
        ctrl_pool = [s for s in meta.index[mask_ctrl]
                     if s in presence_all.index]

        n_case = len(case_ids)
        if n_case < MIN_N_CASE:
            print(f"    {cond_label:<28s}: n_case={n_case} < {MIN_N_CASE} — skip")
            continue

        # Cap case to MAX_REF (weighted subsampling)
        n_ref = min(n_case, MAX_REF)
        if n_case > MAX_REF:
            case_ids = list(rng.choice(case_ids, size=MAX_REF, replace=False))

        # Depth-matched controls from Healthy Industrial pool
        ctrl_ids = _depth_match_controls(case_ids, all_indh_ids,
                                          log_depth, rng, n_control=n_ref)
        if len(ctrl_ids) < MIN_N_CASE:
            print(f"    {cond_label:<28s}: too few controls — skip")
            continue

        # NI reference: subsample equal n
        ni_ref_n = min(n_ref, MAX_REF, len(all_ni_ids))
        ni_ids = list(rng.choice(all_ni_ids, size=ni_ref_n, replace=False))

        result = _project_condition(ni_ids, ctrl_ids, case_ids,
                                     presence_all, N_BOOT, rng)
        if result is None:
            print(f"    {cond_label:<28s}: PCoA failed — skip")
            continue

        cat = CONDITION_CATEGORY.get(cond_label, "other")
        rows.append(dict(
            condition=cond_label,
            category=cat,
            n_case=result["n_case"],
            n_ctrl=result["n_ctrl"],
            n_ni=result["n_ni"],
            p_ctrl=round(result["p_ctrl"], 4),
            p_case=round(result["p_case"], 4),
            p_case_ci_lo=round(result["p_case_ci_lo"], 4)
                         if np.isfinite(result["p_case_ci_lo"]) else np.nan,
            p_case_ci_hi=round(result["p_case_ci_hi"], 4)
                         if np.isfinite(result["p_case_ci_hi"]) else np.nan,
            shift=round(result["shift"], 4),
            shift_ci_lo=round(result["shift_ci_lo"], 4)
                        if np.isfinite(result["shift_ci_lo"]) else np.nan,
            shift_ci_hi=round(result["shift_ci_hi"], 4)
                        if np.isfinite(result["shift_ci_hi"]) else np.nan,
        ))

        sig = ""
        if np.isfinite(result["shift_ci_lo"]):
            if result["shift_ci_lo"] > 0:    sig = "↑ (case > ctrl)"
            elif result["shift_ci_hi"] < 0:  sig = "↓ (case < ctrl)"
            else:                             sig = "ns"
        print(f"    {cond_label:<28s}: n={n_case:4d}  "
              f"p_ctrl={result['p_ctrl']:+.3f}  "
              f"p_case={result['p_case']:+.3f}  "
              f"shift={result['shift']:+.3f}  {sig}")

        if stats is not None:
            tag = (f"health_axis_{cond_label}_{spec_tag}"
                   .replace(" ", "_").replace("/", "_").replace("(", "")
                   .replace(")", ""))
            stats.add(tag, "s3", "main",
                      f"Health axis projection: {cond_label} [{spec_tag}]",
                      f"p_case={result['p_case']:.4f} "
                      f"[{result['p_case_ci_lo']:.4f},{result['p_case_ci_hi']:.4f}], "
                      f"shift={result['shift']:+.4f} "
                      f"[{result['shift_ci_lo']:.4f},{result['shift_ci_hi']:.4f}], "
                      f"n_case={n_case}",
                      n1=n_case, n2=len(ctrl_ids))

    if not rows:
        print("  [health_axis] No conditions passed threshold")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    csv_path = os.path.join(outdir, f"health_axis_projection_{spec_tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    _plot(df, spec_tag, outdir, ref)
    return df


# ── Figure ────────────────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, spec_tag: str, outdir: str, ref: str) -> None:
    try:
        from .plot_utils import current_fs, savefig_dual
    except ImportError:
        def current_fs(): return 12
        def savefig_dual(fig, path, **kw): fig.savefig(path, **kw)
    FS = current_fs()

    # Sort by p_case (position along axis)
    df = df.sort_values("p_case").reset_index(drop=True)
    n_conds = len(df)
    if n_conds == 0:
        return

    fig_h = max(5, n_conds * 0.55 + 3)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    y_pos = np.arange(n_conds)

    for i, (_, row) in enumerate(df.iterrows()):
        cat   = row.get("category", "other")
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        y     = y_pos[i]
        p_case = float(row["p_case"])
        p_ctrl = float(row["p_ctrl"])

        # Line connecting control and case
        ax.plot([p_ctrl, p_case], [y, y],
                color=color, lw=1.2, alpha=0.6, zorder=2)

        # Control point (open circle)
        ax.scatter(p_ctrl, y, color="white", s=55,
                   edgecolors=color, linewidths=1.5,
                   zorder=4, marker="o")

        # Case point (filled circle) with CI bar
        ax.scatter(p_case, y, color=color, s=65,
                   edgecolors="black", linewidths=0.5,
                   zorder=5, marker="o")
        ci_lo = row.get("p_case_ci_lo", np.nan)
        ci_hi = row.get("p_case_ci_hi", np.nan)
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax.plot([ci_lo, ci_hi], [y, y],
                    color=color, lw=2.5, alpha=0.35,
                    solid_capstyle="round", zorder=3)

        # n annotation
        ax.text(max(df["p_case_ci_hi"].fillna(df["p_case"]).max(),
                    df["p_case"].max()) + 0.04,
                y, f"n={int(row['n_case']):,}",
                va="center", ha="left", fontsize=FS - 3,
                color=color)

    # Reference lines
    ax.axvline(0.0, color="#D55E00", lw=1.8, ls="--", alpha=0.7,
               label="Non-Industrial  (0.0)")
    ax.axvline(1.0, color="#56B4E9", lw=1.8, ls="--", alpha=0.7,
               label="Healthy Industrial  (1.0)")
    ax.axvline(0.5, color="gray",    lw=0.8, ls=":",  alpha=0.5,
               label="Midpoint  (0.5)")

    # Y axis: condition labels coloured by category
    ax.set_yticks(y_pos)
    labels = []
    for _, row in df.iterrows():
        cat   = row.get("category", "other")
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        labels.append(row["condition"])
    ax.set_yticklabels(labels, fontsize=FS - 1)
    for label, (_, row) in zip(ax.get_yticklabels(), df.iterrows()):
        cat = row.get("category", "other")
        label.set_color(CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"]))

    # Legend for categories
    from matplotlib.lines import Line2D
    cat_handles = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=CATEGORY_COLORS[c], markersize=9,
               markeredgecolor="black", markeredgewidth=0.5,
               label=c.capitalize())
        for c in ["combined", "metabolic", "gut", "other"]
        if c in df["category"].values
    ]
    cat_handles += [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=7, markeredgecolor="gray", lw=0,
               label="● = cases   ○ = depth-matched controls"),
    ]
    ax.axvline(0.0, color="#D55E00", lw=1.8, ls="--", alpha=0.7)
    ax.axvline(1.0, color="#56B4E9", lw=1.8, ls="--", alpha=0.7)
    ax.legend(handles=cat_handles + [
        Line2D([0],[0], color="#D55E00", lw=1.8, ls="--",
               label="Non-Industrial (0.0)"),
        Line2D([0],[0], color="#56B4E9", lw=1.8, ls="--",
               label="Healthy Industrial (1.0)"),
    ], fontsize=FS - 2, loc="lower right")

    ax.set_xlabel(
        "Fractional position on Non-Industrial → Healthy Industrial axis\n"
        "(0 = NI centroid,  1 = Healthy Industrial centroid;\n"
        " filled ● = disease cases,  open ○ = depth-matched healthy controls;\n"
        " bars = bootstrap 95% CI;  equal group weighting within each condition)",
        fontsize=FS - 1)
    ax.set_title(
        f"Disease-group projection onto NI\u2192Industrial extinction axis\n"
        f"{ref}  [{spec_tag}]  |  "
        f"per-condition Jaccard PCoA with equal weighting (1/3 NI, 1/3 ctrl, 1/3 case)",
        fontsize=FS + 1, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_ylim(-0.7, n_conds - 0.3)

    plt.tight_layout()
    png_path = os.path.join(outdir, f"health_axis_projection_{spec_tag}.png")
    try:
        savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    except Exception:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_health_axis_projection(data, stats, specs=None):
    """Called from __main__.py under --only s3_axis."""
    ref    = data.ref_name
    s3_main = os.path.join("results", ref, "section3_health", "main")
    os.makedirs(s3_main, exist_ok=True)

    if specs is None:
        from .detection import DetectionSpec
        specs = [
            DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    print(f"\n{'='*60}")
    print(f"  Health-axis projection [{ref}]")
    print(f"  (per-condition Jaccard PCoA, equal group weighting)")
    print(f"{'='*60}")

    for spec in specs:
        _run_one_spec(data, spec, outdir=s3_main, ref=ref, stats=stats)
