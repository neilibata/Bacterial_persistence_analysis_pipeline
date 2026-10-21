"""
Table S6B writer — prevalence-matched background-corrected summary.

Resolves Neil's supplement comment that the health directional summary
table is "not corrected for the background taxa, and is currently
misleading."

Column order matches the inferential story in Note 6.4:

    PRIMARY:       mean_z, z_CI_lo/hi, z_p   (distance from the prevalence
                   background; energy_distance/energy_p kept as shape stats)
    Distributional: OVL
    Direction:     direction_vs_bg, mean_shift
    Supportive:    frac_dep/enr + Wilson CIs + binomial p-values
    Context:       bg_frac_*, dep/enr_excess_vs_bg, n_bg_taxa

Table S6A (unadjusted, kept for transparency) written separately.

Standalone repair::

    python -m persistence.table_s7 --ref UHGG --spec flat_bp_1000000
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest

# The stats tag keeps the internal condition key so the manuscript crosswalk
# still matches on it; the description is what a reader sees.
try:
    from .config import display_text as _display
except Exception:                                  # standalone use
    def _display(s):
        return str(s)

PRIMARY_LISTS  = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+",
                  "VanishH+", "BlossumH+"]

# Which column holds the case-control shift, and which the background weight.
# `rd_mh` is the corrected study-stratified shift; `prev_diff` is the pooled one
# the 29 Sep run published. Mirrors section3_health._shift_values so the table
# and the figures can never describe different numbers.
SHIFT_COLUMNS = ("rd_mh", "prev_diff")
BG_WEIGHT_COLUMNS = ("bg_weight", "importance_weight")


def _shift_col(df):
    for c in SHIFT_COLUMNS:
        if c in getattr(df, "columns", ()):
            return c
    return None


def _shift_vals(df):
    c = _shift_col(df)
    return (pd.to_numeric(df[c], errors="coerce").values.astype(float)
            if c else np.array([]))
NULL_RATE      = 0.05
TAXON_SIG_THRESH = 0.05


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def _wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    z     = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2*n)) / denom
    margin = z * ((p_hat*(1-p_hat) + z**2/(4*n)) / n)**0.5 / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── OVL ───────────────────────────────────────────────────────────────────────

def _bg_weights(df):
    """The matched-background weights, or ones if the frame carries none.

    `health_background.matched_background` exists to produce these: the
    weight sum_i 1/|M_i| is the only thing that makes the background set
    prevalence-matched to the list. Any statistic computed over the
    background without them is taken over the union of the matched sets,
    which is dominated by the low-prevalence end where the catalogue is
    dense -- the exact bias the matching was written to remove.
    """
    w = np.ones(len(df))
    col = next((c for c in BG_WEIGHT_COLUMNS if c in df.columns), None)
    if col:
        w = np.maximum(df[col].fillna(1.0).values.astype(float), 0.0)
    return w


def _ovl(lv, bv, n_bins=200, bw=None):
    if len(lv) < 3 or len(bv) < 3:
        return np.nan
    combined = np.concatenate([lv, bv])
    lo, hi   = combined.min(), combined.max()
    if lo >= hi:
        return 1.0
    edges    = np.linspace(lo, hi, n_bins+1)
    hl, _    = np.histogram(lv, bins=edges, density=True)
    hb, _    = np.histogram(bv, bins=edges, density=True, weights=bw)
    return float(np.sum(np.minimum(hl, hb)) * (hi-lo)/n_bins)


# ── IW-weighted background fractions ─────────────────────────────────────────

def _iw_bg_fracs(sub_b, sig_thresh=TAXON_SIG_THRESH):
    if sub_b.empty:
        return np.nan, np.nan
    if "taxon" in sub_b.columns:
        sub_b = sub_b.groupby("taxon", sort=False).first().reset_index()
    w = np.ones(len(sub_b))
    _wc = next((c for c in BG_WEIGHT_COLUMNS if c in sub_b.columns), None)
    if _wc:
        w = np.maximum(sub_b[_wc].fillna(1.0).values.astype(float), 0.0)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return np.nan, np.nan
    if "direction" in sub_b.columns:
        dd = (sub_b["direction"].values == 1).astype(float)
        de = (sub_b["direction"].values == -1).astype(float)
    elif _shift_col(sub_b) is not None:
        _v = _shift_vals(sub_b)
        dd = (_v > 0).astype(float)
        de = (_v < 0).astype(float)
    else:
        return np.nan, np.nan
    sig = np.ones(len(sub_b), dtype=bool)
    if "pval" in sub_b.columns:
        sig = sub_b["pval"].fillna(1.0).values < sig_thresh
    return float(np.sum(w*dd*sig)/w_sum), float(np.sum(w*de*sig)/w_sum)


# ── Direction of mean shift vs background ─────────────────────────────────────

def _direction_vs_bg(sub_l, sub_b):
    if _shift_col(sub_l) is None:
        return "none", np.nan
    _lv = _shift_vals(sub_l)
    focal_mean = float(np.nanmean(_lv)) if len(_lv) else np.nan
    bg_mean    = np.nan
    if not sub_b.empty and _shift_col(sub_b) is not None:
        bg_u = (sub_b.groupby("taxon", sort=False).first().reset_index()
                if "taxon" in sub_b.columns else sub_b)
        _bv = _shift_vals(bg_u)
        # Weighted, like _iw_bg_fracs thirty lines above. An unweighted mean
        # here put the comparison at the catalogue's prevalence rather than
        # the list's, and the direction it reported was the reverse of the
        # mean_z printed in the same row for roughly 40% of cells.
        _bw = _bg_weights(bg_u)
        _ok = np.isfinite(_bv)
        if _ok.any() and _bw[_ok].sum() > 1e-12:
            bg_mean = float(np.average(_bv[_ok], weights=_bw[_ok]))
    shift = (focal_mean - bg_mean
             if np.isfinite(focal_mean) and np.isfinite(bg_mean) else np.nan)
    if np.isnan(shift):   direction = "none"
    elif shift >  0.01:   direction = "depleted"
    elif shift < -0.01:   direction = "enriched"
    else:                 direction = "none"
    return direction, shift


# ── S6B row ───────────────────────────────────────────────────────────────────

def _build_s6b_row(cond, list_name, sub_l, sub_b,
                   energy_stats, sig_thresh=TAXON_SIG_THRESH,
                   curve_stats=None):
    n_taxa = len(sub_l)
    if n_taxa == 0:
        return {}

    if "direction" in sub_l.columns:
        dep_mask = sub_l["direction"].values == 1
        enr_mask = sub_l["direction"].values == -1
    elif _shift_col(sub_l) is not None:
        _v = _shift_vals(sub_l)
        dep_mask = _v > 0
        enr_mask = _v < 0
    else:
        dep_mask = enr_mask = np.zeros(n_taxa, dtype=bool)

    sig_mask = (sub_l["pval"].fillna(1.0).values < sig_thresh
                if "pval" in sub_l.columns else np.ones(n_taxa, dtype=bool))

    n_dep, n_enr = int((dep_mask & sig_mask).sum()), int((enr_mask & sig_mask).sum())
    frac_dep, frac_enr = n_dep/n_taxa, n_enr/n_taxa
    dep_lo, dep_hi = _wilson_ci(n_dep, n_taxa)
    enr_lo, enr_hi = _wilson_ci(n_enr, n_taxa)
    bp_dep = binomtest(n_dep, n_taxa, NULL_RATE, alternative="greater").pvalue
    bp_enr = binomtest(n_enr, n_taxa, NULL_RATE, alternative="greater").pvalue

    bg_frac_dep, bg_frac_enr = _iw_bg_fracs(sub_b, sig_thresh)
    dep_excess = frac_dep - bg_frac_dep if np.isfinite(bg_frac_dep) else np.nan
    enr_excess = frac_enr - bg_frac_enr if np.isfinite(bg_frac_enr) else np.nan
    n_bg = (sub_b["taxon"].nunique() if "taxon" in sub_b.columns
            else len(sub_b)) if not sub_b.empty else 0

    estat = energy_stats.get((cond, list_name), {})
    E, ep = estat.get("E", np.nan), estat.get("p", np.nan)

    ovl_val = np.nan
    if _shift_col(sub_l) is not None and not sub_b.empty:
        sl = _shift_vals(sub_l)
        sl = sl[np.isfinite(sl)]
        bg_u = (sub_b.groupby("taxon", sort=False).first().reset_index()
                if "taxon" in sub_b.columns else sub_b)
        sb_all = _shift_vals(bg_u)
        bw_all = _bg_weights(bg_u)
        _m = np.isfinite(sb_all)
        sb, bw = sb_all[_m], bw_all[_m]
        if len(sl) >= 3 and len(sb) >= 3:
            ovl_val = _ovl(sl, sb, bw=bw)

    direction, mean_shift = _direction_vs_bg(sub_l, sub_b)

    def _f(v):
        return round(float(v), 6) if isinstance(v, (int, float)) and np.isfinite(v) else ""

    # PRIMARY under the corrected method: the mean distance from the
    # background curve, in SDs of the spread among taxa of the same
    # commonness, with a CI that refits the curve on resampled background
    # taxa. The energy distance stays in the table as a shape statistic, but
    # it treats the weighted background as known, so it is not the test the
    # text quotes.
    cs = (curve_stats or {}).get((cond, list_name)) or {}

    return {
        # Identification
        "Condition":         cond,
        "List":              list_name,
        "n_taxa":            n_taxa,
        # PRIMARY — distance from the prevalence background
        "mean_z":            _f(cs.get("mean_z", np.nan)),
        "z_CI_lo":           _f(cs.get("ci_lo", np.nan)),
        "z_CI_hi":           _f(cs.get("ci_hi", np.nan)),
        "z_p":               _f(cs.get("boot_p", np.nan)),
        "z_mde_80":          _f(cs.get("mde_80", np.nan)),
        "n_supported":       cs.get("n_supported", ""),
        "n_unsupported":     cs.get("n_unsupported", ""),
        # Secondary — energy distance (distributional shape only)
        "energy_distance":   _f(E),
        "energy_p":          _f(ep),
        # Distributional
        "OVL":               _f(ovl_val),
        # Direction vs background
        "direction_vs_bg":   direction,
        "mean_shift":        _f(mean_shift),
        # Supportive
        "frac_dep_cases":    round(frac_dep, 4),
        "frac_enr_cases":    round(frac_enr, 4),
        "dep_CI_lo":         round(dep_lo, 4),
        "dep_CI_hi":         round(dep_hi, 4),
        "enr_CI_lo":         round(enr_lo, 4),
        "enr_CI_hi":         round(enr_hi, 4),
        "binom_p_dep":       round(float(bp_dep), 6),
        "binom_p_enr":       round(float(bp_enr), 6),
        # Context
        "bg_frac_dep":       _f(bg_frac_dep),
        "bg_frac_enr":       _f(bg_frac_enr),
        "dep_excess_vs_bg":  _f(dep_excess),
        "enr_excess_vs_bg":  _f(enr_excess),
        "n_bg_taxa":         n_bg,
    }


# ── S6A (unadjusted) row ──────────────────────────────────────────────────────

def _build_s6a_row(cond, list_name, sub_l, sig_thresh=TAXON_SIG_THRESH):
    n_taxa = len(sub_l)
    if n_taxa == 0:
        return {}
    if "direction" in sub_l.columns:
        dep_mask = sub_l["direction"].values == 1
        enr_mask = sub_l["direction"].values == -1
    elif _shift_col(sub_l) is not None:
        _v = _shift_vals(sub_l)
        dep_mask = _v > 0
        enr_mask = _v < 0
    else:
        dep_mask = enr_mask = np.zeros(n_taxa, dtype=bool)
    sig_mask = (sub_l["pval"].fillna(1.0).values < sig_thresh
                if "pval" in sub_l.columns else np.ones(n_taxa, dtype=bool))
    n_dep, n_enr = int((dep_mask & sig_mask).sum()), int((enr_mask & sig_mask).sum())
    frac_dep, frac_enr = n_dep/n_taxa, n_enr/n_taxa
    dep_lo, dep_hi = _wilson_ci(n_dep, n_taxa)
    enr_lo, enr_hi = _wilson_ci(n_enr, n_taxa)
    bp_dep = binomtest(n_dep, n_taxa, NULL_RATE, alternative="greater").pvalue
    bp_enr = binomtest(n_enr, n_taxa, NULL_RATE, alternative="greater").pvalue
    return {
        "Condition":      cond,
        "List":           list_name,
        "n_taxa":         n_taxa,
        "n_dep_sig":      n_dep,
        "n_enr_sig":      n_enr,
        "frac_dep_cases": round(frac_dep, 4),
        "frac_enr_cases": round(frac_enr, 4),
        "dep_CI_lo":      round(dep_lo, 4),
        "dep_CI_hi":      round(dep_hi, 4),
        "enr_CI_lo":      round(enr_lo, 4),
        "enr_CI_hi":      round(enr_hi, 4),
        "binom_p_dep":    round(float(bp_dep), 6),
        "binom_p_enr":    round(float(bp_enr), 6),
        "note":           "UNADJUSTED — no background correction",
    }


# ── Console summary ───────────────────────────────────────────────────────────

def _print_s6b_summary(df, spec_tag):
    if df.empty:
        return
    print(f"\n    Table S6B [{spec_tag}] — background-corrected summary:")
    hdr = f"    {'Condition':<24s} {'List':<12s} {'E':>7s} {'E_p':>10s} {'OVL':>6s} {'direction':<12s} {'dep%':>6s}"
    print(hdr)
    print(f"    {'-'*80}")
    for _, r in df.iterrows():
        def _sv(k, fmt=".4f"):
            v = r.get(k, "")
            try: return format(float(v), fmt)
            except: return "n/a"
        print(f"    {str(r['Condition']):<24s} {str(r['List']):<12s} "
              f"{_sv('energy_distance'):>7s} {_sv('energy_p'):>10s} "
              f"{_sv('OVL','.3f'):>6s} "
              f"{str(r.get('direction_vs_bg','')):<12s} "
              f"{float(r.get('frac_dep_cases',0))*100:>5.1f}%")


# ── Public entry point ────────────────────────────────────────────────────────

def write_table_s7(per_taxon_df, bg_df, cmh_df,
                   energy_stats, s3_main, spec_tag, stats,
                   sig_thresh=TAXON_SIG_THRESH, table_prefix="table_s6",
                   curve_stats=None):
    """
    Write Table S6A (unadjusted) and Table S6B (background-corrected).

    S6B leads with the list's mean distance from the prevalence background
    (`mean_z`, with a bootstrap CI that refits the curve) when `curve_stats`
    is supplied. The energy-distance columns stay beside it as descriptive
    shape statistics.

    `table_prefix` names the output pair. The demographic strata produce the
    same summary from the same columns, and pass their own prefix so the two
    runs do not overwrite each other's tables.
    """
    os.makedirs(s3_main, exist_ok=True)

    if per_taxon_df is None or per_taxon_df.empty:
        print("  [Table S6B] per_taxon_df is empty — skipping")
        return pd.DataFrame(), pd.DataFrame()

    print(f"\n  Table S6A/S6B writer [{spec_tag}]")

    has_bg     = bg_df is not None and not bg_df.empty
    conditions = list(dict.fromkeys(per_taxon_df["Condition"]))
    lists_avail = set(per_taxon_df["Taxon_list"].unique())

    rows_s6a, rows_s6b = [], []

    for cond in conditions:
        for list_name in PRIMARY_LISTS:
            if list_name not in lists_avail:
                continue
            sub_l = per_taxon_df[
                (per_taxon_df["Condition"] == cond) &
                (per_taxon_df["Taxon_list"] == list_name)].copy()
            if sub_l.empty:
                continue

            row_a = _build_s6a_row(cond, list_name, sub_l, sig_thresh)
            if row_a:
                rows_s6a.append(row_a)

            sub_b = pd.DataFrame()
            if has_bg:
                bg_name = f"{list_name}_bg"
                sub_b = bg_df[
                    (bg_df["Condition"] == cond) &
                    (bg_df["Taxon_list"] == bg_name)].copy()

            row_b = _build_s6b_row(cond, list_name, sub_l, sub_b,
                                   energy_stats, sig_thresh,
                                   curve_stats=curve_stats)
            if row_b:
                rows_s6b.append(row_b)
                tag = (f"{table_prefix}b_{cond}_{list_name}_{spec_tag}"
                       .replace(" ", "_").replace("(", "").replace(")", ""))
                ed, ep = row_b.get("energy_distance",""), row_b.get("energy_p","")
                stats.add(
                    tag, "s3", "supplement",
                    f"Table S6B: {_display(cond)} × {list_name} [{spec_tag}]",
                    f"E={ed}, p={ep}, OVL={row_b.get('OVL','')}, "
                    f"direction={row_b['direction_vs_bg']}, "
                    f"frac_dep={row_b['frac_dep_cases']:.3f}",
                    test="IW energy distance (primary) + Wilson CI + binomial",
                    p_value=(float(ep) if isinstance(ep, (int,float)) and ep != ""
                             else None),
                    n1=row_b["n_taxa"],
                )

    def _mark_eligible(df):
        """Flag which contrasts clear the eligibility rule, without dropping any.

        The figures show only contrasts resting on at least
        config.MIN_STRATA_TO_PLOT studies of at least config.CMH_MIN_STRATUM
        samples per arm. The tables deliberately keep every contrast, because
        the excluded ones are still evidence and the table is where a reader
        should be able to see them. Until this column existed, the five-row
        tables in the supplement had to be filtered out of a nine-row CSV by
        hand after every run -- a step that works until the one time someone
        forgets. Sort or filter on `eligible` instead.
        """
        if df.empty or "Condition" not in df.columns:
            return df
        import numpy as _np
        from . import config as _cfg
        ok = None
        if cmh_df is not None and not getattr(cmh_df, "empty", True) \
                and "n_studies" in getattr(cmh_df, "columns", ()):
            _cc = "Condition" if "Condition" in cmh_df.columns else "condition"
            k = cmh_df.groupby(_cc)["n_studies"].max()
            ok = {c for c, v in k.items()
                  if _np.isfinite(v) and v >= _cfg.MIN_STRATA_TO_PLOT}
        if ok is None:
            df["eligible"] = ""          # no counts available; say so rather than guess
            return df
        df["eligible"] = [bool(c in ok) for c in df["Condition"]]
        n_out = int((~df["eligible"].astype(bool)).sum())
        if n_out:
            print(f"    [eligible] {n_out} of {len(df)} rows are contrasts the "
                  f"figures exclude (<{_cfg.MIN_STRATA_TO_PLOT} qualifying "
                  f"studies); kept in the table and flagged")
        return df

    s6a_df = _mark_eligible(pd.DataFrame(rows_s6a) if rows_s6a else pd.DataFrame())
    if not s6a_df.empty:
        path_a = os.path.join(
            s3_main, f"{table_prefix}a_unadjusted_{spec_tag}.csv")
        s6a_df.to_csv(path_a, index=False)
        print(f"    Saved Table S6A (unadjusted): {path_a}")

    s6b_df = _mark_eligible(pd.DataFrame(rows_s6b) if rows_s6b else pd.DataFrame())
    if not s6b_df.empty:
        path_b = os.path.join(
            s3_main, f"{table_prefix}b_energy_first_{spec_tag}.csv")
        s6b_df.to_csv(path_b, index=False)
        print(f"    Saved Table S6B (energy-first): {path_b}")
        _print_s6b_summary(s6b_df, spec_tag)
        if not has_bg:
            print("    [WARN] bg_df was empty — background columns will be blank. "
                  "Re-run --only s3 first.")

    return s6a_df, s6b_df


# ── Standalone repair ─────────────────────────────────────────────────────────

def _standalone_repair(ref, spec_tag):
    s3_main  = os.path.join("results", ref, "section3_health", "main")
    gmw_path = os.path.join(s3_main, f"health_per_taxon_gmw_{spec_tag}.csv")
    bg_path  = os.path.join(s3_main, f"health_per_taxon_prevmatched_{spec_tag}.csv")
    if not os.path.exists(gmw_path):
        raise FileNotFoundError(f"Not found: {gmw_path}")
    per_taxon_df = pd.read_csv(gmw_path)
    bg_df = pd.read_csv(bg_path) if os.path.exists(bg_path) else pd.DataFrame()

    class _Stub:
        def add(self, *a, **kw): pass

    write_table_s7(per_taxon_df, bg_df, None, {}, s3_main, spec_tag, _Stub())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Standalone Table S6B repair.")
    p.add_argument("--ref",  required=True)
    p.add_argument("--spec", required=True)
    a = p.parse_args()
    _standalone_repair(a.ref, a.spec)
