"""Table S12: every list-level p-value, side by side, per detection rule.

The manuscript reports one statistic -- the mean distance from the fitted
prevalence background, combined across detection rules by the harmonic mean
p-value. Four other tests are computed and three of them were only ever drawn
as stars on a figure. This table puts them in one place so a reader can see
what each method says about the same condition-by-list cell, and judge the
threshold-robustness claim rather than take it.

Built from the files a run already writes, so it can also be regenerated from
a finished results tree without rerunning anything:

    python -m persistence.table_methods <section3_health/main> [ref]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SPECS = ("10pct", "300kb")
#: Column order: the primary statistic, then the combination, then the
#: secondary tests in the order 6G.6 introduces them.
_P_COLS = ("p_meanz", "p_energy", "p_iw_fisher", "p_iw_directional")

#: Two-sided and referenced to the background: these ask the same question of
#: the same cell, so counting how many agree is meaningful.
_TWO_SIDED = ("p_meanz", "p_energy")

#: One-sided in whichever direction dominates the list's raw counts. A p near
#: 1 does not mean "no effect": it means the list departs from its background
#: in the OTHER direction. BloSSUM_H+ under Unhealthy has mean z +1.44 at
#: p <= 0.0005 and an IW-directional p of 1.000, because the dominant raw
#: direction is depletion while the list is LESS depleted than its background.
#: Counting these alongside the two-sided tests would report that as a method
#: disagreement, which it is not.
_DIRECTIONAL = ("p_iw_fisher", "p_iw_directional")


def _hmp(ps):
    """Harmonic mean p-value (Wilson 2019), valid under arbitrary dependence."""
    ps = [p for p in ps if p is not None and np.isfinite(p) and p > 0]
    if not ps:
        return np.nan
    return len(ps) / sum(1.0 / p for p in ps)


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def build_method_comparison(s3_main, ref="combined", specs=SPECS,
                            out_name=None):
    """Merge every list-level p-value into one frame and write it."""
    key = ["Condition", "Taxon_list"]
    out = None
    for spec in specs:
        curve = _read(os.path.join(
            s3_main, f"health_background_curve_summary_{spec}.csv"))
        if curve is None:
            continue
        f = curve[["Condition", "Taxon_list", "mean_z", "ci_lo", "ci_hi",
                   "boot_p", "n_list", "n_supported"]].copy()
        f = f.rename(columns={"boot_p": f"p_meanz_{spec}",
                              "mean_z": f"mean_z_{spec}",
                              "ci_lo": f"z_lo_{spec}", "ci_hi": f"z_hi_{spec}"})

        # Energy distance and the binomial counts travel in Table S6B.
        s6b = _read(os.path.join(
            s3_main, f"table_s6b_energy_first_{spec}.csv"))
        if s6b is not None and "List" in s6b.columns:
            e = s6b.rename(columns={"List": "Taxon_list"})
            cols = {"energy_distance": f"energy_{spec}",
                    "energy_p": f"p_energy_{spec}",
                    "binom_p_dep": f"p_binom_dep_{spec}",
                    "binom_p_enr": f"p_binom_enr_{spec}"}
            e = e[key + [c for c in cols if c in e.columns]].rename(columns=cols)
            f = f.merge(e, on=key, how="left")

        # The two importance-weighted tests. Only the contrasts the figure
        # drew have them, because that is the loop they are computed in.
        iw = _read(os.path.join(s3_main, f"health_iw_tests_{ref}_{spec}.csv"))
        if iw is not None:
            iw = iw.rename(columns={"p_iw_fisher": f"p_iw_fisher_{spec}",
                                    "p_iw_directional": f"p_iw_directional_{spec}",
                                    "direction": f"direction_{spec}"})
            f = f.merge(iw, on=key, how="left")

        f = f.rename(columns={"n_list": f"n_list_{spec}",
                              "n_supported": f"n_supported_{spec}"})
        out = f if out is None else out.merge(f, on=key, how="outer")

    if out is None or out.empty:
        print("  [Table S12] no curve summaries found — skipping")
        return pd.DataFrame()

    # The combination the manuscript quotes, plus the same combination applied
    # to each secondary test so the comparison is like for like.
    for stem in _P_COLS:
        cols = [f"{stem}_{s}" for s in specs if f"{stem}_{s}" in out.columns]
        if len(cols) > 1:
            out[f"{stem}_hmp"] = [_hmp(r) for r in out[cols].to_numpy()]

    zc = [f"mean_z_{s}" for s in specs if f"mean_z_{s}" in out.columns]
    if len(zc) > 1:
        out["mean_z_combined"] = out[zc].mean(axis=1, skipna=True)
        out["sign_agree"] = (out[zc].apply(np.sign).nunique(axis=1,
                                                            dropna=True) == 1)

    # How many methods call this cell significant. Counted on the combined
    # values, because those are what the manuscript quotes: counting on one
    # spec made IBD x VANISH_H+ read as a disagreement when both methods
    # agree once their own thresholds are combined. Counted over the
    # two-sided tests only, for the reason given at _DIRECTIONAL.
    def _cols(stems):
        return ([f"{c}_hmp" for c in stems if f"{c}_hmp" in out.columns]
                or [f"{c}_{specs[0]}" for c in stems
                    if f"{c}_{specs[0]}" in out.columns])

    two = _cols(_TWO_SIDED)
    out["n_methods_sig"] = (out[two] < 0.05).sum(axis=1)
    out["n_methods_tested"] = out[two].notna().sum(axis=1)
    out["methods_agree"] = (out["n_methods_sig"] == 0) | \
        (out["n_methods_sig"] == out["n_methods_tested"])
    dirc = _cols(_DIRECTIONAL)
    if dirc:
        out["n_directional_sig"] = (out[dirc] < 0.05).sum(axis=1)
        out["n_directional_tested"] = out[dirc].notna().sum(axis=1)

    # Which contrasts the figures show. Kept as a column, not as a filter:
    # the excluded contrasts are the ones a reader most wants to check the
    # methods against, since they are excluded for want of studies and not
    # for want of signal.
    cmh = _read(os.path.join(
        s3_main, f"health_per_condition_cmh_{specs[0]}.csv"))
    if cmh is not None and "n_studies" in cmh.columns:
        from . import config as _cfg
        _cc = "Condition" if "Condition" in cmh.columns else "condition"
        k = cmh.groupby(_cc)["n_studies"].max()
        ok = {c for c, v in k.items()
              if np.isfinite(v) and v >= _cfg.MIN_STRATA_TO_PLOT}
        out["n_studies"] = out["Condition"].map(k)
        out["eligible"] = [bool(c in ok) for c in out["Condition"]]
    else:
        out["eligible"] = ""

    dcol = f"direction_{specs[0]}"
    if dcol in out.columns:
        out["direction"] = out[dcol]

    front = key + ["eligible", "n_studies", "direction", "mean_z_combined",
                   "p_meanz_hmp", "sign_agree",
                   "n_methods_sig", "n_methods_tested", "methods_agree"]
    rest = [c for c in out.columns if c not in front]
    out = out[[c for c in front if c in out.columns] + rest]
    out = out.sort_values(["Taxon_list", "Condition"]).reset_index(drop=True)

    path = os.path.join(s3_main,
                        out_name or f"table_s12_method_comparison_{ref}.csv")
    out.to_csv(path, index=False)
    print(f"    Saved Table S12 (method comparison): {path}")
    if "p_meanz_hmp" in out.columns:
        n_sig = int((out["p_meanz_hmp"] < 0.05).sum())
        n_all = int((out["n_methods_sig"] == out["n_methods_tested"]).sum())
        n_none = int((out["n_methods_sig"] == 0).sum())
        print(f"      {len(out)} cells: {n_sig} significant by the combined "
              f"mean z; {n_all} where every method tested agrees it is "
              f"significant, {n_none} where none does")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    build_method_comparison(sys.argv[1],
                            sys.argv[2] if len(sys.argv) > 2 else "combined")
