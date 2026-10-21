#!/usr/bin/env python3
"""Expand the parts of a figure that `extras` holds into tabulated panels.

`figure_source.emit` keeps non-tabular material out of `panels` -- an
already-rendered sub-panel, a fitted parameter set, a bootstrap's replicate
curves -- so that the submitted spreadsheet stays one sheet per plotted table.
That is the right default, but for some figures the *plotted quantity itself*
lives there, and `to_workbook` writes panels only. The result is a figure whose
caption describes eight panels and whose Data S1 entry covers two.

Figure 4 is the clearest case: panels A-D are the accumulation ribbons, held as
`extras["aVs"]` and seven siblings, and panels E-F are the per-taxon detection
counts in `extras["sI_soft"]`/`["typ_I"]`. Only G and H (the two volcano
frames) were ever panels. This module turns the rest into tidy frames, so the
data behind every lettered panel reaches the workbook.

Nothing here recomputes anything: each function reshapes what the run already
wrote. A figure with no entry in EXPANDERS is passed through untouched.

    from tools import data_s1_extras
    panels, extra_csvs = data_s1_extras.expand(obj)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Frames larger than this are written as CSVs beside the workbook rather than
#: as sheets. Excel's own limit is 1,048,576 rows; this is far below it, chosen
#: so that one figure's raw distributions cannot dominate the file.
CSV_ROW_THRESHOLD = 400_000

#: Figure 4's accumulation objects: extras key -> (list name, pooling unit).
#: The caption pairs them as (A,C) VANISH and (B,D) BloSSUM, by samples (A,B)
#: and studies (C,D).
_F4_ACCUM = {
    "aVs":   ("VANISH",   "samples"),
    "aVps":  ("VANISH+",  "samples"),
    "aBs":   ("BloSSUM",  "samples"),
    "aBps":  ("BloSSUM+", "samples"),
    "aVst":  ("VANISH",   "studies"),
    "aVpst": ("VANISH+",  "studies"),
    "aBst":  ("BloSSUM",  "studies"),
    "aBpst": ("BloSSUM+", "studies"),
}


def _accum_frame(obj: dict) -> pd.DataFrame:
    """Figure 4A-D: the accumulation ribbons, one row per list x unit x x-value."""
    rows = []
    for key, (lst, unit) in _F4_ACCUM.items():
        a = obj.get("extras", {}).get(key)
        if not isinstance(a, dict) or "ns" not in a:
            continue
        n = len(a["ns"])
        block = {
            "taxon_list": lst,
            "pooling_unit": unit,
            "n_pooled": np.asarray(a["ns"], dtype=float),
        }
        for q in ("q025", "q250", "q500", "q750", "q975"):
            if q in a and len(a[q]) == n:
                block[q] = np.asarray(a[q], dtype=float)
        block["n_list"] = a.get("n_list", np.nan)
        block["N_total"] = a.get("N_total", np.nan)
        rows.append(pd.DataFrame(block))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"q500": "median", "q025": "ci_lo_2.5",
                               "q975": "ci_hi_97.5", "q250": "q25",
                               "q750": "q75"})


def _accum_replicates(obj: dict) -> pd.DataFrame:
    """The bootstrap replicates behind those ribbons, long.

    The ribbons are quantiles of these; published so the interval can be
    checked rather than only re-run.
    """
    rows = []
    for key, (lst, unit) in _F4_ACCUM.items():
        a = obj.get("extras", {}).get(key)
        if not isinstance(a, dict) or "curves" not in a:
            continue
        c = np.asarray(a["curves"], dtype=float)
        ns = np.asarray(a["ns"], dtype=float)
        if c.ndim != 2 or c.shape[1] != len(ns):
            continue
        n_rep, n_pt = c.shape
        rows.append(pd.DataFrame({
            "taxon_list": lst,
            "pooling_unit": unit,
            "replicate": np.repeat(np.arange(n_rep), n_pt),
            "n_pooled": np.tile(ns, n_rep),
            "sgbs": c.reshape(-1),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _detection_frame(obj: dict) -> pd.DataFrame:
    """Figure 4E-F: samples needed for 50% detection, per taxon.

    `sI_soft`/`sN_soft` are the per-taxon soft prevalences the curve is read
    off; `typ_I`/`typ_NI` are the sample counts the panels plot. Membership in
    each list comes from the same object's `V`/`Vp`/`B`/`Bp`.
    """
    e = obj.get("extras", {})
    out = []
    for soft_k, typ_k, arm, n_k, lists in (
            ("sI_soft", "typ_I", "Industrial", "n_I", ("V", "Vp")),
            ("sN_soft", "typ_NI", "Non-Industrial", "n_NI", ("B", "Bp"))):
        soft, typ = e.get(soft_k), e.get(typ_k)
        if soft is None or typ is None:
            continue
        soft = pd.Series(soft)
        typ = np.asarray(typ, dtype=float)
        if len(typ) != len(soft):
            continue
        df = pd.DataFrame({
            "arm": arm,
            "taxon": soft.index.astype(str),
            "soft_prevalence": soft.to_numpy(dtype=float),
            "samples_for_50pct_detection": typ,
            "n_arm": e.get(n_k, np.nan),
        })
        for lk in lists:
            members = set(e.get(lk) or ())
            df[f"on_{lk}"] = df["taxon"].isin(members)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _f4(obj: dict):
    """Figure 4: panels A-D and E-F, which were extras-only."""
    panels, csvs = {}, {}
    acc = _accum_frame(obj)
    if not acc.empty:
        panels["accumulation"] = acc
    rep = _accum_replicates(obj)
    if not rep.empty:
        (csvs if len(rep) > CSV_ROW_THRESHOLD
         else panels)["accumulation_replicates"] = rep
    det = _detection_frame(obj)
    if not det.empty:
        panels["detection_samples"] = det
    return panels, csvs


#: Resolution of the published beta-diversity distributions, in percentiles.
#: 0.5 fixes each drawn distribution to 201 points, which determines a violin
#: or box plot exactly -- median, hinges, whiskers and tail shape.
_BETA_PCTL_STEP = 0.5


def _f2(obj: dict):
    """Figure 2D-E: the distributions behind the two beta-diversity panels.

    `beta_summary` publishes four numbers per distribution, which does not
    determine the drawn shape. The underlying arrays do, but they are 6.57
    million distances -- 205 MB as a CSV, and three of the eight exceed
    Excel's row limit on their own -- which is out of all proportion to a
    panel showing eight summarised distributions.

    So publish the distributions at half-percentile resolution: 201 points
    each, which reproduces every drawn feature, with the pair count and mean
    beside them. The full arrays remain in the archived data object for anyone
    who wants them.
    """
    raw = obj.get("extras", {}).get("beta_raw")
    if not isinstance(raw, dict) or not raw:
        return {}, {}
    qs = np.arange(0.0, 100.0 + _BETA_PCTL_STEP / 2, _BETA_PCTL_STEP)
    rows = []
    for key, vals in raw.items():
        v = np.asarray(vals, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        grp, _, metric = str(key).rpartition("_")
        rows.append(pd.DataFrame({
            "group": grp or str(key),
            "metric": metric or "",
            "percentile": qs,
            "distance": np.percentile(v, qs),
            "n_pairs": v.size,
            "mean_distance": float(v.mean()),
        }))
    if not rows:
        return {}, {}
    out = {"beta_distribution": pd.concat(rows, ignore_index=True)}
    cmp_df = _f2_comparisons(obj)
    if cmp_df is not None:
        out["comparisons"] = cmp_df
    return out, {}


def _f2_comparisons(obj: dict):
    """Figure 2D-E: the P values behind the significance brackets.

    The journal asks for the P values plotted on each figure. Figure 2 draws
    pairwise brackets over the beta-diversity panels, but the object it emits
    carries only the summarised distributions -- so those were the one set of
    P values on any submitted figure that Data S1 did not contain.

    They are recomputed rather than read, because the figure computes them at
    draw time and never stores them: `figures/figure2.py::_approx_p`, a
    two-sided normal approximation on the summary statistics. It is
    deterministic and every input is a column of `beta_summary`, so this
    reproduces the drawn number exactly rather than estimating it.

    The boxplot panel's own stars are left out: that panel is rendered by
    another module and embedded as an image, and its test subsamples the
    distance arrays under a seed, so its drawn value cannot be reproduced
    from what is deposited.
    """
    bs = obj.get("panels", {}).get("beta_summary")
    if bs is None or not len(bs):
        return None
    need = {"Lifestyle", "Metric", "n_samples", "mean", "q25", "q75"}
    if not need <= set(bs.columns):
        return None
    from scipy.stats import norm

    def approx_p(a, b):
        # figures/figure2.py::_approx_p, unchanged.
        na = max(float(a["n_samples"]), 2.0)
        nb = max(float(b["n_samples"]), 2.0)
        sda = max((float(a["q75"]) - float(a["q25"])) / 1.349, 1e-9)
        sdb = max((float(b["q75"]) - float(b["q25"])) / 1.349, 1e-9)
        se = np.sqrt(sda * sda / na + sdb * sdb / nb)
        z = abs(float(a["mean"]) - float(b["mean"])) / max(se, 1e-12)
        return float(2.0 * norm.sf(z))

    def star(p):
        return ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "ns")

    rows = []
    for metric, grp in bs.groupby("Metric", sort=False):
        recs = grp.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                p = approx_p(a, b)
                rows.append({
                    "Metric": metric,
                    "group_a": a["Lifestyle"], "group_b": b["Lifestyle"],
                    "mean_a": a["mean"], "mean_b": b["mean"],
                    "n_samples_a": a["n_samples"],
                    "n_samples_b": b["n_samples"],
                    "p_value": p, "significance": star(p),
                    "test": "two-sided normal approximation on the "
                            "summarised distributions",
                })
    return pd.DataFrame(rows) if rows else None


def _f5(obj: dict):
    """Figure 5E: the energy-distance p-values, held as a list of triples."""
    ep = obj.get("extras", {}).get("energy_p")
    if not ep:
        return {}, {}
    try:
        df = pd.DataFrame(list(ep), columns=["Condition", "Taxon_list",
                                             "p_energy"])
    except Exception:
        return {}, {}
    df["test"] = "energy distance, permutation"
    return {"energy_p": df}, {}


#: slot -> expander. A slot absent from here is passed through untouched.
EXPANDERS = {"F4": _f4, "F2": _f2, "F5": _f5}


def expand(obj: dict):
    """(extra panels, frames to ship as CSVs) for one data object.

    The returned panels are appended after the object's own, so a figure's
    existing sheet order is unchanged.
    """
    fn = EXPANDERS.get(str(obj.get("slot") or ""))
    if fn is None:
        return {}, {}
    try:
        return fn(obj)
    except Exception as exc:                        # pragma: no cover
        print(f"  [WARN] {obj.get('slot')}: could not expand extras: {exc}")
        return {}, {}
