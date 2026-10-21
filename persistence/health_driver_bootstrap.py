"""Per-taxon disease association by a cluster bootstrap over study cohorts.

This replaces the one-sample t-test `_run_taxon_drivers` uses to decide which
taxa drive a list's health association (Figs. S26-S29).  The t-test treats the
per-condition z values as independent observations and estimates their spread
from themselves.  On the four focal contrasts that leaves df = 3, and it
returns ZERO significant taxa for BloSSUM+ and BlossumH+ even though 82 and 79
of their 100 taxa are directionally consistent in at least 70% of conditions.
It is also mis-specified: the five-condition version rewarded correlated
contrasts, so dropping the pooled "Unhealthy" column -- which contains the
other four conditions' cases -- swung the BloSSUM+ count from 52 to 0.

The statistic here keeps Figure 5E's own z,

    z_ic = (shift_ic - mu_c(x_ic)) / max(sigma_c(x_ic), se_ic),
    x = log10(control prevalence),

so a taxon is still measured as its departure from what comparable-prevalence
taxa do in that condition, and the curve is the one the figure draws.  What
changes is only the error term: the background curve is held fixed and the
STUDY COHORTS are resampled.  Each replicate redraws the study clusters with
replacement, recombines the Mantel-Haenszel shift and control prevalence per
condition from whichever cohorts were drawn, re-reads z off the fixed curve,
and averages across conditions with every condition weighted equally.  That
puts the uncertainty where the sampling actually happens -- between cohorts --
and carries the dependence from cohorts serving more than one condition and
from the shared healthy control pool.

Eligibility has two parts, and both matter:

  `supported`   the background curve's own per-condition flag.  It drops taxa
                with too few comparable-prevalence neighbours AND taxa detected
                in no sample of either arm.  The latter is essential: such a
                taxon has shift = 0 because nothing was observed, every one of
                them is floored onto log10(prev_floor), and they therefore all
                collect the same z -- whatever (0 - mu)/sigma happens to be at
                the floor.  Since mu > 0 there (a taxon that rare can only
                gain), they all come out "depleted" with a bootstrap spread of
                exactly zero and p at the 1/NB floor.  Scoring the 124 VANISH
                taxa without this flag produced 35 spurious depletions, 11 of
                them taxa never detected in any sample.  Figure 5E and
                `_run_taxon_drivers` both already apply it.

  `min_cells`   detected in at least this many of the condition's cohorts.
                `supported` only removes taxa seen in zero samples; one seen in
                a single cohort out of 49 still has a shift pinned within a
                hair of zero, so z keeps the sign mu alone dictates in every
                replicate, drawn or not.  Counts are stable from min_cells = 1
                to 5, so nothing here hinges on the choice.

Per-condition q-values are reported alongside the cross-condition ones because
several high-prevalence taxa reverse sign between conditions -- Phocaeicola
dorei is -1.8 in obesity and +7.3 in IBD -- and a single collapsed score hides
that.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config


def _nanmean(a, axis):
    """nanmean without the all-NaN-slice warning; unscored taxa are expected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)

#: Cohorts a taxon must be detected in, per condition, to be scored there.
MIN_CELLS = 3
#: Conditions a taxon must be eligible in for a cross-condition score.
MIN_COND = 3
#: Bootstrap replicates and seed.
N_BOOT = 2000
SEED = 17
#: Prevalence floor, matching health_background's own.
PREV_FLOOR = 1e-4


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    if not ok.any():
        return out
    q = p[ok]
    n = len(q)
    o = np.argsort(q)
    v = q[o] * n / np.arange(1, n + 1)
    v = np.minimum.accumulate(v[::-1])[::-1]
    r = np.empty(n)
    r[o] = np.minimum(v, 1.0)
    out[ok] = r
    return out


def per_stratum_shifts(data, spec, conditions, section_mod=None):
    """Per-cohort case-control shifts, which the MH collapse discards.

    `stratified_shifts_per_taxon` builds weighted present-counts per stratum and
    then reduces them to one Mantel-Haenszel number per taxon per condition.
    The bootstrap needs the per-stratum layer, so this reruns the same
    construction with the repo's own helpers and keeps it.

    Returns {condition: {"shift", "p_ctrl", "W", "studies", "n1", "n2"}} with
    `shift` on the case-minus-control convention the figures use, plus a
    "taxa" key holding the column order.
    """
    from .health_cmh import (_get_study_strata, build_weighted_strata,
                             build_group_masks)
    if section_mod is None:
        from . import section3_health as section_mod

    bdf, thresh = section_mod._build_presence_matrix(data, spec)
    taxa = list(bdf.columns)
    meta = data.meta
    out = {}
    comps = section_mod._health_comparisons()
    for cond, (ctype, dcols) in comps.items():
        if conditions is not None and cond not in conditions:
            continue
        m1, m2 = build_group_masks(meta, ctype, dcols)
        strata = _get_study_strata(meta, m1, m2, bdf, config.CMH_MIN_STRATUM)
        if not strata:
            print(f"    [driver_bootstrap] {cond}: no eligible strata")
            continue
        ws = build_weighted_strata(strata, meta, config.CMH_MIN_STRATUM)
        n_s = len(ws)
        A = np.zeros((n_s, len(taxa)))
        C = np.zeros((n_s, len(taxa)))
        N1 = np.zeros(n_s)
        N2 = np.zeros(n_s)
        names = []
        for k, ((g1, g2, w1, w2), (st, _, _)) in enumerate(zip(ws, strata)):
            present = bdf.loc[g1, taxa].values >= thresh
            A[k] = w1 @ present
            C[k] = w2 @ (bdf.loc[g2, taxa].values >= thresh)
            N1[k] = w1.sum()
            N2[k] = w2.sum()
            names.append(st)
        keep = (N1 > 0) & (N2 > 0)
        A, C, N1, N2 = A[keep], C[keep], N1[keep], N2[keep]
        names = [s for s, k in zip(names, keep) if k]
        p1 = A / N1[:, None]
        p2 = C / N2[:, None]
        out[cond] = {"shift": p2 - p1, "p_ctrl": p1,
                     "W": N1 * N2 / (N1 + N2),
                     "studies": names, "n1": N1, "n2": N2}
        print(f"    [driver_bootstrap] {cond}: {len(names)} cohorts x "
              f"{len(taxa):,} taxa")
    out["taxa"] = taxa
    return out


def _flatten(ps):
    """Per-stratum dict -> cell x taxon arrays plus cell labels."""
    S, P, W, cond, study = [], [], [], [], []
    for c, r in ps.items():
        if c == "taxa":
            continue
        for k in range(r["shift"].shape[0]):
            S.append(r["shift"][k])
            P.append(r["p_ctrl"][k])
            W.append(r["W"][k])
            cond.append(c)
            study.append(r["studies"][k])
    return (np.vstack(S), np.vstack(P), np.asarray(W, float),
            np.asarray(cond), np.asarray(study))


def _zed(shift_pp, p_ctrl, se_pp, gx, gmu, gsd):
    """Figure 5E's z. `shift_pp`, `se_pp`, and the curve are percentage points."""
    x = np.log10(np.maximum(p_ctrl, PREV_FLOOR))
    mu = np.interp(x, gx, gmu)
    sd = np.interp(x, gx, gsd)
    return (shift_pp - mu) / np.maximum(np.maximum(sd, se_pp), 1e-12)


def driver_bootstrap(ps, curve_grid, stratified, list_taxa, taxon_lists,
                     conditions, min_cells=MIN_CELLS, min_cond=MIN_COND,
                     n_boot=N_BOOT, seed=SEED):
    """Cluster-bootstrap driver statistics for every list.

    Parameters
    ----------
    ps          : per_stratum_shifts output
    curve_grid  : health_background_curve_grid_<tag>.csv as a DataFrame
                  (columns x, mu, sd, Condition, Taxon_list)
    stratified  : health_per_taxon_stratified_<tag>.csv as a DataFrame; supplies
                  rd_mh_se, which is a FRACTION and so is scaled by 100 here
    list_taxa   : the background curve's per-taxon frame, carrying `supported`
    taxon_lists : {list name: [taxon, ...]}
    conditions  : the conditions to score over, in display order

    Returns a tidy frame, one row per (Taxon_list, taxon).
    """
    taxa = np.asarray(ps["taxa"])
    idx = {t: i for i, t in enumerate(taxa)}
    S, P, W, cond, study = _flatten(ps)
    conds = [c for c in conditions if (cond == c).any()]
    cells = {c: np.where(cond == c)[0] for c in conds}

    se = {}
    for c in conds:
        s = stratified[stratified["Condition"] == c].set_index("taxon")
        se[c] = 100.0 * s["rd_mh_se"].reindex(taxa).values

    sup = {}
    for (lst, c), g in list_taxa.groupby(["Taxon_list", "Condition"]):
        sup[(lst, c)] = set(g.loc[g["supported"].astype(bool), "taxon"])

    det = {c: ((P[cells[c]] > 0) | ((P + S)[cells[c]] > 0)).sum(0)
           for c in conds}

    curves = {}
    for c in conds:
        for lst in taxon_lists:
            g = curve_grid[(curve_grid["Condition"] == c)
                           & (curve_grid["Taxon_list"] == lst)].sort_values("x")
            if len(g):
                curves[(c, lst)] = (g["x"].values, g["mu"].values,
                                    g["sd"].values)

    def z_of(lst, c, ii, sel=None):
        sel = cells[c] if sel is None else sel
        Wc = W[sel]
        tot = Wc.sum()
        if tot <= 0 or (c, lst) not in curves:
            return np.full(len(ii), np.nan)
        sh = 100.0 * (Wc @ S[sel][:, ii]) / tot
        pc = (Wc @ P[sel][:, ii]) / tot
        gx, gmu, gsd = curves[(c, lst)]
        return _zed(sh, pc, np.nan_to_num(se[c][ii]), gx, gmu, gsd)

    uniq = np.unique(study)
    cells_of = {s: np.where(study == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    draws = [rng.choice(uniq, len(uniq), replace=True) for _ in range(n_boot)]
    # Per replicate, the cells each condition keeps. Shared across lists so the
    # resampling is identical for all of them, which is what makes the lists
    # comparable.
    keep_cells = []
    for pick in draws:
        per_c = {}
        for c in conds:
            parts = [cells_of[s][cond[cells_of[s]] == c] for s in pick
                     if (cond[cells_of[s]] == c).any()]
            per_c[c] = np.concatenate(parts) if parts else np.empty(0, int)
        keep_cells.append(per_c)

    rows = []
    member = {k: set(v) for k, v in taxon_lists.items()}
    for lst, mem in taxon_lists.items():
        mem = [t for t in mem if t in idx]
        if not mem:
            continue
        ii = np.array([idx[t] for t in mem])
        names = taxa[ii]

        elig = np.vstack([
            np.array([t in sup.get((lst, c), set()) for t in names])
            & (det[c][ii] >= min_cells)
            for c in conds])
        n_elig = elig.sum(0)
        scored = n_elig >= min(min_cond, len(conds))

        point = np.vstack([z_of(lst, c, ii) for c in conds])
        masked = np.where(elig, point, np.nan)
        mean_z = _nanmean(masked, axis=0)

        bz = np.full((n_boot, len(ii)), np.nan)
        bc = {c: np.full((n_boot, len(ii)), np.nan) for c in conds}
        for b, per_c in enumerate(keep_cells):
            zs = []
            for j, c in enumerate(conds):
                sel = per_c[c]
                v = (z_of(lst, c, ii, sel) if len(sel)
                     else np.full(len(ii), np.nan))
                bc[c][b] = np.where(elig[j], v, np.nan)
                zs.append(bc[c][b])
            bz[b] = _nanmean(np.vstack(zs), axis=0)

        def two_sided(arr):
            pos = _nanmean(arr > 0, axis=0)
            neg = _nanmean(arr < 0, axis=0)
            return np.clip(2 * np.minimum(pos, neg), 1.0 / n_boot, 1.0)

        p_all = two_sided(bz)
        p_all = np.where(scored, p_all, np.nan)
        q_all = bh(p_all)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanpercentile(bz, 2.5, axis=0)
            hi = np.nanpercentile(bz, 97.5, axis=0)
            cons = np.nanmax(np.vstack([
                np.nanmean(np.sign(masked) > 0, axis=0),
                np.nanmean(np.sign(masked) < 0, axis=0)]), axis=0)

        pc_q = {}
        for j, c in enumerate(conds):
            pv = np.where(elig[j], two_sided(bc[c]), np.nan)
            pc_q[c] = (pv, bh(pv))

        for r, t in enumerate(names):
            row = {
                "Taxon_list": lst, "taxon": t,
                "n_conditions": int(n_elig[r]),
                "scored": bool(scored[r]),
                "driver_score": mean_z[r],
                "mean_z": mean_z[r],
                "ci_lo": lo[r], "ci_hi": hi[r],
                "consistency": cons[r],
                "boot_p": p_all[r], "q_value": q_all[r],
                "direction": ("above" if mean_z[r] > 0 else "below"),
                "driver_metric": "z_cluster_bootstrap",
            }
            for c in conds:
                row[f"z_{c}"] = point[conds.index(c), r]
                row[f"eligible_{c}"] = bool(elig[conds.index(c), r])
                row[f"p_{c}"] = pc_q[c][0][r]
                row[f"q_{c}"] = pc_q[c][1][r]
            for m in taxon_lists:
                row[f"on_{m}"] = t in member[m]
            rows.append(row)
        print(f"    [driver_bootstrap] {lst}: {int(scored.sum())} scored, "
              f"{int(np.nansum(q_all < 0.20))} at q<0.20")

    return pd.DataFrame(rows)
