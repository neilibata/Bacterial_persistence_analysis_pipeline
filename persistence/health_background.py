"""The prevalence background for health associations: curve, z, clade contrasts.

Why this module exists
----------------------
A taxon cannot lose more prevalence than it has. A taxon found in 2% of
controls can fall by at most 2 points; one found in 60% can fall by 60. The
VANISH and BloSSUM lists are not a random sample of the catalogue in this
respect -- the H lists sit near 40% control prevalence against a catalogue
median under 3% -- so "the list taxa are depleted in disease" is partly a
statement about how common they are. Every list-level claim therefore has to be
read against taxa of the *same* commonness.

Up to 2026-10-05 Section 3 did that by importance-weighting the background:
reweighting all non-list taxa so that their weighted prevalence distribution
matched the list's, then comparing distributions of case-control shifts. Three
defects made that comparison wrong, and all three are documented in
`iw_background_review_20261005/`:

1. The weights were realised by rounding to 10 x list-size whole copies of each
   background taxon, and a taxon rounding to zero copies dropped out. For the
   VANISH lists that discarded 41-51% of the total weight, nearly all of it on
   near-zero-prevalence taxa, leaving a tested background ~70% *more* prevalent
   than the list it was supposed to match.
2. A floor at the 5th percentile and a 10x clip on the weights meant BlossumH+
   could never be matched at all: weighted background prevalence 0.20-0.26
   against 0.45 for the list.
3. The shift being compared was `prev_diff` from
   `health_cmh.geom_mean_weighted_cmh_per_taxon`, which pools a/n across
   studies. See `stratified_shifts_per_taxon` for why that is not a
   case-control contrast.

The method here replaces all of it. There are no weights, no bins and no
rounding:

* x = log10(within-study control prevalence), y = case - control shift in
  points, both from `health_cmh.stratified_shifts_per_taxon`.
* Fit mu(x) and sigma(x) on the non-list taxa by local-linear regression with a
  *fixed* Gaussian bandwidth (0.12 log10 units, about +/-30% in relative
  prevalence). Fixed, not nearest-neighbour: a variable bandwidth would reach
  halfway across the catalogue to estimate the sparse high-prevalence end and
  silently borrow from taxa an order of magnitude rarer.
* Each list taxon gets z = (shift - mu(x)) / sigma(x), with sigma floored at
  that taxon's own sampling SE so a taxon cannot look extreme merely because it
  is noisy.
* The list statistic is the mean z over supported taxa, with a bootstrap CI
  that resamples background taxa *and refits the curve*, so the background's
  own sampling noise is inside the interval. The energy-distance test it
  replaces treated the weighted background as known and was anticonservative.

A taxon is `supported` only when at least `HEALTH_BG_MIN_SUPPORT` background
taxa lie within `HEALTH_BG_SUPPORT_WINDOW` of it in log10 prevalence.
Above ~85% prevalence the combined catalogue holds just 5 taxa, 4 of them on
the lists, so their expected shift would be an extrapolation; `clade_contrasts`
drops that range outright and the figures mark it.

Sign conventions, which are easy to get wrong
---------------------------------------------
`rd_mh` and `prev_diff` are control - case: POSITIVE = depleted in disease.
`shift` here is case - control in percentage points (the negated, x100 form),
because that is what the figures plot and what "above the curve" has to mean:
z > 0 = less depleted / more enriched in cases than comparable taxa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ── tuning, all from config so a sweep can move them ──────────────────────────

def _cfg(name, default):
    return getattr(config, name, default)


GRID = np.linspace(-4.0, 0.0, 400)      # log10 prevalence, the fit's own grid


# ── the curve ─────────────────────────────────────────────────────────────────

def local_linear(x, y, x_eval, h=None):
    """Local-linear Gaussian-kernel regression with a fixed bandwidth `h`.

    Chunked over `x_eval` so the kernel matrix stays small: the catalogue has
    ~5,000 taxa and the grid 400 points.
    """
    if h is None:
        h = _cfg("HEALTH_BG_BANDWIDTH", 0.12)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_eval = np.asarray(x_eval, float)
    out = np.empty(len(x_eval))
    for k in range(0, len(x_eval), 256):
        e = x_eval[k:k + 256][:, None]
        w = np.exp(-0.5 * ((x[None, :] - e) / h) ** 2)
        sw = w.sum(1)
        sx = (w * x).sum(1)
        sy = (w * y).sum(1)
        sxx = (w * x * x).sum(1)
        sxy = (w * x * y).sum(1)
        den = sw * sxx - sx ** 2
        ok = den > 1e-12
        b = np.where(ok, (sw * sxy - sx * sy) / np.where(ok, den, 1.0), 0.0)
        a = (sy - b * sx) / np.maximum(sw, 1e-300)
        out[k:k + 256] = a + b * e[:, 0]
    return out


def curves(x, y, x_eval, h=None):
    """(mu, sigma) at `x_eval`, fitted on (x, y) and interpolated off GRID.

    sigma comes from the local mean absolute residual, scaled by sqrt(pi/2) --
    the Gaussian relation between mean |deviation| and SD. A local *variance*
    fit would be dominated by the handful of extreme taxa in every window.
    """
    mu_grid = local_linear(x, y, GRID, h)
    mu_at_x = np.interp(x, GRID, mu_grid)
    sd_grid = np.sqrt(np.pi / 2) * np.maximum(
        local_linear(x, np.abs(np.asarray(y, float) - mu_at_x), GRID, h), 0.0)
    return (np.interp(x_eval, GRID, mu_grid),
            np.maximum(np.interp(x_eval, GRID, sd_grid), 1e-6))


def shift_frame(taxa_df, prev_floor=None):
    """Turn `stratified_shifts_per_taxon` output into the curve's coordinates.

    Returns a frame indexed by taxon with
        p       within-study control prevalence (0-1)
        x       log10(p), floored
        shift   case - control, percentage points
        se      SE of `shift`, percentage points
        p_case  within-study case prevalence, and x_case its log10
    """
    if prev_floor is None:
        prev_floor = _cfg("HEALTH_BG_PREV_FLOOR", 1e-4)
    df = taxa_df.set_index("taxon") if "taxon" in taxa_df.columns else taxa_df
    ok = np.isfinite(df["rd_mh"].values) & np.isfinite(df["p_ctrl_mh"].values)
    df = df[ok]
    out = pd.DataFrame(
        {"p": df["p_ctrl_mh"].values,
         "shift": -df["rd_mh"].values * 100.0,
         "se": df["rd_mh_se"].values * 100.0},
        index=df.index)
    out["x"] = np.log10(np.maximum(out["p"].values, prev_floor))
    # Prevalence in the case arm, on the same within-study weighting. The
    # curve does not use it -- the comparison is made at control prevalence --
    # but S25's coverage row counts taxa per prevalence bin in both arms, and
    # that is the honest way to show how much of the catalogue the comparison
    # can reach at each end.
    if "p_case_mh" in df.columns:
        out["p_case"] = df["p_case_mh"].values
        out["x_case"] = np.log10(np.maximum(out["p_case"].values, prev_floor))
    return out


# ── list vs background ────────────────────────────────────────────────────────

def background_z(frame, list_members, n_boot=None, seed=None,
                 support_window=None, min_support=None, h=None,
                 exclude=None):
    """Standardised residual from the background curve for one list.

    Parameters
    ----------
    frame : DataFrame from `shift_frame`
    list_members : iterable of taxon ids defining the list; every other taxon
        in `frame` is background and is what the curve is fitted on.
    exclude : taxon ids removed from the BACKGROUND without being scored.
        The six published lists are heavily nested -- 98 of VanishH+'s
        100 members and 83 of BlossumH+'s also sit in another list -- so
        excluding only the list under test fitted the curve partly on
        near-duplicates of it, pulling mu toward the list and shrinking
        |z|. The shrinkage was 3-5x stronger for the BloSSUM family than
        the VANISH family, so the two rows of Figure 5E sat on different
        footings. Pass the union of all published lists.

    Returns
    -------
    (per_taxon, summary, curve)
        per_taxon : one row per list taxon (z, expected, bg_sd, supported, ...)
        summary   : dict -- mean z over supported taxa, bootstrap CI and p,
                    the 80%-power minimum detectable mean z, n_eff
        curve     : DataFrame on GRID (x, mu, sd) for drawing, fitted on the
                    background taxa of *this* list
    """
    n_boot = int(_cfg("HEALTH_BG_N_BOOT", 2000) if n_boot is None else n_boot)
    seed = _cfg("HEALTH_BG_SEED", 17) if seed is None else seed
    support_window = (_cfg("HEALTH_BG_SUPPORT_WINDOW", 0.25)
                      if support_window is None else support_window)
    min_support = int(_cfg("HEALTH_BG_MIN_SUPPORT", 15)
                      if min_support is None else min_support)
    rng = np.random.default_rng(seed)

    is_list = frame.index.isin(set(list_members))
    # Background = neither this list nor any sibling list.
    drop = set(exclude or ()) - set(list_members)
    is_bg = (~is_list) & (~frame.index.isin(drop))
    bg, li = frame[is_bg], frame[is_list]
    if len(li) == 0 or len(bg) < min_support:
        return None, None, None

    bx, by = bg["x"].values, bg["shift"].values
    lx, ly, lse = li["x"].values, li["shift"].values, li["se"].values

    mu, sig = curves(bx, by, lx, h)
    sig = np.maximum(sig, lse)                     # never beat the taxon's own noise
    z = (ly - mu) / sig

    # How much of the spread at this prevalence is real taxon-to-taxon
    # variation rather than sampling noise: 0 means the shift is noise.
    tau2 = np.maximum(sig ** 2 - lse ** 2, 0.0)
    rel = tau2 / sig ** 2

    support = np.array([int(np.sum(np.abs(bx - xi) <= support_window))
                        for xi in lx])
    # The high-prevalence limit applies here too, not only in
    # clade_contrasts. Above it the support window reaches only taxa BELOW
    # the focal prevalence, so mu is a one-sided extrapolation of a local
    # line; S26 and Figure 5E already shade that band "no comparable taxa"
    # and draw those markers hollow, while background_z was still scoring
    # them and counting them in mean z and n_supported. Now the statistic
    # says what the figure says.
    # Taxa at or above the high-prevalence limit sit where the support
    # window reaches only taxa BELOW them, so mu there is a one-sided
    # extrapolation of a local line. clade_contrasts drops them outright.
    # background_z does NOT by default, because excluding them moves two
    # headline numbers materially -- Unhealthy BlossumH+ +1.20 -> +0.94 and
    # IBD BlossumH+ +1.16 -> +0.99, on 4 and 4 taxa respectively -- and that
    # is an author's decision about what the manuscript reports, not a bug
    # fix. Set HEALTH_BG_HIGHPREV_EXCLUDE = True to apply it.
    ok = support >= min_support
    if bool(_cfg("HEALTH_BG_HIGHPREV_EXCLUDE", False)):
        ok = ok & (li["p"].values
                   < float(_cfg("HEALTH_BG_HIGHPREV_LIMIT", 0.85)))

    # The other end of the same problem. A taxon detected in no sample of
    # either arm has shift = 0 because nothing was observed, not because
    # nothing moved, and `x` floors every one of them onto log10(prev_floor).
    # They therefore all receive the SAME z -- whatever (0 - mu)/sigma happens
    # to be at the floor -- and each counts once in mean z, so a list carrying
    # several of them is pulled toward an artefact of where the floor sits.
    # In the 9 Oct run nine such taxa sat in VANISH, all with z = -0.475205.
    # Unlike the high-prevalence band this is not a weak measurement, it is
    # the absence of one: the comparison cannot reach a taxon it never saw.
    # `_run_taxon_drivers` already skips them, so excluding them here is what
    # keeps the list statistic and the driver figures describing one dataset.
    _p_case = (li["p_case"].values if "p_case" in li.columns
               else np.where(ly == 0.0, 0.0, 1.0))   # shift==0 at p==0 => p_case==0
    _never = (li["p"].values <= 0.0) & (_p_case <= 0.0)
    if bool(_cfg("HEALTH_BG_DROP_NEVER_DETECTED", True)):
        ok = ok & (~_never)

    per_taxon = pd.DataFrame({
        "taxon": li.index, "prev": li["p"].values, "shift": ly, "se": lse,
        "expected": mu, "bg_sd": sig, "z": z, "reliability": rel,
        "n_bg_near": support, "supported": ok, "never_detected": _never,
    })

    curve = pd.DataFrame({"x": GRID})
    cm, cs = curves(bx, by, GRID, h)
    curve["mu"], curve["sd"] = cm, cs

    if not ok.any():
        return per_taxon, None, curve

    # Bootstrap: resample background taxa (refitting mu and sigma) and list
    # taxa. Refitting is the point -- it is what puts the background's own
    # sampling error into the interval.
    lx_s, ly_s, lse_s = lx[ok], ly[ok], lse[ok]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ib = rng.integers(0, len(bx), len(bx))
        il = rng.integers(0, len(lx_s), len(lx_s))
        m2, s2 = curves(bx[ib], by[ib], lx_s[il], h)
        s2 = np.maximum(s2, lse_s[il])
        boots[b] = np.mean((ly_s[il] - m2) / s2)

    mean_z = float(z[ok].mean())
    se_boot = float(boots.std(ddof=1))
    p_two = (2 * min((boots <= 0).mean(), (boots >= 0).mean())
             if mean_z != 0 else 1.0)
    summary = {
        "n_list": int(len(li)),
        "n_supported": int(ok.sum()),
        "n_unsupported": int((~ok).sum()),
        "n_eff_informative": float(rel[ok].sum()),
        "median_prev": float(np.median(li["p"].values)),
        "mean_z": mean_z,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "boot_p": float(max(p_two, 1.0 / n_boot)),
        "boot_se": se_boot,
        "mde_80": 2.8 * se_boot,
        "mean_resid_pp": float((ly - mu)[ok].mean()),
    }
    return per_taxon, summary, curve


def matched_background(frame, list_members, window=None):
    """A prevalence-matched background with no weights to get wrong.

    Each list taxon brings its own comparison set: the background taxa within
    `window` log10 units of its control prevalence (the same bandwidth the
    curve uses). A background taxon's weight is then

        match_weight_j = sum_i 1[j in M_i] / |M_i|

    which makes any weighted statistic over the background exactly the mean,
    across list taxa, of that statistic within each taxon's own matched set.
    It replaces `_compute_iw_background`: there is no KDE ratio, no 5th-
    percentile floor, no 10x clip, and nothing is rounded to whole copies, so
    no background taxon is silently dropped and the matched distribution really
    does sit where the list sits.

    Returns a frame indexed by background taxon with
        match_count   how many list taxa it is matched to (integer)
        match_weight  the weight above, normalised to mean 1 over matched taxa
    and only the taxa with at least one match.
    """
    if window is None:
        window = _cfg("HEALTH_BG_BANDWIDTH", 0.12)
    is_list = frame.index.isin(set(list_members))
    bg, li = frame[~is_list], frame[is_list]
    bx = bg["x"].values
    count = np.zeros(len(bx))
    weight = np.zeros(len(bx))
    for xi in li["x"].values:
        m = np.abs(bx - xi) <= window
        n = int(m.sum())
        if not n:
            continue
        count[m] += 1
        weight[m] += 1.0 / n
    keep = count > 0
    out = pd.DataFrame({"match_count": count[keep].astype(int),
                        "match_weight": weight[keep]},
                       index=bg.index[keep])
    if len(out) and out["match_weight"].mean() > 0:
        out["match_weight"] /= out["match_weight"].mean()
    return out


def z_against_background(frame, list_members, h=None):
    """z for EVERY taxon in `frame`, with the curve fitted on the non-list taxa.

    `background_z` returns the list taxa; the driver figures (S26-S29) and the
    clade analysis also need background taxa on the same scale.
    """
    is_list = frame.index.isin(set(list_members))
    bx, by = frame["x"].values[~is_list], frame["shift"].values[~is_list]
    mu, sig = curves(bx, by, frame["x"].values, h)
    sig = np.maximum(sig, frame["se"].values)
    return pd.Series((frame["shift"].values - mu) / sig, index=frame.index)


# ── clades (Figure S32) ───────────────────────────────────────────────────────

def _bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def clade_contrasts(frame, lineage, list_members, ranks=("genus", "family"),
                    min_taxa=None, n_perm=None, seed=None,
                    high_prev_limit=None, h=None):
    """Do list taxa differ from their own relatives?

    For each condition the caller loops over, and for each clade with at least
    `min_taxa` members: refit the background curve with the clade *excluded*
    (otherwise a large clade is partly its own control), z every member against
    it, and score the clade by mean z. The null is 5,000 placebo clades whose
    members are drawn from non-clade taxa within +/-bandwidth of each member's
    control prevalence, so a clade cannot score merely by being common.

    Then, within each clade holding at least 3 list taxa and 3 others, the mean
    z of list members minus the rest, with a p-value from permuting the list
    labels inside the clade. That contrast is the point of the figure: it holds
    taxonomy fixed, so it cannot be explained by the lists being full of one
    family.

    Taxa at or above `high_prev_limit` control prevalence are dropped: the
    catalogue has too few taxa there to say what "comparable" means.

    Returns (clades, within): one row per clade, one per clade x list contrast.
    """
    min_taxa = int(_cfg("HEALTH_BG_CLADE_MIN_TAXA", 5)
                   if min_taxa is None else min_taxa)
    n_perm = int(_cfg("HEALTH_BG_CLADE_NPERM", 5000)
                 if n_perm is None else n_perm)
    seed = _cfg("HEALTH_BG_CLADE_SEED", 41) if seed is None else seed
    high_prev_limit = (_cfg("HEALTH_BG_HIGHPREV_LIMIT", 0.85)
                       if high_prev_limit is None else high_prev_limit)
    window = _cfg("HEALTH_BG_BANDWIDTH", 0.12)
    rng = np.random.default_rng(seed)

    df = frame[frame["p"].values < high_prev_limit]
    lin = lineage.reindex(df.index)
    x, y, se = df["x"].values, df["shift"].values, df["se"].values

    clade_rows, within_rows = [], []
    for rank in ranks:
        if rank not in lin.columns:
            continue
        labels = lin[rank].fillna("").values
        counts = pd.Series(labels).value_counts()
        for clade in counts[counts >= min_taxa].index:
            if not clade:
                continue
            inc = labels == clade
            xb, yb, seb = x[~inc], y[~inc], se[~inc]

            mu_b, sd_b = curves(xb, yb, xb, h)
            z_bg = (yb - mu_b) / np.maximum(sd_b, seb)
            mu_c, sd_c = curves(xb, yb, x[inc], h)
            z_cl = (y[inc] - mu_c) / np.maximum(sd_c, se[inc])

            # placebo clades: prevalence-matched draws from the background
            order = np.argsort(xb)
            xs = xb[order]
            lo = np.searchsorted(xs, x[inc] - window)
            hi = np.searchsorted(xs, x[inc] + window, side="right")
            pick = order[lo + (rng.random((n_perm, int(inc.sum())))
                               * np.maximum(hi - lo, 1)).astype(int)]
            null = z_bg[pick].mean(1)
            obs = float(z_cl.mean())
            # Clipped at 1: doubling the smaller tail exceeds 1 whenever both
            # tails do (a clade whose observed mean sits exactly on the null,
            # which happens for a degenerate contrast with diff ~ 0), and a
            # p of 2.0 was reaching the deposited contrast table.
            p = min(1.0, max(2 * min((null >= obs).mean(),
                                     (null <= obs).mean()),
                             1.0 / n_perm))

            ids = df.index[inc]
            row = {"rank": rank, "clade": clade, "n_taxa": int(inc.sum()),
                   "median_prev": float(np.median(df["p"].values[inc])),
                   "mean_z": obs, "null_sd": float(null.std()),
                   "frac_above2": float((z_cl > 2).mean()),
                   "frac_below2": float((z_cl < -2).mean()), "p": p}
            for name, members in list_members.items():
                row[f"n_{name}"] = int(sum(t in members for t in ids))
            clade_rows.append(row)

            for name, members in list_members.items():
                m = np.array([t in members for t in ids])
                if m.sum() < 3 or (~m).sum() < 3:
                    continue
                diff = float(z_cl[m].mean() - z_cl[~m].mean())
                perm = np.empty(n_perm)
                for k in range(n_perm):
                    pm = rng.permutation(m)
                    perm[k] = z_cl[pm].mean() - z_cl[~pm].mean()
                within_rows.append({
                    "rank": rank, "clade": clade, "list": name,
                    "n_list": int(m.sum()), "n_other": int((~m).sum()),
                    "mean_z_list": float(z_cl[m].mean()),
                    "mean_z_other": float(z_cl[~m].mean()), "diff": diff,
                    # Clipped at 1, as above.
                    "p": min(1.0, max(2 * min((perm >= diff).mean(),
                                              (perm <= diff).mean()),
                                      1.0 / n_perm)),
                    "z_list": z_cl[m], "z_other": z_cl[~m],
                    "taxa_list": list(ids[m]),
                })

    clades = pd.DataFrame(clade_rows)
    within = pd.DataFrame(within_rows)
    if not clades.empty:
        clades["q"] = _bh(clades["p"].values)
    if not within.empty:
        within["q"] = within.groupby(["rank"])["p"].transform(
            lambda s: pd.Series(_bh(s.values), index=s.index))
    return clades, within


def lineage_frame(tax_map, taxa):
    """{sgb: GTDB lineage string} -> DataFrame[genus, family, order] for `taxa`.

    Uses whatever `_load_taxonomy_map_s3` gives Section 3, so S30 names clades
    the same way S26-S29 name taxa.
    """
    rows = {}
    for t in taxa:
        parts = [p.strip() for p in str(tax_map.get(t, "") or "").split(";")]
        got = {}
        for p in parts:
            for pre, key in (("g__", "genus"), ("f__", "family"),
                             ("o__", "order")):
                if p.startswith(pre) and len(p) > len(pre):
                    got[key] = p[len(pre):]
        rows[t] = got
    return pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=["genus", "family", "order"])
