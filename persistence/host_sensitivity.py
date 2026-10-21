"""Sensitivity of the lifestyle results to repeat sampling of the same host.

The lifestyle analyses keep every sample, and 3,451 of the 18,075 combined-
cohort samples are repeat samples of a host already represented (Lloyd-Price
2019 alone is 1,125 samples from 104 people). Two results could depend on that,
and this stage re-derives both so the manuscript quotes numbers the run itself
produced (Supplementary Text #5 and #7C, Fig. S14):

1. The Figure 2 headline -- the GMW-corrected Industrial / Non-Industrial
   richness ratio at >=10% breadth -- computed
     * on all samples, bootstrapping samples (as published),
     * on all samples, bootstrapping hosts (a host's samples kept together),
     * on one sample per host, bootstrapping samples.
2. VANISH+/BloSSUM+ and VANISH-H+/BloSSUM-H+ membership: rebuilt with one
   sample per host, and its stability under sample versus host bootstrap.

"One sample per host" is the host's representative -- the sample the health
analyses keep (table_s1a._representatives). GMW weights are refitted on
whichever cohort is analysed, as the pipeline would fit them. A bootstrap
resample is expressed as weight x times-drawn, so no rows are copied.

Outputs, results/<ref>/host_sensitivity/:
    richness_host_sensitivity.csv
    list_host_sensitivity_summary.csv
    list_host_sensitivity_membership.csv   (every taxon that is on either list)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config

TOP_N = 100
N_BOOT_RICHNESS = 1000
N_BOOT_LISTS = 200


def _weights(ld, ld_other):
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights
    return _geomean_weights(ld, _BinnedKDE(ld), _BinnedKDE(ld_other))


def _counts(rng, n, hosts_inv=None):
    """Times each row is drawn: rows directly, or whole hosts (hosts_inv maps
    row -> host index)."""
    if hosts_inv is None:
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    k = int(hosts_inv.max()) + 1
    return np.bincount(rng.integers(0, k, k), minlength=k)[hosts_inv].astype(float)


def _ratio(r1, w1, r2, w2):
    return (np.dot(w1, r1) / w1.sum()) / (np.dot(w2, r2) / w2.sum())


def _rank(P1, w1, P2, w2, taxa):
    """Weighted prevalence difference (g1 - g2) sorted as
    differential.run_differential_gmw sorts it; never-detected taxa dropped."""
    a1, a2 = w1 @ P1, w2 @ P2
    seen = (a1 + a2) > 0
    d = pd.Series(a1 / w1.sum() - a2 / w2.sum(), index=taxa)[seen]
    return d.sort_values(ascending=False, kind="mergesort")


def run_host_sensitivity(data, stats, outdir=None, seed=1,
                         n_boot_richness=N_BOOT_RICHNESS, n_boot_lists=N_BOOT_LISTS):
    from .table_s1a import _representatives

    ref = data.ref_name
    outdir = outdir or os.path.join("results", ref, "host_sensitivity")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    meta = data.meta.loc[data.meta.index.isin(data.breadth.index)]
    ids = meta.index
    taxa = np.asarray(data.breadth.columns)
    P = (data.breadth.loc[ids].values >= config.BREADTH_THRESHOLD).astype(np.float32)
    rich = P.sum(axis=1).astype(float)
    ld = np.log10(pd.to_numeric(data.total_bp.reindex(ids), errors="coerce")
                    .clip(lower=1.0)).values
    subj = meta["Subject_ID"].astype(str).values if "Subject_ID" in meta else ids.values
    rep = _representatives(meta, data.total_bp)
    is_rep = rep.eq(rep.index).reindex(ids).values

    life = meta["Lifestyle"].astype(str).values
    ni = np.isin(life, ["H-G", "Transitional"])
    ind = life == "Industrial"
    ind_h = ind & meta["Unhealthy"].astype(str).eq("Healthy").values
    if ni.sum() < 20 or ind.sum() < 20:
        print(f"  [host sensitivity] {ref}: too few samples per lifestyle; skipped")
        return None

    print(f"\n  [host sensitivity] {ref}: {len(ids):,} samples, "
          f"{len(set(subj)):,} hosts, {int(is_rep.sum()):,} representatives")

    def _group(mask):
        idx = np.where(mask)[0]
        _, inv = np.unique(subj[idx], return_inverse=True)
        return idx, inv

    # ── 1. Figure 2 richness ratio ──────────────────────────────────────────
    rows = []
    for label, m1, m2, boot_hosts in (
            ("all samples, sample bootstrap (as published)", ind, ni, False),
            ("all samples, host bootstrap", ind, ni, True),
            ("one sample per host, sample bootstrap", ind & is_rep, ni & is_rep, False)):
        (i1, h1), (i2, h2) = _group(m1), _group(m2)
        w1, w2 = _weights(ld[i1], ld[i2]), _weights(ld[i2], ld[i1])
        est = _ratio(rich[i1], w1, rich[i2], w2)
        boot = [_ratio(rich[i1], w1 * _counts(rng, len(i1), h1 if boot_hosts else None),
                       rich[i2], w2 * _counts(rng, len(i2), h2 if boot_hosts else None))
                for _ in range(n_boot_richness)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(analysis=label, ratio=round(est, 4), ci_lo=round(lo, 4),
                         ci_hi=round(hi, 4), n_industrial=len(i1), n_non_industrial=len(i2),
                         hosts_industrial=int(h1.max()) + 1,
                         hosts_non_industrial=int(h2.max()) + 1, n_boot=n_boot_richness))
        print(f"    richness {label}: {est:.3f} [{lo:.3f}, {hi:.3f}]")
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(outdir, "richness_host_sensitivity.csv"), index=False)
    for tag, r in zip(("all_sample_boot", "all_host_boot", "one_per_host"), rows):
        stats.add(f"host_sens_richness_{tag}", "s1", "supplement",
                  f"GMW Ind/NI richness ratio (>=10% breadth), {r['analysis']}",
                  f"ratio={r['ratio']:.3f}, 95% CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
                  n1=r["n_industrial"], n2=r["n_non_industrial"])

    # ── 2. VANISH+/BloSSUM+ and H+ membership ───────────────────────────────
    s2 = os.path.join("results", ref, "section2_differential", "main")
    published = {}
    for name, fn in (("VANISH+", "vanish_plus_ids.txt"), ("BloSSUM+", "blossum_plus_ids.txt"),
                     ("VANISH-H+", "vanishH_ids.txt"), ("BloSSUM-H+", "blossumH_ids.txt")):
        p = os.path.join(s2, fn)
        if os.path.exists(p):
            published[name] = {l.strip() for l in open(p) if l.strip()}

    summ, memb = [], []
    for g1, vname, bname in ((ind, "VANISH+", "BloSSUM+"), (ind_h, "VANISH-H+", "BloSSUM-H+")):
        (i1, h1), (i2, h2) = _group(g1), _group(ni)
        w1, w2 = _weights(ld[i1], ld[i2]), _weights(ld[i2], ld[i1])
        P1, P2 = P[i1], P[i2]
        d_all = _rank(P1, w1, P2, w2, taxa)
        (j1, _), (j2, _) = _group(g1 & is_rep), _group(ni & is_rep)
        d_host = _rank(P[j1], _weights(ld[j1], ld[j2]), P[j2], _weights(ld[j2], ld[j1]), taxa)
        lists = {bname: (list(d_all.index[:TOP_N]), list(d_host.index[:TOP_N])),
                 vname: (list(d_all.index[-TOP_N:]), list(d_host.index[-TOP_N:]))}

        freq = {m: {L: pd.Series(0.0, index=taxa) for L in lists} for m in ("sample", "host")}
        for mode in ("sample", "host"):
            for _ in range(n_boot_lists):
                d = _rank(P1, w1 * _counts(rng, len(i1), h1 if mode == "host" else None),
                          P2, w2 * _counts(rng, len(i2), h2 if mode == "host" else None), taxa)
                freq[mode][bname].loc[d.index[:TOP_N]] += 1
                freq[mode][vname].loc[d.index[-TOP_N:]] += 1
        common = d_all.index.intersection(d_host.index)
        rho = float(d_all[common].rank().corr(d_host[common].rank()))

        for L, (full, host) in lists.items():
            fs, fh = freq["sample"][L] / n_boot_lists, freq["host"][L] / n_boot_lists
            pub = published.get(L)
            summ.append(dict(
                list=L, overlap_one_per_host=len(set(full) & set(host)),
                spearman_all_taxa=round(rho, 4),
                n_ge95_sample_boot=int((fs[full] >= 0.95).sum()),
                n_ge95_host_boot=int((fh[full] >= 0.95).sum()),
                expected_overlap_sample_boot=round(float(fs[full].sum()), 1),
                expected_overlap_host_boot=round(float(fh[full].sum()), 1),
                matches_section2_list=(None if pub is None else set(full) == pub),
                n_g1_all=len(i1), n_ni_all=len(i2), n_g1_one_per_host=len(j1),
                n_ni_one_per_host=len(j2), n_boot=n_boot_lists))
            pos_all = {t: k for k, t in enumerate(d_all.index)}
            pos_host = {t: k for k, t in enumerate(d_host.index)}
            vanish = L == vname
            for t in sorted(set(full) | set(host)):
                memb.append(dict(
                    list=L, taxon=t, in_all_samples=t in full, in_one_per_host=t in host,
                    rank_all=(len(d_all) - pos_all[t]) if vanish else pos_all[t] + 1,
                    rank_one_per_host=((len(d_host) - pos_host[t]) if vanish
                                       else pos_host[t] + 1) if t in pos_host else None,
                    prevalence_diff_all=float(d_all[t]),
                    freq_sample_boot=float(fs[t]), freq_host_boot=float(fh[t])))
            s = summ[-1]
            print(f"    {L}: {s['overlap_one_per_host']}/{TOP_N} kept with one sample per "
                  f"host; expected overlap {s['expected_overlap_sample_boot']} (samples) vs "
                  f"{s['expected_overlap_host_boot']} (hosts); "
                  f"matches section 2: {s['matches_section2_list']}")
            stats.add(f"host_sens_list_{L}", "s2", "supplement",
                      f"{L} membership kept with one sample per host",
                      f"{s['overlap_one_per_host']}/{TOP_N}; expected overlap "
                      f"{s['expected_overlap_sample_boot']} (sample boot) vs "
                      f"{s['expected_overlap_host_boot']} (host boot); rho={rho:.3f}")

    pd.DataFrame(summ).to_csv(os.path.join(outdir, "list_host_sensitivity_summary.csv"),
                              index=False)
    pd.DataFrame(memb).to_csv(os.path.join(outdir, "list_host_sensitivity_membership.csv"),
                              index=False)
    print(f"    Saved: {outdir}/")
    return rdf, pd.DataFrame(summ)
