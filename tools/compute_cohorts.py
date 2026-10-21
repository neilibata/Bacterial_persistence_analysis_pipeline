#!/usr/bin/env python3
"""Compute every analytic cohort -- samples (n_s) and hosts (n_h) -- without
running any analysis.

Runs only the pipeline's cohort step, by calling the pipeline's own functions in
the pipeline's order, so the cohorts are identical by construction:

  lifestyle cohorts (per reference)
      load_reference -> run_depth_audit (zero-depth removal) ->
      reinstate_distinct_secondaries                    (as run_reference does)
  health cohorts (combined reference, Section 3 / Figure 5D-E / Table S10)
      section3_health._health_comparisons -> health_cmh.build_group_masks ->
      health_cmh._get_study_strata with config.CMH_MIN_STRATUM

n_h counts distinct Subject_ID. Samples whose host could not be resolved
(Subject_ID_source = "unresolved") carry a per-sample Subject_ID and so count as
one host each; n_h is therefore an upper bound on the true number of people.

Health contrasts are computed twice: as analysed (config.DEDUP_SUBJECTS, one
sample per host) and with de-duplication switched off, so the table shows how
many samples the one-per-host rule collapses in each condition.

Nothing is written under results/ except the depth audit's throwaway directory,
which is removed. Output goes to --out:

    python tools/compute_cohorts.py --out cohort_tables
    python tools/compute_cohorts.py --out cohort_tables --refs combined
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict

sys.path.insert(0, os.getcwd())

import pandas as pd  # noqa: E402

_TMP_AUDIT = "depth_audit_cohort_tables_tmp"
HEALTH_REF = "combined"          # the reference Section 3 is reported on
FIG5_MIN_STRATA = 3              # figures/figure5.MIN_STRATA_TO_PLOT


def _study_names(meta):
    """Study id -> "Author_Year" label (the 83 study datasets of Note 1)."""
    lab = (meta["Author"].astype(str).str.strip() + "_" +
           pd.to_numeric(meta["Year"], errors="coerce").astype("Int64").astype(str))
    return lab.groupby(meta["_study"]).agg(lambda s: s.value_counts().index[0]).to_dict()


def _hosts(meta, ids):
    return int(meta.loc[list(ids), "Subject_ID"].astype(str).nunique())


def _lifestyle_rows(ref, meta):
    groups = OrderedDict([
        ("All",            meta.index),
        ("Industrial",     meta.index[meta["Lifestyle"].eq("Industrial")]),
        ("Non-industrial", meta.index[meta["Lifestyle"].isin(["H-G", "Transitional"])]),
        ("Hunter-gatherer", meta.index[meta["Lifestyle"].eq("H-G")]),
        ("Agro-pastoralist", meta.index[meta["Lifestyle"].eq("Transitional")]),
    ])
    rows = []
    for g, ids in groups.items():
        sub = meta.loc[ids]
        rows.append(dict(reference=ref, group=g, n_samples=len(ids),
                         n_hosts=_hosts(meta, ids),
                         n_studies=int(sub["_study"].nunique()),
                         n_countries=int(sub["Country"].nunique()),
                         unresolved_host_samples=int(
                             sub["Subject_ID_source"].astype(str).eq("unresolved").sum())))
    return rows


def _fig5ac_rows(ref, meta):
    """Figure 5A-C groups exactly as section3_health._run_health_richness_gmw
    builds them (no host de-duplication: richness per sample)."""
    u = meta["Unhealthy"].astype(str).str.strip().str.lower()
    ind = meta["Lifestyle"].eq("Industrial")
    groups = OrderedDict([
        ("Non-industrial", meta["Lifestyle"].isin(["H-G", "Transitional"])),
        ("Industrial healthy", ind & u.isin({"healthy", "no", "false", "0"})),
        ("Industrial unhealthy", ind & u.isin({"unhealthy", "yes", "true", "1"})),
        ("Industrial, health unknown",
         ind & ~u.isin({"healthy", "no", "false", "0", "unhealthy", "yes", "true", "1"})),
    ])
    return [dict(reference=ref, group=g, n_samples=int(m.sum()),
                 n_hosts=_hosts(meta, meta.index[m]),
                 n_studies=int(meta.loc[m, "_study"].nunique()))
            for g, m in groups.items()]


def _per_study_rows(ref, meta, names):
    rows = []
    for (sid, life), sub in meta.groupby(["_study", "Lifestyle"]):
        rows.append(dict(reference=ref, study_id=sid, study=names.get(sid, str(sid)),
                         lifestyle=life, n_samples=len(sub),
                         n_hosts=int(sub["Subject_ID"].astype(str).nunique()),
                         countries=", ".join(sorted(sub["Country"].astype(str).unique()))))
    return rows


def _health(meta, bdf_index, names):
    from persistence import config
    from persistence.health_cmh import build_group_masks, _get_study_strata
    from persistence.section3_health import _health_comparisons

    bdf = pd.DataFrame(index=bdf_index)        # _get_study_strata reads only the index
    display = {"Fatty Liver": config.MASLD_DISPLAY}
    cond_rows, study_rows = [], []
    saved = config.DEDUP_SUBJECTS
    try:
        for label, (ctype, dcols) in _health_comparisons().items():
            res = {}
            for dedup in (True, False):
                config.DEDUP_SUBJECTS = dedup
                g1, g2 = build_group_masks(meta, ctype, dcols)
                strata = _get_study_strata(meta, g1, g2, bdf, config.CMH_MIN_STRATUM)
                res[dedup] = (int(g1.sum()), int(g2.sum()), strata)
            n1_all, n2_all, strata = res[True]
            _, _, strata_nd = res[False]
            ctrl = [i for _, a, _ in strata for i in a]
            case = [i for _, _, b in strata for i in b]
            ctrl_nd = [i for _, a, _ in strata_nd for i in a]
            case_nd = [i for _, _, b in strata_nd for i in b]
            studies = [names.get(s, str(s)) for s, _, _ in strata]
            cond_rows.append(OrderedDict(
                condition=display.get(label, label), comparison_type=ctype,
                disease_columns=", ".join(dcols or []),
                in_figure5=len(strata) >= FIG5_MIN_STRATA,
                n_studies=len(strata),
                controls_samples=len(ctrl), cases_samples=len(case),
                total_samples=len(ctrl) + len(case),
                controls_hosts=_hosts(meta, ctrl) if ctrl else 0,
                cases_hosts=_hosts(meta, case) if case else 0,
                total_hosts=_hosts(meta, ctrl + case) if ctrl or case else 0,
                industrial_controls_before_strata=n1_all,
                industrial_cases_before_strata=n2_all,
                n_studies_without_host_dedup=len(strata_nd),
                controls_samples_without_host_dedup=len(ctrl_nd),
                cases_samples_without_host_dedup=len(case_nd),
                total_samples_without_host_dedup=len(ctrl_nd) + len(case_nd),
                total_hosts_without_host_dedup=(
                    _hosts(meta, ctrl_nd + case_nd) if ctrl_nd or case_nd else 0),
                studies="; ".join(sorted(studies)),
            ))
            nd = {s: (a, b) for s, a, b in strata_nd}
            for s, a, b in strata:
                a_nd, b_nd = nd.get(s, ([], []))
                study_rows.append(dict(
                    condition=display.get(label, label), study_id=s,
                    study=names.get(s, str(s)),
                    controls=len(a), cases=len(b),
                    controls_hosts=_hosts(meta, a), cases_hosts=_hosts(meta, b),
                    controls_without_host_dedup=len(a_nd),
                    cases_without_host_dedup=len(b_nd)))
    finally:
        config.DEDUP_SUBJECTS = saved
    return cond_rows, study_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="cohort_tables")
    ap.add_argument("--refs", nargs="+", default=["UHGG", "lev", "combined", "chimp"])
    args = ap.parse_args()

    from persistence import config
    from persistence.depth_audit import run_depth_audit
    from persistence.loading import load_reference, LAST_LOAD_COUNTS
    from persistence.reinstate_secondary import reinstate_distinct_secondaries
    from persistence.stats_report import StatsReport

    os.makedirs(args.out, exist_ok=True)
    life, fig5ac, per_study, cascade = [], [], [], {}
    health = health_study = None

    for ref in args.refs:
        stats = StatsReport(ref, out_root=os.path.join("results", "_scratch"))
        print(f"\n##### {ref} #####")
        data = load_reference(ref)
        data = run_depth_audit(data, stats, outdir_override=_TMP_AUDIT)
        data = reinstate_distinct_secondaries(data, stats)
        meta = data.meta
        print(f"  cohort: {len(meta):,} samples, "
              f"{meta['Subject_ID'].astype(str).nunique():,} hosts")

        cascade[ref] = dict(LAST_LOAD_COUNTS.get(ref, {}), final=len(meta))
        names = _study_names(meta)
        life += _lifestyle_rows(ref, meta)
        fig5ac += _fig5ac_rows(ref, meta)
        per_study += _per_study_rows(ref, meta, names)
        pd.DataFrame({"Chosen_match_ID": meta.index.astype(str),
                      "Lifestyle": meta["Lifestyle"].astype(str).values,
                      "Subject_ID": meta["Subject_ID"].astype(str).values,
                      "study": meta["_study"].map(names).values,
                      "Unhealthy": meta["Unhealthy"].astype(str).values}
                     ).to_csv(os.path.join(args.out, f"cohort_{ref}.csv"), index=False)

        if ref == HEALTH_REF:
            health, health_study = _health(meta, data.breadth.index, names)

        shutil.rmtree(os.path.join("results", ref, _TMP_AUDIT), ignore_errors=True)
        del data, meta

    shutil.rmtree(os.path.join("results", "_scratch"), ignore_errors=True)

    pd.DataFrame(life).to_csv(os.path.join(args.out, "lifestyle_cohorts.csv"), index=False)
    pd.DataFrame(fig5ac).to_csv(os.path.join(args.out, "figure5AC_cohorts.csv"), index=False)
    pd.DataFrame(per_study).to_csv(os.path.join(args.out, "per_study.csv"), index=False)
    json.dump(cascade, open(os.path.join(args.out, "cascade.json"), "w"), indent=1)
    if health is not None:
        pd.DataFrame(health).to_csv(os.path.join(args.out, "health_cohorts.csv"), index=False)
        pd.DataFrame(health_study).to_csv(
            os.path.join(args.out, "health_cohorts_by_study.csv"), index=False)
    print(f"\nWrote cohort tables to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
