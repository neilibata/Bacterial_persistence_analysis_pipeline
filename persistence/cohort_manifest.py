"""The analytic cohort, published per reference as an explicit sample list.

The post-exclusion cohort is a property of the *matrices*, not of Table M1. The
>=30 Mb floor is applied to mapped bases (``sum(mean_depth x genome_length)``,
`loading.py`) and the companion floor to the number of SGBs a sample detects at
the breadth threshold, so neither cut is derivable from a metadata column. Both
are reference-dependent: a sample maps more bases against a larger catalogue, so
the analytic N differs per reference (UHGG < lev < combined).

`figures/make_metadata_summary.py` had no way to honour that. It re-approximated
the floor against the metadata "Total bases" column -- *sequenced* rather than
mapped bases, ~100x larger -- which dropped 7 samples where the real cut drops a
couple of hundred, and it applied no min-SGBs cut at all. Figure 1 therefore
reported a cohort larger than any analysis ran on.

Writing the surviving index here, immediately after reinstatement (the point at
which cohort N becomes canonical for the paper), lets those scripts intersect
against the real cohort instead of guessing at it.

`Lifestyle` is written as the internal key (`H-G` / `Transitional` /
`Industrial`), not the Table M1 display label -- this is an internal artifact,
and readers never see it. Relabel at draw time via `config.display_lifestyle`.
"""

from __future__ import annotations

import os

import pandas as pd

from . import config

FILENAME = "cohort_samples.csv"


def manifest_path(ref_name: str, root: str = ".") -> str:
    """Path to a reference's cohort manifest. Does not create anything."""
    return os.path.join(root, "results", ref_name, FILENAME)


def write_cohort_manifest(data, stats=None) -> str:
    """Write the post-exclusion, post-reinstatement cohort for `data`.

    Called unconditionally from `run_reference` so the manifest exists for every
    run, including `--only` runs that skip the depth audits.
    """
    meta = data.meta
    mapped = pd.to_numeric(data.total_bp.reindex(meta.index), errors="coerce")
    out = pd.DataFrame({
        "Chosen_match_ID": meta.index.astype(str),
        "Lifestyle": (meta["Lifestyle"].astype(str).values
                      if "Lifestyle" in meta.columns else ""),
        "mapped_bp": mapped.values,
    })

    path = os.path.join(config.outdir(data.ref_name), FILENAME)
    out.to_csv(path, index=False)

    counts = out["Lifestyle"].value_counts().to_dict()
    print(f"    Cohort manifest: {len(out):,} samples -> {path}")
    print(f"      {counts}")
    if stats is not None:
        stats.add("cohort_manifest_n", "qc", "supplement",
                  f"Analytic cohort written to {FILENAME} ({data.ref_name})",
                  len(out))
        # Hosts (n_h) beside samples (n_s), for Table S2. Distinct Subject_ID;
        # a sample with no archived host identifier carries its own, so it
        # counts as one host and n_h is an upper bound on the number of people.
        if "Subject_ID" in meta.columns:
            subj = meta["Subject_ID"].astype(str)
            stats.add("cohort_n_hosts", "qc", "supplement",
                      f"Analytic cohort hosts, distinct Subject_ID ({data.ref_name})",
                      int(subj.nunique()))
            for life, ids in subj.groupby(meta["Lifestyle"].astype(str)):
                stats.add(f"cohort_n_hosts_{life}", "qc", "supplement",
                          f"Analytic cohort hosts, {life} ({data.ref_name})",
                          int(ids.nunique()))
    return path


def read_cohort_manifest(ref_name: str, root: str = ".") -> pd.DataFrame | None:
    """Read a reference's cohort manifest, or None when it has not been written."""
    path = manifest_path(ref_name, root)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"Chosen_match_ID": str})
    if "Chosen_match_ID" not in df.columns:
        return None
    return df
