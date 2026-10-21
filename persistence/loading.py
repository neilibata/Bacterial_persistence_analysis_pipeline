"""
Data loading for a single reference database.

Loads breadth/depth matrices, genome lengths, metadata, and taxon lists.
Returns a ReferenceData namedtuple that is passed to all downstream modules.
"""

import os
import numpy as np
import pandas as pd
from collections import namedtuple

from . import config
from .common import (normalise_country, get_continent, classify_macro_group,
                     classify_group, is_yes, load_id_list)


def _apply_study_exclusion(meta, exclude_studies=None, cols=None):
    """
    Drop samples belonging to excluded studies.

    Mirrors the logic in filter_diag_meta.py:
      - StudyFolder  and study_name columns: str.startswith() match
      - Author column: exact match

    Parameters
    ----------
    meta            : DataFrame (already indexed by Chosen_match_ID)
    exclude_studies : collection of study-name prefixes to exclude.
                      Defaults to config.EXCLUDE_STUDIES.
    cols            : list of column names to search.
                      Defaults to config.STUDY_EXCLUSION_COLS.

    Returns
    -------
    meta_filtered : DataFrame with excluded rows removed
    n_excluded    : int
    breakdown     : dict  {study_folder_value: count}
    """
    if exclude_studies is None:
        exclude_studies = getattr(config, "EXCLUDE_STUDIES", frozenset())
    if cols is None:
        cols = getattr(config, "STUDY_EXCLUSION_COLS",
                       ["StudyFolder",
                        "study_name [MBIOTEMP_study_name]",
                        "Author"])

    if not exclude_studies:
        return meta, 0, {}

    mask = pd.Series(False, index=meta.index)

    for col in cols:
        if col not in meta.columns:
            continue
        s = meta[col].astype(str).str.strip()
        if col == "Author":
            # Exact match for Author (mirrors filter_diag_meta.py)
            mask |= s.isin(exclude_studies)
        else:
            # Prefix match for StudyFolder / study_name
            for prefix in exclude_studies:
                mask |= s.str.startswith(prefix)

    n_excluded = int(mask.sum())
    breakdown = {}
    if n_excluded and "StudyFolder" in meta.columns:
        breakdown = (meta.loc[mask, "StudyFolder"]
                     .value_counts().to_dict())

    return meta[~mask].copy(), n_excluded, breakdown


def _one_sample_per_biosample(meta, col=None):
    """
    Keep one row per BioSample.

    Among rows sharing a BioSample the kept row is the study's preferred
    library (config.BIOSAMPLE_PREFERRED_LIBRARY), then has the most mapped bases
    (Total_depth_bp), then the larger "Total bases", then is the earlier row.
    Rows with no BioSample are kept as they are; a blank is not a shared value.

    Returns (meta_filtered, [(dropped_id, kept_id, biosample), ...]).
    """
    col = col or getattr(config, "BIOSAMPLE_COL", "BioSample")
    if col not in meta.columns:
        raise KeyError(f"ONE_SAMPLE_PER_BIOSAMPLE needs column {col!r}, "
                       f"absent from the metadata")

    bs = meta[col].astype(str).str.strip()
    has_bs = ~bs.isin(["", "nan", "NaN", "None", "NA"])
    shared = has_bs & bs.duplicated(keep=False)
    if not shared.any():
        return meta, []

    preferred = pd.Series(False, index=meta.index)
    folder = meta.get("StudyFolder", pd.Series("", index=meta.index)).astype(str)
    for prefix, (pcol, pattern) in getattr(config, "BIOSAMPLE_PREFERRED_LIBRARY", {}).items():
        in_study = folder.str.startswith(prefix)
        if in_study.any():
            preferred |= in_study & meta[pcol].astype(str).str.match(pattern)
    mapped = pd.to_numeric(meta.get("Total_depth_bp"), errors="coerce").fillna(-1.0)
    tb = pd.to_numeric(meta.get("Total bases"), errors="coerce").fillna(-1.0)
    rank = pd.DataFrame({"bs": bs, "pref": preferred, "mapped": mapped, "tb": tb,
                         "pos": np.arange(len(meta))}, index=meta.index)[shared]
    rank = rank.sort_values(["bs", "pref", "mapped", "tb", "pos"],
                            ascending=[True, False, False, False, True])
    kept = rank.drop_duplicates("bs", keep="first")
    kept_by_bs = dict(zip(kept["bs"], kept.index))
    drop_ids = rank.index.difference(kept.index)
    dropped = sorted((i, kept_by_bs[bs[i]], bs[i]) for i in drop_ids)
    return meta.drop(index=drop_ids), dropped


# Cohort-construction counts from the most recent load of each reference.
# Populated by load_reference and pushed into the stats registry by
# run_reference, so the per-reference cohort cascade is a reproducible artifact
# rather than something only visible in a run log.
LAST_LOAD_COUNTS: dict = {}


ReferenceData = namedtuple("ReferenceData", [
    "ref_name",       # str: "UHGG", "lev", or "chimp"
    "breadth",        # DataFrame: samples x SGBs (% breadth)
    "depth",          # DataFrame: samples x SGBs (mean depth)
    "abundance",      # DataFrame: samples x SGBs (relative abundance)
    "meta",           # DataFrame: indexed by Chosen_match_ID
    "lengths",        # Series: SGB -> genome length (bp)
    "total_bp",       # Series: sample -> total mapped bp (sum mean_depth x genome_len)
    "total_seq_bp",   # Series: sample -> total sequenced bp (from metadata "Total bases")
    "vanish_set",     # set of VANISH SGB IDs
    "blossum_set",    # set of BloSSUM SGB IDs
    "cov_matrix",     # DataFrame: breadth matrix aligned to meta (for CMH scripts)
    # Technical-replicate fields — populated from Secondary_run_IDs column
    "sec_breadth",    # DataFrame: secondary sample rows x SGBs (% breadth); None if none
    "sec_depth",      # DataFrame: secondary sample rows x SGBs (mean depth); None if none
    "sec_map",        # dict primary_id -> [secondary_id, ...]; empty dict if none
], defaults=[None, None, {}])


def _load_matrix(filepath, tag="matrix"):
    """Load a breadth or depth matrix, keep only _mean columns."""
    print(f"    Loading {tag}: {filepath}")
    df = pd.read_csv(filepath, index_col=0, low_memory=False)
    df.index = df.index.astype(str).str.strip()
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    df = df[mean_cols].copy()
    df.columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]
    df = df.fillna(0.0)
    print(f"      {df.shape[0]:,} samples x {df.shape[1]:,} SGBs")
    return df


def _split_member_ids(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    for sep in [";", "|"]:
        text = text.replace(sep, ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_combined_taxon_crosswalk(metadata_file):
    """
    Map UHGG/Leviatan catalogue IDs onto the representative IDs used as
    columns in the combined matrix.

    Shared combined SGBs default to the Leviatan representative when the
    combined metadata chose a Leviatan representative, so original UHGG list
    IDs need this crosswalk before VANISH/BloSSUM overlaps are evaluated.
    """
    combined = pd.read_excel(metadata_file, sheet_name="Combined")
    required = {"combined_id", "representative"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(
            f"{metadata_file} sheet 'Combined' missing columns: {sorted(missing)}")

    rep_by_combined = dict(zip(combined["combined_id"].astype(str),
                               combined["representative"].astype(str)))
    id_to_rep = {}

    for _, row in combined.iterrows():
        rep = str(row["representative"]).strip()
        if rep and rep.lower() != "nan":
            id_to_rep[rep] = rep
        for col in ("leviatan_members", "uhgg_members"):
            if col in combined.columns:
                for member in _split_member_ids(row[col]):
                    id_to_rep[member] = rep

    for sheet in ("Leviatan", "UHGG"):
        try:
            catalog_df = pd.read_excel(metadata_file, sheet_name=sheet)
        except ValueError:
            continue
        if not {"sgb", "combined_id"} <= set(catalog_df.columns):
            continue
        for _, row in catalog_df.iterrows():
            sgb = str(row["sgb"]).strip()
            combined_id = str(row["combined_id"]).strip()
            rep = rep_by_combined.get(combined_id)
            if sgb and rep and sgb.lower() != "nan":
                id_to_rep[sgb] = rep

    return id_to_rep


def _map_taxon_list_for_reference(ids, ref_cfg, valid_taxa, label):
    if ref_cfg.get("taxon_list_crosswalk") != "combined_sgb_metadata":
        return set(ids)

    metadata_file = ref_cfg.get("combined_sgb_metadata_file",
                                "combined_sgb_metadata.xlsx")
    crosswalk = _load_combined_taxon_crosswalk(metadata_file)
    mapped = []
    unmapped = []
    for taxon_id in ids:
        mapped_id = crosswalk.get(taxon_id, taxon_id)
        if mapped_id in valid_taxa:
            mapped.append(mapped_id)
        else:
            unmapped.append(taxon_id)

    mapped_set = set(mapped)
    print(f"    {label} combined crosswalk: {len(ids):,} input IDs -> "
          f"{len(mapped_set):,} combined representatives")
    if unmapped:
        print(f"      {len(unmapped):,} {label} IDs not present in combined matrix "
              f"(first 5: {unmapped[:5]})")
    return mapped_set


def load_reference(ref_name):
    """
    Load all data for one reference database.

    Parameters
    ----------
    ref_name : str
        Key into config.REFERENCES (e.g. "UHGG", "lev", "chimp").

    Returns
    -------
    ReferenceData namedtuple
    """
    ref_cfg = config.REFERENCES[ref_name]
    out = config.outdir(ref_name)

    print(f"\n{'='*65}")
    print(f"  Loading reference: {ref_name}")
    print(f"  Output directory:  {out}")
    print(f"{'='*65}")

    # ---- Matrices ----
    breadth_df = _load_matrix(ref_cfg["breadth_file"], "breadth")
    depth_df   = _load_matrix(ref_cfg["depth_file"],   "depth")

    # Intersect SGBs
    common_sgbs = sorted(set(breadth_df.columns) & set(depth_df.columns))
    breadth_df = breadth_df[common_sgbs]
    depth_df   = depth_df[common_sgbs]
    print(f"    Common SGBs: {len(common_sgbs):,}")

    # ---- Genome lengths ----
    lf = ref_cfg["lengths_file"]
    print(f"    Genome lengths: {lf}")
    length_df = pd.read_csv(lf, sep=r"\s+", header=None, names=["microbe", "length"])
    length_df["microbe"] = length_df["microbe"].str.strip()
    length_dict = dict(zip(length_df["microbe"], length_df["length"]))

    sgbs_with_len = [s for s in common_sgbs if s in length_dict]
    dropped = len(common_sgbs) - len(sgbs_with_len)
    if dropped:
        print(f"    Dropped {dropped} SGBs with no genome length")
    breadth_df = breadth_df[sgbs_with_len]
    depth_df   = depth_df[sgbs_with_len]
    lengths = pd.Series({s: length_dict[s] for s in sgbs_with_len})

    # ---- Abundance: depth x genome_length → per-sample relative ----
    # total_bp = Σ_i (mean_depth_i x genome_length_i)
    # This estimates total mapped bases directly from the depth matrix and
    # genome-size file — no reliance on metadata columns that may be absent.
    abundance_raw = depth_df.mul(lengths, axis=1)
    total_bp = abundance_raw.sum(axis=1)
    total_bp_safe = total_bp.replace(0, np.nan)
    abundance = abundance_raw.div(total_bp_safe, axis=0).fillna(0.0)

    # ---- Metadata ----
    print(f"    Metadata: {config.META_FILE}")
    meta = pd.read_csv(config.META_FILE, low_memory=False)
    if "Chosen_match_ID" not in meta.columns:
        raise ValueError(f"'Chosen_match_ID' not found in {config.META_FILE}")

    meta["Chosen_match_ID"] = meta["Chosen_match_ID"].astype(str).str.strip()
    _counts = LAST_LOAD_COUNTS.setdefault(ref_name, {})
    _counts["matrix_samples"] = int(breadth_df.shape[0])
    _counts["metadata_rows"] = int(len(meta))
    meta = meta[meta["Chosen_match_ID"].isin(breadth_df.index)].copy()
    _counts["matched_to_matrix"] = int(len(meta))
    # Resolve ingestion duplicates deliberately rather than by row order. The
    # same biological sample can appear twice under inconsistent StudyFolder
    # names (e.g. "CarterM_2023" and "CarterM_2023_TZA_fastq"), once per
    # sequencing run. The matrices carry the LARGER run, so the metadata row
    # describing that run is the one to keep — `keep="first"` retained the
    # smaller in 20 of 21 cases, pairing mapped bases from one run with a
    # "Total bases" from the other.
    _dups = meta["Chosen_match_ID"].duplicated(keep=False)
    if _dups.any():
        _n_ids = int(meta.loc[_dups, "Chosen_match_ID"].nunique())
        _tb = pd.to_numeric(meta.get("Total bases"), errors="coerce")
        _rank = meta.assign(_tb=_tb.fillna(-1.0), _pos=np.arange(len(meta)))
        _keep = (_rank.sort_values(["_tb", "_pos"], ascending=[False, True])
                      .drop_duplicates(subset="Chosen_match_ID", keep="first").index)
        meta = meta.loc[sorted(_keep)]          # keep the original row order
        print(f"    Resolved {_n_ids} duplicated Chosen_match_ID(s) by keeping the "
              f"largest 'Total bases' run")
        _counts["duplicate_ids_resolved"] = _n_ids
    else:
        meta = meta.drop_duplicates(subset="Chosen_match_ID", keep="first")
    meta = meta.set_index("Chosen_match_ID")
    print(f"    Linked samples: {len(meta):,}")

    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = config.canonicalize_lifestyle_series(
            meta["Lifestyle"])

    # ---- Technical replicates: extract secondary rows BEFORE filtering --------
    # breadth_df / depth_df still contain all matrix rows at this point.
    # Secondary IDs are checked against the full matrix index so they are found
    # even though they will never appear as primary (Chosen_match_ID) samples.
    def _split_sec_ids(val):
        if pd.isna(val):
            return []
        return [x.strip() for x in str(val).replace(";", ",").split(",")
                if x.strip()]

    sec_map     = {}
    sec_breadth = None
    sec_depth   = None

    if "Secondary_run_IDs" in meta.columns:
        full_matrix_index = set(breadth_df.index)   # all rows, before any filter
        all_sec_ids = []
        for primary_id in meta.index:
            sids = _split_sec_ids(meta.loc[primary_id, "Secondary_run_IDs"])
            sids = [s for s in sids if s in full_matrix_index]
            if sids:
                sec_map[primary_id] = sids
                all_sec_ids.extend(sids)

        all_sec_ids = list(dict.fromkeys(all_sec_ids))   # deduplicate, keep order
        if all_sec_ids:
            sec_breadth = breadth_df.loc[all_sec_ids]
            sec_depth   = depth_df.loc[all_sec_ids]
            print(f"    Technical replicates: {len(sec_map)} primary samples "
                  f"with {len(all_sec_ids)} secondary run(s)")
        else:
            print("    Technical replicates: Secondary_run_IDs present "
                  "but no secondary rows found in the matrix")
    else:
        print("    Technical replicates: no Secondary_run_IDs column in metadata")

    # Align primary samples
    breadth_df = breadth_df.loc[meta.index]
    depth_df   = depth_df.loc[meta.index]
    abundance  = abundance.loc[meta.index]
    total_bp   = total_bp.loc[meta.index]

    # Annotate
    meta["Country_norm"]   = meta["Country"].apply(normalise_country)

    # Fallback: if Country code didn't resolve to a known continent,
    # try the "Country (full)" column which contains the written-out name.
    if "Country (full)" in meta.columns:
        known = set(getattr(config, "COUNTRY_TO_CONTINENT", {}).keys())
        no_continent = meta["Country_norm"].isna() | ~meta["Country_norm"].isin(known)
        if no_continent.any():
            fallback = meta.loc[no_continent, "Country (full)"].apply(normalise_country)
            meta.loc[no_continent, "Country_norm"] = fallback
            print(f"    Country_norm fallback via 'Country (full)': "
                  f"{int(no_continent.sum())} samples updated")

    meta["Continent"]      = meta["Country_norm"].apply(get_continent)

    # Diagnostic: report any samples still without a continent
    n_no_cont = meta["Continent"].isna().sum()
    if n_no_cont:
        unmapped = (meta.loc[meta["Continent"].isna(),
                              ["Country", "Country (full)" if "Country (full)" in meta.columns else "Country",
                               "Country_norm"]]
                        .value_counts()
                        .head(20))
        print(f"    WARNING: {n_no_cont:,} samples have no continent assignment:")
        for vals, cnt in unmapped.items():
            print(f"      {vals}  →  {cnt} samples")
    else:
        cont_counts = meta["Continent"].value_counts().to_dict()
        print(f"    Continent assignment complete: {cont_counts}")
    meta["MacroGroup"]     = meta.apply(classify_macro_group, axis=1)
    meta["Group_fine"]     = meta.apply(classify_group, axis=1)
    meta["Total_depth_bp"] = total_bp

    # Drop missing Lifestyle, and (by default) any Lifestyle that does not
    # resolve to Industrial / Non-Industrial — "Unclear". Those samples were
    # already invisible to every grouped comparison, since classify_macro_group
    # returns None for them; keeping them only made the whole-cohort
    # descriptives disagree with the grouped ones.
    valid = meta["Lifestyle"].notna()
    if getattr(config, "DROP_UNCLASSIFIED_LIFESTYLE", True):
        unclassified = meta["Lifestyle"].notna() & meta["MacroGroup"].isna()
        if unclassified.any():
            labels = meta.loc[unclassified, "Lifestyle"].value_counts().to_dict()
            print(f"    Dropping {int(unclassified.sum()):,} samples with an "
                  f"unclassifiable Lifestyle: {labels}")
            _counts["dropped_unclassifiable_lifestyle"] = int(unclassified.sum())
            valid &= ~unclassified
    meta       = meta[valid]
    breadth_df = breadth_df.loc[meta.index]
    depth_df   = depth_df.loc[meta.index]
    abundance  = abundance.loc[meta.index]
    total_bp   = total_bp.loc[meta.index]

    # Drop named sample exclusions (amplicon libraries registered as WGS)
    excl = getattr(config, "EXCLUDE_SAMPLES", frozenset())
    if excl:
        # meta was set_index("Chosen_match_ID") above, so the ids are the index
        # here, not a column.
        is_excl = pd.Series(meta.index.astype(str).str.strip().isin(excl),
                            index=meta.index)
        if is_excl.any():
            n_drop = int(is_excl.sum())
            print(f"    Dropping {n_drop} samples on the amplicon exclusion list")
            _counts["dropped_excluded_samples"] = n_drop
            keep = ~is_excl
            meta       = meta[keep]
            breadth_df = breadth_df.loc[meta.index]
            depth_df   = depth_df.loc[meta.index]
            abundance  = abundance.loc[meta.index]
            total_bp   = total_bp.loc[meta.index]

    # One sample per BioSample (config.ONE_SAMPLE_PER_BIOSAMPLE)
    if getattr(config, "ONE_SAMPLE_PER_BIOSAMPLE", False):
        meta, dropped = _one_sample_per_biosample(meta)
        if dropped:
            print(f"    One sample per BioSample: {len(dropped):,} further "
                  f"libraries of an already-kept BioSample removed")
            for run, kept, bs in dropped:
                print(f"      {run} (BioSample {bs}; kept {kept})")
            _counts["dropped_duplicate_biosample"] = len(dropped)
            # A further library of a kept BioSample is the same specimen
            # measured again, which is what a technical replicate is. Leave it
            # out of the cohort but pair it with the kept library, as
            # Secondary_run_IDs would have. (Carter's MoBio/PheChl pairs were
            # the only H-G replicates until each library got its own run.)
            # The primary records which of its secondaries are such libraries,
            # on meta so it survives every rebuild of the ReferenceData.
            lib_ids = [run for run, _, _ in dropped]
            lib_b, lib_d = breadth_df.loc[lib_ids], depth_df.loc[lib_ids]
            sec_breadth = lib_b if sec_breadth is None else pd.concat(
                [sec_breadth, lib_b[~lib_b.index.isin(sec_breadth.index)]])
            sec_depth = lib_d if sec_depth is None else pd.concat(
                [sec_depth, lib_d[~lib_d.index.isin(sec_depth.index)]])
            meta["_sibling_library_ids"] = ""
            for run, kept, _ in dropped:
                if run not in sec_map.get(kept, []):
                    sec_map.setdefault(kept, []).append(run)
                prev = meta.at[kept, "_sibling_library_ids"]
                meta.at[kept, "_sibling_library_ids"] = f"{prev},{run}".lstrip(",")
            print(f"      ... kept as technical replicates of the retained "
                  f"library ({len(dropped):,} run(s))")
            breadth_df = breadth_df.loc[meta.index]
            depth_df   = depth_df.loc[meta.index]
            abundance  = abundance.loc[meta.index]
            total_bp   = total_bp.loc[meta.index]

    # Drop samples below minimum depth
    min_bp = getattr(config, 'MIN_TOTAL_BP', 1e8)
    low_depth = total_bp < min_bp
    if low_depth.any():
        n_drop = low_depth.sum()
        print(f"    Dropping {n_drop} samples with < {min_bp/1e6:.0f} Mb (depth-matrix estimate: sum of mean_depth x genome_length)")
        _counts["dropped_below_min_bp"] = int(n_drop)
        _counts["min_total_bp"] = float(min_bp)
        keep = ~low_depth
        meta       = meta[keep]
        breadth_df = breadth_df.loc[meta.index]
        depth_df   = depth_df.loc[meta.index]
        abundance  = abundance.loc[meta.index]
        total_bp   = total_bp.loc[meta.index]

    # Drop samples that detect too few SGBs to be a community
    min_sgbs = getattr(config, 'MIN_SGBS_DETECTED', 0)
    if min_sgbs:
        thr = getattr(config, 'MIN_SGBS_BREADTH_PCT', 10.0)
        n_det = (breadth_df >= thr).sum(axis=1)
        too_few = n_det < min_sgbs
        if too_few.any():
            n_drop = int(too_few.sum())
            print(f"    Dropping {n_drop} samples detecting < {min_sgbs} SGBs at >= {thr:.0f}% breadth")
            _counts["dropped_below_min_sgbs"] = n_drop
            _counts["min_sgbs_detected"] = int(min_sgbs)
            keep = ~too_few
            meta       = meta[keep]
            breadth_df = breadth_df.loc[meta.index]
            depth_df   = depth_df.loc[meta.index]
            abundance  = abundance.loc[meta.index]
            total_bp   = total_bp.loc[meta.index]

    # Study exclusion (config.EXCLUDE_STUDIES)
    meta, n_excl, breakdown = _apply_study_exclusion(meta)
    if n_excl:
        print(f"    Study exclusion: {n_excl:,} samples removed")
        for sf, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"      {sf}: {cnt:,}")
        breadth_df = breadth_df.loc[meta.index]
        depth_df   = depth_df.loc[meta.index]
        abundance  = abundance.loc[meta.index]
        total_bp   = total_bp.loc[meta.index]
    else:
        n_excl_cfg = len(getattr(config, "EXCLUDE_STUDIES", frozenset()))
        print(f"    Study exclusion: 0 samples removed "
              f"(checked {n_excl_cfg} study prefixes)")

    # Recode Non-Industrial NA Unhealthy -> Healthy
    nonind_na = meta["Lifestyle"].isin(["H-G", "Transitional"]) & meta["Unhealthy"].isna()
    if nonind_na.any():
        print(f"    Re-coding {nonind_na.sum()} Non-Industrial NA → Healthy")
        meta.loc[nonind_na, "Unhealthy"] = "Healthy"

    # No flags are derived from the `Diseased (...)` roll-up columns here.
    # They were briefly dropped from the metadata on the grounds that nothing
    # read them; that was wrong (visualization.build_country_summary and the
    # metadata explorer do) and they are back. They remain unused by the
    # analysis: the per-condition `disease_*` columns and the curated
    # `Unhealthy` label are what every contrast is computed from.

    # Consolidated Obesity flag: any of the 3 obesity columns = yes, OR BMI >= 30
    ob_cols = [c for c in config.OBESITY_COLUMNS if c in meta.columns]
    if ob_cols:
        # --- Diagnostic: show value distribution for each obesity column ---
        for oc in ob_cols:
            vc = meta[oc].astype(str).str.strip().str.lower().value_counts()
            print(f"    Obesity column '{oc}' values: {vc.to_dict()}")
        # --- end diagnostic ---

        meta["Obesity_flag"] = meta[ob_cols].apply(
            lambda row: any(is_yes(v) for v in row), axis=1)
        n_pos = meta["Obesity_flag"].sum()
        print(f"    Obesity flag from {ob_cols}: {n_pos:,} positive")

        # Warn if suspiciously low
        if n_pos < 50:
            print(f"    *** WARNING: only {n_pos} obesity-positive samples detected!")
            print(f"    *** is_yes() matches: 1, 1.0, yes, true")
            print(f"    *** If obesity columns use other values (e.g. 'Obese', 'obese'),")
            print(f"    *** update is_yes() in common.py or config.OBESITY_COLUMNS")
    else:
        meta["Obesity_flag"] = False
        print(f"    WARNING: no obesity columns found")

    # Also flag as obese if BMI >= 30 (regardless of obesity columns)
    if "BMI" in meta.columns:
        bmi_numeric = pd.to_numeric(meta["BMI"], errors="coerce")
        bmi_obese = bmi_numeric >= 30
        n_bmi = int(bmi_obese.sum())
        print(f"    BMI >= 30: {n_bmi:,} samples")
        meta["Obesity_flag"] = meta["Obesity_flag"] | bmi_obese.fillna(False)
        print(f"    Obesity flag (after BMI >= 30 OR): {meta['Obesity_flag'].sum():,} positive")
    else:
        print(f"    WARNING: BMI column not found; obesity flag based on metadata columns only")

    # Also report by lifestyle x health
    ind_h = (meta["Lifestyle"] == "Industrial") & (meta["Unhealthy"] == "Healthy")
    print(f"    Obesity within Industrial Healthy: "
          f"{(ind_h & meta['Obesity_flag']).sum():,} / {ind_h.sum():,}")

    # Overweight flag
    if "Overweight" in meta.columns:
        meta["Overweight_flag"] = meta["Overweight"].apply(is_yes)
        print(f"    Overweight flag: {meta['Overweight_flag'].sum():,} positive")
    else:
        meta["Overweight_flag"] = False

    # Depth column for CMH matching.
    #
    # PRIMARY SOURCE: Total_depth_bp = Σ(mean_depth_i x genome_length_i),
    # computed directly from the depth matrix above.  This is guaranteed to
    # be non-NaN for every sample that survived the min-depth filter, including
    # samples reinstated from secondary runs whose metadata columns (number_reads,
    # Total bases, …) may be absent or NaN.
    #
    # SECONDARY: if a metadata depth column is present and non-NaN for a given
    # sample, we log the coverage but do NOT replace the matrix estimate — the
    # matrix estimate is exact and metadata columns are sometimes mismatched
    # across studies.
    meta["_depth"] = total_bp.reindex(meta.index)   # guaranteed complete

    # Diagnostic: report how many samples would have had NaN under the old logic
    _meta_depth_col = None
    for cand in ["Total_mapped_bp", "number_reads", "Total bases"]:
        if cand in meta.columns:
            _meta_depth_col = cand
            break
    if _meta_depth_col:
        meta_d = pd.to_numeric(meta[_meta_depth_col], errors="coerce")
        n_meta_nan = int(meta_d.isna().sum())
        print(f"    CMH depth column: Total_depth_bp (matrix-derived, guaranteed complete)")
        if n_meta_nan:
            print(f"    Note: metadata '{_meta_depth_col}' had {n_meta_nan:,} NaN — "
                  f"these samples now correctly retain matrix-derived depth")
        else:
            print(f"    Note: metadata '{_meta_depth_col}' complete for all samples "
                  f"(matrix estimate used regardless for consistency)")
    else:
        print(f"    CMH depth column: Total_depth_bp (matrix-derived; "
              f"no metadata depth column found)")

    # Confirm no NaN remain
    n_depth_nan = int(meta["_depth"].isna().sum())
    if n_depth_nan:
        print(f"    WARNING: {n_depth_nan:,} samples still have NaN _depth — "
              f"check depth matrix alignment")
    meta["_log_depth"] = np.log10(meta["_depth"].where(meta["_depth"] > 0))

    # Study column
    meta["_study"] = None
    for cand in ["Study", "StudyFolder", "study_name [MBIOTEMP_study_name]", "BioProject ID"]:
        if cand in meta.columns:
            meta["_study"] = meta[cand]
            print(f"    Study column: {cand}")
            break

    # Unhealthy as string for CMH
    meta["Unhealthy"] = meta["Unhealthy"].astype(str)
    meta["Obesity_flag_bool"] = meta.get("Obesity_flag", pd.Series(False, index=meta.index))

    # ---- Taxon lists ----
    vanish_set  = _map_taxon_list_for_reference(
        load_id_list(config.VANISH_FILE), ref_cfg, set(breadth_df.columns),
        "VANISH")
    blossum_set = _map_taxon_list_for_reference(
        load_id_list(config.BLOSSUM_FILE), ref_cfg, set(breadth_df.columns),
        "BloSSUM")

    print(f"\n    VANISH: {len(vanish_set)},  BloSSUM: {len(blossum_set)}")

    print(f"\n    Final: {len(meta):,} samples x {breadth_df.shape[1]:,} SGBs")
    _counts["pre_reinstatement_samples"] = int(len(meta))
    _counts["sgbs"] = int(breadth_df.shape[1])
    # Lifestyle/Health breakdown printed later in __main__.py after secondary
    # reinstatement, so the counts reflect the final analysis dataset.

    # ── Total sequenced bases from metadata ("Total bases" column) ────────────
    # This is the raw sequencer output (FASTQ bases) before any mapping filter,
    # vs total_bp which is Σ(mean_depth x genome_length) — mapped bases only.
    if "Total bases" in meta.columns:
        total_seq_bp = pd.to_numeric(
            meta["Total bases"], errors="coerce"
        ).reindex(meta.index)
        n_missing = total_seq_bp.isna().sum()
        if n_missing:
            print(f"    Note: {n_missing:,} samples lack 'Total bases' metadata")
    else:
        print("    Warning: 'Total bases' column not found in metadata — "
              "total_seq_bp will be all-NaN")
        total_seq_bp = pd.Series(np.nan, index=meta.index, dtype=float)

    return ReferenceData(
        ref_name=ref_name,
        breadth=breadth_df,
        depth=depth_df,
        abundance=abundance,
        meta=meta,
        lengths=lengths,
        total_bp=total_bp,
        total_seq_bp=total_seq_bp,
        vanish_set=vanish_set,
        blossum_set=blossum_set,
        cov_matrix=breadth_df,   # alias for CMH scripts
        sec_breadth=sec_breadth,
        sec_depth=sec_depth,
        sec_map=sec_map,
    )


def load_reference_analytic(ref_name, stats=None, **kwargs):
    """Load a reference **and** apply secondary-run reinstatement.

    This is the cohort every published number is computed on. ``load_reference``
    alone returns the pre-reinstatement cohort, which is *not* what the analyses
    use: ``run_reference`` immediately follows it with
    ``reinstate_distinct_secondaries``, promoting compositionally distinct
    secondary runs to independent samples and changing N.

    Standalone figure scripts that called ``load_reference`` directly silently
    computed on a different cohort from the rest of the paper — for the combined
    reference, 204 samples short. Use this function anywhere a script needs the
    analytic cohort, and keep ``load_reference`` for genuinely pre-reinstatement
    work such as the initial depth audit.
    """
    from .reinstate_secondary import reinstate_distinct_secondaries
    from .stats_report import StatsReport

    data = load_reference(ref_name, **kwargs)
    return reinstate_distinct_secondaries(data, stats or StatsReport(ref_name))
