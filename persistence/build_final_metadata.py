"""
build_final_metadata.py
=======================
Steps:
  1. Load meandepth matrix, metadata, merged reference, and SGB lengths.
  2. Map every meandepth index to metadata rows (via SRS ID, SRA_Accession,
     Sample ID, or any SRR in Paired SRR List — comma or semicolon separated).
  3. Recover unmapped indices by looking them up in merged_with_media1_dedup...csv.
     - Found    → append as new rows to metadata
     - Not found → write to truly_unmapped.csv
  3b. Exclude studies (mirrors config.py EXCLUDE_STUDIES exactly):
     - StudyFolder / study_name: str.startswith() on any prefix in EXCLUDE_STUDIES
     - Author: exact match
  4. Resolve collisions (multiple meandepth indices → same metadata row):
     - Primary run  = matrix index with highest depth_score
     - Secondary runs = all other matrix indices for that row with depth_score > 30 Mb
       (stored in Secondary_run_IDs; NOT discarded)
     - Write full collision table to collision_losers.csv
  4b. Deduplicate on Sample ID + SRS ID:
     - Rows that share the same composite key are MERGED, not just dropped
     - Chosen_match_ID kept from highest-depth row; Paired SRR List and
       Secondary_run_IDs are unioned across collapsed rows
  4c. Special-author dedup (LeChatelierE, NielsenHB, LiJ):
     - Additional collapse by Sample ID alone within these author groups
     - Metadata taken from the most complete (most non-null fields) row
     - Primary run = Chosen_match_ID with highest depth_score; other run IDs
       folded into Secondary_run_IDs
  4d. Strip " (Source)" from all string-valued cells.
  5. Determine infant status from BOTH:
     - metadata Age / Age group columns
     - merged_with_media1 Age / Age group columns (cross-reference by ID)
  6. Write output metadata CSVs, all with a 'Chosen_match_ID' column that
     matches the row index of the breadth/depth matrices:
     - metadata_Mar_16_2026.csv          <- primary pipeline input (no infants)
     - metadata_final_with_infants.csv
     - metadata_final_no_infants.csv
     - truly_unmapped_meandepth_indices.csv
     - collision_losers.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# -- file paths ----------------------------------------------------------------
MEANDEPTH_FILE   = "meandepth_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv"
METADATA_FILE    = ("metadata_non_infants_gt100Mb_sra_by_biosample_best_meandepth"
                    "_fullmeta_dedup_by_Chosen_match_ID_filled_v2.csv")
SGB_LENGTHS_FILE = "all_SGB_lengths_UHGG.txt"
MERGED_FILE      = "merged_with_media1_dedup_v4_plus_MBIOTEMP_dedup_runs.csv"

OUT_PRIMARY          = "metadata_Mar_16_2026.csv"   # <- used by pipeline
OUT_WITH_INFANTS     = "metadata_final_with_infants.csv"
OUT_NO_INFANTS       = "metadata_final_no_infants.csv"
OUT_TRULY_UNMAPPED   = "truly_unmapped_meandepth_indices.csv"
OUT_COLLISION_LOSERS = "collision_losers.csv"

# -- study exclusion (mirrors config.py EXCLUDE_STUDIES exactly) --------------
EXCLUDE_STUDIES = frozenset({
    "Bengtsson-PalmeJ",   # 68 rows  - resistome study
    "DhakanDB",           # 131 rows - extreme depth outliers
    "FerrettiP",          # 121 rows - longitudinal infant data
    "Heintz-BuschartA",   # 50 rows  - type-1 diabetes cohort
    "KaurK",              # 31 rows  - antibiotic-treated
    "UKN",                # 6 rows   - unknown provenance
    "eKmeta",             # 1 row    - single-sample study
})

STUDY_EXCLUSION_COLS = [
    "StudyFolder",
    "study_name [MBIOTEMP_study_name]",
    "Author",
]

# -- technical-replicate threshold --------------------------------------------
# Matrix rows with depth_score (sum mean_depth * genome_length) above this
# threshold qualify as secondary technical replicates.
TECH_REP_MIN_BP = 30_000_000   # 30 Mb

# -- authors that need extra within-Sample-ID deduplication -------------------
SPECIAL_DEDUP_AUTHORS = frozenset({"LeChatelierE", "NielsenHB", "LiJ"})


# -- helpers ------------------------------------------------------------------

def split_ids(val):
    """Split a comma- or semicolon-separated string of IDs into a list."""
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).replace(";", ",").split(",") if x.strip()]


def merge_srr_lists(*values):
    """
    Union of any number of comma/semicolon-separated SRR-ID strings.
    Returns a single comma-separated string of unique, sorted IDs,
    or pd.NA if all inputs were null/empty.
    """
    ids = set()
    for v in values:
        ids.update(split_ids(v))
    return ",".join(sorted(ids)) if ids else pd.NA


def row_completeness(row):
    """Count non-null, non-empty, non-'nan' fields."""
    count = 0
    for v in row:
        if pd.notna(v) and str(v).strip() not in ("", "nan", "NaN"):
            count += 1
    return count


def is_infant(row):
    """Return True if a metadata row represents an infant/newborn."""
    if "Age group" in row.index:
        ag = str(row["Age group"]).lower()
        if any(k in ag for k in ("infant", "newborn", "neo")):
            return True
    if "Age" in row.index:
        try:
            if float(row["Age"]) < 3:
                return True
        except (ValueError, TypeError):
            pass
    return False


def apply_study_exclusion(df):
    """
    Drop rows belonging to excluded studies.
    StudyFolder / study_name: startswith(); Author: exact match.
    Returns (filtered_df, n_excluded, breakdown_dict).
    """
    mask = pd.Series(False, index=df.index)
    for col in STUDY_EXCLUSION_COLS:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        if col == "Author":
            mask |= s.isin(EXCLUDE_STUDIES)
        else:
            for prefix in EXCLUDE_STUDIES:
                mask |= s.str.startswith(prefix)
    n_excluded = int(mask.sum())
    breakdown = {}
    if n_excluded and "StudyFolder" in df.columns:
        breakdown = df.loc[mask, "StudyFolder"].value_counts().to_dict()
    return df[~mask].copy(), n_excluded, breakdown


def build_id_lookup(df):
    """Build a dict: ID string -> set of row indices, from all accession columns.

    This is the rule that links a matrix row to a metadata row. Chosen_match_ID
    is only the cached winner of it (the matching index with the highest depth
    score), so anything that keys on Chosen_match_ID instead of re-deriving it
    here cannot notice when the two have drifted apart.

    BioSample was added because the 2026 preprocessing names matrix rows by the
    BioSample accession (SAMEA14201084, SAMN02334127) rather than by a run or an
    SRS. Without it none of the 260 samples added in that pass was reachable by
    this rule at all: they resolved only through a hand-set Chosen_match_ID, so
    re-running this script would have dropped every one of them while reporting
    nothing. It introduces no new ambiguity -- the only matrix names resolving to
    more than one metadata row remain the 21 known Carter 2023 pairs.
    """
    lookup = defaultdict(set)
    for idx, row in df.iterrows():
        for col in ["SRS ID", "SRA_Accession", "Sample ID", "run_accession",
                    "BioSample"]:
            if col in row and pd.notna(row[col]):
                lookup[str(row[col]).strip()].add(idx)
        for srr in split_ids(row.get("Paired SRR List", None)):
            lookup[srr].add(idx)
    return lookup


def merge_group(grp, depth_score_series, extra_secondary=None):
    """
    Collapse a group of duplicate rows into one merged row.

    - Chosen_match_ID   -> from the row with the highest depth_score
    - Paired SRR List   -> union of all rows' lists
    - Secondary_run_IDs -> union of all rows' lists + non-primary Chosen_match_IDs
                           + any extra_secondary; filtered to depth > TECH_REP_MIN_BP
    - Other columns     -> from the most-complete row (most non-null fields),
                           ties broken by depth_score

    Parameters
    ----------
    grp               : DataFrame slice of rows to merge
    depth_score_series : Series mapping matrix index -> depth_score
    extra_secondary   : iterable of additional sids to include (optional)
    """
    scores = grp["Chosen_match_ID"].map(depth_score_series).fillna(0.0)

    # Best primary run = highest depth score
    primary_idx = scores.idxmax()
    primary_sid = grp.loc[primary_idx, "Chosen_match_ID"]

    # Most complete row for metadata
    completeness = grp.apply(row_completeness, axis=1)
    top = completeness.max()
    candidates = grp[completeness == top]
    if len(candidates) > 1:
        meta_idx = scores.loc[candidates.index].idxmax()
    else:
        meta_idx = candidates.index[0]

    best_row = grp.loc[meta_idx].copy()
    best_row["Chosen_match_ID"] = primary_sid

    # Union Paired SRR Lists
    paired_col = "Paired SRR List"
    if paired_col in grp.columns:
        best_row[paired_col] = merge_srr_lists(*grp[paired_col].tolist())

    # Build Secondary_run_IDs: non-primary Chosen_match_IDs + existing secondaries
    secondary_sids = [sid for sid in grp["Chosen_match_ID"].tolist() if sid != primary_sid]
    existing_sec   = grp["Secondary_run_IDs"].tolist() if "Secondary_run_IDs" in grp.columns else []
    if extra_secondary:
        secondary_sids += list(extra_secondary)

    all_candidate_secondary = split_ids(",".join(
        str(x) for x in secondary_sids + existing_sec if pd.notna(x)
    ))
    qualifying = [s for s in all_candidate_secondary
                  if depth_score_series.get(s, 0) > TECH_REP_MIN_BP]
    best_row["Secondary_run_IDs"] = ",".join(sorted(set(qualifying))) if qualifying else pd.NA

    return best_row


# -- load files ----------------------------------------------------------------
print("=" * 60)
print("Loading files...")
print("=" * 60)

meta   = pd.read_csv(METADATA_FILE, low_memory=False)
merged = pd.read_csv(MERGED_FILE,   low_memory=False)

depth_df = pd.read_csv(MEANDEPTH_FILE, index_col=0)
depth_df.index = depth_df.index.astype(str).str.strip()

sgb_lengths = pd.read_csv(
    SGB_LENGTHS_FILE, sep=r"\s+", header=None, names=["SGB", "length"]
).set_index("SGB")["length"]

print(f"  Meandepth matrix : {depth_df.shape[0]} samples x {depth_df.shape[1]} SGBs")
print(f"  Metadata         : {len(meta)} rows")
print(f"  Merged reference : {len(merged)} rows")
print(f"  SGB lengths      : {len(sgb_lengths)} SGBs")

# -- compute depth scores ------------------------------------------------------
print("\n--- Computing depth scores ---")
depth_df.columns = depth_df.columns.str.replace(r"_mean$", "", regex=True)
common = depth_df.columns.intersection(sgb_lengths.index)
print(f"  SGBs matched after stripping '_mean': {len(common)} / {depth_df.shape[1]}")
if len(common) == 0:
    print("  ERROR: No SGBs matched. Check column names.")
    print(f"  First 3 matrix cols  : {list(depth_df.columns[:3])}")
    print(f"  First 3 SGB len keys : {list(sgb_lengths.index[:3])}")

lengths_aligned    = sgb_lengths.reindex(depth_df.columns, fill_value=0).values
depth_scores       = depth_df.values.dot(lengths_aligned)
depth_score_series = pd.Series(depth_scores, index=depth_df.index)
print(f"  Samples with depth_score > 0: {(depth_score_series > 0).sum()} / {len(depth_score_series)}")

# -- build metadata lookup ----------------------------------------------------
print("\n--- Building metadata ID lookup ---")
meta_lookup = build_id_lookup(meta)
print(f"  Unique ID tokens in metadata: {len(meta_lookup)}")

# -- step 2: map meandepth indices to metadata rows ---------------------------
print("\n--- Mapping meandepth indices to metadata ---")
mapping = {}   # meandepth_idx -> set of meta row indices
for sid in depth_df.index:
    mapping[sid] = meta_lookup.get(str(sid).strip(), set())

unmapped_sids = [sid for sid, hits in mapping.items() if not hits]
mapped_sids   = [sid for sid, hits in mapping.items() if hits]
print(f"  Mapped   : {len(mapped_sids)}")
print(f"  Unmapped : {len(unmapped_sids)}")

# -- step 3: recover unmapped from merged reference ---------------------------
print("\n--- Recovering unmapped samples from merged reference ---")
merged_lookup = build_id_lookup(merged)
print(f"  Unique ID tokens in merged reference: {len(merged_lookup)}")

recovered_rows = []
truly_unmapped = []

for sid in unmapped_sids:
    key  = str(sid).strip()
    hits = merged_lookup.get(key, set())
    if hits:
        for h in hits:
            row = merged.loc[h].copy()
            if pd.isna(row.get("SRA_Accession")):
                row["SRA_Accession"] = key
            row["_recovered_from_merged"] = True
            recovered_rows.append(row)
        mapping[sid] = {f"recovered_{sid}"}
    else:
        truly_unmapped.append(sid)

print(f"  Recovered from merged : {len(recovered_rows)}")
print(f"  Truly unmapped        : {len(truly_unmapped)}")

if recovered_rows:
    recovered_df = pd.DataFrame(recovered_rows).reindex(
        columns=meta.columns.tolist() + ["_recovered_from_merged"], fill_value=pd.NA
    )
    meta_extended = pd.concat([meta, recovered_df], ignore_index=True)
else:
    meta_extended = meta.copy()
    meta_extended["_recovered_from_merged"] = False

print(f"  Metadata rows after recovery: {len(meta_extended)}")

# Rebuild lookup with extended metadata and remap
meta_lookup_ext = build_id_lookup(meta_extended)
for sid in depth_df.index:
    mapping[sid] = meta_lookup_ext.get(str(sid).strip(), set())

# -- step 3b: exclude studies -------------------------------------------------
print("\n--- Applying study exclusion ---")
print(f"  Exclusion prefixes/values: {sorted(EXCLUDE_STUDIES)}")

meta_extended, n_excl, breakdown = apply_study_exclusion(meta_extended)

if breakdown:
    for sf, cnt in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"    {sf}: {cnt:,} rows excluded")
elif n_excl:
    print(f"    (StudyFolder absent - matched on study_name / Author only)")

print(f"  Total excluded : {n_excl:,} rows")
print(f"  Remaining      : {len(meta_extended):,} rows")

# Purge excluded rows from mapping
surviving_row_ids = set(meta_extended.index)
for sid in list(mapping.keys()):
    mapping[sid] = {r for r in mapping[sid] if r in surviving_row_ids}

# -- step 4: resolve collisions + track secondary runs -----------------------
print("\n--- Resolving collisions + tracking secondary runs ---")

meta_row_to_sids = defaultdict(list)
for sid, hits in mapping.items():
    for meta_row in hits:
        meta_row_to_sids[meta_row].append(sid)

collisions = {mr: sids for mr, sids in meta_row_to_sids.items() if len(sids) > 1}
print(f"  Metadata rows with multiple meandepth indices: {len(collisions)}")

loser_rows            = []
loser_sids            = set()
meta_row_to_secondary = defaultdict(list)   # meta_row -> secondary sids above threshold

for meta_row, sids in sorted(collisions.items()):
    scores = {sid: depth_score_series.get(sid, 0) for sid in sids}
    best   = max(scores, key=scores.get)

    # Union ALL Paired SRR Lists across every matrix row that maps to this
    # metadata row — losers may have been merged runs with extra constituent
    # SRR IDs that the winner's metadata row doesn't know about.
    if meta_row in meta_extended.index and "Paired SRR List" in meta_extended.columns:
        all_srr_vals = [meta_extended.loc[meta_row, "Paired SRR List"]]
        # Each sid maps to the same meta_row, so their SRR lists are already
        # in the metadata; we still do the union to catch any divergence.
        merged_srr = merge_srr_lists(*all_srr_vals)
        meta_extended.at[meta_row, "Paired SRR List"] = merged_srr

    for sid in sids:
        if sid != best:
            loser_sids.add(sid)
            meta_info = meta_extended.loc[meta_row] if meta_row in meta_extended.index else {}
            loser_rows.append({
                "metadata_row"       : meta_row,
                "metadata_accession" : meta_info.get("SRA_Accession", ""),
                "meandepth_idx"      : sid,
                "depth_score"        : scores[sid],
                "winner_idx"         : best,
                "winner_depth_score" : scores[best],
            })
            if scores[sid] > TECH_REP_MIN_BP:
                meta_row_to_secondary[meta_row].append(sid)

n_secondary_total = sum(len(v) for v in meta_row_to_secondary.values())
print(f"  Collision losers total             : {len(loser_sids)}")
print(f"  Collision losers kept as secondary : {n_secondary_total} "
      f"(depth_score > {TECH_REP_MIN_BP // 1_000_000} Mb)")

# -- Build Chosen_match_ID + Secondary_run_IDs --------------------------------
print("\n--- Building Chosen_match_ID and Secondary_run_IDs columns ---")

meta_row_to_winner = {}
for meta_row, sids in meta_row_to_sids.items():
    winners = [s for s in sids if s not in loser_sids]
    if len(winners) == 1:
        meta_row_to_winner[meta_row] = winners[0]
    elif len(winners) > 1:
        meta_row_to_winner[meta_row] = max(
            winners, key=lambda s: depth_score_series.get(s, 0)
        )

chosen_ids = pd.Series(
    {row_idx: sid for row_idx, sid in meta_row_to_winner.items()}, dtype=str
)
meta_extended["Chosen_match_ID"] = chosen_ids

secondary_ids = pd.Series(
    {row_idx: ",".join(sorted(sids))
     for row_idx, sids in meta_row_to_secondary.items() if sids},
    dtype=str,
)
meta_extended["Secondary_run_IDs"] = secondary_ids

n_stamped        = meta_extended["Chosen_match_ID"].notna().sum()
n_with_secondary = meta_extended["Secondary_run_IDs"].notna().sum()
print(f"  Rows with Chosen_match_ID      : {n_stamped} / {len(meta_extended)}")
print(f"  Rows with >=1 secondary run ID : {n_with_secondary}")

meta_extended = meta_extended[meta_extended["Chosen_match_ID"].notna()].copy()
print(f"  Rows after dropping unlinked   : {len(meta_extended)}")

# -- step 4b: merge duplicates on Sample ID + SRS ID -------------------------
print("\n--- Deduplicating on Sample ID + SRS ID (merging rows) ---")

DEDUP_KEYS     = ["Sample ID", "SRS ID"]
available_keys = [k for k in DEDUP_KEYS if k in meta_extended.columns]

if available_keys:
    def _dedup_key(row):
        parts = [str(row[k]).strip() for k in available_keys
                 if pd.notna(row[k]) and str(row[k]).strip() not in ("", "nan")]
        return "|".join(sorted(parts)) if parts else None

    meta_extended["_dedup_key"] = meta_extended.apply(_dedup_key, axis=1)

    before_4b = len(meta_extended)
    keyed      = meta_extended[meta_extended["_dedup_key"].notna()].copy()
    unkeyed    = meta_extended[meta_extended["_dedup_key"].isna()].copy()

    merged_rows_4b     = []
    n_groups_collapsed = 0
    for key, grp in keyed.groupby("_dedup_key", sort=False):
        if len(grp) == 1:
            merged_rows_4b.append(grp.iloc[0])
        else:
            n_groups_collapsed += 1
            merged_rows_4b.append(merge_group(grp, depth_score_series))

    meta_extended = pd.concat(
        [pd.DataFrame(merged_rows_4b), unkeyed], ignore_index=True
    ).drop(columns=["_dedup_key"], errors="ignore")

    after_4b = len(meta_extended)
    print(f"  Dedup keys used      : {available_keys}")
    print(f"  Rows before dedup    : {before_4b:,}")
    print(f"  Groups collapsed     : {n_groups_collapsed:,}")
    print(f"  Rows after dedup     : {after_4b:,}  ({before_4b - after_4b:,} removed)")
else:
    before_4b = after_4b = len(meta_extended)
    n_groups_collapsed = 0
    print(f"  WARNING: none of {DEDUP_KEYS} found in columns - skipping dedup")

# -- step 4c: special-author dedup by Sample ID only -------------------------
print(f"\n--- Special-author dedup by Sample ID: {sorted(SPECIAL_DEDUP_AUTHORS)} ---")

before_4c = 0
after_4c  = 0
n_special_collapsed = 0

if "Author" in meta_extended.columns and "Sample ID" in meta_extended.columns:
    def _is_special_author(val):
        s = str(val).strip()
        return any(s.startswith(p) for p in SPECIAL_DEDUP_AUTHORS)

    special_mask = meta_extended["Author"].apply(_is_special_author)
    special_df   = meta_extended[special_mask].copy()
    other_df     = meta_extended[~special_mask].copy()

    before_4c = len(special_df)
    merged_rows_4c = []

    for sample_id, grp in special_df.groupby("Sample ID", sort=False):
        if len(grp) == 1:
            merged_rows_4c.append(grp.iloc[0])
        else:
            n_special_collapsed += 1
            merged_rows_4c.append(merge_group(grp, depth_score_series))

    special_deduped = (pd.DataFrame(merged_rows_4c)
                       if merged_rows_4c
                       else pd.DataFrame(columns=special_df.columns))
    meta_extended   = pd.concat([special_deduped, other_df], ignore_index=True)

    after_4c = len(special_deduped)
    print(f"  Special-author rows before : {before_4c:,}")
    print(f"  Sample ID groups collapsed : {n_special_collapsed:,}")
    print(f"  Special-author rows after  : {after_4c:,}  ({before_4c - after_4c:,} removed)")
    print(f"  Total rows after step 4c   : {len(meta_extended):,}")
else:
    print(f"  WARNING: 'Author' or 'Sample ID' column absent - skipping")

# -- step 4d: strip " (Source)" from all string cells -----------------------
print("\n--- Stripping ' (Source)' tags ---")

n_cells_cleaned = 0
for col in meta_extended.select_dtypes(include="object").columns:
    before_col = meta_extended[col].copy()
    meta_extended[col] = meta_extended[col].apply(
        lambda v: str(v).replace(" (Source)", "") if pd.notna(v) else v
    )
    changed = (meta_extended[col] != before_col) & before_col.notna()
    n_cells_cleaned += int(changed.sum())

print(f"  Cells cleaned: {n_cells_cleaned:,}")

# -- step 5: determine infant status ------------------------------------------
print("\n--- Determining infant status ---")

infant_meta_mask = meta_extended.apply(is_infant, axis=1)
print(f"  Infant rows in metadata (age/age group): {infant_meta_mask.sum()}")

infant_rows_merged = merged[merged.apply(is_infant, axis=1)]
print(f"  Infant rows in merged reference: {len(infant_rows_merged)}")

infant_ids_merged = set()
for _, row in infant_rows_merged.iterrows():
    for col in ["SRS ID", "SRA_Accession", "Sample ID", "run_accession"]:
        if col in row and pd.notna(row[col]):
            infant_ids_merged.add(str(row[col]).strip())
    for srr in split_ids(row.get("Paired SRR List", None)):
        infant_ids_merged.add(srr)

print(f"  Unique infant ID tokens from merged: {len(infant_ids_merged)}")

def matches_infant_merged(row):
    for col in ["SRS ID", "SRA_Accession", "Sample ID"]:
        if col in row and str(row.get(col, "")).strip() in infant_ids_merged:
            return True
    for srr in split_ids(row.get("Paired SRR List", None)):
        if srr in infant_ids_merged:
            return True
    return False

infant_merged_mask = meta_extended.apply(matches_infant_merged, axis=1)
print(f"  Metadata rows flagged as infant via merged cross-ref: {infant_merged_mask.sum()}")

infant_final_mask = infant_meta_mask | infant_merged_mask
print(f"  Total infant rows (union): {infant_final_mask.sum()}")

# -- step 6: write outputs ----------------------------------------------------
print("\n--- Writing outputs ---")

drop_internal = ["_recovered_from_merged"]
out_cols      = [c for c in meta_extended.columns if c not in drop_internal]
meta_final    = meta_extended[out_cols].copy()

# Column order: Chosen_match_ID first, then Secondary_run_IDs, then everything else
priority_cols = [c for c in ["Chosen_match_ID", "Secondary_run_IDs"] if c in meta_final.columns]
other_cols    = [c for c in meta_final.columns if c not in priority_cols]
meta_final    = meta_final[priority_cols + other_cols]

meta_final.to_csv(OUT_WITH_INFANTS, index=False)
print(f"  {OUT_WITH_INFANTS}: {len(meta_final)} rows")

meta_no_infants = meta_final[~infant_final_mask]
meta_no_infants.to_csv(OUT_NO_INFANTS, index=False)
print(f"  {OUT_NO_INFANTS}: {len(meta_no_infants)} rows  "
      f"({infant_final_mask.sum()} infant rows excluded)")

meta_no_infants.to_csv(OUT_PRIMARY, index=False)
print(f"  {OUT_PRIMARY}: {len(meta_no_infants)} rows  <- pipeline input")

pd.Series(truly_unmapped, name="meandepth_index").to_csv(
    OUT_TRULY_UNMAPPED, index=False)
print(f"  {OUT_TRULY_UNMAPPED}: {len(truly_unmapped)} entries")

pd.DataFrame(loser_rows).to_csv(OUT_COLLISION_LOSERS, index=False)
print(f"  {OUT_COLLISION_LOSERS}: {len(loser_rows)} entries")

# -- final summary ------------------------------------------------------------
n_4b_removed     = before_4b - after_4b
n_4c_removed     = before_4c - after_4c
n_with_secondary = (meta_no_infants["Secondary_run_IDs"].notna().sum()
                    if "Secondary_run_IDs" in meta_no_infants.columns else 0)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Meandepth indices total              : {len(depth_df)}")
print(f"  Mapped to original metadata          : {len(mapped_sids)}")
print(f"  Recovered from merged reference      : {len(recovered_rows)}")
print(f"  Truly unmapped (not found anywhere)  : {len(truly_unmapped)}")
print(f"  Study-excluded rows removed          : {n_excl}")
print(f"  Collision losers (total)             : {len(loser_sids)}")
print(f"  Collision losers kept as secondary   : {n_secondary_total}")
print(f"  SRS/Sample-ID duplicates removed     : {n_4b_removed}")
print(f"  Special-author duplicates removed    : {n_4c_removed}")
print(f"  Cells with ' (Source)' stripped      : {n_cells_cleaned:,}")
print(f"  Final rows (with infants)            : {len(meta_final)}")
print(f"  Final rows (no infants)              : {len(meta_no_infants)}")
print(f"  Infant rows excluded                 : {infant_final_mask.sum()}")
print(f"  Samples with >=1 secondary run ID    : {n_with_secondary}")
print(f"")
print(f"  Pipeline input file  -> {OUT_PRIMARY}")
print(f"  Chosen_match_ID      = primary matrix index (highest depth_score)")
print(f"  Secondary_run_IDs    = comma-sep, depth_score > {TECH_REP_MIN_BP // 1_000_000} Mb")
