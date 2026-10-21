"""
Configuration for the microbiome extinctions analysis pipeline (v3).

Cleaned to include only settings directly referenced in the article.
"""

import os
from collections import OrderedDict

# ╔══════════════════════════════════════════════════════════════════╗
# ║  REFERENCE DATABASES                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

REFERENCES = {
    "UHGG": {
        "breadth_file":  "meancoverage_and_std_matrix_high_UHGG_20260820.csv",
        "depth_file":    "meandepth_and_std_matrix_high_UHGG_20260820.csv",
        "lengths_file":  "all_SGB_lengths_UHGG.txt",
    },
    "lev": {
        # 2026-08-12: backfilled with the 398 samples the original Leviatan run
        # was missing (tools/merge_lev_backfill.py). 23,914 -> 24,312 rows. The
        # pre-backfill matrices remain as
        # *_merged_dedup_firstcol.csv, checksummed in ~/matrix_backup/.
        "breadth_file":  "meancoverage_and_std_matrix_high_lev_20260820.csv",
        "depth_file":    "meandepth_and_std_matrix_high_lev_20260820.csv",
        "lengths_file":  "all_SGB_lengths_lev.txt",
    },
    "chimp": {
        "breadth_file":  "meancoverage_and_std_matrix_high_chimp_merged.csv",
        "depth_file":    "meandepth_and_std_matrix_high_chimp_merged.csv",
        "lengths_file":  "all_SGB_lengths_chimp.txt",
    },
    "combined": {
        # 2026-08-12: rebuilt from the backfilled Leviatan matrix on the
        # INTERSECTION of UHGG and Leviatan samples (tools/build_combined_matrix.py).
        # The previous construction took the union, so 421 rows present in UHGG
        # but absent from Leviatan carried all 2,198 Leviatan-derived columns as
        # zeros -- indistinguishable from absent taxa. 23,913 rows; the builder
        # reproduces the published matrix exactly for samples present in both.
        "breadth_file":  "combined_coverage_matrix_20260820.csv",
        "depth_file":    "combined_depth_matrix_20260820.csv",
        "lengths_file":  "all_SGB_lengths_combined.txt",
        "combined_sgb_metadata_file": "combined_sgb_metadata.xlsx",
        "taxon_list_crosswalk": "combined_sgb_metadata",
    },
}

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SHARED INPUT FILES                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

META_FILE_CANDIDATES = [
    # 2026-09-21: obesity counts as unhealthy, consistently. 145 samples were
    # flagged obese and still labelled Healthy -- 143 of them by an explicit
    # obesity column rather than by the BMI >= 30 fallback, and concentrated in
    # studies that recorded obese participants as healthy controls. Since
    # `Unhealthy` is the control/case split for all of Section 3, those samples
    # were sitting in the control group of every other condition's test.
    # See docs/obesity_health_corrections.csv and
    # tools/apply_obesity_health_rule.py.
    #
    # Consequence for the code: "Obese (healthy only)" is now empty by
    # construction and self-skips on the too-few-samples guard.
    # 2026-09-24: the curated ground-truth M1, and the file reviewers see.
    # Built by tools/build_curated_m1.py from metadata_Aug_26_2026_final.csv.
    # Against the 2026-09-21 file it is identical in Age, Gender, BMI,
    # Lifestyle and row order; the differences are deliberate curation:
    #   * 202 samples move Healthy -> Unhealthy (none the other way) from the
    #     disease-flag additions (IBD, STH, IGT, CRC, T2D, MASLD,
    #     hypertension), the Bramble konzo correction, less the melanoma and
    #     HMP CDI drops. Unhealthy 6,310 -> 6,512.
    #   * disease_konzo added (67 -> 68 disease_* columns).
    #   * Eight redundant columns removed: Index, Healthy, Unhealthy (check),
    #     Diseased, Diseased (all), Diseased (gut and metabolic),
    #     Diseased (metabolic) or obese, Diseased (other). Only `Unhealthy`
    #     now carries the health label, which is what the code computes on.
    #     Note: visualization.py's country summary reads the three
    #     "Diseased (...)" columns through _frac_yes, which returns NaN for a
    #     missing column -- frac_gut_metab_disease, frac_other_disease and
    #     frac_metab_or_obese go NaN rather than raising. They are derivable
    #     from the disease_* columns if those fields are wanted back.
    "final_run/metadata_M1.csv",
    "metadata_Sep_21_2026_obesity_unhealthy.csv",
    # 2026-08-26: corrected cohort. Three CarterM 2023 accessions were removed --
    # ENA registers them as Bifidobacterium longum subsp. infantis and the matrix
    # agrees (2 SGBs, top1_share 0.58-0.76), so they are isolate genomes, not gut
    # metagenomes. Also carries 135 Unhealthy corrections, the disease-category
    # assignments, and the appended 2026-08-20 samples merged into their studies.
    # See docs/health_label_corrections.csv and docs/removed_samples.csv.
    "metadata_Aug_26_2026_final.csv",
    "metadata_Aug_20_2026.csv",
    "metadata_Mar_16_2026.csv",
    "metadata_non_infants_gt100Mb_sra_by_biosample_best_meandepth_fullmeta_dedup_by_Chosen_match_ID_filled_v2.csv",
    "metadata_non_infants_gt100Mb_sra_by_biosample_best_meandepth_fullmeta_dedup_by_Chosen_match_ID.csv",
    "diag_depth_abundance_meta_filtered_ge_10000000_no_newborn.csv",
]

META_FILE = None
for _candidate in META_FILE_CANDIDATES:
    if os.path.exists(_candidate):
        META_FILE = _candidate
        break
if META_FILE is None:
    META_FILE = META_FILE_CANDIDATES[0]

# Falling back is never silent. The candidates below the first are *different
# cohorts*, not alternative spellings of one -- the notes above record
# Unhealthy going 6,310 -> 6,512 between two of them -- so a run that picks
# the wrong one finishes normally and reports different numbers. Six files
# named metadata_M1.csv exist in this tree's history (repo root, metadata/,
# final_run/, .rename_backup/, plus a _verified variant), so this is a real
# way to be wrong, not a hypothetical. The reported results come from
# final_run/metadata_M1.csv, sha256 06031e18...; tools/preflight.py checks
# that against docs/INPUT_CHECKSUMS.tsv.
if META_FILE != META_FILE_CANDIDATES[0]:
    import warnings as _warnings
    _warnings.warn(
        f"metadata: {META_FILE_CANDIDATES[0]!r} is absent, so this run will "
        f"use {META_FILE!r} instead. That is a DIFFERENT COHORT and will not "
        f"reproduce the published numbers. Fetch the deposited inputs "
        f"(bash download_data.sh <record>) and re-check with "
        f"`python tools/preflight.py`.",
        RuntimeWarning, stacklevel=2)

VANISH_FILE        = "Vanish_list.txt"
BLOSSUM_FILE       = "Blossum_list.txt"
VANISH_PHYLO_FILE  = "Vanish_phylo.txt"
BLOSSUM_PHYLO_FILE = "Blossom_phylo.txt"

SAMPLE_RUN_FILE   = "Chosen_population_samples.csv"
TRIMMING_FILE     = "trimming_reports_parsed.csv"

# ╔══════════════════════════════════════════════════════════════════╗
# ║  ANALYSIS THRESHOLDS                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

BREADTH_THRESHOLD      = 10.0      # % breadth for detection (breadth mode)
BREADTH_THRESHOLD_FRAC = 0.10
CLR_PSEUDOCOUNT        = 1e-6

# ── Depth matching ──
# Sections 1 & 2: log₁₀ tolerance for 1:1 Ind ↔ NI matching
DEPTH_MATCH_FRAC_TOL   = 0.03      # 3% fractional difference for 1:1 matching

# Section 3: fractional tolerance for triplet matching (NI ↔ IndH ↔ IndU)
# Article: "at most a 1% difference in the number of bacterial bases"
HEALTH_TRIPLET_FRAC_TOL = 0.01     # 1% fractional difference

# ── Anchored 1-to-K depth matching ──
# H-G-anchored: best 90% of H-G -> K x Transitional + K x Industrial
ANCHORED_HG_FRAC_TOL    = 0.05     # 5% fractional depth tolerance
ANCHORED_HG_PCT_KEEP    = 0.90     # keep best 90% of H-G anchors
# NI-anchored: best 80% of Non-Industrial -> K x Industrial
ANCHORED_NI_FRAC_TOL    = 0.05     # 5% fractional depth tolerance
ANCHORED_NI_PCT_KEEP    = 0.80     # keep best 80% of NI anchors
ANCHORED_K_MAX_TRY      = 10       # max K to attempt

# ╔══════════════════════════════════════════════════════════════════╗
# ║  ACCUMULATION CURVES                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

ACCUM_NUM_INNER_BOOT  = 100
ACCUM_NUM_PARAM_BOOT  = 100
ACCUM_TOL_MICROBES    = 5.0

ACCUM_BREADTH_THRESHOLDS = [1.0, 3.0, 10.0, 30.0, 50.0, 90.0]
ACCUM_FLAT_THRESHOLDS_BP = [10_000, 30_000, 100_000, 300_000, 500_000, 1_000_000]

# Main-text thresholds
ACCUM_MAIN_TEXT_BREADTH = 10.0
ACCUM_MAIN_TEXT_FLAT_BP = 300_000

# Minimum total mapped bp — samples below this are excluded globally
MIN_TOTAL_BP = 3e7  # 30 Mb  — computed as sum(mean_depth x genome_length) from the depth matrix

# A sample must detect at least this many SGBs at the main-text breadth threshold
# to be an analysable community. The depth filter above cannot substitute for it:
# 16S libraries mis-registered as WGS carry real mapped bases -- all of them piled
# onto one conserved window -- so they clear 30 Mb while detecting almost nothing.
# 101 samples fall below 5 in the 2026-08-20 cohort, 96 of them Industrial, led by
# MehtaRS 2018 (52) and ZeeviD 2015 (13). It overlaps the amplicon gate in
# preprocessing without being redundant: of the 104 amplicon-flagged samples this
# rule removes 46, while catching 55 that the gate does not.
MIN_SGBS_DETECTED = 5
MIN_SGBS_BREADTH_PCT = 10.0

# Samples whose breadth/depth profile identifies them as amplicon libraries
# mis-registered as WGS. The gate in the preprocessing repo
# (aggregate_coverage.py) catches these at ingest, but only when coverage files
# are built -- it cannot act on matrices that already exist, so the analysis
# needs its own copy of the exclusion. MIN_SGBS_DETECTED removes the 46 of these
# that resolve fewer than five SGBs; the remaining 58 clear that filter with real
# mapped bases piled onto one conserved window, and are excluded here by name.
#
# Criterion: median breadth < 0.02% AND depth-per-covered-base > 2.5, empirical
# from all 24,173 samples. See docs/amplicon_flagged_recommended.csv.
EXCLUDE_SAMPLES_FILE = "docs/amplicon_flagged_recommended.csv"


def _load_excluded_samples(path=EXCLUDE_SAMPLES_FILE):
    """Sample IDs to drop regardless of any other filter. Empty if absent."""
    import csv
    import os
    if not os.path.exists(path):
        return frozenset()
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    col = "Sample_ID" if rows and "Sample_ID" in rows[0] else None
    if col is None:
        return frozenset()
    return frozenset(r[col].strip() for r in rows if r.get(col, "").strip())


EXCLUDE_SAMPLES = _load_excluded_samples()

# ── Sample identity: one sample is one BioSample ────────────────────────────
# A person may contribute several BioSamples (repeat stools, timepoints), and
# subject-level steps handle that through Subject_ID. A BioSample may never
# contribute more than one sample: a second library of the same specimen is a
# re-measurement, not an observation. Enforced in loading.py on the column
# below. Where a BioSample carries more than one row the kept row is the
# preferred library (BIOSAMPLE_PREFERRED_LIBRARY), then the one with the most
# mapped bases, then the larger "Total bases", then M1 order; every drop is
# printed. Only unique BioSamples are affected -- no study loses a specimen.
#
# Verified 2026-09-28 against ENA for 17,965 of the 18,063 UHGG cohort samples
# (the other 98 have no ENA-resolvable accession): the only violations were
# Carter's -- 16 Hadza stools with a MoBio and a PheChl library, and one Nepal
# stool (THA1067YZ) with two MoBio libraries. IaniroG_2022 FMT_68_t90 carried
# FMT_73_t7's BioSample in M1; that was an M1 error, corrected in M1.
ONE_SAMPLE_PER_BIOSAMPLE = True
BIOSAMPLE_COL = "BioSample"

# Which library of a multi-library BioSample to keep, before depth decides.
# {StudyFolder prefix: (column, regex a preferred row matches)}.
#
# Carter's 16 Hadza MoBio/PheChl pairs keep PheChl. Those libraries are deeper
# in 16 of 16 (median 2.05x mapped bases against UHGG) and still detect more
# SGBs after adjusting for depth (median +46 paired; +94 across all 316 Hadza
# libraries). Stated as a preference rather than left to mapped bases because
# stool 2277's libraries are equally deep (ratio 0.99-1.05 by reference), and
# mapped bases are per-reference: the same specimen would be measured by a
# different library in lev than in UHGG. The Nepal MoBio/MoBio pair has no
# preference and falls to depth, which picks ERR7738619 in every reference.
BIOSAMPLE_PREFERRED_LIBRARY = {
    "CarterM_2023": ("Sample ID", r"^[A-Za-z]+_PheChl_"),
}

# Drop samples whose Lifestyle does not resolve to Industrial or Non-Industrial
# (i.e. "Unclear"). classify_macro_group already returns None for them, so they
# never entered a grouped comparison; leaving them in only inflated the
# whole-cohort descriptives (depth audit N, depth-vs-richness correlations,
# Table S1) relative to every grouped statistic. Set False to keep them.
DROP_UNCLASSIFIED_LIFESTYLE = True

# Bray-Curtis threshold above which a secondary run is reinstated as an
# independent sample rather than treated as a technical replicate.
# ── Secondary-run reinstatement ──
# Whether a secondary run whose community differs from its primary is promoted
# to an independent sample. Default OFF.
#
# The mechanism was built for the case where one metadata row conflates
# genuinely different specimens. Auditing which pairs it actually fires on
# showed that case does not occur here -- every contributing study is the same
# subject, and two are the same physical specimen:
#
#   ZeeviD_2015_ISR    105/173  one participant sampled repeatedly across a
#                               dietary intervention (PNP_DietIntervention_14
#                               alone has four secondaries at BC 0.50-0.62).
#                               Reinstating makes one person five "independent"
#                               samples spanning a deliberate perturbation.
#   NielsenHB_2014_DNK  44/173  MetaHIT resequencing, BC 0.11-0.16, borderline.
#   CarterM_2023_TZA    10/173  Hadza_MoBio_* vs Hadza_PheChl_*: one BioSample,
#                               two DNA extraction protocols. BC 0.45-0.79 is
#                               extraction-kit bias, not biology.
#
# The threshold had no empirical basis either: over 3,346 primary/secondary
# pairs the Bray-Curtis distribution decays smoothly with no valley at 0.1
# (153 pairs in [0.05,0.10), 39 in [0.10,0.15), 28 in [0.15,0.20)), so the
# reinstated N is an artefact of where the line is drawn -- 340 at 0.05, 173 at
# 0.10, 118 at 0.15. It was also asymmetric between the compared groups
# (Industrial 5.9% of pairs, Non-Industrial 1.7%), so it was not neutral with
# respect to the main contrast.
#
# Turning it off costs 173 of 18,176 samples (163 Industrial, 10 Non-Industrial)
# and removes that asymmetry. Set True to restore the old behaviour; the
# threshold sweep is the supplementary sensitivity analysis.
REINSTATE_SECONDARIES  = False
BC_REINSTATE_THRESHOLD = 0.1

DETECTION_MODE = "both"

# ── OW-AIPWE depth-bias correction (depth_bias_ipw.py) ──
# Bootstrap iterations for the standalone main figure (higher = tighter CIs)
# Bootstrap replicates — adaptive scaling
# n_boot scales as: clip(BASE x sqrt(min(n_i,n_n) / 500), BASE, CAP)
# Carter study (~900/800 samples): BASE -> 1,000 reps
# Full pipeline NI group (~3,800):         -> ~2,757 reps
# Full pipeline Industrial  (~14k):        -> capped at 5,000
OW_AIPWE_N_BOOT_BASE  = 1000   # minimum reps (also used as Carter default)
OW_AIPWE_N_BOOT_CAP   = 5000   # maximum reps for main comparisons
# Threshold sweep uses a lighter cap (many specs to run):
OW_AIPWE_N_BOOT_SWEEP_CAP = 1000
# Legacy: kept so any external callers that pass n_boot=config.OW_AIPWE_N_BOOT_MAIN
# still work.  The adaptive helper ignores this when n_boot is None.
OW_AIPWE_N_BOOT_MAIN  = 1000
# Bootstrap iterations per threshold in the sweep (lower for speed)
OW_AIPWE_N_BOOT_SWEEP = 200   # base reps for sweep (scales up to OW_AIPWE_N_BOOT_SWEEP_CAP)
# Bootstrap iterations for the robustness figure (runs 4x per pipeline call)
OW_AIPWE_N_BOOT_ROBUSTNESS = 200
OW_AIPWE_N_BOOT_ROBUSTNESS_CAP = 2000  # adaptive cap for robustness-figure GMW bootstrap
# Max samples per group for OW-AIPWE in the robustness figure.
# Equal-sized subsamples are drawn stratified by depth decile.
# KDE class-frequency weighting still uses the TRUE full group sizes
# so the propensity crossover is correct even on balanced subsamples.
# Timing: scipy gaussian_kde eval is O(n_sub^2); 2000 -> ~3-5s per spec pair.
# OW_AIPWE_N_SUBSAMPLE = 2000  # retired

# ── Study exclusion (applied during data loading) ───────────────────────────
# Studies to exclude from analysis (matched via str.startswith() on
# StudyFolder and study_name columns, and exact match on Author).
# Set to an empty set/frozenset to disable.
#
# Derived from audit of full metadata (see filter_diag_meta.py):
#   DhakanDB_2019_IND        131 rows  — extreme depth outliers
#   FerrettiP_2018           121 rows  — longitudinal infant data
#   Bengtsson-PalmeJ_2015_SWE  68 rows  — resistome study
#   Heintz-BuschartA_2017_LUX  50 rows  — type-1 diabetes cohort
#   KaurK_2020                 31 rows  — antibiotic-treated
#   UKN_2017_UK                 6 rows  — unknown provenance
#   eKmeta_2016_AUS             1 row   — single-sample study
EXCLUDE_STUDIES: frozenset = frozenset({
    "Bengtsson-PalmeJ",
    "DhakanDB",
    "FerrettiP",
    "Heintz-BuschartA",
    "KaurK",
    "UKN",
    "eKmeta",
})

# Columns searched (in order) for the study name.
# StudyFolder and MBIOTEMP name use startswith(); Author uses exact match.
STUDY_EXCLUSION_COLS = [
    "StudyFolder",
    "study_name [MBIOTEMP_study_name]",
    "Author",
]

ACCUM_N_MIN       = 5
ACCUM_N_MAX       = 14000
ACCUM_N_NUM       = 30
ACCUM_SWEEP_N_MIN = 30
ACCUM_SWEEP_N_MAX = 14000
ACCUM_SWEEP_N_NUM = 30
ACCUM_USE_WEIGHTS = False
ACCUM_USE_MEDIAN  = True
ACCUM_MIN_SAMPLES = 30

# ╔══════════════════════════════════════════════════════════════════╗
# ║  ALPHA / BETA DIVERSITY                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

DIVERSITY_N_PERMANOVA   = 999
DIVERSITY_BETA_METRIC   = "braycurtis"
DIVERSITY_PCOA_AXES     = 3

# ╔══════════════════════════════════════════════════════════════════╗
# ║  DISEASE COLUMNS (Section 3 health)                             ║
# ╚══════════════════════════════════════════════════════════════════╝

# Three disease super-categories for within-Unhealthy subcategory matching
DISEASE_SUPERCATEGORIES = OrderedDict([
    ("Obesity", {
        "type": "obese",
    }),
    ("Gut & Metabolic", {
        "type": "disease",
        "cols": [
            "disease_T2D", "disease_IGT", "disease_T1D",
            "disease_fatty_liver", "disease_metabolic_syndrome",
            "disease_ACVD", "disease_hypertension",
            "disease_IBD", "disease_Crohn's", "disease_Ulcerative_colitis",
            "disease_CRC", "disease_adenoma", "disease_coeliac",
        ],
    }),
    ("Other Disease", {
        "type": "other",
    }),
])

# ── Demographic strata ───────────────────────────────────────
# The same contrast machinery as DISEASE_NCD_MAP, applied to who the
# participant is rather than what they have. Values are comparison_type
# strings resolved by health_cmh.build_group_masks.
#
# Age and sex contrast within Industrial *Healthy* samples, so that a band
# difference is not just the disease burden that comes with age.
#
# BMI cannot do the same. Obesity sets the Unhealthy label, so
# Industrial & Healthy & Obese is empty by construction (measured: 0 of
# 1,239 obese Industrial samples). The BMI contrasts therefore run across
# all health statuses, exactly as the existing `obese` comparison does and
# for the same reason. Underweight participants are counted in Table S3 but
# excluded from both contrasts: they are a distinct phenotype, not a milder
# version of normal.
#
# Age bands come from the numeric `Age` column, not the `Age group` column,
# which is a coarse and case-inconsistent mix of Adult/adult/Elderly/senior.
# BMI is read from `BMI` through a 10-60 plausibility window (4 values fall
# outside it). Under-18s are their own contrast and never pooled with
# adults: the gut microbiome is still developing, and pooling would confound
# age with maturation.
#: Run the age/sex/BMI strata as part of Section 3. Off only for a quick
#: disease-only rerun; the supplementary demographic figures need it.
RUN_DEMOGRAPHICS = True

AGE_BANDS = [("<18", 0, 18), ("18-29", 18, 30), ("30-44", 30, 45),
             ("45-59", 45, 60), ("60+", 60, 200)]
AGE_REFERENCE_BAND = "18-29"

BMI_BANDS = [("Underweight", 0, 18.5), ("Normal", 18.5, 25),
             ("Overweight", 25, 30), ("Obese", 30, 100)]
BMI_PLAUSIBLE = (10.0, 60.0)

DEMOGRAPHIC_MAP = OrderedDict([
    ("Age <18",              "age_band:<18"),
    ("Age 30-44",            "age_band:30-44"),
    ("Age 45-59",            "age_band:45-59"),
    ("Age 60+",              "age_band:60+"),
    ("Female",               "sex"),
    ("Obese vs rest",        "bmi_obese_vs_rest"),
    ("Overweight or obese",  "bmi_excess_vs_normal"),
])

#: Long form for captions and Table S3, since the keys above are axis labels.
DEMOGRAPHIC_CONTRAST_TEXT = {
    "Age <18":             "under 18 vs 18-29 (Industrial, healthy)",
    "Age 30-44":           "30-44 vs 18-29 (Industrial, healthy)",
    "Age 45-59":           "45-59 vs 18-29 (Industrial, healthy)",
    "Age 60+":             "60+ vs 18-29 (Industrial, healthy)",
    "Female":              "female vs male (Industrial, healthy)",
    "Obese vs rest":       "BMI >=30 vs BMI 18.5-30 (Industrial, all health "
                           "statuses)",
    "Overweight or obese": "BMI >=25 vs BMI 18.5-25 (Industrial, all health "
                           "statuses)",
}

# Fine-grained disease map for per-condition analyses
# NOTE: Per-condition analyses in Section 3 use within-study CMH (not pooled Fisher).
# "Glucose Intolerance" groups T2D + IGT. Type 1 diabetes is excluded: it is
# autoimmune beta-cell loss, not the insulin-resistance spectrum the category
# describes. (It changed no cohort when removed: the 4 T1D samples are all in
# BrownCT_2011, which never has >=5 cases, so never forms a stratum.)
DISEASE_NCD_MAP = OrderedDict([
    ("Obese",               {"type": "obese"}),
    ("Overweight",          {"type": "overweight"}),
    ("Glucose Intolerance", {"type": "disease",
                             "cols": ["disease_T2D", "disease_IGT"]}),
    ("Fatty Liver",         {"type": "disease", "cols": ["disease_fatty_liver"]}),
    ("Metabolic Syndrome",  {"type": "disease", "cols": ["disease_metabolic_syndrome"]}),
    ("ASCVD",               {"type": "disease", "cols": ["disease_ACVD"]}),
    ("Hypertension",        {"type": "disease", "cols": ["disease_hypertension"]}),
    ("IBD",                 {"type": "disease",
                             "cols": ["disease_IBD", "disease_Crohn's",
                                      "disease_Ulcerative_colitis"]}),
    ("CRC / Adenoma",       {"type": "disease",
                             "cols": ["disease_CRC", "disease_adenoma"]}),
    ("Coeliac",             {"type": "disease", "cols": ["disease_coeliac"]}),
])

# ╔══════════════════════════════════════════════════════════════════╗
# ║  EXTERNAL DATA ROOT                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
# The health-metadata sources (the MicrobiomeDB download and the da Silva 2026
# supplement) live outside the repo. On the machine this was assembled on they
# sit in iCloud; in a transferred package they sit in a `metadata_sources/`
# sibling of the repo. Resolved by search rather than a hardcoded absolute path
# so a copy on another machine does not silently look at somewhere that does not
# exist -- the same pattern as META_FILE_CANDIDATES above.
#
# Order: an explicit MICROBIOME_DATA_ROOT always wins, then packaged locations,
# then the original iCloud path.
import os as _os
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]

DATA_ROOT_CANDIDATES = [
    _os.environ.get("MICROBIOME_DATA_ROOT", ""),
    str(_REPO / "metadata_sources"),
    str(_REPO.parent / "metadata_sources"),
    str(_REPO / "data"),
    str(_Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Work/"
        "Postdoc_initial_data/bacterial_evolution"),
]


def resolve_data_root(required=("plots/MBIOTEMP_NCBI_metadata",)):
    """First candidate that contains the required sub-paths, else None.

    Returns a Path. Callers should report the candidate list on failure rather
    than proceeding with a path that does not exist, because the failure mode
    otherwise is an empty source and a silently under-populated rebuild.
    """
    for cand in DATA_ROOT_CANDIDATES:
        if not cand:
            continue
        root = _Path(cand).expanduser()
        if all((root / r).exists() for r in required):
            return root
    return None


def describe_data_root_failure(required=("plots/MBIOTEMP_NCBI_metadata",)):
    lines = ["could not locate the external metadata sources.",
             f"  looked for: {', '.join(required)}",
             "  candidates tried:"]
    for cand in DATA_ROOT_CANDIDATES:
        if cand:
            lines.append(f"    {'OK  ' if _Path(cand).expanduser().exists() else 'no  '}{cand}")
    lines.append("  set MICROBIOME_DATA_ROOT to the directory holding plots/ "
                 "(see docs/PORTING.md)")
    return "\n".join(lines)


#: Restrict case/control contrasts to one sample per subject.
#: Several studies contribute many samples per person -- Lloyd-PriceJL_2019 is
#: 1,125 samples from 104 people, iMSMS_2022 is 1,133 from 611 -- which inflates
#: a stratum's pooling weight and, in the per-taxon CMH, its cell counts
#: outright. `tools/build_verified_m1.py` marks one representative per subject in
#: the `subject_primary` column; this flag makes health_cmh.build_group_masks
#: honour it. Accumulation curves deliberately do NOT use it: there, repeated
#: samples simply add fewer new taxa, so keeping them is conservative for
#: catalogue coverage. Set False to reproduce the pre-correction behaviour.
DEDUP_SUBJECTS = True

# The reference whose analytic cohort the manuscript reports (Figure 1, the
# dataset paragraph, Tables S1-S3: 18,075 samples). Figure 1 and the metadata
# summary figures intersect M1 with results/<MAIN_COHORT_REF>/cohort_samples.csv.
# It was "UHGG" (18,046), so Figure 1 disagreed with every N the text quotes.
MAIN_COHORT_REF = "combined"

# The references Section 3 (health associations) is computed for.
#
# Every health figure is drawn on one reference. The manifest
# (tools/additional_figures_manifest.txt) names `combined` for all of
# S12, S13, S23-S31, figures/figure5.py and figures/figure2.py both default
# to --ref combined, and figure2's own --ref help says UHGG "disagrees with
# the supplementary tables". The two health tables were built on UHGG by
# accident and tools/patch_tables_s9_s10.py exists to move them onto the
# combined basis; their builders now read combined directly.
#
# So nothing published consumes a non-combined Section 3, while running it
# costs about eight hours per reference: the background curve's bootstrap is
# 54 condition-by-list cells plus 42 demographic ones, two detection specs,
# and ~145 s per cell at HEALTH_BG_N_BOOT = 2000 (measured; the loop in
# health_background.background_z is serial). On --all that is ~24 h spent on
# UHGG, lev and chimp for output no figure or table reads.
#
# An explicit `--only s3` for another reference still runs: this governs the
# --all / default sweep only. Set to None to run Section 3 for every
# reference, as before.
#
# No `tuple | None` annotation here: config.py has no
# `from __future__ import annotations`, so a module-level annotation is
# evaluated on import and PEP 604 unions raise TypeError on the 3.8
# interpreter several of the tools still run under.
SECTION3_REFS = ("combined",)

# Minimum samples per arm per study for within-study CMH strata.
#
# This is the first half of the eligibility rule: a study contributes a
# stratum to a contrast only if it has at least this many cases AND at least
# this many controls. MIN_STRATA_TO_PLOT below is the second half. Together
# they decide which contrasts are reported, which is why neither Hypertension
# nor Metabolic Syndrome needs to be named anywhere.
CMH_MIN_STRATUM = 5

# Minimum qualifying studies for a health contrast to be reported -- in Figure
# 5D-E and in every supplementary health figure, which must show the same
# conditions. A qualifying study is one clearing CMH_MIN_STRATUM in both arms.
# One study gives no between-study variance (no interval, no p-value); two
# gives a sandwich SE on a single degree of freedom. Rationale in
# figures/figure5.py.
#
# On the present cohort this admits Obese (29 studies), Unhealthy (41),
# CRC / Adenoma (8), Glucose Intolerance (6) and IBD (6), and excludes
# Fatty Liver (2), Hypertension (2), ASCVD (1) and Metabolic Syndrome (1).
MIN_STRATA_TO_PLOT = 3

# Number of nearest-neighbour background taxa to select per list taxon in
# _run_prevalence_matched_background.  Higher values give a larger, more
# stable background but may include less well-matched taxa.
PREVMATCH_K_NEIGHBOURS = 10

# ── The prevalence background for health associations ──────────────────────────
# How Section 3 compares a list taxon with taxa of the same commonness, which
# sets Figure 5E and Figures S25-S30.
#
#   "curve"  the corrected method (persistence/health_background.py): a
#            study-stratified (Mantel-Haenszel) case-control shift, a
#            fixed-bandwidth local-linear background curve in log10 control
#            prevalence, and a per-taxon standardised residual z. No weights,
#            no bins, no rounding.
#   "iw"     the published importance-weighted background, kept so the 29 Sep
#            figures can be reproduced exactly. It is WRONG in three ways
#            (weight rounding drops 41-51% of the weight; the floor and clip
#            mean BlossumH+ is never matched; the shift it tests pools a/n
#            across studies). See iw_background_review_20261005/ and the
#            module docstring of health_background.py.
HEALTH_BG_METHOD = "curve"

# Kernel bandwidth of the background curve, in log10 prevalence units
# (+/-0.12 is about +/-30% in relative prevalence). Fixed, not
# nearest-neighbour: the catalogue has ~5,000 taxa below 10% prevalence and
# 5 above 85%, so an adaptive bandwidth would estimate the common end from
# taxa an order of magnitude rarer.
HEALTH_BG_BANDWIDTH = 0.12

# A list taxon is tested only when at least HEALTH_BG_MIN_SUPPORT background
# taxa lie within HEALTH_BG_SUPPORT_WINDOW log10 units of its control
# prevalence. Taxa at or above HEALTH_BG_HIGHPREV_LIMIT (P. dorei, P. vulgatus,
# B. uniformis, P. distasonis) have no comparable taxa in the catalogue at all:
# 5 taxa sit there and 4 of them are on the lists.
HEALTH_BG_SUPPORT_WINDOW = 0.25
HEALTH_BG_MIN_SUPPORT = 15
HEALTH_BG_HIGHPREV_LIMIT = 0.85
# Whether that limit also removes those taxa from each list's mean z.
# False reproduces the numbers the talk and the current manuscript report.
# True is the more defensible statistic -- above the limit the background
# curve is a one-sided extrapolation -- and is what clade_contrasts already
# does. Switching it moves Unhealthy BlossumH+ from +1.20 to +0.94 and IBD
# BlossumH+ from +1.16 to +0.99; no cell changes sign.
HEALTH_BG_HIGHPREV_EXCLUDE = False

# Keep EVERY published list out of EVERY list's background, not only the one
# being scored. The six lists are heavily nested (98 of VanishH+'s 100
# members also sit in another list), so scoring one against a background
# still containing the other five fitted the curve partly on near-duplicates
# of it. The bias was conservative -- it shrank |z| -- but 3-5x stronger for
# the BloSSUM family, so the two rows of Figure 5E were not on a common
# footing. Switching it on moves every cell away from zero, e.g. Unhealthy
# BlossumH+ +1.20 -> +1.44 and IBD BlossumH+ +1.16 -> +1.39.
# Set to False to reproduce the pre-2026-10-07 numbers.
HEALTH_BG_EXCLUDE_SIBLING_LISTS = True

# Apply the Rao-Scott first-order design effect to Section 2's weighted
# prevalence chi-square. The contingency entries are weighted sums, not counts
# of independent observations, so an uncorrected chi2 is inflated by about
# n/ESS. On the 29 Sep run (4,721 taxa, 3,595 significant at p<0.05) a design
# effect of 1.41 moves 128 of them (3.6%) out of significance and 202 out of
# Bonferroni; the headline conclusion is unaffected either way.
# False reproduces the uncorrected p-values.
DIFFERENTIAL_DESIGN_EFFECT = True
# Prevalence floor for the curve's x-axis. Every taxon detected in no sample
# of either arm lands on log10 of this value with shift = 0, so they all
# collect the SAME z -- whatever (0 - mu)/sigma is at the floor.
HEALTH_BG_PREV_FLOOR = 1e-4

# Drop those never-detected taxa from each list's mean z. Their shift is zero
# because nothing was observed, not because nothing moved, so they are not a
# weak measurement but the absence of one, and each still counts as a taxon.
# Nine of them sit in VANISH, all carrying z = -0.475205, which was enough to
# set the sign of two of its five cells. `_run_taxon_drivers` has always
# skipped them, so leaving them in the list statistic made Figure 5E and the
# driver figures describe different datasets. Set to False to reproduce the
# 9 Oct run's numbers.
HEALTH_BG_DROP_NEVER_DETECTED = True

# Bootstrap resamples behind each list's mean-z CI. The bootstrap REFITS the
# curve on resampled background taxa, which is the whole point: the energy test
# this replaces treated the weighted background as known and was
# anticonservative at ESS ~120-150.
#
# Raised from 200 to 2000 on 2026-10-08. At 200 the percentile interval was too
# coarse for the cells sitting near a threshold, and the failure is worse than
# a wide interval: the VERDICT depended on the seed. Measured on
# IBD x VanishH+ over four seeds --
#
#     n_boot=200   ci_hi  +0.004, -0.015, +0.023, -0.022   boot_p 0.030-0.090
#     n_boot=2000  ci_hi  +0.001, +0.012, +0.003, +0.004   boot_p 0.053-0.065
#
# -- so at 200 two seeds of four put the upper bound below zero and reported a
# significant depletion, and two put it above and reported nothing, with
# boot_p straddling 0.05. At 2000 every seed agrees: the cell is a near-null
# just outside significance. The point estimate never moved (mean z -0.184
# throughout); it was only ever the interval that was unstable.
# Two further consequences worth knowing:
#   * The floor on boot_p drops from 0.005 to 0.0005, so the "***" band
#     (p < 0.001) becomes reachable for the first time; plot_utils.boot_star
#     drops tiers the resolution cannot support, so it will start admitting
#     *** on its own.
#   * Section 3 costs about 10x more in the curve step. Measured on this
#     catalogue: ~17 s per condition x list cell at 200, ~170 s at 2000, over
#     54 cells and per detection spec.
HEALTH_BG_N_BOOT = 2000
HEALTH_BG_SEED = 17

# Clade contrasts (Figure S30): minimum clade size, placebo-clade draws and
# label permutations.
HEALTH_BG_CLADE_MIN_TAXA = 5
HEALTH_BG_CLADE_NPERM = 5000
HEALTH_BG_CLADE_SEED = 41

# The clades Figure S30 shows, as (condition, rank, clade, list). These are the
# contrasts that survive BH correction and are the ones the manuscript quotes;
# the full table of every clade is written beside the figure.
#: Two of these named the pooled "Unhealthy" contrast, which is now withheld
#: from the reported figures (see HEALTH_PLOT_SKIP below). Rather than let S30
#: fall from six panels to four, each was moved to a focal condition keeping
#: the SAME clade and list, so the figure's structure -- one clade shown under
#: two conditions, per family -- is unchanged and only the contrast differs.
#: Both replacements are the strongest focal evidence available for that
#: clade/list pair, from the figure's own contrast table:
#:
#:   Lachnospiraceae / BloSSUM+    Obese          diff +1.36, q = 0.002
#:   Oscillospiraceae / BlossumH+  CRC / Adenoma  diff +1.19, q = 0.004
#:
#: (The pooled panels they replace were +1.49 and +1.03.) This is an author's
#: choice of what to show, not a rule, so it is spelled out here to be argued
#: with: the other candidates were Glucose Intolerance for both, at +0.76 and
#: +0.92.
HEALTH_BG_CLADE_PANELS = [
    ("Obese",                "genus",  "Prevotella",       "VanishH+"),
    ("IBD",                  "family", "Oscillospiraceae", "VanishH+"),
    ("Obese",                "family", "Lachnospiraceae",  "BloSSUM+"),
    ("IBD",                  "family", "Lachnospiraceae",  "BloSSUM+"),
    ("CRC / Adenoma",        "family", "Oscillospiraceae", "BlossumH+"),
    ("IBD",                  "genus",  "Bacteroides",      "BlossumH+"),
]

# `Obesity (old)` was dropped from M1: it is fully redundant with `Obesity`
# (the union is 1,719 either way and no sample loses obese status).
# `Obesity (metadata)` is only 26% populated but is NOT redundant -- it is the
# only record of a study calling a participant obese where no BMI exists.
OBESITY_COLUMNS = ["Obesity", "Obesity (metadata)"]

# Conditions to exclude from health summary figures, over and above the
# eligibility rule.
#
# Which contrasts are shown is normally decided by the rule -- a study
# contributes a stratum only where both arms carry at least CMH_MIN_STRATUM
# samples, and a contrast is shown only where at least MIN_STRATA_TO_PLOT such
# strata exist -- not by name. A name listed here is excluded even if the rule
# admits it, which is what __main__ uses for the no-obese sensitivity run.
#
# "Unhealthy" is listed because the rule cannot express what is wrong with it.
# It is the pooled industrial case group, so it CONTAINS the cases of the four
# focal contrasts; the rule sees 41 qualifying studies and admits it, but it is
# not a fifth independent condition. Two consequences made it unusable:
#
#   * Aggregation. Its composition is 28% obesity, whose list-level shift runs
#     opposite to the other three, so the pooled column cancels to nothing --
#     VANISH reads +0.02 (p = 0.76) there while reading +0.35 in obesity and
#     -0.16 in IBD. A reader takes the pooled null for "no effect in disease".
#   * Double counting. Any per-taxon test treating the columns as independent
#     observations was scoring a replicate of the other four. Dropping it moved
#     the BloSSUM+ driver count from 52 to 0 under the one-sample t-test, which
#     is what exposed that test as unusable on four conditions and prompted the
#     cluster bootstrap in health_driver_bootstrap.
#
# The contrast is still computed and still written to the result CSVs; it is
# only withheld from the reported figures and tables. Nothing else should be
# listed here without a reason the rule cannot express.
HEALTH_PLOT_SKIP: set = {"Unhealthy"}

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Okabe-Ito colour-blind safe palette ───────────────────────────────────────
OI = {
    "orange":    "#E69F00",
    "sky":       "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermillion":"#D55E00",
    "pink":      "#CC79A7",
    "black":     "#000000",
}

# Lifestyle / population groups
#
# Two vocabularies, deliberately. The *keys* below are what analysis code compares
# against and are never rewritten: 852 references across 63 modules test for
# "Transitional" and "H-G", and renaming them in place would silently change
# dictionary ordering, categorical sort order and colour-map lookups. The
# *display* strings are what the metadata file and every published artefact use.
#
# Table M1 now carries the display names, so the alias map below is what allows a
# file written in the manuscript's vocabulary to drive code written in the
# internal one. Matching is case- and separator-insensitive because the same label
# reaches us as "Agro-pastoralist", "Agro-Pastoralist" and "agro pastoralist"
# across the curated file, the archives and the published supplements.
MIDDLE_LIFESTYLE_KEY = "Transitional"
MIDDLE_LIFESTYLE_DISPLAY = "Agro-pastoralist"
HG_LIFESTYLE_KEY = "H-G"
HG_LIFESTYLE_DISPLAY = "Hunter-gatherer"

#: Metabolic Dysfunction-Associated Steatotic Liver Disease -- the 2023
#: replacement for NAFLD. Expanded on first use in each document and in
#: the first figure caption that carries it; bare thereafter.
MASLD_DISPLAY = "MASLD"

LIFESTYLE_ALIASES = {
    "Transitional": MIDDLE_LIFESTYLE_KEY,
    "Agro-Pastoralist": MIDDLE_LIFESTYLE_KEY,
    "Agro-pastoralist": MIDDLE_LIFESTYLE_KEY,
    "Agropastoralist": MIDDLE_LIFESTYLE_KEY,
    "Agro-Pastorialist": MIDDLE_LIFESTYLE_KEY,
    "Agriculturalist": MIDDLE_LIFESTYLE_KEY,
    "H-G": HG_LIFESTYLE_KEY,
    "HG": HG_LIFESTYLE_KEY,
    "Hunter-Gatherer": HG_LIFESTYLE_KEY,
    "Hunter-gatherer": HG_LIFESTYLE_KEY,
    "HunterGatherer": HG_LIFESTYLE_KEY,
    "Hunter Gatherer": HG_LIFESTYLE_KEY,
}

LIFESTYLE_DISPLAY = {
    HG_LIFESTYLE_KEY:     HG_LIFESTYLE_DISPLAY,
    "Hunter-Gatherer":    HG_LIFESTYLE_DISPLAY,
    "Hunter-gatherer":    HG_LIFESTYLE_DISPLAY,
    MIDDLE_LIFESTYLE_KEY: MIDDLE_LIFESTYLE_DISPLAY,
    "Agro-Pastoralist":   MIDDLE_LIFESTYLE_DISPLAY,
    "Agro-pastoralist":   MIDDLE_LIFESTYLE_DISPLAY,
    "Agro-Pastorialist":  MIDDLE_LIFESTYLE_DISPLAY,
    "Industrial":         "Industrial",
}

PLOT_TEXT_REPLACEMENTS = {
    # The internal key stays "Fatty Liver": it seeds DISEASE_NCD_MAP ordering
    # and is a stored value in results/, so the rename belongs at display time.
    # The wrapped form needs its own entry -- three call sites build the label
    # with a newline, which the spaced form would not match.
    "Fatty\nLiver": MASLD_DISPLAY,
    "Fatty Liver": MASLD_DISPLAY,
    "Agro-Pastorialist": MIDDLE_LIFESTYLE_DISPLAY,
    "Agro-Pastoralist": MIDDLE_LIFESTYLE_DISPLAY,
    "Transitional": MIDDLE_LIFESTYLE_DISPLAY,
    "Hunter-Gatherer": HG_LIFESTYLE_DISPLAY,
    "H-G": HG_LIFESTYLE_DISPLAY,
    # The condition is obesity; "Obese" is the internal key (comparisons,
    # stored results) and stays. Also turns "Obese (healthy only)" into
    # "Obesity (healthy only)". Case-sensitive, so "non-obese" is untouched.
    "Obese": "Obesity",
    # Sentence case, to match "Hunter-gatherer" and "Agro-pastoralist". The
    # figures were internally inconsistent: two of the three group names were
    # sentence case and this one title case, across about twenty rendered
    # labels. The rename has to happen here rather than in the source
    # literals because "Non-Industrial" is also a data key -- LIFESTYLE_COLORS,
    # MacroGroup comparisons, stored CSV values, the figure-data frames -- and
    # renaming those would break group matching. This pass rewrites rendered
    # text only, as a substring, so "(Industrial - Non-Industrial)" is caught
    # too, every key is untouched, and the replacement is the same length, so
    # no label reflows.
    #
    # Deliberately NOT renamed: the "Non-Ind" and "NonInd" abbreviations, set
    # where there is no room for the full name and worse for lowercasing.
    "Non-Industrial": "Non-industrial",
}

# lookup key -> canonical, for tolerant matching
_LIFESTYLE_ALIAS_FOLDED = {
    "".join(ch for ch in k.lower() if ch.isalnum()): v
    for k, v in LIFESTYLE_ALIASES.items()
}


def canonical_lifestyle(value):
    """Return the internal lifestyle key used by analysis code."""
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped in LIFESTYLE_ALIASES:
        return LIFESTYLE_ALIASES[stripped]
    folded = "".join(ch for ch in stripped.lower() if ch.isalnum())
    return _LIFESTYLE_ALIAS_FOLDED.get(folded, stripped)


def canonicalize_lifestyle_series(series):
    """Canonicalize known lifestyle aliases while preserving missing values."""
    return series.map(lambda value: value if not isinstance(value, str)
                      else canonical_lifestyle(value))


def display_lifestyle(value):
    """Return the label shown in plots for a lifestyle/group key."""
    canonical = canonical_lifestyle(value)
    return LIFESTYLE_DISPLAY.get(canonical, canonical)


def display_text(text):
    """Apply publication display-label replacements to arbitrary plot text."""
    if text is None:
        return text
    out = str(text)
    for old, new in PLOT_TEXT_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out

LIFESTYLE_COLORS = {
    "H-G":          OI["vermillion"],   # #D55E00  deep orange-red
    "Transitional": OI["yellow"],       # #F0E442  yellow
    "Industrial":   OI["sky"],          # #56B4E9  sky blue (all Industrial)
    "Non-Industrial": OI["orange"],     # #E69F00  orange
    "IndH":         OI["green"],        # #009E73
    "IndU":         OI["blue"],         # #0072B2
    "Ind (all)":    OI["sky"],          # #56B4E9
    "NI":           OI["orange"],       # #E69F00
    "Hadza":        OI["pink"],         # #CC79A7
    "Carter":       OI["pink"],         # #CC79A7
}
# Default colour for any unlisted group (use pink)
LIFESTYLE_COLOR_DEFAULT = OI["pink"]

# VANISH / BloSSUM taxon lists
# VANISH family: Vermillion -> Yellow -> Orange  (depleted in Industrial)
# BloSSUM family: Blue -> Green -> Sky           (enriched in Industrial)
CMH_LIST_COLORS = {
    "VANISH":    OI["vermillion"],   # #D55E00  — core VANISH list
    "VANISH+":   OI["yellow"],       # #F0E442  — extended VANISH+ list
    "VanishH+":  OI["orange"],       # #E69F00  — health-specific VanishH+
    "BloSSUM":   OI["blue"],         # #0072B2  — core BloSSUM list
    "BloSSUM+":  OI["green"],        # #009E73  — extended BloSSUM+ list
    "BlossumH+": OI["sky"],          # #56B4E9  — health-specific BlossumH+
}

# Continent colours
CONTINENT_COLORS = {
    "Africa":        OI["vermillion"],  # #D55E00
    "Asia":          OI["yellow"],      # #F0E442
    "Latin America": OI["orange"],      # #E69F00
    "South America": OI["orange"],
    "North America": OI["blue"],        # #0072B2
    "Europe":        OI["sky"],         # #56B4E9
    "Oceania":       OI["green"],       # #009E73
    None:            "#888888",
    "Other":         "#888888",
}

# Volcano plots — VANISH shades (vermillion family) / BloSSUM shades (blue family)
VOLCANO_PALETTE_7 = {
    "No list":             "#BBBBBB",
    "VANISH only":         OI["vermillion"],  # #D55E00
    "VANISH+ only":        OI["yellow"],       # #F0E442
    "VANISH ∩ VANISH+":    OI["orange"],       # #E69F00
    "BloSSUM only":        OI["blue"],         # #0072B2
    "BloSSUM+ only":       OI["green"],        # #009E73
    "BloSSUM ∩ BloSSUM+":  OI["sky"],          # #56B4E9
}

VOLCANO_DRAW_ORDER_7 = [
    "No list",
    "VANISH only", "BloSSUM only",
    "VANISH+ only", "BloSSUM+ only",
    "VANISH ∩ VANISH+", "BloSSUM ∩ BloSSUM+",
]

VOLCANO_PALETTE = {
    "VANISH":  OI["vermillion"],
    "BloSSUM": OI["blue"],
    "Both":    OI["pink"],
    "Other":   "#AAAAAA",
    "n.s.":    "#DDDDDD",
}

ACCUM_POOLED_SERIES = [
    ("Industrial (all)",          {"Industrial", "Industrial Unhealthy"},         OI["sky"],        "-"),
    ("Non-Industrial (all)",      {"Non-Industrial", "Non-Industrial Unhealthy"}, OI["orange"],     "-"),
    ("Industrial Unhealthy",      {"Industrial Unhealthy"},                       OI["blue"],       "--"),
    ("Non-Industrial Unhealthy",  {"Non-Industrial Unhealthy"},                   OI["vermillion"], "--"),
]

# LIFESTYLE_COLORS and CONTINENT_COLORS now defined above in OI block


COUNTRY_ALIASES = {
    # Maps every observed Country code (old non-standard AND correct ISO3)
    # plus every full country name -> canonical country name.
    # After fix_metadata_countries.py runs, only the ISO3 keys will appear
    # in the data; the legacy keys are kept so old files still load cleanly.

    # ── Europe ────────────────────────────────────────────────────────────────
    "SWE": "Sweden",          "Sweden": "Sweden",
    "DNK": "Denmark",         "DEN": "Denmark",      "Denmark": "Denmark",
    "DEU": "Germany",         "GER": "Germany",       "Germany": "Germany",
    "FRA": "France",          "France": "France",
    "NLD": "Netherlands",     "Netherlands": "Netherlands",
    "GBR": "United Kingdom",  "UK": "United Kingdom", "United Kingdom": "United Kingdom",
    "ESP": "Spain",            "Spain": "Spain",
    "AUT": "Austria",         "AUS": "Austria",       "Austria": "Austria",
    # NOTE: this dataset uses AUS=Austria (NOT Australia). Canonical ISO3 is AUT.
    "ITA": "Italy",            "Italy": "Italy",
    "KAZ": "Kazakhstan",       "Kazakhstan": "Kazakhstan",
    "FIN": "Finland",          "Finland": "Finland",
    "BEL": "Belgium",          "Belgium": "Belgium",
    "ISL": "Iceland",          "Iceland": "Iceland",
    "RUS": "Russia",           "Russia": "Russia",

    # ── Americas ──────────────────────────────────────────────────────────────
    "USA": "United States of America",
    "United States": "United States of America",
    "United States of America": "United States of America",
    "CAN": "Canada",           "Canada": "Canada",
    "BRA": "Brazil",           "Brazil": "Brazil",
    "ARG": "Argentina",        "Argentina": "Argentina",
    "PER": "Peru",             "Peru": "Peru",
    "COL": "Colombia",         "Colombia": "Colombia",
    "SLV": "El Salvador",      "El Salvador": "El Salvador",
    "HND": "Honduras",         "HON": "Honduras",     "Honduras": "Honduras",

    # ── Africa ────────────────────────────────────────────────────────────────
    "ZAF": "South Africa",     "SA": "South Africa",  "South Africa": "South Africa",
    "ETH": "Ethiopia",         "Ethiopia": "Ethiopia",
    "GHA": "Ghana",            "Ghana": "Ghana",
    "TZA": "Tanzania",         "Tanzania": "Tanzania",
    "GNB": "Guinea-Bissau",    "Guinea-Bissau": "Guinea-Bissau",
    "MDG": "Madagascar",       "MAD": "Madagascar",   "Madagascar": "Madagascar",
    "CMR": "Cameroon",         "Cameroon": "Cameroon",
    "LBR": "Liberia",          "LIB": "Liberia",      "Liberia": "Liberia",
    "BFA": "Burkina Faso",     "BKF": "Burkina Faso", "Burkina Faso": "Burkina Faso",
    "COD": "Democratic Republic of the Congo",
    "DRC": "Democratic Republic of the Congo",
    "DRC Congo": "Democratic Republic of the Congo",

    # ── Asia ──────────────────────────────────────────────────────────────────
    "CHN": "China",  "CHI": "China",  "China": "China",
    "JPN": "Japan",            "Japan": "Japan",
    "NPL": "Nepal",            "NEP": "Nepal",        "Nepal": "Nepal",
    "THA": "Thailand",         "Thailand": "Thailand",
    "MNG": "Mongolia",         "Mongolia": "Mongolia",
    "IDN": "Indonesia",        "IND": "Indonesia",    "Indonesia": "Indonesia",
    # NOTE: this dataset uses IND=Indonesia (NOT India). Canonical ISO3 is IDN.
    "ISR": "Israel",           "Israel": "Israel",
    "KOR": "South Korea",      "South Korea": "South Korea",
    "BGD": "Bangladesh",       "Bangladesh": "Bangladesh",
    "PHL": "Philippines",      "PHN": "Philippines",  "Philippines": "Philippines",

    # ── Oceania ───────────────────────────────────────────────────────────────
    "FJI": "Fiji",             "Fiji": "Fiji",
    "NZL": "New Zealand",      "New Zealand": "New Zealand",
}

COUNTRY_TO_CONTINENT = {
    # Keyed by canonical country name (primary) — these are what Country_norm
    # resolves to. ISO3 keys are NOT needed here because get_continent() always
    # receives an already-normalised name; they're only in COUNTRY_ALIASES above.
    # Europe
    "Sweden": "Europe",        "Denmark": "Europe",       "Germany": "Europe",
    "France": "Europe",        "Netherlands": "Europe",
    "United Kingdom": "Europe","Spain": "Europe",
    "Austria": "Europe",       "Italy": "Europe",
    "Kazakhstan": "Europe",    "Finland": "Europe",
    "Belgium": "Europe",       "Iceland": "Europe",       "Russia": "Europe",
    # Americas
    "United States of America": "North America",
    "Canada": "North America",
    "El Salvador": "North America",  "Honduras": "North America",
    "Brazil": "South America",  "Peru": "South America",
    "Argentina": "South America",  "Colombia": "South America",
    # Africa
    "South Africa": "Africa",  "Ethiopia": "Africa",
    "Ghana": "Africa",          "Tanzania": "Africa",
    "Guinea-Bissau": "Africa",  "Madagascar": "Africa",
    "Cameroon": "Africa",       "Liberia": "Africa",
    "Burkina Faso": "Africa",
    "Democratic Republic of the Congo": "Africa",
    # Asia
    "Israel": "Asia",      "China": "Asia",        "Japan": "Asia",
    "Nepal": "Asia",       "Thailand": "Asia",     "Mongolia": "Asia",
    "Indonesia": "Asia",   "South Korea": "Asia",
    "Bangladesh": "Asia",  "Philippines": "Asia",
    # Oceania
    "Fiji": "Oceania",     "New Zealand": "Oceania",
}

# Natural Earth shapefile name -> our canonical name.
# Used by map-figure scripts when joining on the "admin" column.
WORLD_ALIASES = {
    "United States of America": "United States of America",
    "United States":            "United States of America",
    "Dem. Rep. Congo":          "Democratic Republic of the Congo",
    "Congo":                    "Republic of the Congo",
    "Czechia":                  "Czech Republic",
    "Eswatini":                 "Swaziland",
    "S. Sudan":                 "South Sudan",
    "Tanzania":                 "Tanzania",
    "Russia":                   "Russia",
}

# ╔══════════════════════════════════════════════════════════════════╗
# ║  OUTPUT HELPERS                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

SECTION_DIRS = {
    "s1": "section1_diversity",
    "s2": "section2_differential",
    "s3": "section3_health",
}


def outdir(ref_name):
    d = os.path.join("results", ref_name)
    os.makedirs(d, exist_ok=True)
    return d


def section_dir(ref_name, section, sub="main"):
    d = os.path.join("results", ref_name, SECTION_DIRS[section], sub)
    os.makedirs(d, exist_ok=True)
    return d


def outpath(ref_name, filename):
    return os.path.join(outdir(ref_name), filename)
