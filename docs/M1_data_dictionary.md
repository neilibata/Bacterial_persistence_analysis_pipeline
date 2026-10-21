# Metadata Table M1 — data dictionary

`final_run/metadata_M1.csv`. 18,299 rows × 118 columns, one row per sequencing
run selected to represent a sample. It is built by a chain of nine tools, not
one: `tools/build_curated_m1.py`, then `tools/build_subject_ids.py`,
`tools/build_verified_m1.py`, `tools/fix_carter_runs.py`,
`tools/clean_m1_columns.py`, `tools/fix_unsupported_flags.py`,
`tools/split_disease_bd.py`, `tools/fix_biosample_keyed_row.py` and
`tools/correct_biosample_mismatches.py` (each takes `--apply` and backs up first,
so the eight `final_run/metadata_M1.csv.pre_*` files are the chain in order).
`final_run/metadata_M1_provenance.txt` records the shipped file's hash and
headline counts; `tools/audit_final_run.py` is the check to run after a rebuild.

---

## Which column is authoritative

Several groups of columns describe the same thing. Use the authoritative one.

| question | use this | not these |
|---|---|---|
| Is the sample industrial? | **`Lifestyle`**, via `config.canonicalize_lifestyle_series` | `Country`, `Population` |
| Is the host healthy? | **`Unhealthy`** (`Healthy` / `Unhealthy`) | the three `Diseased*` columns, `disease [MBIOTEMP_disease]`, `mdb_study_condition` |
| Does the host have condition X? | **`disease_<X>`** | the `Diseased*` roll-ups |
| Is the host obese? | **`Obesity`** ∪ **`Obesity (metadata)`** ∪ `BMI ≥ 30` — the union, as `config.OBESITY_COLUMNS` | either column alone; `Overweight` is **not** obesity and **not** a reason to be Unhealthy |
| Which row joins to the matrices? | **`Chosen_match_ID`** (unique, 18,299 distinct, 0 collisions) | `Sample ID` (1,123 duplicates — it is a *subject* id in some studies) |
| Which study? | **`Study`** (numeric; what CMH strata use) | `StudyFolder` (a trailing 3-letter country code is a population split *within* one study) |
| Which person? | **`Subject_ID`** | `Sample ID` — it cannot see repeated sampling: iMSMS_2022 has 1,133 unique `Sample ID`s over 611 people |

## `Unhealthy` — the definition

> **Unhealthy = any named condition OR obesity (BMI ≥ 30 or a curated obesity flag)**

Not derivable from the `disease_*` flags alone: 1,439 Unhealthy samples have no
disease flag and are obesity-only. Audited invariants, all currently hold:

- every Unhealthy sample has a reason — a disease flag, or obesity (**0** without)
- no Healthy sample carries an obesity flag or BMI ≥ 30 (**0**)
- 7 samples carry a disease flag and are still Healthy, deliberately: Yachida's
  `few_polyps` ×5 (corroborated exactly by MicrobiomeDB; a near-normal
  colonoscopy), plus single `coeliac`/`CMV`/`gestational_diabetes`/
  `irritable_bowel` cases
- `Overweight` alone is **never** a reason; 7 such labels were corrected to Healthy

## The three `Diseased*` roll-ups are DERIVED

They are views of the `disease_*` flags, computed by
`tools/_disease_categories.py`, not independent curation:

| column | definition |
|---|---|
| `Diseased (gut and metabolic)` | any of the 19 gut + 12 metabolic flags |
| `Diseased (other)` | any remaining flag |
| `Diseased (metabolic) or obese` | any metabolic flag, or obesity |

They were hand-curated and had drifted badly — `Diseased (gut and metabolic)`
marked 858 flagged samples `No` and 39 unflagged `Yes`; `Diseased (other)` was 802
rows out with 741 `Yes` carrying no flag at all. Deriving them means a flag
correction propagates. `audit_final_run.py` checks they still match.

**The classification is a judgement.** Edit `tools/_disease_categories.py` and
both builders follow.

## `disease_*` flags — provenance and known limits

Two sources, recorded per sample in **`health_source`**:

- **MicrobiomeDB** — 61 of the 69 flags, exploded from its free-text `disease`
- **da Silva et al. 2026** — 8 flags: `Crohn's`, `MS`, `Ulcerative_colitis`,
  `ankylosing_spondylitis`, `myalgic_encephalomyelitis`, `bone_disease` and
  `Behcets` (the two halves of the former `BD`, see below), and `konzo`/
  `Worm-Infected` from elsewhere

Caveats a user needs:

- **`disease_BD` has been split** into `disease_bone_disease` (54, RettedalAE's
  osteoporosis cohort, PMID 33778322) and `disease_Behcets` (20, YeZ's Behçet's
  cohort, PMID 30077182). It previously held both, and the codebase carried three
  different readings of it at once.
- **`disease_ME_CFS` and `disease_myalgic_encephalomyelitis` are the same
  condition from the two sources.** Zero overlap, partitioned by study. Merging
  them double-counts.
- **`health_source` records which source had a matching row, not which supplied a
  value.** Several studies read `dasilva` although da Silva has no column for the
  condition in question.
- 26 flags are verifiable by **neither** source, including `coeliac`,
  `irritable_bowel`, `gestational_diabetes`, `konzo`. They retain their inherited
  values.
- 126 flag instances were cleared as unsupported by any source — `schizofrenia`
  ×27 in an RA cohort, `metastases` ×39 in a probiotic-engraftment study, `STH`
  ×30 in an osteoporosis cohort, `treatment_colitis` ×15 in MetaHIT, `ACVD` ×15 in
  a lifestyle meta-analysis.

## Subject identity and non-independence

| column | meaning |
|---|---|
| `Subject_ID` | `<study root>\|<subject>`; falls back to `Chosen_match_ID` when unresolved, so it is always usable |
| `Subject_ID_source` | where the key came from; `unresolved` for 5,061 samples |
| `subject_primary` | `Yes` for the one row representing that subject (14,700); `No` for 3,599 |

**24 studies contribute several samples per person** — Lloyd-PriceJL_2019 is 1,125
samples from 104 people, HallAB_2017 259 from 32. Use `subject_primary` for any
host-level inference. Do **not** use it for accumulation curves: there, repeated
samples simply add fewer new taxa, so including them is conservative.

`mdb_case_eligible` marks 221 samples that are neither clean cases nor clean
controls — pre-disease states (`LiJ_2017`'s 56 pre-hypertensives) and intervention
arms (post-FMT in `LiSS_2016`, `IaniroG_2022`; post-surgery in `YachidaS_2019`).
Exclude them from both arms of a case/control test.

## `Unhealthy` is blank on 344 rows

Neither `Healthy` nor `Unhealthy`: Valles-ColomerM_2023_ITA 165,
BeghiniF_2023_HND 105, ZeeviD_2015_ISR 37, YachidaS_2019_JPN 30, and a few more.
**None carries a disease flag and none is obese**, so no hidden Unhealthy is
lurking there — the status is genuinely unknown. But `Unhealthy == "Healthy"`
excludes them from *both* arms of a health contrast, so decide explicitly rather
than by accident.

## Traps

- **`Lifestyle` holds display labels** (`Hunter-gatherer`, `Agro-pastoralist`),
  not the analysis keys (`H-G`, `Transitional`). Always read it through
  `config.canonicalize_lifestyle_series`.
- **`Total bases` is the best available value — do not "correct" it against ENA.**
  It disagrees with `number_reads × median_read_length` by >3× on 211 rows, and
  the per-study median ratio varies systematically (Carter ~2.0, RettedalAE 3.97,
  NielsenHB 0.86, LiSS_2016 0.68), which looks like a defect. It was tested at
  scale against ENA `base_count` for all 87 BioProjects (113,136 runs, cached in
  `docs/ena_base_counts.tsv`) under three readings — a single run, the sum of a
  row's listed runs, and the sum over its sample accession. **All three are worse
  than the column as it stands:**

  | candidate | within 10% of `reads × len` | >3× off |
  |---|---|---|
  | **current `Total bases`** | **16,154** | 211 |
  | single run (`SRA_Accession`) | 12,540 | 837 |
  | sum of listed runs | 14,023 | **194** |
  | sum over sample accession | 14,276 | 352 |

  A fix limited to the 211 bad rows would correct only 10 of them. The likely
  explanation is that `Total bases` reflects the FASTQ actually downloaded and
  processed, whereas ENA `base_count` is what the submitter declared — related but
  different quantities. Either way the analysis is unaffected: `loading.py` derives
  the depth used by GMW and depth matching from the depth MATRIX
  (`Σ mean_depth × genome_length`), never from this column, noting that "metadata
  columns are sometimes mismatched across studies".
- The 209 Gbp maximum is **correct**: `Sample ID`
  `Hadza_MoBio_hadza-A-D_C_6_1128` really is a 1.56-billion-read run
  (ERR7803603). Its `Chosen_match_ID` is the BioSample accession SAMEA4765588
  because that run is not in the matrices and the BioSample is — 260 matrix rows
  are legitimately keyed that way.
- `BMI` has four implausible values blanked (0.0, 62.9, 65.7, 83.0); the
  originals are in **`BMI (raw)`**.
- Sparse columns: `Secondary_run_IDs` 10%, `Obesity (metadata)` 26%,
  `Antibiotics` 1%. `Antibiotics` is retained but unusable as a covariate.
- 17 duplicate `BioSample` values remain, covering 34 rows (one BioSample, two
  libraries — see `tools/fix_carter_runs.py`). `Chosen_match_ID` is unique.
- 17 of 118 column names need quoting (spaces, brackets, parentheses).

## Column inventory

| column | populated | values |
|---|---|---|
| `Master_ID` | 100% | 18,299 distinct |
| `Chosen_match_ID` | 100% | 18,299 distinct |
| `Secondary_run_IDs` | 10% | 1,790 distinct |
| `SRA_Accession` | 100% | 18,299 distinct |
| `Sample ID` | 100% | 17,176 distinct |
| `SRS ID` | 99% | 18,157 distinct |
| `Author` | 100% | 80 distinct |
| `Year` | 100% | numeric 2011 – 2023 |
| `StudyFolder` | 100% | 101 distinct |
| `Study` | 100% | numeric 1 – 84 |
| `Population ID` | 100% | numeric 1 – 107 |
| `PubMed ID` | 100% | numeric 2.204e+07 – 3.957e+07 |
| `BioProject ID` | 100% | 88 distinct |
| `Population` | 100% | 46 distinct |
| `Detailed location` | 74% | 180 distinct |
| `Country` | 100% | 43 distinct |
| `Country (full)` | 100% | 43 distinct |
| `Lifestyle` | 100% | `Agro-pastoralist`, `Hunter-gatherer`, `Industrial`, `Unclear` |
| `Age` | 69% | numeric 3 – 93 |
| `Age group` | 96% | 6 distinct |
| `Gender` | 87% | `Female`, `Male`, `unknown` |
| `Unhealthy` | 98% | `Healthy`, `Unhealthy` |
| `Diseased (gut and metabolic)` | 100% | `No`, `Yes` |
| `Diseased (metabolic) or obese` | 100% | `No`, `Yes` |
| `Diseased (other)` | 100% | `No`, `Yes` |
| `BMI` | 63% | numeric 11.16 – 59.58 |
| `Obesity (metadata)` | 26% | `No`, `Yes` |
| `Obesity` | 85% | `No`, `Yes` |
| `Overweight` | 85% | `No`, `Yes` |
| `Antibiotics` | 1% | `No` |
| `NumPairedRuns` | 75% | numeric 1 – 38 |
| `LineType` | 75% | `Merged SRS (paired)`, `Multiple paired runs`, `Single paired run` |
| `Total bases` | 100% | numeric 2.086e+06 – 2.092e+11 |
| `Paired SRR List` | 100% | 18,299 distinct |
| `BioSample` | 100% | 18,282 distinct |
| `study_name [MBIOTEMP_study_name]` | 50% | 53 distinct |
| `disease [MBIOTEMP_disease]` | 50% | 65 distinct |
| `number_reads` | 100% | numeric 1.715e+04 – 1.561e+09 |
| `median_read_length` | 100% | numeric 44 – 302 |
| `non_westernized` | 48% | `No`, `Yes` |
| `median_read_length_source` | 100% | `original`, `study_median` |
| `number_reads_source` | 100% | `computed`, `original` |
| `Subject_ID` | 100% | 14,700 distinct |
| `Subject_ID_source` | 100% | 9 distinct |
| `health_source` | 100% | 7 distinct |
| `mdb_study_condition` | 52% | 27 distinct |
| `mdb_case_eligible` | 52% | `No:intervention`, `No:pre-disease`, `Yes` |
| `subject_primary` | 100% | `No`, `Yes` |
| `BMI (raw)` | 63% | numeric 0 – 83.03 |

69 `disease_*` flag columns, `Yes`/`No`. With the 49 columns above, that is the
full 118.

