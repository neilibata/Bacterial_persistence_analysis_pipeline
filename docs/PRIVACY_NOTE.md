# Human-subjects privacy note (Metadata Table M1)

A review of `final_run/metadata_M1.csv` (18,299 samples × 118 columns) ahead of public
release. That is the curated table as deposited; the analysed cohort is the 18,075-sample
subset of it, so this review covers every row a reader can obtain.

## Provenance
Every field is derived from **public SRA/ENA submissions** and the metadata published by
the source studies. The sample identifiers (`SRA_Accession`, `BioSample`, `SRS ID`,
`BioProject ID`, `Paired SRR List`, …) are **public database accessions**, not personal
identifiers, and resolve to records where the same sample metadata is already public.

## What is present
- **No direct identifiers** — no names, dates of birth, addresses, or contact details.
- **Individual-level quasi-identifiers**, as reported by the source studies:
  - `Age` (exact, range 3–93; the <3 exclusion held — 0 rows below 3), `BMI`, `Gender`, `Age group`.
  - Health/disease status across ~76 flag columns (`Unhealthy`, `Obesity`, `disease_*`).
  - Geography: `Country`, `Population`, and `Detailed location` — the last is **study/site
    level** and occasionally city-level (e.g. "Italy: Trento", "Getxo"), not a home address.

## Assessment
Risk is **low**: the dataset re-publishes metadata that is already public in SRA/ENA for
the same accessions, which is standard for metagenomic meta-analyses. There are no direct
identifiers and no fields finer than the source studies released.

## Recommended checks before public deposit (author decision)
1. Confirm the **data-use terms** of the contributing studies permit redistribution of the
   aggregated per-sample metadata (most public SRA metadata does).
2. If extra caution is desired, **coarsen `Detailed location` to country level** and/or
   **bin `Age`** — neither is required by the pipeline (analyses use `Country`/`Lifestyle`
   and depth-corrected richness, not fine location or exact age).
3. Keep the accession columns (they are the reproducibility link back to SRA/ENA).
