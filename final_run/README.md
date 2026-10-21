# `final_run/` — the curated metadata, and the inputs it is built from

The pipeline reads its per-sample metadata from `final_run/metadata_M1.csv`:
18,299 rows × 118 columns, one row per sequencing run chosen to represent a
sample. `persistence/config.py` resolves it as `META_FILE`.

## Why that file is not in this repository

It arrives with the data deposit instead:

```bash
bash download_data.sh      # fetches it, verifies it, writes it here
```

Two reasons, neither of them a restriction on who may read it.

It is a **pipeline input, not source code**. At 9.2 MB it belongs with the
eight coverage and depth matrices in the Zenodo record, so there is one
authoritative copy with one checksum rather than two that can drift apart.
`docs/INPUT_CHECKSUMS.tsv` carries its SHA-256, and `tools/preflight.py
--checksums` refuses to start a run against a file that does not match.

It is also **written after the pipeline runs**, not before. Committing it as it
is regenerated would leave every superseded version in the history
permanently, which is why `.gitignore` excludes `metadata_*.csv`; that rule
sequences the deposit and nothing more. See the comment above the pattern.

## What it contains

Individual-level metadata for human subjects: exact age, sex, BMI, country and
study site, and 67 `disease_*` status flags. Every field is derived from the
**public SRA/ENA submissions** and from the tables the contributing studies
published themselves; the sample identifiers are public database accessions,
not personal identifiers, and they resolve to records where the same metadata
is already public. There are no direct identifiers — no names, dates of birth,
addresses or contact details — and no field is finer-grained than its source
study released.

[`../docs/PRIVACY_NOTE.md`](../docs/PRIVACY_NOTE.md) is the full human-subjects
review of the table, field by field.
[`metadata_M1_provenance.txt`](metadata_M1_provenance.txt) records which file
the reported run read, its checksum, and how each column was derived.

`docs/metadata_audited.csv` in this repository holds the same per-sample
columns plus the `qc_*` audit columns, for anyone who wants to inspect the
curation without downloading the deposit; `docs/audit_data_dictionary.md`
defines those columns.

## The files committed here

| File | What it is |
| --- | --- |
| `metadata_M1_provenance.txt` | How the shipped M1 was built, and its checksum |
| `disease_flag_curation.csv` | Per-condition curation decisions behind the `disease_*` flags |
| `final_corrections.csv` | Per-sample metadata corrections applied during curation |
| `sample_id_repair.csv` | Accession and sample-ID repairs applied during curation |

These are small, static curation inputs. They are committed because the
provenance of a correction is part of the code, not part of the data.

## How M1 was built

There is no single builder. `tools/build_curated_m1.py` produces the base
table, and eight scripts then apply the corrections recorded in the files
above, in this order:

```
build_curated_m1.py
  -> build_subject_ids.py          subject identity, one sample per host
  -> build_verified_m1.py          case eligibility
  -> fix_carter_runs.py            Carter run accessions
  -> clean_m1_columns.py           column tidying
  -> fix_unsupported_flags.py      disease flags with no source
  -> split_disease_bd.py           separates two conditions sharing a column
  -> fix_biosample_keyed_row.py    a row keyed on the wrong accession
  -> correct_biosample_mismatches.py
```

Each takes `--apply` and leaves a `metadata_M1.csv.pre_*` backup, so the
backups in a working tree are the chain in order.
[`../docs/M1_data_dictionary.md`](../docs/M1_data_dictionary.md) documents the
columns and says which is authoritative where several compete.
