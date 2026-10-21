# `metadata/` — the three tables that describe the analysis

Metadata Tables M1–M3 describe, respectively, the **samples**, the **genome
catalogues** and the **taxon lists** the analysis uses. The manuscript calls
these same three tables **Data S2, S3 and S4**; see
[`../docs/naming_crosswalk.md`](../docs/naming_crosswalk.md).

The workbooks themselves live at the repository root, as
`metadata_M2.xlsx` and `metadata_M3.xlsx` — one copy each, because two copies
of a generated table are two things that can disagree. This directory holds
the taxon lists M3 is built from.

| Path | Committed | What it describes |
| --- | --- | --- |
| `lists_as_analysed/` | yes, 8 files | The derived `"+"` and `"H+"` taxon lists the reported run produced — what M3 describes |
| `metadata_M1.csv` | no | Per-sample curated metadata — see [`../final_run/README.md`](../final_run/README.md) |
| `metadata_M1-M3.xlsx` | no | All three tables in one workbook |
| `metadata_M1_provenance.csv` | no | Row-level change log from the M1 builder |

## M1 — the samples

The pipeline reads one per-sample table, `final_run/metadata_M1.csv`, which
`persistence/config.py` resolves as `META_FILE` and which arrives with the data
deposit. [`../final_run/README.md`](../final_run/README.md) explains where it
comes from and what it contains.

The M1 files in this directory are intermediate renderings of the same cohort,
not committed for the reasons given there. `tools/preflight.py --checksums`
verifies the loaded file against `docs/INPUT_CHECKSUMS.tsv`, so a run cannot
quietly proceed on a different one.

## M2 — the reference catalogues

M2 **is** `combined_sgb_metadata.xlsx`, the catalogue workbook at the
repository root, carried through sheet for sheet with the same names, row order
and columns. Only three things are added, which is why it cannot disagree with
the catalogue it describes: a `genome_length_bp` column from
`all_SGB_lengths_*.txt`, an `in_breadth_matrix` column saying whether each SGB
appears as a column of that reference's breadth matrix, and the
`Analysis_link` sheet.

| Sheet | Rows | What it holds |
| --- | --- | --- |
| `Combined`, `Leviatan`, `UHGG` | one per SGB | The SGBs in each catalogue, plus the two added columns |
| `Shared_Pairs` | 2,484 | SGBs present in both source catalogues: their members on each side and the cross-catalogue minimum ANI |
| `Rep_Choice` | 2,484 | Which of the Leviatan or UHGG representative was chosen for each combined SGB, and the coverage that decided it |
| `Fuzzy_Boundary` | 146 | SGBs whose cross-catalogue partners span more than one family — where the species boundary is ambiguous |
| `Analysis_link` | 4 | One row per catalogue: its `--ref` key, SGB and base-pair totals, and the exact input filenames it corresponds to |
| `GTDB_provenance` | 7 | Which GTDB release the lineages came from |
| `README`, `Summary` | — | The sheet descriptions, and the source workbook's own summary report |

## M3 — the taxon lists

Five sheets: `Members` and `Membership` give list membership per SGB,
`Overlaps` gives the pairwise overlaps between lists, and `README` and
`Summary` describe them and record each list's source path.

M3 covers six list families across two reference ID spaces. The curated
**VANISH** (124 taxa) and **BloSSUM** (63 taxa) lists come from
`Vanish_list.txt` and `Blossum_list.txt` at the repository root. The four
derived top-100 lists are `vanish_plus_ids`, `blossum_plus_ids`, `vanishH_ids`
and `blossumH_ids`.

The derived lists are data-dependent: each is the 100 most differentially
prevalent taxa, so membership moves when the cohort or the matrices change.
So *which* membership M3 describes matters, and the answer is
`lists_as_analysed/`: the lists the reported run derived, which is what every
number in the paper was computed from. `metadata_M3.xlsx` at the repository
root is Data S4 and is built from them; the copy in this directory is the
same table.

A run writes its own lists to
`results/<reference>/section2_differential/main/*_ids.txt`, and
`lists_as_analysed/` is the reported run's copy of those.

`tools/make_submission.py` rebuilds M3 from `results/` after a run, and that
build writes `lists_as_analysed/` back from the lists it read, so the workbook
and the snapshot are produced together and cannot drift apart.

## Rebuilding M2 and M3

The defaults rebuild the published pair — `lists_as_analysed` for the
membership, the repository root for the output:

```bash
python tools/build_metadata_tables.py              # -> metadata_M2.xlsx, metadata_M3.xlsx
```

Point `--lists-dir` at a results tree to describe a different run's lists.

`--m1` defaults to `config.META_FILE`, so `--bundle` embeds the M1 the pipeline
actually reads.

M2 needs the breadth matrices in place: its `in_breadth_matrix` column counts
how many SGBs appear as matrix columns, and without them that column rebuilds
empty. `bash download_data.sh` or `setup_symlinks.sh` puts them where it looks.
