# Persistence of Non-industrial Gut Microbiota Discriminates Health and Disease States in Industrial Populations

Analysis pipeline for the study of that name. The preprocessing that
produces its inputs — raw reads to per-sample coverage matrices — is a
separate repository, `Bacterial_persistence_preprocessing`.

It compares gut microbiomes across **18,075 human gut metagenomes** from 83
studies and 42 countries, grouped by the lifestyle of the people who provided
them: hunter-gatherer, agro-pastoralist, or industrial.

Industrial gut microbiomes are widely reported to be missing bacteria that
non-industrial ones carry. Whether that is true, what "missing" means, and
whether it matters for health are three separate questions, and the pipeline
is built to answer them in that order.

It produces the five main-text figures, 31 supplementary figures, the
supplementary tables, and the metadata tables that describe the dataset.

> **Want to look at the data first?** Download
> [`metadata_explorer.html`](https://doi.org/10.5281/zenodo.23444504) and open
> it in any browser — an interactive world map of all 18,075 samples, their
> studies, countries and health categories, in a single file that needs no
> install and no network. See
> [Explore the cohort](#explore-the-cohort-in-your-browser).

## Three questions

**1. Do individual industrial microbiomes really carry fewer species — or does
the comparison only look that way?** (`--only s1`)

Deeper sequencing finds more species, and depth is not evenly distributed
across lifestyles, so a raw richness difference proves nothing. After depth
correction the difference survives: industrial guts carry **61.4 ± 1.3%** of
non-industrial richness (bootstrap permutation, p < 0.002). Within each
lifestyle, individuals differ enormously — samples from one group share only
18–22% of their detected species — which is why population-level questions
cannot be answered from individual samples.

Agro-pastoralist microbiomes turn out not to be intermediate on a lifestyle
gradient but distinct from both ends, displaced off the hunter-gatherer-to-
industrial axis.

**2. Are the missing species actually gone, or just rarer?** (`--only s2`)

A species absent from every individual sample may still be present in the
population. So diversity is modelled as a discovery problem: pool samples, and
ask what the detection curve converges to. Both lifestyles converge on **at
least 99.5%** of the 5,081-SGB combined catalogue. Pooling the full industrial
cohort detects **115 of the 124** original VANISH taxa; pooling the
non-industrial cohort detects **63 of 63** BloSSUM taxa.

So these are taxa whose *prevalence* has shifted, not taxa near extinction —
which is why the pipeline re-derives the lists by prevalence difference rather
than by rarity.

**3. Does carrying these taxa matter for health?** (`--only s3`)

Tested within industrial populations only, where case-control data exist, for
four conditions with both prior evidence of a microbial role and enough
independent studies to stratify: **obesity, glucose intolerance, IBD, and
colorectal cancer/adenoma**. A pooled `Unhealthy` group spanning 26
non-communicable conditions supports the richness comparison.

The answer is condition-dependent rather than a single direction: non-industrial-
associated taxa track with cases in obesity, both families track with cases in
glucose intolerance, and industrial-associated taxa track with cases in CRC and
IBD. Getting that answer required removing a bias specific to this question —
common taxa have more room to fall than rare ones, so a list of common taxa
looks depleted in disease whatever its biology.

### The taxon lists

- **VANISH** taxa are rare in industrial populations, common in non-industrial
  ones. **BloSSUM** taxa are the reverse.
- The published lists (`Vanish_list.txt`, `Blossum_list.txt`) contrast Hadza
  with industrial samples, so they mix the industrial transition with earlier
  changes. The pipeline therefore also derives **VANISH+/BloSSUM+** — the 100
  taxa most shifted in prevalence across the industrial/non-industrial contrast
  — and **VANISH_H+/BloSSUM_H+**, the same contrast against *healthy*
  industrial samples only.
- The sensitivity sweep re-derives all of them under stricter and looser
  criteria and reruns everything downstream, so no conclusion rests on one
  definition.

## The biases, and what each one forces

Each statistical choice in the pipeline answers a specific way the comparison
could mislead. This table is the short version of why the code looks as it
does; `docs/sections.md` gives the tests in full.

| Bias | Why it would mislead | What the pipeline does |
| --- | --- | --- |
| **Sequencing depth** | Deeper samples yield more species; depth differs by lifestyle and between cases and controls | Geometric-mean-target weighting (GMW), applied before any richness or prevalence estimate |
| **Reference catalogue** | One catalogue was built mostly from non-industrial samples, the other mostly from industrial; each inflates its own side | Every sample aligned to both, plus a combined 5,081-SGB reference; a result must hold under all three |
| **Study membership** | Studies differ in population, protocol and depth, so pooling cases across studies confounds health with batch | Study-stratified CMH; a study contributes only with ≥5 cases and ≥5 controls |
| **Baseline prevalence** | A taxon in 95% of controls cannot rise much; one in 2% cannot fall much, so common taxa look depleted in disease regardless of biology | Each list is scored against a fitted expectation for taxa of the same prevalence, not against the raw background |
| **Repeat sampling** | 23 of 83 studies sampled some people more than once, so samples are not independent | Health analyses keep one sample per host; richness is re-tested resampling hosts, and again one-per-host |
| **Detection threshold** | Presence/absence depends on where the line is drawn | Every headline result reported at both 10% genome breadth and 300 kb mapped bases |
| **Read length** | Read counts are not comparable between studies with different read lengths | Sequencing effort is measured in mapped bases throughout |

### How the depth correction is sequenced

Depth is the bias that touches everything, so it is handled before anything is
measured. Three mechanisms, in order, each depending on the last:

1. *Detection is defined explicitly.* A species counts as present when it
   passes a breadth threshold (the fraction of its genome covered — 10% in the
   main text) or a fixed number of mapped bases. Both are configurable, and
   `--detection-mode both` runs them side by side.
2. *Samples are matched on depth.* Industrial and non-industrial samples are
   paired by mapped bases, so comparisons run on subsets where depth is
   equalised by construction rather than adjusted for afterwards.
3. *Richness is reweighted.* Geometric-mean-target weighting (GMW) corrects
   residual depth dependence in richness estimates that matching cannot remove.

### Other choices that shape the output

**The cohort is defined by the data, not by the metadata.** Whether a sample
qualifies depends on how many bases it maps against a given catalogue and how
many species that yields — both of which depend on the catalogue. So each run
writes `results/<ref>/cohort_samples.csv`, the explicit list of samples that
survived, and anything reporting a sample count reads that file. The counts
differ by reference for exactly this reason:

| Reference | Catalogue | Samples | Hosts |
| --- | --- | ---: | ---: |
| `UHGG` | Carter / UHGG, non-industrial-biased | 18,046 | 14,613 |
| `lev` | Leviatan, industrial-biased | 18,060 | 14,613 |
| `combined` | Carter + Leviatan, 5,081 SGBs | **18,075** | **14,624** |
| `chimp` | Chimpanzee, as a negative control | 3,887 | 2,942 |

`combined` is the reference the manuscript reports (`config.MAIN_COHORT_REF`).
The other three are run to show the result does not depend on which catalogue
the reads were mapped against — a real risk, since one catalogue was built
mostly from industrial samples and the other mostly from non-industrial ones.

**Technical replicates are resolved before the statistics, not after.** Some
samples were sequenced more than once. Where two runs of one sample differ
enough to behave as separate observations they are promoted to independent
samples; where a host contributed several samples, health analyses keep one per
host. Both change the cohort size, so both run before anything is counted.

**Every figure saves the data it was drawn from.** Each figure writes a data
object beside itself (`persistence/figure_source.py`), so any figure can be
redrawn without rerunning the analysis, and the submitted figure data is the
same object the figure was drawn from rather than a separate export.

## Running it

Four commands. Step 3 must print `PREFLIGHT PASSED` before step 4.

```bash
# 1  environment — Python 3.11, versions pinned
conda create -y -n microbiome python=3.11 && conda activate microbiome
pip install -r requirements-pinned.txt

# 2  inputs — fetches the data deposit and verifies every checksum.
#    ~6 GB downloaded, 15.9 GB on disk. No Zenodo account needed.
bash download_data.sh

# 3  check the inputs are the ones the reported run used
python tools/preflight.py --checksums

# 4  run — around 50 hours for all four references, needs ~50 GB free
python -m persistence --all --resume
```

Run step 3 after step 2. Before the inputs exist it reports `PREFLIGHT FAILED`
and lists the missing matrices, which is the correct answer to the question it
was asked.

Run everything from the repository root: inputs and `results/` are resolved
relative to the working directory.

### Smaller runs

```bash
python -m persistence --help                        # all options
python -m persistence --ref combined --only s3      # one section, one reference
python -m persistence --only figures                # redraw the main-text figures
python tools/replot_from_data.py S26                # redraw one supplementary figure
```

`--resume` checkpoints each stage under `results/<ref>/.pipeline_state/` and
skips work whose outputs already postdate their inputs. `--only figures`
regenerates whatever analysis the requested figures depend on, and `--only s3`
derives Section 2's four taxon lists first if no run has written them yet,
because Section 3 is defined on them.

The pipeline skips any reference or analysis whose inputs are absent rather
than failing, so a missing `chimp` matrix is not an error.

### Why it takes two days

The last measured full run was 44.7 hours across four references, and the
budget is ~50. Most of the time is Sections 1 and 2 repeated for each
reference (Carter, Leviatan, combined, chimp).

The largest single component is Section 3's prevalence background, estimated by
bootstrap with the curve refitted on every resample. At
`config.HEALTH_BG_N_BOOT = 2000` that is about 145 seconds per
condition-by-list cell, across 54 health cells, 42 demographic cells and two
detection specifications — about 7.7 hours, in a serial loop, so wall clock.
`config.SECTION3_REFS` restricts Section 3 to `combined`, the only reference it
is published on; computing it for the other three would add roughly 23 hours
and change nothing that is reported.

[`docs/PORTING.md`](docs/PORTING.md) has the step-by-step for running this on a
machine that has never run it before, including the parts that have actually
gone wrong.

## What comes out

```
results/<ref>/          Per-reference analysis output: tables, figures, data objects
  cohort_samples.csv      The samples that survived, for this reference
  .pipeline_state/        --resume checkpoints
main_text_figures/      Figures 1-5
supplementary_figures/  All 31 supplementary figures, flat and in order
staged_figures/         The same figures resampled to 6.3 in for the manuscript
```

`python -m persistence` writes the analysis, the figures and the data objects.
It does **not** write the journal-facing artifacts — the figure-data workbook,
the slot-keyed inventories, the reproducibility manifest. Those are a separate
step, so that they cannot be quietly skipped between a run and a deposit:

```bash
python tools/make_submission.py --check   # what is missing or stale
python tools/make_submission.py           # produce it
```

`--check` exits non-zero if anything is missing or older than the run that
should have produced it, so it works as a gate before packaging a deposit.

### Explore the cohort in your browser

The quickest way to see what 18,075 metagenomes across 42 countries actually
look like. **Download
[`metadata_explorer.html`](https://doi.org/10.5281/zenodo.23444504) from the
results record and open it** — one file, no install, no network, no server.

It is a self-contained Equal-Earth world map of the sample metadata:
per-country and per-study sample counts, the distribution and completeness of
every metadata field, and the health categories the analysis reports. Use it
to check whether a country, study or disease group is represented before
reading any result that depends on it.

To rebuild it from a finished run:

```bash
python tools/metadata_explorer/build_explorer.py    # -> a standalone HTML map
```

It restricts to `results/combined/cohort_samples.csv` so its totals match the
manuscript's; given no such file it reports the full curated table instead and
says so in the page footer. The four `world_map_*.html` pages in the same
record give the per-reference sample maps.

## Repository layout

```
persistence/            The analysis package — `python -m persistence`
  __main__.py             Entry point; run_reference() is the spine
  config.py               Every threshold and path; CLI flags override these
  loading.py              Builds the ReferenceData object everything else reads
  section1_*.py           Diversity, richness, accumulation
  section2_*.py           Differential prevalence
  section3_health*.py     Health associations
  figures/                In-package plotting helpers for Sections 1 and 2
figures/                Standalone figure scripts, run as subprocesses
tools/                  Everything run before or after the pipeline:
                        builders, audits, deposit assembly, one-off checks
docs/                   Reference documentation (see below)
metadata/               Metadata Tables M2 and M3
final_run/              Curation inputs for Metadata Table M1
```

The three vocabularies in play — the manuscript's, this repository's and the
deposit's — are lined up in
**[docs/naming_crosswalk.md](docs/naming_crosswalk.md)**. Figure S31 is
`figure_s0_roadmap.png` here and `S31_figure_s0_roadmap.png` in the deposit,
and Metadata Tables M1, M2 and M3 are the manuscript's Data S2, S3 and S4.

**[docs/code_map.md](docs/code_map.md)** maps each published figure, table and
artifact to the code that produces it. Both documents are generated from the
manifests rather than maintained by hand, so neither can drift:

```bash
python tools/build_naming_crosswalk.py
python tools/build_provenance_map.py
```

## Inputs

The repository carries the small, static inputs that define the analysis: the
VANISH and BloSSUM lists, the genome-length tables, and the catalogue metadata.

`download_data.sh` fetches the rest from the data deposit:

| Input | Size | What it is |
| --- | --- | --- |
| 8 breadth/depth matrices | 15.9 GB | Per-sample × per-species coverage and depth |
| `final_run/metadata_M1.csv` | 9.2 MB | Curated per-sample metadata — see [`final_run/README.md`](final_run/README.md) |

`bash setup_symlinks.sh /path/to/data` links an existing local copy instead.

Rebuilding M1 from the third-party exports it was curated from additionally
requires `metadata_sources/`. Running the pipeline does not: it reads the
finished M1, so preflight reports `metadata_sources/` as a warning and
continues.

### Confirming the inputs are the right ones

Three files named `metadata_M1.csv` describe three different cohorts, so the
pipeline identifies its inputs by hash, not by filename:

- `docs/INPUT_CHECKSUMS.tsv` pins the size, md5 and sha256 of all nine inputs.
  Three of them — both `combined_*` matrices and the metadata — carry
  `attested_by=run_manifest`, meaning that sha256 also appears in the reported
  run's `verification_manifest.json` and identifies the exact file that
  produced the published numbers.
- `tools/preflight.py` hashes the metadata on every invocation and checks the
  matrices by size. `--checksums` hashes the matrices too — a few minutes, and
  the thing to run once after a download.
- `persistence/config.py` resolves `META_FILE` to the first existing path in
  `META_FILE_CANDIDATES` and raises a `RuntimeWarning` naming the file it used
  whenever that is not `final_run/metadata_M1.csv`.

## Verifying reproducibility

`verification_manifest.json` pins hashes for 119 files: the 9 inputs, 25 result
tables, the 5 main-text and 31 supplementary figures, the 36 figure data
objects, and the 8 pinned taxon lists.

```bash
python tools/verify_reproducibility.py            # repo only, ~1 min
python tools/verify_reproducibility.py --tier 2   # + rerun a section, ~20 min
```

Two hashes are recorded per file: a `sha256` over raw bytes, and a `value_hash`
over the parsed table with floats rounded to 8 significant figures, which
survives platform differences in float formatting.

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/sections.md`](docs/sections.md) | Every `--only` key, the sensitivity sweep, the output layout |
| [`docs/naming_crosswalk.md`](docs/naming_crosswalk.md) | What each figure, table and data file is called in the manuscript, here, and in the deposit |
| [`docs/code_map.md`](docs/code_map.md) | Which code produces which published item |
| [`docs/PORTING.md`](docs/PORTING.md) | Running the pipeline on another machine, step by step |
| [`docs/M1_data_dictionary.md`](docs/M1_data_dictionary.md) | The metadata columns, which is authoritative where they compete |
| [`docs/preprocessing_methods.md`](docs/preprocessing_methods.md) | Raw reads to coverage matrices: the upstream half of the method |
| [`docs/audit_data_dictionary.md`](docs/audit_data_dictionary.md) | The `qc_*` columns on the audited metadata, and the per-study audit table |
| [`docs/figure_style.md`](docs/figure_style.md) | How figures are sized and typeset, and why |
| [`docs/PRIVACY_NOTE.md`](docs/PRIVACY_NOTE.md) | Human-subjects review of the metadata |

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
