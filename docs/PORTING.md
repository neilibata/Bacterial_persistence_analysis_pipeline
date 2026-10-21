# Running the pipeline on another machine

The transfer doubles as a portability test: the pipeline reads its inputs by
bare filename relative to the working directory, and a few tools default to an
iCloud path on the original machine. If it runs here, the package is complete.

This document is the step-by-step: every command in order, with the reasoning
behind each one. Work through it one block at a time, checking each result
before starting the next. The README covers the ordinary case, where the inputs
come from the data deposit rather than from a transfer package.

## 1. Unpack

`tar --zstd` needs a `zstd` program on the PATH, which macOS does not provide
(the failure is *Can't initialize filter; unable to run program "zstd -d -qq"*).
Pipe through zstd explicitly; base Anaconda ships one at `~/anaconda3/bin/zstd`:

```bash
ZSTD=$(command -v zstd || ls ~/anaconda3/bin/zstd ~/miniconda3/bin/zstd /opt/homebrew/bin/zstd /usr/local/bin/zstd 2>/dev/null | head -1)
"$ZSTD" -dc microbiome_pipeline_<stamp>.tar.zst | tar -xf -
cd pkg/matrices
shasum -a 256 -c <(grep -A100 'sha256 of the matrices' ../TRANSFER_MANIFEST.txt | tail -n +2)
cd ..
```

The manifest records those hashes as bare filenames (generated with `cd matrices
&& shasum -a 256 *.csv`), so the check has to run from `pkg/matrices` -- from
`pkg` every line reports "No such file or directory". Takes a few minutes over
15 GB, and is worth it before a 50-hour run: AirDrop can truncate.

## 2. Put the matrices where the pipeline looks for them

`config.py` names inputs by bare filename and writes to `results/`, so the
pipeline **must** run from the repo root with the matrices in it. Symlink rather
than copy to avoid a second 15 GB:

```bash
cd repo
for f in ../matrices/*.csv; do ln -sf "$(cd ../matrices && pwd)/$(basename "$f")" .; done
ls -lL *coverage_matrix*.csv | head        # confirm the links resolve
```

## 3. Environment

The pipeline was developed and validated on **Python 3.11.11**, with the exact
library versions in `requirements-pinned.txt`. Use that file, not the unpinned
`requirements.txt`: a fresh install from the latter pulls numpy 2.x and a newer
pandas than this was ever run against.

`cartopy` is listed in `requirements.txt` but is **not installed** in the
validated environment and the pipeline does not need it. Skip it -- it needs GEOS
and PROJ system libraries and is the most likely thing to break an install.

A bare `python3.11` is usually **not** on the PATH even when 3.11 is installed --
`zsh: command not found: python3.11` is the expected first failure. Check what you
actually have before picking a route:

```bash
python3 -V; ls /opt/homebrew/bin/python3.1* /usr/local/bin/python3.1* 2>/dev/null
conda env list 2>/dev/null
```

**Route A -- conda (use this if anaconda is already installed):**

```bash
conda env list | grep -q '^microbiome ' || conda create -y -n microbiome python=3.11
conda activate microbiome
pip install -r requirements-pinned.txt      # always -- also when the env already exists
export MPLCONFIGDIR="$PWD/.mplconfig"
```

Run the `pip install` even when the env exists. `conda activate` does not update
an env, so an env built for an older version of the pipeline keeps whatever it
lacked; that is how a run finished with no Figure 1.

**Route B -- venv, with the real 3.11 path** (substitute what the check above
found; `python3.11` alone will not resolve):

```bash
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pinned.txt
export MPLCONFIGDIR="$PWD/.mplconfig"
```

Either way, confirm the map dependencies landed -- this is what silently skipped
Figure 1 on the second machine, leaving a stale `figure1.png` in place:

```bash
python -c "import geopandas, shapely, plotly; print('map deps OK')"
```

No `MICROBIOME_DATA_ROOT` is needed: `config.resolve_data_root()` tries
`<repo>/../metadata_sources`, which is exactly where the package puts the
MicrobiomeDB download and the da Silva supplement.

## 4. Verify before committing 50 hours

```bash
python tools/preflight.py                              # must end PREFLIGHT PASSED
python -m compileall -q persistence figures tools   # source tree
python -m persistence --help                        # CLI parses
python tools/audit_final_run.py
```

`tools/preflight.py` checks Python 3.11 and the pinned packages, that geopandas
imports and the bundled Natural Earth basemap (`figures/basemap/`) loads, that the
eight matrices resolve, that M1 matches the package manifest, that the external
metadata sources resolve, and free disk space. It exits non-zero on any problem.

**The audit does not pass on a pre-run tree, and it is not supposed to.** Do not
use its totals as the gate -- they move every time a check is added, and they
move often: 70 checks on 2026-09-28, then 99 and 102 within a few hours of each
other on 2026-10-07. Use the rule instead:

**Gate.** Every `FAIL` line must name an artefact under `results/`, which the
package deliberately excludes because the run regenerates it. A failure naming an
import, a package version, a path outside `results/`, or anything in the
`metadata` group means the machine is wrong, not the tree. The absent-`results/`
failures are, by name:

    all 7 demographic outputs written
    disease richness table present
    demographic richness table present
    the run's Figure 5 used the background curve, not the density fallback
    demographic figure still rendered
    demographic figure still publishes a data object
    disease output intact: health_richness_gmw_10pct.csv (+2 more)
    cohort manifest written (combined)

Note `(combined)`, not `(UHGG)`: `config.MAIN_COHORT_REF` is `"combined"`. They
clear once the pipeline has run. What matters here is that **every check in the
`metadata` group passes** -- row count, no dropped curated column, the
`Unhealthy` distribution, `disease_konzo`, Lifestyle canonicalisation and the
Age/Sex/BMI bands. That is the evidence the curated M1 arrived intact.

A check whose inputs are absent prints `----`. Those rows **are counted in the
total**, not excluded from it: the audit prints `len(_rows)` as the total and
increments the failure count only on a hard `False`, so passed + failed + `----`
always equals the number of checks. Every `----` row prints why it was skipped,
and on a pre-run tree every one of those reasons is a missing run output
(`absent`, `no cohort manifest yet`, `... not written yet`). A snapshot,
to show the shape and nothing more: on 2026-10-07 on the development tree the
last line read `77 passed, 11 failed, 102 checks` with 14 `----`
(77 + 11 + 14 = 102). It had read `99` a few hours earlier the same day. Expect
the totals to differ on your tree; expect the ten absent-`results/` failures
named above not to.

One caveat for a tree that carries a *partial* `results/`: the fourth "one
sample per BioSample" check fires for real as soon as that reference's
`cohort_samples.csv` exists. On the development tree a superseded 28 Sep
`results/UHGG/` made it a hard `FAIL` (`one sample per BioSample (UHGG): 34
cohort rows share a BioSample`), which is not an absent-`results/` failure and
should not be waved through as one. On a tree with no `results/` it prints
`----` like its three siblings.

Then the health analysis before all four references:

```bash
python -m persistence --ref combined --only s3      # ~8 h
```

In a tree where Section 2 has not run, that call derives Section 2's four
taxon lists first and then uses them. Section 3 is *defined* on VANISH+,
BloSSUM+, VanishH+ and BlossumH+, so they are an input it has to have, and
deriving them is the same call the full run makes. Add Section 2's own cost to
the estimate above for a first run; a later `--only s3` reuses the `*_ids.txt`
files already under `results/<ref>/section2_differential/main/`. The lists
under `metadata/lists_as_analysed/` are deliberately not used as a substitute:
they are the membership this manuscript reports, kept for comparison, and
running Section 3 over them would report it on a list that was never derived
from the matrices in the tree.

`combined`, not UHGG: `config.SECTION3_REFS` holds Section 3 to the one
reference every health figure and table is published on, and the full run
below skips it elsewhere, so computing it for UHGG would exercise output
nothing reads. An explicit `--only s3` does override the gate, which is the
one way to spend eight hours on a reference the run will not use.

This is not a throwaway smoke test any more. At `HEALTH_BG_N_BOOT = 2000`
the background curve refits on every resample -- ~145 s per
condition-by-list cell, over 54 health cells plus 42 demographic ones and
two detection specs, serially -- so it is about eight of the fifty hours
below. Run it first and the `--resume` in step 5 picks up from it rather
than repeating it. To exercise the contrasts cheaply instead, lower
`HEALTH_BG_N_BOOT` to 200 for this call only and expect roughly a tenth of
the time; do not keep that setting for the run, because at 200 the
percentile interval is coarse enough that the verdict on a near-threshold
cell depends on the seed.

## 5. Full run

Keep this on ONE line -- a pasted backslash continuation gets split by the
shell, which silently runs a truncated command and then tries to execute the log
filename (`exit 127`):

```bash
mkdir -p logs
nohup python -m persistence --all --resume --figure full > logs/final_run_$(date +%Y%m%d_%H%M).log 2>&1 & echo $! > logs/final_run.pid
```

Confirm it is actually running before leaving it:

```bash
sleep 20; kill -0 $(cat logs/final_run.pid) && echo ALIVE; tail -5 logs/final_run_*.log
```

Budget **~50 h** for four references and keep ~50 GB free; `results/` reaches
about 2 GB per run plus figures.

This does not fit in a night. The last full run took 2679 min (44.7 h) with
`HEALTH_BG_N_BOOT = 200`, where Section 3 was ~0.8 h per reference. At 2000 it
is ~7.7 h, now spent once on `combined` instead of four times: about 44.7 − 3
+ 7.7. Start it and leave it; `--resume` checkpoints each stage under
`results/<ref>/.pipeline_state/`, so it survives being interrupted and picks
up where it stopped.

## What is deliberately absent

- `results/` — regenerated by the run.
- The superseded `*_20260812` matrices and the older symlinked variants; only
  the 8 files the four references actually read are included.
- A `chimp` matrix is optional: the pipeline skips any reference whose inputs
  are missing rather than failing.

## Portability: what is guarded, and what is not

These three failure modes are now caught with an explicit error rather than left
to surface as a silent wrong answer.

**1. Missing external metadata sources.** `config.resolve_data_root()` searches,
in order: `$MICROBIOME_DATA_ROOT`, `<repo>/metadata_sources`,
`<repo>/../metadata_sources`, `<repo>/data`, then the original iCloud path. The
transfer package puts the sources where candidate 2 or 3 finds them, so no env
var is needed. If none matches, the three metadata tools exit with the candidate
list and which of them exist.

**2. Missing archive caches.** `tools/build_subject_ids.py` refuses to run if
`docs/archive_subject_ids.tsv` is absent, rather than silently refetching ~18,000
BioSample records. Pass `--refresh-archive` to opt into the refetch (needs
network, ~15-20 min). `docs/ena_run_cache.tsv` and `docs/archive_disease.tsv`
ship in the package alongside it.

**3. Wrong working directory.** `python -m persistence` now checks for
`./persistence/config.py` before doing anything and exits with the expected
path. Previously a run from the wrong directory found no matrices, reported every
reference as input-less, and exited 0 having done nothing. The three metadata
tools `chdir` to the repo root themselves, so they are cwd-independent.

### Still not guarded

- `tools/run_overnight.sh`, `tools/data_progress.sh` and
  `tools/sinf_chain/combined_summary_two_figures.py` contain absolute paths to
  the original machine (a hardcoded interpreter and working directory). They are
  helper scripts, not part of the pipeline, and will need editing if used.
- The Python interpreter path: use the venv from step 3 rather than an absolute
  `python` path.
