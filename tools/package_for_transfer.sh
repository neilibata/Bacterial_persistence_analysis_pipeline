#!/usr/bin/env bash
# Package the pipeline for transfer to another machine (AirDrop, USB, rsync).
#
# Sends: source tree, committed reference inputs, the curated M1, the archive
# caches, the two health-metadata sources, and the 8 matrices the four
# references actually read. Excludes results/ and every figure view a full
# run rebuilds (main_text_figures/, metadata_summary/, supplementary_figures/,
# staged_figures*/, root-level PNGs) -- shipping them let stale copies pass for
# current ones -- plus the superseded dated matrices and scratch dirs.
#
#   bash tools/package_for_transfer.sh [/output/dir]
#   bash tools/package_for_transfer.sh --code-only [/output/dir]
#
# --code-only leaves out the 8 matrices for a machine that already has them,
# but still ships metadata_sources/, the manifest and TRANSFER_MANIFEST.sha256,
# so preflight passes there without anything being supplied by hand.
#
# On the receiving machine see docs/PORTING.md.
set -euo pipefail

# --code-only omits the 8 matrices (14.79 GB) for a receiving machine that
# already has them from an earlier transfer. Everything else is identical --
# metadata_sources/, the manifest and the matrix checksum list included. That
# matters: a hand-rolled lean tar built with ad-hoc rsync flags omitted
# metadata_sources/ and the .sha256 file, and the far side had to supply both
# by hand before it could even run preflight. Leanness belongs in a flag here,
# not in whatever excludes someone types on the day.
CODE_ONLY=0
if [ "${1:-}" = "--code-only" ]; then CODE_ONLY=1; shift; fi

OUT="${1:-$HOME/Desktop}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${MICROBIOME_DATA_ROOT:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/Work/Postdoc_initial_data/bacterial_evolution}"
STAMP="$(date +%Y%m%d_%H%M)"
if [ "$CODE_ONLY" = 1 ]; then
  AR="$OUT/microbiome_pipeline_code_$STAMP.tar.zst"
else
  AR="$OUT/microbiome_pipeline_$STAMP.tar.zst"
fi
STAGE="$(mktemp -d)/pkg"
mkdir -p "$STAGE"

# Untracked files are in the working tree but not on GitHub, so they travel in
# the package and then vanish for anyone who clones. Name them here rather
# than let that be discovered later.
UNTRACKED="$(git -C "$REPO" ls-files --others --exclude-standard \
             -- persistence figures tools 2>/dev/null || true)"
if [ -n "$UNTRACKED" ]; then
  echo "  [WARN] untracked code/data files -- in this package, absent from a clone:"
  echo "$UNTRACKED" | sed 's/^/           /'
fi

echo "staging source tree ..."
rsync -a --quiet \
  --exclude 'results/' --exclude 'results_*/' --exclude 'results_partial_*/' \
  --exclude '.mplconfig*' --exclude 'logs/' --exclude 'logs_*' \
  --exclude 'archive/' --exclude '.rename_backup/' --exclude 'incoming_*/' \
  --exclude 'main_text_figures_pre_*/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude 'combined_*_matrix_*.csv' \
  --exclude '/main_text_figures/' --exclude '/metadata_summary/' \
  --exclude '/supplementary_figures/' --exclude '/staged_figures*/' \
  --exclude '/*.png' \
  --exclude 'mean*_and_std_matrix_*.csv' --exclude 'combined_mean*.csv' \
  "$REPO/" "$STAGE/repo/"

if [ "$CODE_ONLY" = 1 ]; then
  echo "staging the SGB length files (config 'lengths_file' for all 4 refs) ..."
  # These are reference inputs, not matrices: a few hundred KB in total, and
  # excluding them by a *matrix*/*.csv-style glob silently breaks every
  # reference. They are already inside the source tree above; this only
  # asserts they survived.
  for f in "$REPO"/all_SGB_lengths_*.txt; do
    [ -e "$STAGE/repo/$(basename "$f")" ] || cp "$f" "$STAGE/repo/"
  done
  ls "$STAGE/repo"/all_SGB_lengths_*.txt | sed 's|.*/|  |'
  echo "skipping the 8 matrices (--code-only); recording their checksums only"
  mkdir -p "$STAGE/matrices"
fi

if [ "$CODE_ONLY" = 1 ]; then
  echo "checksumming the matrices (not copying them) ..."
else
  echo "staging the matrices the four references read ..."
fi
CODE_ONLY="$CODE_ONLY" python3 - "$REPO" "$STAGE/matrices" "$STAGE" <<'PY'
import hashlib, os, shutil, sys
repo, dest, stage = sys.argv[1], sys.argv[2], sys.argv[3]
code_only = os.environ.get("CODE_ONLY") == "1"
sys.path.insert(0, repo); os.chdir(repo)
from persistence import config

def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()

seen, n, tot, lines = set(), 0, 0, []
for ref, cfg in config.REFERENCES.items():
    for k, v in (cfg.items() if isinstance(cfg, dict) else []):
        if not (isinstance(v, str) and v.endswith(".csv")
                and os.path.exists(v)):
            continue
        base = os.path.basename(v)
        if base in seen:
            continue
        seen.add(base)
        src = os.path.realpath(v)              # follow symlinks
        if not code_only:
            shutil.copy2(src, os.path.join(dest, base))
        tot += os.stat(src).st_size; n += 1
        # The checksum list is written either way. Without it the receiving
        # machine cannot tell a truncated AirDrop from a good one, and a
        # ~50-hour run is the wrong place to find that out.
        lines.append(f"{sha256(src)}  {base}")
        print(f"  {ref:<10} {base}")
with open(os.path.join(stage, "TRANSFER_MANIFEST.sha256"), "w") as f:
    f.write("\n".join(sorted(lines)) + "\n")
verb = "checksummed" if code_only else "staged"
print(f"  {n} matrices {verb}, {tot/1073741824:.2f} GB"
      + ("  (not included in this package)" if code_only else ""))
PY

echo "staging the health-metadata sources ..."
mkdir -p "$STAGE/metadata_sources/plots"
cp -R "$DATA_ROOT/plots/MBIOTEMP_NCBI_metadata" "$STAGE/metadata_sources/plots/" 2>/dev/null \
  || echo "  [warn] MicrobiomeDB dir not found at $DATA_ROOT"
cp "$DATA_ROOT/plots/merged_with_media1.csv" "$STAGE/metadata_sources/plots/" 2>/dev/null \
  || echo "  [warn] merged_with_media1.csv not found"

echo "writing manifest ..."
{ echo "packaged: $(date -u +%FT%TZ)"
  echo "host:     $(hostname)"
  if [ "$CODE_ONLY" = 1 ]; then
    echo "payload:  CODE ONLY -- the 8 matrices are not included; symlink the"
    echo "          existing copies and check them against"
    echo "          TRANSFER_MANIFEST.sha256 before starting the run"
  else
    echo "payload:  full -- source tree, matrices and metadata sources"
  fi
  echo "commit:   $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  _DIRTY="$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)"
  if [ -n "$_DIRTY" ]; then
    echo "          WARNING: tracked files had uncommitted changes"
    echo "$_DIRTY" | sed 's/^/            /'
  else
    echo "          tracked files clean"
  fi
  if [ -n "$UNTRACKED" ]; then
    echo "untracked: present in this package but NOT in the repository, so a"
    echo "           fresh clone will not have them:"
    echo "$UNTRACKED" | sed 's/^/             /'
  fi
  echo "M1:       $(cd "$STAGE/repo" && shasum -a 256 final_run/metadata_M1.csv)"
  # Record the interpreter the pipeline is actually run with, not whatever
  # `python3` happens to resolve to -- they differ here (3.11 env vs 3.9 system)
  # and the wrong number on the receiving end invites a mismatched install.
  # ... which means falling back to `command -v python3` defeats the purpose: on
  # this machine that is 3.9.19 while the pipeline runs on 3.11.11. Prefer the
  # validated env, then PATH, so the line is right without PIPELINE_PYTHON having
  # to be set by hand.
  _VALIDATED="$HOME/anaconda3/envs/plotting_env/bin/python"
  if   [ -n "${PIPELINE_PYTHON:-}" ]; then PYBIN="$PIPELINE_PYTHON"
  elif [ -x "$_VALIDATED" ];          then PYBIN="$_VALIDATED"
  else                                     PYBIN="$(command -v python3)"; fi
  echo "python:   $("$PYBIN" -V 2>&1)  [$PYBIN]"
  echo "validated-on: Python 3.11.11 (conda env 'plotting_env')"
  # Read from the list the staging step wrote, not by globbing the staged
  # matrices: under --code-only that directory is empty and `shasum *.csv`
  # would abort the whole script on `set -e`.
  echo; echo "sha256 of the matrices (also in TRANSFER_MANIFEST.sha256):"
  sed 's/^/  /' "$STAGE/TRANSFER_MANIFEST.sha256"
} > "$STAGE/TRANSFER_MANIFEST.txt"

echo "compressing -> $AR"
mkdir -p "$OUT"
tar -C "$(dirname "$STAGE")" -cf - "$(basename "$STAGE")" \
  | zstd -3 -T0 -o "$AR" -f
rm -rf "$(dirname "$STAGE")"
ls -lh "$AR" | awk '{print "\ndone: "$9"  "$5}'
echo "AirDrop that single file, then follow docs/PORTING.md"
