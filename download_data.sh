#!/usr/bin/env bash
# Download the large external inputs (breadth/depth matrices + curated sample
# metadata) that are not distributed in git, and place them where the pipeline
# opens them.
#
# The deposit holds the eight matrices gzipped, one file each, plus
# metadata_M1.csv uncompressed so it can be previewed in the browser, plus
# DEPOSIT_MANIFEST.tsv. That manifest names which deposited file belongs at
# which path, with an md5 for the deposited file and for the decompressed
# result. This script checks each file, writes it to the path the pipeline
# opens, and re-checks the result against the md5 of the file the reported run
# actually consumed. It refuses to continue on any mismatch.
#
# The manifest also supports a file arriving in several numbered pieces, for a
# deposit too large to upload whole. The published record does not use that,
# but the reassembly path is kept because the manifest format allows it.
#
# Fill ZENODO_RECORD once the deposit exists.
#
# Usage:
#   bash download_data.sh                 # uses the record id below
#   bash download_data.sh 1234567         # override record id
#   KEEP_PARTS=1 bash download_data.sh    # keep the downloaded .gz afterwards
#
# Disk: about 6.1 GiB downloaded plus 15.9 GB decompressed. The downloaded
# copies are deleted once their destination verifies, unless KEEP_PARTS=1.
#
# Alternatively, if you already have the files locally, link them instead:
#   bash setup_symlinks.sh /path/to/data
set -euo pipefail

# Record A of the deposit (the pipeline inputs). Zenodo keeps the
# deposition id as the record id when a draft is published, so this is
# the same number before and after. Confirm it resolves to a public
# record before relying on it: https://zenodo.org/records/23437518
ZENODO_RECORD="${1:-23437518}"
DEST="$(cd "$(dirname "$0")" && pwd)"
KEEP_PARTS="${KEEP_PARTS:-0}"
MANIFEST="DEPOSIT_MANIFEST.tsv"

# The destinations persistence/config.py opens, relative to the repo root
# (every path in config is cwd-relative). Spelled out rather than described
# because the description drifted once: this script used to say the deposit
# held "combined_mean{coverage,depth}_matrix_*" while config has always opened
# combined_coverage_matrix_/combined_depth_matrix_, so a deposit built from the
# old message would load UHGG, lev and chimp and fail on `combined` -- the one
# reference the manuscript reports.
#
# tools/preflight.py checks this same list against config at runtime; if you
# change a filename, change it in config and let preflight tell you.
REQUIRED_FILES=(
  "meancoverage_and_std_matrix_high_UHGG_20260820.csv"
  "meandepth_and_std_matrix_high_UHGG_20260820.csv"
  "meancoverage_and_std_matrix_high_lev_20260820.csv"
  "meandepth_and_std_matrix_high_lev_20260820.csv"
  "meancoverage_and_std_matrix_high_chimp_merged.csv"
  "meandepth_and_std_matrix_high_chimp_merged.csv"
  "combined_coverage_matrix_20260820.csv"
  "combined_depth_matrix_20260820.csv"
  "final_run/metadata_M1.csv"
)

# macOS ships `md5 -q`, Linux ships `md5sum`. Neither is everywhere, so pick
# at run time rather than assuming the author's machine.
if command -v md5sum >/dev/null 2>&1; then
  md5_of() { md5sum "$1" | cut -d' ' -f1; }
elif command -v md5 >/dev/null 2>&1; then
  md5_of() { md5 -q "$1"; }
else
  echo "Neither md5sum nor md5 is available; cannot verify the download." >&2
  exit 1
fi

# A record id is all digits. This used to compare against a placeholder
# string, which stopped being reachable once the default became the real
# record -- so it checked nothing. Validating the shape instead still catches
# the mistake someone actually makes: passing a DOI, a URL, or an empty
# argument where the id belongs.
case "$ZENODO_RECORD" in
  ''|*[!0-9]*)
    echo "Not a Zenodo record id: '${ZENODO_RECORD}'"
    echo "Pass the numeric id, e.g.  bash download_data.sh 23437518"
    echo "(from https://zenodo.org/records/<id>, not the DOI or the full URL)"
    echo
    echo "The deposit reassembles into exactly these files:"
    for f in "${REQUIRED_FILES[@]}"; do echo "    $f"; done
    echo
    echo "The eight matrices belong at the repo root; metadata_M1.csv belongs in final_run/."
    exit 1 ;;
esac

# SKIP_DOWNLOAD=1 means the files are already in DEST. Everything after this
# point then runs unchanged, which is useful for checking an existing copy
# without fetching it again.
# Fetch every file in the record using only what a machine running this
# pipeline already has. `zenodo_get` is pleasant but it is one more thing to
# install, and requiring it meant this script stopped dead -- "zenodo_get not
# found" -- for anyone following the README in order. Zenodo's public record
# API needs no token and no account for a published record, so curl plus the
# python3 this pipeline already requires is enough.
fetch_record() {
  local rec="$1" dest="$2"
  local api="https://zenodo.org/api/records/${rec}"
  echo "  reading the file list from ${api}"
  local listing
  if ! listing=$(curl -fsSL --retry 5 --retry-delay 3 "${api}" \
                 | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit("  the record did not return JSON; is the id right, and is the "
             "record published?")
files = d.get("files") or []
if not files:
    sys.exit("  the record lists no files. An unpublished draft is visible "
             "only to its owner, so check the record is public.")
for f in files:
    key = f.get("key") or f.get("filename")
    link = (f.get("links") or {}).get("self") or ""
    if key and link:
        print(f"{key}\t{link}")
'); then
    echo "  could not list the record's files." >&2
    return 1
  fi
  [ -n "$listing" ] || { echo "  empty file list" >&2; return 1; }

  local n=0
  while IFS=$'\t' read -r key link; do
    [ -n "$key" ] || continue
    n=$((n + 1))
    printf '  [%d] %s ' "$n" "$key"
    # -C - resumes a partial file, so an interrupted run costs only what it
    # had not already fetched. The checksum pass below is what decides
    # whether the result is usable, so a resumed file is still verified.
    if curl -fsSL --retry 5 --retry-delay 3 -C - \
            -o "${dest}/${key}" "${link}"; then
      echo "ok"
    else
      # curl exits 33 when the server will not honour a range request and
      # the local file is already complete; retry once from scratch.
      if curl -fsSL --retry 5 --retry-delay 3 -o "${dest}/${key}" "${link}"; then
        echo "ok"
      else
        echo "FAILED"
        return 1
      fi
    fi
  done <<< "$listing"
  echo "  fetched ${n} file(s)"
}

if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "SKIP_DOWNLOAD=1: using the files already in ${DEST}"
else
  echo "Downloading Zenodo record ${ZENODO_RECORD} -> ${DEST}"
  if command -v zenodo_get >/dev/null 2>&1; then
    zenodo_get -r "${ZENODO_RECORD}" -o "${DEST}"
  elif command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    fetch_record "${ZENODO_RECORD}" "${DEST}" || {
      echo
      echo "The automatic fetch failed. You can download the files by hand from"
      echo "  https://zenodo.org/records/${ZENODO_RECORD}"
      echo "into ${DEST}, then re-run with SKIP_DOWNLOAD=1 to verify and"
      echo "reassemble them."
      exit 1
    }
  else
    echo "Need either zenodo_get, or curl and python3, to fetch the record."
    echo "  pip install zenodo_get"
    echo "or download by hand from https://zenodo.org/records/${ZENODO_RECORD}"
    echo "into ${DEST} and re-run with SKIP_DOWNLOAD=1."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Reassemble
# ---------------------------------------------------------------------------
# A record without the manifest is one of the older whole-file deposits; the
# checks below still apply, there is just nothing to put together.
if [ -s "${DEST}/${MANIFEST}" ]; then
  echo
  echo "Reassembling from ${MANIFEST}"

  # Every part must be present and intact before anything is concatenated. A
  # truncated part would otherwise produce a corrupt CSV that only shows up as
  # a strange number much later in the analysis.
  bad=0
  while IFS=$'\t' read -r dest part part_md5 part_bytes dest_md5 dest_bytes; do
    [ "$dest" = "dest" ] && continue
    if [ ! -s "${DEST}/${part}" ]; then
      printf '  MISSING part %s\n' "$part"; bad=$((bad + 1)); continue
    fi
    got=$(md5_of "${DEST}/${part}")
    if [ "$got" != "$part_md5" ]; then
      printf '  CORRUPT part %s\n    expected %s\n    got      %s\n' \
        "$part" "$part_md5" "$got"
      bad=$((bad + 1))
    fi
  done < "${DEST}/${MANIFEST}"

  if [ "$bad" -gt 0 ]; then
    echo
    echo "${bad} part(s) missing or corrupt. Re-run the download; do not"
    echo "start a run from a partial deposit."
    exit 1
  fi
  echo "  all parts present and checksummed"

  # Group the manifest by destination, preserving part order, and rebuild each
  # one. awk does the grouping so the loop stays one pass per destination.
  for dest in $(awk -F'\t' 'NR>1 && !seen[$1]++ {print $1}' "${DEST}/${MANIFEST}"); do
    want=$(awk -F'\t' -v d="$dest" 'NR>1 && $1==d {print $5; exit}' "${DEST}/${MANIFEST}")
    parts=$(awk -F'\t' -v d="$dest" 'NR>1 && $1==d {print $2}' "${DEST}/${MANIFEST}")

    if [ -s "${DEST}/${dest}" ] && [ "$(md5_of "${DEST}/${dest}")" = "$want" ]; then
      printf '  = %s already present and correct\n' "$dest"
    else
      mkdir -p "$(dirname "${DEST}/${dest}")"
      printf '  > %s ' "$dest"
      # A destination may be deposited uncompressed -- small tables go up as
      # plain .csv so they can be previewed in the browser rather than
      # downloaded and unpacked. Decide by what the manifest actually names,
      # not by assumption: feeding plain text to `gzip -cd` would fail, and
      # feeding a .gz through `cat` would leave a compressed file where the
      # pipeline expects CSV.
      first=$(printf '%s\n' $parts | head -1)
      case "$first" in
        *.gz|*.gz.p[0-9][0-9][0-9])
          # Concatenate the parts into one gzip stream and decompress on the
          # way through, so the parts never also exist as a ~16 GB .gz.
          ( cd "${DEST}" && for p in $parts; do cat "$p"; done ) \
            | gzip -cd > "${DEST}/${dest}" ;;
        *)
          ( cd "${DEST}" && for p in $parts; do cat "$p"; done ) \
            > "${DEST}/${dest}" ;;
      esac
      got=$(md5_of "${DEST}/${dest}")
      if [ "$got" != "$want" ]; then
        printf 'FAILED\n    expected md5 %s\n    got          %s\n' "$want" "$got"
        echo "    The reassembled file does not match the input the pipeline"
        echo "    was run on. Re-run the download; do not use this file."
        exit 1
      fi
      printf 'ok, md5 %s\n' "$got"
    fi

    if [ "$KEEP_PARTS" != "1" ]; then
      ( cd "${DEST}" && for p in $parts; do rm -f "$p"; done )
    fi
  done
  [ "$KEEP_PARTS" = "1" ] || echo "  parts removed (KEEP_PARTS=1 to keep them)"
fi

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------
# Say which ones arrived rather than "Done". A partial deposit otherwise shows
# up much later as a reference that silently skips -- the pipeline is designed
# to skip a missing reference rather than fail, which is right during
# development and wrong here.
echo
mkdir -p "${DEST}/final_run"
missing=0
for f in "${REQUIRED_FILES[@]}"; do
  if [ -s "${DEST}/${f}" ]; then
    printf '  OK      %s\n' "$f"
  else
    printf '  MISSING %s\n' "$f"
    missing=$((missing + 1))
  fi
done

if [ "$missing" -gt 0 ]; then
  echo
  echo "${missing} file(s) missing. The pipeline SKIPS a reference whose inputs are"
  echo "absent rather than failing, so do not start a run until this is clean."
  exit 1
fi

echo
echo "All inputs present. Confirm with:"
echo "  python tools/preflight.py        # expect PREFLIGHT PASSED"
echo "then run:"
echo "  python -m persistence --ref UHGG --resume"
