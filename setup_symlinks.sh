#!/usr/bin/env bash
# setup_symlinks.sh — Link data files into the pipeline root.
#
# Run from:  full_pipeline_clean/
# Usage:
#   bash setup_symlinks.sh              # uses default ../plots
#   bash setup_symlinks.sh /path/to/data   # custom source directory

set -euo pipefail

SRC="${1:-../plots}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: Source directory not found: $SRC"
    echo "Usage: bash setup_symlinks.sh [/path/to/data]"
    exit 1
fi

SRC_ABS="$(cd "$SRC" && pwd)"
SRC_PARENT="$(dirname "$SRC_ABS")"
echo "Source: $SRC_ABS"
echo "Target: $(pwd)"
echo ""

linked=0
skipped=0
missing=0

source_path_for() {
    local fname="$1"
    if [ -f "$SRC_ABS/$fname" ]; then
        printf '%s\n' "$SRC_ABS/$fname"
    elif [ -f "$SRC_PARENT/$fname" ]; then
        printf '%s\n' "$SRC_PARENT/$fname"
    else
        return 1
    fi
}

link_file() {
    local fname="$1"
    local src_path
    if [ -L "$fname" ]; then
        if [ -e "$fname" ]; then
            echo "  SKIP (already linked): $fname"
            ((skipped++)) || true
        else
            rm "$fname"
            if src_path="$(source_path_for "$fname")"; then
                ln -s "$src_path" "$fname"
                echo "  FIX:  $fname"
                ((linked++)) || true
            else
                echo "  MISS: $fname  (broken link removed; not in source dir)"
                ((missing++)) || true
            fi
        fi
    elif [ -f "$fname" ]; then
        echo "  SKIP (real file exists): $fname"
        ((skipped++)) || true
    elif src_path="$(source_path_for "$fname")"; then
        ln -s "$src_path" "$fname"
        echo "  OK:   $fname"
        ((linked++)) || true
    else
        echo "  MISS: $fname  (not in source dir)"
        ((missing++)) || true
    fi
}

echo "=== Matrices (UHGG) ==="
link_file "meancoverage_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv"
link_file "meandepth_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv"

echo ""
echo "=== Matrices (Leviatan) ==="
link_file "meancoverage_and_std_matrix_high_lev_merged_dedup_firstcol.csv"
link_file "meandepth_and_std_matrix_high_lev_merged_dedup_firstcol.csv"

echo ""
echo "=== Matrices (Chimp) ==="
link_file "meancoverage_and_std_matrix_high_chimp_merged.csv"
link_file "meandepth_and_std_matrix_high_chimp_merged.csv"

echo ""
echo "=== Matrices (Combined SGBs) ==="
link_file "combined_meancoverage_matrix_5081_SGBs_best.csv"
link_file "combined_meandepth_matrix_5081_SGBs_best.csv"
link_file "combined_sgb_metadata.xlsx"

echo ""
echo "=== Genome lengths ==="
link_file "all_SGB_lengths_UHGG.txt"
link_file "all_SGB_lengths_lev.txt"
link_file "all_SGB_lengths_chimp.txt"
link_file "all_SGB_lengths_combined.txt"

echo ""
echo "=== Metadata ==="
link_file "metadata_Mar_16_2026.csv"

echo ""
echo "=== Taxon lists (VANISH / BloSSUM) ==="
link_file "Vanish_list.txt"
link_file "Blossum_list.txt"
link_file "Vanish_phylo.txt"
link_file "Blossom_phylo.txt"

echo ""
echo "=== Taxonomy / SGB matching ==="
link_file "Hadza_UHGG_SGB_matching.csv"

echo ""
echo "========================================"
echo "  Linked:  $linked"
echo "  Skipped: $skipped"
echo "  Missing: $missing"
echo "========================================"

if [ $missing -gt 0 ]; then
    echo ""
    echo "NOTE: Missing files may be optional (chimp, etc.)."
    echo "      The pipeline skips analyses whose inputs are absent."
fi

echo ""
echo "Ready! Run:  python -m persistence"
