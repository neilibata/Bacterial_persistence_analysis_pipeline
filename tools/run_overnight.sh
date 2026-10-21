#!/usr/bin/env bash
# Everything, in dependency order, with a log per stage and a verdict at the end.
#
#   bash tools/run_overnight.sh
#
# Stages 1-3 rebuild the S_inf chain for A16/A17 against the current cohort.
# They come first because they are the part that can fail for a reason no
# amount of waiting fixes, and finding that out at 2am is better than at 8am.
# Stage 4 is the rest of the Additional Figures. Stages 5-7 check the result.
set -u
cd "$(dirname "$0")/.."          # always run from the repo root
# The interpreter is whatever the caller's environment provides. Pinning one
# machine's conda path here is what made this script unusable anywhere else;
# environment.yml is where the version is pinned.
PY="${PYTHON:-python3}"
LOGS=logs/overnight_$(date +%Y%m%d_%H%M)
mkdir -p "$LOGS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-.mplconfig}"
START="$LOGS/.started"; touch "$START"

say() { printf '\n=== %s  [%s]\n' "$1" "$(date +%H:%M)" | tee -a "$LOGS/summary.txt"; }
run() { local n="$1"; shift; say "$n"; "$@" >"$LOGS/$n.log" 2>&1; \
        local rc=$?; echo "  exit $rc" | tee -a "$LOGS/summary.txt"; return $rc; }

# ── 1. bootstrap caches for the S_inf chain, current cohort ────────────────
for t in 10 30 50; do
  run "1_fig3_${t}pct" $PY figures/make_combined_figure3.py --threshold "$t" || true
done
for k in 2 3 5; do
  run "1_fig3_min${k}" $PY figures/make_combined_figure3.py --threshold 10 \
      --min-occurrences "$k" --models rsac || true
done

# ── 2. mixture profile scans, then BIC selection ───────────────────────────
# Two steps, not one. profile_sinf_mixture.py runs the profile-likelihood scan
# and writes a CSV per mixture order H; score_mixture_profiles_for_k.py then
# reads those and picks the best by BIC. Calling only the second is what failed
# on the first attempt -- it has nothing to score until the first has run.
for k in 2 3 5; do
  run "2a_scan_k${k}"   $PY tools/sinf_chain/profile_sinf_mixture.py --k "$k" || true
  run "2b_select_k${k}" $PY tools/sinf_chain/score_mixture_profiles_for_k.py \
      --k "$k" --profiles-dir "results/Figure3_combined/profile_sinf_mixture_min${k}" || true
done

# ── 3. A16 and A17 ─────────────────────────────────────────────────────────
run "3_sinf_figures" $PY tools/make_accumulation_sinf_figures.py || true

# ── 4. every other Additional Figure, combined reference ───────────────────
run "4_additional" $PY -m persistence --generate-all-additional-figures || true

# ── 5. contract: all 32 present, and written by this run ───────────────────
say "5_verify"
$PY tools/figure_sets.py --verify --since "$START" 2>&1 | tee "$LOGS/5_verify.log" \
  | tee -a "$LOGS/summary.txt"

# Anything still missing gets a second attempt with the reason recorded, rather
# than being reported as a gap in the morning. The S_inf figures are the likely
# case: stage 3 runs before stage 4 has refreshed anything it depends on.
if ! $PY tools/figure_sets.py --verify >/dev/null 2>&1; then
  say "5b_retry_missing"
  $PY tools/figure_sets.py --verify 2>&1 | grep MISSING | tee -a "$LOGS/summary.txt"
  for k in 2 3 5; do
    [ -f "results/Figure3_combined/profile_sinf_mixture_min${k}/profile_mix_k${k}_best_by_bic_asymptotes.csv" ] || {
      run "5b_scan_k${k}"   $PY tools/sinf_chain/profile_sinf_mixture.py --k "$k" || true
      run "5b_select_k${k}" $PY tools/sinf_chain/score_mixture_profiles_for_k.py \
          --k "$k" --profiles-dir "results/Figure3_combined/profile_sinf_mixture_min${k}" || true
    }
  done
  run "5b_sinf_retry"   $PY tools/make_accumulation_sinf_figures.py || true
  run "5b_additional_retry" $PY -m persistence \
      --generate-all-additional-figures --figure-strict || true
  say "5c_verify_again"
  $PY tools/figure_sets.py --verify 2>&1 | tee -a "$LOGS/summary.txt"
fi

# ── 6. quality: overlaps, aspect, printed size, grids, ticks ───────────────
run "6_standards" $PY tools/figure_size_sweep.py --ref combined --sections \
    depth_audit ipw_richness ipw_pairwise ipw_sweep tech_reps carter axis_angle \
    accumulation richness_cv supp_s5 s2_pooling s1_figures s3_health || true

# ── 7. the review page, and how much is re-styleable without recomputing ───
run "7_review" $PY tools/build_figure_review.py --sweep "$LOGS/6_standards.log"
say "7_cache_coverage"
$PY tools/restyle.py --check 2>&1 | tail -3 | tee -a "$LOGS/summary.txt"

say "done"
echo "  logs in $LOGS" | tee -a "$LOGS/summary.txt"
