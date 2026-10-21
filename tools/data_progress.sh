#!/usr/bin/env bash
# Emit a line whenever the data-object count changes, so progress is visible
# without asking. Also reports any render or rerun that is running.
cd "$(dirname "$0")/.."          # always run from the repo root
PY="${PYTHON:-python3}"
last=""
while true; do
  n=$($PY tools/figure_sets.py --data 2>/dev/null | head -1 | awk '{print $1}')
  jobs=""
  pgrep -f "figure_size_sweep" >/dev/null && jobs="$jobs render"
  pgrep -f "only s1" >/dev/null && jobs="$jobs s1-rerun"
  cur="$n$jobs"
  if [ "$cur" != "$last" ]; then
    echo "$(date +%H:%M)  data objects: $n${jobs:+   running:$jobs}"
    last="$cur"
  fi
  sleep 60
done
