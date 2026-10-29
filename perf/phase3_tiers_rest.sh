#!/usr/bin/env bash
# Phase 3 tiers on the 48 GB card, reordered after the optimized/default
# pass: shipped runs for the before column, then the boundary budgets
# (perf/phase3_tiers_boundary.sh), then expandable segments on the two
# mid budgets.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 OPENCV_FOR_THREADS_NUM=8
R=perf/results/phase3
OUT="$R/tiers_rtx6000ada.jsonl"
while pgrep -f "tier_check.py --variant optimized --allocator [d]efault" > /dev/null; do sleep 15; done
python perf/tier_check.py --variant shipped --allocator default \
  --budgets 10.95 14.95 22.95 46.95 --clips converge_1080 converge_2160 --out "$OUT"
TAG=rtx6000ada bash perf/phase3_tiers_boundary.sh
python perf/tier_check.py --variant optimized --allocator expandable --budgets 10.95 14.95 \
  --clips converge_1080 converge200_1080 converge_1440 converge_3k converge_2160 --out "$OUT"
echo "tiers rest done"
