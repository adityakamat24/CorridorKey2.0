#!/usr/bin/env bash
# Phase 3, Windows-margin check: --optimized auto VRAM with the default
# allocator, the budget reported to the selector but no allocator cap, so
# peak reserved shows what the allocator holds when nothing forces a release
# (Windows spills to system RAM instead of failing). A row passes when peak
# reserved stays under the budget.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase3
OUT="$R/tiers_windows_${TAG:-gpu}.jsonl"
python perf/tier_check.py --variant optimized --allocator default --uncapped \
  --budgets ${BUDGETS:-4.95 6.95 10.95 14.95 22.95} \
  --clips ${CLIPS:-converge_1080 converge_1440 converge_3k converge_2160} --out "$OUT"
python perf/tier_check.py --variant optimized --allocator default --uncapped \
  --budgets ${LONG_BUDGETS:-10.95 14.95 22.95} --clips converge200_1080 --out "$OUT"
echo "tiers windows done"
