#!/usr/bin/env bash
# Phase 3 tier boundaries: run --optimized auto VRAM at the tightest budget
# that still selects each window (perf/tier_boundary_budgets.py), with the
# allocator settings --optimized uses on Windows, so the selector margin is
# tested where it is thinnest.
# The list is a subset of the boundaries: the small-card windows, the
# largest low-VRAM window of each 48-frame clip, and the normal-mode picks.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase3
OUT="$R/tiers_boundary_${TAG:-gpu}.jsonl"
# What --optimized sets on Windows (configure_allocator); ALLOC_CONF overrides.
CONF=${ALLOC_CONF:-max_split_size_mb:256}
while read -r clip budget; do
  [ -z "$clip" ] && continue
  python perf/tier_check.py --variant optimized --allocator default --alloc_conf "$CONF" \
    --budgets "$budget" --clips "$clip" --out "$OUT"
done <<'LIST'
converge_1080 4.744
converge_1080 5.933
converge_1080 7.727
converge_1080 8.916
converge_1080 12.932
converge200_1080 10.115
converge200_1080 13.692
converge200_1080 17.228
converge_1440 4.821
converge_1440 6.142
converge_1440 8.761
converge_1440 12.261
converge_1440 17.918
converge_3k 4.642
converge_3k 6.368
converge_3k 9.819
converge_3k 15.568
converge_3k 23.7
converge_2160 4.302
converge_2160 6.842
converge_2160 10.228
converge_2160 15.308
converge_2160 22.388
converge_2160 34.23
LIST
echo "tiers boundary done"
