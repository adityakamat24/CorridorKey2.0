#!/usr/bin/env bash
# Phase 3 tier validation: run the whole pipeline with auto VRAM on emulated
# card sizes (perf/tier_check.py) and record the window each path picks,
# peak memory and time, and any out-of-memory failure.
#   optimized  --optimized, default allocator (what Windows always gets) and
#              expandable segments (Linux)
#   shipped    the unmodified runtime, default allocator (SHIPPED_CLIPS only)
# BUDGETS and CLIPS come from the environment so the small tiers can run on
# a 12 GB RTX 3060 (same sm_86 as the laptop) and the big ones on a 48 GB card.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase3
BUDGETS=${BUDGETS:-"4.95 6.95 10.95 14.95 22.95 46.95"}
CLIPS=${CLIPS:-"converge_1080 converge200_1080 converge_1440 converge_3k converge_2160"}
SHIPPED_CLIPS=${SHIPPED_CLIPS:-"converge_1080 converge_2160"}
TAG=${TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')}
OUT="$R/tiers_$TAG.jsonl"
python perf/tier_check.py --variant optimized --allocator default --budgets $BUDGETS \
  --clips $CLIPS --out "$OUT"
python perf/tier_check.py --variant optimized --allocator expandable --budgets $BUDGETS \
  --clips $CLIPS --out "$OUT"
python perf/tier_check.py --variant shipped --allocator default --budgets $BUDGETS \
  --clips $SHIPPED_CLIPS --out "$OUT"
echo "tiers validate done"
