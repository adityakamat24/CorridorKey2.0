#!/usr/bin/env bash
# Phase 1 noise floor: compare every Phase 1 render against the goldens.
#   rerun          same command, separate process          -> determinism
#   trial          hair_1080 rendered alone, not concurrent -> concurrency check
#   sdpa_efficient attention forced to the mem-efficient kernel
#   sdpa_cudnn     attention forced to cuDNN
# The SDPA swaps bound what "near-exact" can mean for a code change: any
# optimization whose metrics sit inside these is indistinguishable from a
# different attention kernel.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-2}"
R=perf/results/phase1
mkdir -p "$R"
for cand in rerun trial sdpa_efficient sdpa_cudnn; do
  [ -d "perf/out/$cand" ] || continue
  python perf/compare.py --ref perf/out/golden --cand "perf/out/$cand" \
    --out "$R/cmp_${cand}.json" > "$R/cmp_${cand}.txt" &
done
python perf/compare.py --ref perf/out/golden_hint --cand perf/out/rerun_hint \
  --out "$R/cmp_rerun_hint.json" > "$R/cmp_rerun_hint.txt" &
wait
cat "$R"/cmp_*.txt
for clip in hair_1080 smoke_1080 cyc_1080 glasses_1080; do
  python perf/visual_diff.py --ref perf/out/golden --cand perf/out/sdpa_efficient \
    --cmp "$R/cmp_sdpa_efficient.json" --clip "$clip" --worst 1 \
    --out "$R/vdiff_sdpa_efficient" --gif
done
