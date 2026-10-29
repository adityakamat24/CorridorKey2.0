#!/usr/bin/env bash
# Phase 4, step 4 (RTX 3060): end-to-end check with the banded QKV grid now
# part of --lean_activations, and its timing.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase4
until grep -q "s3 done" perf/logs/p4_step3.log; do sleep 15; done
rm -rf perf/out/p4_s4
python perf/goldens.py --tag p4_s4 --sdpa efficient --clips hair smoke converge --res 1080 \
  "--extra=--optimized --hann_chunk 20 --hann_stride 10"
python perf/compare.py --ref perf/out/p4_base --cand perf/out/p4_s4 \
  --out "$R/cmp_p4_s4_vs_base.json" > "$R/cmp_p4_s4_vs_base.txt"
echo "s4 exact done"
python perf/bench.py --sdpa efficient --mem_cap_gib 4.95 --bf16_weights --lean_activations \
  --prefetch_frames --cache_region_indices --input test_clips/converge200_1080 --num_frames 48 \
  --hann_chunk 20 --hann_stride 10 --low_vram --repeats 2 \
  --label "1080p 20/10 optimized, step 4" --out "$R/bench_1080_20x10_s4.json"
echo "s4 done"
