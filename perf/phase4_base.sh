#!/usr/bin/env bash
# Phase 4, step 0 (RTX 3060, committed code): the reference renders and
# timing every later change is compared against. --optimized 20/10 (the
# 6 GB Windows laptop's auto pick at 1080p) on hair/smoke/converge, and the
# 1080p 20/10 bench under the Windows laptop emulation.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase4
mkdir -p "$R"
rm -rf perf/out/p4_base
python perf/goldens.py --tag p4_base --sdpa efficient --clips hair smoke converge --res 1080 \
  "--extra=--optimized --hann_chunk 20 --hann_stride 10"
python perf/bench.py --sdpa efficient --mem_cap_gib 4.95 --bf16_weights --lean_activations \
  --prefetch_frames --input test_clips/converge200_1080 --num_frames 48 --hann_chunk 20 \
  --hann_stride 10 --low_vram --repeats 2 --label "1080p 20/10 optimized, committed (p4 base)" \
  --out "$R/bench_1080_20x10_base.json"
echo "p4 base done"
