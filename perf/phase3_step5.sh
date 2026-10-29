#!/usr/bin/env bash
# Phase 3, step 5 (RTX 3060): end-to-end check of --lean_activations v4
# (bf16 branch inputs, per-frame QKV and local mix, in-place residual adds).
#   opt8v4   --optimized 8/4, all 8 clips, vs the shipped goldens (g3060)
#   opt16v4  --optimized 16/8, 3 clips, vs the v1 lean renders (opt16)
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4
R=perf/results/phase3
rm -rf perf/out/opt8v4 perf/out/opt16v4
python perf/goldens.py --tag opt8v4 --sdpa efficient \
  --clips hair motionblur smoke glasses sheen cup converge cyc --res 1080 \
  "--extra=--optimized --hann_chunk 8 --hann_stride 4"
python perf/compare.py --ref perf/out/g3060 --cand perf/out/opt8v4 \
  --out "$R/cmp_opt8v4_vs_shipped.json" > "$R/cmp_opt8v4_vs_shipped.txt"
echo "step5 opt8v4 done"
python perf/goldens.py --tag opt16v4 --sdpa efficient --clips hair smoke converge --res 1080 \
  "--extra=--optimized --hann_chunk 16 --hann_stride 8"
python perf/compare.py --ref perf/out/opt16 --cand perf/out/opt16v4 \
  --out "$R/cmp_opt16v4_vs_opt16.json" > "$R/cmp_opt16v4_vs_opt16.txt"
echo "step5 done"
