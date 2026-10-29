#!/usr/bin/env bash
# Phase 4, step 2 (RTX 3060, 4-core host, Windows laptop emulation):
#   exact    --optimized 20/10 renders with the parallel window reader and the
#            region-index cache, vs p4_base (committed code before phase 4)
#   timing   1080p 20/10: optimized flags, then + --cache_region_indices
#   offload  stage-1 peak with the late-skip features parked in host memory
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase4
LAPTOP="--sdpa efficient --mem_cap_gib 4.95"
OPT="--bf16_weights --lean_activations --prefetch_frames"
CLIP=test_clips/converge200_1080

rm -rf perf/out/p4_s2
python perf/goldens.py --tag p4_s2 --sdpa efficient --clips hair smoke converge --res 1080 \
  "--extra=--optimized --hann_chunk 20 --hann_stride 10"
python perf/compare.py --ref perf/out/p4_base --cand perf/out/p4_s2 \
  --out "$R/cmp_p4_s2_vs_base.json" > "$R/cmp_p4_s2_vs_base.txt"
echo "s2 exact done"

python perf/bench.py $LAPTOP $OPT --input $CLIP --num_frames 48 --hann_chunk 20 --hann_stride 10 \
  --low_vram --repeats 2 --label "1080p 20/10 optimized, parallel reader" \
  --out "$R/bench_1080_20x10_reader.json"
python perf/bench.py $LAPTOP $OPT --cache_region_indices --input $CLIP --num_frames 48 \
  --hann_chunk 20 --hann_stride 10 --low_vram --repeats 2 \
  --label "1080p 20/10 optimized, parallel reader + region-index cache" \
  --out "$R/bench_1080_20x10_reader_cache.json"
echo "s2 timing done"

python perf/lean_probe.py --hw 1152 1920 --frames 8 16 24 \
  --candidate offload_skip:Stage1.inference_offload_late_skip=true \
  --out "$R/lean_probe_offload_1080.json"
echo "s2 done"
