#!/usr/bin/env bash
# Phase 4, step 3 (RTX 3060, 4-core host, Windows laptop emulation):
#   gemm     is the QKV projection row-for-row identical on row bands?
#   band     stage-1 peak and bitwise check with the banded QKV grid
#   exact    --optimized 20/10 renders (first-window parallel reader,
#            late-skip offload) vs p4_base
#   timing   1080p 20/10, optimized flags + --cache_region_indices
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase4
python perf/gemm_m_probe.py --hw 80 128 --bands 38 32 22 16 --frames 1 > "$R/gemm_m_probe_1080.txt" 2>&1
python perf/gemm_m_probe.py --hw 144 240 --bands 38 32 22 --frames 1 > "$R/gemm_m_probe_4k.txt" 2>&1
grep -c "equal True" "$R"/gemm_m_probe_*.txt; grep -c "equal False" "$R"/gemm_m_probe_*.txt
python perf/lean_probe.py --hw 1152 1920 --frames 8 16 24 \
  --candidate band:SwinAttention3D.inference_qkv_band=true --out "$R/lean_probe_band_1080.json"
python perf/lean_probe.py --hw 2176 3840 --frames 8 \
  --candidate band:SwinAttention3D.inference_qkv_band=true --out "$R/lean_probe_band_4k.json"
echo "s3 probes done"
rm -rf perf/out/p4_s3
python perf/goldens.py --tag p4_s3 --sdpa efficient --clips hair smoke converge --res 1080 \
  "--extra=--optimized --hann_chunk 20 --hann_stride 10"
python perf/compare.py --ref perf/out/p4_base --cand perf/out/p4_s3 \
  --out "$R/cmp_p4_s3_vs_base.json" > "$R/cmp_p4_s3_vs_base.txt"
echo "s3 exact done"
python perf/bench.py --sdpa efficient --mem_cap_gib 4.95 --bf16_weights --lean_activations \
  --prefetch_frames --cache_region_indices --input test_clips/converge200_1080 --num_frames 48 \
  --hann_chunk 20 --hann_stride 10 --low_vram --repeats 2 \
  --label "1080p 20/10 optimized, step 3" --out "$R/bench_1080_20x10_s3.json"
echo "s3 done"
