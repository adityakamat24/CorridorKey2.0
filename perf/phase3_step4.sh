#!/usr/bin/env bash
# Phase 3, step 4 (RTX 3060, laptop emulation): what is left after the exact
# flags, and stage-2 batching.
#   exact    --optimized 8/4 with --stage2_batch 4 vs --optimized 8/4 (opt8)
#   timing   1080p 8/4 with the optimized flags, --stage2_batch 1, 2, 4
#   nsys     1080p 8/4, 16 frames, optimized flags: GPU busy vs wall, syncs
#   memory   snapshots at windows 8 and 16 (optimized flags, no cap): what is
#            alive at the peak and what the 158 MiB/frame slope is made of
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4
R=perf/results/phase3
LAPTOP="--sdpa efficient --mem_cap_gib 4.95"
OPT="--bf16_weights --lean_activations --prefetch_frames"

rm -rf perf/out/opt8_s2b4
python perf/goldens.py --tag opt8_s2b4 --sdpa efficient --clips converge hair --res 1080 \
  "--extra=--optimized --hann_chunk 8 --hann_stride 4 --stage2_batch 4"
python perf/compare.py --ref perf/out/opt8 --cand perf/out/opt8_s2b4 \
  --out "$R/cmp_s2b4.json" > "$R/cmp_s2b4.txt"
echo "step4 exact done"

for b in 1 2 4; do
  python perf/bench.py $LAPTOP $OPT --input test_clips/converge200_1080 --num_frames 48 \
    --hann_chunk 8 --hann_stride 4 --low_vram --stage2_batch $b --repeats 2 \
    --label "1080p 8/4 optimized, stage2_batch $b" --out "$R/bench_1080_8x4_opt_s2b$b.json"
done
echo "step4 timing done"

NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -n "$NSYS" ]; then
  rm -f "$R"/nsys_1080_8x4_opt*
  "$NSYS" profile -t cuda,nvtx,osrt -f true -o "$R/nsys_1080_8x4_opt" \
    python perf/bench.py $LAPTOP $OPT --input test_clips/converge200_1080 --num_frames 16 \
    --hann_chunk 8 --hann_stride 4 --low_vram --repeats 1 --warmup 0 --nvtx \
    --out "$R/nsys_1080_8x4_opt.json"
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    -f csv -o "$R/nsys_1080_8x4_opt" "$R/nsys_1080_8x4_opt.nsys-rep" > /dev/null
  python perf/nsys_summary.py "$R/nsys_1080_8x4_opt" > "$R/nsys_1080_8x4_opt_summary.txt"
fi
mkdir -p perf/results/mem
for w in 8 16; do
  python perf/bench.py --sdpa efficient $OPT --input test_clips/converge200_1080 --num_frames 16     --hann_chunk $w --hann_stride $((w / 2)) --low_vram --repeats 1 --warmup 0     --mem_snapshot "perf/results/mem/opt_1080_w$w.pickle" --label "1080p optimized w$w snapshot"     --out "$R/snap_opt_1080_w$w.json"
done
python perf/mem_peak.py perf/results/mem/opt_1080_w16.pickle > "$R/mem_peak_opt_1080_w16.txt"
python perf/mem_peak.py --diff perf/results/mem/opt_1080_w8.pickle perf/results/mem/opt_1080_w16.pickle   --frames 8 16 > "$R/mem_slope_opt_1080_w8_w16.txt"
echo "step4 done"
