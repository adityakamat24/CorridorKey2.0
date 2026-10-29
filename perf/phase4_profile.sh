#!/usr/bin/env bash
# Phase 4 (more speed and memory), step 1, RTX 3060 under the Windows laptop
# emulation (mem-efficient attention, 4.95 GiB cap, default allocator), on
# the working-tree code (after phase4_base.sh ran on the committed code):
#   probe    stage-1 peak and bitwise check of the new lean changes
#   new      --optimized 20/10 renders, compared with p4_base bit for bit
#   bench    1080p 20/10 optimized flags, 48 frames: phase times
#   nsys     same, 24 frames, with NVTX phases: GPU busy/idle and syncs
#   profile  torch.profiler kernel time by module, 1080p 20/10, 20 frames
#   memory   snapshot at window 20: what is alive at the stage-1 peak
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
R=perf/results/phase4
mkdir -p "$R" perf/results/mem
LAPTOP="--sdpa efficient --mem_cap_gib 4.95"
OPT="--bf16_weights --lean_activations --prefetch_frames"
CLIP=test_clips/converge200_1080

python perf/lean_probe.py --hw 1152 1920 --frames 8 16 24 \
  --candidate mlp_out_of_place:SwiGLU.inference_inplace_mlp=false \
  --out "$R/lean_probe_1080.json"
python perf/lean_probe.py --hw 1152 1920 --frames 8 16 24 --no_handover \
  --out "$R/lean_probe_1080_no_handover.json"
python perf/lean_probe.py --hw 2176 3840 --frames 8 \
  --candidate mlp_out_of_place:SwiGLU.inference_inplace_mlp=false \
  --out "$R/lean_probe_4k.json"
python perf/foundation_chunk_probe.py --input test_clips/converge_1080 --chunks 4 8 16 \
  --out "$R/foundation_chunk_probe.json"
echo "p4 probe done"

rm -rf perf/out/p4_new
python perf/goldens.py --tag p4_new --sdpa efficient --clips hair smoke converge --res 1080 \
  "--extra=--optimized --hann_chunk 20 --hann_stride 10"
python perf/compare.py --ref perf/out/p4_base --cand perf/out/p4_new \
  --out "$R/cmp_p4_new_vs_base.json" > "$R/cmp_p4_new_vs_base.txt"
echo "p4 new done"

python perf/bench.py $LAPTOP $OPT --input $CLIP --num_frames 48 --hann_chunk 20 --hann_stride 10 \
  --low_vram --repeats 2 --label "1080p 20/10 optimized, working tree" \
  --out "$R/bench_1080_20x10_new.json"
echo "p4 bench done"

NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -n "$NSYS" ]; then
  rm -f "$R"/nsys_1080_20x10*
  "$NSYS" profile -t cuda,nvtx,osrt -f true -o "$R/nsys_1080_20x10" \
    python perf/bench.py $LAPTOP $OPT --input $CLIP --num_frames 24 --hann_chunk 20 \
    --hann_stride 10 --low_vram --repeats 1 --warmup 0 --nvtx --out "$R/nsys_1080_20x10.json"
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    -f csv -o "$R/nsys_1080_20x10" "$R/nsys_1080_20x10.nsys-rep" > /dev/null 2>&1
  python perf/nsys_summary.py "$R/nsys_1080_20x10" > "$R/nsys_1080_20x10_summary.txt"
fi
echo "p4 nsys done"

python perf/bench.py --sdpa efficient $OPT --input $CLIP --num_frames 20 --hann_chunk 20 \
  --hann_stride 10 --low_vram --repeats 1 --warmup 1 --torch_profile "$R/profile_1080_20x10" \
  --out "$R/profile_1080_20x10.json"
echo "p4 profile done"

python perf/bench.py --sdpa efficient $OPT --input $CLIP --num_frames 20 --hann_chunk 20 \
  --hann_stride 10 --low_vram --repeats 1 --warmup 0 \
  --mem_snapshot perf/results/mem/p4_1080_w20.pickle --out "$R/snap_1080_w20.json"
python perf/mem_peak.py perf/results/mem/p4_1080_w20.pickle > "$R/mem_peak_1080_w20.txt"
echo "p4 profile all done"
