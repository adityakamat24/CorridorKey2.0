#!/usr/bin/env bash
# Phase 2 baseline: unmodified runtime, measured from outside by perf/bench.py
# (warmup 1 + 3 measured runs; medians, with SM clocks so throttling shows).
#
# Laptop emulation (when not on the laptop itself), via BENCH_ARGS:
#   --sdpa efficient   Windows torch has no FlashAttention, so the laptop
#                      dispatches every attention to the mem-efficient kernel
#   --mem_cap_gib 5.0  allocator cap = what torch.cuda.mem_get_info reports
#                      free on the 6 GB laptop after CUDA init (5.01 GiB,
#                      desktop and context already taken)
#
#   timing/memory   720p 1/1 4/2 8/4 24/12, 1080p 1/1 4/2 8/4, all --low_vram,
#                   48-frame converge clip, default video output
#   window sweep    1080p 8 10 12 16 and 720p 24 32 48: largest window that
#                   fits under the cap (OOMs are recorded, not fatal)
#   snapshots       1080p windows 4 and 8 on the same 8 frames, for
#                   perf/mem_peak.py --diff (composition of the per-frame slope)
#   block table     1080p 8/4, per-block peak via forward hooks
#   torch.profiler  720p 4/2 on 8 frames, kernels attributed to modules
#   nsys            same tiny config with NVTX ranges, if nsys is installed
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=python; fi
fi
BENCH_ARGS="${BENCH_ARGS:-}"
R=perf/results/phase2
mkdir -p "$R" perf/results/mem perf/logs
CLIP1080=test_clips/converge_1080
CLIP720=test_clips/converge_720
bench() { "$PY" perf/bench.py $BENCH_ARGS "$@"; }

nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,clocks.sm,temperature.gpu,power.limit \
  --format=csv > "$R/idle_nvidia_smi.csv"

for cw in "1 1" "4 2" "8 4" "24 12"; do
  set -- $cw
  bench --input "$CLIP720" --hann_chunk "$1" --hann_stride "$2" --low_vram \
    --num_frames 48 --repeats 3 --warmup 1 --label "720p ${1}/${2} lowvram" \
    --out "$R/bench_720_${1}x${2}.json"
done
for cw in "1 1" "4 2" "8 4"; do
  set -- $cw
  bench --input "$CLIP1080" --hann_chunk "$1" --hann_stride "$2" --low_vram \
    --num_frames 48 --repeats 3 --warmup 1 --label "1080p ${1}/${2} lowvram" \
    --out "$R/bench_1080_${1}x${2}.json"
done

for w in 8 10 12 16; do
  bench --input "$CLIP1080" --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram \
    --num_frames $((2 * w)) --repeats 1 --warmup 0 --label "1080p sweep w$w" \
    --out "$R/sweep_1080_w${w}.json"
done
for w in 24 32 48; do
  bench --input "$CLIP720" --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram \
    --num_frames 48 --repeats 1 --warmup 0 --label "720p sweep w$w" \
    --out "$R/sweep_720_w${w}.json"
done

for w in 4 8; do
  bench --input "$CLIP1080" --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram \
    --num_frames 8 --repeats 1 --warmup 0 --label "1080p w$w snapshot" \
    --mem_snapshot "perf/results/mem/1080_w${w}.pickle" --out "$R/snap_1080_w${w}.json"
done
"$PY" perf/mem_peak.py perf/results/mem/1080_w8.pickle > "$R/mem_peak_1080_w8.txt"
"$PY" perf/mem_peak.py --diff perf/results/mem/1080_w4.pickle perf/results/mem/1080_w8.pickle \
  --frames 4 8 > "$R/mem_slope_1080_w4_w8.txt"

bench --input "$CLIP1080" --hann_chunk 8 --hann_stride 4 --low_vram \
  --num_frames 8 --repeats 1 --warmup 0 --block_mem --label "1080p 8/4 block table" \
  --out "$R/blockmem_1080_8x4.json"

bench --input "$CLIP720" --hann_chunk 4 --hann_stride 2 --low_vram \
  --num_frames 8 --repeats 1 --warmup 1 --torch_profile "$R/profile_720_4x2" \
  --label "720p 4/2 profile" --out "$R/profile_720_4x2.json"

NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -n "$NSYS" ]; then
  "$NSYS" profile -t cuda,nvtx,osrt --cuda-memory-usage=true -f true \
    -o "$R/nsys_720_4x2" "$PY" perf/bench.py $BENCH_ARGS --input "$CLIP720" \
    --hann_chunk 4 --hann_stride 2 --low_vram --num_frames 8 --repeats 1 --warmup 0 \
    --nvtx --out "$R/nsys_720_4x2.json"
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    -f csv -o "$R/nsys_720_4x2" "$R/nsys_720_4x2.nsys-rep" > /dev/null
fi
echo "phase2 runs done"
