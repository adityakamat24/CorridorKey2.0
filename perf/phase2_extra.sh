#!/usr/bin/env bash
# Phase 2 follow-ups on a Linux box (run after perf/phase2_runs.sh):
#   floor      perf/floor_probe.py: stage-1 forward memory at 1080p T=1/4/8
#              with fp32 params (autocast cast cache on), frozen params,
#              cache off, and bf16 Linear/Conv storage; plus exactness
#   flash      1080p 8/4 low-VRAM with the default (flash) attention, no cap:
#              what FlashAttention would buy the Windows laptop
#   normal     1080p 8/4 without --low_vram, no cap (needs ~7.5 GiB): what the
#              low-VRAM staging (model swaps, memmaps, re-reads) costs in time
#   e2e        200-frame clip, 1080p 8/4 low-VRAM under laptop emulation: the
#              brief's end-to-end timing run
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=python; fi
fi
R=perf/results/phase2
mkdir -p "$R"
"$PY" perf/floor_probe.py --hw 1152 1920 --frames 1 4 8 --out "$R/floor_probe_1080.json"
"$PY" perf/bench.py --sdpa default --input test_clips/converge_1080 --hann_chunk 8 \
  --hann_stride 4 --low_vram --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 lowvram flash" --out "$R/extra_1080_8x4_flash.json"
"$PY" perf/bench.py --sdpa efficient --input test_clips/converge_1080 --hann_chunk 8 \
  --hann_stride 4 --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 normal mode" --out "$R/extra_1080_8x4_normal.json"
"$PY" perf/bench.py --sdpa efficient --mem_cap_gib 5.0 --input test_clips/converge200_1080 \
  --hann_chunk 8 --hann_stride 4 --low_vram --repeats 1 --warmup 1 \
  --label "1080p 8/4 lowvram e2e 200f" --out "$R/extra_1080_8x4_e2e200.json"
echo "phase2 extra done (bench)"
# nsys rerun (the first attempt crashed on an nvtx hook bug, fixed in profiling.py)
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -n "$NSYS" ]; then
  rm -f "$R"/nsys_720_4x2*
  "$NSYS" profile -t cuda,nvtx,osrt --cuda-memory-usage=true -f true \
    -o "$R/nsys_720_4x2" "$PY" perf/bench.py --sdpa efficient --mem_cap_gib 5.0 \
    --input test_clips/converge_720 --hann_chunk 4 --hann_stride 2 --low_vram \
    --num_frames 8 --repeats 1 --warmup 0 --nvtx --out "$R/nsys_720_4x2.json"
  "$NSYS" stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    -f csv -o "$R/nsys_720_4x2" "$R/nsys_720_4x2.nsys-rep" > /dev/null
fi
echo "phase2 extra done (nsys)"
