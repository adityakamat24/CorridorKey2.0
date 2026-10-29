#!/usr/bin/env bash
# Phase 3, step 1: weight storage (--frozen_weights, --bf16_weights).
#   exactness   unmodified vs flag, same attention backend (mem-efficient, as
#               on the Windows laptop), float EXR through infer.py, compared
#               bit for bit with perf/compare.py
#   memory/time 1080p 8/4 low-VRAM under laptop emulation, 1 warmup + 3 runs
#   sweep       largest window that fits the 5 GiB cap with --bf16_weights
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=python; fi
fi
R=perf/results/phase3
mkdir -p "$R"
LAPTOP="--sdpa efficient --mem_cap_gib 5.0"

"$PY" perf/goldens.py --tag p3_base --sdpa efficient --clips converge --res 1080 720
"$PY" perf/goldens.py --tag p3_bf16w --sdpa efficient --clips converge --res 1080 720 \
  --extra=--bf16_weights
"$PY" perf/goldens.py --tag p3_frozen --sdpa efficient --clips converge --res 1080 \
  --extra=--frozen_weights
for c in bf16w frozen; do
  "$PY" perf/compare.py --ref perf/out/p3_base --cand "perf/out/p3_$c" \
    --out "$R/cmp_$c.json" > "$R/cmp_$c.txt"
done

for flag in frozen bf16; do
  "$PY" perf/bench.py $LAPTOP --${flag}_weights --input test_clips/converge_1080 \
    --hann_chunk 8 --hann_stride 4 --low_vram --num_frames 48 --repeats 3 --warmup 1 \
    --label "1080p 8/4 ${flag}_weights" --out "$R/bench_1080_8x4_${flag}.json"
done

for w in 10 12 14 16 18 20 24; do
  n=$((2 * w)); [ $n -gt 48 ] && n=48
  "$PY" perf/bench.py $LAPTOP --bf16_weights --input test_clips/converge_1080 \
    --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram --num_frames "$n" \
    --repeats 1 --warmup 0 --label "1080p sweep w$w bf16_weights" \
    --out "$R/sweep_1080_w${w}_bf16.json"
done
for w in 12 16 24 32; do
  "$PY" perf/bench.py $LAPTOP --bf16_weights --input test_clips/converge_720 \
    --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram --num_frames 48 \
    --repeats 1 --warmup 0 --label "720p sweep w$w bf16_weights" \
    --out "$R/sweep_720_w${w}_bf16.json"
done
echo "phase3 weights done"
