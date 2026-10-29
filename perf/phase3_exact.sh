#!/usr/bin/env bash
# Phase 3: bit-exactness of the opt-in flags on real footage, then the
# combined memory/time and the largest window under the laptop cap.
#   p3_base is rendered by perf/phase3_weights.sh (same box, same attention
#   backend); every other tag must compare to it with all-zero metrics.
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=python; fi
fi
R=perf/results/phase3
mkdir -p "$R"
LAPTOP="--sdpa efficient --mem_cap_gib 5.0"

"$PY" perf/goldens.py --tag p3_bf16w --sdpa efficient --clips converge --res 1080 720 \
  --extra=--bf16_weights
"$PY" perf/goldens.py --tag p3_frozen --sdpa efficient --clips converge --res 1080 \
  --extra=--frozen_weights
"$PY" perf/goldens.py --tag p3_lean --sdpa efficient --clips converge --res 1080 720 \
  --extra=--lean_activations
"$PY" perf/goldens.py --tag p3_bf16w_lean --sdpa efficient --clips converge --res 1080 720 \
  "--extra=--bf16_weights --lean_activations"
"$PY" perf/goldens.py --tag p3_all --sdpa efficient --clips converge --res 1080 720 \
  "--extra=--bf16_weights --lean_activations --prefetch_frames"
for c in bf16w frozen lean bf16w_lean all; do
  "$PY" perf/compare.py --ref perf/out/p3_base --cand "perf/out/p3_$c" \
    --out "$R/cmp_$c.json" > "$R/cmp_$c.txt"
done
echo "phase3 exactness done"

"$PY" perf/bench.py $LAPTOP --bf16_weights --lean_activations --input test_clips/converge_1080 \
  --hann_chunk 8 --hann_stride 4 --low_vram --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 bf16+lean" --out "$R/bench_1080_8x4_bf16_lean.json"
"$PY" perf/bench.py $LAPTOP --bf16_weights --lean_activations --prefetch_frames \
  --input test_clips/converge_1080 --hann_chunk 8 --hann_stride 4 --low_vram \
  --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 bf16+lean+prefetch" --out "$R/bench_1080_8x4_all.json"
for w in 16 20 24 28 32; do
  n=$((2 * w)); [ $n -gt 48 ] && n=48
  "$PY" perf/bench.py $LAPTOP --bf16_weights --lean_activations --input test_clips/converge_1080 \
    --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram --num_frames "$n" \
    --repeats 1 --warmup 0 --label "1080p sweep w$w bf16+lean" \
    --out "$R/sweep_1080_w${w}_bf16_lean.json"
done
for w in 24 32 40 48; do
  "$PY" perf/bench.py $LAPTOP --bf16_weights --lean_activations --input test_clips/converge_720 \
    --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram --num_frames 48 \
    --repeats 1 --warmup 0 --label "720p sweep w$w bf16+lean" \
    --out "$R/sweep_720_w${w}_bf16_lean.json"
done
echo "phase3 exact done"
