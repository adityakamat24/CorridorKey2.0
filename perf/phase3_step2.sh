#!/usr/bin/env bash
# Phase 3, step 2: stage-1 ownership hand-off (part of --lean_activations) and
# the allocator's expandable segments, against fragmentation OOMs.
#   exactness  all flags (bf16 weights + lean incl. stage 1 + prefetch) vs the
#              unmodified p3_base renders, bit for bit
#   sweep      1080p windows 14..24 under the 5 GiB cap, with and without
#              PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=python; fi
fi
R=perf/results/phase3
mkdir -p "$R"
LAPTOP="--sdpa efficient --mem_cap_gib 5.0"
ALL="--bf16_weights --lean_activations --prefetch_frames"

rm -rf perf/out/p3_all2
"$PY" perf/goldens.py --tag p3_all2 --sdpa efficient --clips converge --res 1080 720 \
  "--extra=$ALL"
"$PY" perf/compare.py --ref perf/out/p3_base --cand perf/out/p3_all2 \
  --out "$R/cmp_all2.json" > "$R/cmp_all2.txt"
echo "phase3 step2 exactness done"

"$PY" perf/bench.py $LAPTOP $ALL --input test_clips/converge_1080 --hann_chunk 8 \
  --hann_stride 4 --low_vram --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 all (stage-1 lean)" --out "$R/bench_1080_8x4_all2.json"
for alloc in default expandable; do
  if [ "$alloc" = expandable ]; then export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
  for w in 14 16 18 20 24; do
    n=$((2 * w)); [ $n -gt 48 ] && n=48
    "$PY" perf/bench.py $LAPTOP $ALL --input test_clips/converge_1080 \
      --hann_chunk "$w" --hann_stride $((w / 2)) --low_vram --num_frames "$n" \
      --repeats 1 --warmup 0 --label "1080p w$w all $alloc" \
      --out "$R/sweep2_1080_w${w}_${alloc}.json"
  done
  unset PYTORCH_CUDA_ALLOC_CONF
done
echo "phase3 step2 done"
