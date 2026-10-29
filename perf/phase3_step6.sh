#!/usr/bin/env bash
# Phase 3, step 6 (RTX 3060, Windows laptop emulation: mem-efficient
# attention, 4.95 GiB cap, default allocator): time and memory of the final
# --optimized flags (lean v4) at 1080p windows 8..24, then the showcase
# stills/videos rebuilt with fixed labels.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4
R=perf/results/phase3
OPT="--bf16_weights --lean_activations --prefetch_frames"
for w in 8 16 20 24; do
  python perf/bench.py --sdpa efficient --mem_cap_gib 4.95 $OPT \
    --input test_clips/converge200_1080 --num_frames 48 --hann_chunk $w \
    --hann_stride $((w / 2)) --low_vram --repeats 2 \
    --label "1080p $w/$((w / 2)) optimized v4, 4.95 GiB cap, default allocator" \
    --out "$R/bench_1080_w${w}_v4_cap495.json"
done
echo "step6 timing done"
for cfg in "opt16 16-frame" "opt24 24-frame"; do
  set -- $cfg
  python perf/make_showcase.py --clips hair_1080 smoke_1080 converge_1080 motionblur_1080 \
    cup_1080 glasses_1080 sheen_1080 cyc_1080 \
    --before perf/out/g3060 --after "perf/out/$1" \
    --before_label "Shipped: 8-frame window" --after_label "Optimized: $2 window" \
    --out "perf/results/showcase/$1"
done
echo "step6 done"
