#!/usr/bin/env bash
# Phase 3, step 7 (RTX 3060): render the window --optimized auto mode picks
# on the 6 GB Windows laptop (20 frames at 1080p), compare it with the
# shipped goldens and build its showcase panels and visual diffs.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4
R=perf/results/phase3
rm -rf perf/out/opt20
python perf/goldens.py --tag opt20 --sdpa efficient \
  --clips hair motionblur smoke glasses sheen cup converge cyc --res 1080 \
  "--extra=--optimized --hann_chunk 20 --hann_stride 10"
python perf/compare.py --ref perf/out/g3060 --cand perf/out/opt20 \
  --out "$R/cmp_opt20_vs_shipped.json" > "$R/cmp_opt20_vs_shipped.txt"
python perf/make_showcase.py --clips hair_1080 smoke_1080 converge_1080 motionblur_1080 \
  cup_1080 glasses_1080 sheen_1080 cyc_1080 \
  --before perf/out/g3060 --after perf/out/opt20 \
  --before_label "Shipped: 8-frame window" --after_label "Optimized: 20-frame window" \
  --out perf/results/showcase/opt20
for c in cyc smoke glasses; do
  python perf/visual_diff.py --ref perf/out/g3060/${c}_1080 --cand perf/out/opt20/${c}_1080 \
    --cmp "$R/cmp_opt20_vs_shipped.json" --clip ${c}_1080 --worst 1 --gain 2 \
    --out perf/results/showcase/vdiff_opt20
done
echo "step7 done"
