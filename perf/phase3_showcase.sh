#!/usr/bin/env bash
# Phase 3 showcase renders (RTX 3060, mem-efficient attention as on Windows).
#   opt8   --optimized at the shipped 6 GB setting (8/4): must equal g3060
#   opt16  --optimized at 16/8, the window a 6 GB Windows laptop can now fit
#   opt24  --optimized at 24/12, the window a 6 GB Linux card can now fit
# then compares each against the shipped goldens (g3060) and builds the
# before/after stills and videos.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENCV_FOR_THREADS_NUM=4
R=perf/results/phase3
CLIPS="hair motionblur smoke glasses sheen cup converge cyc"
for cfg in "opt8 8 4" "opt16 16 8" "opt24 24 12"; do
  set -- $cfg
  rm -rf "perf/out/$1"
  python perf/goldens.py --tag "$1" --sdpa efficient --clips $CLIPS --res 1080 \
    "--extra=--optimized --hann_chunk $2 --hann_stride $3"
  python perf/compare.py --ref perf/out/g3060 --cand "perf/out/$1" \
    --out "$R/cmp_$1_vs_shipped.json" > "$R/cmp_$1_vs_shipped.txt"
  echo "showcase $1 done"
done
for cfg in "opt16 16-frame" "opt24 24-frame"; do
  set -- $cfg
  python perf/make_showcase.py --clips hair_1080 smoke_1080 converge_1080 motionblur_1080 \
    cup_1080 glasses_1080 sheen_1080 cyc_1080 \
    --before perf/out/g3060 --after "perf/out/$1" \
    --before_label "Shipped: 8-frame window" --after_label "Optimized: $2 window" \
    --out "perf/results/showcase/$1"
done
echo "showcase done"
