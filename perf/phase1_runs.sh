#!/usr/bin/env bash
# Phase 1: golden references plus the noise-floor renders, all through the
# unmodified runtime (perf/goldens.py -> infer.py, or perf/infer_with.py for a
# restricted SDPA backend). Four streams run concurrently on one GPU; every
# tag has its own memmap temp dir, so runs of the same clip never collide.
#
#   golden          all clips, 1080p 8/4 and 720p 24/12, no hint
#   golden_hint     hair + converge, 1080p, rough first-frame hint (q=0.75)
#   rerun*          same as golden, second process (determinism)
#   sdpa_efficient  attention restricted to the mem-efficient kernel
#   sdpa_cudnn      attention restricted to cuDNN
set -u
cd "$(dirname "$0")/.."
# Rented containers often report every host core (64) under a much smaller
# CPU quota (15.4 here); four processes each spinning up 64 OpenMP threads
# throttle each other badly. Size the pools to the quota instead.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-4}"
CLIPS="hair motionblur smoke glasses sheen cup converge cyc"
SUBSET="hair smoke converge cyc"
mkdir -p perf/logs

( python perf/goldens.py --tag golden --clips $CLIPS --res 1080 &&
  python perf/goldens.py --tag golden_hint --clips hair converge --res 1080 --hint
) > perf/logs/p1_golden.log 2>&1 &

( python perf/goldens.py --tag rerun --clips $CLIPS --res 1080 &&
  python perf/goldens.py --tag rerun --clips $SUBSET --res 720 &&
  python perf/goldens.py --tag rerun_hint --clips hair converge --res 1080 --hint
) > perf/logs/p1_rerun.log 2>&1 &

( python perf/goldens.py --tag sdpa_efficient --sdpa efficient --clips $CLIPS --res 1080 &&
  python perf/goldens.py --tag sdpa_efficient --sdpa efficient --clips $SUBSET --res 720
) > perf/logs/p1_efficient.log 2>&1 &

# cuDNN may reject some calls; the 720p goldens run regardless.
( python perf/goldens.py --tag sdpa_cudnn --sdpa cudnn --clips $CLIPS --res 1080;
  python perf/goldens.py --tag golden --clips $CLIPS --res 720
) > perf/logs/p1_cudnn.log 2>&1 &

wait
echo "phase1 renders done"
