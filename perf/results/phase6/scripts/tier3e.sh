#!/usr/bin/env bash
set -u
exec 9> /root/tier3e.lock
flock -n 9 || { echo "tier3e already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null &
pip install -q py-spy > /dev/null 2>&1; log "py-spy $(py-spy --version 2>&1)"
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3e; mkdir -p $OUT
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12 --gpu_prep"
cat /root/tos/04_5h/*.exr > /dev/null
for w in "4 2" "8 4"; do
  set -- $w
  py-spy record --idle -r 100 --format raw -o $OUT/spy_$1x$2.txt -- python perf/bench.py --input /root/tos/04_5h \
    --frame_dir_linear --frame_dir_fps 24 --num_frames 32 --hann_chunk $1 --hann_stride $2 --low_vram $OPT \
    --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "spy 4K $1/$2" --out $OUT/spy_$1x$2.json > $OUT/spy_$1x$2.log 2>&1
  log "spy 4K $1/$2 rc=$?: $(grep '\[bench\] median' $OUT/spy_$1x$2.log)"
  python /root/spy_stage1.py $OUT/spy_$1x$2.txt hann_stage1_stream 100 > $OUT/stage1_$1x$2.txt 2>&1
  python /root/spy_stage1.py $OUT/spy_$1x$2.txt _run_inference 100 > $OUT/stage2_$1x$2.txt 2>&1
done
log "tier3e done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null &
