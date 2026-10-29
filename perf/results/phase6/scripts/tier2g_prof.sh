#!/usr/bin/env bash
set -u
exec 9> /root/tier2g.lock
flock -n 9 || { echo "tier2g already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2e done" /root/tier2e.log 2>/dev/null; do sleep 30; done
sleep 10
while pgrep -f "[t]ier2f_acc.sh" > /dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 3600 > /dev/null 2>&1 < /dev/null &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier2g; mkdir -p $OUT/logs
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12 --gpu_prep"
for w in "4 2" "8 4"; do
  set -- $w
  python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --num_frames 16 \
    --hann_chunk $1 --hann_stride $2 --low_vram $OPT --warmup 1 --repeats 1 --temp_dir $OUT/tmp \
    --label "prof 4K $1/$2" --out $OUT/prof_$1x$2.json --torch_profile $OUT/prof_$1x$2 > $OUT/logs/prof_$1x$2.log 2>&1
  log "profile 4K $1/$2 rc=$?: $(grep '\[bench\] median' $OUT/logs/prof_$1x$2.log)"
  rm -f $OUT/prof_$1x$2/trace.json
done
cd $OUT && tar czf /root/tier2g.tgz *.json prof_*/*.txt prof_*/*.json logs/*.log 2>/dev/null
log "tier2g done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1200 > /dev/null 2>&1 < /dev/null &
