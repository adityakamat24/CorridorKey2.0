#!/usr/bin/env bash
set -u
exec 9> /root/tier3h.lock
flock -n 9 || { echo "tier3h already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3g done" /root/tier3g.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3h; mkdir -p $OUT/logs
run() {
  local tag=$1 dir=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  (cd $dir && python perf/infer_with.py --mem_budget_gib 12 -- --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@") > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier3h.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/tier3h.jsonl) | $(grep -o 'peak allocated.*' $OUT/logs/$tag.log)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
T="--input /root/tos/04_5h --frame_dir_linear --low_vram --fast --gpu_prep --hann_chunk 4 --hann_stride 2"
cat /root/tos/04_5h/*.exr > /dev/null
run base42_a /root/dev_base $T
run final42_a /root/dev2 $T --async_baton
run base42_b /root/dev_base $T
run final42_b /root/dev2 $T --async_baton
log "tier3h done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
