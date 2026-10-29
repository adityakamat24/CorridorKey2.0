#!/usr/bin/env bash
set -u
exec 9> /root/tier3o.lock
flock -n 9 || { echo "tier3o already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3n done" /root/tier3n.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3o; mkdir -p $OUT/logs
run() {
  local tag=$1; shift
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  (cd /root/dev3 && python perf/infer_with.py --mem_budget_gib 12 -- --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@") > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier3o.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/tier3o.jsonl) | $(grep -o 'peak reserved [0-9.]* GiB' $OUT/logs/$tag.log)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
T="--input /root/tos/04_5h --frame_dir_linear --low_vram"
cat /root/tos/04_5h/*.exr > /dev/null
for w in "8 4" "4 2"; do
  set -- $w
  run opt$1$2 $T --hann_chunk $1 --hann_stride $2 --optimized
  run new$1$2 $T --hann_chunk $1 --hann_stride $2 --optimized --gpu_prep --async_baton
done
run newauto --input /root/tos/04_5h --frame_dir_linear --optimized --gpu_prep --async_baton
cd $OUT && tar czf /root/tier3o.tgz *.jsonl logs/*.log 2>/dev/null
log "tier3o done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 < /dev/null 9>&- &
