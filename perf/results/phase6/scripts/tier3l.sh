#!/usr/bin/env bash
set -u
exec 9> /root/tier3l.lock
flock -n 9 || { echo "tier3l already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3m done" /root/tier3m.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 < /dev/null 9>&- &
[ -d /root/dev3 ] || { echo "no dev3"; exit 1; }
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3l; mkdir -p $OUT/logs
(cd /root/dev3 && python -c "from corridorkey import video_io as v; import glob; p = sorted(glob.glob('/root/tos/04_5h/*.exr'))[0]; print('ToS half:', v.exr_rgb_is_half(p)); q = sorted(glob.glob('/root/tos_f32/*.exr')); print('f32 half:', v.exr_rgb_is_half(q[0]) if q else 'no clip')") > $OUT/header.txt 2>&1
log "header check: $(tr '\n' ' ' < $OUT/header.txt)"
run() {
  local tag=$1 dir=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  (cd $dir && python perf/infer_with.py --mem_budget_gib 12 -- --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@") > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier3l.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/tier3l.jsonl) | $(grep -o 'peak reserved [0-9.]* GiB' $OUT/logs/$tag.log)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
T="--input /root/tos/04_5h --frame_dir_linear --low_vram --fast --gpu_prep"
cat /root/tos/04_5h/*.exr > /dev/null
for rep in a b; do
  for w in "4 2" "8 4"; do
    set -- $w
    run base$1$2_$rep /root/dev_base $T --hann_chunk $1 --hann_stride $2
    run sync$1$2_$rep /root/dev3 $T --hann_chunk $1 --hann_stride $2
    run async$1$2_$rep /root/dev3 $T --hann_chunk $1 --hann_stride $2 --async_baton
  done
done
cd $OUT && tar czf /root/tier3l.tgz *.jsonl *.txt logs/*.log 2>/dev/null
log "tier3l done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
