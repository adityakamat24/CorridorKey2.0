#!/usr/bin/env bash
set -u
exec 9> /root/tier3c.lock
flock -n 9 || { echo "tier3c already running"; exit 1; }
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 < /dev/null &
log() { echo "[$(date +%H:%M:%S)] $*"; }
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3c; mkdir -p $OUT/logs
RES=$OUT/tier3c.jsonl
warm() { cat /root/hcw/*/frames/* /root/tos/04_5h/*.exr > /dev/null; }
run() {
  local tag=$1; shift
  rm -rf $OUT/out_$tag $OUT/tmp_$tag; mkdir -p $OUT/out_$tag $OUT/tmp_$tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > $OUT/logs/$tag.smi &
  local smi=$!
  local t0=$(date +%s.%N)
  "$@" > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  kill $smi
  local smi_max=$(sort -n $OUT/logs/$tag.smi | tail -1)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))"), \"nvidia_smi_max_mib\": ${smi_max:-0}}" >> $RES
  log "$tag rc=$rc $(tail -1 $RES) | $(grep -m1 -oE 'hann_chunk=[0-9]+, hann_stride=[0-9]+, low_vram=[A-Za-z]+|Hann stage1: chunk=[0-9]+, stride=[0-9]+' $OUT/logs/$tag.log | head -1)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
ours() { local tag=$1 inp=$2; shift 2; run $tag python perf/infer_with.py --mem_budget_gib 12 -- --input "$inp" --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@"; }
log "code: gpu_prep refs $(grep -c gpu_prep infer.py corridorkey/runtime.py | tr '\n' ' ')"
d=/root/tos/04_5h
L="--frame_dir_linear"
warm
ours opt42_tos $d $L --optimized --hann_chunk 4 --hann_stride 2 --low_vram
ours fast42_tos $d $L --fast --hann_chunk 4 --hann_stride 2 --low_vram
ours best42_tos $d $L --fast --gpu_prep --hann_chunk 4 --hann_stride 2 --low_vram
ours opt84_tos $d $L --optimized --hann_chunk 8 --hann_stride 4 --low_vram
ours fast84_tos $d $L --fast --hann_chunk 8 --hann_stride 4 --low_vram
ours best84_tos $d $L --fast --gpu_prep --hann_chunk 8 --hann_stride 4 --low_vram
ours bestauto_tos $d $L --fast --gpu_prep
for clip in hair_detail motion_blur; do
  dd=/root/hcw/$clip/frames
  warm
  ours orig_$clip $dd
  ours opt_$clip $dd --optimized
  ours fast_$clip $dd --fast
  ours best_$clip $dd --fast --gpu_prep
done
warm
ours orig84_tos $d $L --hann_chunk 8 --hann_stride 4 --low_vram
ours best42b_tos $d $L --fast --gpu_prep --hann_chunk 4 --hann_stride 2 --low_vram
cd $OUT && tar czf /root/tier3c.tgz *.jsonl logs/*.log 2>/dev/null
log "tier3c done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 < /dev/null &
