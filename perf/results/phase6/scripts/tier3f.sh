#!/usr/bin/env bash
set -u
exec 9> /root/tier3f.lock
flock -n 9 || { echo "tier3f already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 < /dev/null 9>&- &
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3f; mkdir -p $OUT/logs
cd /root/dev
python /root/bf16_cast_check.py > $OUT/bf16_cast.txt 2>&1; log "bf16 cast rc=$?: $(cat $OUT/bf16_cast.txt)"
HD="--input test_clips/hair_1080 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30 --no_carry_hint --outputs alpha fg"
UHD="--input /root/tos/04_5h --frame_dir_linear --num_frames 16 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24 --no_carry_hint --outputs alpha fg"
EXR="--output_format exr --exr_dtype float"
for v in default new; do
  x=""; [ $v = new ] && x="--optimized --gpu_prep --async_baton"
  python infer.py $HD $EXR --output_dir $OUT/hd_$v --temp_dir $OUT/tmp $x > $OUT/logs/hd_$v.log 2>&1; log "hd exr $v rc=$?"
  python infer.py $HD --output_dir $OUT/hdv_$v --temp_dir $OUT/tmp $x > $OUT/logs/hdv_$v.log 2>&1; log "hd mp4 $v rc=$?"
  python infer.py $UHD $EXR --output_dir $OUT/uhd_$v --temp_dir $OUT/tmp $x > $OUT/logs/uhd_$v.log 2>&1; log "uhd exr $v rc=$?"
done
log "1080p EXR default vs new: $(python perf/same_files.py $OUT/hd_default $OUT/hd_new | head -1)"
log "1080p mp4 default vs new: $(python perf/same_files.py $OUT/hdv_default $OUT/hdv_new | head -1)"
log "4K EXR    default vs new: $(python perf/same_files.py $OUT/uhd_default $OUT/uhd_new | head -1)"
rm -rf $OUT/hd_* $OUT/hdv_* $OUT/uhd_* $OUT/tmp
run() {
  local tag=$1 dir=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  (cd $dir && python perf/infer_with.py --mem_budget_gib 12 -- --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@") > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier3f.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/tier3f.jsonl) | $(grep -o 'peak allocated.*' $OUT/logs/$tag.log)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
T="--input /root/tos/04_5h --frame_dir_linear --low_vram"
cat /root/tos/04_5h/*.exr /root/hcw/*/frames/* > /dev/null
run base42 /root/dev_base $T --fast --gpu_prep --hann_chunk 4 --hann_stride 2
run new42 /root/dev $T --fast --gpu_prep --hann_chunk 4 --hann_stride 2
run new42a /root/dev $T --fast --gpu_prep --async_baton --hann_chunk 4 --hann_stride 2
run base84 /root/dev_base $T --fast --gpu_prep --hann_chunk 8 --hann_stride 4
run new84a /root/dev $T --fast --gpu_prep --async_baton --hann_chunk 8 --hann_stride 4
run new42a2 /root/dev $T --fast --gpu_prep --async_baton --hann_chunk 4 --hann_stride 2
for clip in hair_detail motion_blur; do
  run base_$clip /root/dev_base --input /root/hcw/$clip/frames --fast --gpu_prep
  run new_$clip /root/dev --input /root/hcw/$clip/frames --fast --gpu_prep --async_baton
done
log "tier3f done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
