#!/usr/bin/env bash
set -u
exec 9> /root/tier3d.lock
flock -n 9 || { echo "tier3d already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "waiting for tier3c"
until grep -q "tier3c done" /root/tier3c.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 < /dev/null &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3d; mkdir -p $OUT/logs
RES=$OUT/tier3d.jsonl
rm -rf /root/jer
git clone -q --depth 1 -b master https://github.com/alphastrata/CorridorKey2.0 /root/jer > $OUT/logs/jer_clone.log 2>&1
for d in weights vfms; do mkdir -p /root/jer/$d; cp -rsn /root/ck/$d/. /root/jer/$d/; done
(cd /root/jer && python infer.py --help) > $OUT/logs/jer_help.txt 2>&1
log "jer: $(cd /root/jer && git log --oneline -1 | cut -c1-100) | ptx: $(ls /root/jer/corridorkey/ptx/bin 2>/dev/null | tr '\n' ' ')"
JT=""; grep -q -- "--ffmpeg_threads" $OUT/logs/jer_help.txt && JT="--ffmpeg_threads 8"
source /root/ckd_env.sh
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
jer() { local tag=$1 inp=$2; shift 2; run $tag bash -c "cd /root/jer && python infer.py --input $inp --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 --outputs alpha fg $*"; }
ckd() { local tag=$1 inp=$2 opts=$3; run $tag env LD_LIBRARY_PATH=$CKD_LD python /root/ckd_drive.py $CKD_ENGINE $CKD_MODELS $CKD_ORT "$inp" $OUT/out_$tag $OUT/tmp_$tag "$opts"; }
ckd_build() {
  local t0=$(date +%s)
  env LD_LIBRARY_PATH=$CKD_LD python /root/ckd_drive.py $CKD_ENGINE $CKD_MODELS $CKD_ORT none /tmp /tmp \
    "{\"_trt_build\": {\"width\": $2, \"height\": $3, \"frames\": $4, \"vram_budget_gb\": $5}}" > $OUT/logs/build_$1.log 2>&1
  log "trt build $1 rc=$? in $(( $(date +%s) - t0 )) s: $(tail -1 $OUT/logs/build_$1.log | cut -c1-200)"
}
d=/root/tos/04_5h
L="--frame_dir_linear"
ckd_build tos12 4096 2160 $(ls $d | grep -c exr) 12
V='"outputs": ["alpha", "fg"], "output_format": "video", "frame_dir_fps": 24, "frame_dir_linear": true'
warm
ckd ckd_cuda12_tos $d "{$V, \"vram_budget_gb\": 12}"
ckd ckd_trt12_tos $d "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
jer jer42_tos $d $L --hann_chunk 4 --hann_stride 2 --low_vram
jer jer84_tos $d $L --hann_chunk 8 --hann_stride 4 --low_vram
[ -n "$JT" ] && jer jer84t8_tos $d $L --hann_chunk 8 --hann_stride 4 --low_vram $JT
ours best84r_tos $d $L --fast --gpu_prep --hann_chunk 8 --hann_stride 4 --low_vram
ckd ckd_trt12r_tos $d "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
V='"outputs": ["alpha", "fg"], "output_format": "video", "frame_dir_fps": 24'
for clip in hair_detail motion_blur; do
  dd=/root/hcw/$clip/frames
  read -r W H <<< "$(python -c "import cv2,sys,glob; im=cv2.imread(sorted(glob.glob(sys.argv[1]+'/*'))[0]); print(im.shape[1], im.shape[0])" $dd)"
  ckd_build $clip $W $H $(ls $dd | wc -l) 12
  warm
  ckd ckd_cuda_$clip $dd "{$V, \"vram_budget_gb\": 12}"
  ckd ckd_trt_$clip $dd "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
  jer jer_$clip $dd
  jer jer8040_$clip $dd --hann_chunk 80 --hann_stride 40
  jer jer4824_$clip $dd --hann_chunk 48 --hann_stride 24 --low_vram
  ours bestr_$clip $dd --fast --gpu_prep
done
cd $OUT && tar czf /root/tier3d.tgz *.jsonl logs/*.log logs/*.txt 2>/dev/null
log "tier3d done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1200 > /dev/null 2>&1 < /dev/null &
