#!/usr/bin/env bash
# Second RTX 5080 box: setup, then the competitor race of tier2f.sh (Desktop CUDA and
# TensorRT, alphastrata/CorridorKey2.0, ours --optimized) at equal settings. Stops itself.
set -u
nohup bash /root/selfstop.sh 12600 > /dev/null 2>&1 &
export DEBIAN_FRONTEND=noninteractive
log() { echo "[$(date +%H:%M:%S)] $*"; }
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
mkdir -p /root/ck && tar xzf /root/code.tgz -C /root/ck && cd /root/ck
find /root/ck -name '*.sh' -exec sed -i 's/\r$//' {} +
PLATES="hair_detail motion_blur" CLIPS="hair" bash perf/setup_vast.sh > /root/setup.log 2>&1
pip uninstall -q -y opencv-python 2>/dev/null; pip install -q opencv-python-headless==4.11.0.86
log "setup: $(tail -1 /root/setup.log)"
mkdir -p /root/tos/04_5h && cd /root/tos/04_5h
curl -s https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/ | grep -o 'href="[^"]*\.exr"' | sed 's/href="//; s/"$//' > /root/tos/list.txt
n=$(wc -l < /root/tos/list.txt); sed -n "$(( (n - 120) / 2 + 1 )),$(( (n - 120) / 2 + 120 ))p" /root/tos/list.txt > /root/tos/take.txt
xargs -P 8 -I{} curl -sSLO "https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/{}" < /root/tos/take.txt
log "tos frames: $(ls /root/tos/04_5h | grep -c exr)"
bash /root/ckd_setup.sh > /root/ckd_setup.log 2>&1; log "ckd: $(tail -1 /root/ckd_setup.log)"
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3; mkdir -p $OUT/logs
bash /root/ckd_trt_setup.sh > $OUT/logs/ckd_trt_setup.log 2>&1; log "ckd trt: $(tail -1 $OUT/logs/ckd_trt_setup.log)"
source /root/ckd_env.sh
git clone -q --depth 1 -b master https://github.com/alphastrata/CorridorKey2.0 /root/jer > $OUT/logs/jer_clone.log 2>&1
for d in weights vfms; do mkdir -p /root/jer/$d; cp -rsn /root/ck/$d/. /root/jer/$d/; done
log "jer: $(cd /root/jer && git log --oneline -1) | ptx sm: $(ls /root/jer/corridorkey/ptx/bin 2>/dev/null | tr '\n' ' ')"
RES=$OUT/tier2f.jsonl
warm() { cat /root/hcw/*/frames/* /root/tos/04_5h/*.exr > /dev/null; }
run() {
  local tag=$1 inp=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag; mkdir -p $OUT/out_$tag $OUT/tmp_$tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > $OUT/logs/$tag.smi &
  local smi=$!
  local t0=$(date +%s.%N)
  "$@" > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  kill $smi
  local smi_max=$(sort -n $OUT/logs/$tag.smi | tail -1)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))"), \"nvidia_smi_max_mib\": ${smi_max:-0}}" >> $RES
  log "$tag rc=$rc $(tail -1 $RES) | $(grep -m1 -oE 'hann_chunk[^,]*, hann_stride[^,]*|Hann stage1: chunk=[0-9]+, stride=[0-9]+|fallback[^"]*' $OUT/logs/$tag.log | head -1)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
ours() { local tag=$1 inp=$2; shift 2; run $tag "$inp" python perf/infer_with.py --mem_budget_gib 12 -- --input "$inp" --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 --optimized "$@"; }
jer() { local tag=$1 inp=$2; shift 2; run $tag "$inp" bash -c "cd /root/jer && python infer.py --input $inp --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 --outputs alpha fg $*"; }
ckd() { local tag=$1 inp=$2 opts=$3; run $tag "$inp" env LD_LIBRARY_PATH=$CKD_LD python /root/ckd_drive.py $CKD_ENGINE $CKD_MODELS $CKD_ORT "$inp" $OUT/out_$tag $OUT/tmp_$tag "$opts"; }
ckd_build() { # tag width height frames budget
  local t0=$(date +%s)
  env LD_LIBRARY_PATH=$CKD_LD python /root/ckd_drive.py $CKD_ENGINE $CKD_MODELS $CKD_ORT none /tmp /tmp \
    "{\"_trt_build\": {\"width\": $2, \"height\": $3, \"frames\": $4, \"vram_budget_gb\": $5}}" > $OUT/logs/build_$1.log 2>&1
  log "trt build $1 rc=$? in $(( $(date +%s) - t0 )) s: $(tail -1 $OUT/logs/build_$1.log | cut -c1-200)"
}
d=/root/tos/04_5h
NT=$(ls $d | grep -c exr)
ckd_build tos12 4096 2160 $NT 12
ckd_build tos16 4096 2160 $NT 16
V='"outputs": ["alpha", "fg"], "output_format": "video", "frame_dir_fps": 24, "frame_dir_linear": true'
warm
ckd ckd_cuda12_tos $d "{$V, \"vram_budget_gb\": 12}"
ckd ckd_trt12_tos $d "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
ours opt42_tos $d --frame_dir_linear --hann_chunk 4 --hann_stride 2 --low_vram
jer jer42_tos $d --frame_dir_linear --hann_chunk 4 --hann_stride 2 --low_vram
ckd ckd_cuda16_tos $d "{$V, \"vram_budget_gb\": 16}"
ckd ckd_trt16_tos $d "{$V, \"vram_budget_gb\": 16, \"tensorrt\": true, \"tensorrt_build\": false}"
ours opt84_tos $d --frame_dir_linear --hann_chunk 8 --hann_stride 4 --low_vram
jer jer84_tos $d --frame_dir_linear --hann_chunk 8 --hann_stride 4 --low_vram
ckd ckd_trt12b_tos $d "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
ours opt42b_tos $d --frame_dir_linear --hann_chunk 4 --hann_stride 2 --low_vram
V='"outputs": ["alpha", "fg"], "output_format": "video", "frame_dir_fps": 24'
for clip in hair_detail motion_blur; do
  dd=/root/hcw/$clip/frames
  read -r W H <<< "$(python -c "import cv2,sys,glob; im=cv2.imread(sorted(glob.glob(sys.argv[1]+'/*'))[0]); print(im.shape[1], im.shape[0])" $dd)"
  ckd_build $clip $W $H $(ls $dd | wc -l) 12
  warm
  ckd ckd_cuda_$clip $dd "{$V, \"vram_budget_gb\": 12}"
  ckd ckd_trt_$clip $dd "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
  ours opt_$clip $dd
  jer jer_$clip $dd
  jer jer4824_$clip $dd --hann_chunk 48 --hann_stride 24 --low_vram
  ours opt4824_$clip $dd --hann_chunk 48 --hann_stride 24 --low_vram
  ckd ckd_trt2_$clip $dd "{$V, \"vram_budget_gb\": 12, \"tensorrt\": true, \"tensorrt_build\": false}"
done
cd $OUT && tar czf /root/tier3.tgz *.jsonl logs/*.log 2>/dev/null
log "tier3 done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
