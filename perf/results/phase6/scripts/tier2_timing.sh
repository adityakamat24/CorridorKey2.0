#!/usr/bin/env bash
# The timing part of tier2.sh only (step 2), after the fixed accuracy step ran in tier2_resume.sh.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12 TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
OUT=/root/tier2; mkdir -p $OUT/logs
RES=$OUT/tier2.jsonl
warm() { cat /root/hcw/*/frames/* /root/tos/04_5h/*.exr > /dev/null; }
run() { # tag input command...
  local tag=$1 inp=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > $OUT/logs/$tag.smi &
  local smi=$!
  local t0=$(date +%s.%N)
  "$@" > $OUT/logs/$tag.log 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  kill $smi
  local wall=$(python -c "print(round($t1 - $t0, 1))")
  local smi_max=$(sort -n $OUT/logs/$tag.smi | tail -1)
  local auto=$(grep -m1 'Auto VRAM' $OUT/logs/$tag.log | sed 's/.*-> //')
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $wall, \"nvidia_smi_max_mib\": ${smi_max:-0}, \"auto\": \"$auto\"}" >> $RES
  log "$tag rc=$rc wall=${wall}s smi=${smi_max}MiB | $auto"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
ours() { local tag=$1 inp=$2; shift 2; run $tag "$inp" python perf/infer_with.py --mem_budget_gib 12 -- --input "$inp" --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@"; }
oursc() { local tag=$1 inp=$2; shift 2; run $tag "$inp" python perf/infer_with.py --mem_budget_gib 12 --compile default -- --input "$inp" --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@"; }
ckd() { local tag=$1 inp=$2 lin=${3:-false}; mkdir -p $OUT/out_$tag $OUT/tmp_$tag; run $tag "$inp" env LD_LIBRARY_PATH=$CKD_LD python /root/ckd_drive.py $CKD_ENGINE $CKD_MODELS $CKD_ORT "$inp" $OUT/out_$tag $OUT/tmp_$tag "{\"outputs\": [\"alpha\", \"fg\"], \"output_format\": \"video\", \"vram_budget_gb\": 12, \"frame_dir_fps\": 24, \"frame_dir_linear\": $lin}"; }
source /root/ckd_env.sh
warm
ours warmup /root/hcw/motion_blur/frames --optimized
for clip in hair_detail motion_blur; do
  dd=/root/hcw/$clip/frames
  warm
  ours opt_$clip $dd --optimized
  oursc optc_cold_$clip $dd --optimized
  oursc optc_$clip $dd --optimized
  ckd ckd_$clip $dd
  ours fast_$clip $dd --fast
  oursc fastc_$clip $dd --fast
  ours opt2_$clip $dd --optimized
  oursc optc2_$clip $dd --optimized
done
d=/root/tos/04_5h
warm
ours opt84L_tos $d --frame_dir_linear --optimized --hann_chunk 8 --hann_stride 4 --low_vram
oursc optc_cold_tos $d --frame_dir_linear --optimized --hann_chunk 8 --hann_stride 4 --low_vram
oursc optc_tos $d --frame_dir_linear --optimized --hann_chunk 8 --hann_stride 4 --low_vram
ckd ckd_tos $d true
oursc fastc_tos $d --frame_dir_linear --fast --hann_chunk 8 --hann_stride 4 --low_vram
ours fast84L_tos $d --frame_dir_linear --fast --hann_chunk 8 --hann_stride 4 --low_vram
cd $OUT && tar czf /root/tier2.tgz *.jsonl cmp_*.txt cmp_*.json logs/*.log 2>/dev/null
log "tier2 timing done"
echo "tier2 done" >> /root/tier2.log
