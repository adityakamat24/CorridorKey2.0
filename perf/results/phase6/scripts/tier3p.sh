#!/usr/bin/env bash
set -u
exec 9> /root/tier3p.lock
flock -n 9 || { echo "tier3p already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3o done" /root/tier3o.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
rm -rf /root/dev4; cp -a /root/dev3 /root/dev4
cp /root/upload_p/runtime.py /root/dev4/corridorkey/runtime.py; cp /root/upload_p/infer.py /root/dev4/infer.py
sed -i "s/\r$//" /root/dev4/corridorkey/runtime.py /root/dev4/infer.py
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3p; mkdir -p $OUT/logs
CPUOPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations"
for clip in hd uhd; do
  case $clip in
    hd) A="--input test_clips/hair_1080 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30";;
    uhd) A="--input /root/tos/04_5h --frame_dir_linear --num_frames 16 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24";;
  esac
  A="$A --no_carry_hint --outputs alpha fg --output_format exr --exr_dtype float"
  (cd /root/dev4 && python infer.py $A --output_dir $OUT/${clip}_def --temp_dir $OUT/tmp) > $OUT/logs/${clip}_def.log 2>&1; log "$clip default rc=$?"
  (cd /root/dev4 && python infer.py $A --output_dir $OUT/${clip}_opt --temp_dir $OUT/tmp --optimized) > $OUT/logs/${clip}_opt.log 2>&1; log "$clip --optimized rc=$?"
  (cd /root/dev4 && python infer.py $A --output_dir $OUT/${clip}_cpu --temp_dir $OUT/tmp $CPUOPT) > $OUT/logs/${clip}_cpu.log 2>&1; log "$clip cpu prep rc=$?"
  log "$clip default vs --optimized: $(cd /root/dev4 && python perf/same_files.py $OUT/${clip}_def $OUT/${clip}_opt | head -1)"
  log "$clip default vs cpu-prep options: $(cd /root/dev4 && python perf/same_files.py $OUT/${clip}_def $OUT/${clip}_cpu | head -1)"
  rm -rf $OUT/${clip}_* $OUT/tmp
done
run() {
  local tag=$1 dir=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  (cd $dir && python perf/infer_with.py --mem_budget_gib 12 -- --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@") > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier3p.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/tier3p.jsonl) | $(grep -o 'peak allocated.*' $OUT/logs/$tag.log)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
T="--input /root/tos/04_5h --frame_dir_linear --low_vram --optimized"
cat /root/tos/04_5h/*.exr > /dev/null
run old84 /root/dev3 $T --hann_chunk 8 --hann_stride 4
run new84 /root/dev4 $T --hann_chunk 8 --hann_stride 4
run old42 /root/dev3 $T --hann_chunk 4 --hann_stride 2
run new42 /root/dev4 $T --hann_chunk 4 --hann_stride 2
run new84b /root/dev4 $T --hann_chunk 8 --hann_stride 4
cd $OUT && tar czf /root/tier3p.tgz *.jsonl logs/*.log 2>/dev/null
log "tier3p done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 < /dev/null 9>&- &
