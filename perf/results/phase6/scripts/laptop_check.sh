#!/usr/bin/env bash
set -u
cd "/d/corridor key/CorridorKey2.0"
OUT=/d/ck_laptop_check; rm -rf $OUT; mkdir -p $OUT/logs
PY=venv/Scripts/python.exe
log() { echo "[$(date +%H:%M:%S)] $*"; }
A="--input test_clips/hair_1080 --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30"
$PY infer.py $A --output_dir $OUT/def --temp_dir $OUT/tmp > $OUT/logs/def.log 2>&1; log "default rc=$?"
$PY infer.py $A --output_dir $OUT/new --temp_dir $OUT/tmp --optimized --gpu_prep --async_baton > $OUT/logs/new.log 2>&1; log "new rc=$?"
log "hair_1080 8/4 carry hint on, default vs --optimized --gpu_prep --async_baton: $($PY perf/same_files.py $OUT/def $OUT/new | head -1)"
$PY perf/compare.py --ref $OUT/def --cand $OUT/new > $OUT/cmp.txt 2>&1; log "$(tail -n1 $OUT/cmp.txt)"
rm -rf $OUT/def $OUT/new $OUT/tmp
for v in old new new2 old2; do
  case $v in old*) x="--optimized";; new*) x="--optimized --gpu_prep --async_baton";; esac
  $PY perf/laptop_run.py --input test_clips/converge_1080 --output_dir $OUT/out_$v --temp_dir $OUT/tmp_$v --frame_dir_fps 30 $x > $OUT/logs/e2e_$v.log 2>&1
  log "converge_1080 $v: $(grep -o 'hann_chunk=[0-9]*, hann_stride=[0-9]*' $OUT/logs/e2e_$v.log | head -1) | $(grep '^\[laptop\] ok\|^\[laptop\] OOM' $OUT/logs/e2e_$v.log)"
  rm -rf $OUT/out_$v $OUT/tmp_$v
done
log "laptop check done"
