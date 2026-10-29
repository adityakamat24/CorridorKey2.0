#!/usr/bin/env bash
set -u
cd "/d/corridor key/CorridorKey2.0"
OUT=/d/ck_laptop_check
PY=venv/Scripts/python.exe
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "laptop check done" "${SCRATCH:-.}/laptop_check.log" 2>/dev/null; do sleep 10; done
A="--input test_clips/hair_1080 --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30 --optimized"
$PY infer.py $A --output_dir $OUT/opt --temp_dir $OUT/tmp > $OUT/logs/opt.log 2>&1; log "old --optimized rc=$?"
$PY infer.py $A --output_dir $OUT/new --temp_dir $OUT/tmp --gpu_prep --async_baton > $OUT/logs/new2.log 2>&1; log "new rc=$?"
log "hair_1080 8/4 carry hint on, --optimized vs + --gpu_prep --async_baton: $($PY perf/same_files.py $OUT/opt $OUT/new | head -1)"
rm -rf $OUT/opt $OUT/new $OUT/tmp
log "laptop exact done"
