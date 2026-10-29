#!/usr/bin/env bash
set -u
cd "/d/corridor key/CorridorKey2.0"
OUT=/d/ck_laptop_check
mkdir -p $OUT/logs
PY=venv/Scripts/python.exe
log() { echo "[$(date +%H:%M:%S)] $*"; }
A="--input test_clips/hair_1080 --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30"
$PY infer.py $A --output_dir $OUT/f_def --temp_dir $OUT/f_tmp > $OUT/logs/final_def.log 2>&1; log "default rc=$?"
$PY infer.py $A --output_dir $OUT/f_new --temp_dir $OUT/f_tmp --optimized > $OUT/logs/final_new.log 2>&1; log "--optimized (now with --gpu_prep --async_baton) rc=$?"
log "hair_1080 8/4 carry hint on, default vs --optimized: $($PY perf/same_files.py $OUT/f_def $OUT/f_new | head -1)"
log "laptop final done"
