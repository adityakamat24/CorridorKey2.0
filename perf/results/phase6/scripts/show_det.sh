#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
D=/root/det; mkdir -p $D/logs
A="--input /root/clips/hair_detail --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24"
python /root/det_run.py $A --output_dir $D/orig --temp_dir $D/tmp > $D/logs/orig.log 2>&1; log "deterministic original rc=$?"
python /root/det_run.py $A --output_dir $D/orig2 --temp_dir $D/tmp > $D/logs/orig2.log 2>&1; log "deterministic original 2 rc=$?"
python /root/det_run.py $A --output_dir $D/opt --temp_dir $D/tmp --optimized > $D/logs/opt.log 2>&1; log "deterministic optimized rc=$?"
log "deterministic cuDNN, original vs original 2: $(python perf/same_files.py $D/orig $D/orig2 | head -1)"
log "deterministic cuDNN, original vs optimized: $(python perf/same_files.py $D/orig $D/opt | head -1)"

log "det done"
