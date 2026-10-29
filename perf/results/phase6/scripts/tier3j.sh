#!/usr/bin/env bash
set -u
exec 9> /root/tier3j.lock
flock -n 9 || { echo "tier3j already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3i done" /root/tier3i.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 3600 > /dev/null 2>&1 < /dev/null 9>&- &
cd /root/dev2
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3j; mkdir -p $OUT
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12 --gpu_prep --async_baton"
cat /root/tos/04_5h/*.exr > /dev/null
py-spy record --idle -r 100 --format raw -o $OUT/spy_4x2.txt -- python perf/bench.py --input /root/tos/04_5h \
  --frame_dir_linear --frame_dir_fps 24 --num_frames 32 --hann_chunk 4 --hann_stride 2 --low_vram $OPT \
  --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "spy new 4K 4/2" --out $OUT/spy_4x2.json > $OUT/spy_4x2.log 2>&1
log "spy rc=$?: $(grep '\[bench\] median' $OUT/spy_4x2.log)"
python /root/spy_stage1.py $OUT/spy_4x2.txt hann_stage1_stream 100 > $OUT/stage1.txt 2>&1
python /root/spy_stage1.py $OUT/spy_4x2.txt _run_inference 100 > $OUT/run.txt 2>&1
python /root/spy_stage1.py $OUT/spy_4x2.txt precompute_prepasses_to_memmap 100 > $OUT/prepass.txt 2>&1
python - "$OUT/spy_4x2.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print({k: round(v["wall_s"], 2) for k, v in r["summary"]["phases"].items()})
PY
log "tier3j done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 < /dev/null 9>&- &
