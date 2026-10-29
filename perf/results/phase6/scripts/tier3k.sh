#!/usr/bin/env bash
set -u
exec 9> /root/tier3k.lock
flock -n 9 || { echo "tier3k already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3j done" /root/tier3j.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3k; mkdir -p $OUT/logs
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12 --gpu_prep"
B="--input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --num_frames 64 --hann_chunk 8 --hann_stride 4 --low_vram $OPT --warmup 0 --repeats 1 --temp_dir $OUT/tmp"
cat /root/tos/04_5h/*.exr > /dev/null
bench() {
  local tag=$1 dir=$2; shift 2
  (cd $dir && python /root/rss_run.py python perf/bench.py $B "$@" --label "$tag" --out $OUT/$tag.json) > $OUT/logs/$tag.log 2>&1
  log "$tag: $(grep '\[bench\] median' $OUT/logs/$tag.log) $(grep '\[rss\]' $OUT/logs/$tag.log)"
  python - "$OUT/$tag.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("    " + "  ".join(f"{k} {v['wall_s']:.1f}s" for k, v in r["summary"]["phases"].items() if v["wall_s"] > 0.5))
PY
}
bench base_84 /root/dev_base
bench sync_84 /root/dev2
bench async_84 /root/dev2 --async_baton
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_use_cuda_host_register:True bench async_reg_84 /root/dev2 --async_baton
bench base_84b /root/dev_base
bench async_84b /root/dev2 --async_baton
log "tier3k done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
