#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null &
for x in third_party scripts; do ln -sfn /root/ck/$x /root/ck2/$x; done
cd /root/ck2
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier2i; mkdir -p $OUT/logs
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for w in "8 4" "4 2"; do
  set -- $w
  for v in cpu gpu; do
    extra=""; [ $v = gpu ] && extra="--gpu_prep"
    python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --num_frames 32 --hann_chunk $1 --hann_stride $2 \
      --low_vram $OPT $extra --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "tos 4K $1/$2 $v prep v2" \
      --out $OUT/bench_$1x$2_$v.json > $OUT/logs/bench_$1x$2_$v.log 2>&1
    log "bench 4K $1/$2 $v prep: $(grep '\[bench\] median' $OUT/logs/bench_$1x$2_$v.log)"
    python - "$OUT/bench_$1x$2_$v.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for name, p in r["summary"]["phases"].items():
    print(f"    {name:<16} {p['wall_s']:8.2f} s   peak reserved {p.get('peak_reserved_gib') or 0:.2f} GiB")
PY
  done
done
log "tier2i done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1500 > /dev/null 2>&1 < /dev/null &
