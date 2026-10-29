#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null &
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/root/tier3q; mkdir -p $OUT/logs
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for w in "8 4" "4 2"; do
  set -- $w
  for v in old new; do
    for prep in cpu gpu; do
      case $v in old) dir=/root/dev3;; new) dir=/root/dev4;; esac
      x=""; [ $prep = gpu ] && x="--gpu_prep"
      (cd $dir && python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --num_frames 32 --hann_chunk $1 --hann_stride $2 \
        --low_vram $OPT $x --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "$v $prep $1/$2" --out $OUT/${v}_${prep}_$1x$2.json) > $OUT/logs/${v}_${prep}_$1x$2.log 2>&1
      python - "$OUT/${v}_${prep}_$1x$2.json" "$v $prep $1/$2" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
p = r["summary"]["phases"]
print(f"[{sys.argv[2]}] prepass {p['prepass']['wall_s']:.2f} s, prepass peak allocated {p['prepass'].get('peak_allocated_gib', 0):.2f} / reserved {p['prepass']['peak_reserved_gib']:.2f} GiB; "
      f"run peak allocated {max(x.get('peak_allocated_gib', 0) for x in p.values()):.2f} / reserved {max(x['peak_reserved_gib'] for x in p.values()):.2f} GiB", flush=True)
PY
    done
  done
done
log "tier3q done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 < /dev/null &
