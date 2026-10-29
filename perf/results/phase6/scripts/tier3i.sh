#!/usr/bin/env bash
set -u
exec 9> /root/tier3i.lock
flock -n 9 || { echo "tier3i already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3h done" /root/tier3h.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3i; mkdir -p $OUT/logs
rm -rf /root/dev_old /root/dev_flip /root/dev_ls
cp -a /root/dev_base /root/dev_old; cp /root/ck/corridorkey/runtime.py /root/ck/corridorkey/video_io.py /root/dev_old/corridorkey/
sed -i "s/\r$//" /root/dev_old/corridorkey/*.py
cp -a /root/dev_base /root/dev_flip; cp /root/dev/corridorkey/runtime.py /root/dev_flip/corridorkey/
cp -a /root/dev_base /root/dev_ls; cp /root/dev/corridorkey/stage1.py /root/dev_ls/corridorkey/
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for w in "8 4" "4 2"; do
  set -- $w
  for v in old new; do
    case $v in old) dir=/root/dev_old;; new) dir=/root/dev2;; esac
    (cd $dir && python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --num_frames 32 \
      --hann_chunk $1 --hann_stride $2 --low_vram $OPT --warmup 0 --repeats 1 --temp_dir $OUT/tmp \
      --label "prepass memory $v $1/$2" --out $OUT/mem_$1x$2_$v.json) > $OUT/logs/mem_$1x$2_$v.log 2>&1
    log "4K $1/$2 cpu prep, $v code: $(grep '\[bench\] median' $OUT/logs/mem_$1x$2_$v.log)"
    python - "$OUT/mem_$1x$2_$v.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
p = r["summary"]["phases"]
print(f"    prepass {p['prepass']['wall_s']:.2f} s peak reserved {p['prepass']['peak_reserved_gib']:.2f} GiB; "
      f"stage1 peak {p['stage1']['peak_reserved_gib']:.2f} GiB")
PY
  done
done
run() {
  local tag=$1 dir=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  (cd $dir && python perf/infer_with.py --mem_budget_gib 12 -- --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@") > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier3i.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/tier3i.jsonl)"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
T="--input /root/tos/04_5h --frame_dir_linear --low_vram --fast --gpu_prep --hann_chunk 4 --hann_stride 2"
run flip42 /root/dev_flip $T
run ls42 /root/dev_ls $T
run base42_c /root/dev_base $T
log "tier3i done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
