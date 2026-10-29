#!/usr/bin/env bash
set -u
exec 9> /root/tier2h.lock
flock -n 9 || { echo "tier2h already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2g done" /root/tier2g.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null &
rm -rf /root/ck2; mkdir -p /root/ck2
cp -r /root/ck/corridorkey /root/ck/perf /root/ck/infer.py /root/ck2/
for d in weights vfms test_clips; do ln -s /root/ck/$d /root/ck2/$d; done
cp /root/upload_h/video_io.py /root/upload_h/runtime.py /root/ck2/corridorkey/
cp /root/upload_h/gpu_prep_probe.py /root/ck2/perf/
cd /root/ck2
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier2h; mkdir -p $OUT/logs
python perf/gpu_prep_probe.py --stride 1 > $OUT/probe.txt 2>&1
log "probe rc=$?"; sed 's/^/    /' $OUT/probe.txt
HD="--input test_clips/hair_1080 --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30 --no_carry_hint"
UHD="--input /root/tos/04_5h --frame_dir_linear --output_format exr --exr_dtype float --outputs alpha fg --num_frames 16 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24 --no_carry_hint"
for v in default opt opt_gpu; do
  case $v in default) x="";; opt) x="--optimized";; opt_gpu) x="--optimized --gpu_prep";; esac
  python infer.py $HD --output_dir $OUT/hd_$v --temp_dir $OUT/tmp $x > $OUT/logs/hd_$v.log 2>&1; log "hd_$v rc=$?"
  python infer.py $UHD --output_dir $OUT/uhd_$v --temp_dir $OUT/tmp $x > $OUT/logs/uhd_$v.log 2>&1; log "uhd_$v rc=$?"
done
for v in opt opt_gpu; do
  log "1080p PNG default vs $v: $(python perf/same_files.py $OUT/hd_default $OUT/hd_$v | head -1)"
  log "4K EXR    default vs $v: $(python perf/same_files.py $OUT/uhd_default $OUT/uhd_$v | head -1)"
done
rm -rf $OUT/hd_* $OUT/uhd_* $OUT/tmp
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for v in cpu gpu; do
  extra=""; [ $v = gpu ] && extra="--gpu_prep"
  python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --hann_chunk 8 --hann_stride 4 \
    --low_vram $OPT $extra --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "tos 4K 8/4 $v prep v2" \
    --out $OUT/bench_tos_$v.json > $OUT/logs/bench_tos_$v.log 2>&1
  log "bench 4K $v prep: $(grep '\[bench\] median' $OUT/logs/bench_tos_$v.log)"
  python - "$OUT/bench_tos_$v.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
run = r["runs"][-1] if r.get("runs") else {}
for name, p in r["summary"]["phases"].items():
    print(f"    {name:<16} {p['wall_s']:8.2f} s   peak reserved {p.get('peak_reserved_gib', p.get('reserved_peak_gib', 0)) or 0:.2f} GiB")
PY
done
for v in fast_gpu84 fast_gpu42 fast_gpu84b; do
  case $v in fast_gpu84*) w="--hann_chunk 8 --hann_stride 4";; fast_gpu42) w="--hann_chunk 4 --hann_stride 2";; esac
  t0=$(date +%s.%N)
  python perf/infer_with.py --mem_budget_gib 12 -- --input /root/tos/04_5h --frame_dir_linear --output_dir $OUT/out_$v \
    --temp_dir $OUT/tmp_$v --frame_dir_fps 24 --fast --gpu_prep $w --low_vram > $OUT/logs/e2e_$v.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  echo "{\"tag\": \"tos_$v\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier2h.jsonl
  log "tos_$v rc=$rc $(tail -1 $OUT/tier2h.jsonl) | $(grep -o 'peak allocated.*' $OUT/logs/e2e_$v.log)"
  rm -rf $OUT/out_$v $OUT/tmp_$v
done
cd $OUT && tar czf /root/tier2h.tgz *.txt *.json *.jsonl logs/*.log 2>/dev/null
log "tier2h done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 1200 > /dev/null 2>&1 < /dev/null &
