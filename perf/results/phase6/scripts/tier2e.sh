#!/usr/bin/env bash
# After tier2d: --gpu_prep (frame conversions on the GPU in the decode_once prepass).
# 1. Output bytes against the CPU path, carry hint off (reproducible): 1080p PNG, 4K linear EXR.
# 2. 4K phase breakdown and end-to-end time with and without it.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2d done" /root/tier2d.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier2e; mkdir -p $OUT/logs
for v in cpu gpu; do
  extra=""; [ $v = gpu ] && extra="--gpu_prep"
  python infer.py --input test_clips/hair_1080 --output_dir $OUT/hd_$v --output_format exr --exr_dtype float \
    --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 30 --temp_dir $OUT/tmp \
    --optimized --no_carry_hint $extra > $OUT/logs/hd_$v.log 2>&1
  python infer.py --input /root/tos/04_5h --frame_dir_linear --output_dir $OUT/uhd_$v --output_format exr \
    --exr_dtype float --outputs alpha fg --num_frames 16 --hann_chunk 8 --hann_stride 4 --low_vram \
    --frame_dir_fps 24 --temp_dir $OUT/tmp --optimized --no_carry_hint $extra > $OUT/logs/uhd_$v.log 2>&1
done
log "gpu_prep vs cpu, 1080p PNG: $(python perf/same_files.py $OUT/hd_cpu $OUT/hd_gpu | head -1)"
log "gpu_prep vs cpu, 4K EXR:    $(python perf/same_files.py $OUT/uhd_cpu $OUT/uhd_gpu | head -1)"
python perf/compare.py --ref $OUT/hd_cpu --cand $OUT/hd_gpu > $OUT/cmp_hd.txt 2>&1; log "1080p: $(tail -n1 $OUT/cmp_hd.txt)"
python perf/compare.py --ref $OUT/uhd_cpu --cand $OUT/uhd_gpu > $OUT/cmp_uhd.txt 2>&1; log "4K: $(tail -n1 $OUT/cmp_uhd.txt)"
rm -rf $OUT/hd_* $OUT/uhd_* $OUT/tmp
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for v in cpu gpu; do
  extra=""; [ $v = gpu ] && extra="--gpu_prep"
  python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --hann_chunk 8 --hann_stride 4 \
    --low_vram $OPT $extra --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "tos 4K 8/4 $v prep" \
    --out $OUT/bench_tos_$v.json > $OUT/logs/bench_tos_$v.log 2>&1
  log "bench 4K $v prep: $(grep '\[bench\] median' $OUT/logs/bench_tos_$v.log)"
  python - "$OUT/bench_tos_$v.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"    load {r.get('load_s', 0):.2f} s")
for name, p in r["summary"]["phases"].items():
    print(f"    {name:<16} {p['wall_s']:8.2f} s   SM median {p.get('sm_mhz_median') or 0:6.0f} MHz")
EOF
done
for v in cpu gpu cpu2 gpu2; do
  extra=""; case $v in gpu*) extra="--gpu_prep";; esac
  t0=$(date +%s.%N)
  python perf/infer_with.py --mem_budget_gib 12 -- --input /root/tos/04_5h --frame_dir_linear --output_dir $OUT/out_$v \
    --temp_dir $OUT/tmp_$v --frame_dir_fps 24 --optimized --hann_chunk 8 --hann_stride 4 --low_vram $extra > $OUT/logs/e2e_$v.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  echo "{\"tag\": \"tos_$v\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))")}" >> $OUT/tier2e.jsonl
  log "tos_$v rc=$rc $(tail -1 $OUT/tier2e.jsonl)"
  rm -rf $OUT/out_$v $OUT/tmp_$v
done
cd $OUT && tar czf /root/tier2e.tgz *.json *.jsonl *.txt logs/*.log 2>/dev/null
log "tier2e done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
