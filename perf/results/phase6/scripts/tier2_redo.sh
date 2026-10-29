#!/usr/bin/env bash
# Redo of the runs that failed while cv2 was missing: tier2 timings (ours, compile cold/warm,
# Desktop again) and the 4K phase breakdown of tier2b. Runs after tier2c.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2c done" /root/tier2c.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12 TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
OUT=/root/tier2; mkdir -p $OUT/logs
mv $OUT/tier2.jsonl $OUT/tier2_nocv2.jsonl 2>/dev/null
RES=$OUT/tier2.jsonl
warm() { cat /root/hcw/*/frames/* /root/tos/04_5h/*.exr > /dev/null; }
run() { # tag input command...
  local tag=$1 inp=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.25; done ) > $OUT/logs/$tag.smi &
  local smi=$!
  local t0=$(date +%s.%N)
  "$@" > $OUT/logs/$tag.log 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  kill $smi
  local wall=$(python -c "print(round($t1 - $t0, 1))")
  local smi_max=$(sort -n $OUT/logs/$tag.smi | tail -1)
  local auto=$(grep -m1 'Auto VRAM' $OUT/logs/$tag.log | sed 's/.*-> //')
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $wall, \"nvidia_smi_max_mib\": ${smi_max:-0}, \"auto\": \"$auto\"}" >> $RES
  log "$tag rc=$rc wall=${wall}s smi=${smi_max}MiB | $auto"
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
}
ours() { local tag=$1 inp=$2; shift 2; run $tag "$inp" python perf/infer_with.py --mem_budget_gib 12 -- --input "$inp" --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@"; }
oursc() { local tag=$1 inp=$2; shift 2; run $tag "$inp" python perf/infer_with.py --mem_budget_gib 12 --compile default -- --input "$inp" --output_dir $OUT/out_$tag --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 "$@"; }
ckd() { local tag=$1 inp=$2 lin=${3:-false}; mkdir -p $OUT/out_$tag $OUT/tmp_$tag; run $tag "$inp" env LD_LIBRARY_PATH=$CKD_LD python /root/ckd_drive.py $CKD_ENGINE $CKD_MODELS $CKD_ORT "$inp" $OUT/out_$tag $OUT/tmp_$tag "{\"outputs\": [\"alpha\", \"fg\"], \"output_format\": \"video\", \"vram_budget_gb\": 12, \"frame_dir_fps\": 24, \"frame_dir_linear\": $lin}"; }
source /root/ckd_env.sh
warm
ours warmup /root/hcw/motion_blur/frames --optimized
for clip in hair_detail motion_blur; do
  dd=/root/hcw/$clip/frames
  warm
  ours opt_$clip $dd --optimized
  oursc optc_cold_$clip $dd --optimized
  oursc optc_$clip $dd --optimized
  ckd ckd_$clip $dd
  ours fast_$clip $dd --fast
  oursc fastc_$clip $dd --fast
  ours opt2_$clip $dd --optimized
  oursc optc2_$clip $dd --optimized
done
d=/root/tos/04_5h
warm
ours opt84L_tos $d --frame_dir_linear --optimized --hann_chunk 8 --hann_stride 4 --low_vram
oursc optc_cold_tos $d --frame_dir_linear --optimized --hann_chunk 8 --hann_stride 4 --low_vram
oursc optc_tos $d --frame_dir_linear --optimized --hann_chunk 8 --hann_stride 4 --low_vram
ckd ckd_tos $d true
oursc fastc_tos $d --frame_dir_linear --fast --hann_chunk 8 --hann_stride 4 --low_vram
ours fast84L_tos $d --frame_dir_linear --fast --hann_chunk 8 --hann_stride 4 --low_vram
OUT=/root/tier2b; mkdir -p $OUT/logs
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for w in "8 4" "4 2"; do
  set -- $w
  python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 \
    --hann_chunk $1 --hann_stride $2 --low_vram $OPT --warmup 0 --repeats 1 \
    --temp_dir $OUT/tmp --label "tos 4K $1/$2 optimized" --out $OUT/bench_tos_$1x$2.json \
    > $OUT/logs/bench_tos_$1x$2.log 2>&1
  log "bench 4K $1/$2: $(grep '\[bench\] median' $OUT/logs/bench_tos_$1x$2.log)"
  python - "$OUT/bench_tos_$1x$2.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"    load {r.get('load_s', 0):.2f} s")
for name, p in r["summary"]["phases"].items():
    print(f"    {name:<16} {p['wall_s']:8.2f} s   SM median {p.get('sm_mhz_median') or 0:6.0f} MHz"
          f"   peak reserved {p['peak_reserved_gib']:.2f} GiB")
EOF
done
cd /root && tar czf /root/tier2_redo.tgz tier2/*.jsonl tier2/logs/*.log tier2b/*.json tier2b/logs/bench*.log 2>/dev/null
log "redo done"
