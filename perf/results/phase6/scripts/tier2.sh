#!/usr/bin/env bash
# Tier 2 on one RTX 5080 (the race3 host: Ryzen 9 5900XT, driver 595).
# 1. Accuracy: every network in fp32 as the reference, then the original
#    (bf16), --optimized and --optimized + torch.compile against it, 1080p
#    goldens (hair, smoke, converge; 48 frames, 8/4 low-VRAM). The bar for an
#    approximate option is the original's own error against fp32.
# 2. Timings with torch.compile on top of --optimized / --fast, cold and warm
#    compile cache, HD plates and 4K Tears of Steel, with CorridorKey
#    Desktop's CUDA engine as the same-session reference (race3 method).
# Stops itself (selfstop.sh) if left alone.
set -u
nohup bash /root/selfstop.sh 14400 > /dev/null 2>&1 &
export DEBIAN_FRONTEND=noninteractive
log() { echo "[$(date +%H:%M:%S)] $*"; }
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
mkdir -p /root/ck && tar xzf /root/code.tgz -C /root/ck && cd /root/ck
find /root/ck -name '*.sh' -exec sed -i 's/\r$//' {} +
PLATES="hair_detail smoke converge motion_blur" CLIPS="hair smoke converge" bash perf/setup_vast.sh > /root/setup.log 2>&1
pip uninstall -q -y opencv-python 2>/dev/null; pip install -q opencv-python-headless==4.11.0.86
log "setup: $(tail -1 /root/setup.log)"
mkdir -p /root/tos/04_5h && cd /root/tos/04_5h
curl -s https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/ | grep -o 'href="[^"]*\.exr"' | sed 's/href="//; s/"$//' > /root/tos/list.txt
n=$(wc -l < /root/tos/list.txt); sed -n "$(( (n - 120) / 2 + 1 )),$(( (n - 120) / 2 + 120 ))p" /root/tos/list.txt > /root/tos/take.txt
xargs -P 8 -I{} curl -sSLO "https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/{}" < /root/tos/take.txt
log "tos frames: $(ls /root/tos/04_5h | grep -c exr)"
bash /root/ckd_setup.sh > /root/ckd_setup.log 2>&1; log "ckd: $(tail -1 /root/ckd_setup.log)"
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12 TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache
OUT=/root/tier2; mkdir -p $OUT/logs
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)" 2>&1 | sed 's/^/[versions] /'

# 1. Accuracy against fp32.
G="python perf/goldens.py --clips hair smoke converge --res 1080"
$G --tag fp32 --fp32 > $OUT/logs/g_fp32.log 2>&1; log "fp32: $(grep -c ': ok in' $OUT/logs/g_fp32.log)/3 ok"
$G --tag orig > $OUT/logs/g_orig.log 2>&1; log "orig: $(grep -c ': ok in' $OUT/logs/g_orig.log)/3 ok"
$G --tag opt --extra "--optimized" > $OUT/logs/g_opt.log 2>&1; log "opt: $(grep -c ': ok in' $OUT/logs/g_opt.log)/3 ok"
TORCH_LOGS="graph_breaks,recompiles" $G --tag optc --compile default --extra "--optimized" > $OUT/logs/g_optc.log 2>&1
log "optc: $(grep -c ': ok in' $OUT/logs/g_optc.log)/3 ok"
for tag in orig opt optc; do
  for clip in hair smoke converge; do
    [ -d perf/out/$tag/${clip}_1080 ] || continue
    python perf/compare.py --ref perf/out/fp32/${clip}_1080 --cand perf/out/$tag/${clip}_1080 \
      --out $OUT/cmp_${tag}_${clip}.json > $OUT/cmp_${tag}_${clip}.txt 2>&1
    log "$tag vs fp32: $(tail -n1 $OUT/cmp_${tag}_${clip}.txt)"
  done
done
for t in fp32 orig opt optc; do
  cp perf/out/$t/runs.jsonl $OUT/runs_$t.jsonl 2>/dev/null
  for f in perf/out/$t/*.log; do cp "$f" "$OUT/logs/g_${t}_$(basename "$f")" 2>/dev/null; done
done
rm -rf perf/out/orig perf/out/opt perf/out/_tmp

# 2. Timings (race3 method), inputs in the page cache, model load included.
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
cd $OUT && tar czf /root/tier2.tgz *.jsonl cmp_*.txt cmp_*.json logs/*.log 2>/dev/null
log "tier2 done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
