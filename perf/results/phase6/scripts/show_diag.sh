#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "show done" /root/show.log 2>/dev/null; do sleep 15; done
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/show; D=/root/diag; mkdir -p $D/logs
python - <<'PY' > $D/flags.txt 2>&1
import torch; print("cudnn.benchmark default:", torch.backends.cudnn.benchmark, "deterministic:", torch.backends.cudnn.deterministic)
PY
grep -rn "cudnn.benchmark\|use_deterministic\|cudnn.deterministic" corridorkey/*.py infer.py | head >> $D/flags.txt
A="--input /root/clips/hair_detail --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24"
python infer.py $A --output_dir $D/orig2 --temp_dir $D/tmp > $D/logs/orig2.log 2>&1; log "original run 2 rc=$?"
log "original vs original run 2: $(python perf/same_files.py $OUT/hair_detail_orig $D/orig2 | head -1)"
python perf/compare.py --ref $OUT/hair_detail_orig --cand $OUT/hair_detail_opt > $D/cmp_orig_opt.txt 2>&1; log "original vs optimized: $(tail -n1 $D/cmp_orig_opt.txt)"
python perf/compare.py --ref $OUT/hair_detail_orig --cand $D/orig2 > $D/cmp_orig_orig2.txt 2>&1; log "original vs original 2: $(tail -n1 $D/cmp_orig_orig2.txt)"
python infer.py $A --no_carry_hint --output_dir $D/orig_nch --temp_dir $D/tmp > $D/logs/orig_nch.log 2>&1
python infer.py $A --no_carry_hint --output_dir $D/opt_nch --temp_dir $D/tmp --optimized > $D/logs/opt_nch.log 2>&1
log "carry hint off, original vs optimized: $(python perf/same_files.py $D/orig_nch $D/opt_nch | head -1)"
python perf/compare.py --ref $D/orig_nch --cand $D/opt_nch > $D/cmp_nch.txt 2>&1; log "carry hint off metrics: $(tail -n1 $D/cmp_nch.txt)"
log "flags: $(tr '\n' ' ' < $D/flags.txt)"
log "diag done"
