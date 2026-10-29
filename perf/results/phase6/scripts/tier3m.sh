#!/usr/bin/env bash
set -u
exec 9> /root/tier3m.lock
flock -n 9 || { echo "tier3m already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3g2 done" /root/tier3g2.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
rm -rf /root/dev3; cp -a /root/dev2 /root/dev3
cp /root/upload_l/runtime.py /root/upload_l/video_io.py /root/upload_l/stage1.py /root/dev3/corridorkey/
cp /root/upload_l/infer.py /root/dev3/infer.py; cp /root/upload_l/gpu_prep_probe.py /root/dev3/perf/
sed -i "s/\r$//" /root/dev3/corridorkey/*.py /root/dev3/infer.py /root/dev3/perf/*.py
cd /root/dev3
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3m; mkdir -p $OUT/logs
python perf/gpu_prep_probe.py > $OUT/probe.txt 2>&1; log "probe rc=$?"; sed 's/^/    /' $OUT/probe.txt
[ -d /root/tos_f32 ] || python /root/make_f32_clip.py /root/tos/04_5h /root/tos_f32 16 > $OUT/f32.txt 2>&1
python -c "from corridorkey import video_io as v; import glob; print('ToS RGB half:', v.exr_rgb_is_half(sorted(glob.glob('/root/tos/04_5h/*.exr'))[0]), '| f32 clip RGB half:', v.exr_rgb_is_half(sorted(glob.glob('/root/tos_f32/*.exr'))[0]))" > $OUT/header.txt 2>&1
log "$(cat $OUT/header.txt)"
for clip in tos_f32 04_5h; do
  case $clip in tos_f32) inp=/root/tos_f32;; *) inp=/root/tos/04_5h;; esac
  A="--input $inp --frame_dir_linear --num_frames 16 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24 --no_carry_hint --outputs alpha fg --output_format exr --exr_dtype float"
  for v in default new; do
    x=""; [ $v = new ] && x="--optimized --gpu_prep --async_baton"
    python infer.py $A --output_dir $OUT/${clip}_$v --temp_dir $OUT/tmp $x > $OUT/logs/${clip}_$v.log 2>&1
    log "$clip $v rc=$?"
  done
  log "$clip default vs --optimized --gpu_prep --async_baton: $(python perf/same_files.py $OUT/${clip}_default $OUT/${clip}_new | head -1)"
  rm -rf $OUT/${clip}_* $OUT/tmp
done
OPT="--bf16_weights --frozen_weights --lean_activations --prefetch_frames --cache_region_indices --activation_arena --pinned_io --lean_reads --read_ahead --fast_load --decode_once --lean_foundations --read_workers 12"
cat /root/tos/04_5h/*.exr > /dev/null
for v in cpu gpu cpu2 gpu2; do
  extra=""; case $v in gpu*) extra="--gpu_prep";; esac
  python perf/bench.py --input /root/tos/04_5h --frame_dir_linear --frame_dir_fps 24 --num_frames 32 --hann_chunk 4 --hann_stride 2 \
    --low_vram $OPT $extra --warmup 0 --repeats 1 --temp_dir $OUT/tmp --label "prep $v" --out $OUT/prep_$v.json > $OUT/logs/prep_$v.log 2>&1
  python - "$OUT/prep_$v.json" "$v" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
p = r["summary"]["phases"]
print(f"[prep {sys.argv[2]}] total {sum(x['wall_s'] for x in p.values()):.2f} s, prepass {p['prepass']['wall_s']:.2f} s, "
      f"prepass peak {p['prepass']['peak_reserved_gib']:.2f} GiB, run peak {max(x['peak_reserved_gib'] for x in p.values()):.2f} GiB")
PY
done
log "tier3m done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
