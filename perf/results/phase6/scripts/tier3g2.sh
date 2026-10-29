#!/usr/bin/env bash
set -u
exec 9> /root/tier3g2.lock
flock -n 9 || { echo "tier3g2 already running"; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier3k done" /root/tier3k.log 2>/dev/null; do sleep 20; done
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 < /dev/null 9>&- &
rm -rf /root/dev2; cp -a /root/dev /root/dev2
cp /root/upload_g/runtime.py /root/upload_g/video_io.py /root/dev2/corridorkey/
cp /root/upload_g/gpu_prep_probe.py /root/dev2/perf/
sed -i "s/\r$//" /root/dev2/corridorkey/*.py /root/dev2/perf/*.py
cd /root/dev2
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier3g2; mkdir -p $OUT/logs
python perf/gpu_prep_probe.py > $OUT/probe.txt 2>&1; log "probe rc=$?"; sed 's/^/    /' $OUT/probe.txt
python /root/make_f32_clip.py /root/tos/04_5h /root/tos_f32 16 > $OUT/f32.txt 2>&1; log "f32 clip: $(cat $OUT/f32.txt | tail -1)"
for clip in tos_f32 04_5h; do
  case $clip in tos_f32) inp=/root/tos_f32;; *) inp=/root/tos/04_5h;; esac
  A="--input $inp --frame_dir_linear --num_frames 16 --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24 --no_carry_hint --outputs alpha fg --output_format exr --exr_dtype float"
  for v in default new; do
    x=""; [ $v = new ] && x="--optimized --gpu_prep --async_baton"
    t0=$(date +%s)
    python infer.py $A --output_dir $OUT/${clip}_$v --temp_dir $OUT/tmp $x > $OUT/logs/${clip}_$v.log 2>&1
    log "$clip $v rc=$? in $(( $(date +%s) - t0 )) s"
  done
  log "$clip default vs --optimized --gpu_prep --async_baton: $(python perf/same_files.py $OUT/${clip}_default $OUT/${clip}_new | head -1)"
  rm -rf $OUT/${clip}_* $OUT/tmp
done
log "tier3g2 done"
pkill -f '[s]elfstop.sh'; setsid nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null 9>&- &
