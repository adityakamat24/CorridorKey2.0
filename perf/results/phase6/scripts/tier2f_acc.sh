#!/usr/bin/env bash
# After tier2e on the tier-2 box: alphastrata/CorridorKey2.0's 1080p goldens against the fp32
# reference here (the competitor timings run on a second box, tier3_race.sh).
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2e done" /root/tier2e.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 3600 > /dev/null 2>&1 &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier2f; mkdir -p $OUT/logs
git clone -q --depth 1 -b master https://github.com/alphastrata/CorridorKey2.0 /root/jer > $OUT/logs/jer_clone.log 2>&1
for d in weights vfms; do mkdir -p /root/jer/$d; cp -rsn /root/ck/$d/. /root/jer/$d/; done
log "jer: $(cd /root/jer && git log --oneline -1)"
for clip in hair smoke converge; do
  (cd /root/jer && python infer.py --input /root/ck/test_clips/${clip}_1080 --output_dir $OUT/jer_g/${clip}_1080 \
    --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram \
    --frame_dir_fps 30 --temp_dir $OUT/tmp_g) > $OUT/logs/jer_g_$clip.log 2>&1
  python perf/compare.py --ref perf/out/fp32/${clip}_1080 --cand $OUT/jer_g/${clip}_1080 > $OUT/cmp_jer_$clip.txt 2>&1
  log "jer vs fp32: $(tail -n1 $OUT/cmp_jer_$clip.txt)"
done
rm -rf $OUT/jer_g $OUT/tmp_g
cd $OUT && tar czf /root/tier2f.tgz *.txt logs/*.log 2>/dev/null
log "tier2f done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
