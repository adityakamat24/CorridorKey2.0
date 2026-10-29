#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 < /dev/null &
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
mkdir -p /root/ck && tar xzf /root/code.tgz -C /root/ck && cd /root/ck
find /root/ck -name '*.sh' -exec sed -i 's/\r$//' {} +
PLATES="hair_detail smoke motion_blur" CLIPS="hair" bash perf/setup_vast.sh > /root/setup.log 2>&1
pip uninstall -q -y opencv-python 2>/dev/null; pip install -q opencv-python-headless==4.11.0.86
log "setup: $(tail -1 /root/setup.log)"
mkdir -p /root/tos/04_5h && cd /root/tos/04_5h
curl -s https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/ | grep -o 'href="[^"]*\.exr"' | sed 's/href="//; s/"$//' > /root/tos/list.txt
n=$(wc -l < /root/tos/list.txt); sed -n "$(( n / 2 - 5 )),$(( n / 2 + 6 ))p" /root/tos/list.txt > /root/tos/take.txt
xargs -P 6 -I{} curl -sSLO "https://media.xiph.org/mango/tearsofsteel-footage-exr/04_5h/linear/{}" < /root/tos/take.txt
log "tos frames: $(ls /root/tos/04_5h | grep -c exr)"
for p in hair_detail smoke motion_blur; do
  src=/root/hcw/$p/frames; n=$(ls $src | wc -l); start=$(( n / 2 - 12 ))
  mkdir -p /root/clips/$p
  ls $src | sort | sed -n "$((start + 1)),$((start + 24))p" | while read f; do ln -sf $src/$f /root/clips/$p/$f; done
done
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/show; mkdir -p $OUT/logs
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/[gpu] /'
for c in hair_detail smoke motion_blur tos; do
  case $c in tos) inp=/root/tos/04_5h; lin="--frame_dir_linear";; *) inp=/root/clips/$c; lin="";; esac
  A="--input $inp $lin --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24"
  python infer.py $A --output_dir $OUT/${c}_orig --temp_dir $OUT/tmp > $OUT/logs/${c}_orig.log 2>&1; log "$c original rc=$?"
  python infer.py $A --output_dir $OUT/${c}_opt --temp_dir $OUT/tmp --optimized > $OUT/logs/${c}_opt.log 2>&1; log "$c optimized rc=$?"
  log "$c original vs optimized: $(python perf/same_files.py $OUT/${c}_orig $OUT/${c}_opt | head -1)"
done
python /root/stills.py $OUT /root/clips /root/tos/04_5h > $OUT/logs/stills.log 2>&1; log "stills rc=$?"; cat $OUT/logs/stills.log
cd $OUT && tar czf /root/show.tgz stills logs
log "show done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 < /dev/null &
