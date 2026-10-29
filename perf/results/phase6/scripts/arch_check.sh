#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 < /dev/null &
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
mkdir -p /root/ck && tar xzf /root/code.tgz -C /root/ck && cd /root/ck
find /root/ck -name '*.sh' -exec sed -i 's/\r$//' {} +
PLATES="hair_detail" CLIPS="hair" bash perf/setup_vast.sh > /root/setup.log 2>&1
pip uninstall -q -y opencv-python 2>/dev/null; pip install -q opencv-python-headless==4.11.0.86
log "setup: $(tail -1 /root/setup.log)"
python -c "import torch; p = torch.cuda.get_device_properties(0); print('[gpu]', p.name, f'sm_{p.major}{p.minor}', round(p.total_memory / 2**30, 1), 'GiB, torch', torch.__version__, 'cuda', torch.version.cuda, 'flash attention', torch.backends.cuda.is_flash_attention_available())"
src=/root/hcw/hair_detail/frames; n=$(ls $src | wc -l); start=$(( n / 2 - 12 ))
mkdir -p /root/clips/hair_detail
ls $src | sort | sed -n "$((start + 1)),$((start + 24))p" | while read f; do ln -sf $src/$f /root/clips/hair_detail/$f; done
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
D=/root/arch; mkdir -p $D/logs
python perf/gpu_prep_probe.py > $D/probe.txt 2>&1; log "probe rc=$?: $(grep -c ' 0 mismatches\|mismatches          0  model mismatches          0' $D/probe.txt) clean lines"
A="--input /root/clips/hair_detail --output_format exr --exr_dtype float --outputs alpha fg --hann_chunk 8 --hann_stride 4 --low_vram --frame_dir_fps 24"
python /root/det_run.py $A --output_dir $D/det_orig --temp_dir $D/tmp > $D/logs/det_orig.log 2>&1; log "deterministic original rc=$?"
python /root/det_run.py $A --output_dir $D/det_orig2 --temp_dir $D/tmp > $D/logs/det_orig2.log 2>&1; log "deterministic original 2 rc=$?"
python /root/det_run.py $A --output_dir $D/det_opt --temp_dir $D/tmp --optimized > $D/logs/det_opt.log 2>&1; log "deterministic optimized rc=$?"
log "deterministic cuDNN: original vs original 2: $(python perf/same_files.py $D/det_orig $D/det_orig2 | head -1)"
log "deterministic cuDNN: original vs optimized: $(python perf/same_files.py $D/det_orig $D/det_opt | head -1)"
python infer.py $A --output_dir $D/orig --temp_dir $D/tmp > $D/logs/orig.log 2>&1; log "default original rc=$?"
python infer.py $A --output_dir $D/opt --temp_dir $D/tmp --optimized > $D/logs/opt.log 2>&1; log "default optimized rc=$?"
log "default cuDNN: original vs optimized: $(python perf/same_files.py $D/orig $D/opt | head -1)"
rm -rf $D/det_* $D/orig $D/opt $D/tmp
for v in orig opt orig2 opt2; do
  x=""; case $v in opt*) x="--optimized";; esac
  t0=$(date +%s.%N)
  python infer.py --input /root/clips/hair_detail --output_dir $D/t_$v --temp_dir $D/tt_$v --frame_dir_fps 24 $x > $D/logs/t_$v.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  log "timing $v rc=$rc wall $(python -c "print(round($t1 - $t0, 1))") s | $(grep -o 'hann_chunk=[0-9]*, hann_stride=[0-9]*, low_vram=[A-Za-z]*' $D/logs/t_$v.log | head -1)"
  rm -rf $D/t_$v $D/tt_$v
done
cd /root && tar czf /root/arch.tgz arch/probe.txt arch/logs 2>/dev/null
log "arch done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1200 > /dev/null 2>&1 < /dev/null &
