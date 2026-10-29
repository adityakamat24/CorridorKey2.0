#!/usr/bin/env bash
# After tier2c: x264's lookahead and the encoder tail, end to end. Same keyed
# frames in every run (--optimized), so each encode is scored by PSNR against
# the lossless (qp=0) encode of the same run.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "redo done" /root/tier2_redo.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
OUT=/root/tier2d; mkdir -p $OUT/logs
RES=$OUT/tier2d.jsonl
run() { # tag input args...
  local tag=$1 inp=$2; shift 2
  rm -rf $OUT/out_$tag $OUT/tmp_$tag
  local t0=$(date +%s.%N)
  python perf/infer_with.py --mem_budget_gib 12 -- --input "$inp" --output_dir $OUT/out_$tag \
    --temp_dir $OUT/tmp_$tag --frame_dir_fps 24 --optimized "$@" > $OUT/logs/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"wall_s\": $(python -c "print(round($t1 - $t0, 1))"), \"mb\": $(( $(du -sb $OUT/out_$tag | cut -f1) / 1000000 ))}" >> $RES
  log "$tag rc=$rc $(tail -1 $RES)"
  rm -rf $OUT/tmp_$tag
}
psnr() { # a b
  ffmpeg -hide_banner -nostats -i "$1" -i "$2" -lavfi "[0:v]format=gbrp16le[a];[1:v]format=gbrp16le[b];[a][b]psnr" -f null - 2>&1 | grep -o 'average:[0-9.inf]*' | tail -1
}
cat /root/hcw/*/frames/* /root/tos/04_5h/*.exr > /dev/null
for item in "hair_detail /root/hcw/hair_detail/frames" "motion_blur /root/hcw/motion_blur/frames" "tos /root/tos/04_5h --frame_dir_linear --hann_chunk 8 --hann_stride 4 --low_vram"; do
  set -- $item; clip=$1; shift
  run ${clip}_lossless "$@" --x264_params qp=0
  run ${clip}_default "$@"
  run ${clip}_la10 "$@" --x264_params rc-lookahead=10
  run ${clip}_la0 "$@" --x264_params rc-lookahead=0
  run ${clip}_default2 "$@"
  for v in default la10 la0; do
    for o in alpha fg; do
      f=$(ls $OUT/out_${clip}_$v/*_$o.mp4 2>/dev/null | head -1); r=$(ls $OUT/out_${clip}_lossless/*_$o.mp4 2>/dev/null | head -1)
      [ -n "$f" ] && [ -n "$r" ] && log "psnr ${clip}_$v $o vs lossless: $(psnr $f $r)"
    done
  done
  rm -rf $OUT/out_${clip}_*
done
cd $OUT && tar czf /root/tier2d.tgz *.jsonl logs/*.log 2>/dev/null
log "tier2d done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
