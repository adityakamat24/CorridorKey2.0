#!/usr/bin/env bash
# After tier2b: lossless encodes (CorridorKey Desktop's default is x264 qp 0),
# x264 and NVENC, on the same 120 4K frames, with PSNR against the input.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2b done" /root/tier2b.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 5400 > /dev/null 2>&1 &
OUT=/root/tier2b
RAW=/root/tos_4k.bgr48
ffmpeg -hide_banner -loglevel error -pattern_type glob -i '/root/tos/04_5h/*.exr' -pix_fmt bgr48le -f rawvideo $RAW 2>>$OUT/logs/raw.log
IN="-f rawvideo -pix_fmt bgr48le -s:v 4096x2160 -r 24 -i $RAW"
enc() {
  local tag=$1; shift
  local t0=$(date +%s.%N)
  ffmpeg -y -hide_banner -loglevel error -nostats $IN -an "$@" /root/enc_$tag.mp4 2>$OUT/logs/enc_$tag.log
  local rc=$? t1=$(date +%s.%N)
  local psnr=$(ffmpeg -hide_banner -nostats -i /root/enc_$tag.mp4 $IN -lavfi "[0:v]format=gbrp16le[a];[1:v]format=gbrp16le[b];[a][b]psnr" -f null - 2>&1 | grep -o 'average:[0-9.inf]*' | tail -1)
  local size=$(stat -c %s /root/enc_$tag.mp4 2>/dev/null || echo 0)
  echo "{\"tag\": \"$tag\", \"rc\": $rc, \"encode_s\": $(python -c "print(round($t1 - $t0, 1))"), \"fps\": $(python -c "print(round(120 / ($t1 - $t0), 2))"), \"psnr\": \"$psnr\", \"mb\": $((size / 1000000))}" >> $OUT/encode.jsonl
  log "$tag rc=$rc $(tail -1 $OUT/encode.jsonl)"
  rm -f /root/enc_$tag.mp4
}
enc lossless_qp0_t2 -c:v libx264 -pix_fmt yuv444p10le -qp 0 -preset medium -threads 2
enc lossless_qp0_t4 -c:v libx264 -pix_fmt yuv444p10le -qp 0 -preset medium -threads 4
enc nvenc_hevc444_lossless -c:v hevc_nvenc -preset p5 -tune lossless -profile:v rext -pix_fmt yuv444p16le
enc nvenc_h264_444_lossless -c:v h264_nvenc -preset p5 -tune lossless -profile:v high444p -pix_fmt yuv444p16le
rm -f $RAW
cd $OUT && tar czf /root/tier2b.tgz *.json *.jsonl logs/*.log 2>/dev/null
log "tier2c done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
