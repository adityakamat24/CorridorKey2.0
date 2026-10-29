#!/usr/bin/env bash
# Follow-up on the tier-2 box, after tier2.sh: where the 4K seconds go.
# 1. perf/bench.py phase breakdown at 4K with the --optimized set, 8/4 (ours)
#    and 4/2 (CorridorKey Desktop's automatic window).
# 2. Encoder throughput and quality at 4K: our writer's exact x264 command
#    (raw bgr48le in, yuv444p10le CRF 12) by preset and thread count, and
#    NVENC HEVC 4:4:4, each with PSNR against the input.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "tier2 done" /root/tier2.log 2>/dev/null; do sleep 30; done
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 7200 > /dev/null 2>&1 &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
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

# 2. Encoder: the writer's input is raw bgr48le; make it once from the plate.
RAW=/root/tos_4k.bgr48
ffmpeg -hide_banner -loglevel error -pattern_type glob -i '/root/tos/04_5h/*.exr' \
  -vf "zscale=transfer=iec61966-2-1:transferin=linear,format=bgr48le" -f rawvideo $RAW 2>$OUT/logs/raw.log \
  || ffmpeg -hide_banner -loglevel error -pattern_type glob -i '/root/tos/04_5h/*.exr' \
       -pix_fmt bgr48le -f rawvideo $RAW 2>>$OUT/logs/raw.log
log "raw frames: $(( $(stat -c %s $RAW) / (4096 * 2160 * 6) ))"
IN="-f rawvideo -pix_fmt bgr48le -s:v 4096x2160 -r 24 -i $RAW"
enc() { # tag codec args...
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
X264="-c:v libx264 -pix_fmt yuv444p10le -crf 12"
for th in 2 4 8; do enc medium_t$th $X264 -preset medium -threads $th; done
for p in fast faster veryfast; do for th in 2 4; do enc ${p}_t$th $X264 -preset $p -threads $th; done; done
enc nvenc_hevc444_qp12 -c:v hevc_nvenc -preset p5 -tune hq -profile:v rext -pix_fmt yuv444p16le -rc constqp -qp 12
enc nvenc_hevc444_qp8 -c:v hevc_nvenc -preset p5 -tune hq -profile:v rext -pix_fmt yuv444p16le -rc constqp -qp 8
enc nvenc_hevc444_p1_qp12 -c:v hevc_nvenc -preset p1 -profile:v rext -pix_fmt yuv444p16le -rc constqp -qp 12
rm -f $RAW
cd $OUT && tar czf /root/tier2b.tgz *.json *.jsonl logs/*.log 2>/dev/null
log "tier2b done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 1800 > /dev/null 2>&1 &
