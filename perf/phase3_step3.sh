#!/usr/bin/env bash
# Phase 3, step 3 (RTX 3060, laptop emulation).
#   exact      --cache_region_indices on top of the exact flags, vs p3_base
#   goldens    all 8 clips at 1080p 8/4 through the unmodified runtime
#              (mem-efficient attention, as on the Windows laptop)
#   floor      same with attention forced to cuDNN: this GPU's noise floor
#   candidate  exact flags + --channels_last_depthwise (near-exact), compared
#              against the goldens and the floor, with visual diffs
#   timing     1080p 8/4 with the exact flags, then + channels-last depthwise
set -u
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  if [ -x venv/Scripts/python.exe ]; then PY=venv/Scripts/python.exe; else PY=python; fi
fi
R=perf/results/phase3
mkdir -p "$R"
LAPTOP="--sdpa efficient --mem_cap_gib 5.0"
EXACT="--bf16_weights --lean_activations --prefetch_frames --cache_region_indices"
CLIPS="hair motionblur smoke glasses sheen cup converge cyc"

rm -rf perf/out/p3_exact4
"$PY" perf/goldens.py --tag p3_exact4 --sdpa efficient --clips converge --res 1080 720 \
  "--extra=$EXACT"
"$PY" perf/compare.py --ref perf/out/p3_base --cand perf/out/p3_exact4 \
  --out "$R/cmp_exact4.json" > "$R/cmp_exact4.txt"
echo "step3 exact done"

# Rebuild all 8 test clips from the HCW plates (byte-identical to the local ones).
"$PY" - <<'EOF'
import os, urllib.request, zipfile
plates = {
    "hair_detail": "keying/hcw_locked_backdrop.zip",
    "motion_blur": "keying/hcw_with_motion_blur.zip",
    "smoke": "keying/hcw_smoke.zip",
    "sheen": "keying/hcw_light_blue_sheen.zip",
    "disney_glasses": "keying/hcw_disney_glasses.zip",
    "drink_from_cup": "gsinteraction/hcw_drink_from_cup.zip",
    "converge": "gsinteraction/hcw_converge.zip",
    "walking_sitting": "gsinteraction/hcw_walking_sitting.zip",
}
for name, rel in plates.items():
    d = f"/root/hcw/{name}/frames"
    if os.path.isdir(d) and len(os.listdir(d)) > 40:
        continue
    os.makedirs(d, exist_ok=True)
    z = f"/root/hcw/{name}/plate.zip"
    urllib.request.urlretrieve(f"https://cdn.hollywoodcamerawork.com/downloads/plates/{rel}", z)
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            base = os.path.basename(m.filename)
            if base.lower().endswith(".png"):
                with zf.open(m) as src, open(os.path.join(d, base), "wb") as dst:
                    dst.write(src.read())
    os.remove(z)
    print(name, len(os.listdir(d)), "frames")
EOF
"$PY" perf/prepare_clips.py --src /root/hcw --clips $CLIPS --res 1080

rm -rf perf/out/g3060 perf/out/nf3060_cudnn perf/out/cand_cl
"$PY" perf/goldens.py --tag g3060 --sdpa efficient --clips $CLIPS --res 1080
"$PY" perf/goldens.py --tag nf3060_cudnn --sdpa cudnn --clips $CLIPS --res 1080
"$PY" perf/goldens.py --tag cand_cl --sdpa efficient --clips $CLIPS --res 1080 \
  "--extra=$EXACT --channels_last_depthwise"
"$PY" perf/compare.py --ref perf/out/g3060 --cand perf/out/nf3060_cudnn \
  --out "$R/cmp_floor3060_cudnn.json" > "$R/cmp_floor3060_cudnn.txt"
"$PY" perf/compare.py --ref perf/out/g3060 --cand perf/out/cand_cl \
  --out "$R/cmp_cand_cl.json" > "$R/cmp_cand_cl.txt"
for clip in hair_1080 smoke_1080 glasses_1080 cyc_1080; do
  "$PY" perf/visual_diff.py --ref perf/out/g3060 --cand perf/out/cand_cl \
    --cmp "$R/cmp_cand_cl.json" --clip "$clip" --worst 1 --out "$R/vdiff_cand_cl" --gif
done
echo "step3 quality done"

"$PY" perf/bench.py $LAPTOP $EXACT --input test_clips/converge_1080 --hann_chunk 8 \
  --hann_stride 4 --low_vram --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 exact flags" --out "$R/bench_1080_8x4_exact4.json"
"$PY" perf/bench.py $LAPTOP $EXACT --channels_last_depthwise --input test_clips/converge_1080 \
  --hann_chunk 8 --hann_stride 4 --low_vram --num_frames 48 --repeats 3 --warmup 1 \
  --label "1080p 8/4 exact + channels-last dw" --out "$R/bench_1080_8x4_cl.json"
echo "step3 done"
