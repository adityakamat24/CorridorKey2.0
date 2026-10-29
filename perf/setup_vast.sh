#!/usr/bin/env bash
# Fresh Vast box (pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel, repo in
# /root/ck): Python deps, weights, HCW plates and the 1080p test clips.
#   PLATES  HCW plate names to fetch (default: the three used for exactness
#           checks plus converge for the 200-frame timing clip)
#   CLIPS   prepare_clips.py clip names to build at 1080p
# Hosts with slow PyPI/GitHub access: set PIP_INDEX_URL to a mirror and copy
# the utils3d tarball (GitHub archive of the pinned commit) to /root first.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENCV_FOR_THREADS_NUM=${OPENCV_FOR_THREADS_NUM:-8}
mkdir -p perf/logs perf/out
if [ -f /root/utils3d.tar.gz ]; then
  grep -v utils3d requirements.txt > /tmp/req.txt
  pip install -q /root/utils3d.tar.gz -r /tmp/req.txt nvidia-ml-py
else
  pip install -q -r requirements.txt nvidia-ml-py
fi
HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/download_weights.py
PLATES=${PLATES:-"hair_detail smoke converge"} python - <<'EOF'
import os, urllib.request, zipfile
urls = {
    "hair_detail": "keying/hcw_locked_backdrop.zip",
    "motion_blur": "keying/hcw_with_motion_blur.zip",
    "smoke": "keying/hcw_smoke.zip",
    "sheen": "keying/hcw_light_blue_sheen.zip",
    "disney_glasses": "keying/hcw_disney_glasses.zip",
    "drink_from_cup": "gsinteraction/hcw_drink_from_cup.zip",
    "converge": "gsinteraction/hcw_converge.zip",
    "walking_sitting": "gsinteraction/hcw_walking_sitting.zip",
}
for name in os.environ["PLATES"].split():
    d = f"/root/hcw/{name}/frames"
    if os.path.isdir(d) and len(os.listdir(d)) > 40:
        continue
    os.makedirs(d, exist_ok=True)
    z = f"/root/hcw/{name}/plate.zip"
    urllib.request.urlretrieve(f"https://cdn.hollywoodcamerawork.com/downloads/plates/{urls[name]}", z)
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            base = os.path.basename(m.filename)
            if base.lower().endswith(".png"):
                with zf.open(m) as src, open(os.path.join(d, base), "wb") as dst:
                    dst.write(src.read())
    os.remove(z)
    print(name, len(os.listdir(d)), "frames", flush=True)
EOF
python perf/prepare_clips.py --src /root/hcw --clips ${CLIPS:-hair smoke converge converge200} --res 1080
echo "setup done"
