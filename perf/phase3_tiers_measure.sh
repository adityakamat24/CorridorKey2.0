#!/usr/bin/env bash
# 48 GB card (RTX 6000 Ada; the A6000 host failed to boot): measure the window tables for
# every resolution the auto selector has to cover.
#   check      shipped path, low-VRAM, a few points: must reproduce the
#              runtime's own _VRAM_PEAK_GB (validates the method)
#   optimized  bf16 weights + lean activations, low-VRAM and normal mode,
#              720p..4K DCI, windows 1..80 (stops at the first OOM)
# Also builds upscaled converge clips (1080p..4K) for the end-to-end tier tests.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 OPENCV_FOR_THREADS_NUM=8
R=perf/results/phase3
mkdir -p "$R" perf/logs
# Hosts with slow PyPI/GitHub access: set PIP_INDEX_URL to a mirror and copy
# the utils3d tarball (GitHub archive of the pinned commit) to /root first.
if [ -f /root/utils3d.tar.gz ]; then
  grep -v utils3d requirements.txt > /tmp/req.txt
  pip install -q /root/utils3d.tar.gz -r /tmp/req.txt nvidia-ml-py
else
  pip install -q -r requirements.txt nvidia-ml-py
fi
HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/download_weights.py

python - <<'EOF'
import os, urllib.request, zipfile
d = "/root/hcw/converge/frames"
os.makedirs(d, exist_ok=True)
z = "/root/hcw/converge/plate.zip"
urllib.request.urlretrieve(
    "https://cdn.hollywoodcamerawork.com/downloads/plates/gsinteraction/hcw_converge.zip", z)
with zipfile.ZipFile(z) as zf:
    for m in zf.infolist():
        b = os.path.basename(m.filename)
        if b.lower().endswith(".png"):
            with zf.open(m) as s, open(os.path.join(d, b), "wb") as o:
                o.write(s.read())
os.remove(z)
EOF
python perf/prepare_clips.py --src /root/hcw --clips converge --res 1080 1440 3k 2160
python perf/prepare_clips.py --src /root/hcw --clips converge200 --res 1080
echo "tiers setup done"

# Peak allocated does not depend on the allocator; expandable segments only
# let the largest windows get closer to 48 GB before fragmentation stops them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python perf/vram_table.py --modes lowvram --sizes 720p 1080p 4k --windows 1 8 24 48 \
  --out "$R/vram_table_shipped_check_ada.json"
python perf/vram_table.py --optimized --modes lowvram normal \
  --out "$R/vram_table_optimized_ada.json"
echo "tiers measure done"
