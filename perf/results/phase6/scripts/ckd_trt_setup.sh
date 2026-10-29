#!/usr/bin/env bash
# TensorRT for CorridorKey Desktop's engine, as its installer sets it up: the
# tensorrt_cu13_libs 10.16.1.11 wheel from pypi.nvidia.com, the
# models-2.0.0-onnx.4-trt release merged into the 2.0.0-onnx.4 models folder,
# and the bundled ck-trt-build builder.
set -u
D=/root/ckd
cd $D
pip download -q --no-deps --only-binary=:all: --extra-index-url https://pypi.nvidia.com \
  -d $D/wheels "tensorrt_cu13_libs==10.16.1.11" 2>&1 | tail -2
f=$(ls $D/wheels/tensorrt_cu13_libs-10.16.1.11-*.whl | head -1)
[ -n "$f" ] || { echo "no tensorrt wheel"; exit 1; }
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$f" rt
mkdir -p $D/trt_rel && cd $D/trt_rel
curl -s https://api.github.com/repos/MarcelineVPQ/corridorkey-desktop/releases/tags/models-2.0.0-onnx.4-trt \
  | python -c "import json,sys; [print(a['browser_download_url']) for a in json.load(sys.stdin).get('assets', [])]" > urls.txt
[ -s urls.txt ] || { echo "no trt release assets"; exit 1; }
xargs -P 6 -n 1 curl -sSLO < urls.txt
M=$D/models/2.0.0-onnx.4
cp -n models.json $M/models_trt.json
for x in *; do case $x in models.json|urls.txt) ;; *) cp -n "$x" $M/ ;; esac; done
python - $M/models.json $M/models_trt.json <<'EOF'
import json, sys
base, trt = (json.load(open(p)) for p in sys.argv[1:3])
def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a[k], v) if isinstance(a.get(k), dict) and isinstance(v, dict) else a.get(k, v)
    return a
merged = merge(base, trt)
json.dump(merged, open(sys.argv[1], "w"), indent=2)
keys = list(merged.get("models", merged).keys()) if isinstance(merged.get("models", merged), dict) else []
print("models.json keys:", keys)
EOF
cat >> /root/ckd_env.sh <<EOS
export CK_ENGINE_TRT_BUILD=$D/squashfs-root/resources/engine/ck-trt-build
export CKD_LD=$(find $D/rt -name "*.so*" -printf "%h\n" | sort -u | tr "\n" ":")
EOS
ls -la $D/squashfs-root/resources/engine/ | head
echo "ckd trt ready: $(ls $M | grep -ci trt) trt files"
