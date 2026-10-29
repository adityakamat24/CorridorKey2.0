#!/usr/bin/env bash
# CorridorKey Desktop 1.0.0-dev.1's engine, runtime and models, as its setup installs them.
set -u
D=/root/ckd; mkdir -p $D/wheels $D/rt $D/models/2.0.0-onnx.4 && cd $D
REL=https://github.com/MarcelineVPQ/corridorkey-desktop/releases/download
curl -sSL -o app.AppImage $REL/v1.0.0-dev.1/CorridorKey-Desktop-1.0.0-dev.1.AppImage
echo "61ed7d4a6fd0d6aa8d43a9452d754006a451a4b3d873252790c6a606cc27eb58  app.AppImage" | sha256sum -c - || exit 1
chmod +x app.AppImage && ./app.AppImage --appimage-extract > /dev/null 2>&1
PYPI=https://files.pythonhosted.org/packages
cat > wheels.txt <<'EOF2'
b52998a075de27ff30e46a125001822541fa5b03f5d31398ab0e50789d80c51c 8f/48/6fbfdbce25e5756c7b9ff8f40645d98b166b8022cf00ede58a2bb4571b8e/onnxruntime_gpu-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl
c11a27fd4379510e5b1f84b367a2514d1e52fe5cc13442117a0e0a1addee3cf2 eb/1b/adc70acf3fde509b1f0ca29d1c0e8d9b00aaecd871f8cf877257c8b68d9c/nvidia_cublas-13.8.1.7-py3-none-manylinux_2_27_x86_64.whl
5ce8c97b00b232c4f50c8c4b5a3b68cafee08bdb82ea86f2052ff01d03194f4a 56/9c/1342ebb460ce2afd014ec5a002adc99108e3c53a6b70f399cf64dd1f267d/nvidia_cuda_nvrtc-13.4.92-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
9641f797da20ce1dd8e779b6e96d08cf9ba564cec8e8225458811ee26423f3a5 98/8a/3431271f6344874b8f1ac03f16b3d679c91493f8da63f716160403e6d0a0/nvidia_cuda_runtime-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
9677e76f21862eb5da7ee5ed69d544738b2d8b5c3ce7e5ec125c5592e6cdbdc8 af/75/96ea5c5368eb595c39d629cde08a66227a864e71ccb0e593add2612bb952/nvidia_cudnn_cu13-9.27.0.42-py3-none-manylinux_2_27_x86_64.whl
0e8385013596b112d29c9ce8c63dc575b308d77636c7169104e18714f03961a8 76/bf/3fea3d1c6262bded26ae00e3106432d63235954965d7785f5051ac146651/nvidia_cufft-12.4.0.43-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
25c3457ae7a224fdd484dab90b0fc5dc0e842fab5db3012afa4a5bd2af4eb7e5 07/73/3ee8e5b4cb891401e603ffd3a59b35c6afe785fd2de123afe7c7029603dc/nvidia_curand-10.4.4.72-py3-none-manylinux_2_27_x86_64.whl
e0391f24ed94ec879b84e3da4d4ec320c879aff681f2c7a638462f7199284323 1d/6b/eef7a9e32872b8f41e145bf10cddc9af26e153c338852811fe9a9baddf9e/nvidia_nvjitlink-13.4.92-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
EOF2
while read -r sha path; do
  f=wheels/$(basename "$path"); curl -sSL -o "$f" "$PYPI/$path"
  echo "$sha  $f" | sha256sum -c --quiet - || { echo "bad wheel $f"; exit 1; }
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$f" rt
done < wheels.txt
cd models/2.0.0-onnx.4
pids=""
for f in manifest.json models.json ck2-cradio-fp16.onnx ck2-cradio-fp16.onnx.data ck2-moge-fp16.onnx ck2-moge-fp16.onnx.data ck2-rvm-fp16.onnx ck2-rvm-fp16.onnx.data ck2-stage1-fp16.onnx ck2-stage1-fp16.onnx.data ck2-stage2-fp16.onnx ck2-stage2-fp16.onnx.data; do
  curl -sSL -o $f $REL/models-2.0.0-onnx.4/$f & pids="$pids $!"
done; wait $pids; cd $D
cat > /root/ckd_env.sh <<EOS
export CKD_ENGINE=$D/squashfs-root/resources/engine/corridorkey-engine
export CKD_MODELS=$D/models/2.0.0-onnx.4
export CKD_ORT=$(find $D/rt -name "libonnxruntime.so*" | sort | head -1)
export CKD_LD=$(find $D/rt -name "*.so*" -printf "%h
" | sort -u | tr "
" ":")
EOS
echo "ckd ready: $(du -sh $D/models | cut -f1) models"
