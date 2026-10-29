import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import glob, os, sys
import cv2
import numpy as np
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
scale = np.float32(1.0 + 2.0 ** -20)
odd = 0
for p in sorted(glob.glob(os.path.join(src, "*.exr")))[:n]:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * scale
    odd += int((img != img.astype(np.float16).astype(np.float32)).sum())
    cv2.imwrite(os.path.join(dst, os.path.basename(p)), img, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
back = cv2.imread(sorted(glob.glob(os.path.join(dst, "*.exr")))[0], cv2.IMREAD_UNCHANGED)
print(f"{n} frames, {odd} values outside half precision, reread dtype {back.dtype}, "
      f"still outside half: {int((back != back.astype(np.float16).astype(np.float32)).sum())}")
