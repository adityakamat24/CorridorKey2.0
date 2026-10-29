import glob
import json
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

out_root, clips_root, tos_dir = sys.argv[1], sys.argv[2], sys.argv[3]
dst = os.path.join(out_root, "stills")
os.makedirs(dst, exist_ok=True)


def srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def lin(x):
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def despill(fg):
    r, g, b = fg[..., 0], fg[..., 1], fg[..., 2]
    spill = np.clip(g - (r + b) * 0.5, 0.0, None)
    full = np.stack([r + spill * 0.5, g - spill, b + spill * 0.5], -1)
    return fg * 0.5 + full * 0.5


def read_exr_rgb(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if im.ndim == 2:
        return im
    return cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB)


def save(name, rgb01):
    im = (np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(dst, name), im)


def fit(im, width):
    h, w = im.shape[:2]
    return cv2.resize(im, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


stats = {}
for clip, src in (("hair_detail", os.path.join(clips_root, "hair_detail")),
                  ("smoke", os.path.join(clips_root, "smoke")),
                  ("motion_blur", os.path.join(clips_root, "motion_blur")),
                  ("tos", tos_dir)):
    name = os.path.basename(src.rstrip("/"))
    a_orig = sorted(glob.glob(os.path.join(out_root, f"{clip}_orig", f"{name}_alpha_exr", "*.exr")))
    a_opt = sorted(glob.glob(os.path.join(out_root, f"{clip}_opt", f"{name}_alpha_exr", "*.exr")))
    f_opt = sorted(glob.glob(os.path.join(out_root, f"{clip}_opt", f"{name}_fg_exr", "*.exr")))
    if not a_orig or len(a_orig) != len(a_opt):
        print(clip, "missing outputs", len(a_orig), len(a_opt))
        continue
    differ = sum(int((cv2.imread(x, cv2.IMREAD_UNCHANGED) != cv2.imread(y, cv2.IMREAD_UNCHANGED)).sum())
                 for x, y in zip(a_orig, a_opt))
    t = len(a_opt) // 2
    plates = sorted(os.listdir(src))
    plate_path = os.path.join(src, plates[t])
    if plate_path.lower().endswith(".exr"):
        plate = srgb(read_exr_rgb(plate_path))
    else:
        plate = cv2.cvtColor(cv2.imread(plate_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = read_exr_rgb(a_opt[t])
    alpha = alpha if alpha.ndim == 2 else alpha[..., 0]
    fg = read_exr_rgb(f_opt[t])
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    top, bottom = np.array([0.10, 0.13, 0.20]), np.array([0.62, 0.34, 0.30])
    bg_srgb = top + (bottom - top) * ((yy / h) * 0.7 + (xx / w) * 0.3)[..., None]
    comp = srgb(despill(fg) * alpha[..., None] + lin(bg_srgb) * (1 - alpha[..., None]))
    plate = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA) if plate.shape[:2] != (h, w) else plate
    edge = ((alpha > 0.02) & (alpha < 0.98)).astype(np.float32)
    side = max(160, min(h, w) // 4)
    energy = cv2.boxFilter(edge, -1, (side, side), normalize=False)
    cy, cx = np.unravel_index(int(np.argmax(energy)), energy.shape)
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    crop = (slice(y0, y0 + side), slice(x0, x0 + side))
    width = 960
    save(f"{clip}_plate.png", fit(plate, width))
    save(f"{clip}_alpha.png", fit(alpha, width))
    save(f"{clip}_comp.png", fit(comp, width))
    for tag, im in (("plate", plate), ("alpha", alpha), ("comp", comp)):
        save(f"{clip}_zoom_{tag}.png", cv2.resize(im[crop], (480, 480), interpolation=cv2.INTER_NEAREST))
    stats[clip] = {"frames": len(a_opt), "alpha_values_differing": differ, "pixels_per_frame": h * w,
                   "still_frame": t, "crop_yx": [y0, x0, side], "size": [w, h]}
    print(clip, stats[clip], flush=True)
json.dump(stats, open(os.path.join(dst, "stats.json"), "w"), indent=1)
