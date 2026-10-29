#!/usr/bin/env python3
"""Before/after showcase: original plate, shipped key, optimized key.

For each clip, composites the predicted foreground over a neutral gradient
background (linear light, alpha from the model) for two renders, and writes:
  <out>/<clip>_still.jpg   original | before | after, one frame, full width
  <out>/<clip>.mp4         the same three panels for every frame (H.264)
  <out>/metrics.json       per clip: frames, mean frame-to-frame alpha change
                           in the edge band for both renders, and their
                           difference to each other

Usage:
    python perf/make_showcase.py --clips hair_1080 smoke_1080 \
        --before perf/out/g3060 --after perf/out/opt24 \
        --before_label "Shipped: 8-frame window" \
        --after_label "Optimized: 24-frame window" --out perf/results/showcase
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def srgb_encode(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(x):
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def background(h: int, w: int) -> np.ndarray:
    """Soft studio-grey vertical gradient, linear light."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array([0.40, 0.42, 0.46], np.float32)
    bottom = np.array([0.16, 0.17, 0.19], np.float32)
    return srgb_decode(np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy())


def read_exr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def comp(render_dir: Path, clip: str, t: int, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = read_exr(render_dir / clip / f"{clip}_alpha_exr" / f"alpha_{t:06d}.exr")
    fg = read_exr(render_dir / clip / f"{clip}_fg_exr" / f"fg_{t:06d}.exr")
    a3 = np.clip(a, 0.0, 1.0)[..., None]
    return srgb_encode(np.maximum(fg, 0.0) * a3 + bg * (1.0 - a3)), a


def label(img_u8: np.ndarray, text: str) -> np.ndarray:
    out = img_u8.copy()
    scale = max(0.6, out.shape[1] / 900.0)
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, max(1, int(1.6 * scale))
    (tw, th), base = cv2.getTextSize(text, font, size, thick)
    pad = int(8 * scale)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, pad + th), font, size, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def edge_flicker(alphas: list[np.ndarray]) -> float:
    vals = []
    for a0, a1 in zip(alphas, alphas[1:]):
        band = (a0 > 0.02) & (a0 < 0.98)
        if band.any():
            vals.append(float(np.abs(a1 - a0)[band].mean()))
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--before_label", default="Before")
    ap.add_argument("--after_label", default="After")
    ap.add_argument("--panel_w", type=int, default=640)
    ap.add_argument("--still_t", type=int, default=24)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for clip in args.clips:
        src_dir = ROOT / "test_clips" / clip
        frames = sorted(src_dir.glob("*.png"))
        h0, w0 = cv2.imread(str(frames[0])).shape[:2]
        bg = background(h0, w0)
        pw = args.panel_w
        ph = int(round(h0 * pw / w0)) // 2 * 2
        ff = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{3 * pw}x{ph}", "-r", str(args.fps), "-i", "pipe:0",
             "-c:v", "libx264", "-crf", "20", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / f"{clip}.mp4")],
            stdin=subprocess.PIPE)
        a_before, a_after, cross = [], [], []
        for t, fpath in enumerate(frames):
            orig = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
            cb, ab = comp(Path(args.before), clip, t, bg)
            ca, aa = comp(Path(args.after), clip, t, bg)
            a_before.append(ab)
            a_after.append(aa)
            cross.append(float(np.abs(aa - ab).mean()))
            full = [orig, (cb * 255 + 0.5).astype(np.uint8), (ca * 255 + 0.5).astype(np.uint8)]
            if t == min(args.still_t, len(frames) - 1):
                still = np.concatenate([label(full[0], "Original"),
                                        label(full[1], args.before_label),
                                        label(full[2], args.after_label)], axis=1)
                cv2.imwrite(str(out / f"{clip}_still.jpg"),
                            cv2.cvtColor(still, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            small = [cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA) for p in full]
            small = [label(small[0], "Original"), label(small[1], args.before_label),
                     label(small[2], args.after_label)]
            ff.stdin.write(np.concatenate(small, axis=1).tobytes())
        ff.stdin.close()
        ff.wait()
        metrics[clip] = {
            "frames": len(frames),
            "edge_flicker_before": edge_flicker(a_before),
            "edge_flicker_after": edge_flicker(a_after),
            "alpha_mae_before_vs_after": float(np.mean(cross)),
        }
        print(clip, metrics[clip], flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
