#!/usr/bin/env python3
"""Build the perf/quality test clips from downloaded source plates.

The source plates are not redistributable, so `test_clips/` is never
committed. This script documents exactly how each clip is derived so the set
can be rebuilt from the original downloads:

  Hollywood Camera Work green screen plates (free to download and use, not
  to re-host): https://www.hollywoodcamerawork.com/green-screen-plates.html
  Each plate zip is extracted to <src>/<plate>/frames/*.png (1280x1080
  DVCPRO HD, 1.5 pixel aspect).

For every clip it writes, per target resolution:
  test_clips/<clip>_<res>/frame_000000.png ...   8-bit sRGB PNG frames
The anamorphic 1280x1080 source is desqueezed to square pixels with Lanczos
(cv2.INTER_LANCZOS4, the same filter the runtime uses), straight to each
target size. Optionally it also writes:
  test_clips/<clip>_<res>.mp4                     x264 yuv444p CRF 10 copy
  test_clips/<clip>_<res>_hint.png                rough first-frame hint matte

The hint is a plain green-difference key of the first frame
(alpha = 1 - clip((g - max(r, b)) * gain, 0, 1)), i.e. a rough automated
hint; run it with --hint_quality 0.75.

Usage:
    python perf/prepare_clips.py --src "D:/corridor key/footage_src/hcw" \
        --out test_clips
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# name: (plate dir, offset into the plate's sorted frames, count, category)
CLIPS = {
    "hair":       ("hair_detail",     0,  48, "fine hair, locked close-up"),
    "motionblur": ("motion_blur",     0,  48, "fast arm swing, heavy motion blur"),
    "smoke":      ("smoke",         100,  48, "semi-transparent smoke, uneven screen, rigging in frame"),
    "glasses":    ("disney_glasses",  0,  48, "tinted transparent plastic, refraction"),
    "sheen":      ("sheen",          60,  48, "thin translucent fabric, spill"),
    "cup":        ("drink_from_cup", 72,  48, "hands and green cup crossing the face, hair"),
    "converge":   ("converge",        0,  48, "foreground marker pole plus walking subject"),
    "cyc":        ("walking_sitting", 40, 48, "wide cyc shot, uneven screen, white coat spill, set pieces"),
    "converge200": ("converge",       0, 200, "end-to-end timing clip"),
}
# Above 1080p the HCW plates are upscaled: fine for memory/speed tests, not
# for judging detail.
RESOLUTIONS = {"1080": (1080, 1920), "720": (720, 1280), "1440": (1440, 2560),
               "3k": (1728, 3072), "2160": (2160, 3840)}
HINT_CLIPS = ("hair", "converge")
VIDEO_CLIPS = ("converge200",)


def plate_frames(src: Path, plate: str) -> list[Path]:
    frames = sorted((src / plate / "frames").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no PNG frames under {src / plate / 'frames'}")
    return frames


def green_difference_hint(rgb: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """rgb: HxWx3 float in [0, 1]. Returns HxW alpha (1 = subject)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    screen = np.clip((g - np.maximum(r, b)) * gain, 0.0, 1.0)
    return 1.0 - screen


def build_clip(src: Path, out: Path, name: str, res: str) -> dict:
    plate, offset, count, category = CLIPS[name]
    frames = plate_frames(src, plate)[offset:offset + count]
    if len(frames) != count:
        raise ValueError(f"{name}: wanted {count} frames from {plate}, got {len(frames)}")
    h, w = RESOLUTIONS[res]
    clip_dir = out / f"{name}_{res}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    first_rgb = None
    for i, path in enumerate(frames):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        resized = cv2.resize(bgr.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LANCZOS4)
        resized = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
        cv2.imwrite(str(clip_dir / f"frame_{i:06d}.png"), resized)
        if i == 0:
            first_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    entry = {
        "clip": f"{name}_{res}", "plate": plate, "category": category,
        "source_frames": [frames[0].name, frames[-1].name], "frames": count,
        "size": [w, h], "frames_dir": str(clip_dir.relative_to(out.parent)),
    }
    if name in HINT_CLIPS:
        hint = green_difference_hint(first_rgb)
        hint_path = out / f"{name}_{res}_hint.png"
        cv2.imwrite(str(hint_path), np.rint(hint * 255).astype(np.uint8))
        entry["hint_first_frame"] = str(hint_path.relative_to(out.parent))
    if name in VIDEO_CLIPS:
        mp4 = out / f"{name}_{res}.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-framerate", "30",
            "-i", str(clip_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "10", "-preset", "slow",
            "-pix_fmt", "yuv444p", str(mp4),
        ], check=True)
        entry["video"] = str(mp4.relative_to(out.parent))
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="Root holding <plate>/frames/*.png from the HCW zips.")
    ap.add_argument("--out", default=str(ROOT / "test_clips"))
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--res", nargs="*", default=list(RESOLUTIONS))
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in args.clips:
        for res in args.res:
            entry = build_clip(src, out, name, res)
            manifest.append(entry)
            print(f"{entry['clip']}: {entry['frames']} frames {entry['size']} "
                  f"from {entry['plate']} {entry['source_frames']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
