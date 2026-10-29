#!/usr/bin/env python3
"""Visual diffs of a candidate CorridorKey EXR render against the golden one.

For the worst frames of a clip (taken from a compare.py JSON, or given with
--frames) this writes, per frame:
  <clip>_tNNN_overview.png  golden comp | candidate comp | |da| x gain heatmap,
                            full frame, downscaled
  <clip>_tNNN_crop.png      golden comp | candidate comp | golden a |
                            candidate a | |da| x gain, a 1:1 crop (shown 2x)
                            centred on the worst edge-band error
  <clip>_tNNN_flip.gif      (with --gif) the crop flipping golden/candidate

Comps are the raw predicted foreground (no despill) over a 64 px checker,
composited in linear light and shown in sRGB.

Usage:
    python perf/visual_diff.py --ref perf/out/golden/hair_1080 \
        --cand perf/out/cand/hair_1080 --cmp perf/results/cmp_hair.json \
        --out perf/results/vdiff_hair --gif
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from compare import EDGE_HI, EDGE_LO, clips_in, read_exr, srgb_encode


def srgb_decode(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def checker(h: int, w: int, size: int = 64) -> np.ndarray:
    y, x = np.indices((h, w))
    gray = np.where(((y // size) + (x // size)) % 2 == 0, 0.75, 0.5)
    return np.repeat(gray[..., None], 3, axis=2).astype(np.float32)


def comp_srgb(fg_lin: np.ndarray, alpha: np.ndarray, bg_lin: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return srgb_encode(np.maximum(fg_lin, 0.0) * a + bg_lin * (1.0 - a))


def heat(err: np.ndarray, gain: float) -> np.ndarray:
    u8 = (np.clip(err * gain, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def gray3(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(a, 0.0, 1.0)[..., None], 3, axis=2)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8).copy()
    cv2.rectangle(out, (0, 0), (8 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_rgb(path: Path, rgb_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def worst_window(err: np.ndarray, band: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size window with the most edge-band error."""
    h, w = err.shape
    score = cv2.boxFilter((err * band).astype(np.float32), -1, (size, size),
                          normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    y0 = int(np.clip(cy - size // 2, 0, max(h - size, 0)))
    x0 = int(np.clip(cx - size // 2, 0, max(w - size, 0)))
    return y0, x0


def render_frame(ref: Path, cand: Path, clip: str, t: int, out: Path, *,
                 gain: float, crop: int, overview_w: int, gif: bool) -> None:
    name = f"{t:06d}.exr"
    a_r = read_exr(ref / f"{clip}_alpha_exr" / f"alpha_{name}")
    a_c = read_exr(cand / f"{clip}_alpha_exr" / f"alpha_{name}")
    fg_r = read_exr(ref / f"{clip}_fg_exr" / f"fg_{name}")
    fg_c = read_exr(cand / f"{clip}_fg_exr" / f"fg_{name}")
    h, w = a_r.shape
    bg = srgb_decode(checker(h, w))
    c_r = comp_srgb(fg_r, a_r, bg)
    c_c = comp_srgb(fg_c, a_c, bg)
    err = np.abs(a_c - a_r)
    band = (a_r > EDGE_LO) & (a_r < EDGE_HI)

    scale = overview_w / float(w)
    small = lambda x: cv2.resize(x, (overview_w, int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
    tiles = [label(small(c_r), "golden"), label(small(c_c), "candidate"),
             label(small(heat(err, gain)), f"|da| x{gain:g}  max {err.max():.3g}")]
    save_rgb(out / f"{clip}_t{t:03d}_overview.png", np.concatenate(tiles, axis=1))

    size = min(crop, h, w)
    y0, x0 = worst_window(err, band, size)
    sl = (slice(y0, y0 + size), slice(x0, x0 + size))
    up = lambda x: cv2.resize(x, (2 * size, 2 * size), interpolation=cv2.INTER_NEAREST)
    crop_tiles = [
        label(up(c_r[sl]), "golden"),
        label(up(c_c[sl]), "candidate"),
        label(up(gray3(a_r[sl])), "golden a"),
        label(up(gray3(a_c[sl])), "candidate a"),
        label(up(heat(err[sl], gain)), f"|da| x{gain:g} @ ({x0},{y0})"),
    ]
    save_rgb(out / f"{clip}_t{t:03d}_crop.png", np.concatenate(crop_tiles, axis=1))

    if gif:
        from PIL import Image
        frames = [Image.fromarray(crop_tiles[0]), Image.fromarray(crop_tiles[1])]
        frames[0].save(out / f"{clip}_t{t:03d}_flip.gif", save_all=True,
                       append_images=frames[1:], duration=600, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmp", default=None, help="compare.py JSON for worst frames.")
    ap.add_argument("--clip", default=None, help="Only this clip.")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="Explicit frame indices (overrides --cmp ranking).")
    ap.add_argument("--worst", type=int, default=3)
    ap.add_argument("--gain", type=float, default=20.0,
                    help="Multiplier on |alpha diff| before the heatmap clips at 1.")
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--overview_w", type=int, default=640)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    ref, cand, out = Path(args.ref), Path(args.cand), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int]] = []
    if args.frames is not None:
        if not args.clip:
            ap.error("--frames needs --clip")
        jobs = [(args.clip, t) for t in args.frames]
    else:
        if not args.cmp:
            ap.error("give --cmp (compare.py JSON) or --clip with --frames")
        data = json.loads(Path(args.cmp).read_text())
        for r in data["clips"]:
            if args.clip and r["clip"] != args.clip:
                continue
            jobs += [(r["clip"], w["t"]) for w in r["worst_frames"][:args.worst]]
    ref_clips, cand_clips = clips_in(ref), clips_in(cand)
    for clip, t in jobs:
        render_frame(ref_clips[clip], cand_clips[clip], clip, t, out,
                     gain=args.gain, crop=args.crop,
                     overview_w=args.overview_w, gif=args.gif)
        print(f"{clip} t={t} -> {out}")


if __name__ == "__main__":
    main()
