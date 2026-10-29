#!/usr/bin/env python3
"""Candidate faster versions of the prepass's sRGB conversion, checked bitwise.

The foundation view of a linear EXR frame is clip(linear_to_srgb(max(x, 0)),
0, 1). linear_to_srgb runs about ten whole-frame torch ops; at 4K each one
writes a 107 MB temporary. This compares, on every frame of a clip:

  ref      cu.linear_to_srgb (what the default path runs)
  nonneg   cu.linear_to_srgb_nonneg (--decode_once: no negative branch)
  inplace  the same ops as nonneg, reusing its buffers (in-place variants)
  tiled    inplace on row bands of `--rows` rows, each band a multiple of
           the vector width, so the vector/scalar split per element is the
           same as on the whole frame

and prints whether each is bitwise equal to ref and its time per frame.

Usage:
    python perf/srgb_probe.py --input /root/tos/04_5h --frames 24 --rows 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from corridorkey import color_utils as cu
from corridorkey import video_io as v


def srgb_inplace(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x)
    pos_t = t.clamp(min=0.0)
    pos_pow = pos_t.clamp(min=0.0031308)
    pos_pow.pow_(1.0 / 2.4)
    pos_pow.mul_(1.055).sub_(0.055)
    low = pos_t <= 0.0031308
    pos_t.mul_(12.92)
    return torch.where(low, pos_t, pos_pow).numpy()


def srgb_tiled(x: np.ndarray, rows: int) -> np.ndarray:
    out = np.empty_like(x)
    for r0 in range(0, x.shape[0], rows):
        out[r0:r0 + rows] = srgb_inplace(np.ascontiguousarray(x[r0:r0 + rows]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()
    paths = v.frame_paths(args.input)[:args.frames]
    times = {k: 0.0 for k in ("ref", "nonneg", "inplace", "tiled")}
    equal = {k: True for k in ("nonneg", "inplace", "tiled")}
    for p in paths:
        x = np.maximum(v.read_image_rgb(p).astype(np.float32, copy=False), 0.0)
        t0 = time.perf_counter(); ref = cu.linear_to_srgb(x); times["ref"] += time.perf_counter() - t0
        for name, fn in (("nonneg", cu.linear_to_srgb_nonneg), ("inplace", srgb_inplace),
                         ("tiled", lambda a: srgb_tiled(a, args.rows))):
            t0 = time.perf_counter()
            out = fn(x)
            times[name] += time.perf_counter() - t0
            equal[name] &= bool(np.array_equal(out.view(np.uint32), ref.view(np.uint32)))
    n = len(paths)
    print(f"{n} frames {x.shape[1]}x{x.shape[0]}, torch threads {torch.get_num_threads()}")
    for name, secs in times.items():
        eq = "" if name == "ref" else f"  bitwise equal to ref: {equal[name]}"
        print(f"  {name:8s} {secs / n * 1000:7.1f} ms/frame{eq}")


if __name__ == "__main__":
    main()
