#!/usr/bin/env python3
"""--gpu_prep against the CPU frame preparation, input value by input value.

Builds frames that hold every value a decoded frame can contain, runs each
through the CPU path (video_io.preprocess_raw_both) and the GPU path
(video_io.frame_dir_views_on_device), and compares the bf16 bits of both
views:

  png8      every 8-bit value (k / 255): every non-EXR frame
  half      every finite half float, as linear and as display input: every
            EXR frame that --gpu_prep converts on the GPU
  float     with --float, float32 bit patterns from 0 to 65504 and from -0 to
            -65504, every --stride-th one; EXR frames with values outside
            half precision are converted on the CPU, since a few of these
            round differently on the GPU

For png8 and half it also runs the CPU path's per-value functions through
NumPy's and PyTorch's scalar loops (a stride-2 view, and one value at a
time) and compares them with the vectorized loop, since which loop an
element takes depends on how a frame is split across threads.

Prints the mismatch count per set; exit status 1 if png8 or half differs.

Usage:
    python perf/gpu_prep_probe.py [--float --stride 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from corridorkey import color_utils as cu
from corridorkey import video_io as v


def compare(frames, meta, device):
    mismatches = [0, 0]
    examples = []
    for raw in frames:
        cpu = v.preprocess_raw_both(raw, meta)
        gpu = v.frame_dir_views_on_device(raw, meta, device)
        for i, (a, b) in enumerate(zip(cpu, gpu)):
            a16 = a.view(torch.int16)
            b16 = b.cpu().contiguous().view(torch.int16)
            diff = a16 != b16
            n = int(diff.sum())
            if n:
                mismatches[i] += n
                c, y, x = (int(k) for k in diff.nonzero()[0])
                h0 = (meta["shape_hw"][0] - raw.shape[0]) // 2
                y0 = min(max(y - h0, 0), raw.shape[0] - 1)
                examples.append((("display", "model")[i], float(raw[y0, x, c]),
                                 float(a[c, y, x]), float(b[c, y, x])))
    return mismatches, examples[:4]


def cpu_loop_mismatches(values, linear):
    def views(x):
        if linear:
            display = np.clip(cu.linear_to_srgb_nonneg(np.maximum(x, 0.0)), 0.0, 1.0)
            model = cu.linear_to_asinh(x)
        else:
            display = np.clip(x, 0.0, 1.0)
            model = cu.linear_to_asinh(cu.srgb_to_linear(display))
        return [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                .to(torch.bfloat16).view(torch.int16) for a in (display, model)]

    values = values.astype(np.float32)
    vector = [v[:values.size] for v in views(np.tile(values, 64))]
    spaced = np.empty(2 * values.size, dtype=np.float32)
    spaced[::2] = values
    stride2 = views(spaced[::2])
    singles = [views(values[i:i + 1]) for i in range(values.size)]
    singles = [torch.cat([s[k] for s in singles]) for k in range(2)]
    return sum(int((vector[k] != other[k]).sum()) for other in (stride2, singles) for k in range(2))


def tiled(values, h, w):
    per = h * w * 3
    n = -(-values.size // per)
    return np.resize(values.astype(np.float32), n * per).reshape(n, h, w, 3)


def sweep(lo, hi, stride, h, w):
    per = h * w * 3
    for start in range(lo, hi, per * stride):
        bits = np.arange(start, min(hi, start + per * stride), stride, dtype=np.uint64)
        bits = np.resize(bits.astype(np.uint32), per)
        yield bits.view(np.float32).reshape(h, w, 3)


def meta_for(h, w, h_out, linear):
    return {"shape_hw": (h_out, w), "model_native_hw": (h, w), "native_hw": (h, w),
            "exr_linear": linear}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"numpy {np.__version__}")
    halves = np.arange(65536, dtype=np.uint16).view(np.float16)
    halves = halves[np.isfinite(halves)].astype(np.float32)
    sets = [
        ("png8", tiled(np.arange(256).astype(np.float32) / 255.0, 1080, 1920), meta_for(1080, 1920, 1152, False)),
        ("png8 odd size", tiled(np.arange(256).astype(np.float32) / 255.0, 1001, 1333), meta_for(1001, 1333, 1024, False)),
        ("half linear", tiled(halves, 2160, 4096), meta_for(2160, 4096, 2176, True)),
        ("half display", tiled(halves, 2160, 4096), meta_for(2160, 4096, 2176, False)),
    ]
    if args.float:
        sets += [
            ("float linear +", sweep(0, 0x477FE001, args.stride, 2160, 4096), meta_for(2160, 4096, 2176, True)),
            ("float linear -", sweep(0x80000000, 0xC77FE001, args.stride, 2160, 4096), meta_for(2160, 4096, 2176, True)),
        ]
    failed = False
    for name, values, linear in (("png8", np.arange(256).astype(np.float32) / 255.0, False),
                                 ("half linear", halves, True), ("half display", halves, False)):
        n = cpu_loop_mismatches(values, linear)
        failed |= bool(n)
        print(f"{name:<16} CPU scalar loops vs vectorized: {n} mismatches", flush=True)
    for name, frames, meta in sets:
        mism, examples = compare(frames, meta, args.device)
        failed |= any(mism) and not name.startswith("float")
        print(f"{name:<16} display mismatches {mism[0]:>10}  model mismatches {mism[1]:>10}"
              + (f"  e.g. {examples}" if examples else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
