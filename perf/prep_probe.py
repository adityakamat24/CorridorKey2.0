#!/usr/bin/env python3
"""Where a frame's CPU preparation goes, and how the prepass reader scales.

Times each step of preparing one frame (decode, the display view the
foundations get, the model view stages 1 and 2 get), then the throughput of
iter_foundation_frames (the prepass reader) and of model-view reads at a few
worker counts. CPU only; nothing here touches the GPU.

Usage:
    python perf/prep_probe.py --input /root/tos/04_5h --frame_dir_linear --frames 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from corridorkey import color_utils as cu
from corridorkey import video_io as v


def timed(fn, *args, repeat=3):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return out, best * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--frame_dir_linear", action="store_true")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = v.probe_frame_dir_meta(args.input, num_frames=args.frames, shape_quantum=128,
                                  fps=24.0, exr_linear=args.frame_dir_linear, model_scale=1.0)
    h_out, w_out = meta["shape_hw"]
    path = meta["frame_paths"][0]
    steps = {}
    raw, steps["decode"] = timed(v.read_image_rgb, path)
    if args.frame_dir_linear:
        linear = raw.astype(np.float32, copy=False)
        pos, steps["maximum"] = timed(np.maximum, linear, 0.0)
        srgb, steps["linear_to_srgb"] = timed(cu.linear_to_srgb, pos)
        display, steps["clip"] = timed(np.clip, srgb, 0.0, 1.0)
    else:
        display, steps["clip"] = timed(np.clip, raw, 0.0, 1.0)
        linear, steps["srgb_to_linear"] = timed(cu.srgb_to_linear, display)
    padded, steps["pad"] = timed(lambda x: v.center_pad_to_shape(np.clip(x, 0.0, 1.0), h_out, w_out)[0], display)
    _, steps["to_bf16"] = timed(lambda x: torch.from_numpy(x.transpose(2, 0, 1)).to(dtype=torch.bfloat16), padded)
    lin_pad, _ = timed(lambda x: v.center_pad_to_shape(x, h_out, w_out)[0], linear)
    _, steps["linear_to_asinh"] = timed(cu.linear_to_asinh, lin_pad)
    _, steps["foundation_view_total"] = timed(v.preprocess_frame_dir_foundation, path, meta)
    _, steps["model_view_total"] = timed(
        lambda: v.preprocess_frame_dir_path(path, meta, need_model=True, need_display=False))
    _, steps["all_views_total"] = timed(lambda: v.preprocess_frame_dir_path(path, meta, need_model=True))
    print(f"one frame {w_out}x{h_out}, torch threads {torch.get_num_threads()}:")
    for k, ms in steps.items():
        print(f"  {k:24s} {ms:8.1f} ms")
    rates = {}
    for w in args.workers:
        t0 = time.perf_counter()
        for _t, _f in v.iter_foundation_frames(args.input, meta, reader_kind="frame_dir", workers=w):
            pass
        fps = args.frames / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as pool:
            list(pool.map(lambda p: v.preprocess_frame_dir_path(p, meta, need_model=True,
                                                                 need_display=False),
                          meta["frame_paths"]))
        mfps = args.frames / (time.perf_counter() - t0)
        rates[w] = {"foundation_fps": fps, "model_view_fps": mfps}
        print(f"  workers {w:2d}: prepass reader {fps:5.2f} frames/s, model views {mfps:5.2f} frames/s")
    if args.out:
        Path(args.out).write_text(json.dumps({"input": args.input, "shape_hw": [h_out, w_out],
                                              "torch_threads": torch.get_num_threads(),
                                              "steps_ms": steps, "rates": rates}, indent=2))


if __name__ == "__main__":
    main()
