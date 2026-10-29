#!/usr/bin/env python3
"""Is stage 2 bit-identical when frames are batched, and is it faster?

Runs stage 2 (the --optimized weight policy, autocast bf16, as the runtime
does) on the first frames of a clip with a fixed random baton, one frame per
call and then `--batch` frames per call (--stage2_batch), and compares the
logits frame by frame bitwise. Prints the time per frame for each batch size.

Usage:
    python perf/batch_probe.py --input test_clips/converge200_1080 --frames 8 --batch 2 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from corridorkey import runtime as rt
from corridorkey.blocks import set_region_index_cache


@torch.no_grad()
def run(model, frames, baton, n):
    outs = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, len(frames), n):
        rgb = torch.stack(frames[s:s + n]).unsqueeze(0).cuda()
        bt = torch.from_numpy(baton[s:s + n]).unsqueeze(0).cuda().float()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.stage2(rgb, bt)
            if isinstance(logits, tuple):
                logits = logits[0]
        outs.extend(logits[0, i].float().cpu() for i in range(logits.shape[1]))
    torch.cuda.synchronize()
    return outs, (time.perf_counter() - t0) / len(frames) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="test_clips/converge200_1080")
    ap.add_argument("--frame_dir_linear", action="store_true")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--batch", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device="cuda", skip_init=True)
    rt.apply_weight_policy(model, bf16_linear_conv=True)
    cfg = rt.InferenceConfig(low_vram=True, lean_activations=True, cache_region_indices=True)
    rt.configure_low_vram_inference(model, cfg)
    rt.configure_lean_activations(model, True)
    set_region_index_cache(True)
    rt._phase2_setup(model)
    meta = rt.probe_frame_dir_meta(args.input, num_frames=args.frames, shape_quantum=128,
                                   fps=24.0, exr_linear=args.frame_dir_linear, model_scale=1.0)
    H, W = meta["shape_hw"]
    reader = rt.SequentialFrameDirWindowReader(args.input, meta, model_only=True)
    model_chunk, _, _ = reader.window(0, args.frames)
    frames = [model_chunk[t] for t in range(args.frames)]
    baton = (np.random.default_rng(0).standard_normal(
        (args.frames, model.stage1.d_16, H // 16, W // 16), dtype=np.float32).astype(np.float16))
    run(model, frames[:2], baton[:2], 1)  # warm-up
    ref, ms1 = run(model, frames, baton, 1)
    rows = [{"batch": 1, "ms_per_frame": ms1, "bitwise_equal": True, "max_abs_diff": 0.0}]
    print(f"batch 1: {ms1:.1f} ms/frame")
    for n in args.batch:
        run(model, frames[:n], baton[:n], n)  # warm-up for this shape
        out, ms = run(model, frames, baton, n)
        equal = all(torch.equal(a, b) for a, b in zip(ref, out))
        diff = max(float((a - b).abs().max()) for a, b in zip(ref, out))
        rows.append({"batch": n, "ms_per_frame": ms, "bitwise_equal": equal, "max_abs_diff": diff})
        print(f"batch {n}: {ms:.1f} ms/frame (x{ms1 / ms:.2f}), bitwise equal {equal}, "
              f"max abs diff {diff:.3g}")
    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0),
                                              "shape_hw": [H, W], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
