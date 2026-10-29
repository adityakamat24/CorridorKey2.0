#!/usr/bin/env python3
"""Stage-1 window time with the default attention kernels vs --fast's
(rt.attention_backends), alternating in one process so clock drift hits both.

Loads the model with the --optimized weight policy and lean activations,
puts only stage 1 on the GPU (the low-VRAM stage-1 phase), and runs one
window of random inputs per call (perf/floor_probe.make_inputs). Prints the
median time per window for each kernel choice and how far the fast kernels
move the baton and aux outputs.

Usage:
    python perf/stage1_attention_probe.py --hw 1152 1920 --frames 20 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from corridorkey import runtime as rt
from floor_probe import make_inputs
from lean_probe import run


def timed(stage1, inputs, fast: bool):
    with rt.attention_backends(fast):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _base, peak = run(stage1, inputs)
        return out, time.perf_counter() - t0, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs=2, default=[1152, 1920])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device="cuda")
    rt.apply_weight_policy(model, bf16_linear_conv=True)
    rt.configure_low_vram_inference(model, rt.InferenceConfig())
    rt.configure_lean_activations(model, True)
    model.foundations.cpu()
    model.stage2.cpu()
    model.stage1.cuda()
    torch.cuda.empty_cache()
    H, W = args.hw
    inputs = {k: (v.cpu() if v is not None else None)
              for k, v in make_inputs(model.stage1, args.frames, H, W).items()}
    torch.cuda.empty_cache()

    ref = None
    times = {False: [], True: []}
    peaks = {}
    diffs = {}
    for fast in (False, True):  # warm-up both
        timed(model.stage1, inputs, fast)
    for r in range(args.rounds):
        for fast in ((False, True) if r % 2 == 0 else (True, False)):
            out, secs, peak = timed(model.stage1, inputs, fast)
            times[fast].append(secs)
            peaks[fast] = peak / 2**30
            out = {k: v.cpu() for k, v in out.items()}
            if not fast and ref is None:
                ref = out
            elif fast:
                diffs = {k: float((out[k] - ref[k]).abs().max()) for k in out} if ref else diffs
            del out
            torch.cuda.empty_cache()
        print(f"round {r + 1}: default {times[False][-1]:.2f}s, fast {times[True][-1]:.2f}s",
              flush=True)
    if ref is not None and not diffs:
        out, _, _ = timed(model.stage1, inputs, True)
        diffs = {k: float((out[k].cpu() - ref[k]).abs().max()) for k in out}
    base, fast = statistics.median(times[False]), statistics.median(times[True])
    print(f"T={args.frames} {H}x{W}: default {base:.2f}s, fast attention {fast:.2f}s "
          f"(x{base / fast:.3f}); peak {peaks[False]:.2f} / {peaks[True]:.2f} GiB; "
          f"max abs diff {diffs}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(0), "hw": [H, W], "frames": args.frames,
            "default_s": times[False], "fast_s": times[True], "speedup": base / fast,
            "peak_gib": {"default": peaks[False], "fast": peaks[True]},
            "max_abs_diff": diffs}, indent=2))


if __name__ == "__main__":
    main()
