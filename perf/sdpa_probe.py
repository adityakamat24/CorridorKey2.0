#!/usr/bin/env python3
"""Time each SDPA backend at the model's own attention shapes, and how far
its output is from the memory-efficient kernel (the one Windows builds of
PyTorch use, which have no flash attention).

Shapes: stage 2's full-frame attention (16 heads x H/16*W/16 tokens x 80)
and its 16x16 windows, stage 1's Swin-3D windows (T x 8 x 8 tokens x 64).

Usage:
    python perf/sdpa_probe.py --hw 1152 1920 --frames 24
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {"efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}


def timed(fn, reps=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(reps):
        out = fn()
    ev[1].record()
    torch.cuda.synchronize()
    return out, ev[0].elapsed_time(ev[1]) / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs=2, default=[1152, 1920])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--backends", nargs="+", default=["efficient", "flash", "cudnn"])
    args = ap.parse_args()
    H, W = args.hw
    shapes = {
        "stage2 full-frame": (1, 16, (H // 16) * (W // 16), 80),
        "stage2 16x16 windows": ((H // 16) * (W // 16) // 256, 16, 256, 80),
        "stage1 Swin-3D windows": (16, 8, args.frames * 64, 64),
    }
    torch.manual_seed(0)
    for name, (b, h, n, d) in shapes.items():
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        ref = None
        for be in args.backends:
            try:
                with sdpa_kernel([BACKENDS[be]]):
                    out, ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
            except RuntimeError as e:
                print(f"{name:24s} {be:9s}: not available ({str(e).splitlines()[0][:80]})")
                continue
            if ref is None:
                ref = out
                print(f"{name:24s} {be:9s}: {ms:8.3f} ms  (reference)")
            else:
                diff = (out.float() - ref.float()).abs()
                print(f"{name:24s} {be:9s}: {ms:8.3f} ms  x{ref_ms / ms:4.2f}  max diff {diff.max():.3g}, "
                      f"mean {diff.mean():.3g}, equal {torch.equal(out, ref)}")
            if be == args.backends[0]:
                ref_ms = ms
        del q, k, v, ref


if __name__ == "__main__":
    main()
