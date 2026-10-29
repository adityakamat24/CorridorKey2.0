#!/usr/bin/env python3
"""Are the model's transposed convolutions deterministic run to run?

ConvTranspose2d runs as cuDNN's backward-data convolution, and some of its
algorithms accumulate with atomics, so the same input can give different
bits on every call. Which algorithm cuDNN picks depends on the shape. This
runs each depatchify layer (stage-1 aux head 512 -> 4, kernel 16, stride 16;
stage-2 head 1280 -> 4, kernel 32, stride 16, padding 8) a few times on one
fixed input per size, under bf16 autocast with bf16 weights as the optimized
path has them, and reports whether every call matches the first.

Usage:
    python perf/deconv_probe.py --hw 1152 1920 1536 2560 --frames 24 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs="+", default=[1152, 1920, 1536, 2560])
    ap.add_argument("--frames", type=int, nargs="+", default=[8, 24, 48])
    ap.add_argument("--calls", type=int, default=4)
    args = ap.parse_args()
    layers = {}
    with safe_open(str(ROOT / "weights/corridorkey_v2.safetensors"), "pt") as f:
        for name in ("stage1.aux_depatchify", "stage2.depatchify"):
            w = f.get_tensor(f"{name}.deconv.weight").cuda().to(torch.bfloat16)
            b = f.get_tensor(f"{name}.deconv.bias").cuda().to(torch.bfloat16)
            layers[name] = (w, b)
    sizes = list(zip(args.hw[0::2], args.hw[1::2]))
    for H, W in sizes:
        for name, (w, b) in layers.items():
            k = w.shape[-1]
            pad = (k - 16) // 2
            # stage 2 runs per frame; stage 1's aux head over the whole window
            for T in ([1] if name.startswith("stage2") else args.frames):
                torch.manual_seed(0)
                x = torch.randn(T, w.shape[0], H // 16, W // 16, device="cuda").to(torch.bfloat16)
                same, diff = True, 0.0
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    first = F.conv_transpose2d(x, w, b, stride=16, padding=pad)
                    for _ in range(args.calls - 1):
                        out = F.conv_transpose2d(x, w, b, stride=16, padding=pad)
                        if not torch.equal(first, out):
                            same = False
                            diff = max(diff, float((first - out).abs().max()))
                        del out
                print(f"{H}x{W} {name} T={T}: deterministic {same} (max diff {diff:.3g})",
                      flush=True)
                del x, first
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
