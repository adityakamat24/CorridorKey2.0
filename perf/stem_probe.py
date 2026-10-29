#!/usr/bin/env python3
"""Stage-1 stem conv (4 -> 512 channels, kernel 32, stride 16, padding 8):
how much memory does one call over a whole window take beyond its input and
output, and is calling it a few frames at a time bit-identical?

The stem runs per frame (batch = B*T), so chunking over frames is the same
math; the question is whether cuDNN picks the same algorithm, and so the
same rounding, for a smaller batch. Uses the checkpoint's stem weights
under bf16 autocast, as the runtime does (with --bf16_weights the weight is
already bf16; both are checked).

Usage:
    python perf/stem_probe.py --hw 1152 1920 --frames 24 --chunks 1 2 4 8
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
    ap.add_argument("--hw", type=int, nargs=2, default=[1152, 1920])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()
    H, W = args.hw
    with safe_open(str(ROOT / "weights/corridorkey_v2.safetensors"), "pt") as f:
        w32 = f.get_tensor("stage1.stem.conv.weight").cuda()
        b32 = f.get_tensor("stage1.stem.conv.bias").cuda()
    pad = (w32.shape[-1] - 16) // 2
    for wdtype in (torch.float32, torch.bfloat16):
        w, b = w32.to(wdtype), b32.to(wdtype)
        for trial in range(args.trials):
            torch.manual_seed(trial)
            x = (torch.randn(args.frames, 4, H, W, device="cuda") * 0.5).to(torch.bfloat16)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                full = F.conv2d(x, w, b, stride=16, padding=pad)
            torch.cuda.synchronize()
            extra = (torch.cuda.max_memory_allocated() - base
                     - full.numel() * full.element_size()) / 2**20
            print(f"weights {str(wdtype)[6:]} trial {trial}: full call {tuple(x.shape)} -> "
                  f"{tuple(full.shape)} {full.dtype}, transient beyond output {extra:.0f} MiB",
                  flush=True)
            for c in args.chunks:
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    parts = [F.conv2d(x[t:t + c], w, b, stride=16, padding=pad)
                             for t in range(0, args.frames, c)]
                torch.cuda.synchronize()
                out = torch.cat(parts)
                extra_c = (torch.cuda.max_memory_allocated() - base
                           - 2 * full.numel() * full.element_size()) / 2**20
                diff = float((out.float() - full.float()).abs().max())
                print(f"   chunk {c}: equal {torch.equal(out, full)} (max diff {diff:.3g}), "
                      f"transient {extra_c:.0f} MiB", flush=True)
                del parts, out
            del x, full
            torch.cuda.empty_cache()
    wb, bb = w32.to(torch.bfloat16), b32.to(torch.bfloat16)
    x = torch.randn(args.frames, 4, H, W, device="cuda").to(torch.bfloat16)

    def timed(c):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        times = []
        for _ in range(5):
            ev[0].record()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(0, args.frames, c):
                    F.conv2d(x[t:t + c], wb, bb, stride=16, padding=pad)
            ev[1].record()
            torch.cuda.synchronize()
            times.append(ev[0].elapsed_time(ev[1]))
        return sorted(times)[2]
    for c in [args.frames] + args.chunks:
        print(f"time, chunk {c}: {timed(c):.1f} ms for {args.frames} frames", flush=True)
    del x
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        x = torch.randn(args.frames, 4, H, W, device="cuda").to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            F.conv2d(x, w32.to(torch.bfloat16), b32.to(torch.bfloat16), stride=16, padding=pad)
            F.conv2d(x[:1], w32.to(torch.bfloat16), b32.to(torch.bfloat16), stride=16, padding=pad)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
