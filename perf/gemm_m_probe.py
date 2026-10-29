#!/usr/bin/env python3
"""Is a bf16 Linear row-for-row identical when fed a subset of its rows?

For the stage-1 Swin-3D QKV projection shape (dim 512 -> 1536, bf16 weights,
autocast) this runs F.linear on a full padded /16 frame (H x W tokens) and on
row bands of it (band rows x W tokens, the band a window batch plus a
depthwise-conv halo would need), then checks the band's outputs bitwise
against the matching rows of the full result. cuBLAS may pick a different
kernel for a different row count, which can change the rounding.

Usage:
    python perf/gemm_m_probe.py --hw 80 128 --bands 38 32 22 16 --frames 1 4
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs=2, default=[80, 128], help="padded /16 grid")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--out_dim", type=int, default=1536)
    ap.add_argument("--bands", type=int, nargs="+", default=[38, 32, 22, 16])
    ap.add_argument("--frames", type=int, nargs="+", default=[1])
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    H, W = args.hw
    torch.manual_seed(0)
    w = (torch.randn(args.out_dim, args.dim, device="cuda") * 0.02).to(torch.bfloat16)
    b = (torch.randn(args.out_dim, device="cuda") * 0.02).to(torch.bfloat16)
    for T in args.frames:
        for trial in range(args.trials):
            x = torch.randn(T, H, W, args.dim, device="cuda").to(torch.bfloat16)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                full = F.linear(x, w, b)
                for rows in args.bands:
                    for r0 in sorted({0, (H - rows) // 2, H - rows}):
                        band = F.linear(x[:, r0:r0 + rows].contiguous(), w, b)
                        ref = full[:, r0:r0 + rows]
                        eq = torch.equal(band, ref)
                        diff = float((band.float() - ref.float()).abs().max())
                        print(f"T={T} trial={trial} rows={rows} r0={r0} M={T * rows * W}"
                              f" vs {T * H * W}: equal {eq} (max diff {diff:.3g})", flush=True)


if __name__ == "__main__":
    main()
