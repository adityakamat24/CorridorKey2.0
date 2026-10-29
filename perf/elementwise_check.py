#!/usr/bin/env python3
"""Do the CPU kernels behind the frame conversions give the same result per
element whatever path computes it?

PyTorch's elementwise CPU kernels run a vectorized loop over contiguous data
and the scalar function for leftovers and for strided data; NumPy's ufuncs
likewise. Which elements take which path depends on how a tensor is split
across threads, so processing a frame in tiles is exact only if both paths
agree. This feeds every relevant float32 bit pattern through each function
twice, contiguous (vector path) and through a stride-2 view (scalar path),
and counts disagreements:

  pow      torch .pow(1/2.4) over [0.0031308, inf] (linear_to_srgb's input
           after its clamp)
  arcsinh  numpy np.arcsinh over every finite float32 (linear_to_asinh)

Usage:
    python perf/elementwise_check.py
"""

from __future__ import annotations

import time

import numpy as np
import torch


def bits_range(lo: float, hi: float):
    a = int(np.array(lo, dtype=np.float32).view(np.uint32))
    b = int(np.array(hi, dtype=np.float32).view(np.uint32))
    return a, b


def check(name, fn_contig, fn_strided, ranges, step=1 << 24):
    t0 = time.time()
    total = mismatch = 0
    for a, b in ranges:
        for lo in range(a, b + 1, step):
            bits = np.arange(lo, min(lo + step, b + 1), dtype=np.uint32)
            x = bits.view(np.float32)
            wide = np.empty(2 * x.size, dtype=np.float32)
            wide[::2] = x
            c = fn_contig(x)
            s = fn_strided(wide[::2])
            mismatch += int(np.count_nonzero(c.view(np.uint32) != s.view(np.uint32)))
            total += x.size
    print(f"{name}: {total} values, {mismatch} differ between vector and scalar paths "
          f"({time.time() - t0:.0f}s)", flush=True)
    return mismatch


def main():
    e = 1.0 / 2.4
    pow_c = lambda x: torch.from_numpy(x).pow(e).numpy()
    pow_s = lambda x: torch.from_numpy(x).pow(e).contiguous().numpy()
    check("torch pow(1/2.4)", pow_c, pow_s, [bits_range(0.0031308, np.inf)])
    pos = bits_range(0.0, np.finfo(np.float32).max)
    neg = (pos[0] | 0x80000000, pos[1] | 0x80000000)
    check("numpy arcsinh", np.arcsinh, lambda x: np.ascontiguousarray(np.arcsinh(x)),
          [pos, neg])
    print(f"torch {torch.__version__}, numpy {np.__version__}, "
          f"cpu capability {torch.backends.cpu.get_cpu_capability()}")


if __name__ == "__main__":
    main()
