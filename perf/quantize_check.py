"""Exhaustive check that uint16 quantization on the GPU matches the CPU path.

The default renderer copies float32 frames to the host and quantizes there:
`(x.cpu().numpy() * 65535.0).astype(np.uint16)`, with x already clamped to
[0, 1]. --pinned_output quantizes on the GPU instead,
`(x * 65535.0).to(torch.uint16)`. This feeds every float32 bit pattern in
[0, 1] (and -0.0, NaN) through both and counts mismatches.

    python perf/quantize_check.py
"""

import time

import numpy as np
import torch


def main():
    one = int(np.array(1.0, dtype=np.float32).view(np.uint32))
    step = 1 << 24
    mismatches = 0
    t0 = time.time()
    for lo in range(0, one + 1, step):
        bits = np.arange(lo, min(lo + step, one + 1), dtype=np.uint32)
        x = bits.view(np.float32)
        ref = (x * 65535.0).astype(np.uint16)
        gpu = (torch.from_numpy(x).cuda() * 65535.0).to(torch.uint16).cpu().numpy()
        mismatches += int(np.count_nonzero(ref != gpu))
    special = np.array([-0.0, np.nan], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        ref = (special * 65535.0).astype(np.uint16)
    gpu = (torch.from_numpy(special).cuda() * 65535.0).to(torch.uint16).cpu().numpy()
    print(f"checked {one + 1} float32 values in [0, 1] in {time.time() - t0:.1f}s: "
          f"{mismatches} mismatches; -0.0 -> {ref[0]}/{gpu[0]}, NaN -> {ref[1]}/{gpu[1]} "
          f"(cpu/gpu)")


if __name__ == "__main__":
    main()
