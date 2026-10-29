#!/usr/bin/env python3
"""How fast are device-to-host copies into pageable memory here, and does
staging them through a pinned buffer help?

`tensor.to("cpu")` copies into fresh pageable memory, which the driver
stages internally; on some hosts that runs far below the PCIe link. This
times, for one tensor size: the plain copy, a copy into a reused pinned
buffer, and a copy into fresh pageable memory through a reused pinned
staging buffer in chunks (what a drop-in replacement for `.to("cpu")`
would do). All three give the same bytes.

Usage:
    python perf/d2h_probe.py --mib 64 512 --chunk_mib 64
"""

from __future__ import annotations

import argparse
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mib", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--chunk_mib", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    chunk = args.chunk_mib << 20
    staging = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
    for mib in args.mib:
        n = mib << 20
        src = torch.randint(0, 255, (n,), dtype=torch.uint8, device="cuda")
        pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)

        def plain():
            return src.to("cpu")

        def into_pinned():
            pinned.copy_(src)
            return pinned

        def staged():
            out = torch.empty(n, dtype=torch.uint8)
            for i in range(0, n, chunk):
                k = min(chunk, n - i)
                staging[:k].copy_(src[i:i + k])
                out[i:i + k].copy_(staging[:k])
            return out

        ref = plain()
        for name, fn in (("pageable .to('cpu')", plain), ("into pinned buffer", into_pinned),
                         ("staged via pinned", staged)):
            fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = sorted(ts)[len(ts) // 2]
            print(f"{mib} MiB {name:22s}: {t * 1e3:7.1f} ms  {mib / 1024 / t:6.2f} GiB/s  "
                  f"same bytes {torch.equal(out, ref)}", flush=True)
        del src, pinned


if __name__ == "__main__":
    main()
