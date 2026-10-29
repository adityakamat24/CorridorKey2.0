#!/usr/bin/env python3
"""Where is the GPU idle? Reads a torch.profiler chrome trace (bench.py
--torch_profile), merges all GPU activity (kernels, memcpy, memset) into busy
intervals, and attributes every idle gap to the CPU work in flight at that
moment: the innermost CPU op or record_function range covering the gap's
midpoint, summed by name. Also reports the busy fraction overall.

Usage:
    python perf/trace_gaps.py perf/results/phase4/profile_1080_20x10/trace.json
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel", "Memcpy", "Memset"}
CPU_CATS = {"cpu_op", "user_annotation", "python_function", "Operator"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--min_gap_us", type=float, default=200.0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    events = json.load(open(args.trace))
    events = events["traceEvents"] if isinstance(events, dict) else events
    gpu, cpu = [], []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        if cat in GPU_CATS:
            gpu.append((e["ts"], e["ts"] + e["dur"]))
        elif cat in CPU_CATS:
            cpu.append((e["ts"], e["ts"] + e["dur"], e.get("name", "?")))
    gpu.sort()
    busy = []
    for s, t in gpu:
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], t)
        else:
            busy.append([s, t])
    start, end = busy[0][0], busy[-1][1]
    busy_us = sum(t - s for s, t in busy)
    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1)
            if busy[i + 1][0] - busy[i][1] >= args.min_gap_us]
    print(f"GPU span {(end - start) / 1e6:.2f} s, busy {busy_us / 1e6:.2f} s "
          f"({100 * busy_us / (end - start):.1f}%), "
          f"{len(gaps)} gaps >= {args.min_gap_us:.0f} us totalling "
          f"{sum(t - s for s, t in gaps) / 1e6:.2f} s")

    cpu.sort()
    starts = [c[0] for c in cpu]
    by_name = defaultdict(float)
    for s, t in gaps:
        mid = (s + t) / 2
        i = bisect.bisect_right(starts, mid)
        best = None
        for j in range(max(0, i - 4000), i):  # innermost = latest start covering mid
            c0, c1, name = cpu[j]
            if c1 >= mid and (best is None or c0 >= best[0]):
                best = (c0, name)
        by_name[best[1] if best else "<no CPU op>"] += t - s
    print(f"\nidle time by CPU op in flight (top {args.top}):")
    for name, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {us / 1e6:8.3f} s  {name[:110]}")


if __name__ == "__main__":
    main()
