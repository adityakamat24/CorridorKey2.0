#!/usr/bin/env python3
"""Summarize `nsys stats -f csv` output for one profiled bench run.

Reads <prefix>_cuda_gpu_kern_sum.csv, _cuda_api_sum.csv and
_cuda_gpu_mem_time_sum.csv (and the bench JSON for wall time) and prints:
GPU kernel busy time vs wall time, host-sync API calls (stream/device/event
synchronize), memcpy calls by direction, and the top kernels.

Usage:
    python perf/nsys_summary.py perf/results/phase2/nsys_720_4x2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ns(row: dict, key: str = "Total Time (ns)") -> float:
    return float(row.get(key, 0) or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    p = args.prefix

    kern = rows(Path(f"{p}_cuda_gpu_kern_sum.csv"))
    api = rows(Path(f"{p}_cuda_api_sum.csv"))
    mem = rows(Path(f"{p}_cuda_gpu_mem_time_sum.csv"))
    bench = json.loads(Path(f"{p}.json").read_text()) if Path(f"{p}.json").exists() else None

    kern_ms = sum(ns(r) for r in kern) / 1e6
    mem_ms = sum(ns(r) for r in mem) / 1e6
    print(f"GPU kernel time {kern_ms:.1f} ms over {sum(int(r['Instances']) for r in kern)} launches; "
          f"memcpy/memset time {mem_ms:.1f} ms")
    if bench and bench["summary"].get("measured_runs"):
        wall = bench["summary"]["total_s_median"] * 1000
        print(f"run wall time {wall:.1f} ms -> GPU busy {100 * (kern_ms + mem_ms) / wall:.1f}% "
              f"(kernels {100 * kern_ms / wall:.1f}%), idle {100 - 100 * (kern_ms + mem_ms) / wall:.1f}%")

    print("\nhost-blocking / copy APIs:")
    for r in api:
        name = r["Name"]
        if any(k in name for k in ("Synchronize", "Memcpy", "Memset", "HostAlloc", "Malloc", "Free")):
            print(f"  {name:<32} calls {int(r['Num Calls']):>7}  total {ns(r) / 1e6:9.1f} ms  "
                  f"avg {ns(r, 'Avg (ns)') / 1e3:8.1f} us")
    print("\nmemory operations:")
    for r in mem:
        print(f"  {r['Operation']:<40} count {int(r['Count']):>7}  total {ns(r) / 1e6:9.1f} ms")
    print("\ntop kernels:")
    for r in sorted(kern, key=ns, reverse=True)[:args.top]:
        print(f"  {float(r['Time (%)']):5.1f}%  {ns(r) / 1e6:8.1f} ms  {int(r['Instances']):>6}x  {r['Name'][:100]}")


if __name__ == "__main__":
    main()
