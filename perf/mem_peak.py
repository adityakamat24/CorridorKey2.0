#!/usr/bin/env python3
"""What is alive at the allocator peak, and what grows with the window size.

Reads snapshots written by `perf/bench.py --mem_snapshot` (i.e.
torch.cuda.memory._record_memory_history + _dump_snapshot), replays the
device trace to find the moment of maximum allocated bytes, and attributes
every block alive at that moment to the deepest stack frame inside the
corridorkey package (file:line function), falling back to the deepest frame
overall.

One snapshot: prints the top allocation sites at the peak.
Two snapshots (--diff A B, same resolution, different window sizes):
prints per-site byte deltas between the two peaks, divided by the frame
difference, i.e. the composition of the per-frame VRAM slope.

Usage:
    python perf/mem_peak.py perf/results/mem/1080_w4.pickle
    python perf/mem_peak.py --diff perf/results/mem/1080_w4.pickle \
        perf/results/mem/1080_w8.pickle --frames 4 8
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict


def site_of(frames: list[dict]) -> str:
    """Deepest corridorkey frame, else the deepest frame with a filename."""
    if not frames:
        return "<no stack>"
    for fr in frames:  # snapshot frames are innermost first
        name = fr.get("filename", "")
        if "corridorkey" in name.replace("\\", "/") and "/perf/" not in name.replace("\\", "/"):
            short = name.replace("\\", "/").split("corridorkey/")[-1]
            return f"{short}:{fr.get('line')} {fr.get('name')}"
    fr = frames[0]
    return f"{fr.get('filename', '?').replace(chr(92), '/').split('/')[-1]}:{fr.get('line')} {fr.get('name')}"


def peak_blocks(snapshot: dict, device: int = 0) -> tuple[int, dict[int, tuple[int, str]]]:
    """Replay alloc/free events; return (peak_bytes, {key: (size, site)} at peak).

    Blocks allocated before recording started (e.g. weights already on the
    GPU) only show up as frees, or not at all if they outlive the trace.
    They are counted as alive from the start until their free, so the peak
    and the live set include them (site "<before recording>").
    """
    trace = snapshot["device_traces"][device]

    # Pass 1: frees of blocks the trace never allocated -> pre-existing.
    live_sz: dict[int, int] = {}
    pre_freed_at: dict[int, int] = {}  # trace index -> size
    for i, ev in enumerate(trace):
        if ev["action"] == "alloc":
            live_sz[ev["addr"]] = ev["size"]
        elif ev["action"] == "free_completed":
            if ev["addr"] in live_sz:
                del live_sz[ev["addr"]]
            else:
                pre_freed_at[i] = ev["size"]
    pre_forever = [
        (blk["address"], blk["size"])
        for seg in snapshot["segments"] for blk in seg["blocks"]
        if blk["state"] == "active_allocated" and blk["address"] not in live_sz
    ]

    # Pass 2: total allocated over time, find the peak index.
    total = sum(pre_freed_at.values()) + sum(s for _, s in pre_forever)
    best_total, best_i = total, -1
    live_sz = {}
    for i, ev in enumerate(trace):
        if ev["action"] == "alloc":
            live_sz[ev["addr"]] = ev["size"]
            total += ev["size"]
        elif ev["action"] == "free_completed":
            if ev["addr"] in live_sz:
                total -= live_sz.pop(ev["addr"])
            else:
                total -= pre_freed_at.get(i, 0)
        if total > best_total:
            best_total, best_i = total, i

    # Pass 3: rebuild the live set at the peak, with allocation sites.
    live: dict[int, tuple[int, str]] = {}
    for ev in trace[:best_i + 1]:
        if ev["action"] == "alloc":
            live[ev["addr"]] = (ev["size"], site_of(ev.get("frames", [])))
        elif ev["action"] == "free_completed":
            live.pop(ev["addr"], None)
    for j, size in pre_freed_at.items():
        if j > best_i:
            live[-1 - j] = (size, "<before recording, freed later>")
    for addr, size in pre_forever:
        live[-(10**12) - addr] = (size, "<before recording, still alive>")
    return best_total, live


def by_site(live: dict[int, tuple[int, str]]) -> dict[str, tuple[int, int]]:
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for size, site in live.values():
        acc[site][0] += size
        acc[site][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", nargs="?")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    ap.add_argument("--frames", nargs=2, type=int, metavar=("TA", "TB"),
                    help="Window sizes of A and B, to report bytes per frame.")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    gib = 2**30

    if args.diff:
        (ta, tb) = args.frames or (0, 1)
        pa, la = peak_blocks(load(args.diff[0]))
        pb, lb = peak_blocks(load(args.diff[1]))
        sa, sb = by_site(la), by_site(lb)
        df = max(tb - ta, 1)
        print(f"peak A {pa / gib:.3f} GiB, peak B {pb / gib:.3f} GiB, "
              f"delta {(pb - pa) / gib:.3f} GiB = {(pb - pa) / df / 2**20:.1f} MiB/frame")
        rows = []
        for site in set(sa) | set(sb):
            d = sb.get(site, (0, 0))[0] - sa.get(site, (0, 0))[0]
            rows.append((d, site, sa.get(site, (0, 0)), sb.get(site, (0, 0))))
        rows.sort(key=lambda r: abs(r[0]), reverse=True)
        print(f"{'MiB/frame':>10} {'A MiB':>9} {'B MiB':>9}  site")
        for d, site, a, b in rows[:args.top]:
            print(f"{d / df / 2**20:10.1f} {a[0] / 2**20:9.1f} {b[0] / 2**20:9.1f}  {site}")
        return

    peak, live = peak_blocks(load(args.snapshot))
    sites = sorted(by_site(live).items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"peak allocated {peak / gib:.3f} GiB across {len(live)} blocks")
    print(f"{'MiB':>9} {'blocks':>7}  site")
    for site, (size, n) in sites[:args.top]:
        print(f"{size / 2**20:9.1f} {n:7d}  {site}")


if __name__ == "__main__":
    main()
