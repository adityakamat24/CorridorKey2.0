#!/usr/bin/env python3
"""Why does the default caching allocator reserve more than it allocates?

Reads a snapshot written by `perf/bench.py --mem_snapshot`, replays the
device trace (alloc, free_completed, segment_alloc, segment_free) and
reports:

  * the moment reserved memory peaks: reserved, allocated, and every
    segment alive then (size, bytes in use, largest free gap), grouped by
    the allocation site that created the segment;
  * every segment the run allocated while the large pool already held
    enough free bytes for the request, just not in one piece
    (fragmentation growth), summed by the site of the request.

Without expandable segments one cudaMalloc is one segment, and a free gap
can only serve a request that fits inside it, so a cache full of gaps
that are each too small grows the pool even when the total would do.

Usage:
    python perf/frag_analyze.py perf/results/phase4/snap_1080_24L.pickle
"""

from __future__ import annotations

import argparse
import bisect
import pickle
from collections import defaultdict

from mem_peak import site_of

SMALL_SEGMENT = 2 << 20  # small-pool segments (requests up to 1 MiB) are 2 MiB
MIB = 2**20


def replay(snapshot: dict, device: int = 0):
    trace = snapshot["device_traces"][device]
    # Segments and blocks that existed before recording started show up only
    # as frees, or in the end-of-run segment list.
    seen_seg, seen_blk = set(), set()
    pre_segs, pre_blks = {}, {}
    for ev in trace:
        a = ev["action"]
        if a == "segment_alloc":
            seen_seg.add(ev["addr"])
        elif a == "segment_free" and ev["addr"] not in seen_seg:
            pre_segs[ev["addr"]] = ev["size"]
        elif a == "alloc":
            seen_blk.add(ev["addr"])
        elif a == "free_completed" and ev["addr"] not in seen_blk:
            pre_blks[ev["addr"]] = ev["size"]
    for seg in snapshot["segments"]:
        if seg["address"] not in seen_seg:
            pre_segs.setdefault(seg["address"], seg["total_size"])
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated" and blk["address"] not in seen_blk:
                pre_blks.setdefault(blk["address"], blk["size"])
    return trace, pre_segs, pre_blks


class State:
    def __init__(self, segs: dict, blks: dict):
        self.segs = dict(segs)            # base -> size
        self.seg_site = {b: "<before recording>" for b in segs}
        self.blks = dict(blks)            # addr -> size
        self.reserved = sum(segs.values())
        self.allocated = sum(blks.values())

    def layout(self):
        """Per segment: (size, used bytes, largest free gap, site)."""
        bases = sorted(self.segs)
        per = defaultdict(list)
        for addr, size in self.blks.items():
            i = bisect.bisect_right(bases, addr) - 1
            if i >= 0 and addr < bases[i] + self.segs[bases[i]]:
                per[bases[i]].append((addr, size))
        out = {}
        for b in bases:
            size = self.segs[b]
            cur, used, gap = b, 0, 0
            for addr, s in sorted(per.get(b, [])):
                gap = max(gap, addr - cur)
                cur = addr + s
                used += s
            gap = max(gap, b + size - cur)
            out[b] = (size, used, gap, self.seg_site.get(b, "?"))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)
    trace, pre_segs, pre_blks = replay(snap)
    st = State(pre_segs, pre_blks)

    peak_res, peak_i, peak_alloc_overall = st.reserved, -1, st.allocated
    growth = []  # (index, seg size, request size, free large, largest gap, site)
    pending = None
    for i, ev in enumerate(trace):
        a = ev["action"]
        if a == "segment_alloc":
            if ev["size"] != SMALL_SEGMENT:
                lay = st.layout()
                free_large = sum(s - u for s, u, _, _ in lay.values() if s != SMALL_SEGMENT)
                gap = max((g for s, _, g, _ in lay.values() if s != SMALL_SEGMENT), default=0)
                pending = [i, ev["size"], None, free_large, gap, site_of(ev.get("frames", []))]
            st.segs[ev["addr"]] = ev["size"]
            st.seg_site[ev["addr"]] = site_of(ev.get("frames", []))
            st.reserved += ev["size"]
        elif a == "segment_free":
            st.reserved -= st.segs.pop(ev["addr"], ev["size"])
        elif a == "alloc":
            st.blks[ev["addr"]] = ev["size"]
            st.allocated += ev["size"]
            if pending is not None:
                pending[2] = ev["size"]
                growth.append(tuple(pending))
                pending = None
        elif a == "free_completed":
            st.allocated -= st.blks.pop(ev["addr"], 0)
        peak_alloc_overall = max(peak_alloc_overall, st.allocated)
        if st.reserved > peak_res:
            peak_res, peak_i = st.reserved, i

    # Rebuild the state at the reserved peak.
    st2 = State(pre_segs, pre_blks)
    for ev in trace[:peak_i + 1]:
        a = ev["action"]
        if a == "segment_alloc":
            st2.segs[ev["addr"]] = ev["size"]
            st2.seg_site[ev["addr"]] = site_of(ev.get("frames", []))
            st2.reserved += ev["size"]
        elif a == "segment_free":
            st2.reserved -= st2.segs.pop(ev["addr"], ev["size"])
        elif a == "alloc":
            st2.blks[ev["addr"]] = ev["size"]
            st2.allocated += ev["size"]
        elif a == "free_completed":
            st2.allocated -= st2.blks.pop(ev["addr"], 0)
    lay = st2.layout()
    gib = 2**30
    print(f"{len(trace)} trace events; peak reserved {peak_res / gib:.3f} GiB at event "
          f"{peak_i}, allocated then {st2.allocated / gib:.3f} GiB; peak allocated in the "
          f"run {peak_alloc_overall / gib:.3f} GiB")
    small = [v for v in lay.values() if v[0] == SMALL_SEGMENT]
    large = [v for v in lay.values() if v[0] != SMALL_SEGMENT]
    print(f"small pool: {len(small)} segments, {sum(v[0] for v in small) / MIB:.0f} MiB, "
          f"{sum(v[1] for v in small) / MIB:.0f} MiB in use")
    print(f"large pool: {len(large)} segments, {sum(v[0] for v in large) / MIB:.0f} MiB, "
          f"{sum(v[1] for v in large) / MIB:.0f} MiB in use, free "
          f"{sum(v[0] - v[1] for v in large) / MIB:.0f} MiB, largest gap "
          f"{max((v[2] for v in large), default=0) / MIB:.0f} MiB")

    print(f"\nlarge segments alive at the reserved peak, by creating site (top {args.top}):")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for size, used, gap, site in large:
        r = by[site]
        r[0] += 1; r[1] += size; r[2] += used; r[3] = max(r[3], gap)
    print(f"{'segs':>5} {'MiB':>8} {'used':>8} {'free':>8} {'gap':>7}  site")
    for site, (n, size, used, gap) in sorted(by.items(), key=lambda kv: -(kv[1][1] - kv[1][2]))[:args.top]:
        print(f"{n:5d} {size / MIB:8.0f} {used / MIB:8.0f} {(size - used) / MIB:8.0f} "
              f"{gap / MIB:7.0f}  {site}")

    print("\nlarge segments the run allocated while enough was cached but split up:")
    frag = [g for g in growth if g[2] is not None and g[3] >= g[2]]
    by2 = defaultdict(lambda: [0, 0, 0.0])
    for _i, seg, req, free, gap, site in frag:
        r = by2[site]
        r[0] += 1; r[1] += seg; r[2] = max(r[2], free)
    print(f"{len(frag)} of {len(growth)} large segment allocations; "
          f"{sum(g[1] for g in frag) / MIB:.0f} MiB of {sum(g[1] for g in growth) / MIB:.0f} MiB")
    print(f"{'n':>4} {'MiB':>8} {'max free then':>14}  site")
    for site, (n, seg, free) in sorted(by2.items(), key=lambda kv: -kv[1][1])[:args.top]:
        print(f"{n:4d} {seg / MIB:8.0f} {free / MIB:14.0f}  {site}")
    print("\nfirst 30 large segment allocations (request, segment, free large, largest gap):")
    for i, seg, req, free, gap, site in growth[:30]:
        print(f"  ev {i:8d} req {(req or 0) / MIB:8.1f} seg {seg / MIB:8.1f} free {free / MIB:8.1f} "
              f"gap {gap / MIB:8.1f}  {site}")


if __name__ == "__main__":
    main()
