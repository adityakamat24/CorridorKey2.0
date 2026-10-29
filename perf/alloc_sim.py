#!/usr/bin/env python3
"""Replay a recorded allocation trace through a model of PyTorch's CUDA
caching allocator (expandable segments off, no memory fraction), to try
allocation strategies offline instead of one GPU run each.

The model follows c10/cuda/CUDACachingAllocator.cpp: requests round up to
512 bytes; up to 1 MiB they come from the small pool (2 MiB segments),
above it from the large pool (20 MiB segments under 10 MiB, else the
request rounded up to 2 MiB); a request takes the smallest free block of
its pool that fits (lowest address on ties) and splits it when more than
1 MiB (large pool) or 512 bytes (small pool) would remain, the allocated
part first; a freed block merges with free neighbours in its segment; a
segment is returned to the driver only by empty_cache, and only when it is
one free block. Each burst of segment_free events in the trace is taken as
an empty_cache call. Ties between equal-size free blocks go to the lower
address, so segment addresses matter: while the simulated run creates the
same segments as the recorded one it reuses their recorded addresses, and
after it diverges it places new segments below every address so far, as
the driver does for fresh address space.

Input: a snapshot from `perf/bench.py --mem_snapshot`. Prints the recorded
peak reserved next to the simulated one, then for each strategy given:

  --arena GIB [GIB ...]   reserve one free segment of GIB GiB right before
                          the first stage-1 allocation (what
                          perf/arena_probe.py does on the GPU)
  --roundup N             roundup_power2_divisions:N for every size
  --drop SITE:MIB [...]   remove every allocation (and its free) made at a
                          site (substring of the deepest corridorkey frame,
                          e.g. blocks.py:3379) of at least MIB MiB, to see
                          what a code change that avoids it would save
  --estimate GIB --factor F [F ...]
                          the arena sized the way the runtime would size it:
                          F x (estimated peak - allocated at that point)

Usage:
    python perf/alloc_sim.py snap.pickle --arena 1.9 2.0 2.2
"""

from __future__ import annotations

import argparse
import bisect
import pickle

KB512 = 512
SMALL = 1 << 20          # kSmallSize: requests up to 1 MiB use the small pool
SMALL_BUF = 2 << 20      # kSmallBuffer
LARGE_BUF = 20 << 20     # kLargeBuffer
MIN_LARGE = 10 << 20     # kMinLargeAlloc
ROUND_LARGE = 2 << 20    # kRoundLarge
GIB = 2**30


def round_size(size: int, divisions: int = 0) -> int:
    if size < KB512:
        return KB512
    if divisions > 1 and size > KB512 * divisions:
        p = 1 << (size - 1).bit_length()          # next power of two >= size
        lo = p >> 1
        step = lo // divisions
        if step == 0:
            return p
        return min(p, lo + -(-(size - lo) // step) * step)
    return -(-size // KB512) * KB512


def allocation_size(size: int) -> int:
    if size <= SMALL:
        return SMALL_BUF
    if size < MIN_LARGE:
        return LARGE_BUF
    return -(-size // ROUND_LARGE) * ROUND_LARGE


class Block:
    __slots__ = ("addr", "size", "seg", "free", "prev", "next")

    def __init__(self, addr, size, seg, free):
        self.addr, self.size, self.seg, self.free = addr, size, seg, free
        self.prev = self.next = None


class Allocator:
    def __init__(self, divisions: int = 0, real_segments=()):
        self.divisions = divisions
        self.free_sets = {True: [], False: []}   # small? -> sorted [(size, addr)]
        self.by_addr: dict[int, Block] = {}      # block start -> Block (free or not)
        self.segments: dict[int, tuple[int, bool]] = {}  # base -> (size, small?)
        self.reserved = 0
        self.allocated = 0
        self.peak_reserved = 0
        self.real_segments = list(real_segments)[::-1]  # pop() takes the next one
        self.lowest = min((a for a, _ in real_segments), default=1 << 48)

    # -- free-set bookkeeping
    def _add_free(self, b: Block):
        bisect.insort(self.free_sets[self.segments[b.seg][1]], (b.size, b.addr))

    def _del_free(self, b: Block):
        fs = self.free_sets[self.segments[b.seg][1]]
        i = bisect.bisect_left(fs, (b.size, b.addr))
        assert fs[i] == (b.size, b.addr)
        fs.pop(i)

    def add_segment(self, base: int, size: int, small: bool, live: list[tuple[int, int]]):
        """A segment that exists at trace start, with its live blocks."""
        self.segments[base] = (size, small)
        self.reserved += size
        cur, prev = base, None
        pieces = []
        for addr, s in sorted(live):
            if addr > cur:
                pieces.append((cur, addr - cur, True))
            pieces.append((addr, s, False))
            cur = addr + s
        if cur < base + size:
            pieces.append((cur, base + size - cur, True))
        for addr, s, free in pieces:
            b = Block(addr, s, base, free)
            b.prev = prev
            if prev is not None:
                prev.next = b
            prev = b
            self.by_addr[addr] = b
            if free:
                self._add_free(b)
            else:
                self.allocated += s
        self.peak_reserved = max(self.peak_reserved, self.reserved)

    def _new_segment(self, size: int, small: bool) -> Block:
        if self.real_segments and self.real_segments[-1][1] == size:
            base = self.real_segments.pop()[0]
        else:
            self.real_segments = []  # diverged from the recording
            base = self.lowest - size - (2 << 20)
        self.lowest = min(self.lowest, base)
        self.segments[base] = (size, small)
        self.reserved += size
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        b = Block(base, size, base, True)
        self.by_addr[base] = b
        return b

    def malloc(self, req: int) -> Block:
        size = round_size(req, self.divisions)
        small = size <= SMALL
        fs = self.free_sets[small]
        i = bisect.bisect_left(fs, (size, -1))
        if i < len(fs):
            bsize, baddr = fs.pop(i)
            b = self.by_addr[baddr]
        else:
            b = self._new_segment(allocation_size(size), small)
        remaining = b.size - size
        if (remaining >= KB512) if small else (remaining > SMALL):
            rest = Block(b.addr + size, remaining, b.seg, True)
            rest.prev, rest.next = b, b.next
            if b.next is not None:
                b.next.prev = rest
            b.next = rest
            b.size = size
            self.by_addr[rest.addr] = rest
            self._add_free(rest)
        b.free = False
        self.allocated += b.size
        return b

    def free(self, b: Block):
        self.allocated -= b.size
        b.free = True
        for nb in (b.prev, b.next):
            if nb is not None and nb.free:
                self._del_free(nb)
                if nb is b.prev:
                    nb.size += b.size
                    nb.next = b.next
                    if b.next is not None:
                        b.next.prev = nb
                    del self.by_addr[b.addr]
                    b = nb
                else:
                    b.size += nb.size
                    b.next = nb.next
                    if nb.next is not None:
                        nb.next.prev = b
                    del self.by_addr[nb.addr]
        self._add_free(b)

    def empty_cache(self):
        for base, (size, small) in list(self.segments.items()):
            b = self.by_addr.get(base)
            if b is not None and b.free and b.size == size:
                self._del_free(b)
                del self.by_addr[base]
                del self.segments[base]
                self.reserved -= size


def initial_state(snapshot: dict, trace: list) -> tuple[dict, dict]:
    """Segments and live blocks at trace start (they appear only as frees,
    or in the end-of-run segment list)."""
    seen_seg, seen_blk = set(), set()
    segs, blks = {}, {}
    for ev in trace:
        a = ev["action"]
        if a == "segment_alloc":
            seen_seg.add(ev["addr"])
        elif a == "segment_free" and ev["addr"] not in seen_seg:
            segs[ev["addr"]] = ev["size"]
        elif a == "alloc":
            seen_blk.add(ev["addr"])
        elif a == "free_completed" and ev["addr"] not in seen_blk:
            blks[ev["addr"]] = ev["size"]
    for seg in snapshot["segments"]:
        if seg["address"] not in seen_seg:
            segs.setdefault(seg["address"], seg["total_size"])
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated" and blk["address"] not in seen_blk:
                blks.setdefault(blk["address"], blk["size"])
    return segs, blks


def first_index(trace: list, needle: str) -> int:
    for i, ev in enumerate(trace):
        if ev["action"] == "alloc":
            for fr in ev.get("frames", []):
                if needle in fr.get("name", ""):
                    return i
    return -1


def drop_sites(trace: list, drops: list[tuple[str, int]]) -> list:
    """The trace without the allocations at the given sites and sizes."""
    from mem_peak import site_of
    gone, out = set(), []
    for ev in trace:
        a = ev["action"]
        if a == "alloc":
            site = site_of(ev.get("frames", []))
            if any(s in site and ev["size"] >= n for s, n in drops):
                gone.add(ev["addr"])
                continue
        elif a in ("free_requested", "free_completed") and ev["addr"] in gone:
            if a == "free_completed":
                gone.discard(ev["addr"])
            continue
        out.append(ev)
    return out


def simulate(snapshot: dict, *, arena_gib: float = 0.0, divisions: int = 0,
             arena_at: str = "hann_stage1_stream", estimate_gib: float = 0.0,
             factor: float = 0.0, drops=()) -> dict:
    trace = snapshot["device_traces"][0]
    if drops:
        trace = drop_sites(trace, list(drops))
    segs, blks = initial_state(snapshot, trace)
    al = Allocator(divisions, [(ev["addr"], ev["size"]) for ev in trace
                               if ev["action"] == "segment_alloc"])
    bases = sorted(segs)
    live_by_seg = {b: [] for b in bases}
    for addr, size in blks.items():
        i = bisect.bisect_right(bases, addr) - 1
        if i >= 0 and addr < bases[i] + segs[bases[i]]:
            live_by_seg[bases[i]].append((addr, round_size(size)))
    for base in bases:
        al.add_segment(base, segs[base], segs[base] == SMALL_BUF, live_by_seg[base])
    pre = {addr: al.by_addr[addr] for _seg in live_by_seg.values() for addr, _ in _seg
           if addr in al.by_addr}
    arena_i = first_index(trace, arena_at) if (arena_gib > 0 or factor > 0) else -1
    arena_used = 0
    live: dict[int, Block] = dict(pre)
    in_free_burst = False
    peak_alloc = al.allocated
    stage_peaks = {}
    for i, ev in enumerate(trace):
        a = ev["action"]
        if i == arena_i:
            al.empty_cache()
            nbytes = int(arena_gib * GIB)
            if factor > 0:
                nbytes = int(factor * (estimate_gib * GIB - al.allocated))
            arena_used = nbytes
            if nbytes > 0:
                al.free(al.malloc(nbytes))
        if a == "segment_free":
            if not in_free_burst:
                al.empty_cache()
            in_free_burst = True
            continue
        if a in ("alloc", "free_completed"):
            in_free_burst = False
        if a == "alloc":
            live[ev["addr"]] = al.malloc(ev["size"])
            peak_alloc = max(peak_alloc, al.allocated)
        elif a == "free_completed":
            b = live.pop(ev["addr"], None)
            if b is not None:
                al.free(b)
    return {"peak_reserved": al.peak_reserved, "peak_allocated": peak_alloc,
            "arena": arena_used}


def recorded_peak(snapshot: dict) -> float:
    trace = snapshot["device_traces"][0]
    segs, _ = initial_state(snapshot, trace)
    res = peak = sum(segs.values())
    known = dict(segs)
    for ev in trace:
        if ev["action"] == "segment_alloc":
            known[ev["addr"]] = ev["size"]
            res += ev["size"]
        elif ev["action"] == "segment_free":
            res -= known.pop(ev["addr"], ev["size"])
        peak = max(peak, res)
    return peak


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("--arena", type=float, nargs="*", default=[])
    ap.add_argument("--roundup", type=int, nargs="*", default=[])
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--estimate", type=float, default=0.0)
    ap.add_argument("--factor", type=float, nargs="*", default=[])
    args = ap.parse_args()
    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)
    print(f"recorded peak reserved {recorded_peak(snap) / GIB:.3f} GiB")
    drops = [(d.rsplit(":", 1)[0], int(float(d.rsplit(":", 1)[1]) * 2**20)) for d in args.drop]
    r = simulate(snap)
    if drops:
        print(f"simulated as recorded: peak reserved {r['peak_reserved'] / GIB:.3f} GiB, "
              f"peak allocated {r['peak_allocated'] / GIB:.3f} GiB")
        r = simulate(snap, drops=drops)
        print(f"without {args.drop}:")
    print(f"simulated: peak reserved {r['peak_reserved'] / GIB:.3f} GiB, "
          f"peak allocated {r['peak_allocated'] / GIB:.3f} GiB")
    for g in args.arena:
        r = simulate(snap, arena_gib=g, drops=drops)
        print(f"arena {g:.2f} GiB: peak reserved {r['peak_reserved'] / GIB:.3f} GiB")
    for f in args.factor:
        r = simulate(snap, estimate_gib=args.estimate, factor=f, drops=drops)
        print(f"arena {f:.2f} x (estimate {args.estimate:.2f} - allocated) = "
              f"{r['arena'] / GIB:.3f} GiB: peak reserved {r['peak_reserved'] / GIB:.3f} GiB "
              f"({r['peak_reserved'] / r['peak_allocated']:.3f}x allocated)")
    for d in args.roundup:
        r = simulate(snap, divisions=d, drops=drops)
        print(f"roundup_power2_divisions {d}: peak reserved {r['peak_reserved'] / GIB:.3f} GiB")


if __name__ == "__main__":
    main()
