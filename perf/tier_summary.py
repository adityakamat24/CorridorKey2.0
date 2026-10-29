#!/usr/bin/env python3
"""Markdown summary of perf/tier_check.py results.

One row per (budget, clip): the window each variant's auto VRAM picked
(L = low-VRAM mode, N = normal mode), peak allocated / reserved GiB, wall
seconds (model load included) and out-of-memory failures.

Usage:
    python perf/tier_summary.py perf/results/phase3/tiers_rtx3060.jsonl \
        perf/results/phase3/tiers_rtx6000ada.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

VARIANTS = (("shipped", "default"), ("optimized", "default"), ("optimized", "expandable"))
LABELS = {("shipped", "default"): "shipped",
          ("optimized", "default"): "optimized, default allocator (Windows)",
          ("optimized", "expandable"): "optimized, expandable segments (Linux)"}


def cell(r: dict | None) -> str:
    if r is None:
        return ""
    if r["hann_chunk"] is None:
        return f"failed rc={r['returncode']}"
    mode = "L" if r["low_vram"] else "N"
    head = f"{r['hann_chunk']}{mode}"
    if r["oom"]:
        return f"**{head} OOM**"
    return (f"{head}: {r['peak_allocated_gib']:.2f} / {r['peak_reserved_gib']:.2f} GiB, "
            f"{r['wall_s']:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    args = ap.parse_args()
    rows = defaultdict(dict)
    for path in args.jsonl:
        for line in open(path):
            r = json.loads(line)
            rows[(r["budget_gib"], r["clip"])][(r["variant"], r["allocator"])] = r
    print("| budget GiB | clip | " + " | ".join(LABELS[v] for v in VARIANTS) + " |")
    print("|---|---|" + "---|" * len(VARIANTS))
    for (budget, clip) in sorted(rows):
        got = rows[(budget, clip)]
        print(f"| {budget:g} | {clip} | " + " | ".join(cell(got.get(v)) for v in VARIANTS) + " |")
    ooms = [r for got in rows.values() for r in got.values() if r["oom"]]
    print(f"\n{sum(len(g) for g in rows.values())} runs, {len(ooms)} out of memory"
          + (": " + ", ".join(f"{r['variant']}/{r['allocator']} {r['clip']} @ {r['budget_gib']:g}"
                              for r in ooms) if ooms else ""))


if __name__ == "__main__":
    main()
