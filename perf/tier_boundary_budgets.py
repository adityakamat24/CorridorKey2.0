#!/usr/bin/env python3
"""Smallest budget at which --optimized auto VRAM picks each configuration.

For every window the selector can pick for a clip (resolution and length),
prints `clip budget` lines where budget = estimate x scale + extra GiB for
the given allocator margin, plus 0.005: the tightest budget that still
selects that window. Running the pipeline at exactly these budgets tests
the margin where it is thinnest.

Usage:
    python perf/tier_boundary_budgets.py --allocator default \
        converge_1080:1152x1920:48 converge_2160:2176x3840:48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corridorkey import runtime as rt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allocator", choices=("default", "expandable"), default="default")
    ap.add_argument("--max_budget", type=float, default=46.9)
    ap.add_argument("clips", nargs="+", help="name:HxW:frames (padded model size)")
    args = ap.parse_args()
    table = rt._VRAM_PEAK_GB_OPTIMIZED
    expandable = args.allocator == "expandable"
    for spec in args.clips:
        name, hw, frames = spec.split(":")
        H, W = (int(v) for v in hw.split("x"))
        frames = int(frames)
        margin = rt.optimized_vram_margin(H * W / 1e6, expandable)
        seen = set()
        for chunk in table[True][0]:
            for low_vram in (True, False):
                est = rt.estimate_peak_vram_gb(H * W / 1e6, min(chunk, frames), low_vram, table)
                scale, extra = margin[low_vram] if isinstance(margin, dict) else margin
                budget = round(est * scale + extra + 0.005, 3)
                if budget > args.max_budget:
                    continue
                pick = rt.select_vram_profile((H, W), frames, budget, table, margin)
                key = (min(pick[0], frames), pick[2])
                if key in seen or (min(pick[0], frames), pick[2]) != (min(chunk, frames), low_vram):
                    continue
                seen.add(key)
                print(f"{name} {budget}")


if __name__ == "__main__":
    main()
