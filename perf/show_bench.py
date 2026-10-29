#!/usr/bin/env python3
"""Print perf/bench.py JSON results as compact tables.

Usage:
    python perf/show_bench.py perf/results/phase2/bench_*.json
    python perf/show_bench.py --phases perf/results/phase2/bench_1080_8x4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--phases", action="store_true", help="Per-phase table for each file.")
    args = ap.parse_args()

    print(f"{'config':<28}{'frames':>7}{'total s':>9}{'s/frame':>9}{'peak alloc':>11}"
          f"{'peak res':>10}{'table':>7}  {'peak phase':<18}{'s1 win ms':>10}{'s2 ms/f':>9}")
    for path in args.files:
        r = json.loads(Path(path).read_text())
        s = r["summary"]
        label = r.get("label") or Path(path).stem
        if not s.get("measured_runs"):
            oom = s.get("oom") or {}
            print(f"{label:<28}  OOM in {oom.get('phase')}: {str(oom.get('message'))[:70]}")
            continue
        frames = max(r["runs"][-1]["frames"], 1)
        print(f"{label:<28}{frames:>7}{s['total_s_median']:>9.2f}"
              f"{s['total_s_median'] / frames:>9.3f}{s['peak_allocated_gib']:>11.3f}"
              f"{s['peak_reserved_gib']:>10.3f}{fmt(r.get('vram_table_estimate_gib')):>7}  "
              f"{s['peak_owner_phase']:<18}{fmt(s.get('stage1_window_ms_median'), 0):>10}"
              f"{fmt(s.get('stage2_forward_ms_per_frame'), 1):>9}")
        if args.phases:
            for name, p in s["phases"].items():
                print(f"    {name:<18}{p['wall_s']:>8.2f}s  alloc {p['peak_allocated_gib']:6.3f}"
                      f"  res {p['peak_reserved_gib']:6.3f}  nvml {fmt(p.get('nvml_mem_used_max_gib'), 3)}"
                      f"  sm {fmt(p.get('sm_mhz_median'), 0)} MHz")


if __name__ == "__main__":
    main()
