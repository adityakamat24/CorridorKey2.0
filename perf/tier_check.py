#!/usr/bin/env python3
"""End-to-end check of the auto VRAM selector on emulated GPU sizes.

For every (budget, clip) pair, runs infer.py in a fresh process through
perf/infer_with.py --mem_budget_gib: the caching allocator is capped at the
budget and torch.cuda.mem_get_info reports it, so the runtime's auto VRAM
picks the Hann window it would pick on a card that leaves that much memory
to PyTorch. Records the window it picked, the peak allocated and reserved
memory, the wall time, and out-of-memory failures.

Budgets are GiB left to PyTorch, not the card's nominal size. A 6 GB RTX 3060
Laptop on Windows has 4.95 GiB free after the CUDA context and display; the
default budgets assume the same 1.05 GiB overhead on every card:
  6 GB -> 4.95, 8 GB -> 6.95, 12 GB -> 10.95, 16 GB -> 14.95,
  24 GB -> 22.95, 48 GB -> 46.95

--allocator default keeps PyTorch's default caching allocator (what Windows
builds always use); --allocator expandable sets expandable_segments:True.

Usage:
    python perf/tier_check.py --variant optimized --allocator default \
        --clips converge_1080 converge_2160 --out perf/results/phase3/tiers.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [4.95, 6.95, 10.95, 14.95, 22.95, 46.95]
AUTO_RE = re.compile(r"Auto VRAM: ([\d.]+) GB available at (\d+)x(\d+) -> hann_chunk=(\d+), "
                     r"hann_stride=(\d+), low_vram=(\w+) \(estimated peak ([\d.]+) GB\)")
PEAK_RE = re.compile(r"peak allocated ([\d.]+) GiB, peak reserved ([\d.]+) GiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("shipped", "optimized"), required=True)
    ap.add_argument("--allocator", choices=("default", "expandable"), default="default")
    ap.add_argument("--uncapped", action="store_true",
                    help="Report the budget to auto VRAM without capping the allocator; "
                         "the row's `fits` says whether peak reserved stayed under it.")
    ap.add_argument("--alloc_conf", default=None,
                    help="Exact PYTORCH_CUDA_ALLOC_CONF for the run (overrides --allocator's).")
    ap.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    ap.add_argument("--clips", nargs="+", required=True,
                    help="Folder names under test_clips/.")
    ap.add_argument("--sdpa", default="efficient")
    ap.add_argument("--extra", default="", help="Extra infer.py arguments.")
    ap.add_argument("--out", required=True, help="Append-only JSONL results.")
    args = ap.parse_args()

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = args.alloc_conf or (
        "expandable_segments:True" if args.allocator == "expandable" else "expandable_segments:False")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / "perf/out/_tiers"
    for clip in args.clips:
        for budget in args.budgets:
            conf_tag = "" if args.alloc_conf is None else "_" + "".join(
                c if c.isalnum() else "-" for c in args.alloc_conf)
            tag = (f"{args.variant}_{args.allocator}{conf_tag}"
                   f"{'_uncapped' if args.uncapped else ''}_{clip}_{budget:g}")
            out_dir = work / tag
            cmd = [sys.executable, str(ROOT / "perf/infer_with.py"),
                   "--sdpa", args.sdpa, "--mem_budget_gib", str(budget)]
            if args.uncapped:
                cmd.append("--uncapped")
            cmd += ["--",
                   "--input", str(ROOT / "test_clips" / clip), "--output_dir", str(out_dir),
                   "--frame_dir_fps", "30", "--temp_dir", str(work / "_tmp" / tag)]
            if args.variant == "optimized":
                cmd.append("--optimized")
            cmd += args.extra.split()
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            log = proc.stdout + proc.stderr
            (work / f"{tag}.log").write_text(log)
            auto = AUTO_RE.search(log)
            peak = PEAK_RE.search(log)
            row = {
                "variant": args.variant, "allocator": args.allocator,
                "alloc_conf": env["PYTORCH_CUDA_ALLOC_CONF"], "clip": clip,
                "budget_gib": budget, "uncapped": args.uncapped,
                "returncode": proc.returncode,
                "oom": proc.returncode == 3, "wall_s": round(wall, 1),
                "hann_chunk": int(auto.group(4)) if auto else None,
                "hann_stride": int(auto.group(5)) if auto else None,
                "low_vram": auto.group(6) == "True" if auto else None,
                "estimate_gb": float(auto.group(7)) if auto else None,
                "peak_allocated_gib": float(peak.group(1)) if peak else None,
                "peak_reserved_gib": float(peak.group(2)) if peak else None,
                "cmd": " ".join(cmd),
            }
            row["fits"] = (row["peak_reserved_gib"] is not None and not row["oom"]
                           and row["peak_reserved_gib"] <= budget)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{tag}: rc={proc.returncode} window={row['hann_chunk']}/{row['hann_stride']} "
                  f"low_vram={row['low_vram']} est={row['estimate_gb']} "
                  f"alloc={row['peak_allocated_gib']} reserved={row['peak_reserved_gib']} "
                  f"fits={row['fits']} {wall:.0f}s", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(work / "_tmp" / tag, ignore_errors=True)


if __name__ == "__main__":
    main()
