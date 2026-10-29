#!/usr/bin/env python3
"""How much does PyTorch's default caching allocator hold above the
allocated peak when nothing forces a release (the Windows situation)?

Runs the whole --optimized pipeline (perf/infer_with.py, no allocator cap,
expandable segments off) at fixed windows, low-VRAM and normal mode, and
records peak allocated vs peak reserved per run, next to the table's
estimate. The auto-VRAM margin for allocators without expandable segments
has to cover reserved/estimate for every configuration it can pick.

Usage:
    python perf/overhead_sweep.py --clips converge_1080 converge_2160 \
        --windows 8 24 48 --modes lowvram normal \
        --out perf/results/phase4/overhead_sweep.jsonl
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
sys.path.insert(0, str(ROOT))
PEAK_RE = re.compile(r"peak allocated ([\d.]+) GiB, peak reserved ([\d.]+) GiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[8, 24, 48])
    ap.add_argument("--modes", nargs="+", default=["lowvram", "normal"],
                    choices=("lowvram", "normal"))
    ap.add_argument("--sdpa", default="efficient")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from corridorkey import runtime as rt
    from corridorkey import video_io as vio
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False")
    work = ROOT / "perf/out/_sweep"
    for clip in args.clips:
        meta = vio.probe_frame_dir_meta(str(ROOT / "test_clips" / clip), num_frames=-1,
                                        shape_quantum=128, fps=30.0, exr_linear=False,
                                        model_scale=1.0)
        H, W = meta["shape_hw"]
        T_all = int(meta["num_frames"])
        for mode in args.modes:
            for w in args.windows:
                low = mode == "lowvram"
                est = rt.estimate_peak_vram_gb(H * W / 1e6, min(w, T_all), low,
                                               rt._VRAM_PEAK_GB_OPTIMIZED)
                tag = f"{clip}_{mode}_{w}"
                cmd = [sys.executable, str(ROOT / "perf/infer_with.py"), "--sdpa", args.sdpa,
                       "--mem_budget_gib", "47", "--uncapped", "--",
                       "--input", str(ROOT / "test_clips" / clip),
                       "--output_dir", str(work / tag), "--frame_dir_fps", "30",
                       "--temp_dir", str(work / "_tmp" / tag), "--optimized",
                       "--hann_chunk", str(w), "--hann_stride", str(max(1, w // 2))]
                if low:
                    cmd.append("--low_vram")
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                peak = PEAK_RE.search(proc.stdout + proc.stderr)
                row = {"clip": clip, "megapixels": H * W / 1e6, "mode": mode, "window": w,
                       "frames": T_all, "estimate_gib": est, "returncode": proc.returncode,
                       "peak_allocated_gib": float(peak.group(1)) if peak else None,
                       "peak_reserved_gib": float(peak.group(2)) if peak else None,
                       "wall_s": round(time.perf_counter() - t0, 1)}
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                ratio = (row["peak_reserved_gib"] / est) if peak else float("nan")
                print(f"{tag}: rc={proc.returncode} est={est:.2f} alloc={row['peak_allocated_gib']} "
                      f"reserved={row['peak_reserved_gib']} reserved/est={ratio:.2f} "
                      f"{row['wall_s']}s", flush=True)
                shutil.rmtree(work / tag, ignore_errors=True)
                shutil.rmtree(work / "_tmp" / tag, ignore_errors=True)


if __name__ == "__main__":
    main()
