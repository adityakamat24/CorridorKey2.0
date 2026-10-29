#!/usr/bin/env python3
"""Does the low-VRAM foundation prepass give the same features at a larger
frame chunk, and how much faster is it?

Runs runtime.precompute_foundations_to_memmap on the same clip with each
--chunks value, then compares the C-RADIO and MoGe feature memmaps bitwise
against the first chunk size and reports wall time and peak allocated
memory per chunk size.

Usage:
    python perf/foundation_chunk_probe.py --input test_clips/converge_1080 \
        --chunks 4 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from corridorkey import runtime as rt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--chunks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--num_frames", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device="cuda")
    rt.apply_weight_policy(model, bf16_linear_conv=True)
    model.stage1.cpu()
    model.stage2.cpu()
    model.foundations.cuda()
    torch.cuda.empty_cache()
    meta = rt.probe_frame_dir_meta(args.input, num_frames=args.num_frames, shape_quantum=128,
                                   fps=30.0, exr_linear=False, model_scale=1.0)
    tmp = Path(tempfile.mkdtemp(prefix="fchunk_"))
    rows, ref = [], None
    for chunk in args.chunks:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        paths = rt.precompute_foundations_to_memmap(
            model, args.input, meta, tmp / f"c{chunk}", chunk=chunk, reader_kind="frame_dir")
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        feats = {k: np.array(np.memmap(paths[k], dtype=np.float32, mode="r",
                                        shape=tuple(paths[k + "_shape"])))
                 for k in ("moge", "cradio") if k in paths}
        row = {"chunk": chunk, "s": secs,
               "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        if ref is None:
            ref = feats
        else:
            row["bitwise_equal"] = all(np.array_equal(ref[k], feats[k]) for k in ref)
            row["max_abs_diff"] = max(float(np.abs(ref[k] - feats[k]).max()) for k in ref)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
