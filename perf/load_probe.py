#!/usr/bin/env python3
"""Model load: does skipping random initialization (load_model(skip_init=True),
infer.py --fast_load) give the same model, and how much time does it save?

--verify builds the model twice the normal way and once with skip_init, on
the CPU, one at a time, and hashes every parameter and buffer (persistent or
not). Tensors that differ between the two normal builds are random and
unused by inference (there should be none); every other tensor must hash the
same with skip_init.

--time loads the model onto the GPU the way infer.py --optimized does
(load_model, then bf16 Linear/Conv weights), alternating normal and
skip_init, and prints the wall time of each.

Usage:
    python perf/load_probe.py --verify
    python perf/load_probe.py --time --rounds 2
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from corridorkey import runtime as rt


def tensor_hashes(model) -> dict[str, str]:
    out = {}
    named = list(model.named_parameters(remove_duplicate=False))
    named += list(model.named_buffers(remove_duplicate=False))
    for name, t in named:
        t = t.detach().cpu().contiguous()
        h = hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        h.update(f"{tuple(t.shape)}{t.dtype}".encode())
        out[name] = h.hexdigest()
    return out


def build(skip_init: bool, device: str):
    t0 = time.perf_counter()
    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device=device, skip_init=skip_init)
    secs = time.perf_counter() - t0
    return model, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = {}
    if args.verify:
        hashes = []
        for skip in (False, False, True):
            model, secs = build(skip, "cpu")
            hashes.append(tensor_hashes(model))
            print(f"built on cpu, skip_init={skip}: {secs:.1f}s, {len(hashes[-1])} tensors",
                  flush=True)
            del model
            gc.collect()
        a, b, c = hashes
        random = sorted(k for k in a if a[k] != b.get(k))
        differ = sorted(k for k in a if k not in random and a[k] != c.get(k))
        missing = sorted(set(a) ^ set(c))
        print(f"{len(a)} tensors: {len(random)} differ between two normal builds "
              f"{random[:5]}; with skip_init {len(differ)} differ {differ[:5]}, "
              f"{len(missing)} missing on one side")
        result["verify"] = {"tensors": len(a), "random_between_normal_builds": random,
                            "differ_with_skip_init": differ, "missing": missing}
    if args.time:
        times = {False: [], True: []}
        for r in range(args.rounds):
            for skip in ((False, True) if r % 2 == 0 else (True, False)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model, load_s = build(skip, "cuda")
                t1 = time.perf_counter()
                rt.apply_weight_policy(model, bf16_linear_conv=True)
                torch.cuda.synchronize()
                secs = time.perf_counter() - t0
                times[skip].append(secs)
                print(f"round {r + 1}: skip_init={skip} load {load_s:.2f}s + bf16 weights "
                      f"{time.perf_counter() - t1:.2f}s = {secs:.2f}s", flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        result["time"] = {"normal_s": times[False], "skip_init_s": times[True]}
        print(f"normal {min(times[False]):.2f}s, skip_init {min(times[True]):.2f}s (best of "
              f"{args.rounds})")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
