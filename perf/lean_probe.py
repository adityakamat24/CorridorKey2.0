#!/usr/bin/env python3
"""Stage-1 peak memory and bitwise output check for lean-path candidates.

Loads the model with the --optimized weight policy (bf16 Linear/Conv weights,
frozen), the runtime's low-VRAM chunking and --lean_activations, puts only
stage 1 on the GPU (as in the low-VRAM stage-1 phase), then runs stage-1
forwards on the same random inputs for the baseline and for each candidate.
A candidate is a set of module attributes, applied to every module of the
named class for its run and reset afterwards. Prints the peak allocated
memory and time (one warm-up run, then one timed run) per window and
whether baton and aux are bitwise equal to the baseline.

Usage:
    python perf/lean_probe.py --hw 1152 1920 --frames 8 16 \
        --candidate qkv1:SwinAttention3D.inference_qkv_frame_chunk=1
"""

from __future__ import annotations

import argparse
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
from floor_probe import make_inputs

GIB = 2**30
_MISSING = object()


def parse_candidate(text: str):
    name, _, spec = text.partition(":")
    assigns = []
    for item in spec.split(","):
        target, _, value = item.partition("=")
        cls, _, attr = target.partition(".")
        assigns.append((cls, attr, json.loads(value)))
    return name, assigns


def apply(model, assigns):
    saved = []
    for cls, attr, value in assigns:
        for m in model.modules():
            if m.__class__.__name__ == cls:
                saved.append((m, attr, getattr(m, attr, _MISSING)))
                setattr(m, attr, value)
    return saved


def restore(saved):
    for m, attr, old in reversed(saved):
        if old is _MISSING:
            delattr(m, attr)
        else:
            setattr(m, attr, old)


@torch.no_grad()
def run(stage1, inputs_cpu: dict, handover: bool = True):
    """One stage-1 forward on fresh GPU copies of the inputs. With
    `handover` and lean mode on, stage 1 gets them in a dict it empties,
    as the runtime does, so it can free them once consumed."""
    gpu = {k: (v.cuda() if v is not None else None) for k, v in inputs_cpu.items()}
    hq = gpu.pop("hint_quality")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if handover and getattr(stage1, "inference_lean", False):
            out = stage1(inputs=gpu, t_stride=1.0, hint_quality=hq)
        else:
            out = stage1(gpu["rgb"], gpu["hint"], gpu["moge_feat"],
                         cradio_feat=gpu["cradio_feat"], rvm_feat=gpu["rvm_feat"],
                         t_stride=1.0, hint_quality=hq)
    del gpu
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {"baton": out["baton"].float(), "aux": out["aux"].float()}
    del out
    return result, base, peak


def timed_run(stage1, inputs, handover=True):
    t0 = time.perf_counter()
    result = run(stage1, inputs, handover)
    return result, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs=2, default=[1152, 1920])
    ap.add_argument("--frames", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--candidate", action="append", default=[],
                    help="name:Class.attr=json[,Class.attr=json]")
    ap.add_argument("--no_handover", action="store_true",
                    help="Call stage 1 with positional inputs (the caller keeps them).")
    ap.add_argument("--sdpa", default="default",
                    choices=("default", "flash", "efficient", "cudnn", "math"),
                    help="Restrict SDPA to one backend (perf/infer_with.py).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from infer_with import sdpa_context
    with sdpa_context(args.sdpa):
        probe(args)


def probe(args):
    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device="cuda")
    rt.apply_weight_policy(model, bf16_linear_conv=True)
    rt.configure_low_vram_inference(model, rt.InferenceConfig())
    rt.configure_lean_activations(model, True)
    model.foundations.cpu()
    model.stage2.cpu()
    model.stage1.cuda()  # as the runtime does: modules shared with stage 1 come back
    torch.cuda.empty_cache()
    candidates = [parse_candidate(c) for c in args.candidate]
    H, W = args.hw
    rows = []
    for T in args.frames:
        inputs = {k: (v.cpu() if v is not None else None)
                  for k, v in make_inputs(model.stage1, T, H, W).items()}
        torch.cuda.empty_cache()
        handover = not args.no_handover
        run(model.stage1, inputs, handover)  # warm-up
        (ref, base, peak), secs = timed_run(model.stage1, inputs, handover)
        ref = {k: v.cpu() for k, v in ref.items()}
        torch.cuda.empty_cache()
        rows.append({"candidate": "baseline", "frames": T, "peak_gib": peak / GIB, "s": secs})
        print(f"T={T} baseline: peak {peak / GIB:.3f} GiB, {secs:.2f} s", flush=True)
        for name, assigns in candidates:
            saved = apply(model, assigns)
            try:
                run(model.stage1, inputs, handover)  # warm-up
                (out, base, peak), secs = timed_run(model.stage1, inputs, handover)
            finally:
                restore(saved)
            out = {k: v.cpu() for k, v in out.items()}
            torch.cuda.empty_cache()
            per_key = {k: torch.equal(ref[k], out[k]) for k in ref}
            equal = all(per_key.values())
            diff = max(float((ref[k] - out[k]).abs().max()) for k in ref)
            rows.append({"candidate": name, "frames": T, "peak_gib": peak / GIB, "s": secs,
                         "bitwise_equal": equal, "max_abs_diff": diff, "equal_by_output": per_key})
            print(f"T={T} {name}: peak {peak / GIB:.3f} GiB, {secs:.2f} s, bitwise equal "
                  f"{equal} (max abs diff {diff:.3g}; {per_key})", flush=True)
        del inputs
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps({"hw": args.hw, "gpu": torch.cuda.get_device_name(0),
                                              "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
