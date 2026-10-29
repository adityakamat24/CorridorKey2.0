#!/usr/bin/env python3
"""Find the first stage-1 module whose output changes under a weight-storage
variant (default fp32 vs nn.Linear/Conv weights stored as bf16).

Runs one stage-1 forward per variant on identical random inputs with forward
hooks on every submodule, recording a bitwise checksum of each tensor output
(sum of the raw bits as integers) in execution order. Prints the first
modules whose checksum differs, with the max abs difference of that output.

Usage:
    python perf/exactness_probe.py --hw 768 1280 --frames 1
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from corridorkey.runtime import (InferenceConfig, RuntimePaths,
                                 configure_low_vram_inference, load_model)
from floor_probe import make_inputs, to_bf16_linconv

_INT_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16,
             torch.float16: torch.int16}


def bits_sum(t: torch.Tensor) -> int:
    v = t.detach().contiguous()
    return int(v.view(_INT_VIEW[v.dtype]).to(torch.int64).sum()) if v.dtype in _INT_VIEW else 0


def first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if torch.is_tensor(o):
                return o
    if isinstance(out, dict):
        for o in out.values():
            if torch.is_tensor(o):
                return o
    return None


@torch.no_grad()
def trace(stage1, inputs, keep_names: set[str] | None = None):
    rows, keep = [], {}
    handles = []
    for name, mod in stage1.named_modules():
        if not name:
            continue

        def hook(_m, inp, out, name=name):
            t = first_tensor(out)
            if t is None or not t.is_floating_point():
                return
            outs = [o for o in (out if isinstance(out, (tuple, list)) else [out])
                    if torch.is_tensor(o) and o.is_floating_point()]
            ins = [x for x in inp if torch.is_tensor(x) and x.is_floating_point()]
            rows.append((name, str(t.dtype), tuple(t.shape),
                         tuple(bits_sum(o) for o in outs),
                         tuple((str(x.dtype), bits_sum(x)) for x in ins)))
            if keep_names is not None and name in keep_names:
                keep[name + f"#{len(rows)}"] = t.detach().float().clone()
        handles.append(mod.register_forward_hook(hook))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        stage1(inputs["rgb"], inputs["hint"], inputs["moge_feat"],
               cradio_feat=inputs["cradio_feat"], rvm_feat=inputs["rvm_feat"],
               t_stride=1.0, hint_quality=inputs["hint_quality"])
    for h in handles:
        h.remove()
    return rows, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs=2, default=[768, 1280])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    H, W = args.hw

    model = load_model(RuntimePaths.from_package(ROOT), device="cuda")
    model.foundations.cpu()
    model.stage2.cpu()
    torch.cuda.empty_cache()
    configure_low_vram_inference(model, InferenceConfig(low_vram=True))

    ref = copy.deepcopy(model.stage1).cuda()
    inputs = make_inputs(ref, args.frames, H, W)
    rows_ref, _ = trace(ref, inputs)
    del ref
    torch.cuda.empty_cache()

    var = copy.deepcopy(model.stage1)
    to_bf16_linconv(var)
    var = var.cuda()
    rows_var, _ = trace(var, inputs)

    if len(rows_ref) != len(rows_var):
        print(f"call counts differ: {len(rows_ref)} vs {len(rows_var)}")
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(rows_ref, rows_var)) if a != b]
    print(f"{len(rows_ref)} module calls traced, {len(diffs)} differ")
    for i, a, b in diffs[:args.show]:
        same_in = a[4] == b[4]
        print(f"#{i:5d} {a[0]:<40} var name {b[0]:<40} inputs "
              f"{'same' if same_in else 'DIFFER'} {a[4] if not same_in else ''} -> "
              f"{b[4] if not same_in else ''}  outputs ref {a[1]} var {b[1]} shape {a[2]}")

    if diffs:
        names = {a[0] for _, a, _ in diffs[:args.show]}
        torch.cuda.empty_cache()
        ref = copy.deepcopy(model.stage1).cuda()
        _, keep_ref = trace(ref, inputs, names)
        del ref
        _, keep_var = trace(var, inputs, names)
        for k in list(keep_ref)[:args.show]:
            if k in keep_var:
                d = (keep_ref[k] - keep_var[k]).abs().max().item()
                print(f"  max abs diff {d:.3e}  {k}")


if __name__ == "__main__":
    main()
