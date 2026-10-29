#!/usr/bin/env python3
"""Probe: what makes up stage-1 memory, and is bf16 weight storage exact?

Runs single stage-1 forwards on random inputs shaped exactly like
hann_stage1_stream builds them (low-VRAM chunking configured the same way),
under torch.no_grad() + autocast(bf16), and reports the weight footprint and
the forward's peak allocated memory for several modes:

  default        the runtime as shipped: fp32 parameters with requires_grad
                 left on. Autocast caches the bf16 cast of every parameter
                 that requires grad for the whole autocast region.
  frozen         parameters requires_grad_(False): autocast stops caching
                 casts (they become per-call temporaries)
  cache_off      autocast(cache_enabled=False), parameters untouched
  bf16_linconv   nn.Linear / nn.Conv* weights and biases stored as bf16, all
                 other parameters (norm weights, scales) kept fp32

Outputs of every mode are compared with `default` (max abs difference of the
baton and aux heads); 0.0 means bit-identical.

Usage:
    python perf/floor_probe.py --hw 1152 1920 --frames 1 4 8 \
        --out perf/results/phase2/floor_probe_1080.json
"""

from __future__ import annotations

import argparse
import copy
import json
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

GIB = 2**30


def make_inputs(stage1, T: int, H: int, W: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h64, w64 = H // 64, W // 64
    rnd = lambda *s, dtype=torch.float32: torch.randn(*s, generator=g, device="cuda").to(dtype)
    return dict(
        rgb=rnd(1, T, 3, H, W, dtype=torch.bfloat16),
        hint=torch.zeros(1, T, 1, H, W, device="cuda", dtype=torch.bfloat16),
        moge_feat=rnd(1, T, stage1.moge_dim, h64, w64),
        cradio_feat=rnd(1, T, stage1.cradio_dim, h64, w64),
        rvm_feat=rnd(1, T, stage1.rvm_dim, h64, w64),
        hint_quality=torch.full((1, T), -1.0, device="cuda", dtype=torch.bfloat16),
    )


def to_bf16_linconv(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            m.weight.data = m.weight.data.to(torch.bfloat16)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(torch.bfloat16)


@torch.no_grad()
def run(stage1, inputs: dict, cache_enabled: bool = True):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache_enabled):
        out = stage1(inputs["rgb"], inputs["hint"], inputs["moge_feat"],
                     cradio_feat=inputs["cradio_feat"], rvm_feat=inputs["rvm_feat"],
                     t_stride=1.0, hint_quality=inputs["hint_quality"])
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    result = {"baton": out["baton"].float(), "aux": out["aux"].float()}
    del out
    return result, base, peak


def param_bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", type=int, nargs=2, default=[1152, 1920],
                    help="Padded model input size (1080p pads to 1152x1920).")
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--no_low_vram", action="store_true",
                    help="Skip configure_low_vram_inference (normal-mode chunking).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    H, W = args.hw

    model = load_model(RuntimePaths.from_package(ROOT), device="cuda")
    model.foundations.cpu()
    model.stage2.cpu()
    torch.cuda.empty_cache()
    if not args.no_low_vram:
        configure_low_vram_inference(model, InferenceConfig(low_vram=True))
    stage1_cpu = model.stage1  # fp32, requires_grad left as load_model leaves it

    rows = []
    for T in args.frames:
        refs = None
        for mode in ("default", "frozen", "cache_off", "bf16_linconv", "bf16_frozen"):
            s1 = copy.deepcopy(stage1_cpu)
            if mode in ("frozen", "bf16_frozen"):
                s1.requires_grad_(False)
            if mode in ("bf16_linconv", "bf16_frozen"):
                to_bf16_linconv(s1)
            s1 = s1.cuda()
            torch.cuda.empty_cache()
            inputs = make_inputs(s1, T, H, W)
            out, base, peak = run(s1, inputs, cache_enabled=(mode != "cache_off"))
            row = {
                "frames": T, "mode": mode,
                "param_gib": param_bytes(s1) / GIB,
                "allocated_before_gib": base / GIB,
                "peak_gib": peak / GIB,
                "forward_extra_gib": (peak - base) / GIB,
            }
            if refs is None:
                refs = out
            else:
                row["max_abs_diff_baton"] = (out["baton"] - refs["baton"]).abs().max().item()
                row["max_abs_diff_aux"] = (out["aux"] - refs["aux"]).abs().max().item()
            rows.append(row)
            print(json.dumps(row), flush=True)
            del s1, inputs, out
            torch.cuda.empty_cache()
        del refs
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hw": [H, W], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
