#!/usr/bin/env python3
"""Measure `_VRAM_PEAK_GB`-style tables (peak allocated GiB per resolution and
Hann window) for the shipped or the optimized runtime path.

The auto window selector (`runtime.select_vram_profile`) reads peak memory
from a measured table, so memory savings only turn into bigger windows once
the table is re-measured. Only stage 1 grows with the window, so this runs
single stage-1 forwards on shape-exact random inputs and records the peak
allocated memory:

  low-VRAM mode  only stage 1 resident, low-VRAM chunking (as in the runtime's
                 stage-1 phase); fixed phases (foundation prepass chunk, one
                 stage-2 frame) measured once per resolution
  normal mode    foundations + stage 1 + stage 2 resident, no low-VRAM
                 chunking, the window's foundation forward included

Each table entry is the max over the phases. Windows stop growing at the
first OOM on the measuring GPU.

Usage:
    python perf/vram_table.py --optimized --modes lowvram normal \
        --out perf/results/phase3/vram_table_optimized.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from corridorkey import runtime as rt
from floor_probe import make_inputs

GIB = 2**30
# Padded model sizes (shape_quantum 128).
SIZES = {
    "720p": (768, 1280),       # 1280x720
    "1080p": (1152, 1920),     # 1920x1080
    "1440p": (1536, 2560),     # 2560x1440
    "3k": (1792, 3072),        # 3072x1728
    "4k": (2176, 3840),        # 3840x2160
    "4k_dci": (2176, 4096),    # 4096x2160
}
WINDOWS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80]


def peak_of(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / GIB
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")


def place(model, foundations: bool, stage1: bool, stage2: bool) -> None:
    for mod, on in ((model.foundations, foundations), (model.stage1, stage1),
                    (model.stage2, stage2)):
        if not on:
            mod.cpu()
    for mod, on in ((model.foundations, foundations), (model.stage1, stage1),
                    (model.stage2, stage2)):
        if on:
            mod.cuda()
    torch.cuda.empty_cache()


def measure(model, mode: str, H: int, W: int, windows: list[int]) -> dict:
    low = mode == "lowvram"
    chunks = rt.InferenceConfig() if low else rt.InferenceConfig(
        low_vram_mlp_chunk_tokens=0, low_vram_enc_frame_chunk=0,
        low_vram_swin_window_batch=0, low_vram_full_attn_query_chunk_tokens=0)
    rt.configure_low_vram_inference(model, chunks)

    # Fixed phases.
    place(model, foundations=True, stage1=not low, stage2=not low)
    found_frames = 4 if low else 1
    rgb = torch.rand(1, found_frames, 3, H, W, device="cuda", dtype=torch.bfloat16)

    def found():
        r, m = rt._build_foundation_inputs(rgb)
        model.foundations(rgb_cradio=r[0], rgb_moge=m[0])
    found_peak = peak_of(found)
    del rgb
    place(model, foundations=not low, stage1=not low, stage2=True)
    x = torch.rand(1, 1, 3, H, W, device="cuda", dtype=torch.bfloat16)
    baton = torch.rand(1, 1, model.stage1.d_16, H // 16, W // 16, device="cuda")
    s2_peak = peak_of(lambda: model.stage2(x, baton))
    del x, baton
    place(model, foundations=not low, stage1=True, stage2=not low)

    rows = {}
    for T in windows:
        inputs = make_inputs(model.stage1, T, H, W)
        if not low:
            rgb_f = torch.rand(1, T, 3, H, W, device="cuda", dtype=torch.bfloat16)

        def window():
            if not low:  # normal mode runs the window's foundations inline
                r, m = rt._build_foundation_inputs(rgb_f)
                model.foundations(rgb_cradio=r[0], rgb_moge=m[0])
            if getattr(model.stage1, "inference_lean", False):
                # As the runtime does: stage 1 holds the only references.
                owned = {k: inputs.pop(k) for k in
                         ("rgb", "hint", "moge_feat", "cradio_feat", "rvm_feat")}
                model.stage1(inputs=owned, t_stride=1.0, hint_quality=inputs["hint_quality"])
            else:
                model.stage1(inputs["rgb"], inputs["hint"], inputs["moge_feat"],
                             cradio_feat=inputs["cradio_feat"], rvm_feat=inputs["rvm_feat"],
                             t_stride=1.0, hint_quality=inputs["hint_quality"])
        p = peak_of(window)
        del inputs
        if not low:
            del rgb_f
        rows[T] = {"window": p, "peak": max(p, found_peak, s2_peak)}
        print(f"  {mode} {W}x{H} T={T}: window {p:.3f} GiB, global {rows[T]['peak']:.3f}",
              flush=True)
        if p != p:  # NaN: OOM on this GPU
            break
    return {"foundation_peak": found_peak, "stage2_peak": s2_peak, "windows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimized", action="store_true",
                    help="bf16 weights + lean activations (the --optimized path)")
    ap.add_argument("--modes", nargs="+", default=["lowvram", "normal"],
                    choices=("lowvram", "normal"))
    ap.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device="cuda")
    if args.optimized:
        rt.apply_weight_policy(model, bf16_linear_conv=True)
    rt.configure_lean_activations(model, args.optimized)
    result = {"optimized": args.optimized, "gpu": torch.cuda.get_device_name(0),
              "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "tables": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        result["tables"][mode] = {}
        for name in args.sizes:
            H, W = SIZES[name]
            print(f"{mode} {name} ({W}x{H}, {H * W / 1e6:.2f} MP)", flush=True)
            entry = measure(model, mode, H, W, args.windows)
            entry["megapixels"] = H * W / 1e6
            result["tables"][mode][name] = entry
            out.write_text(json.dumps(result, indent=2))  # keep partial results


if __name__ == "__main__":
    main()
