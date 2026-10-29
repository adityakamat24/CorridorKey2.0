#!/usr/bin/env python3
"""Why is bf16 weight storage not bit-exact? Isolate one Linear.

Takes the first stage-1 module that diverged in perf/exactness_probe.py
(dec_16.2.attn.qkv.qkv, the CSwin strip-batch QKV projection), and runs it on
identical inputs as (a) fp32 parameters under autocast, the shipped path, and
(b) the same parameters pre-cast to bf16. Checks that the bf16 weight values
are identical, then compares outputs for several input forms.

Usage:
    python perf/bf16_linear_check.py
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

from corridorkey.runtime import RuntimePaths, load_model


def main():
    model = load_model(RuntimePaths.from_package(ROOT), device="cuda")
    model.foundations.cpu()
    torch.cuda.empty_cache()
    lin = model.stage1.dec_16[2].attn.qkv.qkv
    ref = copy.deepcopy(lin).cuda()
    var = copy.deepcopy(lin)
    var.weight.data = var.weight.data.to(torch.bfloat16)
    var.bias.data = var.bias.data.to(torch.bfloat16)
    var = var.cuda()
    print("weights identical after cast:",
          bool((ref.weight.to(torch.bfloat16) == var.weight).all()),
          "| bias identical:", bool((ref.bias.to(torch.bfloat16) == var.bias).all()))

    g = torch.Generator(device="cuda").manual_seed(0)
    big = torch.randn(80, 48, 512, generator=g, device="cuda")  # fp32, like the strip input
    cases = {
        "3D contiguous fp32 (16,48,512)": big[:16].contiguous(),
        "3D slice of bigger fp32 tensor": big[16:32],
        "2D fp32 (768,512)": big[:16].reshape(-1, 512).contiguous(),
        "3D bf16 input": big[:16].to(torch.bfloat16),
    }
    with torch.no_grad():
        for name, x in cases.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_ref = ref(x)
                y_var = var(x)
            same = bool((y_ref == y_var).all())
            d = (y_ref.float() - y_var.float()).abs().max().item()
            print(f"{name:<34} autocast: identical={same}  max|diff|={d:.3e}")
        # Outside autocast, explicitly the same bf16 tensors through F.linear.
        x = big[:16].to(torch.bfloat16)
        w_cast, b_cast = ref.weight.to(torch.bfloat16), ref.bias.to(torch.bfloat16)
        y1 = F.linear(x, w_cast, b_cast)
        y2 = F.linear(x, var.weight, var.bias)
        print("explicit F.linear, cast vs stored bf16 weights: identical=",
              bool((y1 == y2).all()), " max|diff|=", (y1.float() - y2.float()).abs().max().item())
        # Is the autocast path even computing with the bias in bf16?
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_ac = ref(x)
        y_manual = F.linear(x, w_cast, b_cast)
        print("autocast(ref) vs manual bf16 F.linear: identical=",
              bool((y_ac == y_manual).all()), " max|diff|=", (y_ac.float() - y_manual.float()).abs().max().item())


def replay():
    """Capture the diverging Linear's real input/output in both full stage-1
    forwards, then replay the GEMM in isolation with identical bf16 tensors."""
    from floor_probe import make_inputs, to_bf16_linconv
    from corridorkey.runtime import InferenceConfig, configure_low_vram_inference

    model = load_model(RuntimePaths.from_package(ROOT), device="cuda")
    model.foundations.cpu()
    model.stage2.cpu()
    torch.cuda.empty_cache()
    configure_low_vram_inference(model, InferenceConfig(low_vram=True))
    captured = {}

    def run(mode):
        s1 = copy.deepcopy(model.stage1)
        if mode == "var":
            to_bf16_linconv(s1)
        s1 = s1.cuda()
        lin = s1.dec_16[2].attn.qkv.qkv

        def hook(_m, inp, out):
            captured.setdefault(mode, (inp[0].detach().clone(), out.detach().clone()))
        h = lin.register_forward_hook(hook)
        inputs = make_inputs(s1, 1, 768, 1280)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s1(inputs["rgb"], inputs["hint"], inputs["moge_feat"],
               cradio_feat=inputs["cradio_feat"], rvm_feat=inputs["rvm_feat"],
               t_stride=1.0, hint_quality=inputs["hint_quality"])
        h.remove()
        w = lin.weight.detach().to(torch.bfloat16).clone()
        b = lin.bias.detach().to(torch.bfloat16).clone()
        del s1
        torch.cuda.empty_cache()
        return w, b

    w_ref, b_ref = run("ref")
    w_var, b_var = run("var")
    x_ref, y_ref = captured["ref"]
    x_var, y_var = captured["var"]
    print("captured inputs identical:", bool((x_ref == x_var).all()), x_ref.dtype, tuple(x_ref.shape))
    print("bf16 weights identical:", bool((w_ref == w_var).all()), "bias:", bool((b_ref == b_var).all()))
    print("full-forward outputs identical:", bool((y_ref == y_var).all()),
          "max|diff|", (y_ref.float() - y_var.float()).abs().max().item())
    with torch.no_grad():
        y_rep = F.linear(x_ref.to(torch.bfloat16), w_ref, b_ref)
    print("replay == full ref:", bool((y_rep == y_ref).all()),
          "| replay == full var:", bool((y_rep == y_var).all()))


if __name__ == "__main__":
    replay() if "--replay" in sys.argv else main()
