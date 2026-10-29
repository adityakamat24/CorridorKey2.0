#!/usr/bin/env python3
"""Run the unmodified `infer.py` with process-wide perf/debug settings.

Everything after `--` (or every argument this wrapper does not recognize) is
passed to infer.py unchanged. The wrapper only changes the environment the
runtime executes in; it never edits runtime code.

  --sdpa {default,flash,efficient,cudnn,math}
        Restrict F.scaled_dot_product_attention to one backend for the whole
        run (torch.nn.attention.sdpa_kernel). "flash" also allows the
        memory-efficient kernel, because flash cannot take the boolean
        padding masks some Swin calls pass; flash is still preferred
        wherever it applies.
  --mem_budget_gib G
        Emulate a GPU that leaves G GiB to PyTorch: cap the caching
        allocator at G GiB (set_per_process_memory_fraction) and make
        torch.cuda.mem_get_info report the same, so auto VRAM picks the
        window it would pick on that card. Prints the peak allocated and
        reserved memory at the end; exits with code 3 on out-of-memory.
  --uncapped
        With --mem_budget_gib: report the budget but leave the allocator
        uncapped, to see what it reserves when nothing forces a release.
  --fp32
        Run every network in fp32, the reference that approximate options
        are measured against: bf16 autocast off, TF32 off for matmuls and
        convolutions, and floating inputs cast to fp32 where they enter the
        foundations, RVM, stage 1 and stage 2. The weights stay fp32, so do
        not combine with --optimized or --bf16_weights. What passes between
        stages keeps the pipeline's own formats (bf16 model views, fp16 RVM
        features, bf16 carry hint), so only the networks' arithmetic changes.
  --compile MODE
        torch.compile the foundations, RVM, stage 1 and stage 2 in place
        (nn.Module.compile, mode MODE: default, max-autotune,
        max-autotune-no-cudagraphs, reduce-overhead). Needs Triton, so
        Linux. Not bit-exact: fused kernels round differently.

Usage:
    python perf/infer_with.py --sdpa efficient -- --input ... --output_dir ...
    python perf/infer_with.py --mem_budget_gib 4.95 -- --input ... --optimized
    python perf/infer_with.py --fp32 -- --input ... --hann_chunk 8 --hann_stride 4 --low_vram
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sdpa_context(name: str):
    if name == "default":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = {
        "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "cudnn": [SDPBackend.CUDNN_ATTENTION],
        "math": [SDPBackend.MATH],
    }[name]
    return sdpa_kernel(backends, set_priority=True)


def emulate_budget(budget_gib: float, capped: bool = True) -> None:
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(budget_gib * 2**30)
    if capped:
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total))
    torch.cuda.mem_get_info = lambda device=None: (
        max(0, budget - torch.cuda.memory_reserved()), budget)


def fp32_everywhere() -> None:
    import torch

    class Fp32Autocast(torch.autocast):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            super().__init__(device_type, dtype=dtype, enabled=False,
                             cache_enabled=cache_enabled)

    torch.autocast = Fp32Autocast
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    def to_fp32(x):
        if torch.is_tensor(x):
            return x.float() if x.is_floating_point() else x
        if isinstance(x, tuple) and hasattr(x, "_fields"):
            return type(x)(*(to_fp32(v) for v in x))
        if isinstance(x, (list, tuple)):
            return type(x)(to_fp32(v) for v in x)
        if isinstance(x, dict):
            return {k: to_fp32(v) for k, v in x.items()}
        return x

    def inputs_to_fp32(_module, args, kwargs):
        return to_fp32(args), to_fp32(kwargs)

    import infer
    load_model = infer.load_model

    def load_fp32_model(*args, **kwargs):
        model = load_model(*args, **kwargs)
        for module in (model.foundations, getattr(model.foundations, "rvm", None),
                       model.stage1, model.stage2):
            if module is not None:
                module.register_forward_pre_hook(inputs_to_fp32, with_kwargs=True)
        return model

    infer.load_model = load_fp32_model


def compile_modules(mode: str) -> None:
    import infer
    load_model = infer.load_model

    def load_compiled_model(*args, **kwargs):
        model = load_model(*args, **kwargs)
        for module in (model.foundations, getattr(model.foundations, "rvm", None),
                       model.stage1, model.stage2):
            if module is not None:
                module.compile(mode=mode, dynamic=False)
        return model

    infer.load_model = load_compiled_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdpa", default="default",
                    choices=("default", "flash", "efficient", "cudnn", "math"))
    ap.add_argument("--mem_budget_gib", type=float, default=None)
    ap.add_argument("--uncapped", action="store_true",
                    help="With --mem_budget_gib: only report the budget to auto VRAM, "
                         "do not cap the allocator (Windows spills to system RAM "
                         "instead of failing, so what counts there is whether peak "
                         "reserved stays under the budget without a cap).")
    ap.add_argument("--fp32", action="store_true",
                    help="Run every network in fp32 (the reference for approximate "
                         "options); see above.")
    ap.add_argument("--compile", default=None,
                    choices=("default", "max-autotune", "max-autotune-no-cudagraphs",
                             "reduce-overhead"),
                    help="torch.compile the networks with this mode; see above.")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    sys.path.insert(0, str(ROOT))
    import infer

    if args.fp32:
        fp32_everywhere()
    if args.compile:
        compile_modules(args.compile)
    sys.argv = [str(ROOT / "infer.py")] + rest
    if args.mem_budget_gib is None:
        with sdpa_context(args.sdpa):
            infer.main()
        return

    import torch
    emulate_budget(args.mem_budget_gib, capped=not args.uncapped)
    try:
        with sdpa_context(args.sdpa):
            infer.main()
    except torch.OutOfMemoryError as e:
        print(f"[infer_with] OOM: {str(e).splitlines()[0]}", flush=True)
        status = 3
    else:
        status = 0
    print(f"[infer_with] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB, "
          f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.3f} GiB, "
          f"budget {args.mem_budget_gib:.2f} GiB", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
