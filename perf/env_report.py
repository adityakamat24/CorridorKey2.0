#!/usr/bin/env python3
"""Phase 0 environment and model inventory for the CorridorKey perf work.

Writes a JSON report (and prints a summary) covering:
  - GPU, driver, CUDA/cuDNN/torch versions, idle VRAM, clocks
  - which SDPA backends actually run on this GPU for the head dims the model
    uses (64 at /16, 96 at /64), in bf16
  - per-module parameter/buffer counts, bytes and dtypes, both as stored in
    the checkpoint files (header only, no tensor data is read) and, with
    --load, as loaded by the unmodified runtime `load_model`

Usage:
    python perf/env_report.py --out perf/results/phase0_env.json [--load]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
                "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def nvidia_smi(fields: str) -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    return dict(zip(fields.split(","), (v.strip() for v in out.split(","))))


def safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def stored_inventory_safetensors(path: Path, prefix_depth: int = 1) -> dict:
    """Group tensors by their first `prefix_depth` dotted name parts."""
    header = safetensors_header(path)
    groups: dict[str, dict] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        key = ".".join(name.split(".")[:prefix_depth])
        g = groups.setdefault(key, {"tensors": 0, "elements": 0, "bytes": 0,
                                    "dtypes": Counter()})
        numel = int(np.prod(info["shape"])) if info["shape"] else 1
        g["tensors"] += 1
        g["elements"] += numel
        g["bytes"] += numel * _DTYPE_BYTES[info["dtype"]]
        g["dtypes"][info["dtype"]] += numel
    for g in groups.values():
        g["dtypes"] = dict(g["dtypes"])
    return groups


def stored_inventory_npz(path: Path) -> dict:
    total = {"tensors": 0, "elements": 0, "bytes": 0, "dtypes": Counter()}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format._read_array_header(f, version)
            numel = int(np.prod(shape)) if shape else 1
            total["tensors"] += 1
            total["elements"] += numel
            total["bytes"] += numel * dtype.itemsize
            total["dtypes"][str(dtype)] += numel
    total["dtypes"] = dict(total["dtypes"])
    return total


def stored_inventory_torch(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    total = {"tensors": 0, "elements": 0, "bytes": 0, "dtypes": Counter()}
    for v in sd.values():
        if torch.is_tensor(v):
            total["tensors"] += 1
            total["elements"] += v.numel()
            total["bytes"] += v.numel() * v.element_size()
            total["dtypes"][str(v.dtype).replace("torch.", "")] += v.numel()
    total["dtypes"] = dict(total["dtypes"])
    del sd
    return total


def module_inventory(module: torch.nn.Module) -> dict:
    params = list(module.parameters())
    buffers = list(module.buffers())
    dtypes = Counter()
    devices = Counter()
    for t in params + buffers:
        dtypes[str(t.dtype).replace("torch.", "")] += t.numel()
        devices[str(t.device)] += t.numel()
    return {
        "params": sum(p.numel() for p in params),
        "param_bytes": sum(p.numel() * p.element_size() for p in params),
        "buffers": sum(b.numel() for b in buffers),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in buffers),
        "dtypes": dict(dtypes),
        "devices": dict(devices),
    }


def sdpa_backend_matrix() -> dict:
    """Try each SDPA backend alone on bf16 shapes matching the model's heads."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cases = {
        # (batch, heads, tokens, head_dim): /16 Swin-3D window at T=8,
        # /16 full-spatial frame at 1080p, /64 full spatiotemporal at T=8.
        "swin3d_16_T8_hd64": (40, 8, 8 * 256, 64),
        "fullspatial_16_1080p_hd64": (1, 8, 8640, 64),
        "full3d_64_1080p_T8_hd96": (1, 16, 8 * 540, 96),
    }
    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }
    result = {}
    for case, shape in cases.items():
        row = {}
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ref = None
        for name, backend in backends.items():
            if name == "math" and shape[2] * shape[2] * shape[0] * shape[1] > 2**31:
                row[name] = "skipped (score matrix too large)"
                continue
            try:
                with sdpa_kernel([backend]):
                    out = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                if ref is None:
                    ref = out.float()
                    row[name] = "ok"
                else:
                    err = (out.float() - ref).abs().max().item()
                    row[name] = f"ok (max|diff| vs first ok = {err:.2e})"
            except RuntimeError as e:
                row[name] = "unavailable: " + str(e).splitlines()[0][:120]
        # Which backend the default dispatcher picks for this shape.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
        kernels = sorted({e.name for e in prof.events()
                          if e.device_type == torch.autograd.DeviceType.CUDA})
        row["default_dispatch_kernels"] = kernels
        result[case] = row
        del q, k, v, ref
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "perf/results/phase0_env.json"))
    ap.add_argument("--load", action="store_true",
                    help="Also load the model with the runtime loader and "
                         "inventory the loaded modules.")
    args = ap.parse_args()

    report: dict = {}
    smi_fields = ("name,driver_version,memory.total,memory.used,memory.free,"
                  "clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
                  "temperature.gpu,power.draw,pstate,pcie.link.gen.current,"
                  "pcie.link.width.current")
    report["nvidia_smi_before_cuda_init"] = nvidia_smi(smi_fields)
    report["platform"] = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    props = torch.cuda.get_device_properties(0)
    report["gpu"] = {
        "name": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_gib": round(props.total_memory / 2**30, 3),
    }
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    report["vram_after_cuda_init"] = {
        "free_gib": round(free / 2**30, 3),
        "total_gib": round(total / 2**30, 3),
        "nvidia_smi": nvidia_smi("memory.used,memory.free"),
    }
    report["sdpa_flags"] = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    report["sdpa_backends"] = sdpa_backend_matrix()

    stored = {}
    ck = stored_inventory_safetensors(ROOT / "weights/corridorkey_v2.safetensors")
    for key, g in sorted(ck.items()):
        stored[f"checkpoint:{key}"] = g
    stored["vfms:cradio(model.safetensors)"] = {
        k: v for k, v in _merge(stored_inventory_safetensors(
            ROOT / "vfms/cradio/C-RADIOv4-SO400M/model.safetensors", 0)).items()}
    stored["vfms:moge(model.pt)"] = stored_inventory_torch(ROOT / "vfms/moge/model.pt")
    stored["vfms:rvm(npz)"] = stored_inventory_npz(
        ROOT / "vfms/rvm/pretrain_rvm_small16_256_204031069.npz")
    report["stored"] = stored
    header = safetensors_header(ROOT / "weights/corridorkey_v2.safetensors")
    meta = header.get("__metadata__", {})
    report["checkpoint_metadata"] = {
        "format": meta.get("format"),
        "metadata": json.loads(meta.get("metadata", "{}")),
        "model_config": json.loads(meta.get("model_config", "{}")),
    }

    if args.load:
        from corridorkey.runtime import RuntimePaths, load_model
        paths = RuntimePaths.from_package(ROOT)
        torch.cuda.reset_peak_memory_stats()
        model = load_model(paths, device="cuda")
        loaded = {}
        for name in ("foundations.cradio", "foundations.moge", "foundations.rvm",
                     "stage1", "stage2"):
            mod = model.get_submodule(name)
            loaded[name] = module_inventory(mod)
        loaded["model_total"] = module_inventory(model)
        report["loaded"] = loaded
        report["vram_after_load"] = {
            "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "nvidia_smi": nvidia_smi("memory.used,memory.free"),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _merge(groups: dict) -> dict:
    total = {"tensors": 0, "elements": 0, "bytes": 0, "dtypes": Counter()}
    for g in groups.values():
        total["tensors"] += g["tensors"]
        total["elements"] += g["elements"]
        total["bytes"] += g["bytes"]
        total["dtypes"].update(g["dtypes"])
    total["dtypes"] = dict(total["dtypes"])
    return total


if __name__ == "__main__":
    main()
