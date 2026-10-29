#!/usr/bin/env python3
"""Benchmark one CorridorKey inference config: time and GPU memory per phase.

Config in, JSON out. Loads the model once with the runtime's own
`load_model`, then calls the unmodified `run_inference` for --warmup +
--repeats runs. Phases are measured from the outside (perf/profiling.py
wraps the runtime's phase functions), with a CUDA synchronize at each phase
boundary:

  setup         probe, writers, initial device moves
  rvm           RVM continuous prepass (memmap)
  foundations   C-RADIO + MoGe prepass to memmaps (low-VRAM only)
  stage1        Hann stage-1 windows (normal mode: includes stage 2)
  stage2        low-VRAM stage-2 pass, including its device moves
  encode_tail   writer drain after the last frame is queued
  after_*       gaps between phases (device moves, cleanup)

Per phase: wall seconds, torch allocator peak allocated/reserved (reset at
each phase start), and NVML device memory, SM clock, power and temperature.
The summary reports medians over the measured runs, the global peak and the
phase that owns it, next to the runtime's own `_VRAM_PEAK_GB` estimate.

Usage:
    python perf/bench.py --input test_clips/converge200_1080 --num_frames 48 \
        --hann_chunk 8 --hann_stride 4 --low_vram --repeats 3 \
        --out perf/results/bench/b_1080_8x4.json
Optional, on the last measured run only: --mem_snapshot FILE.pickle,
--torch_profile DIR, --block_mem, --nvtx.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from corridorkey import runtime as rt
from corridorkey import video_io as vio
from corridorkey.runtime import InferenceConfig, RuntimePaths, WriterConfig, load_model
from profiling import (NvmlSampler, PhaseTracker, block_memory_hooks,
                       instrument_runtime, nvtx_hooks)


def parse_args():
    d = InferenceConfig()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Inference config (same meaning and defaults as infer.py).
    ap.add_argument("--input", required=True)
    ap.add_argument("--hann_chunk", type=int, required=True)
    ap.add_argument("--hann_stride", type=int, default=None)
    ap.add_argument("--low_vram", action="store_true")
    ap.add_argument("--num_frames", type=int, default=-1)
    ap.add_argument("--start_mode", default="begin")
    ap.add_argument("--hint_first_frame", default=None)
    ap.add_argument("--hint_quality", type=float, default=d.hint_quality)
    ap.add_argument("--no_carry_hint", action="store_true")
    ap.add_argument("--outputs", nargs="*", default=["alpha", "fg"])
    ap.add_argument("--output_format", choices=("video", "exr"), default="video")
    ap.add_argument("--exr_dtype", choices=("half", "float"), default="half")
    ap.add_argument("--frame_dir_fps", type=float, default=30.0)
    ap.add_argument("--frame_dir_linear", action="store_true",
                    help="InferenceConfig.frame_dir_linear (scene-linear EXR frames)")
    ap.add_argument("--stage2_batch", type=int, default=d.stage2_batch)
    ap.add_argument("--low_vram_foundation_chunk", type=int, default=d.low_vram_foundation_chunk)
    ap.add_argument("--low_vram_mlp_chunk_tokens", type=int, default=d.low_vram_mlp_chunk_tokens)
    ap.add_argument("--low_vram_enc_frame_chunk", type=int, default=d.low_vram_enc_frame_chunk)
    ap.add_argument("--low_vram_swin_window_batch", type=int, default=d.low_vram_swin_window_batch)
    ap.add_argument("--low_vram_full_attn_query_chunk_tokens", type=int,
                    default=d.low_vram_full_attn_query_chunk_tokens)
    ap.add_argument("--temp_dir", default=str(ROOT / "perf/out/_tmp/bench"))
    # Bench options.
    ap.add_argument("--sdpa", default="default",
                    choices=("default", "flash", "efficient", "cudnn", "math"),
                    help="Restrict attention to one backend (see perf/infer_with.py). "
                         "'efficient' mirrors Windows, whose torch has no FlashAttention.")
    ap.add_argument("--mem_cap_gib", type=float, default=None,
                    help="Cap the caching allocator (set_per_process_memory_fraction) to "
                         "emulate a smaller card. An OOM is recorded in the JSON, with "
                         "the phase it hit, instead of raised.")
    ap.add_argument("--frozen_weights", action="store_true",
                    help="runtime.apply_weight_policy(freeze=True)")
    ap.add_argument("--bf16_weights", action="store_true",
                    help="runtime.apply_weight_policy(bf16_linear_conv=True)")
    ap.add_argument("--lean_activations", action="store_true",
                    help="InferenceConfig.lean_activations")
    ap.add_argument("--prefetch_frames", action="store_true",
                    help="InferenceConfig.prefetch_frames")
    ap.add_argument("--cache_region_indices", action="store_true",
                    help="InferenceConfig.cache_region_indices")
    ap.add_argument("--activation_arena", action="store_true",
                    help="InferenceConfig.activation_arena")
    ap.add_argument("--pinned_io", action="store_true",
                    help="InferenceConfig.pinned_io")
    ap.add_argument("--async_baton", action="store_true",
                    help="InferenceConfig.async_baton")
    ap.add_argument("--lean_reads", action="store_true",
                    help="InferenceConfig.lean_reads")
    ap.add_argument("--fast_load", action="store_true",
                    help="load_model(skip_init=True)")
    ap.add_argument("--lean_foundations", action="store_true",
                    help="runtime.drop_unused_foundation_parts")
    ap.add_argument("--decode_once", action="store_true",
                    help="InferenceConfig.decode_once")
    ap.add_argument("--gpu_prep", action="store_true",
                    help="InferenceConfig.gpu_prep")
    ap.add_argument("--read_workers", type=int, default=4,
                    help="InferenceConfig.read_workers")
    ap.add_argument("--read_ahead", action="store_true",
                    help="InferenceConfig.read_ahead")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True, help="JSON result path.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--keep_outputs", action="store_true")
    ap.add_argument("--nvml_interval", type=float, default=0.02)
    ap.add_argument("--mem_snapshot", default=None)
    ap.add_argument("--torch_profile", default=None)
    ap.add_argument("--block_mem", action="store_true")
    ap.add_argument("--nvtx", action="store_true")
    return ap.parse_args()


def build_config(args) -> InferenceConfig:
    return InferenceConfig(
        hann_chunk=args.hann_chunk,
        hann_stride=args.hann_stride or max(1, args.hann_chunk // 2),
        auto_vram=False,
        carry_hint=not args.no_carry_hint,
        hint_quality=args.hint_quality,
        num_frames=args.num_frames,
        start_mode=args.start_mode,
        low_vram=args.low_vram,
        low_vram_foundation_chunk=args.low_vram_foundation_chunk,
        low_vram_mlp_chunk_tokens=args.low_vram_mlp_chunk_tokens,
        low_vram_enc_frame_chunk=args.low_vram_enc_frame_chunk,
        low_vram_swin_window_batch=args.low_vram_swin_window_batch,
        low_vram_full_attn_query_chunk_tokens=args.low_vram_full_attn_query_chunk_tokens,
        stage2_batch=args.stage2_batch,
        outputs=set(args.outputs),
        temp_dir=Path(args.temp_dir),
        frame_dir_fps=args.frame_dir_fps,
        frame_dir_linear=args.frame_dir_linear,
        output_format=args.output_format,
        exr_dtype=args.exr_dtype,
        lean_activations=args.lean_activations,
        prefetch_frames=args.prefetch_frames,
        cache_region_indices=args.cache_region_indices,
        activation_arena=args.activation_arena,
        pinned_io=args.pinned_io,
        async_baton=args.async_baton,
        lean_reads=args.lean_reads,
        read_ahead=args.read_ahead,
        decode_once=args.decode_once,
        gpu_prep=args.gpu_prep,
        read_workers=args.read_workers,
    )


def reset_to_load_state(model) -> None:
    """Put the modules where `load_model` leaves them (foundations on the GPU,
    stage1/stage2 on the CPU), i.e. the state a fresh infer.py process starts
    from. Without this, run N+1 starts with stage 2 still on the GPU from run
    N, and the runtime's setup uploads the foundations on top of it."""
    model.stage2.cpu()
    model.stage1.cpu()
    model.foundations.cuda()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def summarize(runs: list[dict]) -> dict:
    measured = [r for r in runs if not r["warmup"] and not r.get("oom")]
    ooms = [r["oom"] for r in runs if r.get("oom")]
    if not measured:
        return {"measured_runs": 0, "oom": ooms[0] if ooms else None}
    names = []
    for r in measured:
        for p in r["phases"]:
            if p["phase"] not in names:
                names.append(p["phase"])
    phases = {}
    for name in names:
        rows = [p for r in measured for p in r["phases"] if p["phase"] == name]
        med = lambda k: statistics.median(p[k] for p in rows if p.get(k) is not None)
        phases[name] = {
            "wall_s": med("wall_s"),
            "wall_s_all": [round(p["wall_s"], 3) for p in rows],
            "peak_allocated_gib": med("peak_allocated_gib"),
            "peak_reserved_gib": med("peak_reserved_gib"),
            "nvml_mem_used_max_gib": (med("nvml_mem_used_max_gib")
                                      if any("nvml_mem_used_max_gib" in p for p in rows) else None),
            "sm_mhz_median": (med("sm_mhz_median")
                              if any("sm_mhz_median" in p for p in rows) else None),
        }
    owner = max(phases, key=lambda n: phases[n]["peak_reserved_gib"])
    windows = [w for r in measured for w in r["stage1_windows_ms"]]
    return {
        "measured_runs": len(measured),
        "total_s_median": statistics.median(r["total_s"] for r in measured),
        "total_s_all": [round(r["total_s"], 3) for r in measured],
        "peak_allocated_gib": max(p["peak_allocated_gib"] for p in phases.values()),
        "peak_reserved_gib": max(p["peak_reserved_gib"] for p in phases.values()),
        "peak_owner_phase": owner,
        "stage1_window_ms_median": statistics.median(windows) if windows else None,
        "stage2_forward_ms_per_frame": statistics.median(
            r["stage2_forward_ms_total"] / max(r["frames"], 1) for r in measured),
        "phases": phases,
    }


def main():
    args = parse_args()
    paths = RuntimePaths.from_package(ROOT)
    cfg = build_config(args)
    sampler = NvmlSampler(args.nvml_interval).start()
    idle = sampler.idle_snapshot()
    tracker = PhaseTracker(sampler)
    if args.mem_cap_gib:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gib * 2**30 / total))
    from infer_with import sdpa_context

    t0 = time.perf_counter()
    model = load_model(paths, device="cuda", skip_init=args.fast_load)
    rt.apply_weight_policy(model, freeze=args.frozen_weights,
                           bf16_linear_conv=args.bf16_weights)
    if args.lean_foundations:
        rt.drop_unused_foundation_parts(model)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    runs = []
    total_runs = args.warmup + args.repeats
    (ROOT / "perf/out").mkdir(parents=True, exist_ok=True)
    for i in range(total_runs):
        last = i == total_runs - 1
        reset_to_load_state(model)
        out_dir = Path(tempfile.mkdtemp(prefix="bench_", dir=ROOT / "perf/out"))
        with ExitStack() as stack:
            if last and args.mem_snapshot:
                torch.cuda.memory._record_memory_history(
                    max_entries=1_000_000, stacks="python")
            if last and args.block_mem:
                block_rows = stack.enter_context(block_memory_hooks(model))
            if last and args.nvtx:
                stack.enter_context(nvtx_hooks(model))
            prof = None
            if last and args.torch_profile:
                from profiling import record_function_hooks
                stack.enter_context(record_function_hooks(model))
                prof = stack.enter_context(torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True))
            stack.enter_context(sdpa_context(args.sdpa))
            timings = stack.enter_context(instrument_runtime(rt, vio, model, tracker))
            t_run = time.perf_counter()
            oom = None
            try:
                rt.run_inference(model, args.input, str(out_dir), cfg, WriterConfig(),
                                 initial_hint_path=args.hint_first_frame,
                                 hint_video_path=None)
            except torch.OutOfMemoryError as exc:
                oom = {"phase": timings.get("phase_before_encode") or tracker.current,
                       "message": str(exc).splitlines()[0]}
            torch.cuda.synchronize()
            total_s = time.perf_counter() - t_run
        phases = tracker.finish()
        frames = timings.get("stage2_forwards", 0) * cfg.stage2_batch
        run = {"warmup": i < args.warmup, "total_s": total_s, "frames": frames,
               "phases": phases, "oom": oom, **timings}
        runs.append(run)
        if oom:
            print(f"[bench] run {i + 1}: OOM in {oom['phase']} ({oom['message'][:90]})",
                  flush=True)
            torch.cuda.empty_cache()
            shutil.rmtree(out_dir, ignore_errors=True)
            break
        print(f"[bench] run {i + 1}/{total_runs}{' (warmup)' if run['warmup'] else ''}: "
              f"{total_s:.2f}s, peak reserved "
              f"{max(p['peak_reserved_gib'] for p in phases):.2f} GiB", flush=True)
        if last and args.mem_snapshot:
            torch.cuda.memory._dump_snapshot(args.mem_snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)
        if last and args.block_mem:
            run["block_mem"] = block_rows
        if prof is not None:
            pdir = Path(args.torch_profile)
            pdir.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(pdir / "trace.json"))
            (pdir / "ops_by_cuda_time.txt").write_text(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=80))
            from profiling import attribute_kernels
            (pdir / "kernels_by_module.json").write_text(
                json.dumps(attribute_kernels(prof), indent=2))
        if not args.keep_outputs:
            shutil.rmtree(out_dir, ignore_errors=True)
    sampler.stop()

    meta_hw = rt.probe_frame_dir_meta(
        args.input, num_frames=args.num_frames, shape_quantum=cfg.shape_quantum,
        fps=args.frame_dir_fps, exr_linear=args.frame_dir_linear, model_scale=1.0)["shape_hw"] \
        if os.path.isdir(args.input) else None
    estimate = (rt.estimate_peak_vram_gb(meta_hw[0] * meta_hw[1] / 1e6,
                                         cfg.hann_chunk, cfg.low_vram)
                if meta_hw else None)
    report = {
        "label": args.label,
        "argv": sys.argv,
        "config": {k: (sorted(v) if isinstance(v, set) else str(v) if isinstance(v, Path) else v)
                   for k, v in vars(cfg).items()},
        "sdpa": args.sdpa,
        "mem_cap_gib": args.mem_cap_gib,
        "weights": ("bf16_linear_conv+frozen" if args.bf16_weights
                    else "frozen" if args.frozen_weights else "as loaded"),
        "env": {
            "platform": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "nvml_idle": idle,
        },
        "model_shape_hw": meta_hw,
        "vram_table_estimate_gib": estimate,
        "load_s": load_s,
        "summary": summarize(runs),
        "runs": runs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    s = report["summary"]
    if not s.get("measured_runs"):
        print(f"[bench] no completed measured run; oom: {s.get('oom')}")
        sys.exit(3)
    print(f"[bench] median total {s['total_s_median']:.2f}s; peak reserved "
          f"{s['peak_reserved_gib']:.2f} GiB (allocated {s['peak_allocated_gib']:.2f}) "
          f"in {s['peak_owner_phase']}; table estimate {estimate}")
    for name, p in s["phases"].items():
        print(f"  {name:<18} {p['wall_s']:8.2f}s  peak alloc {p['peak_allocated_gib']:6.2f}  "
              f"reserved {p['peak_reserved_gib']:6.2f} GiB  sm {p['sm_mhz_median']}")


if __name__ == "__main__":
    main()
