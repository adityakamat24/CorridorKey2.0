#!/usr/bin/env python3
"""Stage-2 phase throughput: default host copies vs --pinned_io, eager vs
--cuda_graphs, on one GPU in one process.

Runs the low-VRAM phase-2 loop (stage 2, render, hand-off to the writers)
over the first frames of a clip with a fixed random baton, the variants
alternating in short rounds (order rotated each round) so clock drift, such
as a laptop heating up, hits them alike. The writers stand in for the video
writer: a queue of two batches drained by a thread that copies each frame to
bytes, as the pipe write does, but no encoder to set the pace. Prints the
median wall time per frame for each variant, then runs each variant once
more hashing every frame it hands over and checks the hashes match the
default path's.

Usage:
    python perf/stage2_probe.py --input test_clips/converge200_1080 \
        --frames 12 --rounds 4 --out perf/results/phase4/stage2_probe_laptop.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from corridorkey import runtime as rt
from corridorkey.blocks import set_region_index_cache

VARIANTS = {
    "default": dict(pinned=False, graph=False),
    "pinned_io": dict(pinned=True, graph=False),
    "graph": dict(pinned=False, graph=True),
    "graph+pinned_io": dict(pinned=True, graph=True),
    # Not bit-exact: --fast's attention kernel choice.
    "fast_attention": dict(pinned=False, graph=False, fast=True),
    "graph+pinned_io+fast_attention": dict(pinned=True, graph=True, fast=True),
}
DEFAULT_VARIANTS = ["default", "pinned_io", "graph", "graph+pinned_io"]


class StandInWriter:
    def __init__(self, hash_frames: bool = False):
        self.q = queue.Queue(maxsize=2)
        self.digest = hashlib.sha256() if hash_frames else None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            data = item.tobytes()
            if self.digest is not None:
                self.digest.update(data)
            self.q.task_done()

    def write_many(self, frames: np.ndarray):
        self.q.put(np.ascontiguousarray(frames))

    def release(self) -> str | None:
        self.q.put(None)
        self.thread.join()
        return self.digest.hexdigest() if self.digest is not None else None


def run_variant(model, call, frames, baton, meta, outputs, checker_t, *, pinned: bool,
                fast: bool = False, hash_frames: bool = False):
    with rt.attention_backends(fast):
        return _run_variant(model, call, frames, baton, meta, outputs, checker_t,
                            pinned=pinned, hash_frames=hash_frames)


@torch.no_grad()
def _run_variant(model, call, frames, baton, meta, outputs, checker_t, *, pinned: bool,
                 hash_frames: bool = False):
    """The runtime's write_prediction_batches loop for stage2_batch=1."""
    writers = {name: StandInWriter(hash_frames) for name in sorted(outputs)}
    copies = rt._HostCopies(writers) if pinned else None
    device = next(model.stage2.parameters()).device
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(len(frames)):
        rgb_t = torch.stack([frames[t]], dim=0)
        if pinned:
            rgb_t = rgb_t.pin_memory()
        rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
        baton_np = np.stack([np.asarray(baton[t]).copy()], axis=0)
        baton_t = torch.from_numpy(baton_np).unsqueeze(0)
        if pinned:
            baton_t = baton_t.pin_memory()
        baton_t = baton_t.to(device, non_blocking=True).float()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = call(rgb_t, baton_t)
            if isinstance(logits, tuple):
                logits = logits[0]
        rendered = rt._render_predictions_torch(meta, outputs, checker_t, logits[0],
                                                on_device=pinned)
        if pinned:
            copies.submit(rendered)
            copies.flush(keep=1)
        else:
            for name, out in rendered.items():
                writers[name].write_many(out)
        del logits, baton_t, rgb_t, baton_np, rendered
    if pinned:
        copies.flush()
    for w in writers.values():
        w.q.join()
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    digests = {name: w.release() for name, w in writers.items()}
    return secs, digests


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="test_clips/converge200_1080",
                    help="Frame directory.")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--outputs", nargs="+", default=["alpha", "fg"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = rt.load_model(rt.RuntimePaths.from_package(ROOT), device="cuda")
    rt.apply_weight_policy(model, bf16_linear_conv=True)
    cfg = rt.InferenceConfig(low_vram=True, lean_activations=True, cache_region_indices=True)
    rt.configure_low_vram_inference(model, cfg)
    rt.configure_lean_activations(model, True)
    set_region_index_cache(True)
    rt._phase2_setup(model)

    meta = rt.probe_frame_dir_meta(args.input, num_frames=args.frames, shape_quantum=128,
                                   fps=30.0, exr_linear=False, model_scale=1.0)
    meta["cutout_linear"] = False
    meta["despill_strength"] = 0.5
    H, W = meta["shape_hw"]
    H_out, W_out = meta["output_hw"]
    reader = rt.SequentialFrameDirWindowReader(args.input, meta)
    model_chunk, _display, _review = reader.window(0, args.frames)
    frames = [model_chunk[t] for t in range(args.frames)]
    reader.close()
    rng = np.random.default_rng(0)
    baton = rng.standard_normal((args.frames, model.stage1.d_16, H // 16, W // 16),
                                dtype=np.float32).astype(np.float16)
    checker_t = torch.from_numpy(rt.make_checker(H_out, W_out))
    outputs = set(args.outputs)

    calls = {}
    for name in args.variants:
        calls[name] = (rt._GraphedForward(model.stage2) if VARIANTS[name]["graph"]
                       else model.stage2)
        run_variant(model, calls[name], frames, baton, meta, outputs, checker_t,
                    pinned=VARIANTS[name]["pinned"],
                    fast=VARIANTS[name].get("fast", False))  # warm-up (and graph capture)
    times = {name: [] for name in args.variants}
    for r in range(args.rounds):
        order = args.variants[r % len(args.variants):] + args.variants[:r % len(args.variants)]
        for name in order:
            secs, _ = run_variant(model, calls[name], frames, baton, meta, outputs, checker_t,
                                  pinned=VARIANTS[name]["pinned"],
                                  fast=VARIANTS[name].get("fast", False))
            times[name].append(secs / args.frames * 1000)
        print(f"round {r + 1}/{args.rounds}: "
              + ", ".join(f"{n} {times[n][-1]:.1f}" for n in args.variants) + " ms/frame",
              flush=True)
    digests = {}
    for name in args.variants:
        _, digests[name] = run_variant(model, calls[name], frames, baton, meta, outputs,
                                       checker_t, pinned=VARIANTS[name]["pinned"],
                                       fast=VARIANTS[name].get("fast", False),
                                       hash_frames=True)
    ref = digests[args.variants[0]]
    base = statistics.median(times[args.variants[0]])
    rows = []
    for name in args.variants:
        med = statistics.median(times[name])
        same = digests[name] == ref
        rows.append({"variant": name, "ms_per_frame_median": med,
                     "ms_per_frame": times[name], "speedup": base / med,
                     "same_bytes_as_first": same})
        print(f"{name:16s} {med:8.1f} ms/frame  x{base / med:.3f}  "
              f"same bytes as {args.variants[0]}: {same}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(0), "input": args.input,
            "shape_hw": [H, W], "output_hw": [H_out, W_out], "frames": args.frames,
            "rounds": args.rounds, "outputs": sorted(outputs), "rows": rows,
            "digests": digests}, indent=2))


if __name__ == "__main__":
    main()
