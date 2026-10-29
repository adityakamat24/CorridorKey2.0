"""Profiling helpers for perf/bench.py.

Everything here observes the runtime from the outside (wrapping functions,
module hooks, a sampling thread); nothing changes what the model computes.

  NvmlSampler       background thread sampling device memory, SM/mem clocks,
                    power, temperature and utilization, tagged by phase
  PhaseTracker      wall time and torch allocator peaks per named phase
  instrument_runtime  wraps corridorkey.runtime phase functions so a
                    run_inference call reports per-phase numbers
  block_memory_hooks  per-block peak allocator memory via forward hooks
  nvtx_hooks        NVTX ranges per block, for nsys timelines
"""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager

import torch


class NvmlSampler:
    def __init__(self, interval_s: float = 0.02, device_index: int = 0):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:  # NVML missing: keep running without samples
            print(f"[bench] NVML unavailable ({exc}); no clock/power samples")
            self._nvml = None

    def _sample(self) -> dict:
        n, h = self._nvml, self._handle
        mem = n.nvmlDeviceGetMemoryInfo(h)
        util = n.nvmlDeviceGetUtilizationRates(h)
        try:
            power_w = n.nvmlDeviceGetPowerUsage(h) / 1000.0
        except n.NVMLError:
            power_w = None
        return {
            "t": time.perf_counter(), "phase": self.phase,
            "mem_used": mem.used,
            "sm_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
            "mem_mhz": n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
            "gpu_util": util.gpu, "power_w": power_w,
            "temp_c": n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
        }

    def _loop(self):
        # Windows drivers occasionally fail a query with NVMLError_Unknown;
        # drop that sample rather than the whole run.
        while not self._stop.is_set():
            try:
                self.samples.append(self._sample())
            except self._nvml.NVMLError:
                pass
            time.sleep(self.interval_s)

    def start(self):
        if self._nvml is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def idle_snapshot(self) -> dict | None:
        if self._nvml is None:
            return None
        try:
            return self._sample()
        except self._nvml.NVMLError:
            return None

    def summarize(self, phase: str, t0: float, t1: float) -> dict:
        rows = [s for s in self.samples if t0 <= s["t"] <= t1 and s["phase"] == phase]
        if not rows:
            return {}
        med = lambda k: statistics.median(r[k] for r in rows if r[k] is not None)
        return {
            "nvml_samples": len(rows),
            "nvml_mem_used_max_gib": max(r["mem_used"] for r in rows) / 2**30,
            "sm_mhz_median": med("sm_mhz"),
            "mem_mhz_median": med("mem_mhz"),
            "gpu_util_mean": statistics.fmean(r["gpu_util"] for r in rows),
            "power_w_median": (med("power_w")
                               if any(r["power_w"] is not None for r in rows) else None),
            "temp_c_max": max(r["temp_c"] for r in rows),
        }


class PhaseTracker:
    """Contiguous named phases; switching closes the previous one."""

    def __init__(self, sampler: NvmlSampler | None = None):
        self.sampler = sampler
        self.records: list[dict] = []
        self._cur: dict | None = None

    @property
    def current(self) -> str | None:
        return self._cur["phase"] if self._cur else None

    def switch(self, name: str | None):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._cur is not None:
            rec = self._cur
            rec["wall_s"] = now - rec.pop("_t0")
            rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            rec["allocated_end_gib"] = torch.cuda.memory_allocated() / 2**30
            rec["reserved_end_gib"] = torch.cuda.memory_reserved() / 2**30
            rec["_t1"] = now
            self.records.append(rec)
        self._cur = None
        if name is not None:
            torch.cuda.reset_peak_memory_stats()
            self._cur = {"phase": name, "_t0": now,
                         "allocated_start_gib": torch.cuda.memory_allocated() / 2**30}
            if self.sampler is not None:
                self.sampler.phase = name

    def finish(self) -> list[dict]:
        self.switch(None)
        out = []
        for rec in self.records:
            t1 = rec.pop("_t1")
            if self.sampler is not None:
                rec.update(self.sampler.summarize(rec["phase"], t1 - rec["wall_s"], t1))
            out.append(rec)
        self.records = []
        return out


@contextmanager
def instrument_runtime(rt, vio, model, tracker: PhaseTracker):
    """Patch corridorkey.runtime so run_inference reports phases to `tracker`.

    Low-VRAM phase order: setup, rvm, after_rvm, foundations,
    after_foundations, stage1, after_stage1, stage2, encode_tail (with
    --prefetch_frames the two prepasses are one phase, prepass).
    Normal mode has no foundations/stage2 phases: stage 2 runs inside
    stage1 through the on_baton_frames callback, so its GPU time is also
    accumulated separately from CUDA events around model.stage2.forward.
    """
    originals = {}
    stage1_events: list[tuple] = []
    stage2_events: list[tuple] = []
    state = {"encode_started": False}
    timings: dict = {"stage1_windows_ms": [], "stage2_forward_ms_total": 0.0}

    def wrap_phase(name):
        fn = getattr(rt, name)
        originals[name] = fn
        label = {"precompute_rvm_to_memmap": "rvm",
                 "precompute_foundations_to_memmap": "foundations",
                 "precompute_prepasses_to_memmap": "prepass",
                 "hann_stage1_stream": "stage1",
                 "_phase2_setup": "stage2"}[name]

        def wrapper(*args, **kwargs):
            tracker.switch(label)
            out = fn(*args, **kwargs)
            if label != "stage2":  # stage2 runs on in run_inference after setup
                tracker.switch(f"after_{label}")
            return out
        setattr(rt, name, wrapper)

    for name in ("precompute_rvm_to_memmap", "precompute_foundations_to_memmap",
                 "precompute_prepasses_to_memmap", "hann_stage1_stream", "_phase2_setup"):
        if hasattr(rt, name):
            wrap_phase(name)

    release_originals = {}
    for cls in (vio.ExrSequenceWriter, vio.ThreadedVideoWriter):
        release_originals[cls] = cls.release

        def make_release(orig):
            def release(self):
                if not state["encode_started"]:
                    state["encode_started"] = True
                    # If run_inference is unwinding from an exception, its
                    # finally block lands here; keep the phase that raised.
                    timings["phase_before_encode"] = tracker.current
                    tracker.switch("encode_tail")
                return orig(self)
            return release
        cls.release = make_release(cls.release)

    def timed_hooks(module, sink):
        # Calls made while a CUDA graph is being captured are not timed (their
        # events would belong to the graph); replays do not call the hooks, so
        # with --cuda_graph_stage2 only the stage2 phase time covers stage 2.
        def pre(_m, _inp):
            if torch.cuda.is_current_stream_capturing():
                sink.append(None)
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            sink.append([ev, None])

        def post(_m, _inp, _out):
            if sink and sink[-1] is None:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            sink[-1][1] = ev
        return (module.register_forward_pre_hook(pre),
                module.register_forward_hook(post))

    handles = timed_hooks(model.stage1, stage1_events) + timed_hooks(model.stage2, stage2_events)
    try:
        tracker.switch("setup")
        yield timings
    finally:
        for h in handles:
            h.remove()
        for name, fn in originals.items():
            setattr(rt, name, fn)
        for cls, fn in release_originals.items():
            cls.release = fn
        torch.cuda.synchronize()
        done = lambda evs: [(e[0], e[1]) for e in evs  # skip captures and forwards that raised
                            if e is not None and e[1] is not None]
        timings["stage1_windows_ms"] = [a.elapsed_time(b) for a, b in done(stage1_events)]
        timings["stage2_forward_ms_total"] = sum(a.elapsed_time(b) for a, b in done(stage2_events))
        timings["stage2_forwards"] = len(done(stage2_events))


@contextmanager
def block_memory_hooks(model, names_prefixes=("stage1.", "stage2.")):
    """Per-block peak allocator memory above the block's input state.

    Hooks every direct child of stage1/stage2, and every element of their
    ModuleLists (one row per transformer block). Each forward call is
    bracketed with a synchronize + reset_peak_memory_stats, so use it in a
    dedicated profiling run: it skews timing and clobbers PhaseTracker peaks.
    """
    rows: list[dict] = []
    handles = []
    targets = []
    for name, mod in model.named_modules():
        if not name.startswith(names_prefixes):
            continue
        parent = name.rsplit(".", 1)[0]
        depth = name.count(".")
        if isinstance(mod, torch.nn.ModuleList):
            continue
        parent_mod = model.get_submodule(parent) if "." in name else None
        if depth == 1 or (depth == 2 and isinstance(parent_mod, torch.nn.ModuleList)):
            targets.append((name, mod))
    for name, mod in targets:
        def pre(_m, _inp, name=name):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            rows.append({"block": name, "_base": torch.cuda.memory_allocated()})

        def post(_m, _inp, _out, name=name):
            torch.cuda.synchronize()
            row = next(r for r in reversed(rows) if r["block"] == name and "_base" in r)
            base = row.pop("_base")
            row["transient_peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2**30
            row["resident_delta_gib"] = (torch.cuda.memory_allocated() - base) / 2**30
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    try:
        yield rows
    finally:
        for h in handles:
            h.remove()


@contextmanager
def record_function_hooks(model, names_prefixes=("stage1.", "stage2.", "foundations.")):
    """torch.profiler ranges per block (depth <= 2), so kernels in a trace
    can be attributed to the module that launched them."""
    handles = []
    for name, mod in model.named_modules():
        if not name.startswith(names_prefixes) or name.count(".") > 2:
            continue

        def pre(m, _i, name=name):
            rf = torch.autograd.profiler.record_function(f"mod::{name}")
            rf.__enter__()
            m.__dict__.setdefault("_perf_rf", []).append(rf)

        def post(m, _i, _o):
            m.__dict__["_perf_rf"].pop().__exit__(None, None, None)
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def attribute_kernels(prof, top: int = 40) -> dict:
    """GPU kernel time per module, from a torch.profiler run that used
    record_function_hooks. Each kernel is charged to the nearest enclosing
    `mod::` range of the op that launched it. Attention kernels are tagged by
    backend so each attention type's actual dispatch is visible."""
    from collections import defaultdict

    def backend(kname: str) -> str | None:
        k = kname.lower()
        if "flash" in k:
            return "flash"
        if "fmha_cutlass" in k or "memeffattention" in k:
            return "mem_efficient"
        if "cudnn" in k and ("fmha" in k or "sdpa" in k or "attn" in k):
            return "cudnn"
        return None

    by_mod = defaultdict(float)
    by_mod_kernel = defaultdict(float)
    by_kernel = defaultdict(lambda: [0.0, 0])
    attn = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for evt in prof.events():
        kernels = getattr(evt, "kernels", None)
        if not kernels:
            continue
        mod, p = None, evt
        while p is not None:
            if p.name.startswith("mod::"):
                mod = p.name[5:]
                break
            p = p.cpu_parent
        mod = mod or "<outside modules>"
        for k in kernels:
            us = float(k.duration)
            total += us
            by_mod[mod] += us
            by_mod_kernel[(mod, k.name)] += us
            by_kernel[k.name][0] += us
            by_kernel[k.name][1] += 1
            b = backend(k.name)
            if b:
                attn[mod][b] += us
    group = defaultdict(float)  # e.g. stage1.body_64.3 -> stage1.body_64
    for mod, us in by_mod.items():
        parts = mod.split(".")
        group[".".join(parts[:2]) if len(parts) > 2 else mod] += us
    pct = lambda us: round(100.0 * us / total, 2) if total else 0.0
    return {
        "total_kernel_ms": total / 1000.0,
        "by_module_group": sorted(([g, us / 1000.0, pct(us)] for g, us in group.items()),
                                  key=lambda r: -r[1]),
        "by_module": sorted(([m, us / 1000.0, pct(us)] for m, us in by_mod.items()),
                            key=lambda r: -r[1])[:top],
        "top_kernels": sorted(([k, v[0] / 1000.0, v[1], pct(v[0])] for k, v in by_kernel.items()),
                              key=lambda r: -r[1])[:top],
        "top_module_kernels": sorted(([m, k, us / 1000.0, pct(us)]
                                      for (m, k), us in by_mod_kernel.items()),
                                     key=lambda r: -r[2])[:top],
        "attention_backend_ms_by_module": {m: {b: us / 1000.0 for b, us in d.items()}
                                           for m, d in sorted(attn.items())},
    }


@contextmanager
def nvtx_hooks(model, names_prefixes=("stage1.", "stage2.", "foundations.")):
    handles = []
    for name, mod in model.named_modules():
        if not name.startswith(names_prefixes) or name.count(".") > 2:
            continue

        # Hooks must return None: a pre-hook's return value replaces the
        # module's inputs and a forward hook's replaces its output, and
        # range_push/range_pop return ints.
        def pre(_m, _i, name=name):
            torch.cuda.nvtx.range_push(name)

        def post(_m, _i, _o):
            torch.cuda.nvtx.range_pop()
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
