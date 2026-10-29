#!/usr/bin/env python3
"""Does one pre-reserved segment cut the default allocator's overhead?

Runs perf/bench.py with corridorkey.runtime.hann_stage1_stream patched to
empty the cache and reserve a single free segment of CK_ARENA_GIB GiB
(allocate, then free: the caching allocator keeps it) before stage 1
starts. Later large requests are carved out of that one segment, so
neighbouring blocks freed together coalesce into one gap, where separate
cudaMalloc segments of the same sizes never merge. Nothing about the
computation changes.

Usage:
    CK_ARENA_GIB=1.9 python perf/arena_probe.py <bench.py arguments>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf"))

import torch  # noqa: E402

from corridorkey import runtime as rt  # noqa: E402

_orig = rt.hann_stage1_stream


def _with_arena(*args, **kwargs):
    gib = float(os.environ.get("CK_ARENA_GIB", "0") or 0)
    if gib > 0:
        torch.cuda.empty_cache()
        block = torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda")
        del block
        print(f"[arena] reserved {gib:.3f} GiB, reserved now "
              f"{torch.cuda.memory_reserved() / 2**30:.3f} GiB", flush=True)
    return _orig(*args, **kwargs)


rt.hann_stage1_stream = _with_arena

import bench  # noqa: E402

if __name__ == "__main__":
    sys.argv = [str(ROOT / "perf/bench.py")] + sys.argv[1:]
    bench.main()
