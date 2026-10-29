"""Run infer.py in-process and report what the machine actually did: free
VRAM at start, then (after infer.py's own output, which includes the window
auto mode picked) peak allocated/reserved and the wall time, model load
included. Written for the Windows laptop runs in REPORT.md.

Usage:
    python perf/laptop_run.py <infer.py args...>
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

import infer  # noqa: E402

free, total = torch.cuda.mem_get_info()
print(f"[laptop] {torch.cuda.get_device_name(0)}: {free / 2**30:.2f} GiB free of "
      f"{total / 2**30:.2f} GiB at start", flush=True)
sys.argv = [os.path.join(ROOT, "infer.py")] + sys.argv[1:]
t0 = time.perf_counter()
try:
    infer.main()
    status = "ok"
except torch.OutOfMemoryError as e:
    status = f"OOM: {str(e).splitlines()[0]}"
dt = time.perf_counter() - t0
print(f"[laptop] {status}; peak allocated {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB, "
      f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.3f} GiB, wall {dt:.1f} s",
      flush=True)
