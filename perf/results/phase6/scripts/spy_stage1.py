"""Aggregate a py-spy raw profile: main-thread samples inside a function,
grouped by the line of that function being executed (and the callee).

usage: spy_stage1.py profile.txt [func] [rate]
"""
import re
import sys
from collections import Counter

path = sys.argv[1]
func = sys.argv[2] if len(sys.argv) > 2 else "hann_stage1_stream"
rate = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
by_line, by_leaf, threads = Counter(), Counter(), Counter()
total_in = 0
for ln in open(path, encoding="utf-8", errors="replace"):
    ln = ln.rstrip("\n")
    if not ln:
        continue
    stack, _, n = ln.rpartition(" ")
    n = int(n)
    frames = stack.split(";")
    root = frames[0]
    threads["main" if "bench.py" in root else root[:60]] += n
    if "bench.py" not in root:
        continue
    idx = [i for i, f in enumerate(frames) if f.startswith(func + " ")]
    if not idx:
        continue
    i = idx[0]
    total_in += n
    here = frames[i]
    callee = frames[i + 1] if i + 1 < len(frames) else "<self>"
    m = re.search(r"\((.*):(\d+)\)", here)
    line = m.group(2) if m else "?"
    by_line[f"{func}:{line} -> {callee[:70]}"] += n
    by_leaf[frames[-1][:90]] += n
print(f"{total_in / rate:.1f} s of main-thread samples in {func}")
for k, v in by_line.most_common(25):
    print(f"  {v / rate:7.2f} s  {k}")
print("leaf frames:")
for k, v in by_leaf.most_common(20):
    print(f"  {v / rate:7.2f} s  {k}")
print("threads:", {k: round(v / rate, 1) for k, v in threads.most_common(8)})
