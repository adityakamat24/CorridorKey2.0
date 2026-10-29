import json
import re
import sys
from collections import defaultdict

CATS = [
    ("attention: flash", r"flash"),
    ("attention: mem-efficient", r"fmha_cutlass|memeffattention"),
    ("attention: cudnn", r"cudnn.*(fmha|sdpa|attn)"),
    ("depthwise conv", r"conv_depthwise"),
    ("deconv (dgrad)", r"dgrad"),
    ("conv (cudnn/implicit)", r"fprop|implicit_convolve|conv2d|convolve|winograd|xmma_fprop"),
    ("gemm", r"gemm|s16816|cutlass.*tensorop|sm\d+_xmma|cublas|matmul"),
    ("memcpy", r"^Memcpy|^Memset"),
    ("copy/cast", r"direct_copy|copy_kernel|cat_|CatArray"),
    ("norm", r"norm"),
    ("softmax/reduce", r"softmax|reduce"),
    ("pad", r"reflection_pad|replication_pad|constant_pad"),
    ("elementwise", r"elementwise|vectorized|unrolled"),
]


def category(name):
    for cat, pat in CATS:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


def main(path, wall_s=None):
    d = json.load(open(path))
    total = d["total_kernel_ms"]
    print(f"{path}: total kernel time {total / 1000:.2f} s"
          + (f" of {wall_s:.1f} s wall ({100 * total / 1000 / wall_s:.0f}% busy)" if wall_s else ""))
    stages = defaultdict(float)
    for name, ms, _pct in d["by_module_group"]:
        stages[name.split(".")[0]] += ms
    print("  by stage: " + ", ".join(f"{k} {v / 1000:.2f} s ({100 * v / total:.0f}%)"
                                      for k, v in sorted(stages.items(), key=lambda kv: -kv[1])))
    print("  by module group:")
    for name, ms, pct in d["by_module_group"][:14]:
        print(f"    {name:<28} {ms / 1000:7.2f} s  {pct:5.1f}%")
    cats = defaultdict(float)
    for row in d["top_kernels"]:
        cats[category(row[0])] += row[1]
    covered = sum(cats.values())
    print(f"  top kernels by category ({100 * covered / total:.0f}% of kernel time listed):")
    for cat, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<26} {ms / 1000:7.2f} s  {100 * ms / total:5.1f}%")
    print("  top kernels:")
    for name, ms, calls, pct in d["top_kernels"][:15]:
        short = re.sub(r"[(<].*", "", name.replace("(anonymous namespace)::", "").replace("void ", ""))[:90]
        print(f"    {pct:5.1f}%  {ms / 1000:6.2f} s  {calls:6d}x  {short}")
    att = d.get("attention_backend_ms_by_module") or []
    if att:
        per = defaultdict(float)
        for mod, backends in (att.items() if isinstance(att, dict) else att):
            for b, ms in backends.items():
                per[(mod.split(".")[0] + "." + mod.split(".")[1] if "." in mod else mod, b)] += ms
        print("  attention by module group and backend:")
        for (mod, b), ms in sorted(per.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {mod:<28} {b:<14} {ms / 1000:6.2f} s")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
