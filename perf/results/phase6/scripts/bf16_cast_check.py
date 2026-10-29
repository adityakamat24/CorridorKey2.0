import sys
import numpy as np
import torch

lo, hi = 0, 0x3F800000 + 1
step = 1 << 26
mismatch = 0
for start in range(lo, hi, step):
    bits = torch.arange(start, min(hi, start + step), dtype=torch.int64).to(torch.int32)
    x = bits.view(torch.float32)
    cpu = x.to(torch.bfloat16).view(torch.int16)
    gpu = x.cuda().to(torch.bfloat16).cpu().view(torch.int16)
    mismatch += int((cpu != gpu).sum())
    neg = (-x)
    mismatch += int((neg.to(torch.bfloat16).view(torch.int16) != neg.cuda().to(torch.bfloat16).cpu().view(torch.int16)).sum())
print(f"float32 -> bfloat16, every value in [-1, 1]: {2 * (hi - lo)} values, CPU vs GPU mismatches {mismatch}")
sys.exit(1 if mismatch else 0)
