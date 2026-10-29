#!/usr/bin/env bash
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 2400 > /dev/null 2>&1 < /dev/null &
cd /root/ck
export OMP_NUM_THREADS=12 OPENCV_FOR_THREADS_NUM=12
D=/root/arch2; mkdir -p $D/logs
python - > $D/sdpa.txt 2>&1 <<'PY'
import warnings
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
p = torch.cuda.get_device_properties(0)
print(p.name, f"sm_{p.major}{p.minor}", torch.__version__)
for dt in (torch.bfloat16, torch.float16):
    for hd in (32, 64, 128):
        q = torch.randn(2, 8, 512, hd, device="cuda", dtype=dt)
        row = []
        for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    with sdpa_kernel([be]):
                        F.scaled_dot_product_attention(q, q, q)
                    row.append(f"{be.name} ok")
                except RuntimeError:
                    why = " / ".join(str(x.message).strip().splitlines()[0] for x in w if "because" not in str(x.message)) or "no reason given"
                    row.append(f"{be.name} unavailable ({why[:160]})")
        print(str(dt).replace("torch.", ""), f"head_dim {hd}:", " | ".join(row))
PY
log "sdpa: $(grep -c unavailable $D/sdpa.txt) unavailable combos"
for v in orig opt orig2 opt2; do
  x=""; case $v in opt*) x="--optimized";; esac
  t0=$(date +%s.%N)
  python infer.py --input /root/clips/hair_detail --output_dir $D/t_$v --temp_dir $D/tt_$v --frame_dir_fps 24 --low_vram --output_format exr --exr_dtype float --outputs alpha fg $x > $D/logs/t_$v.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  log "timing --low_vram $v rc=$rc wall $(python -c "print(round($t1 - $t0, 1))") s"
  case $v in orig2|opt2) rm -rf $D/t_$v;; esac
  rm -rf $D/tt_$v
done
log "low_vram default chunk: original vs optimized: $(python perf/same_files.py $D/t_orig $D/t_opt | head -1)"
rm -rf $D/t_orig $D/t_opt
cd /root && tar czf /root/arch2.tgz arch2/sdpa.txt arch2/logs 2>/dev/null
log "extra done"
pkill -f '[s]elfstop.sh'; nohup bash /root/selfstop.sh 900 > /dev/null 2>&1 < /dev/null &
