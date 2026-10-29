#!/usr/bin/env python3
"""Render reference (or candidate) EXR outputs for the test clips.

Runs `infer.py` once per clip with explicit, fixed Hann settings per
resolution and full-float EXR output, so later runs can be compared with
perf/compare.py without video-codec quantization in the way:

  1080p: --hann_chunk 8  --hann_stride 4  --low_vram   (6 GB default profile)
  720p:  --hann_chunk 24 --hann_stride 12 --low_vram

With --sdpa other than "default", with --fp32 (every network in fp32, the
reference for approximate options) or with --compile MODE (torch.compile),
the run goes through perf/infer_with.py (same runtime, process-wide
setting). Every command is appended to
<out>/commands.sh exactly as executed, and wall times to <out>/runs.jsonl.

Usage:
    python perf/goldens.py --tag golden --clips hair converge --res 1080 720
    python perf/goldens.py --tag golden_hint --clips hair converge --res 1080 --hint
    python perf/goldens.py --tag sdpa_efficient --sdpa efficient --clips hair
    python perf/goldens.py --tag fp32_ref --fp32 --clips hair smoke converge
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"1080": (8, 4), "720": (24, 12)}
FPS = 30


def build_cmd(clip: str, res: str, out_dir: Path, *, hint: bool, sdpa: str,
              temp_dir: Path, extra: list[str], fp32: bool = False,
              compile_mode: str | None = None) -> list[str]:
    chunk, stride = CONFIGS[res]
    clip_dir = ROOT / "test_clips" / f"{clip}_{res}"
    args = [
        "--input", str(clip_dir), "--output_dir", str(out_dir),
        "--output_format", "exr", "--exr_dtype", "float",
        "--outputs", "alpha", "fg",
        "--hann_chunk", str(chunk), "--hann_stride", str(stride), "--low_vram",
        "--frame_dir_fps", str(FPS), "--temp_dir", str(temp_dir),
    ]
    if hint:
        args += ["--hint_first_frame",
                 str(ROOT / "test_clips" / f"{clip}_{res}_hint.png"),
                 "--hint_quality", "0.75"]
    args += extra
    if sdpa == "default" and not fp32 and not compile_mode:
        return [sys.executable, str(ROOT / "infer.py")] + args
    return ([sys.executable, str(ROOT / "perf/infer_with.py"), "--sdpa", sdpa]
            + (["--fp32"] if fp32 else [])
            + (["--compile", compile_mode] if compile_mode else []) + ["--"] + args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="Output folder name under perf/out/.")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--res", nargs="+", default=["1080"], choices=list(CONFIGS))
    ap.add_argument("--hint", action="store_true",
                    help="First-frame hint run (test_clips/<clip>_<res>_hint.png).")
    ap.add_argument("--sdpa", default="default",
                    choices=("default", "flash", "efficient", "cudnn", "math"))
    ap.add_argument("--fp32", action="store_true",
                    help="Every network in fp32 (perf/infer_with.py --fp32).")
    ap.add_argument("--compile", default=None,
                    help="torch.compile mode (perf/infer_with.py --compile).")
    ap.add_argument("--out_root", default=str(ROOT / "perf/out"))
    ap.add_argument("--temp_dir", default=None,
                    help="Memmap temp dir. Default perf/out/_tmp/<tag>; runs of "
                         "the same clip must not share one (file names are per clip).")
    ap.add_argument("--extra", default="",
                    help="Extra infer.py arguments, as one quoted string.")
    args = ap.parse_args()

    out_root = Path(args.out_root) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir) if args.temp_dir else Path(args.out_root) / "_tmp" / args.tag
    runs_path = out_root / "runs.jsonl"  # append-only: several streams may share a tag
    for res in args.res:
        for clip in args.clips:
            out_dir = out_root / f"{clip}_{res}"
            cmd = build_cmd(clip, res, out_dir, hint=args.hint, sdpa=args.sdpa,
                            temp_dir=temp_dir, extra=shlex.split(args.extra), fp32=args.fp32,
                            compile_mode=args.compile)
            line = " ".join(shlex.quote(c) for c in cmd)
            with open(out_root / "commands.sh", "a") as f:
                f.write(line + "\n")
            print(f"[{args.tag}] {clip}_{res}: {line}", flush=True)
            t0 = time.perf_counter()
            log_path = out_root / f"{clip}_{res}.log"
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            wall = time.perf_counter() - t0
            with open(runs_path, "a") as f:
                f.write(json.dumps({"clip": clip, "res": res, "hint": args.hint,
                                    "sdpa": args.sdpa, "fp32": args.fp32,
                                    "compile": args.compile,
                                    "returncode": proc.returncode,
                                    "wall_s": round(wall, 2), "cmd": line}) + "\n")
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"[{args.tag}] {clip}_{res}: {status} in {wall:.1f}s (log {log_path})",
                  flush=True)
            if proc.returncode != 0:
                sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
