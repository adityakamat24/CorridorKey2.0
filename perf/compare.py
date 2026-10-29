#!/usr/bin/env python3
"""Compare a candidate CorridorKey EXR render against a golden render.

Both directories are `infer.py --output_format exr --outputs alpha fg`
output dirs: <dir>/<clip>_alpha_exr/alpha_NNNNNN.exr and
<dir>/<clip>_fg_exr/fg_NNNNNN.exr. Every clip present in both is compared,
frame by frame (streaming; only the previous frame is kept for the temporal
metric).

Metrics per clip (alpha on its 0-1 scale):
  alpha_mae, alpha_max           |a_c - a_r| over all pixels and frames
  edge_mae, edge_max, edge_frac  same, inside the edge band: reference alpha
                                 in (0.02, 0.98), dilated by --band_px pixels
  fg_mae_lin                     |fg_c - fg_r| weighted by reference alpha,
                                 scene-linear, mean over RGB
  fg_mae_srgb                    same after clamp to [0, 1] and sRGB encode
                                 (what a viewer sees)
  flicker                        mean |(a_c,t - a_c,t-1) - (a_r,t - a_r,t-1)|
  worst_frames                   frames ranked by edge-band MAE

Usage:
    python perf/compare.py --ref perf/out/golden/hair_1080 \
        --cand perf/out/cand/hair_1080 --out perf/results/cmp_hair.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

EDGE_LO, EDGE_HI = 0.02, 0.98


def read_exr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32, copy=False)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def clips_in(dir_path: Path) -> dict[str, Path]:
    """Map clip name -> folder holding <clip>_alpha_exr (and <clip>_fg_exr).

    Accepts an infer.py output dir, or a perf/goldens.py tag dir with one
    output dir per clip.
    """
    found = {}
    for pattern in ("*_alpha_exr", "*/*_alpha_exr"):
        for p in sorted(dir_path.glob(pattern)):
            if p.is_dir():
                found.setdefault(p.name[:-len("_alpha_exr")], p.parent)
    return found


def edge_band(alpha_ref: np.ndarray, band_px: int) -> np.ndarray:
    band = ((alpha_ref > EDGE_LO) & (alpha_ref < EDGE_HI)).astype(np.uint8)
    if band_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2)
        band = cv2.dilate(band, k)
    return band.astype(bool)


def compare_clip(ref_dir: Path, cand_dir: Path, clip: str, *,
                 band_px: int, worst_n: int) -> dict:
    ref_alpha = sorted((ref_dir / f"{clip}_alpha_exr").glob("alpha_*.exr"))
    cand_alpha = sorted((cand_dir / f"{clip}_alpha_exr").glob("alpha_*.exr"))
    if len(ref_alpha) != len(cand_alpha):
        raise ValueError(f"{clip}: {len(ref_alpha)} golden vs "
                         f"{len(cand_alpha)} candidate alpha frames")
    has_fg = ((ref_dir / f"{clip}_fg_exr").is_dir()
              and (cand_dir / f"{clip}_fg_exr").is_dir())

    totals = dict(abs_sum=0.0, n=0, max=0.0, edge_sum=0.0, edge_n=0, edge_max=0.0,
                  fg_lin=0.0, fg_srgb=0.0, fg_w=0.0, flick_sum=0.0, flick_n=0)
    per_frame = []
    prev_r = prev_c = None
    for t, (pr, pc) in enumerate(zip(ref_alpha, cand_alpha)):
        a_r = read_exr(pr)
        a_c = read_exr(pc)
        if a_r.shape != a_c.shape:
            raise ValueError(f"{clip} frame {t}: shape {a_r.shape} vs {a_c.shape}")
        err = np.abs(a_c - a_r)
        band = edge_band(a_r, band_px)
        e_band = err[band]
        frame = {
            "t": t,
            "alpha_mae": float(err.mean()),
            "alpha_max": float(err.max()),
            "edge_mae": float(e_band.mean()) if e_band.size else 0.0,
            "edge_max": float(e_band.max()) if e_band.size else 0.0,
            "edge_frac": float(band.mean()),
        }
        totals["abs_sum"] += float(err.sum(dtype=np.float64))
        totals["n"] += err.size
        totals["max"] = max(totals["max"], frame["alpha_max"])
        totals["edge_sum"] += float(e_band.sum(dtype=np.float64))
        totals["edge_n"] += e_band.size
        totals["edge_max"] = max(totals["edge_max"], frame["edge_max"])

        if has_fg:
            name = pr.name.replace("alpha_", "fg_")
            fg_r = read_exr(ref_dir / f"{clip}_fg_exr" / name)
            fg_c = read_exr(cand_dir / f"{clip}_fg_exr" / name)
            w = np.clip(a_r, 0.0, 1.0)[..., None]
            lin = (np.abs(fg_c - fg_r) * w).sum(dtype=np.float64) / 3.0
            srgb = (np.abs(srgb_encode(fg_c) - srgb_encode(fg_r)) * w).sum(
                dtype=np.float64) / 3.0
            wsum = float(w.sum(dtype=np.float64))
            totals["fg_lin"] += float(lin)
            totals["fg_srgb"] += float(srgb)
            totals["fg_w"] += wsum
            frame["fg_mae_srgb"] = float(srgb / wsum) if wsum > 0 else 0.0

        if prev_r is not None:
            flick = np.abs((a_c - prev_c) - (a_r - prev_r))
            frame["flicker"] = float(flick.mean())
            totals["flick_sum"] += float(flick.sum(dtype=np.float64))
            totals["flick_n"] += flick.size
        prev_r, prev_c = a_r, a_c
        per_frame.append(frame)

    worst = sorted(per_frame, key=lambda f: (f["edge_mae"], f["alpha_max"]),
                   reverse=True)[:worst_n]
    return {
        "clip": clip,
        "frames": len(per_frame),
        "alpha_mae": totals["abs_sum"] / max(totals["n"], 1),
        "alpha_max": totals["max"],
        "edge_mae": totals["edge_sum"] / max(totals["edge_n"], 1),
        "edge_max": totals["edge_max"],
        "edge_frac": totals["edge_n"] / max(totals["n"], 1),
        "fg_mae_lin": (totals["fg_lin"] / totals["fg_w"]) if totals["fg_w"] else None,
        "fg_mae_srgb": (totals["fg_srgb"] / totals["fg_w"]) if totals["fg_w"] else None,
        "flicker": totals["flick_sum"] / max(totals["flick_n"], 1),
        "worst_frames": [{k: f[k] for k in ("t", "edge_mae", "alpha_max", "alpha_mae")}
                         for f in worst],
        "per_frame": per_frame,
    }


def fmt(x) -> str:
    return "-" if x is None else f"{x:.2e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="Golden output dir.")
    ap.add_argument("--cand", required=True, help="Candidate output dir.")
    ap.add_argument("--out", default=None, help="Write the full JSON here.")
    ap.add_argument("--band_px", type=int, default=3,
                    help="Edge-band dilation radius in pixels.")
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args()

    ref, cand = Path(args.ref), Path(args.cand)
    ref_clips, cand_clips = clips_in(ref), clips_in(cand)
    common = sorted(set(ref_clips) & set(cand_clips))
    if not common:
        sys.exit(f"no common *_alpha_exr clips between {ref} and {cand}")
    results = [compare_clip(ref_clips[c], cand_clips[c], c,
                            band_px=args.band_px, worst_n=args.worst)
               for c in common]

    print(f"{'clip':<24}{'frames':>7}{'a_mae':>10}{'a_max':>10}{'edge_mae':>10}"
          f"{'edge_max':>10}{'fg_srgb':>10}{'flicker':>10}  worst t")
    for r in results:
        print(f"{r['clip']:<24}{r['frames']:>7}{fmt(r['alpha_mae']):>10}"
              f"{fmt(r['alpha_max']):>10}{fmt(r['edge_mae']):>10}"
              f"{fmt(r['edge_max']):>10}{fmt(r['fg_mae_srgb']):>10}"
              f"{fmt(r['flicker']):>10}  "
              + ",".join(str(w["t"]) for w in r["worst_frames"]))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ref": str(ref), "cand": str(cand),
                                   "band_px": args.band_px, "clips": results},
                                  indent=2))


if __name__ == "__main__":
    main()
