#!/usr/bin/env python3
"""Turn perf/vram_table.py JSON into a `_VRAM_PEAK_GB`-format Python literal.

Each entry is the measured global peak (max of the window, foundation and
stage-2 phases) in GiB. Windows the measuring GPU could not fit (out of
memory, or skipped after the first OOM) are extrapolated linearly from the
last two measured windows of the same resolution and listed in the trailing
comment, the same way the runtime extrapolates past the end of its table.

Usage:
    python perf/build_vram_table.py perf/results/phase3/vram_table_optimized.json
"""

from __future__ import annotations

import argparse
import json
import math


def build(tables: dict, mode: str):
    rows, extrapolated = [], []
    windows = sorted({int(t) for entry in tables[mode].values() for t in entry["windows"]})
    for name, entry in sorted(tables[mode].items(), key=lambda kv: kv[1]["megapixels"]):
        measured = {int(t): v["peak"] for t, v in entry["windows"].items()
                    if not math.isnan(v["peak"]) and not math.isnan(v["window"])}
        ts = sorted(measured)
        peaks = []
        for t in windows:
            if t in measured:
                peaks.append(measured[t])
                continue
            (t0, p0), (t1, p1) = [(x, measured[x]) for x in ts[-2:]]
            peaks.append(p1 + (t - t1) * (p1 - p0) / (t1 - t0))
            extrapolated.append(f"{name}@{t}")
        rows.append((round(entry["megapixels"], 2), tuple(round(p, 2) for p in peaks)))
    return tuple(windows), tuple(rows), extrapolated


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="vram_table.py outputs; later files win.")
    ap.add_argument("--name", default="_VRAM_PEAK_GB_OPTIMIZED")
    args = ap.parse_args()

    tables = {}
    for path in args.json:
        for mode, sizes in json.load(open(path))["tables"].items():
            tables.setdefault(mode, {}).update(sizes)
    print(f"{args.name} = {{")
    notes = []
    for low, mode in ((False, "normal"), (True, "lowvram")):
        if mode not in tables:
            continue
        windows, rows, extrapolated = build(tables, mode)
        print(f"    {low}: ({windows}, (")
        for mp, peaks in rows:
            print(f"        ({mp}, {peaks}),")
        print("    )),")
        if extrapolated:
            notes.append(f"{mode}: " + ", ".join(extrapolated))
    print("}")
    for note in notes:
        print(f"# Extrapolated (out of memory on the measuring GPU) - {note}")


if __name__ == "__main__":
    main()
