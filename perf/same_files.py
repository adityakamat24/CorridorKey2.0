#!/usr/bin/env python3
"""Byte-level comparison of two infer.py output directories.

Hashes every file under each directory (videos, EXR frames) except the
settings logs, which record the output path, and reports files that differ
or exist on one side only. compare.py measures how much two mattes differ;
this checks an option leaves the written files byte-identical, including
encoded video, which compare.py does not read.

    python perf/same_files.py perf/out/vid_ref perf/out/vid_pinned
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def hashes(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith("_settings.txt"):
            out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("cand")
    args = ap.parse_args()
    ref, cand = hashes(Path(args.ref)), hashes(Path(args.cand))
    differ = [k for k in ref if k in cand and ref[k] != cand[k]]
    only = sorted(set(ref) ^ set(cand))
    print(f"{len(ref)} files in ref, {len(cand)} in cand: {len(ref) - len(differ) - len(set(ref) - set(cand))} "
          f"identical, {len(differ)} differ, {len(only)} on one side only")
    for k in differ[:20]:
        print(f"  differs: {k}")
    for k in only[:20]:
        print(f"  one side only: {k}")
    sys.exit(1 if differ or only or not ref else 0)


if __name__ == "__main__":
    main()
