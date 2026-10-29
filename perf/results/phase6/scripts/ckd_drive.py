#!/usr/bin/env python3
"""Run one clip through corridorkey-engine (CorridorKey Desktop's engine) over
its JSON-lines protocol (docs/ENGINE_PROTOCOL.md): start, ping, one run, quit.
Prints the engine's replies that matter and the wall time from start to exit,
which is what their bench calls Total (model load included)."""

import json
import subprocess
import sys
import time

engine, models, ort, inp, out_dir, temp_dir = sys.argv[1:7]
opts = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
t0 = time.perf_counter()
p = subprocess.Popen([engine, "--models", models, "--ort", ort],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def send(obj):
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()


send({"cmd": "ping"})
build = opts.pop("_trt_build", None)
if build is not None:
    send({"cmd": "trt_build", "id": "j1", **build})
else:
    send({"cmd": "run", "id": "j1", "input": inp, "output_dir": out_dir, "temp_dir": temp_dir,
          "options": opts})
result = None
for line in p.stdout:
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    kind = msg.get("type")
    if kind in ("pong", "memory", "provider", "result", "error") or msg.get("id") == "j1" and build or (
            kind == "log" and any(w in str(msg.get("text", msg.get("message", ""))).lower()
                                  for w in ("window", "budget", "vram", "auto", "tensorrt", "fallback"))):
        print(json.dumps(msg)[:600], flush=True)
    if msg.get("id") == "j1" and (kind in ("result", "error") or build is not None and kind not in (
            "log", "progress", "status", "memory", "provider", "pong")):
        result = msg
        send({"cmd": "quit"})
        break
p.stdin.close()
p.wait()
wall = time.perf_counter() - t0
print(json.dumps({"wall_s": round(wall, 1), "ok": bool(result and result.get("type") == "result"),
                  "exit": p.returncode}), flush=True)
sys.exit(0 if result and result.get("type") == "result" else 1)
