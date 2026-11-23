#!/usr/bin/env python3
"""
tools/live_gui.py -- live browser view of the CPU wave simulation (owner: Krish)

    make gui                      # builds libwavesim_live.so
    python3 tools/live_gui.py     # then open http://localhost:8000

Runs the real C++ CpuSolver (through src/live_api.cpp) in a background thread
and serves a small web page that shows a 2D slice of the 3D pressure field as
it evolves. Python standard library only: no numpy or matplotlib needed.

On Lonestar6 (no screen), forward the port from your laptop first:
    ssh -L 8000:localhost:8000 <user>@ls6.tacc.utexas.edu
then run the two commands above and open http://localhost:8000 locally.
"""
import argparse
import base64
import ctypes
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(ROOT, "libwavesim_live.so")
PAGE_PATH = os.path.join(ROOT, "tools", "live_gui.html")

C = 343.0          # m/s, same as the configs
DX = 0.01          # m
STABLE_CFL = 1 / math.sqrt(3)
BLOWUP = 1e6       # treat |p| above this (or NaN/inf) as blown up
HISTORY = 600      # points kept for the max|p| chart


def load_lib():
    if not os.path.exists(LIB_PATH):
        sys.exit(f"{LIB_PATH} not found. Build it first:  make gui")
    lib = ctypes.CDLL(LIB_PATH)
    vp, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.ws_create.restype = vp
    lib.ws_create.argtypes = [i, i, i, f, f, f]
    lib.ws_destroy.argtypes = [vp]
    lib.ws_add_source.restype = i
    lib.ws_add_source.argtypes = [vp, i, i, i, i, f, f]
    lib.ws_clear_sources.argtypes = [vp]
    lib.ws_reset.argtypes = [vp]
    lib.ws_run.restype = ctypes.c_double
    lib.ws_run.argtypes = [vp, i]
    lib.ws_steps.restype = i
    lib.ws_steps.argtypes = [vp]
    lib.ws_dt.restype = f
    lib.ws_dt.argtypes = [vp]
    lib.ws_max_abs.restype = f
    lib.ws_max_abs.argtypes = [vp]
    lib.ws_slice.restype = i
    lib.ws_slice.argtypes = [vp, ctypes.c_char, i, ctypes.POINTER(f)]
    return lib


class Sim:
    """Owns one simulation. Every call into the C library holds self.lock."""

    def __init__(self, lib, n):
        self.lib = lib
        self.lock = threading.Lock()
        self.h = None
        self.n = n
        self.cfl = 0.5
        self.sources = []          # dicts: i j k type freq amp
        self.playing = False
        self.steps_per_frame = 2
        self.blew_up = False
        self.ms_per_step = 0.0
        self.history = []          # [step, max|p|]
        self.rebuild(keep_sources=False)

    # -- grid / sources ---------------------------------------------------
    def rebuild(self, keep_sources=True):
        if self.h:
            self.lib.ws_destroy(self.h)
        self.h = self.lib.ws_create(self.n, self.n, self.n, DX, C, self.cfl)
        old = self.sources if keep_sources else []
        self.sources = []
        mid = self.n // 2
        if not old:
            old = [dict(i=mid, j=mid, k=mid, type=0, freq=2000.0, amp=1.0)]
        for s in old:
            clamp = lambda v: max(1, min(self.n - 2, int(v)))
            self._add(dict(s, i=clamp(s["i"]), j=clamp(s["j"]), k=clamp(s["k"])))
        self.reset()

    def _add(self, s):
        if self.lib.ws_add_source(self.h, s["i"], s["j"], s["k"], s["type"], s["freq"], s["amp"]) == 0:
            self.sources.append(s)
            return True
        return False

    def reset(self):
        self.lib.ws_reset(self.h)
        self.blew_up = False
        self.history = [[0, 0.0]]

    # -- stepping ---------------------------------------------------------
    def advance(self, n):
        ms = self.lib.ws_run(self.h, n)
        self.ms_per_step = ms / n
        m = self.lib.ws_max_abs(self.h)
        step = self.lib.ws_steps(self.h)
        self.history.append([step, m if math.isfinite(m) else None])
        del self.history[:-HISTORY]
        if not math.isfinite(m) or m > BLOWUP:
            self.blew_up = True
            self.playing = False

    def loop(self):
        while True:
            with self.lock:
                running = self.playing and not self.blew_up
                if running:
                    self.advance(self.steps_per_frame)
            time.sleep(0.004 if running else 0.03)

    # -- state for the page ----------------------------------------------
    def state(self, axis, index):
        n = self.n
        if index < 0 or index >= n:
            index = n // 2
        buf = (ctypes.c_float * (n * n))()
        count = self.lib.ws_slice(self.h, axis.encode(), index, buf)
        step = self.lib.ws_steps(self.h)
        m = self.lib.ws_max_abs(self.h)
        return {
            "n": n, "axis": axis, "index": index,
            "slice": base64.b64encode(bytes(buf)[: 4 * max(count, 0)]).decode(),
            "step": step,
            "time_ms": step * self.lib.ws_dt(self.h) * 1e3,
            "max_abs": m if math.isfinite(m) else None,
            "ms_per_step": self.ms_per_step,
            "playing": self.playing, "blew_up": self.blew_up,
            "cfl": self.cfl, "stable_cfl": STABLE_CFL,
            "steps_per_frame": self.steps_per_frame,
            "sources": self.sources, "history": self.history,
        }

    def command(self, cmd):
        op = cmd.get("op")
        if op == "play":
            if self.blew_up:
                self.reset()
            self.playing = True
        elif op == "pause":
            self.playing = False
        elif op == "step":
            self.playing = False
            if not self.blew_up:
                self.advance(int(cmd.get("count", 1)))
        elif op == "reset":
            self.reset()
        elif op == "speed":
            self.steps_per_frame = max(1, min(50, int(cmd["value"])))
        elif op == "grid":
            self.n = max(9, min(160, int(cmd["n"])))
            self.rebuild()
        elif op == "cfl":
            self.cfl = max(0.05, min(0.75, float(cmd["value"])))
            self.rebuild()
        elif op == "clear":
            self.sources = []
            self.lib.ws_clear_sources(self.h)
            self.reset()
        elif op == "add_source":
            s = {"type": 1 if cmd.get("type") == "sine" else 0,
                 "freq": float(cmd.get("freq", 2000)), "amp": float(cmd.get("amp", 1))}
            axis, index, a, b = cmd["axis"], int(cmd["index"]), int(cmd["a"]), int(cmd["b"])
            # a = column, b = row of the clicked slice -> 3D cell
            s.update({"z": dict(i=a, j=b, k=index), "y": dict(i=a, j=index, k=b),
                      "x": dict(i=index, j=a, k=b)}[axis])
            if s["freq"] <= 0 or not self._add(s):
                return {"error": "pick an interior cell (not a wall); at most 8 sources"}
            self.reset()
        else:
            return {"error": f"unknown op {op!r}"}
        return {"ok": True}


def make_handler(sim):
    page = open(PAGE_PATH, "rb").read()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):    # keep the terminal quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/state":
                q = parse_qs(url.query)
                axis = q.get("axis", ["z"])[0]
                axis = axis if axis in ("x", "y", "z") else "z"
                index = int(q.get("index", ["-1"])[0])
                with sim.lock:
                    body = json.dumps(sim.state(axis, index)).encode()
                return self._send(200, body, "application/json")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/cmd":
                return self._send(404, b"not found", "text/plain")
            try:
                cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with sim.lock:
                    result = sim.command(cmd)
            except (ValueError, KeyError, TypeError) as e:
                result = {"error": str(e)}
            self._send(200, json.dumps(result).encode(), "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default 127.0.0.1; use 0.0.0.0 only if localhost doesn't reach it)")
    ap.add_argument("--n", type=int, default=64, help="starting grid size per axis (default 64)")
    args = ap.parse_args()

    sim = Sim(load_lib(), args.n)
    threading.Thread(target=sim.loop, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sim))
    print(f"WaveSim live view: open http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
