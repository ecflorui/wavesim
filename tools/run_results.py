#!/usr/bin/env python3
"""
tools/run_results.py -- every CPU/GPU/hybrid measurement in one go (owner: Krish)

Run on a Lonestar6 GPU node from the repository root, after `make gpu`:

    python3 tools/run_results.py            # about 5 minutes
    python3 tools/run_results.py --quick    # 1 repeat, no 256^3 (about 1 minute)

or submit tools/results.slurm from a login node. It runs:

  1. Correctness: the four reference configs on cpu, gpu and hybrid, and
     compare.py of gpu vs cpu and hybrid vs cpu (PASS = relative L2 < 1%).
  2. Speed-up: the same runs timed (median of the repeats). The CPU run is
     pinned to one core, as in bench.sh.
  3. Scaling: grid sizes 64^3 .. 256^3, 1000 steps, cpu vs gpu.
  4. Hybrid split: 128^3 with the CPU owning 1, 2, 4, 8 and 16 z-layers.

Raw numbers go to out/results/results.csv and out/results/summary.md, and the
tables between the <!-- results:... --> markers in docs/RESULTS.md are
replaced, so you can commit docs/RESULTS.md straight afterwards.
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "results")
REFS = [("tiny_33", 0.04977), ("one_source_64", 0.02026),
        ("two_source_64", 0.03303), ("bench_128", 0.02129)]


def pinned_core():
    try:
        return str(min(os.sched_getaffinity(0)))
    except AttributeError:
        return "0"


def run(cfg, backend, extra=()):
    """Run ./wavesim once. Returns dict with dir, max, init, steps, copy, total."""
    cmd = ["./wavesim", cfg, "--backend", backend, *extra]
    if backend == "cpu":
        cmd = ["taskset", "-c", pinned_core()] + cmd
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stdout}{p.stderr}")
    d = re.search(r"^run_dir=(.*)$", p.stdout, re.M).group(1)
    m = float(re.search(r"max \|p\| = (\S+)", p.stdout).group(1))
    init, steps, copy, total = map(float, open(os.path.join(ROOT, d, "timing.txt")).read().split())
    return dict(dir=d, max=m, init=init, steps=steps, copy=copy, total=total)


def timed(cfg, backend, repeats, extra=()):
    runs = [run(cfg, backend, extra) for _ in range(repeats)]
    med = lambda k: statistics.median(r[k] for r in runs)
    return dict(dir=runs[-1]["dir"], max=runs[-1]["max"],
                **{k: med(k) for k in ("init", "steps", "copy", "total")})


def compare(ref_dir, test_dir):
    p = subprocess.run([sys.executable, "tools/compare.py", ref_dir, test_dir],
                       cwd=ROOT, capture_output=True, text=True)
    text = (p.stdout + p.stderr).strip()
    m = re.search(r"relative_l2=(\S+)", text)
    return ("PASS" if p.returncode == 0 else "FAIL"), (float(m.group(1)) if m else None), text


def make_cfg(n, steps=1000):
    os.makedirs(os.path.join(OUT, "cfg"), exist_ok=True)
    path = os.path.join(OUT, "cfg", f"scale_{n}.cfg")
    c = n // 2
    with open(path, "w") as f:
        f.write(f"nx = {n}\nny = {n}\nnz = {n}\ndx = 0.01\nc = 343\ncfl = 0.5\n"
                f"steps = {steps}\nsnapshot_every = 0\noutput_dir = out/results/scale_{n}\n"
                f"source = {c} {c} {c} ricker 2000 1.0\n")
    return os.path.relpath(path, ROOT)


def fmt_err(e):
    return "-" if e is None else f"{e:.2e}"


def table(header, rows, align):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---:" if a == "r" else "---" for a in align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def replace_block(text, name, body):
    start, end = f"<!-- results:{name}:start -->", f"<!-- results:{name}:end -->"
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
    block = f"{start}\n{body}\n{end}"
    return pattern.sub(lambda _: block, text) if pattern.search(text) else text + "\n" + block + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1 repeat and no 256^3 run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu-layers", type=int, default=2, help="hybrid split for the correctness runs")
    ap.add_argument("--no-docs", action="store_true", help="don't rewrite docs/RESULTS.md")
    args = ap.parse_args()
    reps = 1 if args.quick else args.repeats
    os.makedirs(OUT, exist_ok=True)
    csv_rows, t0 = [], time.time()
    hybrid = ["--cpu-layers", str(args.cpu_layers)]

    try:
        run("configs/tiny_33.cfg", "gpu", ["--steps", "1"])
    except RuntimeError as e:
        sys.exit(f"The GPU backend does not run here. Build with `make gpu` on a GPU node.\n{e}")
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    gpu = gpu[0] if gpu else "unknown GPU"
    print(f"host {host}, {gpu}, CPU pinned to core {pinned_core()}, {reps} repeat(s)\n")

    # 1 + 2: correctness and speed-up on the reference configs ---------------
    print("== reference configs ==")
    corr_rows, speed_rows = [], []
    for name, ref in REFS:
        cfg = f"configs/{name}.cfg"
        cpu = timed(cfg, "cpu", reps)
        g = timed(cfg, "gpu", reps)
        verdict_g, err_g, _ = compare(cpu["dir"], g["dir"])
        try:
            h = timed(cfg, "hybrid", reps, hybrid)
            verdict_h, err_h, _ = compare(cpu["dir"], h["dir"])
        except RuntimeError as e:
            h, verdict_h, err_h = None, "ERROR", None
            print(f"  hybrid failed on {name}: {str(e).splitlines()[-1]}")
        cpu_ok = abs(cpu["max"] - ref) / ref < 5e-4
        corr_rows.append([f"`{name}`", f"{ref:.5f}", f"{cpu['max']:.5f}" + ("" if cpu_ok else " (!)"),
                          f"{g['max']:.5f}", f"{verdict_g} ({fmt_err(err_g)})",
                          f"{verdict_h} ({fmt_err(err_h)})"])
        speed_rows.append([f"`{name}`", f"{cpu['total']:.1f}", f"{g['total']:.1f}",
                           f"{cpu['total'] / g['total']:.1f}x", f"{cpu['steps']:.1f}", f"{g['steps']:.1f}",
                           f"{cpu['steps'] / g['steps']:.1f}x", f"{g['init']:.1f}",
                           f"{h['total']:.1f}" if h else "-"])
        for backend, r in (("cpu", cpu), ("gpu", g), ("hybrid", h)):
            if r:
                csv_rows.append(dict(section="reference", case=name, backend=backend, **{
                    k: r[k] for k in ("max", "init", "steps", "copy", "total")}))
        print(f"  {name:14s} gpu {verdict_g} {fmt_err(err_g)}  hybrid {verdict_h} {fmt_err(err_h)}  "
              f"cpu {cpu['total']:.0f} ms  gpu {g['total']:.0f} ms")

    # 3: scaling ---------------------------------------------------------------
    print("\n== scaling (1000 steps) ==")
    scale_rows = []
    for n in ([64, 128, 192] if args.quick else [64, 128, 192, 256]):
        cfg = make_cfg(n)
        cpu, g = timed(cfg, "cpu", reps), timed(cfg, "gpu", reps)
        verdict, err, _ = compare(cpu["dir"], g["dir"])
        cells = n ** 3
        scale_rows.append([f"{n}³", f"{cells / 1e6:.2f} M", f"{cpu['total']:.0f}", f"{g['total']:.0f}",
                           f"{cpu['total'] / g['total']:.1f}x", f"{cpu['steps'] / g['steps']:.1f}x",
                           f"{cells * 1000 / (g['steps'] / 1e3) / 1e9:.1f}",
                           f"{verdict} ({fmt_err(err)})"])
        for backend, r in (("cpu", cpu), ("gpu", g)):
            csv_rows.append(dict(section="scaling", case=f"{n}^3", backend=backend,
                                 **{k: r[k] for k in ("max", "init", "steps", "copy", "total")}))
        print(f"  {n:4d}^3  cpu {cpu['total']:8.0f} ms  gpu {g['total']:7.0f} ms  {verdict}")

    # 4: hybrid split ----------------------------------------------------------
    print("\n== hybrid split, 128^3 ==")
    hyb_rows = []
    cfg = "configs/bench_128.cfg"
    cpu, g = timed(cfg, "cpu", reps), timed(cfg, "gpu", reps)
    hyb_rows.append(["GPU only", "0", f"{g['steps']:.1f}", f"{g['total']:.1f}", "-"])
    for layers in [1, 2, 4, 8, 16]:
        try:
            h = timed(cfg, "hybrid", reps, ["--cpu-layers", str(layers)])
            verdict, err, _ = compare(cpu["dir"], h["dir"])
            hyb_rows.append([f"hybrid", f"{layers} of 126", f"{h['steps']:.1f}", f"{h['total']:.1f}",
                             f"{verdict} ({fmt_err(err)})"])
            csv_rows.append(dict(section="hybrid", case=f"cpu_layers={layers}", backend="hybrid",
                                 **{k: h[k] for k in ("max", "init", "steps", "copy", "total")}))
            print(f"  cpu_layers {layers:2d}: steps {h['steps']:8.1f} ms  {verdict}")
        except RuntimeError as e:
            hyb_rows.append(["hybrid", f"{layers} of 126", "-", "-", "ERROR"])
            print(f"  cpu_layers {layers}: failed: {str(e).splitlines()[-1]}")
    hyb_rows.append(["CPU only", "126 of 126", f"{cpu['steps']:.1f}", f"{cpu['total']:.1f}", "-"])

    # write everything ---------------------------------------------------------
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "case", "backend", "max", "init", "steps", "copy", "total"])
        w.writeheader()
        w.writerows(csv_rows)

    stamp = time.strftime("%Y-%m-%d")
    note = (f"Measured {stamp} on `{host}` ({gpu}) by `tools/run_results.py`, median of {reps} run(s). "
            "Times in ms. CPU pinned to one core. Total = init + steps + final copy; GPU init includes "
            "CUDA start-up.")
    blocks = {
        "correctness": note + "\n\n" + table(
            ["Config", "Reference", "CPU max", "GPU max", "GPU vs CPU", "Hybrid vs CPU"],
            corr_rows, "lrrrll"),
        "speedup": table(["Config", "CPU total", "GPU total", "Speed-up (total)", "CPU steps", "GPU steps",
                          "Speed-up (steps)", "GPU init", "Hybrid total"], speed_rows, "lrrrrrrrr"),
        "scaling": table(["Grid", "Cells", "CPU total", "GPU total", "Speed-up (total)",
                          "Speed-up (steps)", "GPU G cell-updates/s", "GPU vs CPU"], scale_rows, "lrrrrrrl"),
        "hybrid": table(["Backend", "CPU z-layers", "Steps (ms)", "Total (ms)", "vs CPU"], hyb_rows, "lrrrl"),
    }
    summary = "\n\n".join(f"### {k}\n\n{v}" for k, v in blocks.items())
    open(os.path.join(OUT, "summary.md"), "w").write(summary + "\n")
    if not args.no_docs:
        path = os.path.join(ROOT, "docs", "RESULTS.md")
        text = open(path).read()
        for k, v in blocks.items():
            text = replace_block(text, k, v)
        open(path, "w").write(text)
        print("\nupdated docs/RESULTS.md")
    print(f"wrote out/results/results.csv and out/results/summary.md in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
