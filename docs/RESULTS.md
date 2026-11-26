# WaveSim results

## CPU validation

The CPU sanity tests passed on commit `4d346f1`. The built-in `tiny_33` case
completed 200 steps and reached a peak absolute pressure of `0.0497707`.
Its `final.bin` contained 143,748 bytes, as expected for a 33 x 33 x 33
float32 field. Comparing the CPU field against itself returned zero relative
L2 error and zero maximum absolute difference.

Since then the real config reader landed; `make check` runs every reference
config through `./wavesim` and all four match the reference values (see
[CPU.md](CPU.md)).

## Lonestar6 CPU / GPU / hybrid results

The tables below are written by `tools/run_results.py` (run it on a GPU
compute node after `make gpu`, or submit `tools/results.slurm`). Each table is
replaced between its markers on every run, so don't edit inside them by hand.
Speed-up = CPU time / GPU time; "total" includes setup and the final copy,
"steps" is the stepping time alone. The CPU run is pinned to one core.

### Correctness (relative L2 error against the CPU, limit 0.01)

<!-- results:correctness:start -->
Measured 2026-11-26 on `c301-004.ls6.tacc.utexas.edu` (NVIDIA A100-PCIE-40GB) by `tools/run_results.py`, median of 3 run(s). Times in ms. CPU pinned to one core. Total = init + steps + final copy; GPU init includes CUDA start-up.

| Config | Reference | CPU max | GPU max | GPU vs CPU | Hybrid vs CPU |
| --- | ---: | ---: | ---: | --- | --- |
| `tiny_33` | 0.04977 | 0.04977 | 0.04977 | PASS (9.63e-07) | PASS (9.16e-07) |
| `one_source_64` | 0.02026 | 0.02026 | 0.02026 | PASS (2.35e-06) | PASS (2.36e-06) |
| `two_source_64` | 0.03303 | 0.03303 | 0.03303 | PASS (2.35e-06) | PASS (2.34e-06) |
| `bench_128` | 0.02129 | 0.02129 | 0.02129 | PASS (2.58e-06) | PASS (2.58e-06) |
<!-- results:correctness:end -->

### Speed-up on the reference configs

<!-- results:speedup:start -->
| Config | CPU total | GPU total | Speed-up (total) | CPU steps | GPU steps | Speed-up (steps) | GPU init | Hybrid total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny_33` | 8.6 | 436.9 | 0.02x | 8.4 | 2.0 | 4.2x | 434.9 | 443.4 |
| `one_source_64` | 332.0 | 472.7 | 0.7x | 330.5 | 12.6 | 26.1x | 460.0 | 493.9 |
| `two_source_64` | 332.7 | 466.2 | 0.7x | 330.7 | 12.6 | 26.2x | 453.5 | 490.8 |
| `bench_128` | 2819.4 | 494.8 | 5.7x | 2812.1 | 33.1 | 85.0x | 458.2 | 542.8 |
<!-- results:speedup:end -->

### Scaling with grid size (1000 steps)

<!-- results:scaling:start -->
| Grid | Cells | CPU total | GPU total | Speed-up (total) | Speed-up (steps) | GPU G cell-updates/s | GPU vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64³ | 0.26 M | 331 | 468 | 0.7x | 26.9x | 21.4 | PASS (2.35e-06) |
| 128³ | 2.10 M | 2822 | 489 | 5.8x | 85.5x | 63.7 | PASS (2.58e-06) |
| 192³ | 7.08 M | 9671 | 548 | 17.6x | 100.5x | 73.7 | PASS (2.50e-06) |
| 256³ | 16.78 M | 25119 | 575 | 43.7x | 143.1x | 95.8 | PASS (2.51e-06) |
<!-- results:scaling:end -->

### Hybrid CPU+GPU split (128³, 1000 steps)

The CPU owns the bottom z-layers and the GPU the rest; they swap one boundary
layer each step (see `include/hybrid.hpp`).

<!-- results:hybrid:start -->
| Backend | CPU z-layers | Steps (ms) | Total (ms) | vs CPU |
| --- | ---: | ---: | ---: | --- |
| GPU only | 0 | 33.1 | 490.6 | - |
| hybrid | 1 of 126 | 70.8 | 527.6 | PASS (2.57e-06) |
| hybrid | 2 of 126 | 81.9 | 541.3 | PASS (2.58e-06) |
| hybrid | 4 of 126 | 127.7 | 583.0 | PASS (2.57e-06) |
| hybrid | 8 of 126 | 219.1 | 674.0 | PASS (2.59e-06) |
| hybrid | 16 of 126 | 401.7 | 862.7 | PASS (2.60e-06) |
| CPU only | 126 of 126 | 2812.2 | 2819.1 | - |
<!-- results:hybrid:end -->

## What the numbers say

- **Correct everywhere.** GPU and hybrid match the CPU to about 2.6e-6
  relative L2 error on every case, nearly 4,000 times under the 1% limit.
- **Target met: 5.7× at 128³** (total time, CUDA start-up included). In the
  simulation loop alone the GPU is 85× faster.
- **Start-up dominates small runs.** About 0.46 s of GPU initialization is a
  fixed cost (93% of the GPU's total time at 128³). It acts like the serial
  fraction in Amdahl's law: 64³ is slower on the GPU (0.7×), 256³ is 43.7×.
- **The GPU also gets more efficient as the grid grows** (21 → 96 billion
  cell updates per second from 64³ to 256³), so stepping-only speed-up rises
  from 27× to 143×.
- **Hybrid is correct but slower than GPU alone.** Each CPU layer adds time
  (33 ms GPU-only stepping → 71 ms with one CPU layer → 402 ms with 16),
  because one core is about 85× slower than the A100 and both sides must meet
  every step.

## Earlier standalone GPU check

Before integration, Connor's standalone A100 run (see [GPU.md](GPU.md))
measured relative L2 errors of about 1e-6 on the three nonzero cases, and
128³ × 1000 steps in 519.5 ms total (483.8 ms initialization including CUDA
start-up, 32.9 ms stepping).
