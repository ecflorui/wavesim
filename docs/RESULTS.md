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
Pending: run `tools/run_results.py` on a GPU node.
<!-- results:correctness:end -->

### Speed-up on the reference configs

<!-- results:speedup:start -->
Pending.
<!-- results:speedup:end -->

### Scaling with grid size (1000 steps)

<!-- results:scaling:start -->
Pending.
<!-- results:scaling:end -->

### Hybrid CPU+GPU split (128³, 1000 steps)

The CPU owns the bottom z-layers and the GPU the rest; they swap one boundary
layer each step (see `include/hybrid.hpp`).

<!-- results:hybrid:start -->
Pending.
<!-- results:hybrid:end -->

## Earlier standalone GPU check

Before integration, Connor's standalone A100 run (see [GPU.md](GPU.md))
measured relative L2 errors of about 1e-6 on the three nonzero cases, and
128³ × 1000 steps in 519.5 ms total (483.8 ms initialization including CUDA
start-up, 32.9 ms stepping).
