# GPU baseline (Connor / cjvb)

`src/gpu_solver.cu` implements the shared Solver interface using CUDA. The
solver and standalone test driver were compiled and run on Lonestar6 on
November 11, 2026 (job 3575998). This contribution preserves that tested solver.
The team's current config/runner pipeline has not yet been revalidated with it.

## Implementation

- Three zero-initialized float32 GPU fields: previous, current, next.
- A (32, 4, 4) thread block, one thread per cell, with interior bounds checks.
- The update kernel calls `update_cell()` from `physics.hpp`.
- A second kernel adds `source_value()` at `(n + 1) * dt` after each update.
- Sources at the same cell are accumulated by one thread in source order.
- Pointer rotation reuses the fields without copying arrays every step.
- Kernels use the same default stream; `step()` does not synchronize.
- `sync()` waits for GPU completion; final copying returns the current field.
- CUDA calls and launch errors are checked; allocations are freed on cleanup.

This is an unoptimized baseline: no shared-memory tiling, multi-GPU processing,
or hybrid CPU/GPU grid split. Correct input validation belongs to load_config().

## Reproduce standalone checks on an allocated A100 node

From the repository root, after obtaining a GPU allocation:

```bash
module load cuda
mkdir -p out/gpu-check
nvcc -O2 -std=c++17 -arch=sm_80 -Iinclude \
  src/gpu_solver.cu tools/test_gpu.cpp -o out/gpu-check/test_gpu
out/gpu-check/test_gpu --quick
out/gpu-check/test_gpu --full
out/gpu-check/test_gpu --bench
```

Alternatively, from the LS6 login node, submit:

```bash
mkdir -p out/gpu-check
sbatch tools/check_gpu.slurm
```

Use `sbatch -A YOUR_ALLOCATION tools/check_gpu.slurm` if an allocation must be
selected. The script saves a log under out/gpu-check using the job ID.

The standalone driver uses a small serial reference calling the shared physics;
it is not Krish's CpuSolver. It checks initialization, zero input, the first three
steps, non-cubic and partial-block grids, wall zeros, sine and Ricker sources,
coincident sources, eight sources, step counts, and reinitialization. `--full`
also compares the two 64-cubed cases. `--bench` measures only the GPU, checks
finiteness, and does not establish CPU agreement for the 128-cubed benchmark.

## Recorded standalone results

Hardware: one NVIDIA A100-PCIE-40GB, device 0. Compiler: CUDA 11.4.120.
Flags: `-O2 -std=c++17 -arch=sm_80 -Iinclude`. All requested diagnostics passed.
These are historical measurements from the returned job log, not new runs.

| Case | Grid | Steps | Relative L2 error | GPU max absolute field |
| --- | --- | ---: | ---: | ---: |
| tiny_33 | 33 x 33 x 33 | 200 | 9.6291645e-7 | 0.049770743 |
| one_source_64 | 64 x 64 x 64 | 1000 | 2.3547258e-6 | 0.020262033 |
| two_source_64 | 64 x 64 x 64 | 1000 | 2.3548831e-6 | 0.033025682 |

All nonzero cases passed the relative L2 threshold of 0.01; zero-field
diagnostics require an exactly zero difference instead of dividing by zero.

128-cubed, 1000 steps, one Ricker source:

| Initialization (ms) | Steps (ms) | Final copy (ms) | Total (ms) |
| ---: | ---: | ---: | ---: |
| 483.797 | 32.874 | 2.849 | 519.520 |

Final max absolute field: 0.021287825. Timers synchronize before stopping.
Initialization includes CUDA startup overhead. This is one measurement, not a
repeated-run performance study. No CPU/GPU speedup is claimed.

## Next: shared application integration

On a GPU compute node, use the real CPU solver, parser, output writer and checker:

```bash
module load cuda
make gpu
for case_name in tiny_33 one_source_64 two_source_64 bench_128; do
  bash tools/bench.sh "configs/${case_name}.cfg" || break
done
```

Record the resulting CPU/GPU errors and timings in docs/RESULTS.md only after
running these integrated tests. The standalone measurements above do not fill
the pending integrated result rows. Follow with repeated timings across grid
sizes and profiling before selecting optimizations.
