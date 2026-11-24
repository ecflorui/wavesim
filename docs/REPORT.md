# GPU-Accelerated 3D Waveform Simulator: final report (draft)

ECE 379K · Krish Singh, Connor McDonald, Rishik Boddeti, Venthan Dinesh · due Dec 6, 2026

> Draft. Sections marked **[fill in]** need numbers from `tools/run_results.py`
> (see docs/RESULTS.md) or a teammate's input. Everything else is backed by
> code and checks in this repository.

## 1. Goal

Simulate sound spreading through a closed 3D box, and show that a GPU version
runs at least 5× faster than one CPU core on a 128³ grid for 1,000 steps, while
staying within 1% relative error of the CPU answer. Timing includes setup and
data movement but not file output. The proposal's reduced scope kept a hybrid
CPU+GPU split, correctness checks and reproducible timings.

## 2. Method

**Physics.** The 3D scalar wave equation ∂²p/∂t² = c²∇²p + sources, with
c = 343 m/s on a grid with 1 cm spacing.

**Discretisation.** Leapfrog in time and a 7-point Laplacian in space:

    p_next = 2·p_now − p_prev + r2 · (sum of 6 neighbours − 6·p_now),   r2 = CFL²

The CFL number c·dt/dx is 0.5; the scheme is stable for CFL ≤ 1/√3 ≈ 0.577.
Walls are pinned at p = 0 (reflecting, sign-inverting). Sources are Ricker
pulses (2 kHz) or sine tones, added after the update at time (n+1)·dt.

**One shared math file.** `include/physics.hpp` holds the update and source
formulas, and both the CPU and GPU code call it, so the two backends can only
differ in how they loop, never in the formula. Neighbours are summed in a fixed
order so both round alike.

## 3. Implementation

| Part | Owner | Files |
|---|---|---|
| Physics, CPU reference solver, hybrid split, live view | Krish | `physics.hpp`, `cpu_solver.cpp`, `hybrid_solver.cpp`, `gpu_slab.cu`, `live_api.cpp`, `tools/live_gui.*` |
| GPU solver | Connor | `gpu_solver.cu` |
| Config format and visualisation | Rishik | `config.hpp`, `config.cpp`, `configs/`, `plot_slice.py` **[fill in: final ownership]** |
| Runner, output format, comparison, benchmark | Venthan | `main.cpp`, `output.cpp`, `compare.py`, `bench.sh` |

- **CPU reference:** one thread, three buffers rotated by pointer swap, loops
  in memory order (x fastest). It is both the answer key and the one-core baseline.
- **GPU:** one CUDA thread per cell, (32, 4, 4) blocks, a second small kernel
  adds sources, buffers stay on the GPU and rotate by pointer swap.
- **Hybrid:** the grid is split along z. The CPU owns the bottom N layers, the
  GPU the rest; both step at the same time and swap one boundary layer per step
  through ghost layers.
- **Runner:** times init, stepping and the final copy separately, calls
  `sync()` before stopping each timer, and excludes snapshot writes.

## 4. Correctness

- **CPU checks** (`make check`): a silent box stays exactly zero; a centred
  source is mirror- and axis-symmetric to within 1e-5 of the peak; 1,000 steps
  at CFL 0.5 stay bounded; CFL 0.6 blows past 10⁶ by step 52; all four
  reference peaks match to 4 digits through the real program.
- **Hybrid split logic:** with two CPU slabs, the split reproduces the CPU
  solver bit for bit for every split position tested.
- **GPU and hybrid against CPU:** **[fill in from RESULTS.md, correctness table]**.
  Connor's standalone check before integration gave relative L2 errors of
  about 1e-6.

## 5. Performance

**[fill in from RESULTS.md: speed-up table, scaling table, hybrid table]**

Points to discuss:
- Total vs stepping-only speed-up. The standalone GPU run spent 484 ms of
  520 ms on initialization (mostly CUDA start-up) and 33 ms stepping, so the
  answer to "5× faster" depends on whether start-up counts. The proposal
  defined timing to include setup. **[fill in: the measured numbers and the
  instructors' answer]**
- How the speed-up grows with grid size (GPU start-up is a fixed cost).
- Why the hybrid split does not help: one CPU core is about 100× slower per
  layer than the A100, and swapping layers costs about as much as a GPU step.

## 6. Visualisation

- `tools/plot_slice.py`: saved slices to PNG strips and GIFs, on one fixed
  colour scale.
- `tools/live_gui.py`: a browser view of the running solver with play/pause,
  grid size, CFL (including the unstable range), slice direction and position,
  and click-to-add sources.

## 7. What changed from the proposal

- Schedule: the checked CPU reference was planned for Oct 12 and landed
  Nov 7; the first GPU comparison moved from Oct 12 to Nov 11/23. **[fill in: why]**
- Hybrid split implemented, but as a measured result rather than a speed-up.
  **[fill in: if the instructors waived it]**
- Not done: shared-memory tiling and other GPU optimisations, absorbing walls,
  rooms other than boxes. **[fill in: anything done after Nov 24]**

## 8. Prior art and tools

- Finite-difference time-domain method: Yee (1966); acoustic FDTD applies it to pressure.
- Leapfrog time stepping and 7-point Laplacian: standard numerical PDE methods.
- CFL stability condition: Courant, Friedrichs and Lewy (1928).
- Ricker wavelet: Ricker (1940s), standard in seismology.
- GPU 3D stencils: NVIDIA CUDA samples (FDTD3d); Micikevicius, "3D Finite
  Difference Computation on GPUs using CUDA" (2009).
- Tools: C++17, CUDA, Python, NumPy, Matplotlib. An AI coding assistant
  (Claude) wrote parts of the code, including the CPU solver, tests, config
  reader, hybrid split, results script and live view; commits made from those
  sessions show "Claude" as the author. The team reviewed this code and can
  explain it. **[fill in: anything else the course requires you to disclose]**

## 9. How to reproduce

```bash
make cpu && make check                       # any Linux machine
module load cuda && make gpu                 # Lonestar6 GPU node
python3 tools/run_results.py                 # all comparisons and timings
```
