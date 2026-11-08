# Physics (owner: Krish)

All of the math is in `include/physics.hpp`. The CPU (`src/cpu_solver.cpp`) and the
GPU (`src/gpu_solver.cu`) both call the same functions, so they can only differ in
*how they loop*, never in *what they compute*.

## The equation

We solve the 3D scalar wave equation for pressure `p(x, y, z, t)`:

    ∂²p/∂t² = c² (∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²) + sources

`c` is the wave speed (343 m/s, sound in air). The grid spacing `dx` is the same on
all three axes, and time moves in steps of `dt`.

**Discretisation (leapfrog in time, 7-point Laplacian in space):**

    lap   = (p[i-1] + p[i+1]) + (p[j-1] + p[j+1]) + (p[k-1] + p[k+1]) - 6·p
    p_next = 2·p_cur - p_prev + r2 · lap,      r2 = (c·dt/dx)² = cfl²

Each new value needs the current field and the previous one, which is why the
solver keeps three buffers (`prev`, `cur`, `next`) and rotates the pointers after
every step instead of copying arrays. The neighbours are summed in a fixed order
(x pair, then y pair, then z pair) so the CPU and the GPU round the same way.

**Sources** are added to one cell *after* the interior update, at the time of the
new field, `t = (n+1)·dt`. A Ricker source is a single zero-mean pulse that peaks at
`t0 = 1/f`; a sine source is a continuous tone.

## The CFL limit

`cfl = c·dt/dx` is how many cells a wave moves per step. For this scheme in 3D it
must satisfy

    cfl ≤ 1/√3 ≈ 0.577

Intuition: in one step information can only travel one cell along each axis. If
the real wave would travel farther than the stencil can "see" (the worst case is the
cube diagonal, which is why the 3 appears), the numerical errors with the shortest
wavelength get multiplied by a factor larger than 1 every step and grow
exponentially. We use `cfl = 0.5`. `test_cpu` check (d) shows that `cfl = 0.6`
blows past 1e6 in about 50 steps of tiny_33, which is the instability, not a bug.

## Why the walls reflect

The boundary cells (index 0 and N-1 on every axis) are never written, so they stay
at `p = 0` forever (a "pressure-release" / Dirichlet wall). When a wave hits the
wall, the only way to keep the wall at zero is for a reflected wave of the *opposite
sign* to come back, so the pulse returns inverted. No energy leaves the box, which
is why check (c) sees the field stay bounded rather than decaying to zero.

## How we check it (`make test-cpu`)

| Check | What it proves |
|---|---|
| (a) no sources → all zeros after 100 steps | nothing appears from nothing |
| (b) centred source on tiny_33 → mirror and axis-swap symmetric to 1e-5·max\|p\| | the stencil treats ±x and x/y/z identically, indexing is right |
| (c) cfl 0.5, 1000 steps → finite, late peak ≤ early peak | the scheme is stable below the CFL limit |
| (d) cfl 0.6 → blows up | the CFL limit is real and the check above is meaningful |
| reference: max\|p\| after the last step | the GPU's target numbers |

Reference numbers (Ricker 2000 Hz, amplitude 1, cfl 0.5, dx 0.01):

| Config | Grid | Steps | Sources | max\|p\| |
|---|---|---|---|---|
| tiny_33 | 33³ | 200 | (16,16,16) | 0.04977 |
| one_source_64 | 64³ | 1000 | (32,32,32) | 0.02026 |
| two_source_64 | 64³ | 1000 | (20,32,32), (44,32,32) | 0.03303 |
| bench_128 | 128³ | 1000 | (64,64,64) | 0.02129 |

`./test_cpu --all-refs` reproduces all four.
