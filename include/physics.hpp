// ============================================================================
// physics.hpp  --  SHARED CONTRACT (owner: Krish)
// The ONLY place the math lives. CPU and GPU both call these functions, so
// they cannot disagree on the formula. Do not copy this math anywhere else.
// ============================================================================
#pragma once
#include <cstddef>
#include <cmath>
#include "config.hpp"

#ifdef __CUDACC__
  #define HD __host__ __device__
#else
  #define HD
#endif

// Memory layout: x fastest, then y, then z.  index = (k*ny + j)*nx + i
HD inline size_t idx(int i, int j, int k, int nx, int ny) {
    return ((size_t)k * ny + j) * nx + i;
}

// Value a source adds to its cell at time t (seconds).
//   Ricker: zero-mean pulse peaking at t0 = 1/freq
//   Sine:   continuous tone
HD inline float source_value(const Source& s, float t) {
    const float PI = 3.14159265358979f;
    if (s.type == SRC_RICKER) {
        float a = PI * s.freq * (t - 1.0f / s.freq);
        a = a * a;
        return s.amp * (1.0f - 2.0f * a) * expf(-a);
    }
    return s.amp * sinf(2.0f * PI * s.freq * t);
}

// New value of ONE interior cell (1 <= i,j,k <= n-2).  r2 = cfg.r2().
// Leapfrog finite difference for the 3D wave equation, 7-point Laplacian.
// Summation order is fixed (x pair, y pair, z pair) so CPU and GPU round alike.
HD inline float update_cell(const float* cur, const float* prev,
                            int i, int j, int k, int nx, int ny, float r2) {
    const size_t s  = idx(i, j, k, nx, ny);
    const size_t sy = (size_t)nx, sz = (size_t)nx * ny;
    const float  c0 = cur[s];
    const float lap = (cur[s - 1]  + cur[s + 1])
                    + (cur[s - sy] + cur[s + sy])
                    + (cur[s - sz] + cur[s + sz])
                    - 6.0f * c0;
    return 2.0f * c0 - prev[s] + r2 * lap;
}
