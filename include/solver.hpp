// ============================================================================
// solver.hpp  --  SHARED CONTRACT (owners: Krish + Connor)
// CpuSolver and GpuSolver both implement exactly this. main.cpp only ever
// talks to a Solver*, so it cannot tell which backend it is running.
// ============================================================================
#pragma once
#include <memory>
#include <vector>
#include "config.hpp"

class Solver {
public:
    virtual ~Solver() = default;

    // Allocate 3 buffers (prev, cur, next) of cfg.n_cells() floats, all zero.
    // GPU: cudaMalloc + cudaMemset + copy cfg.sources to the device.
    virtual void init(const SimConfig& cfg) = 0;

    // Advance exactly one time step (see "The step" in the plan).
    virtual void step() = 0;

    // Advance n steps. Default just loops; GPU may override to avoid overhead.
    virtual void run(int n) { for (int s = 0; s < n; ++s) step(); }

    // Copy the CURRENT field to host memory. Resizes out to cfg.n_cells().
    virtual void copy_field_to_host(std::vector<float>& out) = 0;

    // Block until all queued work is finished. CPU: no-op.
    // GPU: cudaDeviceSynchronize(). main.cpp calls this before stopping a timer.
    virtual void sync() {}

    virtual int         steps_done() const = 0;   // 0 right after init()
    virtual const char* name()       const = 0;   // "cpu" or "gpu"
};

std::unique_ptr<Solver> make_cpu_solver();   // src/cpu_solver.cpp
std::unique_ptr<Solver> make_gpu_solver();   // src/gpu_solver.cu (or gpu_stub.cpp)
