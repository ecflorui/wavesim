// ============================================================================
// hybrid.hpp  --  CPU+GPU split of one simulation (owner: Krish)
//
// Not a contract header: only main.cpp, live_api.cpp and the hybrid tests use
// it. The physics still lives only in physics.hpp.
//
// How the split works (z is the slowest axis, so a z-layer is contiguous):
//
//     global k:   0     1 .. L            L+1 .. nz-2        nz-1
//                wall   CPU slab           GPU slab           wall
//
// Each slab stores its own layers plus one GHOST layer on each side:
//
//     local k:    0        1 .. n          n+1
//               ghost    owned layers     ghost
//
// A ghost layer holds a copy of the neighbour's boundary layer (or the wall,
// which stays 0). One hybrid step:
//   1. both slabs update their own layers at the same time (GPU asynchronously)
//   2. wait for the GPU
//   3. swap boundary layers: CPU top owned layer -> GPU bottom ghost,
//                            GPU bottom owned layer -> CPU top ghost
// update_cell() only reads the CURRENT field of the six neighbours, so only
// the current field's ghosts need refreshing, once per step.
// ============================================================================
#pragma once
#include <memory>
#include "config.hpp"
#include "solver.hpp"

class Slab {
public:
    virtual ~Slab() = default;
    // Own global layers k0..k1 (inclusive, all interior). Field starts at 0.
    // Sources outside k0..k1 are ignored (the other slab owns them).
    virtual void init(const SimConfig& cfg, int k0, int k1) = 0;
    // Start one full step (update, sources, rotate). GPU: asynchronous.
    virtual void step_begin() = 0;
    // Wait until the step started by step_begin() is finished. CPU: no-op.
    virtual void step_end() = 0;
    // Copy owned layer global_k of the CURRENT field into out (nx*ny floats).
    virtual void get_layer(int global_k, float* out) = 0;
    // Overwrite a ghost layer (global k0-1 or k1+1) of the CURRENT field.
    virtual void set_ghost(int global_k, const float* in) = 0;
    // Write the owned layers of the CURRENT field into a full nx*ny*nz array.
    virtual void copy_owned(float* field) = 0;
    virtual const char* name() const = 0;
};

std::unique_ptr<Slab> make_cpu_slab();
std::unique_ptr<Slab> make_gpu_slab();   // src/gpu_slab.cu (stub in CPU-only builds)

// CPU owns layers 1..cpu_layers, the GPU owns the rest. Needs a CUDA build.
std::unique_ptr<Solver> make_hybrid_solver(int cpu_layers);

// Same split with any two slabs; used by the tests (two CPU slabs must give
// exactly the CpuSolver answer). lower owns 1..lower_layers.
std::unique_ptr<Solver> make_split_solver(std::unique_ptr<Slab> lower,
                                          std::unique_ptr<Slab> upper,
                                          int lower_layers, const char* name);
