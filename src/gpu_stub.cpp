// CPU-only builds (`make cpu`) link this instead of the CUDA files.
#include "solver.hpp"
#include "hybrid.hpp"
#include <stdexcept>

std::unique_ptr<Solver> make_gpu_solver() {
    throw std::runtime_error("this binary was built without CUDA (use `make gpu`)");
}

std::unique_ptr<Slab> make_gpu_slab() {
    throw std::runtime_error("the hybrid backend needs the GPU: build with `make gpu`");
}
