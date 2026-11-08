// STUB (owner: Connor) -- linked by `make cpu` when CUDA is not available.
#include "solver.hpp"
#include <stdexcept>

std::unique_ptr<Solver> make_gpu_solver() {
    throw std::runtime_error("this binary was built without CUDA (use `make gpu`)");
}
