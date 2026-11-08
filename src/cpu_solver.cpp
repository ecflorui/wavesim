// ============================================================================
// cpu_solver.cpp  --  CPU reference solver (owner: Krish)
//
// Single-threaded, one-core baseline. This is the "answer key" the GPU is
// graded against, so it is written to be obviously correct, not fast.
//
// All of the math lives in include/physics.hpp (idx, update_cell,
// source_value). This file only decides WHICH cells to update and in what
// order -- exactly the same step the GPU performs:
//
//   1. Interior update   next = update_cell(cur, prev) for 1 <= i,j,k <= N-2
//   2. Walls             never written, stay 0 (reflecting, sign flip)
//   3. Sources           next[src] += source_value(src, (n+1)*dt)
//   4. Rotate            prev <- cur, cur <- next, next <- old prev (pointers)
//   5. Count             n++
// ============================================================================
#include <memory>
#include <vector>

#include "config.hpp"
#include "physics.hpp"
#include "solver.hpp"

namespace {

class CpuSolver : public Solver {
public:
    void init(const SimConfig& cfg) override {
        cfg_ = cfg;
        const size_t n = cfg_.n_cells();
        // assign() both sizes and zero-fills, so re-init on the same object
        // is safe and the field at step 0 is zero everywhere.
        buf_a_.assign(n, 0.0f);
        buf_b_.assign(n, 0.0f);
        buf_c_.assign(n, 0.0f);
        prev_ = buf_a_.data();
        cur_  = buf_b_.data();
        next_ = buf_c_.data();
        n_ = 0;
    }

    void step() override {
        const int nx = cfg_.nx, ny = cfg_.ny, nz = cfg_.nz;
        const float r2 = cfg_.r2();

        // 1. Interior update. Loop order k, j, i so i (x) is innermost and
        //    memory is walked contiguously (x is the fastest index).
        //    2. Walls: the i/j/k = 0 and N-1 cells are skipped on purpose.
        for (int k = 1; k <= nz - 2; ++k)
            for (int j = 1; j <= ny - 2; ++j)
                for (int i = 1; i <= nx - 2; ++i)
                    next_[idx(i, j, k, nx, ny)] =
                        update_cell(cur_, prev_, i, j, k, nx, ny, r2);

        // 3. Sources, AFTER the update, evaluated at the time of the new
        //    field: t = (n+1)*dt. Added (+=) so two sources can share a cell.
        const float t = (float)(n_ + 1) * cfg_.dt();
        for (const Source& s : cfg_.sources)
            next_[idx(s.i, s.j, s.k, nx, ny)] += source_value(s, t);

        // 4. Rotate the three buffers by swapping pointers (no copies).
        //    The old prev becomes the scratch "next"; every interior cell of it
        //    is overwritten next step and its walls are still 0.
        float* old_prev = prev_;
        prev_ = cur_;
        cur_  = next_;
        next_ = old_prev;

        // 5. Count.
        ++n_;
    }

    void copy_field_to_host(std::vector<float>& out) override {
        // cur_ always holds the newest field (after the rotate).
        out.assign(cur_, cur_ + cfg_.n_cells());
    }

    int         steps_done() const override { return n_; }
    const char* name()       const override { return "cpu"; }

private:
    SimConfig cfg_;
    std::vector<float> buf_a_, buf_b_, buf_c_;   // own the memory
    float* prev_ = nullptr;                       // field at step n-1
    float* cur_  = nullptr;                       // field at step n
    float* next_ = nullptr;                       // scratch for step n+1
    int n_ = 0;                                   // steps completed
};

}  // namespace

std::unique_ptr<Solver> make_cpu_solver() {
    return std::make_unique<CpuSolver>();
}
