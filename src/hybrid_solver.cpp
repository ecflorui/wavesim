// ============================================================================
// hybrid_solver.cpp  --  CPU slab + the split solver (owner: Krish)
// See include/hybrid.hpp for the picture. The CPU slab does exactly what
// CpuSolver does, but on a slice of layers with ghost layers at each end.
// ============================================================================
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "hybrid.hpp"
#include "physics.hpp"

namespace {

class CpuSlab : public Slab {
public:
    void init(const SimConfig& cfg, int k0, int k1) override {
        cfg_ = cfg;
        k0_ = k0;
        nl_ = k1 - k0 + 1;
        const size_t n = layer() * (size_t)(nl_ + 2);
        a_.assign(n, 0.0f); b_.assign(n, 0.0f); c_.assign(n, 0.0f);
        prev_ = a_.data(); cur_ = b_.data(); next_ = c_.data();
        // Keep only our sources, shifted to local k.
        cfg_.sources.clear();
        for (Source s : cfg.sources)
            if (s.k >= k0 && s.k <= k1) { s.k = s.k - k0 + 1; cfg_.sources.push_back(s); }
        n_ = 0;
    }

    void step_begin() override {
        const int nx = cfg_.nx, ny = cfg_.ny;
        const float r2 = cfg_.r2();
        for (int k = 1; k <= nl_; ++k)                       // owned layers only
            for (int j = 1; j <= ny - 2; ++j)
                for (int i = 1; i <= nx - 2; ++i)
                    next_[idx(i, j, k, nx, ny)] = update_cell(cur_, prev_, i, j, k, nx, ny, r2);
        const float t = (float)(n_ + 1) * cfg_.dt();
        for (const Source& s : cfg_.sources)
            next_[idx(s.i, s.j, s.k, nx, ny)] += source_value(s, t);
        float* old_prev = prev_;
        prev_ = cur_; cur_ = next_; next_ = old_prev;
        ++n_;
    }
    void step_end() override {}

    void get_layer(int global_k, float* out) override {
        std::memcpy(out, cur_ + local(global_k) * layer(), layer() * sizeof(float));
    }
    void set_ghost(int global_k, const float* in) override {
        std::memcpy(cur_ + local(global_k) * layer(), in, layer() * sizeof(float));
    }
    void copy_owned(float* field) override {
        std::memcpy(field + (size_t)k0_ * layer(), cur_ + layer(), nl_ * layer() * sizeof(float));
    }
    const char* name() const override { return "cpu"; }

private:
    size_t layer() const { return (size_t)cfg_.nx * cfg_.ny; }
    size_t local(int global_k) const { return (size_t)(global_k - k0_ + 1); }
    SimConfig cfg_;
    std::vector<float> a_, b_, c_;
    float *prev_ = nullptr, *cur_ = nullptr, *next_ = nullptr;
    int k0_ = 1, nl_ = 0, n_ = 0;
};

class SplitSolver : public Solver {
public:
    SplitSolver(std::unique_ptr<Slab> lower, std::unique_ptr<Slab> upper, int lower_layers,
                const char* name)
        : lo_(std::move(lower)), hi_(std::move(upper)), lower_layers_(lower_layers), name_(name) {}

    void init(const SimConfig& cfg) override {
        cfg_ = cfg;
        const int interior = cfg.nz - 2;
        if (lower_layers_ < 1 || lower_layers_ > interior - 1)
            throw std::runtime_error("hybrid split: --cpu-layers must be between 1 and " +
                                     std::to_string(interior - 1) + " for nz = " +
                                     std::to_string(cfg.nz));
        split_ = lower_layers_;                 // lower owns 1..split_, upper owns split_+1..nz-2
        lo_->init(cfg, 1, split_);
        hi_->init(cfg, split_ + 1, interior);
        buf_lo_.assign((size_t)cfg.nx * cfg.ny, 0.0f);
        buf_hi_.assign((size_t)cfg.nx * cfg.ny, 0.0f);
        n_ = 0;
    }

    void step() override {
        hi_->step_begin();          // GPU starts (asynchronous)
        lo_->step_begin();          // CPU works meanwhile
        hi_->step_end();            // wait for the GPU
        lo_->get_layer(split_, buf_lo_.data());       // swap the two boundary layers
        hi_->get_layer(split_ + 1, buf_hi_.data());
        hi_->set_ghost(split_, buf_lo_.data());
        lo_->set_ghost(split_ + 1, buf_hi_.data());
        ++n_;
    }

    void copy_field_to_host(std::vector<float>& out) override {
        out.assign(cfg_.n_cells(), 0.0f);       // walls are 0
        lo_->copy_owned(out.data());
        hi_->copy_owned(out.data());
    }
    void sync() override { hi_->step_end(); lo_->step_end(); }
    int steps_done() const override { return n_; }
    const char* name() const override { return name_; }

private:
    std::unique_ptr<Slab> lo_, hi_;
    int lower_layers_, split_ = 0, n_ = 0;
    const char* name_;
    SimConfig cfg_;
    std::vector<float> buf_lo_, buf_hi_;
};

}  // namespace

std::unique_ptr<Slab> make_cpu_slab() { return std::make_unique<CpuSlab>(); }

std::unique_ptr<Solver> make_split_solver(std::unique_ptr<Slab> lower, std::unique_ptr<Slab> upper,
                                          int lower_layers, const char* name) {
    return std::make_unique<SplitSolver>(std::move(lower), std::move(upper), lower_layers, name);
}

std::unique_ptr<Solver> make_hybrid_solver(int cpu_layers) {
    return make_split_solver(make_cpu_slab(), make_gpu_slab(), cpu_layers, "hybrid");
}
