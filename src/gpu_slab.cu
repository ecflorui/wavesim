// ============================================================================
// gpu_slab.cu  --  GPU half of the hybrid split (owner: Krish, built on
// Connor's GpuSolver kernels). See include/hybrid.hpp.
//
// Same kernels as src/gpu_solver.cu, run on a slab of nl owned layers stored
// with one ghost layer below and above (nl + 2 layers). The ghost layers play
// the role of walls for the kernel: it never writes them, and the hybrid
// solver refreshes them from the CPU slab after every step.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "hybrid.hpp"
#include "physics.hpp"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

namespace {

__global__ void slab_step_kernel(const float* cur, const float* prev, float* next,
                                 int nx, int ny, int nl, float r2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z + 1;     // local owned layers 1..nl
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k <= nl)
        next[idx(i, j, k, nx, ny)] = update_cell(cur, prev, i, j, k, nx, ny, r2);
}

// Identical to Connor's source_kernel: coincident sources are summed by one
// thread in CPU order, so there are no racing writes.
__global__ void slab_source_kernel(float* next, const Source* sources, int count,
                                   int nx, int ny, float t) {
    const int q = threadIdx.x;
    if (q >= count) return;
    const Source s = sources[q];
    const size_t cell = idx(s.i, s.j, s.k, nx, ny);
    for (int p = 0; p < q; ++p)
        if (idx(sources[p].i, sources[p].j, sources[p].k, nx, ny) == cell) return;
    float value = next[cell];
    for (int p = q; p < count; ++p)
        if (idx(sources[p].i, sources[p].j, sources[p].k, nx, ny) == cell)
            value += source_value(sources[p], t);
    next[cell] = value;
}

class GpuSlab : public Slab {
public:
    GpuSlab() = default;
    GpuSlab(const GpuSlab&) = delete;
    GpuSlab& operator=(const GpuSlab&) = delete;
    ~GpuSlab() override { release(); }

    void init(const SimConfig& cfg, int k0, int k1) override {
        release();
        cfg_ = cfg;
        k0_ = k0;
        nl_ = k1 - k0 + 1;
        n_ = 0;
        const size_t bytes = layer() * (size_t)(nl_ + 2) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_prev_, bytes));
        CUDA_CHECK(cudaMalloc(&d_cur_, bytes));
        CUDA_CHECK(cudaMalloc(&d_next_, bytes));
        CUDA_CHECK(cudaMemset(d_prev_, 0, bytes));
        CUDA_CHECK(cudaMemset(d_cur_, 0, bytes));
        CUDA_CHECK(cudaMemset(d_next_, 0, bytes));
        std::vector<Source> mine;                       // our sources, local k
        for (Source s : cfg.sources)
            if (s.k >= k0 && s.k <= k1) { s.k = s.k - k0 + 1; mine.push_back(s); }
        nsrc_ = (int)mine.size();
        if (nsrc_ > 0) {
            CUDA_CHECK(cudaMalloc(&d_src_, nsrc_ * sizeof(Source)));
            CUDA_CHECK(cudaMemcpy(d_src_, mine.data(), nsrc_ * sizeof(Source),
                                  cudaMemcpyHostToDevice));
        }
    }

    void step_begin() override {
        const dim3 block(32, 4, 4);
        const dim3 grid((cfg_.nx + block.x - 1) / block.x,
                        (cfg_.ny + block.y - 1) / block.y,
                        (nl_ + block.z - 1) / block.z);
        slab_step_kernel<<<grid, block>>>(d_cur_, d_prev_, d_next_, cfg_.nx, cfg_.ny, nl_,
                                          cfg_.r2());
        CUDA_CHECK(cudaGetLastError());
        if (nsrc_ > 0) {
            slab_source_kernel<<<1, nsrc_>>>(d_next_, d_src_, nsrc_, cfg_.nx, cfg_.ny,
                                             (n_ + 1) * cfg_.dt());
            CUDA_CHECK(cudaGetLastError());
        }
        float* old_prev = d_prev_;
        d_prev_ = d_cur_; d_cur_ = d_next_; d_next_ = old_prev;
        ++n_;
    }
    void step_end() override { CUDA_CHECK(cudaDeviceSynchronize()); }

    void get_layer(int global_k, float* out) override {
        CUDA_CHECK(cudaMemcpy(out, d_cur_ + local(global_k) * layer(), layer() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    void set_ghost(int global_k, const float* in) override {
        CUDA_CHECK(cudaMemcpy(d_cur_ + local(global_k) * layer(), in, layer() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    void copy_owned(float* field) override {
        CUDA_CHECK(cudaMemcpy(field + (size_t)k0_ * layer(), d_cur_ + layer(),
                              nl_ * layer() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const char* name() const override { return "gpu"; }

private:
    size_t layer() const { return (size_t)cfg_.nx * cfg_.ny; }
    size_t local(int global_k) const { return (size_t)(global_k - k0_ + 1); }
    void release() {
        if (d_prev_) CUDA_CHECK(cudaFree(d_prev_));
        if (d_cur_) CUDA_CHECK(cudaFree(d_cur_));
        if (d_next_) CUDA_CHECK(cudaFree(d_next_));
        if (d_src_) CUDA_CHECK(cudaFree(d_src_));
        d_prev_ = d_cur_ = d_next_ = nullptr;
        d_src_ = nullptr;
    }
    SimConfig cfg_;
    float *d_prev_ = nullptr, *d_cur_ = nullptr, *d_next_ = nullptr;
    Source* d_src_ = nullptr;
    int k0_ = 1, nl_ = 0, n_ = 0, nsrc_ = 0;
};

}  // namespace

std::unique_ptr<Slab> make_gpu_slab() { return std::make_unique<GpuSlab>(); }
