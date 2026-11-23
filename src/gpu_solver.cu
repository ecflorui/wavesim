// OWNER: Connor. Baseline GpuSolver (see plan, Connor steps 2-6).
// Kernels MUST call update_cell() and source_value() from physics.hpp.
#include "solver.hpp"
#include "physics.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

namespace {
__global__ void step_kernel(const float* cur, const float* prev, float* next,
                            int nx, int ny, int nz, float r2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    // Skip both walls and extra threads in rounded-up blocks.
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1)
        next[idx(i, j, k, nx, ny)] = update_cell(cur, prev, i, j, k, nx, ny, r2);
}

__global__ void source_kernel(float* next, const Source* sources, int count,
                              int nx, int ny, float t) {
    const int q = threadIdx.x;
    if (q >= count) return;
    const Source s = sources[q];
    const size_t cell = idx(s.i, s.j, s.k, nx, ny);
    // Coincident sources are legal. Their first thread adds all contributions
    // in CPU order; other threads at this cell return. No racing += writes.
    for (int p = 0; p < q; ++p)
        if (idx(sources[p].i, sources[p].j, sources[p].k, nx, ny) == cell) return;
    float value = next[cell];
    for (int p = q; p < count; ++p)
        if (idx(sources[p].i, sources[p].j, sources[p].k, nx, ny) == cell)
            value += source_value(sources[p], t);
    next[cell] = value;
}
} // namespace

class GpuSolver : public Solver {
public:
    GpuSolver() = default;
    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;
    ~GpuSolver() override { release(); }

    void init(const SimConfig& cfg) override {
        // load_config() owns validation under the shared contract.
        release();
        cfg_ = cfg;
        n_ = 0;
        int device = 0;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (!printed_device_) {
            std::printf("[gpu] device %d: %s\n", device, prop.name);
            printed_device_ = true;
        }
        const size_t bytes = cfg_.n_cells() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_prev_, bytes));
        CUDA_CHECK(cudaMalloc(&d_cur_, bytes));
        CUDA_CHECK(cudaMalloc(&d_next_, bytes));
        CUDA_CHECK(cudaMemset(d_prev_, 0, bytes));
        CUDA_CHECK(cudaMemset(d_cur_, 0, bytes));
        CUDA_CHECK(cudaMemset(d_next_, 0, bytes));
        // Empty sources are supported for the zero-field diagnostic.
        if (!cfg_.sources.empty()) {
            const size_t source_bytes = cfg_.sources.size() * sizeof(Source);
            CUDA_CHECK(cudaMalloc(&d_src_, source_bytes));
            CUDA_CHECK(cudaMemcpy(d_src_, cfg_.sources.data(), source_bytes,
                                  cudaMemcpyHostToDevice));
        }
    }

    void step() override {
        require_initialized();
        const dim3 block(32, 4, 4); // 512 threads; x varies fastest within a warp
        const dim3 grid((cfg_.nx + block.x - 1) / block.x,
                        (cfg_.ny + block.y - 1) / block.y,
                        (cfg_.nz + block.z - 1) / block.z);
        step_kernel<<<grid, block>>>(d_cur_, d_prev_, d_next_,
                                    cfg_.nx, cfg_.ny, cfg_.nz, cfg_.r2());
        CUDA_CHECK(cudaGetLastError());
        const int count = static_cast<int>(cfg_.sources.size());
        if (count > 0) {
            source_kernel<<<1, count>>>(d_next_, d_src_, count, cfg_.nx, cfg_.ny,
                                       (n_ + 1) * cfg_.dt());
            CUDA_CHECK(cudaGetLastError());
        }
        // Same stream orders the kernels. Launches capture pointer values,
        // so rotating host pointers needs no device synchronization.
        float* old_prev = d_prev_;
        d_prev_ = d_cur_;
        d_cur_ = d_next_;
        d_next_ = old_prev;
        ++n_;
    }

    void copy_field_to_host(std::vector<float>& out) override {
        require_initialized();
        out.resize(cfg_.n_cells());
        CUDA_CHECK(cudaMemcpy(out.data(), d_cur_, out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    void sync() override              { CUDA_CHECK(cudaDeviceSynchronize()); }
    int  steps_done() const override  { return n_; }
    const char* name() const override { return "gpu"; }
private:
    void require_initialized() const {
        if (!d_cur_) throw std::runtime_error("GpuSolver: call init() first");
    }
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
    int n_ = 0;
    bool printed_device_ = false;
};

std::unique_ptr<Solver> make_gpu_solver() { return std::make_unique<GpuSolver>(); }
