// Standalone diagnostics, not a replacement for Krish's solver or Venthan's tests.
// Build with src/gpu_solver.cu only; no parser/output/CPU stubs are needed.
#include "solver.hpp"
#include "physics.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Serial oracle reuses the contract math, including source addition order.
struct Reference {
    SimConfig cfg;
    std::vector<float> prev, cur, next;
    int n = 0;
    explicit Reference(const SimConfig& c) : cfg(c), prev(c.n_cells()),
        cur(c.n_cells()), next(c.n_cells()) {}
    void step() {
        for (int k = 1; k < cfg.nz - 1; ++k)
            for (int j = 1; j < cfg.ny - 1; ++j)
                for (int i = 1; i < cfg.nx - 1; ++i)
                    next[idx(i,j,k,cfg.nx,cfg.ny)] =
                        update_cell(cur.data(), prev.data(), i,j,k,cfg.nx,cfg.ny,cfg.r2());
        for (const auto& s : cfg.sources)
            next[idx(s.i,s.j,s.k,cfg.nx,cfg.ny)] += source_value(s, (n+1)*cfg.dt());
        prev.swap(cur);
        cur.swap(next);
        ++n;
    }
};

static void compare(const SimConfig& c, const std::vector<float>& ref,
                    const std::vector<float>& got, const char* label) {
    require(got.size() == ref.size(), "wrong output size");
    double error2 = 0, norm2 = 0;
    float maximum = 0, max_diff = 0;
    for (int k = 0; k < c.nz; ++k)
        for (int j = 0; j < c.ny; ++j)
            for (int i = 0; i < c.nx; ++i) {
                const size_t q = idx(i,j,k,c.nx,c.ny);
                require(std::isfinite(got[q]) && std::isfinite(ref[q]), "nonfinite field");
                const double d = double(got[q]) - ref[q];
                error2 += d*d;
                norm2 += double(ref[q])*ref[q];
                maximum = std::max(maximum, std::fabs(got[q]));
                max_diff = std::max(max_diff, float(std::fabs(d)));
                if (i == 0 || j == 0 || k == 0 || i == c.nx-1 || j == c.ny-1 || k == c.nz-1)
                    require(got[q] == 0, "wall changed");
            }
    // Zero-norm diagnostics use exact zero; production comparison rejects zero norm.
    const double rel = norm2 > 0 ? std::sqrt(error2/norm2) : 0;
    require(norm2 > 0 ? rel < 0.01 : error2 == 0, "CPU/GPU mismatch");
    std::printf("PASS %-20s rel=%.8g max_diff=%.8g max|p|=%.8g\n",
                label, rel, max_diff, maximum);
}

static SimConfig config(int nx, int ny, int nz) {
    SimConfig c;
    c.nx=nx; c.ny=ny; c.nz=nz;
    c.sources={{nx/2,ny/2,nz/2,SRC_RICKER,2000.0f,1.0f}};
    return c;
}

static void check(Solver& gpu, const SimConfig& c, int steps, const char* label,
                  bool every_step=false) {
    gpu.init(c); // Reusing this instance also exercises cleanup and re-init.
    require(gpu.steps_done() == 0, "init did not reset step count");
    Reference ref(c);
    std::vector<float> field;
    gpu.copy_field_to_host(field);
    compare(c, ref.cur, field, "initial zeros");
    if (every_step) {
        for (int n=0; n<steps; ++n) {
            gpu.step(); ref.step();
            require(gpu.steps_done() == n+1, "wrong step count");
            gpu.copy_field_to_host(field); // Also tests copying without explicit sync.
            compare(c, ref.cur, field, label);
        }
    } else {
        gpu.run(steps);
        gpu.sync();
        for (int n=0; n<steps; ++n) ref.step();
        gpu.copy_field_to_host(field);
        compare(c, ref.cur, field, label);
    }
    require(gpu.steps_done() == steps, "wrong final step count");
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "--quick";
        require(mode == "--quick" || mode == "--full" || mode == "--bench", "use --quick, --full or --bench");
        auto gpu = make_gpu_solver();
        if (mode == "--bench") {
            const SimConfig c=config(128,128,128);
            using Clock=std::chrono::steady_clock;
            auto t0=Clock::now(); gpu->init(c); gpu->sync(); auto t1=Clock::now();
            gpu->run(1000); gpu->sync(); auto t2=Clock::now();
            std::vector<float> field;
            gpu->copy_field_to_host(field); gpu->sync(); auto t3=Clock::now();
            auto ms=[](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
            float maximum=0;
            for (float v:field) { require(std::isfinite(v), "nonfinite benchmark field"); maximum=std::max(maximum,std::fabs(v)); }
            std::printf("128^3 / 1000 steps: GPU init_ms=%.3f steps_ms=%.3f copy_ms=%.3f total_ms=%.3f max|p|=%.8g\n",
                        ms(t0,t1),ms(t1,t2),ms(t2,t3),ms(t0,t3),maximum);
            std::puts("GPU-only timing; obtain team CPU timing and comparison after merging.");
            return 0;
        }
        SimConfig c=config(11,9,13); // Non-cubic and not divisible by block dimensions.
        c.sources.clear(); check(*gpu,c,100,"no sources");
        c=config(11,9,13);
        check(*gpu,c,3,"steps 1,2,3",true);
        c.sources={{1,1,1,SRC_SINE,1000,1}, {9,7,11,SRC_RICKER,2000,1}};
        check(*gpu,c,25,"near walls / sine");
        c.sources={{5,4,6,SRC_RICKER,2000,1}, {2,2,2,SRC_SINE,1000,0.2f},
                   {5,4,6,SRC_SINE,1000,-0.25f}};
        check(*gpu,c,25,"coincident sources");
        c.sources.clear();
        for(int q=0;q<MAX_SOURCES;++q) c.sources.push_back({5,4,6,SRC_RICKER,2000,0.125f});
        check(*gpu,c,25,"eight sources");
        check(*gpu,config(33,33,33),200,"tiny_33");
        if(mode == "--full") {
            check(*gpu,config(64,64,64),1000,"one_source_64");
            c=config(64,64,64);
            c.sources={{20,32,32,SRC_RICKER,2000,1},{44,32,32,SRC_RICKER,2000,1}};
            check(*gpu,c,1000,"two_source_64");
        }
        std::puts("All requested local diagnostics passed. Team integration tests are still required.");
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
