// ============================================================================
// live_api.cpp  --  small C interface for the live GUI (owner: Krish)
//
// Built into libwavesim_live.so by `make gui` and loaded from Python with
// ctypes (tools/live_gui.py). It drives the SAME CpuSolver the real program
// uses, through the Solver interface, so the GUI shows exactly what
// ./wavesim computes. No physics lives here: it only creates the solver,
// runs steps, and cuts a 2D slice out of the field for display.
// ============================================================================
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "config.hpp"
#include "physics.hpp"
#include "solver.hpp"

namespace {

struct Live {
    SimConfig cfg;
    std::unique_ptr<Solver> solver;
    std::vector<float> field;      // last copy of the current field
    bool field_fresh = false;      // field matches the solver's current step
};

const std::vector<float>& current_field(Live* L) {
    if (!L->field_fresh) {
        L->solver->copy_field_to_host(L->field);
        L->field_fresh = true;
    }
    return L->field;
}

}  // namespace

extern "C" {

// Create a simulation with no sources yet. Returns nullptr on bad sizes.
void* ws_create(int nx, int ny, int nz, float dx, float c, float cfl) {
    if (nx < 3 || ny < 3 || nz < 3 || dx <= 0 || c <= 0 || cfl <= 0) return nullptr;
    auto* L = new Live;
    L->cfg.nx = nx; L->cfg.ny = ny; L->cfg.nz = nz;
    L->cfg.dx = dx; L->cfg.c = c; L->cfg.cfl = cfl;
    L->cfg.sources.clear();
    L->solver = make_cpu_solver();
    L->solver->init(L->cfg);
    return L;
}

void ws_destroy(void* h) { delete static_cast<Live*>(h); }

// Add a source (interior cells only). Like any config change it takes effect
// at the next ws_reset(). Returns 0 on success, -1 if the cell is not
// interior or the list is full.
int ws_add_source(void* h, int i, int j, int k, int type, float freq, float amp) {
    auto* L = static_cast<Live*>(h);
    const SimConfig& g = L->cfg;
    if (i < 1 || j < 1 || k < 1 || i > g.nx - 2 || j > g.ny - 2 || k > g.nz - 2) return -1;
    if ((int)L->cfg.sources.size() >= MAX_SOURCES) return -1;
    L->cfg.sources.push_back(Source{i, j, k, type, freq, amp});
    return 0;
}

void ws_clear_sources(void* h) { static_cast<Live*>(h)->cfg.sources.clear(); }

// Back to step 0, field all zeros, same grid and sources.
void ws_reset(void* h) {
    auto* L = static_cast<Live*>(h);
    L->solver->init(L->cfg);
    L->field_fresh = false;
}

// Advance n steps. Returns the wall-clock milliseconds the steps took.
double ws_run(void* h, int n) {
    auto* L = static_cast<Live*>(h);
    const auto t0 = std::chrono::steady_clock::now();
    L->solver->run(n);
    L->solver->sync();
    L->field_fresh = false;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int   ws_steps(void* h) { return static_cast<Live*>(h)->solver->steps_done(); }
float ws_dt(void* h)    { return static_cast<Live*>(h)->cfg.dt(); }

// Largest |p| in the whole 3D field; +inf if anything is NaN/inf (blown up).
float ws_max_abs(void* h) {
    float m = 0.0f;
    for (float v : current_field(static_cast<Live*>(h))) {
        if (!std::isfinite(v)) return INFINITY;
        m = std::fmax(m, std::fabs(v));
    }
    return m;
}

// Copy one plane into out. axis 'x' (ny cols, nz rows), 'y' (nx cols, nz
// rows) or 'z' (nx cols, ny rows); row 0 is the low edge. Returns the number
// of floats written, or -1 for a bad axis/index.
int ws_slice(void* h, char axis, int index, float* out) {
    auto* L = static_cast<Live*>(h);
    const SimConfig& g = L->cfg;
    const auto& f = current_field(L);
    int n = 0;
    if (axis == 'z') {
        if (index < 0 || index >= g.nz) return -1;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) out[n++] = f[idx(i, j, index, g.nx, g.ny)];
    } else if (axis == 'y') {
        if (index < 0 || index >= g.ny) return -1;
        for (int k = 0; k < g.nz; ++k)
            for (int i = 0; i < g.nx; ++i) out[n++] = f[idx(i, index, k, g.nx, g.ny)];
    } else if (axis == 'x') {
        if (index < 0 || index >= g.nx) return -1;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j) out[n++] = f[idx(index, j, k, g.nx, g.ny)];
    } else {
        return -1;
    }
    return n;
}

}  // extern "C"
