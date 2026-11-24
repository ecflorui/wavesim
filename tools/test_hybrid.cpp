// ============================================================================
// tools/test_hybrid.cpp  --  checks the CPU+GPU split logic (owner: Krish)
//
//   make test-hybrid
//
// Builds the split solver from TWO CPU slabs. The split only changes which
// code updates which layers, never the arithmetic, so the result must equal
// CpuSolver bit for bit. If it does, the layer bookkeeping and ghost-layer
// swap are right, and the GPU slab only has to match the same interface
// (checked on Lonestar6 with compare.py against the CPU run).
// ============================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "config.hpp"
#include "hybrid.hpp"
#include "solver.hpp"

namespace {
int failures = 0;

std::vector<float> run(std::unique_ptr<Solver> s, const SimConfig& cfg) {
    s->init(cfg);
    s->run(cfg.steps);
    s->sync();
    std::vector<float> f;
    s->copy_field_to_host(f);
    return f;
}

void check(const char* label, SimConfig cfg, int lower_layers) {
    const auto ref = run(make_cpu_solver(), cfg);
    const auto got = run(make_split_solver(make_cpu_slab(), make_cpu_slab(), lower_layers, "split"), cfg);
    size_t diff = 0;
    double maxabs = 0;
    for (size_t n = 0; n < ref.size(); ++n) {
        if (ref[n] != got[n]) ++diff;
        maxabs = std::fmax(maxabs, std::fabs(ref[n]));
    }
    const bool ok = diff == 0 && got.size() == ref.size() && maxabs > 0;
    std::printf("[%s] %-44s split after layer %3d: %zu cells differ (max|p| %.5f)\n",
                ok ? "PASS" : "FAIL", label, lower_layers, diff, maxabs);
    if (!ok) ++failures;
}

SimConfig cube(int n, int steps) {
    SimConfig c;
    c.nx = c.ny = c.nz = n;
    c.steps = steps;
    return c;
}
}  // namespace

int main() {
    SimConfig tiny = cube(33, 200);
    tiny.sources = {{16, 16, 16, SRC_RICKER, 2000, 1}};
    check("tiny_33, centred source", tiny, 1);
    check("tiny_33, centred source", tiny, 15);          // source just above split
    check("tiny_33, centred source", tiny, 16);          // source on the CPU's top layer
    check("tiny_33, centred source", tiny, 30);          // GPU slab is one layer

    SimConfig two = cube(40, 300);
    two.sources = {{20, 20, 10, SRC_RICKER, 2000, 1}, {20, 20, 30, SRC_SINE, 1500, 0.3f}};
    check("40^3, one source in each slab", two, 20);

    SimConfig odd;                                        // non-cubic, sources on both sides
    odd.nx = 37; odd.ny = 21; odd.nz = 26; odd.steps = 250;
    odd.sources = {{18, 10, 12, SRC_RICKER, 2000, 1}, {18, 10, 13, SRC_RICKER, 2000, 1},
                   {5, 5, 12, SRC_RICKER, 2000, 1}};
    check("37x21x26, sources either side of split", odd, 12);

    // A bad split must be refused, not silently wrong.
    bool threw = false;
    try { run(make_split_solver(make_cpu_slab(), make_cpu_slab(), 31, "split"), tiny); }
    catch (const std::exception&) { threw = true; }
    std::printf("[%s] cpu_layers = 31 on a 33^3 grid is rejected\n", threw ? "PASS" : "FAIL");
    if (!threw) ++failures;

    std::printf("%s (%d failure%s)\n", failures ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
