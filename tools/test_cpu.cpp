// ============================================================================
// tools/test_cpu.cpp  --  CPU solver sanity checks (owner: Krish)
//
//   make test-cpu              # 4 sanity checks + tiny_33 reference number
//   ./test_cpu --all-refs      # also runs the full reference table (slower)
//
// Configs are built in code, so this does not depend on the config parser.
// Exit code 0 only if every check passes.
//
//   (a) no sources        -> field is exactly zero after 100 steps
//   (b) symmetry          -> tiny_33, centred source: mirror (x -> -x) and
//                            axis swap (x <-> y, x <-> z) agree to 1e-5*max|p|
//   (c) stable  cfl 0.5   -> 1000 steps stay finite and do not grow
//   (d) unstable cfl 0.6  -> blows up (NaN/inf or huge), proving the CFL limit
//   (+) reference         -> tiny_33 max|p| after 200 steps = 0.04977
// ============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "config.hpp"
#include "physics.hpp"
#include "solver.hpp"

namespace {

int g_failures = 0;

void report(const char* name, bool ok, const std::string& detail) {
    std::printf("[%s] %-34s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++g_failures;
}

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
    char buf[256];
    std::snprintf(buf, sizeof buf, f, a, b, c);
    return buf;
}

SimConfig cube(int n, int steps, float cfl = 0.5f) {
    SimConfig cfg;
    cfg.nx = cfg.ny = cfg.nz = n;
    cfg.steps = steps;
    cfg.cfl = cfl;
    return cfg;
}

Source ricker(int i, int j, int k) { return Source{i, j, k, SRC_RICKER, 2000.0f, 1.0f}; }

// max |p| over the field; returns +inf if any value is NaN/inf.
double max_abs(const std::vector<float>& f) {
    double m = 0.0;
    for (float v : f) {
        if (!std::isfinite(v)) return INFINITY;
        m = std::fmax(m, std::fabs((double)v));
    }
    return m;
}

std::vector<float> run(const SimConfig& cfg) {
    auto s = make_cpu_solver();
    s->init(cfg);
    s->run(cfg.steps);
    std::vector<float> f;
    s->copy_field_to_host(f);
    return f;
}

// (a) Nothing should ever appear from nothing.
void check_no_sources() {
    SimConfig cfg = cube(33, 100);          // no sources pushed
    auto f = run(cfg);
    size_t nonzero = 0;
    for (float v : f) nonzero += (v != 0.0f);
    report("(a) no sources -> all zero", nonzero == 0,
           fmt("nonzero cells after 100 steps: %.0f", (double)nonzero));
}

// (b) Source at the exact centre of a cube -> field must be symmetric.
void check_symmetry() {
    SimConfig cfg = cube(33, 200);
    cfg.sources.push_back(ricker(16, 16, 16));
    auto f = run(cfg);
    const int n = cfg.nx;
    auto p = [&](int i, int j, int k) { return (double)f[idx(i, j, k, n, n)]; };

    const double m = max_abs(f);
    double mirror = 0.0, swap = 0.0;
    for (int d = 1; d <= 15; ++d) {
        mirror = std::fmax(mirror, std::fabs(p(16 + d, 16, 16) - p(16 - d, 16, 16)));
        swap   = std::fmax(swap,   std::fabs(p(16 + d, 16, 16) - p(16, 16 + d, 16)));
        swap   = std::fmax(swap,   std::fabs(p(16 + d, 16, 16) - p(16, 16, 16 + d)));
    }
    const double tol = 1e-5 * m;
    report("(b) tiny_33 mirror symmetry", mirror <= tol,
           fmt("max diff %.3g  (tol %.3g)", mirror, tol));
    report("(b) tiny_33 axis-swap symmetry", swap <= tol,
           fmt("max diff %.3g  (tol %.3g)", swap, tol));
}

// (c) At cfl 0.5 the scheme is stable: 1000 steps, values finite, no growth.
//     The Ricker pulse is over after ~70 steps; after that the closed box only
//     moves energy around, so late max|p| must not exceed the early peak.
void check_stable() {
    SimConfig cfg = cube(33, 1000, 0.5f);
    cfg.sources.push_back(ricker(16, 16, 16));
    auto s = make_cpu_solver();
    s->init(cfg);
    std::vector<float> f;
    double early = 0.0, late = 0.0;
    bool finite = true;
    for (int n = 1; n <= cfg.steps; ++n) {
        s->step();
        if (n % 10) continue;                  // sample every 10 steps
        s->copy_field_to_host(f);
        double m = max_abs(f);
        if (!std::isfinite(m)) { finite = false; break; }
        (n <= 500 ? early : late) = std::fmax(n <= 500 ? early : late, m);
    }
    report("(c) cfl 0.5, 1000 steps stable", finite && late <= early,
           fmt("peak steps 1-500 %.4g, peak 501-1000 %.4g", early, late));
}

// (d) Above the CFL limit (1/sqrt(3) = 0.577) the same run must blow up.
void check_unstable() {
    SimConfig cfg = cube(33, 1000, 0.6f);
    cfg.sources.push_back(ricker(16, 16, 16));
    auto s = make_cpu_solver();
    s->init(cfg);
    std::vector<float> f;
    double m = 0.0;
    int blew_at = -1;
    for (int n = 1; n <= cfg.steps; ++n) {
        s->step();
        s->copy_field_to_host(f);
        m = max_abs(f);
        if (!std::isfinite(m) || m > 1e6) { blew_at = n; break; }
    }
    report("(d) cfl 0.6 blows up", blew_at > 0,
           blew_at > 0 ? fmt("max|p| > 1e6 (or NaN) at step %.0f", blew_at)
                       : fmt("still bounded: max|p| = %.3g", m));
}

struct Ref { const char* name; int n, steps; std::vector<Source> src; double expect; };

void check_reference(const Ref& r) {
    SimConfig cfg = cube(r.n, r.steps);
    cfg.sources = r.src;
    auto t0 = std::chrono::steady_clock::now();
    auto f = run(cfg);
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
    double m = max_abs(f);
    double rel = std::fabs(m - r.expect) / r.expect;
    std::string label = std::string("(ref) ") + r.name;
    report(label.c_str(), rel < 5e-4,          // table is given to 4 digits
           fmt("max|p| = %.5f  expected %.5f  (%.0f ms)", m, r.expect, ms));
}

}  // namespace

int main(int argc, char** argv) {
    bool all_refs = argc > 1 && !std::strcmp(argv[1], "--all-refs");

    check_no_sources();
    check_symmetry();
    check_stable();
    check_unstable();

    std::vector<Ref> refs = {
        {"tiny_33",        33,  200, {ricker(16, 16, 16)},                     0.04977},
        {"one_source_64",  64, 1000, {ricker(32, 32, 32)},                     0.02026},
        {"two_source_64",  64, 1000, {ricker(20, 32, 32), ricker(44, 32, 32)}, 0.03303},
        {"bench_128",     128, 1000, {ricker(64, 64, 64)},                     0.02129},
    };
    for (size_t r = 0; r < (all_refs ? refs.size() : 1); ++r) check_reference(refs[r]);

    std::printf("%s (%d failure%s)\n", g_failures ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
