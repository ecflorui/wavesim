// STUB (owner: Rishik) -- placeholder until the real parser lands.
// Ignores the path and returns the tiny_33 reference config so the CPU solver
// can be exercised before configs exist.
#include "config.hpp"
#include <cstdio>

SimConfig load_config(const std::string& /*path*/) {
    SimConfig cfg;
    cfg.nx = cfg.ny = cfg.nz = 33;
    cfg.steps = 200;
    cfg.output_dir = "out/tiny_33";
    cfg.sources.push_back(Source{16, 16, 16, SRC_RICKER, 2000.0f, 1.0f});
    return cfg;
}

void print_config(const SimConfig& cfg) {
    std::printf("grid %dx%dx%d dx=%g c=%g cfl=%g dt=%g steps=%d sources=%zu\n",
                cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.c, cfg.cfl, cfg.dt(),
                cfg.steps, cfg.sources.size());
}
