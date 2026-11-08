// STUB (owner: Venthan) -- placeholder runner. Loads a config, runs the chosen
// backend, prints max |p|. Timers, --steps and file output come in the real one.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include "config.hpp"
#include "solver.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <cfg> [--backend cpu|gpu]\n", argv[0]); return 2; }
    std::string backend = "cpu";
    for (int a = 2; a + 1 < argc; ++a)
        if (!std::strcmp(argv[a], "--backend")) backend = argv[a + 1];
    try {
        SimConfig cfg = load_config(argv[1]);
        print_config(cfg);
        auto solver = backend == "gpu" ? make_gpu_solver() : make_cpu_solver();
        solver->init(cfg);
        solver->run(cfg.steps);
        solver->sync();
        std::vector<float> field;
        solver->copy_field_to_host(field);
        float m = 0.0f;
        for (float v : field) m = std::fmax(m, std::fabs(v));
        std::printf("[%s] steps=%d  max |p| = %.5f\n", solver->name(), solver->steps_done(), m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
