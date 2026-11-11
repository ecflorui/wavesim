#include "config.hpp"
#include "solver.hpp"
#include "output.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <regex>
#include <stdexcept>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int parse_steps(std::string_view value) {
    int steps = 0;
    const char* end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, steps);
    if (result.ec != std::errc{} || result.ptr != end || steps < 1) {
        throw std::runtime_error("--steps requires a positive integer");
    }
    return steps;
}

void clear_old_slices(const std::string& dir) {
    const std::regex slice_name(R"(slice_[xyz]_[0-9]{6,}\.bin)");
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() &&
            std::regex_match(entry.path().filename().string(), slice_name)) {
            std::filesystem::remove(entry.path());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <config.cfg> [--backend cpu|gpu] [--steps N]\n", argv[0]);
        return 2;
    }

    try {
        std::string_view backend = "cpu";
        int steps_override = 0;
        for (int a = 2; a < argc; ++a) {
            const std::string_view option = argv[a];
            if (option != "--backend" && option != "--steps") {
                throw std::runtime_error("unknown option: " + std::string(option));
            }
            if (++a == argc) {
                throw std::runtime_error("missing value for " + std::string(option));
            }
            if (option == "--backend") {
                backend = argv[a];
                if (backend != "cpu" && backend != "gpu") {
                    throw std::runtime_error("--backend requires cpu or gpu");
                }
            } else {
                steps_override = parse_steps(argv[a]);
            }
        }

        SimConfig cfg = load_config(argv[1]);
        if (steps_override > 0) {
            cfg.steps = steps_override;
        }
        print_config(cfg);

        auto start = Clock::now();
        auto solver = backend == "gpu" ? make_gpu_solver() : make_cpu_solver();
        solver->init(cfg);
        solver->sync();
        const double init_ms = elapsed_ms(start);

        const std::string dir = run_dir(cfg, solver->name());
        clear_old_slices(dir);

        const int interval = cfg.snapshot_every;
        const int chunk_size = interval > 0 ? interval : cfg.steps;
        double steps_ms = 0.0;
        {
            std::vector<float> snapshot;
            for (int done = 0; done < cfg.steps;) {
                const int chunk = std::min(chunk_size, cfg.steps - done);
                start = Clock::now();
                solver->run(chunk);
                solver->sync();
                steps_ms += elapsed_ms(start);
                done += chunk;

                if (interval > 0 && done < cfg.steps) {
                    solver->copy_field_to_host(snapshot);
                    solver->sync();
                    write_slice(dir, snapshot, cfg, done);
                }
            }
        }

        std::vector<float> field;
        start = Clock::now();
        solver->copy_field_to_host(field);
        solver->sync();
        const double copy_ms = elapsed_ms(start);

        // Reuse the final field if the last step lands on a snapshot interval.
        if (interval > 0 && cfg.steps % interval == 0) {
            write_slice(dir, field, cfg, cfg.steps);
        }

        float max_abs = 0.0f;
        for (float value : field) {
            max_abs = std::fmax(max_abs, std::fabs(value));
        }
        std::printf("[%s] %d steps done, max |p| = %g\n",
                    solver->name(), solver->steps_done(), max_abs);

        write_meta(dir, cfg, solver->name());
        write_field(dir, field);
        write_timing(dir, init_ms, steps_ms, copy_ms);
        std::printf("run_dir=%s\n", dir.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
