#include "output.hpp"
#include "physics.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

int snapshot_index(const SimConfig& cfg) {
    int extent;
    switch (cfg.snapshot_axis) {
        case 'x': extent = cfg.nx; break;
        case 'y': extent = cfg.ny; break;
        case 'z': extent = cfg.nz; break;
        default: throw std::runtime_error("snapshot axis must be x, y, or z");
    }

    const int index = cfg.snapshot_index == -1 ? extent / 2 : cfg.snapshot_index;
    if (index < 0 || index >= extent) {
        throw std::runtime_error("snapshot index is outside the grid");
    }
    return index;
}

void close_text(std::ofstream& out, const fs::path& path) {
    out.close();
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

void write_floats(const fs::path& path, const std::vector<float>& values) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "WaveSim binary output requires IEEE-754 float32");

    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    const auto count = std::fwrite(values.data(), sizeof(float), values.size(), file);
    const int close_status = std::fclose(file);
    if (count != values.size() || close_status != 0) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

}  // namespace

std::string run_dir(const SimConfig& cfg, const std::string& backend) {
    const fs::path dir = fs::path(cfg.output_dir) / backend;
    fs::create_directories(dir);
    return dir.generic_string();
}

void write_meta(const std::string& dir, const SimConfig& cfg, const std::string& backend) {
    const int index = snapshot_index(cfg);
    const fs::path path = fs::path(dir) / "meta.txt";
    std::ofstream out(path);
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "nx " << cfg.nx << '\n'
        << "ny " << cfg.ny << '\n'
        << "nz " << cfg.nz << '\n'
        << "dx " << cfg.dx << '\n'
        << "dt " << cfg.dt() << '\n'
        << "steps " << cfg.steps << '\n'
        << "backend " << backend << '\n'
        << "snapshot_axis " << cfg.snapshot_axis << '\n'
        << "snapshot_index " << index << '\n';
    close_text(out, path);
}

void write_field(const std::string& dir, const std::vector<float>& field) {
    write_floats(fs::path(dir) / "final.bin", field);
}

void write_slice(const std::string& dir, const std::vector<float>& field,
                 const SimConfig& cfg, int step) {
    if (field.size() != cfg.n_cells()) {
        throw std::runtime_error("slice field size does not match the grid");
    }

    const char axis = cfg.snapshot_axis;
    const int index = snapshot_index(cfg);
    const int rows = axis == 'z' ? cfg.ny : cfg.nz;
    const int cols = axis == 'x' ? cfg.ny : cfg.nx;
    const std::size_t first = idx(axis == 'x' ? index : 0,
                                  axis == 'y' ? index : 0,
                                  axis == 'z' ? index : 0, cfg.nx, cfg.ny);

    // Columns advance along x (or y for an x slice). Rows advance along y or z.
    const std::size_t row_stride = axis == 'z' ? cfg.nx : std::size_t(cfg.nx) * cfg.ny;
    const std::size_t col_stride = axis == 'x' ? cfg.nx : 1;
    std::vector<float> slice;
    slice.reserve(std::size_t(rows) * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            slice.push_back(field[first + row * row_stride + col * col_stride]);
        }
    }

    std::ostringstream name;
    name << "slice_" << axis << '_' << std::setfill('0') << std::setw(6) << step << ".bin";
    write_floats(fs::path(dir) / name.str(), slice);
}

void write_timing(const std::string& dir, double init_ms, double steps_ms, double copy_ms) {
    const fs::path path = fs::path(dir) / "timing.txt";
    const double total_ms = init_ms + steps_ms + copy_ms;
    std::ofstream out(path);
    out << std::setprecision(12) << init_ms << ' ' << steps_ms << ' '
        << copy_ms << ' ' << total_ms << '\n';
    close_text(out, path);
}
