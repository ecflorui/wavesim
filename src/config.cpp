// ============================================================================
// config.cpp  --  .cfg parser and validation (contract owner: Rishik)
//
// Written to the format and rules in the week plan so the CPU baseline can
// run every example config end to end. If Rishik's version lands, it replaces
// this file; nothing else depends on the internals, only on config.hpp.
//
// Format: one `key = value` per line, `#` starts a comment, blank lines are
// skipped, `source = i j k type freq amp` repeats once per source (1..8).
// Every error names the file and line, and load_config never returns an
// invalid SimConfig.
// ============================================================================
#include "config.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

std::string fmt(float v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
    throw std::runtime_error(where + ": " + msg);
}

// Parse the whole value as one number; reject trailing junk like "64abc".
template <typename T>
T parse_number(const std::string& value, const std::string& where, const std::string& key) {
    std::istringstream in(value);
    T out{};
    in >> out;
    const bool ok = !in.fail();
    std::string rest;
    in.clear();
    in >> rest;  // anything left over ("64abc") is an error
    if (!ok || !rest.empty() || value.empty())
        fail(where, "'" + key + "' expects a number, got '" + value + "'");
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(out)) fail(where, "'" + key + "' must be finite");
    }
    return out;
}

Source parse_source(const std::string& value, const std::string& where) {
    std::istringstream in(value);
    Source s{};
    std::string type;
    if (!(in >> s.i >> s.j >> s.k >> type >> s.freq >> s.amp))
        fail(where, "source expects 'i j k type freq_hz amplitude', got '" + value + "'");
    std::string extra;
    if (in >> extra) fail(where, "unexpected text after source: '" + extra + "'");
    if (type == "ricker")      s.type = SRC_RICKER;
    else if (type == "sine")   s.type = SRC_SINE;
    else fail(where, "source type must be 'ricker' or 'sine', got '" + type + "'");
    if (!std::isfinite(s.freq) || !std::isfinite(s.amp))
        fail(where, "source frequency and amplitude must be finite");
    return s;
}

void check(bool ok, const std::string& where, const std::string& msg) {
    if (!ok) fail(where, msg);
}

}  // namespace

SimConfig load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open config file '" + path + "'");

    SimConfig cfg;
    std::map<std::string, int> seen;   // key -> line it was set on
    std::vector<int> source_lines;
    std::string line;
    int line_no = 0;

    while (std::getline(file, line)) {
        ++line_no;
        const std::string where = path + ":" + std::to_string(line_no);
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        const size_t eq = line.find('=');
        if (eq == std::string::npos) fail(where, "expected 'key = value', got '" + line + "'");
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty()) fail(where, "missing key before '='");
        if (value.empty()) fail(where, "missing value for '" + key + "'");

        if (key == "source") {
            cfg.sources.push_back(parse_source(value, where));
            source_lines.push_back(line_no);
            continue;
        }
        if (!seen.emplace(key, line_no).second)
            fail(where, "'" + key + "' is set twice (first on line " + std::to_string(seen[key]) + ")");

        if      (key == "nx")             cfg.nx = parse_number<int>(value, where, key);
        else if (key == "ny")             cfg.ny = parse_number<int>(value, where, key);
        else if (key == "nz")             cfg.nz = parse_number<int>(value, where, key);
        else if (key == "dx")             cfg.dx = parse_number<float>(value, where, key);
        else if (key == "c")              cfg.c = parse_number<float>(value, where, key);
        else if (key == "cfl")            cfg.cfl = parse_number<float>(value, where, key);
        else if (key == "steps")          cfg.steps = parse_number<int>(value, where, key);
        else if (key == "snapshot_every") cfg.snapshot_every = parse_number<int>(value, where, key);
        else if (key == "snapshot_index") cfg.snapshot_index = parse_number<int>(value, where, key);
        else if (key == "snapshot_axis") {
            check(value == "x" || value == "y" || value == "z", where,
                  "snapshot_axis must be x, y or z, got '" + value + "'");
            cfg.snapshot_axis = value[0];
        }
        else if (key == "output_dir")     cfg.output_dir = value;
        else fail(where, "unknown key '" + key + "' (typo?)");
    }

    // ---- validation (each rule from the plan) ------------------------------
    // Point each error at the line that set the value (or the file if it was defaulted).
    auto at = [&](const std::string& key) {
        auto it = seen.find(key);
        return it == seen.end() ? path : path + ":" + std::to_string(it->second);
    };
    for (auto [name, n] : {std::pair{"nx", cfg.nx}, {"ny", cfg.ny}, {"nz", cfg.nz}})
        check(n >= 8 && n <= 512, at(name), std::string(name) + " must be between 8 and 512, got " + std::to_string(n));
    check(cfg.dx > 0, at("dx"), "dx must be > 0");
    check(cfg.c > 0, at("c"), "c must be > 0");
    check(cfg.cfl > 0 && cfg.cfl <= 0.577f, at("cfl"),
          "cfl must be in (0, 0.577] for a stable 3D simulation (limit 1/sqrt(3)), got " + fmt(cfg.cfl));
    check(cfg.steps >= 1, at("steps"), "steps must be >= 1");
    check(cfg.snapshot_every >= 0, at("snapshot_every"), "snapshot_every must be >= 0");
    const int axis_len = cfg.snapshot_axis == 'x' ? cfg.nx : cfg.snapshot_axis == 'y' ? cfg.ny : cfg.nz;
    check(cfg.snapshot_index == -1 || (cfg.snapshot_index >= 0 && cfg.snapshot_index < axis_len), at("snapshot_index"),
          "snapshot_index must be -1 or inside 0.." + std::to_string(axis_len - 1));
    check(!cfg.output_dir.empty(), at("output_dir"), "output_dir must not be empty");
    check(!cfg.sources.empty() && cfg.sources.size() <= (size_t)MAX_SOURCES, path,
          "need between 1 and " + std::to_string(MAX_SOURCES) + " source lines, got " + std::to_string(cfg.sources.size()));

    for (size_t n = 0; n < cfg.sources.size(); ++n) {
        const Source& s = cfg.sources[n];
        const std::string where = path + ":" + std::to_string(source_lines[n]);
        auto interior = [](int v, int len) { return v >= 1 && v <= len - 2; };
        check(interior(s.i, cfg.nx) && interior(s.j, cfg.ny) && interior(s.k, cfg.nz), where,
              "source must be an interior cell (1..n-2 on every axis), got " +
              std::to_string(s.i) + " " + std::to_string(s.j) + " " + std::to_string(s.k));
        check(s.freq > 0, where, "source frequency must be > 0");
        // Warning only: below ~10 grid points per wavelength the wave is under-resolved.
        const double ppw = cfg.c / (s.freq * cfg.dx);
        if (ppw < 10)
            std::fprintf(stderr, "warning: %s: only %.1f grid points per wavelength (want >= 10)\n",
                         where.c_str(), ppw);
    }
    return cfg;
}

void print_config(const SimConfig& cfg) {
    std::printf("grid      %d x %d x %d   dx = %g m\n", cfg.nx, cfg.ny, cfg.nz, cfg.dx);
    std::printf("physics   c = %g m/s   cfl = %g   dt = %.6g s\n", cfg.c, cfg.cfl, cfg.dt());
    std::printf("steps     %d   (%.4g ms simulated)\n", cfg.steps, cfg.steps * cfg.dt() * 1e3);
    if (cfg.snapshot_every > 0)
        std::printf("snapshots every %d steps, axis %c, index %d\n",
                    cfg.snapshot_every, cfg.snapshot_axis, cfg.snapshot_index);
    else
        std::printf("snapshots off\n");
    std::printf("output    %s\n", cfg.output_dir.c_str());
    for (size_t n = 0; n < cfg.sources.size(); ++n) {
        const Source& s = cfg.sources[n];
        std::printf("source %zu  (%d, %d, %d) %s %g Hz amp %g\n", n + 1, s.i, s.j, s.k,
                    s.type == SRC_RICKER ? "ricker" : "sine", s.freq, s.amp);
    }
}
