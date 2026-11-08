// ============================================================================
// config.hpp  --  SHARED CONTRACT (owner: Rishik)
// Do not change without a PR approved by the whole team.
// ============================================================================
#pragma once
#include <string>
#include <vector>

constexpr int MAX_SOURCES = 8;

enum SourceType : int { SRC_RICKER = 0, SRC_SINE = 1 };

// Plain-old-data on purpose: the GPU copies an array of these to the device.
struct Source {
    int   i, j, k;      // grid cell, must be interior: 1 .. n-2 on every axis
    int   type;         // SourceType
    float freq;         // Hz
    float amp;          // amplitude added to the cell each step
};

struct SimConfig {
    // grid
    int   nx = 64, ny = 64, nz = 64;   // points per axis (including the wall cells)
    float dx = 0.01f;                  // m, same spacing on all three axes
    // physics
    float c   = 343.0f;                // wave speed, m/s
    float cfl = 0.5f;                  // Courant number c*dt/dx, must be <= 0.577
    int   steps = 1000;                // number of time steps
    // output
    int   snapshot_every = 0;          // 0 = no slice snapshots
    char  snapshot_axis  = 'z';        // 'x' | 'y' | 'z'
    int   snapshot_index = -1;         // -1 = middle of that axis
    std::string output_dir = "out/default";
    // sources (1 .. MAX_SOURCES)
    std::vector<Source> sources;

    float  dt()      const { return cfl * dx / c; }
    float  r2()      const { return cfl * cfl; }            // (c*dt/dx)^2, used by update_cell
    size_t n_cells() const { return (size_t)nx * ny * nz; }
};

// Parses a .cfg file, validates it, throws std::runtime_error with a clear
// message on any bad input. Never returns an invalid config.
SimConfig load_config(const std::string& path);

// Prints every field (and dt) to stdout so every run log shows its settings.
void print_config(const SimConfig& cfg);
