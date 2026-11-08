// ============================================================================
// output.hpp  --  SHARED CONTRACT (owner: Venthan)
// File formats every tool (compare.py, plot_slice.py, bench.sh) reads.
// All .bin files: raw little-endian float32, x fastest, no header.
// ============================================================================
#pragma once
#include <string>
#include <vector>
#include "config.hpp"

// Run folder = <cfg.output_dir>/<backend>/   e.g. out/one_source_64/gpu/
std::string run_dir(const SimConfig& cfg, const std::string& backend);

// <run_dir>/meta.txt  -- one "key value" per line:
//   nx ny nz dx dt steps backend snapshot_axis snapshot_index
void write_meta(const std::string& dir, const SimConfig& cfg, const std::string& backend);

// <run_dir>/final.bin  -- full 3D field after the last step (nx*ny*nz floats)
void write_field(const std::string& dir, const std::vector<float>& field);

// <run_dir>/slice_<axis>_<step 6 digits>.bin  e.g. slice_z_000050.bin
//   axis z: ny rows of nx floats   axis y: nz rows of nx   axis x: nz rows of ny
void write_slice(const std::string& dir, const std::vector<float>& field,
                 const SimConfig& cfg, int step);

// <run_dir>/timing.txt  -- "init_ms steps_ms copy_ms total_ms" on one line
void write_timing(const std::string& dir, double init_ms, double steps_ms,
                  double copy_ms);
