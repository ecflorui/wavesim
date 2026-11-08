// STUB (owner: Venthan) -- placeholder until the real output writer lands.
#include "output.hpp"

std::string run_dir(const SimConfig& cfg, const std::string& backend) {
    return cfg.output_dir + "/" + backend + "/";
}
void write_meta(const std::string&, const SimConfig&, const std::string&) {}
void write_field(const std::string&, const std::vector<float>&) {}
void write_slice(const std::string&, const std::vector<float>&, const SimConfig&, int) {}
void write_timing(const std::string&, double, double, double) {}
