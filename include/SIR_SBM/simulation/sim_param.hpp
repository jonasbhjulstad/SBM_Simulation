#pragma once

#include <cstdint>
#include <filesystem>
namespace SIR_SBM {
struct Sim_Param {
  float p_I0;
  float p_I;
  float p_R;
  uint32_t Nt;
  uint32_t N_sims;
  uint32_t seed;
  static Sim_Param parse(const std::filesystem::path &fname);
};

} // namespace SIR_SBM