#pragma once
#include <cstdint>
#include <filesystem>
namespace SIR_SBM {
struct SBM_Param {
  uint32_t N_pop = 100;
  uint32_t N_communities = 2;
  float p_in = 1.0;
  float p_out = 1.0;
  uint32_t seed = 42;
  static SBM_Param parse(const std::filesystem::path &);
};
} // namespace SIR_SBM