#pragma once
#include <SIR_SBM/epidemiological/types.hpp>
#include <cstdint>
#include <filesystem>
#include <vector>
namespace SIR_SBM {
struct Population_Data : public std::vector<Population_Count> {
  uint32_t N_sims;
  uint32_t N_partitions;
  uint32_t Nt;
  Population_Data(uint32_t N_sims, uint32_t N_connections, uint32_t Nt);
  Population_Data(const std::vector<Population_Count> &data, uint32_t N_sims,
                  uint32_t N_connections, uint32_t Nt);
  Population_Count operator()(uint32_t sim_idx, uint32_t connection_idx,
                              uint32_t t) const;
  void write(const std::filesystem::path &fname) const;
  uint32_t get_partition_idx(uint32_t sim_idx, uint32_t connection_idx,
                              uint32_t t) const;
};
} // namespace SIR_SBM