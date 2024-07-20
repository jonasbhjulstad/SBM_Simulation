#pragma once
#include <SIR_SBM/epidemiological/epidemiological.hpp>
#include <SIR_SBM/graph/graph.hpp>
#include <SIR_SBM/simulation/sim_param.hpp>
#include <cstdint>
#include <filesystem>

namespace SIR_SBM {
struct Sim_Result {
  explicit Sim_Result(const Sim_Param &p, const SBM_Graph &G);
  void resize(const Sim_Param &p, const SBM_Graph &G);
  std::vector<uint32_t> contact_events;
  std::vector<Population_Count> population_count;
  std::vector<uint32_t> N_pops;
  uint32_t N_contact_events;

  void write(const std::filesystem::path &dir);

  void validate() const;
  uint32_t partition_idx(uint32_t sim_idx, uint32_t p_idx,
                         uint32_t t_idx) const;
  uint32_t from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                               uint32_t t_idx) const;
  uint32_t to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                             uint32_t t_idx) const;
  uint32_t N_partitions, N_connections, N_sims, Nt;

private:
  void validate_partition_size(uint32_t sim_idx) const;
  std::vector<uint32_t> get_t_infections(uint32_t sim_idx, uint32_t t) const;

  void validate_step(uint32_t sim_idx, uint32_t t) const;

  std::vector<uint32_t> merge_connection_infections(uint32_t sim_idx,
                                                    uint32_t t_idx) const;
};
} // namespace SIR_SBM