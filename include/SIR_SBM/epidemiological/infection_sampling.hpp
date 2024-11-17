#pragma once

#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/sim_result.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/connection_data.hpp>
#include <SIR_SBM/simulation/population_data.hpp>

namespace SIR_SBM {

struct Infection_Sampler {
  uint32_t N_sims, N_partitions, N_connections, Nt;
  std::vector<uint32_t>
  sample_infections(const Connection_Data &contact_events,
                    const Population_Data &population_count,
                    int seed);
  Infection_Sampler(uint32_t N_sims, uint32_t N_partitions,
                    uint32_t N_connections, uint32_t Nt);
  uint32_t partition_idx(uint32_t sim_idx, uint32_t p_idx,
                         uint32_t t_idx) const;
  uint32_t from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                               uint32_t t_idx) const;

  uint32_t to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                             uint32_t t_idx) const;

private:
  std::vector<int> get_connection_indices(int p_idx) const;
  std::vector<uint32_t>
  get_t_connections(const Connection_Data &contact_events,
                    uint32_t sim_idx, uint32_t t);


  std::vector<uint32_t>
  sample_infections(const Connection_Data &contact_events,
                    const Population_Data &population_count,
                    uint32_t sim_idx, uint32_t p_idx, uint32_t t_idx,
                    std::mt19937 &rng);
  void assign_t_infections(Connection_Data &infections,
                           std::vector<uint32_t> &infections_pt,
                           uint32_t sim_idx, uint32_t t);
};

} // namespace SIR_SBM