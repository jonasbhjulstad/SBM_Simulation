#pragma once
#include <SIR_SBM/graph/graph.hpp>
#include <SIR_SBM/simulation/sim_param.hpp>
#include <filesystem>
#include <cstdint>

namespace SIR_SBM {
struct Sim_Result {
  explicit Sim_Result(const Sim_Param &p, const SBM_Graph &G);
  void resize(const Sim_Param &p, const SBM_Graph &G);
  std::shared_ptr<uint32_t> contact_events;
  std::shared_ptr<Population_Count> population_count;
  uint32_t N_pop_count;
  uint32_t N_contact_events;

  void write(const std::filesystem::path &dir);

  void write_contact_events(const std::filesystem::path &dir);

  void write_population_count(const std::filesystem::path &dir);

  void validate() const;

  uint32_t N_partitions, N_connections, N_sims, Nt;

private:
  void validate_partition_size(uint32_t sim_idx) const;

  std::vector<int> get_t_dI(uint32_t sim_idx, uint32_t t_idx) const;

  void validate_step(uint32_t sim_idx, uint32_t t) const;

  std::vector<uint32_t> merge_connection_infections(uint32_t sim_idx,
                                                    uint32_t t_idx) const;
};
} // namespace SIR_SBM