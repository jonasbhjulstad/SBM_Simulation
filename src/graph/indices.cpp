#include <SIR_SBM/graph/indices.hpp>
#include <SIR_SBM/math/numeric.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
namespace SIR_SBM {
uint32_t get_from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                                 uint32_t t_idx, uint32_t N_connections,
                                 uint32_t Nt) {
  return sim_idx * 2 * N_connections * Nt + (2 * con_idx) * Nt + t_idx;
}

uint32_t get_to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                               uint32_t t_idx, uint32_t N_connections,
                               uint32_t Nt) {
  return sim_idx * 2 * N_connections * Nt + (2 * con_idx + 1) * Nt + t_idx;
}

uint32_t get_partition_idx(uint32_t sim_idx, uint32_t p_idx, uint32_t t_idx,
                           uint32_t N_partitions, uint32_t Nt) {
  return sim_idx * N_partitions * Nt + p_idx * Nt + t_idx;
}

std::vector<int> get_connection_indices(int p_idx, uint32_t N_partitions) {
  std::vector<int> result;
  uint32_t con_idx = 0;
  for (auto comb :
       iter::combinations_with_replacement(make_iota(N_partitions), 2)) {
    auto from = comb[0];
    auto to = comb[1];
    if (to == p_idx)
      result.push_back(2 * con_idx);
    if (from == p_idx)
      result.push_back(2 * con_idx + 1);
    con_idx++;
  }
  return result;
}

std::vector<uint32_t>
get_partition_connection_contacts(const std::vector<uint32_t> &contact_events,
                                  int p_idx, uint32_t N_partitions) {
  auto indices = get_connection_indices(p_idx, N_partitions);
  uint32_t N_connections = contact_events.size() / 2;
  std::vector<uint32_t> result(2 * N_connections, 0);
  for (int idx = 0; idx < indices.size(); idx++) {
    result[indices[idx]] = contact_events[idx];
  }
  return result;
}

} // namespace SIR_SBM