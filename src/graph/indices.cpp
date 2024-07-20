#include <SIR_SBM/graph/indices.hpp>

namespace SIR_SBM {
uint32_t get_from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                                 uint32_t t_idx, uint32_t N_connections, uint32_t Nt) {
  return sim_idx * 2 * N_connections * Nt + (2*con_idx) * Nt + t_idx;
}

uint32_t get_to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                               uint32_t t_idx, uint32_t N_connections, uint32_t Nt) {
  return sim_idx * 2 * N_connections * Nt + (2*con_idx + 1) * Nt + t_idx;
}

uint32_t get_partition_idx(uint32_t sim_idx, uint32_t p_idx, uint32_t t_idx, uint32_t N_partitions, uint32_t Nt) {
  return sim_idx * N_partitions * Nt + p_idx * Nt + t_idx;
}
} // namespace SIR_SBM