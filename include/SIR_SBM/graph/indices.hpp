#pragma once
#include <cstdint>
#include <vector>

namespace SIR_SBM {
uint32_t get_from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                                 uint32_t t_idx, uint32_t N_connections,
                                 uint32_t Nt);
uint32_t get_to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                               uint32_t t_idx, uint32_t N_connections,
                               uint32_t Nt);

uint32_t get_partition_idx(uint32_t sim_idx, uint32_t p_idx, uint32_t t_idx,
                           uint32_t N_partitions, uint32_t Nt);
std::vector<int> get_connection_indices(int p_idx, uint32_t N_partitions);
std::vector<uint32_t>
get_partition_connection_contacts(const std::vector<uint32_t> &contact_events,
                                  int p_idx, uint32_t N_partitions);

} // namespace SIR_SBM