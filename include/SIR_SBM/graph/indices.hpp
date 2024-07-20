#pragma once
#include <cstdint>
namespace SIR_SBM {
uint32_t get_from_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                                 uint32_t t_idx, uint32_t N_connections,
                                 uint32_t Nt);
uint32_t get_to_connection_idx(uint32_t sim_idx, uint32_t con_idx,
                               uint32_t t_idx, uint32_t N_connections,
                               uint32_t Nt);

uint32_t get_partition_idx(uint32_t sim_idx, uint32_t p_idx, uint32_t t_idx,
                           uint32_t N_partitions, uint32_t Nt);
} // namespace SIR_SBM