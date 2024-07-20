#pragma once
#include <cstdint>
#include <vector>
#include <SIR_SBM/epidemiological/types.hpp>
namespace SIR_SBM {
uint32_t get_new_infections(const std::vector<Population_Count> &pop_count,
                            uint32_t sim_idx, uint32_t p_idx,
                            uint32_t N_partitions, uint32_t t_idx, uint32_t Nt);
}