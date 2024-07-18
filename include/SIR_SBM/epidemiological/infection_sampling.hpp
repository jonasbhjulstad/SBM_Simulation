#pragma once

#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/sim_result.hpp>
#include <SIR_SBM/utils/csv.hpp>

#include <cppitertools/combinations_with_replacement.hpp>
#include <filesystem>
#include <fstream>
#include <random>

#include <SIR_SBM/utils/occurrence.hpp>


namespace SIR_SBM {
uint32_t get_new_infections(const std::shared_ptr<Population_Count> &pop_count,
                            uint32_t p_idx, uint32_t t_idx)

    std::vector<int> get_connection_indices(int N_partitions, int p_idx);

std::vector<uint32_t>
get_partition_connection_contacts(const Vec1D<uint32_t> &contact_events,
                                  int N_partitions, int p_idx);

std::vector<uint32_t>
get_column(const std::shared_ptr<uint32_t> &data,
           std::tuple<uint32_t, uint32_t> idx,
           std::tuple<uint32_t, uint32_t, uint32_t> shape);

std::vector<uint32_t> sample_infections(
    const std::shared_ptr<uint32_t> &contact_events,
    const std::shared_ptr<Population_Count> &population_count,
    std::tuple<uint32_t, uint32 _t, uint32_t> idx,
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> shape,
    std::mt19937_64 &
        rng) void assign_to_column(std::shared_ptr<uint32_t> &infections,
                                   const std::vector<uint32_t> &infections_pt,
                                   std::tuple<uint32_t, uint32_t, uint32_t> idx,
                                   std::tuple<uint32_t, uint32_t, uint32_t>
                                       shape);

std::shared_ptr<uint32_t>
sample_infections(const std::shared_ptr<uint32_t> &contact_events,
                  const std::shared_ptr<Population_Count> &population_count,
                  uint32_t N_sims, uint32_t N_partitions,
                  uint32_t N_connections, uint32_t Nt, int seed);
} // namespace SIR_SBM