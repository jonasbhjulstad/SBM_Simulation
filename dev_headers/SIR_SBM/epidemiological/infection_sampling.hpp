#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological/population_count.hpp>
#include <SIR_SBM/simulation/sim_result.hpp>
#include <SIR_SBM/utils/csv.hpp>
#include <SIR_SBM/vector/vector.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#end

#src
#include <SIR_SBM/utils/occurrence.hpp>
#include <SIR_SBM/vector/vector.hpp>
#end

namespace SIR_SBM {
uint32_t get_new_infections(const std::shared_ptr<Population_Count> &pop_count,
                            uint32_t p_idx, uint32_t t_idx) {
  auto dI = pop_count[p_idx][t_idx + 1].I - pop_count[p_idx][t_idx].I;
  auto dR = pop_count[p_idx][t_idx + 1].R - pop_count[p_idx][t_idx].R;
  if (dI > dR) {
    return dI - dR;
  } else {
    return 0;
  }
}
std::vector<int> get_connection_indices(int N_partitions, int p_idx) {
  std::vector<int> result;
  for (auto comb :
       iter::combinations_with_replacement(make_iota(N_partitions), 2)) {
    auto from = comb[0];
    auto to = comb[1];
    if (to == p_idx)
      result.push_back(2 * to);
    if (from == p_idx)
      result.push_back(2 * from + 1);
  }
  return result;
}

std::vector<uint32_t>
get_partition_connection_contacts(const Vec1D<uint32_t> &contact_events,
                                  int N_partitions, int p_idx) {
  auto indices = get_connection_indices(N_partitions, p_idx);
  std::vector<uint32_t> result(indices.size());
  for (int c_idx = 0; c_idx < indices.size(); c_idx++) {
    result[c_idx] = contact_events[indices[c_idx]];
  }
  return result;
}

std::vector<uint32_t> get_column(const std::shared_ptr<uint32_t>& data, std::tuple<uint32_t, uint32_t> idx, std::tuple<uint32_t, uint32_t, uint32_t> shape)
{
  auto [N0, N1, N2] = shape;
  auto [n0, n2] = idx;
  std::vector<uint32_t> result(N1);
  for (int k = 0; k < N1; k++)
  {
    result[k] = data.get()[n0 * N1 * N2 + k * N2 + n2];
  }
  return result;
}

std::vector<uint32_t>
sample_infections(const std::shared_ptr<uint32_t> &contact_events,
                  const std::shared_ptr<Population_Count> &population_count,
                  std::tuple<uint32_t, uint32
                  _t, uint32_t> idx,
                  std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> shape,
                  std::mt19937_64 &rng) {

  auto [sim_idx, p_idx, t_idx] = idx;
  auto [N_sims, N_connections, N_partitions, Nt] = shape;
  auto con_indices = get_connection_indices(N_partitions, p_idx);
  std::vector<uint32_t> connection_contacts = get_partition_connection_contacts(
      get_column(contact_events, {sim_idx, t_idx}, {N_sims, N_connections*2, Nt}), N_partitions, p_idx);

  auto new_infs = get_new_infections(population_count, p_idx, t_idx);
  if (new_infs) {
    auto inf_index_samples =
        discrete_finite_sample(rng, connection_contacts, new_infs);
    return count_occurrences(inf_index_samples, 2 * N_connections);
  } else {
    return std::vector<uint32_t>(2 * N_connections, 0);
  }
}

void assign_to_column(std::shared_ptr<uint32_t> &infections,
                      const std::vector<uint32_t> &infections_pt,
                      std::tuple<uint32_t, uint32_t, uint32_t> idx,
                      std::tuple<uint32_t, uint32_t, uint32_t> shape) {
  auto [sim_idx, p_idx, t_idx] = idx;
  auto [N_sims, N_connections*2, Nt] = shape;
  for (int i = 0; i < N_connections; i++) {
    infections[sim_idx * N_connections*2 * Nt + i * Nt + t_idx] =
        infections_pt[p_idx];
  }
}

std::shared_ptr<uint32_t>
sample_infections(const std::shared_ptr<uint32_t> &contact_events,
                  const std::shared_ptr<Population_Count> &population_count,
                  uint32_t N_sims, uint32_t N_partitions,
                  uint32_t N_connections, uint32_t Nt, int seed) {

  auto rngs = generate_rngs<std::mt19937_64>(N_sims, seed);
  auto infections = make_shared_array<uint32_t>(N_sims * N_partitions * Nt);
  auto infections_pt = std::vector<uint32_t>(N_connections*2, 0);
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        infections_pt = sample_infections(contact_events, population_count,
                                          {sim_idx, p_idx, t_idx},
                                          {N_sims, N_connections, N_partitions, Nt},
                                          rngs[sim_idx]);
        assign_to_column(infections, infections_pt, {sim_idx, p_idx, t_idx},
                         {N_sims, N_connections, Nt});
      }
      std::fill(infections_pt.begin(), infections_pt.end(), 0);
    }
  }
  return result;
}
} // namespace SIR_SBM