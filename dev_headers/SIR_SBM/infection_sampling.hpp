#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/csv.hpp>
#include <SIR_SBM/population_count.hpp>
#include <SIR_SBM/sim_result.hpp>
#include <SIR_SBM/vector.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#end

namespace SIR_SBM {

std::vector<uint32_t> get_connection_indices(uint32_t N_partitions,
                                             uint32_t p_idx) {
  std::vector<uint32_t> result;
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
sample_infections(const LinearVector2D<uint32_t> &contact_events,
                  const LinearVector2D<Population_Count> &population_count,
                  uint32_t p_idx, uint32_t t_idx, std::mt19937_64 &rng) {
  auto N_connections = contact_events.N1 / 2;
  auto N_partitions = population_count.N1;
  auto Nt = contact_events.N2;

  auto connection_infections = get_at(
      contact_events(p_idx), get_connection_indices(N_partitions, p_idx));
  auto new_infs = get_new_infections(population_count, p_idx, t_idx);
  auto inf_index_samples =
      discrete_finite_sample(rng, connection_infections, new_infs);
  return count_occurrences(inf_index_samples, 2 * N_connections);
}

LinearVector2D<uint32_t> simulation_sample_infections(
    const LinearVector2D<uint32_t> &contact_events,
    const LinearVector2D<Population_Count> &population_count,
    std::mt19937_64 &rng) {
  auto N_connections = contact_events.N1 / 2;
  auto N_partitions = population_count.N1;
  auto Nt = contact_events.N2;
  auto vector_plus = [](const std::vector<uint32_t> &a,
                        const std::vector<uint32_t> &b) {
    std::vector<uint32_t> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<uint32_t>());
    return result;
  };

  LinearVector2D<uint32_t> result(2 * N_connections, Nt);
  for (auto t_idx : make_iota(Nt)) {
    auto p_idx = make_iota(N_partitions);
    auto infections = std::transform_reduce(
        p_idx.begin(), p_idx.end(), std::vector<uint32_t>(2 * N_connections, 0),
        vector_plus, [&](uint32_t p_idx) {
          return sample_infections(contact_events, population_count, p_idx,
                                   t_idx, rng);
        });
    set_at_row(result, infections, t_idx);
  }
  return result;
}

LinearVector3D<uint32_t>
sample_infections(const LinearVector3D<uint32_t> &contact_events,
                  const LinearVector3D<Population_Count> &population_count,
                  uint32_t seed) {
  auto N_connections = contact_events.N1 / 2;
  auto N_partitions = population_count.N1;
  auto Nt = contact_events.N2;
  auto N_sims = contact_events.N3;

  auto rngs = generate_rngs<std::mt19937_64>(N_sims, seed);

  LinearVector3D<uint32_t> result(N_sims, 2 * N_connections, Nt);
  auto sim_vec = make_iota(N_sims);
  std::vector<LinearVector2D<uint32_t>> unmerged;
  std::transform(std::execution::par_unseq, sim_vec.begin(), sim_vec.end(), rngs.begin(), 
                 unmerged.begin(), [&](auto sim_idx, auto& rng) {
                   return simulation_sample_infections(
                       contact_events.get_row(sim_idx),
                       population_count.get_row(sim_idx), rng);
                 });
  for (auto sim_idx : sim_vec) {
    set_at_row(result, unmerged[sim_idx], sim_idx);
  }
  return result;
}
} // namespace SIR_SBM