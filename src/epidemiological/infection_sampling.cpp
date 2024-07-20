#include <SIR_SBM/epidemiological/infection_sampling.hpp>
#include <SIR_SBM/epidemiological/infection_count.hpp>
#include <SIR_SBM/graph/indices.hpp>
#include <SIR_SBM/random/random.hpp>
#include <SIR_SBM/utils/numeric.hpp>
#include <SIR_SBM/utils/occurrence.hpp>
#include <cppitertools/combinations_with_replacement.hpp>

namespace SIR_SBM {

Infection_Sampler::Infection_Sampler(uint32_t N_sims, uint32_t N_partitions,
                                     uint32_t N_connections, uint32_t Nt)
    : N_sims(N_sims), N_partitions(N_partitions), N_connections(N_connections),
      Nt(Nt) {}
uint32_t Infection_Sampler::partition_idx(uint32_t sim_idx, uint32_t p_idx,
                                          uint32_t t_idx) const {
  return get_partition_idx(sim_idx, p_idx, t_idx, N_partitions, Nt + 1);
}
uint32_t Infection_Sampler::from_connection_idx(uint32_t sim_idx,
                                                uint32_t con_idx,
                                                uint32_t t_idx) const {
  return get_from_connection_idx(sim_idx, con_idx, t_idx, N_connections, Nt);
}

uint32_t Infection_Sampler::to_connection_idx(uint32_t sim_idx,
                                              uint32_t con_idx,
                                              uint32_t t_idx) const {
  return get_to_connection_idx(sim_idx, con_idx, t_idx, N_connections, Nt);
}
std::vector<int> Infection_Sampler::get_connection_indices(int p_idx) const {
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
std::vector<uint32_t> Infection_Sampler::get_t_connections(uint32_t sim_idx,
                                                           uint32_t t) {
  std::vector<uint32_t> result(2 * N_connections);
  for (int c_idx = 0; c_idx < N_connections; c_idx++) {
    result[2 * c_idx] = to_connection_idx(sim_idx, c_idx, t);
    result[2 * c_idx + 1] = from_connection_idx(sim_idx, c_idx, t);
  }
  return result;
}
std::vector<uint32_t> Infection_Sampler::get_partition_connection_contacts(
    const std::vector<uint32_t> &contact_events, int p_idx) const {
  auto indices = get_connection_indices(p_idx);
  std::vector<uint32_t> result(indices.size());
  for (int c_idx = 0; c_idx < indices.size(); c_idx++) {
    result[c_idx] = contact_events[indices[c_idx]];
  }
  return result;
}

std::vector<uint32_t> Infection_Sampler::sample_infections(
    const std::vector<uint32_t> &contact_events,
    const std::vector<Population_Count> &population_count, uint32_t sim_idx,
    uint32_t p_idx, uint32_t t_idx, std::mt19937 &rng) {

  auto con_indices = get_connection_indices(p_idx);
  std::vector<uint32_t> connection_contacts = get_partition_connection_contacts(
      get_t_connections(sim_idx, t_idx), p_idx);

  uint32_t new_infs = get_new_infections(population_count, sim_idx, p_idx,
                                         N_partitions, t_idx, Nt + 1);
  if (new_infs) {
    auto inf_index_samples =
        discrete_finite_sample(rng, connection_contacts, new_infs);
    return count_occurrences(inf_index_samples, 2 * N_connections);
  } else {
    return std::vector<uint32_t>(2 * N_connections, 0);
  }
}
void Infection_Sampler::assign_t_infections(
    std::vector<uint32_t> &infections, std::vector<uint32_t> &infections_pt,
    uint32_t sim_idx, uint32_t t) {
  for (int con_idx = 0; con_idx < N_connections; con_idx++) {
    infections[from_connection_idx(sim_idx, con_idx, t)] +=
        infections_pt[2 * con_idx];
    infections[to_connection_idx(sim_idx, con_idx, t)] +=
        infections_pt[2 * con_idx + 1];
  }
}

std::vector<uint32_t> Infection_Sampler::sample_infections(
    const std::vector<uint32_t> &contact_events,
    const std::vector<Population_Count> &population_count, int seed) {

  auto rngs = generate_rngs(N_sims, seed);
  auto infections = std::vector<uint32_t>(N_sims * (2 * N_connections) * Nt, 0);
  auto infections_pt = std::vector<uint32_t>(N_connections * 2, 0);
  for (int sim_idx = 0; sim_idx < N_sims; sim_idx++) {
    for (int t_idx = 0; t_idx < Nt; t_idx++) {
      for (int p_idx = 0; p_idx < N_partitions; p_idx++) {
        infections_pt = sample_infections(contact_events, population_count,
                                          sim_idx, p_idx, t_idx, rngs[sim_idx]);
        assign_t_infections(infections, infections_pt, sim_idx, t_idx);
      }

      std::fill(infections_pt.begin(), infections_pt.end(), 0);
    }
  }

  return infections;
}
} // namespace SIR_SBM