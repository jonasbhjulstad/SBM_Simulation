#pragma once
#hdr
#include <SIR_SBM/csv.hpp>
#include <casadi/casadi.hpp>
#include <filesystem>
#include <fstream>
#include <tuple>
#end

#src
#include <SIR_SBM/vector.hpp>
#include <cppitertools/combinations_with_replacement.hpp>
#end

namespace SIR_SBM {

std::tuple<casadi::DM, casadi::DM>
connection_expand_population(const std::tuple<casadi::DM, casadi::DM> &data, uint32_t N_connections) {
  using namespace casadi;
  auto [population_counts, infection_counts] = data;
  // population_counts dim 1 is N_communities
  // infection_counts dim 1 is N_connections

  auto N_communities = population_counts.size1();
  auto N_directed_connections = infection_counts.size1();

  DM sources = DM::zeros(population_counts.size1(), N_directed_connections);
  DM targets = DM::zeros(population_counts.size1(), N_directed_connections);
  uint32_t con_idx = 0;
  auto population_slice = [](int idx) {
    return Slice<int>(idx * 3, idx * 3 + 3);
  };
  for (auto comb : iter::combinations_with_replacement(make_iota(N_communities), 2)) {
    auto from_idx = comb[0];
    auto to_idx = comb[1];
    auto con_slice = population_slice(con_idx);
    auto from_slice = Slice(from_idx * 3, from_idx * 3 + 3);
    auto to_slice = Slice(to_idx * 3, to_idx * 3 + 3);

    sources(Slice(), population_slice(con_idx)) =
        population_counts(Slice(), population_slice(from_idx));
    targets(Slice(), population_slice(con_idx)) =
        population_counts(Slice(), population_slice(to_idx));
    con_idx++;
  }
  return std::make_tuple(sources, targets);
}
// load csv into MX matrix
std::tuple<casadi::DM, casadi::DM, casadi::DM>
regression_data_from_simulations(const std::filesystem::path &filenameprefix,
                                 uint32_t N_communities, uint32_t N_connections,
                                 uint32_t N_sims, uint32_t Nt) {

  auto community_state = read_csv<uint32_t>(filenameprefix / "population_count_",
                                 N_communities, N_sims, Nt + 1);
  auto infection_count = read_csv<uint32_t>(filenameprefix / "infected_count_",
                                 N_connections * 2, N_sims, Nt);

  using namespace casadi;
  auto linvec_to_dm = [](const Vec3D<uint32_t> &vec, uint32_t start, uint32_t end) {
    auto shape = vec.dimensions();
    auto N0 = shape[0];
    auto N1 = shape[1];
    auto N2 = shape[2];
    DM result(N0 * N1 * (end - start));
    for (uint32_t i = 0; i < N0; i++) {
      for (uint32_t j = 0; j < N1; j++) {
        for (uint32_t k = start; k < end; k++) {
          result(i * N1 * N2 + j * N2 + k) = vec(i,j,k);
        }
      }
    }
    return result;
  };

  DM population_counts =
      linvec_to_dm(community_state, 0, community_state.dimensions()[2] - 1);

  DM infection_counts = linvec_to_dm(infection_count, 0, infection_count.dimensions()[2]);

  auto [population_sources, population_targets] = connection_expand_population(
      std::make_tuple(population_counts, infection_counts),
      N_connections);
  return std::make_tuple(population_sources, population_targets,
                         infection_counts);
}

} // namespace SIR_SBM