#include <SIR_SBM/regression/regression.hpp>
#include <SIR_SBM/utils/csv.hpp>
#include <SIR_SBM/utils/numeric.hpp>
#include <cppitertools/combinations_with_replacement.hpp>

#include <fstream>

namespace SIR_SBM {

std::tuple<casadi::DM, casadi::DM>
connection_expand_population(const std::tuple<casadi::DM, casadi::DM> &data,
                             uint32_t N_connections) {
  using namespace casadi;
  auto [population_counts, infection_counts] = data;
  // population_counts dim 1 is N_communitiesdim
  // infection_counts dim 1 is N_connections

  auto N_communities = population_counts.size1();
  auto N_directed_connections = infection_counts.size1();

  DM sources = DM::zeros(population_counts.size1(), N_directed_connections);
  DM targets = DM::zeros(population_counts.size1(), N_directed_connections);
  uint32_t con_idx = 0;
  auto population_slice = [](int idx) {
    return Slice(idx * 3, idx * 3 + 3, 1);
  };
  for (auto comb :
       iter::combinations_with_replacement(make_iota(N_communities), 2)) {
    int from_idx = comb[0];
    int to_idx = comb[1];
    auto con_slice = population_slice(con_idx);
    auto from_slice = Slice(from_idx * 3, from_idx * 3 + 3, 1);
    auto to_slice = Slice(to_idx * 3, to_idx * 3 + 3, 1);

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

  auto community_state = read_csv_flat(filenameprefix / "population_count_",
                                       N_communities, N_sims, Nt + 1);
  auto infection_count = read_csv_flat(filenameprefix / "infected_count_",
                                       N_connections * 2, N_sims, Nt);

  using namespace casadi;

  auto population_counts = DM(community_state);
  auto infection_counts = DM(infection_count);

  auto [population_sources, population_targets] = connection_expand_population(
      std::make_tuple(population_counts, infection_counts), N_connections);
  return std::make_tuple(population_sources, population_targets,
                         infection_counts);
}

} // namespace SIR_SBM