// infection_sampling.cpp
//

#include "infection_sampling.hpp"
#define LZZ_INLINE inline
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  std::vector <int> get_connection_indices (int N_partitions, int p_idx)
#line 17 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
                                                                     {
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
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 31 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  std::vector <uint32_t> get_partition_connection_contacts (Vec2DView <uint32_t> const & contact_events, int N_partitions, int p_idx)
#line 33 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
               {
  auto indices = get_connection_indices(N_partitions, p_idx);
  std::vector<uint32_t> result(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    result[i] = contact_events(p_idx, indices[i]);
  }
  return result;
}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 42 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  std::vector <uint32_t> sample_infections (Vec2DView <uint32_t> const & contact_events, Vec2DView <Population_Count> const & population_count, int p_idx, int t_idx, std::mt19937_64 & rng)
#line 45 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
                                     {

  auto N_connections = contact_events.N0 / 2;
  auto N_partitions = population_count.N0;
  auto Nt = contact_events.N1;
  auto con_indices = get_connection_indices(N_partitions, p_idx);
  std::vector<uint32_t> connection_contacts = get_partition_connection_contacts(
      contact_events.column_view(t_idx), N_partitions, p_idx);

  auto new_infs = get_new_infections(population_count, p_idx, t_idx);
  auto inf_index_samples =
      discrete_finite_sample(rng, connection_contacts, new_infs);
  return count_occurrences(inf_index_samples, 2 * N_connections);
}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 60 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  LinVec2D <uint32_t> simulation_sample_infections (Vec2DView <uint32_t> const & contact_events, Vec2DView <Population_Count> const & population_count, std::mt19937_64 & rng)
#line 63 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
                          {

  auto N_connections = contact_events.N0 / 2;
  auto N_partitions = population_count.N0;
  auto Nt = contact_events.N1;

  LinVec2D<uint32_t> result(2 * N_connections, Nt);

  auto add_to_row = [](Vec2DView<uint32_t> &res,
                       const std::vector<uint32_t> &&infs, int t) {
    for (int p_idx = 0; p_idx < infs.size(); p_idx++) {
      res(t, p_idx) += infs[p_idx];
    }
  };

  for (auto t_idx : make_iota(Nt)) {
    for (auto p_idx : make_iota(N_partitions)) {
      result(p_idx) += sample_infections(contact_events, population_count,
                                               p_idx, t_idx, rng);
    }
  }
  return result;
}
}
#line 15 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
namespace SIR_SBM
{
#line 88 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
  Vec3D <uint32_t> sample_infections (Vec3D <uint32_t> const & contact_events, Vec3D <Population_Count> const & population_count, int seed)
#line 90 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//infection_sampling.hpp"
                            {
  auto shape = contact_events.dimensions();
  auto N_sims = shape[0];
  auto N_connections = contact_events.dimensions()[1] / 2;
  auto Nt = contact_events.dimensions()[2];
  auto N_partitions = population_count.dimensions()[1];

  auto rngs = generate_rngs<std::mt19937_64>(N_sims, seed);
  
  Vec3D<uint32_t> result(N_sims, 2 * N_connections, Nt);
  for (auto sim_idx : make_iota(N_sims)) {
    result(sim_idx) = simulation_sample_infections(
        contact_events(sim_idx), population_count(sim_idx),
        rngs[sim_idx]);
  }
  return result;
}
}
#undef LZZ_INLINE
