#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Routines.hpp>
#include <Sycl_Graph/SBM_write.hpp>
#include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <execution>
namespace Sycl_Graph::SBM {
template <>
void linewrite<>(std::ofstream &file, const std::vector<uint32_t> &iter);

void linewrite(std::ofstream &file, const std::vector<uint32_t> &state_iter) {
  for (const auto &t_i_i : state_iter) {
    file << t_i_i;
    if (&t_i_i != &state_iter.back())
      file << ",";
    else
      file << "\n";
  }
}

void linewrite(std::ofstream &file,
               const std::vector<std::array<uint32_t, 3>> &state_iter) {
  for (const auto &t_i : state_iter) {
    for (const auto &t_i_i : t_i) {
      file << t_i_i;
      file << ",";
    }
  }
  file << "\n";
}

void linewrite(std::ofstream &file, const std::vector<Edge_t> &iter) {
  for (auto &t_i_i : iter) {
    file << t_i_i.first << "," << t_i_i.second;
    if (&t_i_i != &iter.back())
      file << ",";
    else
      file << "\n";
  }
}

void columnwrite(std::ofstream &file, const std::vector<Edge_t> &iter) {
  for (auto &t_i_i : iter) {
    file << t_i_i.first << "," << t_i_i.second << "\n";
  }
}

void simulate_to_file(const SBM_Graph_t &G, const SIR_SBM_Param_t &param,
                      sycl::queue &q, const std::string &file_path,
                      uint32_t sim_idx, uint32_t seed, uint32_t N_wg) {
  uint32_t Nt = param.p_I.size();
  SIR_SBM_Network network(G, param.p_I0, param.p_R, q, seed, N_wg, param.p_R0);
  auto sim_event = network.simulate(param);
  sim_event.wait();
  // auto [community_trajectory, connection_events_trajectory,
  //       connection_infections_trajectory] = network.read_trajectory();

  // std::filesystem::create_directories(file_path);

  // std::ofstream community_traj_f(file_path + "community_trajectory_" +
  //                                std::to_string(sim_idx) + ".csv");
  // std::ofstream connection_events_f(file_path + "connection_events_" +
  //                                   std::to_string(sim_idx) + ".csv");

  // std::ofstream connection_infections_f(file_path + "connection_infections_" +
  //                                       std::to_string(sim_idx) + ".csv");

  // std::for_each(community_trajectory.begin(), community_trajectory.end(),
  //               [&](auto &community_trajectory_i) {
  //                 linewrite(community_traj_f, community_trajectory_i);
  //               });
  // std::for_each(connection_events_trajectory.begin(),
  //               connection_events_trajectory.end(),
  //               [&](auto &connection_events_i) {
  //                 linewrite(connection_events_f, connection_events_i);
  //               });

  // std::for_each(connection_infections_trajectory.begin(),
  //               connection_infections_trajectory.end(),
  //               [&](auto &connection_infections_i) {
  //                 linewrite(connection_infections_f, connection_infections_i);
  //               });
  // std::ofstream p_I_f(file_path + "p_Is_" + std::to_string(sim_idx) + ".csv");
  // std::vector<std::vector<float>> p_I_duplicated;
  // std::transform(param.p_I.begin(), param.p_I.end(),
  //                std::back_inserter(p_I_duplicated),
  //                [&](const auto &p_I_i) { std::vector<float> p_I_dup = p_I_i;
  //                p_I_dup.insert(p_I_dup.end(), p_I_i.begin(), p_I_i.end());
  //                return p_I_dup;});
  // std::for_each(p_I_duplicated.begin(), p_I_duplicated.end(),
  //               [&](auto &p_I_i) { linewrite(p_I_f, p_I_i); });
}

void parallel_simulate_to_file(const SBM_Graph_t &G,
                               const std::vector<SIR_SBM_Param_t> &params,
                               std::vector<sycl::queue> &qs,
                               const std::string &file_path, uint32_t N_sim,
                               uint32_t seed) {
#ifdef DEBUG
  assert(params.size() == qs.size() &&
         "Number of parameters and queues must be equal");
  assert(params[0].size() == G.N_connections &&
         "Number of parameters must be equal to number of connections");
  assert(qs.size() == N_sim &&
         "Number of queues must be equal to number of simulations");
#endif
  uint32_t N_sims = params.size();
  std::vector<uint32_t> seeds(N_sims);
  Static_RNG::default_rng rng(seed);
  std::generate(seeds.begin(), seeds.end(),
                [&rng]() { return (uint32_t)rng(); });
  std::vector<SBM_Graph_t> Gs(N_sim, G);

  std::vector<std::tuple<const SBM_Graph_t *, const SIR_SBM_Param_t *,
                         sycl::queue *, const std::string, uint32_t, uint32_t>>
      zip;
  for (uint32_t i = 0; i < N_sim; i++) {
    zip.push_back(
        std::make_tuple(&Gs[i], &params[i], &qs[i], file_path, i, seeds[i]));
  }

  std::for_each(std::execution::par_unseq, zip.begin(), zip.end(), [&](auto z) {
    simulate_to_file(*std::get<0>(z), *std::get<1>(z), *std::get<2>(z),
                     std::get<3>(z), std::get<4>(z), std::get<5>(z));
  });
}

void parallel_simulate_to_file(
    const std::vector<SBM_Graph_t> &Gs,
    const std::vector<std::vector<SIR_SBM_Param_t>> &params,
    std::vector<std::vector<sycl::queue>> &qs,
    const std::vector<std::string> &file_paths, uint32_t seed) {
#ifdef DEBUG
  assert(Gs.size() == params.size() &&
         "Number of graphs and parameters must be equal");
  assert(Gs.size() == qs.size() && "Number of graphs and queues must be equal");
  assert(file_paths.size() == Gs.size() &&
         "Number of file paths and graphs must be equal");
  assert(std::all_of(params.begin(), params.end(),
                     [](auto p) { return p.size() == params[0].size(); }) &&
         "Number of parameters must be equal for each graph");
#endif
  std::vector<uint32_t> seeds(Gs.size());
  std::mt19937 rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<uint32_t> N_sims(Gs.size());
  std::transform(params.begin(), params.end(), N_sims.begin(),
                 [](auto p) { return p.size(); });

  // zip
  std::vector<
      std::tuple<const SBM_Graph_t *, const std::vector<SIR_SBM_Param_t> *,
                 std::vector<sycl::queue> *, std::string, uint32_t, uint32_t>>
      zip(Gs.size());
  for (uint32_t i = 0; i < Gs.size(); i++) {
    zip[i] = std::make_tuple(&Gs[i], &params[i], &qs[i], file_paths[i],
                             N_sims[i], seeds[i]);
  }

  std::for_each(std::execution::par_unseq, zip.begin(), zip.end(),
                [&](const auto &z) {
                  parallel_simulate_to_file(*std::get<0>(z), *std::get<1>(z),
                                            *std::get<2>(z), std::get<3>(z),
                                            std::get<4>(z), std::get<5>(z));
                });
}

} // namespace Sycl_Graph::SBM
