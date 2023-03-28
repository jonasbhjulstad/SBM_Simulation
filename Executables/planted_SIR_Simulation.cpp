#include "Static_RNG/distributions.hpp"
#include <Sycl_Graph/SBM_Generation.hpp>
#include <Sycl_Graph/SIR_SBM.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/SBM_write.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>

using namespace Sycl_Graph::SBM;



auto generate_p_Is(uint32_t N_community_connections, float p_I_min,
                   float p_I_max, uint32_t Nt, uint32_t seed = 42) {
  std::vector<Static_RNG::default_rng> rngs(Nt);
  std::mt19937 rd(seed);
  std::vector<uint32_t> seeds(Nt);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::transform(seeds.begin(), seeds.end(), rngs.begin(),
                 [](auto seed) { return Static_RNG::default_rng(seed); });

  std::vector<std::vector<float>> p_Is(
      Nt, std::vector<float>(N_community_connections));
  // std::for_each(p_Is.begin(), p_Is.end(),
  //               [&](auto &p_I) { p_I.resize(N_community_connections); });

  std::transform(
      std::execution::par_unseq, rngs.begin(), rngs.end(), p_Is.begin(),
      [&](auto &rng) {
        Static_RNG::uniform_real_distribution<> dist(p_I_min, p_I_max);
        std::vector<float> p_I(N_community_connections);
        std::generate(p_I.begin(), p_I.end(), [&]() { return dist(rng); });
        return p_I;
      });

  return p_Is;
}

void simulate_to_file(const auto &SBM_node_list, const auto &SBM_edge_list,
                      const auto &SBM_connection_targets, const auto &p_Is,
                      float p_R, float p_I0, float p_R0, float Nt, uint32_t Ng,
                      sycl::queue &q, const std::string &file_path,
                      uint32_t sim_idx, uint32_t seed = 42) {

  auto [init_state, iteration_bufs] = SBM_simulate(
      p_Is, p_R, p_I0, p_R0, SBM_node_list, SBM_edge_list,SBM_connection_targets, q, seed);
  std::vector<std::vector<uint32_t>> inf_events;
  std::vector<std::vector<uint32_t>> community_infs;
  std::vector<std::vector<uint32_t>> connection_infs;
  std::vector<std::vector<uint32_t>> community_recs;
  inf_events.reserve(iteration_bufs.size());
  community_infs.reserve(iteration_bufs.size());
  community_recs.reserve(iteration_bufs.size());
  connection_infs.reserve(iteration_bufs.size());

  for(int i = 0; i < iteration_bufs.size(); i++)
  {
    auto [inf_event, community_inf, community_rec, connection_inf] = read_iteration_buffer(iteration_bufs[i]);
    inf_events.push_back(inf_event);
    community_infs.push_back(community_inf);
    community_recs.push_back(community_rec);
    connection_infs.push_back(connection_inf);
  }
  write_tot_traj(init_state, inf_events, connection_infs, community_recs, file_path, sim_idx);

  write_community_traj(init_state, inf_events, connection_infs, community_recs, SBM_connection_targets, community_infs, file_path, sim_idx);
  iterations_to_file(inf_events, community_infs, community_recs, connection_infs, file_path, sim_idx);
}

auto create_seeds_idx(uint32_t N_sims, uint32_t seed) {
  std::vector<uint32_t> sim_idx(N_sims);
  std::iota(sim_idx.begin(), sim_idx.end(), 0);
  std::vector<uint32_t> seeds(N_sims);
  std::mt19937 rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  // zip sim_idx, seeds
  std::vector<std::pair<uint32_t, uint32_t>> zip(N_sims);
  for (uint32_t i = 0; i < N_sims; i++) {
    zip[i] = std::make_pair(sim_idx[i], seeds[i]);
  }
  return zip;
}

void single_simulation() {
  float p_I0 = 0.1;
  float p_R0 = 0.0;
  uint32_t N_clusters = 10;
  uint32_t N_pop = 100;
  float p_in = 1.0f;
  float p_out = 0.0f;
  uint32_t N_sims = 1;
  sycl::queue q(sycl::gpu_selector_v);
  uint32_t Nt = 70;
  uint32_t seed = 47;
  uint32_t N_threads = 10;

  const auto res =
      create_planted_SBM(N_pop, N_clusters, p_in, p_out, true, N_threads, seed);
  const auto SBM_node_list = std::get<0>(res);
  const auto SBM_edge_list = std::get<1>(res);
  const auto SBM_connection_targets = std::get<2>(res);
  uint32_t N_community_connections = SBM_edge_list.size();

  float p_I_min = 1e-3f;
  float p_I_max = 1e-2f;
  float p_R = 1e-1f;

  auto p_Is =
      generate_p_Is(N_community_connections, p_I_min, p_I_max, Nt, seed);

  auto seeds_idx = create_seeds_idx(N_sims, seed);

  // write to file
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");

  std::for_each(std::execution::par_unseq, seeds_idx.begin(), seeds_idx.end(),
                [&](auto &si) {
                  simulate_to_file(
                      SBM_node_list, SBM_edge_list, SBM_connection_targets,
                      p_Is, p_R, p_I0, p_R0, Nt, N_sims, q,
                      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) +
                          "/SIR_sim/",
                      si.first, si.second);
                });
}
int main() {

  std::vector<uint32_t> dummy(1);
  std::for_each(std::execution::seq, dummy.begin(), dummy.end(),
                [](auto &d) { single_simulation(); });

  return 0;
}