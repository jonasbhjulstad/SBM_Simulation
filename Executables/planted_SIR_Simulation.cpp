#include "Static_RNG/distributions.hpp"
#define USE_TBB_DEBUG
#include <Sycl_Graph/SBM_Generation.hpp>
#include <Sycl_Graph/SIR_SBM.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>

using namespace Sycl_Graph::SBM;

// void simulate_to_file(Network_t &network, const tp_list_t &tp_list,
//                       const std::string &file_path, uint32_t sim_idx)
// {
//   auto [initial_state, iter_data] = network.simulate_groups(tp_list);

//   auto linewrite([&](std::ofstream &file, const auto &iter)
//                  {
//       std::for_each(iter.begin(), iter.end(),
//                     [&](auto &t_i_i) { file << t_i_i << ","; });
//       file << "\n"; });
//   std::ofstream tot_traj_f(file_path + "traj_tot_" + std::to_string(sim_idx)
//   +
//                            ".csv");
//   std::ofstream p_I_f(file_path + "p_Is_" + std::to_string(sim_idx) +
//   ".csv"); std::ofstream delta_I_f(file_path + "delta_Is_" +
//   std::to_string(sim_idx) +
//                           ".csv");
//   std::ofstream delta_R_f(file_path + "delta_Rs_" + std::to_string(sim_idx) +
//                           ".csv");
//   std::vector<uint32_t> state = initial_state;
//   linewrite(tot_traj_f, state);
//   std::for_each(iter_data.begin(), iter_data.end(), [&](auto &t)
//                 {

//     state = t.next_state(state);
//     linewrite(tot_traj_f, state);
//     linewrite(delta_I_f, t.community_infs);
//     linewrite(delta_R_f, t.community_rec); });
//   // filewrite(tot_traj_f, tot_traj);
//   // filewrite(delta_I_f, delta_Is);
//   // filewrite(delta_R_f, delta_Rs);

//   std::for_each(tp_list.begin(), tp_list.end(), [&](auto &t_i)
//                 {
//       std::for_each(t_i.p_Is.begin(), t_i.p_Is.end(),
//                     [&](auto &t_i_i) { p_I_f << t_i_i << ","; });
//       p_I_f << "\n"; });
// }

// void simulate_to_file(std::vector<Network_t> &networks,
//                       const std::vector<tp_list_t> &tp_lists,
//                       const std::string &file_path)
// {
//   std::vector<uint32_t> file_idx(networks.size());
//   std::generate(file_idx.begin(), file_idx.end(),
//                 [n = 0]() mutable
//                 { return n++; });

//   std::vector<std::tuple<Network_t *, const tp_list_t *, uint32_t>> tup(
//       networks.size());
//   std::generate(tup.begin(), tup.end(),
//                 [n = -1, &networks, &tp_lists, &file_idx]() mutable
//                 {
//                   n++;
//                   return std::make_tuple(&networks[n], &tp_lists[n], n);
//                 });

//   std::for_each(std::execution::par_unseq, tup.begin(), tup.end(),
//                 [&](auto &t)
//                 {
//                   auto network = std::get<0>(t);
//                   auto tp_list = std::get<1>(t);
//                   auto sim_idx = std::get<2>(t);
//                   simulate_to_file(*network, *tp_list, file_path, sim_idx);
//                 });
// }

std::vector<uint32_t> iteration_lists_to_community_state(
    const std::vector<uint32_t> &prev_state,
    const std::vector<uint32_t> &connection_infs,
    const std::vector<uint32_t> &recoveries, const auto &connection_targets) {
  // find largest element in connection_targets pairs
uint32_t N_communities = prev_state.size()/3;

  // create vector of vectors to hold community states
  std::vector<uint32_t> community_state = prev_state;
  std::vector<uint32_t> delta_Is(N_communities, 0);
  for (int i = 0; i < connection_infs.size(); i++) {
    // get community of connection
    auto community = connection_targets[i];
    delta_Is[community] += connection_infs[i];
  }

  for(int i = 0; i < N_communities; i++)
  {
    community_state[3*i] -= delta_Is[i];
    community_state[3*i + 1] += delta_Is[i] - recoveries[i];
    community_state[3*i + 2] += recoveries[i];
  }

  return community_state;
}

auto linewrite(std::ofstream &file, const auto &iter) {
  std::for_each(iter.begin(), iter.end(),
                [&](auto &t_i_i) { file << t_i_i << ","; });
  file << "\n";
}

void iterations_to_file(const auto& inf_events, const auto& community_infs, const auto& community_recs, const std::string &file_path,
                        uint32_t sim_idx) {

  std::ofstream inf_events_f(file_path + "inf_events_" +
                             std::to_string(sim_idx) + ".csv");
  std::ofstream community_infs_f(file_path + "community_infs_" +
                                 std::to_string(sim_idx) + ".csv");
  std::ofstream community_recs_f(file_path + "community_recs_" +
                                 std::to_string(sim_idx) + ".csv");
  for(int i = 0; i < inf_events.size(); i++)
  {
    linewrite(inf_events_f, inf_events[i]);
    linewrite(community_infs_f, community_infs[i]);
    linewrite(community_recs_f, community_recs[i]);
  }
}

void write_community_traj(const auto &init_state, const auto& inf_events, const auto& connection_infs, const auto& community_recs, const auto& connection_targets,
                          const std::string &file_path, uint32_t sim_idx) {
  std::vector<uint32_t> state;
  //flatten init_state
  for (auto &community : init_state) {
    state.insert(state.end(), community.begin(), community.end());
  }

  std::ofstream traj_f(file_path + "/community_traj_" +
                       std::to_string(sim_idx) + ".csv");
  for(int i = 0; i < inf_events.size(); i++)
  {
    state = iteration_lists_to_community_state(state, connection_infs[i], community_recs[i],
                                               connection_targets);
    linewrite(traj_f, state);
  }
}

void write_tot_traj(const auto &init_state, const auto& inf_events, const auto& connection_infs, const auto& community_recs,
                    const std::string &file_path, uint32_t sim_idx) {
  std::ofstream tot_traj_f(file_path + "/tot_traj_" + std::to_string(sim_idx) +
                           ".csv");
  auto state = std::vector<uint32_t>(init_state[0].size(), 0);

  for (int i = 0; i < init_state.size(); i++) {
    for (int j = 0; j < init_state[i].size(); j++) {
      state[j] += init_state[i][j];
    }
  }
  for(int i = 0; i < inf_events.size(); i++)
  {

    auto delta_I =
        std::accumulate(connection_infs[i].begin(), connection_infs[i].end(), 0);
    auto delta_R =
        std::accumulate(community_recs[i].begin(), community_recs[i].end(), 0);
    delta_I = (delta_I > state[0]) ? state[0] : delta_I;
    delta_R = (delta_R > state[1]) ? state[1] : delta_R;

    state[0] -= delta_I;
    state[1] += delta_I - delta_R;
    state[2] += delta_R;
    linewrite(tot_traj_f, state);
  }
}

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

auto generate_Ng_Nt_p_Is(uint32_t Ng, float p_I_min, float p_I_max, uint32_t Nt,
                         const auto &edge_list_sizes, uint32_t seed) {
  std::vector<uint32_t> p_I_seeds(Ng);
  std::mt19937 rd(seed);
  std::generate(p_I_seeds.begin(), p_I_seeds.end(), [&rd]() { return rd(); });

  // generate
  std::vector<std::vector<std::vector<std::vector<float>>>> p_Is(Ng);
  std::transform(std::execution::par_unseq, p_I_seeds.begin(), p_I_seeds.end(),
                 p_Is.begin(), [&](auto p_I_seed) {
                   return generate_p_Is(edge_list_sizes, p_I_min, p_I_max, Nt,
                                        p_I_seed);
                 });
  return p_Is;
}

auto run_Ng_Nt_simulations(const auto &SBM_lists, const auto &p_Is, float p_R,
                           float p_I0, float p_R0, float Nt, uint32_t Ng,
                           std::vector<sycl::queue> &qs, uint32_t seed = 42) {
  std::mt19937 rd(seed);
  std::vector<uint32_t> seeds(Ng);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  // zip sbm_lists, p_Is, seeds

  std::vector<std::tuple<std::pair<SBM_Node_List_t, SBM_Edge_List_t>,
                         std::vector<std::vector<std::vector<float>>>, uint32_t,
                         sycl::queue>>
      zip(Ng);
  for (uint32_t i = 0; i < Ng; i++) {
    zip[i] = std::make_tuple(SBM_lists[i], p_Is[i], seeds[i], qs[i]);
  }

  std::vector<std::pair<std::vector<uint32_t>,
                        std::vector<std::vector<Iteration_Buffers_t>>>>
      result(Ng);
  std::transform(std::execution::par_unseq, zip.begin(), zip.end(),
                 result.begin(), [p_I0, p_R0, p_R, Nt](const auto &z) {
                   auto SBM_list = std::get<0>(z);
                   auto p_I = std::get<1>(z);
                   auto sim_seed = std::get<2>(z);
                   auto q = std::get<3>(z);
                   // create

                   return SBM_simulate(p_I, p_R, p_I0, p_R0, SBM_list.first,
                                       SBM_list.second, q, sim_seed);
                 });
  return result;
}

void simulate_to_file(const auto &SBM_node_list, const auto &SBM_edge_list,
                      const auto &SBM_connection_targets, const auto &p_Is,
                      float p_R, float p_I0, float p_R0, float Nt, uint32_t Ng,
                      sycl::queue &q, const std::string &file_path,
                      uint32_t sim_idx, uint32_t seed = 42) {

  auto [init_state, iteration_bufs] = SBM_simulate(
      p_Is, p_R, p_I0, p_R0, SBM_node_list, SBM_edge_list, q, seed);
  // std::vector<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>> iteration_data(iteration_bufs.size());
  // std::transform(iteration_bufs.begin(), iteration_bufs.end(),
  //                iteration_data.begin(), [](auto &buf) {
  //                  return read_iteration_buffer(buf);
  //                });
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
    connection_infs.push_back(connection_inf)
  }
  write_tot_traj(init_state, inf_events, connection_infs, community_recs, file_path, sim_idx);

  write_community_traj(init_state, inf_events, connection_infs, community_recs, SBM_connection_targets, file_path, sim_idx);
  iterations_to_file(inf_events, community_infs, community_recs, file_path, sim_idx);
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
  // auto edge_list_sizes = std::vector<uint32_t>(Ng);
  // std::transform(SBM_lists.begin(), SBM_lists.end(), edge_list_sizes.begin(),
  //                [](auto &SBM_list) { return SBM_list.second.size(); });
  uint32_t N_community_connections = SBM_edge_list.size();

  float p_I_min = 1e-4f;
  float p_I_max = 1e-2f;
  float p_R = 0.1f;

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