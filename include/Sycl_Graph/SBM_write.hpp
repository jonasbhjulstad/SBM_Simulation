#ifndef SYCL_GRAPH_SBM_WRITE_HPP
#define SYCL_GRAPH_SBM_WRITE_HPP
#include <vector>
#include <filesystem>
#include <Sycl_Graph/SBM_types.hpp>
#include <Sycl_Graph/SIR_SBM.hpp>
namespace Sycl_Graph::SBM
{

  std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>
  read_iteration_buffer(const Iteration_Buffers_t &buffers)
  {
    auto inf_event_buf = std::get<0>(buffers);
    auto community_infs_buf = std::get<1>(buffers);
    auto community_recs = std::get<2>(buffers);
    std::vector<uint32_t> connection_infs = std::get<3>(buffers);
    auto event = std::get<3>(buffers);
    auto inf_event_acc = inf_event_buf.get_access<sycl::access::mode::read>();
    auto community_infs_acc =
        community_infs_buf.get_access<sycl::access::mode::read>();
    std::vector<uint32_t> inf_event(inf_event_acc.size());
    std::vector<uint32_t> community_infs(community_infs_acc.size());
    for (int i = 0; i < inf_event_acc.size(); i++)
    {
      inf_event[i] = inf_event_acc[i];
    }
    for (int i = 0; i < community_infs_acc.size(); i++)
    {
      community_infs[i] = community_infs_acc[i];
    }
    return std::make_tuple(inf_event, community_infs, community_recs, connection_infs);
  }

std::vector<uint32_t> iteration_lists_to_community_state(
    const std::vector<uint32_t> &prev_state,
    const std::vector<uint32_t> &connection_infs,
    const std::vector<uint32_t> &recoveries, const std::vector<uint32_t> &connection_targets, const std::vector<uint32_t>& community_infs) {
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
  std::vector<int> c_state(3*N_communities);
  std::vector<uint32_t> total_state(3,0);
  for(int i = 0; i < N_communities; i++)
  {
    community_state[3*i] -= delta_Is[i];
    community_state[3*i + 1] += delta_Is[i] - recoveries[i];
    community_state[3*i + 2] += recoveries[i];
    total_state[0] += community_state[3*i];
    total_state[1] += community_state[3*i + 1];
    total_state[2] += community_state[3*i + 2];
    c_state[3*i] = community_state[3*i];
    c_state[3*i + 1] = community_state[3*i + 1];
    c_state[3*i + 2] = community_state[3*i + 2];
  }

  return community_state;
}

auto linewrite(std::ofstream &file, const auto &iter) {
  // std::for_each(iter.begin(), iter.end(),
  //               [&](auto &t_i_i) { file << t_i_i << ","; });
  for (auto &t_i_i : iter) {
    file << t_i_i;
    if(&t_i_i != &iter.back())
      file << ",";
    else
      file << "\n";
  }
}

void iterations_to_file(const auto& inf_events, const auto& community_infs, const auto& community_recs, const auto& connection_infs, const auto& p_Is, const std::string &file_path,
                        uint32_t sim_idx) {

  std::ofstream inf_events_f(file_path + "infection_events_" +
                             std::to_string(sim_idx) + ".csv");
  std::ofstream community_infs_f(file_path + "community_infs_" +
                                 std::to_string(sim_idx) + ".csv");
  std::ofstream community_recs_f(file_path + "community_recs_" +
                                 std::to_string(sim_idx) + ".csv");
  std::ofstream connection_infs_f(file_path + "connection_infs_" +
                                 std::to_string(sim_idx) + ".csv");
  std::ofstream p_Is_f(file_path + "p_Is_" +
                                 std::to_string(sim_idx) + ".csv");
  for(int i = 0; i < inf_events.size(); i++)
  {
    linewrite(inf_events_f, inf_events[i]);
    linewrite(community_infs_f, community_infs[i]);
    linewrite(community_recs_f, community_recs[i]);
    linewrite(connection_infs_f, connection_infs[i]);
    linewrite(p_Is_f, p_Is[i]);
  }
}

void write_community_traj(const auto &init_state, const auto& inf_events, const auto& connection_infs, const auto& community_recs, const auto& connection_targets, const auto& community_infs,
                          const std::string &file_path, uint32_t sim_idx) {
  std::vector<uint32_t> state;
  //flatten init_state
  for (auto &community : init_state) {
    state.insert(state.end(), community.begin(), community.end());
  }

  std::ofstream traj_f(file_path + "/community_traj_" +
                       std::to_string(sim_idx) + ".csv");
  linewrite(traj_f, state);
  for(int i = 0; i < inf_events.size(); i++)
  {
    state = iteration_lists_to_community_state(state, connection_infs[i], community_recs[i],
                                               connection_targets, community_infs[i]);
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
  linewrite(tot_traj_f, state);
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

std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections, float p_I_min,
                   float p_I_max, uint32_t Nt, uint32_t seed = 42) {
  std::vector<Static_RNG::default_rng> rngs(Nt);
  Static_RNG::default_rng rd(seed);
  std::vector<uint32_t> seeds(Nt);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::transform(seeds.begin(), seeds.end(), rngs.begin(),
                 [](auto seed) { return Static_RNG::default_rng(seed); });

  std::vector<std::vector<float>> p_Is(
      Nt, std::vector<float>(N_community_connections));

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

std::vector<std::vector<std::vector<float>>> generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, float p_I_min,
                   float p_I_max, uint32_t Nt, uint32_t seed = 42) {
  std::vector<uint32_t> seeds(N_sims);
  Static_RNG::default_rng rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<std::vector<std::vector<float>>> p_Is(N_sims);
  std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), p_Is.begin(),
                 [&](auto seed) {
                   return generate_p_Is(N_community_connections, p_I_min, p_I_max, Nt, seed);
                 });
  return p_Is;
}

std::vector<std::vector<std::vector<std::vector<float>>>> generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, uint32_t Ng, float p_I_min,
                   float p_I_max, uint32_t Nt, uint32_t seed = 42) {
  std::vector<uint32_t> seeds(Ng);
  Static_RNG::default_rng rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<std::vector<std::vector<std::vector<float>>>> p_Is(Ng);
  std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), p_Is.begin(),
                 [&](auto seed) {
                   return generate_p_Is(N_community_connections, N_sims, p_I_min, p_I_max, Nt, seed);
                 });
  return p_Is;
}


void simulate_to_file(const SBM_Graph_t& G, const SIR_SBM_Param_t& param,
                      sycl::queue &q, const std::string &file_path,
                      uint32_t sim_idx, uint32_t seed = 42) {

  auto [init_state, iteration_bufs] = SBM_simulate(G, param, q, seed);
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
  std::filesystem::create_directory(
      file_path);
  write_tot_traj(init_state, inf_events, connection_infs, community_recs, file_path, sim_idx);

  write_community_traj(init_state, inf_events, connection_infs, community_recs, G.connection_targets, community_infs, file_path, sim_idx);
  iterations_to_file(inf_events, community_infs, community_recs, connection_infs, param.p_I, file_path, sim_idx);
  std::ofstream ctm_f(file_path + "/connection_targets_" + std::to_string(sim_idx) + ".csv");
  linewrite(ctm_f, G.connection_targets);

  //sources
  std::ofstream csm_f(file_path + "/connection_sources_" + std::to_string(sim_idx) + ".csv");
  linewrite(csm_f, G.connection_sources);
}



void parallel_simulate_to_file(const SBM_Graph_t& G, const std::vector<SIR_SBM_Param_t>& params,
                                std::vector<sycl::queue> &qs, const std::string &file_path, uint32_t N_sim, uint32_t seed = 42) {

  uint32_t N_sims = params.size();
  std::vector<uint32_t> seeds(N_sims);
  Static_RNG::default_rng rng(seed);
  std::generate(seeds.begin(), seeds.end(), [&rng]() { return (uint32_t) rng(); });
  std::vector<SBM_Graph_t> Gs(N_sim, G);

  std::vector<std::tuple<const SBM_Graph_t*, const SIR_SBM_Param_t*, sycl::queue*, const std::string, uint32_t, uint32_t>> zip;
  for (uint32_t i = 0; i < N_sim; i++) {
    zip.push_back(std::make_tuple(&Gs[i], &params[i], &qs[i], file_path, i, seeds[i]));
  }

  std::for_each(std::execution::par_unseq, zip.begin(), zip.end(),
                [&](auto z) {
                  simulate_to_file(*std::get<0>(z), *std::get<1>(z), *std::get<2>(z), std::get<3>(z), std::get<4>(z), std::get<5>(z));
                });
}


void parallel_simulate_to_file(const std::vector<SBM_Graph_t>& Gs, const std::vector<std::vector<SIR_SBM_Param_t>>& params,
                                std::vector<std::vector<sycl::queue>> &qs, const std::vector<std::string> &file_paths, uint32_t seed = 42) {

  std::vector<uint32_t> seeds(Gs.size());
  std::mt19937 rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<uint32_t> N_sims(Gs.size());
  std::transform(params.begin(), params.end(), N_sims.begin(), [](auto p) { return p.size(); });

  //zip
  std::vector<std::tuple<const SBM_Graph_t*, const std::vector<SIR_SBM_Param_t>*, std::vector<sycl::queue>* ,std::string, uint32_t, uint32_t>> zip(Gs.size());
  for (uint32_t i = 0; i < Gs.size(); i++) {
    zip[i] = std::make_tuple(&Gs[i], &params[i], &qs[i], file_paths[i], N_sims[i], seeds[i]);
  }

  std::for_each(std::execution::par_unseq, zip.begin(), zip.end(),
                [&](const auto& z) {
                  parallel_simulate_to_file(*std::get<0>(z), *std::get<1>(z), *std::get<2>(z), std::get<3>(z), std::get<4>(z), std::get<5>(z));
                });
}

}
#endif // SYCL_GRAPH_SBM_WRITE_HPP