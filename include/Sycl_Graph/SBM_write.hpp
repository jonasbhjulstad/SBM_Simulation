#ifndef SYCL_GRAPH_SBM_WRITE_HPP
#define SYCL_GRAPH_SBM_WRITE_HPP
#include <vector>
#include <filesystem>
#include <Sycl_Graph/SBM_types.hpp>
namespace Sycl_Graph::SBM
{

  std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>
  read_iteration_buffer(Iteration_Buffers_t &buffers)
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
    const std::vector<uint32_t> &recoveries, const auto &connection_targets, const auto& community_infs) {
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
  std::for_each(iter.begin(), iter.end(),
                [&](auto &t_i_i) { file << t_i_i << ","; });
  file << "\n";
}

void iterations_to_file(const auto& inf_events, const auto& community_infs, const auto& community_recs, const auto& connection_infs, const std::string &file_path,
                        uint32_t sim_idx) {

  std::ofstream inf_events_f(file_path + "infection_events_" +
                             std::to_string(sim_idx) + ".csv");
  std::ofstream community_infs_f(file_path + "community_infs_" +
                                 std::to_string(sim_idx) + ".csv");
  std::ofstream community_recs_f(file_path + "community_recs_" +
                                 std::to_string(sim_idx) + ".csv");
  std::ofstream connection_infs_f(file_path + "connection_infs_" +
                                 std::to_string(sim_idx) + ".csv");
  for(int i = 0; i < inf_events.size(); i++)
  {
    linewrite(inf_events_f, inf_events[i]);
    linewrite(community_infs_f, community_infs[i]);
    linewrite(community_recs_f, community_recs[i]);
    linewrite(connection_infs_f, connection_infs[i]);
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
}
#endif // SYCL_GRAPH_SBM_WRITE_HPP