#include <Sycl_Graph/SBM_write.hpp>
#include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <Static_RNG/distributions.hpp>
#include <execution>
namespace Sycl_Graph::SBM
{
  template <>
  void linewrite<>(std::ofstream &file, const std::vector<uint32_t> &iter);

  void linewrite(std::ofstream &file, const std::array<uint32_t, 3> &state_iter)
  {
    for (auto &t_i_i : iter)
    {
      file << t_i_i.from << "," << t_i_i.to;
      if (&t_i_i != &iter.back())
        file << ",";
      else
        file << "\n";
    }
  }

  void linewrite(std::ofstream &file, const std::array<uint32_t, 3> &state_iter)
  {
    for (auto &t_i : state_iter)
    {
      for (auto &&t_i_i : iter)
      {
        file << t_i_i;
        if (&t_i_i != &iter.back())
          file << ",";
        else
          file << "\n";
      }
    }
  }

  void linewrite(std::ofstream &file, const std::vector<Edge_t> &iter)
  {
    for (auto &t_i_i : iter)
    {
      file << t_i_i.from << "," << t_i_i.to;
      if (&t_i_i != &iter.back())
        file << ",";
      else
        file << "\n";
    }
  }

  template <>
  std::vector<std::vector<uint32_t>> read_buffers(std::vector<sycl::buffer<uint32_t>> &bufs,
                                                  std::vector<sycl::event> &events,
                                                  uint32_t N, uint32_t N_elem);

  void simulate_to_file(const SBM_Graph_t &G, const SIR_SBM_Param_t &param,
                        sycl::queue &q, const std::string &file_path,
                        uint32_t sim_idx, uint32_t seed)
  {
    uint32_t Nt = param.p_I.size();
    SIR_SBM_Network network(G, param.p_I0, param.p_R, q, seed, param.p_R0);
    auto [community_state_bufs, connection_events_bufs, acs_events] =
        network.simulate(param);

    auto community_trajectory = read_buffers(community_state_bufs, acs_events,
                                             network.N_communities, Nt + 1);

    auto connection_events_trajectory = read_buffers(
        connection_events_bufs, acs_events, network.N_connections, Nt);

    std::ofstream community_traj_f(file_path + "community_trajectory_" +
                                   std::to_string(sim_idx) + ".csv");
    std::ofstream connection_events_f(file_path + "connection_events_" +
                                      std::to_string(sim_idx) + ".csv");
    std::for_each(community_trajectory.begin(), community_trajectory.end(),
                  [&](auto &community_trajectory_i)
                  {
                    linewrite(community_traj_f, community_trajectory_i);
                  });
    std::for_each(connection_events_trajectory.begin(),
                  connection_events_trajectory.end(),
                  [&](auto &connection_events_i)
                  {
                    linewrite(connection_events_f, connection_events_i);
                  });
  }

  void parallel_simulate_to_file(const SBM_Graph_t &G,
                                 const std::vector<SIR_SBM_Param_t> &params,
                                 std::vector<sycl::queue> &qs,
                                 const std::string &file_path, uint32_t N_sim,
                                 uint32_t seed)
  {
    assert(params.size() == qs.size() &&
           "Number of parameters and queues must be equal");
    uint32_t N_sims = params.size();
    std::vector<uint32_t> seeds(N_sims);
    Static_RNG::default_rng rng(seed);
    std::generate(seeds.begin(), seeds.end(),
                  [&rng]()
                  { return (uint32_t)rng(); });
    std::vector<SBM_Graph_t> Gs(N_sim, G);

    std::vector<std::tuple<const SBM_Graph_t *, const SIR_SBM_Param_t *,
                           sycl::queue *, const std::string, uint32_t, uint32_t>>
        zip;
    for (uint32_t i = 0; i < N_sim; i++)
    {
      zip.push_back(
          std::make_tuple(&Gs[i], &params[i], &qs[i], file_path, i, seeds[i]));
    }

    std::for_each(std::execution::par_unseq, zip.begin(), zip.end(), [&](auto z)
                  { simulate_to_file(*std::get<0>(z), *std::get<1>(z), *std::get<2>(z),
                                     std::get<3>(z), std::get<4>(z), std::get<5>(z)); });
  }

  void parallel_simulate_to_file(
      const std::vector<SBM_Graph_t> &Gs,
      const std::vector<std::vector<SIR_SBM_Param_t>> &params,
      std::vector<std::vector<sycl::queue>> &qs,
      const std::vector<std::string> &file_paths, uint32_t seed)
  {
    assert(Gs.size() == params.size() &&
           "Number of graphs and parameters must be equal");
    assert(Gs.size() == qs.size() && "Number of graphs and queues must be equal");

    std::vector<uint32_t> seeds(Gs.size());
    std::mt19937 rd(seed);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<uint32_t> N_sims(Gs.size());
    std::transform(params.begin(), params.end(), N_sims.begin(),
                   [](auto p)
                   { return p.size(); });

    // zip
    std::vector<
        std::tuple<const SBM_Graph_t *, const std::vector<SIR_SBM_Param_t> *,
                   std::vector<sycl::queue> *, std::string, uint32_t, uint32_t>>
        zip(Gs.size());
    for (uint32_t i = 0; i < Gs.size(); i++)
    {
      zip[i] = std::make_tuple(&Gs[i], &params[i], &qs[i], file_paths[i],
                               N_sims[i], seeds[i]);
    }

    std::for_each(std::execution::par_unseq, zip.begin(), zip.end(),
                  [&](const auto &z)
                  {
                    parallel_simulate_to_file(*std::get<0>(z), *std::get<1>(z),
                                              *std::get<2>(z), std::get<3>(z),
                                              std::get<4>(z), std::get<5>(z));
                  });
  }

}