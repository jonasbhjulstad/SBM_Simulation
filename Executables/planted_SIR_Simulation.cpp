#define USE_TBB_DEBUG
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/SBM_Generation.hpp>
#include <Sycl_Graph/SIR_SBM.hpp>
#include <cstdint>
#include <string>
#include <filesystem>
#include <algorithm>

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
//   std::ofstream tot_traj_f(file_path + "traj_tot_" + std::to_string(sim_idx) +
//                            ".csv");
//   std::ofstream p_I_f(file_path + "p_Is_" + std::to_string(sim_idx) + ".csv");
//   std::ofstream delta_I_f(file_path + "delta_Is_" + std::to_string(sim_idx) +
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

void iterations_to_file(auto& iteration_data, const std::string& file_path, uint32_t sim_idx)
{
  std::ofstream inf_events_f(file_path + "inf_events_" + std::to_string(sim_idx) + ".csv");
  std::ofstream community_infs_f(file_path + "community_infs_" + std::to_string(sim_idx) +
                          ".csv");
  std::ofstream community_recs_f(file_path + "community_recs_" + std::to_string(sim_idx) +
                          ".csv");

  auto linewrite([&](std::ofstream &file, const auto &iter)
                 {
      std::for_each(iter.begin(), iter.end(),
                    [&](auto &t_i_i) { file << t_i_i << ","; });
      file << "\n"; });

  std::for_each(iteration_data.begin(), iteration_data.end(), [&](auto &t)
                {
                  auto [inf_events, community_infs, community_rec] = read_iteration_buffer(t);
                  linewrite(inf_events_f, inf_events);
                  linewrite(community_infs_f, community_infs);
                  linewrite(community_recs_f, community_rec);
                });
}


auto generate_p_Is(const auto& SBM_sizes, uint32_t Nt, uint32_t seed = 42)
{
  std::vector<Static_RNG::default_rng> rngs(SBM_sizes.size());
  std::mt19937 rd(seed);
  std::vector<uint32_t> seeds(SBM_sizes.size());
  std::generate(seeds.begin(), seeds.end(), [&rd]()
                { return rd(); });
  std::transform(seeds.begin(), seeds.end(), rngs.begin(), [](auto seed)
                 { return Static_RNG::default_rng(seed); });

  std::vector<std::vector<std::vector<float>>> p_Is(SBM_sizes.size());
  std::transform(std::execution::par_unseq, SBM_sizes.begin(), SBM_sizes.end(),
                 rngs.begin(), p_Is.begin(), [Nt](auto &SBM_size, auto& rng)
                 {
                   std::vector<std::vector<float>> p_Is_sub(SBM_size);
                   std::generate(p_Is_sub.begin(),
                                  p_Is_sub.end(), [&]()
                                  {
                                    auto p_I = std::vector<float>(Nt);
                                    std::generate(p_I.begin(), p_I.end(),
                                                  [&rng]()
                                                  { return rng(); });
                                                  return p_I;
                                  });
                   return p_Is_sub;
                 });

  return p_Is;
}

auto generate_Ng_Nt_p_Is(uint32_t Ng, uint32_t Nt, const auto& edge_list_sizes, uint32_t seed)
{
  std::vector<uint32_t> p_I_seeds(Ng);
  std::mt19937 rd(seed);
  std::generate(p_I_seeds.begin(), p_I_seeds.end(), [&rd]()
                { return rd(); });

  // generate
  std::vector<std::vector<std::vector<std::vector<float>>>> p_Is(Ng);
  std::transform(std::execution::par_unseq, p_I_seeds.begin(), p_I_seeds.end(), p_Is.begin(), [&](auto p_I_seed)
                { return generate_p_Is(edge_list_sizes, Nt, p_I_seed); });
  return p_Is;
}

auto run_Ng_Nt_simulations(const auto& SBM_lists, const auto& p_Is, float p_R, float p_I0, float p_R0, float Nt, uint32_t Ng, std::vector<sycl::queue>& qs, uint32_t seed = 42)
{
  std::mt19937 rd(seed);
  std::vector<uint32_t> seeds(Ng);  
  std::generate(seeds.begin(), seeds.end(), [&rd]()
                { return rd(); });
  //zip sbm_lists, p_Is, seeds

  std::vector<std::tuple<std::pair<SBM_Node_List_t, SBM_Edge_List_t>, std::vector<std::vector<std::vector<float>>>, uint32_t, sycl::queue>> zip(Ng);
  for(uint32_t i = 0; i < Ng; i++)
  {
    zip[i] = std::make_tuple(SBM_lists[i], p_Is[i], seeds[i], qs[i]);
  }


  


  std::vector<std::vector<std::vector<Iteration_Buffers_t>>> result(Ng);
  std::transform(std::execution::par_unseq, zip.begin(), zip.end(), result.begin(), [p_I0, p_R0, p_R, Nt](const auto &z)
                 {
                    auto SBM_list = std::get<0>(z);
                    auto p_I = std::get<1>(z);
                    auto sim_seed = std::get<2>(z);
                    auto q = std::get<3>(z);
                    //create 

                    return SBM_simulate(p_I, p_R, p_I0, p_R0, SBM_list.first, SBM_list.second, q, sim_seed);
                 });
  return result;
}


void single_simulation()
{
    float p_I0 = 0.1;
  float p_R0 = 0.0;
  uint32_t N_clusters = 10;
  uint32_t N_pop = 100;
  float p_in = 1.0f;
  float p_out = 0.1f;
  uint32_t Ng = 1;
  std::vector<sycl::queue> qs;
  for(uint32_t i = 0; i < 10; i++)
  {
    qs.push_back(sycl::queue(sycl::gpu_selector_v));
  }
  uint32_t Nt = 70;
  uint32_t seed = 47;
  uint32_t N_threads = 10;

  auto SBM_lists = create_planted_SBMs(Ng, N_pop, N_clusters, p_in, p_out, true, N_threads,
                                       seed);

  auto edge_list_sizes = std::vector<uint32_t>(Ng);
  std::transform(SBM_lists.begin(), SBM_lists.end(), edge_list_sizes.begin(),
                 [](auto &SBM_list)
                 { return SBM_list.second.size(); });

  float p_I_min = 1e-5;
  float p_I_max = 1e-3f;
  float p_R = 0.1f;
  
  auto p_Is = generate_Ng_Nt_p_Is(Ng, Nt, edge_list_sizes, seed);

  auto result = run_Ng_Nt_simulations(SBM_lists, p_Is, p_R, p_I0, p_R0, Nt, Ng, qs);



  // auto par_res = simulate_N_parallel_copied(networks, tp_list);

  // write to file
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");

  iterations_to_file(result[0][0], std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) +
                                 "/SIR_sim/", 0);

}


int main()
{

  std::vector<uint32_t> dummy(1);
  std::for_each(std::execution::seq, dummy.begin(), dummy.end(), [](auto &d)
                { single_simulation();});

  return 0;
}