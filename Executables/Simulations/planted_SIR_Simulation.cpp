#include "Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_SBM_Types.hpp"
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Generation/Graph_Generation.hpp>
#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli_SBM.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <itertools.hpp>
#include <algorithm>
#include <execution>
#include <cstdint>
#include <filesystem>

using Sycl_Graph::Dynamic::Network_Models::generate_SBM;
using namespace Sycl_Graph::Sycl::Network_Models;
typedef SIR_Bernoulli_SBM_Network Network_t;
typedef typename Network_t::Graph_t Graph_t;


typedef std::vector<SIR_Bernoulli_SBM_Temporal_Param<float>> tp_list_t;

void simulate_to_file(Network_t &network, const tp_list_t &tp_list, const std::string &file_path, uint32_t sim_idx)
{
  auto traj = network.simulate_groups(tp_list);

  std::ofstream tot_traj_f(file_path + "traj_tot_" + std::to_string(sim_idx) + ".csv");
  std::ofstream p_I_f(file_path + "p_Is_" + std::to_string(sim_idx) + ".csv");
      std::ofstream community_traj_f(file_path + "traj_community_" + std::to_string(sim_idx) + ".csv");

  std::for_each(traj.first.begin(), traj.first.end(), [&](auto &t_i)
                { 
                  std::for_each(t_i.begin(), t_i.end(), [&](auto &t_i_i)
                                { tot_traj_f << t_i_i << ","; });
                  tot_traj_f <<"\n"; });

  std::for_each(traj.second.begin(), traj.second.end(), [&](auto &t_i)
                { 
                  std::for_each(t_i.begin(), t_i.end(), [&](auto &t_i_i)
                                { community_traj_f << t_i_i << ","; });
                });

  std::for_each(tp_list.begin(), tp_list.end(), [&](auto &tp)
                {
    std::for_each(tp.p_Is.begin(), tp.p_Is.end(), [&](auto& tp_i)
    {
      p_I_f << tp_i << ",";
    });
    p_I_f << "\n"; });
}

void simulate_to_file(std::vector<Network_t> &networks, const std::vector<tp_list_t> &tp_lists, const std::string &file_path)
{
  std::vector<uint32_t> file_idx(networks.size());
  std::generate(file_idx.begin(), file_idx.end(), [n = 0]() mutable { return n++;});

  std::vector<std::tuple<Network_t*, const tp_list_t*, uint32_t>> tup(networks.size());
  std::generate(tup.begin(), tup.end(), [n = -1, &networks, &tp_lists, &file_idx]() mutable { 
    n++;
    return std::make_tuple(&networks[n], &tp_lists[n], n);});


  std::for_each(std::execution::par_unseq, tup.begin(), tup.end(), [&](auto& t)
                 { 
                  auto network = std::get<0>(t);
                  auto tp_list = std::get<1>(t);
                  auto sim_idx = std::get<2>(t);
                  simulate_to_file(*network, *tp_list, file_path, sim_idx);});
}

typedef std::pair<uint32_t, uint32_t> ID_Pair_t;
typedef std::vector<ID_Pair_t> Partition_Edge_List_t;
typedef std::vector<uint32_t> Nodelist_t;
template <typename RNG = Static_RNG::default_rng>
Partition_Edge_List_t random_connect(const Nodelist_t& to_nodes, const Nodelist_t& from_nodes, float p, bool self_loop = true, uint32_t N_threads = 4, uint32_t seed = 47)
{
  uint32_t N_edges_max = self_loop ? to_nodes.size() * from_nodes.size() : to_nodes.size() * (from_nodes.size() - 1);
  Partition_Edge_List_t edge_list(N_edges_max);
  std::random_device rd;
  std::vector<RNG> rngs;
  uint32_t n = 0;
  for(auto&& prod: iter::product(to_nodes, from_nodes))
  {
    edge_list[n] = std::make_pair(std::get<0>(prod), std::get<1>(prod));
    n++;
  }
  RNG rng(seed);
  edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(), [&](auto& e){return rng() > p;}), edge_list.end());
  return edge_list;
}


std::vector<Partition_Edge_List_t> random_connect(const std::vector<Nodelist_t>& nodelists, const std::vector<float>& p_list, bool self_loop = true, bool undirected = true, uint32_t N_threads = 4, uint32_t seed = 47)
{
  uint32_t N_node_pairs = nodelists.size() * (nodelists.size() - 1);
  std::random_device rd;
  std::vector<uint32_t> seeds(N_threads);
  std::generate(seeds.begin(), seeds.end(), [&rd](){return rd();});
  std::vector<std::tuple<Nodelist_t, Nodelist_t, float, uint32_t>> node_pairs(N_node_pairs);
  uint32_t n = 0;
  for(auto&& comb: iter::combinations(nodelists, 2))
  {

    node_pairs[n] = std::make_tuple(comb[0], comb[1], p_list[n], seeds[n]);
    n++;
  }
  std::vector<Partition_Edge_List_t> edge_lists(N_node_pairs);

  std::transform(std::execution::par_unseq, node_pairs.begin(), node_pairs.end(), edge_lists.begin(), [&](const auto& t)
                 {
                  auto to_nodes = std::get<0>(t);
                  auto from_nodes = std::get<1>(t);
                  auto p = std::get<2>(t);
                  auto seed = std::get<3>(t);
                  return random_connect(to_nodes, from_nodes, p, self_loop, N_threads, seed);});

  return edge_lists;
}


// create pybind11 module
auto create_SIR_Bernoulli_SBM(sycl::queue &q, const std::vector<uint32_t> N_pop, const std::vector<float> p_SBM, const float p_I0, const float p_R0 = 0.0, bool undirected = true, uint32_t N_threads = 4, uint32_t seed = 47)
{

  std::vector<Nodelist_t> nodelists(N_pop.size());

  uint32_t N_nodes = 0;
  std::generate(nodelists.begin(), nodelists.end(), [&, n = 0] () mutable 
                {
                  auto nodelist = Nodelist_t(N_pop[n]);
                  std::iota(nodelist.begin(), nodelist.end(), N_nodes);
                  N_nodes += N_pop[n];
                  n++;
                  return nodelist;
                });

  std::vector<uint32_t> vertex_ids(N_nodes);
  std::iota(vertex_ids.begin(), vertex_ids.end(), 0);

  std::vector<Partition_Edge_List_t> edge_id_lists = random_connect(nodelists, p_SBM, true, undirected, N_threads, seed);
  uint32_t N_edges = std::accumulate(edge_id_lists.begin(), edge_id_lists.end(), 0, [](const auto& sum, auto& edge_id_list){return sum + edge_id_list.size();});

  std::vector<std::pair<uint32_t, uint32_t>> edge_ids(N_edges);
  //flatten edge_id_lists
  std::for_each(edge_id_lists.begin(), edge_id_lists.end(), [&](auto& edge_id_list)
                {
                  edge_ids.insert(edge_ids.end(), edge_id_list.begin(), edge_id_list.end());
                });


  Graph_t G(q, 0, 0);
  G.add_vertex(vertex_ids);
  G.add_edge(edge_ids);

  // create network
  Network_t sir(G, p_I0, p_R0, edge_id_lists, seed);

  return sir;
}

auto create_SIR_Bernoulli_planted(sycl::queue &q, uint32_t N_pop, uint32_t N, float p_in, float p_out, float p_I0, float p_R0 = 0.0, bool undirected = true, uint32_t N_threads = 4, uint32_t seed = 47)
{
  std::vector<float> p_SBM(N * N);
  for (uint32_t i = 0; i < N; i++)
  {
    for (uint32_t j = 0; j < N; j++)
    {
      if (i == j)
      {
        p_SBM[i * N + j] = p_in;
      }
      else
      {
        p_SBM[i * N + j] = p_out;
      }
    }
  }

  std::vector<uint32_t> N_pop_vec(N, N_pop);
  return create_SIR_Bernoulli_SBM(q, N_pop_vec, p_SBM, p_I0, p_R0, undirected, N_threads, seed);
}

auto create_SIR_Bernoulli_planted(uint32_t Ng, sycl::queue &q, uint32_t N_pop, uint32_t N, float p_in, float p_out, float p_I0, float p_R0 = 0.0, bool undirected = true, uint32_t seed = 47)
{
  std::vector<std::pair<std::vector<Network_t>, uint32_t>> net_seed(Ng);
  std::mt19937 rd(seed);
  std::generate(net_seed.begin(),net_seed.end(), [&rd](){return std::make_pair(std::vector<Network_t>(), rd());});
  std::for_each(std::execution::par_unseq, net_seed.begin(), net_seed.end(), [&q, N_pop, N, p_in, p_out, p_I0, p_R0, undirected](auto& ns)
                {
                  auto sir = create_SIR_Bernoulli_planted(q, N_pop, N, p_in, p_out, p_I0, p_R0, undirected, 4, ns.second);
                  sir.initialize();
                  ns.first.push_back(sir);
                });
  //flatten
  std::vector<Network_t> networks_flat;
  for (auto&& network_vec: net_seed)
  {
    networks_flat.insert(networks_flat.end(), network_vec.first.begin(), network_vec.first.end());
  }
  return networks_flat;
}
int main()
{

  float p_I0 = 0.1;
  float p_R0 = 0.0;
  uint32_t N_clusters = 10;
  uint32_t N_pop = 100;
  float p_in = 1.0f;
  float p_out = 0.1f;
  sycl::queue q(sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{});
  uint32_t Nt = 100;
  uint32_t Ng = 200;
  uint32_t seed = 47;

  auto networks = create_SIR_Bernoulli_planted(Ng, q, N_pop, N_clusters, p_in, p_out, p_I0, p_R0, seed);

  uint32_t N_p_I = networks[0].SBM_ids.size();
  std::vector<float> p_Is(N_p_I, 1e-2);
  SIR_Bernoulli_SBM_Temporal_Param<> sir_param;
  sir_param.p_Is = p_Is;

  tp_list_t sir_param_vec;
  for (int i = 0; i < Nt; i++)
  {
    sir_param_vec.push_back(sir_param);
  }
  std::vector<tp_list_t> tp_list(Ng, sir_param_vec);
  std::generate(tp_list.begin(), tp_list.end(), [&]()
                { return sir_param_vec; });

  // auto par_res = simulate_N_parallel_copied(networks, tp_list);

  // write to file
  std::filesystem::create_directory(
      std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");

  simulate_to_file(networks, tp_list, std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/");

  return 0;
}