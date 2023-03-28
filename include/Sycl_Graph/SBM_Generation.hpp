#ifndef SBM_GENERATION_HPP
#define SBM_GENERATION_HPP
#include <CL/sycl.hpp>
#include <vector>
#include <utility>
#include <tuple>
#include <execution>
#include <algorithm>
#include <random>
#include <utility>
#include <Static_RNG/distributions.hpp>
#include <itertools.hpp>

namespace Sycl_Graph::SBM
{

  typedef std::pair<uint32_t, uint32_t> ID_Pair_t;
  typedef std::vector<ID_Pair_t> Partition_Edge_List_t;
  typedef std::vector<uint32_t> Nodelist_t;
  template <typename RNG = Static_RNG::default_rng>
  Partition_Edge_List_t
  random_connect(const Nodelist_t &to_nodes, const Nodelist_t &from_nodes,
                 float p, bool self_loop = true, uint32_t N_threads = 4,
                 uint32_t seed = 47)
  {
    uint32_t N_edges_max = self_loop ? to_nodes.size() * from_nodes.size()
                                     : to_nodes.size() * (from_nodes.size() - 1);
    Partition_Edge_List_t edge_list(N_edges_max);
    std::random_device rd;
    std::vector<RNG> rngs;
    uint32_t n = 0;
    for (auto &&prod : iter::product(to_nodes, from_nodes))
    {
      edge_list[n] = std::make_pair(std::get<0>(prod), std::get<1>(prod));
      n++;
    }
    RNG rng(seed);
    Static_RNG::bernoulli_distribution<float> dist(p);
    if (p == 1)
      return edge_list;
    if (p == 0)
      return Partition_Edge_List_t();
    edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(),
                                   [&](auto &e)
                                   { return !dist(rng); }),
                    edge_list.end());
    return edge_list;
  }

  long long n_choose_k(int n, int k)
  {
    long long product = 1;
    for (int i = 1; i <= k; i++)
      product = product * (n - k + i) / i; // Must do mul before div
    return product;
  }

  auto random_connect(const std::vector<Nodelist_t> &nodelists,
                      float p_in, float p_out, bool self_loop = true,
                      bool undirected = true, uint32_t N_threads = 4,
                      uint32_t seed = 47)
  {
    uint32_t N_node_pairs = n_choose_k(nodelists.size(), 2);
    N_node_pairs += self_loop ? nodelists.size() : 0;
    std::random_device rd;
    std::vector<uint32_t> seeds(N_threads);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<std::tuple<Nodelist_t, Nodelist_t, float, uint32_t>> node_pairs;
    node_pairs.reserve(N_node_pairs);
    uint32_t n = 0;

    for (auto &&comb : iter::combinations(nodelists, 2))
    {
      node_pairs.push_back(std::make_tuple(comb[0], comb[1], p_out, seeds[n]));
      n++;
    }
    if (self_loop)
    {
      for (auto &&nodelist : nodelists)
      {
        node_pairs.push_back(std::make_tuple(nodelist, nodelist, p_in, seeds[n]));
      }
    }
    std::vector<Partition_Edge_List_t> edge_lists(N_node_pairs);

    std::transform(std::execution::par_unseq, node_pairs.begin(),
                   node_pairs.end(), edge_lists.begin(), [&](const auto &t)
                   {
                   auto to_nodes = std::get<0>(t);
                   auto from_nodes = std::get<1>(t);
                   auto p = std::get<2>(t);
                   auto seed = std::get<3>(t);
                   return random_connect(to_nodes, from_nodes, p, self_loop,
                                         N_threads, seed); });
    std::vector<uint32_t> community_sizes(nodelists.size());
    std::transform(nodelists.begin(), nodelists.end(), community_sizes.begin(),
                   [](const auto &nodelist)
                   { return nodelist.size(); });

    std::vector<uint32_t> community_idx(nodelists.size());
    std::iota(community_idx.begin(), community_idx.end(), 0);
    std::vector<uint32_t> connection_targets;
    for (auto &&comb : iter::combinations(community_idx, 2))
    {
      connection_targets.push_back(comb[1]);
    }


    if (self_loop)
    {
      for (auto &&idx : community_idx)
      {
        connection_targets.push_back(idx);
      }
    }
    for (auto &&comb : iter::combinations(community_idx, 2))
    {
      connection_targets.push_back(comb[0]);
    }

    return std::make_tuple(nodelists, edge_lists, connection_targets);
  }

  // create pybind11 module
  auto create_SBM(const std::vector<uint32_t> N_pop,
                  float p_in, float p_out, bool undirected = true,
                  uint32_t N_threads = 4, uint32_t seed = 47)
  {

    std::vector<Nodelist_t> nodelists(N_pop.size());

    uint32_t N_nodes = 0;
    std::generate(nodelists.begin(), nodelists.end(), [&, n = 0]() mutable
                  {
    auto nodelist = Nodelist_t(N_pop[n]);
    std::iota(nodelist.begin(), nodelist.end(), N_nodes);
    N_nodes += N_pop[n];
    n++;
    return nodelist; });

    return random_connect(nodelists, p_in, p_out, true, undirected, N_threads, seed);
  }

  auto create_planted_SBM(uint32_t N_pop, uint32_t N,
                          float p_in, float p_out, bool undirected = true,
                          uint32_t N_threads = 4, uint32_t seed = 47)
  {
    std::vector<uint32_t> N_pop_vec(N, N_pop);
    return create_SBM(N_pop_vec, p_in, p_out, undirected,
                      N_threads, seed);
  }

  typedef std::vector<Partition_Edge_List_t> SBM_Edge_List_t;
  typedef std::vector<Nodelist_t> SBM_Node_List_t;

  auto create_planted_SBMs(uint32_t Ng, uint32_t N_pop,
                           uint32_t N, float p_in, float p_out, bool undirected = true, uint32_t N_threads = 4, uint32_t seed = 47)
  {
    std::mt19937 rd(seed);
    std::vector<uint32_t> seeds(Ng);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });

    std::vector<std::tuple<SBM_Node_List_t, SBM_Edge_List_t, std::vector<uint32_t>>> edge_lists(Ng);
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), edge_lists.begin(), [&](auto seed)
                   { return create_planted_SBM(N_pop, N, p_in, p_out, undirected, N_threads, seed); });

    // create vertex community maps
    std::vector<std::vector<uint32_t>> vertex_community_maps(Ng);

    return edge_lists;
  }
}
#endif