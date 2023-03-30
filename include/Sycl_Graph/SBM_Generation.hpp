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
#include <Sycl_Graph/SBM_types.hpp>

namespace Sycl_Graph::SBM
{

  template <typename RNG = Static_RNG::default_rng>
  Edge_List_t
  random_connect(const Node_List_t &to_nodes, const Node_List_t &from_nodes,
                 float p, bool self_loop = true, uint32_t N_threads = 4,
                 uint32_t seed = 47)
  {
    uint32_t N_edges_max = self_loop ? to_nodes.size() * from_nodes.size()
                                     : to_nodes.size() * (from_nodes.size() - 1);
    Edge_List_t edge_list(N_edges_max);
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
      return Edge_List_t();
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

  struct SBM_Graph_t
  {
    std::vector<Node_List_t> node_list;
    std::vector<Edge_List_t> edge_lists;
    std::vector<uint32_t> connection_targets;
    std::vector<uint32_t> connection_sources;
    std::vector<std::pair<uint32_t, uint32_t>> community_connections;
  };

  SBM_Graph_t random_connect(const std::vector<Node_List_t> &nodelists,
                             float p_in, float p_out, bool self_loop = true, uint32_t N_threads = 4,
                             uint32_t seed = 47)
  {
    uint32_t N_node_pairs = n_choose_k(nodelists.size(), 2);
    N_node_pairs += self_loop ? nodelists.size() : 0;
    std::random_device rd;
    std::vector<uint32_t> seeds(N_threads);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<std::tuple<Node_List_t, Node_List_t, float, uint32_t>> node_pairs;
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
    std::vector<Edge_List_t> edge_lists(N_node_pairs);

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
    std::vector<uint32_t> connection_sources;
    std::vector<std::pair<uint32_t, uint32_t>> community_connections;
    for (auto &&comb : iter::combinations(community_idx, 2))
    {
      connection_targets.push_back(comb[1]);
      connection_sources.push_back(comb[0]);
    }

    if (self_loop)
    {
      for (auto &&idx : community_idx)
      {
        connection_targets.push_back(idx);
        connection_sources.push_back(idx);
      }
    }
    for (auto &&comb : iter::combinations(community_idx, 2))
    {
      connection_targets.push_back(comb[0]);
      connection_sources.push_back(comb[1]);
    }

    return {nodelists, edge_lists, connection_targets, connection_sources};
  }

  SBM_Graph_t rearrange_SBM_with_cmap(const std::vector<uint32_t> &cmap, const SBM_Graph_t &G)
  {
    SBM_Graph_t G_new;
    uint32_t N_new_communities = *std::max_element(cmap.begin(), cmap.end()) + 1;
    std::vector<uint32_t> community_idx(N_new_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    G_new.node_list.resize(N_new_communities);
    for (int i = 0; i < G.node_list.size(); i++)
    {
      auto mapped_idx = cmap[i];
      G_new.node_list[mapped_idx].insert(G_new.node_list[mapped_idx].end(),
                                         G.node_list[i].begin(), G.node_list[i].end());
    }

    // G.connection_targets.reserve(community_idx.size());
    // G.connection_sources.reserve(community_idx.size());
    uint32_t N_connections = n_choose_k(community_idx.size(), 2) + community_idx.size();
    G_new.edge_lists.resize(N_connections);

    std::vector<uint32_t> community_idx_old(G.node_list.size());
    std::iota(community_idx_old.begin(), community_idx_old.end(), 0);
    std::vector<uint32_t> connection_idx_old(G.edge_lists.size());
    std::iota(connection_idx_old.begin(), connection_idx_old.end(), 0);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> remapped_edges(N_connections);

    std::vector<std::pair<uint32_t, uint32_t>> community_connections;

    for (auto &&comb : iter::combinations(community_idx, 2))
    {
      community_connections.push_back(std::make_pair(comb[0], comb[1]));
    }

    uint32_t n = n_choose_k(community_idx.size(), 2);
    for (int i = 0; i < G.edge_lists.size(); i++)
    {
      auto mapped_target_idx = cmap[G.connection_targets[i]];
      auto mapped_source_idx = cmap[G.connection_sources[i]];
      // find index of connection in community_connections
      auto it = std::find_if(community_connections.begin(), community_connections.end(), [&](const auto &a)
                             {
                             return (a.first == mapped_target_idx && a.second == mapped_source_idx) ||
                                    (a.first == mapped_source_idx && a.second == mapped_target_idx);
                             });
      if (it != community_connections.end())
      {
        std::cout << "Assigning " << i << " to " << std::distance(community_connections.begin(), it) << std::endl;
        auto idx = std::distance(community_connections.begin(), it);
        remapped_edges[idx].insert(remapped_edges[idx].end(), G.edge_lists[i].begin(), G.edge_lists[i].end());
      }
      else{
        remapped_edges[n + cmap[mapped_target_idx]].insert(remapped_edges[n + cmap[mapped_target_idx]].end(), G.edge_lists[i].begin(), G.edge_lists[i].end());
      }
    }


    uint32_t N_communities_old = G.node_list.size();

    for (int i = 0; i < N_communities_old; i++)
    {
      auto mapped_target_idx = cmap[i];
      auto mapped_source_idx = cmap[i];
    }
    for (int i = 0; i < G_new.node_list.size(); i++)
    {
      G_new.connection_sources.push_back(i);
      G_new.connection_targets.push_back(i);
    }

    G_new.edge_lists = remapped_edges;

    for (auto &&comb : iter::combinations(community_idx, 2))
    {
      G_new.connection_targets.push_back(comb[0]);
      G_new.connection_sources.push_back(comb[1]);
    }

    return G_new;
  }

  // create pybind11 module
  SBM_Graph_t create_SBM(const std::vector<uint32_t> N_pop,
                         float p_in, float p_out,
                         uint32_t N_threads = 4, uint32_t seed = 47)
  {

    std::vector<Node_List_t> nodelists(N_pop.size());

    uint32_t N_nodes = 0;
    std::generate(nodelists.begin(), nodelists.end(), [&, n = 0]() mutable
                  {
    auto nodelist = Node_List_t(N_pop[n]);
    std::iota(nodelist.begin(), nodelist.end(), N_nodes);
    N_nodes += N_pop[n];
    n++;
    return nodelist; });

    return random_connect(nodelists, p_in, p_out, true, N_threads, seed);
  }

  SBM_Graph_t create_planted_SBM(uint32_t N_pop, uint32_t N,
                                 float p_in, float p_out,
                                 uint32_t N_threads = 4, uint32_t seed = 47)
  {
    std::vector<uint32_t> N_pop_vec(N, N_pop);
    return create_SBM(N_pop_vec, p_in, p_out,
                      N_threads, seed);
  }

  std::vector<SBM_Graph_t> create_planted_SBMs(uint32_t Ng, uint32_t N_pop,
                                               uint32_t N, float p_in, float p_out, uint32_t N_threads = 4, uint32_t seed = 47)
  {
    std::mt19937 rd(seed);
    std::vector<uint32_t> seeds(Ng);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });

    std::vector<SBM_Graph_t> result(Ng);
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), result.begin(), [&](auto seed)
                   { return create_planted_SBM(N_pop, N, p_in, p_out, N_threads, seed); });

    return result;
  }
}
#endif