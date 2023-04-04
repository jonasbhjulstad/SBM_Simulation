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
#include <Sycl_Graph/Buffer_Routines.hpp>

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
      edge_list[n] = {std::get<0>(prod), std::get<1>(prod)};
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
    uint32_t N_vertices() const
    {
      uint32_t N = 0;
      for (auto &&nodelist : node_list)
      {
        N += nodelist.size();
      }
      return N;
    }

    uint32_t N_edges() const
    {
      uint32_t N = 0;
      for (auto &&edge_list : edge_lists)
      {
        N += edge_list.size();
      }
      return N;
    }

    uint32_t N_connections() const
    {
      return edge_lists.size();
    }

    uint32_t N_communities() const
    {
      return node_list.size();
    }

    auto create_edge_buffer(sycl::queue& q) const
    {
      std::vector<Edge_t> edges;
      edges.reserve(N_edges());
      for (auto &&edge_list : edge_lists)
      {
        edges.insert(edges.end(), edge_list.begin(), edge_list.end());
      }
      sycl::buffer<Edge_t> buf(edges.data(), edges.size());
      auto event = copy_to_buffer(buf, edges, q);

      return std::make_tuple(buf, event);
    }

    auto create_community_buffer(sycl::queue& q)
    {
      std::vector<uint32_t> community;
      community.reserve(N_vertices());
      for (uint32_t i = 0; i < N_communities(); i++)
      {
        for (auto &&node : node_list[i])
        {
          community.push_back(i);
        }
      }
      sycl::buffer<uint32_t> buf(community.data(), community.size());
      auto event = copy_to_buffer(buf, community, q);
      return std::make_tuple(buf, event);
    }
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

  std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections, float p_I_min,
                                                float p_I_max, uint32_t Nt, uint32_t seed = 42)
  {
    std::vector<Static_RNG::default_rng> rngs(Nt);
    Static_RNG::default_rng rd(seed);
    std::vector<uint32_t> seeds(Nt);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::transform(seeds.begin(), seeds.end(), rngs.begin(),
                   [](auto seed)
                   { return Static_RNG::default_rng(seed); });

    std::vector<std::vector<float>> p_Is(
        Nt, std::vector<float>(N_community_connections));

    std::transform(
        std::execution::par_unseq, rngs.begin(), rngs.end(), p_Is.begin(),
        [&](auto &rng)
        {
          Static_RNG::uniform_real_distribution<> dist(p_I_min, p_I_max);
          std::vector<float> p_I(N_community_connections);
          std::generate(p_I.begin(), p_I.end(), [&]()
                        { return dist(rng); });
          return p_I;
        });

    return p_Is;
  }

  std::vector<std::vector<std::vector<float>>> generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, float p_I_min,
                                                             float p_I_max, uint32_t Nt, uint32_t seed = 42)
  {
    std::vector<uint32_t> seeds(N_sims);
    Static_RNG::default_rng rd(seed);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<std::vector<std::vector<float>>> p_Is(N_sims);
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), p_Is.begin(),
                   [&](auto seed)
                   {
                     return generate_p_Is(N_community_connections, p_I_min, p_I_max, Nt, seed);
                   });
    return p_Is;
  }

  std::vector<std::vector<std::vector<std::vector<float>>>> generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, uint32_t Ng, float p_I_min,
                                                                          float p_I_max, uint32_t Nt, uint32_t seed = 42)
  {
    std::vector<uint32_t> seeds(Ng);
    Static_RNG::default_rng rd(seed);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<std::vector<std::vector<std::vector<float>>>> p_Is(Ng);
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), p_Is.begin(),
                   [&](auto seed)
                   {
                     return generate_p_Is(N_community_connections, N_sims, p_I_min, p_I_max, Nt, seed);
                   });
    return p_Is;
  }
}
#endif