#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Buffer_Routines.hpp>
#include <Sycl_Graph/SBM_Generation.hpp>
#include <algorithm>
#include <execution>
#include <itertools.hpp>
#include <random>

namespace Sycl_Graph::SBM {

static constexpr uint32_t RANDOM_CONNECT_MAX_NODE_SIZE = 10000000;
static constexpr uint32_t RANDOM_CONNECT_MAX_EDGE_SIZE = 10000000;
Edge_List_t random_connect(const Node_List_t &to_nodes,
                           const Node_List_t &from_nodes, float p,
                           bool self_loop, uint32_t N_threads, uint32_t seed) {
  uint32_t N_edges_max = 2 * to_nodes.size() * from_nodes.size();
  //    : to_nodes.size() * (from_nodes.size() - 1);
#ifdef DEBUG
  assert(N_edges_max < RANDOM_CONNECT_MAX_EDGE_SIZE);
  assert(to_nodes.size() < RANDOM_CONNECT_MAX_NODE_SIZE);
  assert(from_nodes.size() < RANDOM_CONNECT_MAX_NODE_SIZE);
#endif
  Edge_List_t edge_list;
  edge_list.reserve(N_edges_max);
  std::random_device rd;
  std::vector<Static_RNG::default_rng> rngs;
  uint32_t n = 0;
  for (auto &&prod : iter::product(to_nodes, from_nodes)) {
    edge_list.push_back({std::get<0>(prod), std::get<1>(prod)});
    n++;
  }
  Static_RNG::default_rng rng(seed);
  Static_RNG::bernoulli_distribution<float> dist(p);
  if (!self_loop)
    edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(),
                                   [&](auto &e) { return e.from == e.to; }),
                    edge_list.end());
  if (p == 1)
    return edge_list;
  if (p == 0)
    return Edge_List_t();
  edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(),
                                 [&](auto &e) { return !dist(rng); }),
                  edge_list.end());
  return edge_list;
}

long long n_choose_k(int n, int k) {
  long long product = 1;
  for (int i = 1; i <= k; i++)
    product = product * (n - k + i) / i; // Must do mul before div
  return product;
}

SBM_Graph_t random_connect(const std::vector<Node_List_t> &nodelists,
                           float p_in, float p_out, bool self_loop,
                           uint32_t N_threads, uint32_t seed) {
  uint32_t N_node_pairs = n_choose_k(nodelists.size(), 2) + nodelists.size();
#ifdef DEBUG
  std::for_each(nodelists.begin(), nodelists.end(), [&](const auto &n_vec) {
    assert(n_vec.size() < RANDOM_CONNECT_MAX_NODE_SIZE);
    assert(std::all_of(n_vec.begin(), n_vec.end(), [&](const auto &n) {
      return n < RANDOM_CONNECT_MAX_NODE_SIZE;
    }));
  });
#endif

  N_node_pairs += self_loop ? nodelists.size() : 0;
  std::random_device rd;
  std::vector<uint32_t> seeds(N_node_pairs);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<std::tuple<Node_List_t, Node_List_t, float, uint32_t>> node_pairs;
  node_pairs.reserve(N_node_pairs);
  uint32_t n = 0;

  for (auto &&comb : iter::combinations(nodelists, 2)) {
    node_pairs.push_back(std::make_tuple(comb[0], comb[1], p_out, seeds[n]));
#ifdef DEBUG
    assert(n < N_node_pairs.size());
#endif
    n++;
  }
  for (auto &&nodelist : nodelists) {
    node_pairs.push_back(std::make_tuple(nodelist, nodelist, p_in, seeds[n]));
    n++;
  }
#ifdef DEBUG
  assert(node_pairs.size() == N_node_pairs);
#endif
  std::vector<Edge_List_t> edge_lists(N_node_pairs);

  std::transform(std::execution::par_unseq, node_pairs.begin(),
                 node_pairs.end(), edge_lists.begin(), [&](const auto &t) {
                   auto to_nodes = std::get<0>(t);
                   auto from_nodes = std::get<1>(t);
                   auto p = std::get<2>(t);
                   auto seed = std::get<3>(t);
                   return random_connect(to_nodes, from_nodes, p, self_loop,
                                         N_threads, seed);
                 });
  std::vector<uint32_t> community_sizes(nodelists.size());
  std::transform(nodelists.begin(), nodelists.end(), community_sizes.begin(),
                 [](const auto &nodelist) { return nodelist.size(); });

  uint32_t N_nodes =
      std::accumulate(community_sizes.begin(), community_sizes.end(), 0);
  // flatten node and edge lists
  std::vector<uint32_t> nodelist_flat;
  nodelist_flat.reserve(N_nodes);
  for (int i = 0; i < nodelists.size(); i++) {
    nodelist_flat.insert(nodelist_flat.end(), nodelists[i].begin(),
                         nodelists[i].end());
  }

  uint32_t N_edges = std::accumulate(edge_lists.begin(), edge_lists.end(), 0,
                                     [](uint32_t acc, const auto &edge_list) {
                                       return acc + edge_list.size();
                                     });
  return SBM_Graph_t(nodelists, edge_lists);
}

// create pybind11 module
SBM_Graph_t create_SBM(const std::vector<uint32_t> N_pop, float p_in,
                       float p_out, uint32_t N_threads, uint32_t seed) {

  std::vector<Node_List_t> nodelists(N_pop.size());

  uint32_t N_nodes = 0;
  std::generate(nodelists.begin(), nodelists.end(), [&, n = 0]() mutable {
    auto nodelist = Node_List_t(N_pop[n]);
    std::iota(nodelist.begin(), nodelist.end(), N_nodes);
    N_nodes += N_pop[n];
    n++;
    return nodelist;
  });

  return random_connect(nodelists, p_in, p_out, false, N_threads, seed);
}

SBM_Graph_t create_planted_SBM(uint32_t N_pop, uint32_t N, float p_in,
                               float p_out, uint32_t N_threads, uint32_t seed) {
  std::vector<uint32_t> N_pop_vec(N, N_pop);
  return create_SBM(N_pop_vec, p_in, p_out, N_threads, seed);
}

std::vector<SBM_Graph_t> create_planted_SBMs(uint32_t Ng, uint32_t N_pop,
                                             uint32_t N, float p_in,
                                             float p_out, uint32_t N_threads,
                                             uint32_t seed) {
  std::mt19937 rd(seed);
  std::vector<uint32_t> seeds(Ng);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });

  std::vector<SBM_Graph_t> result(Ng);
  std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(),
                 result.begin(), [&](auto seed) {
                   return create_planted_SBM(N_pop, N, p_in, p_out, N_threads,
                                             seed);
                 });

  return result;
}

std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections,
                                              float p_I_min, float p_I_max,
                                              uint32_t Nt, uint32_t seed) {
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

std::vector<std::vector<std::vector<float>>>
generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, float p_I_min,
              float p_I_max, uint32_t Nt, uint32_t seed) {
  std::vector<uint32_t> seeds(N_sims);
  Static_RNG::default_rng rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<std::vector<std::vector<float>>> p_Is(N_sims);
  std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(),
                 p_Is.begin(), [&](auto seed) {
                   return generate_p_Is(N_community_connections, p_I_min,
                                        p_I_max, Nt, seed);
                 });
  return p_Is;
}

std::vector<std::vector<std::vector<std::vector<float>>>>
generate_p_Is(uint32_t N_community_connections, uint32_t N_sims, uint32_t Ng,
              float p_I_min, float p_I_max, uint32_t Nt, uint32_t seed) {
  std::vector<uint32_t> seeds(Ng);
  Static_RNG::default_rng rd(seed);
  std::generate(seeds.begin(), seeds.end(), [&rd]() { return rd(); });
  std::vector<std::vector<std::vector<std::vector<float>>>> p_Is(Ng);
  std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(),
                 p_Is.begin(), [&](auto seed) {
                   return generate_p_Is(N_community_connections, N_sims,
                                        p_I_min, p_I_max, Nt, seed);
                 });
  return p_Is;
}

} // namespace Sycl_Graph::SBM