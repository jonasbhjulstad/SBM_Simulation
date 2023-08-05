#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <itertools.hpp>
#include <execution>
#include <algorithm>
#include <random>
#include <tuple>
std::vector<std::pair<uint32_t, uint32_t>> random_connect(const std::vector<uint32_t> &to_nodes,
                                                          const std::vector<uint32_t> &from_nodes, float p, uint32_t connection_idx, uint32_t seed)
{

    std::vector<std::pair<uint32_t, uint32_t>> edge_list(to_nodes.size() * from_nodes.size());

    uint32_t n = 0;
    auto prod = iter::product(to_nodes, from_nodes);
    std::transform(std::execution::par_unseq, prod.begin(), prod.end(), edge_list.begin(),
                   [&](auto &pair)
                   {
                       return std::make_pair(std::get<0>(pair), std::get<1>(pair));
                   });

    edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(),
                                   [&](auto &e)
                                   { return e.first == e.second; }),
                    edge_list.end());
    if (p == 1)
        return edge_list;
    if (p == 0)
        return {};
    Static_RNG::default_rng rng(seed);
    Static_RNG::bernoulli_distribution<float> dist(p);
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

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> random_connect(const std::vector<std::vector<uint32_t>> &nodelists,
                                                                       float p_in, float p_out, uint32_t seed)
{
    uint32_t N_node_pairs = n_choose_k(nodelists.size(), 2) + nodelists.size();
    std::random_device rd;
    std::vector<uint32_t> seeds(N_node_pairs);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, float, uint32_t>> node_pairs;
    node_pairs.reserve(N_node_pairs);
    uint32_t n = 0;
    std::vector<uint32_t> nodelist_idx(nodelists.size());
    std::iota(nodelist_idx.begin(), nodelist_idx.end(), 0);

    for (auto &&comb : iter::combinations_with_replacement(nodelist_idx, 2))
    {
        auto from_idx = comb[0];
        auto to_idx = comb[1];
        if (from_idx == to_idx)
        {
            node_pairs.push_back(std::make_tuple(nodelists[from_idx], nodelists[to_idx], p_in, seeds[n]));
        }
        else
        {
            node_pairs.push_back(std::make_tuple(nodelists[from_idx], nodelists[to_idx], p_out, seeds[n]));
        }
        n++;
    }
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists(N_node_pairs);
    std::vector<uint32_t> connection_idx(N_node_pairs);
    std::iota(connection_idx.begin(), connection_idx.end(), 0);

    std::transform(std::execution::par_unseq, node_pairs.begin(),
                   node_pairs.end(), connection_idx.begin(), edge_lists.begin(), [&](const auto &t, const auto idx)
                   {
                   auto from_nodes = std::get<0>(t);
                   auto to_nodes = std::get<1>(t);
                   auto p = std::get<2>(t);
                   auto seed = std::get<3>(t);
                   return random_connect(from_nodes, to_nodes, p, idx, seed); });

    return edge_lists;
}

std::tuple<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>,std::vector<std::vector<uint32_t>>> generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_clusters, float p_in, float p_out, uint32_t seed)
{
    std::vector<std::vector<uint32_t>> nodelists(N_clusters);
    std::generate(nodelists.begin(), nodelists.end(), [N_pop, offset = 0]()
                   mutable {
                    std::vector<uint32_t> nodes(N_pop);
                    std::iota(nodes.begin(), nodes.end(), offset);
                    offset += N_pop;
                    return nodes;
                  });
    return std::make_tuple(random_connect(nodelists, p_in, p_out, seed), nodelists);
}

std::vector<std::pair<uint32_t, uint32_t>> complete_ccm(uint32_t N_communities)
{
    uint32_t N_edges = N_communities*(N_communities-1);
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    ccm.reserve(N_edges);
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    for(auto&& prod : iter::combinations_with_replacement(community_idx, 2))
    {
        ccm.push_back(std::make_pair(prod[0], prod[1]));
    }

    return ccm;

}
