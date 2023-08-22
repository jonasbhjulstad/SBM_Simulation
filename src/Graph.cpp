#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <itertools.hpp>
#include <execution>
#include <algorithm>
#include <random>
#include <tuple>
#include <unordered_map>
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


using Node_Edge_Tuple_t = std::tuple<Edge_List_t, Node_List_t>;

Node_Edge_Tuple_t generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed)
{
    std::vector<std::vector<uint32_t>> nodelists(N_communities);
    std::generate(nodelists.begin(), nodelists.end(), [N_pop, offset = 0]()
                   mutable {
                    std::vector<uint32_t> nodes(N_pop);
                    std::iota(nodes.begin(), nodes.end(), offset);
                    offset += N_pop;
                    return nodes;
                  });
    return std::make_tuple(random_connect(nodelists, p_in, p_out, seed), nodelists);
}


std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>> generate_N_SBM_graphs(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng)
{
    std::vector<Node_Edge_Tuple_t> result(Ng);
    std::vector<uint32_t> seeds(Ng);
    std::mt19937 gen(seed);
    std::generate_n(seeds.begin(), Ng, [&gen]()
    {
        return gen();
    });
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), result.begin(), [&](auto seed)
    {
        return generate_planted_SBM_edges(N_pop, N_communities, p_in, p_out, seed);
    });

    std::vector<Edge_List_t> edge_data(Ng);
    std::vector<Node_List_t> node_data(Ng);

    std::transform(std::execution::par_unseq, result.begin(), result.end(), edge_data.begin(), [](auto& t)
    {
        return std::get<0>(t);
    });
    std::transform(std::execution::par_unseq, result.begin(), result.end(), node_data.begin(), [](auto& t)
    {
        return std::get<1>(t);
    });

    return std::make_tuple(edge_data, node_data);
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

  std::vector<uint32_t> create_ecm(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_lists)
  {
    std::vector<uint32_t> list_sizes(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), list_sizes.begin(), [](auto& edge_list){return edge_list.size();});
    uint32_t N_edges = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uint32_t> ecm(N_edges);
    uint32_t offset = 0;
    for(int i = 0; i < edge_lists.size(); i++)
    {
        std::fill(ecm.begin() + offset, ecm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return ecm;
  }

  std::vector<uint32_t> create_vcm(const std::vector<std::vector<uint32_t>> node_lists)
  {
    std::vector<uint32_t> list_sizes(node_lists.size());
    std::transform(node_lists.begin(), node_lists.end(), list_sizes.begin(), [](auto& node_list){return node_list.size();});
    uint32_t N_nodes = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uint32_t> vcm(N_nodes);
    uint32_t offset = 0;
    for(int i = 0; i < node_lists.size(); i++)
    {
        std::fill(vcm.begin() + offset, vcm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return vcm;
  }


std::vector<uint32_t> ecm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>>& edges, const std::vector<uint32_t>& vcm)
{
    std::vector<uint32_t> ecm(edges.size());
    auto get_edge_communities = [vcm](auto edge)
    {
        return std::make_pair(vcm[edge.first], vcm[edge.second]);
    };
    uint32_t N_communities = *std::max_element(vcm.begin(), vcm.end())+1;
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    std::vector<std::pair<uint32_t, uint32_t>> connection_pairs;
    uint32_t N_connections = 0;

    for(auto&& comb : iter::combinations_with_replacement(community_idx, 2))
    {
        connection_pairs.push_back(std::make_pair(comb[0], comb[1]));
        N_connections++;
    }

    std::transform(edges.begin(), edges.end(), ecm.begin(), [&](const auto& edge)
    {
        auto edge_communities = get_edge_communities(edge);
        //find index of edge_communities in connection_pairs
        auto it = std::find(connection_pairs.begin(), connection_pairs.end(), edge_communities);
        return std::distance(connection_pairs.begin(), it);
    });

    return ecm;
}


std::vector<uint32_t> ccm_weights_from_ecm(const std::vector<uint32_t>& ecm)
{
    uint32_t N_connections = std::max_element(ecm.begin(), ecm.end())[0]+1;
    std::vector<uint32_t> ccm_weights(N_connections, 0);
    std::for_each(ecm.begin(), ecm.end(), [&](const uint32_t idx)
    {
        ccm_weights[idx]++;
    });
    return ccm_weights;
}

std::vector<std::vector<uint32_t>> ccm_weights_from_ecms(const std::vector<uint32_t>& ecms, const std::vector<uint32_t>& edge_counts)
{
    std::vector<std::vector<uint32_t>> result(ecms.size());

    return result;
}
