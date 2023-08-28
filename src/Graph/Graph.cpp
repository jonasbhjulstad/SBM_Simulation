#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Utils/Vector_Utils.hpp>
#include <algorithm>
#include <execution>
#include <itertools.hpp>
#include <random>
#include <thread>
#include <tuple>
#include <unordered_map>



auto initialize_rngs = [](uint32_t seed, uint32_t N)
{
    std::mt19937 gen(seed);
    std::vector<uint32_t> seeds(N);
    std::generate_n(seeds.begin(), N, [&gen]()
                    { return gen(); });
    std::vector<std::mt19937> rngs(N);
    std::transform(seeds.begin(), seeds.end(), rngs.begin(), [](auto s)
                    { return std::mt19937(s); });
    return rngs;
};

std::vector<std::pair<uint32_t, uint32_t>> random_connect(const std::vector<uint32_t> &to_nodes,
                                                          const std::vector<uint32_t> &from_nodes, float p, uint32_t seed)
{
    auto max_size = std::min<std::size_t>({to_nodes.max_size(), std::numeric_limits<uint32_t>::max()});

    const auto N_threads = std::thread::hardware_concurrency();
    uint32_t N_nodes = to_nodes.size() + from_nodes.size();
    auto max_edges = (N_nodes * N_nodes) / 4;
    auto expected_edges  = max_edges * p;
    if (max_edges > max_size)
    {
        throw std::runtime_error("Error: Number of edges exceeds max value");
    }
    auto rngs = initialize_rngs(seed, N_threads);
    auto to_node_lists = vector_split(to_nodes, N_threads);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists(N_threads);
    std::transform(std::execution::par_unseq, to_node_lists.begin(), to_node_lists.end(), rngs.begin(), edge_lists.begin(), [=](auto &to_node_list, auto &rng)
                   {
                        std::bernoulli_distribution dist(p);
                       std::vector<std::pair<uint32_t, uint32_t>> edge_list;
                       edge_list.reserve(expected_edges/N_threads);
                       for (auto &to_node : to_node_list)
                       {
                           for (auto &from_node : from_nodes)
                           {
                               if (to_node != from_node)
                               {
                                   if (dist(rng))
                                   {
                                       edge_list.push_back(std::make_pair(to_node, from_node));
                                   }
                               }
                           }
                       }
                       return edge_list; });
    auto edge_list = merge_vectors(edge_lists);
    return edge_list;
}

std::vector<std::pair<uint32_t, uint32_t>> self_connect(const std::vector<uint32_t> &nodes, float p, uint32_t seed)
{
    auto max_size = std::min<std::size_t>({nodes.max_size(), std::numeric_limits<uint32_t>::max()});

    const auto N_threads = std::thread::hardware_concurrency();
    uint32_t N_nodes = nodes.size();
    auto max_edges = (N_nodes * N_nodes) / 4;
    auto expected_edges  = max_edges * p;
    if (max_edges > max_size)
    {
        throw std::runtime_error("Error: Number of edges exceeds max value");
    }
    auto rngs = initialize_rngs(seed, N_threads);
    auto to_node_lists = vector_split(nodes, N_threads);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists(N_threads);
    std::transform(std::execution::par_unseq, to_node_lists.begin(), to_node_lists.end(), rngs.begin(), edge_lists.begin(), [=](auto &to_node_list, auto &rng)
                   {
                        std::bernoulli_distribution dist(p);
                       std::vector<std::pair<uint32_t, uint32_t>> edge_list;
                       edge_list.reserve(expected_edges/N_threads);
                       for (auto &to_node : to_node_list)
                       {
                           for (auto &from_node : nodes)
                           {
                               if (to_node > from_node)
                               {
                                   if (dist(rng))
                                   {
                                       edge_list.push_back(std::make_pair(to_node, from_node));
                                   }
                               }
                           }
                       }
                       return edge_list; });
    auto edge_list = merge_vectors(edge_lists);
    return edge_list;
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
                   if (std::equal(from_nodes.begin(), from_nodes.end(), to_nodes.begin(), to_nodes.end()))
                     {
                          return self_connect(from_nodes, p, seed);
                     }
                     else
                     {
                          return random_connect(from_nodes, to_nodes, p, seed);
                     } });
    return edge_lists;
}

std::vector<std::pair<uint32_t, uint32_t>> merge_vecs(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &vecs)
{
    std::vector<std::pair<uint32_t, uint32_t>> result;
    uint32_t size = 0;
    for (int i = 0; i < vecs.size(); i++)
    {
        size += vecs[i].size();
    }
    result.reserve(size);
    for (auto &v : vecs)
    {
        result.insert(result.end(), v.begin(), v.end());
    }
    return result;
}

std::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<uint32_t>> generate_planted_SBM(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed)
{
    std::vector<std::vector<uint32_t>> nodelists(N_communities);
    std::vector<uint32_t> node_idx(N_pop);
    for (int i = 0; i < N_communities; i++)
    {
        std::fill(node_idx.begin(), node_idx.end(), i);
        nodelists[i] = node_idx;
    }
    auto edgelists = random_connect(nodelists, p_in, p_out, seed);

    auto edgelist = merge_vecs(edgelists);

    std::vector<uint32_t> vcm = create_vcm(N_pop, N_communities);

    return std::make_tuple(edgelist, vcm);
}

std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>> generate_N_SBM_graphs(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng)
{
    std::vector<Node_Edge_Tuple_t> result(Ng);
    std::vector<uint32_t> seeds(Ng);
    std::mt19937 gen(seed);
    // generate seeds
    std::generate_n(seeds.begin(), Ng, [&gen]()
                    { return gen(); });
    assert(seeds[0] != seeds[1]);
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), result.begin(), [&](auto s)
                   { return generate_planted_SBM(N_pop, N_communities, p_in, p_out, s); });

    std::vector<Edge_List_t> edge_data(Ng);
    std::vector<Node_List_t> node_data(Ng);

    std::transform(std::execution::par_unseq, result.begin(), result.end(), edge_data.begin(), [](auto &t)
                   { return std::get<0>(t); });
    std::transform(std::execution::par_unseq, result.begin(), result.end(), node_data.begin(), [](auto &t)
                   { return std::get<1>(t); });

    return std::make_tuple(edge_data, node_data);
}

std::vector<std::pair<uint32_t, uint32_t>> complete_ccm(uint32_t N_communities)
{
    uint32_t N_edges = N_communities * (N_communities - 1);
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    ccm.reserve(N_edges);
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    for (auto &&prod : iter::combinations_with_replacement(community_idx, 2))
    {
        ccm.push_back(std::make_pair(prod[0], prod[1]));
    }

    return ccm;
}

std::vector<uint32_t> create_ecm(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_lists)
{
    std::vector<uint32_t> list_sizes(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), list_sizes.begin(), [](auto &edge_list)
                   { return edge_list.size(); });
    uint32_t N_edges = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uint32_t> ecm(N_edges);
    uint32_t offset = 0;
    for (int i = 0; i < edge_lists.size(); i++)
    {
        std::fill(ecm.begin() + offset, ecm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return ecm;
}

std::vector<uint32_t> create_vcm(const std::vector<std::vector<uint32_t>> node_lists)
{
    std::vector<uint32_t> list_sizes(node_lists.size());
    std::transform(node_lists.begin(), node_lists.end(), list_sizes.begin(), [](auto &node_list)
                   { return node_list.size(); });
    uint32_t N_nodes = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uint32_t> vcm(N_nodes);
    uint32_t offset = 0;
    for (int i = 0; i < node_lists.size(); i++)
    {
        std::fill(vcm.begin() + offset, vcm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return vcm;
}

std::vector<uint32_t> create_vcm(uint32_t N_pop, uint32_t N_communities)
{
    std::vector<uint32_t> result(N_pop * N_communities);
    for (int i = 0; i < N_communities; i++)
    {
        std::vector<uint32_t> idx(N_pop, i);
        std::copy(idx.begin(), idx.end(), result.begin() + i * N_pop);
    }
    return result;
}

std::vector<uint32_t> ecm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm)
{
    std::vector<uint32_t> ecm(edges.size());
    auto get_edge_communities = [vcm](auto edge)
    {
        return std::make_pair(vcm[edge.first], vcm[edge.second]);
    };
    uint32_t N_communities = *std::max_element(vcm.begin(), vcm.end()) + 1;
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    std::vector<std::pair<uint32_t, uint32_t>> connection_pairs;
    uint32_t N_connections = 0;

    for (auto &&comb : iter::combinations_with_replacement(community_idx, 2))
    {
        connection_pairs.push_back(std::make_pair(comb[0], comb[1]));
        N_connections++;
    }

    std::transform(edges.begin(), edges.end(), ecm.begin(), [&](const auto &edge)
                   {
        auto edge_communities = get_edge_communities(edge);
        //find index of edge_communities in connection_pairs
        auto it = std::find(connection_pairs.begin(), connection_pairs.end(), edge_communities);
        return std::distance(connection_pairs.begin(), it); });

    return ecm;
}

std::vector<uint32_t> ccm_weights_from_ecm(const std::vector<uint32_t> &ecm)
{
    uint32_t N_connections = std::max_element(ecm.begin(), ecm.end())[0] + 1;
    std::vector<uint32_t> ccm_weights(N_connections, 0);
    std::for_each(ecm.begin(), ecm.end(), [&](const uint32_t idx)
                  { ccm_weights[idx]++; });
    return ccm_weights;
}

std::vector<std::vector<uint32_t>> ccm_weights_from_ecms(const std::vector<uint32_t> &ecms, const std::vector<uint32_t> &edge_counts)
{
    std::vector<std::vector<uint32_t>> result(ecms.size());

    return result;
}
