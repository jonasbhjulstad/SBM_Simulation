#ifndef SYCL_GRAPH_GRAPH_IMPL_HPP
#define SYCL_GRAPH_GRAPH_IMPL_HPP

#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Utils/Vector_Utils.hpp>
#include <algorithm>
#include <execution>
#include <itertools.hpp>
#include <random>
#include <tuple>
#include <CL/sycl.hpp>
#include <oneapi/dpl/random>

auto initialize_rngs = [](uI_t seed, uI_t N)
{
    std::mt19937 gen(seed);
    std::vector<uI_t> seeds(N);
    std::generate_n(seeds.begin(), N, [&gen]()
                    { return gen(); });
    std::vector<std::mt19937> rngs(N);
    std::transform(seeds.begin(), seeds.end(), rngs.begin(), [](auto s)
                    { return std::mt19937(s); });
    return rngs;
};

template <std::unsigned_integral uI_t, typename RNG>

std::vector<std::pair<uI_t, uI_t>> random_connect(sycl::queue& q, const std::vector<uI_t> &to, const std::vector<uI_t> &from, float p, uI_t seed)
{
    auto max_size = std::min<std::size_t>({to.max_size(), std::numeric_limits<uI_t>::max()});

    const auto N_threads = std::thread::hardware_concurrency();
    uI_t N = to.size() + from.size();
    auto max_edges = (N * N) / 4;
    auto expected_edges  = max_edges * p;
    if (max_edges > max_size)
    {
        throw std::runtime_error("Error: Number of edges exceeds max value");
    }
    auto rngs = initialize_rngs(seed, N_threads);
    auto to_node_lists = vector_split(to, N_threads);
    std::vector<std::vector<std::pair<uI_t, uI_t>>> edge_lists(N_threads);
    std::transform(std::execution::par_unseq, to_node_lists.begin(), to_node_lists.end(), rngs.begin(), edge_lists.begin(), [=](auto &to_node_list, auto &rng)
                   {
                        std::bernoulli_distribution dist(p);
                       std::vector<std::pair<uI_t, uI_t>> edge_list;
                       edge_list.reserve(expected_edges/N_threads);
                       for (auto &to_node : to_node_list)
                       {
                           for (auto &from_node : from)
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

std::vector<std::pair<uI_t, uI_t>> self_connect(const std::vector<uI_t> &nodes, float p, uI_t seed)
{
    auto max_size = std::min<std::size_t>({nodes.max_size(), std::numeric_limits<uI_t>::max()});

    const auto N_threads = std::thread::hardware_concurrency();
    uI_t N = nodes.size();
    auto max_edges = (N * N) / 4;
    auto expected_edges  = max_edges * p;
    if (max_edges > max_size)
    {
        throw std::runtime_error("Error: Number of edges exceeds max value");
    }
    auto rngs = initialize_rngs(seed, N_threads);
    auto to_node_lists = vector_split(nodes, N_threads);
    std::vector<std::vector<std::pair<uI_t, uI_t>>> edge_lists(N_threads);
    std::transform(std::execution::par_unseq, to_node_lists.begin(), to_node_lists.end(), rngs.begin(), edge_lists.begin(), [=](auto &to_node_list, auto &rng)
                   {
                        std::bernoulli_distribution dist(p);
                       std::vector<std::pair<uI_t, uI_t>> edge_list;
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


std::vector<std::vector<std::pair<uI_t, uI_t>>> random_connect(const std::vector<std::vector<uI_t>> &nodelists,
                                                                       float p_in, float p_out, uI_t seed)
{
    uI_t N_node_pairs = n_choose_k(nodelists.size(), 2) + nodelists.size();
    std::random_device rd;
    std::vector<uI_t> seeds(N_node_pairs);
    std::generate(seeds.begin(), seeds.end(), [&rd]()
                  { return rd(); });
    std::vector<std::tuple<std::vector<uI_t>, std::vector<uI_t>, float, uI_t>> node_pairs;
    node_pairs.reserve(N_node_pairs);
    uI_t n = 0;
    std::vector<uI_t> nodelist_idx(nodelists.size());
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
    std::vector<std::vector<std::pair<uI_t, uI_t>>> edge_lists(N_node_pairs);
    std::vector<uI_t> connection_idx(N_node_pairs);
    std::iota(connection_idx.begin(), connection_idx.end(), 0);

    std::transform(std::execution::par_unseq, node_pairs.begin(),
                   node_pairs.end(), connection_idx.begin(), edge_lists.begin(), [&](const auto &t, const auto idx)
                   {
                   auto from = std::get<0>(t);
                   auto to = std::get<1>(t);
                   auto p = std::get<2>(t);
                   auto seed = std::get<3>(t);
                   if (std::equal(from.begin(), from.end(), to.begin(), to.end()))
                     {
                          return self_connect(from, p, seed);
                     }
                     else
                     {
                          return random_connect(from, to, p, seed);
                     } });
    return edge_lists;
}
std::tuple<std::vector<std::pair<uI_t, uI_t>>, std::vector<uI_t>> generate_planted_SBM(uI_t N_pop, uI_t N_communities, float p_in, float p_out, uI_t seed)
{
    std::vector<std::vector<uI_t>> nodelists(N_communities);
    std::vector<uI_t> node_idx(N_pop);
    for (int i = 0; i < N_communities; i++)
    {
        std::fill(node_idx.begin(), node_idx.end(), i);
        nodelists[i] = node_idx;
    }
    auto edgelists = random_connect(nodelists, p_in, p_out, seed);

    auto edgelist = merge_vecs(edgelists);

    std::vector<uI_t> vcm = create_vcm(N_pop, N_communities);

    return std::make_tuple(edgelist, vcm);
}

std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>> generate_N_SBM_graphs(uI_t N_pop, uI_t N_communities, float p_in, float p_out, uI_t seed, std::size_t Ng)
{
    std::vector<Node_Edge_Tuple_t> result(Ng);
    std::vector<uI_t> seeds(Ng);
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

std::vector<std::pair<uI_t, uI_t>> complete_ccm(uI_t N_communities)
{
    uI_t N_edges = N_communities * (N_communities - 1);
    std::vector<std::pair<uI_t, uI_t>> ccm;
    ccm.reserve(N_edges);
    std::vector<uI_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    for (auto &&prod : iter::combinations_with_replacement(community_idx, 2))
    {
        ccm.push_back(std::make_pair(prod[0], prod[1]));
    }

    return ccm;
}

std::vector<uI_t> create_ecm(const std::vector<std::vector<std::pair<uI_t, uI_t>>> &edge_lists)
{
    std::vector<uI_t> list_sizes(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), list_sizes.begin(), [](auto &edge_list)
                   { return edge_list.size(); });
    uI_t N_edges = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uI_t> ecm(N_edges);
    uI_t offset = 0;
    for (int i = 0; i < edge_lists.size(); i++)
    {
        std::fill(ecm.begin() + offset, ecm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return ecm;
}

std::vector<uI_t> create_vcm(const std::vector<std::vector<uI_t>> node_lists)
{
    std::vector<uI_t> list_sizes(node_lists.size());
    std::transform(node_lists.begin(), node_lists.end(), list_sizes.begin(), [](auto &node_list)
                   { return node_list.size(); });
    uI_t N = std::accumulate(list_sizes.begin(), list_sizes.end(), 0);
    std::vector<uI_t> vcm(N);
    uI_t offset = 0;
    for (int i = 0; i < node_lists.size(); i++)
    {
        std::fill(vcm.begin() + offset, vcm.begin() + offset + list_sizes[i], i);
        offset += list_sizes[i];
    }
    return vcm;
}

std::vector<uI_t> create_vcm(uI_t N_pop, uI_t N_communities)
{
    std::vector<uI_t> result(N_pop * N_communities);
    for (int i = 0; i < N_communities; i++)
    {
        std::vector<uI_t> idx(N_pop, i);
        std::copy(idx.begin(), idx.end(), result.begin() + i * N_pop);
    }
    return result;
}

std::vector<uI_t> ecm_from_vcm(const std::vector<std::pair<uI_t, uI_t>> &edges, const std::vector<uI_t> &vcm)
{
    std::vector<uI_t> ecm(edges.size());
    auto get_edge_communities = [vcm](auto edge)
    {
        return std::make_pair(vcm[edge.first], vcm[edge.second]);
    };
    uI_t N_communities = *std::max_element(vcm.begin(), vcm.end()) + 1;
    std::vector<uI_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    std::vector<std::pair<uI_t, uI_t>> connection_pairs;
    uI_t N_connections = 0;

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

std::vector<uI_t> ccm_weights_from_ecm(const std::vector<uI_t> &ecm)
{
    uI_t N_connections = std::max_element(ecm.begin(), ecm.end())[0] + 1;
    std::vector<uI_t> ccm_weights(N_connections, 0);
    std::for_each(ecm.begin(), ecm.end(), [&](const uI_t idx)
                  { ccm_weights[idx]++; });
    return ccm_weights;
}

std::vector<std::vector<uI_t>> ccm_weights_from_ecms(const std::vector<uI_t> &ecms, const std::vector<uI_t> &edge_counts)
{
    std::vector<std::vector<uI_t>> result(ecms.size());

    return result;
}


#endif
