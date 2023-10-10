#include <Sycl_Graph/Graph/Graph.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <execution>
#include <itertools.hpp>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <Sycl_Graph/Utils/math.hpp>


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
long long n_choose_2(long long n)
{
    return n * (n - 1) / 2;
}

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> random_connect(const std::vector<std::vector<uint32_t>> &nodelists,
                                                                       float p_in, float p_out, uint32_t seed)
{
    uint32_t N_node_pairs = n_choose_2(nodelists.size()) + nodelists.size();
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

std::tuple<Edge_List_t, Node_List_t> generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed)
{
    std::vector<std::vector<uint32_t>> nodelists(N_communities);
    std::generate(nodelists.begin(), nodelists.end(), [N_pop, offset = 0]() mutable
                  {
                    std::vector<uint32_t> nodes(N_pop);
                    std::iota(nodes.begin(), nodes.end(), offset);
                    offset += N_pop;
                    return nodes; });
    auto edge_lists = random_connect(nodelists, p_in, p_out, seed);

    return std::make_tuple(edge_lists, nodelists);
}

std::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<uint32_t>> generate_planted_SBM_flat(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed)
{
    auto G_data = generate_planted_SBM_edges(N_pop, N_communities, p_in, p_out, seed);
    auto edge_list = std::get<0>(G_data);
    uint32_t N_edges = std::accumulate(edge_list.begin(), edge_list.end(), 0, [](auto sum, auto &edge_list)
                                       { return sum + edge_list.size(); });
    // flatten edge_list
    std::vector<std::pair<uint32_t, uint32_t>> flat_edge_list(N_edges);
    uint32_t offset = 0;
    for (auto &edge_list : edge_list)
    {
        std::copy(edge_list.begin(), edge_list.end(), flat_edge_list.begin() + offset);
        offset += edge_list.size();
    }
    auto node_list = std::get<1>(G_data);
    std::vector<uint32_t> flat_node_list(N_pop * N_communities);
    offset = 0;
    for (auto &node_list : node_list)
    {
        std::copy(node_list.begin(), node_list.end(), flat_node_list.begin() + offset);
        offset += node_list.size();
    }

    return std::make_tuple(flat_edge_list, flat_node_list);
}

std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>> generate_N_SBM_graphs(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng)
{
    std::vector<std::tuple<Edge_List_t, Node_List_t>> result(Ng);
    std::vector<uint32_t> seeds(Ng);
    std::mt19937 gen(seed);
    std::generate_n(seeds.begin(), Ng, [&gen]()
                    { return gen(); });
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), result.begin(), [&](auto seed)
                   { return generate_planted_SBM_edges(N_pop, N_communities, p_in, p_out, seed); });

    std::vector<Edge_List_t> edge_data(Ng);
    std::vector<Node_List_t> node_data(Ng);
    std::transform(std::execution::par_unseq, result.begin(), result.end(), edge_data.begin(), [](auto &t)
                   { return std::get<0>(t); });
    std::transform(std::execution::par_unseq, result.begin(), result.end(), node_data.begin(), [](auto &t)
                   { return std::get<1>(t); });
    return std::make_tuple(edge_data, node_data);
}

std::tuple<std::vector<Edge_List_Flat_t>, std::vector<Node_List_Flat_t>> generate_N_SBM_graphs_flat(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng)
{
    std::vector<std::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<uint32_t>>> result(Ng);
    std::vector<uint32_t> seeds(Ng);
    std::mt19937 gen(seed);
    // generate seeds
    std::generate_n(seeds.begin(), Ng, [&gen]()
                    { return gen(); });
    assert(seeds[0] != seeds[1]);
    std::transform(std::execution::par_unseq, seeds.begin(), seeds.end(), result.begin(), [&](auto s)
                   { return generate_planted_SBM_flat(N_pop, N_communities, p_in, p_out, s); });

    std::vector<Edge_List_Flat_t> edge_data(Ng);
    std::vector<Node_List_Flat_t> node_data(Ng);

    std::transform(std::execution::par_unseq, result.begin(), result.end(), edge_data.begin(), [](auto &t)
                   { return std::get<0>(t); });
    std::transform(std::execution::par_unseq, result.begin(), result.end(), node_data.begin(), [](auto &t)
                   { return std::get<1>(t); });

    return std::make_tuple(edge_data, node_data);
}



Dataframe_t<std::pair<uint32_t, uint32_t>, 2> mirror_duplicate_edge_list(const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list)
{
    Dataframe_t<std::pair<uint32_t, uint32_t>, 2> result(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), result.begin(), [](const auto &edge_list_elem)
                   { Dataframe_t<std::pair<uint32_t, uint32_t>, 1> result_elem(edge_list_elem.size() * 2);
                       std::transform(edge_list_elem.begin(), edge_list_elem.end(), result_elem.begin(), [](const auto &edge)
                                      { return edge; });
                       std::transform(edge_list_elem.begin(), edge_list_elem.end(), result_elem.begin() + edge_list_elem.size(), [](const auto &edge)
                                      { return std::make_pair(edge.second, edge.first); });
                       return result_elem; });
    return result;
}


void write_edgelist(const std::string &fname, const Dataframe_t<std::pair<uint32_t, uint32_t>, 1> &edges)
{
    std::ofstream f(fname);
    for (auto &&e : edges)
    {
        f << e.first << "," << e.second << "\n";
    }
}

void read_edgelist(const std::string &fname, std::vector<std::pair<uint32_t, uint32_t>> &edges)
{
    std::ifstream f(fname);
    std::string line;
    while (std::getline(f, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ','))
        {
            tokens.push_back(token);
        }
        edges.push_back(std::make_pair(std::stoi(tokens[0]), std::stoi(tokens[1])));
    }
}
std::vector<uint32_t> read_vec(const std::string& fpath, size_t N)
{
    //read
    std::vector<uint32_t> result(N);
    std::fstream f(fpath);
    std::string line;
    for(int i = 0; i < N; i++)
    {
        std::getline(f, line);
        result[i] = std::stoi(line);
    }
    return result;
}

std::vector<std::pair<uint32_t, uint32_t>> read_edgelist(const std::string& fname)
{
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    read_edgelist(fname, edges);
    return edges;
}


void write_vector(const std::string &fname, const std::vector<uint32_t> &vec)
{
    std::ofstream f(fname);
    for (auto &&e : vec)
    {
        f << e << "\n";
    }
}

// Project value onto the edges in connection_index
