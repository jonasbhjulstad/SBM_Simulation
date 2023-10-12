#include <Sycl_Graph/Graph/Community_Mappings.hpp>
#include <itertools.hpp>
std::vector<Edge_t> combine_ccm(const std::vector<std::pair<uint32_t, uint32_t>> &ccm_indices, const std::vector<uint32_t> &ccm_weights)
{
    auto make_edges = [](const auto pair, const auto weight)
    {
        return Edge_t{pair.first, pair.second, weight};
    };
    std::vector<Edge_t> result;
    result.reserve(ccm_indices.size());
    std::transform(ccm_indices.begin(), ccm_indices.end(), ccm_weights.begin(), std::back_inserter(result), make_edges);
    return result;
}
Dataframe_t<Edge_t, 2> make_ccm_df(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccm_indices, const std::vector<std::vector<uint32_t>> &ccm_weights)
{
    Dataframe_t<Edge_t, 2> df;
    auto N_graphs = ccm_weights.size();
    df.data.reserve(N_graphs);
    std::vector<Dataframe_t<Edge_t, 1>> ccms(N_graphs);
    std::transform(ccm_indices.begin(), ccm_indices.end(), ccm_weights.begin(), ccms.begin(), [](const std::vector<std::pair<uint32_t, uint32_t>>& ccm_i, const auto &ccm_w)
                   { return Dataframe_t<Edge_t, 1>(combine_ccm(ccm_i, ccm_w)); });
    return Dataframe_t<Edge_t, 2>(ccms);
}

std::vector<uint32_t> ecm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &ccm)
{
    auto directed_equal = [](const auto& e0, const auto& e1)
    {
        return ((e0.first == e1.first) && (e0.second == e1.second));
    };
    std::vector<uint32_t> ecm(edges.size());
    std::transform(edges.begin(), edges.end(), ecm.begin(), [&](const auto& edge)
    {
        auto c_0 = vcm[edge.first];
        auto c_1 = vcm[edge.second];
        auto idx = std::find_if(ccm.begin(), ccm.end(), [&](const auto& cc)
        {
            return directed_equal(cc, std::make_pair(c_0, c_1));
        });
        auto res = std::distance(ccm.begin(), idx);
        return res;
    });
    return ecm;
}

std::vector<uint32_t> ecm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm, const std::vector<Edge_t> &ccm)
{
    auto directed_equal = [](const auto& e0, const auto& e1)
    {
        return ((e0.from == e1.first) && (e0.to == e1.second));
    };
    std::vector<uint32_t> ecm(edges.size());
    std::transform(edges.begin(), edges.end(), ecm.begin(), [&](const auto& edge)
    {
        auto c_0 = vcm[edge.first];
        auto c_1 = vcm[edge.second];
        auto idx = std::find_if(ccm.begin(), ccm.end(), [&](const auto& cc)
        {
            return directed_equal(cc, std::make_pair(c_0, c_1));
        });
        auto res = std::distance(ccm.begin(), idx);
        return res;
    });
    return ecm;
}

std::vector<std::pair<uint32_t, uint32_t>> complete_ccm(uint32_t N_communities, bool directed)
{
    uint32_t N_edges = N_communities * (N_communities - 1);
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    ccm.reserve((directed ? 2 * N_edges : N_edges));
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);

    if (directed)
    {
        for (auto &&prod : iter::combinations_with_replacement(community_idx, 2))
        {
            ccm.push_back(std::make_pair(prod[0], prod[1]));
            if (prod[0] != prod[1])
                ccm.push_back(std::make_pair(prod[1], prod[0]));
        }
    }
    else
    {
        for (auto &&prod : iter::combinations_with_replacement(community_idx, 2))
        {
            ccm.push_back(std::make_pair(prod[0], prod[1]));
        }
    }

    return ccm;
}

std::vector<std::pair<uint32_t, uint32_t>> ccm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm)
{
    // std::vector<std::pair<uint32_t, uint32_t>> ccm;
    auto N_communities = *std::max_element(vcm.begin(), vcm.end()) + 1;
    auto ccm = complete_ccm(N_communities, true);
    return ccm;
}

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccms_from_vcms(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edges, const std::vector<std::vector<uint32_t>> &vcms)
{
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> result(edges.size());
    std::transform(edges.begin(), edges.end(), vcms.begin(), result.begin(), [](const auto &edge, const auto &vcm)
                   { return ccm_from_vcm(edge, vcm); });
    return result;
}

template <typename T>
std::vector<std::vector<uint32_t>> ecms_from_vcms(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edges, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<T>> &ccms)
{
    std::vector<std::vector<uint32_t>> result(edges.size());
    for(int i = 0; i < edges.size(); i++)
    {
        result[i] = ecm_from_vcm(edges[i], vcms[i], ccms[i]);
    }
    return result;
}

template std::vector<std::vector<uint32_t>> ecms_from_vcms(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edges, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccms);
template std::vector<std::vector<uint32_t>> ecms_from_vcms(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edges, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<Edge_t>> &ccms);


std::vector<uint32_t> ccm_weights_from_ecm(const std::vector<uint32_t> &ecm, uint32_t N_connections)
{
    std::vector<uint32_t> ccm_weights(N_connections, 0);
    std::for_each(ecm.begin(), ecm.end(), [&](const uint32_t idx)
                  { ccm_weights[idx]++; });
    return ccm_weights;
}

std::vector<std::vector<uint32_t>> ccm_weights_from_ecms(const std::vector<std::vector<uint32_t>> &ecms, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& ccms)
{
    std::vector<std::vector<uint32_t>> result(ecms.size());

    for (int i = 0; i < ecms.size(); i++)
    {
        result[i] = ccm_weights_from_ecm(ecms[i], ccms[i].size());
    }
    return result;
}


std::vector<std::pair<uint32_t, uint32_t>> ccm_from_edgelist(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm)
{
    auto N_communities = *std::max_element(vcm.begin(), vcm.end()) + 1;
    std::vector<uint32_t> community_idx(N_communities);
    std::iota(community_idx.begin(), community_idx.end(), 0);
    std::vector<std::pair<uint32_t, uint32_t>> ccm;

    auto is_edge_in_list = [&](const auto &e_list, auto e_from, auto e_to)
    {
        for (auto &&e : e_list)
        {
            if ((e.first == e_from && e.second == e_to) || (e.first == e_to && e.second == e_from))
                return true;
        }
        return false;
    };
    auto edge_connects_communities = [&vcm](const auto &e, auto c_0, auto c_1)
    {
        return ((vcm[e.first] == c_0) && (vcm[e.second] == c_1)) || ((vcm[e.first] == c_1) && (vcm[e.second] == c_0));
    };
    for (auto &&comb : iter::combinations_with_replacement(community_idx, 2))
    {
        for (int i = 0; i < edges.size(); i++)
        {
            if (((edge_connects_communities(edges[i], comb[0], comb[1]))) && !is_edge_in_list(ccm, comb[0], comb[1]))
                ccm.push_back(std::make_pair(comb[0], comb[1]));
        }
    }
    return ccm;
}


std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> create_community_mappings(const std::vector<std::pair<uint32_t, uint32_t>>& edge_list, const std::vector<std::vector<uint32_t>>& vertex_list)
{
    auto vcm = create_vcm(vertex_list);
    auto ccm = ccm_from_vcm(edge_list, vcm);
    auto ecm = ecm_from_vcm(edge_list, vcm, ccm);
    return std::make_tuple(ecm, vcm, ccm);
}


std::tuple<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> create_community_mappings(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_list, const std::vector<std::vector<std::vector<uint32_t>>>& vertex_list)
{
    std::tuple<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> result;
    auto N_mappings = edge_list.size();
    std::get<0>(result).resize(N_mappings);
    std::get<1>(result).resize(N_mappings);
    std::get<2>(result).resize(N_mappings);

    for(int i = 0; i < N_mappings; i++)
    {
        std::tie(std::get<0>(result)[i], std::get<1>(result)[i], std::get<2>(result)[i]) = create_community_mappings(edge_list[i], vertex_list[i]);
    }
    return result;
}

std::tuple<Dataframe_t<uint32_t, 1>, Dataframe_t<uint32_t, 1>, Dataframe_t<std::pair<uint32_t, uint32_t>, 1>> create_community_mappings(const Dataframe_t<std::pair<uint32_t, uint32_t>, 1>& edge_list, const Dataframe_t<uint32_t, 2>& vertex_list)
{
    auto vcm = create_vcm(vertex_list);
    auto ccm = ccm_from_vcm(edge_list, vcm);
    auto ecm = ecm_from_vcm(edge_list, vcm, ccm);
    return std::make_tuple(ecm, vcm, ccm);
}

std::tuple<Dataframe_t<uint32_t, 2>, Dataframe_t<uint32_t, 2>, Dataframe_t<std::pair<uint32_t, uint32_t>, 2>> create_community_mappings(const Dataframe_t<std::pair<uint32_t, uint32_t>, 2>& edge_list, const Dataframe_t<uint32_t, 3>& vertex_list)
{
    std::tuple<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> result;
    auto N_mappings = edge_list.size();
    std::get<0>(result).resize(N_mappings);
    std::get<1>(result).resize(N_mappings);
    std::get<2>(result).resize(N_mappings);

    for(int i = 0; i < N_mappings; i++)
    {
        std::tie(std::get<0>(result)[i], std::get<1>(result)[i], std::get<2>(result)[i]) = create_community_mappings(edge_list[i], vertex_list[i]);
    }
    return result;
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

std::vector<uint32_t> create_vcm(size_t N_pop, size_t N_clusters)
{
    std::vector<uint32_t> vcm(N_pop * N_clusters);
    for (int i = 0; i < N_clusters; i++)
    {
        std::fill(vcm.begin() + i * N_pop, vcm.begin() + (i + 1) * N_pop, i);
    }
    return vcm;
}


std::vector<float> project_on_connection(const std::vector<uint32_t> &ecm, float value, uint32_t connection_index)
{
    std::vector<float> result(ecm.size(), 0);
    // insert value at connection_index
    std::transform(ecm.begin(), ecm.end(), result.begin(), [&](auto idx)
                   {
        if(idx == connection_index)
        {
            return value;
        }
        else
        {
            return 0.0f;
        } });
    return result;
}

auto read_ccm(const std::string &ccm_path)
{
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::ifstream ccm_file(ccm_path);
    std::string line;
    while (std::getline(ccm_file, line))
    {
        std::stringstream line_stream(line);
        std::string from_idx_str;
        std::string to_idx_str;
        std::getline(line_stream, from_idx_str, ',');
        std::getline(line_stream, to_idx_str, '\n');
        ccm.push_back(std::make_pair(std::stoi(from_idx_str), std::stoi(to_idx_str)));
    }
    return ccm;
}



// void validate_maps(const std::vector<std::vector<uint32_t>> &ecms, const std::vector<std::vector<uint32_t>> &vcms, const auto &N_communities, const auto &N_connections, const auto &edge_list)
// {
//     for (auto g_idx = 0; g_idx < ecms.size(); g_idx++)
//     {
//         if_false_throw(std::all_of(ecms[g_idx].begin(), ecms[g_idx].end(), [&](const auto &e)
//                                    { return e < N_connections[g_idx]; }),
//                        "ecms[" + std::to_string(g_idx) + "] has index values higher than N_connections: " + std::to_string(N_connections[g_idx]));

//         if_false_throw(std::all_of(vcms[g_idx].begin(), vcms[g_idx].end(), [&](const auto &v)
//                                    { return v < N_communities[g_idx]; }),
//                        "vcms[" + std::to_string(g_idx) + "] has index values higher than N_communities: " + std::to_string(N_communities[g_idx]));
//         if_false_throw(edge_list[g_idx].size() == ecms[g_idx].size(), "edge_list[" + std::to_string(g_idx) + "] and ecms[" + std::to_string(g_idx) + "] have different sizes: " + std::to_string(edge_list[g_idx].size()) + " vs " + std::to_string(ecms[g_idx].size()));
//     }
// }

// void validate_ecm(const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<uint32_t> &ecm, const std::vector<> &ccm, const auto &vcm)
// {
//     auto directed_equal = [](const auto &e_0, const auto &e_1)
//     {
//         return (e_0.first == e_1.first) && (e_0.second == e_1.second);
//     };
//     auto is_edge_in_connection = [&](auto edge, auto ecm_id)
//     {
//         auto c_edge = std::make_pair(vcm[edge.first], vcm[edge.second]);
//         for (int i = 0; i < ccm.size(); i++)
//         {

//             if (directed_equal(c_edge, ccm[i]))
//             {
//                 return ecm_id == i;
//             }
//         }
//         return false;
//     };
//     for (int i = 0; i < edge_list.size(); i++)
//     {
//         if_false_throw(is_edge_in_connection(edge_list[i], ecm[i]), "edge_list[" + std::to_string(i) + "] is not in the correct connection");
//     }
// }
