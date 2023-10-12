#ifndef SYCL_GRAPH_GRAPH_CONNECTION_COMMUNITY_MAP_HPP
#define SYCL_GRAPH_GRAPH_CONNECTION_COMMUNITY_MAP_HPP
#include <Sycl_Graph/Dataframe/Dataframe.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
std::vector<Edge_t> combine_ccm(const std::vector<std::pair<uint32_t, uint32_t>> &ccm_indices, const std::vector<uint32_t> &ccm_weights);

Dataframe_t<Edge_t, 2> make_ccm_df(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccm_indices, const std::vector<std::vector<uint32_t>> &ccm_weights);

std::vector<std::pair<uint32_t, uint32_t>> complete_ccm(uint32_t N_communities, bool directed = false);

std::vector<std::pair<uint32_t, uint32_t>> ccm_from_edgelist(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm);

std::vector<uint32_t> create_ecm(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_lists);

std::vector<uint32_t> create_vcm(const std::vector<std::vector<uint32_t>> node_lists);
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> create_community_mappings(const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, const std::vector<std::vector<uint32_t>> &vertex_list);
std::tuple<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> create_community_mappings(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<std::vector<uint32_t>>> &vertex_list);

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccms_from_vcms(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edges, const std::vector<std::vector<uint32_t>> &vcms);
template <typename T>
std::vector<std::vector<uint32_t>> ecms_from_vcms(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edges, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<T>> &ccms);

std::vector<uint32_t> ecm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &ccm);
std::vector<std::pair<uint32_t, uint32_t>> ccm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm);
std::vector<uint32_t> ccm_weights_from_ecm(const std::vector<uint32_t> &ecm, uint32_t N_connections);
auto read_ccm(const std::string &ccm_path);
std::vector<std::vector<uint32_t>> ccm_weights_from_ecms(const std::vector<std::vector<uint32_t>> &ecms, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccms);

std::vector<float> project_on_connection(const std::vector<uint32_t> &ecm, float value, uint32_t connection_index);

#endif
