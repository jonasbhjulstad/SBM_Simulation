#ifndef GRAPH_GENERATION_HPP
#define GRAPH_GENERATION_HPP
#include <Sycl_Graph/Utils/Common.hpp>
#include <Sycl_Graph/Utils/Dataframe.hpp>

using Edge_List_t = std::vector<std::vector<std::pair<uint32_t, uint32_t>>>;
using Node_List_t = std::vector<std::vector<uint32_t>>;
using Edge_List_Flat_t = std::vector<std::pair<uint32_t, uint32_t>>;
using Node_List_Flat_t = std::vector<uint32_t>;
using Node_Edge_Tuple_Flat_t = std::tuple<Edge_List_Flat_t, Node_List_Flat_t>;
std::size_t complete_graph_max_edges(std::size_t N, bool self_loops=true, bool directed = false);

std::vector<std::pair<uint32_t, uint32_t>> random_connect(const std::vector<uint32_t> &to_nodes,
                                                          const std::vector<uint32_t> &from_nodes, float p, uint32_t connection_idx, uint32_t seed);
std::vector<std::vector<std::pair<uint32_t, uint32_t>>> random_connect(const std::vector<std::vector<uint32_t>> &nodelists,
                                                                       float p_in, float p_out, uint32_t seed);
std::tuple<Edge_List_t, Node_List_t, std::vector<uint32_t>, std::vector<uint32_t>> generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed);

std::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<uint32_t>, std::vector<uint32_t>> generate_planted_SBM_flat(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed);

std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>, std::vector<std::vector<uint32_t>>,std::vector<std::vector<uint32_t>>> generate_N_SBM_graphs(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng);

std::tuple<std::vector<Edge_List_Flat_t>, std::vector<Node_List_Flat_t>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> generate_N_SBM_graphs_flat(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng);

std::vector<std::pair<uint32_t, uint32_t>> complete_ccm(uint32_t N_communities, bool directed = false);

std::vector<std::pair<uint32_t, uint32_t>> ccm_from_edgelist(const std::vector<std::pair<uint32_t, uint32_t>>& edges, const std::vector<uint32_t>& vcm);


std::vector<uint32_t> create_ecm(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_lists);

std::vector<uint32_t> create_vcm(const std::vector<std::vector<uint32_t>> node_lists);

std::vector<uint32_t> ecm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>>& ccm);
std::vector<std::pair<uint32_t, uint32_t>> ccm_from_vcm(const std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<uint32_t> &vcm);
std::vector<std::pair<uint32_t, uint32_t>> complete_graph(size_t N);
std::size_t complete_graph_size(size_t N, bool directed = true, bool self_loops = false);


std::vector<uint32_t> ccm_weights_from_ecm(const std::vector<uint32_t> &ecm, uint32_t N_connections);
void write_edgelist(const std::string &fname, const Dataframe_t<std::pair<uint32_t, uint32_t>, 1> &edges);
void read_edgelist(const std::string& fname, std::vector<std::pair<uint32_t, uint32_t>>& edges);
std::vector<std::pair<uint32_t, uint32_t>> read_edgelist(const std::string& fname);
std::vector<uint32_t> read_vec(const std::string& fpath, size_t N);

void write_vector(const std::string& fname, const std::vector<uint32_t>& vec);
// std::vector<float> project_on_connection(const std::vector<uint32_t>& ecm, float value, uint32_t connection_index);
std::vector<float> project_on_connection(const std::vector<uint32_t>& ecm, const std::vector<float>& values, uint32_t connection_index);
auto read_ccm(const std::string& ccm_path);

#endif
