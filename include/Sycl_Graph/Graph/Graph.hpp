#ifndef GRAPH_GENERATION_HPP
#define GRAPH_GENERATION_HPP
#include <Sycl_Graph/Dataframe/Dataframe.hpp>
using Edge_List_t = std::vector<std::vector<std::pair<uint32_t, uint32_t>>>;
using Node_List_t = std::vector<std::vector<uint32_t>>;
using Edge_List_Flat_t = std::vector<std::pair<uint32_t, uint32_t>>;
using Node_List_Flat_t = std::vector<uint32_t>;
using Node_Edge_Tuple_Flat_t = std::tuple<Edge_List_Flat_t, Node_List_Flat_t>;

std::vector<std::pair<uint32_t, uint32_t>> random_connect(const std::vector<uint32_t> &to_nodes,
                                                          const std::vector<uint32_t> &from_nodes, float p, uint32_t connection_idx, uint32_t seed);
std::vector<std::vector<std::pair<uint32_t, uint32_t>>> random_connect(const std::vector<std::vector<uint32_t>> &nodelists,
                                                                       float p_in, float p_out, uint32_t seed);


std::tuple<Edge_List_Flat_t, Node_List_t> generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed);

std::tuple<std::vector<Edge_List_Flat_t>, std::vector<Node_List_t>> generate_N_SBM_graphs(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng);


// std::tuple<Edge_List_t, Node_List_t, std::vector<uint32_t>, std::vector<uint32_t>> generate_planted_SBM_edges(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed);

// std::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<uint32_t>, std::vector<uint32_t>> generate_planted_SBM_flat(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed);

// std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>, std::vector<std::vector<uint32_t>>,std::vector<std::vector<uint32_t>>> generate_N_SBM_graphs(uint32_t N_pop, uint32_t N_communities, float p_in, float p_out, uint32_t seed, std::size_t Ng);

Dataframe_t<std::pair<uint32_t, uint32_t>, 2> mirror_duplicate_edge_list(const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list);


void write_edgelist(const std::string &fname, const Dataframe_t<std::pair<uint32_t, uint32_t>, 1> &edges);
void read_edgelist(const std::string& fname, std::vector<std::pair<uint32_t, uint32_t>>& edges);
std::vector<std::pair<uint32_t, uint32_t>> read_edgelist(const std::string& fname);
std::vector<uint32_t> read_vec(const std::string& fpath, size_t N);

void write_vector(const std::string& fname, const std::vector<uint32_t>& vec);
// std::vector<float> project_on_connection(const std::vector<uint32_t>& ecm, float value, uint32_t connection_index);
std::vector<float> project_on_connection(const std::vector<uint32_t>& ecm, const std::vector<float>& values, uint32_t connection_index);

#endif
