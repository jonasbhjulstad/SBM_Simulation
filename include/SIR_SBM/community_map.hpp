#pragma once
#include <SIR_SBM/graph.hpp>
std::vector<uint32_t> edge_connection_map(const SBM_Graph& G)
{
    std::vector<uint32_t> ecm;
    ecm.reserve(G.N_edges());
    for(int i = 0; i < G.edges.size(); i++)
    {
        ecm.insert(ecm.end(), G.edges[i].size(), i);
    }
    return ecm;
}

std::vector<uint32_t> vertex_partition_map(const SBM_Graph& G)
{
    std::vector<uint32_t> vpm;
    vpm.reserve(G.N_vertices());
    for(const auto& v: G.vertices)
    {
        vpm.insert(vpm.end(), v.size(), v[0]);
    }
    return vpm;
}