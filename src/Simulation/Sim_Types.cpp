#include <Sycl_Graph/Graph/Complete_Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <algorithm>
#include <numeric>
std::vector<uint32_t> Sim_Param::N_connections() const
{
    std::vector<uint32_t> result(N_graphs);
    std::transform(N_communities.begin(), N_communities.end(), result.begin(), [](uint32_t N)
                   { return complete_graph_max_edges(N, false, true); });
}

std::size_t N_communities_max() const { return std::max_element(N_communities.begin(), N_communities.end())[0]; }
std::size_t N_connections_max() const
{
    return complete_graph_max_edges(N_communities_max(), false, true);
}


std::size_t N_connections_tot() const
{
    auto N_con = N_connections();
    return std::accumulate(N_con.begin(), N_con.end(), 0);
}
