#include <SBM_Simulation/Graph/Complete_Graph.hpp>
#include <SBM_Simulation/Simulation/Sim_Types.hpp>
#include <SBM_Simulation/Utils/String_Manipulation.hpp>
#include <algorithm>
#include <numeric>
std::vector<uint32_t> Sim_Param::N_connections() const
{
    std::vector<uint32_t> result(N_communities.size());
    std::transform(N_communities.begin(), N_communities.end(), result.begin(), [](uint32_t N)
                   { return complete_graph_max_edges(N, true, true); });
    return result;
}

std::size_t Sim_Param::N_communities_max() const { return std::max_element(N_communities.begin(), N_communities.end())[0]; }
std::size_t Sim_Param::N_connections_max() const
{
    return complete_graph_max_edges(N_communities_max(), true, true);
}

std::size_t Sim_Param::N_connections_tot() const
{
    auto N_con = N_connections();
    return std::accumulate(N_con.begin(), N_con.end(), 0);
}

std::vector<std::string> Sim_Param::string_param_names()
{
    return std::vector<std::string>({"N_pop", "p_in", "p_out", "N_graphs", "N_sims", "Nt", "Nt_alloc", "seed", "p_I_min", "p_I_max", "p_R", "p_I0", "p_R0", "N_communities"});
}


std::vector<std::string> Sim_Param::string_param_types()
{

    return std::vector<std::string>({"INTEGER", "real", "real", "INTEGER", "INTEGER", "INTEGER", "INTEGER", "INTEGER", "real", "real", "real", "real", "real", "INTEGER[]"});

}

std::string Sim_Param::string_values() const
{
    return tuple_to_string(std::make_tuple(N_pop, p_in, p_out, N_graphs, N_sims, Nt, Nt_alloc, seed, p_I_min, p_I_max, p_out_idx, p_R, p_I0, p_R0)) + ", " + vector_to_string(N_communities);
}
