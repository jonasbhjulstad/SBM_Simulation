
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Sim_Types.hpp>
#include <SBM_Database/Timeseries.hpp>
#include <SBM_Database/SBM_Database.hpp>
#include <Dataframe/Dataframe.hpp>
namespace SBM_Database
{

void construct_simulation_tables(soci::session &sql, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt)
{
    using namespace Dataframe;
    using namespace SBM_Database;

    std::vector<std::string> timeseries_indices = {"p_out", "graph", "sim", "t"};
    table_create(sql, "community_state", {"p_out", "graph", "sim", "t", "community"}, {"state"}, {"INTEGER[3]"});
    table_create(sql, "connection_events", {"p_out", "graph", "sim", "t", "connection"}, {"event"}, {"INTEGER"});
    table_create(sql, "infection_events", {"p_out", "graph", "sim", "t", "connection"}, {"event"}, {"INTEGER"});
    table_create(sql, "p_Is", {"p_out", "graph", "sim", "t", "connection"}, {"p_I"}, {"REAL"});

    std::vector<std::string> graph_indices({"p_out", "graph", "edge"});
    std::array<std::string, 1> ccm_data_names({"data"});
    std::array<std::string, 1> ccm_data_types({"INTEGER[3]"});

    std::vector<std::string> ccm_indices({"p_out", "graph", "edge"});
    table_create(sql, "connection_community_map", ccm_indices, ccm_data_names, ccm_data_types);

    std::array<std::string, 3> edgelists_data_name({"from", "to", "weight"});
    std::array<std::string, 3> edgelists_data_types({"INTEGER", "INTEGER", "INTEGER"});
    table_create(sql, "edgelists", graph_indices, edgelists_data_name, edgelists_data_types);
    std::array<std::string, 1> vcm_data_name({"community"});
    std::array<std::string, 1> vcm_data_types({"INTEGER"});
    std::vector<std::string> vcm_indices({"p_out", "graph", "vertex"});

    table_create(sql, "vertex_community_map", vcm_indices, vcm_data_name, vcm_data_types);

    table_create<Sim_Param::N_parameters>(sql, "simulation_parameters", {}, Sim_Param::string_param_names(), Sim_Param::string_param_types());
}


void drop_simulation_tables(soci::session &sql)
{
    drop_table(sql, "community_state");
    drop_table(sql, "connection_events");
    drop_table(sql, "infection_events");
    drop_table(sql, "p_Is");
    drop_table(sql, "connection_community_map");
    drop_table(sql, "edgelists");
    drop_table(sql, "vertex_community_map");
    drop_table(sql, "simulation_parameters");
}

} // namespace SBM_Database
