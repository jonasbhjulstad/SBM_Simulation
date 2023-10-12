#include <Sycl_Graph/Database/Simulation_Tables.hpp>
#include <Sycl_Graph/Database/Dataframe.hpp>
#include <Sycl_Graph/Database/Table.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Utils/String_Manipulation.hpp>

void construct_sim_param_table(pqxx::connection& con, uint32_t Np)
{
    std::vector<std::string> param_indices({"p_out"});
    std::vector<std::string> param_data_names({"N_pop", "N_communities", "p_in", "p_out", "N_graphs", "N_sims", "Nt", "Nt_alloc", "seed"});
}

void construct_simulation_tables(pqxx::connection &con, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt)
{
    create_timeseries_table(con, Np, Ng, Ns, Nt, "community_state");
    create_timeseries_table(con, Np, Ng, Ns, Nt, "connection_events");
    create_timeseries_table(con, Np, Ng, Ns, Nt, "infection_events");
    create_timeseries_table(con, Np, Ng, Ns, Nt, "p_Is");
    std::vector<std::string> graph_indices({"p_out", "graph", "edge"});
    std::vector<std::string> ccm_data_names({"from", "to", "weight"});
    std::vector<std::string> ccm_data_types({"INTEGER", "INTEGER", "real"});

    std::vector<std::string> ccm_indices({"p_out", "graph", "edge"});
    create_table(con, "connection_community_map", ccm_indices, ccm_data_names, ccm_data_types);

    std::vector<std::string> edgelists_data_name({"from", "to"});
    std::vector<std::string> edgelists_data_types({"INTEGER", "INTEGER"});
    create_table(con, "edgelists", graph_indices, edgelists_data_name, edgelists_data_types);
    std::vector<std::string> vcm_data_name({"community"});
    std::vector<std::string> vcm_data_types({"INTEGER"});
    create_table(con, "vcms", graph_indices, vcm_data_name, vcm_data_types);
    create_table(con, "vertex_community_map", {"p_out", "graph", "vertex"}, {"community"}, {"INTEGER"});
    create_table(con, "simulation_parameters", {"p_out_id"}, Sim_Param::string_param_names(), Sim_Param::string_param_types());
}

void sim_param_write(pqxx::connection& con, const Sim_Param& p)
{
    auto work = pqxx::work(con);
    std::string insert_str = "INSERT INTO simulation_parameters (" + merge_strings(p.string_param_names(), ", ") + ") VALUES (";
    std::string command = insert_str + p.string_values() + ");";
    work.exec(command);
    work.commit();
}

void drop_simulation_tables(pqxx::connection &con)
{
    drop_table(con, "community_state");
    drop_table(con, "connection_events");
    drop_table(con, "infection_events");
    drop_table(con, "p_Is");
    drop_table(con, "connection_community_map");
    drop_table(con, "edgelists");
    drop_table(con, "vcms");
    drop_table(con, "vertex_community_map");
    drop_table(con, "simulation_parameters");
}
