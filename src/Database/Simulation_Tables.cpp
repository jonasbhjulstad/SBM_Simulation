#include <Sycl_Graph/Database/Simulation_Tables.hpp>
#include <Sycl_Graph/Database/Dataframe.hpp>
void construct_simulation_tables(pqxx::connection &con, uint32_t Np, uint32_t Ng, uint32_t Ns, uint32_t Nt)
{
    create_timeseries_table(con, Np, Ng, Ns, Nt, "community_state");
    create_timeseries_table(con, Np, Ng, Ns, Nt, "connection_events");
    create_timeseries_table(con, Np, Ng, Ns, Nt, "infection_events");
    create_timeseries_table(con, Np, Ng, Ns, Nt, "p_Is");
    std::vector<std::string> ccm_indices({"p_out", "graph"});
    std::vector<std::string> ccm_data_names({"edge", "weight"});
    std::vector<std::string> ccm_data_types({"INTEGER[2]", "real"});

    create_table(con, "connection_community_map", ccm_indices, ccm_data_names, ccm_data_types);

}
