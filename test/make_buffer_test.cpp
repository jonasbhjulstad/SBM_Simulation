
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Graph/Community_Mappings.hpp>
#include <Sycl_Graph/Simulation/Simulation.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <CL/sycl.hpp>
#include <chrono>
#include <Sycl_Graph/Database/Simulation_Tables.hpp>


Sim_Param create_sim_param(uint32_t N_communities)
{

    Sim_Param p;
    p.N_pop = 100;
    p.N_graphs = 2;
    p.N_communities = std::vector<uint32_t>(p.N_graphs,N_communities);
    p.p_in = 0.5f;
    p.p_out = 0.1f;
    p.N_sims = 2;
    p.Nt = 56;
    p.Nt_alloc = 20;
    p.seed = 234;
    p.p_I_min = 0.1f;
    p.p_I_max = 0.2f;
    p.p_out_idx = 0;
    p.p_R = 0.1f;
    p.p_I0 = 0.1f;
    p.p_R0 = 0.0f;
    return p;
}
int main()
{


    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";

    sycl::queue q(sycl::cpu_selector_v);
    auto N_communities = 10;
    auto p = create_sim_param(N_communities);

    auto con = pqxx::connection("dbname=postgres user=postgres");
    drop_simulation_tables(con);
    construct_simulation_tables(con, 1, p.N_graphs, p.N_sims, p.Nt+1);


    auto [edge_lists, vertex_list] = generate_N_SBM_graphs(p.N_pop, N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);

    auto vcms = std::vector<std::vector<uint32_t>>(p.N_graphs, create_vcm(vertex_list[0]));

    auto p_Is = generate_duplicated_p_Is(p.Nt, p.N_sims, p.N_connections_tot(), p.p_I_min, p.p_I_max, p.seed);

    Simulation_t sim(q, con, p, edge_lists, vcms);

    return 0;
}
