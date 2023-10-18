
// #define _GLIBCXX_USE_CXX11_ABI 0
#include <SBM_Simulation/Graph/Graph.hpp>
#include <SBM_Simulation/Simulation/Sim_Types.hpp>
#include <SBM_Simulation/Graph/Community_Mappings.hpp>
// #include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>

#include <CL/sycl.hpp>
#include <chrono>

// target_link_libraries(Simulation PUBLIC Sim_Buffers SIR_Dynamics Sim_Infection_Sampling String_Manipulation math State_Accumulation)


Sim_Param create_sim_param(uint32_t N_communities)
{
    //     uint32_t N_pop = 100;
    // std::vector<uint32_t> N_communities;
    // float p_in = 1.0f;
    // float p_out = 0.5f;
    // uint32_t N_graphs = 2;
    // uint32_t N_sims = 2;
    // uint32_t Nt = 56;
    // uint32_t Nt_alloc = 20;
    // uint32_t seed = 234;
    // float p_I_min = 0.1f;
    // float p_I_max = 0.2f;
    // uint32_t p_out_idx = 0;
    // float p_R = 0.1f;
    // float p_I0 = 0.1f;
    // float p_R0 = 0.0f;
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


    //project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::gpu_selector_v);
    uint32_t seed = 283;
    auto N_communities = 2;
    auto p = create_sim_param(N_communities);


    auto con = pqxx::connection("dbname=postgres user=postgres");
    SBM_Database::drop_simulation_tables(sql);
    SBM_Database::construct_simulation_tables(sql, 1, p.N_graphs, p.N_sims, p.Nt+1);

    auto [edge_lists, vertex_list] = generate_N_SBM_graphs(p.N_pop, N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);

    //create sim_buffers
    Sim_Buffers b(q, p, sql, edge_lists, vertex_list[0]);

    sycl::range<1> compute_range(p.N_sims_tot());
    sycl::range<1> wg_range(p.N_sims_tot());
    // std::vector<sycl::event> infect(sycl::queue &q,
    //                             const Sim_Param &p,
    //                             Sim_Buffers &b,
    //                             uint32_t t,
    //                             sycl::range<1> compute_range,
    //                             sycl::range<1> wg_range,
    //                             std::vector<sycl::event> &dep_event)
    auto inf_event = infect(q, p, b, 0, compute_range, wg_range, b.construction_events);

    return 0;
}
