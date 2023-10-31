
#include <CL/sycl.hpp>
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <SBM_Graph/Community_Mappings.hpp>
#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/Types/Sim_Types.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <Sycl_Buffer_Routines/Buffer_Routines.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <chrono>

using namespace SBM_Simulation;




int main()
{
    using namespace SBM_Database;

    // project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;
    auto N_communities = 2;
    auto N_pop = 10;
    auto N_graphs = 2;
    auto N_communities = 2;
    auto p_in = 0.5f;
    auto p_out = 0.1f;
    auto N_sims = 2;
    auto Nt = 56;
    auto Nt_alloc = 20;
    auto seed = 234;
    auto p_I_min = 0.1f;
    auto p_I_max = 0.2f;
    auto p_out_id = 0;
    auto p_R = 0.1f;
    auto p_I0 = 0.1f;
    auto p_R0 = 0.0f;

    Sim_Param p{N_pop, N_communities, p_in, p_out, N_graphs, N_sims, Nt, Nt_alloc, seed, p_I_min, p_I_max, p_out_id, p_R, p_I0, p_R0};

    generate_SBM_to_db(p);

    return 0;
}
