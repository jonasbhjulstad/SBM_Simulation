#include <SBM_Simulation/SBM_Simulation.hpp>
#include <SBM_Graph/Graph.hpp>
#include <tom/tom_config.hpp>
int main()
{
    using namespace SBM_Database;
    using namespace SBM_Simulation;
    auto DB = tom_config::default_db_connection();
    // project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;

    sycl::queue q(sycl::cpu_selector_v);
    uint32_t seed = 283;
    auto N_communities = 2;
    auto p_out_id = 0;
    auto p = SBM_Database::sim_param_read(p_out_id);
    Simulation_t sim(q, p, "Community");
    sim.run();
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();

    return 0;
}