#include <SBM_Simulation/SBM_Simulation.hpp>
#include <SBM_Graph/Graph.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>
#include <tom/tom_config.hpp>
int main()
{
    using namespace SBM_Database;
    using namespace SBM_Simulation;
    auto DB = tom_config::default_db_connection_postgres();
    // project root
    std::string root_dir = "/home/man/Documents/ER_Bernoulli_Robust_MPC/";
    std::string output_dir = root_dir + "data/";
    std::chrono::high_resolution_clock::time_point t1, t2;
    sycl::queue q(sycl::gpu_selector_v);
    auto info = Buffer_Routines::get_device_info(q);
    info.print();
    uint32_t seed = 283;
    auto p_out_id = 0;
    auto graph_id = 0;
    auto p = SBM_Database::sim_param_read(p_out_id, graph_id);

    Sim_Buffers b(q, p, "Community");
    q.wait();
    t2 = std::chrono::high_resolution_clock::now();

    return 0;
}