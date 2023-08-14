
//enable tbb debug
#define TBB_USE_DEBUG 1

#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/Simulation.hpp>
#include <Sycl_Graph/Profiling.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>

int main()
{

    uint32_t N_communities = 2;
    uint32_t N_pop = 100;
    Sim_Param p;
    p.N_communities = N_communities;
    p.N_pop = N_pop;
    p.p_in = 1.0f;
    p.p_out = 0.5f;
    p.p_R0 = 0.0f;
    p.p_I0 = 0.1f;
    p.p_R = 1e-1f;
    p.Nt = 2;
    sycl::queue q(sycl::cpu_selector_v, {cl::sycl::property::queue::enable_profiling{}});
    auto device_info = get_device_info(q);
    device_info.print();

    // p.N_sims = device_info.max_compute_units*device_info.max_work_group_size;
    p.N_sims = 8;
    p.output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_0/";
    uint32_t seed = 238;

    float p_I_min = 1e-6f;
    float p_I_max = 1e-4f;

    auto [edge_lists, vertex_lists] = generate_planted_SBM_edges(N_pop, N_communities, p.p_in, p.p_out, seed);

    auto vcm = create_vcm(vertex_lists);

    auto edge_list_flat = merge_vectors(edge_lists);

    auto sim = make_SIR_simulation(q, p, edge_list_flat, vcm, p_I_min, p_I_max);
    sim.run();
    return 0;
}
