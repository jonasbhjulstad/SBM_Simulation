
//enable tbb debug
#define TBB_USE_DEBUG 1

#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/Simulation.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Buffer_Utils.hpp>

int main()
{
    std::string output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_0/";

    uint32_t N_clusters = 10;
    uint32_t N_pop = 100;
    Sim_Param p;
    p.N_clusters = N_clusters;
    p.N_pop = N_pop;
    p.p_in = 1.0f;
    p.p_out = 0.5f;
    p.p_R0 = 0.0f;
    p.p_I0 = 0.1f;
    p.p_R = 1e-1f;
    p.sim_idx = 0;
    p.Nt = 50;
    uint32_t seed = 238;
    uint32_t N_sims = 100;

    float p_I_min = 1e-3f;
    float p_I_max = 1e-1f;

    auto [edge_lists, vertex_lists] = generate_planted_SBM_edges(N_pop, N_clusters, p.p_in, p.p_out, seed);

    auto vcm = create_vcm(vertex_lists);

    auto edge_list_flat = merge_vectors(edge_lists);
    sycl::queue q(sycl::gpu_selector_v);
    excite_simulate(q, p, vcm, edge_list_flat, p_I_min, p_I_max, output_dir, N_sims, seed);

    return 0;
}
