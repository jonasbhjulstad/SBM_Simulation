
//enable tbb debug
#define TBB_USE_DEBUG 1

#include <Sycl_Graph/Utils/path_config.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <Sycl_Graph/Graph.hpp>
template <typename T>
std::vector<T> merge_vectors(const std::vector<std::vector<T>> &vectors)
{
    std::vector<T> merged;
    uint32_t size = 0;
    for(int i = 0; i < vectors.size(); i++)
    {
        size += vectors[i].size();
    }
    merged.reserve(size);
    for (auto &v : vectors)
    {
        merged.insert(merged.end(), v.begin(), v.end());
    }
    return merged;
}

int main(int argc, char **argv)
{
    //parse argv which is N_sims, N_communities, N_pop
    if (argc != 5)
    {
        std::cout << "Usage: ./parallel_SIR_Network N_compute N_work_group_size N_communities N_pop" << std::endl;
        return 1;
    }
    uint32_t N_wg = std::stoi(argv[1]);
    uint32_t N_compute = std::stoi(argv[2]);
    uint32_t N_communities = std::stoi(argv[3]);
    uint32_t N_pop = std::stoi(argv[4]);


    sycl::queue q(sycl::gpu_selector_v);//, {cl::sycl::property::queue::enable_profiling{}});


    Sim_Param p(q);
    p.N_communities = N_communities;
    p.N_pop = N_pop;
    p.p_in = 1.0f;
    p.p_out = 0.5f;
    p.p_R0 = 0.0f;
    p.p_I0 = 0.1f;
    p.p_R = 1e-1f;
    p.Nt = 56;
    p.Nt_alloc = 3;
    p.p_I_max = 1e-2f;
    p.p_I_min = 1e-4f;


    p.compute_range = sycl::range<1>(N_compute);
    p.wg_range = sycl::range<1>(N_wg);
    p.N_sims = p.compute_range[0]*p.wg_range[0];



    auto device_info = get_device_info(q);
    device_info.print();


    // p.N_sims = device_info.max_compute_units*device_info.max_work_group_size;
    p.output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Graph_0/";
    uint32_t seed = 238;

    auto [edge_lists, vertex_lists] = generate_planted_SBM_edges(N_pop, N_communities, p.p_in, p.p_out, seed);

    auto vcm = create_vcm(vertex_lists);

    auto edge_list_flat = merge_vectors(edge_lists);
    p.print();


    auto buffers = Sim_Buffers::make(q, p, edge_list_flat, vcm);
    std::cout << "Sim_Buffers size:\t" << buffers.byte_size() << "byte" << std::endl;
    q.wait();
    run(q, p, buffers);

    return 0;
}
