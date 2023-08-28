
//enable tbb debug
#define TBB_USE_DEBUG 1

#include <Sycl_Graph/Utils/path_config.hpp>
#include <Sycl_Graph/Simulation/Sim_Routines.hpp>
#include <Sycl_Graph/Utils/Profiling.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <execution>
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

auto generate_graphs(const std::vector<Sim_Param>& ps, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::vector<uint32_t> seeds(ps.size());
    std::generate(seeds.begin(), seeds.end(), [&gen]()
    {
        return gen();
    });
    std::vector<std::tuple<std::vector<Edge_List_t>, std::vector<Node_List_t>>> graphs(ps.size());
    std::transform(std::execution::par_unseq, ps.begin(), ps.end(), seeds.begin(),  graphs.begin(), [&](const auto& p, auto s)
    {

        return generate_N_SBM_graphs(p.N_pop, p.N_communities, p.p_in, p.p_out, p.seed, p.N_graphs);
    });
    return graphs;
}


std::vector<std::tuple<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>,std::vector<std::vector<uint32_t>>>> sbm_data();


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
    //get all gpus
    std::vector<sycl::queue> qs;
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto &d : devices)
    {
        qs.emplace_back(d);
    }
    uint32_t seed = 283;
    std::vector<Sim_Param> ps;
    std::mt19937 gen(seed);


    std::transform(qs.begin(), qs.end(), std::back_inserter(ps), [&, n = 0](sycl::queue& q)mutable {auto p = Sim_Param(q);
    p.N_communities = N_communities;
    p.N_pop = N_pop;
    p.p_in = 1.0f;
    p.p_out = 0.0f;
    p.p_R0 = 0.0f;
    p.p_I0 = 0.1f;
    p.p_R = 1e-1f;
    p.Nt = 20;
    p.Nt_alloc = 3;
    p.p_I_max = 1e-3f;
    p.p_I_min = 1e-5f;
    p.seed = gen();
    p.compute_range = sycl::range<1>(N_compute);
    p.wg_range = sycl::range<1>(N_wg);
    p.N_sims = p.compute_range[0]*p.wg_range[0];
    p.file_idx_offset = N_compute*n;
    p.N_graphs = 2;
    // p.output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "/SIR_sim/Batch_0/";
    n++;
    return p;
    });
    std::string output_dir = std::string(Sycl_Graph::SYCL_GRAPH_DATA_DIR) + "SIR_sim/Batch_0";
    auto graphs = generate_graphs(ps, seed);

    std::vector<Sim_Buffers> buffers;
    std::vector<std::tuple<std::decay_t<decltype(graphs[0])>, sycl::queue, Sim_Param>> buffer_params;
    for(int i = 0; i < graphs.size(); i++)
    {
        buffer_params.push_back(std::make_tuple(graphs[i], qs[i], ps[i]));
    }

    std::transform(buffer_params.begin(), buffer_params.end(), std::back_inserter(buffers), [](auto &t)
    {
        auto graph = std::get<0>(t);
        auto q = std::get<1>(t);
        auto p = std::get<2>(t);
        auto edge_lists = std::get<0>(graph);
        //flatten edge_lists
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> edge_lists_flat(edge_lists.size());
        std::transform(std::execution::par_unseq, edge_lists.begin(), edge_lists.end(), edge_lists_flat.begin(), [](auto& el)
        {
            return merge_vectors(el);
        });

        auto vcm = create_vcm(std::get<1>(graph)[0]);

        auto vcms = std::vector<decltype(vcm)>(p.N_sims, vcm);
        return Sim_Buffers::make(q, p, edge_lists_flat, vcms, {});
    });
    std::for_each(qs.begin(), qs.end(), [](auto& q){q.wait();});
    // std::cout << "Sim_Buffers size:\t" << buffers.byte_size() << "byte" << std::endl;

    std::vector<uint32_t> result(qs.size());
    // std::transform(std::execution::par_unseq, buffer_params.begin(), buffer_params.end(), buffers.begin(), result.begin(), [](auto& bp, auto& b){
    //     run(std::get<1>(bp), std::get<2>(bp), b);
    //     return 0;
    // });
    run(std::get<1>(buffer_params[0]), std::get<2>(buffer_params[0]), buffers[0]);


    return 0;
}
