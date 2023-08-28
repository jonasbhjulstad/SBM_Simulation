#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <functional>
#include <execution>



std::size_t Sim_Buffers::byte_size() const { return b_size; }

std::vector<uint32_t> join_vectors(const std::vector<std::vector<uint32_t>>& vs)
{
    std::vector<uint32_t> result;
    for (auto& v : vs)
    {
        result.insert(result.end(), v.begin(), v.end());
    }
    return result;
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> separate_flatten_edge_lists(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_lists)
{
    std::vector<uint32_t> edge_counts_init(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), edge_counts_init.begin(), [](auto &e)
                   { return e.size(); });
    uint32_t N_tot_edges = std::accumulate(edge_counts_init.begin(), edge_counts_init.end(), 0);

    std::vector<uint32_t> edge_from_init(N_tot_edges);
    std::vector<uint32_t> edge_to_init(N_tot_edges);
    std::vector<std::vector<uint32_t>> edge_froms(edge_lists.size());
    std::vector<std::vector<uint32_t>> edge_tos(edge_lists.size());
    std::transform(edge_lists.begin(), edge_lists.end(), edge_froms.begin(), [](auto &e)
                   { std::vector<uint32_t> result(e.size());
                       std::transform(e.begin(), e.end(), result.begin(), [](auto &e)
                                      { return e.first; });
                       return result; });
    std::transform(edge_lists.begin(), edge_lists.end(), edge_tos.begin(), [](auto &e)
                     { std::vector<uint32_t> result(e.size());
                          std::transform(e.begin(), e.end(), result.begin(), [](auto &e)
                                          { return e.second; });
                          return result; });
    //insert into edge_from_init
    std::size_t offset = 0;
    for (auto &e : edge_froms)
    {
        std::copy(e.begin(), e.end(), edge_from_init.begin() + offset);
        offset += e.size();
    }
    //insert into edge_to_init
    offset = 0;
    for (auto &e : edge_tos)
    {
        std::copy(e.begin(), e.end(), edge_to_init.begin() + offset);
        offset += e.size();
    }
    return std::make_tuple(edge_from_init, edge_to_init, edge_counts_init);
}


Sim_Buffers Sim_Buffers::make(sycl::queue &q, const Sim_Param &p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcms, std::vector<float> p_Is_init)
{


    std::vector<std::vector<uint32_t>> ecms(edge_list.size());
    std::transform(std::execution::par_unseq, edge_list.begin(), edge_list.end(), vcms.begin(), ecms.begin(), [&vcms](const auto& e_list, const auto& vcm)
    {
        return ecm_from_vcm(e_list, vcm);
    });
    auto ecm_init = join_vectors(ecms);
    uint32_t N_connections = std::max_element(ecm_init.begin(), ecm_init.end())[0] + 1;
    if (p_Is_init.size() == 0)
    {
        p_Is_init = generate_floats(p.Nt * p.N_sims * N_connections, p.p_I_min, p.p_I_max, p.seed);
    }
    uint32_t N_vertices = vcms[0].size();
    auto vcm_init = join_vectors(vcms);

    auto e_data = separate_flatten_edge_lists(edge_list);
    std::vector<uint32_t> edge_from_init = std::get<0>(e_data);
    std::vector<uint32_t> edge_to_init = std::get<1>(e_data);
    std::vector<uint32_t> edge_counts_init = std::get<2>(e_data);
    std::vector<uint32_t> edge_offsets_init(edge_counts_init.size());
    std::partial_sum(edge_counts_init.begin(), edge_counts_init.end()-1, edge_offsets_init.begin() + 1);
    uint32_t N_tot_edges = std::accumulate(edge_counts_init.begin(), edge_counts_init.end(), 0);


    std::vector<State_t> community_state_init(p.N_communities * p.N_sims * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(p.N_sims * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> event_init(p.N_sims * N_connections * p.Nt_alloc, 0);
    auto seeds = generate_seeds(p.N_sims, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(p.N_sims);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
                   { return Static_RNG::default_rng(seed); });

    std::vector<std::size_t> sizes = {p.Nt*p.N_sims*N_connections*sizeof(float),
    edge_from_init.size()*sizeof(uint32_t),
    edge_to_init.size()*sizeof(uint32_t),
    ecm_init.size()*sizeof(uint32_t),
    vcm_init.size()*sizeof(uint32_t),
    (p.Nt_alloc + 1)*p.N_sims*p.N_communities*sizeof(State_t),
    p.Nt_alloc*p.N_sims*N_connections*sizeof(uint32_t),
    p.Nt_alloc*p.N_sims*N_connections*sizeof(uint32_t),
    rng_init.size()*sizeof(Static_RNG::default_rng)};

    assert(p.global_mem_size > std::accumulate(sizes.begin(), sizes.end(), 0) && "Not enough global memory to allocate all buffers");

    Sim_Buffers b;
    b.p_Is = sycl::malloc_device<float>(p_Is_init.size(), q);
    b.edge_from = sycl::malloc_device<uint32_t>(edge_from_init.size(), q);
    b.edge_to = sycl::malloc_device<uint32_t>(edge_to_init.size(), q);
    b.ecm = sycl::malloc_device<uint32_t>(ecm_init.size(), q);
    b.vcm = sycl::malloc_device<uint32_t>(vcm_init.size(), q);
    b.edge_counts = sycl::malloc_device<uint32_t>(edge_counts_init.size(), q);
    b.edge_offsets = sycl::malloc_device<uint32_t>(edge_offsets_init.size(), q);
    b.community_state = sycl::malloc_device<State_t>(community_state_init.size(), q);
    b.vertex_state = sycl::malloc_device<SIR_State>(traj_init.size(), q);
    b.events_from = sycl::malloc_device<uint32_t>(event_init.size(), q);
    b.events_to = sycl::malloc_device<uint32_t>(event_init.size(), q);
    b.rngs = sycl::malloc_device<Static_RNG::default_rng>(rng_init.size(), q);

    std::vector<sycl::event> alloc_events(11);
    alloc_events[0] = initialize_device_buffer<float, 3>(q, p_Is_init, b.p_Is);
    alloc_events[1] = initialize_device_buffer<uint32_t, 1>(q, edge_from_init, b.edge_from);
    alloc_events[2] = initialize_device_buffer<uint32_t, 1>(q, edge_to_init, b.edge_to);
    alloc_events[3] = initialize_device_buffer<uint32_t, 1>(q, ecm_init, b.ecm);
    alloc_events[4] = initialize_device_buffer<uint32_t, 2>(q, vcm_init, b.vcm);
    alloc_events[5] = initialize_device_buffer<uint32_t, 1>(q, edge_counts_init, b.edge_counts);
    alloc_events[5] = initialize_device_buffer<uint32_t, 1>(q, edge_offsets_init, b.edge_offsets);
    alloc_events[6] = initialize_device_buffer<State_t, 3>(q, community_state_init, b.community_state);
    alloc_events[7] = initialize_device_buffer<SIR_State, 3>(q, traj_init, b.vertex_state);
    alloc_events[8] = initialize_device_buffer<uint32_t, 3>(q, event_init, b.events_from);
    alloc_events[9] = initialize_device_buffer<uint32_t, 3>(q, event_init, b.events_to);
    alloc_events[10] = initialize_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, b.rngs);

    b.b_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto ccm = complete_ccm(p.N_graphs);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccms(p.N_graphs, ccm);
    std::vector<std::vector<uint32_t>> ccm_weights(p.N_graphs);
    std::transform(std::execution::par_unseq, ecms.begin(), ecms.end(), ccm_weights.begin(), ccm_weights_from_ecm);

    b.N_edges = ecm_init.size();
    b.N_connections = ccm.size();
    // get global memory size
    auto global_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
    // get max alloc size
    //  auto max_alloc_size = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    assert(b.byte_size() < global_mem_size && "Not enough global memory to allocate all buffers");

    return b;
}



//
