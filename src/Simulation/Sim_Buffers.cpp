#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <functional>
#include <execution>
// #include <Sycl_Graph/Utils/Buffer_Utils.hpp>
// std::vector<float> generate_floats(uint32_t N, float min, float max, uint32_t seed)
// {
//     std::mt19937_64 rng(seed);
//     std::uniform_real_distribution<float> dist(min, max);
//     std::vector<float> result(N);
//     std::generate(result.begin(), result.end(), [&]()
//                   { return dist(rng); });
//     return result;
// }

Sim_Buffers::Sim_Buffers(cl::sycl::buffer<Static_RNG::default_rng> &rngs,
                cl::sycl::buffer<SIR_State, 3> &vertex_state,
                cl::sycl::buffer<uint32_t, 3> &events_from,
                cl::sycl::buffer<uint32_t, 3> &events_to,
                cl::sycl::buffer<float, 3> &p_Is,
                cl::sycl::buffer<uint32_t> &edge_from,
                cl::sycl::buffer<uint32_t> &edge_to,
                cl::sycl::buffer<uint32_t> &ecm,
                cl::sycl::buffer<uint32_t, 2> &vcm,
                cl::sycl::buffer<uint32_t> &edge_counts,
                cl::sycl::buffer<uint32_t> & edge_offsets,
                cl::sycl::buffer<State_t, 3> &community_state,
                const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &ccm,
                const std::vector<std::vector<uint32_t>> &ccm_weights) : rngs(std::move(rngs)),
                                                                     vertex_state(std::move(vertex_state)),
                                                                     events_from(std::move(events_from)),
                                                                     events_to(std::move(events_to)),
                                                                     p_Is(std::move(p_Is)),
                                                                     edge_from(std::move(edge_from)),
                                                                     edge_to(std::move(edge_to)),
                                                                     ecm(std::move(ecm)),
                                                                     vcm(std::move(vcm)),
                                                                     edge_counts(std::move(edge_counts)),
                                                                     edge_offsets(std::move(edge_offsets)),
                                                                     community_state(std::move(community_state)),
                                                                     ccm(ccm),
                                                                     ccm_weights(ccm_weights) {}

std::size_t Sim_Buffers::byte_size() const { return rngs.byte_size() + vertex_state.byte_size() + events_from.byte_size() + events_to.byte_size() + p_Is.byte_size() + edge_from.byte_size() + edge_to.byte_size() + ecm.byte_size() + vcm.byte_size() + community_state.byte_size(); }

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


Sim_Buffers Sim_Buffers::make(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<uint32_t>>& ecms, std::vector<float> p_Is_init, uint32_t N_connections)
{

    if ((p.global_mem_size == 0) || p.local_mem_size == 0)
    {
        auto device = q.get_device();
        p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }
    auto N_sims_tot = p.compute_range[0];
    auto ecm_init = join_vectors(ecms);
    // uint32_t N_connections = std::max_element(ecm_init.begin(), ecm_init.end())[0] + 1;

    uint32_t N_vertices = vcms[0].size();
    auto vcm_init = join_vectors(vcms);

    auto e_data = separate_flatten_edge_lists(edge_list);
    std::vector<uint32_t> edge_from_init = std::get<0>(e_data);
    std::vector<uint32_t> edge_to_init = std::get<1>(e_data);
    std::vector<uint32_t> edge_counts_init = std::get<2>(e_data);
    std::vector<uint32_t> edge_offsets_init(edge_counts_init.size());
    std::partial_sum(edge_counts_init.begin(), edge_counts_init.end()-1, edge_offsets_init.begin() + 1);
    uint32_t N_tot_edges = std::accumulate(edge_counts_init.begin(), edge_counts_init.end(), 0);


    std::vector<State_t> community_state_init(p.N_communities * N_sims_tot * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(N_sims_tot * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> event_init_from(N_sims_tot * N_connections * p.Nt_alloc, 0);
    std::vector<uint32_t> event_init_to(N_sims_tot * N_connections * p.Nt_alloc, 0);
    auto seeds = generate_seeds(N_sims_tot, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(N_sims_tot);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
                   { return Static_RNG::default_rng(seed); });

    std::vector<std::size_t> sizes = {p.Nt*N_sims_tot*N_connections*sizeof(float),
    edge_from_init.size()*sizeof(uint32_t),
    edge_to_init.size()*sizeof(uint32_t),
    ecm_init.size()*sizeof(uint32_t),
    vcm_init.size()*sizeof(uint32_t),
    (p.Nt_alloc + 1)*N_sims_tot*p.N_communities*sizeof(State_t),
    p.Nt_alloc*N_sims_tot*N_connections*sizeof(uint32_t),
    p.Nt_alloc*N_sims_tot*N_connections*sizeof(uint32_t),
    rng_init.size()*sizeof(Static_RNG::default_rng)};

    assert(p.global_mem_size > std::accumulate(sizes.begin(), sizes.end(), 0) && "Not enough global memory to allocate all buffers");



    auto p_Is = sycl::buffer<float, 3>(sycl::range<3>(p.Nt, N_sims_tot, N_connections));
    auto edge_from = sycl::buffer<uint32_t>((sycl::range<1>(N_tot_edges)));
    auto edge_to = sycl::buffer<uint32_t>((sycl::range<1>(N_tot_edges)));
    auto ecm = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_tot_edges)));
    auto vcm = sycl::buffer<uint32_t, 2>(sycl::range<2>(p.N_graphs, vcms[0].size()));
    auto edge_counts = sycl::buffer<uint32_t>((sycl::range<1>(p.compute_range[0])));
    auto edge_offsets = sycl::buffer<uint32_t>((sycl::range<1>(p.compute_range[0])));
    auto community_state = sycl::buffer<State_t, 3>(sycl::range<3>(p.Nt_alloc + 1, N_sims_tot, p.N_communities));
    auto trajectory = sycl::buffer<SIR_State, 3>(sycl::range<3>(p.Nt_alloc + 1, N_sims_tot, N_vertices));
    auto events_from = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_connections));
    auto events_to = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_connections));
    auto rngs = sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(rng_init.size()));

    std::vector<sycl::event> alloc_events(13);

    if (p_Is_init.size() == 0)
    {
        p_Is_init = generate_floats(p.Nt * N_sims_tot * N_connections, p.p_I_min, p.p_I_max, 8, p.seed);
    }

    alloc_events[1] = initialize_device_buffer<float, 3>(q, p_Is_init, p_Is);
    alloc_events[2] = initialize_device_buffer<uint32_t, 1>(q, edge_from_init, edge_from);
    alloc_events[3] = initialize_device_buffer<uint32_t, 1>(q, edge_to_init, edge_to);
    alloc_events[4] = initialize_device_buffer<uint32_t, 1>(q, ecm_init, ecm);
    alloc_events[5] = initialize_device_buffer<uint32_t, 2>(q, vcm_init, vcm);
    alloc_events[6] = initialize_device_buffer<uint32_t, 1>(q, edge_counts_init, edge_counts);
    alloc_events[7] = initialize_device_buffer<uint32_t, 1>(q, edge_offsets_init, edge_offsets);
    alloc_events[8] = initialize_device_buffer<State_t, 3>(q, community_state_init, community_state);
    alloc_events[9] = initialize_device_buffer<SIR_State, 3>(q, traj_init, trajectory);
    alloc_events[10] = initialize_device_buffer<uint32_t, 3>(q, event_init_from, events_from);
    alloc_events[11] = initialize_device_buffer<uint32_t, 3>(q, event_init_to, events_to);
    alloc_events[12] = initialize_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, rngs);

    auto ccm = complete_ccm(p.N_communities);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccms(p.N_graphs, ccm);
    std::vector<std::vector<uint32_t>> ccm_weights(p.N_graphs);
    std::transform(std::execution::par_unseq, ecms.begin(), ecms.end(), ccm_weights.begin(), [N_connections](const auto& ecm){return ccm_weights_from_ecm(ecm, N_connections);});



    auto byte_size = rngs.byte_size() + trajectory.byte_size() + events_from.byte_size() + events_to.byte_size() + p_Is.byte_size() + edge_from.byte_size() + edge_to.byte_size() + ecm.byte_size() + vcm.byte_size() + community_state.byte_size();
    // get global memory size
    auto global_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
    // get max alloc size
    //  auto max_alloc_size = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    assert(byte_size < global_mem_size && "Not enough global memory to allocate all buffers");

    return Sim_Buffers(rngs,
                       trajectory,
                       events_from,
                       events_to,
                       p_Is,
                       edge_from,
                       edge_to,
                       ecm,
                       vcm,
                       edge_counts,
                       edge_offsets,
                       community_state,
                       ccms,
                       ccm_weights);
}
