
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Graph/Community_Mappings.hpp>
#include <Sycl_Graph/Database/Dataframe.hpp>

Sim_Buffers::Sim_Buffers(sycl::buffer<Static_RNG::default_rng> &rngs,
                         sycl::buffer<SIR_State, 3> &vertex_state,
                         sycl::buffer<uint32_t, 3> &accumulated_events,
                         sycl::buffer<float, 3> &p_Is,
                         sycl::buffer<uint32_t> &edge_from,
                         sycl::buffer<uint32_t> &edge_to,
                         sycl::buffer<uint32_t> &ecm,
                         sycl::buffer<uint32_t, 2> &vcm,
                         sycl::buffer<uint32_t> &edge_counts,
                         sycl::buffer<uint32_t> &edge_offsets,
                         sycl::buffer<uint32_t> &N_connections,
                         sycl::buffer<State_t, 3> &community_state,
                         const Dataframe_t<Edge_t, 2> &ccm) : rngs(std::move(rngs)),
                                                                       vertex_state(std::move(vertex_state)),
                                                                       accumulated_events(std::move(accumulated_events)),
                                                                       p_Is(std::move(p_Is)),
                                                                       edge_from(std::move(edge_from)),
                                                                       edge_to(std::move(edge_to)),
                                                                       ecm(std::move(ecm)),
                                                                       vcm(std::move(vcm)),
                                                                       edge_counts(std::move(edge_counts)),
                                                                       edge_offsets(std::move(edge_offsets)),
                                                                       N_connections(N_connections),
                                                                       community_state(std::move(community_state)),
                                                                       ccm(ccm)
{
}



void validate_buffer_init_sizes(Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<uint32_t>> &ecms, std::vector<float> p_Is_init)
{
    if (ecms.size() != p.N_graphs)
    {
        throw std::runtime_error("ecms.size() != p.N_graphs");
    }
    if (vcms.size() != p.N_graphs)
    {
        throw std::runtime_error("vcms.size() != p.N_graphs");
    }

    if (edge_list.size() != p.N_graphs)
    {
        throw std::runtime_error("edge_list.size() != p.N_graphs");
    }

    if ((p_Is_init.size() != p.N_graphs * p.N_connections_tot() * p.Nt) && (p_Is_init.size() != 0))
    {
        throw std::runtime_error("p_Is_init.size() != p.N_graphs*N_connections*p.Nt");
    }
}

void Sim_Buffers::validate_sizes(const Sim_Param &p) const
{

    auto check_buffer_dims = [](auto &buf, auto N0, auto N1, auto N2)
    {
        return (buf.get_range()[0] == N0) && (buf.get_range()[1] == N1) && (buf.get_range()[2] == N2);
    };
    auto buf_dim_string = [](auto &buf)
    {
        return "(" + std::to_string(buf.get_range()[0]) + ", " + std::to_string(buf.get_range()[1]) + ", " + std::to_string(buf.get_range()[2]) + ")";
    };
    auto dim_string = [](auto N0, auto N1, auto N2)
    {
        return "(" + std::to_string(N0) + ", " + std::to_string(N1) + ", " + std::to_string(N2) + ")";
    };
    auto N_connections_max = p.N_connections_max();
    if_false_throw(check_buffer_dims(p_Is, p.Nt, p.N_graphs * p.N_sims, p.N_connections_max()), "p_Is has wrong dimensions: " + buf_dim_string(p_Is) + " vs " + dim_string(p.Nt, p.N_graphs, N_connections_max));
    if_false_throw(rngs.size() == p.N_sims * p.N_graphs, "rngs has wrong size: " + std::to_string(rngs.size()) + " vs " + std::to_string(p.N_sims * p.N_graphs));
    if_false_throw(check_buffer_dims(accumulated_events, p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max), "events has wrong dimensions: " + buf_dim_string(accumulated_events) + " vs " + dim_string(p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max));
}



Dataframe_t<float, 3> generate_duplicated_p_Is(uint32_t Nt, uint32_t N_sims_tot, const uint32_t N_connections_tot, float p_I_min, float p_I_max, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(p_I_min, p_I_max);
    Dataframe_t<float, 3> p_Is(Nt);
    auto t = 0;
    while (t < Nt)
    {
        Dataframe_t<float, 2> p_Is_t(N_sims_tot, N_connections_tot);
        for (auto &p_I : p_Is_t)
        {
            std::generate(p_I.begin(), p_I.end(), [&]()
                          { return dist(gen); });
        }
        for (int i = 0; i < 7; i++)
        {
            if ((t + i) >= Nt)
                break;
            p_Is[t + i] = p_Is_t;
        }
        t += 7;
    }
    return p_Is;
}

// Sim_Buffers Sim_Buffers::make(sycl::queue &q, Sim_Param p, pqxx::connection &con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list_undirected, const Dataframe_t<uint32_t, 2> &vcms, Dataframe_t<float, 3> p_Is_init)
// {

//     if (p_Is_init.size() == 0)
//         p_Is_init = generate_duplicated_p_Is(p.Nt, p.N_sims_tot(), p.N_connections_tot(), p.p_I_min, p.p_I_max, p.seed);
//     auto edge_list = mirror_duplicate_edge_list(edge_list_undirected);
//     return Sim_Buffers::make_impl(q, p, con, edge_list, vcms, p_Is_init);
// }

// Sim_Buffers Sim_Buffers::make_impl(sycl::queue &q, Sim_Param p, pqxx::connection &con, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcms, Dataframe_t<float, 3> p_Is_init)
// {
//     const auto N_connections_max = p.N_connections_max();
//     const auto N_communities_max = p.N_communities_max();
//     const auto N_sims_tot = p.N_sims_tot();
//     auto N_vertices = vcms[0].size();
//     std::vector<uint32_t> N_connections_vec(p.N_graphs, N_connections_max);
//     auto N_edges = edge_list.template apply<1, uint32_t>([](const Dataframe_t<std::pair<uint32_t, uint32_t>, 1> &data)
//                                                          { return data.size(); });
//     auto N_tot_edges = std::accumulate(N_edges.begin(), N_edges.end(), 0);
//     std::vector<uint32_t> edge_offsets_init(p.N_graphs + 1);
//     std::partial_sum(N_edges.begin(), N_edges.end(), edge_offsets_init.begin() + 1);
//     std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccm_indices(p.N_graphs);
//     std::transform(p.N_communities.begin(), p.N_communities.end(), ccm_indices.begin(), [&](const auto &N)
//                    { return complete_ccm(N, false); });
//     Dataframe_t<uint32_t, 2> ecms(p.N_graphs);



//     for (int i = 0; i < edge_list.size(); i++)
//     {
//         ecms[i] = ecm_from_vcm(edge_list[i], vcms[i], ccm_indices[i]);
//     }


//     validate_buffer_init_sizes(p, edge_list, vcms, ecms, p_Is_init);
//     auto ccm_weights = std::vector<std::vector<uint32_t>>(p.N_graphs);
//     std::transform(ecms.begin(), ecms.end(), ccm_weights.begin(), [&](const auto &ecm)
//                    { return ccm_weights_from_ecm(ecm, N_connections_max); });

//     auto ccm = make_ccm_df(ccm_indices, ccm_weights);
//     auto elist_flat = edge_list.flatten();
//     auto edge_from_init = Edge_t::get_from(elist_flat);
//     auto edge_to_init = Edge_t::get_to(elist_flat);


//     std::vector<State_t> community_state_init(p.N_communities_max() * N_sims_tot * (p.Nt_alloc + 1), {0, 0, 0});
//     std::vector<SIR_State> traj_init(N_sims_tot * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
//     std::vector<uint32_t> accumulate_event_init(N_sims_tot * N_connections_max * p.Nt_alloc * 2, 0);

//     auto seeds = generate_seeds(N_sims_tot, p.seed);
//     std::vector<Static_RNG::default_rng> rng_init;
//     rng_init.reserve(N_sims_tot);
//     std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
//                    { return Static_RNG::default_rng(seed); });

//     std::vector<std::size_t> sizes = {p.Nt * N_sims_tot * N_connections_max * sizeof(float),
//                                       edge_from_init.size() * sizeof(uint32_t),
//                                       edge_to_init.size() * sizeof(uint32_t),
//                                       ecms.byte_size(),
//                                       vcms.byte_size(),
//                                       (p.Nt_alloc + 1) * N_sims_tot * N_communities_max * sizeof(State_t),
//                                       p.Nt_alloc * N_sims_tot * N_connections_max * sizeof(uint32_t),
//                                       p.Nt_alloc * N_sims_tot * N_connections_max * sizeof(uint32_t),
//                                       rng_init.size() * sizeof(Static_RNG::default_rng)};

//     auto p_Is = sycl::buffer<float, 3>(sycl::range<3>(p.Nt, N_sims_tot, N_connections_max));
//     auto N_connections = sycl::buffer<uint32_t>(sycl::range<1>(p.N_graphs));
//     auto edge_from = sycl::buffer<uint32_t>((sycl::range<1>(N_tot_edges)));
//     auto edge_to = sycl::buffer<uint32_t>((sycl::range<1>(N_tot_edges)));
//     auto ecm = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_tot_edges)));
//     auto vcm = sycl::buffer<uint32_t, 2>(sycl::range<2>(p.N_graphs, vcms[0].size()));
//     auto edge_counts = sycl::buffer<uint32_t>((sycl::range<1>(p.N_graphs)));
//     auto edge_offsets = sycl::buffer<uint32_t>((sycl::range<1>(p.N_graphs)));
//     auto community_state = sycl::buffer<State_t, 3>(sycl::range<3>(p.Nt_alloc + 1, N_sims_tot, N_communities_max));
//     auto trajectory = sycl::buffer<SIR_State, 3>(sycl::range<3>(p.Nt_alloc + 1, N_sims_tot, N_vertices));
//     auto accumulated_events = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_connections_max));
//     // auto edge_events = sycl::buffer<uint8_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_tot_edges * p.N_sims));
//     auto rngs = sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(rng_init.size()));

//     std::vector<sycl::event> alloc_events(14);

//     alloc_events[1] = initialize_device_buffer<float, 3>(q, p_Is_init.flatten(), p_Is);
//     alloc_events[2] = initialize_device_buffer<uint32_t, 1>(q, edge_from_init, edge_from);
//     alloc_events[3] = initialize_device_buffer<uint32_t, 1>(q, edge_to_init, edge_to);
//     alloc_events[4] = initialize_device_buffer<uint32_t, 1>(q, ecms.flatten(), ecm);
//     alloc_events[5] = initialize_device_buffer<uint32_t, 2>(q, vcms.flatten(), vcm);
//     alloc_events[6] = initialize_device_buffer<uint32_t, 1>(q, N_edges, edge_counts);
//     alloc_events[7] = initialize_device_buffer<uint32_t, 1>(q, edge_offsets_init, edge_offsets);
//     alloc_events[8] = initialize_device_buffer<State_t, 3>(q, community_state_init, community_state);
//     alloc_events[9] = initialize_device_buffer<SIR_State, 3>(q, traj_init, trajectory);
//     alloc_events[10] = initialize_device_buffer<uint32_t, 3>(q, accumulate_event_init, accumulated_events);
//     alloc_events[12] = initialize_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, rngs);
//     alloc_events[13] = initialize_device_buffer<uint32_t, 1>(q, N_connections_vec, N_connections);
//     auto byte_size = rngs.byte_size() + trajectory.byte_size() + accumulated_events.byte_size() + p_Is.byte_size() + edge_from.byte_size() + edge_to.byte_size() + ecm.byte_size() + vcm.byte_size() + community_state.byte_size();
//     auto global_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
//     if_false_throw(ccm.size() == p.N_graphs, "ccm.size() != p.N_graphs");
//     assert(byte_size < global_mem_size && "Not enough global memory to allocate all buffers");

//     write_edgelist(con, p.p_out_idx, edge_list);
//     write_ccm(con, p.p_out_idx, ccm);
//     write_vcm(con, p.p_out_idx, vcms);

//     return Sim_Buffers(rngs,
//                        trajectory,
//                        accumulated_events,
//                        p_Is,
//                        edge_from,
//                        edge_to,
//                        ecm,
//                        vcm,
//                        edge_counts,
//                        edge_offsets,
//                        N_connections,
//                        community_state,
//                        ccm);
// }
