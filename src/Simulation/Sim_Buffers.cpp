#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Validation.hpp>
#include <execution>
#include <functional>

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
                         cl::sycl::buffer<uint32_t> &edge_offsets,
                         cl::sycl::buffer<uint32_t> &N_connections,
                         cl::sycl::buffer<State_t, 3> &community_state,
                         const Dataframe_t<Edge_t, 4> &ccm,
                         const std::vector<uint32_t> &N_connections_vec,
                         const std::vector<uint32_t> &N_communities) : rngs(std::move(rngs)),
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
                                                                       N_connections(N_connections),
                                                                       community_state(std::move(community_state)),
                                                                       ccm(ccm),
                                                                       N_connections_vec(N_connections_vec),
                                                                       N_connections_max(std::max_element(N_connections_vec.begin(), N_connections_vec.end())[0]),
                                                                       N_communities_vec(N_communities),
                                                                       N_communities_max(std::max_element(N_communities_vec.begin(), N_communities_vec.end())[0]) {}

std::size_t Sim_Buffers::byte_size() const { return rngs.byte_size() + vertex_state.byte_size() + events_from.byte_size() + events_to.byte_size() + p_Is.byte_size() + edge_from.byte_size() + edge_to.byte_size() + ecm.byte_size() + vcm.byte_size() + community_state.byte_size(); }

std::vector<uint32_t> join_vectors(const std::vector<std::vector<uint32_t>> &vs)
{
    std::vector<uint32_t> result;
    for (auto &v : vs)
    {
        result.insert(result.end(), v.begin(), v.end());
    }
    return result;
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> separate_flatten_edge_lists(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_lists)
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
    // insert into edge_from_init
    std::size_t offset = 0;
    for (auto &e : edge_froms)
    {
        std::copy(e.begin(), e.end(), edge_from_init.begin() + offset);
        offset += e.size();
    }
    // insert into edge_to_init
    offset = 0;
    for (auto &e : edge_tos)
    {
        std::copy(e.begin(), e.end(), edge_to_init.begin() + offset);
        offset += e.size();
    }
    return std::make_tuple(edge_from_init, edge_to_init, edge_counts_init);
}

void validate_buffer_init_sizes(Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcms, const std::vector<std::vector<uint32_t>> &ecms, std::vector<float> p_Is_init, uint32_t N_connections)
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

    if ((p_Is_init.size() != p.N_graphs * N_connections * p.Nt) && (p_Is_init.size() != 0))
    {
        throw std::runtime_error("p_Is_init.size() != p.N_graphs*N_connections*p.Nt");
    }

    // if (std::any_of(vcms.begin(), vcms.end(), [&](const auto &vc)
    //                 { return !(vc.size() == N_communities_max * p.N_pop); }))
    // {
    //     throw std::runtime_error("vcm.size() != N_communities_max*p.N_pop");
    // }
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

    if_false_throw(check_buffer_dims(p_Is, p.Nt, p.N_graphs * p.N_sims, N_connections_max), "p_Is has wrong dimensions: " + buf_dim_string(p_Is) + " vs " + dim_string(p.Nt, p.N_graphs, N_connections_max));
    if_false_throw(rngs.size() == p.N_sims * p.N_graphs, "rngs has wrong size: " + std::to_string(rngs.size()) + " vs " + std::to_string(p.N_sims * p.N_graphs));
    // if_false_throw(check_buffer_dims(vertex_state, p.Nt_alloc + 1, p.N_sims*p.N_graphs, p.N_pop*N_communities_max), "vertex_state has wrong dimensions: " + buf_dim_string(vertex_state) + " vs " + dim_string(p.Nt_alloc + 1, p.N_sims*p.N_graphs, p.N_pop*N_communities_max));
    if_false_throw(check_buffer_dims(events_to, p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max), "events_to has wrong dimensions: " + buf_dim_string(events_to) + " vs " + dim_string(p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max));
    if_false_throw(check_buffer_dims(events_from, p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max), "events_from has wrong dimensions: " + buf_dim_string(events_from) + " vs " + dim_string(p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max));
}

// count unique elements
template <typename T>
uint32_t count_unique(const std::vector<T> &v)
{
    auto n = 1 + std::inner_product(std::next(v.begin()), v.end(),
                                    v.begin(), size_t(0),
                                    std::plus<size_t>(),
                                    std::not_equal_to<double>());
    return n;
}

void validate_maps(const std::vector<std::vector<uint32_t>> &ecms, const std::vector<std::vector<uint32_t>> &vcms, const auto &N_communities, const auto &N_connections, const auto& edge_list)
{
    for (auto g_idx = 0; g_idx < ecms.size(); g_idx++)
    {
        if_false_throw(std::all_of(ecms[g_idx].begin(), ecms[g_idx].end(), [&](const auto &e)
                                   { return e < N_connections[g_idx]; }),
                       "ecms[" + std::to_string(g_idx) + "] has index values higher than N_connections: " + std::to_string(N_connections[g_idx]));

        if_false_throw(std::all_of(vcms[g_idx].begin(), vcms[g_idx].end(), [&](const auto &v)
                                   { return v < N_communities[g_idx]; }),
                       "vcms[" + std::to_string(g_idx) + "] has index values higher than N_communities: " + std::to_string(N_communities[g_idx]));
        if_false_throw(edge_list[g_idx].size() == ecms[g_idx].size(), "edge_list[" + std::to_string(g_idx) + "] and ecms[" + std::to_string(g_idx) + "] have different sizes: " + std::to_string(edge_list[g_idx].size()) + " vs " + std::to_string(ecms[g_idx].size()));
    }
}

void validate_ecm(const auto& edge_list, const auto& ecm, const auto& ccm, const auto& vcm)
{
    auto is_edge_in_connection = [&](auto edge, auto ecm_id)
    {
        for(int i = 0; i < ccm.size(); i++)
        {
            if (((vcm[edge.first] == ccm[i].first) && (vcm[edge.second] == ccm[i].second)) || ((vcm[edge.first] == ccm[i].second) && (vcm[edge.second] == ccm[i].first)))
            {
                return ecm_id == i;
            }
        }
        return false;
    };
    for(int i = 0; i < edge_list.size(); i++)
    {
        if_false_throw(is_edge_in_connection(edge_list[i], ecm[i]), "edge_list[" + std::to_string(i) + "] is not in the correct connection");
    }
}

std::tuple<std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> create_mappings(const std::vector<std::pair<uint32_t, uint32_t>>& edge_list, const std::vector<uint32_t>& vcm)
{
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    auto undirected_equal = [](const std::pair<uint32_t, uint32_t> e0, const std::pair<uint32_t, uint32_t> e1)
    {
        return ((e0.first == e1.first) && (e0.second == e1.second)) || ((e0.first == e1.second) && (e0.second == e1.first));
    };
    auto rising_directed = [](const std::pair<uint32_t, uint32_t> e0)
    {
        return (e0.first < e0.second) ? e0 : std::make_pair(e0.second, e0.first);
    };
    std::vector<uint32_t> ecm(edge_list.size());
    auto idx = 0;
    for(int e_idx = 0; e_idx < edge_list.size(); e_idx++)
    {
        auto edge = edge_list[e_idx];
        auto v1 = vcm[edge.first];
        auto v2 = vcm[edge.second];
        auto ccm_edge = std::make_pair(v1, v2);
        auto it = std::find_if(ccm.begin(), ccm.end(), [&](auto e)
                               { return undirected_equal(e, ccm_edge); });
        if(it == ccm.end())
        {
            ccm.push_back(rising_directed(ccm_edge));
            idx = ccm.size() - 1;
        }
        else
        {
            idx = std::distance(ccm.begin(), it);
        }
        ecm[e_idx] = idx;
    }
    return std::make_tuple(ecm, ccm);
}

std::vector<Edge_t> combine_ccm(const std::vector<std::pair<uint32_t, uint32_t>>& ccm_indices, const std::vector<uint32_t>& ccm_weights)
{
    auto make_edges = [](const auto pair, const auto weight)
    {
        return Edge_t{pair.first, pair.second, weight};
    };
    std::vector<Edge_t> result;
    result.reserve(ccm_indices.size());
    std::transform(ccm_indices.begin(), ccm_indices.end(), ccm_weights.begin(), std::back_inserter(result), make_edges);
    return result;
}

auto combine_ccms(const auto& ccm_indices, const auto& ccm_weights)
{
    std::vector<std::vector<Edge_t>> result;
    result.reserve(ccm_indices.size());
    std::transform(ccm_indices.begin(), ccm_indices.end(), ccm_weights.begin(), std::back_inserter(result), combine_ccm);
    return result;
}

Dataframe_t<Edge_t, 2> duplicate_ccm(const std::vector<Edge_t>& ccm, auto N)
{
    Dataframe_t<Edge_t, 2> result;
    result.data = std::vector<Dataframe_t<Edge_t, 1>>(N, ccm);
    return result;
}

Dataframe_t<Edge_t, 4> make_ccm_df(const auto& ccms, auto N_graphs, auto N_sims)
{
    Dataframe_t<Edge_t, 4> df;
    df.data.reserve(N_graphs);
    for(int i = 0; i < N_graphs; i++)
    {
        auto df_graph = Dataframe_t<Edge_t, 3>(std::vector<Dataframe_t<Edge_t, 2>>(N_sims, duplicate_ccm(ccms[i], N_sims)));
        df.data.push_back(df_graph);
    }
    return df;
}


Sim_Buffers Sim_Buffers::make(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcms, std::vector<float> p_Is_init)
{

    if ((p.global_mem_size == 0) || p.local_mem_size == 0)
    {
        auto device = q.get_device();
        p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }

    std::vector<std::vector<uint32_t>> ecms(p.N_graphs);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> ccm_indices(p.N_graphs);
    for(int i = 0; i < p.N_graphs; i++)
    {
        std::tie(ecms[i], ccm_indices[i]) = create_mappings(edge_list[i], vcms[i]);
    }


    std::vector<uint32_t> N_connections_vec(p.N_graphs);
    std::transform(ccm_indices.begin(), ccm_indices.end(), N_connections_vec.begin(), [](const auto &c)
                   { return c.size(); });

    auto N_connections_max = std::max_element(N_connections_vec.begin(), N_connections_vec.end())[0];
    std::vector<uint32_t> N_communities(p.N_graphs);
    std::transform(vcms.begin(), vcms.end(), N_communities.begin(), [](auto &v)
                   { return *std::max_element(v.begin(), v.end()) + 1; });
    auto N_communities_max = std::max_element(N_communities.begin(), N_communities.end())[0];
    validate_buffer_init_sizes(p, edge_list, vcms, ecms, p_Is_init, N_connections_max);
    auto N_sims_tot = p.compute_range[0];

    auto ecm_init = join_vectors(ecms);
    // uint32_t N_connections = std::max_element(ecm_init.begin(), ecm_init.end())[0] + 1;

    uint32_t N_vertices = vcms[0].size();
    auto vcm_init = join_vectors(vcms);


    validate_maps(ecms, vcms, N_communities, N_connections_vec, edge_list);
    auto e_data = separate_flatten_edge_lists(edge_list);
    std::vector<uint32_t> edge_from_init = std::get<0>(e_data);
    std::vector<uint32_t> edge_to_init = std::get<1>(e_data);
    std::vector<uint32_t> edge_counts_init = std::get<2>(e_data);
    std::vector<uint32_t> edge_offsets_init(edge_counts_init.size());
    std::partial_sum(edge_counts_init.begin(), edge_counts_init.end() - 1, edge_offsets_init.begin() + 1);
    uint32_t N_tot_edges = std::accumulate(edge_counts_init.begin(), edge_counts_init.end(), 0);

    std::vector<State_t> community_state_init(N_communities_max * N_sims_tot * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(N_sims_tot * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> event_init_from(N_sims_tot * N_connections_max * p.Nt_alloc, 0);
    std::vector<uint32_t> event_init_to(N_sims_tot * N_connections_max * p.Nt_alloc, 0);
    auto seeds = generate_seeds(N_sims_tot, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(N_sims_tot);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
                   { return Static_RNG::default_rng(seed); });

    std::vector<std::size_t> sizes = {p.Nt * N_sims_tot * N_connections_max * sizeof(float),
                                      edge_from_init.size() * sizeof(uint32_t),
                                      edge_to_init.size() * sizeof(uint32_t),
                                      ecm_init.size() * sizeof(uint32_t),
                                      vcm_init.size() * sizeof(uint32_t),
                                      (p.Nt_alloc + 1) * N_sims_tot * N_communities_max * sizeof(State_t),
                                      p.Nt_alloc * N_sims_tot * N_connections_max * sizeof(uint32_t),
                                      p.Nt_alloc * N_sims_tot * N_connections_max * sizeof(uint32_t),
                                      rng_init.size() * sizeof(Static_RNG::default_rng)};

    assert(p.global_mem_size > std::accumulate(sizes.begin(), sizes.end(), 0) && "Not enough global memory to allocate all buffers");

    auto p_Is = sycl::buffer<float, 3>(sycl::range<3>(p.Nt, N_sims_tot, N_connections_max));
    auto N_connections = sycl::buffer<uint32_t>(sycl::range<1>(p.N_graphs));
    auto edge_from = sycl::buffer<uint32_t>((sycl::range<1>(N_tot_edges)));
    auto edge_to = sycl::buffer<uint32_t>((sycl::range<1>(N_tot_edges)));
    auto ecm = sycl::buffer<uint32_t, 1>((sycl::range<1>(N_tot_edges)));
    auto vcm = sycl::buffer<uint32_t, 2>(sycl::range<2>(p.N_graphs, vcms[0].size()));
    auto edge_counts = sycl::buffer<uint32_t>((sycl::range<1>(p.compute_range[0])));
    auto edge_offsets = sycl::buffer<uint32_t>((sycl::range<1>(p.compute_range[0])));
    auto community_state = sycl::buffer<State_t, 3>(sycl::range<3>(p.Nt_alloc + 1, N_sims_tot, N_communities_max));
    auto trajectory = sycl::buffer<SIR_State, 3>(sycl::range<3>(p.Nt_alloc + 1, N_sims_tot, N_vertices));
    auto events_from = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_connections_max));
    auto events_to = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_connections_max));
    auto rngs = sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(rng_init.size()));

    std::vector<sycl::event> alloc_events(14);

    if (p_Is_init.size() == 0)
    {
        p_Is_init = generate_floats(p.Nt * N_sims_tot * N_connections_max, p.p_I_min, p.p_I_max, 8, p.seed);
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
    alloc_events[13] = initialize_device_buffer<uint32_t, 1>(q, N_connections_vec, N_connections);


    std::vector<std::vector<uint32_t>> ccm_weights(p.N_graphs);
    for (int i = 0; i < ecms.size(); i++)
    {
        ccm_weights[i] = ccm_weights_from_ecm(ecms[i], N_connections_vec[i]);
    }

    auto ccms = combine_ccms(ccm_indices, ccm_weights);
    auto ccm_df = make_ccm_df(ccms, p.N_graphs, N_sims_tot);

    for(int i = 0; i < ecms.size(); i++)
    {
        validate_ecm(edge_list[i], ecms[i], ccm_indices[i], vcms[i]);
    }

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
                       N_connections,
                       community_state,
                       ccm_df,
                       N_connections_vec,
                       N_communities);
}
