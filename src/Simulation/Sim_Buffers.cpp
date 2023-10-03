
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>

#include <execution>
#include <functional>

Sim_Buffers::Sim_Buffers(sycl::buffer<Static_RNG::default_rng> &rngs,
                         sycl::buffer<SIR_State, 3> &vertex_state,
                         sycl::buffer<uint32_t, 3> &accumulated_events,
                         //  sycl::buffer<uint8_t, 3> &edge_events,
                         sycl::buffer<float, 3> &p_Is,
                         sycl::buffer<uint32_t> &edge_from,
                         sycl::buffer<uint32_t> &edge_to,
                         sycl::buffer<uint32_t> &ecm,
                         sycl::buffer<uint32_t, 2> &vcm,
                         const std::vector<uint32_t> &vcm_vec,
                         sycl::buffer<uint32_t> &edge_counts,
                         sycl::buffer<uint32_t> &edge_offsets,
                         sycl::buffer<uint32_t> &N_connections,
                         sycl::buffer<State_t, 3> &community_state,
                         const Dataframe_t<Edge_t, 2> &ccm,
                         const std::vector<uint32_t> &N_connections_vec,
                         const std::vector<uint32_t> &N_communities) : rngs(std::move(rngs)),
                                                                       vertex_state(std::move(vertex_state)),
                                                                       accumulated_events(std::move(accumulated_events)),
                                                                       //    edge_events(std::move(edge_events)),
                                                                       p_Is(std::move(p_Is)),
                                                                       edge_from(std::move(edge_from)),
                                                                       edge_to(std::move(edge_to)),
                                                                       ecm(std::move(ecm)),
                                                                       vcm(std::move(vcm)),
                                                                       vcm_vec(vcm_vec),
                                                                       edge_counts(std::move(edge_counts)),
                                                                       edge_offsets(std::move(edge_offsets)),
                                                                       N_connections(N_connections),
                                                                       community_state(std::move(community_state)),
                                                                       N_connections_vec(N_connections_vec),
                                                                       N_connections_max(std::max_element(N_connections_vec.begin(), N_connections_vec.end())[0]),
                                                                       ccm(ccm),
                                                                       N_communities_vec(N_communities),
                                                                       N_communities_max(std::max_element(N_communities_vec.begin(), N_communities_vec.end())[0]),
                                                                       N_edges_tot(edge_from.get_range()[0])
{
}

std::vector<uint32_t> join_vectors(const std::vector<std::vector<uint32_t>> &vs)
{
    std::vector<uint32_t> result;
    for (auto &v : vs)
    {
        result.insert(result.end(), v.begin(), v.end());
    }
    return result;
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> separate_flatten_edge_lists(const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_lists)
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

void align_edges_with_ccm(std::vector<std::pair<uint32_t, uint32_t>> &edges, const std::vector<std::pair<uint32_t, uint32_t>> &ccm, const std::vector<uint32_t> &vcm)
{
    auto unaligned_equal = [](const std::pair<uint32_t, uint32_t> e0, const std::pair<uint32_t, uint32_t> e1)
    {
        return ((e0.first == e1.first) && (e0.second == e1.second)) || ((e0.first == e1.second) && (e0.second == e1.first));
    };
    for (auto &e : edges)
    {
        for (auto &c : ccm)
        {
            if (unaligned_equal(std::make_pair(vcm[e.first], vcm[e.second]), c))
            {
                e = std::make_pair(e.second, e.first);
            }
        }
    }
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
    if_false_throw(check_buffer_dims(accumulated_events, p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max), "events has wrong dimensions: " + buf_dim_string(accumulated_events) + " vs " + dim_string(p.Nt_alloc, p.N_sims * p.N_graphs, N_connections_max));
}

// count unique elements
template <typename T>
uint32_t count_unique(const std::vector<T> &v)
{
    auto n = 1 + std::inner_product(std::next(v.begin()), v.end(),
                                    v.begin(), size_t(0),
                                    std::plus<size_t>(),
                                    std::not_equal_to<float>());
    return n;
}

void validate_maps(const std::vector<std::vector<uint32_t>> &ecms, const std::vector<std::vector<uint32_t>> &vcms, const auto &N_communities, const auto &N_connections, const auto &edge_list)
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

void validate_ecm(const auto &edge_list, const auto &ecm, const auto &ccm, const auto &vcm)
{
    auto directed_equal = [](const auto &e_0, const auto &e_1)
    {
        return (e_0.first == e_1.first) && (e_0.second == e_1.second);
    };
    auto is_edge_in_connection = [&](auto edge, auto ecm_id)
    {
        auto c_edge = std::make_pair(vcm[edge.first], vcm[edge.second]);
        for (int i = 0; i < ccm.size(); i++)
        {

            if (directed_equal(c_edge, ccm[i]))
            {
                return ecm_id == i;
            }
        }
        return false;
    };
    for (int i = 0; i < edge_list.size(); i++)
    {
        if_false_throw(is_edge_in_connection(edge_list[i], ecm[i]), "edge_list[" + std::to_string(i) + "] is not in the correct connection");
    }
}

std::vector<Edge_t> combine_ccm(const std::vector<std::pair<uint32_t, uint32_t>> &ccm_indices, const std::vector<uint32_t> &ccm_weights)
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

auto combine_ccms(const auto &ccm_indices, const auto &ccm_weights)
{
    std::vector<std::vector<Edge_t>> result;
    result.reserve(ccm_indices.size());
    std::transform(ccm_indices.begin(), ccm_indices.end(), ccm_weights.begin(), std::back_inserter(result), combine_ccm);
    return result;
}

Dataframe_t<Edge_t, 2> make_ccm_df(const auto &ccm_indices, const std::vector<std::vector<uint32_t>> &ccm_weights)
{
    Dataframe_t<Edge_t, 2> df;
    auto N_graphs = ccm_weights.size();
    df.data.reserve(N_graphs);
    std::vector<Dataframe_t<Edge_t, 1>> ccms(N_graphs);
    std::transform(ccm_weights.begin(), ccm_weights.end(), ccms.begin(), [ccm_indices](const auto &ccm_w)
                   { return Dataframe_t<Edge_t, 1>(combine_ccm(ccm_indices, ccm_w)); });

    return Dataframe_t<Edge_t, 2>(ccms);
}

auto get_max_element(const std::vector<std::vector<uint32_t>> &vec)
{
    std::vector<uint32_t> vec_max(vec.size());
    std::transform(vec.begin(), vec.end(), vec_max.begin(), [](const auto &v)
                   { return *std::max_element(v.begin(), v.end()); });
    return std::max_element(vec_max.begin(), vec_max.end())[0];
}

auto get_max_elements(const std::vector<std::vector<uint32_t>> &vec)
{
    std::vector<uint32_t> vec_max(vec.size());
    std::transform(vec.begin(), vec.end(), vec_max.begin(), [](const auto &v)
                   { return std::max_element(v.begin(), v.end())[0]; });
    return vec_max;
}


template <typename T>
auto get_sizes(const std::vector<std::vector<T>> &vec)
{
    std::vector<uint32_t> vec_max(vec.size());
    std::transform(vec.begin(), vec.end(), vec_max.begin(), [](const auto &v)
                   { return v.size(); });
    return vec_max;
}

Dataframe_t<std::pair<uint32_t, uint32_t>, 2> mirror_duplicate_edge_list(const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list)
{
    Dataframe_t<std::pair<uint32_t, uint32_t>, 2> result(edge_list.size());
    std::transform(edge_list.begin(), edge_list.end(), result.begin(), [](const auto &edge_list_elem)
                   { Dataframe_t<std::pair<uint32_t, uint32_t>, 1> result_elem(edge_list_elem.size() * 2);
                       std::transform(edge_list_elem.begin(), edge_list_elem.end(), result_elem.begin(), [](const auto &edge)
                                      { return edge; });
                       std::transform(edge_list_elem.begin(), edge_list_elem.end(), result_elem.begin() + edge_list_elem.size(), [](const auto &edge)
                                      { return std::make_pair(edge.second, edge.first); });
                       return result_elem; });
    return result;
}

Dataframe_t<float,1> generate_duplicated_p_Is(auto Nt, auto N_sims_tot, auto N_connections, float p_I_min, float p_I_max, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(p_I_min, p_I_max);
    Dataframe_t<float,3> p_Is(Nt, Dataframe_t<float,2>(N_sims_tot, std::vector<float>(N_connections)));
    auto t = 0;
    while (t < Nt)
    {
        Dataframe_t<float,2> p_Is_t(N_sims_tot, std::vector<float>(N_connections));
        for (auto &p_I : p_Is_t)
        {
            std::generate(p_I.begin(), p_I.end(), [&]()
                          { return dist(gen); });
        }
        for(int i = 0; i < 7; i++)
        {
            if ((t + i) >= Nt)
                break;
            p_Is[t + i] = p_Is_t;
        }
        t += 7;
    }
    //reshape p_Is to flat
    std::vector<float> p_Is_flat(Nt*N_sims_tot*N_connections);
    for (int t = 0; t < Nt; t++)
    {
        for (int s = 0; s < N_sims_tot; s++)
        {
            for (int c = 0; c < N_connections; c++)
            {
                p_Is_flat[t * N_sims_tot * N_connections + s * N_connections + c] = p_Is[t][s][c];
            }
        }
    }
    return p_Is_flat;
}

Sim_Buffers Sim_Buffers::make(sycl::queue &q, Sim_Param p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2>& edge_list_undirected, const Dataframe_t<uint32_t, 2>& vcms, std::vector<float> p_Is_init)
{

    if ((p.global_mem_size == 0) || p.local_mem_size == 0)
    {
        auto device = q.get_device();
        p.local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
        p.global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    }
    auto edge_list = mirror_duplicate_edge_list(edge_list_undirected);
    auto N_sims_tot = p.N_graphs * p.N_sims;
    auto N_vertices = vcms[0].size();


    auto N_communities = vcms.template apply<1, uint32_t>([](const Dataframe_t<uint32_t, 1>& data){return (uint32_t)std::max_element(data.begin(), data.end())[0] + 1;});
    auto N_communities_max = std::max_element(N_communities.begin(), N_communities.end())[0] + 1;
    auto ccm_indices = complete_ccm(N_communities_max, true);
    auto N_connections_max = ccm_indices.size();
    std::for_each(N_communities.begin(), N_communities.end(), [](auto &N)
                  { N++; });
    std::vector<uint32_t> N_connections_vec(p.N_graphs, N_connections_max);
    auto N_edges = edge_list.template apply<1, uint32_t>([](const Dataframe_t<std::pair<uint32_t, uint32_t>, 1>& data){return data.size();});
    auto N_tot_edges = std::accumulate(N_edges.begin(), N_edges.end(), 0);
    std::vector<uint32_t> edge_offsets_init(p.N_graphs + 1);
    std::partial_sum(N_edges.begin(), N_edges.end(), edge_offsets_init.begin() + 1);

    Dataframe_t<uint32_t,2> ecms(p.N_graphs);
    for (int i = 0; i < edge_list.size(); i++)
    {
        ecms[i] = ecm_from_vcm(edge_list[i], vcms[i], ccm_indices);
    }

    auto ccm_weights = std::vector<std::vector<uint32_t>>(p.N_graphs);
    std::transform(ecms.begin(), ecms.end(), ccm_weights.begin(), [&](const auto &ecm)
                   { return ccm_weights_from_ecm(ecm, N_connections_max); });

    auto ccm = make_ccm_df(ccm_indices, ccm_weights);

    auto [edge_from_init, edge_to_init, edge_counts_init] = separate_flatten_edge_lists(edge_list);

    std::vector<State_t> community_state_init(N_communities_max * N_sims_tot * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(N_sims_tot * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> accumulate_event_init(N_sims_tot * N_connections_max * p.Nt_alloc * 2, 0);
    // std::vector<uint8_t> edge_event_init(N_tot_edges * p.N_sims * p.Nt_alloc * 2, 0);

    auto seeds = generate_seeds(N_sims_tot, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(N_sims_tot);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
                   { return Static_RNG::default_rng(seed); });

    std::vector<std::size_t> sizes = {p.Nt * N_sims_tot * N_connections_max * sizeof(float),
                                      edge_from_init.size() * sizeof(uint32_t),
                                      edge_to_init.size() * sizeof(uint32_t),
                                      ecms.byte_size(),
                                      vcms.byte_size(),
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
    auto accumulated_events = sycl::buffer<uint32_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_connections_max));
    // auto edge_events = sycl::buffer<uint8_t, 3>(sycl::range<3>(p.Nt_alloc, N_sims_tot, N_tot_edges * p.N_sims));
    auto rngs = sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(rng_init.size()));

    std::vector<sycl::event> alloc_events(14);
    if (p_Is_init.size() == 0)
    {
        p_Is_init = generate_duplicated_p_Is(p.Nt, p.N_sims*p.N_graphs, N_connections_max, p.p_I_min, p.p_I_max, p.seed);
    }

    // if (p_Is_init.size() == 0)
    // {
    //     p_Is_init = generate_floats(p.Nt * N_sims_tot * N_connections_max, p.p_I_min, p.p_I_max, 8, p.seed);
    // }

    else
    {
        if_false_throw(p_Is_init.size() == p.Nt * N_sims_tot * N_connections_vec[0], "p_Is_init.size() != p.Nt*N_sims_tot*N_connections");
    }

    alloc_events[1] = initialize_device_buffer<float, 3>(q, p_Is_init, p_Is);
    alloc_events[2] = initialize_device_buffer<uint32_t, 1>(q, edge_from_init, edge_from);
    alloc_events[3] = initialize_device_buffer<uint32_t, 1>(q, edge_to_init, edge_to);
    alloc_events[4] = initialize_device_buffer<uint32_t, 1>(q, ecms.flatten(), ecm);
    alloc_events[5] = initialize_device_buffer<uint32_t, 2>(q, vcms.flatten(), vcm);
    alloc_events[6] = initialize_device_buffer<uint32_t, 1>(q, edge_counts_init, edge_counts);
    alloc_events[7] = initialize_device_buffer<uint32_t, 1>(q, edge_offsets_init, edge_offsets);
    alloc_events[8] = initialize_device_buffer<State_t, 3>(q, community_state_init, community_state);
    alloc_events[9] = initialize_device_buffer<SIR_State, 3>(q, traj_init, trajectory);
    alloc_events[10] = initialize_device_buffer<uint32_t, 3>(q, accumulate_event_init, accumulated_events);
    // alloc_events[11] = initialize_device_buffer<uint8_t, 3>(q, edge_event_init, edge_events);
    alloc_events[12] = initialize_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, rngs);
    alloc_events[13] = initialize_device_buffer<uint32_t, 1>(q, N_connections_vec, N_connections);

    for (int i = 0; i < ecms.size(); i++)
    {
        validate_ecm(edge_list[i], ecms[i], ccm_indices, vcms[i]);
    }

    auto byte_size = rngs.byte_size() + trajectory.byte_size() + accumulated_events.byte_size() + p_Is.byte_size() + edge_from.byte_size() + edge_to.byte_size() + ecm.byte_size() + vcm.byte_size() + community_state.byte_size();
    // get global memory size
    auto global_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
    // get max alloc size
    //  auto max_alloc_size = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    if_false_throw(ccm.size() == p.N_graphs, "ccm.size() != p.N_graphs");
    assert(byte_size < global_mem_size && "Not enough global memory to allocate all buffers");
    return Sim_Buffers(rngs,
                       trajectory,
                       accumulated_events,
                       //    edge_events,
                       p_Is,
                       edge_from,
                       edge_to,
                       ecm,
                       vcm,
                       vcms.flatten(),
                       edge_counts,
                       edge_offsets,
                       N_connections,
                       community_state,
                       ccm,
                       N_connections_vec,
                       N_communities);
}
