
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <SBM_Simulation/Utils/Buffer_Utils.hpp>
#include <SBM_Simulation/Graph/Graph_Types.hpp>
#include <SBM_Simulation/Graph/Graph.hpp>
#include <SBM_Simulation/Graph/Community_Mappings.hpp>
#include <Dataframe/Dataframe.hpp>



void validate_buffer_init_sizes(Sim_Param p, const Dataframe::Dataframe_t<std::pair<uint32_t, uint32_t>,2> &edge_list, const Dataframe::Dataframe_t<uint32_t,2> &vcms, const Dataframe::Dataframe_t<uint32_t, 2> &ecms, std::vector<float> p_Is_init)
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
        auto msg = std::string("p_Is_init.size() != p.N_graphs*N_connections*p.Nt, ") + std::to_string(p_Is_init.size()) + " vs " + std::to_string(p.N_graphs * p.N_connections_tot() * p.Nt);
        throw std::runtime_error(msg);
    }
}




Dataframe::Dataframe_t<float, 3> generate_duplicated_p_Is(uint32_t Nt, uint32_t N_sims_tot, const uint32_t N_connections_max, float p_I_min, float p_I_max, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(p_I_min, p_I_max);
    Dataframe::Dataframe_t<float, 3> p_Is(Nt);
    auto t = 0;
    while (t < Nt)
    {
        Dataframe::Dataframe_t<float, 2> p_Is_t(N_sims_tot, N_connections_max);
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

Dataframe::Dataframe_t<float, 3> Sim_Buffers::generate_random_p_Is(sycl::queue &q, Sim_Param p, soci::session &sql, const Dataframe::Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list_undirected, const Dataframe::Dataframe_t<uint32_t, 2> &vcms)
{

    auto p_Is_init = generate_duplicated_p_Is(p.Nt, p.N_sims_tot(), p.N_connections_max(), p.p_I_min, p.p_I_max, p.seed);
    return p_Is_init;
}

Sim_Buffers::Sim_Buffers(sycl::queue &q, Sim_Param p, soci::session &sql, const Dataframe::Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list_undirected, const Dataframe::Dataframe_t<uint32_t, 2> &vcms, Dataframe::Dataframe_t<float, 3> p_Is_init)
{

    const auto N_connections_max = p.N_connections_max();
    const auto N_communities_max = p.N_communities_max();
    const auto N_sims_tot = p.N_sims_tot();
    auto N_vertices = vcms[0].size();
    std::vector<uint32_t> N_connections_vec(p.N_graphs, N_connections_max);
    auto edge_list = mirror_duplicate_edge_list(edge_list_undirected);
    auto N_edges = edge_list.template apply<1, uint32_t>([](const Dataframe::Dataframe_t<std::pair<uint32_t, uint32_t>, 1> &data)
                                                         { return data.size(); });
    uint32_t N_tot_edges = std::accumulate(N_edges.begin(), N_edges.end(), 0);
    std::vector<uint32_t> edge_offsets_init(p.N_graphs + 1);
    std::partial_sum(N_edges.begin(), N_edges.end(), edge_offsets_init.begin() + 1);

    auto ccm_indices = ccms_from_vcms(edge_list, vcms);
    auto ecms = Dataframe::Dataframe_t<uint32_t, 2>(ecms_from_vcms(edge_list, vcms, ccm_indices));
    auto ccms = make_ccm_df(ccm_indices, ccm_weights_from_ecms(ecms, ccm_indices));

    validate_buffer_init_sizes(p, edge_list, vcms, ecms, p_Is_init.flatten());

    if (!p_Is_init.size())
    {
        p_Is_init = generate_random_p_Is(q, p, sql, edge_list_undirected, vcms);
    }

    auto elist_flat = edge_list.flatten();
    auto edge_from_init = Edge_t::get_from(elist_flat);
    auto edge_to_init = Edge_t::get_to(elist_flat);

    std::vector<State_t> community_state_init(p.N_communities_max() * N_sims_tot * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(N_sims_tot * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> accumulate_event_init(N_sims_tot * N_connections_max * p.Nt_alloc * 2, 0);

    auto seeds = generate_seeds(N_sims_tot, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(N_sims_tot);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
                   { return Static_RNG::default_rng(seed); });


    std::vector<sycl::event> init_event(12);
    rngs = make_shared_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, {rng_init.size()}, init_event[0]);
    vertex_state = make_shared_device_buffer<SIR_State, 3>(q, traj_init, {p.Nt+1, p.N_sims_tot(), N_vertices}, init_event[1]);
    accumulated_events = make_shared_device_buffer<uint32_t, 3>(q, accumulate_event_init, {p.Nt_alloc, p.N_sims_tot(), p.N_connections_max()}, init_event[2]);
    p_Is = make_shared_device_buffer<float, 3>(q, p_Is_init.flatten(), {p.Nt, p.N_sims_tot(), p.N_connections_max()}, init_event[3]);
    edge_from = make_shared_device_buffer<uint32_t, 1>(q, edge_from_init, {N_tot_edges}, init_event[4]);
    edge_to = make_shared_device_buffer<uint32_t, 1>(q, edge_to_init, {N_tot_edges}, init_event[5]);
    ecm = make_shared_device_buffer<uint32_t, 1>(q, ecms.flatten(), {ecms.size()}, init_event[6]);
    vcm = make_shared_device_buffer<uint32_t, 2>(q, vcms.flatten(), {p.N_graphs, vcms[0].size()}, init_event[7]);
    edge_counts = make_shared_device_buffer<uint32_t, 1>(q, N_connections_vec, {N_connections_vec.size()}, init_event[8]);
    edge_offsets = make_shared_device_buffer<uint32_t, 1>(q, edge_offsets_init, {edge_offsets_init.size()}, init_event[9]);
    community_state = make_shared_device_buffer<State_t, 3>(q, community_state_init, {p.Nt_alloc + 1, N_sims_tot, N_communities_max}, init_event[10]);
    N_connections = make_shared_device_buffer<uint32_t, 1>(q, N_connections_vec, {N_connections_vec.size()}, init_event[11]);


    auto check_sizes = [](const auto& args, auto sizes, const std::string& msg){
        for(int i = 0; i < args.size(); i++)
        {
            if_false_throw(args[i].size() == sizes[i], msg + ", " + std::to_string(args[i].size()) + " vs " + std::to_string(sizes[i]));
        }
    };

    // if_false_throw(edge_from_init.size() == edge_from.size(), "edge_from_init.size() != edge_from.size(), should be " + std::to_string(N_tot_edges));
    // if_false_throw(edge_to_init.size() == edge_to.size(), "edge_to_init.size() != edge_to.size(), should be " + std::to_string(N_tot_edges));
    // check_sizes(ecms, N_edges, "Wrong ecm size");
    // if_false_throw(vcm.get_range()[0] == p.N_graphs, "vcm.get_range()[0] != p.N_graphs, should be " + std::to_string(p.N_graphs));
    // if_false_throw(vcm.get_range()[1] == vcms[0].size(), "vcm.get_range()[1] != vcms[0].size(), should be " + std::to_string(vcms[0].size()));
    // if_false_throw(edge_counts.size() == p.N_graphs, "edge_counts.size() != p.N_graphs, should be " + std::to_string(p.N_graphs));
    // if_false_throw(edge_offsets.size() == (p.N_graphs + 1), "edge_offsets.size() != p.N_graphs, should be " + std::to_string(p.N_graphs));
    // if_false_throw(community_state.size() == (p.Nt_alloc + 1) * N_sims_tot * N_communities_max, "community_state.size() != (p.Nt_alloc + 1) * N_sims_tot * N_communities_max, should be " + std::to_string((p.Nt_alloc + 1) * N_sims_tot * N_communities_max));
    // if_false_throw(accumulated_events.size() == p.Nt_alloc * N_sims_tot * N_connections_max, "accumulated_events.size() != p.Nt_alloc * N_sims_tot * N_connections_max, should be " + std::to_string(p.Nt_alloc * N_sims_tot * N_connections_max));
    // if_false_throw(rngs.size() == rng_init.size(), "rngs.size() != rng_init.size(), should be " + std::to_string(rng_init.size()));
    // if_false_throw(ccms.size() == p.N_graphs, "ccm.size() != p.N_graphs");
    q.wait();




    SBM_Database::write_edgelist(sql, p.p_out_idx, edge_list);
    SBM_Database::write_ccm(sql, p.p_out_idx, ccms);
    SBM_Database::write_vcm(sql, p.p_out_idx, vcms);
}
