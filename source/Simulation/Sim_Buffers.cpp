
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <Sycl_Buffer_Routines/Buffer_Utils.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>
#include <SBM_Simulation/Graph/Graph_Types.hpp>
#include <SBM_Simulation/Graph/Graph.hpp>
#include <SBM_Simulation/Graph/Community_Mappings.hpp>
#include <Dataframe/Dataframe.hpp>



void validate_buffer_init_sizes(QJsonObject p, const Dataframe::Dataframe_t<Edge_t,2> &edge_list, const Dataframe::Dataframe_t<uint32_t,2> &vcms, const Dataframe::Dataframe_t<uint32_t, 2> &ecms, std::vector<float> p_Is_init)
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

Dataframe::Dataframe_t<float, 3> Sim_Buffers::generate_random_p_Is(sycl::queue &q, QJsonObject p,  const Dataframe::Dataframe_t<Edge_t, 2> &edge_list_undirected, const Dataframe::Dataframe_t<uint32_t, 2> &vcms)
{

    auto p_Is_init = generate_duplicated_p_Is(p.Nt, p.N_sims_tot(), p.N_connections_max(), p.p_I_min, p.p_I_max, p.seed);
    return p_Is_init;
}

Sim_Buffers::Sim_Buffers(sycl::queue &q, QJsonObject p,  const Dataframe::Dataframe_t<Edge_t, 2> &edge_list_undirected, const Dataframe::Dataframe_t<uint32_t, 2> &vcms, Dataframe::Dataframe_t<float, 3> p_Is_init)
{

    const auto N_connections_max = p.N_connections_max();
    const auto N_communities_max = p.N_communities_max();
    const auto N_sims_tot = p.N_sims_tot();
    auto N_vertices = vcms[0].size();
    std::vector<uint32_t> N_connections_vec(p.N_graphs, N_connections_max);
    auto edge_list = mirror_duplicate_edge_list(edge_list_undirected);
    auto N_edges = edge_list.template apply<1, uint32_t>([](const Dataframe::Dataframe_t<Edge_t, 1> &data)
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
        p_Is_init = generate_random_p_Is(q, p, edge_list_undirected, vcms);
    }

    auto elist_flat = edge_list.flatten();
    auto edge_from_init = Edge_t::get_from(elist_flat);
    auto edge_to_init = Edge_t::get_to(elist_flat);

    std::vector<State_t> community_state_init(p.N_communities_max() * N_sims_tot * (p.Nt_alloc + 1), {0, 0, 0});
    std::vector<SIR_State> traj_init(N_sims_tot * (p.Nt_alloc + 1) * N_vertices, SIR_INDIVIDUAL_S);
    std::vector<uint32_t> accumulate_event_init(N_sims_tot * N_connections_max * p.Nt_alloc, 0);

    auto seeds = Buffer_Routines::generate_seeds(N_sims_tot, p.seed);
    std::vector<Static_RNG::default_rng> rng_init;
    rng_init.reserve(N_sims_tot);
    std::transform(seeds.begin(), seeds.end(), std::back_inserter(rng_init), [](const auto seed)
                   { return Static_RNG::default_rng(seed); });
    auto ecms_flat = ecms.flatten();
    auto vcms_flat = vcms.flatten();
    std::vector<sycl::event> init_event(12);
    rngs = Buffer_Routines::make_shared_device_buffer<Static_RNG::default_rng, 1>(q, rng_init, {rng_init.size()}, init_event[0]);
    vertex_state = Buffer_Routines::make_shared_device_buffer<SIR_State, 3>(q, traj_init, {p.Nt_alloc+1, p.N_sims_tot(), N_vertices}, init_event[1]);
    accumulated_events = Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(q, accumulate_event_init, {p.Nt_alloc, p.N_sims_tot(), p.N_connections_max()}, init_event[2]);
    p_Is = Buffer_Routines::make_shared_device_buffer<float, 3>(q, p_Is_init.flatten(), {p.Nt, p.N_sims_tot(), p.N_connections_max()}, init_event[3]);
    edge_from = Buffer_Routines::make_shared_device_buffer<uint32_t, 1>(q, edge_from_init, {N_tot_edges}, init_event[4]);
    edge_to = Buffer_Routines::make_shared_device_buffer<uint32_t, 1>(q, edge_to_init, {N_tot_edges}, init_event[5]);
    ecm = Buffer_Routines::make_shared_device_buffer<uint32_t, 1>(q, ecms_flat, {ecms_flat.size()}, init_event[6]);
    vcm = Buffer_Routines::make_shared_device_buffer<uint32_t, 2>(q, vcms_flat, {p.N_graphs, vcms[0].size()}, init_event[7]);
    edge_offsets = Buffer_Routines::make_shared_device_buffer<uint32_t, 1>(q, edge_offsets_init, {edge_offsets_init.size()}, init_event[9]);
    community_state = Buffer_Routines::make_shared_device_buffer<State_t, 3>(q, community_state_init, {p.Nt_alloc + 1, N_sims_tot, N_communities_max}, init_event[10]);
    N_connections = Buffer_Routines::make_shared_device_buffer<uint32_t, 1>(q, N_connections_vec, {N_connections_vec.size()}, init_event[11]);


    auto check_sizes = [](const auto& args, auto sizes, const std::string& msg){
        for(int i = 0; i < args.size(); i++)
        {
            Buffer_Routines::if_false_throw(args[i].size() == sizes[i], msg + ", " + std::to_string(args[i].size()) + " vs " + std::to_string(sizes[i]));
        }
    };

    check_sizes(ecms, N_edges, "Wrong ecm size");
    Buffer_Routines::if_false_throw(edge_from_init.size() == edge_from->size(), "edge_from_init->size() != edge_from->size(), should be " + std::to_string(N_tot_edges));
    Buffer_Routines::if_false_throw(edge_to_init.size() == edge_to->size(), "edge_to_init->size() != edge_to->size(), should be " + std::to_string(N_tot_edges));
    Buffer_Routines::if_false_throw(vcm->get_range()[0] == p.N_graphs, "vcm.get_range()[0] != p.N_graphs, should be " + std::to_string(p.N_graphs));
    Buffer_Routines::if_false_throw(vcm->get_range()[1] == vcms[0].size(), "vcm.get_range()[1] != vcms[0]->size(), should be " + std::to_string(vcms[0].size()));
    Buffer_Routines::if_false_throw(edge_offsets->size() == (p.N_graphs + 1), "edge_offsets->size() != p.N_graphs, should be " + std::to_string(p.N_graphs));
    Buffer_Routines::if_false_throw(community_state->size() == (p.Nt_alloc + 1) * N_sims_tot * N_communities_max, "community_state->size() != (p.Nt_alloc + 1) * N_sims_tot * N_communities_max, should be " + std::to_string((p.Nt_alloc + 1) * N_sims_tot * N_communities_max));
    Buffer_Routines::if_false_throw(accumulated_events->size() == p.Nt_alloc * N_sims_tot * N_connections_max, "accumulated_events->size() != p.Nt_alloc * N_sims_tot * N_connections_max, should be " + std::to_string(p.Nt_alloc * N_sims_tot * N_connections_max));
    Buffer_Routines::if_false_throw(rngs->size() == rng_init.size(), "rngs->size() != rng_init->size(), should be " + std::to_string(rng_init.size()));
    Buffer_Routines::if_false_throw(ccms.size() == p.N_graphs, "ccm->size() != p.N_graphs");
    q.wait();


    auto N_pop_max = std::max_element(vcms.data.begin(), vcms.data.end(), [](const auto& v0, const auto& v1){return v0.size() < v1.size();})->size();
    Buffer_Routines::validate_buffer_elements<uint32_t, 2>(q, *vcm, [](auto elem){return (elem >= 0) && (elem <= 20);}).wait();
    Buffer_Routines::validate_buffer_elements<uint32_t, 1>(q, *ecm, [](auto elem){return (elem >= 0) && (elem <= 200);}).wait();
    Buffer_Routines::validate_buffer_elements<uint32_t, 1>(q, *edge_from, [=](auto elem){return (elem >= 0) && (elem <= N_pop_max);}).wait();
    Buffer_Routines::validate_buffer_elements<uint32_t, 1>(q, *edge_to, [=](auto elem){return (elem >= 0) && (elem <= N_pop_max);}).wait();
    Buffer_Routines::validate_buffer_elements<uint32_t, 1>(q, *edge_offsets, [=](auto elem){return (elem >= 0) && (elem <= N_tot_edges);}).wait();
    Buffer_Routines::validate_buffer_elements<uint32_t, 1>(q, *N_connections, [=](auto elem){return (elem >= 0) && (elem <= N_connections_max);}).wait();
    Buffer_Routines::validate_buffer_elements<uint32_t, 3>(q, *accumulated_events, [=](auto elem){return (elem >= 0) && (elem <= N_pop_max);}).wait();
    Buffer_Routines::validate_buffer_elements<float, 3>(q, *p_Is, [=](auto elem){return (elem >= 0) && (elem <= 1);}).wait();

}
