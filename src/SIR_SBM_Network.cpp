#include <Sycl_Graph/SIR_SBM_Network.hpp>
#include <Sycl_Graph/Buffer_Routines.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <execution>
#include <iostream>
#include <memory>
#include <random>
namespace Sycl_Graph::SBM
{

    uint32_t get_susceptible_id_if_infected(sycl::accessor<SIR_State> &v_acc, uint32_t id_from,
                                            uint32_t id_to)
    {
        if (((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
             (v_acc[id_from] == SIR_INDIVIDUAL_I)) ||
            ((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
             (v_acc[id_to] == SIR_INDIVIDUAL_I)))
        {
            return id_to;
        }
        else
        {
            return std::numeric_limits<uint32_t>::max();
        }
    }

    SIR_SBM_Network::SIR_SBM_Network(const SBM_Graph_t &G, float p_I0, float p_R, sycl::queue &q,
                                     uint32_t seed, float p_R0)
        : N_communities(G.N_communities()), N_vertices(G.N_vertices()),
          N_edges(G.N_edges()), N_connections(G.N_connections()),
          ecm_buf(sycl::range<1>(G.N_edges())),
          vcm_buf(sycl::range<1>(G.N_vertices())), edges(sycl::range<1>(G.N_edges())),
          q(q), p_R(p_R), p_I0(p_I0), p_R0(p_R0),
          seed_buf(generate_seeds(N_edges, q, seed))
    {
        auto [edges, e_event] = G.create_edge_buffer(q);
        init_events.push_back(e_event);
        init_events.push_back(create_ecm(G));
        init_events.push_back(create_vcm(G.node_list));
    }

    sycl::event initialize_vertices(float p_I0, float p_R0, uint32_t N,
                                    sycl::queue &q,
                                    sycl::buffer<uint32_t, 1> &seed_buf,
                                    sycl::buffer<SIR_State> &buf)
    {
        if (trajectory.size() == 0)
        {
            trajectory.push_back(sycl::buffer<SIR_State, 1>(sycl::range<1>(N)));
        }
        return q.submit([&](sycl::handler &h)
                        {
      auto state_acc = buf.template get_access<sycl::access::mode::write,
                                               sycl::access::target::device>(h);
      auto seed_acc =
          seed_buf.template get_access<sycl::access::mode::read_write,
                                       sycl::access::target::device>(h);
      h.parallel_for(N, [=](sycl::id<1> id) {
        Static_RNG::default_rng rng(seed_acc[id]);
        seed_acc[id]++;
        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);

        if (bernoulli_I(rng)) {
          state_acc[id] = SIR_INDIVIDUAL_I;
        } else if (bernoulli_R(rng)) {
          state_acc[id] = SIR_INDIVIDUAL_R;
        } else {
          state_acc[id] = SIR_INDIVIDUAL_S;
        }
      }); });
    }

    std::vector<sycl::event> SIR_SBM_Network::remap(const std::vector<uint32_t> &cmap)
    {

        auto ecm_idx_map_old = create_ecm_index_map(N_communities);
        auto cmap_buf =
            sycl::buffer<uint32_t>(cmap.data(), sycl::range<1>(cmap.size()));

        auto vcm_remap_event = q.submit([&](sycl::handler &h)
                                        {
      auto cmap_acc =
          cmap_buf.template get_access<sycl::access::mode::read,
                                       sycl::access::target::device>(h);
      auto vcm_acc =
          vcm_buf.template get_access<sycl::access::mode::read_write,
                                      sycl::access::target::device>(h);
      h.parallel_for(vcm_acc.size(),
                     [=](sycl::id<1> i) { vcm_acc[i] = cmap_acc[vcm_acc[i]]; }); });
        N_communities = *std::max_element(cmap.begin(), cmap.end()) + 1;
        auto ecm_idx_map_new = create_ecm_index_map(N_communities);
        std::vector<uint32_t> ecm_new;
        auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read>();
        uint32_t n = 0;
        for (int i = 0; i < N_edges; i++)
        {
            auto map_elem = ecm_idx_map_old[ecm_acc[i]];
            auto old_to = map_elem.community_to;
            auto old_from = map_elem.community_from;

            auto new_to = cmap[old_to];
            auto new_from = cmap[old_from];

            auto it = std::find_if(
                ecm_idx_map_new.begin(), ecm_idx_map_new.end(), [&](auto &&el)
                { return (el.community_to == new_to &&
                          el.community_from == new_from) ||
                         (el.community_to == new_from && el.community_from == new_to); });

            assert(it != ecm_idx_map_new.end());
            ecm_new.push_back(std::distance(ecm_idx_map_new.begin(), it));
        }

        sycl::buffer<uint32_t> tmp2(ecm_new.data(), sycl::range<1>(ecm_new.size()));
        auto ecm_remap_event = q.submit([&](sycl::handler &h)
                                        {
      auto tmp_acc = tmp2.template get_access<sycl::access::mode::read,
                                              sycl::access::target::device>(h);
      auto ecm_acc =
          ecm_buf.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h);
      h.parallel_for(ecm_acc.size(),
                     [=](sycl::id<1> i) { ecm_acc[i] = tmp_acc[i]; }); });
        return {vcm_remap_event, ecm_remap_event};
    }

    sycl::event SIR_SBM_Network::infect(sycl::buffer<SIR_State> &state,
                                        sycl::buffer<SIR_State> &state_next,
                                        sycl::buffer<Edge_t> &connection_events_buf,
                                        sycl::buffer<float> &p_I, auto &dep_event)
    {
        uint32_t state_size, state_next_size, connection_inf_size, p_I_size;
        state_size = state.size();
        state_next_size = state_next.size();
        connection_inf_size = connection_events_buf.size();
        p_I_size = p_I.size();

        return q.submit([&](sycl::handler &h)
                        {
      h.depends_on(dep_event);
      auto ecm_acc =
          ecm_buf.template get_access<sycl::access::mode::read,
                                      sycl::access::target::device>(h);
      auto p_I_acc = p_I.template get_access<sycl::access::mode::read,
                                             sycl::access::target::device>(h);
      auto seed_acc =
          seed_buf.template get_access<sycl::access_mode::read_write,
                                       sycl::access::target::device>(h);
      auto v_acc = state.template get_access<sycl::access::mode::read,
                                             sycl::access::target::device>(h);
      auto v_next_acc =
          state_next.template get_access<sycl::access::mode::write,
                                         sycl::access::target::device>(h);
      auto edge_acc =
          edges.template get_access<sycl::access::mode::read,
                                    sycl::access::target::device>(h);
      auto connection_events_acc = connection_events_buf.template get_access<
          sycl::access::mode::write, sycl::access::target::device>(h);

      sycl::stream out(1024, 256, h);
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
        auto id_from = edge_acc[id].from;
        auto id_to = edge_acc[id].to;
        auto sus_id = get_susceptible_id_if_infected(v_acc, id_from, id_to);
        if (sus_id != std::numeric_limits<uint32_t>::max()) {
          Static_RNG::default_rng rng(seed_acc[id]);
          seed_acc[id]++;
          auto p_I = p_I_acc[ecm_acc[id]];
          Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
          if (bernoulli_I(rng)) {
            out << "Infecting " << sus_id << " from " << id_from << " to "
                << id_to << sycl::endl;
            v_next_acc[sus_id] = SIR_INDIVIDUAL_I;
          }
          if (sus_id == id_from) {
            connection_events_acc[id].from++;
          } else {
            connection_events_acc[id].to++;
          }
        }
      }); });
    }
    sycl::event recover(sycl::buffer<SIR_State> &state,
                        sycl::buffer<SIR_State> &state_next, auto &dep_event)
    {
        float p_R = this->p_R;
        return q.submit([&](sycl::handler &h)
                        {
      h.depends_on(dep_event);
      auto seed_acc =
          seed_buf.template get_access<sycl::access_mode::read_write,
                                       sycl::access::target::device>(h);
      auto v_acc = state.template get_access<sycl::access::mode::read,
                                             sycl::access::target::device>(h);
      auto v_next_acc =
          state_next.template get_access<sycl::access::mode::write,
                                         sycl::access::target::device>(h);
      h.parallel_for(v_acc.size(), [=](sycl::id<1> i) {
        if (v_acc[i] == SIR_INDIVIDUAL_I) {
          Static_RNG::default_rng rng(seed_acc[i]);
          seed_acc[i]++;
          Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
          if (bernoulli_R(rng)) {
            v_next_acc[i] = SIR_INDIVIDUAL_R;
          }
        }
      }); });
    }

    sycl::event SIR_SBM_Network::advance(sycl::buffer<SIR_State> &state,
                                         sycl::buffer<SIR_State> &state_next,
                                         sycl::buffer<Edge_t> &connection_events_buf,
                                         sycl::buffer<float> &p_I, auto &dep_event)
    {
        auto rec_event = recover(state, state_next, dep_event);
        return infect(state, state_next, connection_events_buf, p_I, rec_event);
    }

    std::vector<sycl::event>
    SIR_SBM_Network::accumulate_community_state(std::vector<sycl::buffer<State_t>> &result,
                                                std::vector<sycl::buffer<SIR_State>> &v_bufs,
                                                auto &dep_event)
    {
        assert(std::all_of(result.begin(), result.end(), [&](auto &buf)
                           { return buf.size() == 3 * N_communities; }));
        assert(std::all_of(v_bufs.begin(), v_bufs.end(),
                           [&](auto &buf)
                           { return buf.size() == N_vertices; }));
        std::vector<sycl::event> events(result.size());

        auto accumulate_timestep = [&, this](sycl::buffer<SIR_State> &v_buf,
                                             sycl::buffer<State_t> &res)
        {
            return q.submit([&](sycl::handler &h)
                            {
        h.depends_on(dep_event);
        auto v_acc = v_buf.template get_access<sycl::access::mode::read,
                                               sycl::access::target::device>(h);
        auto result_acc =
            res.template get_access<sycl::access::mode::write,
                                    sycl::access::target::device>(h);
        auto vcm_acc =
            vcm_buf.template get_access<sycl::access::mode::read,
                                        sycl::access::target::device>(h);
        h.parallel_for(v_acc.size(), [=](sycl::id<1> id) {
          result_acc[vcm_acc[id]][v_acc[id]]++;
        }); });
        };
        std::transform(v_bufs.begin(), v_bufs.end(), result.begin(), events.begin(),
                       accumulate_timestep);
        // print_buffer(result[0]);
        return events;
    }

    SIR_SBM_Network::init_t SIR_SBM_Network::sim_init(const std::vector<std::vector<float>> &p_Is)
    {
        const uint32_t Nt = p_Is.size();

        auto [p_I_buf, p_I_events] =
            initialize_buffer_vector(p_Is, q);
        auto [connection_events_buf, ce_events] =
            initialize_buffer_vector(N_connections, Nt, Edge_t{0, 0}, q);
        auto [trajectory_buf, t_events] =
            initialize_buffer_vector(N_vertices, Nt + 1, SIR_INDIVIDUAL_S, q);
        auto [community_state_buf, cs_events] = initialize_buffer_vector(3 * N_communities, Nt + 1, State_t{0, 0, 0}, q);

        std::vector<sycl::event> events;
        events.push_back(initialize_vertices(p_I0, p_R0, N_vertices, q, seed_buf,
                                             trajectory_buf[0]));
        events.insert(events.end(), t_events.begin(), t_events.end());
        events.insert(events.end(), ce_events.begin(), ce_events.end());
        events.insert(events.end(), p_I_events.begin(), p_I_events.end());
        events.insert(events.end(), cs_events.begin(), cs_events.end());

        return std::make_tuple(p_I_buf, connection_events_buf, trajectory_buf,
                               community_state_buf, events);
    }

    SIR_SBM_Network::trajectory_t SIR_SBM_Network::simulate(const SIR_SBM_Param_t &param)
    {
        uint32_t Nt = param.p_I.size();
        trajectory.resize(Nt + 1,
                          sycl::buffer<SIR_State>(sycl::range<1>(N_vertices)));

        auto [p_I_buf, connection_events, trajectory, community_state_buf,
              sim_init_events] = sim_init(param.p_I);
        // advance(trajectory[0], trajectory[1], connection_events[0], p_I_buf[0],
        //         sim_init_events);
        sycl::event advance_event;
        for (int i = 0; i < Nt; i++)
        {
            advance_event = advance(trajectory[i], trajectory[i + 1],
                                    connection_events[i], p_I_buf[i], advance_event);
        }

        auto acs_events = accumulate_community_state(community_state_buf,
                                                     trajectory, advance_event);

        return std::make_tuple(community_state_buf, connection_events, acs_events);
    }

    sycl::buffer<uint32_t, 1> SIR_SBM_Network::generate_seeds(uint32_t N_rng, sycl::queue &q,
                                                              unsigned long seed)
    {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint32_t> dis(0, 1000000);
        std::vector<uint32_t> rngs(N_rng);
        std::generate(rngs.begin(), rngs.end(), [&]()
                      { return dis(gen); });

        sycl::buffer<uint32_t, 1> seed_buf((sycl::range<1>(N_rng)));
        copy_to_buffer(seed_buf, rngs, q).wait();
        return seed_buf;
    }
    sycl::event SIR_SBM_Network::create_ecm(const SBM_Graph_t &G)
    {
        std::vector<std::vector<uint32_t>> fills(G.edge_lists.size());
        std::transform(G.edge_lists.begin(), G.edge_lists.end(), fills.begin(),
                       [n = 0](auto &v) mutable
                       {
                           std::vector<uint32_t> res(v.size(), n);
                           n++;
                           return res;
                       });
        std::vector<uint32_t> ecm;
        // insert all
        for (auto &v : fills)
        {
            ecm.insert(ecm.end(), v.begin(), v.end());
        }
        sycl::buffer<uint32_t> tmp(ecm.data(), sycl::range<1>(ecm.size()));

        return q.submit([&](sycl::handler &h)
                        {
      auto tmp_acc = tmp.template get_access<sycl::access::mode::read,
                                             sycl::access::target::device>(h);
      auto ecm_acc =
          ecm_buf.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h);
      h.parallel_for(ecm.size(),
                     [=](sycl::id<1> i) { ecm_acc[i] = tmp_acc[i]; }); });
    }

    sycl::event create_vcm(const std::vector<std::vector<uint32_t>> &node_lists)
    {
        std::vector<uint32_t> vcm;
        vcm.reserve(N_vertices);
        uint32_t n = 0;
        for (auto &&v_list : node_lists)
        {
            std::vector<uint32_t> vs(v_list.size(), n);
            vcm.insert(vcm.end(), vs.begin(), vs.end());
            n++;
        }
        sycl::buffer<uint32_t> tmp(vcm.data(), sycl::range<1>(vcm.size()));

        return q.submit([&](sycl::handler &h)
                        {
      auto tmp_acc = tmp.template get_access<sycl::access::mode::read,
                                             sycl::access::target::device>(h);
      auto vcm_acc =
          vcm_buf.template get_access<sycl::access::mode::write,
                                      sycl::access::target::device>(h);
      h.parallel_for(vcm.size(),
                     [=](sycl::id<1> i) { vcm_acc[i] = tmp_acc[i]; }); });
    }

    std::vector<SIR_SBM_Network::ecm_map_elem_t> SIR_SBM_Network::create_ecm_index_map(uint32_t N)
    {
        uint32_t n = 0;
        std::vector<uint32_t> vcm_indices(N);
        std::iota(vcm_indices.begin(), vcm_indices.end(), 0);
        std::vector<ecm_map_elem_t> idx_map;
        for (auto &&comb : iter::combinations(vcm_indices, 2))
        {
            idx_map.push_back({comb[0], comb[1]});
            n++;
        }
        for (uint32_t i = 0; i < N_communities; i++)
        {
            idx_map.push_back({i, i});
            n++;
        }
        return idx_map;
    }

    sycl::event SIR_SBM_Network::create_state_buf(sycl::buffer<SIR_State> &state_buf)
    {
        return q.submit([&](sycl::handler &h)
                        {
      auto state_acc =
          state_buf.template get_access<sycl::access::mode::write,
                                        sycl::access::target::device>(h);
      h.parallel_for(state_acc.size(),
                     [=](sycl::id<1> id) { state_acc[id] = SIR_INDIVIDUAL_S; }); });
    }

} // namespace Sycl_Graph::SBM
