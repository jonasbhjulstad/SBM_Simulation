#ifndef SIR_SBM_NETWORK_HPP
#define SIR_SBM_NETWORK_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <execution>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stddef.h>
#include <tuple>
#include <Sycl_Graph/SBM_types.hpp>
namespace Sycl_Graph
{
    struct Edge_t
    {
        uint32_t from;
        uint32_t to;
    };

    template <sycl::access_mode Mode, sycl::access::target Target = sycl::access::target::device>
    struct Edge_Accessor_t
    {
        Edge_Accessor_t(sycl::handler &h) : to(h), from(h), self(h) {}
        sycl::accessor<uint32_t, 1, Mode, Target> to;
        sycl::accessor<uint32_t, 1, Mode, Target> from;
        sycl::accessor<uint32_t, 1, Mode, Target> self;
    };

    struct Edge_Buffer_t
    {

        Edge_Buffer_t(uint32_t N_edges, uint32_t N_communities)
            : to((sycl::range<1>(N_edges))),
              from((sycl::range<1>(N_edges))),
              self((sycl::range<1>(N_communities)))
        {
            auto to_acc = to.template get_access<sycl::access::mode::write>();
            auto from_acc = from.template get_access<sycl::access::mode::write>();
            auto self_acc = self.template get_access<sycl::access::mode::write>();
            for (uint32_t i = 0; i < N_edges; i++)
            {
                to_acc[i] = invalid_id;
                from_acc[i] = invalid_id;
            }
            for (uint32_t i = 0; i < N_communities; i++)
            {
                self_acc[i] = invalid_id;
            }
        }
        Edge_Buffer_t(uint32_t N_edges) : Edge_Buffer_t(N_edges, 1) {}
        sycl::buffer<uint32_t, 1> to;
        sycl::buffer<uint32_t, 1> from;
        sycl::buffer<uint32_t, 1> self;
        static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
        template <sycl::access_mode Mode>
        auto get_access(sycl::handler &h)
        {
            return Edge_Accessor_t<Mode>(h);
        }

        sycl::event fill(uint32_t val, sycl::queue &q)
        {
            q.submit([&](sycl::handler &h)
                     {
      auto to_acc = to.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      auto from_acc = from.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      auto self_acc = self.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      h.parallel_for(to_acc.size(), [=](sycl::id<1> i){
        to_acc[i] = val;
        from_acc[i] = val;
      }); });
            q.submit([&](sycl::handler &h)
                     {
      auto self_acc = self.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
      h.parallel_for(self_acc.size(), [=](sycl::id<1> i){
        self_acc[i] = val;
      }); });
        }
    };

    uint32_t get_susceptible_id_if_infected(auto &v_acc, uint32_t id_from,
                                            uint32_t id_to)
    {
        if ((v_acc[id_to] == SIR_INDIVIDUAL_S) &&
            (v_acc[id_from] == SIR_INDIVIDUAL_I))
            || ((v_acc[id_from] == SIR_INDIVIDUAL_S) &&
                (v_acc[id_to] == SIR_INDIVIDUAL_I))
            {
                return id_to;
            }
        else
        {
            return std::numeric_limits<uint32_t>::max();
        }
    }

    struct SIR_SBM_Network
    {

        SIR_SBM_Network(const SBM_Graph_t &G, float p_I0, float p_R, sycl::queue &q, uint32_t seed = 52, float p_R0 = .0f)
            : N_communities(G.N_communities()),
              N_vertices(G.N_vertices()),
              N_edges(G.N_edges()),
              N_connections(G.N_connections()),
              vertices(sycl::range<1>(N_vertices)),
              edge_connection_map(sycl::range<1>(N_edges)),
              edges(G.create_edge_buffer()),
              edge_community_map(sycl::range<1>(N_edges)),
              vertex_community_map(sycl::range<1>(N_vertices)),
              infection_events_buf(N_edges, 1), q(q),
              rng(seed), p_R(p_R), ecm(sycl::range<1>(N_edges)),
              vcm(sycl::range<1>(N_vertices))
        {
            seed_buf = generate_seeds(N_edges, q, seed);
            init_events.push_back(initialize_vertices(p_I0, p_R0, N_vertices, q, seed_buf));
            init_events.push_back(create_ecm(G));
            init_events.push_back(create_vcm(G.node_list));
        }
        uint32_t N_communities;
        uint32_t N_connections;
        uint32_t N_vertices;
        uint32_t N_edges;
        const float p_R;
        const float p_I0;
        const float p_R0;
        sycl::queue &q;
        sycl::buffer<Edge_t> edges;
        std::vector<sycl::buffer<uint32_t>> trajectory;
        std::vector<sycl::buffer<bool>> infection_events;
        sycl::buffer<uint32_t> seed_buf;
        sycl::buffer<uint32_t> ecm;
        sycl::buffer<uint32_t> vcm;
        std::vector<sycl::event> init_events;

        auto reset()
        {
            return initialize
        }

        auto initialize_vertices(float p_I0, float p_R0, uint32_t N, sycl::queue &q,
                                 sycl::buffer<uint32_t, 1> &seed_buf)
        {
            if (trajectory.size() == 0)
            {
                trajectory.push_back(sycl::buffer<uint32_t, 1>(sycl::range<1>(N)));
            }
            return q.submit([&](sycl::handler &h)
                            {
            auto state_acc = trajectory[0].template get_access<sycl::access::mode::write,
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

        sycl::event create_ecm(const SBM_Graph_t &G)
        {
            std::vector<std::vector<uint32_t>> fills(G.edge_lists.size());
            std::transform(G.edge_lists.begin(), G.edge_lists.end(), fills.begin(), [n = 0](auto &v) mutable
                           { std::vector<uint32_t> res(v.size(), n);
                        n++;
                        return res; });
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
                h.parallel_for(ecm.size(), [=](sycl::id<1> i)
                               { ecm_acc[i] = tmp_acc[i]; }); });
        }

        sycl::event create_vcm(const std::vector<std::vector<uint32_t>> &node_lists)
        {
            uint32_t N_nodes =
                std::accumulate(node_lists.begin(), node_lists.end(), 0,
                                [](auto acc, const auto &el)
                                { return acc + el.size(); });
            std::vector<uint32_t> vcm;
            vcm.reserve(N_nodes);
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
                h.parallel_for(vcm.size(), [=](sycl::id<1> i)
                               { vcm_acc[i] = tmp_acc[i]; }); });
        }

        sycl::buffer<uint32_t, 1> generate_seeds(uint32_t N_rng, sycl::queue &q,
                                                 unsigned long seed = 42)
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

        sycl::event infect(sycl::buffer<SIR_State> &state, sycl::buffer<SIR_State> &state_next, sycl::buffer<bool> &infection_event_buf, sycl::buffer<Infection_Events_t> &connection_inf_buf, sycl::buffer<float> &p_I, auto &dep_event)
        {

            return q.submit([&](sycl::handler &h)
                            {
                h.depends_on(dep_event);
                auto ecm_acc = ecm_buf.template get_access<sycl::access::mode::read,
                                                    sycl::access::target::device>(h);
                auto p_I_acc = p_I.template get_access<sycl::access::mode::read,
                                                    sycl::access::target::device>(h);
                auto seed_acc =
                        seed_buf.template get_access<sycl::access_mode::read_write,
                                    sycl::access::target::device>(h);
                auto p_I_acc = p_I_buf.get_access<sycl::access::mode::read,
                                    sycl::access::target::device>(h);
                auto v_acc = state.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto v_next_acc = state_next.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                auto edge_acc = edges.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto connection_inf_acc = connection_inf_buf.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                h.parallel_for(edge_acc.size(), [=](sycl::id<1> id){
                    auto id_from = edge_acc[id].from;
                    auto id_to = edge_acc[id].to;
                    auto sus_id = get_susceptible_id_if_infected(v_acc, id_from, id_to);
                    if(sus_id != std::numeric_limits<uint32_t>::max())
                    {
                        Static_RNG::default_rng rng(seed_acc[id]);
                        seed_acc[id]++;
                        auto p_I = p_I_acc[ecm_acc[id]]
                        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                        if(bernoulli_I(rng))
                        {
                            v_next_acc[sus_id] = SIR_INDIVIDUAL_I;
                        }
                        if (sus_id == id_from)
                        {
                            connection_inf_acc[id].from++;
                        }
                        else
                        {
                            connection_inf_acc[id].to++;
                        }

                    }
                }); });
        }
        sycl::event recover(sycl::buffer<SIR_State> &state, sycl::buffer<SIR_State> &state_next, auto &dep_event)
        {
            float p_R = this->p_R;
            return q.submit([&](sycl::handler &h)
                            {
                h.depends_on(dep_event);
                auto seed_acc =
                        seed_buf.template get_access<sycl::access_mode::read_write,
                                    sycl::access::target::device>(h);
                auto v_acc = state.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto v_next_acc = state_next.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                h.parallel_for(v_acc.size(), [=](sycl::id<1> i){
                    if(v_acc[i] == SIR_INDIVIDUAL_I)
                    {
                        Static_RNG::default_rng rng(seed_acc[i]);
                        seed_acc[i]++;
                        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R_acc[0]);
                        if(bernoulli_R(rng))
                        {
                            v_next_acc[i] = SIR_INDIVIDUAL_R;
                        }
                    }
                }); });
        }

        sycl::event advance(sycl::buffer<SIR_State> &state, sycl::buffer<SIR_State> &state_next, sycl::buffer<bool> &infection_event_buf, sycl::buffer<float> &p_I, auto &dep_event)
        {
            auto rec_event = recover(state, state_next, dep_event);
            return infect(state, state_next, infection_event_buf, p_I, rec_event);
        }

        struct State_t : public std::vector<uint32_t>
        {
            State_t() : std::vector<uint32_t>(3) {}
        };
        typedef std::vector<State_t> Community_States_t;
        typedef std::vector<Community_States_t> Community_Trajectory_t;

        struct Infection_Events_t
        {
            uint32_t from = 0;
            uint32_t to = 0;
        };

        Community_States_t read_state(sycl::buffer<SIR_State> &state_prev, sycl::buffer<SIR_State> &state, sycl::buffer<bool> &infection_event_buf)
        {
            q.submit([&](sycl::handler &h)
                     {
                auto v_acc = state.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto vcm_acc = vcm_buf.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto state_acc = state.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                h.parallel_for(state_acc.size(), [=](sycl::id<1> id){
                    state_acc[id] = {0,0,0};
                    for(int i = 0; i < N_vertices; i++)
                    {
                        auto community_idx = vcm_acc[i];
                        state_acc[community_idx][v_acc[i]]++;
                    }
                }); })
                .wait();

            auto state_acc = state_buf.template get_access<sycl::access::mode::read, sycl::access::target::host>();
            Community_States_t states(N_communities);
            for (int i = 0; i < N_communities; i++)
            {
                states[i][0] = state_acc[i][0];
                states[i][1] = state_acc[i][1];
                states[i][2] = state_acc[i][2];
            }
            return states;
        }

        std::vector<Infection_Events_t> read_infection_events(sycl::buffer<bool> &infection_event_buf)
        {
            auto infection_event_acc = infection_event_buf.template get_access<sycl::access::mode::read, sycl::access::target::host>();
            std::vector<Infection_Events_t> infection_events(N_edges);
            for (int i = 0; i < N_edges; i++)
            {
                if (infection_event_acc[i])
                {
                    auto edge = edges[i];
                    infection_events[i].from = edge.from;
                    infection_events[i].to = edge.to;
                }
            }
            return infection_events;
        }

        auto simulate(const SIR_SBM_Param_t &param)
        {
            trajectory.resize(Nt + 1, sycl::buffer<uint32_t>(sycl::range<1>(N_vertices)));
            infection_events.resize(Nt, sycl::buffer<bool>(sycl::range<1>(N_edges)));
            for (int i = 0; i < Nt; i++)
            {
                auto advance_event = advance(trajectory[i], trajectory[i + 1], infection_events[i], param.p_Is[i], i == 0 ? init_events : advance_event);
            }

            Community_Trajectory_t community_trajectory(Nt + 1);
            std::transform(trajectory.begin(), trajectory.end(), community_trajectory.begin(), [&](auto &state)
                           { return read_state(state); });
            std::vector<std::vector<Infection_Events_t>> infection_events_trajectory(Nt);
            std::transform(infection_events.begin(), infection_events.end(), infection_events_trajectory.begin(), [&](auto &infection_event)
                           { return read_infection_events(infection_event); });
            return std::make_pair(community_trajectory, infection_events_trajectory);
        }
    }

void linewrite(std::ofstream &file, const std::vector<uint32_t> &iter)
    {
        // std::for_each(iter.begin(), iter.end(),
        //               [&](auto &t_i_i) { file << t_i_i << ","; });
        for (auto &t_i_i : iter)
        {
            file << t_i_i;
            if (&t_i_i != &iter.back())
                file << ",";
            else
                file << "\n";
        }
    }

    void linewrite(std::ofstream &file, const std::vector<Edge_t> &iter)
    {
        for (auto &t_i_i : iter)
        {
            file << t_i_i.from << "," << t_i_i.to;
            if (&t_i_i != &iter.back())
                file << ",";
            else
                file << "\n";
        }
    }
    void simulate_to_file(const SBM_Graph_t &G, const SIR_SBM_Param_t &param,
                          sycl::queue &q, const std::string &file_path,
                          uint32_t sim_idx, uint32_t seed = 42)
    {
        SIR_SBM_Network(G, param.p_I0, param.p_R, q, seed, param.p_R0);
        auto [community_trajectory, infection_events_trajectory] = simulate(param);
        std::ofstream community_traj_f(file_path + "community_trajectory_" + std::to_string(sim_idx) + ".csv");
        std::ofstream infection_events_f(file_path + "infection_events_" + std::to_string(sim_idx) + ".csv");
        linewrite(community_traj_f, community_trajectory[i]);
        for (int i = 0; i < Nt; i++)
        {
            linewrite(community_traj_f, community_trajectory[i+1]);
            linewrite(infection_events_f, infection_events_trajectory[i]);
        }
    }

    void
    parallel_simulate_to_file(const SBM_Graph_t &G, const std::vector<SIR_SBM_Param_t> &params,
                              std::vector<sycl::queue> &qs, const std::string &file_path, uint32_t N_sim, uint32_t seed = 42)
    {

        uint32_t N_sims = params.size();
        std::vector<uint32_t> seeds(N_sims);
        Static_RNG::default_rng rng(seed);
        std::generate(seeds.begin(), seeds.end(), [&rng]()
                      { return (uint32_t)rng(); });
        std::vector<SBM_Graph_t> Gs(N_sim, G);

        std::vector<std::tuple<const SBM_Graph_t *, const SIR_SBM_Param_t *, sycl::queue *, const std::string, uint32_t, uint32_t>> zip;
        for (uint32_t i = 0; i < N_sim; i++)
        {
            zip.push_back(std::make_tuple(&Gs[i], &params[i], &qs[i], file_path, i, seeds[i]));
        }

        std::for_each(std::execution::par_unseq, zip.begin(), zip.end(),
                      [&](auto z)
                      {
                          simulate_to_file(*std::get<0>(z), *std::get<1>(z), *std::get<2>(z), std::get<3>(z), std::get<4>(z), std::get<5>(z));
                      });
    }

    void parallel_simulate_to_file(const std::vector<SBM_Graph_t> &Gs, const std::vector<std::vector<SIR_SBM_Param_t>> &params,
                                   std::vector<std::vector<sycl::queue>> &qs, const std::vector<std::string> &file_paths, uint32_t seed = 42)
    {

        std::vector<uint32_t> seeds(Gs.size());
        std::mt19937 rd(seed);
        std::generate(seeds.begin(), seeds.end(), [&rd]()
                      { return rd(); });
        std::vector<uint32_t> N_sims(Gs.size());
        std::transform(params.begin(), params.end(), N_sims.begin(), [](auto p)
                       { return p.size(); });

        // zip
        std::vector<std::tuple<const SBM_Graph_t *, const std::vector<SIR_SBM_Param_t> *, std::vector<sycl::queue> *, std::string, uint32_t, uint32_t>> zip(Gs.size());
        for (uint32_t i = 0; i < Gs.size(); i++)
        {
            zip[i] = std::make_tuple(&Gs[i], &params[i], &qs[i], file_paths[i], N_sims[i], seeds[i]);
        }

        std::for_each(std::execution::par_unseq, zip.begin(), zip.end(),
                      [&](const auto &z)
                      {
                          parallel_simulate_to_file(*std::get<0>(z), *std::get<1>(z), *std::get<2>(z), std::get<3>(z), std::get<4>(z), std::get<5>(z));
                      });
    }

}

#endif