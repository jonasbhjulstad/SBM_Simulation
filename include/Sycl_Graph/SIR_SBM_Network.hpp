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
namespace Sycl_Graph::SBM
{

    uint32_t get_susceptible_id_if_infected(auto &v_acc, uint32_t id_from,
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

    template <typename T>
    sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &vec,
                               sycl::queue &q)
    {
        assert(buf.size() == vec.size());
        sycl::buffer<T, 1> tmp(vec.data(), sycl::range<1>(vec.size()));
        return q.submit([&](sycl::handler &h)
                        {
    auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    h.copy(tmp_acc, acc); });
    }

    template <typename T>
    sycl::event copy_to_buffer(sycl::buffer<T, 1> &buf, const std::vector<T> &&vec,
                               sycl::queue &q)
                               {
                                return copy_to_buffer(buf, vec, q);
                               }


    struct SIR_SBM_Network
    {

        SIR_SBM_Network(const SBM_Graph_t &G, float p_I0, float p_R, sycl::queue &q, uint32_t seed = 52, float p_R0 = .0f)
            : N_communities(G.N_communities()),
              N_vertices(G.N_vertices()),
              N_edges(G.N_edges()),
              N_connections(G.N_connections()),
              ecm_buf(sycl::range<1>(G.N_edges())),
              vcm_buf(sycl::range<1>(G.N_vertices())),
              edges(G.create_edge_buffer()),
              q(q), p_R(p_R), p_I0(p_I0), p_R0(p_R0),
              seed_buf(generate_seeds(N_edges, q, seed))
        {
            // init_events.push_back(initialize_vertices(p_I0, p_R0, N_vertices, q, seed_buf));
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
        std::vector<sycl::buffer<SIR_State>> trajectory;
        std::vector<sycl::buffer<Edge_t>> connection_events;
        sycl::buffer<uint32_t> seed_buf;
        sycl::buffer<uint32_t> ecm_buf;
        sycl::buffer<uint32_t> vcm_buf;
        std::vector<sycl::event> init_events;

        sycl::event initialize_vertices(float p_I0, float p_R0, uint32_t N, sycl::queue &q,
                                        sycl::buffer<uint32_t, 1> &seed_buf, sycl::buffer<SIR_State>& buf)
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

        std::vector<sycl::event> remap(const std::vector<uint32_t> &cmap)
        {

            auto ecm_idx_map_old = create_ecm_index_map(N_communities);
            auto cmap_buf = sycl::buffer<uint32_t>(cmap.data(), sycl::range<1>(cmap.size()));

            auto vcm_remap_event = q.submit([&](sycl::handler &h)
                                            {
                auto cmap_acc = cmap_buf.template get_access<sycl::access::mode::read,
                                                            sycl::access::target::device>(h);
                auto vcm_acc =
                    vcm_buf.template get_access<sycl::access::mode::read_write,
                                                    sycl::access::target::device>(h);
                h.parallel_for(vcm_acc.size(), [=](sycl::id<1> i)
                               { vcm_acc[i] = cmap_acc[vcm_acc[i]]; }); });
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

                auto it = std::find_if(ecm_idx_map_new.begin(), ecm_idx_map_new.end(), [&](auto &&el)
                                       { return (el.community_to == new_to && el.community_from == new_from) || (el.community_to == new_from && el.community_from == new_to); });

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
                h.parallel_for(ecm_acc.size(), [=](sycl::id<1> i)
                               { ecm_acc[i] = tmp_acc[i]; }); });
            return {vcm_remap_event, ecm_remap_event};
        }

        sycl::event infect(sycl::buffer<SIR_State> &state, sycl::buffer<SIR_State> &state_next, sycl::buffer<Edge_t> &connection_events_buf, sycl::buffer<float> &p_I, auto &dep_event)
        {
            uint32_t state_size, state_next_size, connection_inf_size, p_I_size;
            state_size = state.get_count();
            state_next_size = state_next.get_count();
            connection_inf_size = connection_events_buf.get_count();
            p_I_size = p_I.get_count();

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
                auto v_acc = state.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto v_next_acc = state_next.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                auto edge_acc = edges.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                auto connection_events_acc = connection_events_buf.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                h.parallel_for(edge_acc.size(), [=](sycl::id<1> id){
                    auto id_from = edge_acc[id].from;
                    auto id_to = edge_acc[id].to;
                    auto sus_id = get_susceptible_id_if_infected(v_acc, id_from, id_to);
                    if(sus_id != std::numeric_limits<uint32_t>::max())
                    {
                        Static_RNG::default_rng rng(seed_acc[id]);
                        seed_acc[id]++;
                        auto p_I = p_I_acc[ecm_acc[id]];
                        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I);
                        if(bernoulli_I(rng))
                        {
                            v_next_acc[sus_id] = SIR_INDIVIDUAL_I;
                        }
                        if (sus_id == id_from)
                        {
                            connection_events_acc[id].from++;
                        }
                        else
                        {
                            connection_events_acc[id].to++;
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
                        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
                        if(bernoulli_R(rng))
                        {
                            v_next_acc[i] = SIR_INDIVIDUAL_R;
                        }
                    }
                }); });
        }

        sycl::event advance(sycl::buffer<SIR_State> &state, sycl::buffer<SIR_State> &state_next, sycl::buffer<Edge_t> &connection_events_buf, sycl::buffer<float> &p_I, auto &dep_event)
        {
            auto rec_event = recover(state, state_next, dep_event);
            return infect(state, state_next, connection_events_buf, p_I, rec_event);
        }

        // typedef std::array<uint32_t, 3> State_t;
        struct State_t : std::array<uint32_t, 3>
        {
            // write operator<<
            friend std::ostream &operator<<(std::ostream &os, const State_t &state)
            {
                os << "," << state[0] << "," << state[1] << "," << state[2];
                return os;
            }
        };

        typedef std::vector<State_t> Community_States_t;
        typedef std::vector<Community_States_t> Community_Trajectory_t;

        std::vector<sycl::event> accumulate_community_state(std::vector<sycl::buffer<State_t>> &result, std::vector<sycl::buffer<SIR_State>> &v_bufs, auto &dep_event)
        {
            assert(std::all_of(result.begin(), result.end(), [&](auto &buf){ return buf.size() == 3*N_communities; }));
            assert(std::all_of(v_bufs.begin(), v_bufs.end(), [&](auto &buf){ return buf.size() == N_vertices; }));
            std::vector<sycl::event> events(result.size());

            auto accumulate_timestep = [&](sycl::buffer<SIR_State> &v_buf, sycl::buffer<State_t> &res)
            {
                return q.submit([&](sycl::handler &h)
                                {
                    h.depends_on(dep_event);
                    auto v_acc = v_buf.template get_access<sycl::access::mode::read, sycl::access::target::device>(h);
                    auto result_acc = res.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                    h.parallel_for(v_acc.size(), [=](sycl::id<1> id){
                        result_acc[id][v_acc[id]]++;
                    }); });
            };
            std::transform(v_bufs.begin(), v_bufs.end(), result.begin(), events.begin(), accumulate_timestep);
            return events;
        }

        auto sim_init(const auto& p_Is)
        {
            const uint32_t Nt = p_Is.size();
            auto trajectory_buf = std::vector<sycl::buffer<SIR_State>>(Nt+1, sycl::buffer<SIR_State>(sycl::range<1>(N_vertices)));  
            auto p_I_buf = std::vector<sycl::buffer<float>>(Nt, sycl::buffer<float>(sycl::range<1>(N_connections)));
            auto connection_events_buf = std::vector<sycl::buffer<Edge_t>>(Nt, sycl::buffer<Edge_t>(sycl::range<1>(N_connections)));
            std::vector<sycl::event> ce_events(Nt);
            std::vector<sycl::event> p_I_events(Nt);

            std::transform(connection_events_buf.begin(), connection_events_buf.end(), ce_events.begin(), [&](auto &ce)
                           { return copy_to_buffer(ce, std::vector<Edge_t>(N_connections, Edge_t{0,0}), q); });
            std::transform(p_Is.begin(), p_Is.end(), p_I_buf.begin(), p_I_events.begin(), [&](const auto &p_I, auto &p_I_buf)
                           { return copy_to_buffer(p_I_buf, p_I, q); });

            std::vector<sycl::event> events;
            events.push_back(initialize_vertices(p_I0, p_R0, N_vertices, q, seed_buf, trajectory_buf[0]));
            events.insert(events.end(), ce_events.begin(), ce_events.end());
            events.insert(events.end(), p_I_events.begin(), p_I_events.end());

            return std::make_tuple(p_I_buf, connection_events_buf, trajectory_buf, events);
        }

        auto simulate(const SIR_SBM_Param_t &param)
        {
            uint32_t Nt = param.p_I.size();
            trajectory.resize(Nt + 1, sycl::buffer<SIR_State>(sycl::range<1>(N_vertices)));

            auto [p_I_buf, connection_events, trajectory, sim_init_events] = sim_init(param.p_I);
            advance(trajectory[0], trajectory[1], connection_events[0], p_I_buf[0], sim_init_events);
            sycl::event advance_event;
            for (int i = 1; i < Nt; i++)
            {
                advance_event = advance(trajectory[i], trajectory[i + 1], connection_events[i], p_I_buf[i], advance_event);
            }

            std::vector<sycl::event> accumulate_events;

            auto community_state_bufs = std::vector<sycl::buffer<State_t>>(Nt + 1, sycl::buffer<State_t>(sycl::range<1>(3*N_communities)));
            auto acs_events = accumulate_community_state(community_state_bufs, trajectory, advance_event);

            return std::make_tuple(community_state_bufs, connection_events, acs_events);
        }

    private:
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

        struct ecm_map_elem_t
        {
            uint32_t community_from = std::numeric_limits<uint32_t>::max();
            uint32_t community_to = std::numeric_limits<uint32_t>::max();
        };

        std::vector<ecm_map_elem_t>
        create_ecm_index_map(uint32_t N)
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

        sycl::event create_state_buf(sycl::buffer<SIR_State> &state_buf)
        {
            return q.submit([&](sycl::handler &h)
                            {
                auto state_acc = state_buf.template get_access<sycl::access::mode::write, sycl::access::target::device>(h);
                h.parallel_for(state_acc.size(), [=](sycl::id<1> id){
                    state_acc[id] = SIR_INDIVIDUAL_S;
                }); });
        }
    };

    template <typename T>
    void linewrite(std::ofstream &file, const std::vector<T> &iter)
    {
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

    template <typename T>
    std::vector<std::vector<T>> read_buffers(std::vector<sycl::buffer<T>> &bufs, std::vector<sycl::event> &events, uint32_t N, uint32_t N_elem)
    {
        auto data_vec = std::vector<std::vector<T>>(N_elem, std::vector<T>(N));
        std::transform(std::execution::par_unseq, bufs.begin(), bufs.end(), events.begin(), data_vec.begin(),
                       [&](auto &buf, auto &event)
                       {
                           event.wait();
                           auto buf_acc = buf.template get_access<sycl::access::mode::read>();
                           std::vector<T> res(N);
                           for (int i = 0; i < N; i++)
                           {
                               res[i] = buf_acc[i];
                           }

                           return res;
                       });
        return data_vec;
    }

    void simulate_to_file(const SBM_Graph_t &G, const SIR_SBM_Param_t &param,
                          sycl::queue &q, const std::string &file_path,
                          uint32_t sim_idx, uint32_t seed = 42)
    {
        uint32_t Nt = param.p_I.size();
        SIR_SBM_Network network(G, param.p_I0, param.p_R, q, seed, param.p_R0);
        auto [community_state_bufs, connection_events_bufs, acs_events] = network.simulate(param);

        auto community_trajectory = read_buffers(community_state_bufs, acs_events, network.N_communities, Nt + 1);

        auto connection_events_trajectory = read_buffers(connection_events_bufs, acs_events, network.N_connections, Nt);

        std::ofstream community_traj_f(file_path + "community_trajectory_" + std::to_string(sim_idx) + ".csv");
        std::ofstream connection_events_f(file_path + "connection_events_" + std::to_string(sim_idx) + ".csv");
        std::for_each(community_trajectory.begin(), community_trajectory.end(),
                      [&](auto &community_trajectory_i)
                      {
                          linewrite(community_traj_f, community_trajectory_i);
                      });
        std::for_each(connection_events_trajectory.begin(), connection_events_trajectory.end(),
                      [&](auto &connection_events_i)
                      {
                          linewrite(connection_events_f, connection_events_i);
                      });
    }

    void
    parallel_simulate_to_file(const SBM_Graph_t &G, const std::vector<SIR_SBM_Param_t> &params,
                              std::vector<sycl::queue> &qs, const std::string &file_path, uint32_t N_sim, uint32_t seed = 42)
    {
        assert(params.size() == qs.size() && "Number of parameters and queues must be equal");
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
        assert(Gs.size() == params.size() && "Number of graphs and parameters must be equal");
        assert(Gs.size() == qs.size() && "Number of graphs and queues must be equal");

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