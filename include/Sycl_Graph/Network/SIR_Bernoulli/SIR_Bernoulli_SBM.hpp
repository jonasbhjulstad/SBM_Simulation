#ifndef SIR_Bernoulli_SBM_SYCL_IMPL_HPP
#define SIR_Bernoulli_SBM_SYCL_IMPL_HPP
#include "SIR_Bernoulli_SBM_Types.hpp"
#include <stddef.h>
#include <sycl/sycl.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Static_RNG/distributions.hpp>
#ifdef SYCL_GRAPH_USE_ONEAPI
#include <oneapi/dpl/algorithm>
#endif
#include <utility>
#include <type_traits>
template <>
struct sycl::is_device_copyable<Sycl_Graph::Network_Models::SIR_Edge> : std::true_type
{
};

template <> 
struct sycl::is_device_copyable<Sycl_Graph::Network_Models::SIR_Individual_State>: std::true_type {};
namespace Sycl_Graph
{
    namespace Sycl::Network_Models
    {
        // sycl::is_device_copyable_v<SIR_Individual_State> is_copyable_SIR_Invidual_State;
        using namespace Sycl_Graph::Network_Models;
        template <typename T>
        using SIR_vector_t = std::vector<T, std::allocator<T>>;
        using SIR_Graph =
            Sycl_Graph::Sycl::Graph<SIR_Individual_State, SIR_Edge, uint32_t>;

        struct SIR_Bernoulli_SBM_Network
            : public Network<SIR_Bernoulli_SBM_Network, std::vector<uint32_t>, SIR_Bernoulli_SBM_Temporal_Param<>>
        {
            using Graph_t = SIR_Graph;
            using Vertex_t = typename Graph_t::Vertex_t;
            using Edge_t = typename Graph_t::Edge_t;
            using Base_t = Network<SIR_Bernoulli_SBM_Network, std::vector<uint32_t>, SIR_Bernoulli_SBM_Temporal_Param<>>;
            const float p_I0;
            const float p_R0;
            sycl::buffer<int, 1> seed_buf;
            bool record_group_infections = true;

            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> SBM_ids;

            // SIR_Bernoulli_SBM_Network(): p_R0(0), p_I0(0),seed_buf(sycl::range<1>(0)), G(q, 0, 0) {}

            SIR_Bernoulli_SBM_Network(const Graph_t &G, float p_I0, float p_R0, auto& SBM_ids, int seed = 777)
                : q(G.q), G(G), p_I0(p_I0), p_R0(p_R0), seed_buf(sycl::range<1>(G.N_vertices())), SBM_ids(SBM_ids)
            {
                assert(G.N_vertices() > 0 && "Graph must have at least one vertex");
                // generate G.N_vertices() random numbers
                // create rng
                std::mt19937 rng(seed);
                std::uniform_int_distribution<int> dist(0, 1000000);
                std::vector<int> seeds(G.N_vertices());
                std::generate(seeds.begin(), seeds.end(), [&]()
                              { return dist(rng); });
                // copy seeds to buffer
                sycl::buffer<int, 1> seeds_buf(seeds.data(), sycl::range<1>(G.N_vertices()));
                q.submit([&](sycl::handler &h)
                         {
                        auto seeds = seeds_buf.get_access<sycl::access::mode::read>(h);
                        auto seed = seed_buf.get_access<sycl::access::mode::write>(h);
                        h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> i) {
                            seed[i] = seeds[i];
                        }); });
            }

            uint32_t N_communities() const
            {
                //inverse of n(n−1)/2
                return (1 + std::sqrt(1 + 8 * SBM_ids.size())) / 2;
                // return SBM_ids.size();
            }
            void initialize()
            {

                const float p_I0 = this->p_I0;
                const float p_R0 = this->p_R0;
                
                
                // generate seeds for
                sycl::buffer<float, 1> p_I_buf(sycl::range<1>(G.N_vertices()));
                sycl::buffer<float, 1> p_R_buf(sycl::range<1>(G.N_vertices()));
                std::vector<SIR_Individual_State> v_host(G.N_vertices());
                sycl::buffer<SIR_Individual_State, 1> v_buf(v_host.data(), sycl::range<1>(G.N_vertices()));

                q.submit([&](sycl::handler &h)
                         {
                    auto seed = seed_buf.get_access<sycl::access::mode::read>(h);
                    auto p_I = p_I_buf.get_access<sycl::access::mode::write>(h);
                    auto p_R = p_R_buf.get_access<sycl::access::mode::write>(h);
                    auto v_acc = G.get_vertex_access<sycl::access::mode::write>(h);
                    h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> i) {
                    Static_RNG::uniform_real_distribution<float> d_I;
                    Static_RNG::uniform_real_distribution<float> d_R;
                    Static_RNG::default_rng rng(seed[i]);

                    if(d_I(rng) < p_I0)
                        {
                            v_acc.data[i] = SIR_INDIVIDUAL_I;
                        }
                        else if(d_R(rng) < p_R0)
                        {
                            v_acc.data[i] = SIR_INDIVIDUAL_R;
                        }
                        else
                        {
                            v_acc.data[i] = SIR_INDIVIDUAL_S;
                        }
                }); });
            }

            std::vector<uint32_t> read_state(const SIR_Bernoulli_SBM_Temporal_Param<>& tp)
            {
                std::vector<uint32_t> count(3, 0);
                sycl::buffer<uint32_t, 1> count_buf(count.data(), sycl::range<1>(3));
                const uint32_t N_vertices = G.N_vertices();

                q.submit([&](sycl::handler &h)
                         {
                             auto count_acc = count_buf.get_access<sycl::access::mode::write>(h);
                             auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);

                             h.single_task([=]
                                           {
                    for (int i = 0; i < N_vertices; i++)
                    {   
                        auto v_i = v_acc.data[i];
                        if (v_i == SIR_INDIVIDUAL_S)
                        {
                            count_acc[0]++;
                        }
                        else if (v_i == SIR_INDIVIDUAL_I)
                        {
                            count_acc[1]++;
                        }
                        else if (v_i == SIR_INDIVIDUAL_R)
                        {
                            count_acc[2]++;
                        }
                    } }); });
                q.wait();
                sycl::host_accessor count_acc(count_buf, sycl::read_only);
                // read into vector
                std::vector<uint32_t> res(3);
                for (int i = 0; i < 3; i++)
                {
                    res[i] = count_acc[i];
                }

                return res;
            }

            // function for infection step
            uint32_t infection_step(float p_I, const std::vector<std::pair<uint32_t, uint32_t>>& SBM_connection)
            {
                using Sycl_Graph::Network_Models::SIR_Individual_State;

                sycl::buffer<std::pair<uint32_t, uint32_t>> SBM_id_buf(SBM_connection.data(), sycl::range<1>(SBM_connection.size()));

                auto sus_inf_edge_comp([&](auto &v_acc, const uint32_t e0_to, const uint32_t e0_from, const uint32_t e1_to, const uint32_t e1_from)
                                       {
                                           // if invalid
                                           if (e0_to == Graph_t::invalid_id || e0_from == Graph_t::invalid_id)
                                               return false;
                                           if (e1_to == Graph_t::invalid_id || e1_from == Graph_t::invalid_id)
                                               return true;

                                           SIR_Individual_State v0_data[2] = {v_acc[e0_from], v_acc[e0_to]};
                                           SIR_Individual_State v1_data[2] = {v_acc[e1_from], v_acc[e1_to]};
                                           bool e0_valid = (v0_data[0] == SIR_INDIVIDUAL_S && v0_data[1] == SIR_INDIVIDUAL_I) || (v0_data[0] == SIR_INDIVIDUAL_I && v0_data[1] == SIR_INDIVIDUAL_S);
                                           bool e1_valid = (v1_data[0] == SIR_INDIVIDUAL_S && v1_data[1] == SIR_INDIVIDUAL_I) || (v1_data[0] == SIR_INDIVIDUAL_I && v1_data[1] == SIR_INDIVIDUAL_S);

                                           if (e0_valid && e1_valid){
                                               uint32_t e0_sus_idx = (v0_data[0] == SIR_INDIVIDUAL_S) ? e0_from : e0_to;
                                                uint32_t e1_sus_idx = (v1_data[0] == SIR_INDIVIDUAL_S) ? e1_from : e1_to;
                                           return e0_sus_idx > e1_sus_idx;
                                           }
                                           else if (e0_valid) return true;
                                           else return false; });
                auto get_susceptible_neighbor([&](auto &v_acc, const uint32_t e_to, const uint32_t e_from)
                                              {
                    SIR_Individual_State v_data[2] = {v_acc[e_from], v_acc[e_to]};
                    if (v_data[0] == SIR_INDIVIDUAL_S && v_data[1] == SIR_INDIVIDUAL_I)
                        return e_from;
                    else if (v_data[0] == SIR_INDIVIDUAL_I && v_data[1] == SIR_INDIVIDUAL_S)
                        return e_to;
                    else
                        return Graph_t::invalid_id; });

                if (G.N_edges() > 0)
                {
                sycl::buffer<bool> sn_buf(sycl::range<1>(G.N_vertices()));
                q.submit([&, this](sycl::handler &h)
                         {
                    auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
                    auto e_SBM_acc = SBM_id_buf.get_access<sycl::access::mode::read>(h);
                    auto sn_acc = sn_buf.get_access<sycl::access::mode::write>(h);
                    // parallel for

                    h.parallel_for<class edge_validity>(sycl::range<1>(e_SBM_acc.size()), [=](sycl::id<1> index)
                                                       {
                        // get the index of the element to sort
                        int i = index[0];
                        uint32_t sn = get_susceptible_neighbor(v_acc.data, e_SBM_acc[i].first, e_SBM_acc[i].second);
                        if (sn != Graph_t::invalid_id)
                        {sn_acc[sn] = true;
                        }
                    }); }).wait();

                uint32_t count = 0;
                sycl::buffer<uint32_t> sn_count_buf(&count, sycl::range<1>(1));

                q.submit([&](sycl::handler &h)
                         {
                    auto sn_count_acc = sn_count_buf.get_access<sycl::access::mode::write>(h);
                    auto sn_acc = sn_buf.get_access<sycl::access::mode::read>(h);
                    h.single_task([=]()
                    {
                        for (int i = 0; i < sn_acc.size(); i++)
                        {
                            if (sn_acc[i])
                            {
                                sn_count_acc[0]++;
                            }
                        }
                    }); });
                q.wait();

                //read count from buffer
                auto count_acc = sn_count_buf.get_access<sycl::access::mode::read>();
                count = count_acc[0];

                if (count > 0)
                {
                // get neighbor count
                sycl::buffer<uint32_t> sn_ids_buf((sycl::range<1>(count)));

                q.submit([&](sycl::handler &h)
                         {
                    auto sn_ids_acc = sn_ids_buf.get_access<sycl::access::mode::write>(h);
                    auto sn_acc = sn_buf.get_access<sycl::access::mode::read>(h);
                    h.single_task([=]()
                    {
                        int j = 0;
                        for (int i = 0; i < sn_acc.size(); i++)
                        {
                            if (sn_acc[i])
                            {
                                sn_ids_acc[j] = i;
                                j++;
                            }
                        }
                    }); });

                auto state_old = read_state({});


                q.submit([&](sycl::handler &h)
                         {
                    auto sn_ids_acc = sn_ids_buf.get_access<sycl::access::mode::read>(h);
                    auto v_acc = G.get_vertex_access<sycl::access_mode::read_write>(h);
                    auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(sycl::range<1>(sn_ids_acc.size()), [=](sycl::id<1> id)
                    {
                        Static_RNG::default_rng rng(seed_acc[id]);
                        Static_RNG::uniform_real_distribution<float> d_R(0, 1);
                        if (v_acc.data[sn_ids_acc[id[0]]] == SIR_INDIVIDUAL_S)
                        {
                            if (d_R(rng) < p_I)
                            {
                                v_acc.data[sn_ids_acc[id[0]]] = SIR_INDIVIDUAL_I;
                            }
                        }
                        seed_acc[id] += 1;
                    }); });
                q.wait();
                auto state_new = read_state({});
                return state_new[1] - state_old[1];
                }
                else
                {
                    return 0;
                }
                }
                return 0;
            }

            void recovery_step(float p_R)
            {

                q.submit([&](sycl::handler &h)
                         {
                    auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write>(h);
                    auto v_acc = G.get_vertex_access<sycl::access::mode::write>(h);
                    
                    sycl::stream out(1024, 256, h);
                    //  auto nv = neighbors_buf.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id)
                                   {
                        Static_RNG::default_rng rng(seed_acc[id]);
                        seed_acc[id]++;
                        Static_RNG::uniform_real_distribution<float> d_R(0, 1);
                            if(v_acc.data[id] == SIR_INDIVIDUAL_I)
                            {
                                if(d_R(rng) < p_R)
                                {
                                    v_acc.data[id] = SIR_INDIVIDUAL_R;
                                }
                            } }); });
                q.wait();
            }
            void advance(const SIR_Bernoulli_SBM_Temporal_Param<> &p)
            {
                for (int i = 0; i < SBM_ids.size(); i++)
                {
                    infection_step(p.p_Is[i], SBM_ids[i]);
                }
                recovery_step(p.p_R);
            }

            void advance(const SIR_Bernoulli_SBM_Temporal_Param<> &p, std::vector<uint32_t>& N_infected)
            {

                for (int i = 0; i < SBM_ids.size(); i++)
                {
                    N_infected[i] = infection_step(p.p_Is[i], SBM_ids[i]);
                }
                recovery_step(p.p_R);
            }

            bool terminate(const std::vector<uint32_t> &x, const SIR_Bernoulli_SBM_Temporal_Param<> &p)
            {
                static int t = 0;
                bool early_termination = ((t > p.Nt_min) && (x[1] < p.N_I_min));
                return early_termination;
            }

            void reset()
            {
                q.submit([&](sycl::handler &h)
                         {
                    auto v_acc = G.get_vertex_access<sycl::access::mode::write>(h);
        
                    h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id)
                                   {
                                       v_acc.data[id[0]] = SIR_INDIVIDUAL_S;
                                   }); });
            }

            SIR_Bernoulli_SBM_Network& operator=(SIR_Bernoulli_SBM_Network other)
            {
                q = other.q;
                std::swap(G, other.G);
                std::swap(SBM_ids, other.SBM_ids);
                std::swap(seed_buf, other.seed_buf);
                return *this;
            }

        using Base_t::simulate;

        typedef std::vector<std::vector<uint32_t>> Trajectory_t;
        typedef std::pair<Trajectory_t, Trajectory_t> Trajectory_pair_t;

        Trajectory_pair_t simulate_groups(const std::vector<SIR_Bernoulli_SBM_Temporal_Param<>> tp)
        {

            assert(tp[0].p_Is.size() == SBM_ids.size() && "Must have one p_I for each SBM_id_group");
            auto Nt = tp.size()-1;
            std::vector<std::vector<uint32_t>> group_trajectory(Nt);
            //resize group_trajectory
            for (int i = 0; i < Nt; i++)
            {
                group_trajectory[i].resize(SBM_ids.size(), 0);
            }
            std::vector<std::vector<uint32_t>> trajectory(Nt+1);
            uint32_t t = 0;
            auto tp_i = tp[0];
            trajectory[0] = read_state(tp[0]);
            for (int i = 0; i < Nt; i++)
            {
                auto tp_i = tp[i+1];
                advance(tp_i, group_trajectory[i]);
                trajectory[i+1] = read_state(tp_i);
                if (terminate(trajectory[i+1], tp_i))
                {
                    break;
                }
            }
            return std::make_pair(trajectory, group_trajectory);
        }
        auto byte_size() const { return G.byte_size(); }

            Graph_t G;
        private:
            sycl::queue &q;
        };
    } // namespace Sycl
} // namespace Network_Models
#endif