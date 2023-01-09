#ifndef SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#define SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#ifdef SYCL_GRAPH_USE_SYCL
#include "SIR_Metapopulation_Types.hpp"
#include <stddef.h>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <oneapi/dpl/random>
#include <oneapi/dpl/algorithm>
#include <utility>
namespace Sycl_Graph
{

    namespace Sycl::Network_Models
    {
        // sycl::is_device_copyable_v<SIR_Metapopulation_State> is_copyable_SIR_Invidual_State;
        using namespace Sycl_Graph::Network_Models;
        template <typename T>
        using SIR_vector_t = std::vector<T, std::allocator<T>>;
        struct SIR_Metapopulation_Node
        {
            SIR_Metapopulation_State state;
            SIR_Metapopulation_Param param;
        };
        using SIR_Metapopulation_Graph =
            Sycl_Graph::Sycl::Graph<SIR_Metapopulation_Node, SIR_Metapopulation_Param, uint32_t>;
        struct SIR_Metapopulation_Network
            : public Network<SIR_Metapopulation_Param<>, SIR_Metapopulation_Network>
        {
            using Graph_t = SIR_Metapopulation_Graph;
            using Vertex_t = typename Graph_t::Vertex_t;
            using Edge_t = typename Graph_t::Edge_t;
            using Base_t = Network<SIR_Metapopulation_Param<>, SIR_Metapopulation_Network>;
            const uint32_t t = 0;
            sycl::buffer<int, 1> seed_buf;
            SIR_Metapopulation_Network(Graph_t &G, float E_I0, float std_I0, float E_R0, float std_R0, int seed = 777)
                : Base_t(3), q(G.q), G(G), seed_buf(sycl::range<1>(G.NV))
            {
                assert(G.NV > 0);
                // generate G.NV random numbers
                // create rng
                std::mt19937 rng(seed);
                std::uniform_int_distribution<int> dist(0, 1000000);
                std::vector<int> seeds(G.NV);
                std::generate(seeds.begin(), seeds.end(), [&]()
                              { return dist(rng); });
                // copy seeds to buffer
                sycl::buffer<int, 1> seeds_buf(seeds.data(), sycl::range<1>(G.NV));
                q.submit([&](sycl::handler &h)
                         {
                        auto seeds = seeds_buf.get_access<sycl::access::mode::read>(h);
                        auto seed = seed_buf.get_access<sycl::access::mode::write>(h);
                        h.parallel_for(sycl::range<1>(G.NV), [=](sycl::id<1> i) {
                            seed[i] = seeds[i];
                        }); });
            }

            SIR_Metapopulation_Network
            operator=(const SIR_Metapopulation_Network &other)
            {
                return SIR_Metapopulation_Network(other.G, other.p_I0, other.p_R0);
            }

            void initialize()
            {

                q.submit([&](sycl::handler &h)
                         {
                    auto seed = seed_buf.get_access<sycl::access::mode::read>(h);
                    auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
                    h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id) {
                    //total population stored in susceptible state
                    float N_pop = v[id].S;
                    float I0_mean = N_pop*p_I0;
                    float R0_mean = N_pop*p_R0;
                    Sycl_Graph::random::normal_distribution d_I(I0_mean, std_I0);
                    Sycl_Graph::random::normal_distribution d_R(R0_mean, std_R0);
                    Sycl_Graph::random::default_rng rng(seed[i]);
                    SIR_Metapopulation_State v_i;
                    v_i.I = d_I(rng);
                    v_i.R = d_R(rng);
                    v_i.S = N_pop - v_i.I - v_i.R;
                    v[id] = v_i;
                    }); });
            }

            std::vector<uint32_t> population_count()
            {
                std::vector<uint32_t> count(3, 0);
                sycl::buffer<uint32_t, 1> count_buf(count.data(), sycl::range<1>(3));
                const uint32_t N_vertices = G.N_vertices();

                q.submit([&](sycl::handler &h)
                         {
                             auto count_acc = count_buf.get_access<sycl::access::mode::write>(h);
                             auto v = G.vertex_access(h);

                             h.single_task([=]
                                           {
                    for (int i = 0; i < N_vertices; i++)
                    {   
                        count_acc[0] += v[i].S;
                        count_acc[1] += v[i].I;
                        count_acc[2] += v[i].R;
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
            void infection_step(float p_I)
            {
                using Sycl_Graph::Network_Models::SIR_Metapopulation_State;

                sycl::buffer<float, 1> edge_inf_buf(sycl::range<1>(G.N_edges));

                q.submit([&](sycl::handler &h)
                         {
                    auto edge_inf_acc = edge_inf_buf.get_access<sycl::access::mode::write>(h);
                    auto e_acc = G.edge_access(h, cl::sycl::access::mode::read);
                    auto v_acc = G.vertex_access(h, cl::sycl::access::mode::read);
                    h.parallel_for(sycl::range<1>(G.N_edges), [=](sycl::id<1> id)
                    {
                        Sycl_Graph::random::default_rng rng(seed_acc[id]);
                        Sycl_Graph::random::binomial_distribution d_I(v_acc[id].S, p_I);
                        float p_I = e_acc[id].

                    }); });

                q.submit([&](sycl::handler &h)
                         {
                    auto v_acc = G.vertex_access(h);
                    auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(sycl::range<1>(v_acc.size()), [=](sycl::id<1> id)
                    {
                        Sycl_Graph::random::default_rng rng(seed_acc[id]);
                        Sycl_Graph::random::binomial_distribution d_I(v_acc[id].S, p_I);
                        float n_infected = d_I(rng);
                        v_acc[id].S -= n_infected;
                        v_acc[id].I += n_infected;
                        seed_acc[id] += 1;
                    }); 
                    });
                


                int a = 0;
            }

            void recovery_step(float p_R)
            {

                q.submit([&](sycl::handler &h)
                         {
                    auto seed_acc = seed_buf.get_access<sycl::access::mode::read_write>(h);
                    auto v = G.vertex_access(h);
                    sycl::stream out(1024, 256, h);
                    //  auto nv = neighbors_buf.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id)
                                   {
                        Sycl_Graph::random::default_rng rng(seed_acc[id]);
                        seed_acc[id]++;
                        Sycl_Graph::random::uniform_real_distribution<float> d_R(0, 1);
                            if(v[id] == SIR_INDIVIDUAL_I)
                            {
                                if(d_R(rng) < p_R)
                                {
                                    v[id] = SIR_INDIVIDUAL_R;
                                }
                            } }); });
                q.wait();
            }
            void advance(const SIR_Metapopulation_Param<> &p)
            {
                infection_step(p.p_I);
                recovery_step(p.p_R);
            }

            bool terminate(const SIR_Metapopulation_Param<> &p, const std::vector<uint32_t> &x)
            {
                bool early_termination = ((t > p.Nt_min) && (x[1] < p.N_I_min));
                return early_termination;
            }

            void reset()
            {
                q.submit([&](sycl::handler &h)
                         {
                    auto v = G.vertex_access(h);
                    h.parallel_for(sycl::range<1>(G.N_vertices()), [=](sycl::id<1> id)
                                   {
                                       v[id[0]] = SIR_INDIVIDUAL_S;
                                   }); });
            }

        private:
            Graph_t &G;
            sycl::queue &q;
        };
    } // namespace Sycl
} // namespace Network_Models
#endif
#endif