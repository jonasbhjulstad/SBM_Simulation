#ifndef SYCL_GRAPH_SYCL_SIR_BERNOULLI_NETWORK_HPP
#define SYCL_GRAPH_SYCL_SIR_BERNOULLI_NETWORK_HPP

#include <Graph_Generation.hpp>
#include <SIR_Bernoulli_Network.hpp>
#include <Sycl_Graph_Math.hpp>
#include <FROLS_Random.hpp>
#include <stddef.h>
#include <utility>
#include <vector>
#include <FROLS_Execution.hpp>
#include <FROLS_Sycl.hpp>
#include <ranges>

namespace Network_Models
{
    template <typename RNG, uint32_t Nt, uint32_t NV, uint32_t NE, typename dType = float>
    struct Sycl_SIR_Bernoulli_Network : public Network<SIR_Param<>, 3, Nt, SIR_Bernoulli_Network<RNG, Nt, NV, NE>>
    {
        using Vertex_t = typename SIR_Graph<NV, NE>::Vertex_t;
        using Edge_t = typename SIR_Graph<NV, NE>::Edge_t;
        using Edge_Prop_t = typename SIR_Graph<NV, NE>::Edge_Prop_t;
        using Vertex_Prop_t = typename SIR_Graph<NV, NE>::Vertex_Prop_t;
        // using RNG_accessor = sycl::accessor<FROLS::random::default_rng, 1, sycl::access::mode::read_write, sycl::access::target::local>;
        using RNG_accessor = sycl::accessor<FROLS::random::default_rng, 1, sycl::access::mode::read_write, sycl::access::target::device>;
        const uint32_t t = 0;

        Sycl_SIR_Bernoulli_Network(const SIR_Graph<NV, NE> &G, dType p_I0, dType p_R0, sycl::queue &q, const std::array<size_t, NV> &seeds) : G(G),
                                                                                                                                              q(q), p0{p_I0, p_R0}
        {
            std::transform(seeds.begin(), seeds.end(), rng_vec.begin(), [](const auto &seed)
                           { return FROLS::random::default_rng(seed); });
        }

        void initialize()
        {
            q.submit([&](sycl::handler &h)
                     {
                sycl::accessor rng_acc{rng_buffer, h, sycl::read_write};
                sycl::accessor p0_acc{p0_buffer, h, sycl::read_constant};
                sycl::accessor G_acc{G_buffer, h, sycl::read_write};
                        h.parallel_for<>(
                           sycl::range<1>{NV}, [=](sycl::id<1> i)
                           {
                            auto p_I0 = p0_acc[0];
                            auto p_R0 = p0_acc[1];
                    FROLS::random::uniform_real_distribution<dType> d_I;
                    FROLS::random::uniform_real_distribution<dType> d_R;
                    SIR_State state = d_I(rng_acc[i]) < p_I0 ? SIR_I : SIR_S;
                    state = d_R(rng_acc[i]) < p_R0 ? SIR_R : state;
                    G_acc[0].assign_vertex(state, i); }); });
            auto pop = population_count();
            std::cout << "Population count: " << pop[0] << ", " << pop[1] << ", " << pop[2] << std::endl;
        }

        std::array<uint32_t, 3> population_count()
        {
            std::array<uint32_t, 3> count = {0, 0, 0};
            std::for_each(G.begin(), G.end(), [&count](const Vertex_t &v)
                          { count[v.data]++; });
            return count;
        }

        // function for infection step
        void infection_step(dType p_I)
        {
            q.submit([&](sycl::handler &h)
                     {

            FROLS::random::uniform_real_distribution<dType> d_I;
            RNG_accessor rng_acc{rng_buffer, h};

            h.parallel_for(sycl::range<1>{NV}, [=](sycl::id<1> it)
                           {
                auto rng = rng_acc[it[0]];
                if (G.get_vertex_prop(it[0]) == SIR_I)
                {
                    for(const auto& v: G.neighbors(it[0]))
                    {
                        if (G.get_vertex_prop(v) == SIR_S && d_I(rng) < p_I)
                        {
                            G.assign_vertex(v, SIR_I);
                        }
                    }
                } }); });
        }

        void recovery_step(dType p_R)
        {
            q.submit([&](sycl::handler &h)
                     {
                FROLS::random::uniform_real_distribution<dType> d_R;
                RNG_accessor rng_acc{rng_buffer, h};
                h.parallel_for<>(sycl::range<1>{NV}, [=](sycl::id<1> it)
                                 { 
                                auto rng = rng_acc[it[0]];
                                G.assign(it[0], (G[it[0]].data == SIR_I && d_R(rng) < p_R) ? SIR_R : G[it[0]].data); }); });
        }

        bool terminate(const SIR_Param<> &p, const std::array<uint32_t, 3> &x)
        {
            bool early_termination = ((t > p.Nt_min) && x[1] < p.N_I_min);
            return early_termination || (t >= Nt);
        }

        void advance(const SIR_Param<> &p)
        {
            infection_step(p.p_I);
            recovery_step(p.p_R);
        }

        void reset()
        {
            for (const auto &v : G)
            {
                G.assign(v.id, SIR_S);
            }
        }

    private:
        sycl::queue &q;
        SIR_Graph<NV, NE> G;
        sycl::buffer<SIR_Graph<NV, NE>> G_buffer{&G, sycl::range<1>{1}};
        std::array<RNG, NV> rng_vec;
        sycl::buffer<RNG, 1> rng_buffer{rng_vec};
        // Initial state probabilities
        const std::array<dType, 2> p0;
        sycl::buffer<dType, 1> p0_buffer{p0};
    };
} // namespace Network_Models
#endif