#ifndef FROLS_SIR_BERNOULLI_NETWORK_HPP
#define FROLS_SIR_BERNOULLI_NETWORK_HPP

#include "Graph_Generation.hpp"
#include "Network.hpp"
#include <FROLS_Math.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Random.hpp>
#include <stddef.h>
#include <utility>
#include <vector>
#include <thread>
#include <FROLS_Execution.hpp>
#include <ranges>

namespace Network_Models
{
    enum SIR_State
    {
        SIR_S = 0,
        SIR_I = 1,
        SIR_R = 2
    };
    struct SIR_Edge
    {
    };

    template <typename dType = float>
    struct SIR_Param
    {
        dType p_I;
        dType p_R;
        uint32_t Nt_min;
        uint32_t N_I_min;
    };
    template <uint32_t NV, uint32_t NE>
    using SIR_ArrayGraph = FROLS::Graph::ArrayGraph<SIR_State, SIR_Edge, NV, NE>;

    using SIR_VectorGraph = FROLS::Graph::VectorGraph<SIR_State, SIR_Edge>;

    template <typename SIR_Graph, typename RNG, uint32_t Nt, typename dType = float>
    struct SIR_Bernoulli_Network : public Network<SIR_Param<>, 3, Nt, SIR_Bernoulli_Network<SIR_Graph, RNG, Nt, dType>>
    {
        using Vertex_t = typename SIR_Graph::Vertex_t;
        using Edge_t = typename SIR_Graph::Edge_t;
        using Edge_Prop_t = typename SIR_Graph::Edge_Prop_t;
        using Vertex_Prop_t = typename SIR_Graph::Vertex_Prop_t;
        const dType p_I0;
        const dType p_R0;
        const uint32_t t = 0;

        SIR_Bernoulli_Network(SIR_Graph &G, dType p_I0, dType p_R0, RNG rng) : G(G), rng(rng),
                                                                                       p_I0(p_I0), p_R0(p_R0) {}

        void initialize()
        {

            FROLS::random::uniform_real_distribution<dType> d_I;
            FROLS::random::uniform_real_distribution<dType> d_R;
            std::for_each(std::execution::par_unseq, G.begin(), G.end(), [&](auto v)
                          {
                if (d_I(rng) < p_I0) {
                    G.assign(v.id, SIR_I);
                } else if (d_R(rng) < p_R0) {
                    G.assign(v.id, SIR_R);
                } else {
                    G.assign(v.id, SIR_S);
                } });
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

            FROLS::random::uniform_real_distribution<dType> d_I;

            // print distance between G.begin() and G.end/()
            //  std::cout << std::distance(G.begin(), G.end()) << std::endl;
            std::for_each(std::execution::par_unseq, G.begin(), G.end(),  [&](auto v0)
                          {
                //print id of thread
                if (v0.data == SIR_I) {
                    for (const auto v: G.neighbors(v0.id)) {
                        if(!v) break;
                        bool trigger = d_I(rng) < p_I;
                        if (v->data == SIR_S && trigger) {
                        G.assign(v->id, ((v->data == SIR_S) && (trigger)) ? SIR_I : v->data);
                        }
                    };
                } });
        }

        void recovery_step(dType p_R)
        {

            FROLS::random::uniform_real_distribution<dType> d_R;
            std::for_each(std::execution::par_unseq, G.begin(), G.end(), [&](const auto &v)
                          {
                bool recover_trigger = (v.data == SIR_I) && d_R(rng) < p_R;
                G.assign(v.id, (recover_trigger) ? SIR_R : v.data); });
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
            std::for_each(G.begin(), G.end(), [&](auto v)
                          {G.assign(v.id, SIR_S);});
        }

    private:
        SIR_Graph &G;
        RNG rng;
    };
}
#endif