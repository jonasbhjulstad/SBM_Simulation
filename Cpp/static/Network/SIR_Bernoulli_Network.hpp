#ifndef SYCL_GRAPH_SIR_BERNOULLI_NETWORK_HPP
#define SYCL_GRAPH_SIR_BERNOULLI_NETWORK_HPP

#include "Graph_Generation.hpp"
#include "Network.hpp"
#include <Sycl_Graph_Math.hpp>
#include <Sycl_Graph.hpp>
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
        SIR_Param(){}
        dType p_I = 0;
        dType p_R = 0;
        uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
        uint32_t N_I_min = 0;
    };
    template <uint32_t NV, uint32_t NE>
    using SIR_ArrayGraph = FROLS::Graph::ArrayGraph<SIR_State, SIR_Edge, NV, NE>;

    using SIR_VectorGraph = FROLS::Graph::VectorGraph<SIR_State, SIR_Edge>;

    template <typename RNG, uint32_t NV, uint32_t NE, uint32_t Nt, typename dType = float>
    struct Array_SIR_Bernoulli_Network : public ArrayNetwork<SIR_Param<>, 3, Nt, Array_SIR_Bernoulli_Network<RNG, NV, NE, Nt, dType>>
    {
        using Vertex_t = typename SIR_ArrayGraph<NV, NE>::Vertex_t;
        using Edge_t = typename SIR_ArrayGraph<NV, NE>::Edge_t;
        using Edge_Prop_t = typename SIR_ArrayGraph<NV, NE>::Edge_Prop_t;
        using Vertex_Prop_t = typename SIR_ArrayGraph<NV, NE>::Vertex_Prop_t;
        const dType p_I0;
        const dType p_R0;
        const uint32_t t = 0;

        Array_SIR_Bernoulli_Network(SIR_ArrayGraph<NV, NE> &G, dType p_I0, dType p_R0, RNG rng) : G(G), rng(rng),
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
            bool early_termination = ((t > p.Nt_min) || x[1] < p.N_I_min);
            return early_termination;
        }

        void advance(const SIR_Param<> &p)
        {
            infection_step(p.p_I);
            recovery_step(p.p_R);
        }

        void reset()
        {
            std::cout << "Resetting..." << std::endl;
            std::for_each(G.begin(), G.end(), [&](auto v)
                          {G.assign(v.id, SIR_S);});
        }

    private:
        SIR_ArrayGraph<NV, NE> &G;
        RNG rng;
    };


    template <typename RNG, typename dType>
    struct Vector_SIR_Bernoulli_Network : public VectorNetwork<SIR_Param<>, Network_Models::Vector_SIR_Bernoulli_Network<RNG, dType>>
    {
        using Vertex_t = typename SIR_VectorGraph::Vertex_t;
        using Edge_t = typename SIR_VectorGraph::Edge_t;
        using Edge_Prop_t = typename SIR_VectorGraph::Edge_Prop_t;
        using Vertex_Prop_t = typename SIR_VectorGraph::Vertex_Prop_t;
        const dType p_I0;
        const dType p_R0;
        const uint32_t t = 0;

        Vector_SIR_Bernoulli_Network(SIR_VectorGraph &G, dType p_I0, dType p_R0, RNG rng) : G(G), rng(rng),
                                                                                       p_I0(p_I0), p_R0(p_R0) {}
        
        Vector_SIR_Bernoulli_Network operator=(const Vector_SIR_Bernoulli_Network &other)
        {
            return Vector_SIR_Bernoulli_Network(other.G, other.p_I0, other.p_R0, other.rng);
        }

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

        std::vector<uint32_t> population_count()
        {
            std::vector<uint32_t> count = {0, 0, 0};
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

        bool terminate(const SIR_Param<> &p, const std::vector<uint32_t> &x)
        {
            bool early_termination = ((t > p.Nt_min) || (x[1] < p.N_I_min));
            return early_termination;
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
        SIR_VectorGraph &G;
        RNG rng;
    };
}
#endif