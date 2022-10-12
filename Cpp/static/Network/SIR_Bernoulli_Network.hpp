#ifndef FROLS_SIR_BERNOULLI_NETWORK_HPP
#define FROLS_SIR_BERNOULLI_NETWORK_HPP

#include "Graph_Generation.hpp"
#include "Network.hpp"
#include <FROLS_Math.hpp>
#include <FROLS_Graph.hpp>
#include <graph_lite.h>
#include <random>
#include <stddef.h>
#include <oneapi/dpl/random>
#include <utility>
#include <vector>
#include <execution>
#include <ranges>

namespace Network_Models {
    enum SIR_State {
        SIR_S = 0, SIR_I = 1, SIR_R = 2
    };
    struct SIR_Edge {
    };

    struct SIR_Param {
        double p_I;
        double p_R;
        size_t Nt_min;
        size_t N_I_min;
    };
    template<size_t NV, size_t NE>
    using SIR_Graph = FROLS::Graph::Graph<SIR_State, SIR_Edge, NV, NE>;

    template<typename RNG, size_t Nt, size_t NV, size_t NE>
    struct SIR_Bernoulli_Network : public Network<SIR_Param, 3, Nt, SIR_Bernoulli_Network<RNG, Nt, NV, NE>> {
        using Vertex_t = typename SIR_Graph<NV, NE>::Vertex_t;
        using Edge_t = typename SIR_Graph<NV, NE>::Edge_t;
        using Edge_Prop_t = typename SIR_Graph<NV, NE>::Edge_Prop_t;
        using Vertex_Prop_t = typename SIR_Graph<NV, NE>::Vertex_Prop_t;
        const double p_I0;
        const double p_R0;
        const size_t t = 0;

        SIR_Bernoulli_Network(const SIR_Graph<NV, NE> &G, double p_I0, double p_R0, RNG rng) : G(G), rng(rng),
                                                                                               p_I0(p_I0), p_R0(p_R0) {}

        void initialize() {
#ifdef FROLS_USE_DPCPP
            using oneapi::dpl::bernoulli_distribution;
#else
            using std::bernoulli_distribution;
#endif
            bernoulli_distribution d_I(p_I0);
            bernoulli_distribution d_R(p_R0);
            for(auto& v: G)
            {
                SIR_State state = d_I(rng) ? SIR_I : SIR_S;
                state = d_R(rng) ? SIR_R : state;
                v.data = state;
            }
        }

        std::array<size_t, 3> population_count() {
            std::array<size_t, 3> count = {0, 0, 0};
            for (const auto& v: G)
            {
                count[v.data]++;
            }
            return count;
        }

// function for infection step
        void infection_step(double p_I) {

#ifdef FROLS_USE_DPCPP
            using oneapi::dpl::bernoulli_distribution;
#else
            using std::bernoulli_distribution;
#endif
            bernoulli_distribution d_I(p_I);

            std::for_each(G.begin(), G.end(), [&](auto v0) {
                if (v0.data == SIR_I) {
                    for (const auto &v: G.neighbors(v0.id)) {
                        G.assign(v.id, ((v.data == SIR_S) && d_I(rng)) ? SIR_I : v.data);
                    };
                }
            });
        }

        void recovery_step(double p_R) {
#ifdef FROLS_USE_DPCPP
            using oneapi::dpl::bernoulli_distribution;
#else
            using std::bernoulli_distribution;
#endif
            bernoulli_distribution d_R(p_R);
            for (const auto &v: G) {
                G.assign(v.id, ((v.data == SIR_I) && d_R(rng)) ? SIR_R : v.data);
            }
        }

        bool terminate(const SIR_Param &p, const std::array<size_t, 3> &x) {
            bool early_termination = ((t > p.Nt_min) && x[1] < p.N_I_min);
            return early_termination || (t >= Nt);
        }

        void advance(const SIR_Param &p) {
            infection_step(p.p_I);
            recovery_step(p.p_R);
        }

        void reset() {
            for (const auto& v: G)
            {
                G.assign(v.id, SIR_S);
            }
        }

    private:
        SIR_Graph<NV, NE> G;
        RNG rng;
    };
}
#endif