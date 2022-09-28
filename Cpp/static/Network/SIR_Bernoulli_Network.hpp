#ifndef FROLS_SIR_BERNOULLI_NETWORK_HPP
#define FROLS_SIR_BERNOULLI_NETWORK_HPP

#include "Graph_Generation.hpp"
#include "Network.hpp"
#include <FROLS_Math.hpp>
#include <graph_lite.h>
#include <random>
#include <stddef.h>
#include <utility>
#include <vector>
#include <FROLS_Execution.hpp>

namespace Network_Models {
    enum SIR_State {
        SIR_S = 0, SIR_I = 1, SIR_R = 2
    };
    typedef graph_lite::Graph<int, SIR_State> SIR_Network;

    struct SIR_Param
    {
        double p_I; double p_R;
        size_t Nt_min;
        size_t N_I_min;
    };
    template<typename RNG, size_t Nt>
    struct SIR_Bernoulli_Network : public Network<SIR_Param, 3, Nt>{
        const double p_I0;
        const double p_R0;
        const size_t t = 0;
        SIR_Bernoulli_Network(size_t N_pop, double p_ER, double p_I0, double p_R0, RNG rng) : rng(rng), p_I0(p_I0), p_R0(p_R0) {
            G = generate_erdos_renyi<RNG, SIR_State>(N_pop, p_ER, rng);
        }

        void initialize() {
            std::bernoulli_distribution d_I(p_I0);
            std::bernoulli_distribution d_R(p_R0);
            std::for_each(std::execution::par_unseq, G.begin(), G.end(), [&](auto &v) {
                SIR_State state = d_I(rng) ? SIR_I : SIR_S;
                state = d_R(rng) ? SIR_R : state;
                G.node_prop(v) = state;
            });
        }

        std::array<size_t, 3> population_count() {
            std::array<size_t, 3> count = {0, 0, 0};
            std::for_each(G.begin(), G.end(), [&](const auto &v) { count[G.node_prop(v)] += 1; });
            return count;
        }

// function for infection step
        void infection_step(double p_I) {
            std::bernoulli_distribution d_I(p_I);
            std::for_each(std::execution::par_unseq, G.begin(), G.end(), [&](auto v0) {
                if (G.node_prop(v0) == SIR_I) {
                    auto [nbr_start, nbr_end] = G.neighbors(v0);
                    std::for_each(std::execution::par_unseq, nbr_start, nbr_end, [&](auto &nbr_it) {
                        G.node_prop(nbr_it) =  ((G.node_prop(nbr_it) == SIR_S) && d_I(rng)) ? SIR_I : G.node_prop(nbr_it);
                    });
                }
            });
        }

        void recovery_step(double p_R) {
            std::bernoulli_distribution d_R(p_R);
            std::for_each(std::execution::par_unseq, G.begin(), G.end(), [&](auto &v) {
                SIR_State &state = G.node_prop(v);
                state = ((state == SIR_I) && d_R(rng)) ? SIR_R : state;
            });
        }

        bool terminate(const SIR_Param& p, const std::array<size_t, 3>& x)
        {
            bool early_termination = ((t > p.Nt_min) && x[1] < p.N_I_min);
            return early_termination || (t >= Nt);
        }

        void advance(const SIR_Param& p)
        {
            infection_step(p.p_I);
            recovery_step(p.p_R);
        }

        void reset()
        {
            std::for_each(G.begin(), G.end(), [&](auto& v){G.node_prop(v) = SIR_S;});
        }

    private:
        SIR_Network G;
        RNG rng;
    };
}
#endif