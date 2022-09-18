#ifndef FROLS_SIS_BERNOULLI_NETWORK_HPP
#define FROLS_SIS_BERNOULLI_NETWORK_HPP

#include "Graph_Generation.hpp"
#include <FROLS_Math.hpp>
#include <graph_lite.h>
#include <random>
#include <stddef.h>
#include <utility>
#include <vector>

namespace Network_Models {
    enum SIS_State {
        SIS_S = 0, SIS_I = 1
    };
    typedef graph_lite::Graph<int, SIS_State> SIS_Network;

    template<typename RNG>
    struct SIS_Bernoulli_Network {

        SIS_Bernoulli_Network(size_t N_pop, double p_ER, RNG rng) : rng(rng) {
            G = generate_erdos_renyi<RNG, SIS_State>(N_pop, p_ER, rng);
        }

        void generate_initial_infections(double p_I0) {
            std::bernoulli_distribution d_I(p_I0);
            std::for_each(G.begin(), G.end(), [&](auto &v) {
                SIS_State state = d_I(rng) ? SIS_I : SIS_S;
                G.node_prop(v) = state;
            });
        }

        std::vector<size_t> population_count() {
            std::vector<size_t> count = {0, 0};
            std::for_each(G.begin(), G.end(), [&](const auto &v) { count[G.node_prop(v)] += 1; });
            return count;
        }

// function for infection step
        void infection_step(double p_I) {
            std::bernoulli_distribution d_I(p_I);
            std::for_each(G.begin(), G.end(), [&](auto v0) {
                if (G.node_prop(v0) == SIS_I) {
                    auto [nbr_start, nbr_end] = G.neighbors(v0);
                    std::for_each(nbr_start, nbr_end, [&](auto &nbr_it) {
                        G.node_prop(nbr_it) =  ((G.node_prop(nbr_it) == SIS_S) && d_I(rng)) ? SIS_I : G.node_prop(nbr_it);
                    });
                }
            });
        }

        void recovery_step(double p_S) {
            std::bernoulli_distribution d_S(p_S);
            std::for_each(G.begin(), G.end(), [&](auto &v) {
                SIS_State &state = G.node_prop(v);
                state = ((state == SIS_I) && d_S(rng)) ? SIS_S : state;
            });
        }

        std::vector<std::vector<size_t>>
        simulate(const std::vector<std::pair<double, double>>& p_vec, size_t infection_count_tolerance = 0, size_t Nt_min = 15) {

            std::vector<std::vector<size_t>> trajectory;
            trajectory.reserve(p_vec.size());
            trajectory.push_back(population_count());
            size_t t = 0;
            for (const auto& p: p_vec)
            {

                auto &[p_I, p_S] = p;
                infection_step(p_I);
                recovery_step(p_S);
                trajectory.push_back(population_count());
                if ((trajectory.back()[1] < infection_count_tolerance) && (t > Nt_min))
                {
                    break;
                }
                t++;
            }
            return FROLS::transpose(trajectory);
        }

        std::vector<std::vector<size_t>> simulate(size_t Nt, double p_I, double p_S) {
            std::vector<std::pair<double, double>> p_vec(Nt);
            std::fill(p_vec.begin(), p_vec.end(), std::make_pair(p_I, p_S));
            return simulate(p_vec);
        }
        std::vector<std::vector<size_t>> simulate(const std::vector<double>& p_I, const std::vector<double>& p_S)
        {
            return simulate(FROLS::zip(p_I, p_S));
        }

        void reset()
        {
            for (auto& v: G)
            {
                G.node_prop(v) = SIS_S;
            }
        }


    private:
        SIS_Network G;
        RNG rng;
    };
}
#endif