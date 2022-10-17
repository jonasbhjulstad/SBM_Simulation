#ifndef FROLS_NETWORK_HPP
#define FROLS_NETWORK_HPP

#include "Graph_Generation.hpp"
#include <FROLS_Math.hpp>
#include <graph_lite.h>
#include <random>
#include <stddef.h>
#include <vector>
#include <FROLS_Execution.hpp>

namespace Network_Models {

    template <typename Param, uint16_t Nx, uint16_t Nt, class Derived>
    struct Network
    {
        std::array<uint16_t, Nx> population_count()
        {
            return static_cast<Derived*>(this)->population_count();
        }
        void advance(const Param& p)
        {
            static_cast<Derived*>(this)->advance(p);
        }
        void reset()
        {
            static_cast<Derived*>(this)->reset();
        }
        bool terminate(const Param& p, const std::array<uint16_t, Nx>& x)
        {
            return static_cast<Derived*>(this)->terminate(p, x);
        }
        
        std::array<std::array<uint16_t, Nt+1>, Nx>
        simulate(const std::array<Param, Nt>& p_vec, uint16_t infection_count_tolerance = 0, uint16_t Nt_min = 15) {

            std::array<std::array<uint16_t, Nx>,Nt+1> trajectory;
            uint16_t t = 0;
            trajectory[0] = population_count();
            for (int i = 0; i < Nt; i++)
            {
                advance(p_vec[i]);
                trajectory[i+1] = population_count();
                if (terminate(p_vec[i], trajectory[i+1]))
                {
                    break;
                }
            }
            return FROLS::transpose(trajectory);
        }
    };
}
#endif