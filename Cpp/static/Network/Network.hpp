#ifndef FROLS_NETWORK_HPP
#define FROLS_NETWORK_HPP

#include "Graph_Generation.hpp"
#include <FROLS_Math.hpp>
#include <random>
#include <stddef.h>
#include <vector>
#include <FROLS_Execution.hpp>

namespace Network_Models {

    template <typename Param, uint32_t Nx, uint32_t Nt, class Derived>
    struct Network
    {

        using Trajectory = std::array<uint32_t, Nx>;
        Trajectory population_count()
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
        bool terminate(const Param& p, const std::array<uint32_t, Nx>& x)
        {
            return static_cast<Derived*>(this)->terminate(p, x);
        }
        
        std::array<std::array<uint32_t, Nt+1>, Nx>
        simulate(const std::array<Param, Nt>& p_vec, uint32_t infection_count_tolerance = 0, uint32_t Nt_min = 15) {

            std::array<Trajectory,Nt+1> trajectory;
            uint32_t t = 0;
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

    size_t get_required_network_space(size_t Nx, size_t Nt)
    {
        return Nx * (Nt + 1) * sizeof(uint32_t);
    }
}
#endif