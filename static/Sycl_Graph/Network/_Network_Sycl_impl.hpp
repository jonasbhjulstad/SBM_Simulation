#ifndef SYCL_GRAPH_NETWORK_SYCL_IMPL_HPP
#define SYCL_GRAPH_NETWORK_SYCL_IMPL_HPP
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/execution.hpp>
#include <random>
#include <stddef.h>
#include <array>

namespace Sycl_Graph::Sycl
{
    namespace Network_Models
    {
    using namespace Sycl_Graph::Network_Models;
    template <typename Param, uint32_t Nx, uint32_t Nt, class Derived>
    struct Network
    {

        using Trajectory = std::array<uint32_t, Nx>;
        Trajectory population_count()
        {
            return static_cast<Derived *>(this)->population_count();
        }

        void scatter(const Param& p) { static_cast<Derived *>(this)->scatter(p); }
        void gather(const Param& p) { static_cast<Derived *>(this)->gather(p); }
        void advance(const Param &p) { scatter(p); gather(p); }
        void reset() { static_cast<Derived *>(this)->reset(); }
        bool terminate(const Param &p, const std::array<uint32_t, Nx> &x)
        {
            return static_cast<Derived *>(this)->terminate(p, x);
        }

        std::array<std::array<uint32_t, Nt + 1>, Nx>
        simulate(const std::array<Param, Nt> &p_vec, uint32_t Nt_min = 15)
        {

            std::array<Trajectory, Nt + 1> trajectory;
            uint32_t t = 0;
            trajectory[0] = population_count();
            for (int i = 0; i < Nt; i++)
            {
                advance(p_vec[i]);
                trajectory[i + 1] = population_count();
                if (terminate(p_vec[i], trajectory[i + 1]))
                {
                    break;
                }
            }
            return Sycl_Graph::transpose(trajectory);
        }
    };
} // namespace Fixed
} // namespace Network_Models
#endif