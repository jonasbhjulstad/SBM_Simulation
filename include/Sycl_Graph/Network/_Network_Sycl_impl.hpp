#ifndef SYCL_GRAPH_NETWORK_SYCL_IMPL_HPP
#define SYCL_GRAPH_NETWORK_SYCL_IMPL_HPP
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/execution.hpp>
#include <random>
#include <stddef.h>
#include <vector>

namespace Sycl_Graph::Sycl
{
    namespace Network_Models
    {
    using namespace Sycl_Graph::Network_Models;
    template <typename Param, class Derived>
    struct Network
    {
        Network(uint32_t Nx): Nx(Nx) {}
        uint32_t Nx;
        std::vector<uint32_t> population_count()
        {
            return static_cast<Derived *>(this)->population_count();
        }
        void advance(const Param &p) { static_cast<Derived *>(this)->advance(p); }
        void reset() { static_cast<Derived *>(this)->reset(); }
        bool terminate(const Param &p, const std::vector<uint32_t> &x)
        {
            return static_cast<Derived *>(this)->terminate(p, x);
        }

        std::vector<std::vector<uint32_t>>
        simulate(const std::vector<Param> &p_vec, uint32_t Nt, uint32_t Nt_min = 15)
        {
            std::vector<std::vector<uint32_t>> trajectory;
            trajectory.resize(p_vec.size()+1);
            //reserve space for the trajectories
            std::for_each(trajectory.begin(), trajectory.end(), [this](auto &x) {
                x.resize(Nx);
            });
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