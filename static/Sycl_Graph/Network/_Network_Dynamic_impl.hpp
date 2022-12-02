#ifndef SYCL_GRAPH_NETWORK_DYNAMIC_IMPL_HPP
#define SYCL_GRAPH_NETWORK_DYNAMIC_IMPL_HPP
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/execution.hpp>
#include <random>
#include <stddef.h>
#include <vector>


    namespace Sycl_Graph::Dynamic
    {
        namespace Network_Models
        {
        template <typename Param, class Derived>
        struct Network
        {

            using Trajectory = std::vector<uint32_t>;
            Trajectory population_count()
            {
                return static_cast<Derived *>(this)->population_count();
            }
            void advance(const Param &p) { static_cast<Derived *>(this)->advance(p); }
            void reset() { static_cast<Derived *>(this)->reset(); }
            bool terminate(const Param &p, const std::vector<uint32_t> &x)
            {
                return static_cast<Derived *>(this)->terminate(p, x);
            }

            void initialize() { return static_cast<Derived *>(this)->initialize(); }

            std::vector<std::vector<uint32_t>>
            simulate(const std::vector<Param> &p_vec, uint32_t Nt, uint32_t Nt_min = 15)
            {

                std::vector<Trajectory> trajectory(3);
                std::for_each(trajectory.begin(), trajectory.end(),
                              [Nt](auto &x)
                              { x.reserve(Nt + 1); });
                auto count = population_count();
                trajectory[0].push_back(count[0]);
                trajectory[1].push_back(count[1]);
                trajectory[2].push_back(count[2]);
                for (int i = 0; i < Nt; i++)
                {
                    advance(p_vec[i]);
                    auto count = population_count();
                    trajectory[0].push_back(count[0]);
                    trajectory[1].push_back(count[1]);
                    trajectory[2].push_back(count[2]);
                    if (terminate(p_vec[i], trajectory[i + 1]))
                    {
                        break;
                    }
                }
                return trajectory;
            }
        };
    } // namespace Sycl_Graph::Dynamic
} // namespace Network_Models

#endif