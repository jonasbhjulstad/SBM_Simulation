#ifndef SYCL_GRAPH_STATISTICAL_TYPEDEFS_HPP
#define SYCL_GRAPH_STATISTICAL_TYPEDEFS_HPP
#include <sycl/CL/sycl.hpp>
#include <oneapi/dpl/random>
#include <type_traits>
namespace Sycl_Graph
{
    template <typename dType = float>
    struct Normal_Distribution
    {
        dType mean = 0.f;
        dType std = 0.f;
        dType operator()(auto rng)
        {
            oneapi::dpl::normal_distribution<dType> dist(mean, std);
            return dist(rng);
        }
    };

}
    template <>
    struct sycl::is_device_copyable<Sycl_Graph::Normal_Distribution<float>> : std::true_type
    {
    };
    template <>
    struct sycl::is_device_copyable<Sycl_Graph::Normal_Distribution<double>> : std::true_type
    {
    };

#endif