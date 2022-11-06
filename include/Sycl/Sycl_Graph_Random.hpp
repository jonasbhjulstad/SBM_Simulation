#ifndef SYCL_GRAPH_RANDOM_HPP
#define SYCL_GRAPH_RANDOM_HPP

#ifdef SYCL_GRAPH_USE_INTEL_SYCL
#include<oneapi/dpl/random>
#else
#include <random>
#endif
namespace Sycl::Graph::random {
#ifdef SYCL_GRAPH_USE_INTEL_SYCL
    using default_rng = oneapi::dpl::ranlux48;
    using oneapi::dpl::uniform_real_distribution;
#else
    using std::mt19937_64;
    using std::uniform_real_distribution;
    typedef mt19937_64 default_rng;
#endif
}
#endif