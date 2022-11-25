//
// Created by arch on 9/28/22.
//
#ifndef SYCL_GRAPH_SYCL_GRAPH_EXECUTION_HPP
#define SYCL_GRAPH_SYCL_GRAPH_EXECUTION_HPP

#include <pstl/glue_execution_defs.h>
#ifdef SYCL_GRAPH_USE_INTEL_SYCL
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#else
#include <execution>
#endif
#include <shared_mutex>
#include <mutex>
namespace Sycl_Graph::execution {
#ifdef SYCL_GRAPH_USE_INTEL_SYCL
    using namespace oneapi::dpl;
    using namespace oneapi::dpl::execution;
    static auto default_policy = dpcpp_default;
#else
    static auto seq = std::execution::seq;
    static auto par = std::execution::par;
    static auto par_unseq = std::execution::par_unseq;
    static auto default_policy = std::execution::seq;
#endif
}

#endif //SYCL_GRAPH_SYCL_GRAPH_EXECUTION_HPP
