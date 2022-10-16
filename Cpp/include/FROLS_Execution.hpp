//
// Created by arch on 9/28/22.
//
#ifndef FROLS_FROLS_EXECUTION_HPP
#define FROLS_FROLS_EXECUTION_HPP

#ifdef FROLS_USE_INTEL_SYCL
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#else
#include <execution>
#endif
namespace FROLS::execution {
#ifdef FROLS_USE_INTEL_SYCL
    using namespace oneapi::dpl;
    using namespace oneapi::dpl::execution;
#else
    static auto par_unseq = std::execution::par_unseq;
    static auto par = std::execution::par;
    static auto seq = std::execution::seq;
#endif
}

#endif //FROLS_FROLS_EXECUTION_HPP
