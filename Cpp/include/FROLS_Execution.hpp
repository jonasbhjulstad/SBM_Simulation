//
// Created by arch on 9/28/22.
//
#ifndef FROLS_FROLS_EXECUTION_HPP
#define FROLS_FROLS_EXECUTION_HPP

#include <execution>
#include <algorithm>
#ifdef FROLS_USE_INTEL_SYCL
#include<oneapi/dpl/execution>
#endif
namespace FROLS::execution {
#ifdef FROLS_USE_INTEL_SYCL
    using namespace oneapi::dpl;
    using namespace oneapi::dpl::execution;
    static auto frols_par_unseq = oneapi::dpl::execution::dpcpp_default;
#else
    static auto par_unseq = std::execution::par_unseq;
    static auto par = std::execution::par;
    static auto seq = std::execution::seq;
#endif
}

#endif //FROLS_FROLS_EXECUTION_HPP
