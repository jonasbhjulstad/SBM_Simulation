#ifndef SYCL_GRAPH_UTILS_COMPUTE_RANGE_HPP
#define SYCL_GRAPH_UTILS_COMPUTE_RANGE_HPP
#include <CL/sycl.hpp>
#include <cstdint>

sycl::range<1> get_wg_range(sycl::queue &q);
sycl::range<1> get_compute_range(sycl::queue &q, uint32_t N_sims);

#endif
