#ifndef SBM_SIMULATION_UTILS_COMPUTE_RANGE_HPP
#define SBM_SIMULATION_UTILS_COMPUTE_RANGE_HPP
#include <CL/sycl.hpp>
#include <cstdint>



sycl::range<1> get_compute_range(sycl::queue &q, uint32_t N_sims);


#endif
