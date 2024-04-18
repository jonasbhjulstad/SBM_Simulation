#pragma once
#include <sycl/sycl.hpp>
namespace SIR_SBM {

sycl::queue default_queue() {
#ifdef SIR_SBM_USE_GPU
  return sycl::queue(sycl::gpu_selector_v);
#else
  return sycl::queue(sycl::cpu_selector_v);
#endif
}

} // namespace SIR_SBM