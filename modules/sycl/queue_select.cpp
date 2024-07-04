module;

#include <SIR_SBM/common.hpp>
#include <sycl/sycl.hpp>

export module SIR_SBM.sycl.queue;

export sycl::queue default_queue() {
#ifdef SIR_SBM_USE_GPU
  return sycl::queue(sycl::gpu_selector_v);
#else
  return sycl::queue(sycl::cpu_selector_v);
if
}

