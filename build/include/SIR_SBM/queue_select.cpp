// queue_select.cpp
//

#include "queue_select.hpp"
#define LZZ_INLINE inline
#line 6 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//queue_select.hpp"
namespace SIR_SBM
{
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//queue_select.hpp"
  sycl::queue default_queue ()
#line 8 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//queue_select.hpp"
                            {
#ifdef SIR_SBM_USE_GPU
  return sycl::queue(sycl::gpu_selector_v);
#else
  return sycl::queue(sycl::cpu_selector_v);
#endif
}
}
#undef LZZ_INLINE
