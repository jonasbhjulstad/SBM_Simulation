// queue_select.hpp
//

#ifndef LZZ_queue_select_hpp
#define LZZ_queue_select_hpp
#include <SIR_SBM/common.hpp>
#include <sycl/sycl.hpp>
#define LZZ_INLINE inline
namespace SIR_SBM
{
  sycl::queue default_queue ();
}
#undef LZZ_INLINE
#endif
