// sycl_routines.cpp
//

#include "sycl_routines.hpp"
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
#include <SIR_SBM/reshape.hpp>
#define LZZ_INLINE inline
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
namespace SIR_SBM
{
#line 92 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
  sycl::event zero_fill (sycl::queue & q, sycl::buffer <uint32_t, 3> & buf, sycl::range <3> range, sycl::range <3> offset, sycl::event dep_event)
#line 94 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//sycl_routines.hpp"
                                                  {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read_write>(
        buf, h, range, offset);
    h.parallel_for(acc.get_range(), [=](sycl::id<3> idx) { acc[idx] = 0; });
  });
}
}
#undef LZZ_INLINE
