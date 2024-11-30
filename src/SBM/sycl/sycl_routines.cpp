#include <SBM/sycl/sycl_routines.hpp>
namespace SBM {
sycl::event zero_fill(sycl::queue &q, sycl::buffer<uint32_t, 3> &buf,
                      sycl::range<3> range, sycl::range<3> offset,
                      sycl::event dep_event) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto acc = sycl::accessor<uint32_t, 3, sycl::access::mode::read_write>(
        buf, h, range, offset);
    h.parallel_for(acc.get_range(), [=](sycl::id<3> idx) { acc[idx] = 0; });
  });
}
} // namespace SBM