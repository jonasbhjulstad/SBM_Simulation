#include <CL/sycl.hpp>
#include <fstream>
#include <tuple>
int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    // get work group range
    auto range = q.get_device().get_info<sycl::info::device::max_work_group_size>();

    q.submit([&](sycl::handler &h)
             { h.parallel_for(range, [=](sycl::id<1> idx)
                              { int a = 0; }); });
    return 0;
}
