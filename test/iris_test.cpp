#include <CL/sycl.hpp>
#include <Sycl_Buffer_Routines/Profiling.hpp>

int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    auto dev_info = get_device_info(q);
    dev_info.print();

    q.submit([](sycl::handler& h)
    {
        h.parallel_for(sycl::range<1>(256), [](sycl::id<1> id)
        {
            float a = 0;
            a++;
        });
    }).wait();
    return 0;
}
