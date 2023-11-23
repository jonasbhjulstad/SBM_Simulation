#include <CL/sycl.hpp>



int main()
{
    sycl::queue q(sycl::gpu_selector_v);


    sycl::buffer<int> buf(sycl::range<1>(10));

    q.submit([&](sycl::handler& cgh)
    {
        auto acc = buf.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class test>(sycl::range<1>(10), [=](sycl::id<1> idx)
        {
            acc[idx] = idx[0];
        });
    }).wait();

    return 0;
}