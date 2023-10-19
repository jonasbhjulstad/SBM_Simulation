#include <doctest/doctest.h>
#include <CL/sycl.hpp>

void parallel_for_test()
{
    sycl::queue q(sycl::gpu_selector_v);
    sycl::buffer<uint32_t> data(sycl::range<1>(100));
    q.submit([&](sycl::handler& h)
    {
        auto acc = data.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(100, [=](sycl::id<1> id)
        {
            acc[id] = id;
        });
    }).wait();

}

void nd_range_test()
{
sycl::queue q(sycl::gpu_selector_v);
    sycl::buffer<uint32_t> data(sycl::range<1>(100));
    q.submit([&](sycl::handler& h)
    {
        auto acc = data.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(100), sycl::range<1>(10)), [=](sycl::nd_item<1> id)
        {
            acc[id.get_global_id()] = id.get_global_id();
        });
    }).wait();
}

TEST_CASE("parallel_for")
{
    parallel_for_test();
}

TEST_CASE("nd_range")
{
    nd_range_test();
}
