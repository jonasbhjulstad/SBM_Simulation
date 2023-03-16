#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>

int foo(int a)
{
    sycl::queue q(sycl::gpu_selector_v);
    sycl::buffer<int, 1> b(&a, sycl::range<1>(1));
    q.submit([&](sycl::handler& cgh){
        auto acc = b.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class foo>([=](){
            acc[0] +=1;
        });
    });
    //accessor
    auto acc = b.get_access<sycl::access::mode::read>();
    return acc[0];
}

PYBIND11_MODULE(test_binder, m) {
    m.def("foo", &foo);
}

