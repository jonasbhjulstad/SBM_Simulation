#include <CL/sycl.hpp>
#include <iostream>

int main()
{
    //demonstrate the usage of kernel bundles
    //get context
    cl::sycl::context ctx = cl::sycl::context();
    cl::sycl::kernel_bundle<cl::sycl::bundle_state::input> kb = cl::sycl::get_kernel_bundle<cl::sycl::bundle_state::input>(ctx);
    std::cout << "kernel bundle size: " << kb.get_kernel_ids().size() << std::endl;
    return 0;

}