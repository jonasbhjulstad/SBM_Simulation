#include <Sycl_Graph/Graph/Graph_Base.hpp>
#include <random>
#include <algorithm>
#include <iostream>
int main()
{

    //select gpu
    sycl::queue q;

    std::vector<uint32_t> counts(3);
    //Create buffer
    sycl::buffer<uint32_t> counts_buf(counts.data(), counts.size());

    //submit kernel
    q.submit([&](sycl::handler &cgh) {
        //get access to buffer
        auto counts_acc = counts_buf.get_access<sycl::access::mode::read_write>(cgh);
        //submit kernel
        cgh.single_task([=]() {
            counts_acc[0] = 1;
            counts_acc[1] = 2;
            counts_acc[2] = 3;
        });
    });

    //
    sycl::host_accessor counts_acc(counts_buf, sycl::read_only);
    for (int i = 0; i < 3; i++) {
        std::cout << counts_acc[i] << std::endl;
    }
    return 0;

}