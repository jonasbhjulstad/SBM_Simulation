#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <cmath>
int main()
{
    sycl::queue q(sycl::cpu_selector_v);
    std::vector<int> vec(16);
    auto N = std::sqrt(vec.size());
    std::iota(vec.begin(), vec.end(), 0);
    sycl::buffer<int, 2> data(vec.data(), sycl::range<2>(N, N));

    q.submit([&](sycl::handler &h)
             {
        auto acc = sycl::accessor<int, 2, sycl::access::mode::read_write, sycl::access::target::device>(data, h, sycl::range<2>(N, 1), sycl::range<2>(0,1));
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i)
        {
            acc[i][0]+=10;
        }); })
        .wait();
    std::vector<int> result(vec.size());
    q.submit([&](sycl::handler &h)
             {
                auto acc = data.template get_access<sycl::access::mode::read>(h);
                h.copy(acc, result.data()); })
        .wait();
    for(auto i : result)
        std::cout << i << " ";
}
