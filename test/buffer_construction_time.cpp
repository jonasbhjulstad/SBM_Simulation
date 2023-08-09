#include <CL/sycl.hpp>

#include <cstdint>
#include <chrono>
#include <iostream>

template <typename T>
sycl::buffer<T, 1> buffer_create_1D(sycl::queue &q, const std::vector<T> &data, sycl::event &res_event)
{
    sycl::buffer<T> tmp(data.data(), data.size());
    sycl::buffer<T> result(sycl::range<1>(data.size()));

    res_event = q.submit([&](sycl::handler &h)
                         {
                auto tmp_acc = tmp.template get_access<sycl::access::mode::read>(h);
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(tmp_acc, res_acc); });
    return result;
}

int main()
{
    std::vector<sycl::buffer<uint32_t>> data;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    sycl::queue q(sycl::gpu_selector_v);
    std::vector<sycl::event> events(100);
    std::vector<uint32_t> vec(30000, 1);
    for(int i = 0;i < 100; i++)
    {
        start = std::chrono::system_clock::now();
        data.push_back(buffer_create_1D(q, vec, events[i]));
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }

    return 0;
}
