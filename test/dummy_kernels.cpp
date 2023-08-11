#include <CL/sycl.hpp>

#include <chrono>
#include <cstdint>
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

std::vector<sycl::event> dispatch(sycl::queue &q)
{
    uint32_t N = 50*1024;
    std::vector<uint32_t> data_0(N, 2);
    uint32_t N_kernels = 100000;
    std::vector<sycl::event> events(2);
    std::vector<sycl::event> res_events(N_kernels);

    auto buf_0 = buffer_create_1D(q, data_0, events[0]);
    auto buf_1 = buffer_create_1D(q, data_0, events[1]);

    auto kernel_dispatch = [&](auto &b0, auto &b1, auto &dep_events)
    { return
          [&](sycl::handler &h)
      {
          h.depends_on(dep_events);
          auto read_acc = b0.template get_access<sycl::access::mode::read>(h);
          auto res_acc = b1.template get_access<sycl::access::mode::read_write>(h);
          h.parallel_for(N, [=](sycl::id<1> id)
                         { res_acc[id] += read_acc[id]; });
      }; };

    std::cout << "Enqueue" << std::endl;
    for (int i = 0; i < N_kernels; i++)
    {
        res_events[i] = q.submit(kernel_dispatch(buf_0, buf_1, events));
        events[1] = res_events[i];
    }
    return res_events;
}

int main()
{
    sycl::queue q_cpu(sycl::cpu_selector_v);
    sycl::queue q_gpu(sycl::gpu_selector_v);

    auto cpu_events = dispatch(q_cpu);
    auto gpu_events = dispatch(q_gpu);
    std::cout << "Wait" << std::endl;

    std::for_each(cpu_events.begin(), cpu_events.end(), [&](auto &e)
                  { e.wait(); });
    std::for_each(gpu_events.begin(), gpu_events.end(), [&](auto &e)
                  { e.wait(); });
    return 0;
}
