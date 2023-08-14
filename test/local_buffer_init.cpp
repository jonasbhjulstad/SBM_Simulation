#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <iostream>

template <typename T>
sycl::buffer<T, 1> buffer_create_1D(sycl::queue &q, const std::vector<T> &data, sycl::event &res_event)
{
    sycl::buffer<T> result(sycl::range<1>(data.size()));

    res_event = q.submit([&](sycl::handler &h)
                         {
                auto res_acc = result.template get_access<sycl::access::mode::write>(h);
                h.copy(data.data(), res_acc); });
    return result;
}

int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    auto device = q.get_device();
    auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
    uint32_t N = max_work_group_size * max_compute_units;

    std::cout << max_work_group_size << ", " << max_compute_units << std::endl;

    std::vector<int> vec(N, 1);
    sycl::event res_event;
    auto buf = buffer_create_1D(q, vec, res_event);

    auto event = q.submit([&](sycl::handler &h)
             {
        h.depends_on(res_event);
        auto acc =  buf.template get_access<sycl::access::mode::read_write>(h);
        auto local_acc = sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local>(sycl::range<1>(max_work_group_size), h);
        h.parallel_for_work_group(sycl::range<1>(max_compute_units), sycl::range<1>(max_work_group_size), [=](sycl::group<1> gr)
        {
            gr.parallel_for_work_item([&](sycl::h_item<1> it)
            {
                auto gid = it.get_global_id();
                auto lid = it.get_local_id();
                acc[gid] = lid;
            });
            // gr.parallel_for_work_item([&](sycl::h_item<1> it)
            // {
            //     auto gid = it.get_global_id();
            //     auto lid = it.get_local_id();
            //     out << "Global id: " << gid << " Local id: " << lid << " Local acc: " << local_acc[lid] << sycl::endl;
            //     local_acc[lid] += 1;

            // });

        }); });

    std::vector<int> out(N);
    q.submit([&](sycl::handler &h)
             {
        h.depends_on(event);
        auto acc =  buf.template get_access<sycl::access::mode::read_write>(h);
        h.copy(acc, out.data());
    }).wait();
    std::for_each(out.begin(), out.end(), [](int i){ if (i != 0){std::cout << i << ", ";} });
    std::cout << std::endl;

    return 0;
}
