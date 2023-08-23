#include <CL/sycl.hpp>
#include <iostream>
int main()
{
    std::vector<sycl::queue> qs;
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto &d : devices)
    {
        qs.emplace_back(d);
    }
    auto q = qs[0];
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto N_compute = device.get_info<sycl::info::device::max_compute_units>();
    // auto device_info = get_device_info(q);
    // device_info.print();
    // auto N0 = N_compute;
    auto N0 = N_compute;
    auto N1 = max_wg_size;
    auto N2 = 1;
    auto buf_range = sycl::range<3>(N0, N1, N2);
    sycl::buffer<uint8_t, 3> buf(buf_range);
    for (int wg_size = 0; wg_size < max_wg_size; wg_size++)
    {
        q.submit([&](sycl::handler &h)
                 {
        auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for_work_group(sycl::range<1>(N0), sycl::range<1>(wg_size), [=](sycl::group<1> gr)
        {
            auto group_id = gr.get_id(0);
            gr.parallel_for_work_item([&](sycl::h_item<1> h)
            {
                auto local_id = h.get_local_id(0);
                for(int i = 0; i < N2; i++)
                {
                    acc[group_id][local_id][i] = 1;
                }

            });
        }); })
            .wait();
    }
    return 0;
}
