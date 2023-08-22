#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>

int main()
{
    sycl::queue q(sycl::gpu_selector_v);
    auto device = q.get_device();
    //get number of compute units
    auto N_compute_units = device.get_info<sycl::info::device::max_compute_units>();
    //get max work group size
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    //get max work item sizes
    std::cout << "Running over " << N_compute_units << " compute units, with a max work group size of " << max_wg_size << std::endl;

    sycl::buffer<int> buf((sycl::range<1>(N_compute_units*max_wg_size)));
    q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for_work_group(sycl::range<1>(N_compute_units), sycl::range<1>(max_wg_size), [=](sycl::group<1> gr)
        {
            gr.parallel_for_work_item([&](sycl::h_item<1> h)
            {
                auto gid = h.get_global_id(0);
                acc[gid] = gid;
            });
        });
    }).wait();

    std::vector<int> data(N_compute_units*max_wg_size);
    q.submit([&](sycl::handler& h)
    {
        auto acc = buf.template get_access<sycl::access::mode::read>(h);
        h.copy(acc, data.data());
    });

    std::vector<int> true_vec(N_compute_units*max_wg_size);
    std::iota(true_vec.begin(), true_vec.end(), 0);

    assert(std::equal(data.begin(), data.end(), true_vec.begin()));
    return 0;
}
