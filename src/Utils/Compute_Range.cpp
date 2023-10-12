#include <Sycl_Graph/Utils/Compute_Range.hpp>
#include <cmath>
sycl::range<1> get_wg_range(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    return max_wg_size;
}

sycl::range<1> get_compute_range(sycl::queue &q)
{
    auto device = q.get_device();
    auto wg_size = get_wg_range(q);
    auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
    return max_compute_units * wg_size;
}

sycl::range<1> get_compute_range(sycl::queue &q, uint32_t N_sims)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    float d_sims = N_sims;
    float d_max_wg_size = max_wg_size;
    auto N_compute_units = static_cast<uint32_t>(std::ceil(d_sims / d_max_wg_size));
    return N_compute_units;
}
