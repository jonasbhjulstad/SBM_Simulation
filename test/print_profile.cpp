#include <CL/sycl.hpp>
#include <SBM_Simulation/Simulation/Sim_Types.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
struct Device_Info
{
    sycl::info::device_type type;
    std::string vendor;
    std::string version;
    std::string name;
    uint32_t max_compute_units;
    uint32_t max_work_group_size;
    uint32_t max_clock_frequency;
    uint64_t global_mem_size;
    uint64_t local_mem_size;
    uint64_t max_mem_alloc_size;
    uint64_t global_mem_cache_size;
    uint32_t global_mem_cacheline_size;
    uint32_t max_work_item_sizes_1D;
    std::tuple<uint32_t, uint32_t> max_work_item_sizes_2D;
    std::tuple<uint32_t, uint32_t, uint32_t> max_work_item_sizes_3D;
    void print() const;
    std::string info_string() const;
};
Device_Info get_device_info(sycl::queue& q);

std::vector<Device_Info> get_device_info(std::vector<sycl::queue> &qs);
std::vector<uint32_t> determine_device_workload(const Sim_Param& p, uint32_t N_sims, std::vector<sycl::queue>& qs);

uint32_t get_event_execution_time(sycl::event& event);

void Device_Info::print() const
{
    std::cout << "Device Info:" << std::endl;
    std::cout << "name: " << name << std::endl;
    std::cout << "vendor: " << vendor << std::endl;
    std::cout << "version: " << version << std::endl;
    std::cout << "max_compute_units: " << max_compute_units << std::endl;
    std::cout << "max_work_group_size: " << max_work_group_size << std::endl;
    std::cout << "max_clock_frequency: " << max_clock_frequency << std::endl;
    std::cout << "global_mem_size: " << global_mem_size << std::endl;
    std::cout << "local_mem_size: " << local_mem_size << std::endl;
    std::cout << "max_mem_alloc_size: " << max_mem_alloc_size << std::endl;
    std::cout << "global_mem_cache_size: " << global_mem_cache_size << std::endl;
    std::cout << "global_mem_cacheline_size: " << global_mem_cacheline_size << std::endl;
    std::cout << "max_work_item_sizes_1D: " << max_work_item_sizes_1D << std::endl;
    std::cout << "max_work_item_sizes_2D: " << std::get<0>(max_work_item_sizes_2D) << ", " << std::get<1>(max_work_item_sizes_2D) << std::endl;
    std::cout << "max_work_item_sizes_3D: " << std::get<0>(max_work_item_sizes_3D) << ", " << std::get<1>(max_work_item_sizes_3D) << ", " << std::get<2>(max_work_item_sizes_3D) << std::endl;
}

std::string Device_Info::info_string() const
{
    std::stringstream ss;
    ss << "Device Info:" << std::endl;
    ss << "name: " << name << std::endl;
    ss << "vendor: " << vendor << std::endl;
    ss << "version: " << version << std::endl;
    ss << "max_compute_units: " << max_compute_units << std::endl;
    ss << "max_work_group_size: " << max_work_group_size << std::endl;
    ss << "max_clock_frequency: " << max_clock_frequency << std::endl;
    ss << "global_mem_size: " << global_mem_size << std::endl;
    ss << "local_mem_size: " << local_mem_size << std::endl;
    ss << "max_mem_alloc_size: " << max_mem_alloc_size << std::endl;
    ss << "global_mem_cache_size: " << global_mem_cache_size << std::endl;
    ss << "global_mem_cacheline_size: " << global_mem_cacheline_size << std::endl;
    ss << "max_work_item_sizes_1D: " << max_work_item_sizes_1D << std::endl;
    ss << "max_work_item_sizes_2D: " << std::get<0>(max_work_item_sizes_2D) << ", " << std::get<1>(max_work_item_sizes_2D) << std::endl;
    ss << "max_work_item_sizes_3D: " << std::get<0>(max_work_item_sizes_3D) << ", " << std::get<1>(max_work_item_sizes_3D) << ", " << std::get<2>(max_work_item_sizes_3D) << std::endl;
    return ss.str();
}

Device_Info get_device_info(sycl::queue& q)
{
    auto qs = std::vector<sycl::queue>({q});
    return get_device_info(qs)[0];
}
uint32_t get_event_execution_time(sycl::event& event)
{
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    //return in ms
    return (uint32_t)(end - start);
}
std::vector<Device_Info> get_device_info(std::vector<sycl::queue> &qs)
{
    std::vector<Device_Info> device_infos(qs.size(), Device_Info{});
    std::transform(qs.begin(), qs.end(), device_infos.begin(), [&](auto &q)
                   {
        Device_Info info{};
        auto device = q.get_device();
        info.name = device.template get_info<sycl::info::device::name>();
        info.type = device.template get_info<sycl::info::device::device_type>();
        info.vendor = device.template get_info<sycl::info::device::vendor>();
        info.version = device.template get_info<sycl::info::device::version>();
        info.max_compute_units = device.template get_info<sycl::info::device::max_compute_units>();
        info.max_work_group_size = device.template get_info<sycl::info::device::max_work_group_size>();
        info.max_clock_frequency = device.template get_info<sycl::info::device::max_clock_frequency>();
        info.global_mem_size = device.template get_info<sycl::info::device::global_mem_size>();
        info.local_mem_size = device.template get_info<sycl::info::device::local_mem_size>();
        info.max_mem_alloc_size = device.template get_info<sycl::info::device::max_mem_alloc_size>();
        info.global_mem_cache_size = device.template get_info<sycl::info::device::global_mem_cache_size>();
        auto size_1D = device.template get_info<sycl::info::device::max_work_item_sizes<1>>();
        auto size_2D = device.template get_info<sycl::info::device::max_work_item_sizes<2>>();
        auto size_3D = device.template get_info<sycl::info::device::max_work_item_sizes<3>>();
        info.max_work_item_sizes_1D = size_1D[0];
        info.max_work_item_sizes_2D = {size_2D[0], size_2D[1]};
        info.max_work_item_sizes_3D = {size_3D[0], size_3D[1], size_3D[2]};
        return info; });
    return device_infos;
}


std::vector<uint32_t> determine_device_workload(const Sim_Param& p, uint32_t N_sims, std::vector<sycl::queue>& qs)
{
    std::vector<Device_Info> infos = get_device_info(qs);
    std::vector<uint32_t> workloads(qs.size());
    //prioritize high clock frequency
    //assign workloads based on clock frequency and compute_unit capacity
    std::vector<uint32_t> clock_frequencies(qs.size());
    std::transform(infos.begin(), infos.end(), clock_frequencies.begin(), [](auto &info) { return info.max_clock_frequency; });
    std::vector<uint32_t> compute_units(qs.size());
    std::transform(infos.begin(), infos.end(), compute_units.begin(), [](auto &info) { return info.max_compute_units; });
    std::vector<uint32_t> max_work_group_sizes(qs.size());
    std::transform(infos.begin(), infos.end(), max_work_group_sizes.begin(), [](auto &info) { return info.max_work_group_size; });

    std::vector<uint32_t> effective_work_group_flops(qs.size());
    std::transform(max_work_group_sizes.begin(), max_work_group_sizes.end(), clock_frequencies.begin(), effective_work_group_flops.begin(), [&](auto size, auto freq) { return size*freq; });

    std::vector<uint32_t> q_idxs(qs.size());
    std::iota(q_idxs.begin(), q_idxs.end(), 0);


    uint32_t N_sims_left = N_sims;
    for (auto i : q_idxs)
    {
        uint32_t N_sims_i = std::round(N_sims * (float)effective_work_group_flops[i] / std::accumulate(effective_work_group_flops.begin(), effective_work_group_flops.end(), 0));
        workloads[i] = N_sims_i;
        N_sims_left -= N_sims_i;
    }

    return workloads;
}



int main()
{
    //get all cpu queues
    std::vector<sycl::queue> cpu_qs;
    for (const auto &dev : sycl::device::get_devices(sycl::info::device_type::cpu))
    {
        cpu_qs.emplace_back(dev);
    }
    //get all gpu queues
    std::vector<sycl::queue> gpu_qs;
    for (const auto &dev : sycl::device::get_devices(sycl::info::device_type::gpu))
    {
        gpu_qs.emplace_back(dev);
    }

    auto cpu_infos = get_device_info(cpu_qs);
    auto gpu_infos = get_device_info(gpu_qs);

    //wall of line

    std::cout << std::string(80, '-') << std::endl;

    std::for_each(cpu_infos.begin(), cpu_infos.end(), [](const auto &info)
                  { info.print();});

    std::cout << std::string(80, '-') << std::endl;

    std::for_each(gpu_infos.begin(), gpu_infos.end(), [](const auto &info)
                  { info.print();});

}
