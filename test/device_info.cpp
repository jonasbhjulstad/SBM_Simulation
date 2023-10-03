#include <CL/sycl.hpp>
#include <fstream>
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
    // uint32_t max_work_item_sizes_1D;
    // std::tuple<uint32_t, uint32_t> max_work_item_sizes_2D;
    // std::tuple<uint32_t, uint32_t, uint32_t> max_work_item_sizes_3D;
    void print();
    std::string info_string() const;
};
Device_Info get_device_info(sycl::queue& q);

std::vector<Device_Info> get_device_info(std::vector<sycl::queue> &qs);
void Device_Info::print()
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
    // std::cout << "max_work_item_sizes_1D: " << max_work_item_sizes_1D << std::endl;
    // std::cout << "max_work_item_sizes_2D: " << std::get<0>(max_work_item_sizes_2D) << ", " << std::get<1>(max_work_item_sizes_2D) << std::endl;
    // std::cout << "max_work_item_sizes_3D: " << std::get<0>(max_work_item_sizes_3D) << ", " << std::get<1>(max_work_item_sizes_3D) << ", " << std::get<2>(max_work_item_sizes_3D) << std::endl;
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
    // ss << "max_work_item_sizes_1D: " << max_work_item_sizes_1D << std::endl;
    // ss << "max_work_item_sizes_2D: " << std::get<0>(max_work_item_sizes_2D) << ", " << std::get<1>(max_work_item_sizes_2D) << std::endl;
    // ss << "max_work_item_sizes_3D: " << std::get<0>(max_work_item_sizes_3D) << ", " << std::get<1>(max_work_item_sizes_3D) << ", " << std::get<2>(max_work_item_sizes_3D) << std::endl;
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
        // auto size_1D = device.template get_info<sycl::info::device::max_work_item_sizes<1>>();
        // auto size_2D = device.template get_info<sycl::info::device::max_work_item_sizes<2>>();
        // auto size_3D = device.template get_info<sycl::info::device::max_work_item_sizes<3>>();
        // info.max_work_item_sizes_1D = size_1D[0];
        // info.max_work_item_sizes_2D = {size_2D[0], size_2D[1]};
        // info.max_work_item_sizes_3D = {size_3D[0], size_3D[1], size_3D[2]};
        return info; });
    return device_infos;
}


int main()
{
    //create queues for all gpu devices
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    std::vector<sycl::queue> queues;
    for(auto &device : devices)
        queues.emplace_back(device);

    std::vector<Device_Info> device_infos;
    for(auto &queue : queues)
        device_infos.emplace_back(get_device_info(queue));

    auto idx = 0;
    for(auto&& info: device_infos)
    {
        //ofstream with name of device
        std::ofstream file(info.name + "_" + std::to_string(idx) + ".csv");
        //
        file << info.info_string();
    }
    return 0;
}
