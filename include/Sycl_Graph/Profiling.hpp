#ifndef SYCL_GRAPH_DEVICE_PROFILING_HPP
#define SYCL_GRAPH_DEVICE_PROFILING_HPP

#include <Sycl_Graph/SIR_Types.hpp>
#include <string>
#include <CL/sycl.hpp>
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
    void print();
};
std::vector<Device_Info> get_device_info(std::vector<sycl::queue> &qs);
std::vector<uint32_t> determine_device_workload(const Sim_Param& p, uint32_t N_sims, std::vector<sycl::queue>& qs);


#endif
