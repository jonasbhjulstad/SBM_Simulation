#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <algorithm>
Sim_Data::Sim_Data(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections) : events_to_timeseries(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0))),
                                                                                                   events_from_timeseries(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0))),
                                                                                                   state_timeseries(N_sims, std::vector<std::vector<State_t>>(Nt+1, std::vector<State_t>(N_communities, {0, 0, 0}))),
                                                                                                   connection_infections(N_sims, std::vector<std::vector<uint32_t>>(Nt, std::vector<uint32_t>(N_connections, 0)))
{
}

std::size_t get_sim_data_byte_size(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections)
{
    return sizeof(uint32_t) * Nt * N_sims * N_connections * 2 + sizeof(State_t) * Nt * N_sims * N_communities + sizeof(uint32_t) * Nt * N_sims * N_connections;
}

sycl::range<1> get_wg_range(sycl::queue &q)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    return max_wg_size;
}

sycl::range<1> get_compute_range(sycl::queue &q, uint32_t N_sims)
{
    auto device = q.get_device();
    auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    double d_sims = N_sims;
    double d_max_wg_size = max_wg_size;
    auto N_compute_units = static_cast<uint32_t>(std::ceil(d_sims / d_max_wg_size));
    return N_compute_units;
}

Sim_Param::Sim_Param(sycl::queue &q, uint32_t N_sims): wg_range(sycl::range<1>(std::min<uint32_t>({(uint32_t)get_wg_range(q)[0], N_sims}))), compute_range(get_compute_range(q, N_sims)), N_sims(N_sims) {}
