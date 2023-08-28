#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <algorithm>


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


void Sim_Param::print() const
{
    std::cout << "SBM:\t" << N_communities << " communities, with " << N_pop << " individuals" << "\n";
    std::cout << "N_sims:\t" << N_sims << "\n";
    std::cout << "p_in:\t" << p_in << "\n";
    std::cout << "p_out:\t" << p_out << "\n";
    std::cout << "Nt:\t" << Nt << "\n";
    std::cout << "Initial probabilities:\t " << p_I0 << ", " << p_R0 << "\n";
    std::cout << "p_I in:\t [" << p_I_min << "," << p_I_max << "]" << "\n";
    std::cout << "compute_range:\t" << compute_range[0] << "\n";
    std::cout << "work_group_range:\t" << wg_range[0] << "\n";

}
