#ifndef SIM_TYPES_HPP
#define SIM_TYPES_HPP
#include <cstdint>
#include <string>
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
struct Sim_Data
{
    Sim_Data(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections);
    std::vector<std::vector<std::vector<uint32_t>>> events_to_timeseries;
    std::vector<std::vector<std::vector<uint32_t>>> events_from_timeseries;
    std::vector<std::vector<std::vector<State_t>>> state_timeseries;
    std::vector<std::vector<std::vector<uint32_t>>> connection_infections;
};

struct Sim_Param
{
    Sim_Param(sycl::queue& q, uint32_t N_sims);
    std::tuple<sycl::range<1>, sycl::range<1>> sim_ranges(sycl::queue& q, uint32_t N_sims) const;
    uint32_t N_communities = 4;
    uint32_t N_pop = 100;
    uint32_t N_sims = 2;
    float p_in = 1.0f;
    float p_out = .0f;
    uint32_t Nt = 30;
    float p_R0 = .0f;
    float p_I0;
    float p_R;
    float p_I_min = 0.0f;
    float p_I_max = 0.0f;
    uint32_t Nt_alloc = 2;
    uint32_t N_threads = 1024;
    uint32_t seed = 238;
    uint32_t max_infection_samples = 1000;
    sycl::range<1> compute_range;
    sycl::range<1> wg_range;
    std::size_t N_vertices() const {return N_communities * N_pop;}
    std::string output_dir;
};
std::size_t get_sim_data_byte_size(uint32_t Nt, uint32_t N_sims, uint32_t N_communities, uint32_t N_connections);

#endif
