#ifndef SIR_SIMULATION_HPP
#define SIR_SIMULATION_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>
struct Sim_Data
{
    Sim_Data(): trajectory(sycl::range<2>(0, 0)), event_from_buf(sycl::range<2>(0, 0)), event_to_buf(sycl::range<2>(0, 0)) {}
    Sim_Data(sycl::buffer<SIR_State, 2> &trajectory,
             sycl::buffer<uint32_t, 2> &event_from_buf,
             sycl::buffer<uint32_t, 2> &event_to_buf) : trajectory(std::move(trajectory)), event_from_buf(std::move(event_from_buf)), event_to_buf(std::move(event_to_buf)) {}
    sycl::buffer<SIR_State, 2> trajectory;
    sycl::buffer<uint32_t, 2> event_from_buf;
    sycl::buffer<uint32_t, 2> event_to_buf;
    sycl::event event;
    std::string output_dir;
    uint32_t seed;
    uint32_t sim_idx;
    std::vector<std::pair<uint32_t, uint32_t>> ccm;
    std::vector<uint32_t> ccm_weights;
    std::vector<std::vector<float>> p_I_vec;
    std::vector<uint32_t> vcm;
};
Sim_Data excite_simulate(const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir = "./", bool debug_flag = false, sycl::queue q = sycl::queue(sycl::gpu_selector_v));
void parallel_excite_simulate(const Sim_Param &p, const std::vector<uint32_t> &vcm, const std::vector<std::pair<uint32_t, uint32_t>> &edge_list, float p_I_min, float p_I_max, const std::string output_dir, uint32_t N_simulations, sycl::queue q = sycl::queue(sycl::gpu_selector_v), bool debug_flag = false);

#endif
