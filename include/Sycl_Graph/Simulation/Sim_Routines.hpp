#ifndef SIM_ROUTINES_HPP
#define SIM_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>

void run(sycl::queue& q, Sim_Param p, Sim_Buffers& b);
void run(sycl::queue& q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_list, const std::vector<std::vector<uint32_t>>& vcm, const std::vector<std::vector<uint32_t>>& ecm);

void p_I_run(sycl::queue& q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& edge_list, const std::vector<std::vector<uint32_t>>& vcm, const std::vector<std::vector<uint32_t>>& ecm, const std::vector<std::vector<std::vector<float>>>& p_Is);
#endif
