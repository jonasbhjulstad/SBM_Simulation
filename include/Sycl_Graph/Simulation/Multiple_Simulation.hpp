#ifndef SYCL_GRAPH_SIMULATION_HPP
#define SYCL_GRAPH_SIMULATION_HPP
#include <Sycl_Graph/Utils/Common.hpp>
#include <Sycl_Graph/Dynamics.hpp>
#include <Sycl_Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
void run(sycl::queue &q, Multiple_Sim_Param_t p, Sim_Buffers &b);

void run(sycl::queue &q, Multiple_Sim_Param_t p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm);

void p_I_run(sycl::queue &q, Multiple_Sim_Param_t p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm, const std::vector<std::vector<std::vector<float>>> &p_Is);

void multiple_sim_param_run(sycl::queue &q, const Multiple_Sim_Param_t &p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 3> &edge_list, const Dataframe_t<uint32_t, 3> &vcm, Dataframe_t<float, 4> p_Is = {});

#endif
