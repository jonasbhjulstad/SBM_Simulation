#ifndef SYCL_GRAPH_SIMULATION_HPP
#define SYCL_GRAPH_SIMULATION_HPP
#include <Sycl_Graph/Epidemiological/SIR_Dynamics.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Simulation/Sim_Infection_Sampling.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
void run(sycl::queue &q, Sim_Param p, Sim_Buffers &b);

sycl::event accumulate_community_state(sycl::queue &q, std::vector<sycl::event> &dep_events, sycl::buffer<SIR_State, 3> &v_buf, sycl::buffer<uint32_t, 2> &vcm_buf, sycl::buffer<State_t, 3> community_buf, sycl::range<1> compute_range, sycl::range<1> wg_range, uint32_t N_sims);
void run(sycl::queue &q, Sim_Param p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm);


void run(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm);

void p_I_run(sycl::queue &q, Sim_Param p, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &edge_list, const std::vector<std::vector<uint32_t>> &vcm, const std::vector<std::vector<std::vector<float>>> &p_Is);

void p_I_run(sycl::queue &q, Sim_Param p, const Dataframe_t<std::pair<uint32_t, uint32_t>, 2> &edge_list, const Dataframe_t<uint32_t, 2> &vcm, const Dataframe_t<float, 3> &p_Is);

#endif
