#ifndef SIM_ROUTINES_HPP
#define SIM_ROUTINES_HPP
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
void community_state_to_timeseries(sycl::queue &q,
                                   sycl::buffer<SIR_State, 3> &vertex_state,
                                   sycl::buffer<State_t, 3> &community_state,
                                   std::vector<std::vector<std::vector<State_t>>> &community_timeseries,
                                   sycl::buffer<uint32_t> &vcm,
                                   uint32_t t_offset,
                                   std::vector<sycl::event> &dep_events);
void connection_events_to_timeseries(sycl::queue &q,
                                     sycl::buffer<uint32_t, 3> events_from,
                                     sycl::buffer<uint32_t, 3> &events_to,
                                     std::vector<std::vector<std::vector<State_t>>> &community_timeseries,
                                     sycl::buffer<uint32_t> &vcm,
                                     uint32_t t_offset,
                                     std::vector<sycl::event> &dep_events);

void run(sycl::queue& q, const Sim_Param& p, Sim_Buffers& b);


#endif
