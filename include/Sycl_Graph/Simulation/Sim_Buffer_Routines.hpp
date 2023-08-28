#ifndef SIM_BUFFER_ROUTINES_HPP
#define SIM_BUFFER_ROUTINES_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <CL/sycl.hpp>
SYCL_EXTERNAL void single_community_state_accumulate(sycl::item<1> &it, const auto &vcm,
                                       const auto &vertex_state,
                                       auto &state_acc,
                                       uint32_t N_communities,
                                       uint32_t N_vertices,
                                       uint32_t Nt);

sycl::event accumulate_community_state(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b, std::vector<sycl::event> &dep_events);

#endif
