#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Static_RNG/distributions.hpp>
std::vector<sycl::event> recover(sycl::queue &q,
                                 const Sim_Param &p,
                                 sycl::buffer<SIR_State, 3> &vertex_state,
                                 sycl::buffer<Static_RNG::default_rng> &rngs,
                                 uint32_t t,
                                 std::vector<sycl::event> &dep_event);
sycl::event initialize_vertices(sycl::queue &q, const Sim_Param &p,
sycl::buffer<SIR_State, 3> &vertex_state,
sycl::buffer<Static_RNG::default_rng> &rngs);

std::vector<sycl::event> infect(sycl::queue &q,
                                           const Sim_Param &p,
                                           Sim_Buffers &b,
                                           uint32_t t,
                                           std::vector<sycl::event> &dep_event);
#endif