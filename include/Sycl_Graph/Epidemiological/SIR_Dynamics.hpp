#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
#include <Sycl_Graph/Utils/Buffer_Validation.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>


std::vector<sycl::event> recover(sycl::queue &q,
                                 const Sim_Param &p,
                                 sycl::buffer<SIR_State, 3> &vertex_state,
                                 sycl::buffer<Static_RNG::default_rng> &rngs,
                                 uint32_t t,
                                 sycl::range<1> compute_range,
                                sycl::range<1> wg_range,
                                 std::vector<sycl::event> &dep_event);
sycl::event initialize_vertices(sycl::queue &q, const Sim_Param &p,
                                sycl::buffer<SIR_State, 3> &vertex_state,
                                sycl::buffer<Static_RNG::default_rng> &rngs,
                                sycl::range<1> compute_range,
                                sycl::range<1> wg_range,
                                std::vector<sycl::event> dep_events);

std::vector<sycl::event> infect(sycl::queue &q,
                                const Sim_Param &p,
                                Sim_Buffers &b,
                                uint32_t t,
                                sycl::range<1> compute_range,
                                sycl::range<1> wg_range,
                                std::vector<sycl::event> &dep_event);
#endif
