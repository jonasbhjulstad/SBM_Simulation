#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP
#include <CL/sycl.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <Static_RNG/Distributions/Bernoulli.hpp>
#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <Sycl_Buffer_Routines/Buffer_Utils.hpp>
#include <Sycl_Buffer_Routines/Buffer_Validation.hpp>

namespace SBM_Simulation {
sycl::event recover(sycl::queue &q, const SBM_Database::Sim_Param &p,
                    sycl::buffer<SIR_State, 3> &vertex_state,
                    sycl::buffer<Static_RNG::default_rng> &rngs, uint32_t t,
                    const sycl::nd_range<1>&nd_range, sycl::event &dep_event);

sycl::event initialize_vertices(sycl::queue &q,
                                const SBM_Database::Sim_Param &p,
                                sycl::buffer<SIR_State, 3> &vertex_state,
                                sycl::buffer<Static_RNG::default_rng> &rngs,
                                const sycl::nd_range<1>&nd_range,
                                std::vector<sycl::event>& dep_events); 

sycl::event infect(sycl::queue &q, const SBM_Database::Sim_Param &p,
                   Sim_Buffers &b, uint32_t t, const sycl::nd_range<1>&nd_range,
                   sycl::event &dep_event); 
} // namespace SBM_Simulation
#endif
