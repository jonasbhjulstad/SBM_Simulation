#pragma once
#include <memory>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SIR_SBM/graph/graph.hpp>
#include <SIR_SBM/simulation/sim_buffers.hpp>

namespace SIR_SBM {

sycl::event simulation_step(sycl::queue &q, Sim_Buffers &SB,
                            float p_I, float p_R, uint32_t t, uint32_t t_offset,
                            sycl::event dep_event = {});

sycl::event simulation_alloc_step(sycl::queue &q,
                                  Sim_Buffers &SB, float p_I,
                                  float p_R, uint32_t t_offset,
                                  sycl::event dep_event = {});

sycl::event run_simulation(sycl::queue &q, Sim_Buffers &SB,
                           const Sim_Param &p);
} // namespace SIR_SBM