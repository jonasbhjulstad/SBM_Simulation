#pragma once
#include <SIR_SBM/epidemiological/types.hpp>
#include <SIR_SBM/graph/graph.hpp>
#include <cstdint>
#include <oneapi/dpl/random>
#include <sycl/sycl.hpp>
#include <tuple>

namespace SIR_SBM {
Population_Count state_to_count(SIR_State s);

std::tuple<uint32_t, uint32_t, uint32_t>
init_range(sycl::buffer<SIR_State, 3> &state);

sycl::event initialize(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                       sycl::buffer<oneapi::dpl::ranlux48> &rngs, float p_I0);

sycl::event state_copy(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                       uint32_t t_src, uint32_t t_dest,
                       sycl::event dep_event = {});

// runs inplace recovery on vertices at time t
sycl::event recover(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                    sycl::buffer<oneapi::dpl::ranlux48> &rngs, float p_R,
                    uint32_t t, sycl::event dep_event = {});

sycl::event infect(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                   sycl::buffer<Edge_t> &edges, sycl::buffer<uint32_t> &ecc,
                   sycl::buffer<uint32_t, 3> &contact_events,
                   sycl::buffer<oneapi::dpl::ranlux48> &rngs, float p_I,
                   uint32_t t, sycl::event dep_event = {});

} // namespace SIR_SBM