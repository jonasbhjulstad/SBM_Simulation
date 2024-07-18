#pragma once
#include <sycl/sycl.hpp>
#include <SIR_SBM/epidemiological/epidemiological.hpp>

namespace SIR_SBM {

void validate_population(sycl::queue &q, sycl::buffer<SIR_State, 3> &state);

sycl::event partition_population_count(sycl::queue &q,
                                       sycl::buffer<SIR_State, 3> &state,
                                       sycl::buffer<Population_Count, 3> &count,
                                       sycl::buffer<uint32_t> &vpc,
                                       uint32_t t_offset,
                                       sycl::event dep_event = {});

std::vector<Population_Count>
partition_population_count(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                           sycl::buffer<uint32_t> &vpc, uint32_t t_offset);

} // namespace SIR_SBM