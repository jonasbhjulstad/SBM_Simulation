#pragma once
#include <SIR_SBM/graph/graph.hpp>
#include <cstdint>
#include <oneapi/dpl/random>
#include <sycl/sycl.hpp>
#include <tuple>

namespace SIR_SBM {

enum class SIR_State : char {
  Susceptible = 0,
  Infected = 1,
  Recovered = 2,
  Invalid = 3
};

struct Population_Count {
  int S, I, R;
  Population_Count();
  Population_Count(int S, int I, int R);
  Population_Count(const std::array<int, 3> &arr);
  Population_Count operator+(const Population_Count &other) const;
  bool is_zero() const;
  int &operator[](SIR_State s);
};

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
                   uint32_t t, uint32_t t_offset, sycl::event dep_event = {});

} // namespace SIR_SBM