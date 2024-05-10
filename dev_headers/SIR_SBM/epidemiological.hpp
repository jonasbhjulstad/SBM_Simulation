#pragma once
#hdr
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/graph.hpp>
#include <SIR_SBM/sycl_routines.hpp>
#include <oneapi/dpl/random>
#end
namespace SIR_SBM {
#hdr
enum class SIR_State : char {
  Susceptible = 0,
  Infected = 1,
  Recovered = 2,
  Invalid = 3
};
#end

struct Population_Count {
  int S, I, R;
  Population_Count() : S(0), I(0), R(0) {}
  Population_Count(int S, int I, int R) : S(S), I(I), R(R) {}
  Population_Count(const std::array<int, 3> &arr)
      : S(arr[0]), I(arr[1]), R(arr[2]) {}
  Population_Count operator+(const Population_Count &other) const {
    return Population_Count{S + other.S, I + other.I, R + other.R};
  }
  bool is_zero() const { return S == 0 && I == 0 && R == 0; }
  int &operator[](SIR_State s) {
    switch (s) {
    case SIR_State::Susceptible:
      return S;
    case SIR_State::Infected:
      return I;
    case SIR_State::Recovered:
      return R;
    default:
      return S;
    }
  }
};

Population_Count state_to_count(SIR_State s) {
  switch (s) {
  case SIR_State::Susceptible:
    return Population_Count{1, 0, 0};
  case SIR_State::Infected:
    return Population_Count{0, 1, 0};
  case SIR_State::Recovered:
    return Population_Count{0, 0, 1};
  default:
    return Population_Count{0, 0, 0};
  }
}

std::tuple<size_t, size_t, size_t>
init_range(sycl::buffer<SIR_State, 3> &state) {
  return std::make_tuple(state.get_range()[0], state.get_range()[1], 1);
}

sycl::event initialize(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                       sycl::buffer<oneapi::dpl::ranlux48> &rngs, float p_I0) {

  auto [N_vertices, N_sims, _] = init_range(state);
  validate_range<3>(sycl::range<3>(N_vertices, N_sims, 1), state.get_range());
  return q.submit([&](sycl::handler &h) {
    auto init_range = sycl::range<3>({N_vertices, N_sims, 1});
    auto state_acc = sycl::accessor<SIR_State, 3, sycl::access::mode::write>(
        state, h, init_range);
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      oneapi::dpl::bernoulli_distribution dist(p_I0);
      auto &rng = rng_acc[idx];
      for (int i = 0; i < N_vertices; i++) {
        if (dist(rng)) {
          state_acc[sycl::range<3>(i, idx[0], 0)] = SIR_State::Infected;
        }
      }
    });
  });
}

sycl::event state_copy(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                       size_t t_src, size_t t_dest,
                       sycl::event dep_event = {}) {
  throw_if(t_dest >= state.get_range()[2], "Invalid dest time step");
  throw_if(t_src >= state.get_range()[2], "Invalid source time step");

  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto timestep_range =
        sycl::range<3>(state.get_range()[0], state.get_range()[1], 1);
    auto src_acc = sycl::accessor<SIR_State, 3, sycl::access::mode::read>(
        state, h, timestep_range, sycl::range<3>(0, 0, t_src));
    auto dest_acc = sycl::accessor<SIR_State, 3, sycl::access::mode::write>(
        state, h, timestep_range, sycl::range<3>(0, 0, t_dest));
    h.copy(src_acc, dest_acc);
  });
}

// runs inplace recovery on vertices at time t

sycl::event recover(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                    sycl::buffer<oneapi::dpl::ranlux48> &rngs, float p_R,
                    uint32_t t, sycl::event dep_event = {}) {
  throw_if(t > state.get_range()[2], "Invalid time step");
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    size_t N_sims = state.get_range()[1];
    size_t N_vertices = state.get_range()[0];
    auto state_acc =
        sycl::accessor<SIR_State, 3, sycl::access::mode::read_write>(state, h);
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      oneapi::dpl::bernoulli_distribution dist(p_R);
      auto rng = rng_acc[idx];
      for (int i = 0; i < N_vertices; i++) {
        if (state_acc[sycl::range<3>(i, idx[0], t)] == SIR_State::Infected) {
          if (dist(rng)) {
            state_acc[sycl::range<3>(i, idx[0], t)] = SIR_State::Recovered;
          }
        }
      }
      rng_acc[idx] = rng;
    });
  });
}

sycl::event infect(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                   sycl::buffer<Edge_t> &edges, sycl::buffer<uint32_t> &ecc,
                   sycl::buffer<uint32_t, 3> &contact_events,
                   sycl::buffer<oneapi::dpl::ranlux48> &rngs, float p_I,
                   uint32_t t, uint32_t t_offset, sycl::event dep_event = {}) {
  throw_if(t_offset > contact_events.get_range()[2], "Invalid time step");
  size_t N_vertices = state.get_range()[0];
  size_t N_sims = state.get_range()[1];
  size_t N_edges = edges.size();
  size_t N_connections = contact_events.get_range()[0] / 2;
  size_t t_offset_inf_count = t + t_offset - 1;
  auto zero_evt =
      zero_fill(q, contact_events, sycl::range<3>(2 * N_connections, N_sims, 1),
                sycl::range<3>(0, 0, t_offset_inf_count), dep_event);

  return q.submit([&](sycl::handler &h) {
    h.depends_on(zero_evt);
    // uint32_t Nt_alloc = state.get_range()[2];
    auto state_acc =
        sycl::accessor<SIR_State, 3, sycl::access::mode::read_write>(
            state, h, sycl::range<3>(N_vertices, N_sims, 1),
            sycl::range<3>(0, 0, t));
    auto edges_acc = edges.template get_access<sycl::access::mode::read>(h);
    auto ecc_acc = ecc.template get_access<sycl::access::mode::read>(h);
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    auto infected_count_acc =
        sycl::accessor<uint32_t, 3, sycl::access::mode::read_write>(
            contact_events, h, sycl::range<3>(2 * N_connections, N_sims, 1),
            sycl::range<3>(0, 0, t_offset_inf_count));
    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      auto rng = rng_acc[idx];
      auto is_directed_sus_inf_pair = [](SIR_State from, SIR_State to) {
        return from == SIR_State::Infected && to == SIR_State::Susceptible;
      };

      oneapi::dpl::bernoulli_distribution dist(p_I);
      uint32_t e_offset = 0;
      for (int c_idx = 0; c_idx < N_connections; c_idx++) {
        auto N_connection_edges = ecc_acc[c_idx];
        for (int e_idx = e_offset; e_idx < e_offset + N_connection_edges;
             e_idx++) {
          auto edge = edges_acc[e_idx];
          auto [from_id, to_id] = edge;
          auto connection_id = c_idx;

          SIR_State &from = state_acc[sycl::range<3>(from_id, idx[0], 0)];
          SIR_State &to = state_acc[sycl::range<3>(to_id, idx[0], 0)];
          if (is_directed_sus_inf_pair(from, to) && dist(rng)) {
            to = SIR_State::Infected;
            infected_count_acc[sycl::range<3>(2 * connection_id, idx[0], 0)] +=
                1;
          } else if (is_directed_sus_inf_pair(to, from) && dist(rng)) {
            from = SIR_State::Infected;
            infected_count_acc[sycl::range<3>(2 * connection_id + 1, idx[0],
                                              0)] += 1;
          }
        }
        e_offset += N_connection_edges;
      }
      rng_acc[idx] = rng;
    });
  });
}

} // namespace SIR_SBM