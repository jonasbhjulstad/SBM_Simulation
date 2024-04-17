#pragma once
#include <SIR_SBM/common.hpp>
#include <SIR_SBM/epidemiological.hpp>
#include <SIR_SBM/exceptions.hpp>
#include <SIR_SBM/sycl_routines.hpp>
namespace SIR_SBM {
struct Sim_Param {
  float p_I0;
  size_t Nt;
  size_t Nt_alloc;
  size_t N_I_terminate;
  size_t N_sims;
  uint32_t seed;
};
template <typename RNG> struct Sim_Buffers {
  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p) {
    events.push_back(buffer_copy(q, ecm, edge_connection_map(G)));
    events.push_back(buffer_copy(q, vpm, vertex_partition_map(G)));
    events.push_back(buffer_copy(q, edges, G.flat_edges()));
    events.push_back(
        buffer_copy(q, rngs, generate_rngs<RNG>(p.seed, p.N_sims)));
    state = sycl::buffer<SIR_State, 3>(
        sycl::range<3>(G.N_vertices(), p.N_sims, p.Nt_alloc));
    infected_count = sycl::buffer<uint32_t, 3>(
        sycl::range<3>(G.N_connections() * 2, p.N_sims, p.Nt_alloc));
  }

  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p,
              const std::vector<uint32_t> &pcm)
      : Sim_Buffers(q, G, p) {
    throw_if(pcm.size() != G.N_partitions(),
             "Invalid number of partitions in partition community map");
    events.push_back(buffer_copy(q, this->pcm, pcm));
  }

  void wait() const { sycl::event::wait(events); }

  sycl::buffer<uint32_t> ecm = dummy_buf_1<uint32_t>();
  sycl::buffer<uint32_t> vpm = dummy_buf_1<uint32_t>();
  sycl::buffer<uint32_t> pcm = dummy_buf_1<uint32_t>();
  sycl::buffer<Edge_t> edges = dummy_buf_1<Edge_t>();
  sycl::buffer<SIR_State, 3> state = dummy_buf_3<SIR_State>();
  sycl::buffer<uint32_t, 3> infected_count = dummy_buf_3<uint32_t>();
  sycl::buffer<RNG, 1> rngs = dummy_buf_1<RNG>();

private:
  std::vector<sycl::event> events;
};

std::tuple<size_t, size_t, size_t>
init_range(sycl::buffer<SIR_State, 3> &state) {
  return std::make_tuple(state.get_range()[0], state.get_range()[1], 1);
}

template <typename RNG>
sycl::event initialize(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                       sycl::buffer<RNG> &rngs, float p_I0) {

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
                       size_t t) {
  throw_if((t + 1) >= state.get_range()[2], "Invalid time step");

  return q.submit([&](sycl::handler &h) {
    auto state_acc =
        state.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<3>(state.get_range()[0], state.get_range()[1], 1), [=](sycl::id<3> idx) {
      state_acc[sycl::range<3>(idx[0
      ], idx[1], t + 1)] =
          state_acc[sycl::range<3>(idx[0], idx[1], t)];
      });
  });
}

template <typename RNG>
sycl::event recover(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                    sycl::buffer<RNG> &rngs, float p_R, uint32_t t) {
  throw_if((t + 1) > state.get_range()[2], "Invalid time step");
  auto cpy_event = state_copy(q, state, t);
  return q.submit([&](sycl::handler &h) {
    h.depends_on(cpy_event);
    size_t N_sims = state.get_range()[0];
    size_t N_vertices = state.get_range()[1];
    auto state_acc =
        sycl::accessor<SIR_State, 3, sycl::access::mode::read_write>(state, h);
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      oneapi::dpl::bernoulli_distribution dist(p_R);
      auto rng = rng_acc[idx];
      for (int i = 0; i < N_vertices; i++) {
        if (state_acc[sycl::range<3>(idx[0], i, t + 1)] ==
            SIR_State::Infected) {
          if (dist(rng)) {
            state_acc[sycl::range<3>(idx[0], i, t + 1)] = SIR_State::Recovered;
          }
        }
      }
      rng_acc[idx] = rng;
    });
  });
}

template <typename RNG>
sycl::event infect(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                   sycl::buffer<Edge_t> &edges, sycl::buffer<uint32_t> &ecm,
                   sycl::buffer<uint32_t, 3> &infected_count,
                   sycl::buffer<RNG> &rngs, float p_I, uint32_t t) {
  throw_if(t > state.get_range()[2], "Invalid time step");
  return q.submit([&](sycl::handler &h) {
    size_t N_sims = state.get_range()[0];
    size_t N_vertices = state.get_range()[1];
    size_t N_edges = edges.size();
    auto state_acc =
        sycl::accessor<SIR_State, 3, sycl::access::mode::read_write>(state, h);
    auto edges_acc = edges.template get_access<sycl::access::mode::read>(h);
    auto ecm_acc = ecm.template get_access<sycl::access::mode::read>(h);
    auto infected_count_acc =
        infected_count.template get_access<sycl::access::mode::read_write>(h);
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      oneapi::dpl::bernoulli_distribution dist(p_I);
      for (int i = 0; i < N_vertices; i++) {
        if (state_acc[sycl::range<3>(idx[0], i, t)] == SIR_State::Susceptible) {
          for (int j = ecm_acc[i]; j < ecm_acc[i + 1]; j++) {
            if (dist(rng_acc[idx]) &&
                state_acc[sycl::range<3>(idx[0], edges_acc[j].first, t)] ==
                    SIR_State::Infected) {
              state_acc[sycl::range<3>(idx[0], i, t)] = SIR_State::Infected;
              infected_count_acc[sycl::range<3>(j, idx[0], t)]++;
              break;
            }
          }
        }
      }
    });
  });
}
} // namespace SIR_SBM