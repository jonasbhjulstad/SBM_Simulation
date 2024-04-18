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

struct Sim_Result {
  std::vector<uint32_t> infection_count;
  std::vector<Population_Count> population_count;
};

template <typename RNG> struct Sim_Buffers {
  Sim_Buffers(size_t N_vertices, size_t N_sims, size_t Nt, size_t Nt_alloc,
              size_t N_edges, size_t N_partitions)
      : ecm{N_edges}, vpm{N_vertices}, edges{N_edges}, rngs{N_sims},
        state{sycl::range<3>(N_vertices, N_sims, Nt_alloc)},
        infected_count{sycl::range<3>(N_partitions * 2, N_sims, Nt + 1)},
        population_count{sycl::range<3>(N_partitions, N_sims, Nt + 1)} {}

  Sim_Buffers(size_t N_vertices, size_t N_sims, size_t Nt, size_t Nt_alloc,
              size_t N_edges, size_t N_partitions, Sim_Result &result)
      : ecm{N_edges}, vpm{N_vertices}, edges{N_edges}, rngs{N_sims},
        state{sycl::range<3>(N_vertices, N_sims, Nt_alloc)},
        infected_count{result.infection_count.data(),
                       sycl::range<3>(N_partitions * 2, N_sims, Nt + 1)},
        population_count{result.population_count.data(),
                         sycl::range<3>(N_partitions, N_sims, Nt + 1)} {}

  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p,
              Sim_Result &result)
      : Sim_Buffers(G.N_vertices(), p.N_sims, p.Nt, p.Nt_alloc, G.N_edges(),
                    G.N_partitions(), result) {
    result.infection_count.resize(G.N_partitions() * 2 * p.N_sims * p.Nt);
    result.population_count.resize(G.N_partitions() * p.N_sims * p.Nt);
    buffer_copy_init(q, G, p);
  }

  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p)
      : Sim_Buffers(G.N_vertices(), p.N_sims, p.Nt, p.Nt_alloc, G.N_edges(),
                    G.N_partitions()) {
    buffer_copy_init(q, G, p);
  }


  Sim_Buffers(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p,
              const std::vector<uint32_t> &pcm)
      : Sim_Buffers(q, G, p) {
    throw_if(pcm.size() != G.N_partitions(),
             "Invalid number of partitions in partition community map");
    events.push_back(buffer_copy(q, this->pcm, pcm));
  }

  void wait() const { sycl::event::wait(events); }

  sycl::buffer<uint32_t> ecm;
  sycl::buffer<uint32_t> vpm;
  sycl::buffer<Edge_t> edges;
  sycl::buffer<SIR_State, 3> state;
  sycl::buffer<uint32_t, 3> infected_count;
  sycl::buffer<Population_Count, 3> population_count;
  sycl::buffer<RNG, 1> rngs;

private:

  void buffer_copy_init(sycl::queue &q, const SBM_Graph &G, const Sim_Param &p)
  {
    events.push_back(buffer_copy(q, ecm, edge_connection_map(G)));
    events.push_back(buffer_copy(q, vpm, vertex_partition_map(G)));
    events.push_back(buffer_copy(q, edges, G.flat_edges()));
    events.push_back(
    buffer_copy(q, rngs, generate_rngs<RNG>(p.seed, p.N_sims)));
  }

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
    h.parallel_for(
        sycl::range<3>(state.get_range()[0], state.get_range()[1], 1),
        [=](sycl::id<3> idx) {
          state_acc[sycl::range<3>(idx[0], idx[1], t + 1)] =
              state_acc[sycl::range<3>(idx[0], idx[1], t)];
        });
  });
}

// runs inplace recovery on vertices at time t
template <typename RNG>
sycl::event recover(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                    sycl::buffer<RNG> &rngs, float p_R, uint32_t t,
                    sycl::event dep_event = {}) {
  throw_if(t > state.get_range()[2], "Invalid time step");
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    size_t N_sims = state.get_range()[0];
    size_t N_vertices = state.get_range()[1];
    auto state_acc =
        sycl::accessor<SIR_State, 3, sycl::access::mode::read_write>(state, h);
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      oneapi::dpl::bernoulli_distribution dist(p_R);
      auto rng = rng_acc[idx];
      for (int i = 0; i < N_vertices; i++) {
        if (state_acc[sycl::range<3>(idx[0], i, t)] == SIR_State::Infected) {
          if (dist(rng)) {
            state_acc[sycl::range<3>(idx[0], i, t)] = SIR_State::Recovered;
          }
        }
      }
      rng_acc[idx] = rng;
    });
  });
}

// runs inplace infection on vertices at time t
template <typename RNG>
sycl::event infect(sycl::queue &q, sycl::buffer<SIR_State, 3> &state,
                   sycl::buffer<Edge_t> &edges, sycl::buffer<uint32_t> &ecm,
                   sycl::buffer<uint32_t, 3> &infected_count,
                   sycl::buffer<RNG> &rngs, float p_I, uint32_t t,
                   sycl::event dep_event = {}) {
  throw_if(t > state.get_range()[2], "Invalid time step");
  return q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
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
    auto is_directed_sus_inf_pair = [](SIR_State from, SIR_State to) {
      return from == SIR_State::Susceptible && to == SIR_State::Infected;
    };

    h.parallel_for(sycl::range<1>(N_sims), [=](sycl::id<1> idx) {
      auto rng = rng_acc[idx];
      oneapi::dpl::bernoulli_distribution dist(p_I);
      for (int edge_idx = 0; edge_idx < N_edges; edge_idx++) {
        auto edge = edges_acc[edge_idx];
        auto [from_id, to_id] = edge;
        auto connection_id = ecm_acc[edge_idx];
        SIR_State &from = state_acc[sycl::range<3>(from_id, idx[0], t)];
        SIR_State &to = state_acc[sycl::range<3>(to_id, idx[0], t)];
        if (is_directed_sus_inf_pair(from, to) && dist(rng)) {
          to = SIR_State::Infected;
          infected_count_acc[sycl::range<3>(2 * connection_id, idx[0], t)] += 1;
        } else if (is_directed_sus_inf_pair(to, from) && dist(rng)) {
          from = SIR_State::Infected;
          infected_count_acc[sycl::range<3>(2 * connection_id + 1, idx[0],
                                            t)] += 1;
        }
      }
      rng_acc[idx] = rng;
    });
  });
}

template <typename RNG>
sycl::event simulation_step(sycl::queue &q, Sim_Buffers<RNG> &sb, float p_I,
                            float p_R, uint32_t t) {
  auto cpy_event = state_copy(q, sb.state, t);
  // auto rec_event = recover(q, sb.state, sb.rngs, p_R, t+1, cpy_event);
  // auto inf_event =
  // infect(q, sb.state, sb.edges, sb.ecm, sb.infected_count, sb.rngs, p_I, t+1,
  // rec_event);
  // return inf_event;
  return cpy_event;
}
} // namespace SIR_SBM