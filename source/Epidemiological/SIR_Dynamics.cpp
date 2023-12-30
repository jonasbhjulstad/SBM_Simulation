
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <SBM_Simulation/Utils/Math.hpp>
// SYCL_EXTERNAL auto SBM_Simulation::floor_div(auto a, auto b) {
//   return static_cast<uint32_t>(
//       std::floor(static_cast<float>(a) / static_cast<float>(b)));
// }
namespace SBM_Simulation {
sycl::event recover(sycl::queue &q, const SBM_Database::Sim_Param &p,
                    sycl::buffer<SIR_State, 3> &vertex_state,
                    sycl::buffer<Static_RNG::default_rng> &rngs, uint32_t t,
                    const sycl::nd_range<1>&nd_range, sycl::event &dep_event) {
  float p_R = p.p_R;
  uint32_t N_vertices = vertex_state.get_range()[2];
  uint32_t N_sims = p.N_sims;
  auto cpy_event = q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto v_prev =
        Buffer_Routines::construct_validate_accessor<SIR_State, 3,
                                                     sycl::access_mode::read>(
            vertex_state, h, sycl::range<3>(N_sims, 1, N_vertices),
            sycl::range<3>(0, t, 0));
    auto v_next =
        Buffer_Routines::construct_validate_accessor<SIR_State, 3,
                                                     sycl::access_mode::write>(
            vertex_state, h, sycl::range<3>(N_sims, 1, N_vertices),
            sycl::range<3>(0, t + 1, 0));
    h.depends_on(dep_event);
    h.copy(v_prev, v_next);
  });

  auto event = q.submit([&](sycl::handler &h) {
    h.depends_on(cpy_event);
    auto rng_acc = rngs.template get_access<sycl::access_mode::read_write>(h);
    auto v_acc = Buffer_Routines::construct_validate_accessor<
        SIR_State, 3, sycl::access_mode::read_write>(
        vertex_state, h, sycl::range<3>(N_sims, 1, N_vertices),
        sycl::range<3>(0, t + 1, 0));
    h.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
      auto sim_id = it.get_global_id();

      Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R);
      for (int v_idx = 0; v_idx < N_vertices; v_idx++) {
        auto state_prev = v_acc[sim_id][0][v_idx];
        if (state_prev == SIR_INDIVIDUAL_I) {
          if (bernoulli_R(rng_acc[sim_id])) {
            v_acc[sim_id][0][v_idx] = SIR_INDIVIDUAL_R;
          }
        }
      }
    });
  });
  return event;
}

sycl::event initialize_vertices(sycl::queue &q,
                                const SBM_Database::Sim_Param &p,
                                sycl::buffer<SIR_State, 3> &vertex_state,
                                sycl::buffer<Static_RNG::default_rng> &rngs,
                                const sycl::nd_range<1>&nd_range,
                                std::vector<sycl::event>& dep_event) {
  uint32_t N_vertices = vertex_state.get_range()[2];
  float p_I0 = p.p_I0;
  float p_R0 = p.p_R0;
  auto init_event = q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto v_acc =
        Buffer_Routines::construct_validate_accessor<SIR_State, 3,
                                                     sycl::access::mode::write>(
            vertex_state, h, sycl::range<3>(p.N_sims, 1, N_vertices),
            sycl::range<3>(0, 0, 0));
    auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
      auto sim_id = it.get_global_id();
      for (int vertex_idx = 0; vertex_idx < N_vertices; vertex_idx++) {
        auto &rng = rng_acc[sim_id];
        Static_RNG::bernoulli_distribution<float> bernoulli_I(p_I0);
        Static_RNG::bernoulli_distribution<float> bernoulli_R(p_R0);
        if (bernoulli_I(rng)) {
          v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_I;
        } else if (bernoulli_R(rng)) {
          v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_R;
        } else {
          v_acc[sim_id][0][vertex_idx] = SIR_INDIVIDUAL_S;
        }
      }
    });
  });

  return init_event;
}

sycl::event infect(sycl::queue &q, const SBM_Database::Sim_Param &p,
                   Sim_Buffers &b, uint32_t t, const sycl::nd_range<1>&nd_range,
                   sycl::event &dep_event) {

  uint32_t N_edges = b.ecm.size();
  uint32_t N_vertices = b.vertex_state.get_range()[2];
  uint32_t N_sims = p.N_sims;
  // auto N_threads_per_sim = floor_div(compute_range[0], N_sims);
  // auto N_edges_per_thread = ceil_div(N_edges, N_threads_per_sim);


  auto inf_event = q.submit([&](sycl::handler &h) {
    h.depends_on(dep_event);
    auto ecm_acc = b.ecm.template get_access<sycl::access::mode::read>(h);
    auto p_I_acc =
        Buffer_Routines::construct_validate_accessor<float, 3,
                                                     sycl::access::mode::read>(
            b.p_Is, h, sycl::range<3>(N_sims, 1, p.N_connections),
            sycl::range<3>(0, t, 0));
    auto rng_acc =
        b.rngs.template get_access<sycl::access::mode::read_write>(h);
    auto v_next =
        Buffer_Routines::construct_validate_accessor<SIR_State, 3,
                                                     sycl::access_mode::write>(
            b.vertex_state, h, sycl::range<3>(N_sims, 1, N_vertices),
            sycl::range<3>(0, t + 1, 0));
    auto e_acc = b.edges.template get_access<sycl::access::mode::read>(h);
    auto event_acc = Buffer_Routines::construct_validate_accessor<
        SBM_Graph::Edge_t, 3, sycl::access::mode::read_write>(
        b.accumulated_events, h, sycl::range<3>(N_sims, 1, p.N_connections),
        sycl::range<3>(0, t, 0));
    const auto N_edges = b.ecm.size();
    h.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id();
      auto lid = it.get_local_id();
      auto &rng = rng_acc[gid]; // was lid
      //  auto sim_id = floor_div(gid, N_threads_per_sim);
      Static_RNG::bernoulli_distribution<float> bernoulli_I(0.f);
      for (uint32_t edge_idx = 0;

           edge_idx < N_edges; edge_idx++) {
        auto connection_id = ecm_acc[edge_idx];
        float p_I = p_I_acc[gid][0][connection_id];
        bernoulli_I.p = p_I;
        auto v_from_id = e_acc[edge_idx].from;
        auto v_to_id = e_acc[edge_idx].to;
        const auto v_prev_from = v_next[gid][0][v_from_id];
        const auto v_prev_to = v_next[gid][0][v_to_id];

        //Sample first direction
        if ((v_prev_from == SIR_INDIVIDUAL_I) &&
            (v_prev_to == SIR_INDIVIDUAL_S)) {
          if (bernoulli_I(rng)) {
            v_next[gid][0][v_from_id] = SIR_INDIVIDUAL_I;
            event_acc[gid][0][connection_id].to += 1;
          }
        }
        // Sample second direction
        if ((v_prev_from == SIR_INDIVIDUAL_S) &&
            (v_prev_to == SIR_INDIVIDUAL_I)) {
          if (bernoulli_I(rng)) {
            v_next[gid][0][v_from_id] = SIR_INDIVIDUAL_I;
            event_acc[gid][0][connection_id].from += 1;
          }
        }
      }
    });
  });

  return inf_event;
}
} // namespace SBM_Simulation
