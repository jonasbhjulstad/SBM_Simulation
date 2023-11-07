#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <Dataframe/Dataframe.hpp>
#include <chrono>
namespace SBM_Simulation {
Simulation_t::Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
                           const std::string &control_type,
                           const std::string &simulation_type)
    : q(q), p(sim_param), b(q, sim_param, control_type, simulation_type),
      compute_range(Buffer_Routines::get_compute_range(q, p.N_sims)),
      wg_range(std::min<uint32_t>({(uint32_t)Buffer_Routines::get_wg_range(q)[0], sim_param.N_sims})) {}

Simulation_t::Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
                           const Sim_Buffers &sim_buffers)
    : q(q), p(sim_param), b(sim_buffers),
      compute_range(Buffer_Routines::get_compute_range(q, p.N_sims)),
      wg_range(Buffer_Routines::get_wg_range(q)) {}

void Simulation_t::write_allocated_steps(uint32_t t,
                                         std::vector<sycl::event> &dep_events,
                                         uint32_t N_max_steps) {
  auto N_steps = t % p.Nt_alloc;
  N_steps = (N_steps == 0) ? p.Nt_alloc : N_steps;
  N_steps =
      (N_max_steps) ? std::min<uint32_t>({N_steps, N_max_steps}) : N_steps;
  std::chrono::high_resolution_clock::time_point t1, t2;
  N_steps = std::min<size_t>({N_steps, p.Nt_alloc});
  t1 = std::chrono::high_resolution_clock::now();
  auto acc_event = accumulate_community_state(
      q, dep_events, b.vertex_state, b.vcm, b.community_state, compute_range,
      wg_range, p.N_sims);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Accumulate community state: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms\n";
  t1 = t2;

  auto state_df = Dataframe::Dataframe_t<State_t, 3>(q, *b.community_state);
  auto event_df = Dataframe::Dataframe_t<uint32_t, 3>(q, *b.accumulated_events);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Read graphseries: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms\n";
  t1 = t2;

  auto t_offset = t - p.Nt_alloc;
  SBM_Database::community_state_upsert(p.p_out, p.graph_id, state_df, t_offset);
  SBM_Database::connection_upsert<uint32_t>("connection_events", p.p_out_id, p.graph_id,
                              event_df, t_offset);

  state_df.resize_dim(2, N_steps + 1);
  event_df.resize_dim(2, N_steps);
  auto inf_gs = sample_infections(state_df, event_df, b.ccm, p.seed);

  SBM_Database::connection_upsert<uint32_t>("infection_events", p.p_out_id, p.graph_id,
                              inf_gs, t_offset);

  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Inf sample/ write graphseries: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms\n";
}

void Simulation_t::write_initial_steps(sycl::queue &q, const SBM_Database::Sim_Param &p,
                                       Sim_Buffers &b,
                                       std::vector<sycl::event> &dep_events) {

  auto acc_event = accumulate_community_state(
      q, dep_events, b.vertex_state, b.vcm, b.community_state, compute_range,
      wg_range, p.N_sims);
  auto state_df = Dataframe::Dataframe_t<State_t, 3>(q, *b.community_state);
  state_df.resize_dim(2, 1);
  SBM_Database::community_state_upsert(p.p_out, p.graph_id, state_df);
}

void Simulation_t::run() {
  // if ((!compute_range[0]) || (!wg_range[0])) {
  //   std::tie(compute_range, wg_range) = Buffer_Routines::default_compute_range(q);
  // }
  std::vector<sycl::event> events(1);
  q.wait();

  events[0] = initialize_vertices(q, p, b.vertex_state, b.rngs, compute_range, wg_range, b.construction_events);
  write_initial_steps(q, p, b, events);
  uint32_t t = 0;
  for (t = 0; t < p.Nt; t++) {
    bool is_initial_write = (t == 0);
    if (is_allocated_space_full(t, p.Nt_alloc)) {
      q.wait();
      write_allocated_steps(t, events);
      events[0] = Buffer_Routines::clear_buffer<uint32_t, 3>(
          q, *b.accumulated_events, events);
      events[0] = move_buffer_row(q, b.vertex_state, p.Nt_alloc, events);
    }
    events = recover(q, p, *b.vertex_state, *b.rngs, t, compute_range, wg_range,
                     events);
    events = infect(q, p, b, t, compute_range, wg_range, events);
    std::cout << "t: " << t << "\n";
  }
  write_allocated_steps(t, events);
}

} // namespace SBM_Simulation
