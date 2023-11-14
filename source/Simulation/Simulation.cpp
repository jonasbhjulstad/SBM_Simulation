#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <Dataframe/Dataframe.hpp>
namespace SBM_Simulation {
Simulation_t::Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
                           const char *control_type,
                           const char *regression_type)
    : q(q), p(sim_param), b(Sim_Buffers(q, sim_param, control_type, regression_type)),
      compute_range(p.N_sims),
      wg_range(std::min<uint32_t>({(uint32_t)Buffer_Routines::get_wg_range(q)[0], sim_param.N_sims})), control_type(control_type), regression_type(regression_type) 
      {
      }
Simulation_t::Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
                           const char *control_type)
    : q(q), p(sim_param), b(Sim_Buffers(q, sim_param, control_type)),
      compute_range(p.N_sims),
      wg_range(std::min<uint32_t>({(uint32_t)Buffer_Routines::get_wg_range(q)[0], sim_param.N_sims})), control_type(control_type)
      {
      }

Simulation_t::Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
                           const Sim_Buffers &sim_buffers)
    : q(q), p(sim_param), b(sim_buffers),
      compute_range(p.N_sims),
      wg_range(Buffer_Routines::get_wg_range(q)) {}

void Simulation_t::write_allocated_steps(uint32_t t,
                                         sycl::event& event) {
  auto N_steps = t % p.Nt_alloc;
  N_steps = (N_steps == 0) ? p.Nt_alloc : N_steps;
  std::chrono::high_resolution_clock::time_point t1, t2;
  N_steps = std::min<size_t>({N_steps, p.Nt_alloc});
  t1 = std::chrono::high_resolution_clock::now();
  event = accumulate_community_state(
      q, event, b.vertex_state, b.vcm, b.community_state, compute_range,
      wg_range, p.N_sims, 1, N_steps);
  event.wait();
  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Accumulate community state: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms\n";
  t1 = t2;

  auto state_df = Dataframe::make_dataframe<State_t>(q, b.community_state);
  auto event_df = Dataframe::make_dataframe<uint32_t>(q, b.accumulated_events);
  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Read graphseries: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms\n";
  t1 = t2; 

  auto t_offset = t - p.Nt_alloc + 1;
  SBM_Database::community_state_upsert(p.p_out, p.graph_id, state_df, t_offset,control_type, regression_type);
  SBM_Database::connection_upsert<uint32_t>("connection_events", p.p_out_id, p.graph_id,
                              event_df, t_offset, control_type, regression_type);

  // state_df.resize_dim(2, N_steps + 1);
  // event_df.resize_dim(2, N_steps);
  auto inf_gs = sample_infections(state_df, event_df, b.ccm, p.seed);

  SBM_Database::connection_upsert<uint32_t>("infection_events", p.p_out_id, p.graph_id,
                              inf_gs, t_offset,control_type, regression_type);

  t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Inf sample/ write graphseries: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms\n";
}

void Simulation_t::write_initial_steps(sycl::queue &q, const SBM_Database::Sim_Param &p,
                                       Sim_Buffers &b,
                                       sycl::event &dep_event) {

  auto acc_event = accumulate_community_state(
      q, dep_event, b.vertex_state, b.vcm, b.community_state, compute_range,
      wg_range, p.N_sims, 0, 1);
  acc_event.wait();
  auto state_df = Dataframe::make_dataframe<State_t>(q, b.community_state);
  state_df.resize_dim(1, 1);
  SBM_Database::community_state_upsert(p.p_out, p.graph_id, state_df, 0, control_type, regression_type);
}

void Simulation_t::run() {
  // if ((!compute_range[0]) || (!wg_range[0])) {
  //   std::tie(compute_range, wg_range) = Buffer_Routines::default_compute_range(q);
  // }
  q.wait();

  sycl::event event = initialize_vertices(q, p, b.vertex_state, b.rngs, compute_range, wg_range, b.construction_events);
  event.wait();
  write_initial_steps(q, p, b, event);
  event = Buffer_Routines::clear_buffer<uint32_t, 3>(
        q, b.accumulated_events, event);
  uint32_t t = 0;
  for (t = 0; t < p.Nt; t++) {
    bool is_initial_write = (t == 0);
    if (is_allocated_space_full(t, p.Nt_alloc)) {
      q.wait();
      write_allocated_steps(t, event);
      q.wait();
      event = Buffer_Routines::clear_buffer<uint32_t, 3>(
          q, b.accumulated_events, event);
      event = move_buffer_row(q, b.vertex_state, p.Nt_alloc, event);
    }
    event = recover(q, p, b.vertex_state, b.rngs, t, compute_range, wg_range,
                     event);
    event = infect(q, p, b, t, compute_range, wg_range, event);
    std::cout << "t: " << t << "\n";
  }
  // write_allocated_steps(t, event);
}

} // namespace SBM_Simulation
