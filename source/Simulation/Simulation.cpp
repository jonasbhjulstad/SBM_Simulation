#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <future>
#include <tom/tom_config.hpp>
namespace SBM_Simulation {


void write_simulation_steps(sycl::queue &q, std::vector<Sim_Buffers> &bs,
                            const std::vector<SBM_Database::Sim_Param> &ps,
                            std::vector<sycl::event> &dep_events,
                            std::vector<sycl::nd_range<1>> nd_ranges,
                            const QString &control_type,
                            const QString &simulation_type, uint32_t t_start,
                            uint32_t t_end) {
  std::vector<sycl::event> events(ps.size());
  for (int i = 0; i < ps.size(); i++) {
    events[i] = accumulate_community_state(q, dep_events[i], bs[i].vertex_state,
                                           bs[i].vpm, bs[i].community_state,
                                           nd_ranges[i]);
  }

  std::vector<Dataframe::Dataframe_t<State_t, 3>> state_dfs(ps.size());
  std::transform(bs.begin(), bs.end(), state_dfs.begin(), [&](Sim_Buffers &b) {
    return Dataframe::make_dataframe<State_t>(q, b.community_state);
  });
  std::vector<Dataframe::Dataframe_t<SBM_Graph::Edge_t, 3>> event_dfs(ps.size());

  std::transform(bs.begin(), bs.end(), event_dfs.begin(), [&](Sim_Buffers &b) {
    return Dataframe::make_dataframe<SBM_Graph::Edge_t>(q, b.accumulated_events);
  });

  std::for_each(ps.begin(), ps.end(),
                [&, i = 0](const SBM_Database::Sim_Param &p) mutable {

                  SBM_Database::community_state_to_table(
                      p.p_out_id, p.graph_id, state_dfs[i], control_type,
                      simulation_type, t_start+1, t_end + 1);
                  SBM_Database::edge_to_table(
                      "connection_events", p.p_out_id, p.graph_id, event_dfs[i],
                      control_type, simulation_type, t_start, t_end);
                  i++;
                });
}


void write_initial_steps(sycl::queue &q,
                         const std::vector<SBM_Database::Sim_Param> &ps,
                         std::vector<Sim_Buffers> &b,
                         std::vector<sycl::nd_range<1>> nd_ranges,
                         const QString &control_type,
                         const QString &simulation_type,
                         std::vector<sycl::event> &dep_events) {
  std::vector<sycl::event> acc_events(ps.size());

  for (int i = 0; i < ps.size(); i++) {
    acc_events[i] = accumulate_community_state(
        q, dep_events[i], b[i].vertex_state, b[i].vpm, b[i].community_state,
        nd_ranges[i]);
  }
  sycl::event::wait_and_throw(acc_events);
  std::vector<Dataframe::Dataframe_t<State_t, 3>> state_dfs(ps.size());
  std::transform(b.begin(), b.end(), state_dfs.begin(), [&](Sim_Buffers &b) {
    return Dataframe::make_dataframe<State_t>(q, b.community_state);
  });

  std::for_each(ps.begin(), ps.end(),
                [&, i = 0](const SBM_Database::Sim_Param &p) mutable {
                  SBM_Database::community_state_upsert(
                      p.p_out_id, p.graph_id, state_dfs[i], control_type,
                      simulation_type, 0, 1);
                  i++;
                });
}


std::vector<sycl::event> initialize_simulations(
    sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
    std::vector<Sim_Buffers> &bs, const QString &control_type,
    const QString &simulation_type, std::vector<sycl::nd_range<1>> nd_ranges) {
  std::vector<sycl::event> events(ps.size());
  for (int i = 0; i < ps.size(); i++) {
    events[i] = initialize_vertices(q, ps[i], bs[i].vertex_state, bs[i].rngs,
                                    nd_ranges[i], bs[i].construction_events);
  }
  write_initial_steps(q, ps, bs, nd_ranges, control_type, simulation_type,
                      events);
  return events;
}

sycl::event p_I_table_to_buffer(sycl::queue &q,
                                const SBM_Database::Sim_Param &p,
                                sycl::buffer<float, 3> &p_Is,
                                const QString &control_type,
                                const QString &simulation_type,
                                uint32_t t_start, uint32_t t_end,
                                 sycl::event dep_event) {
  if (t_end == t_start)
    {
      return sycl::event{};
    }
  auto table_name = QString("p_Is_") +
                    ((simulation_type.isEmpty()) ? "excitation" : "validation");
  QVector<Orm::WhereItem> const_indices = {{"p_out", p.p_out_id},
                                           {"graph", p.graph_id},
                                           {"Control_Type", control_type}};
  QStringList var_indices = {"simulation", "t", "connection"};
  std::array<uint32_t, 3> start = {0, t_start, 0};
  std::array<uint32_t, 3> end = {p.N_sims, t_end,
                                 p.N_connections};
  return Buffer_Routines::table_to_buffer<float, 3>(
      q, p_Is, table_name, const_indices, var_indices, "value", start, end,
      "simulation", "t", dep_event);
}

std::vector<sycl::event> simulate_allocated_steps(
    sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
    std::vector<Sim_Buffers> &bs, std::vector<sycl::event> &events,
    std::vector<sycl::nd_range<1>> nd_ranges, const QString &control_type,
    const QString &simulation_type, uint32_t t_start, uint32_t N_steps) {
  auto t_end = std::min<uint32_t>({t_start + N_steps, ps[0].Nt});
  for (int i = 0; i < ps.size(); i++) {
    events[i] = p_I_table_to_buffer(q, ps[i], bs[i].p_Is, control_type,
                                    simulation_type, t_start, t_end, events[i]);
    for (int t = t_start; t < t_end; t++) {
      events[i] = recover(q, ps[i], bs[i].vertex_state, bs[i].rngs, t,
                          nd_ranges[i], events[i]);
      events[i] = infect(q, ps[i], bs[i], t, nd_ranges[i], events[i]);
    }
  }
  write_simulation_steps(q, bs, ps, events, nd_ranges, control_type,
                         simulation_type, t_start, t_end);
  std::vector<sycl::event> new_events(ps.size());
  std::transform(bs.begin(), bs.end(), new_events.begin(), [&](Sim_Buffers &b) {
    return shift_buffer(q, b.vertex_state);
  });
  return new_events;
}
void run_simulation(sycl::queue &q, const SBM_Database::Sim_Param &p,
                    Sim_Buffers &b, const QString &control_type,
                    const QString &simulation_type) {
  auto nd_range = Buffer_Routines::get_nd_range(q, p.N_sims);

  auto event =
      initialize_simulation(q, p, b, control_type, simulation_type, nd_range);
  uint32_t N_bulks = std::floor((double)p.Nt_alloc / p.Nt);
  for (int i = 0; i < N_bulks; i++) {
    simulate_allocated_steps(q, p, b, event, nd_range, control_type,
                             simulation_type, i * p.Nt_alloc + 1, p.Nt_alloc);
  }
  simulate_allocated_steps(q, p, b, event, nd_range, control_type,
                           simulation_type, N_bulks * p.Nt_alloc + 1,
                           p.Nt % p.Nt_alloc);
}

void run_simulations(sycl::queue &q,
                     const std::vector<SBM_Database::Sim_Param> &ps,
                     std::vector<Sim_Buffers> &bs, const QString &control_type,
                     const QString &simulation_type, bool verbose) {
  std::vector<sycl::nd_range<1>> nd_ranges;
  std::transform(ps.begin(), ps.end(), std::back_inserter(nd_ranges),
                 [&](const SBM_Database::Sim_Param &p) {
                   return Buffer_Routines::get_nd_range(q, p.N_sims);
                 });
  auto events = initialize_simulations(q, ps, bs, control_type, simulation_type,
                                       nd_ranges);
  uint32_t N_bulks = std::floor((double)ps[0].Nt / (ps[0].Nt_alloc - 1));
  std::cout << "Running concurrent MC-simulations with (p_out, N_graphs, "
               "N_sims, Nt, Nt_alloc) = ("
            << ps[0].p_out_id << ", " << ps.size() << ", " << ps[0].N_sims
            << ", " << ps[0].Nt << ", " << ps[0].Nt_alloc << ") \n over "
            << N_bulks << " bulks" << std::endl;
  for (int i = 0; i < N_bulks; i++) {
    std::cout << i + 1 << " of " << N_bulks << std::endl;
    events = simulate_allocated_steps(q, ps, bs, events, nd_ranges,
                                      control_type, simulation_type,
                                      i * (ps[0].Nt_alloc-1), ps[0].Nt_alloc-1);
  }
  events = simulate_allocated_steps(q, ps, bs, events, nd_ranges, control_type,
                                    simulation_type, N_bulks * (ps[0].Nt_alloc - 1),
                                    ps[0].Nt % (ps[0].Nt_alloc - 1));
  sycl::event::wait_and_throw(events);
}

} // namespace SBM_Simulation
