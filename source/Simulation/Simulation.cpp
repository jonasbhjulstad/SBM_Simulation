#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <SBM_Simulation/Simulation/Simulation.hpp>
#include <SBM_Simulation/Simulation/State_Accumulation.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <future>
#include <tom/tom_config.hpp>
namespace SBM_Simulation
{

  void write_community_state(sycl::queue &q, std::vector<Sim_Buffers> &bs,
                              const std::vector<SBM_Database::Sim_Param> &ps,
                              std::vector<sycl::event> &dep_events,
                              std::vector<sycl::nd_range<1>> nd_ranges,
                              const QString &control_type,
                              const QString &simulation_type, uint32_t t_start, uint32_t t_end)
  {
    std::vector<sycl::event> events(ps.size());
    for (int i = 0; i < ps.size(); i++)
    {
      events[i] = accumulate_community_state(q, dep_events[i], bs[i].vertex_state,
                                             bs[i].vpm, bs[i].community_state,
                                             nd_ranges[i]);
    }


    std::vector<Dataframe::Dataframe_t<State_t, 3>> state_dfs(ps.size());
    std::transform(bs.begin(), bs.end(), events.begin(), state_dfs.begin(), [&](Sim_Buffers &b, sycl::event& e)
                   { return Dataframe::make_dataframe<State_t>(q, b.community_state, e); });
    std::for_each(ps.begin(), ps.end(),
                  [&, i = 0](const SBM_Database::Sim_Param &p) mutable
                  {
                    SBM_Database::community_state_to_table(
                        p.p_out_id, p.graph_id, state_dfs[i], control_type,
                        simulation_type, t_start, t_end);
                    i++;
                  });
  }

  void write_connection_events(sycl::queue &q, std::vector<Sim_Buffers> &bs,
                               const std::vector<SBM_Database::Sim_Param> &ps,
                               std::vector<sycl::event> &dep_events,
                               std::vector<sycl::nd_range<1>> nd_ranges,
                               const QString &control_type,
                               const QString &simulation_type, uint32_t t_start)
  {
    std::vector<Dataframe::Dataframe_t<SBM_Graph::Edge_t, 3>> event_dfs(ps.size());

    std::transform(bs.begin(), bs.end(), event_dfs.begin(), [&](Sim_Buffers &b)
                   { return Dataframe::make_dataframe<SBM_Graph::Edge_t>(q, b.accumulated_events); });
    std::for_each(ps.begin(), ps.end(),
                  [&, i = 0](const SBM_Database::Sim_Param &p) mutable
                  {
                    SBM_Database::edge_to_table(
                        "connection_events", p.p_out_id, p.graph_id, event_dfs[i],
                        control_type, simulation_type, t_start, t_start + p.Nt_alloc);
                    i++;
                  });
  }


  std::vector<sycl::event> initialize_simulations(
      sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
      std::vector<Sim_Buffers> &bs, const QString &control_type,
      const QString &simulation_type, std::vector<sycl::nd_range<1>> nd_ranges)
  {
    std::vector<sycl::event> events(ps.size());
    for (int i = 0; i < ps.size(); i++)
    {
      events[i] = initialize_vertices(q, ps[i], bs[i].vertex_state, bs[i].rngs,
                                      nd_ranges[i], bs[i].construction_events);
    }
    return events;
  }

  sycl::event p_I_table_to_buffer(sycl::queue &q,
                                  const SBM_Database::Sim_Param &p,
                                  sycl::buffer<float, 3> &p_Is,
                                  const QString &control_type,
                                  const QString &simulation_type,
                                  uint32_t t_start,
                                  sycl::event dep_event)
  {
    auto table_name = QString("p_Is_") +
                      ((simulation_type.isEmpty()) ? "excitation" : "validation");
    QVector<Orm::WhereItem> const_indices = {{"p_out", p.p_out_id},
                                             {"graph", p.graph_id},
                                             {"Control_Type", control_type}};
    QStringList var_indices = {"simulation", "t", "connection"};
    std::array<uint32_t, 3> start = {0, t_start, 0};
    auto t_end = t_start + (p.Nt_alloc-1);
    std::array<uint32_t, 3> end = {p.N_sims, t_end,
                                   p.N_connections};

    // auto query = Orm::DB::table("p_Is")->select()
    auto N = p.N_sims * (p.Nt_alloc-1) * p.N_connections;
    auto query = Orm::DB::unprepared("select value from \"p_Is\" where (p_out, graph, \"Control_Type\") = (" + QString::number(p.p_out_id) + ", " + QString::number(p.graph_id) + ", '" + control_type + "') and t between " + QString::number(t_start) + " and " + QString::number(t_end - 1) + " order by simulation, t, connection asc");
    std::vector<float> data(N);
    auto idx = 0;
    while (query.next())
    {

      data[idx] = query.value(0).toFloat();
      idx++;
    }
    assert(idx == N);
    return q.submit([&](sycl::handler &h)
                    {
    auto acc = p_Is.template get_access<sycl::access_mode::write>(h);
    h.copy(data.data(), acc); });
  }

  std::vector<sycl::event> simulate_allocated_steps(
      sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
      std::vector<Sim_Buffers> &bs, std::vector<sycl::event> &events,
      std::vector<sycl::nd_range<1>> nd_ranges, const QString &control_type,
      const QString &simulation_type, uint32_t t_start, bool is_last)
  {
    for (int i = 0; i < ps.size(); i++)
    {
      events[i] = p_I_table_to_buffer(q, ps[i], bs[i].p_Is, control_type,
                                      simulation_type, t_start, events[i]);
      for (int t = 0; t < ps[0].Nt_alloc-1; t++)
      {
        events[i] = recover(q, ps[i], bs[i].vertex_state, bs[i].rngs, t,
                            nd_ranges[i], events[i]);
        events[i] = infect(q, ps[i], bs[i], t, nd_ranges[i], events[i]);
      }
    }
    auto t_end = t_start + ps[0].Nt_alloc;
    t_end = (is_last) ? t_end : t_end - 1;
    write_community_state(q, bs, ps, events, nd_ranges, control_type,
                          simulation_type, t_start, t_end);
    write_connection_events(q, bs, ps, events, nd_ranges, control_type,
                            simulation_type, t_start);
    std::vector<sycl::event> new_events(ps.size());
    std::transform(bs.begin(), bs.end(), new_events.begin(), [&](Sim_Buffers &b)
                   { return shift_buffer(q, b.vertex_state); });
    return new_events;
  }

  void run_simulations(sycl::queue &q,
                       const std::vector<SBM_Database::Sim_Param> &ps,
                       std::vector<Sim_Buffers> &bs, const QString &control_type,
                       const QString &simulation_type, bool verbose)
  {
    std::vector<sycl::nd_range<1>> nd_ranges;
    std::transform(ps.begin(), ps.end(), std::back_inserter(nd_ranges),
                   [&](const SBM_Database::Sim_Param &p)
                   {
                     return sycl::nd_range<1>(sycl::range<1>(p.N_sims),
                                              sycl::range<1>(p.N_sims));
                   });
    auto events = initialize_simulations(q, ps, bs, control_type, simulation_type,
                                         nd_ranges);
    uint32_t N_bulks = std::ceil((double)ps[0].Nt / (ps[0].Nt_alloc - 1));
    std::cout << "Running concurrent MC-simulations with (p_out, N_graphs, "
                 "N_sims, Nt, Nt_alloc) = ("
              << ps[0].p_out_id << ", " << ps.size() << ", " << ps[0].N_sims
              << ", " << ps[0].Nt << ", " << ps[0].Nt_alloc << ") \n over "
              << N_bulks << " bulks" << std::endl;
    uint32_t t_start = 0;
    uint32_t N_steps = ps[0].Nt_alloc - 1;
    for (int i = 0; i < N_bulks; i++)
    {
      std::cout << i + 1 << " of " << N_bulks << std::endl;
      events = simulate_allocated_steps(q, ps, bs, events, nd_ranges,
                                        control_type, simulation_type,
                                        t_start, i == N_bulks - 1);
      t_start += N_steps;
    }
    sycl::event::wait_and_throw(events);
  }

} // namespace SBM_Simulation
