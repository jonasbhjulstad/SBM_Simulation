
#include <Dataframe/Dataframe.hpp>
#include <SBM_Graph/Community_Mappings.hpp>
#include <SBM_Graph/Graph.hpp>
#include <SBM_Graph/Graph_Types.hpp>
#include <SBM_Simulation/Database/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <orm/db.hpp>

namespace SBM_Simulation {

Sim_Buffers::Sim_Buffers(sycl::queue &q, const Sim_Param &p,
                         const std::string &control_type,
                         const std::string &simulation_type) {
  construction_events.resize(4);
  auto [compute_range, wg_range] = Buffer_Routines::get_nd_range(q, p.N_sims);
  rngs = Buffer_Routines::generate_rngs(q, compute_range[0], p.seed,
                                        construction_events[0]);

  // std::shared_ptr<sycl::buffer<T, N>> make_shared_device_buffer(sycl::queue
  // &q, const std::vector<T> &vec, sycl::range<N> range, sycl::event
  // &res_event)
  vertex_state = Buffer_Routines::make_shared_device_buffer<SIR_State, 3>(
      q, sycl::range<3>(p.N_sims, p.N_pop, p.Nt_alloc));
  accumulated_events = Buffer_Routines::make_shared_device_buffer<uint32_t, 3>(
      q, sycl::range<3>(p.N_sims, p.N_connections, p.Nt_alloc));

  std::string p_I_table_name = "p_Is_" + control_type + "_" + simulation_type;

  QVector<Orm::WhereItem> const_indices = {{"p_out", p.p_out_id},
                                           {"graph", p.graph_id}};

  p_Is = Buffer_Routines::table_to_buffer<float, 3>(
      q, p_I_table_name.c_str(), const_indices,
      {"simulation", "t", "connection"});
  ecm = SBM_Graph::Sycl::read_ecm(q, p.p_out_id, p.graph_id,
                                  construction_events[1]);
  vcm = SBM_Graph::Sycl::read_vcm(q, p.p_out_id, p.graph_id,
                                  construction_events[2]);
  community_state = Buffer_Routines::make_shared_device_buffer<State_t, 3>(
      q, sycl::range<3>(p.N_sims, p.N_communities, p.Nt_alloc));

  //   connection_read(const QString &table_name, uint32_t p_out, uint32_t
  //   graph,
  // uint32_t N_sims, uint32_t Nt, uint32_t N_cols)
}
} // namespace SBM_Simulation
