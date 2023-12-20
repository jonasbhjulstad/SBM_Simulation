
#include <SBM_Database/Graph/Graph_Tables.hpp>
#include <SBM_Database/Graph/Sycl/Graph_Tables.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
namespace SBM_Simulation {

Sim_Buffers::Sim_Buffers(sycl::queue &q, const SBM_Database::Sim_Param &p,
                         const QString &control_type)
    : rngs{Buffer_Routines::generate_rngs(q, p.N_sims, p.seed)},
      vertex_state{sycl::buffer<SIR_State, 3>(
          sycl::range<3>(p.N_sims, p.Nt_alloc, p.N_pop * p.N_communities))},
      accumulated_events{sycl::buffer<SBM_Graph::Edge_t, 3>(
          sycl::range<3>(p.N_sims, p.Nt_alloc-1, p.N_connections))},
      edges{SBM_Database::Sycl::read_edgelist(q, p.p_out_id, p.graph_id)},
      ecm{SBM_Database::Sycl::read_ecm(q, p.p_out_id, p.graph_id)},
      vpm{SBM_Database::Sycl::read_vpm(q, p.p_out_id, p.graph_id)},
      community_state{sycl::buffer<State_t, 3>(
          sycl::range<3>(p.N_sims, p.Nt_alloc, p.N_communities))},
      p_Is{sycl::buffer<float, 3>(
          sycl::range<3>(p.N_sims, p.Nt_alloc-1, p.N_connections))},
      ccm{Dataframe::Dataframe_t<SBM_Graph::Weighted_Edge_t, 1>(
          SBM_Database::ccm_read(p.p_out_id, p.graph_id))} {}

} // namespace SBM_Simulation
