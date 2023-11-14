
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <SBM_Database/Simulation/Simulation_Tables.hpp>
#include <Sycl_Buffer_Routines/Database.hpp>
#include <Sycl_Buffer_Routines/ND_range.hpp>
#include <Sycl_Buffer_Routines/Random.hpp>
#include <SBM_Database/Graph/Sycl/Graph_Tables.hpp>
#include <SBM_Simulation/Utils/P_I_Generation.hpp>
namespace SBM_Simulation {

Sim_Buffers::Sim_Buffers(sycl::queue &q,
                                     const SBM_Database::Sim_Param &p,
                                     const QString &control_type): 
                                     rngs{Buffer_Routines::generate_rngs(q, p.N_sims, p.seed)},
                                     vertex_state{sycl::buffer<SIR_State, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc+1, p.N_pop*p.N_communities))},
                                      accumulated_events{sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc, p.N_connections))},
                                      edges{SBM_Database::Sycl::read_edgelist(q, p.p_out_id, p.graph_id)},
                                      ecm{SBM_Database::Sycl::read_ecm(q, p.p_out_id, p.graph_id)},
                                      vcm{SBM_Database::Sycl::read_vcm(q, p.p_out_id, p.graph_id)},
                                      community_state{sycl::buffer<State_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc+1,p.N_communities))},
                                      p_Is{generate_p_Is_excitation(q, p, control_type)}
{
}

Sim_Buffers::Sim_Buffers(sycl::queue &q,
                                     const SBM_Database::Sim_Param &p,
                                     const QString &control_type,
                                     const QString &regression_type): 
                                     rngs{Buffer_Routines::generate_rngs(q, p.N_sims, p.seed)},
                                     vertex_state{sycl::buffer<SIR_State, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc+1, p.N_pop*p.N_communities))},                                      accumulated_events{sycl::buffer<uint32_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc, p.N_connections))},
                                      edges{SBM_Database::Sycl::read_edgelist(q, p.p_out_id, p.graph_id)},
                                      ecm{SBM_Database::Sycl::read_ecm(q, p.p_out_id, p.graph_id)},
                                      vcm{SBM_Database::Sycl::read_vcm(q, p.p_out_id, p.graph_id)},
                                      community_state{sycl::buffer<State_t, 3>(sycl::range<3>(p.N_sims, p.Nt_alloc+1,p.N_communities))},
                                      p_Is{generate_p_Is_validation(q, p, control_type, regression_type)}
{
  
}


} // namespace SBM_Simulation
