#ifndef SBM_SIMULATION_SIMULATION_SIM_BUFFERS_HPP
#define SBM_SIMULATION_SIMULATION_SIM_BUFFERS_HPP
#include <CL/sycl.hpp>
#include <Dataframe/Dataframe.hpp>
#include <SBM_Database/Simulation/SIR_Types.hpp>
#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <SBM_Graph/Graph_Types.hpp>
#include <Static_RNG/distributions.hpp>
namespace SBM_Simulation {
struct Sim_Buffers {


  sycl::buffer<SIR_State, 3> vertex_state;
  sycl::buffer<uint32_t, 3> accumulated_events;
  sycl::buffer<float, 3> p_Is;
  sycl::buffer<Static_RNG::default_rng, 1> rngs;
  sycl::buffer<std::pair<uint32_t, uint32_t>, 1> edges;
  sycl::buffer<uint32_t, 1> ecm;
  sycl::buffer<uint32_t, 1> vcm;
  sycl::buffer<State_t, 3> community_state;
  std::vector<sycl::event> construction_events = std::vector<sycl::event>(4);
  const Dataframe::Dataframe_t<Weighted_Edge_t, 1> ccm;

  Sim_Buffers(sycl::queue &q, const SBM_Database::Sim_Param &p,
              const QString &control_type);

  Sim_Buffers(sycl::queue &q, const SBM_Database::Sim_Param &p,
              const QString &control_type, const QString &regression_type);
};
sycl::buffer<float, 3> generate_upsert_p_Is(sycl::queue &q,
                                            const SBM_Database::Sim_Param &p,
                                            const QString &table_name,
                                            const QString &control_type);
} // namespace SBM_Simulation
#endif
