#ifndef SBM_SIMULATION_SIMULATION_SIM_BUFFERS_HPP
#define SBM_SIMULATION_SIMULATION_SIM_BUFFERS_HPP
#include <CL/sycl.hpp>
#include <SBM_Graph/Graph_Types.hpp>
#include <SBM_Database/Graph/Sycl/Graph_Tables.hpp>
#include <SBM_Database/Simulation/SIR_Types.hpp>
#include <SBM_Database/Simulation/Sim_Types.hpp>
#include <Dataframe/Dataframe.hpp>
#include <Static_RNG/distributions.hpp>
namespace SBM_Simulation {
struct Sim_Buffers {

  std::shared_ptr<sycl::buffer<Static_RNG::default_rng>>
      rngs; // =
            // std::make_shared(sycl::buffer<Static_RNG::default_rng>(sycl::range<1>(1)));
  std::shared_ptr<sycl::buffer<SIR_State, 3>>
      vertex_state; // = std::make_shared(sycl::buffer<SIR_State,
                    // 3>(sycl::range<3>(1,1,1)));
  std::shared_ptr<sycl::buffer<uint32_t, 3>>
      accumulated_events; // = std::make_shared(sycl::buffer<uint32_t,
                          // 3>(sycl::range<3>(1,1,1)));
  std::shared_ptr<sycl::buffer<float, 3>>
      p_Is; // = std::make_shared(sycl::buffer<float,
            // 3>(sycl::range<3>(1,1,1)));
  std::shared_ptr<sycl::buffer<std::pair<uint32_t, uint32_t>>>
      edges; //= std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
  std::shared_ptr<sycl::buffer<uint32_t,1>>
      ecm; // = std::make_shared(sycl::buffer<uint32_t>(sycl::range<1>(1)));
  std::shared_ptr<sycl::buffer<uint32_t,1>>
      vcm; // = std::make_shared(sycl::buffer<uint32_t,
           // 2>(sycl::range<2>(1,1)));
  std::shared_ptr<sycl::buffer<State_t, 3>>
      community_state; //= std::make_shared(sycl::buffer<State_t,
                       //3>(sycl::range<3>(1,1,1)));
  std::vector<sycl::event> construction_events;
  const Dataframe::Dataframe_t<Weighted_Edge_t, 1> ccm;
  Sim_Buffers(sycl::queue &q, const SBM_Database::Sim_Param &p,
              const std::string &control_type = "Community",
              const std::string &simulation_type = "Excitation");
};
} // namespace SBM_Simulation
#endif
