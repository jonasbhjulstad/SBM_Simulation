#ifndef SBM_SIMULATION_SIMULATION_HPP
#define SBM_SIMULATION_SIMULATION_HPP
#include <CL/sycl.hpp>
#include <QString>
#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>
#include <spdlog/async_logger.h>
namespace SBM_Simulation {
struct Simulation_t {

  Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
               const char *control_type, const char *regression_type);
  Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
               const char *control_type);
  Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
               const Sim_Buffers &sim_buffers);

  Sim_Buffers b;
  SBM_Database::Sim_Param p;
  sycl::queue &q;
  sycl::range<1> compute_range = sycl::range<1>(1);
  sycl::range<1> wg_range = sycl::range<1>(1);
  QString control_type;
  QString regression_type;
  std::shared_ptr<spdlog::logger> logger;


  void run();

private:
  void write_initial_steps(sycl::queue &q, const SBM_Database::Sim_Param &p,
                           Sim_Buffers &b, sycl::event &dep_event);

  void write_allocated_steps(uint32_t t, sycl::event &event,
                             uint32_t t_start = 1, uint32_t t_end = 0);
  
};
} // namespace SBM_Simulation
#endif
