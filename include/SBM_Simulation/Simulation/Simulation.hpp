#ifndef SBM_SIMULATION_SIMULATION_HPP
#define SBM_SIMULATION_SIMULATION_HPP
#include <CL/sycl.hpp>
#include <SBM_Graph/Graph.hpp>
#include <SBM_Simulation/Epidemiological/SIR_Dynamics.hpp>
#include <SBM_Simulation/Simulation/Sim_Buffers.hpp>
#include <SBM_Simulation/Simulation/Sim_Infection_Sampling.hpp>

namespace SBM_Simulation {
struct Simulation_t {

  Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param, const std::string &control_type = "Community", const std::string &simulation_type = "Excitation");

  Simulation_t(sycl::queue &q, const SBM_Database::Sim_Param &sim_param,
               const Sim_Buffers &sim_buffers);

  Sim_Buffers b;
  SBM_Database::Sim_Param p;
  sycl::queue &q;
  sycl::range<1> compute_range = sycl::range<1>(1);
  sycl::range<1> wg_range = sycl::range<1>(1);

  void run();

private:
  void write_initial_steps(sycl::queue &q, const SBM_Database::Sim_Param &p, Sim_Buffers &b,
                           std::vector<sycl::event> &dep_events);

  void write_allocated_steps(uint32_t t, std::vector<sycl::event> &dep_events,
                             uint32_t N_max_steps = 0);
};
} // namespace SBM_Simulation
#endif
