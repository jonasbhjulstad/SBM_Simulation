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


std::vector<sycl::event> initialize_simulations(
    sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
    std::vector<Sim_Buffers> &bs, const QString &control_type,
    const QString &regression_type,
    std::vector<sycl::nd_range<1>> nd_ranges);
std::vector<sycl::event> simulate_allocated_steps(
    sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
    std::vector<Sim_Buffers> &bs, std::vector<sycl::event> &events,
    std::vector<sycl::nd_range<1>> nd_ranges, const QString &control_type,
    const QString &regression_type, uint32_t t_start);


void run_simulation(sycl::queue &q, const SBM_Database::Sim_Param &p,
                    Sim_Buffers &b, const sycl::nd_range<1>&nd_range,
                    const QString &control_type,
                    const QString &regression_type = "");
void run_simulations(sycl::queue &q, const std::vector<SBM_Database::Sim_Param> &ps,
                              std::vector<Sim_Buffers> &bs,
                              const QString &control_type,
                              const QString &regression_type = "",
                              bool verbose = true);
} // namespace SBM_Simulation
#endif
