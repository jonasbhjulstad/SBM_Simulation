#ifndef SIM_ROUTINES_HPP
#define SIM_ROUTINES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>

void run(sycl::queue& q, const Sim_Param& p, Sim_Buffers& b);

void run_allocated(sycl::queue &q, const Sim_Param &p, Sim_Buffers &b);

#endif
