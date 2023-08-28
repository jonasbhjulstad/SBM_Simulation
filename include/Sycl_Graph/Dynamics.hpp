#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>
std::vector<sycl::event> recover(sycl::queue &q,
                                 const Sim_Param &p,
                                 Sim_Buffers& b,
                                 uint32_t t,
                                 std::vector<sycl::event> &dep_event);
sycl::event initialize_vertices(sycl::queue &q, const Sim_Param &p,
                                Sim_Buffers& b);

std::vector<sycl::event> infect(sycl::queue &q,
                                const Sim_Param &p,
                                Sim_Buffers &b,
                                uint32_t t,
                                std::vector<sycl::event> &dep_event);
#endif
