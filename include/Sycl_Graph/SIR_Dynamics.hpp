#ifndef SIR_DYNAMICS_HPP
#define SIR_DYNAMICS_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <tuple>
sycl::event initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                sycl::buffer<SIR_State, 2> &buf, sycl::buffer<uint32_t> &seed_buf, std::vector<sycl::event> event);

sycl::event recover(sycl::queue &q, uint32_t t, sycl::event &dep_event, float p_R, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<SIR_State, 2> &trajectory);

sycl::event infect(sycl::queue &q, const std::shared_ptr<sycl::buffer<uint32_t>> &ecm_buf, sycl::buffer<float, 2> &p_I_buf, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<uint32_t, 2> &event_from_buf, sycl::buffer<uint32_t, 2> &event_to_buf, sycl::buffer<SIR_State, 2> &trajectory, const std::shared_ptr<sycl::buffer<uint32_t, 1>> &edge_from_buf, const std::shared_ptr<sycl::buffer<uint32_t, 1>> &edge_to_buf, sycl::buffer<uint32_t>& infection_indices_buf, uint32_t t, uint32_t N_connections, sycl::event dep_event);

std::vector<std::vector<float>> generate_p_Is(uint32_t N_community_connections,
                                              float p_I_min, float p_I_max,
                                              uint32_t Nt, uint32_t seed);

#endif
