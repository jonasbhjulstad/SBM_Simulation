#ifndef SIR_DYNAMICS_HPP
#define SIR_DYNAMICS_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <tuple>
std::tuple<uint32_t, uint32_t> get_susceptible_id_if_infected(const sycl::accessor<SIR_State, 1> &v_acc, uint32_t id_from,
                                                              uint32_t id_to);
sycl::event initialize_vertices(float p_I0, float p_R0, sycl::queue &q,
                                sycl::buffer<SIR_State, 2> &buf, sycl::buffer<uint32_t> &seed_buf, sycl::event event);

sycl::event recover(sycl::queue &q, uint32_t t, sycl::event &dep_event, float p_R, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<uint32_t, 2> &trajectory, sycl::buffer<uint32_t> &vcm_buf)

sycl::event infect(sycl::queue &q, sycl::buffer<uint32_t> &ecm_buf, sycl::buffer<float, 2> &p_I_buf, sycl::buffer<uint32_t> &seed_buf, sycl::buffer<uint32_t, 2> &event_from_buf, sycl::buffer<uint32_t, 2> &event_to_buf, sycl::buffer<SIR_State, 2> &trajectory, sycl::buffer<uint32_t, 1> &edge_from_buf, sycl::buffer<uint32_t, 1> &edge_to_buf, uint32_t N_wg, uint32_t t, uint32_t N_connections, sycl::event dep_event);

#endif
