#ifndef SIM_TIMESERIES_HPP
#define SIM_TIMESERIES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
void community_state_to_timeseries(sycl::queue &q,
                                   sycl::buffer<SIR_State, 3> &vertex_state,
                                   sycl::buffer<State_t, 3> &community_state,
                                   std::vector<std::vector<std::vector<State_t>>> &community_timeseries,
                                   sycl::buffer<uint32_t, 2> &vcm,
                                   uint32_t t_offset,
                                   sycl::range<1> compute_range,
                                   sycl::range<1> wg_range,
                                   std::vector<sycl::event> &dep_events);
void connection_events_to_timeseries(sycl::queue &q,
                                     sycl::buffer<uint32_t, 3> events_from,
                                     sycl::buffer<uint32_t, 3> &events_to,
                                     std::vector<std::vector<std::vector<uint32_t>>> &events_from_timeseries,
                                     std::vector<std::vector<std::vector<uint32_t>>> &events_to_timeseries,
                                     sycl::buffer<uint32_t, 2> &vcm,
                                     uint32_t t_offset,
                                     std::vector<sycl::event> &dep_events);
void community_state_append_to_file(sycl::queue &q,
                                    sycl::buffer<SIR_State, 3> &vertex_state,
                                    sycl::buffer<State_t, 3> &community_state,
                                    sycl::buffer<uint32_t, 2> &vcm,
                                    sycl::range<1> compute_range,
                                    sycl::range<1> wg_range,
                                    const std::string &output_dir,
                                    std::vector<sycl::event> &dep_events);
void community_state_init_to_file(sycl::queue &q,
                                  sycl::buffer<SIR_State, 3> &vertex_state,
                                  sycl::buffer<State_t, 3> &community_state,
                                  sycl::buffer<uint32_t, 2> &vcm,
                                  sycl::range<1> compute_range,
                                  sycl::range<1> wg_range,
                                  const std::string &output_dir,
                                  std::vector<sycl::event> &dep_events);
void connection_events_append_to_file(sycl::queue &q,
                                      sycl::buffer<uint32_t, 3> events_from,
                                      sycl::buffer<uint32_t, 3> &events_to,
                                      sycl::buffer<uint32_t, 2> &vcm,
                                      const std::string &output_dir,
                                      std::vector<sycl::event> &dep_events,
                                      bool append = true);

#endif
