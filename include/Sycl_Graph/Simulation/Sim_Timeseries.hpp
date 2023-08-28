#ifndef SIM_TIMESERIES_HPP
#define SIM_TIMESERIES_HPP
#include <Sycl_Graph/Simulation/Sim_Timeseries_impl.hpp>

extern template Graphseries_t<uint32_t> read_timeseries(sycl::queue &q,
                                uint32_t N_graphs,
                                uint32_t N_sims,
                                uint32_t Nt,
                                uint32_t N_columns,
                                uint32_t *p_usm,
                                std::vector<sycl::event> &dep_events);
extern template Graphseries_t<State_t> read_timeseries(sycl::queue &q,
                                uint32_t N_graphs,
                                uint32_t N_sims,
                                uint32_t Nt,
                                uint32_t N_columns,
                                State_t *p_usm,
                                std::vector<sycl::event> &dep_events);

#endif
