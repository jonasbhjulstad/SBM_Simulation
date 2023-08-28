#include <Sycl_Graph/Simulation/Sim_Timeseries.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffer_Routines.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Simulation/Sim_Timeseries_impl.hpp>

template Graphseries_t<uint32_t> read_timeseries(sycl::queue &q,
                                uint32_t N_graphs,
                                uint32_t N_sims,
                                uint32_t Nt,
                                uint32_t N_columns,
                                uint32_t *p_usm,
                                std::vector<sycl::event> &dep_events);
template Graphseries_t<State_t> read_timeseries(sycl::queue &q,
                                uint32_t N_graphs,
                                uint32_t N_sims,
                                uint32_t Nt,
                                uint32_t N_columns,
                                State_t *p_usm,
                                std::vector<sycl::event> &dep_events);

Graphseries_t<State_t> read_community_state(
                                sycl::queue &q,
                                const Sim_Param& p,
                                State_t *p_usm,
                                std::vector<sycl::event> &dep_events)
                                {
                                    return read_timeseries(q, p.N_graphs, p.N_sims, p.Nt_alloc+1, p.N_communities, p_usm, dep_events);
                                }

std::tuple<Graphseries_t<uint32_t>, Graphseries_t<uint32_t>> read_connection_events(
                                sycl::queue &q,
                                const Sim_Param& p,
                                Sim_Buffers& b,
                                std::vector<sycl::event> &dep_events)
                                {
                                    auto events_from = read_timeseries(q, p.N_graphs, p.N_sims, p.Nt_alloc+1, b.N_connections, b.events_from, dep_events);
                                    auto events_to = read_timeseries(q, p.N_graphs, p.N_sims, p.Nt_alloc+1, b.N_connections, b.events_to, dep_events);
                                    return std::make_tuple(events_from, events_to);
                                }
