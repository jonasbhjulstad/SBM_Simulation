#ifndef SIM_TIMESERIES_IMPL_HPP
#define SIM_TIMESERIES_IMPL_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <Sycl_Graph/Utils/Vector_Remap.hpp>
#include <Sycl_Graph/Simulation/Sim_Buffers.hpp>

template <typename T>
Graphseries_t<T> read_timeseries(sycl::queue &q,
                                uint32_t N_graphs,
                                uint32_t N_sims,
                                uint32_t Nt,
                                uint32_t N_columns,
                                T *p_usm,
                                std::vector<sycl::event> &dep_events)
{
    std::vector<T> result(Nt * N_sims * N_graphs * N_columns);
    auto event = q.submit([&](sycl::handler &h)
                          {
        h.depends_on(dep_events);
        h.memcpy(result.data(), p_usm, sizeof(T)*result.size()); });
    event.wait();
    return remap_linear_data(N_graphs, N_sims, Nt, N_columns, result);
}

Graphseries_t<State_t> read_community_state(
    sycl::queue &q,
    const Sim_Param &p,
    State_t *p_usm,
    std::vector<sycl::event> &dep_events);
std::tuple<Graphseries_t<uint32_t>, Graphseries_t<uint32_t>> read_connection_events(
    sycl::queue &q,
    const Sim_Param &p,
    Sim_Buffers &b,
    std::vector<sycl::event> &dep_events);

#endif
