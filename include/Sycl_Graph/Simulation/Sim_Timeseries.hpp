#ifndef SIM_TIMESERIES_HPP
#define SIM_TIMESERIES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/SIR_Types.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <cmath>
template <typename T>
Graphseries_t<T> read_graphseries(sycl::queue& q, sycl::buffer<T, 3>& buf, const Sim_Param& p, uint32_t Nt, uint32_t N_columns, auto& dep_events)
{
    auto floor_div = [](auto a, auto b){return static_cast<uint32_t>(std::floor(static_cast<double>(a) / static_cast<double>(b)));};

    auto N_sims_tot = p.N_graphs*p.N_sims;
    std::vector<T> data(buf.size());
    auto event = read_buffer<T, 3>(buf, q, data, dep_events);
    event.wait();
    Graphseries_t<T> graphseries(p.N_graphs, p.N_sims, Nt, N_columns);
    //data has linear position defined by: t*N_sims_tot*N_columns + sim_idx*N_columns + c
    for(int sim_tot_idx = 0; sim_tot_idx < N_sims_tot; sim_tot_idx++)
    {
        auto graph_idx = floor_div(sim_tot_idx, p.N_sims);
        auto sim_idx = sim_tot_idx % p.N_sims;
        for(int t = 0; t < Nt; t++)
        {
            for(int c = 0; c < N_columns; c++)
            {
                graphseries[graph_idx][sim_idx][t][c] = data[t*N_sims_tot*N_columns + graph_idx*p.N_sims*N_columns + sim_idx*N_columns + c];
            }
        }
    }
    return graphseries;
}


template <typename T>
Graphseries_t<T> get_N_timesteps(const Graphseries_t<T>&& gs, size_t N, size_t offset)
{

    auto N_graphs = gs.Ng;
    auto N_sims = gs.N_sims;
    auto Nt = gs.Nt;
    auto N_columns = gs.N_cols;
    Graphseries_t<T> new_gs(N_graphs, N_sims, N, N_columns);
    assert(offset + N <= Nt && "offset + N must be less than Nt");
    for(int g = 0; g < N_graphs; g++)
    {
        for(int s = 0; s < N_sims; s++)
        {
            for(int t = offset; t < N + offset; t++)
            {
                for(int c = 0; c < N_columns; c++)
                {
                    new_gs[g][s][t-offset][c] = gs[g][s][t][c];
                }
            }
        }
    }
    return new_gs;
}
#endif
