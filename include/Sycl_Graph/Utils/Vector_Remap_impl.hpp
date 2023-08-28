#ifndef VECTOR_REMAP_IMPL_HPP
#define VECTOR_REMAP_IMPL_HPP
#include <Sycl_Graph/SIR_Types.hpp>
#include <CL/sycl.hpp>
#include <cassert>
#include <cmath>

SYCL_EXTERNAL std::size_t get_linear_offset(uint32_t N_sims, uint32_t Nt, uint32_t N_columns, uint32_t sim_id, uint32_t t, uint32_t c_id);

//4-D Graph, N_sims, Nt, N_columns
template <typename T>
Graphseries_t<T> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<T>& linear_data)
{
    assert(linear_data.size() == Ng*N_sims * Nt * N_columns);
    Graphseries_t<T> result(Ng, N_sims, Nt, N_columns);

    for(uint32_t g = 0; g < Ng; g++)
    {
        for(uint32_t s = 0; s < N_sims; s++)
        {
            for(uint32_t t = 0; t < Nt; t++)
            {
                for(uint32_t c = 0; c < N_columns; c++)
                {
                    result[g][s][t][c] = linear_data[get_linear_offset(N_sims, Nt, N_columns, s, t, c)];
                }
            }
        }
    }
    return result;
}

template <typename T>
Graphseries_t<T> get_N_timesteps(const Graphseries_t<T>& ts, uint32_t Nt)
{
    auto Ng = ts.size();
    auto Ns = ts[0].size();
    auto N_cols = ts[0][0][0].size();
    Graphseries_t<T> result(Ng, Ns, Nt, N_cols);
    for (uint32_t g = 0; g < Ng; ++g)
    {
        for (uint32_t s = 0; s < Ns; ++s)
        {

            for(uint32_t t = 0; t < Nt; t++)
            {
                for(uint32_t c = 0; c < N_cols; c++)
                {
                    result[g][s][t][c] = ts[g][s][t][c];
                }
            }
        }
    }
    return result;
}

Graphseries_t<uint32_t> zip_merge_timeseries(const Graphseries_t<uint32_t>& ts_0, const Graphseries_t<uint32_t>& ts_1);

#endif
