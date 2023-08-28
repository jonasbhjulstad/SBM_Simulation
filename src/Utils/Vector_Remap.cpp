#include <Sycl_Graph/Utils/Vector_Remap.hpp>
template Graphseries_t<uint32_t> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<uint32_t>& linear_data);
template Graphseries_t<SIR_State> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<SIR_State>& linear_data);
template Graphseries_t<State_t> remap_linear_data(uint32_t Ng, uint32_t N_sims, uint32_t Nt, uint32_t N_columns, const std::vector<State_t>& linear_data);

std::size_t get_linear_offset(uint32_t N_sims, uint32_t Nt, uint32_t N_columns, uint32_t sim_id, uint32_t t, uint32_t c_id)
{
    auto sim_frac = std::floor(static_cast<double>(sim_id) / static_cast<double>(N_sims));
    auto graph_offset = sim_frac*N_sims*Nt*N_columns;
    auto sim_offset = sim_id*Nt*N_columns;
    auto t_offset = t*N_columns;
    return graph_offset + sim_offset + t_offset + c_id;
}



template Graphseries_t<uint32_t> get_N_timesteps(const Graphseries_t<uint32_t>& ts, uint32_t Nt);
template Graphseries_t<State_t> get_N_timesteps(const Graphseries_t<State_t>& ts, uint32_t Nt);

Graphseries_t<uint32_t> zip_merge_timeseries(const Graphseries_t<uint32_t>& ts_0, const Graphseries_t<uint32_t>& ts_1)
{
    auto N_graphs = ts_0.size();
    auto N_sims = ts_0[0].size();
    auto Nt = ts_0[0][0].size();
    auto N_columns = ts_0[0][0][0].size();
    Graphseries_t<uint32_t> result(N_graphs, N_sims, Nt, N_columns*2);
    for (uint32_t g = 0; g < N_graphs; ++g)
    {
        for (uint32_t s = 0; s < N_sims; ++s)
        {
            for (uint32_t t = 0; t < Nt; ++t)
            {
                for (uint32_t c = 0; c < N_columns; ++c)
                {
                    result[g][s][t][c*2] = ts_0[g][s][t][c];
                    result[g][s][t][c*2+1] = ts_1[g][s][t][c];
                }
            }
        }
    }
    return result;
}
