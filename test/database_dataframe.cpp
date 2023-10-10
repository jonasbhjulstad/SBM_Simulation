#include <Sycl_Graph/Database/Dataframe.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Types.hpp>

template <typename T>
auto generate_timeseries(uint32_t Nt, uint32_t N_cols);

template <>
auto generate_timeseries<State_t>(uint32_t Nt, uint32_t N_cols)
{
    Dataframe_t<State_t, 2> timeseries(std::array{Nt, N_cols});
    for (int t = 0; t < Nt; t++)
    {
        for (int i = 0; i < N_cols; i++)
        {
            timeseries(t, i) = State_t{0,1,2};
        }
    }
    return timeseries;
}
template <>
auto generate_timeseries<float>(uint32_t Nt, uint32_t N_cols)
{
    Dataframe_t<float, 2> timeseries(std::array{Nt, N_cols});
    for (int t = 0; t < Nt; t++)
    {
        for (int i = 0; i < N_cols; i++)
        {
            timeseries(t, i) = 0.0f;
        }
    }
    return timeseries;
}

template <typename T>
auto generate_simseries(uint32_t N_sims, uint32_t Nt, uint32_t N_cols)
{
    Dataframe_t<T, 3> simseries(N_sims);
    for (int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        simseries[sim_idx] = generate_timeseries<T>(Nt, N_cols);
    }
    return simseries;
}
template <typename T>
auto generate_graphseries(uint32_t N_graphs, uint32_t N_sims, uint32_t Nt, uint32_t N_cols)
{
    Dataframe_t<T, 4> graphseries(N_graphs);

    for (int graph_idx = 0; graph_idx < N_graphs; graph_idx++)
    {
        graphseries[graph_idx] = generate_simseries<T>(N_sims, Nt, N_cols);
    }
    return graphseries;
}

int main()
{
    pqxx::connection con("dbname=postgres user=postgres password=postgres");

    uint32_t Np = 10;
    uint32_t Ng = 10;
    uint32_t Ns = 10;
    uint32_t Nt = 10;
    drop_table(con, "community_state");

    create_timeseries_table(con, Np, Ng, Ns, Nt, "community_state");

    auto N_cols = 3;

    Dataframe_t<State_t, 2> timeseries(std::array{10, N_cols});

    write_timeseries(con, 0, 0, 0, timeseries, "community_state");

    auto simseries = generate_simseries<State_t>(Ns, Nt, N_cols);
    write_simseries(con, 0, 0, simseries, "community_state");

    auto graphseries = generate_graphseries<State_t>(Ng, Ns, Nt, N_cols);
    write_graphseries(con, 0, graphseries, "community_state");

    drop_table(con, "community_state");
    return 0;

}
