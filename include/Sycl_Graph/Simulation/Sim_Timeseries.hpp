#ifndef SIM_TIMESERIES_HPP
#define SIM_TIMESERIES_HPP
#include <Sycl_Graph/Utils/Common.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Dataframe.hpp>
#include <cmath>

template <typename T>
Dataframe_t<T, 4> read_3D_buffer(sycl::queue& q, sycl::buffer<T, 3>& buf, uint32_t N_graphs, std::vector<sycl::event> dep_events)
{
    auto Nt = buf.get_range()[0];
    auto N_sims_tot = buf.get_range()[1];
    auto N3 = buf.get_range()[2];
    auto N_sims = N_sims_tot/N_graphs;
    auto floor_div = [](auto a, auto b){return static_cast<uint32_t>(std::floor(static_cast<float>(a) / static_cast<float>(b)));};
    std::vector<T> data(buf.size());
    auto event = read_buffer<T, 3>(buf, q, data, dep_events);
    event.wait();

    Dataframe_t<T,4> dataframe(N_graphs, N_sims, Nt, N3);
    //data has linear position defined by: t*N_sims_tot*N3 + sim_idx*N3 + c
    for(int sim_tot_idx = 0; sim_tot_idx < N_sims_tot; sim_tot_idx++)
    {
        auto graph_idx = floor_div(sim_tot_idx, N_sims);
        auto sim_idx = sim_tot_idx % N_sims;
        for(int t = 0; t < Nt; t++)
        {
            for(int c = 0; c < N3; c++)
            {
                dataframe[graph_idx][sim_idx][t][c] = data[t*N_sims_tot*N3 + graph_idx*N_sims*N3 + sim_idx*N3 + c];
            }
        }
    }
    return dataframe;
}

template <typename T>
void write_dataframe(const std::string& fname, const Dataframe_t<T, 1>& df, bool append = false, const size_t offset = 0)
{
    if_false_throw(df.size(), "Empty df");
    std::fstream f;

    if(append)
    {
        f.open(fname, std::ios::app);
    }
    else
    {
        f.open(fname, std::ios::out);
    }
    for(int row = offset; row < df.size(); row++)
    {
        f << df[row] << "\n";
    }
    f.close();
}

template <typename T>
void write_dataframe(const std::string& dir, const std::string& fname, const Dataframe_t<T, 2>& df, bool append = false, const std::array<std::size_t,2> offsets = {0,0})
{
    for(int i = 0; i < df.size(); i++)
    {
        write_dataframe(dir + std::to_string(i) + "/" + fname, df[i], append, offsets[0]);
    }
}

template <typename T>
void write_dataframe(const std::string& fname, const Dataframe_t<T, 2>& df, bool append = false, const std::array<std::size_t,2> offsets = {0,0})
{
    std::fstream f;
    if(append)
    {
        f.open(fname, std::ios::app);
    }
    else
    {
        f.open(fname, std::ios::out);
    }
    for(int row = offsets[0]; row < df.size(); row++)
    {
        for(int col = offsets[1]; col < df[row].size(); col++)
        {
            f << df(row, col);
            if(col != df[row].size() - 1)
            {
                f << ",";
            }
        }
        f << "\n";
    }
    f.close();
}

template <typename T>
void write_dataframe(const std::string& basename, const Dataframe_t<T, 3>& df, bool append = false, std::array<std::size_t, 2> offsets = {0,0})
{
    for(int i = 0; i < df.size(); i++)
    {
        write_dataframe(basename + std::to_string(i) + ".csv", df[i], append, offsets);
    }
}

template <typename T>
void write_dataframe(const std::string& base_dir, const std::string& basename, const Dataframe_t<T, 4>& df, bool append = false, std::array<std::size_t, 2> offsets = {0,0})
{
    for(int i = 0; i < df.size(); i++)
    {
        write_dataframe(base_dir + std::to_string(i) + "/" + basename, df[i], append, offsets);
    }
}



#endif
