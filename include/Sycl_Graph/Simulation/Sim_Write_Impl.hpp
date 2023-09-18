#ifndef SIM_WRITE_IMPL_HPP
#define SIM_WRITE_IMPL_HPP
#include <Sycl_Graph/Simulation/Sim_Types.hpp>
#include <fstream>
#include <filesystem>
template <typename T>
void write_timeseries(const Timeseries_t<T>&& ts, std::fstream& f)
{
    auto Nt = ts.Nt;
    auto N_cols = ts.N_cols;
    for(int t = 0; t < Nt; t++)
    {
        for(int c = 0; c < N_cols; c++)
        {
            f << ts[t][c];
            if(c < N_cols - 1)
            {
                f << ",";
            }
        }
        f << "\n";
    }
}

template <>
void write_timeseries(const Timeseries_t<State_t>&& ts, std::fstream& f)
{
    auto Nt = ts.Nt;
    auto N_cols = ts.N_cols;
    for(int t = 0; t < Nt; t++)
    {
        for(int c = 0; c < N_cols; c++)
        {
            f << ts[t][c][0] << "," << ts[t][c][1] << "," << ts[t][c][2];
            if(c < N_cols - 1)
            {
                f << ",";
            }
        }
        f << "\n";
    }
}


template <typename T>
void write_simseries(const Simseries_t<T>&& ss, const std::string& abs_fname, bool append)
{
    auto N_sims = ss.N_sims;
    std::fstream f;
    for(int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        auto fname = abs_fname + "_" + std::to_string(sim_idx) + ".csv";
        if (append)
        {
            f.open(fname, std::ios::out | std::ios::app);
        }
        else
        {
            f.open(fname, std::ios::out);
        }
        write_timeseries(std::forward<const Timeseries_t<T>>(ss[sim_idx]), f);
        f.close();
    }
}

template <typename T>
void write_graphseries(const Graphseries_t<T>&& gs, const std::string& base_dir, const std::string& base_fname, bool append)
{
    auto N_graphs = gs.Ng;
    for(auto graph_idx = 0; graph_idx < N_graphs; graph_idx++)
    {
        auto iter_dir = base_dir + "/Graph_" + std::to_string(graph_idx) + "/";
        std::filesystem::create_directories(iter_dir);
        write_simseries(std::forward<const Simseries_t<T>>(gs[graph_idx]), iter_dir + base_fname, append);
    }
}


Graphseries_t<uint32_t> zip_merge_graphseries(const Graphseries_t<uint32_t>&& gs0, const Graphseries_t<uint32_t>&& gs1)
{
    auto N_graphs = gs0.Ng;
    auto N_sims = gs0.N_sims;
    auto Nt = gs0.Nt;
    auto N_columns = gs0.N_cols;
    Graphseries_t<uint32_t> result(N_graphs, N_sims, Nt, N_columns*2);
    for(int graph_idx = 0; graph_idx < N_graphs; graph_idx++)
    {
        for(int sim_idx = 0; sim_idx < N_sims; sim_idx++)
        {
            for(int t = 0; t < Nt; t++)
            {
                for(int c = 0; c < N_columns; c++)
                {
                    result[graph_idx][sim_idx][t][2*c] = gs0[graph_idx][sim_idx][t][c];
                    result[graph_idx][sim_idx][t][2*c + 1] = gs1[graph_idx][sim_idx][t][c];
                }
            }
        }
    }
    return result;
}

void write_to_file(const std::vector<std::pair<uint32_t, uint32_t>>& ccm, const std::string& fname)
{
    std::ofstream f(fname);
    for(auto& [c0, c1] : ccm)
    {
        f << c0 << "," << c1 << "\n";
    }
    f.close();
}
void ccms_to_file(const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& ccms, const std::string& output_dir)
{
    auto n = 0;
    for(auto&& ccm: ccms)
    {
        write_to_file(ccm, output_dir + "/Graph_" + std::to_string(n) + "/ccm.csv");
        n++;
    }
}


#endif
