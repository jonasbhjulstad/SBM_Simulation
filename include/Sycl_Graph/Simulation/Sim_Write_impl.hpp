#ifndef SIM_WRITE_IMPL_HPP
#define SIM_WRITE_IMPL_HPP
#include <vector>
#include <filesystem>
#include <fstream>
#include <Sycl_Graph/Simulation/Sim_Write.hpp>
#include <Sycl_Graph/SIR_Types.hpp>

template <typename T>
void write_timeseries(const Timeseries_t<T>&& data, std::fstream& f)
{
    auto N0 = data.size();
    auto N1 = data[0].size();
    for(auto n0 = 0; n0 < N0; n0++)
    {
        for(auto n1 = 0; n1 < N1; n1++)
        {
            f << data[n0][n1];
            if(n1 < N1 - 1)
            {
                f << ",";
            }
        }
        f << "\n";
    }
}

template <>
void write_timeseries(const Timeseries_t<State_t>&& data, std::fstream& f)
{
    auto N0 = data.size();
    auto N1 = data[0].size();
    for(auto n0 = 0; n0 < N0; n0++)
    {
        for(auto n1 = 0; n1 < N1; n1++)
        {
            f << data[n0][n1][0] << "," << data[n0][n1][1] << "," << data[n0][n1][2];
        }
        f << "\n";
    }
}


template <typename T>
void write_timeseries(const Simseries_t<T>&& data, const std::string& fname, bool append = false)
{
    auto N0 = data.size();
    auto N1 = data[0].size();
    auto N2 = data[0][0].size();
    std::fstream f;
    for(auto n0 = 0; n0 < N0; n0++)
    {
        f.open(fname + "_" + std::to_string(n0) + ".csv", std::ios::out | (append ? std::ios::app : std::ios::trunc));
        write_timeseries(std::forward<const Timeseries_t<T>>(data[n0]), f);
        f.close();
    }
}

template <typename T>
void write_timeseries(const Graphseries_t<T>&& data, const std::string& base_dir, const std::string& fname, bool append = false)
{
    auto N0 = data.size();
    auto N1 = data[0].size();
    auto N2 = data[0][0].size();
    auto N3 = data[0][0][0].size();
    for(int n0 = 0; n0 < N0; n0++)
    {
        std::filesystem::create_directories(base_dir + "Graph_" + std::to_string(n0) + "/");
        write_timeseries(std::forward<const Simseries_t<T>>(data[n0]), base_dir + "Graph_" + std::to_string(n0) + "/" + fname, append);
    }
}



#endif
