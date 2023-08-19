#include <Sycl_Graph/Simulation/Sim_Write.hpp>
#include <fstream>
#include <algorithm>
#include <array>
auto merge(const std::vector<std::vector<std::vector<uint32_t>>>& t0, const std::vector<std::vector<std::vector<uint32_t>>>& t1)
{
    auto N_sims = t0.size();
    auto Nt = t0[0].size();
    auto N_cols = t0[0][0].size();

    std::vector<std::vector<std::vector<uint32_t>>> merged(t0.size(), std::vector<std::vector<uint32_t>>(t0[0].size(), std::vector<uint32_t>(2*t0[0][0].size())));
    for(int sim_idx = 0; sim_idx < N_sims; sim_idx++)
    {
        for(int t = 0; t < Nt; t++)
        {
            for(int col = 0; col < N_cols; col++)
            {
                merged[sim_idx][t][2*col] = t0[sim_idx][t][col];
                merged[sim_idx][t][2*col+1] = t1[sim_idx][t][col];
            }
        }
    }
    return merged;

}

void events_to_file(const std::vector<std::vector<std::vector<uint32_t>>>& e_to, const std::vector<std::vector<std::vector<uint32_t>>>& e_from, const std::string& abs_fname)
{
    auto merged = merge(e_to, e_from);
    std::ofstream f;
    std::for_each(merged.begin(), merged.end(), [&, n = 0](const auto& sim_ts) mutable
                  {
        f.open(abs_fname + "_" + std::to_string(n) + ".csv");
        std::for_each(sim_ts.begin(), sim_ts.end(), [&](const auto& v)
                      {
            std::for_each(v.begin(), v.end(), [&](const auto& e)
                          {
                f << e << ",";
            });
            f << "\n";
        });
        f.close();
        n++;
    });
}
void timeseries_to_file(const std::vector<std::vector<std::vector<std::array<uint32_t, 3>>>>& ts, const std::string& abs_fname)
{
    std::ofstream f;
    std::for_each(ts.begin(), ts.end(), [&, n = 0](const auto& sim_ts) mutable
                  {
        f.open(abs_fname + "_" + std::to_string(n) + ".csv");
        std::for_each(sim_ts.begin(), sim_ts.end(), [&](const auto& v)
                      {
            std::for_each(v.begin(), v.end(), [&](const auto& e)
                          {
                f << e[0] << "," << e[1] << "," << e[2] << ",";
            });
            f << "\n";
        });
        f.close();
        n++;
    });
}

void timeseries_to_file(const std::vector<std::vector<std::vector<uint32_t>>>& ts, const std::string& abs_fname)
{
    std::ofstream f;
    std::for_each(ts.begin(), ts.end(), [&, n = 0](const auto& sim_ts) mutable
                  {
        f.open(abs_fname + "_" + std::to_string(n) + ".csv");
        std::for_each(sim_ts.begin(), sim_ts.end(), [&](const auto& v)
                      {
            std::for_each(v.begin(), v.end(), [&](const auto& e)
                          {
                f << e;
            });
            f << "\n";
        });
        f.close();
        n++;
    });
}
