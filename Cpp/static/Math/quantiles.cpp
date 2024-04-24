#include "quantiles.hpp"
#include <FROLS_Path_Config.hpp>
#include <FROLS_Math.hpp>

#include <thread>
namespace FROLS {
    float quantile(std::vector<float> list, float tau) {
        typename std::vector<float>::iterator b = list.begin();
        typename std::vector<float>::iterator e = list.end();
        typename std::vector<float>::iterator quant = list.begin();
        const uint32_t pos = tau * std::distance(b, e);
        std::advance(quant, pos);

        std::nth_element(b, quant, e);
        return *quant;
    }

    std::vector<float> dataframe_quantiles(DataFrameStack &dfs,
                                            std::string col_name, float tau) {
        uint32_t N_rows = 0;
        for (int i = 0; i < dfs.get_N_frames(); i++) {
            N_rows = std::max({N_rows, (uint32_t) dfs[i].get_N_rows()});
        }
        uint32_t N_frames = dfs.get_N_frames();
        std::vector<float> result(N_rows);
        std::vector<float> xk;
        xk.reserve(N_rows);
        for (int i = 0; i < N_rows; i++) {
            for (int j = 0; j < N_frames; j++) {
                if (dfs[j].get_N_rows() > i) {
                    xk.push_back((*dfs[j][col_name])[i]);

                }
            }
            result[i] = quantile(xk, tau);
            xk.clear();
        }
        return result;
    }

    void quantiles_to_file(uint32_t N_simulations, const std::vector<std::string>& colnames, std::function<std::string(uint32_t)> MC_fname_f, std::function<std::string(uint32_t)> q_fname_f) {
        static std::thread::id thread_0 = std::this_thread::get_id();
        std::vector<std::string> filenames(N_simulations);
        uint32_t iter = 0;
        for (int i = 0; i < N_simulations; i++)
        {
            filenames[i] = MC_fname_f(i);
        }
        {
            using namespace FROLS;
            DataFrameStack dfs(filenames);
            uint32_t N_rows = dfs[0].get_N_rows();
            std::vector<float> t = (*dfs[0]["t"]);
            std::vector<float> xk(N_simulations);

            std::vector<float> tau = FROLS::arange(0.05f, 1.00f, 0.05f);

            std::vector<std::vector<uint32_t>> q_trajectories(tau.size());
            for (auto &traj: q_trajectories) {
                traj.resize(N_rows);
            }
            for (int i = 0; i < q_trajectories.size(); i++) {
                if (std::this_thread::get_id() == thread_0)
                {
                    // std::cout << "Quantile " << i+1 << " of " << q_trajectories.size() << std::endl;
                }
                DataFrame df;
                df.assign("t", t);
                df.resize(t.size());

                std::for_each(colnames.begin(), colnames.end(), [&](const auto& colname){df.assign(colname, dataframe_quantiles(dfs, colname, tau[i]));});
                df.write_csv(q_fname_f(uint32_t(tau[i]*100)), ",");

            }

        }
    }
}