//
// Created by arch on 9/22/22.
//

#include "Regression_Algorithm.hpp"
namespace FROLS::Regression
{
    void from_file_regression(std::function<std::string(size_t)> MC_fname_f, const std::vector<std::string> &colnames_x,
                              const std::vector<std::string> &colnames_u, size_t N_sims,
                              FROLS::Regression::Regressor &regressor, FROLS::Features::Feature_Model  &feature_model,
                              std::function<std::string(size_t)> outsim_f) {
        std::vector<std::string> df_names(N_sims);
        std::generate(df_names.begin(), df_names.end(), [&, n = 0]()mutable {
            n++;return MC_fname_f(n);});
        size_t Nt_max = 0;
        size_t Nx = colnames_x.size();
        using namespace FROLS;
        DataFrameStack dfs(df_names);
        Mat X = dataframe_to_matrix(dfs, colnames_x,
                                    0, -2);
        Mat Y = dataframe_to_matrix(dfs, colnames_x, 1, -1);
        Mat U = dataframe_to_matrix(dfs, colnames_u, 0, -2);

        std::vector<size_t> N_rows = dfs.get_N_rows();

        using namespace FROLS::Features;

        regressor.transform_fit(X, U, Y, feature_model);
        feature_model.feature_summary();

        Vec t = df_to_vec(dfs[0], "t");
        size_t offset = 0;
        for (int i = 0; i < N_sims; i++) {
            DataFrame df(MC_fname_f(i));
            Vec u = df_to_vec(df, "p_I");
            Vec x0 = X.row(offset);
            Mat X_sim = feature_model.simulate(x0, u, N_rows[i] - 1);
            offset += N_rows[i]-1;
            DataFrame er_traj;
            er_traj.assign(colnames_x, X_sim);
            er_traj.assign(colnames_u, u);
            er_traj.assign("t", t);
            er_traj.write_csv(outsim_f(i), ",");
        }

    }
}