//
// Created by arch on 9/22/22.
//

#include "Regression_Algorithm.hpp"
namespace FROLS::Regression
{
    void from_file_regression(std::function<std::string(uint16_t)> MC_fname_f, const std::vector<std::string> &colnames_x,
                              const std::vector<std::string> &colnames_u, uint16_t N_sims,
                              FROLS::Regression::Regressor &regressor, FROLS::Features::Feature_Model  &feature_model,
                              std::function<std::string(uint16_t)> outsim_f) {
        std::vector<std::string> df_names(N_sims);
        std::generate(df_names.begin(), df_names.end(), [&, n = 0]()mutable {
            n++;return MC_fname_f(n);});
        uint16_t Nt_max = 0;
        uint16_t Nx = colnames_x.size();
        using namespace FROLS;
        DataFrameStack dfs(df_names);
        Mat X = dataframe_to_matrix(dfs, colnames_x,
                                    0, -2);
        Mat Y = dataframe_to_matrix(dfs, colnames_x, 1, -1);
        Mat U = dataframe_to_matrix(dfs, colnames_u, 0, -2);

        std::vector<uint16_t> N_rows = dfs.get_N_rows();

        using namespace FROLS::Features;

        regressor.transform_fit(X, U, Y, feature_model);
        feature_model.feature_summary();
        for (int i = 0; i < N_sims; i++) {
            Mat u = vecs_to_mat(dfs[i][colnames_u])(Eigen::seq(0, Eigen::last - 1), Eigen::all);
            Vec x0 = vecs_to_mat(dfs[i][colnames_x]).row(0);
            Mat X_sim = feature_model.simulate(x0, u, u.rows());
            FROLS::DataFrame df;
            df.assign({colnames_u}, u);
            df.assign("t", FROLS::range(0, u.rows()));
            df.assign(colnames_x, X_sim);
            df.write_csv(outsim_f(i), ",");
        }

    }
}