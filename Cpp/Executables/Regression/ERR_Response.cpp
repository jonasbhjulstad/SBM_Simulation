#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <ERR_Regressor.hpp>
#include <iostream>
#include <algorithm>

int main(int argc, char** argv) {
    std::string network_type = "SIS";
    std::vector<std::string> colnames = {"S", "I"};
    size_t Nx = 2;

    if (argc == 1)
    {
        Nx = 3;
        network_type = "SIR";
        colnames = {"S", "I", "R"};
    }
    size_t N_sims = 10000; // 10000;
    size_t N_pop = 80;
    double p_ER = 1.0;
    using namespace FROLS;
    std::vector<std::string> df_names(N_sims);

    for (int i = 0; i < N_sims; i++) {
//        df_names[i] = MC_sim_filename(N_pop, p_ER, i, network_type);
        df_names[i] = std::string(FROLS_DATA_DIR) + std::string("/SIR_Sine_Trajectory_") + std::to_string(i) + ".csv";
    }

    size_t Nt_max = 25;
    DataFrameStack dfs(df_names);
    Mat X = dataframe_to_matrix(dfs, colnames,
                                0, Nt_max-1);
    Mat Y = dataframe_to_matrix(dfs, colnames, 1, Nt_max);
    Mat U = dataframe_to_matrix(dfs, {"p_I"}, 0,  Nt_max-1);
//    size_t Nt_max = 0;
//
//    for (int i = 0;i < N_sims; i++)
//    {
//        Nt_max = std::max({Nt_max, (size_t) dataframe_to_vector(dfs[i], "t").rows()});
//    }

    size_t d_max = 1;
    size_t N_output_features = 16;
    using namespace FROLS::Features;
    size_t Nu = U.cols();
    std::vector<size_t> ignore_idx = {0, 1,2, 3,4, 5, 6,7,8};//{0, 1, 2, 3,4};
    double ERR_tolerance = 1e-1;
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max, ignore_idx);
    double theta_tol = 1e-6;
    size_t N_terms_max = 2;
    Regression::ERR_Regressor regressor(ERR_tolerance, theta_tol, N_terms_max);
    regressor.transform_fit(X, U, Y, model);

//    double u_max = U.maxCoeff()/2;
    Vec x0 = X.row(0);
//    Vec u = Vec::Ones(Nt_max) * u_max;
    model.feature_summary();
    Vec t = df_to_vec(dfs[0], "t").head(Nt_max);
    for (int i = 0; i < N_sims; i++)
    {
        x0 = dataframe_to_matrix(dfs[i], colnames).row(0);
        Vec u = df_to_vec(dfs[i], "p_I").topRows(Nt_max-1);
        Mat X_sim = model.simulate(x0, u, Nt_max-1);
        DataFrame er_traj;
        er_traj.assign(colnames, X_sim);
        er_traj.assign("p_I", u);
        er_traj.assign("t", t);
        er_traj.write_csv(FROLS_DATA_DIR + std::string("/ERR_Trajectory_" + network_type +  "_" + std::to_string(i) + ".csv"), ",");
    }


    return 0;
}