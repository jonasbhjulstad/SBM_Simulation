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

    if (argc > 1)
    {
        Nx = 3;
        network_type = "SIR";
        colnames = {"S", "I", "R"};
    }
    size_t N_sims = 450; // 10000;
    size_t N_pop = 60;
    double p_ER = 1.0;
    using namespace FROLS;
    std::vector<std::string> df_names(N_sims);

    for (int i = 0; i < N_sims; i++) {
        df_names[i] = MC_sim_filename(N_pop, p_ER, i, network_type);
    }


    DataFrameStack dfs(df_names);
    Mat X = dataframe_to_matrix(dfs, colnames,
                                0, -2);
    Mat Y = dataframe_to_matrix(dfs, colnames, 1, -1);
    Mat U = dataframe_to_matrix(dfs, {"p_I"}, 0, -2);
    size_t Nt_max = 0;

    for (int i = 0;i < N_sims; i++)
    {
        Nt_max = std::max({Nt_max, (size_t) dataframe_to_vector(dfs[i], "t").rows()});
    }

    size_t d_max = 1;
    size_t N_output_features = 16;
    using namespace FROLS::Features;
    size_t Nu = U.cols();
    std::vector<size_t> ignore_idx = {0, 1,2, 3,4,5,6};//{0, 1, 2, 3,4};
    double ERR_tolerance = 1e-1;
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max, ignore_idx);
    double theta_tol = 10;
    Regression::ERR_Regressor regressor(ERR_tolerance, theta_tol);
    regressor.transform_fit(X, U, Y, model);

    double u_max = U.maxCoeff()/2;
    Vec x0 = X.row(0);
    Vec u = Vec::Ones(Nt_max) * u_max;
    x0(1) = 10;
    model.feature_summary();
    Mat X_sim = model.simulate(x0, u, Nt_max);
    auto t = FROLS::range(0, Nt_max);
    DataFrame err_traj;
    err_traj.assign(colnames, X_sim);
    err_traj.assign("p_I", u);
    err_traj.assign("t", t);

    err_traj.write_csv(FROLS_DATA_DIR + std::string("/ERR_Trajectory_" + network_type + ".csv"), ",");


    return 0;
}