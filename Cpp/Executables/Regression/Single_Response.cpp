#include <ERR_Regressor.hpp>
#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <iostream>
int main()
{
    size_t N_sims = 10000;
    size_t N_pop = 60;
    double p_ER = 1.0;
    using namespace FROLS;
    std::vector<std::string> df_names(N_sims);

    for (int i = 0; i < N_sims; i++)
    {
        df_names[i] = MC_sim_filename(N_pop, p_ER, i);
    }

    DataFrameStack dfs(df_names);
    Mat X_raw = dataframe_to_matrix(dfs[0], {"S", "I", "R", "p_I"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    size_t d_min = 1;
    size_t d_max = 1;
    auto N_raw_features = X_raw.cols();
    // Features::PolynomialLibrary lib(N_raw_features, d_min, d_max);
    // Mat X_poly = lib.transform(X_raw);

    using namespace FROLS::Features;
    int a = 0;

    


    Mat Y = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(1, Eigen::last), Eigen::all);

    size_t N_output_features = 5;
    double ERR_tolerance = 1e-4;

    //generate x with X.rows ones
    Mat X_quad(Y.rows(), 2);
    X_quad.col(0) = Vec::LinSpaced(Y.rows(), 1, Y.rows());
    X_quad.col(1) = X_quad.col(1).array().square();
    size_t N_input_features = X_quad.cols();
    size_t Nx = N_input_features;
    size_t Nu = 0;

    double theta_tol = 10;
    FROLS::Regression::ERR_Regressor regressor(ERR_tolerance, theta_tol);

    auto rd = regressor.fit(X_quad,  X_quad.col(1));


    return 0;

}