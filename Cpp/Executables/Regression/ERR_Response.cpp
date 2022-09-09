#include "Features.hpp"
#include <ERR_Regressor.hpp>
#include <Polynomial.hpp>
#include <DataFrame.hpp>
#include <Path_Config.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <iostream>
int main()
{
    size_t N_sims = 100;//10000;
    size_t N_pop = 60;
    double p_ER = 1.0;
    using namespace FROLS;
    std::vector<std::string> df_names(N_sims);

    for (int i = 0; i < N_sims; i++)
    {
        df_names[i] = MC_sim_filename(N_pop, p_ER, i);
    }

    DataFrameStack dfs(df_names);
    Mat X = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    Mat Y = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(1, Eigen::last), Eigen::all);
    Mat U = dataframe_to_matrix(dfs[0], {"p_I"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    size_t d_max = 1;
    size_t N_output_features = 8;
    using namespace FROLS::Features;
    size_t Nx = X.cols();
    size_t Nu = U.cols();

    double ERR_tolerance = 1e-2;
    FROLS::Regression::ERR_Regressor regressor(ERR_tolerance);
    
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);

    regressor.transform_fit(X, U, Y, model);
    // auto rds = (Y,  X_poly, ERR_tolerance);
    std::cout << model.model_equations() << std::endl;
    // std::cout << Regression::regression_data_summary(rds) << std::endl;

    Vec x0 = X.row(0);
    double u0 = U(0,0);
    size_t Nt = 30;
    Vec u = Vec::Ones(Nt)*u0;
    //print x0, u
    std::cout << "x0 = " << x0.transpose() << std::endl;
    std::cout << "u0 = " << u0 << std::endl;
    model.feature_summary();
    model.simulate(x0, u, Nt);



    return 0;

}