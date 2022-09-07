#include "FROLS_Polynomial.hpp"
#include <FROLS_Algorithm.hpp>
#include <FROLS_DataFrame.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Eigen.hpp>
#include <FROLS_Features.hpp>
#include <iostream>
#include <itertools.hpp>
int main()
{
    size_t N_sims = 100;//10000;
    size_t N_pop = 60;
    double p_ER = 0.1;
    using namespace FROLS;
    std::vector<std::string> df_names(N_sims);

    for (int i = 0; i < N_sims; i++)
    {
        df_names[i] = MC_sim_filename(N_pop, p_ER, i);
    }

    DataFrameStack dfs(df_names);
    Mat X_raw = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    Mat Y = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(1, Eigen::last), Eigen::all);
    Mat U = dataframe_to_matrix(dfs[0], {"p_I"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    size_t d_max = 1;
    size_t N_output_features = 100;
    using namespace FROLS::Features;
    int a = 0;
    Mat X_poly = Polynomial::feature_transform(X_raw.leftCols(3), d_max, N_output_features);

    double ERR_tolerance = 1e-4;
    FROLS::Features::Polynomial::Polynomial_Model model(N_output_features, d_max);

    model.multiple_response_regression(X_raw, U, Y, ERR_tolerance);
    // auto rds = (Y,  X_poly, ERR_tolerance);
    model.print();
    // std::cout << Regression::regression_data_summary(rds) << std::endl;

    Vec x0 = X_raw.row(0);
    double u0 = U(0,0);
    size_t Nt = 100;
    Vec u = Vec::Ones(Nt)*u0;
    //print x0, u
    std::cout << "x0 = " << x0.transpose() << std::endl;
    std::cout << "u0 = " << u0 << std::endl;
    model.feature_summary();
    model.simulate(x0, u, Nt);



    return 0;

}