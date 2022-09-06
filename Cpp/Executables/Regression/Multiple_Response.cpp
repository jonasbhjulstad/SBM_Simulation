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
    Mat X_raw = dataframe_to_matrix(dfs[0], {"S", "I", "R", "p_I"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    Mat Y = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(1, Eigen::last), Eigen::all);
    size_t d_min = 1;
    size_t d_max = 1;
    auto N_raw_features = X_raw.cols();
    // Features::PolynomialLibrary lib(N_raw_features, d_min, d_max);
    // Mat X_poly = lib.transform(X_raw);

    using namespace FROLS::Features;
    int a = 0;
    size_t N_input_features = 3;
    size_t N_output_features = 100;
    Mat X_poly = Polynomial::feature_transform(X_raw.leftCols(3), d_max, N_output_features);

    // Polynomial::feature_display(X_poly, d_max, N_input_features);    
    



    double ERR_tolerance = 1e-4;

    auto rds = Regression::multiple_response_regression(Y,  X_poly, ERR_tolerance);
    
    std::cout << Regression::regression_data_summary(rds) << std::endl;

    std::cout << Polynomial::model_print(rds, d_max, N_input_features, N_output_features);


    return 0;

}