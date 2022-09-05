#include <FROLS.hpp>
#include <FROLS_DataFrame.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Eigen.hpp>
#include <FROLS_Features.hpp>
#include <iostream>
#include <itertools.hpp>
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
    size_t d_max = 2;
    auto N_raw_features = X_raw.cols();
    // Features::PolynomialLibrary lib(N_raw_features, d_min, d_max);
    // Mat X_poly = lib.transform(X_raw);

    using namespace FROLS::Features;
    int a = 0;
    Mat X_poly = Polynomial::feature_transform(X_raw, d_max, 5);

    Polynomial::feature_display(X_poly, d_max, 3);    
    


    Mat Y = dataframe_to_matrix(dfs[0], {"S", "I", "R"})(Eigen::seq(1, Eigen::last), Eigen::all);

    auto rd = Regression::single_response_regression(X_raw.leftCols(3),  Y.col(0), 1e-1);
    
    std::cout << regression_data_summary(rd) << std::endl;

    

    return 0;

}