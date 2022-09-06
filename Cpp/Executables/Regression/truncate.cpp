#include <FROLS_Algorithm.hpp>
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
    // Mat X_raw = dataframe_to_matrix(dfs[0], {"S", "I", "R", "p_I"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    double threshold = 1e-1;

    Mat X_trunc = dmd_truncate(dfs, {"S", "I", "R"}, threshold); 
    std::cout << X_trunc.topRows(3) << std::endl;
}