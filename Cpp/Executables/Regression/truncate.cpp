
#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <iostream>
int main()
{
    uint32_t N_sims = 10000;
    uint32_t N_pop = 60;
    float p_ER = 1.0;
    using namespace FROLS;
    std::vector<std::string> df_names(N_sims);

    for (int i = 0; i < N_sims; i++)
    {
        df_names[i] = MC_filename(N_pop, p_ER, i);
    }

    DataFrameStack dfs(df_names);
    // Mat X_raw = dataframe_to_matrix(dfs[0], {"S", "I", "R", "p_I"})(Eigen::seq(0, Eigen::last - 1), Eigen::all);
    float threshold = 1e-1;

    Mat X_trunc = dmd_truncate(dfs, {"S", "I", "R"}, threshold); 
    std::cout << X_trunc.topRows(3) << std::endl;
}