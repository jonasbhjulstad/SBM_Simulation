#ifndef FROLS_QUANTILES_HPP
#define FROLS_QUANTILES_HPP
#include "DataFrame.hpp"
#include <FROLS_Math.hpp>
#include <vector>
#include <string>
#include <FROLS_Execution.hpp>
#include <functional>
namespace FROLS
{
float quantile(std::vector<float> list, float tau = .95);

std::vector<float> dataframe_quantiles(FROLS::DataFrameStack &dfs,
                                       std::string col_name, float tau = .95);
void quantiles_to_file(uint32_t N_simulations, const std::vector<std::string>& colnames, std::function<std::string(uint32_t)> MC_fname_f, std::function<std::string(uint32_t)> q_fname_f);

}
#endif