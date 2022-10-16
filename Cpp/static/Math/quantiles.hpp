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
double quantile(std::vector<double> list, double tau = .95);

std::vector<double> dataframe_quantiles(FROLS::DataFrameStack &dfs,
                                       std::string col_name, double tau = .95);
void quantiles_to_file(size_t N_simulations, const std::vector<std::string>& colnames, std::function<std::string(size_t)> MC_fname_f, std::function<std::string(size_t)> q_fname_f);

}
#endif