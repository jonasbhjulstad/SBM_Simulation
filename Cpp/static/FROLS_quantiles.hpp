#ifndef FROLS_QUANTILES_HPP
#define FROLS_QUANTILES_HPP
#include "FROLS_DataFrame.hpp"
#include <vector>
#include <string>
namespace FROLS
{
double quantile(std::vector<double> list, double tau = .95);

std::vector<double> dataframe_quantiles(FROLS::DataFrameStack &dfs,
                                       std::string col_name, double tau = .95);
}
#endif