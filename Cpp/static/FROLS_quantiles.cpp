#include "FROLS_quantiles.hpp"
#include <algorithm>
namespace FROLS
{
double quantile(std::vector<double> list, double tau) {
  typename std::vector<double>::iterator b = list.begin();
  typename std::vector<double>::iterator e = list.end();
  typename std::vector<double>::iterator quant = list.begin();
  const size_t pos = tau * std::distance(b, e);
  std::advance(quant, pos);

  std::nth_element(b, quant, e);
  return *quant;
}

std::vector<double> dataframe_quantiles(FROLS::DataFrameStack &dfs,
                                       std::string col_name, double tau) {
  size_t N_rows = dfs[0].get_N_rows();
  size_t N_frames = dfs.get_N_frames();
  std::vector<double> result(N_rows);
  std::vector<double> xk;
  xk.reserve(N_rows);
  for (int i = 0; i < N_rows; i++) {
    for (int j = 0; j < N_frames; j++) {
      xk.push_back((*dfs[j][col_name])[i]);
    }
    result[i] = quantile(xk, tau);
    xk.clear();
  }
  return result;
}
}