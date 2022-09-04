#ifndef FROLS_EIGEN_HPP
#define FROLS_EIGEN_HPP
#include "FROLS_DataFrame.hpp"
#include <FROLS_Typedefs.hpp>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace FROLS
{
    Vec dataframe_to_vector(DataFrame& df, const std::string& col_name);
    Vec dataframe_to_vector(DataFrameStack& dfs, const std::string& col_name);
    Mat dataframe_to_matrix(DataFrame& df, const std::vector<std::string>& col_names);
    Mat dataframe_to_matrix(DataFrameStack& dfs, const std::vector<std::string>& col_names);
}


#endif
