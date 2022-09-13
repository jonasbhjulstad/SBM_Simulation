#ifndef FROLS_EIGEN_HPP
#define FROLS_EIGEN_HPP
#include <DataFrame.hpp>
#include <Typedefs.hpp>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace FROLS
{
    Vec dataframe_to_vector(DataFrame& df, const std::string& col_name, int start_idx = 0, int end_idx = -1);
    Vec dataframe_to_vector(DataFrameStack& dfs, const std::string& col_name, int start_idx = 0, int end_idx = -1);
    Mat dataframe_to_matrix(DataFrame& df, const std::vector<std::string>& col_names, int start_row = 0, int end_row = -1);
    Mat dataframe_to_matrix(DataFrameStack& dfs, const std::vector<std::string>& col_names, int start_row = 0, int end_row = -1);
    Mat dmd_truncate_timeseries(DataFrameStack& dfs, const std::string& col_name, int truncation);
    Vec df_to_vec(const DataFrame& df, const std::string& col_name);
    Vec df_to_vec(const DataFrameStack& dfs, const std::string& col_name);

    Mat dmd_truncate(const Mat& X, double threshold);
    Mat dmd_truncate(DataFrameStack &dfs, const std::vector<std::string>& col_names, double threshold);
}


#endif
