#ifndef SYCL_GRAPH_EIGEN_HPP
#define SYCL_GRAPH_EIGEN_HPP

#include <DataFrame.hpp>
#include <Typedefs.hpp>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace Sycl::Graph {
    Vec dataframe_to_vector(DataFrame &df, const std::string &col_name, int start_idx = 0, int end_idx = -1);

    Vec dataframe_to_vector(DataFrameStack &dfs, const std::string &col_name, int start_idx = 0, int end_idx = -1);

    Mat
    dataframe_to_matrix(DataFrame &df, const std::vector<std::string> &col_names, int start_row = 0, int end_row = -1);

    Mat dataframe_to_matrix(DataFrameStack &dfs, const std::vector<std::string> &col_names, int start_row = 0,
                            int end_row = -1);

    Mat dmd_truncate_timeseries(DataFrameStack &dfs, const std::string &col_name, int truncation);

    Vec df_to_vec(const DataFrame &df, const std::string &col_name);

    Vec df_to_vec(const DataFrameStack &dfs, const std::string &col_name);

    Mat dmd_truncate(const Mat &X, float threshold);

    Mat dmd_truncate(DataFrameStack &dfs, const std::vector<std::string> &col_names, float threshold);

    Mat diff_dataframe_to_matrix(DataFrame& df, const std::vector<std::string> &col_names, int start_idx, int end_idx);

    Mat diff_dataframe_to_matrix(DataFrameStack& dfs, const std::vector<std::string> &col_names, int start_idx, int end_idx);


    template <typename T>
    Mat vecs_to_mat(const std::vector<std::vector<T>> &vectors) {
        uint32_t N_rows = std::max_element(vectors.begin(), vectors.end(),
                                         [](auto &vec0, auto &vec1) { return vec0.size() > vec1.size(); })->size();
        Mat res(N_rows, vectors.size());
        for (int i = 0; i < res.cols(); i++) {
            for (int j = 0; j < vectors[i].size(); j++) {
                res.col(i)(j) = (float) vectors[i][j];
            }
        }
        return res;
    }

}


#endif
