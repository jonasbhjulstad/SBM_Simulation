#include "FROLS_Eigen.hpp"

namespace FROLS
{

    Vec dataframe_to_vector(DataFrame& df, const std::string& col_name)
    {
        return Eigen::Map<Eigen::VectorXd>(df[col_name]->data(), df[col_name]->size());
    }
    Vec dataframe_to_vector(DataFrameStack& dfs, const std::string& col_name)
    {
        Vec res(dfs.get_N_frames()*dfs[0][col_name]->size());
        for (size_t i = 0; i < dfs.get_N_frames(); i++)
        {
            res.segment(i*dfs[0][col_name]->size(), dfs[0][col_name]->size()) = dataframe_to_vector(dfs[i], col_name);
        }
        return res;
    }
    Mat dataframe_to_matrix(DataFrame &df, const std::vector<std::string> &col_names)
    {
        Mat res(df[col_names[0]]->size(), col_names.size());
        for (size_t i = 0; i < col_names.size(); i++)
        {
            res.col(i) = dataframe_to_vector(df, col_names[i]);
        }
        return res;
    }
    Mat dataframe_to_matrix(DataFrameStack &dfs, const std::vector<std::string> &col_names)
    {
        Mat res(dfs.get_N_frames()*dfs[0][col_names[0]]->size(), col_names.size());
        for (size_t i = 0; i < dfs.get_N_frames(); i++)
        {
            res.block(i*dfs[0][col_names[0]]->size(), 0, dfs[0][col_names[0]]->size(), col_names.size()) = dataframe_to_matrix(dfs[i], col_names);
        }
        return res;
    }
}
