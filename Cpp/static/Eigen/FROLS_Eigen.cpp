#include "FROLS_Eigen.hpp"
#include <Eigen/SVD>

namespace FROLS
{

    Vec dataframe_to_vector(DataFrame& df, const std::string& col_name, int start_idx, int end_idx)
    {
        auto sequence = Eigen::all;
        size_t N_rows = df.get_N_rows();
        if (N_rows <= 0)
        {
            throw std::invalid_argument("[DataFrame] Error: Dataframe must contain elements");
        }
        if (end_idx < 0)
        {
            return Eigen::Map<Eigen::VectorXd>(df[col_name]->data(), N_rows)(Eigen::seq(start_idx, N_rows + end_idx));
        }
        else
        {
            return Eigen::Map<Eigen::VectorXd>(df[col_name]->data(), N_rows)(Eigen::seq(start_idx, end_idx));
        }

    }
    Vec dataframe_to_vector(DataFrameStack& dfs, const std::string& col_name, int start_idx, int end_idx)
    {
        Vec res(dfs.get_N_frames()*dfs[0][col_name]->size());
        for (size_t i = 0; i < dfs.get_N_frames(); i++)
        {
            res.segment(i*dfs[0][col_name]->size(), dfs[0][col_name]->size()) = dataframe_to_vector(dfs[i], col_name, start_idx, end_idx);
        }
        return res;
    }
    Mat dataframe_to_matrix(DataFrame &df, const std::vector<std::string> &col_names, int start_idx, int end_idx)
    {
        size_t N_rows = (end_idx >= 0) ? end_idx - start_idx : df.get_N_rows() + end_idx+1 - start_idx;
        Mat res(N_rows, col_names.size());
        for (size_t i = 0; i < col_names.size(); i++)
        {
            res.col(i) = dataframe_to_vector(df, col_names[i], start_idx, end_idx);
        }
        return res;
    }
    Mat dataframe_to_matrix(DataFrameStack &dfs, const std::vector<std::string> &col_names, int start_idx, int end_idx)
    {
        size_t N_frame_rows = (end_idx >= 0) ? end_idx - start_idx : dfs[0].get_N_rows() + end_idx+1 - start_idx;

        Mat res(dfs.get_N_frames()*N_frame_rows, col_names.size());
        for (size_t i = 0; i < dfs.get_N_frames(); i++)
        {
            res(Eigen::seqN(i*N_frame_rows, N_frame_rows), Eigen::all) = dataframe_to_matrix(dfs[i], col_names, start_idx, end_idx);
        }
        return res;
    }

    Vec df_to_vec(const DataFrame& df, const std::string& col_name)
    {
        return Eigen::Map<Eigen::VectorXd>(df[col_name]->data(), df[col_name]->size());
    }

    Vec df_to_vec(DataFrameStack& dfs, const std::string& col_name)
    {
        size_t N_samples = dfs[0].get_N_rows();
        Vec res(dfs.get_N_frames()*N_samples);
        for (size_t i = 0; i < dfs.get_N_frames(); i++)
        {
            res.segment(i*N_samples, i*(N_samples+1)) = df_to_vec(dfs[i], col_name);
        }
        return res;
    }

    Mat dmd_truncate(const Mat& Xk, const Mat& Xk_1, double threshold)
    {
        Eigen::JacobiSVD<Mat> svd(Xk, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // svd.setThreshold(threshold);
        Mat U = svd.matrixU();
        Mat V = svd.matrixV();
        Vec sigma = svd.singularValues();
        return svd.matrixU().transpose()*(Xk_1*svd.matrixV()*(sigma.asDiagonal().inverse())); 
    }

    Mat dmd_truncate(DataFrameStack &dfs, const std::vector<std::string>& col_names, double threshold)
    {
        size_t Nt = dfs[0].get_N_rows();
        size_t N_frames = dfs.get_N_frames();
        size_t N_features = col_names.size();
        Mat X(N_features, N_frames*Nt);
        Mat Xk(N_features, N_frames*(Nt-1)+1);
        Mat Xk_1(N_features, N_frames*(Nt-1)+1);
        using Eigen::seq;
        for (size_t i = 0; i < N_features; i++)
        {
            for (size_t j = 0; j < N_frames; j++)
            {
                Vec xij = df_to_vec(dfs[j], col_names[i]);
                Xk(i, seq(j*(Nt-1), (j+1)*(Nt-1)-1)) = xij.head(Nt-1);
                Xk_1(i, seq(j*(Nt-1), (j+1)*(Nt-1)-1)) = xij.tail(Nt-1);
            
            }
        }
        return dmd_truncate(Xk, Xk_1, threshold);
    }
}
