#ifndef SYCL_GRAPH_REGRESSION_HPP
#define SYCL_GRAPH_REGRESSION_HPP
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/Quantile_Regression.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <sstream>

namespace Sycl_Graph
{
    using Mat = Eigen::MatrixXf;
    using Vec = Eigen::VectorXf;
    using namespace std;
    static constexpr size_t MAXBUFSIZE = 100000;

    using namespace Eigen;
    using namespace Sycl_Graph;
    MatrixXf openData(string fileToOpen)
    {

        vector<float> matrixEntries;

        ifstream matrixDataFile(fileToOpen);

        string matrixRowString;

        string matrixEntry;

        int matrixRowNumber = 0;

        while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
        {
            stringstream matrixRowStringStream(matrixRowString); // convert matrixRowString that is a string to a stream variable.

            while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
            {
                matrixEntries.push_back(stod(matrixEntry)); // here we convert the string to double and fill in the row vector storing all the matrix entries
            }
            matrixRowNumber++; // update the column numbers
        }

        return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    }

    std::pair<Mat, Mat> load_beta_regression(const std::string datapath, uint32_t idx)
    {
        Mat connection_infs = openData(datapath + connection_infs_filename(idx));
        Mat connection_community_map = openData(datapath + connection_community_map_filename());
        Mat community_traj = openData(datapath + community_traj_filename(idx));

        assert((connection_infs.array() <= 1000).all());
        assert((connection_infs.array() >= 0).all());
        assert((community_traj.array() <= 1000).all());
        auto N_connections = connection_infs.cols();
        auto Nt = community_traj.rows() - 1;
        uint32_t N_communities = community_traj.cols() / 3;
        Mat p_Is = openData(datapath + p_Is_filename(idx));

        std::vector<float> thetas_LS(N_connections);
        std::vector<float> thetas_QR(N_connections);
        Mat F_beta_rs_mat(Nt, N_connections);
        for (int i = 0; i < connection_community_map.rows(); i++)
        {
            auto target_idx = connection_community_map(i, 1);
            auto source_idx = connection_community_map(i, 0);
            // community_traj[:, 3*target_idx:3*target_idx+3]
            Mat target_traj = community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * target_idx, 3));
            Mat source_traj = community_traj(Eigen::seqN(0, Nt), Eigen::seqN(3 * source_idx, 3));
            
            Vec S_r = source_traj.col(0);
            Vec I_r = source_traj.col(1);
            Vec R_r = source_traj.col(2);
            Vec S_s = target_traj.col(0);
            Vec I_s = target_traj.col(1);
            Vec R_s = target_traj.col(2);
            Vec p_I = p_Is.col(i);

            Vec denom = (S_s.array() + I_r.array() + R_s.array()).matrix();
            Vec nom = (S_s.array() * I_r.array()).matrix();
            Vec connection_inf = connection_infs.col(i);
            F_beta_rs_mat.col(i) = p_I.array() * nom.array() / denom.array();
        }
        // not all zero
        assert(F_beta_rs_mat.array().sum() != 0);
        assert(connection_infs.array().sum() != 0);
        return std::make_pair(F_beta_rs_mat, connection_infs);
    }

    std::tuple<Mat, Mat> load_N_datasets(const std::string &datapath, uint32_t N, uint32_t offset = 0)
    {
        std::vector<uint32_t> idx(N);
        std::iota(idx.begin(), idx.end(), offset);
        std::vector<std::pair<Mat, Mat>> datasets;
        std::transform(idx.begin(), idx.end(), std::back_inserter(datasets), [&](auto idx)
                       { return load_beta_regression(datapath, idx); });

        std::vector<std::pair<uint32_t, uint32_t>> sizes;
        std::transform(datasets.begin(), datasets.end(), std::back_inserter(sizes), [](auto &dataset)
                       { return std::make_pair(dataset.first.rows(), dataset.first.cols()); });
        uint32_t tot_rows = std::accumulate(sizes.begin(), sizes.end(), 0, [](auto acc, auto &size)
                                            { return acc + size.first; });
        uint32_t cols = sizes[0].second;
        Mat connection_infs_tot(tot_rows, cols);
        Mat F_beta_rs_mat(tot_rows, cols);
        // fill mats with datasets
        uint32_t row_offset = 0;
        for (int i = 0; i < datasets.size(); i++)
        {
            auto &dataset = datasets[i];
            auto &size = sizes[i];
            connection_infs_tot(Eigen::seqN(row_offset, size.first), Eigen::all) = dataset.second;
            F_beta_rs_mat(Eigen::seqN(row_offset, size.first), Eigen::all) = dataset.first;
            row_offset += size.first;

        }
        assert(F_beta_rs_mat.array().sum() != 0);
        assert(connection_infs_tot.array().sum() != 0);

        return std::make_tuple(F_beta_rs_mat, connection_infs_tot);
    }

    std::pair<std::vector<float>, std::vector<float>> beta_regression(const Mat &F_beta_rs_mat, const Mat &connection_infs, float tau)
    {

        uint32_t N_connections = connection_infs.cols();
        std::vector<float> thetas_LS(N_connections);
        std::vector<float> thetas_QR(N_connections);
        for (int i = 0; i < thetas_QR.size(); i++)
        {
            thetas_LS[i] = connection_infs.col(i).dot(F_beta_rs_mat.col(i)) / F_beta_rs_mat.col(i).dot(F_beta_rs_mat.col(i));
            thetas_QR[i] = quantile_regression(F_beta_rs_mat.col(i), connection_infs.col(i), tau);
        }

        return std::make_pair(thetas_LS, thetas_QR);
    }

    float alpha_regression(const Vec &x, const Vec &y)
    {
        return y.dot(x) / x.dot(x);
    }

    std::tuple<std::vector<float>, std::vector<float>> regression_on_datasets(const std::string &datapath, uint32_t N, float tau, uint32_t offset)
    {
        auto [F_beta_rs_mat, connection_infs] = load_N_datasets(datapath, N, offset);
        // float alpha = alpha_regression(x_recovery, y_recovery);
        auto [thetas_LS, thetas_QR] = beta_regression(F_beta_rs_mat, connection_infs, tau);
        return std::make_tuple(thetas_LS, thetas_QR);
    }

    std::tuple<std::vector<float>, std::vector<float>> regression_on_datasets(const std::vector<std::string> &datapaths, uint32_t N, float tau, uint32_t offset = 0)
    {
        std::vector<std::tuple<Mat, Mat>> datasets(datapaths.size());
        std::transform(datapaths.begin(), datapaths.end(), datasets.begin(), [&](auto &datapath)
                       { return load_N_datasets(datapath, N, offset); });



        Mat F_beta_rs_mat_tot;
        Mat connection_infs_tot;

        uint32_t row_offset = 0;
        //stack datasets
        for (int i = 0; i < datasets.size(); i++)
        {
            auto &dataset = datasets[i];
            auto &F_beta_rs_mat = std::get<0>(dataset);
            auto &connection_infs = std::get<1>(dataset);
            uint32_t N_rows = F_beta_rs_mat.rows();

            F_beta_rs_mat_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) = F_beta_rs_mat;
            connection_infs_tot(Eigen::seqN(row_offset, N_rows), Eigen::all) = connection_infs;
            row_offset += N_rows;
        }

        auto [thetas_LS, thetas_QR] = beta_regression(F_beta_rs_mat_tot, connection_infs_tot, tau);
        return std::make_tuple(thetas_LS, thetas_QR);
    }

}

#endif