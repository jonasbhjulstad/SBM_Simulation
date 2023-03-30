#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <Sycl_Graph/path_config.hpp>
#include <Sycl_Graph/Quantile_Regression.hpp>
#include <string>
#include <sstream>
using Mat = Eigen::MatrixXf;
using Vec = Eigen::VectorXf;
using namespace std;
static constexpr size_t MAXBUFSIZE = 100000;

using namespace Eigen;
using namespace Sycl_Graph;
MatrixXf openData(string fileToOpen)
{

    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format

    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    vector<float> matrixEntries;

    // in this object we store the data from the matrix
    ifstream matrixDataFile(fileToOpen);

    // this variable is used to store the row of the matrix that contains commas
    string matrixRowString;

    // this variable is used to store the matrix entry;
    string matrixEntry;

    // this variable is used to track the number of rows
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

    // here we convet the vector variable into the matrix and return the resulting object,
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

std::pair<Mat, Mat> connection_regression(const std::string datapath, uint32_t idx)
{
    Mat connection_infs = openData(datapath + connection_infs_filename(idx));
    Mat connection_targets = openData(datapath + connection_targets_filename(idx));
    Mat connection_sources = openData(datapath + connection_sources_filename(idx));
    Mat community_traj = openData(datapath + community_traj_filename(idx));
    auto N_connections = connection_infs.cols();
    auto Nt = community_traj.rows() - 1;
    uint32_t N_communities = community_traj.cols() / 3;
    Mat p_Is_to = openData(datapath + p_Is_filename(idx));

    Mat p_Is_from = p_Is_to(Eigen::all, Eigen::seq(0, Eigen::last - N_communities));
    Mat p_Is(Nt, N_connections);
    p_Is << p_Is_to, p_Is_from;
    std::vector<float> thetas_LS(N_connections);
    std::vector<float> thetas_QR(N_connections);
    Mat F_beta_rs_mat(Nt, N_connections);
    for (int i = 0; i < connection_targets.cols(); i++)
    {
        auto target_idx = connection_targets(0, i);
        auto source_idx = connection_sources(0, i);
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
    //not all zero
    assert(F_beta_rs_mat.array().sum() != 0);
    assert(connection_infs.array().sum() != 0);
    return std::make_pair(F_beta_rs_mat, connection_infs);
}

std::tuple<Mat, Mat, Vec, Vec> load_N_datasets(const std::string& datapath, uint32_t N)
{
    std::vector<uint32_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<std::pair<Mat, Mat>> datasets;
    std::transform(idx.begin(), idx.end(), std::back_inserter(datasets), [&](auto idx) { return connection_regression(datapath, idx); });
    
    std::vector<std::pair<uint32_t, uint32_t>> sizes; 
    std::transform(datasets.begin(), datasets.end(), std::back_inserter(sizes), [](auto& dataset) { return std::make_pair(dataset.first.rows(), dataset.first.cols()); });
    uint32_t tot_rows = std::accumulate(sizes.begin(), sizes.end(), 0, [](auto acc, auto& size) { return acc + size.first; });
    uint32_t cols = sizes[0].second;
    Mat connection_infs_tot(tot_rows, cols);
    Mat F_beta_rs_mat(tot_rows, cols);
    //fill mats with datasets
    uint32_t row_offset = 0;

    Vec y_recovery(tot_rows);
    Vec x_recovery(tot_rows);
    uint32_t rec_offset = 0;
    for (int i = 0; i < datasets.size(); i++)
    {
        auto& dataset = datasets[i];
        auto& size = sizes[i];
        connection_infs_tot(Eigen::seqN(row_offset, size.first), Eigen::all) = dataset.second;
        F_beta_rs_mat(Eigen::seqN(row_offset, size.first), Eigen::all) = dataset.first;
        row_offset += size.first;


        Mat tot_traj = openData(datapath + tot_traj_filename(i));
        uint32_t N_elems = tot_traj.rows()-1;
        x_recovery(Eigen::seqN(rec_offset, N_elems)) = tot_traj.col(1).head(N_elems);
        y_recovery(Eigen::seqN(rec_offset, N_elems)) = tot_traj.col(2).tail(N_elems) - tot_traj.col(2).head(N_elems);
        rec_offset += N_elems;

        Vec x = tot_traj.col(1).head(tot_traj.rows() - 1);

    }
    assert(F_beta_rs_mat.array().sum() != 0);
    assert(connection_infs_tot.array().sum() != 0);

    return std::make_tuple(F_beta_rs_mat, connection_infs_tot, x_recovery, y_recovery);

}

void beta_regression(const Mat& F_beta_rs_mat, const Mat& connection_infs, float tau, std::string datapath)
{
    
    uint32_t N_connections = connection_infs.cols();
    std::vector<float> thetas_LS(N_connections);
    std::vector<float> thetas_QR(N_connections);
    for(int i = 0; i < thetas_QR.size(); i++)
    {
        thetas_LS[i] = connection_infs.col(i).dot(F_beta_rs_mat.col(i)) / F_beta_rs_mat.col(i).dot(F_beta_rs_mat.col(i));
        thetas_QR[i] = quantile_regression(F_beta_rs_mat.col(i), connection_infs.col(i), tau);
    }

    auto writeline = [](auto& f, const auto& v)
    {
        for (int i = 0; i < v.size(); i++)
        {
            f << v[i];
            if (i != v.size() - 1)
            {
                f << ",";
            }
        }
        f << std::endl;
    };

    std::ofstream theta_LS_f(datapath + "/theta_LS.csv");
    writeline(theta_LS_f, thetas_LS);
    std::ofstream theta_QR_f(datapath + "/theta_QR.csv");
    writeline(theta_QR_f, thetas_QR);
}

void alpha_regression(const Vec& x, const Vec& y, std::string datapath)
{
    float alpha = y.dot(x) / x.dot(x);
    std::ofstream alpha_f(datapath + "/alpha.csv");
    alpha_f << alpha;
}

int main()
{
    float tau = .5f;
    std::string path = Sim_Datapath + "/Graph_0/";
    auto [F_beta_rs_mat,connection_infs, x_recovery, y_recovery] = load_N_datasets(path, 10);
    beta_regression(F_beta_rs_mat, connection_infs, tau, path);
    alpha_regression(x_recovery, y_recovery, path);
    return 0;
}