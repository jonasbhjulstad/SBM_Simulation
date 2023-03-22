
#include "../Generate/Bernoulli_SIR_MC_Dynamic.hpp"
#include <quantiles.hpp>
#include <FROLS_Path_Config.hpp>
#include <FROLS_Graph.hpp>
#include <FROLS_Execution.hpp>
#include <functional>
#include <utility>
#include <chrono>
#include <fmt/format.h>

#include <FROLS_Eigen.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <Regression_Algorithm.hpp>
#include <Polynomial_Discrete.hpp>
#include <memory>

std::string simulation_filename(uint32_t iter)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/SIR_Sine_Trajectory_Discrete_" << iter
       << ".csv";
    return ss.str();
}


static bool qr_invoked = false;
using namespace Eigen;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
constexpr uint32_t Nt = 50;
constexpr uint32_t N_sims = 50;
const std::string network_type = "SIR";
int main(int argc, char **argv)
{
    using namespace std::placeholders;

    using namespace FROLS::Regression;
    using namespace FROLS;
    using namespace Network_Models;
    // mark start timestamp
    auto start = std::chrono::high_resolution_clock::now();


    // auto G = generate_Bernoulli_SIR_Network(G_structure, p.p_I0, seeds[1], 0.f);

    const std::vector<std::string> colnames_x = {"S", "I", "R"};
    const std::vector<std::string> colnames_u = {"p_I"};
    const std::vector<std::string> colnames_X = {"S", "I", "R", "p_I"};

    // mark end timestamp
    auto end = std::chrono::high_resolution_clock::now();
    // print time
    uint32_t N_terms_max = 2;
    uint32_t d_max = 1;
    uint32_t N_output_features = 80;
    uint32_t Nu = 1;
    uint32_t Nx = 3;
    FROLS::Regression::Regressor_Param er_param;
    er_param.tol = 1e-6;
    er_param.theta_tol = 1e-3;
    er_param.N_terms_max = N_terms_max;
    // ERR-Regression

    FROLS::Features::Polynomial_Model er_model(Nx, Nu, d_max);
    FROLS::Features::Polynomial_Model qr_model(Nx, Nu, d_max);


    FROLS::Regression::ERR_Regressor er_regressor(er_param);




    std::vector<std::string> df_names(N_sims);
    std::generate(df_names.begin(), df_names.end(), [&, n = -1]() mutable
                    {
                    n++;
        return simulation_filename(n); });
    uint32_t Nt_max = 0;
    DataFrameStack dfs(df_names);
    fmt::print("ERR-Regression: d_max = {}, N_output_features = {}, tolerance = {}, max terms = {}\n", d_max, er_model.N_output_features, er_param.tol, er_param.N_terms_max);

    std::vector<Mat> X_list(N_sims);
    std::vector<Mat> U_list(N_sims);
    std::vector<Mat> Y_list(N_sims);
    for (int i = 0; i < N_sims; i++)
    {
        X_list[i] = dataframe_to_matrix(dfs.dataframes[i], colnames_x, 0, -2);
        U_list[i] = dataframe_to_matrix(dfs.dataframes[i], colnames_u, 0, -2);
        Y_list[i] = dataframe_to_matrix(dfs.dataframes[i], colnames_x, 1, -1) - X_list[i];
    }

    using namespace FROLS::Features;
    auto er_features = er_regressor.transform_fit(X_list, U_list, Y_list, er_model);

    er_model.feature_summary(er_features);

    FROLS::Regression::Quantile_Param qr_param;
    float MAE_tol = 1e-6;
    float tau = .8;
    qr_param.N_terms_max = N_terms_max;
    qr_param.tol = MAE_tol;
    qr_param.tau = 1-tau;
    qr_param.theta_tol = 1e-3;
    qr_param.N_rows = X_list[0].rows();
    FROLS::Regression::Quantile_Regressor qr_regressor(qr_param);
    //Regressor for S
    // Quantile-Regression
    fmt::print("Quantile-Regression: d_max = {}, N_output_features = {}, tolerance = {}, max terms = {}\n", d_max, qr_model.N_output_features, qr_param.tol, qr_param.N_terms_max);
    qr_invoked = true;
    // auto qr_features = qr_regressors[0].transform_fit(X_list, U_list, Y_list, qr_model);
    

    auto qr_features = qr_regressor.transform_fit(X_list, U_list, Y_list, qr_model);
    qr_model.feature_summary(qr_features);

    auto er_outfile_f = [&](int i)
    {
         std::stringstream ss;
        ss << FROLS::FROLS_DATA_DIR << "/ERR_Sine_" << i
       << ".csv";
        return ss.str();
    };

        auto qr_outfile_f = [&](int i)
    {
         std::stringstream ss;
        ss << FROLS::FROLS_DATA_DIR << "/QR_Sine_" << i
       << ".csv";
        return ss.str();
    };

    for (int i = 0; i < N_sims; i++)
    {
        Mat u = vecs_to_mat(dfs[i][colnames_u])(Eigen::seq(0, Eigen::last - 1), Eigen::all);
        Vec x0 = vecs_to_mat(dfs[i][colnames_x]).row(0);
        Mat X_sim = er_model.simulate(x0, u, u.rows(), er_features);
        {
        FROLS::DataFrame df;
        df.assign(colnames_u, u);
        df.assign("t", FROLS::range(0, u.rows()));
        df.assign(colnames_x, X_sim);
        df.resize(u.rows());
        df.write_csv(er_outfile_f(i), ",", 1e-6);
        }
        u = vecs_to_mat(dfs[i][colnames_u])(Eigen::seq(0, Eigen::last - 1), Eigen::all);
        x0 = vecs_to_mat(dfs[i][colnames_x]).row(0);
        {
        FROLS::DataFrame df;
        Mat X_sim = qr_model.simulate(x0, u, u.rows(), qr_features);
        df.assign(colnames_u, u);
        df.assign("t", FROLS::range(0, u.rows()));
        df.assign(colnames_x, X_sim);
        df.resize(u.rows());
        df.write_csv(qr_outfile_f(i), ",", 1e-6);
        }
    }



    std::vector<std::string> latex_colnames_x = {"S_t", "I_t", "R_t"};
    std::vector<std::string> latex_colnames_u = {"p_I"};
    std::vector<std::string> latex_colnames_y = {"S_{t+1}", "I_{t+1}", "R_{t+1}"};
    //convert p_ER to string with 1 decimal place
    er_model.write_csv(path_dirname(er_outfile_f(0).c_str()) + std::string("/param.csv"), er_features);
    er_model.write_latex(er_features, FROLS_DATA_DIR+ std::string("/latex/er_param_Sine.tex"), latex_colnames_x, latex_colnames_u, latex_colnames_y);

    qr_model.write_csv(path_dirname(qr_outfile_f(0).c_str()) + std::string("/param.csv"), qr_features);
    qr_model.write_latex(qr_features, FROLS_DATA_DIR+ std::string("/latex/qr_param_Sine.tex"), latex_colnames_x, latex_colnames_u, latex_colnames_y);
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    std::ofstream er_file(FROLS_DATA_DIR+ std::string("/latex/ERR_Sine.tex"));
    std::ofstream qr_file(FROLS_DATA_DIR+ std::string("/latex/MAE_Sine.tex"));
    for (int i = 0; i < 3; i++)
    {
        er_file << fmt::format("{:.2e}", er_features[i].back().f_ERR) << ",";
        qr_file << fmt::format("{:.2e}", qr_features[i].back().f_ERR) << ",";
    }

    return 0;
}
