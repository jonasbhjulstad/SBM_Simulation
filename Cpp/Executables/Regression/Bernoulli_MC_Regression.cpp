
#include "../Generate/Bernoulli_SIR_MC_Dynamic.hpp"
#include <quantiles.hpp>
#include <Sycl_Graph_Path_Config.hpp>
#include <Sycl_Graph.hpp>
#include <FROLS_Execution.hpp>
#include <functional>
#include <utility>
#include <chrono>
#include <fmt/format.h>

#include <Sycl_Graph_Eigen.hpp>
#include <Quantile_Regressor.hpp>
#include <ERR_Regressor.hpp>
#include <Regression_Algorithm.hpp>
#include <Polynomial_Discrete.hpp>
#include <memory>

std::string err_simulation_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/ERR_Simulation_" << network_type << "_" << N_pop << "_" << p_ER << "/trajectory_" << iter
       << ".csv";
    return ss.str();
}

std::string quantile_simulation_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Quantile_Simulation_" << network_type << "_" << N_pop << "_" << p_ER << "/trajectory_" << iter
       << ".csv";
    return ss.str();
}

static bool qr_invoked = false;
using namespace Eigen;
const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
constexpr uint32_t Nt = 50;
constexpr uint32_t N_sims = 50;
const std::string network_type = "SIR";
void simulation_loop(uint32_t N_pop, float p_ER)
{

    using namespace std::placeholders;

    auto MC_fname_f = std::bind(FROLS::MC_filename, N_pop, p_ER, _1, network_type);
    auto er_outfile_f = std::bind(err_simulation_filename, N_pop, p_ER, _1, network_type);
    auto qr_outfile_f = std::bind(quantile_simulation_filename, N_pop, p_ER, _1, network_type);
    using namespace SYCL::Graph::Regression;
    using namespace SYCL::Graph;
    using namespace Network_Models;
    MC_SIR_Params<> p;
    p.N_pop = N_pop;
    p.p_ER = p_ER;
    p.N_I_min = (N_pop*10/100);
    p.N_sim = N_sims;
    p.p_I0 = 0.2;
    p.R0_max = 1.5;
    uint32_t NV = N_pop;
    size_t nk = FROLS::n_choose_k(NV, 2);
    uint32_t NE = 1.5 * nk;
    std::cout << "Simulating for network with N_pop = " << N_pop << " and p_ER = " << p_ER << std::endl;

    // mark start timestamp
    auto start = std::chrono::high_resolution_clock::now();

    std::random_device rd{};
    std::vector<uint32_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), [&]()
                  { return rd(); });
    auto enum_seeds = enumerate(seeds);


    auto G_structure = generate_SIR_ER_graph(N_pop, p_ER, seeds[0]);
    // auto G = generate_Bernoulli_SIR_Network(G_structure, p.p_I0, seeds[1], 0.f);

    auto trajectories = MC_SIR_simulations(G_structure, p, seeds, Nt);

    std::for_each(trajectories.begin(), trajectories.end(), [&, n = 0](const auto &simdata) mutable
                  { traj_to_file(p, simdata, n++, Nt); });

    const std::vector<std::string> colnames_x = {"S", "I", "R"};
    const std::vector<std::string> colnames_u = {"p_I"};
    const std::vector<std::string> colnames_X = {"S", "I", "R", "p_I"};
    auto q_fname_f = std::bind(quantile_filename, p.N_pop, p.p_ER, _1, "SIR");
    quantiles_to_file(p.N_sim, colnames_X, MC_fname_f, q_fname_f);

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


    std::vector<std::vector<Feature>> preselected_features(4);
    FROLS::Features::Polynomial_Model er_model(Nx, Nu, d_max);
    FROLS::Features::Polynomial_Model qr_model(Nx, Nu, d_max);


    FROLS::Regression::ERR_Regressor er_regressor(er_param);




    std::vector<std::string> df_names(N_sims);
    std::generate(df_names.begin(), df_names.end(), [&, n = -1]() mutable
                    {
                    n++;
        return MC_fname_f(n); });
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

    using namespace SYCL::Graph::Features;
    auto er_features = er_regressor.transform_fit(X_list, U_list, Y_list, er_model);

    er_model.feature_summary(er_features);

    std::vector<FROLS::Regression::Quantile_Regressor> qr_regressors;
    float MAE_tol = 1e-6;
    float tau = .95;
    FROLS::Regression::Quantile_Param qr_param;
    qr_param.N_terms_max = N_terms_max;
    qr_param.tol = MAE_tol;
    qr_param.tau = 1-tau;
    qr_param.theta_tol = 1e-3;
    qr_param.N_rows = X_list[0].rows();
    //Regressor for S
    qr_regressors.push_back(FROLS::Regression::Quantile_Regressor(qr_param));
    //Regressor for I
    qr_param.tau = tau;
    qr_regressors.push_back(FROLS::Regression::Quantile_Regressor(qr_param));
    //Regressor for R
    qr_param.tau = tau;
    qr_regressors.push_back(FROLS::Regression::Quantile_Regressor(qr_param));
    // Quantile-Regression
    fmt::print("Quantile-Regression: d_max = {}, N_output_features = {}, tolerance = {}, max terms = {}\n", d_max, qr_model.N_output_features, qr_param.tol, qr_param.N_terms_max);
    std::vector<std::vector<Vec>> Y_sep(3);
    std::vector<std::vector<Feature>> qr_features(3);
    qr_invoked = true;
    for (int i = 0; i < Y_sep.size();i++)
    {
        Y_sep[i].reserve(N_sims);
        for (int j = 0; j < N_sims; j++)
        {
            Y_sep[i].push_back(Y_list[j].col(i));
        }
        qr_features[i] = qr_regressors[i].transform_fit(X_list, U_list, Y_sep[i], qr_model);
    }
    // auto qr_features = qr_regressor.transform_fit(X_list, U_list, Y_list, qr_model);
    qr_model.feature_summary(qr_features);
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
    std::string p_ER_str = fmt::format("{:.1f}", p.p_ER);
    er_model.write_csv(path_dirname(er_outfile_f(0).c_str()) + std::string("/param.csv"), er_features);
    er_model.write_latex(er_features, FROLS_DATA_DIR+ std::string("/latex/er_param_") + std::to_string(N_pop) + "_" + p_ER_str + ".tex", latex_colnames_x, latex_colnames_u, latex_colnames_y);

    qr_model.write_csv(path_dirname(qr_outfile_f(0).c_str()) + std::string("/param.csv"), qr_features);
    qr_model.write_latex(qr_features, FROLS_DATA_DIR+ std::string("/latex/qr_param_") + std::to_string(N_pop) + "_" + p_ER_str + ".tex", latex_colnames_x, latex_colnames_u, latex_colnames_y);
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    std::ofstream er_file(FROLS_DATA_DIR+ std::string("/latex/ERR_") + std::to_string(N_pop) + "_" + p_ER_str + ".tex");
    std::ofstream qr_file(FROLS_DATA_DIR+ std::string("/latex/MAE_") + std::to_string(N_pop) + "_" + p_ER_str + ".tex");
    for (int i = 0; i < 3; i++)
    {
        er_file << fmt::format("{:.2e}", er_features[i].back().f_ERR) << ",";
        qr_file << fmt::format("{:.2e}", qr_features[i].back().f_ERR) << ",";
    }

}

int main(int argc, char **argv)
{
    // auto N_pop_vec = FROLS::arange((uint32_t)10, (uint32_t)100, (uint32_t)10);
    std::vector<float> N_pop_vec = {20};
    std::vector<float> p_ER_vec = {0.1, 0.5,1.0};
    std::reverse(N_pop_vec.begin(), N_pop_vec.end());
    std::reverse(p_ER_vec.begin(), p_ER_vec.end());
    for (const float &p_ER : p_ER_vec)
    {
        for (const uint32_t N_pop : N_pop_vec)
        {
            simulation_loop(N_pop, p_ER);
        }
    }

    return 0;
}
