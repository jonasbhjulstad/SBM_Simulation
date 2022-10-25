
#include "../Generate/Bernoulli_SIR_MC.hpp"
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

constexpr uint32_t Nt = 50;
constexpr uint32_t N_sims = 200;
const std::string network_type = "SIR";
void simulation_loop(uint32_t N_pop, float p_ER)
{

    using namespace std::placeholders;

    auto MC_fname_f = std::bind(FROLS::MC_filename, N_pop, p_ER, _1, network_type);
    auto er_outfile_f = std::bind(err_simulation_filename, N_pop, p_ER, _1, network_type);
    auto qr_outfile_f = std::bind(quantile_simulation_filename, N_pop, p_ER, _1, network_type);

    using namespace FROLS;
    using namespace Network_Models;
    MC_SIR_Params<> p;
    p.N_pop = N_pop;
    p.p_ER = p_ER;
    p.N_sim = N_sims;
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

    std::mt19937_64 rng(rd());
    typedef Network_Models::SIR_Bernoulli_Network<SIR_VectorGraph, decltype(rng), Nt> SIR_Bernoulli_Network;
    std::vector<std::shared_ptr<std::mutex>> v_mx(NV + 1);
    // create mutexes
    for (auto &mx : v_mx)
    {
        mx = std::make_unique<std::mutex>();
    }
    std::vector<std::shared_ptr<std::mutex>> e_mx(NE + 1);
    // create mutexes
    for (auto &mx : e_mx)
    {
        mx = std::make_unique<std::mutex>();
    }

    SIR_VectorGraph G(v_mx, e_mx);
    generate_erdos_renyi<SIR_VectorGraph, decltype(rng)>(G, p.N_pop, p.p_ER, SIR_S, rng);
    std::vector<MC_SIR_SimData<Nt>> simdatas(p.N_sim);
    std::transform(enum_seeds.begin(), enum_seeds.end(), simdatas.begin(), [&](auto &es)
                   {
        uint32_t iter = es.first;
        uint32_t seed = es.second;
        if ((iter % (p.N_sim / 10)) == 0)
        {
            // std::cout << "Simulation " << iter << " of " << p.N_sim << std::endl;
        }
        return MC_SIR_simulation<decltype(G), Nt>(G, p, seed); });

    std::for_each(simdatas.begin(), simdatas.end(), [&, n = 0](const auto &simdata) mutable
                  { traj_to_file<Nt>(p, simdata, n++); });

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
    er_param.tol = 1e-4;
    er_param.theta_tol = 1e-3;
    er_param.N_terms_max = N_terms_max;
    // ERR-Regression


    std::vector<std::vector<Feature>> preselected_features(4);
    FROLS::Features::Polynomial_Model er_model(Nx, Nu, N_output_features, d_max);
    FROLS::Features::Polynomial_Model qr_model(Nx, Nu, N_output_features, d_max);

    FROLS::Regression::ERR_Regressor er_regressor(er_param);

    float MAE_tol = 1e-6;
    FROLS::Regression::Quantile_Param qr_param;
    qr_param.N_terms_max = N_terms_max;
    qr_param.tol = MAE_tol;
    qr_param.tau = 0.95;
    qr_param.theta_tol = 1e-3;

    // Quantile-Regression



    FROLS::Regression::Quantile_Regressor qr_regressor(qr_param);

    std::vector<std::string> df_names(N_sims);
    std::generate(df_names.begin(), df_names.end(), [&, n = -1]() mutable
                    {
                    n++;
        return MC_fname_f(n); });
    uint32_t Nt_max = 0;
    DataFrameStack dfs(df_names);
    fmt::print("ERR-Regression: d_max = {}, N_output_features = {}, tolerance = {}, max terms = {}\n", d_max, N_output_features, er_param.tol, er_param.N_terms_max);

    Mat X, Y, U;

    X = dataframe_to_matrix(dfs, colnames_x,
                            0, -2);
    Y = dataframe_to_matrix(dfs, colnames_x, 1, -1);
    U = dataframe_to_matrix(dfs, colnames_u, 0, -2);

    using namespace FROLS::Features;

    er_regressor.transform_fit(X, U, Y, er_model);
    er_model.feature_summary();

    fmt::print("Quantile-Regression: d_max = {}, N_output_features = {}, tolerance = {}, max terms = {}\n", d_max, N_output_features, qr_param.tol, qr_param.N_terms_max);

    X = dataframe_to_matrix(dfs, colnames_x,
                            0, -2);
    Y = dataframe_to_matrix(dfs, colnames_x, 1, -1);
    U = dataframe_to_matrix(dfs, colnames_u, 0, -2);
    qr_regressor.transform_fit(X, U, Y, qr_model);
    qr_model.feature_summary();
    for (int i = 0; i < N_sims; i++)
    {
        Mat u = vecs_to_mat(dfs[i][colnames_u])(Eigen::seq(0, Eigen::last - 1), Eigen::all);
        Vec x0 = vecs_to_mat(dfs[i][colnames_x]).row(0);
        Mat X_sim = er_model.simulate(x0, u, u.rows());
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
        Mat X_sim = qr_model.simulate(x0, u, u.rows());
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
    er_model.write_csv(path_dirname(er_outfile_f(0).c_str()) + std::string("/param.csv"));
    er_model.write_latex(FROLS_DATA_DIR+ std::string("/latex/er_param_") + std::to_string(N_pop) + "_" + p_ER_str + ".tex", latex_colnames_x, latex_colnames_u, latex_colnames_y);
    
    qr_model.write_csv(path_dirname(qr_outfile_f(0).c_str()) + std::string("/param.csv"));
    qr_model.write_latex(FROLS_DATA_DIR+ std::string("/latex/qr_param_") + std::to_string(N_pop) + "_" + p_ER_str + ".tex", latex_colnames_x, latex_colnames_u, latex_colnames_y);
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

}

int main(int argc, char **argv)
{
    // auto N_pop_vec = FROLS::arange((uint32_t)10, (uint32_t)100, (uint32_t)10);
    auto N_pop_vec = {20, 100};
    std::vector<float> p_ER_vec = {0.1, 1.0};
    // std::reverse(N_pop_vec.begin(), N_pop_vec.end());
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
