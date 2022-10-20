#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <ERR_Regressor.hpp>
#include <algorithm>
#include <FROLS_Path_Config.hpp>
#include <Regression_Algorithm.hpp>
#include <FROLS_Eigen.hpp>

std::string err_simulation_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Quantile_ERR_Simulation_" << network_type << "_" << N_pop << "_" << p_ER << "/" << iter
       << ".csv";
    return ss.str();
}
std::string Quantile_MC_filename(uint32_t N_pop, float p_ER, uint32_t tau, std::string network_type, uint32_t iter)
{
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Quantile_Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "/" << tau
       << ".csv";
    return ss.str();
}

int main(int argc, char **argv)
{
    const uint32_t Nx = 3;
    const std::string network_type = "SIR";
    const std::vector<std::string> colnames = {"S", "I", "R"};
    uint32_t N_pop = 50;
    float p_ER = 1.0;
    size_t N_sims = 1000;
    using namespace FROLS;
    using namespace std::placeholders;
    uint32_t d_max = 1;
    uint32_t N_output_features = 16;
    uint32_t Nu = 1;
    uint32_t tau = 95;
    auto Quantile_fname_f = std::bind(Quantile_MC_filename, N_pop, p_ER, tau, network_type, _1);
    auto outfile_f = std::bind(err_simulation_filename, N_pop, p_ER, _1, network_type);
    std::vector<std::vector<Feature>> preselected_features(4);
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
    // model.preselect("x0", 1.0, 0, FEATURE_PRESELECTED_IGNORE);
    // model.preselect("x1", 1.0, 1, FEATURE_PRESELECTED_IGNORE);
    // model.preselect("x2", 1.0, 2, FEATURE_PRESELECTED_IGNORE);
    std::vector<uint32_t> ignore_idx = {}; //;{6, 8, 9, 13};
    std::for_each(ignore_idx.begin(), ignore_idx.end(), [&](auto &ig_idx)
                  { model.ignore(ig_idx); });
    FROLS::Regression::Regressor_Param reg_param;
    reg_param.tol = 1e-4;
    reg_param.theta_tol = 1e-10;
    reg_param.N_terms_max = 4;
    FROLS::Regression::ERR_Regressor regressor(reg_param);

    Regression::from_file_regression(Quantile_fname_f, {"S", "I", "R"}, {"p_I"}, 1, regressor, model, outfile_f, true);
    auto fnames = model.feature_names();
    std::for_each(fnames.begin(), fnames.end(), [n = 0](auto &name) mutable
                  {
        std::cout << n << ": " << name << std::endl;
        n++; });

    std::vector<std::string> colnames_u = {"p_I"};
    std::vector<std::string> colnames_x = {"S", "I", "R"};
    auto MC_fname_f = std::bind(MC_filename, N_pop, p_ER, _1, network_type);

    std::vector<std::string> df_names(N_sims);
    std::generate(df_names.begin(), df_names.end(), [&, n = -1]() mutable
                    {
                    n++;
        return MC_fname_f(n); });
    DataFrameStack dfs(df_names);

    for (int i = 0; i < N_sims; i++)
    {
        Mat u = vecs_to_mat(dfs[i][colnames_u])(Eigen::seq(0, Eigen::last - 1), Eigen::all);
        Vec x0 = vecs_to_mat(dfs[i][colnames_x]).row(0);
        Mat X_sim = model.simulate(x0, u, u.rows());
        FROLS::DataFrame df;
        df.assign({colnames_u}, u);
        df.assign("t", FROLS::range(0, u.rows()));
        df.assign(colnames_x, X_sim);
        df.resize(u.rows());
        df.write_csv(outfile_f(i), ",", 1e-6);
    }


    const char *out_dir = basename(outfile_f(0).c_str());
    model.write_csv(out_dir + std::string("/Quantile_ERR_param.csv"));
    model.read_csv(out_dir + std::string("/Quantile_ERR_param.csv"));
    model.feature_summary();

    return 0;
}
