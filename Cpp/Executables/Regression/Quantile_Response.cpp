#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <Quantile_Regressor.hpp>
#include <algorithm>
#include <FROLS_Path_Config.hpp>
#include <Regression_Algorithm.hpp>

std::string quantile_simulation_filename(size_t N_pop, double p_ER, size_t iter, std::string network_type) {
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Quantile_Simulation_" << network_type << "_" <<  N_pop << "_" << p_ER << "_" << iter
       << ".csv";
    return ss.str();
}


int main(int argc, char **argv) {
    const size_t Nx = 3;
    const std::string network_type = "SIR";
    const std::vector<std::string> colnames = {"S", "I", "R"};
    size_t N_sims = 500; // 10000;
    size_t N_pop = 80;
    double p_ER = 1.0;
    using namespace FROLS;
    using namespace std::placeholders;
    size_t d_max = 1;
    size_t N_output_features = 16;
    size_t Nu = 1;
    auto MC_fname_f = std::bind(MC_filename, N_pop, p_ER, _1, network_type);
    auto outfile_f = std::bind(quantile_simulation_filename, N_pop, p_ER, _1, network_type);
    std::vector<size_t> ignore_idx = {};//{0, 1, 2, 3, 4, 5, 6, 7, 8};//{0, 1, 2, 3,4};
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max, ignore_idx);
    double theta_tol = 1e-6;
    size_t N_terms_max = 2;
    double MAE_tol = 1e-3;
    FROLS::Regression::Quantile_Param reg_param;
    reg_param.N_terms_max = 2;
    reg_param.tol = MAE_tol;
    FROLS::Regression::Quantile_Regressor regressor(reg_param);

    from_file_regression(MC_fname_f, {"S", "I", "R"}, {"p_I"}, N_sims, regressor, model, outfile_f);

    return 0;
}