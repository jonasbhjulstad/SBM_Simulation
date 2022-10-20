#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <Quantile_Regressor.hpp>
#include <algorithm>
#include <FROLS_Path_Config.hpp>
#include <Regression_Algorithm.hpp>

std::string quantile_simulation_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type) {
    std::stringstream ss;
    ss << FROLS::FROLS_DATA_DIR << "/Quantile_Simulation_" << network_type << "_" <<  N_pop << "_" << p_ER << "/" << iter
       << ".csv";
    return ss.str();
}


int main(int argc, char **argv) {
    const uint32_t Nx = 3;
    const std::string network_type = "SIR";
    const std::vector<std::string> colnames = {"S", "I", "R"};
    uint32_t N_sims = 100; // 10000;
    uint32_t N_pop = 50;
    float p_ER = 1.0;
    using namespace FROLS;
    using namespace std::placeholders;
    uint32_t d_max = 1;
    uint32_t N_output_features = 16;
    uint32_t Nu = 1;
    auto MC_fname_f = std::bind(MC_filename, N_pop, p_ER, _1, network_type);
    auto outfile_f = std::bind(quantile_simulation_filename, N_pop, p_ER, _1, network_type);
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
    // model.preselect("x0", 1.0, 0, FEATURE_PRESELECTED_IGNORE);
    // model.preselect("x1", 1.0, 1, FEATURE_PRESELECTED_IGNORE);
    // model.preselect("x2", 1.0, 2, FEATURE_PRESELECTED_IGNORE);
    // model.ignore(0);
    float theta_tol = 1e-6;
    uint32_t N_terms_max = 2;
    float MAE_tol = 1e-3;
    FROLS::Regression::Quantile_Param reg_param;
    reg_param.N_terms_max = 2;
    reg_param.tol = MAE_tol;
    FROLS::Regression::Quantile_Regressor regressor(reg_param);

    from_file_regression(MC_fname_f, {"S", "I", "R"}, {"p_I"}, N_sims, regressor, model, outfile_f);

    return 0;
}