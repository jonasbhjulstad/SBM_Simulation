#include <DataFrame.hpp>
#include <FROLS_Eigen.hpp>
#include <Features.hpp>
#include <ERR_Regressor.hpp>
#include <algorithm>
#include <FROLS_Path_Config.hpp>
#include <Regression_Algorithm.hpp>



int main(int argc, char **argv) {
    const uint32_t Nx = 3;
    const std::string network_type = "SIR";
    const std::vector<std::string> colnames = {"S", "I", "R"};
    uint32_t N_sims = 1000; // 10000;
    uint32_t N_pop = 200;
    float p_ER = 1.0;
    using namespace FROLS;
    using namespace std::placeholders;
    uint32_t d_max = 1;
    uint32_t N_output_features = 16;
    uint32_t Nu = 1;
    std::vector<std::vector<Feature>> preselected_features(4);
    FROLS::Features::Polynomial_Model model(Nx, Nu, N_output_features, d_max);
    std::vector<uint32_t> ignore_idx = {};//;{6, 8, 9, 13};
    std::for_each(ignore_idx.begin(), ignore_idx.end(), [&](auto& ig_idx){
        model.ignore(ig_idx);
    });
    FROLS::Regression::Regressor_Param reg_param;
    reg_param.tol = 1e-4;
    reg_param.theta_tol = 1e-10;
    reg_param.N_terms_max = 4;
    FROLS::Regression::ERR_Regressor regressor(reg_param);

    std::vector<std::string> filenames(10);
    std::generate(filenames.begin(), filenames.end(), [i = 0]() mutable { return FROLS::FROLS_DATA_DIR + std::string("/SIR_Sine_Trajectory_Discrete_") + std::to_string(i++) + ".csv"; });

    auto features = regressor.transform_fit(filenames, {"S", "I", "R"}, {"p_I"}, "I", model);
    // model.feature_summary();
    return 0;
}

