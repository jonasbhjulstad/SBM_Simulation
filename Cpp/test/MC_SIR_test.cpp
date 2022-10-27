#include <Bernoulli_SIR_MC_Dynamic.hpp>
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
int main()
{
    using namespace FROLS;
    using namespace Network_Models;
    // std::vector<MC_SIR_VectorData> MC_SIR_simulations(uint32_t N_pop, float p_ER, float p_I0, const std::vector<uint32_t> &seeds, uint32_t Nt, uint32_t N_sims)
    //create test for MC_SIR_simulations
        // std::vector<MC_SIR_VectorData> MC_SIR_simulations(uint32_t N_pop, float p_ER, float p_I0, const std::vector<uint32_t> &seeds, uint32_t Nt, uint32_t N_sims)
    MC_SIR_Params<> p;
    p.N_sim = 20;
    //generate p.N_sim random seeds
    std::vector<uint32_t> seeds(p.N_sim);
    std::generate(seeds.begin(), seeds.end(), std::rand);
    uint32_t Nt = 20;
    auto G_structure = generate_SIR_ER_graph(100, 0.1, seeds[0]);
    auto G_temp = generate_Bernoulli_SIR_Network(G_structure, 0.1, seeds[0], 0.f);


    auto rd = MC_SIR_simulations_to_regression(G_structure, p, seeds, Nt);
    std::cout << rd.X.topRows(5) << std::endl;
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

    FROLS::Regression::ERR_Regressor er_regressor(er_param);

    er_regressor.transform_fit(rd.X, rd.U, rd.Y.col(0), er_model);

}