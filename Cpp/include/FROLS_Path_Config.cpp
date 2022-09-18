#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP
#include <string>
namespace FROLS
{
// const char *FROLS_ROOT_DIR = "";
// const char *FROLS_INCLUDE_DIR = "";
const char *FROLS_DATA_DIR = "C:/Users/jonas/Documents/Network_Robust_MPC/Cpp/data";
const char *FROLS_LOG_DIR = "C:/Users/jonas/Documents/Network_Robust_MPC/Cpp/log";

std::string MC_sim_filename(size_t N_pop, double p_ER, size_t idx, const std::string network_type = "SIR")
{
    return FROLS_DATA_DIR + std::string("/Bernoulli_" + network_type + "_MC_") +
           std::to_string(idx) + "_" + std::to_string(N_pop) + "_" +
           std::to_string(p_ER) + ".csv";
}

std::string quantile_filename(size_t N_pop, double p_ER, double tau)
{
    return FROLS_DATA_DIR + std::string("/Bernoulli_SIR_MC_Quantiles_") +
           std::to_string(N_pop) + "_" + std::to_string(p_ER) + "_" +
           std::to_string(tau) + ".csv";
}
}
#endif
