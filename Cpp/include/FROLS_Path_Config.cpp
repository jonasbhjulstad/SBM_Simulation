#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP
#include <string>
namespace FROLS
{
// const char *FROLS_ROOT_DIR = "";
// const char *FROLS_INCLUDE_DIR = "";
const char *FROLS_DATA_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/data";
const char *FROLS_LOG_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/log";

std::string MC_sim_filename(size_t N_pop, double p_ER, size_t idx)
{
    return FROLS_DATA_DIR + std::string("/Bernoulli_SIR_MC_") +
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
