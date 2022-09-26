#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP

#include <string>
#include <sstream>

namespace FROLS {
// const char *FROLS_ROOT_DIR = "";
// const char *FROLS_INCLUDE_DIR = "";
    const char *FROLS_DATA_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/data";
    const char *FROLS_LOG_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/log";


    std::string MC_filename(size_t N_pop, double p_ER, size_t iter, std::string network_type) {
        std::stringstream ss;
        ss << FROLS_DATA_DIR << "/Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "_" << iter
           << ".csv";
        return ss.str();
    }
    std::string quantile_filename(size_t N_pop, double p_ER, size_t iter, std::string network_type) {
        std::stringstream ss;
        ss << FROLS_DATA_DIR << "/Quantile_Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "_" << iter
           << ".csv";
        return ss.str();
    }


} //FROLS
#endif
