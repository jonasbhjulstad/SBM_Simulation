#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP

#include <string>
#include <sstream>

namespace FROLS {
// const char *FROLS_ROOT_DIR = "";
// const char *FROLS_INCLUDE_DIR = "";
    const char *FROLS_DATA_DIR = "/home/deb/Downloads/SBM_Simulation/Cpp/data";
    const char *FROLS_LOG_DIR = "/home/deb/Downloads/SBM_Simulation/Cpp/log";


    std::string MC_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type) {
        std::stringstream ss;
        ss << FROLS_DATA_DIR << "/Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "/" << iter
           << ".csv";
        return ss.str();
    }
    std::string quantile_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type) {
        std::stringstream ss;
        ss << FROLS_DATA_DIR << "/Quantile_Bernoulli_" << network_type << "_MC_" << N_pop << "_" << p_ER << "/" << iter
           << ".csv";
        return ss.str();
    }
    std::string path_dirname(const std::string &fname)
    {
        size_t pos = fname.find_last_of("\\/");
        return (std::string::npos == pos)
            ? ""
            : fname.substr(0, pos);
    }


} //FROLS
#endif
