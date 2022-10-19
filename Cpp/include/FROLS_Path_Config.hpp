#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP

#include <string>

namespace FROLS
{
    // const char *FROLS_ROOT_DIR = "";
    // const char *FROLS_INCLUDE_DIR = "";
    extern const char *FROLS_DATA_DIR;
    extern const char *FROLS_LOG_DIR;

    std::string MC_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type);

    std::string quantile_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type);
    std::string path_dirname(const std::string &fname);
} // FROLS
#endif
