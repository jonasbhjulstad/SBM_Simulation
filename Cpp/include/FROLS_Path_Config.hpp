#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP

#include <string>

namespace FROLS {
// const char *FROLS_ROOT_DIR = "";
// const char *FROLS_INCLUDE_DIR = "";
    extern const char *FROLS_DATA_DIR;
    extern const char *FROLS_LOG_DIR;

    std::string MC_filename(uint16_t N_pop, float p_ER, uint16_t iter, std::string network_type);

    std::string quantile_filename(uint16_t N_pop, float p_ER, uint16_t iter, std::string network_type);
} //FROLS
#endif
