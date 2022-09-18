#ifndef FROLS_PATH_CONFIG_HPP
#define FROLS_PATH_CONFIG_HPP
#include <string>
namespace FROLS
{
// const char *FROLS_ROOT_DIR = "";
// const char *FROLS_INCLUDE_DIR = "";
extern const char *FROLS_DATA_DIR;
extern const char *FROLS_LOG_DIR;
std::string MC_sim_filename(size_t N_pop, double p_ER, size_t idx, const std::string network_type = "SIR");

std::string quantile_filename(size_t N_pop, double p_ER, double tau);
}
#endif
