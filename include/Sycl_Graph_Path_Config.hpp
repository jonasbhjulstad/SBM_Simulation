#ifndef SYCL_GRAPH_PATH_CONFIG_HPP
#define SYCL_GRAPH_PATH_CONFIG_HPP

#include <string>

namespace SYCL::Graph
{
    extern const char *SYCL_Graph_DATA_DIR;
    extern const char *SYCL_Graph_LOG_DIR;

    std::string MC_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type);

    std::string quantile_filename(uint32_t N_pop, float p_ER, uint32_t iter, std::string network_type);
    std::string path_dirname(const std::string &fname);
} // FROLS
#endif
