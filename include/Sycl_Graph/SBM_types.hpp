#ifndef SYCL_GRAPH_SBM_TYPES_HPP
#define SYCL_GRAPH_SBM_TYPES_HPP
#include <tuple>
#include <CL/sycl.hpp>
#include <vector>
namespace Sycl_Graph::SBM
{
      typedef std::tuple<sycl::buffer<uint32_t, 1>, sycl::buffer<uint32_t, 1>,
                     std::vector<uint32_t>, std::vector<uint32_t>, sycl::event>
      Iteration_Buffers_t;

}

#endif