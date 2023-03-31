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

  struct SIR_SBM_Param_t
  {
    std::vector<std::vector<float>> p_I;
    float p_R;
    float p_I0;
    float p_R0;
  };

  struct Edge_t
  {
    uint32_t from = std::numeric_limits<uint32_t>::max();
    uint32_t to = std::numeric_limits<uint32_t>::max();
  };
  typedef std::vector<Edge_t> Edge_List_t;
  
  typedef std::vector<uint32_t> Node_List_t;



}

#endif