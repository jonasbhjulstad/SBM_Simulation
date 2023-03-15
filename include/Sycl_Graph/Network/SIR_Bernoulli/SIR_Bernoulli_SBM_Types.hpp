#ifndef SYCL_GRAPH_SIR_BERNOULLI_TYPES_HPP
#define SYCL_GRAPH_SIR_BERNOULLI_TYPES_HPP
#include <stdint.h>
#include <CL/sycl.hpp>
#include <numeric>
#include <Sycl_Graph/Math/math.hpp>
namespace Sycl_Graph::Network_Models {
enum SIR_Individual_State: int { SIR_INDIVIDUAL_S = 0, SIR_INDIVIDUAL_I = 1, SIR_INDIVIDUAL_R = 2 };
struct SIR_Edge {};

template <typename dType = float> struct SIR_Bernoulli_SBM_Temporal_Param {
  SIR_Bernoulli_SBM_Temporal_Param<>(){};
  SIR_Bernoulli_SBM_Temporal_Param<>(const std::vector<dType>& p_Is): p_Is(p_Is){}

  std::array<dType, N_p_Is> p_Is;
  
  dType p_R = 0.01;
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
};
} // namespace Network_Models
#endif