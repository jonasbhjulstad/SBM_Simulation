#ifndef SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#define SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#include <stdint.h>
#include <array>
namespace Sycl_Graph::Network_Models {
struct SIR_Metapopulation_State
{
  explicit SIR_Metapopulation_State(float S): S(S), I(0), R(0) {}
  float S;
  float I;
  float R;
};
struct SIR_Metapopulation_Param
{
  float beta = 0;
  float alpha = 0;
};
template <typename dType = float>  {
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
};


} // namespace Network_Models
#endif