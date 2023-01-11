#ifndef SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#define SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#include <stdint.h>
#include <array>
namespace Sycl_Graph::Network_Models {
struct SIR_Metapopulation_State
{
  SIR_Metapopulation_State() = default;
  SIR_Metapopulation_State(float S): S(S), I(0), R(0) {}
  SIR_Metapopulation_State(float S, float I, float R): S(S), I(I), R(R) {}
  float S;
  float I;
  float R;
};
struct SIR_Metapopulation_Param
{
  float beta = 0;
  float alpha = 0;
};

struct SIR_Metapopulation_Node_Param
{
    float E_I0 = 0.1;
    float std_I0 = 0.01;
    float E_R0 = 0.05;
    float std_R0 = 0.01;
    float alpha = 0.05;
    float beta = 0.01;
};


template <typename dType = float>  
struct SIR_Metapopulation_Temporal_Param
{
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
};


} // namespace Network_Models
#endif