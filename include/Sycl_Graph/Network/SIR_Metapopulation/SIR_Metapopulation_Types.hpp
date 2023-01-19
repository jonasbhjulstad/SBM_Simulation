#ifndef SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#define SYCL_GRAPH_SIR_METAPOPULATION_TYPES_HPP
#include <stdint.h>
#include <array>
#include <ostream>
#include <limits>
namespace Sycl_Graph::Network_Models {
struct SIR_Metapopulation_State
{
  SIR_Metapopulation_State() = default;
  SIR_Metapopulation_State(float S): S(S), I(0), R(0) {}
  SIR_Metapopulation_State(float S, float I, float R): S(S), I(I), R(R) {}

  //create default operator+=
  SIR_Metapopulation_State& operator+=(const SIR_Metapopulation_State &other)
  {
    S += other.S;
    I += other.I;
    R += other.R;
    return *this;
  }

  SIR_Metapopulation_State& operator-=(const SIR_Metapopulation_State &other)
  {
    S -= other.S;
    I -= other.I;
    R -= other.R;
    return *this;
  }

  SIR_Metapopulation_State operator+(const SIR_Metapopulation_State &other) const
  {
    return SIR_Metapopulation_State(S + other.S, I + other.I, R + other.R);
  }
  float S = 0.f;
  float I = 0.f;
  float R = 0.f;
  friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_State& state)
  {
    os << state.S << " " << state.I << " " << state.R;
    return os;
  }
};
struct SIR_Metapopulation_Param
{
  float beta = 0;
  float alpha = 0;
  friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_Param& param)
  {
    os << param.beta << " " << param.alpha;
    return os;
  }
};

struct SIR_Metapopulation_Node_Param
{
    float E_I0 = 0.1;
    float std_I0 = 0.01;
    float E_R0 = 0.05;
    float std_R0 = 0.01;
    float alpha = 0.05;
    float beta = 0.01;
    friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_Node_Param& param)
    {
      os << param.E_I0 << " " << param.std_I0 << " " << param.E_R0 << " " << param.std_R0 << " " << param.alpha << " " << param.beta;
      return os;
    }
};


template <typename dType = float>  
struct SIR_Metapopulation_Temporal_Param
{
  uint32_t Nt_min = std::numeric_limits<uint32_t>::max();
  uint32_t N_I_min = 0;
  friend std::ostream& operator<<(std::ostream& os, const SIR_Metapopulation_Temporal_Param& param)
  {
    os << param.Nt_min << " " << param.N_I_min;
    return os;
  }
};


} // namespace Network_Models
#endif