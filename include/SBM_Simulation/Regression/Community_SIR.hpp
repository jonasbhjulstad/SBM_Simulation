#pragma once
#include <SBM_Simulation/Regression/Regression_Types.hpp>
namespace SBM_Regression
{
  Vec compute_beta_rs_col(uint32_t from_idx, uint32_t to_idx, const Vec &p_I);
}