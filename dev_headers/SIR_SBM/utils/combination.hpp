#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <string>
#end

namespace SIR_SBM {

uint32_t n_choose_k(uint32_t n, uint32_t k) {
  if (k > n) {
    return 0;
  }
  if (k * 2 > n) {
    k = n - k;
  }
  if (k == 0) {
    return 1;
  }

  uint32_t result = n;
  for (uint32_t i = 2; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}



} // namespace SIR_SBM