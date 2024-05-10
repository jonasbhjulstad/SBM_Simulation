#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <string>
#end

namespace SIR_SBM {

size_t n_choose_k(size_t n, size_t k) {
  if (k > n) {
    return 0;
  }
  if (k * 2 > n) {
    k = n - k;
  }
  if (k == 0) {
    return 1;
  }

  size_t result = n;
  for (size_t i = 2; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}



} // namespace SIR_SBM