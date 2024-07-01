// combination.cpp
//

#include "combination.hpp"
#define LZZ_INLINE inline
#line 7 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//combination.hpp"
namespace SIR_SBM
{
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//combination.hpp"
  size_t n_choose_k (size_t n, size_t k)
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//combination.hpp"
                                      {
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
}
#undef LZZ_INLINE
