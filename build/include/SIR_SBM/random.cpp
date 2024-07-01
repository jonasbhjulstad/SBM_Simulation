// random.cpp
//

#include "random.hpp"
#line 9 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
#include <numeric>
#define LZZ_INLINE inline
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
namespace SIR_SBM
{
#line 24 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  std::vector <uint32_t> repeat_N_indices (std::vector <uint32_t> const weights)
#line 25 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
{
  auto N_indices = std::accumulate(weights.begin(), weights.end(), 0);
  std::vector<uint32_t> indices(N_indices);
  uint32_t idx = 0;
  for (size_t i = 0; i < weights.size(); i++)
  {
    for (size_t j = 0; j < weights[i]; j++)
    {
      indices[idx] = i;
      idx++;
    }
  }
  return indices;
}
}
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
namespace SIR_SBM
{
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  std::vector <uint32_t> discrete_finite_sample (std::mt19937_64 & rng, std::vector <uint32_t> const & weights, size_t N_samples)
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
                                                                                                                           {
  auto indices = repeat_N_indices(weights);
  std::vector<uint32_t> result(N_samples);
  std::sample(indices.begin(), indices.end(), result.begin(), N_samples, rng);
  return result;
}
}
#undef LZZ_INLINE
