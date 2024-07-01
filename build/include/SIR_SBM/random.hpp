// random.hpp
//

#ifndef LZZ_SIR_SBM_LZZ_random_hpp
#define LZZ_SIR_SBM_LZZ_random_hpp
#line 3 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
#include <SIR_SBM/common.hpp>
#include <oneapi/dpl/random>
#include <random>
#include <sycl.hpp>
#define LZZ_INLINE inline
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
namespace SIR_SBM
{
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  template <typename RNG>
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  std::vector <RNG> generate_rngs (uint32_t seed, size_t N);
}
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
namespace SIR_SBM
{
#line 24 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  std::vector <uint32_t> repeat_N_indices (std::vector <uint32_t> const weights);
}
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
namespace SIR_SBM
{
#line 40 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  std::vector <uint32_t> discrete_finite_sample (std::mt19937_64 & rng, std::vector <uint32_t> const & weights, size_t N_samples);
}
#line 11 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
namespace SIR_SBM
{
#line 12 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  template <typename RNG>
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
  std::vector <RNG> generate_rngs (uint32_t seed, size_t N)
#line 13 "/home/deb/Documents/SBM_Simulation/dev_headers/SIR_SBM//random.hpp"
                                                        {
  RNG seeder(seed);
  std::vector<RNG> rngs(N);
  std::generate_n(rngs.begin(), N, [&seeder]() {
    uint32_t seed = seeder();
    return RNG(seed);
  });
  return rngs;
}
}
#undef LZZ_INLINE
#endif
