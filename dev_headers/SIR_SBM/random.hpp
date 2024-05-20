#pragma once
#hdr
#include <SIR_SBM/common.hpp>
#include <oneapi/dpl/random>
#include <random>
#include <sycl.hpp>
#end
#src
#include <numeric>
#end
namespace SIR_SBM {
template <typename RNG>
std::vector<RNG> generate_rngs(uint32_t seed, size_t N) {
  RNG seeder(seed);
  std::vector<RNG> rngs(N);
  std::generate_n(rngs.begin(), N, [&seeder]() {
    uint32_t seed = seeder();
    return RNG(seed);
  });
  return rngs;
}


std::vector<uint32_t> repeat_N_indices(const std::vector<uint32_t> weights)
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

std::vector<uint32_t> discrete_finite_sample(std::mt19937_64& rng, const std::vector<uint32_t>& weights, size_t N_samples) {
  auto indices = repeat_N_indices(weights);
  std::vector<uint32_t> result(N_samples);
  std::sample(indices.begin(), indices.end(), result.begin(), N_samples, rng);
  return result;
}


} // namespace SIR_SBM
