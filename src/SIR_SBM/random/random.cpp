
#include <SIR_SBM/random/random.hpp>
#include <numeric>
#include <random>

namespace SIR_SBM {

std::vector<oneapi::dpl::ranlux48> generate_rngs_dpl(uint32_t seed, uint32_t N) {
  std::mt19937 seeder(seed);
  std::vector<oneapi::dpl::ranlux48> rngs(N);
  std::generate_n(rngs.begin(), N, [&seeder]() {
    uint32_t seed = seeder();
    return oneapi::dpl::ranlux48(seed);
  });
  return rngs;
}


std::vector<std::mt19937> generate_rngs(uint32_t seed, uint32_t N) {
  std::mt19937 seeder(seed);
  std::vector<std::mt19937> rngs(N);
  std::generate_n(rngs.begin(), N, [&seeder]() {
    uint32_t seed = seeder();
    return std::mt19937(seed);
  });
  return rngs;
}

std::vector<uint32_t> repeat_N_indices(const std::vector<uint32_t> weights) {
  auto N_indices = std::accumulate(weights.begin(), weights.end(), 0);
  std::vector<uint32_t> indices(N_indices);
  uint32_t idx = 0;
  for (uint32_t i = 0; i < weights.size(); i++) {
    for (uint32_t j = 0; j < weights[i]; j++) {
      indices[idx] = i;
      idx++;
    }
  }
  return indices;
}

std::vector<uint32_t>
discrete_finite_sample(std::mt19937 &rng,
                       const std::vector<uint32_t> &weights,
                       uint32_t N_samples) {
  auto indices = repeat_N_indices(weights);
  std::vector<uint32_t> result(N_samples);
  std::sample(indices.begin(), indices.end(), result.begin(), N_samples, rng);
  return result;
}

} // namespace SIR_SBM
